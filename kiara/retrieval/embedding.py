"""Gemeinsame Embedding-Logik fuer alle VectorStore-Backends.

Unterstuetzt drei Provider:
- ``ollama``: Lokale Embeddings via Ollama-Server
- ``openai``: OpenAI Embedding-API (text-embedding-3-small/large)
- ``voyage``: Voyage AI Embedding-API (voyage-3, voyage-3-lite, ...)
"""

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from functools import wraps

import httpx
import ollama as ollama_client

from kiara.backend_selector import (
    BackendUnavailableError,
    get_all_active,
    get_embedding_backend_for_model,
    normalize_model_name,
)
from kiara.retrieval.embedding_contract import ProfileKind
from kiara.retrieval.embedding_roles import (
    SEARCH_DOCUMENT,
    SEARCH_QUERY,
    EmbeddingRequestProfile,
    emit_embedding_role_telemetry,
    normalize_semantic_query_input,
    request_profile_from_guard,
    voyage_input_type_for_role,
)

logger = logging.getLogger(__name__)

# Batch-Limits pro Provider (max Texte pro API-Request)
_BATCH_LIMITS: dict[str, int] = {
    "openai": 2048,
    "voyage": 1000,
}

# Char-Budget pro Embedding-Request als konservative Token-Approximation.
# Vermeidet Per-Request-Token-Limits der Cloud-Provider, ohne tiktoken/voyageai-
# tokenize als Hard-Dependency zu binden. Worst-Case-Annahme: ~2 chars/token
# (deutsche Wortzusammensetzungen). Werte enthalten ~50% Sicherheitsmarge auf
# das dokumentierte Provider-Token-Limit.
# - openai 500_000: 300k Tokens-Limit -> 600k chars (worst-case), 500k mit Marge
# - voyage 200_000: 120k Tokens-Limit (Free Tier) -> 240k chars, 200k mit Marge
_BATCH_CHAR_BUDGETS: dict[str, int] = {
    "openai": 500_000,
    "voyage": 200_000,
}


def _guard_dense_embedding(operation: str, role: str):
    """Fence dense documents or queries for the selected primary index."""

    def decorate(function):
        @wraps(function)
        def wrapped(config: dict, model: str, *args, **kwargs):
            # Invalid input cannot reach an embedding or index side effect;
            # preserve the public validation error before constructing a
            # physical index target from an empty model.
            if not normalize_model_name(model):
                return function(config, model, *args, _request_profile=None, **kwargs)
            from kiara.retrieval.index_compatibility import (
                dense_index_target,
                guard_index_for_active_job,
                primary_index_target,
            )

            target = (
                primary_index_target(config, model)
                if role == SEARCH_QUERY
                else dense_index_target(config, model)
            )
            with guard_index_for_active_job(
                config,
                target,
                operation=operation,
                write=False,
            ) as token:
                if "_request_profile" in kwargs:
                    raise TypeError("_request_profile ist ausschliesslich intern")
                profile = request_profile_from_guard(
                    token,
                    role=role,
                    expected_kind=ProfileKind(target.vector_kind.value),
                    expected_endpoint=(
                        "/api/embed_late"
                        if target.vector_kind.value == "late_chunking"
                        else "/api/embed"
                    ),
                )
                return function(
                    config,
                    model,
                    *args,
                    _request_profile=profile,
                    **kwargs,
                )

        return wrapped

    return decorate


def _guard_late_embedding(operation: str):
    """Late Chunking is a distinct profile and may never borrow dense proof."""

    def decorate(function):
        @wraps(function)
        def wrapped(config: dict, model: str, *args, **kwargs):
            if not normalize_model_name(model):
                return function(config, model, *args, _request_profile=None, **kwargs)
            from kiara.retrieval.index_compatibility import (
                guard_index_for_active_job,
                late_index_target,
            )

            target = late_index_target(config, model)
            with guard_index_for_active_job(
                config,
                target,
                operation=operation,
                write=False,
            ) as token:
                if "_request_profile" in kwargs:
                    raise TypeError("_request_profile ist ausschliesslich intern")
                profile = request_profile_from_guard(
                    token,
                    role=SEARCH_DOCUMENT,
                    expected_kind=ProfileKind.LATE_CHUNKING,
                    expected_endpoint="/api/embed_late",
                )
                return function(
                    config,
                    model,
                    *args,
                    _request_profile=profile,
                    **kwargs,
                )

        return wrapped

    return decorate


def _iter_token_aware_batches(
    texts: list[str], item_limit: int, char_budget: int
) -> Iterator[list[str]]:
    """Splittet Texte primaer nach char_budget (Token-Approximation),
    sekundaer nach item_limit. Ueberlange Einzeltexte werden als Solo-Batch
    durchgereicht — die Provider-API entscheidet ueber das Per-Item-Limit.
    """
    current: list[str] = []
    current_chars = 0
    for text in texts:
        text_chars = len(text)
        if text_chars >= char_budget:
            if current:
                yield current
                current = []
                current_chars = 0
            yield [text]
            continue
        if current and (current_chars + text_chars > char_budget or len(current) >= item_limit):
            yield current
            current = []
            current_chars = 0
        current.append(text)
        current_chars += text_chars
    if current:
        yield current


# Client-Caches — ein Client pro Endpoint (Connection-Pooling).
# Ein Lock fuer alle drei Dicts (Ollama, OpenAI, Voyage).
_CLIENT_CACHE_MAXSIZE = 32
_ollama_clients: OrderedDict[str, "ollama_client.Client"] = OrderedDict()
_ollama_http_clients: OrderedDict[str, httpx.Client] = OrderedDict()
_openai_clients: OrderedDict[tuple[str, str], object] = (
    OrderedDict()
)  # (api_key, base_url) → openai.OpenAI
_voyage_clients: OrderedDict[str, object] = OrderedDict()  # api_key → voyageai.Client
_client_lock = threading.Lock()


def _close_evicted_client(client) -> None:
    """Best-effort-Cleanup: schliesst Connection-Pool eines evicted Clients."""
    closer = getattr(client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _remember_client(cache: OrderedDict, key, client):
    cache[key] = client
    cache.move_to_end(key)
    while len(cache) > _CLIENT_CACHE_MAXSIZE:
        _, evicted = cache.popitem(last=False)
        _close_evicted_client(evicted)
    return client


def invalidate_clients(category: str | None = None) -> None:
    """Leert gecachte Embedding-Clients nach Backend-Konfigurationsaenderungen."""
    if category not in (None, "embedding"):
        return

    with _client_lock:
        clients = [
            *list(_ollama_clients.values()),
            *list(_ollama_http_clients.values()),
            *list(_openai_clients.values()),
            *list(_voyage_clients.values()),
        ]
        _ollama_clients.clear()
        _ollama_http_clients.clear()
        _openai_clients.clear()
        _voyage_clients.clear()

    for client in clients:
        _close_evicted_client(client)


def _get_ollama_client(url: str) -> "ollama_client.Client":
    """Cached Ollama-Client pro URL (Connection-Pooling, thread-safe)."""
    with _client_lock:
        client = _ollama_clients.get(url)
        if client is None:
            client = ollama_client.Client(host=url, timeout=120.0)
            return _remember_client(_ollama_clients, url, client)
        _ollama_clients.move_to_end(url)
        return client


def _get_ollama_http_client(url: str) -> httpx.Client:
    """Cached httpx-Client pro Ollama-/KIron-URL fuer /api/embed."""
    base_url = url.rstrip("/")
    with _client_lock:
        client = _ollama_http_clients.get(base_url)
        if client is None:
            client = httpx.Client(base_url=base_url, timeout=120.0)
            return _remember_client(_ollama_http_clients, base_url, client)
        _ollama_http_clients.move_to_end(base_url)
        return client


def _get_openai_client(api_key: str, base_url: str = ""):
    """Cached OpenAI-Client pro (api_key, base_url), thread-safe."""
    cache_key = (api_key, base_url or "")
    with _client_lock:
        client = _openai_clients.get(cache_key)
        if client is None:
            try:
                import openai
            except ImportError:
                raise BackendUnavailableError(
                    "OpenAI-SDK nicht installiert. Bitte 'pip install openai' ausfuehren."
                )
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=30.0,  # Client-Level-Timeout, bewusst grosszuegig: mehrtaegige Ingest-Jobs sollen nicht abbrechen wenn Backend kurzzeitig im Wartungsmodus ist
            )
            return _remember_client(_openai_clients, cache_key, client)
        _openai_clients.move_to_end(cache_key)
        return client


def _get_voyage_client(api_key: str):
    """Cached Voyage-Client pro api_key, thread-safe."""
    with _client_lock:
        client = _voyage_clients.get(api_key)
        if client is None:
            try:
                import voyageai
            except ImportError:
                raise BackendUnavailableError(
                    "Voyage-AI-SDK nicht installiert. Bitte 'pip install voyageai' ausfuehren."
                )
            client = voyageai.Client(api_key=api_key)
            return _remember_client(_voyage_clients, api_key, client)
        _voyage_clients.move_to_end(api_key)
        return client


_GPU_GATE_BODY_MARKER = "GPU-Operation blockiert"
"""Antwort-Marker des Kiron-Proxy-GPU-Gates (Docling-Start, Modellwechsel).

Diese Blockaden sind transient (Sekunden bis wenige Minuten) und der Proxy
bittet im Hint selbst um Wiederholung — deshalb wird hier begrenzt retried
statt sofort fail-closed abzubrechen (3 Vorfaelle am 2026-07-07 rissen sonst
ganze Benchmark-/Chat-Retrievals ab).
"""

EMBED_GATE_RETRY_MAX_WAIT_S = 90.0
"""Maximale Gesamtwartezeit auf ein blockiertes GPU-Gate (Docling-Start ~25s,
Ollama-Modellwechsel bis ~90s). Danach regulaerer BackendUnavailableError."""

EMBED_GATE_RETRY_DELAY_S = 3.0
"""Pause zwischen GPU-Gate-Retries."""


def _ollama_embed(
    url: str,
    model: str,
    inputs: list[str],
    input_type: str,
    *,
    keep_alive: str | None = None,
) -> list[list[float]]:
    """POST /api/embed mit kanonischer, expliziter RAG-Rolle.

    Nutzt httpx statt des ollama-Python-SDKs, weil letzterer unbekannte
    Felder wie input_type nicht durchreicht. Konsistent mit embed_texts_late.
    """
    if input_type not in (SEARCH_DOCUMENT, SEARCH_QUERY):
        raise ValueError(f"Ungueltiger input_type fuer /api/embed: {input_type!r}")
    payload: dict = {"model": model, "input": inputs, "input_type": input_type}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    client = _get_ollama_http_client(url)
    deadline = time.monotonic() + EMBED_GATE_RETRY_MAX_WAIT_S
    gate_attempts = 0
    while True:
        try:
            response = client.post("/api/embed", json=payload)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            body = exc.response.text or ""
            status = exc.response.status_code
            if (
                status in (409, 503)
                and _GPU_GATE_BODY_MARKER in body
                and time.monotonic() + EMBED_GATE_RETRY_DELAY_S < deadline
            ):
                gate_attempts += 1
                logger.warning(
                    "Embedding vom Kiron-GPU-Gate blockiert (HTTP %d), Retry %d in %.0fs",
                    status,
                    gate_attempts,
                    EMBED_GATE_RETRY_DELAY_S,
                )
                time.sleep(EMBED_GATE_RETRY_DELAY_S)
                continue
            if status == 404 or "not found" in body.lower():
                raise BackendUnavailableError(
                    f"Modell '{model}' nicht auf Embedding-Server gefunden: {body}"
                ) from exc
            raise BackendUnavailableError(
                f"Embedding-Server-Fehler (HTTP {status}): {body}"
            ) from exc
        except httpx.RequestError as exc:
            raise BackendUnavailableError(
                f"Embedding-Server {url} nicht erreichbar: {exc.__class__.__name__}"
            ) from exc
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError(
            f"Embedding-Server {url} lieferte ungueltige JSON-Antwort: {exc.__class__.__name__}"
        ) from exc
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise BackendUnavailableError(f"Embedding-Server {url} lieferte ungueltige Antwort: {data}")
    return embeddings


_SAFE_COLLECTION_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyz0123456789-")


def _hex_encode(name: str) -> str:
    """UTF-8-Bytes zu _hh-escaped ASCII (ohne Prefix)."""
    body = []
    for b in name.encode("utf-8"):
        if b in _SAFE_COLLECTION_BYTES:
            body.append(chr(b))
        else:
            body.append(f"_{b:02x}")
    return "".join(body)


def _hex_decode(encoded: str, context: str = "") -> str:
    """Umkehrung von _hex_encode. Alle Fehler als ValueError."""
    error_ref = context or encoded
    raw = bytearray()
    i = 0
    while i < len(encoded):
        ch = encoded[i]
        if ch != "_":
            raw.append(ord(ch))
            i += 1
            continue
        if i + 2 >= len(encoded):
            raise ValueError(f"Ungueltige Escape-Sequenz in {error_ref!r}")
        try:
            raw.append(int(encoded[i + 1 : i + 3], 16))
        except ValueError as exc:
            raise ValueError(f"Ungueltige Escape-Sequenz in {error_ref!r}") from exc
        i += 3
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Ungueltiger UTF-8-Body in {error_ref!r}") from exc


def collection_name_for_model(model_name: str) -> str:
    """Erzeugt einen reversiblen Qdrant-Collection-Namen aus einem Modellnamen.

    Verwendet byte-basiertes ``_hh``-Escape ueber UTF-8:
    - Safe-Zeichen (a-z, 0-9, -) bleiben unveraendert
    - Jedes andere UTF-8-Byte wird als ``_hh`` kodiert (2 lowercase Hex-Digits)
    - Prefix: ``kb_``

    Input wird zentral normalisiert und darf danach nicht leer sein.
    """
    normalized = normalize_model_name(model_name)
    if not normalized:
        raise ValueError("collection_name_for_model erwartet einen nicht-leeren Modellnamen")
    return "kb_" + _hex_encode(normalized)


def model_name_from_collection(collection_name: str) -> str:
    """Rekonstruiert den Modellnamen aus einem Qdrant-Collection-Namen.

    Inverse von ``collection_name_for_model()``. Liest den Body nach ``kb_``
    links-nach-rechts: bei ``_`` werden die naechsten zwei Hex-Digits als
    ein UTF-8-Byte interpretiert; alle anderen Zeichen sind Literal-ASCII.
    """
    if not collection_name.startswith("kb_"):
        raise ValueError(f"Kein KIara-Collection-Name: {collection_name!r}")
    return _hex_decode(collection_name[3:], collection_name)


def _resolve_backend(model: str) -> dict:
    """Ermittelt das Backend fuer ein Embedding-Modell (DB-only).

    Returns:
        Dict mit ``url``, ``provider``, ``api_key``, ``base_url``, ``model``.

    Raises:
        BackendUnavailableError: Kein erreichbares Backend gefunden.
    """
    from cryptography.fernet import InvalidToken

    from kiara.retrieval.index_compatibility import exact_backend_from_guard
    from kiara.shared.crypto import decrypt, is_encrypted

    backend = exact_backend_from_guard(model)
    if backend is None:
        backend = get_embedding_backend_for_model(model)
    if backend:
        provider = backend.get("provider", "ollama")
        api_key = backend.get("api_key", "")
        base_url = backend.get("base_url", "")

        # API-Key-Handling fuer Cloud-Provider: backend_selector entschluesselt
        # bereits best-effort. Nur Restfall behandeln (is_encrypted noch True).
        if provider in ("openai", "voyage"):
            if api_key and is_encrypted(api_key):
                try:
                    api_key = decrypt(api_key)
                except InvalidToken:
                    raise BackendUnavailableError(
                        f"API-Key fuer Backend '{backend.get('name', '?')}' "
                        f"konnte nicht entschluesselt werden (Schluessel-Mismatch)"
                    )
            if not api_key:
                raise BackendUnavailableError(
                    f"API-Key fehlt fuer Cloud-Embedding-Backend "
                    f"'{backend.get('name', '?')}' (Provider: {provider})"
                )

        return {
            "url": backend.get("url", ""),
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": backend.get("model") or model,
        }

    # Kein erreichbares Backend gefunden — Ursache differenzieren
    all_embedding = get_all_active("embedding")
    if all_embedding:
        norm_model = normalize_model_name(model)
        matching = [b for b in all_embedding if normalize_model_name(b["model"]) == norm_model]
        if matching:
            urls = [b["url"] for b in matching]
            raise BackendUnavailableError(
                f"Embedding-Backend fuer Modell '{model}' gefunden aber nicht erreichbar "
                f"(Health-Check fehlgeschlagen). URLs: {urls}"
            )
        raise BackendUnavailableError(
            f"Kein Embedding-Backend fuer Modell '{model}' konfiguriert. "
            f"Verfuegbare Modelle: {[normalize_model_name(b['model']) for b in all_embedding]}"
        )
    raise BackendUnavailableError(
        f"Kein Embedding-Backend konfiguriert. Modell '{model}' kann nicht "
        "verwendet werden. Bitte ein Embedding-Backend in der Admin-UI anlegen."
    )


def fetch_ollama_model_info(url: str, model: str, timeout: float = 5.0) -> dict:
    """Synchrone Abfrage von model_info via Ollama /api/show.

    Gibt Dict mit verfuegbaren Keys zurueck:
    - 'embedding_dim': int (wenn model_info.*.embedding_length vorhanden)
    - 'num_ctx': int (wenn model_info.*.context_length vorhanden)

    Wird genutzt von:
    - _fetch_embedding_dim_sync() in embedding.py (VectorStore-Init)
    - _fetch_model_info() in api_ki_backends.py (Backend-Setup)
    """
    try:
        client = ollama_client.Client(host=url, timeout=timeout)
        info = client.show(model)
        model_info = info.get("modelinfo") or info.get("model_info") or {}
        result = {}
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                if key.endswith("embedding_length"):
                    result["embedding_dim"] = int(value)
                elif key.endswith("context_length"):
                    result["num_ctx"] = int(value)
        return result
    except Exception as exc:
        logger.debug("Ollama model-info fuer %s nicht abrufbar: %s", model, exc)
        return {}


def _fetch_embedding_dim_sync(url: str, model: str) -> int | None:
    """Synchrone Abfrage der Embedding-Dimension via Ollama /api/show."""
    info = fetch_ollama_model_info(url, model, timeout=5.0)
    return info.get("embedding_dim") if info else None


def _cache_embedding_dim(backend_id: int, dim: int) -> None:
    """Speichert ermittelte Dimension opportunistisch in extra_config."""
    import json

    try:
        from kiara.repositories.backend import KiBackendRepository

        _backend_repo = KiBackendRepository()
        row = _backend_repo.get_by_id(backend_id)
        if row:
            extra = json.loads(row.get("extra_config") or "{}")
            extra["embedding_dim"] = dim
            _backend_repo.update(backend_id, extra_config=json.dumps(extra))
    except Exception as exc:
        logger.debug("Embedding-Dim-Caching fehlgeschlagen (best-effort): %s", exc)


def get_embedding_dimension(model: str, config: dict) -> int:
    """Ermittelt die Vektordimension fuer ein Embedding-Modell.

    Kaskade:
    0. Dimension des atomar validierten Profils im aktiven Index-Guard
    1. extra_config['embedding_dim'] aus ki_backends (gecacht, OHNE Health-Check)
    2. Ollama /api/show (Live, nur wenn 1. fehlt)
    3. KNOWN_EMBEDDING_DIMENSIONS Lookup
    4. config['vectordb']['vector_size'] (Fallback)
    5. DEFAULT_VECTOR_SIZE (768)

    Schritt 1 liest die gecachte Dimension unabhaengig von der Backend-
    Erreichbarkeit — die Dimension ist eine Modell-Konstante und aendert
    sich nicht durch Downtime.
    """
    from kiara.shared.constants import DEFAULT_VECTOR_SIZE, KNOWN_EMBEDDING_DIMENSIONS
    from kiara.web.config_utils import safe_json_loads

    norm = normalize_model_name(model)

    # 1. Gecachte Dimension aus ki_backends (ohne Health-Check).
    #    get_all_active() liefert alle aktiven Backends unabhaengig
    #    von der Erreichbarkeit — die Dimension ist statisch.
    from kiara.retrieval.index_compatibility import current_guard_token

    active_guard = current_guard_token(model=model)
    if active_guard is not None:
        snapshot_text = getattr(active_guard.resolved, "contract_snapshot_json_text", None)
        if snapshot_text:
            snapshot = json.loads(snapshot_text)
            for profile in snapshot.get("profiles", []):
                if profile.get("profile_id") == active_guard.resolved.profile_id:
                    dim = profile.get("dimensions")
                    if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                        return dim
    exact_backend = active_guard.resolved.backend if active_guard else None
    all_embedding = [exact_backend] if exact_backend else get_all_active("embedding")
    matching = [b for b in all_embedding if normalize_model_name(b["model"]) == norm]
    for b in matching:
        extra = safe_json_loads(b.get("extra_config", "{}"), default={})
        if not isinstance(extra, dict):
            extra = {}
        dim = extra.get("embedding_dim")
        if isinstance(dim, int) and dim > 0:
            return dim

    # 2. Live-Abfrage via Ollama /api/show (mit 5s Timeout)
    backend = exact_backend or get_embedding_backend_for_model(model)
    if backend and backend.get("provider") == "ollama":
        dim = _fetch_embedding_dim_sync(backend["url"], model)
        if isinstance(dim, int) and dim > 0:
            # Kein Config-Write unter dem Shared-Contract-Lock: der Trigger
            # benoetigt exklusiv denselben Lock. Opportunistisches Caching ist
            # ausserhalb eines Guards weiterhin erlaubt.
            if active_guard is None:
                _cache_embedding_dim(backend["id"], dim)
            return dim

    # 3. Lookup-Tabelle
    dim = KNOWN_EMBEDDING_DIMENSIONS.get(norm)
    if dim:
        return dim

    # 4/5. Config-Fallback
    return config.get("vectordb", {}).get("vector_size", DEFAULT_VECTOR_SIZE)


def _validate_embedding_dim(
    embeddings: list[list[float]],
    expected_dim: int,
    model: str,
    expected_count: int | None = None,
) -> None:
    """Prueft Cardinality und Per-Vektor-Dimension der Embedding-Antwort.

    Fail-closed bei:
    - Cardinality-Mismatch (``len(embeddings) != expected_count``): Provider-
      Anomalien wie unvollstaendige Batches wuerden sonst erst beim Qdrant-
      Insert mit kryptischer Meldung auffliegen oder via
      ``embeddings[0]`` einen IndexError werfen.
    - Per-Vektor-Dimensions-Mismatch: Wenn das Backend eine andere Vektor-
      Dimension liefert als ``get_embedding_dimension()`` zwischengespeichert
      hat (Modell-Drift) oder ein einzelner Vektor abweicht.
    """
    if expected_count is not None and len(embeddings) != expected_count:
        raise ValueError(
            f"Embedding-Cardinality-Mismatch fuer Modell '{model}': "
            f"erwartet {expected_count} Vektoren, erhalten {len(embeddings)}. "
            "Provider-Antwort unvollstaendig — Backend-Konfiguration und API-Limits pruefen."
        )
    for idx, vec in enumerate(embeddings):
        if not isinstance(vec, list):
            raise ValueError(
                f"Embedding-Typ-Fehler fuer Modell '{model}' (Vektor {idx}): "
                f"erwartet list, erhalten {type(vec).__name__}."
            )
        actual_dim = len(vec)
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding-Dimension-Mismatch fuer Modell '{model}' "
                f"(Vektor {idx}): erwartet {expected_dim}, erhalten {actual_dim}. "
                "Modell-Drift erkannt — Backend-Konfiguration und ki_backends.extra_config pruefen."
            )


def _embed_openai(client, model: str, texts: list[str]) -> list[list[float]]:
    """OpenAI-Embedding mit Batch-Splitting (Item-Count + Token-Budget)."""
    try:
        import openai as openai_mod
    except ImportError:
        raise BackendUnavailableError(
            "OpenAI-SDK nicht installiert. Bitte 'pip install openai' ausfuehren."
        )
    item_limit = _BATCH_LIMITS["openai"]
    char_budget = _BATCH_CHAR_BUDGETS["openai"]
    all_embeddings: list[list[float]] = []
    try:
        for batch in _iter_token_aware_batches(texts, item_limit, char_budget):
            response = client.embeddings.create(model=model, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
    except openai_mod.AuthenticationError as exc:
        raise BackendUnavailableError(
            f"OpenAI-Authentifizierung fehlgeschlagen: {exc.__class__.__name__}"
        ) from exc
    except openai_mod.RateLimitError as exc:
        raise BackendUnavailableError(
            f"OpenAI Rate-Limit erreicht: {exc.__class__.__name__}"
        ) from exc
    except (openai_mod.APIConnectionError, openai_mod.APITimeoutError) as exc:
        raise BackendUnavailableError(f"OpenAI nicht erreichbar: {exc.__class__.__name__}") from exc
    except openai_mod.APIError as exc:
        raise BackendUnavailableError(f"OpenAI API-Fehler: {exc.__class__.__name__}") from exc
    return all_embeddings


def _embed_voyage(client, model: str, texts: list[str], input_type: str) -> list[list[float]]:
    """Voyage-Embedding mit Batch-Splitting (Item-Count + Token-Budget)."""
    try:
        from voyageai import error as voyage_error
    except ImportError:
        raise BackendUnavailableError(
            "Voyage-AI-SDK nicht installiert. Bitte 'pip install voyageai' ausfuehren."
        )
    item_limit = _BATCH_LIMITS["voyage"]
    char_budget = _BATCH_CHAR_BUDGETS["voyage"]
    voyage_input_type = voyage_input_type_for_role(input_type)
    all_embeddings: list[list[float]] = []
    try:
        for batch in _iter_token_aware_batches(texts, item_limit, char_budget):
            result = client.embed(batch, model=model, input_type=voyage_input_type)
            all_embeddings.extend(result.embeddings)
    except voyage_error.AuthenticationError as exc:
        raise BackendUnavailableError(
            f"Voyage-Authentifizierung fehlgeschlagen: {exc.__class__.__name__}"
        ) from exc
    except Exception as exc:
        raise BackendUnavailableError(f"Voyage API-Fehler: {exc.__class__.__name__}") from exc
    return all_embeddings


@_guard_dense_embedding("document_embedding", SEARCH_DOCUMENT)
def embed_texts(
    config: dict,
    model: str,
    texts: list[str],
    *,
    keep_alive: str | None = None,
    _request_profile: EmbeddingRequestProfile,
) -> list[list[float]]:
    """Embeddings fuer Dokumente generieren.

    Unterstuetzt Ollama, OpenAI und Voyage AI als Provider.
    config bleibt in der Signatur fuer Abwaertskompatibilitaet.
    """
    model = normalize_model_name(model)
    if not model:
        raise ValueError(
            "Kein Embedding-Modell angegeben. Bitte ein Modell in der "
            "Chat-UI oder API-Anfrage auswaehlen."
        )
    if not texts:
        return []
    if any(not t or not t.strip() for t in texts):
        raise ValueError("embed_texts(): Leere/Whitespace-Strings im Batch sind nicht erlaubt")

    backend = _resolve_backend(model)
    provider = backend["provider"]
    expected_dim = get_embedding_dimension(model, config)
    emit_embedding_role_telemetry(
        _request_profile,
        operation="document_embedding",
        input_count=len(texts),
    )

    if provider == "openai":
        client = _get_openai_client(backend["api_key"], backend["base_url"])
        embeddings = _embed_openai(client, model, texts)
    elif provider == "voyage":
        client = _get_voyage_client(backend["api_key"])
        embeddings = _embed_voyage(client, model, texts, input_type=SEARCH_DOCUMENT)
    else:
        embeddings = _ollama_embed(
            backend["url"],
            backend["model"],
            texts,
            SEARCH_DOCUMENT,
            keep_alive=keep_alive,
        )

    _validate_embedding_dim(embeddings, expected_dim, model, expected_count=len(texts))
    return embeddings


@_guard_dense_embedding("query_embedding", SEARCH_QUERY)
def embed_query(
    config: dict,
    model: str,
    query: str,
    *,
    _request_profile: EmbeddingRequestProfile,
) -> list[float]:
    """Embedding fuer eine Query generieren.

    Unterstuetzt Ollama, OpenAI und Voyage AI als Provider.
    config bleibt in der Signatur fuer Abwaertskompatibilitaet.
    """
    model = normalize_model_name(model)
    if not model:
        raise ValueError(
            "Kein Embedding-Modell angegeben. Bitte ein Modell in der "
            "Chat-UI oder API-Anfrage auswaehlen."
        )
    query = normalize_semantic_query_input(query)

    backend = _resolve_backend(model)
    provider = backend["provider"]
    expected_dim = get_embedding_dimension(model, config)
    emit_embedding_role_telemetry(
        _request_profile,
        operation="query_embedding",
        input_count=1,
    )

    if _request_profile.kind == ProfileKind.LATE_CHUNKING.value:
        if provider != "ollama":
            raise ValueError("Late-Query benoetigt ein verifiziertes KIron-Late-Profil")
        response = httpx.post(
            backend["url"].rstrip("/") + "/api/embed_late",
            json={
                "model": backend["model"],
                "input_type": SEARCH_QUERY,
                "document": query,
                "chunks": [{"text": query, "char_start": 0, "char_end": len(query)}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Late-Query lieferte keine Embedding-Liste")
    elif provider == "openai":
        client = _get_openai_client(backend["api_key"], backend["base_url"])
        embeddings = _embed_openai(client, model, [query])
    elif provider == "voyage":
        client = _get_voyage_client(backend["api_key"])
        embeddings = _embed_voyage(client, model, [query], input_type=SEARCH_QUERY)
    else:
        embeddings = _ollama_embed(backend["url"], backend["model"], [query], SEARCH_QUERY)

    _validate_embedding_dim(embeddings, expected_dim, model, expected_count=1)
    return embeddings[0]


@_guard_late_embedding("late_document_embedding")
def embed_texts_late(
    config: dict,
    model: str,
    document: str,
    chunks: list[str],
    chunk_spans: list[tuple[int, int]],
    *,
    _request_profile: EmbeddingRequestProfile,
) -> list[list[float]] | None:
    """Late-Chunking-Embeddings: Kontextbewusste Vektoren via KIron.

    Sendet das vollstaendige Dokument + Chunk-Texte + char spans an den
    /api/embed_late Endpoint via httpx POST.

    Kein interner Fallback auf embed_texts(): Bei Fehler wird None
    zurueckgegeben. Der Caller muss die Datei als fehlgeschlagen behandeln;
    ein Late-Index darf keine Dense-Ersatzvektoren aufnehmen.

    Die Backend-URL wird via _resolve_backend(model) ermittelt.
    Nur fuer Ollama-Backends — Cloud-Provider haben keinen Late-Endpoint.

    Args:
        config:      Globale Konfiguration.
        model:       Embedding-Modellname (normalisiert via normalize_model_name).
        document:    Vollstaendiger Dokumenttext.
        chunks:      Liste der Chunk-Texte.
        chunk_spans: Liste (char_start, char_end) in gleicher Reihenfolge wie chunks.

    Returns:
        Liste von Vektoren (gleiche Laenge wie chunks) bei Erfolg.
        None bei Fehler (Warning wird geloggt).
    """
    model = normalize_model_name(model)
    if not model:
        raise ValueError(
            "Kein Embedding-Modell angegeben. Bitte ein Modell in der "
            "Chat-UI oder API-Anfrage auswaehlen."
        )
    if not chunks:
        return []
    if len(chunk_spans) != len(chunks):
        logger.warning(
            "embed_texts_late: chunk_spans-Laenge (%d) != chunks-Laenge (%d)",
            len(chunk_spans),
            len(chunks),
        )
        return None
    try:
        backend = _resolve_backend(model)

        if backend.get("provider") != "ollama":
            logger.warning(
                "embed_texts_late: Provider '%s' unterstuetzt kein Late Chunking",
                backend.get("provider"),
            )
            return None

        url = f"{backend['url'].rstrip('/')}/api/embed_late"
        payload = {
            "model": backend["model"],
            "input_type": SEARCH_DOCUMENT,
            "document": document,
            "chunks": [
                {"text": text, "char_start": start, "char_end": end}
                for text, (start, end) in zip(chunks, chunk_spans, strict=True)
            ],
        }

        emit_embedding_role_telemetry(
            _request_profile,
            operation="late_document_embedding",
            input_count=len(chunks),
        )

        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(chunks):
            logger.warning(
                "embed_texts_late: Unerwartete Antwort — erwartet %d Embeddings, erhalten: %s",
                len(chunks),
                len(embeddings) if isinstance(embeddings, list) else type(embeddings).__name__,
            )
            return None

        expected_dim = get_embedding_dimension(model, config)
        _validate_embedding_dim(embeddings, expected_dim, model)

        return embeddings

    except json.JSONDecodeError as exc:
        logger.warning("embed_texts_late: JSON-Dekodierungsfehler: %s", exc)
        return None
    except (ValueError, BackendUnavailableError):
        raise
    except Exception as exc:
        logger.warning("embed_texts_late fehlgeschlagen: %s", exc)
        return None
