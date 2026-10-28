"""
KI-Backend CRUD Service — extrahiert aus api_ki_backends.py.
Wird von Admin-Funktionen (Chat) und Auto-Router (REST) genutzt.
"""

import json
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {"embedding", "llm", "cloud", "vlm", "ddu"}

_VALID_PROVIDERS = {
    "embedding": ["ollama", "openai", "voyage"],
    "llm": ["ollama"],
    "cloud": ["anthropic", "openai"],
    "vlm": ["ollama"],
    "ddu": ["docling"],
}

_API_KEY_PROVIDERS = {"anthropic", "openai", "voyage"}


def _provider_needs_api_key(provider: str) -> bool:
    return provider in _API_KEY_PROVIDERS


def _validate_url(value: str, field_name: str = "URL") -> None:
    """Prueft, ob ein nicht-leerer URL-String ein gueltiges Format hat."""
    if not value:
        return
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{field_name}: Schema muss http oder https sein, erhalten: {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ValueError(f"{field_name}: Hostname fehlt")


def _validate_backend_url_requirements(category: str, provider: str, url: str) -> None:
    """Prueft Provider-spezifische URL-Pflichtfelder."""
    if provider == "ollama" and not url:
        raise ValueError("url ist fuer Ollama-Backends erforderlich")
    if category == "ddu" and provider == "docling" and not url:
        raise ValueError("url ist fuer DDU-Backends erforderlich; lokales Docling ist deaktiviert")


def validate_backend_params(category: str, provider: str) -> None:
    """Validiert category und provider. Wirft ValueError bei Fehler."""
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Ungueltige Kategorie: {category}. Erlaubt: {', '.join(sorted(_VALID_CATEGORIES))}"
        )
    valid = _VALID_PROVIDERS.get(category, [])
    if provider not in valid:
        raise ValueError(
            f"Ungueltiger Provider '{provider}' fuer Kategorie '{category}'. "
            f"Erlaubt: {', '.join(valid)}"
        )


async def create_backend(
    *,
    category: str,
    provider: str,
    name: str = "",
    url: str = "",
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    is_active: bool = True,
    priority: int = 10,
    extra_config: dict | None = None,
    trusted: bool = False,
) -> dict:
    """Erstellt ein neues KI-Backend. Gibt {"id": ..., "message": ...} zurueck."""
    from kiara.backend_selector import invalidate_cache
    from kiara.repositories.backend import KiBackendRepository
    from kiara.shared.constants import PASSWORD_MASK
    from kiara.shared.crypto import encrypt

    validate_backend_params(category, provider)
    url = url.strip() if isinstance(url, str) else url
    base_url = base_url.strip() if isinstance(base_url, str) else base_url
    _validate_url(url, "url")
    _validate_url(base_url, "base_url")
    api_key = api_key.strip() if isinstance(api_key, str) else api_key
    _validate_backend_url_requirements(category, provider, url)
    if _provider_needs_api_key(provider) and (not api_key or api_key == PASSWORD_MASK):
        raise ValueError(f"api_key ist fuer Provider '{provider}' erforderlich")

    from kiara.backend_selector import normalize_model_name

    model = normalize_model_name(model)
    if category == "ddu" and not model:
        model = "docling"
    if not model:
        raise ValueError("Modell ist erforderlich")

    effective_name = name or f"{category.title()} ({provider})"
    extra = extra_config or {}

    # Cloud-Embedding: embedding_dim kanonisch setzen / validieren
    if category == "embedding" and provider in ("openai", "voyage"):
        extra = _normalize_embedding_dim(extra, model)

    # API-Key verschluesseln
    encrypted_key = ""
    if api_key and api_key != PASSWORD_MASK:
        encrypted_key = encrypt(api_key)

    # trusted: nur cloud-Backends mit anthropic/openai duerfen trusted=True
    if category == "cloud" and provider in ("anthropic", "openai"):
        effective_trusted = bool(trusted)
    else:
        effective_trusted = False

    backend_repo = KiBackendRepository()
    new_id = backend_repo.create(
        category=category,
        provider=provider,
        name=effective_name,
        url=url,
        base_url=base_url,
        model=model,
        api_key=encrypted_key,
        is_active=bool(is_active),
        priority=priority,
        extra_config=json.dumps(extra, ensure_ascii=False),
        trusted=effective_trusted,
    )

    # Ollama: num_ctx + embedding_dim automatisch ermitteln (non-fatal)
    if provider == "ollama" and url and model:
        needed = {"num_ctx", "embedding_dim"} - set(extra.keys())
        if needed:
            try:
                model_info = await _fetch_model_info(url, model)
            except Exception as exc:
                logger.warning("Ollama model-info nicht abrufbar: %s", exc)
                model_info = None
            if model_info:
                merged = dict(extra, **{k: v for k, v in model_info.items() if k not in extra})
                if merged != extra:
                    backend_repo.update(new_id, extra_config=json.dumps(merged, ensure_ascii=False))

    invalidate_cache(category)
    if category == "ddu":
        _invalidate_ddu()

    logger.info("Backend %d (%s/%s) erstellt", new_id, category, provider)
    return {"id": new_id, "message": "Backend erstellt", "name": effective_name}


async def update_backend(
    backend_id: int,
    *,
    category: str | None = None,
    provider: str | None = None,
    name: str | None = None,
    url: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    is_active: bool | None = None,
    priority: int | None = None,
    extra_config: dict | None = None,
    trusted: bool | None = None,
) -> dict:
    """Aktualisiert ein bestehendes KI-Backend."""
    from kiara.backend_selector import invalidate_cache
    from kiara.repositories.backend import KiBackendRepository
    from kiara.repositories.base import RawSQL
    from kiara.shared.constants import PASSWORD_MASK
    from kiara.shared.crypto import encrypt

    backend_repo = KiBackendRepository()
    existing = backend_repo.get_by_id(backend_id)
    if not existing:
        raise ValueError(f"Backend {backend_id} nicht gefunden")

    # Effektive Werte: uebergebene oder bestehende
    eff_category = category if category is not None else existing["category"]
    eff_provider = provider if provider is not None else existing["provider"]
    validate_backend_params(eff_category, eff_provider)
    eff_url = url if url is not None else existing["url"]
    eff_base_url = base_url if base_url is not None else existing.get("base_url", "")
    eff_url = eff_url.strip() if isinstance(eff_url, str) else eff_url
    eff_base_url = eff_base_url.strip() if isinstance(eff_base_url, str) else eff_base_url
    _validate_url(eff_url, "url")
    _validate_url(eff_base_url, "base_url")
    _validate_backend_url_requirements(eff_category, eff_provider, eff_url)

    from kiara.backend_selector import normalize_model_name

    eff_model = model if model is not None else existing["model"]
    eff_model = normalize_model_name(eff_model)
    if eff_category == "ddu" and not eff_model:
        eff_model = "docling"
    if not eff_model:
        raise ValueError("Modell ist erforderlich")

    eff_name = name if name is not None else existing["name"]
    if not eff_name:
        eff_name = f"{eff_category.title()} ({eff_provider})"

    credential_scope_changed = (
        eff_category != existing["category"] or eff_provider != existing["provider"]
    )
    target_needs_api_key = _provider_needs_api_key(eff_provider)
    if (
        credential_scope_changed
        and target_needs_api_key
        and (api_key is None or api_key == PASSWORD_MASK or not api_key)
    ):
        raise ValueError("Providerwechsel zu einem API-Key-Provider erfordert einen neuen API-Key")

    # API-Key: PASSWORD_MASK → behalten, leer → loeschen, sonst → verschluesseln.
    # Beim Wechsel zu einem Provider ohne API-Key wird ein altes Secret verworfen.
    if credential_scope_changed and not target_needs_api_key:
        eff_api_key = ""
    elif api_key is None or api_key == PASSWORD_MASK:
        eff_api_key = existing["api_key"]
    elif api_key:
        eff_api_key = encrypt(api_key)
    else:
        eff_api_key = ""

    if extra_config is not None:
        eff_extra = extra_config
    else:
        try:
            eff_extra = json.loads(existing.get("extra_config") or "{}")
        except (ValueError, TypeError):
            eff_extra = {}
    if not isinstance(eff_extra, dict):
        try:
            eff_extra = json.loads(eff_extra) if isinstance(eff_extra, str) else {}
        except (json.JSONDecodeError, TypeError):
            eff_extra = {}

    # Cloud-Embedding: embedding_dim kanonisch setzen / validieren
    if eff_category == "embedding" and eff_provider in ("openai", "voyage"):
        eff_extra = _normalize_embedding_dim(eff_extra, eff_model)

    eff_extra = json.dumps(eff_extra, ensure_ascii=False)

    # trusted: nur cloud-Backends mit anthropic/openai duerfen trusted=True
    if eff_category == "cloud" and eff_provider in ("anthropic", "openai"):
        eff_trusted = bool(trusted) if trusted is not None else bool(existing.get("trusted", False))
    else:
        eff_trusted = False

    backend_repo.update(
        backend_id,
        category=eff_category,
        provider=eff_provider,
        name=eff_name,
        url=eff_url,
        base_url=eff_base_url,
        model=eff_model,
        api_key=eff_api_key,
        is_active=bool(is_active) if is_active is not None else bool(existing["is_active"]),
        priority=priority if priority is not None else existing["priority"],
        extra_config=eff_extra,
        trusted=eff_trusted,
        updated_at=RawSQL("NOW()"),
    )

    invalidate_cache(eff_category)
    if existing["category"] != eff_category:
        invalidate_cache(existing["category"])
        if existing["category"] == "ddu":
            _invalidate_ddu()
    if eff_category == "ddu":
        _invalidate_ddu()

    logger.info("Backend %d aktualisiert", backend_id)
    return {"message": "Backend aktualisiert", "name": eff_name}


async def delete_backend(backend_id: int) -> dict:
    """Loescht ein Backend."""
    from kiara.backend_selector import invalidate_cache
    from kiara.repositories.backend import KiBackendRepository

    backend_repo = KiBackendRepository()
    existing = backend_repo.get_by_id(backend_id)
    if not existing:
        raise ValueError(f"Backend {backend_id} nicht gefunden")

    category = existing["category"]

    backend_repo.delete(backend_id)
    invalidate_cache(category)
    if category == "ddu":
        _invalidate_ddu()

    logger.info("Backend %d geloescht", backend_id)
    return {"message": "Backend geloescht", "name": existing.get("name", "")}


def _decrypt_api_key(backend: dict) -> tuple[str, str]:
    """API-Key aus Backend-Record entschluesseln.

    Returns: (api_key, error) — api_key ist "" bei Fehler, error ist "" bei Erfolg.
    """
    from kiara.shared.crypto import decrypt, is_encrypted

    api_key = backend.get("api_key", "")
    if api_key:
        try:
            if is_encrypted(api_key):
                api_key = decrypt(api_key)
        except Exception:
            return "", "API-Key konnte nicht entschluesselt werden (Schluessel-Mismatch)"
    if not api_key:
        return "", "Kein API-Key konfiguriert"
    return api_key, ""


def _test_cloud_llm(
    provider: str, api_key: str, base_url: str | None = None
) -> tuple[bool, str, list[str]]:
    """Cloud-LLM-Test (Anthropic/OpenAI): models.list() ausfuehren.

    Returns: (success, error, models_list)
    """
    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, timeout=10.0)
            result = client.models.list()
            models = [m.id for m in result.data] if hasattr(result, "data") else []
            return True, "", models
        else:  # openai
            import openai

            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
            result = client.models.list()
            return True, "", [m.id for m in result.data]
    except Exception as exc:
        return False, str(exc), []


async def test_backend(backend_id: int) -> dict:
    """Verbindungstest fuer ein Backend. Gibt {"success": ..., "models": [...]} zurueck."""
    from kiara.backend_selector import invalidate_cache
    from kiara.config import DEFAULT_TIMEOUTS
    from kiara.repositories.backend import KiBackendRepository

    backend_repo = KiBackendRepository()
    backend = backend_repo.get_by_id(backend_id)
    if not backend:
        raise ValueError(f"Backend {backend_id} nicht gefunden")

    provider = backend["provider"]
    category = backend["category"]
    success = False
    error = ""
    models: list[str] = []
    dimension: int | None = None

    if provider == "ollama":
        from kiara.web.api_ki import _test_ollama_connection

        result = await _test_ollama_connection(backend["url"])
        success = result["success"]
        error = result.get("error", "")
        models = result.get("models", [])

    elif provider in ("anthropic", "openai"):
        api_key, error = _decrypt_api_key(backend)
        if not error:
            if category == "embedding" and provider == "openai":
                success, error, dimension = _test_embedding_cloud(
                    provider, api_key, backend.get("model", ""), backend.get("base_url", "")
                )
            else:
                success, error, models = _test_cloud_llm(
                    provider, api_key, backend.get("base_url") or None
                )

    elif provider == "voyage":
        api_key, error = _decrypt_api_key(backend)
        if not error:
            success, error, dimension = _test_embedding_cloud(
                provider, api_key, backend.get("model", ""), ""
            )

    elif provider == "docling":
        url = (backend.get("url") or "").strip()
        if url:
            import httpx

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{url.rstrip('/')}/health",
                        timeout=DEFAULT_TIMEOUTS.get("http_health", 10),
                    )
                success = resp.status_code == 200
                if not success:
                    error = f"HTTP {resp.status_code}"
            except Exception as exc:
                error = str(exc)
        else:
            error = "Remote-DDU-URL ist erforderlich; lokales Docling ist in KIara deaktiviert"

    # Unbekannter Provider
    if not success and not error:
        error = f"Unbekannter Provider: {provider}"

    # extra_config bei erfolgreichem Test aktualisieren
    extra_config_update = None
    if success and provider == "ollama" and backend.get("url") and backend.get("model"):
        try:
            model_info = await _fetch_model_info(backend["url"], backend["model"])
        except Exception as exc:
            logger.warning("Ollama model-info nach Test nicht abrufbar: %s", exc)
            model_info = None
        if model_info:
            try:
                existing_extra = json.loads(backend.get("extra_config") or "{}")
            except (json.JSONDecodeError, TypeError):
                existing_extra = {}
            existing_extra.update(model_info)
            extra_config_update = json.dumps(existing_extra, ensure_ascii=False)

    # Cloud-Embedding: dimension in extra_config cachen
    if success and dimension and category == "embedding" and provider in ("openai", "voyage"):
        try:
            existing_extra = json.loads(backend.get("extra_config") or "{}")
        except (json.JSONDecodeError, TypeError):
            existing_extra = {}
        existing_extra["embedding_dim"] = dimension
        extra_config_update = json.dumps(existing_extra, ensure_ascii=False)

    # Testergebnis persistieren
    backend_repo.update_test_result(
        backend_id, ok=success, error=error, extra_config=extra_config_update
    )
    invalidate_cache(category)

    if success:
        result = {"success": True, "models": models}
        if dimension:
            result["dimension"] = dimension
        return result
    raise ValueError(error or "Verbindung fehlgeschlagen")


# --- Interne Helfer ---


async def _fetch_model_info(url: str, model: str) -> dict:
    """Ermittelt num_ctx und embedding_dim via Ollama /api/show."""
    import asyncio

    from kiara.retrieval.embedding import fetch_ollama_model_info

    return await asyncio.to_thread(fetch_ollama_model_info, url, model, 10.0)


def _invalidate_ddu():
    """DDU-Cache invalidieren (Import vermeidet zirkulaere Abhaengigkeit)."""
    try:
        from kiara.web.api_datasource_features import invalidate_ddu_cache

        invalidate_ddu_cache()
    except ImportError:
        pass


def _normalize_embedding_dim(extra: dict, model: str) -> dict:
    """Setzt embedding_dim fuer Cloud-Embedding-Backends kanonisch.

    - Bekannte Modelle: Dimension aus KNOWN_EMBEDDING_DIMENSIONS (ueberschreibt stale Werte)
    - Unbekannte Modelle: embedding_dim muss als positive Ganzzahl vorliegen
    """
    from kiara.backend_selector import normalize_model_name
    from kiara.shared.constants import KNOWN_EMBEDDING_DIMENSIONS

    extra = dict(extra)  # Kopie, original nicht mutieren
    norm = normalize_model_name(model)
    known_dim = KNOWN_EMBEDDING_DIMENSIONS.get(norm)

    if known_dim:
        extra["embedding_dim"] = known_dim
    else:
        dim = extra.get("embedding_dim")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"Embedding-Dimension unbekannt fuer Modell '{model}'. "
                "Bitte Modell zuerst testen oder embedding_dim angeben."
            )
    return extra


def _test_embedding_cloud(
    provider: str, api_key: str, model: str, base_url: str = ""
) -> tuple[bool, str, int | None]:
    """Fuehrt einen Probe-Embedding-Test fuer Cloud-Provider durch.

    Returns:
        (success, error, dimension)
    """
    if not model:
        return False, "Kein Modell angegeben", None

    try:
        if provider == "openai":
            try:
                import openai
            except ImportError:
                return False, "OpenAI-SDK nicht installiert (pip install openai)", None
            client = openai.OpenAI(api_key=api_key, base_url=base_url or None, timeout=10.0)
            response = client.embeddings.create(model=model, input=["test"])
            dim = len(response.data[0].embedding) if response.data else None
            return True, "", dim

        elif provider == "voyage":
            try:
                import voyageai
            except ImportError:
                return False, "Voyage-AI-SDK nicht installiert (pip install voyageai)", None
            client = voyageai.Client(api_key=api_key, timeout=10)
            from kiara.retrieval.embedding_roles import (
                SEARCH_DOCUMENT,
                voyage_input_type_for_role,
            )

            result = client.embed(
                ["test"],
                model=model,
                input_type=voyage_input_type_for_role(SEARCH_DOCUMENT),
            )
            dim = len(result.embeddings[0]) if result.embeddings else None
            return True, "", dim

        else:
            return False, f"Unbekannter Embedding-Provider: {provider}", None

    except Exception as exc:
        return False, f"{exc.__class__.__name__}: Probe-Embedding fehlgeschlagen", None
