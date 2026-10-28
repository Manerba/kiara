"""Backend-Warmup — Modelle vor Verarbeitung laden.

Sendet Dummy-Requests an Ollama/DDU-Backends, um Modelle in den VRAM
zu laden, bevor die eigentliche Ingestion startet. Verhindert Timeouts
beim ersten produktiven Request nach Idle-Phase.

Alle Ollama-Requests setzen keep_alive, damit Modelle waehrend der
Sync-/Hashing-Phase nicht entladen werden.

Beziehungen:
    - kiara.backend_selector -> get_all_active(), normalize_model_name(), mark_healthy()
    - kiara.config           -> get_ddu_backends()
    - kiara.shared.constants -> WARMUP_* Konstanten
    - kiara.web.ingestion_worker -> ruft build_warmup_tasks() + BackendWarmer.warm() auf
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import ollama as ollama_client

from kiara.shared.constants import (
    WARMUP_DUMMY_TEXT,
    WARMUP_KEEP_ALIVE,
    WARMUP_RETRY_DELAY,
    WARMUP_TIMEOUT_DEFAULT,
    WARMUP_TIMEOUT_MULTIPLIER,
    WARMUP_TIMEOUT_OFFSET,
    WARMUP_TRANSIENT_RETRIES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class WarmupTask:
    """Ein einzelner Warmup-Auftrag."""

    category: str  # "embedding" | "llm" | "vlm" | "ddu"
    model: str  # Modell-Name (leer bei DDU)
    url: str  # Backend-URL
    backend_name: str  # Anzeigename fuer Logs/UI
    profile_id: str | None = None  # kanonischer KIron-Profilbezug, falls konfiguriert


@dataclass
class WarmupResult:
    success: bool
    duration_s: float
    details: list[dict]  # [{category, model, url, ok, error, duration_s}, ...]


# ---------------------------------------------------------------------------
# Fehlerklassifikation
# ---------------------------------------------------------------------------

# Ollama-Library wirft:
# - ollama.RequestError  -> Netzwerk/Connection-Fehler (wrapped httpx-Exceptions)
# - ollama.ResponseError -> API-Fehler (Modell nicht gefunden, Server-Fehler)
#
# Nur fuer DDU (direktes httpx) sind die rohen httpx-Exceptions relevant.

TRANSIENT_EXCEPTIONS = (
    ollama_client.RequestError,  # Netzwerk-Fehler (Connection refused, Timeout, Reset)
    httpx.TimeoutException,  # Direkte httpx-Timeouts (DDU/ColBERT)
    httpx.NetworkError,  # Direkte httpx-Connection/Read/Write/Close-Fehler
    httpx.ProtocolError,  # Direkte httpx-Protokollfehler
    httpx.ConnectTimeout,  # Nur fuer DDU-Warmup (direkter httpx-Call)
    httpx.ReadTimeout,  # Nur fuer DDU-Warmup
    ConnectionResetError,  # Fallback
    ConnectionRefusedError,  # Fallback
)

_PERMANENT_ERROR_PATTERNS = (
    "not found",
    "invalid model",
    "unknown model",
    "file does not exist",
    "out of memory",
    "model not found",
    "no such model",
)
_TRANSIENT_HTTP_STATUS_CODES = frozenset({409, 429})


def _normalize_error_text(value: object) -> str:
    """Normalisiert Backend-Fehlertexte fuer robuste Pattern-Erkennung."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _has_permanent_error_text(value: object) -> bool:
    msg = _normalize_error_text(value)
    if any(pattern in msg for pattern in _PERMANENT_ERROR_PATTERNS):
        return True
    return bool(re.search(r"\boom\b", msg))


def _is_permanent_ollama_error(exc: Exception) -> bool:
    """Prueft ob ein Ollama-Fehler permanent ist (kein Retry sinnvoll)."""
    if isinstance(exc, ollama_client.ResponseError):
        return _has_permanent_error_text(exc)
    return False


def _is_transient_http_error(exc: Exception) -> bool:
    """Prueft ob ein httpx.HTTPStatusError transient ist.

    Transient sind Warmup-Contestion/Rate-Limits (409/429) und HTTP 5xx,
    ausser der Body enthaelt permanente Modell-/OOM-Fehler.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        if _has_permanent_error_text(getattr(exc.response, "text", "")):
            return False
        return (
            exc.response.status_code in _TRANSIENT_HTTP_STATUS_CODES
            or exc.response.status_code >= 500
        )
    return False


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_warmup_client(url: str, timeout: float) -> ollama_client.Client:
    """Erstellt Ollama-Client mit explizitem Timeout fuer Warmup.

    Eigener Client statt gecachtem aus embedding.py:
    1. Explizite Timeout-Kontrolle (verhindert haengende Threads)
    2. Warmup-Timeout unabhaengig vom Produktions-Client
    """
    return ollama_client.Client(host=url, timeout=timeout)


_warmup_tiny_png: bytes | None = None
_warmup_tiny_png_lock = threading.Lock()


def _get_tiny_png() -> bytes:
    """Lazy-generiertes 1x1 weisses PNG. PIL-Import nur bei VLM-Warmup."""
    global _warmup_tiny_png
    if _warmup_tiny_png is not None:
        return _warmup_tiny_png
    with _warmup_tiny_png_lock:
        if _warmup_tiny_png is None:
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (1, 1), (255, 255, 255)).save(buf, format="PNG")
            _warmup_tiny_png = buf.getvalue()
    return _warmup_tiny_png


def _mark_healthy_safe(url: str) -> None:
    """Wrapper fuer mark_healthy() mit Fehler-Toleranz. DDU wird uebersprungen."""
    try:
        from kiara.backend_selector import mark_healthy

        mark_healthy(url)
    except Exception:
        logger.debug("Health-Cache-Update fuer %s fehlgeschlagen (nicht kritisch)", url)


# ---------------------------------------------------------------------------
# BackendWarmer
# ---------------------------------------------------------------------------


class BackendWarmer:
    DEFAULT_TIMEOUT = WARMUP_TIMEOUT_DEFAULT
    TRANSIENT_RETRIES = WARMUP_TRANSIENT_RETRIES

    # -- Warmup-Strategien pro Kategorie --

    def _warm_embedding(self, task, timeout):
        from kiara.retrieval.embedding_contract import ProfileKind
        from kiara.retrieval.embedding_roles import (
            SEARCH_DOCUMENT,
            emit_embedding_role_telemetry,
            resolve_server_request_profile,
        )

        profile = resolve_server_request_profile(
            task.url,
            task.model,
            role=SEARCH_DOCUMENT,
            kind=ProfileKind.DENSE,
            endpoint="/api/embed",
            profile_id=task.profile_id,
        )
        emit_embedding_role_telemetry(
            profile,
            operation="warmup_document_embedding",
            input_count=1,
        )
        response = httpx.post(
            f"{task.url.rstrip('/')}/api/embed",
            json={
                "model": task.model,
                "input": [WARMUP_DUMMY_TEXT],
                "input_type": SEARCH_DOCUMENT,
                "keep_alive": WARMUP_KEEP_ALIVE,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise TypeError("Embedding-Warmup lieferte kein einzelnes Embedding")

    def _warm_llm(self, task, timeout):
        client = _make_warmup_client(task.url, timeout)
        client.generate(
            model=task.model,
            prompt="Hi",
            stream=False,
            options={"num_predict": 1},
            keep_alive=WARMUP_KEEP_ALIVE,
        )

    def _warm_vlm(self, task, timeout):
        client = _make_warmup_client(task.url, timeout)
        client.chat(
            model=task.model,
            stream=False,
            keep_alive=WARMUP_KEEP_ALIVE,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in one word.",
                    "images": [_get_tiny_png()],
                }
            ],
        )

    def _warm_ddu(self, task, timeout):
        # Kleines PNG mit OCR senden, damit Docling seine OCR-/Vision-Modelle
        # in den VRAM laedt. Eine Textdatei ohne OCR wuerde die Modelle nicht
        # laden und der erste echte PDF-Request koennte in einen Timeout laufen.
        dummy_png = io.BytesIO(_get_tiny_png())
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            resp = client.post(
                f"{task.url}/v1/convert/file",
                files=[("files", ("warmup.png", dummy_png, "image/png"))],
                data={
                    "to_formats": "md",
                    "do_ocr": "true",
                    "table_mode": "fast",
                    "ocr_lang": ["en"],
                },
            )
            resp.raise_for_status()

    # -- Retry-Logik --

    def _attempt(self, fn, task, timeout, retries=TRANSIENT_RETRIES):
        """Fuehrt fn aus, mit optionalem Retry bei transienten Fehlern."""
        last_exc = None
        for attempt in range(1 + retries):
            try:
                return fn(task, timeout)
            except ollama_client.ResponseError as exc:
                if _is_permanent_ollama_error(exc):
                    raise
                last_exc = exc
                if attempt < retries:
                    logger.warning(
                        "Warmup %s transient error (ResponseError), retry in %ds: %s",
                        task.backend_name,
                        WARMUP_RETRY_DELAY,
                        exc,
                    )
                    time.sleep(WARMUP_RETRY_DELAY)
            except TRANSIENT_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < retries:
                    logger.warning(
                        "Warmup %s transient error, retry in %ds: %s",
                        task.backend_name,
                        WARMUP_RETRY_DELAY,
                        exc,
                    )
                    time.sleep(WARMUP_RETRY_DELAY)
            except httpx.HTTPStatusError as exc:
                if not _is_transient_http_error(exc):
                    raise
                last_exc = exc
                if attempt < retries:
                    logger.warning(
                        "Warmup %s transient HTTP %d, retry in %ds",
                        task.backend_name,
                        exc.response.status_code,
                        WARMUP_RETRY_DELAY,
                    )
                    time.sleep(WARMUP_RETRY_DELAY)
            except Exception:
                raise  # Unbekannte Fehler sofort weiterleiten
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"Warmup {task.backend_name}: alle {1 + retries} Versuche fehlgeschlagen"
        )

    def _timed_attempt(self, fn, task, timeout, retries=TRANSIENT_RETRIES):
        """Wrapper um _attempt(). Misst Dauer, haengt sie bei Fehler an die Exception."""
        t_start = time.monotonic()
        try:
            self._attempt(fn, task, timeout, retries)
            return time.monotonic() - t_start
        except Exception as exc:
            exc._warmup_duration = time.monotonic() - t_start
            raise

    # -- Warmup-Strategie-Dispatch --

    _CATEGORY_FN_MAP = {
        "embedding": "_warm_embedding",
        "llm": "_warm_llm",
        "vlm": "_warm_vlm",
        "ddu": "_warm_ddu",
        "colbert": "_warm_colbert",
    }

    def _warm_colbert(self, task: "WarmupTask", timeout: float = 60.0) -> None:
        """Waermt den ColBERT-Endpoint auf KIron auf."""
        from kiara.retrieval.embedding_contract import ProfileKind
        from kiara.retrieval.embedding_roles import (
            SEARCH_DOCUMENT,
            emit_embedding_role_telemetry,
            resolve_server_request_profile,
        )

        profile = resolve_server_request_profile(
            task.url,
            task.model,
            role=SEARCH_DOCUMENT,
            kind=ProfileKind.MULTI_VECTOR,
            endpoint="/api/embed_colbert",
            profile_id=task.profile_id,
        )
        emit_embedding_role_telemetry(
            profile,
            operation="colbert_warmup_document_embedding",
            input_count=1,
        )
        resp = httpx.post(
            f"{task.url.rstrip('/')}/api/embed_colbert",
            json={
                "model": task.model,
                "input": ["warmup"],
                "input_type": SEARCH_DOCUMENT,
            },
            timeout=timeout,
        )
        resp.raise_for_status()

    # -- Parallele Ausfuehrung --

    # Poll-Intervall fuer check_control-Abfragen waehrend des Warmups
    _CONTROL_POLL_INTERVAL = 2.0

    def _execute_warmup_tasks(
        self,
        tasks: list[WarmupTask],
        timeout: float,
        on_progress: Callable | None,
        t0: float,
        check_control: Callable[[], str | None] | None = None,
    ) -> list[dict]:
        """Fuehrt alle Warmup-Tasks parallel im ThreadPool aus.

        Args:
            tasks: Liste der Warmup-Auftraege.
            timeout: Timeout in Sekunden pro einzelnem Backend-Request.
            on_progress: Optional — Callback(task, status_str) fuer Live-Updates.
            t0: Startzeitpunkt (monotonic) fuer Fallback-Dauermessung.
            check_control: Optional — Callback() -> "cancel"/"shutdown"/None
                fuer reaktionsschnelle Abbrueche. Wird alle _CONTROL_POLL_INTERVAL
                Sekunden ausgewertet.

        Returns:
            Liste von Detail-Dicts mit Ergebnis pro Task. Bei Abbruch
            enthalten alle noch laufenden Tasks ein "abgebrochen"-Detail.
        """
        details: list[dict] = []

        # Worst-Case pro Task: 2 * timeout + Offset (s.o.)
        overall_timeout = WARMUP_TIMEOUT_MULTIPLIER * timeout + WARMUP_TIMEOUT_OFFSET
        deadline = time.monotonic() + overall_timeout

        # Kein Context-Manager: shutdown(wait=False, cancel_futures=True) bei Cancel
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(tasks)),
            thread_name_prefix="warmup",
        )
        try:
            futures = {}
            for task in tasks:
                fn = getattr(self, self._CATEGORY_FN_MAP[task.category])
                futures[pool.submit(self._timed_attempt, fn, task, timeout)] = task

            pending = set(futures.keys())
            cancelled = False

            while pending:
                # Cancel/Shutdown-Check
                if check_control is not None:
                    ctrl = check_control()
                    if ctrl in ("cancel", "shutdown"):
                        logger.info("Warmup abgebrochen via check_control=%s", ctrl)
                        for f in pending:
                            f.cancel()
                        cancelled = True
                        break

                # Verbleibende Zeit gegen den Gesamt-Timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._handle_overall_timeout(
                        futures,
                        details,
                        overall_timeout,
                        on_progress,
                        t0,
                    )
                    return details

                done_set, pending = concurrent.futures.wait(
                    pending,
                    timeout=min(self._CONTROL_POLL_INTERVAL, remaining),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done_set:
                    task = futures[future]
                    self._collect_future_result(
                        future,
                        task,
                        details,
                        on_progress,
                        t0,
                    )

            if cancelled:
                self._handle_cancel(futures, details, on_progress, t0)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        return details

    def _handle_cancel(
        self,
        futures: dict[concurrent.futures.Future, WarmupTask],
        details: list[dict],
        on_progress: Callable | None,
        t0: float,
    ) -> None:
        """Ergaenzt Cancel-Details fuer alle Tasks ohne Ergebnis."""
        seen = {(d["category"], d["model"], d["url"]) for d in details}
        for _future, task in futures.items():
            key = (task.category, task.model, task.url)
            if key in seen:
                continue
            details.append(
                {
                    "category": task.category,
                    "model": task.model,
                    "url": task.url,
                    "ok": False,
                    "error": "Warmup abgebrochen",
                    "duration_s": round(time.monotonic() - t0, 1),
                }
            )
            if on_progress:
                on_progress(task, "cancelled")

    def _collect_future_result(
        self,
        future: concurrent.futures.Future,
        task: WarmupTask,
        details: list[dict],
        on_progress: Callable | None,
        t0: float,
    ) -> None:
        """Sammelt das Ergebnis eines einzelnen Warmup-Futures ein.

        Haengt ein Detail-Dict an ``details`` an — sowohl bei Erfolg als auch
        bei Fehler. Aktualisiert den Health-Cache bei Ollama-Backends und
        ruft den optionalen Progress-Callback auf.

        Args:
            future: Der abgeschlossene Future.
            task: Der zugehoerige WarmupTask.
            details: Ziel-Liste fuer das Detail-Dict (wird in-place ergaenzt).
            on_progress: Optionaler Callback(task, status_str).
            t0: Startzeitpunkt (monotonic) fuer Fallback-Dauermessung.
        """
        try:
            task_duration = future.result()
            details.append(
                {
                    "category": task.category,
                    "model": task.model,
                    "url": task.url,
                    "ok": True,
                    "error": None,
                    "duration_s": round(task_duration, 1),
                }
            )
            if task.category in ("embedding", "llm", "vlm"):
                _mark_healthy_safe(task.url)
            logger.info(
                "Warmup %s(%s): OK (%.1fs)", task.category, task.model or task.url, task_duration
            )
            if on_progress:
                on_progress(task, "ok")
        except Exception as exc:
            dur = getattr(exc, "_warmup_duration", time.monotonic() - t0)
            details.append(
                {
                    "category": task.category,
                    "model": task.model,
                    "url": task.url,
                    "ok": False,
                    "error": str(exc),
                    "duration_s": round(dur, 1),
                }
            )
            logger.warning(
                "Warmup %s(%s): FEHLER (%.1fs): %s", task.category, task.model or task.url, dur, exc
            )
            if on_progress:
                on_progress(task, f"error: {exc}")

    def _handle_overall_timeout(
        self,
        futures: dict[concurrent.futures.Future, WarmupTask],
        details: list[dict],
        overall_timeout: float,
        on_progress: Callable | None,
        t0: float,
    ) -> None:
        """Behandelt den Gesamt-Timeout des ThreadPools.

        Fuer alle Tasks, die noch kein Ergebnis in ``details`` haben,
        wird ein Fehler-Detail-Dict mit Timeout-Meldung angehaengt.

        Args:
            futures: Mapping Future -> WarmupTask.
            details: Bisherige Ergebnis-Details (wird in-place ergaenzt).
            overall_timeout: Der ueberschrittene Gesamt-Timeout in Sekunden.
            on_progress: Optionaler Callback(task, status_str).
            t0: Startzeitpunkt (monotonic) fuer Dauermessung.
        """
        # Noch nicht gestartete Futures canceln
        for future in futures:
            future.cancel()

        seen = {(d["category"], d["model"], d["url"]) for d in details}
        for _future, task in futures.items():
            key = (task.category, task.model, task.url)
            if key not in seen:
                details.append(
                    {
                        "category": task.category,
                        "model": task.model,
                        "url": task.url,
                        "ok": False,
                        "error": f"Gesamt-Timeout nach {overall_timeout}s",
                        "duration_s": round(time.monotonic() - t0, 1),
                    }
                )
                if on_progress:
                    on_progress(task, f"error: Gesamt-Timeout nach {overall_timeout}s")

    # -- Ergebnis-Aggregation --

    def _aggregate_warmup_results(self, details: list[dict]) -> bool:
        """Prueft ob der Warmup insgesamt erfolgreich war.

        Erfolg bedeutet: Fuer jede beteiligte Kategorie existiert mindestens
        ein erfolgreiches Ergebnis. Bei Gesamt-Erfolg werden fehlgeschlagene
        Einzelbackends als nicht-kritische Warnings geloggt.

        Args:
            details: Liste der Detail-Dicts aus _execute_warmup_tasks().

        Returns:
            True wenn alle Kategorien mindestens ein OK haben.
        """
        required_categories = {d["category"] for d in details}
        ok_categories = {d["category"] for d in details if d["ok"]}
        all_ok = required_categories == ok_categories

        # Warnings fuer fehlgeschlagene Einzelbackends (bei Gesamt-Erfolg)
        if all_ok:
            failed = [d for d in details if not d["ok"]]
            for d in failed:
                logger.warning(
                    "Warmup %s(%s) fehlgeschlagen (nicht kritisch, andere Backends OK): %s",
                    d["category"],
                    d.get("model") or d["url"],
                    d["error"],
                )

        return all_ok

    # -- Hauptmethode --

    def warm(
        self,
        tasks: list[WarmupTask],
        timeout: float = DEFAULT_TIMEOUT,
        on_progress: Callable | None = None,
        check_control: Callable[[], str | None] | None = None,
    ) -> WarmupResult:
        """Waermt alle Backends parallel vor.

        Args:
            tasks: Liste der Warmup-Auftraege.
            timeout: Timeout in Sekunden pro einzelnem Backend-Request.
            on_progress: Optional — Callback(task, status_str) fuer Live-Updates.
                Aktuell nicht vom Worker genutzt; vorbereitet fuer kuenftige
                Admin-UI-Integration (siehe TODO_ADMIN_UI_WARMUP_LIVE_STATUS).
            check_control: Optional — Callback() -> "cancel"/"shutdown"/None
                fuer reaktionsschnelle Abbrueche waehrend des Warmups. Wird
                alle 2 Sekunden ausgewertet.

        Returns:
            WarmupResult mit Gesamt-Erfolg (mindestens 1 OK pro Kategorie)
            und Detail-Infos. Bei Cancel ist ``success=False``.
        """
        if not tasks:
            # Leere Tasks-Liste ist legitim: alle aktiven Backends sind
            # Cloud-Provider (openai/voyage/anthropic) und brauchen kein
            # Vorab-Warmup. Pipeline darf weiterlaufen.
            logger.info("Warmup: keine Tasks zu warmen — alle aktiven Backends sind Cloud-Provider")
            return WarmupResult(success=True, duration_s=0.0, details=[])

        t0 = time.monotonic()
        logger.info(
            "Warmup gestartet: %d Tasks (%s)",
            len(tasks),
            ", ".join(sorted({t.category for t in tasks})),
        )

        details = self._execute_warmup_tasks(
            tasks,
            timeout,
            on_progress,
            t0,
            check_control=check_control,
        )
        all_ok = self._aggregate_warmup_results(details)

        total_dur = time.monotonic() - t0
        return WarmupResult(success=all_ok, duration_s=round(total_dur, 1), details=details)


# ---------------------------------------------------------------------------
# Warmup-Tasks dynamisch aus Job-Kontext bauen
# ---------------------------------------------------------------------------


def build_warmup_tasks(ds_row, embedding_model, config) -> list[WarmupTask]:
    """Erstellt Warmup-Tasks basierend auf Datasource-Settings und Config.

    Args:
        ds_row: dict — DB-Row mit Bracket-Zugriff.
        embedding_model: Modellname fuer Embedding (aus ingestion_jobs).
        config: Geladene kiara.yaml-Config.

    Raises:
        RuntimeError: Wenn kein Embedding-Modell angegeben oder kein
            Embedding-Backend fuer das Modell konfiguriert ist.

    Hinweis: Cloud-Backends (Anthropic, OpenAI) brauchen kein Warmup — Embedding,
    VLM und DDU sind Ollama-/Docling-only. LLM bekommt einen expliziten
    provider=="ollama"-Guard.
    """
    tasks = []

    if not embedding_model:
        raise RuntimeError("Kein Embedding-Modell fuer Ingestion-Job angegeben")

    # 1. Embedding — immer (jeder Ingestion-Job braucht Embeddings)
    #
    # get_all_active() statt get_embedding_backend_for_model() um den
    # Health-Cache zu umgehen (Henne-Ei-Problem: unhealthy-Cache blockierte
    # Warmup-Versuch). Der Warmup-Request IST der Health-Check.
    from kiara.backend_selector import get_all_active, normalize_model_name

    norm_model = normalize_model_name(embedding_model)
    emb_backends = [
        b for b in get_all_active("embedding") if normalize_model_name(b["model"]) == norm_model
    ]

    if not emb_backends:
        raise RuntimeError(
            f"Kein Embedding-Backend fuer Modell '{embedding_model}' konfiguriert "
            f"(ki_backends-Tabelle). Pruefen Sie die Backend-Konfiguration in der "
            f"Admin-UI oder fuehren Sie die DB-Migration erneut aus."
        )

    # Cloud-Embedding-Backends (openai/voyage) sind reaktive APIs ohne
    # lokales Modell-Loading — kein Warmup notwendig. Analog zur LLM-Logik
    # unten werden nur Ollama-Provider als WarmupTask erstellt; bei reinen
    # Cloud-Setups bleibt die Embedding-Liste leer.
    ollama_emb = [b for b in emb_backends if b.get("provider", "ollama") == "ollama"]
    if ollama_emb:
        for emb in ollama_emb:
            from kiara.retrieval.embedding_discovery import (
                select_configured_embedding_profile_id,
            )

            profile_id, profile_issues = select_configured_embedding_profile_id(emb)
            if profile_issues:
                raise RuntimeError(
                    "Embedding-Warmup wegen ungueltigem Profilbezug blockiert: "
                    f"{profile_issues[0].reason_code}"
                )
            tasks.append(
                WarmupTask(
                    category="embedding",
                    model=emb["model"],
                    url=emb["url"],
                    backend_name=emb.get("name", embedding_model),
                    profile_id=profile_id,
                )
            )
    else:
        logger.debug(
            "Embedding-Warmup uebersprungen fuer Modell '%s' — nur Cloud-Backends konfiguriert",
            embedding_model,
        )

    # 2. DDU — nur wenn fuer Datasource aktiviert
    #
    # Gleiche Fallback-Logik wie build_datasource_overrides() (datasource_runtime.py).
    # Spalte ds_ddu_enabled existiert erst nach DB-Migration Phase 9.
    try:
        ddu_enabled = ds_row["ds_ddu_enabled"]
    except (KeyError, IndexError):
        ddu_enabled = 0
    if ddu_enabled:
        from kiara.config import get_ddu_backends

        for ddu in get_ddu_backends(config):
            if ddu["url"]:
                tasks.append(
                    WarmupTask(
                        category="ddu",
                        model="",
                        url=ddu["url"],
                        backend_name=f"DDU ({ddu['url']})",
                    )
                )

    # 3. VLM — nur wenn fuer Datasource aktiviert
    #    Alle aktiven VLM-Backends waermen (Failover-Abdeckung).
    #
    # Gleiche Fallback-Logik wie build_datasource_overrides() (datasource_runtime.py).
    # Spalte ds_vlm_enabled existiert erst nach DB-Migration Phase 10.
    try:
        vlm_enabled = ds_row["ds_vlm_enabled"]
    except (KeyError, IndexError):
        vlm_enabled = 0
    if vlm_enabled:
        vlm_backends = get_all_active("vlm")
        for vlm in vlm_backends:
            tasks.append(
                WarmupTask(
                    category="vlm",
                    model=vlm["model"],
                    url=vlm["url"],
                    backend_name=f"VLM ({vlm.get('name', vlm['model'])})",
                )
            )

    # 4. LLM — nur wenn GraphRAG Entity-Extraction aktiv
    #
    #    get_all_active("llm") statt get_llm_backend(config), um den
    #    RR-Counter nicht zu inkrementieren. Nur provider=="ollama"
    #    (Cloud-Backends brauchen kein Warmup).
    graph_cfg = config.get("graph", {})
    if graph_cfg.get("enabled", False) and graph_cfg.get("extract_on_ingestion", True):
        extraction_model = graph_cfg.get("extraction_model", "")
        all_llms = get_all_active("llm")
        ollama_llms = [b for b in all_llms if b.get("provider") == "ollama"]
        if ollama_llms:
            for llm in ollama_llms:
                model = extraction_model or llm["model"]
                tasks.append(
                    WarmupTask(
                        category="llm",
                        model=model,
                        url=llm["url"],
                        backend_name=f"LLM ({model})",
                    )
                )
        elif all_llms:
            logger.debug("LLM-Warmup uebersprungen — nur Cloud-Backends konfiguriert")
        else:
            logger.warning("Kein LLM-Backend konfiguriert — LLM-Warmup uebersprungen")

    # 5. ColBERT — nur wenn global + per-DS aktiviert
    colbert_cfg = config.get("retrieval", {}).get("colbert", {})
    colbert_global = bool(colbert_cfg.get("enabled", False))
    try:
        ds_colbert = ds_row["ds_colbert_enabled"]
    except (KeyError, IndexError):
        ds_colbert = 0
    if colbert_global and ds_colbert:
        from kiara.retrieval.colbert import _resolve_colbert_url
        from kiara.shared.constants import COLBERT_DEFAULT_MODEL

        colbert_model = normalize_model_name(colbert_cfg.get("model", COLBERT_DEFAULT_MODEL))
        try:
            colbert_url = _resolve_colbert_url(config)
            tasks.append(
                WarmupTask(
                    category="colbert",
                    model=colbert_model,
                    url=colbert_url,
                    backend_name=f"ColBERT ({colbert_model})",
                )
            )
        except Exception as exc:
            logger.warning("ColBERT-Warmup-Task konnte nicht erstellt werden: %s", exc)

    return tasks
