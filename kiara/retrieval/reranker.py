"""Cross-Encoder Reranking fuer Retrieval-Ergebnisse (lokal oder remote via Kiron)."""

import json
import logging
import math
import threading
import time

import httpx

from kiara.shared.constants import (
    RERANK_BACKEND_DEFAULT,
    RERANK_BATCH_SIZE_DEFAULT,
    RERANK_FAIL_MODE_DEFAULT,
    RERANK_REMOTE_MAX_DOC_CHARS,
    RERANK_REMOTE_MAX_DOCUMENTS,
    RERANK_TIMEOUT_SECS_DEFAULT,
    RERANK_TOP_K_DEFAULT,
)
from kiara.shared.processing_warnings import report_processing_warning
from kiara.shared.tracing import get_tracer

logger = logging.getLogger(__name__)
security_log = logging.getLogger("kiara.security")

_tracer = get_tracer(__name__, pipeline="retrieval")


def _is_oom_error(exc: BaseException) -> bool:
    """Erkennt OOM-Fehler aus PyTorch (CUDA) und CPU/Python."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


class CrossEncoderReranker:
    """Rerankt Retrieval-Ergebnisse mit einem Cross-Encoder-Modell.

    Lazy-Loading: Modell wird erst beim ersten Aufruf geladen.
    Optional: Funktioniert auch ohne sentence-transformers
    (gibt dann Ergebnisse unveraendert zurueck).
    """

    def __init__(self, config: dict):
        self._enabled = config.get("retrieval", {}).get("reranking", False)
        self._fail_mode = (
            str(config.get("retrieval", {}).get("rerank_fail_mode", RERANK_FAIL_MODE_DEFAULT) or "")
            .strip()
            .lower()
            or RERANK_FAIL_MODE_DEFAULT
        )
        self._model_name = config.get("retrieval", {}).get(
            "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._top_k = config.get("retrieval", {}).get("rerank_top_k", 5)
        batch_size_cfg = config.get("retrieval", {}).get(
            "rerank_batch_size", RERANK_BATCH_SIZE_DEFAULT
        )
        try:
            batch_size = int(batch_size_cfg)
        except (TypeError, ValueError):
            batch_size = RERANK_BATCH_SIZE_DEFAULT
        self._batch_size = max(1, batch_size)
        self._cross_encoder = None
        self._load_lock = threading.Lock()  # Double-Checked Locking
        self._predict_lock = threading.Lock()  # PyTorch forward pass nicht thread-safe

        # Eager-Import-Probe: ohne sentence-transformers darf _enabled nicht True bleiben,
        # sonst aktiviert chain._retrieve_and_rerank Query-Expansion + rerank_fetch_multiplier
        # und truncatet das Ergebnis am Ende ungeranked — schlechter als kein Reranking.
        # Breite Exception-Klausel analog zum Lazy-Load-Pfad (siehe _load_model): ein
        # zerschossenes torch-Binary kann beim sentence_transformers-Import statt
        # ImportError z.B. RuntimeError, OSError oder AttributeError werfen — die alle
        # bedeuten "Reranker nicht funktionsfaehig", nicht "Konstruktor crashen lassen".
        if self._enabled:
            try:
                from sentence_transformers import CrossEncoder  # noqa: F401
            except Exception as exc:
                logger.error(
                    "Reranking aktiviert, aber CrossEncoder-Import fehlgeschlagen "
                    "(%s: %s) — Fallback auf disabled. Bei 'No module named' Installation "
                    "pruefen: pip install sentence-transformers",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                self._enabled = False
                self._report_failure("initialization_failed", exc)

    def _report_failure(self, reason: str, exc: BaseException | None = None) -> None:
        if self._fail_mode == "error":
            raise RerankBackendError(f"Lokales Reranking fehlgeschlagen ({reason})") from exc
        report_processing_warning("reranking", reason)

    def _load_model(self) -> None:
        """Lazy-Loading des Cross-Encoder-Modells."""
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self._model_name)
            logger.info("Cross-Encoder-Modell '%s' geladen", self._model_name)
        except ImportError as exc:
            logger.error(
                "sentence-transformers nicht installiert – Reranking aktiviert aber "
                "nicht funktionsfaehig! Installation: pip install sentence-transformers"
            )
            self._enabled = False
            self._report_failure("initialization_failed", exc)
        except Exception as exc:
            logger.error(
                "Cross-Encoder-Modell '%s' konnte nicht geladen werden – "
                "Reranking aktiviert aber nicht funktionsfaehig: %s",
                self._model_name,
                exc,
                exc_info=True,
            )
            self._enabled = False
            self._report_failure("initialization_failed", exc)

    def rerank(self, query: str, results: list[dict], top_k: int | None = None) -> list[dict]:
        """Rerankt Ergebnisse nach Relevanz.

        Input:  Ergebnisse vom Retriever [{text, metadata, score}, ...]
        Output: Top-K nach Cross-Encoder-Score, absteigend sortiert.

        Wenn nicht enabled oder keine Ergebnisse: direkt Top-K zurueckgeben.

        Args:
            query:   Die Suchanfrage.
            results: Ergebnisliste vom Retriever.
            top_k:   Anzahl der zurueckzugebenden Ergebnisse (ueberschreibt
                     den Config-Wert ``rerank_top_k``).

        Returns:
            Sortierte und gekuerzte Ergebnisliste.
        """
        with _tracer.start_as_current_span("rerank") as span:
            span.set_attribute("input_count", len(results))
            span.set_attribute("model", self._model_name)
            span.set_attribute("enabled", self._enabled)

            effective_k = top_k if top_k is not None else self._top_k

            if not self._enabled or not results or not query:
                final_results = results[:effective_k]
                span.set_attribute("output_count", len(final_results))
                return final_results

            if self._cross_encoder is None:
                with self._load_lock:
                    if self._cross_encoder is None:
                        self._load_model()
                        if not self._enabled:
                            final_results = results[:effective_k]
                            span.set_attribute("output_count", len(final_results))
                            return final_results

            # Results ohne text-Feld filtern — CrossEncoder bewertet leere
            # Strings mit ~0.0 Score und verfaelscht so das Ranking.
            results_with_text = [r for r in results if r.get("text")]
            results_no_text = [r for r in results if not r.get("text")]

            if results_no_text:
                logger.warning(
                    "Reranking: %d von %d Ergebnissen ohne text-Feld uebersprungen",
                    len(results_no_text),
                    len(results),
                )
                span.set_attribute("skipped_no_text", len(results_no_text))

            if not results_with_text:
                final_results = results[:effective_k]
                span.set_attribute("output_count", len(final_results))
                return final_results

            pairs = [(query, r["text"]) for r in results_with_text]

            scores = None
            current_batch = self._batch_size
            while True:
                try:
                    with self._predict_lock:
                        scores = self._cross_encoder.predict(pairs, batch_size=current_batch)
                    break
                except Exception as exc:
                    if _is_oom_error(exc) and current_batch > 1:
                        new_batch = max(1, current_batch // 2)
                        logger.warning(
                            "Cross-Encoder OOM bei batch_size=%d, retry mit %d",
                            current_batch,
                            new_batch,
                        )
                        span.set_attribute("rerank_oom_retry", True)
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            # CUDA-Cache-Reset ist Best-Effort; OOM-Retry laeuft weiter.
                            logger.debug(
                                "torch.cuda.empty_cache() nach OOM fehlgeschlagen",
                                exc_info=True,
                            )
                        current_batch = new_batch
                        continue
                    logger.error("Cross-Encoder-Predict fehlgeschlagen: %s", exc, exc_info=True)
                    span.set_attribute("rerank_fallback", True)
                    span.set_attribute("rerank_fallback_reason", "predict_error")
                    self._report_failure(
                        "out_of_memory" if _is_oom_error(exc) else "predict_error", exc
                    )
                    final_results = results[:effective_k]
                    span.set_attribute("output_count", len(final_results))
                    return final_results

            if len(scores) != len(results_with_text):
                logger.error(
                    "Cross-Encoder: %d Scores fuer %d Results",
                    len(scores),
                    len(results_with_text),
                )
                span.set_attribute("rerank_fallback", True)
                span.set_attribute("rerank_fallback_reason", "score_count_mismatch")
                self._report_failure("score_count_mismatch")
                final_results = results[:effective_k]
                span.set_attribute("output_count", len(final_results))
                return final_results

            try:
                rerank_scores = [float(scores[i]) for i in range(len(results_with_text))]
            except (TypeError, ValueError) as exc:
                logger.error(
                    "Cross-Encoder lieferte nicht-skalare Scores (%s) – "
                    "Fallback auf Originalreihenfolge: %s",
                    type(scores).__name__,
                    exc,
                )
                span.set_attribute("rerank_fallback", True)
                span.set_attribute("rerank_fallback_reason", "non_scalar_scores")
                self._report_failure("non_scalar_scores", exc)
                final_results = results[:effective_k]
                span.set_attribute("output_count", len(final_results))
                return final_results

            for i, result in enumerate(results_with_text):
                score = rerank_scores[i]
                result["rerank_score"] = score if math.isfinite(score) else float("-inf")

            results_sorted = sorted(
                results_with_text, key=lambda x: x.get("rerank_score", 0), reverse=True
            )

            final_results = results_sorted[:effective_k]
            span.set_attribute("output_count", len(final_results))
            return final_results

    @property
    def enabled(self) -> bool:
        """Ob Reranking aktiviert ist."""
        return self._enabled


class RerankBackendError(RuntimeError):
    """Remote-Reranking fehlgeschlagen (Transport-, HTTP- oder Antwortformatfehler).

    Wird nur im fail_mode "error" geworfen — Benchmarks erzwingen diesen Modus,
    damit eine Messung mit totem Reranker nie wie eine gueltige Variante aussieht.
    """


class KironReranker:
    """Rerankt Retrieval-Ergebnisse remote ueber den Kiron-Rerank-Endpoint.

    Gleiches Interface wie CrossEncoderReranker (``enabled``, ``rerank()``).
    Der Kiron-Service (kiron-deberta) haelt das Cross-Encoder-Modell auf der
    GPU und liefert pro Dokument einen Score. Sortiert und gekuerzt wird
    lokal, damit die Ergebnis-Semantik (``rerank_score`` an jedem gescorten
    Kandidaten, absteigend, Top-K) identisch zum lokalen Pfad bleibt.

    Fehlerverhalten (``retrieval.rerank_fail_mode``):
      - ``degrade``: RRF-Reihenfolge unveraendert zurueckgeben; Details in
        ``last_diagnostics`` und im Tracing-Span. Der Chat bleibt benutzbar.
      - ``error``: RerankBackendError propagieren (fail-closed, Messpfad).
    """

    def __init__(self, config: dict):
        retrieval_cfg = config.get("retrieval", {})
        self._enabled = retrieval_cfg.get("reranking", False)
        self._model_name = retrieval_cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._top_k = retrieval_cfg.get("rerank_top_k", RERANK_TOP_K_DEFAULT)
        self._fail_mode = (
            str(retrieval_cfg.get("rerank_fail_mode", RERANK_FAIL_MODE_DEFAULT) or "")
            .strip()
            .lower()
            or RERANK_FAIL_MODE_DEFAULT
        )
        try:
            self._timeout_s = float(
                retrieval_cfg.get("rerank_timeout_s", RERANK_TIMEOUT_SECS_DEFAULT)
            )
        except (TypeError, ValueError):
            self._timeout_s = float(RERANK_TIMEOUT_SECS_DEFAULT)
        self._base_url = str(retrieval_cfg.get("rerank_url", "") or "").strip().rstrip("/")
        self.last_diagnostics: dict | None = None

        # Analog zum Eager-Import-Check des lokalen Rerankers: Ohne Backend-URL
        # darf _enabled nicht True bleiben, sonst aktiviert chain._retrieve_and_rerank
        # Query-Expansion + rerank_fetch_multiplier und truncatet ungeranked.
        if self._enabled and not self._base_url:
            logger.error(
                "Reranking mit rerank_backend=kiron aktiviert, aber retrieval.rerank_url "
                "ist leer — Fallback auf disabled."
            )
            self._enabled = False
            if self._fail_mode == "error":
                raise RerankBackendError("Kiron-Reranking aktiviert, aber rerank_url fehlt")
            report_processing_warning("reranking", "missing_url")

    @property
    def enabled(self) -> bool:
        """Ob Reranking aktiviert ist."""
        return self._enabled

    def _set_diagnostics(self, status: str, **fields) -> dict:
        diag = {"backend": "kiron", "model": self._model_name, "status": status, **fields}
        self.last_diagnostics = diag
        return diag

    def _fail(
        self,
        span,
        results: list[dict],
        effective_k: int,
        reason: str,
        error: str,
        latency_ms: int,
        candidates_in: int,
        warning_reason: str | None = None,
        response: httpx.Response | None = None,
    ) -> list[dict]:
        """Behandelt Backend-Fehler gemaess fail_mode (degrade|error)."""
        diagnostics = self._set_diagnostics(
            "error" if self._fail_mode == "error" else "degraded",
            fallback_reason=reason,
            error=error,
            latency_ms=latency_ms,
            candidates_in=candidates_in,
        )
        span.set_attribute("rerank_fallback", True)
        span.set_attribute("rerank_fallback_reason", reason)
        details = {
            **diagnostics,
            "component": "reranking",
            "endpoint": f"{self._base_url}/api/rerank",
            "fail_mode": self._fail_mode,
        }
        log_error = error
        if response is not None:
            # Das System-Log behaelt die vollstaendige Backend-Antwort,
            # auch wenn die normale Diagnose nur einen Auszug enthaelt.
            details["http_status"] = response.status_code
            details["response_body"] = response.text
            if response.status_code != 200:
                log_error = response.text
            try:
                details["backend_response"] = response.json()
            except ValueError:
                pass
        security_log.log(
            logging.ERROR if self._fail_mode == "error" else logging.WARNING,
            "RERANK_BACKEND_FAILED reason=%s error=%s",
            reason,
            # Quoting haelt key=value aus Backend-Texten im error-Feld.
            json.dumps(log_error, ensure_ascii=False),
            extra={"kiara_details": details},
        )
        if self._fail_mode == "error":
            raise RerankBackendError(f"Kiron-Reranking fehlgeschlagen ({reason}): {error}")
        report_processing_warning("reranking", warning_reason or reason)
        logger.warning(
            "Kiron-Reranking fehlgeschlagen (%s): %s — Fallback auf Retrieval-Reihenfolge",
            reason,
            error,
        )
        final_results = results[:effective_k]
        span.set_attribute("output_count", len(final_results))
        return final_results

    def _post_rerank(self, query: str, documents: list[str]) -> httpx.Response:
        """POST an den Kiron-Rerank-Endpoint; 1 Retry nur bei Verbindungsaufbau-Fehlern."""
        payload = {"model": self._model_name, "query": query, "documents": documents}
        url = f"{self._base_url}/api/rerank"
        for attempt in (1, 2):
            try:
                return httpx.post(url, json=payload, timeout=self._timeout_s)
            except httpx.ConnectError:
                if attempt == 2:
                    raise
                logger.warning("Kiron-Rerank ConnectError, Retry 1/1 (%s)", url)
        raise RuntimeError("unreachable")  # pragma: no cover

    def rerank(self, query: str, results: list[dict], top_k: int | None = None) -> list[dict]:
        """Rerankt Ergebnisse via Kiron; Signatur und Semantik wie CrossEncoderReranker.rerank."""
        with _tracer.start_as_current_span("rerank") as span:
            span.set_attribute("input_count", len(results))
            span.set_attribute("model", self._model_name)
            span.set_attribute("enabled", self._enabled)
            span.set_attribute("backend", "kiron")

            effective_k = top_k if top_k is not None else self._top_k

            if not self._enabled or not results or not query:
                self._set_diagnostics("disabled" if not self._enabled else "skipped")
                final_results = results[:effective_k]
                span.set_attribute("output_count", len(final_results))
                return final_results

            # Semantik wie im lokalen Pfad: Ergebnisse ohne text-Feld werden
            # nicht gescored und fallen aus dem Ranking.
            results_with_text = [r for r in results if r.get("text")]
            results_no_text = [r for r in results if not r.get("text")]
            if results_no_text:
                logger.warning(
                    "Reranking: %d von %d Ergebnissen ohne text-Feld uebersprungen",
                    len(results_no_text),
                    len(results),
                )
                span.set_attribute("skipped_no_text", len(results_no_text))
            if not results_with_text:
                self._set_diagnostics("skipped", candidates_in=len(results))
                final_results = results[:effective_k]
                span.set_attribute("output_count", len(final_results))
                return final_results

            candidates = results_with_text
            if len(candidates) > RERANK_REMOTE_MAX_DOCUMENTS:
                logger.warning(
                    "Reranking: %d Kandidaten ueberschreiten das API-Limit %d — "
                    "nur die ersten %d (Retrieval-Reihenfolge) werden gescored",
                    len(candidates),
                    RERANK_REMOTE_MAX_DOCUMENTS,
                    RERANK_REMOTE_MAX_DOCUMENTS,
                )
                span.set_attribute("capped_to_api_limit", True)
                candidates = candidates[:RERANK_REMOTE_MAX_DOCUMENTS]

            truncated = 0
            documents = []
            for r in candidates:
                text = r["text"]
                if len(text) > RERANK_REMOTE_MAX_DOC_CHARS:
                    truncated += 1
                    text = text[:RERANK_REMOTE_MAX_DOC_CHARS]
                documents.append(text)
            if truncated:
                span.set_attribute("truncated_docs", truncated)

            started = time.monotonic()
            try:
                response = self._post_rerank(query, documents)
            except httpx.HTTPError as exc:
                return self._fail(
                    span,
                    results,
                    effective_k,
                    reason=type(exc).__name__,
                    error=str(exc),
                    latency_ms=int((time.monotonic() - started) * 1000),
                    candidates_in=len(candidates),
                    warning_reason="timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "connection_error",
                )
            latency_ms = int((time.monotonic() - started) * 1000)

            if response.status_code != 200:
                # Auch GPU-Gate-Antworten des Proxys (409/503) landen hier.
                body_excerpt = response.text[:300]
                # Nur ein strukturierter Backend-Code belegt GPU-OOM.
                # Ein HTTP-Status oder beliebiger Fehlertext genuegt nicht.
                warning_reason = f"http_{response.status_code}"
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = None
                if isinstance(error_payload, dict):
                    error = error_payload.get("error")
                    if isinstance(error, dict):
                        error = error.get("code") or error.get("type")
                    if error == "cuda_oom" or error_payload.get("code") == "cuda_oom":
                        warning_reason = "cuda_oom"
                return self._fail(
                    span,
                    results,
                    effective_k,
                    reason=f"http_{response.status_code}",
                    error=body_excerpt,
                    warning_reason=warning_reason,
                    latency_ms=latency_ms,
                    candidates_in=len(candidates),
                    response=response,
                )

            # Zweiphasig: erst vollstaendig validieren, dann anwenden — sonst
            # blieben bei einem Parse-Abbruch partielle rerank_scores an den
            # Dicts kleben, obwohl die Originalreihenfolge zurueckgeht.
            try:
                payload = response.json()
                items = payload["results"]
                if not isinstance(items, list):
                    raise TypeError(f"results ist {type(items).__name__}, erwartet list")
                parsed_scores: dict[int, float] = {}
                for item in items:
                    idx = item["index"]
                    score = float(item["score"])
                    if not isinstance(idx, int) or not 0 <= idx < len(candidates):
                        raise ValueError(f"index {idx!r} ausserhalb der Kandidatenliste")
                    parsed_scores[idx] = score
            except (ValueError, TypeError, KeyError) as exc:
                return self._fail(
                    span,
                    results,
                    effective_k,
                    reason="malformed_response",
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=latency_ms,
                    candidates_in=len(candidates),
                    response=response,
                )

            if len(parsed_scores) != len(candidates):
                # Der Service scored ohne top_k alle Dokumente; fehlende Scores
                # deuten auf einen Vertragsbruch — nicht still weiterranken.
                return self._fail(
                    span,
                    results,
                    effective_k,
                    reason="score_count_mismatch",
                    error=f"{len(parsed_scores)} Scores fuer {len(candidates)} Kandidaten",
                    latency_ms=latency_ms,
                    candidates_in=len(candidates),
                    response=response,
                )

            scored = 0
            for idx, score in parsed_scores.items():
                candidates[idx]["rerank_score"] = score if math.isfinite(score) else float("-inf")
                scored += 1

            results_sorted = sorted(
                candidates, key=lambda x: x.get("rerank_score", 0), reverse=True
            )
            final_results = results_sorted[:effective_k]
            self._set_diagnostics(
                "success",
                candidates_in=len(candidates),
                scored=scored,
                latency_ms=latency_ms,
            )
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("output_count", len(final_results))
            return final_results


def create_reranker(config: dict) -> "CrossEncoderReranker | KironReranker":
    """Erzeugt den konfigurierten Reranker (retrieval.rerank_backend: local|kiron).

    Zentrale Weiche fuer alle Aufrufer — Chain, Agent-Tools, Retrieval-only
    und Benchmarks erhalten dieselbe Backend-Auswahl.
    """
    backend = (
        str(config.get("retrieval", {}).get("rerank_backend", RERANK_BACKEND_DEFAULT) or "")
        .strip()
        .lower()
        or RERANK_BACKEND_DEFAULT
    )
    if backend == "kiron":
        return KironReranker(config)
    return CrossEncoderReranker(config)
