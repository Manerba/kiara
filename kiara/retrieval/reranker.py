"""Cross-Encoder Reranking fuer Retrieval-Ergebnisse."""

import logging
import math
import threading

from kiara.shared.constants import RERANK_BATCH_SIZE_DEFAULT
from kiara.shared.tracing import get_tracer

logger = logging.getLogger(__name__)

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

    def _load_model(self) -> None:
        """Lazy-Loading des Cross-Encoder-Modells."""
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self._model_name)
            logger.info("Cross-Encoder-Modell '%s' geladen", self._model_name)
        except ImportError:
            logger.error(
                "sentence-transformers nicht installiert – Reranking aktiviert aber "
                "nicht funktionsfaehig! Installation: pip install sentence-transformers"
            )
            self._enabled = False
        except Exception as exc:
            logger.error(
                "Cross-Encoder-Modell '%s' konnte nicht geladen werden – "
                "Reranking aktiviert aber nicht funktionsfaehig: %s",
                self._model_name,
                exc,
                exc_info=True,
            )
            self._enabled = False

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
