"""NLI-Grounding: prueft, ob eine generierte Antwort vom Kontext gedeckt ist.

Nutzt den Kiron-Cross-Encoder-Service (/api/score) mit einem multilingualen
NLI-Modell: Pro Kontext-Chunk wird das Paar (Chunk-Text, Antwort) klassifiziert;
das Maximum der Entailment-Wahrscheinlichkeit ueber alle Chunks ist der
Grounding-Score. Eine Antwort gilt als gedeckt, wenn mindestens ein Chunk sie
entailt — genau die Frage, die der Reranker-Relevanz-Score strukturell nicht
beantworten kann (PDCA-Befund 2026-07-07: Relevanz != Answerability).
"""

import logging
import time

import httpx

from kiara.shared.constants import (
    GROUNDING_HYPOTHESIS_MAX_CHARS,
    GROUNDING_MAX_CHUNKS_DEFAULT,
    GROUNDING_MIN_ENTAILMENT_DEFAULT,
    GROUNDING_MODEL_DEFAULT,
    GROUNDING_PREMISE_MAX_CHARS,
    GROUNDING_TIMEOUT_SECS_DEFAULT,
)

logger = logging.getLogger(__name__)


def resolve_grounding_config(config: dict) -> dict:
    """Effektive Grounding-Einstellungen; grounding_url faellt auf rerank_url zurueck."""
    retrieval_cfg = config.get("retrieval", {})
    url = str(retrieval_cfg.get("grounding_url") or "").strip().rstrip("/")
    if not url:
        url = str(retrieval_cfg.get("rerank_url") or "").strip().rstrip("/")
    try:
        threshold = float(
            retrieval_cfg.get("grounding_min_entailment", GROUNDING_MIN_ENTAILMENT_DEFAULT)
        )
    except (TypeError, ValueError):
        threshold = GROUNDING_MIN_ENTAILMENT_DEFAULT
    try:
        timeout_s = float(retrieval_cfg.get("grounding_timeout_s", GROUNDING_TIMEOUT_SECS_DEFAULT))
    except (TypeError, ValueError):
        timeout_s = float(GROUNDING_TIMEOUT_SECS_DEFAULT)
    return {
        "enabled": bool(retrieval_cfg.get("grounding_retry", False)),
        "url": url,
        "model": str(retrieval_cfg.get("grounding_model") or GROUNDING_MODEL_DEFAULT),
        "threshold": threshold,
        "timeout_s": timeout_s,
    }


def check_entailment(config: dict, answer: str, chunks: list[dict]) -> dict:
    """Prueft die Antwort gegen die Kontext-Chunks via Kiron-NLI.

    Returns:
        {"status": "success", "max_entailment": float, "per_chunk": [...]}
        oder {"status": "error"|"skipped", "error": str}.
        Fail-closed-Nutzung obliegt dem Aufrufer: Bei error/skipped darf eine
        Retry-Antwort NICHT als gedeckt behandelt werden.
    """
    settings = resolve_grounding_config(config)
    if not settings["url"]:
        return {"status": "skipped", "error": "grounding_url/rerank_url nicht konfiguriert"}

    hypothesis = (answer or "").strip()[:GROUNDING_HYPOTHESIS_MAX_CHARS]
    premises = [
        str(c.get("text") or "")[:GROUNDING_PREMISE_MAX_CHARS]
        for c in chunks[:GROUNDING_MAX_CHUNKS_DEFAULT]
        if str(c.get("text") or "").strip()
    ]
    if not hypothesis or not premises:
        return {"status": "skipped", "error": "keine Hypothese oder keine Praemissen"}

    payload = {
        "model": settings["model"],
        "pairs": [[premise, hypothesis] for premise in premises],
    }
    started = time.monotonic()
    try:
        response = httpx.post(
            f"{settings['url']}/api/score", json=payload, timeout=settings["timeout_s"]
        )
    except httpx.HTTPError as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    latency_ms = int((time.monotonic() - started) * 1000)
    if response.status_code != 200:
        return {
            "status": "error",
            "error": f"HTTP {response.status_code}: {response.text[:200]}",
        }
    try:
        results = response.json()["results"]
        per_chunk = [float(item["entailment"]) for item in results]
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "error", "error": f"malformed_response {type(exc).__name__}: {exc}"}
    if len(per_chunk) != len(premises):
        return {
            "status": "error",
            "error": f"{len(per_chunk)} Scores fuer {len(premises)} Praemissen",
        }
    return {
        "status": "success",
        "max_entailment": max(per_chunk),
        "per_chunk": [round(v, 4) for v in per_chunk],
        "threshold": settings["threshold"],
        "model": settings["model"],
        "latency_ms": latency_ms,
    }
