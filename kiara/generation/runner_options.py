"""Feste Ollama-Runner-Optionen je LLM-Backend.

Kiron betreibt grosse Modelle mit einem gemessenen Ressourcenprofil
(Kontextfenster, Batch, Threads, GPU-Layer). Ollama laedt ein Modell neu,
sobald eine Anfrage andere Runner-Optionen schickt; Kiara waehlt ``num_ctx``
aber je Funktion und Anfrage unterschiedlich (Chat-Schieber, Agent, Rewriter,
dynamisches Kontextbudget). Ist in ``ki_backends.extra_config`` eines aktiven
Ollama-LLM-Backends ``runner_options`` gesetzt, gehen genau diese Werte bei
jeder Anfrage an dieses Backend mit und ersetzen die aufrufspezifischen.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

RUNNER_OPTION_KEYS = ("num_ctx", "num_batch", "num_thread", "num_gpu")


def fixed_runner_options(url: str, model: str) -> dict[str, int]:
    """Runner-Optionen des aktiven LLM-Backends mit dieser URL und diesem Modell."""
    try:
        from kiara.backend_selector import get_all_active

        backends = get_all_active("llm")
    except Exception as exc:
        logger.warning("Runner-Optionen nicht lesbar (%s): %s", model, exc)
        return {}
    wanted_url = (url or "").rstrip("/")
    for backend in backends:
        if backend.get("provider", "ollama") != "ollama":
            continue
        if str(backend.get("url") or "").rstrip("/") != wanted_url or backend.get("model") != model:
            continue
        extra = backend.get("extra_config") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except ValueError:
                return {}
        options = extra.get("runner_options") if isinstance(extra, dict) else None
        if not isinstance(options, dict):
            return {}
        return {
            key: int(options[key])
            for key in RUNNER_OPTION_KEYS
            if isinstance(options.get(key), int) and not isinstance(options.get(key), bool)
        }
    return {}
