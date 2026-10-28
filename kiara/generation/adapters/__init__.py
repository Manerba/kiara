"""Backend-agnostische Tool-Adapter fuer Agentic RAG.

Stellt ein einheitliches Interface (BaseToolAdapter) fuer verschiedene
LLM-Backends bereit. Die Factory-Funktion create_tool_adapter() waehlt
automatisch den passenden Adapter basierend auf dem Provider.

Module:
    base       — BaseToolAdapter (ABC), ToolStep, ToolAdapterError
    cloud_base — CloudAdapterBase (gemeinsame Cloud-Logik)
    ollama     — OllamaToolAdapter (prompt-basiertes TOOL_CALL:-Parsing)
    anthropic  — AnthropicToolAdapter (native tool_use API)
    openai     — OpenAIToolAdapter (native tool_calls API)

Beziehungen:
    - kiara.generation.agent_chain nutzt BaseToolAdapter + ToolStep
    - kiara.web.api_chat nutzt create_tool_adapter()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiara.anonymization.base import BaseAnonymizer

from kiara.generation.adapters.anthropic import AnthropicToolAdapter
from kiara.generation.adapters.base import (
    BaseToolAdapter,
    ToolAdapterError,
    ToolStep,
    _extract_balanced_params,
)
from kiara.generation.adapters.cloud_base import CloudAdapterBase
from kiara.generation.adapters.ollama import OllamaToolAdapter
from kiara.generation.adapters.openai import OpenAIToolAdapter


def create_tool_adapter(
    backend: dict, config: dict, anonymizer: BaseAnonymizer | None = None
) -> BaseToolAdapter:
    """Factory: Erstellt den passenden Adapter fuer den Provider.

    Args:
        backend: Backend-Dict mit provider, url, model, api_key etc.
        config: KIara-Konfiguration.
        anonymizer: BaseAnonymizer-Instanz (nur fuer Cloud-Backends).

    Returns:
        Passende BaseToolAdapter-Instanz.
    """
    provider = backend.get("provider", "ollama")
    if provider == "anthropic":
        return AnthropicToolAdapter(backend, config, anonymizer)
    elif provider == "openai":
        return OpenAIToolAdapter(backend, config, anonymizer)
    else:
        return OllamaToolAdapter(backend, config)
