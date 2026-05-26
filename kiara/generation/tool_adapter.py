"""Backward-Compat-Wrapper — alle Symbole kommen aus kiara.generation.adapters.

Dieses Modul existiert nur, damit bestehende Imports
``from kiara.generation.tool_adapter import X`` weiterhin funktionieren.
Neuer Code sollte direkt aus ``kiara.generation.adapters`` importieren.
"""

from kiara.generation.adapters import (  # noqa: F401 — re-export
    AnthropicToolAdapter,
    BaseToolAdapter,
    OllamaToolAdapter,
    OpenAIToolAdapter,
    ToolAdapterError,
    ToolStep,
    create_tool_adapter,
)
