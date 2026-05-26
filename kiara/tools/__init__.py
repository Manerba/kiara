"""Tools: Registry und Basisklassen fuer erweiterbare LLM-Werkzeuge.

Stellt das Plugin-System fuer Tool-augmentierte RAG-Queries bereit.
Tools koennen vom LLM aufgerufen werden, um Dokumente zu suchen,
Berechnungen durchzufuehren oder Datenquellen aufzulisten.

Beziehungen:
    - generation.agent_chain.AgentChain orchestriert Tool-Aufrufe
    - generation.tool_adapter liefert backend-agnostisches Tool-Calling
    - web.api_tools stellt die Admin-API fuer Tool-Verwaltung bereit

Module:
    base     — BaseTool ABC, ToolResult Dataclass, ToolDefinition
    registry — ToolRegistry (Tool-Registrierung und -Lookup)
"""

from .base import BaseTool, ToolResult
from .registry import ToolRegistry
