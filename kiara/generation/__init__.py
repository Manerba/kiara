"""Generation-Schicht: LLM-Backends, RAG-Chain, Cloud-Routing.

Module:
    backends/       — LLMBackend-ABC + Provider (Ollama/Anthropic/OpenAI) + AnonymizingBackend + Factory
    chain           — RAGChain (Backend-injiziert)
    cloud_routing   — Shared Routing-/Backend-Helper fuer api_chat.py und mcp_server.py
    agent_chain     — AgentChain (ReAct-Loop mit Tool-Aufrufen)
    adapters/       — Backend-agnostische Tool-Calling-Abstraktion (Ollama/Anthropic/OpenAI)
    tool_adapter    — Backward-Compat-Wrapper fuer adapters/
    unchained_chain — Direkt-LLM ohne RAG-Kontext (Backend-injiziert)
    rewriter_chain  — Query-Rewrite + RAG (Backend-injiziert)
"""

from kiara.generation.chain import RAGChain
