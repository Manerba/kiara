"""Retrieval-Schicht: Vektorsuche, Embedding, Reranking, Query-Expansion.

Module:
    vectorstore    — VectorStore-Factory mit Ollama-Embeddings (Default-Backend)
    vectorstore_qdrant   — Qdrant-Backend
    retriever      — Hybrid-Retriever mit Metadaten-Filter
    reranker       — Cross-Encoder Reranking (optional)
    query_expansion — LLM-basierte Query-Erweiterung
"""

from kiara.retrieval.retriever import Retriever
from kiara.retrieval.vectorstore import VectorStore
