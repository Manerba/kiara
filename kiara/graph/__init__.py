"""KIara GraphRAG — Wissensgraph fuer Entity-basiertes Retrieval.

Ergaenzt die Vektor-Suche um strukturierte Entity-Relationen.
Entities werden bei der Ingestion automatisch extrahiert (LLM-basiert)
und in PostgreSQL gespeichert.

Module:
    store      — GraphStore (PostgreSQL-basiert, Nodes + Edges)
    extractor  — EntityExtractor (LLM-basierte Extraktion)
    retriever  — GraphRetriever (Entity-Suche + Kontext-Anreicherung)
"""

from kiara.graph.extractor import EntityExtractor
from kiara.graph.retriever import GraphRetriever
from kiara.graph.store import GraphStore
