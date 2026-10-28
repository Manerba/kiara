"""Query-Cache: LRU-basierter Cache fuer RAG-Anfragen.

Cached LLM-Antworten inklusive Quellen fuer identische oder aehnliche
Anfragen. Reduziert Latenz und Ollama-Last bei wiederholten Fragen.
TTL und maximale Groesse sind per Config steuerbar.

Beziehungen:
    - web.app nutzt den Cache vor der RAG-Chain
    - web.api_cache stellt die Admin-API (Clear, Stats) bereit
    - ingestion.pipeline invalidiert den Cache nach Re-Indexierung

Module:
    query_cache — QueryCache (LRU-Cache mit TTL und Hash-basiertem Key)
"""

from .query_cache import QueryCache
