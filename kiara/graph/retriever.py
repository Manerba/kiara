"""Graph-basiertes Retrieval fuer GraphRAG."""

import logging
import re

from kiara.graph.store import GraphStore
from kiara.shared.constants import STOP_WORDS as _STOP_WORDS

logger = logging.getLogger(__name__)

_ENTITY_MATCH_LIMIT = 10
_MAX_ENTITY_SEARCH_TERMS = 8
_MAX_FULL_QUERY_CHARS = 160


class GraphRetriever:
    """Findet relevante Chunks ueber Graph-Traversal."""

    def __init__(self, config: dict, graph_store: GraphStore, embedding_model: str | None = None):
        self._config = config
        self._graph = graph_store
        self._embedding_model = embedding_model
        self._max_depth = max(1, config.get("graph", {}).get("max_traversal_depth", 2))
        self._max_chunks = max(1, config.get("graph", {}).get("max_chunks", 20))

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Graph-basiertes Retrieval.

        1. Entity-Erkennung in der Query (find_nodes mit Query-Tokens)
        2. Subgraph-Expansion (get_neighbors mit max_depth)
        3. Chunk-IDs sammeln (get_connected_chunks)
        4. Chunks aus VectorDB laden (collection.get(ids=[...]))
        5. Scoring: Chunks naeher am Query-Entity bekommen hoeheren Score

        DS-Filterung findet NICHT hier statt, sondern in HybridRetriever.retrieve()
        per chunk_ds.filter_allowed() (Punkt-Lookup). GraphRetriever hat keinen
        Zugriff auf chunk_datasources.

        Returns:
            Liste von Dicts im gleichen Format wie HybridRetriever:
            [{"text": str, "metadata": dict, "score": float, "_doc_id": str}]
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 1. Entity-Erkennung
        unique_nodes = self._match_entities(query_tokens, query)
        if not unique_nodes:
            return []

        # 2. Subgraph-Expansion
        all_node_ids_with_depth = self._expand_subgraph(unique_nodes)

        # 3. Chunk-IDs deterministisch nach Graph-Distanz sammeln + begrenzen.
        chunk_ids, chunk_depth = self._collect_chunk_candidates(all_node_ids_with_depth)
        if not chunk_ids:
            return []

        result = self._load_chunks_from_graph(chunk_ids)
        if not result:
            return []

        # 4. Scoring (ohne DS-Filterung — wird in HybridRetriever gemacht)
        return self._score_results(
            result,
            chunk_depth,
            k,
        )

    def _match_entities(self, query_tokens: list[str], full_query: str) -> list[dict]:
        """Erkennt Entities im Graph anhand von Query-Tokens.

        Sucht sowohl einzelne Tokens als auch den vollstaendigen Query-String
        und dedupliziert die Ergebnisse.

        Args:
            query_tokens: Tokenisierte Query-Begriffe (ohne Stoppwoerter).
            full_query: Der vollstaendige Query-String.

        Returns:
            Liste eindeutiger Graph-Knoten.
        """
        matched_nodes = []
        for term in self._entity_search_terms(query_tokens, full_query):
            nodes = self._graph.find_nodes(term, limit=_ENTITY_MATCH_LIMIT)
            matched_nodes.extend(nodes)

        if not matched_nodes:
            return []

        # Deduplicate by node ID
        seen_ids = set()
        unique_nodes = []
        for node in matched_nodes:
            if node["id"] not in seen_ids:
                seen_ids.add(node["id"])
                unique_nodes.append(node)

        return unique_nodes

    @staticmethod
    def _entity_search_terms(query_tokens: list[str], full_query: str) -> list[str]:
        """Dedupliziert und begrenzt Suchbegriffe fuer Entity-Lookups."""
        terms = []
        seen = set()
        for token in query_tokens:
            term = str(token or "").strip()
            if not term:
                continue
            key = term.casefold()
            if key in seen:
                continue
            terms.append(term)
            seen.add(key)
            if len(terms) >= _MAX_ENTITY_SEARCH_TERMS:
                break

        compact_query = " ".join(str(full_query or "").split())[:_MAX_FULL_QUERY_CHARS]
        if compact_query:
            key = compact_query.casefold()
            if key not in seen:
                terms.append(compact_query)

        return terms

    def _expand_subgraph(self, seed_nodes: list[dict]) -> dict:
        """Expandiert den Subgraph ausgehend von den Seed-Knoten.

        Traversiert Nachbarn bis zur konfigurierten max_depth und
        merkt sich die minimale Tiefe pro Knoten.

        Args:
            seed_nodes: Liste der initialen Graph-Knoten.

        Returns:
            Dict node_id -> min_depth fuer alle erreichbaren Knoten.
        """
        all_node_ids_with_depth = {}  # node_id -> min_depth
        for node in seed_nodes:
            subgraph = self._graph.get_neighbors(node["id"], max_depth=self._max_depth)
            for sg_node in subgraph["nodes"]:
                nid = sg_node["id"]
                depth = sg_node.get("depth", 0)
                if nid not in all_node_ids_with_depth or depth < all_node_ids_with_depth[nid]:
                    all_node_ids_with_depth[nid] = depth

        return all_node_ids_with_depth

    def _collect_chunk_candidates(self, node_depths: dict) -> tuple[list[str], dict[str, int]]:
        """Sammelt Chunk-Kandidaten begrenzt nach minimaler Graph-Distanz."""
        chunk_depth: dict[str, int] = {}
        ordered_nodes = sorted(node_depths.items(), key=lambda item: (item[1], item[0]))
        if not ordered_nodes:
            return [], {}
        node_ids = [node_id for node_id, _depth in ordered_nodes]
        node_depth_map = dict(ordered_nodes)
        for node_id, chunk_id in self._graph.get_connected_chunks_by_node(node_ids):
            depth = node_depth_map.get(node_id)
            if depth is None:
                continue
            if chunk_id not in chunk_depth or depth < chunk_depth[chunk_id]:
                chunk_depth[chunk_id] = depth

        ordered_chunks = sorted(chunk_depth, key=lambda cid: (chunk_depth[cid], cid))
        limited_ids = ordered_chunks[: self._max_chunks]
        limited_depth = {cid: chunk_depth[cid] for cid in limited_ids}
        return limited_ids, limited_depth

    def _load_chunks_from_graph(self, chunk_ids: list[str]) -> dict | None:
        """Laedt Chunks aus VectorDB anhand ihrer IDs.

        Args:
            chunk_ids: Liste der zu ladenden Chunk-IDs.

        Returns:
            VectorDB-Ergebnis-Dict mit 'ids', 'documents', 'metadatas'
            oder None bei Fehler/leerem Ergebnis.
        """
        try:
            from kiara.retrieval.vectorstore import VectorStore

            store = VectorStore(self._config, self._embedding_model)
            result = store.get_documents_by_ids(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("VectorDB-Abfrage fuer Graph-Chunks fehlgeschlagen: %s", exc)
            return None

        if not result or not result.get("documents"):
            return None

        return result

    def _score_results(
        self,
        query_result: dict,
        chunk_depth: dict[str, int],
        k: int,
    ) -> list[dict]:
        """Bewertet Chunks basierend auf Graph-Distanz.

        Naehere Chunks (geringere Tiefe zum Query-Entity) bekommen hoehere Scores:
        depth 0 -> 1.0, depth 1 -> 0.7, depth 2 -> 0.4, sonst -> 0.2.

        DS-Filterung findet NICHT hier statt — wird in HybridRetriever.retrieve()
        per chunk_ds.filter_allowed() gemacht.

        Args:
            query_result: VectorDB-Ergebnis mit ids/documents/metadatas.
            chunk_depth: Dict chunk_id -> min_depth aus der Subgraph-Expansion.
            k: Maximale Anzahl zurueckgegebener Ergebnisse.

        Returns:
            Nach Score sortierte Liste von Result-Dicts.
        """
        results = []
        for _i, (doc_id, text, metadata) in enumerate(
            zip(
                query_result["ids"],
                query_result["documents"],
                query_result.get("metadatas", []),
                strict=False,
            )
        ):
            if not text:
                continue

            # Score by depth
            depth = chunk_depth.get(doc_id, self._max_depth)
            if depth == 0:
                score = 1.0
            elif depth == 1:
                score = 0.7
            elif depth == 2:
                score = 0.4
            else:
                score = 0.2

            results.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "score": score,
                    "_doc_id": doc_id,
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Einfache Tokenisierung: Split, Stoppwoerter entfernen, min 2 Zeichen."""
        if not isinstance(text, str) or not text:
            return []
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
