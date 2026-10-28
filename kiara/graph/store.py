"""Persistenter Wissensgraph in PostgreSQL fuer GraphRAG."""

import json
import logging
from collections import deque

from kiara.repositories.graph import EdgeRepository, NodeRepository
from kiara.shared.db import db_query, db_transaction

logger = logging.getLogger(__name__)

# Module-level Repo-Instanzen
_node_repo = NodeRepository()
_edge_repo = EdgeRepository()


def _load_props(raw):
    """Laedt JSON-Properties robust.

    Spalte ``properties`` ist derzeit ``text``; psycopg2 gibt den Wert als
    String zurueck. Wenn das Schema spaeter auf ``jsonb`` umgestellt wird,
    gibt psycopg2 bereits ein ``dict``/``list`` zurueck. Beide Faelle sind
    abgedeckt.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        result = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return result if isinstance(result, (dict, list)) else {}


class GraphStore:
    """Persistenter Wissensgraph in PostgreSQL."""

    def __init__(self):
        pass

    def add_node(
        self,
        node_id: str,
        entity_type: str,
        label: str,
        properties: dict | None = None,
        source: str = "extraction",
        chunk_id: str | None = None,
        *,
        conn=None,
    ) -> None:
        """Fuegt Knoten hinzu oder aktualisiert ihn (UPSERT).

        Optionale chunk_id verknuepft den Knoten in graph_node_chunks mit
        einem Chunk — wichtig fuer Entity-only Chunks, die sonst ueber
        den GraphRetriever nicht auffindbar waeren (Bug #685).

        Upsert + Link laufen in EINER Transaktion — sonst kann ein
        fehlgeschlagener Link einen bereits committed Node zuruecklassen.
        """
        from contextlib import nullcontext

        ctx = nullcontext(conn) if conn is not None else db_transaction()
        with ctx as c:
            _node_repo.upsert_node(
                node_id,
                entity_type,
                label,
                properties=properties,
                source=source,
                conn=c,
            )
            if chunk_id:
                _node_repo.link_to_chunk(node_id, chunk_id, conn=c)

    def link_node_to_chunk(self, node_id: str, chunk_id: str, *, conn=None) -> None:
        """Verknuepft einen bestehenden Node mit einem Chunk ohne Upsert."""
        _node_repo.link_to_chunk(node_id, chunk_id, conn=conn)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict | None = None,
        chunk_id: str | None = None,
        source: str = "extraction",
        *,
        conn=None,
    ) -> None:
        """Fuegt Kante hinzu. Chunk-Provenienz wird additiv in Junction gespeichert
        (mehrere Chunks koennen dieselbe Relation extrahieren).

        source unterscheidet 'extraction' von 'erp_import' — ERP-Resync
        loescht nur 'erp_import'-Kanten (Bug #683).
        """
        _edge_repo.add_edge(
            source_id,
            target_id,
            relation,
            properties=properties,
            chunk_id=chunk_id,
            source=source,
            conn=conn,
        )

    def node_exists(self, node_id: str, *, conn=None) -> bool:
        """Prueft ob ein Knoten existiert."""
        return _node_repo.get_by_id(node_id, conn=conn) is not None

    def get_neighbors(
        self,
        node_id: str,
        max_depth: int = 2,
        relation_filter: list[str] | None = None,
        max_nodes: int = 10_000,
    ) -> dict:
        """BFS-Traversal ab node_id. Gibt Subgraph zurueck:
        {"nodes": [...], "edges": [...]}"""
        # BFS benoetigt mehrere Queries in einer Connection fuer Konsistenz
        with db_query() as conn:
            visited_nodes = set()
            visited_edges = set()
            queue = deque([(node_id, 0)])
            queued_nodes = {node_id}
            nodes = []
            edges = []
            max_nodes_logged = False

            while queue:
                current_id, depth = queue.popleft()
                queued_nodes.discard(current_id)
                if current_id in visited_nodes:
                    continue
                if len(visited_nodes) >= max_nodes:
                    logger.warning(
                        "BFS abgebrochen: max_nodes=%d erreicht (start=%s)", max_nodes, node_id
                    )
                    break
                visited_nodes.add(current_id)

                node_row = _node_repo.get_by_id(current_id, conn=conn)
                if not node_row:
                    logger.warning("Knoten %s existiert nicht (dangling FK?)", current_id)
                if node_row:
                    nodes.append(
                        {
                            "id": node_row["id"],
                            "entity_type": node_row["entity_type"],
                            "label": node_row["label"],
                            "properties": _load_props(node_row["properties"]),
                            "source": node_row["source"],
                            "depth": depth,
                        }
                    )

                if depth >= max_depth:
                    continue

                edge_rows = _edge_repo.get_edges_for_node(current_id, conn=conn)
                for edge_row in edge_rows:
                    if relation_filter and edge_row["relation"] not in relation_filter:
                        continue
                    edge_id = edge_row["id"]
                    if edge_id in visited_edges:
                        continue
                    neighbor_id = (
                        edge_row["target_id"]
                        if edge_row["source_id"] == current_id
                        else edge_row["source_id"]
                    )
                    if (
                        neighbor_id not in visited_nodes
                        and neighbor_id not in queued_nodes
                        and len(visited_nodes) + len(queued_nodes) >= max_nodes
                    ):
                        if not max_nodes_logged:
                            logger.warning(
                                "BFS abgebrochen: max_nodes=%d erreicht (start=%s)",
                                max_nodes,
                                node_id,
                            )
                            max_nodes_logged = True
                        continue

                    visited_edges.add(edge_id)
                    edges.append(
                        {
                            "id": edge_id,
                            "source_id": edge_row["source_id"],
                            "target_id": edge_row["target_id"],
                            "relation": edge_row["relation"],
                            "properties": _load_props(edge_row["properties"]),
                            "source": edge_row["source"],
                        }
                    )

                    if neighbor_id not in visited_nodes and neighbor_id not in queued_nodes:
                        queue.append((neighbor_id, depth + 1))
                        queued_nodes.add(neighbor_id)

            returned_node_ids = {node["id"] for node in nodes}
            edges = [
                edge
                for edge in edges
                if (
                    edge["source_id"] in returned_node_ids
                    and edge["target_id"] in returned_node_ids
                )
            ]
            return {"nodes": nodes, "edges": edges}

    def find_nodes(self, query: str, entity_type: str = None, limit: int = 20) -> list[dict]:
        """Volltextsuche ueber node labels (ILIKE %query%).
        Optional nach entity_type filtern."""
        rows = _node_repo.search(query, entity_type=entity_type, limit=limit)
        return [
            {
                "id": r["id"],
                "entity_type": r["entity_type"],
                "label": r["label"],
                "properties": _load_props(r["properties"]),
                "source": r["source"],
            }
            for r in rows
        ]

    def list_nodes(self, limit: int = 200) -> list[dict]:
        """Letzte N Knoten als Snapshot fuer Entity-Linking.

        Ohne Suchterm — ``find_nodes('')`` waere leer, weil die Repo-Suche
        leere Terms ablehnt. Genutzt vom Ingestion-Pfad zum Vorbefuellen
        des ``existing_entities``-Prompt-Kontexts.
        """
        rows = _node_repo.list_recent(limit=limit)
        return [
            {
                "id": r["id"],
                "entity_type": r["entity_type"],
                "label": r["label"],
                "properties": _load_props(r["properties"]),
                "source": r["source"],
            }
            for r in rows
        ]

    def get_connected_chunks(self, node_ids: list[str]) -> list[str]:
        """Gibt alle chunk_ids zurueck, die ueber Kanten mit den Knoten verbunden sind."""
        return _edge_repo.get_connected_chunks(node_ids)

    def get_connected_chunks_by_node(self, node_ids: list[str]) -> list[tuple[str, str]]:
        """Gibt (node_id, chunk_id)-Paare fuer mehrere Nodes in einem DB-Call zurueck."""
        return _edge_repo.get_connected_chunks_by_node(node_ids)

    def get_stats(self) -> dict:
        """Gibt Statistiken zurueck."""
        return {
            "total_nodes": _node_repo.count_total(),
            "total_edges": _edge_repo.count_total(),
            "nodes_by_type": _node_repo.count_by_type(),
            "edges_by_relation": _edge_repo.count_by_relation(),
            "sources": _node_repo.count_by_source(),
        }

    def clear(self, source: str | None = None, *, conn=None) -> int:
        """Loescht Knoten und Kanten. Optional nur fuer eine source.

        Mit source-Filter werden NUR Kanten dieser Provenienz geloescht
        (Bug #683: ERP-Resync darf Extraction-Kanten zu ERP-Knoten nicht
        ausradieren). Node-Loeschung cascadet via FK auf alle Kanten — deshalb
        Edges zuerst selektiv loeschen, danach Nodes.

        Gibt Anzahl geloeschter Knoten zurueck.
        """
        from contextlib import nullcontext

        ctx = nullcontext(conn) if conn is not None else db_transaction()
        with ctx as c:
            if source:
                node_ids = _node_repo.get_ids_by_source(source, conn=c)
                edge_count = _edge_repo.delete_by_source(source, conn=c)
                if edge_count:
                    logger.info(
                        "Graph-Clear: %d Kanten mit source=%s entfernt",
                        edge_count,
                        source,
                    )
                deleted_count = 0
                if node_ids:
                    # Nodes haben FK→edges mit ON DELETE CASCADE. Wenn noch
                    # Extraction-Kanten zu ihnen existieren, wuerden diese
                    # beim Delete Cascade-geloescht. Daher nur Nodes loeschen
                    # die keine Kanten mehr haben.
                    deleted_count = _node_repo.delete_by_ids_if_orphan(node_ids, conn=c)
                return deleted_count
            count = _node_repo.count_total(conn=c)
            _edge_repo.delete_all(conn=c)
            _node_repo.delete_all(conn=c)
            return count

    def delete_by_chunk_id(self, chunk_id: str) -> None:
        """Entfernt alle Chunk-Verknuepfungen (Edge + Node-Chunks) fuer einen
        Chunk und bereinigt Orphan-Extraction-Nodes.

        Wird bei inkrementeller Neuindexierung aufgerufen.
        """
        with db_transaction() as conn:
            edge_rows = _edge_repo.get_edges_by_chunk(chunk_id, conn=conn)
            affected_node_ids = set()
            for row in edge_rows:
                affected_node_ids.add(row["source_id"])
                affected_node_ids.add(row["target_id"])
            # Auch Node-Chunk-Links dieses Chunks sammeln — fuer Orphan-Cleanup
            node_link_rows = conn.execute(
                "SELECT node_id FROM graph_node_chunks WHERE chunk_id = %s",
                (chunk_id,),
            ).fetchall()
            for r in node_link_rows:
                affected_node_ids.add(r["node_id"])

            _edge_repo.delete_by_chunk_id(chunk_id, conn=conn)
            conn.execute(
                "DELETE FROM graph_node_chunks WHERE chunk_id = %s",
                (chunk_id,),
            )

            for nid in affected_node_ids:
                _node_repo.delete_extraction_orphans(nid, conn=conn)

    def delete_by_chunk_ids(self, chunk_ids: list[str], *, conn=None) -> None:
        """Loescht alle Kanten fuer mehrere Chunks und verwaiste Knoten.

        Batch-Version von delete_by_chunk_id. Akzeptiert optional eine
        bestehende Connection, damit der Aufrufer Graph-Cleanup und
        chunk_datasources-Cleanup in derselben Transaktion kombinieren kann.
        """
        if not chunk_ids:
            return

        def _run(c):
            affected_node_ids = set()
            for chunk_id in chunk_ids:
                edge_rows = _edge_repo.get_edges_by_chunk(chunk_id, conn=c)
                for row in edge_rows:
                    affected_node_ids.add(row["source_id"])
                    affected_node_ids.add(row["target_id"])
                node_link_rows = c.execute(
                    "SELECT node_id FROM graph_node_chunks WHERE chunk_id = %s",
                    (chunk_id,),
                ).fetchall()
                for r in node_link_rows:
                    affected_node_ids.add(r["node_id"])
                _edge_repo.delete_by_chunk_id(chunk_id, conn=c)
                c.execute(
                    "DELETE FROM graph_node_chunks WHERE chunk_id = %s",
                    (chunk_id,),
                )
            for nid in affected_node_ids:
                _node_repo.delete_extraction_orphans(nid, conn=c)

        if conn is not None:
            _run(conn)
        else:
            with db_transaction() as c:
                _run(c)
