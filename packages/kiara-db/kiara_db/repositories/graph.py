"""Graph-Repositories (Knoten + Kanten).

Ersetzt: graph/store.py (23 execute + 1 executescript)

Schema-Design:
- graph_nodes.properties / graph_edges.properties: JSONB
- graph_edges.source: Provenienz ('extraction' vs 'erp_import')
- graph_node_chunks: n:m Node ↔ Chunk (Entity-only Chunks auffindbar)
- graph_edge_chunks: n:m Edge ↔ Chunk (Multi-Chunk-Provenienz)
"""

from __future__ import annotations

from psycopg2.extras import Json

from kiara_db.base import BaseRepository


class NodeRepository(BaseRepository):
    table = "graph_nodes"
    pk = "id"  # TEXT PK
    default_sort = "created_at DESC"

    _MAX_NODE_ID_LEN = 256

    def upsert_node(
        self,
        node_id: str,
        entity_type: str,
        label: str,
        properties: dict | None = None,
        source: str = "extraction",
        *,
        conn=None,
    ) -> None:
        """Knoten einfuegen oder aktualisieren.

        source wird nur beim INSERT gesetzt — existierende Nodes behalten
        ihren urspruenglichen source-Wert (z.B. 'erp_import').
        """
        if len(node_id) > self._MAX_NODE_ID_LEN:
            raise ValueError(
                f"node_id too long ({len(node_id)} chars, max {self._MAX_NODE_ID_LEN})"
            )
        with self._write(conn) as db:
            db.execute(
                """INSERT INTO graph_nodes (id, entity_type, label, properties, source)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       entity_type = EXCLUDED.entity_type,
                       label = EXCLUDED.label,
                       properties = (
                           COALESCE(graph_nodes.properties, '{}')::jsonb
                           || COALESCE(EXCLUDED.properties, '{}')::jsonb
                       )::text""",
                (node_id, entity_type, label, Json(properties or {}), source),
            )

    def link_to_chunk(
        self,
        node_id: str,
        chunk_id: str,
        *,
        conn=None,
    ) -> None:
        """Verknuepft einen Node mit einem Chunk (n:m). Idempotent."""
        with self._write(conn) as db:
            db.execute(
                """INSERT INTO graph_node_chunks (node_id, chunk_id)
                   VALUES (%s, %s)
                   ON CONFLICT (node_id, chunk_id) DO NOTHING""",
                (node_id, chunk_id),
            )

    def list_recent(self, limit: int = 200, *, conn=None) -> list[dict]:
        """Letzte N Knoten nach ``created_at DESC``.

        Wird fuer den Existing-Entity-Snapshot beim Start einer
        Graph-Extraction genutzt (Entity-Linking auf bereits importierte
        ERP-/Stammdaten-Knoten). ``search`` blockt leere Terms bewusst, daher
        gibt es hier eine eigene Methode statt ``search('', ...)``.
        """
        safe_limit = max(1, min(int(limit), 10000))
        return self.list_all(limit=safe_limit, conn=conn)

    def search(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
        *,
        conn=None,
    ) -> list[dict]:
        """Suche ueber Node-ID, Label und JSONB-Properties mit literalem LIKE."""
        term = str(query or "").strip()
        if not term:
            return []
        safe_limit = max(1, min(int(limit), 100))
        pattern = f"%{self._escape_like(term)}%"

        where = [
            """(
                id = %s
                OR id ILIKE %s ESCAPE '\\'
                OR label ILIKE %s ESCAPE '\\'
                OR properties::text ILIKE %s ESCAPE '\\'
            )"""
        ]
        params = [term, pattern, pattern, pattern]
        if entity_type:
            where.append("entity_type = %s")
            params.append(entity_type)

        order_params = [term, term, pattern, pattern]
        with self._read(conn) as db:
            rows = db.execute(
                f"""SELECT * FROM graph_nodes
                    WHERE {" AND ".join(where)}
                    ORDER BY
                        CASE
                            WHEN id = %s THEN 0
                            WHEN lower(id) = lower(%s) THEN 1
                            WHEN id ILIKE %s ESCAPE '\\' THEN 2
                            WHEN label ILIKE %s ESCAPE '\\' THEN 3
                            ELSE 4
                        END,
                        created_at DESC
                    LIMIT %s""",
                params + order_params + [safe_limit],
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escaped LIKE-Wildcards, damit %, _ und \\ literal gesucht werden."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def count_total(self, *, conn=None) -> int:
        """Gesamtanzahl Knoten."""
        with self._read(conn) as db:
            row = db.execute("SELECT COUNT(*) AS c FROM graph_nodes").fetchone()
            return row["c"]

    def count_by_type(self, *, conn=None) -> dict[str, int]:
        """Knotenanzahl gruppiert nach entity_type."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT entity_type, COUNT(*) AS c FROM graph_nodes GROUP BY entity_type"
            ).fetchall()
            return {r["entity_type"]: r["c"] for r in rows}

    def count_by_source(self, *, conn=None) -> dict[str, int]:
        """Knotenanzahl gruppiert nach source."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT source, COUNT(*) AS c FROM graph_nodes GROUP BY source"
            ).fetchall()
            return {r["source"]: r["c"] for r in rows}

    def get_ids_by_source(self, source: str, *, conn=None) -> list[str]:
        """Alle Node-IDs einer bestimmten source."""
        with self._read(conn) as db:
            rows = db.execute("SELECT id FROM graph_nodes WHERE source = %s", (source,)).fetchall()
            return [r["id"] for r in rows]

    def delete_by_ids(self, node_ids: list[str], *, conn=None) -> int:
        """Loescht Knoten anhand einer Liste von IDs."""
        if not node_ids:
            return 0
        with self._write(conn) as db:
            cursor = db.execute("DELETE FROM graph_nodes WHERE id = ANY(%s)", (node_ids,))
            return cursor.rowcount

    def delete_by_ids_if_orphan(
        self,
        node_ids: list[str],
        *,
        conn=None,
    ) -> int:
        """Loescht nur Nodes ohne verbleibende Edges und Chunk-Links.

        Wichtig fuer ERP-Resync (Bug #683): ERP-Nodes die auch von
        Extraction-Kanten referenziert werden oder an Chunks haengen,
        duerfen nicht geloescht werden — sonst gehen die Extraction-Kanten
        per ON DELETE CASCADE mit verloren.
        """
        if not node_ids:
            return 0
        with self._write(conn) as db:
            cursor = db.execute(
                """DELETE FROM graph_nodes
                   WHERE id = ANY(%s)
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_edges
                       WHERE source_id = graph_nodes.id
                          OR target_id = graph_nodes.id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM graph_node_chunks
                       WHERE node_id = graph_nodes.id
                   )""",
                (node_ids,),
            )
            return cursor.rowcount

    def delete_extraction_orphans(self, node_id: str, *, conn=None) -> None:
        """Loescht einen Knoten wenn er keine Kanten und keine Chunk-Links mehr hat
        und source='extraction'."""
        with self._write(conn) as db:
            has_edges = db.execute(
                "SELECT 1 FROM graph_edges WHERE source_id = %s OR target_id = %s LIMIT 1",
                (node_id, node_id),
            ).fetchone()
            if has_edges:
                return
            has_chunk_links = db.execute(
                "SELECT 1 FROM graph_node_chunks WHERE node_id = %s LIMIT 1",
                (node_id,),
            ).fetchone()
            if has_chunk_links:
                return
            db.execute(
                "DELETE FROM graph_nodes WHERE id = %s AND source = 'extraction'",
                (node_id,),
            )

    def delete_all(self, *, conn=None) -> int:
        """Loescht ALLE Knoten."""
        with self._write(conn) as db:
            cursor = db.execute("DELETE FROM graph_nodes")
            return cursor.rowcount


class EdgeRepository(BaseRepository):
    table = "graph_edges"
    default_sort = "id DESC"

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
    ) -> int:
        """Edge einfuegen oder aktualisieren (UPSERT via Provenienz-Key).

        Chunk-Provenienz wird in graph_edge_chunks (n:m) gespeichert —
        mehrere Chunks koennen dieselbe Relation extrahieren ohne sich
        gegenseitig zu ueberschreiben.
        ERP- und Extraction-Provenienz bleiben getrennte Rows, damit
        clear(source='erp_import') keine Extraction-Kante entfernt.

        Returns:
            edge_id der eingefuegten oder aktualisierten Edge.
        """
        with self._write(conn) as db:
            required_node_ids = {source_id, target_id}
            existing_rows = db.execute(
                "SELECT id FROM graph_nodes WHERE id = ANY(%s)",
                (list(required_node_ids),),
            ).fetchall()
            existing_node_ids = {row["id"] for row in existing_rows}
            missing_node_ids = sorted(required_node_ids - existing_node_ids)
            if missing_node_ids:
                raise ValueError(
                    "Graph-Edge verweist auf fehlende Nodes: " + ", ".join(missing_node_ids)
                )
            row = db.execute(
                """INSERT INTO graph_edges
                   (source_id, target_id, relation, properties, source)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (source_id, target_id, relation, source)
                   DO UPDATE SET properties = EXCLUDED.properties
                   RETURNING id""",
                (
                    source_id,
                    target_id,
                    relation,
                    Json(properties or {}),
                    source,
                ),
            ).fetchone()
            edge_id = row["id"]
            if chunk_id:
                db.execute(
                    """INSERT INTO graph_edge_chunks (edge_id, chunk_id)
                       VALUES (%s, %s)
                       ON CONFLICT (edge_id, chunk_id) DO NOTHING""",
                    (edge_id, chunk_id),
                )
            return edge_id

    def get_connected_chunks(
        self,
        node_ids: list[str],
        *,
        conn=None,
    ) -> list[str]:
        """Alle chunk_ids die mit den gegebenen Nodes verbunden sind.

        Quellen:
        1. Edges zu/von den Nodes → chunk_id via graph_edge_chunks
        2. Direkte Node-Chunk-Links (graph_node_chunks) — Entity-only Chunks
        """
        if not node_ids:
            return []
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DISTINCT chunk_id FROM (
                       SELECT gec.chunk_id FROM graph_edge_chunks gec
                       JOIN graph_edges ge ON ge.id = gec.edge_id
                       WHERE ge.source_id = ANY(%s) OR ge.target_id = ANY(%s)
                       UNION
                       SELECT chunk_id FROM graph_node_chunks
                       WHERE node_id = ANY(%s)
                   ) AS combined""",
                (node_ids, node_ids, node_ids),
            ).fetchall()
            return [r["chunk_id"] for r in rows]

    def get_connected_chunks_by_node(
        self,
        node_ids: list[str],
        *,
        conn=None,
    ) -> list[tuple[str, str]]:
        """Alle (node_id, chunk_id)-Paare fuer Graph-Retrieval in einem Batch."""
        if not node_ids:
            return []
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DISTINCT node_id, chunk_id FROM (
                       SELECT ge.source_id AS node_id, gec.chunk_id
                       FROM graph_edge_chunks gec
                       JOIN graph_edges ge ON ge.id = gec.edge_id
                       WHERE ge.source_id = ANY(%s)
                       UNION
                       SELECT ge.target_id AS node_id, gec.chunk_id
                       FROM graph_edge_chunks gec
                       JOIN graph_edges ge ON ge.id = gec.edge_id
                       WHERE ge.target_id = ANY(%s)
                       UNION
                       SELECT node_id, chunk_id
                       FROM graph_node_chunks
                       WHERE node_id = ANY(%s)
                   ) AS combined""",
                (node_ids, node_ids, node_ids),
            ).fetchall()
            return [(r["node_id"], r["chunk_id"]) for r in rows]

    def delete_by_chunk_id(self, chunk_id: str, *, conn=None) -> int:
        """Entfernt Chunk-Provenienz aus graph_edge_chunks fuer einen Chunk.

        Edges ohne verbleibende Chunk-Links werden entfernt (da sie keine
        Herkunft mehr haben) — Edges die auch von anderen Chunks extrahiert
        wurden bleiben bestehen.
        Gibt Anzahl entfernter Junction-Eintraege zurueck.
        """
        with self._write(conn) as db:
            affected_rows = db.execute(
                "SELECT edge_id FROM graph_edge_chunks WHERE chunk_id = %s",
                (chunk_id,),
            ).fetchall()
            affected_edge_ids = [r["edge_id"] for r in affected_rows]
            cursor = db.execute(
                "DELETE FROM graph_edge_chunks WHERE chunk_id = %s",
                (chunk_id,),
            )
            removed = cursor.rowcount
            if affected_edge_ids:
                db.execute(
                    """DELETE FROM graph_edges ge
                       WHERE ge.id = ANY(%s)
                         AND ge.source = 'extraction'
                         AND NOT EXISTS (
                           SELECT 1 FROM graph_edge_chunks gec
                           WHERE gec.edge_id = ge.id
                         )""",
                    (affected_edge_ids,),
                )
            return removed

    def count_total(self, *, conn=None) -> int:
        """Gesamtanzahl Kanten."""
        with self._read(conn) as db:
            row = db.execute("SELECT COUNT(*) AS c FROM graph_edges").fetchone()
            return row["c"]

    def count_by_relation(self, *, conn=None) -> dict[str, int]:
        """Kantenanzahl gruppiert nach relation."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT relation, COUNT(*) AS c FROM graph_edges GROUP BY relation"
            ).fetchall()
            return {r["relation"]: r["c"] for r in rows}

    def get_edges_for_node(self, node_id: str, *, conn=None) -> list[dict]:
        """Alle logischen Kanten die mit einem Knoten verbunden sind.

        Physische Edge-Rows sind nach Provenienz getrennt. Traversal soll
        dieselbe Beziehung aber nur einmal sehen; Extraction gewinnt dabei
        gegen ERP, weil sie Chunk-Provenienz tragen kann.
        """
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DISTINCT ON (source_id, target_id, relation) *
                   FROM graph_edges
                   WHERE source_id = %s OR target_id = %s
                   ORDER BY source_id, target_id, relation,
                            CASE WHEN source = 'extraction' THEN 0 ELSE 1 END,
                            id""",
                (node_id, node_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_edges_by_chunk(self, chunk_id: str, *, conn=None) -> list[dict]:
        """Alle Kanten die diesen Chunk als Herkunft haben (via Junction)."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT ge.source_id, ge.target_id, ge.relation, ge.id
                   FROM graph_edges ge
                   JOIN graph_edge_chunks gec ON gec.edge_id = ge.id
                   WHERE gec.chunk_id = %s""",
                (chunk_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_by_node_ids(
        self,
        node_ids: list[str],
        source: str | None = None,
        *,
        conn=None,
    ) -> int:
        """Loescht Kanten die mit den gegebenen Nodes verbunden sind.

        Mit source-Filter werden nur Kanten dieser Provenienz entfernt —
        z.B. source='erp_import' loescht Extraction-Kanten NICHT.
        """
        if not node_ids:
            return 0
        sql = "DELETE FROM graph_edges WHERE (source_id = ANY(%s) OR target_id = ANY(%s))"
        params: list = [node_ids, node_ids]
        if source is not None:
            sql += " AND source = %s"
            params.append(source)
        with self._write(conn) as db:
            cursor = db.execute(sql, tuple(params))
            return cursor.rowcount

    def delete_by_source(self, source: str, *, conn=None) -> int:
        """Loescht alle Kanten einer Provenienz, unabhaengig von Node-Source."""
        with self._write(conn) as db:
            cursor = db.execute("DELETE FROM graph_edges WHERE source = %s", (source,))
            return cursor.rowcount

    def delete_all(self, *, conn=None) -> int:
        """Loescht ALLE Kanten."""
        with self._write(conn) as db:
            cursor = db.execute("DELETE FROM graph_edges")
            return cursor.rowcount
