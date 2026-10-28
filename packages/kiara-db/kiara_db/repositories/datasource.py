"""Datasource-Repository.

Ersetzt SQL in: web/api_datasource.py (16 execute-Aufrufe)
"""

from __future__ import annotations

from kiara_db.base import BaseRepository, RawSQL


class DatasourceRepository(BaseRepository):
    table = "datasources"
    default_sort = "id ASC"

    def get_active(self, *, conn=None) -> list[dict]:
        """Alle aktiven Datenquellen mit Chunks."""
        rows, _ = self.list_filtered(
            {"is_active": True},
            sort="name ASC",
            limit=10000,
            offset=0,
            conn=conn,
        )
        return rows

    def list_active_with_chunks(self, *, conn=None) -> list[dict]:
        """Aktive Datenquellen mit Chunks (ueber datasource_embeddings)."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT d.id, d.name, d.type, "
                "COALESCE(SUM(de.chunk_count), 0) AS chunk_count, "
                "COALESCE(SUM(de.file_count), 0) AS file_count "
                "FROM datasources d "
                "LEFT JOIN datasource_embeddings de ON de.datasource_id = d.id "
                "WHERE d.is_active = true "
                "GROUP BY d.id, d.name, d.type "
                "HAVING COALESCE(SUM(de.chunk_count), 0) > 0 "
                "ORDER BY d.name"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_restriction_levels(
        self,
        ds_ids: list[int],
        *,
        conn=None,
    ) -> list[dict]:
        """Restriction-Level fuer eine Liste von DS-IDs."""
        if not ds_ids:
            return []
        ph = ", ".join(["%s"] * len(ds_ids))
        with self._read(conn) as db:
            rows = db.execute(
                f"SELECT restriction_level FROM datasources WHERE id IN ({ph})",
                list(ds_ids),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_cascade(self, ds_id: int, *, conn=None) -> dict:
        """Loescht DS mit allen abhaengigen Tabellen.

        Reihenfolge (FK-Abhaengigkeiten):
        1. ingestion_file_logs (FK auf ingestion_jobs)
        2. file_hashes, hash_mode_meta, chunk_datasources
        3. datasource_embeddings, datasource_groups, datasource_users
        4. ingestion_jobs
        5. cache_datasources, cache_jobs
        6. datasources

        Fail-closed bei laufendem Job (Bug #778): der Worker wuerde
        sonst in die verwaiste DS schreiben (FK-Fehler) oder Ghost-Chunks
        hinterlassen. Aufrufer muss den Job vorher abbrechen.

        Raises:
            RuntimeError: Wenn ein Job mit status='running' fuer diese DS
                existiert.

        Hinweis: Qdrant-Vektoren muessen separat geloescht werden!
        """
        with self._write(conn) as db:
            db.execute("LOCK TABLE ingestion_jobs IN SHARE ROW EXCLUSIVE MODE")
            row = db.execute(
                "SELECT type, name FROM datasources WHERE id = %s", (ds_id,)
            ).fetchone()
            if not row:
                return {"deleted": False}

            running = db.execute(
                "SELECT id FROM ingestion_jobs "
                "WHERE datasource_id = %s AND status = 'running' "
                "LIMIT 1",
                (ds_id,),
            ).fetchone()
            if running:
                raise RuntimeError(
                    f"Datasource {ds_id} ({row['name']}) hat einen laufenden "
                    f"Ingestion-Job (job_id={running['id']}). Bitte Job erst "
                    f"abbrechen bevor die Datenquelle geloescht wird."
                )

            counts = {}

            # FK-Abhaengigkeit: ingestion_file_logs ZUERST
            cursor = db.execute(
                """DELETE FROM ingestion_file_logs WHERE job_id IN (
                       SELECT id FROM ingestion_jobs WHERE datasource_id = %s
                   )""",
                (ds_id,),
            )
            counts["ingestion_file_logs"] = cursor.rowcount

            for dep_table in [
                "file_hashes",
                "hash_mode_meta",
                "chunk_datasources",
                "datasource_embeddings",
                "datasource_groups",
                "datasource_users",
                "ingestion_jobs",
                "cache_datasources",
                "cache_jobs",
            ]:
                cursor = db.execute(f"DELETE FROM {dep_table} WHERE datasource_id = %s", (ds_id,))
                counts[dep_table] = cursor.rowcount

            db.execute("DELETE FROM datasources WHERE id = %s", (ds_id,))
            return {
                "deleted": True,
                "name": row["name"],
                "type": row["type"],
                "dependents": counts,
            }


class DatasourceEmbeddingRepository(BaseRepository):
    table = "datasource_embeddings"
    default_sort = "datasource_id ASC"

    def get_active_models(self, *, conn=None) -> list[str]:
        """Alle Embedding-Modelle mit mindestens einem Chunk."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DISTINCT embedding_model FROM datasource_embeddings
                   WHERE chunk_count > 0 ORDER BY embedding_model""",
            ).fetchall()
            return [r["embedding_model"] for r in rows]

    def get_for_datasource_model(
        self,
        datasource_id: int,
        embedding_model: str,
        *,
        conn=None,
    ) -> dict | None:
        """Statistik fuer genau eine DS-/Embedding-Modell-Kombination."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT datasource_id, embedding_model, chunk_count,
                          file_count, last_ingested_at
                   FROM datasource_embeddings
                   WHERE datasource_id = %s AND embedding_model = %s""",
                (datasource_id, embedding_model),
            ).fetchone()
            return dict(row) if row else None

    def upsert_stats(
        self,
        datasource_id: int,
        embedding_model: str,
        chunk_count: int,
        file_count: int,
        *,
        conn=None,
    ) -> None:
        """Aktualisiert oder erstellt Embedding-Statistiken."""
        self.upsert(
            conflict_columns=["datasource_id", "embedding_model"],
            conn=conn,
            datasource_id=datasource_id,
            embedding_model=embedding_model,
            chunk_count=chunk_count,
            file_count=file_count,
            last_ingested_at=RawSQL("NOW()"),
        )

    def upsert_chunk_count(
        self,
        datasource_id: int,
        embedding_model: str,
        chunk_count: int,
        *,
        conn=None,
    ) -> None:
        """Aktualisiert nur chunk_count (ohne file_count zu ueberschreiben).

        Fuer Fehler-Jobs: Chunk-Count aktualisieren ohne file_count zu verlieren.
        """
        with self._write(conn) as db:
            db.execute(
                """INSERT INTO datasource_embeddings
                       (datasource_id, embedding_model, chunk_count, last_ingested_at)
                   VALUES (%s, %s, %s, NOW())
                   ON CONFLICT(datasource_id, embedding_model) DO UPDATE SET
                       chunk_count = EXCLUDED.chunk_count,
                       last_ingested_at = EXCLUDED.last_ingested_at""",
                (datasource_id, embedding_model, chunk_count),
            )

    def get_last_model_for_datasource(
        self,
        datasource_id: int,
        *,
        conn=None,
    ) -> str:
        """Ermittelt das zuletzt verwendete Embedding-Modell einer Datenquelle.

        Returns:
            Modellname oder leerer String wenn keine Ingestion bekannt.
        """
        with self._read(conn) as db:
            row = db.execute(
                "SELECT embedding_model FROM datasource_embeddings "
                "WHERE datasource_id = %s ORDER BY last_ingested_at DESC NULLS LAST LIMIT 1",
                (datasource_id,),
            ).fetchone()
            return row["embedding_model"] if row else ""

    def get_ds_ids_with_chunks(self, *, conn=None) -> set[int]:
        """IDs aller Datenquellen mit mind. einem Chunk."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT DISTINCT datasource_id FROM datasource_embeddings WHERE chunk_count > 0"
            ).fetchall()
            return {r["datasource_id"] for r in rows}
