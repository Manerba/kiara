"""FileHash- und ChunkDatasource-Repositories.

Verwaltet: file_hashes, hash_mode_meta, chunk_datasources.
Erbt NICHT von BaseRepository (Batch-Ops, Composite-PKs).

Ersetzt SQL in: ingestion/_file_utils.py (43 execute-Aufrufe)
"""

from __future__ import annotations

from contextlib import contextmanager

from kiara_db.connection import db_query, db_transaction


class FileHashRepository:
    """Verwaltet File-Hashes fuer inkrementelle Indexierung."""

    @contextmanager
    def _read(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_query() as db:
                yield db

    @contextmanager
    def _write(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_transaction() as db:
                yield db

    def get_hash(
        self,
        ds_id: int,
        embedding_model: str,
        file_path: str,
        *,
        conn=None,
    ) -> str | None:
        """Hash einer Datei lesen."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT file_hash FROM file_hashes
                   WHERE datasource_id = %s AND embedding_model = %s AND file_path = %s""",
                (ds_id, embedding_model, file_path),
            ).fetchone()
            return row["file_hash"] if row else None

    def upsert_hash(
        self,
        ds_id: int,
        embedding_model: str,
        file_path: str,
        file_hash: str,
        file_size: int | None = None,
        *,
        conn=None,
    ) -> None:
        """Hash + Dateigroesse speichern oder aktualisieren.

        file_size ist datasource-/dateibezogen, nicht modellbezogen.
        Deshalb wird die Groesse anschliessend ueber alle Modell-Zeilen
        derselben Datei synchronisiert, aber nur wenn `file_size` explizit
        uebergeben wurde.
        """
        with self._write(conn) as db:
            if file_size is None:
                db.execute(
                    """INSERT INTO file_hashes
                           (datasource_id, embedding_model, file_path, file_hash)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (datasource_id, embedding_model, file_path)
                       DO UPDATE SET file_hash = EXCLUDED.file_hash""",
                    (ds_id, embedding_model, file_path, file_hash),
                )
                return

            size = int(file_size)
            if size < 0:
                raise ValueError(f"file_size darf nicht negativ sein: {size}")

            db.execute(
                """INSERT INTO file_hashes
                       (datasource_id, embedding_model, file_path, file_hash, file_size)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (datasource_id, embedding_model, file_path)
                   DO UPDATE SET
                       file_hash = EXCLUDED.file_hash,
                       file_size = EXCLUDED.file_size""",
                (ds_id, embedding_model, file_path, file_hash, size),
            )
            db.execute(
                """UPDATE file_hashes
                   SET file_size = %s
                   WHERE datasource_id = %s
                     AND file_path = %s""",
                (size, ds_id, file_path),
            )

    def delete_hash(
        self,
        ds_id: int,
        embedding_model: str,
        file_path: str,
        *,
        conn=None,
    ) -> bool:
        """Einzelnen Hash loeschen (z.B. bei Fehler)."""
        with self._write(conn) as db:
            cursor = db.execute(
                """DELETE FROM file_hashes
                   WHERE datasource_id = %s AND embedding_model = %s AND file_path = %s""",
                (ds_id, embedding_model, file_path),
            )
            return cursor.rowcount > 0

    def get_all_hashes(
        self,
        ds_id: int,
        embedding_model: str,
        *,
        conn=None,
    ) -> dict[str, str]:
        """Alle Hashes fuer DS+Model als {path: hash} Dict."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT file_path, file_hash FROM file_hashes
                   WHERE datasource_id = %s AND embedding_model = %s""",
                (ds_id, embedding_model),
            ).fetchall()
            return {r["file_path"]: r["file_hash"] for r in rows}

    def get_paths_missing_size(
        self,
        ds_id: int,
        *,
        conn=None,
    ) -> set[str]:
        """Alle Dateipfade einer Datasource mit fehlender/ungeheilter file_size laden."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DISTINCT file_path FROM file_hashes
                   WHERE datasource_id = %s
                    AND file_size IS NULL""",
                (ds_id,),
            ).fetchall()
            return {r["file_path"] for r in rows}

    def delete_by_paths(
        self,
        ds_id: int,
        embedding_model: str,
        paths: list[str],
        *,
        conn=None,
    ) -> int:
        """Hashes fuer mehrere Pfade loeschen."""
        if not paths:
            return 0
        batch_size = 500
        deleted = 0
        with self._write(conn) as db:
            for i in range(0, len(paths), batch_size):
                batch = paths[i : i + batch_size]
                ph = ", ".join(["%s"] * len(batch))
                cursor = db.execute(
                    f"""DELETE FROM file_hashes
                        WHERE datasource_id = %s AND embedding_model = %s
                        AND file_path IN ({ph})""",
                    [ds_id, embedding_model, *batch],
                )
                deleted += cursor.rowcount
        return deleted

    def delete_by_datasource(
        self,
        ds_id: int,
        *,
        conn=None,
    ) -> int:
        """Alle Hashes einer Datenquelle loeschen."""
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM file_hashes WHERE datasource_id = %s",
                (ds_id,),
            )
            return cursor.rowcount


class ChunkDatasourceRepository:
    """Verwaltet die M2M-Beziehung Chunk <-> Datasource."""

    @contextmanager
    def _read(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_query() as db:
                yield db

    @contextmanager
    def _write(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_transaction() as db:
                yield db

    def upsert(
        self,
        chunk_id: str,
        ds_id: int,
        embedding_model: str = "",
        *,
        conn=None,
    ) -> None:
        """Chunk-Datasource-Zuordnung speichern."""
        with self._write(conn) as db:
            db.execute(
                """INSERT INTO chunk_datasources (chunk_id, datasource_id, embedding_model)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (chunk_id, datasource_id, embedding_model) DO NOTHING""",
                (chunk_id, ds_id, embedding_model),
            )

    def get_chunks_for_datasource(
        self,
        ds_id: int,
        embedding_model: str = "",
        *,
        conn=None,
    ) -> list[str]:
        """Alle Chunk-IDs einer Datenquelle."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT chunk_id FROM chunk_datasources
                   WHERE datasource_id = %s AND embedding_model = %s""",
                (ds_id, embedding_model),
            ).fetchall()
            return [r["chunk_id"] for r in rows]

    def delete_by_datasource(self, ds_id: int, *, conn=None) -> int:
        """Alle Zuordnungen einer Datenquelle loeschen."""
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM chunk_datasources WHERE datasource_id = %s",
                (ds_id,),
            )
            return cursor.rowcount
