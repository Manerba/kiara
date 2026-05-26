"""Index-State fuer das Kiara-Monorepo.

Monorepo-internes Helper-Modul fuer HashStore und ChunkDatasources.
Kein stabiler Standalone-Contract: Das Modul darf weiterhin `kiara.shared.*`,
`kiara.repositories.*` und die Monorepo-DB-Infrastruktur direkt voraussetzen.

Extrahiert aus kiara.ingestion._file_utils (Sprint 2, Package-Extraktion).
Verwaltet den persistenten Zustand fuer inkrementelle Ingestion:
- HashStore: File-Hashes in PostgreSQL (file_hashes + hash_mode_meta)
- ChunkDatasources: n:m-Zuordnung Chunk <-> Datasource (chunk_datasources)
"""

import logging
import os
from collections.abc import Generator

from kiara.repositories.file_hash import ChunkDatasourceRepository, FileHashRepository
from kiara.shared.constants import INGESTION_BATCH_SIZE
from kiara.shared.db import db_query, db_transaction
from kiara_ingestion_core.file_diff import compute_file_hash

logger = logging.getLogger(__name__)

# Module-level repo instances
_hash_repo = FileHashRepository()
_chunk_ds_repo = ChunkDatasourceRepository()

_BATCH_SIZE = INGESTION_BATCH_SIZE
_GRAPH_CHUNK_CLEANUP_LOCK_KEY = 681029


def _in_batches(
    items: list, batch_size: int = _BATCH_SIZE
) -> Generator[tuple[list, str], None, None]:
    """Yields (batch, placeholders) fuer batched IN-clause Queries.

    Teilt eine Liste in Gruppen und erzeugt fuer jede Gruppe
    die passenden SQL-Placeholders.

    Yields:
        Tuple (batch: list, placeholders: str) — z.B. (["a","b"], "%s,%s")
    """
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        placeholders = ",".join(["%s"] * len(batch))
        yield batch, placeholders


def _upsert_hash_rows(conn, rows: list[tuple[int, str, str, str, int]]) -> None:
    """Schreibt file_hashes und synchronisiert file_size ueber alle Modelle."""
    conn.executemany(
        "INSERT INTO file_hashes "
        "(datasource_id, embedding_model, file_path, file_hash, file_size) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (datasource_id, embedding_model, file_path) "
        "DO UPDATE SET file_hash = EXCLUDED.file_hash, "
        "file_size = EXCLUDED.file_size",
        rows,
    )
    conn.executemany(
        "UPDATE file_hashes SET file_size = %s WHERE datasource_id = %s AND file_path = %s",
        [(fs, ds_id, fp) for ds_id, _model, fp, _fh, fs in rows],
    )


class HashStore:
    """PostgreSQL-basierter Hash-Store fuer inkrementelle Ingestion.

    Speichert File-Hashes in der PostgreSQL-Tabelle ``file_hashes``
    mit O(1) Lookups und inkrementellen Writes.
    """

    def __init__(self):
        pass

    @staticmethod
    def _ds_id(datasource_id) -> int:
        """Normalisiert datasource_id: None → 0 (Legacy-CLI-Pfad)."""
        return 0 if datasource_id is None else int(datasource_id)

    def load(self, datasource_id, embedding_model) -> dict[str, str]:
        """Laedt alle Hashes fuer eine Datasource+Modell-Kombination."""
        ds_id = self._ds_id(datasource_id)
        model = embedding_model
        return _hash_repo.get_all_hashes(ds_id, model)

    def upsert_many(
        self,
        datasource_id,
        embedding_model,
        hashes: dict[str, tuple[str, int]],
    ) -> None:
        """Fuegt Hashes + Dateigroessen ein oder aktualisiert sie."""
        if not hashes:
            return

        ds_id = self._ds_id(datasource_id)
        model = embedding_model
        rows = []

        for fp, value in hashes.items():
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(
                    f"HashStore.upsert_many erwartet (hash, file_size), bekam {value!r} fuer {fp}"
                )
            fh, fs = value
            if not isinstance(fh, str):
                raise TypeError(f"file_hash muss str sein, bekam {type(fh).__name__} fuer {fp}")
            if fs is None:
                raise TypeError(f"file_size darf nicht None sein fuer {fp}")
            try:
                size = int(fs)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"file_size muss int-castbar sein, bekam {fs!r} fuer {fp}") from exc
            if size < 0:
                raise ValueError(f"file_size darf nicht negativ sein: {size} fuer {fp}")
            rows.append((ds_id, model, fp, fh, size))

        with db_transaction() as conn:
            _upsert_hash_rows(conn, rows)

    def get_missing_size_paths(self, datasource_id) -> set[str]:
        """Dateipfade mit fehlender file_size fuer eine Datasource."""
        ds_id = self._ds_id(datasource_id)
        return _hash_repo.get_paths_missing_size(ds_id)

    def remove_files(self, datasource_id, embedding_model, paths: list[str], conn=None) -> None:
        """Loescht Hashes fuer bestimmte Dateipfade (batched in 500er-Gruppen).

        Args:
            conn: Optionale externe DB-Connection fuer kombinierte
                  Transaktionen (z.B. Cross-DS-Cleanup).
        """
        if not paths:
            return
        ds_id = self._ds_id(datasource_id)
        model = embedding_model
        if conn is not None:
            for batch, ph in _in_batches(paths):
                conn.execute(
                    f"DELETE FROM file_hashes "
                    f"WHERE datasource_id = %s AND embedding_model = %s "
                    f"AND file_path IN ({ph})",
                    [ds_id, model, *batch],
                )
        else:
            _hash_repo.delete_by_paths(ds_id, model, paths)

    def get_hash_mode(self, datasource_id, embedding_model) -> str:
        """Gibt den gespeicherten hash_mode fuer diese DS+Modell-Kombination zurueck."""
        ds_id = self._ds_id(datasource_id)
        model = embedding_model
        with db_query() as conn:
            row = conn.execute(
                "SELECT hash_mode FROM hash_mode_meta "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, model),
            ).fetchone()
        return row["hash_mode"] if row else "mtime"

    def set_hash_mode(self, datasource_id, embedding_model, hash_mode: str) -> None:
        """Speichert den hash_mode fuer diese DS+Modell-Kombination."""
        ds_id = self._ds_id(datasource_id)
        model = embedding_model
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO hash_mode_meta "
                "(datasource_id, embedding_model, hash_mode) VALUES (%s, %s, %s) "
                "ON CONFLICT (datasource_id, embedding_model) "
                "DO UPDATE SET hash_mode = EXCLUDED.hash_mode",
                (ds_id, model, hash_mode),
            )

    def migrate_hash_mode(self, datasource_id, embedding_model, new_mode: str) -> int:
        """Migriert gespeicherte Hashes auf einen neuen hash_mode.

        Liest alle gespeicherten Dateipfade, berechnet den Hash mit dem neuen
        Modus neu und aktualisiert die DB. Nicht mehr existierende Dateien
        werden entfernt.

        Die gesamte Migration (upsert + remove + mode-update) laeuft in einer
        einzigen Transaktion fuer Atomizitaet.

        Returns:
            Anzahl erfolgreich migrierter Hashes.
        """
        stored = self.load(datasource_id, embedding_model)
        if not stored:
            self.set_hash_mode(datasource_id, embedding_model, new_mode)
            return 0

        ds_id = self._ds_id(datasource_id)
        model = embedding_model

        migrated = {}
        missing = []
        for fp in stored:
            try:
                stat = os.stat(fp)
                migrated[fp] = (compute_file_hash(fp, new_mode), stat.st_size)
            except OSError:
                missing.append(fp)

        # Atomare Transaktion: Hashes aktualisieren + fehlende entfernen + Mode setzen
        with db_transaction() as conn:
            if migrated:
                _upsert_hash_rows(
                    conn,
                    [(ds_id, model, fp, fh, fs) for fp, (fh, fs) in migrated.items()],
                )
            if missing:
                for batch, ph in _in_batches(missing):
                    conn.execute(
                        f"DELETE FROM file_hashes "
                        f"WHERE datasource_id = %s AND embedding_model = %s "
                        f"AND file_path IN ({ph})",
                        [ds_id, model, *batch],
                    )
            conn.execute(
                "INSERT INTO hash_mode_meta "
                "(datasource_id, embedding_model, hash_mode) VALUES (%s, %s, %s) "
                "ON CONFLICT (datasource_id, embedding_model) "
                "DO UPDATE SET hash_mode = EXCLUDED.hash_mode",
                (ds_id, model, new_mode),
            )

        logger.info(
            "Hash-Migration: ds=%s model=%s → %s (%d/%d migriert, %d entfernt)",
            datasource_id,
            embedding_model,
            new_mode,
            len(migrated),
            len(stored),
            len(missing),
        )
        return len(migrated)

    def delete_datasource(self, datasource_id, embedding_model=None) -> None:
        """Loescht alle Hashes + hash_mode_meta einer Datasource."""
        ds_id = self._ds_id(datasource_id)
        with db_transaction() as conn:
            if embedding_model is not None:
                model = embedding_model
                conn.execute(
                    "DELETE FROM file_hashes WHERE datasource_id = %s AND embedding_model = %s",
                    (ds_id, model),
                )
                conn.execute(
                    "DELETE FROM hash_mode_meta WHERE datasource_id = %s AND embedding_model = %s",
                    (ds_id, model),
                )
            else:
                conn.execute(
                    "DELETE FROM file_hashes WHERE datasource_id = %s",
                    (ds_id,),
                )
                conn.execute(
                    "DELETE FROM hash_mode_meta WHERE datasource_id = %s",
                    (ds_id,),
                )


class ChunkDatasources:
    """Verwaltet die n:m-Beziehung Chunk <-> Datasource in PostgreSQL.

    Alle DS-Zuordnungen leben in der Tabelle ``chunk_datasources``.
    Schreibende Methoden nutzen ``db_transaction`` fuer Atomizitaet.
    """

    def __init__(self):
        pass

    @staticmethod
    def acquire_graph_cleanup_lock(conn) -> None:
        """Nimmt den gemeinsamen Lock fuer Graph-Chunk-Cleanup/Writer.

        Der Lock ist transaktionsgebunden. Aufrufer, die erst
        ``chunk_datasources`` mutieren und danach Graph-Cleanup entscheiden,
        muessen ihn in derselben Transaktion vor der Mutation halten.
        """
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (_GRAPH_CHUNK_CLEANUP_LOCK_KEY,),
        )

    @staticmethod
    def acquire_graph_cleanup_session_lock(conn) -> None:
        """Nimmt denselben Lock sessionweit fuer mehrtransaktionale Repairs."""
        conn.execute(
            "SELECT pg_advisory_lock(%s)",
            (_GRAPH_CHUNK_CLEANUP_LOCK_KEY,),
        )
        conn.commit()

    @staticmethod
    def release_graph_cleanup_session_lock(conn) -> None:
        """Gibt den sessionweiten Graph-Cleanup-Lock frei."""
        conn.execute(
            "SELECT pg_advisory_unlock(%s)",
            (_GRAPH_CHUNK_CLEANUP_LOCK_KEY,),
        )
        conn.commit()

    def chunks_exist(self, chunk_ids: list[str], model: str) -> set[str]:
        """Gibt chunk_ids zurueck die bereits in chunk_datasources existieren."""
        if not chunk_ids:
            return set()
        m = model
        result = set()
        with db_query() as conn:
            for batch, ph in _in_batches(chunk_ids):
                rows = conn.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                ).fetchall()
                result.update(r["chunk_id"] for r in rows)
        return result

    def add_associations(
        self,
        chunk_ids: list[str],
        ds_id: int,
        model: str,
        *,
        conn=None,
        lock_already_held: bool = False,
    ) -> None:
        """INSERT mit ON CONFLICT DO NOTHING fuer alle chunk_ids."""
        if not chunk_ids:
            return
        m = model
        sql = (
            "INSERT INTO chunk_datasources "
            "(chunk_id, datasource_id, embedding_model) VALUES (%s, %s, %s) "
            "ON CONFLICT (chunk_id, datasource_id, embedding_model) DO NOTHING"
        )

        def _do(c):
            if not lock_already_held:
                self.acquire_graph_cleanup_lock(c)
            c.executemany(
                sql,
                [(cid, ds_id, m) for cid in chunk_ids],
            )

        if conn is not None:
            _do(conn)
        else:
            with db_transaction() as c:
                _do(c)

    def filter_unreferenced_chunk_ids(
        self,
        chunk_ids: list[str],
        *,
        conn=None,
    ) -> list[str]:
        """Filtert Chunk-IDs, die ueber alle Keyspaces unreferenziert sind."""
        if not chunk_ids:
            return []
        seen = set()
        candidates = []
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            candidates.append(cid)

        def _do(c):
            unreferenced = []
            for batch, ph in _in_batches(candidates):
                rows = c.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_datasources WHERE chunk_id IN ({ph})",
                    batch,
                ).fetchall()
                referenced = {r["chunk_id"] for r in rows}
                unreferenced.extend(cid for cid in batch if cid not in referenced)
            return unreferenced

        if conn is not None:
            return _do(conn)
        with db_query() as c:
            return _do(c)

    def preview_graph_orphan_chunk_ids_after_datasource_removal(
        self,
        ds_id: int,
        *,
        conn=None,
    ) -> list[str]:
        """Simuliert DS-Entfernung ueber alle Keyspaces fuer Graph-Cleanup."""

        def _do(c):
            rows = c.execute(
                "SELECT DISTINCT chunk_id FROM chunk_datasources WHERE datasource_id = %s",
                (ds_id,),
            ).fetchall()
            candidate_ids = [r["chunk_id"] for r in rows]
            if not candidate_ids:
                return []

            graph_orphans = []
            for batch, ph in _in_batches(candidate_ids):
                rows = c.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_datasources "
                    f"WHERE datasource_id != %s AND chunk_id IN ({ph})",
                    [ds_id, *batch],
                ).fetchall()
                still_referenced = {r["chunk_id"] for r in rows}
                graph_orphans.extend(cid for cid in batch if cid not in still_referenced)
            return graph_orphans

        if conn is not None:
            return _do(conn)
        with db_query() as c:
            return _do(c)

    def remove_datasource(self, ds_id: int, model: str) -> list[str]:
        """Entfernt alle Eintraege einer DS. Gibt orphan chunk_ids zurueck.

        SELECT + DELETE + Orphan-Check in EINER Transaktion.
        """
        m = model
        with db_transaction() as conn:
            # Alle chunk_ids dieser DS holen
            rows = conn.execute(
                "SELECT chunk_id FROM chunk_datasources "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, m),
            ).fetchall()
            chunk_ids = [r["chunk_id"] for r in rows]
            if not chunk_ids:
                return []

            # Eintraege loeschen
            for batch, ph in _in_batches(chunk_ids):
                conn.execute(
                    f"DELETE FROM chunk_datasources "
                    f"WHERE datasource_id = %s AND embedding_model = %s "
                    f"AND chunk_id IN ({ph})",
                    [ds_id, m, *batch],
                )

            # Orphan-Check: Chunks die KEINE andere DS mehr referenziert
            orphans = []
            for batch, ph in _in_batches(chunk_ids):
                still_referenced = conn.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                ).fetchall()
                still_set = {r["chunk_id"] for r in still_referenced}
                orphans.extend(cid for cid in batch if cid not in still_set)

            return orphans

    def preview_datasource_removal(
        self, ds_id: int, model: str
    ) -> tuple[list[str], dict[str, list[int]]]:
        """Read-only Vorschau: Ermittelt Orphans + verbleibende ds_ids OHNE zu loeschen.

        Simuliert das Ergebnis von remove_datasource_full() ohne DB-Aenderung.
        Verwendung: Qdrant-Cleanup VOR DB-Aenderung ausfuehren (Recovery-sicher).

        Returns:
            (orphan_ids, non_orphan_ds_ids_map)
            - orphan_ids: Chunk-IDs die nach Entfernung dieser DS verwaist waeren
            - non_orphan_ds_ids_map: Dict chunk_id -> [ds_id, ...] (ohne ds_id selbst)
        """
        m = model

        with db_query() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunk_datasources "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, m),
            ).fetchall()
            all_chunk_ids = [r["chunk_id"] for r in rows]
            if not all_chunk_ids:
                return [], {}

            orphans = []
            non_orphan_ds_ids: dict[str, list[int]] = {}

            for batch, ph in _in_batches(all_chunk_ids):
                rows = conn.execute(
                    f"SELECT chunk_id, datasource_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                ).fetchall()

                ds_map: dict[str, list[int]] = {}
                for r in rows:
                    ds_map.setdefault(r["chunk_id"], []).append(r["datasource_id"])

                for cid in batch:
                    other_ds = [d for d in ds_map.get(cid, []) if d != ds_id]
                    if other_ds:
                        non_orphan_ds_ids[cid] = other_ds
                    else:
                        orphans.append(cid)

        return orphans, non_orphan_ds_ids

    def delete_datasource_entries(self, ds_id: int, model: str) -> int:
        """Loescht alle chunk_datasources-Eintraege fuer eine Datasource.

        Nur die DELETE-Operation, ohne Orphan-Analyse.
        Fuer den Fall, dass die Analyse bereits separat via
        preview_datasource_removal() durchgefuehrt wurde.

        Returns:
            Anzahl geloeschter Eintraege.
        """
        m = model
        with db_transaction() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunk_datasources "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, m),
            ).fetchall()
            all_chunk_ids = [r["chunk_id"] for r in rows]
            if not all_chunk_ids:
                return 0
            deleted = 0
            for batch, ph in _in_batches(all_chunk_ids):
                conn.execute(
                    f"DELETE FROM chunk_datasources "
                    f"WHERE datasource_id = %s AND embedding_model = %s "
                    f"AND chunk_id IN ({ph})",
                    [ds_id, m, *batch],
                )
                deleted += len(batch)
            return deleted

    def remove_datasource_full(
        self, ds_id: int, model: str
    ) -> tuple[list[str], dict[str, list[int]]]:
        """Entfernt DS-Zuordnungen und gibt Orphans + verbleibende ds_ids atomar zurueck.

        Kombiniert get_chunk_ids + remove_datasource + get_datasource_ids_for_chunks
        in EINER Transaktion (keine Race Condition zwischen den Schritten).

        Returns:
            (orphan_ids, non_orphan_ds_ids_map)
            - orphan_ids: Chunk-IDs ohne weitere DS-Zuordnung (zum Loeschen in Qdrant)
            - non_orphan_ds_ids_map: Dict chunk_id -> [ds_id, ...] fuer Chunks die
              noch anderen DS zugeordnet sind (zum Update in Qdrant)
        """
        m = model

        with db_transaction() as conn:
            # 1. Alle chunk_ids dieser DS erfassen
            rows = conn.execute(
                "SELECT chunk_id FROM chunk_datasources "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, m),
            ).fetchall()
            all_chunk_ids = [r["chunk_id"] for r in rows]
            if not all_chunk_ids:
                return [], {}

            # 2. DS-Zuordnungen loeschen
            for batch, ph in _in_batches(all_chunk_ids):
                conn.execute(
                    f"DELETE FROM chunk_datasources "
                    f"WHERE datasource_id = %s AND embedding_model = %s "
                    f"AND chunk_id IN ({ph})",
                    [ds_id, m, *batch],
                )

            # 3. Orphan-Check + verbleibende ds_ids in einem Durchgang
            orphans = []
            non_orphan_ds_ids: dict[str, list[int]] = {}

            for batch, ph in _in_batches(all_chunk_ids):
                rows = conn.execute(
                    f"SELECT chunk_id, datasource_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                ).fetchall()

                # Noch referenzierte Chunks mit ihren DS-IDs sammeln
                still_map: dict[str, list[int]] = {}
                for r in rows:
                    still_map.setdefault(r["chunk_id"], []).append(r["datasource_id"])

                for cid in batch:
                    if cid in still_map:
                        non_orphan_ds_ids[cid] = still_map[cid]
                    else:
                        orphans.append(cid)

        return orphans, non_orphan_ds_ids

    def remove_chunks_for_datasource(
        self,
        ds_id: int,
        model: str,
        chunk_ids: list[str],
        conn=None,
    ) -> tuple[list[str], list[str]]:
        """Entfernt Zuordnungen fuer bestimmte Chunks einer DS mit Orphan-Check.

        Die chunk_datasources-Tabelle hat keine 'source'-Spalte
        (nur chunk_id, datasource_id, embedding_model). Der Aufrufer muss
        die chunk_ids VORHER ueber store.get_ids_by_sources() aufloesen.

        Returns:
            (orphan_chunk_ids, non_orphan_chunk_ids)
        """
        if not chunk_ids:
            return [], []
        m = model

        def _do(c):
            all_orphan = []
            all_non_orphan = []
            for batch, ph in _in_batches(chunk_ids):
                # DELETE: DS-Zuordnung entfernen
                c.execute(
                    f"DELETE FROM chunk_datasources "
                    f"WHERE chunk_id IN ({ph}) "
                    f"AND datasource_id = %s AND embedding_model = %s",
                    [*batch, ds_id, m],
                )
                # SELECT DISTINCT: welche Chunks haben noch andere DS?
                rows = c.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_datasources "
                    f"WHERE chunk_id IN ({ph}) AND embedding_model = %s",
                    [*batch, m],
                ).fetchall()
                still_referenced = {r["chunk_id"] for r in rows}
                for cid in batch:
                    if cid in still_referenced:
                        all_non_orphan.append(cid)
                    else:
                        all_orphan.append(cid)
            return all_orphan, all_non_orphan

        if conn is not None:
            return _do(conn)
        else:
            with db_transaction() as c:
                return _do(c)

    def get_chunk_ids(self, ds_id: int, model: str) -> list[str]:
        """Gibt alle chunk_ids zurueck die dieser DS zugeordnet sind."""
        m = model
        return _chunk_ds_repo.get_chunks_for_datasource(ds_id, m)

    def delete_by_chunk_ids(self, chunk_ids: list[str], model: str, conn=None) -> None:
        """Loescht alle DS-Zuordnungen fuer gegebene chunk_ids.

        Optional: externe Connection fuer kombinierte Transaktionen.
        """
        if not chunk_ids:
            return
        m = model

        def _do(c):
            for batch, ph in _in_batches(chunk_ids):
                c.execute(
                    f"DELETE FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                )

        if conn is not None:
            _do(conn)
        else:
            with db_transaction() as c:
                _do(c)

    def get_other_datasources(self, chunk_ids: list[str], exclude_ds: int, model: str) -> set[int]:
        """Gibt DS-IDs zurueck die chunk_ids referenzieren (ohne exclude_ds)."""
        if not chunk_ids:
            return set()
        m = model
        result = set()
        with db_query() as conn:
            for batch, ph in _in_batches(chunk_ids):
                rows = conn.execute(
                    f"SELECT DISTINCT datasource_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND datasource_id != %s "
                    f"AND chunk_id IN ({ph})",
                    [m, exclude_ds, *batch],
                ).fetchall()
                result.update(r["datasource_id"] for r in rows)
        return result

    def get_datasource_ids_for_chunks(
        self, chunk_ids: list[str], model: str
    ) -> dict[str, list[int]]:
        """Gibt ds_ids fuer mehrere Chunks zurueck (batched).

        Returns:
            Dict chunk_id -> [ds_id_1, ds_id_2, ...].
            Chunks ohne Zuordnung fehlen im Dict.
        """
        if not chunk_ids:
            return {}
        m = model
        result: dict[str, list[int]] = {}
        with db_query() as conn:
            for batch, ph in _in_batches(chunk_ids):
                rows = conn.execute(
                    f"SELECT chunk_id, datasource_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                ).fetchall()
                for r in rows:
                    result.setdefault(r["chunk_id"], []).append(r["datasource_id"])
        return result

    def get_datasource_ids_for_chunk(self, chunk_id: str, model: str) -> set[int]:
        """Gibt ds_ids fuer einen einzelnen Chunk zurueck (fuer Repair)."""
        m = model
        with db_query() as conn:
            rows = conn.execute(
                "SELECT datasource_id FROM chunk_datasources "
                "WHERE chunk_id = %s AND embedding_model = %s",
                (chunk_id, m),
            ).fetchall()
        return {r["datasource_id"] for r in rows}

    def get_all_ds_ids_grouped(
        self, model: str, batch_size: int = 1000
    ) -> Generator[dict[str, list[int]], None, None]:
        """Iteriert ueber alle Chunks mit ds_ids (fuer Migration).

        Yields:
            Dict chunk_id -> [ds_id_1, ds_id_2, ...] in Batches.
        """
        m = model
        offset = 0
        while True:
            with db_query() as conn:
                cid_rows = conn.execute(
                    "SELECT DISTINCT chunk_id FROM chunk_datasources "
                    "WHERE embedding_model = %s ORDER BY chunk_id LIMIT %s OFFSET %s",
                    (m, batch_size, offset),
                ).fetchall()
            if not cid_rows:
                break
            batch_ids = [r["chunk_id"] for r in cid_rows]
            yield self.get_datasource_ids_for_chunks(batch_ids, model)
            offset += batch_size

    def filter_allowed(self, chunk_ids: list[str], ds_ids: list[int], model: str) -> set[str]:
        """Punkt-Lookup: Gibt nur chunk_ids zurueck die einem erlaubten ds_id zugeordnet sind.

        Fuer BM25/ES/Graph Post-Filter (max ~20-40 Chunks pro Query).
        Eine SQL-Query fuer alle chunk_ids (batched).
        """
        if not chunk_ids or not ds_ids:
            return set()
        m = model
        result = set()
        d_ph = ",".join(["%s"] * len(ds_ids))
        with db_query() as conn:
            for c_batch, c_ph in _in_batches(chunk_ids):
                rows = conn.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s "
                    f"AND chunk_id IN ({c_ph}) "
                    f"AND datasource_id IN ({d_ph})",
                    [m, *c_batch, *list(ds_ids)],
                ).fetchall()
                result.update(r["chunk_id"] for r in rows)
        return result

    def preview_chunk_removal_for_datasource(
        self,
        ds_id: int,
        model: str,
        chunk_ids: list[str],
    ) -> tuple[list[str], dict[str, list[int]]]:
        """Read-only Vorschau fuer datei-/subset-basiertes Cleanup.

        Wie preview_datasource_removal(), aber nur fuer eine Teilmenge
        von chunk_ids statt fuer ALLE Chunks einer DS.

        Returns:
            (orphan_ids, non_orphan_ds_ids_map)
            - orphan_ids: Chunk-IDs die nach Entfernung dieser DS-Eintraege
              verwaist waeren
            - non_orphan_ds_ids_map: Dict chunk_id -> [restliche_ds_ids]

        Keine DB-Mutation.
        """
        if not chunk_ids:
            return [], {}

        m = model
        orphans = []
        non_orphan_ds_ids: dict[str, list[int]] = {}

        with db_query() as conn:
            for batch, ph in _in_batches(chunk_ids):
                rows = conn.execute(
                    f"SELECT chunk_id, datasource_id FROM chunk_datasources "
                    f"WHERE embedding_model = %s AND chunk_id IN ({ph})",
                    [m, *batch],
                ).fetchall()

                ds_map: dict[str, list[int]] = {}
                for r in rows:
                    ds_map.setdefault(r["chunk_id"], []).append(r["datasource_id"])

                for cid in batch:
                    other_ds = [d for d in ds_map.get(cid, []) if d != ds_id]
                    if other_ds:
                        non_orphan_ds_ids[cid] = other_ds
                    else:
                        orphans.append(cid)

        return orphans, non_orphan_ds_ids

    def count_by_datasource(self, ds_id: int, model: str) -> int:
        """Zaehlt Chunk-Zuordnungen fuer eine DS."""
        m = model
        with db_query() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM chunk_datasources "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, m),
            ).fetchone()
        return row["cnt"] if row else 0

    def distinct_models(self) -> list[str]:
        """Liefert alle in chunk_datasources vorhandenen Embedding-Modelle."""
        with db_query() as conn:
            rows = conn.execute(
                "SELECT DISTINCT embedding_model FROM chunk_datasources "
                "WHERE embedding_model != '' ORDER BY embedding_model"
            ).fetchall()
        return [r["embedding_model"] for r in rows]

    def counts_by_datasource_all_models(
        self,
        ds_id: int,
        models: list[str],
    ) -> dict[str, int]:
        """Chunk-Counts fuer eine DS ueber mehrere Modelle in einem Query."""
        if not models:
            return {}
        placeholders = ",".join("%s" for _ in models)
        with db_query() as conn:
            rows = conn.execute(
                "SELECT embedding_model, COUNT(*) AS cnt "
                "FROM chunk_datasources "
                f"WHERE datasource_id = %s AND embedding_model IN ({placeholders}) "
                "GROUP BY embedding_model",
                [ds_id, *list(models)],
            ).fetchall()
        result = {m: 0 for m in models}
        for r in rows:
            if r["embedding_model"] in result:
                result[r["embedding_model"]] = r["cnt"]
        return result
