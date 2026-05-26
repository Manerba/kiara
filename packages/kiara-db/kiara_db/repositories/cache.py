"""Cache-Repository (Denormalisierte Cache-Tabellen).

Verwaltet: cache_datasources, cache_jobs, cache_vectordb.
Erbt NICHT von BaseRepository (3 Tabellen, Bulk-Refresh).

Ersetzt SQL in: web/cache_refresh.py (20 execute-Aufrufe)
"""

from __future__ import annotations

import json
from contextlib import contextmanager

from kiara_db.connection import db_query, db_transaction


class CacheRepository:
    """Bulk-Refresh fuer denormalisierte Cache-Tabellen."""

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

    # ----- Datasource Cache -----

    def get_cached_datasources(self, *, conn=None) -> list[dict]:
        """Liest alle gecachten Datasource-Payloads."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT payload_json FROM cache_datasources ORDER BY datasource_id",
            ).fetchall()
            # JSONB: psycopg2 liefert dict direkt; Fallback fuer Alt-Pfade
            return [
                r["payload_json"]
                if isinstance(r["payload_json"], dict)
                else json.loads(r["payload_json"] or "{}")
                for r in rows
            ]

    def refresh_datasources(
        self,
        entries: list[dict],
        *,
        conn=None,
    ) -> None:
        """Ersetzt den gesamten DS-Cache."""
        with self._write(conn) as db:
            db.execute("DELETE FROM cache_datasources")
            if not entries:
                return
            params_list = [
                (entry["id"], json.dumps(entry, ensure_ascii=False)) for entry in entries
            ]
            db.executemany(
                """INSERT INTO cache_datasources
                   (datasource_id, payload_json)
                   VALUES (%s, %s)""",
                params_list,
            )

    # ----- Job Cache -----

    def get_cached_jobs(
        self,
        ds_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        *,
        conn=None,
    ) -> tuple[list[dict], int]:
        """Liest gecachte Jobs mit optionalem Filter."""
        conditions = []
        params: list = []
        if ds_id is not None:
            conditions.append("datasource_id = %s")
            params.append(ds_id)
        if status:
            status_list = [s.strip() for s in status.split(",")]
            if len(status_list) == 1:
                conditions.append("status = %s")
                params.append(status_list[0])
            else:
                ph = ", ".join(["%s"] * len(status_list))
                conditions.append(f"status IN ({ph})")
                params.extend(status_list)

        where = ""
        if conditions:
            where = " WHERE " + " AND ".join(conditions)

        with self._read(conn) as db:
            count_row = db.execute(
                f"SELECT COUNT(*) AS count FROM cache_jobs{where}",
                params,
            ).fetchone()
            total = count_row["count"] if count_row else 0

            rows = db.execute(
                f"SELECT payload_json FROM cache_jobs{where} "
                "ORDER BY job_id DESC LIMIT %s OFFSET %s",
                [*params, limit, offset],
            ).fetchall()

        # JSONB: psycopg2 liefert dict direkt; Fallback fuer Alt-Pfade
        return (
            [
                r["payload_json"]
                if isinstance(r["payload_json"], dict)
                else json.loads(r["payload_json"] or "{}")
                for r in rows
            ],
            total,
        )

    def refresh_jobs(self, entries: list[dict], *, conn=None) -> None:
        """Ersetzt den gesamten Job-Cache."""
        with self._write(conn) as db:
            db.execute("DELETE FROM cache_jobs")
            if not entries:
                return
            params_list = [
                (
                    entry["id"],
                    entry.get("datasource_id"),
                    entry.get("status"),
                    json.dumps(entry, ensure_ascii=False),
                )
                for entry in entries
            ]
            db.executemany(
                """INSERT INTO cache_jobs
                   (job_id, datasource_id, status, payload_json)
                   VALUES (%s, %s, %s, %s)""",
                params_list,
            )
