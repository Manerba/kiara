"""Ingestion-Job- und FileLog-Repositories.

Ersetzt SQL in: web/api_ingestion_jobs.py (17), web/ingestion_worker.py (37)
"""

from __future__ import annotations

from kiara_db.base import BaseRepository


class IngestionJobRepository(BaseRepository):
    table = "ingestion_jobs"
    default_sort = "id DESC"

    def list_paginated(
        self,
        ds_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        *,
        conn=None,
    ) -> tuple[list[dict], int]:
        """Jobs paginiert mit optionalem DS- und Status-Filter."""
        filters: dict = {}
        if ds_id is not None:
            filters["datasource_id"] = ds_id
        if status:
            status_list = [s.strip() for s in status.split(",")]
            if len(status_list) == 1:
                filters["status"] = status_list[0]
            else:
                filters["status__in"] = status_list
        return self.list_filtered(
            filters,
            limit=limit,
            offset=offset,
            conn=conn,
        )

    def request_action(
        self,
        job_id: int,
        action: str | None,
        *,
        conn=None,
    ) -> bool:
        """Setzt requested_action (cancel/pause/resume/None).

        Nur fuer Jobs mit status IN ('pending', 'initializing', 'running') — verhindert,
        dass abgeschlossene Jobs (success/error/cancelled) nachtraeglich
        ein cancel/pause/resume erhalten.
        """
        count = self.update_where(
            {"id": job_id, "status__in": ["pending", "initializing", "running"]},
            conn=conn,
            requested_action=action,
        )
        return count > 0

    def find_stale(
        self,
        max_age_seconds: int = 3600,
        *,
        conn=None,
    ) -> list[dict]:
        """Findet RUNNING Jobs ohne Heartbeat."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT * FROM ingestion_jobs
                   WHERE status = 'running'
                   AND started_at < NOW() - make_interval(secs => %s)""",
                (max_age_seconds,),
            ).fetchall()
            return [dict(r) for r in rows]

    def claim_next(self, *, conn=None) -> dict | None:
        """Atomar naechsten pending Job claimen.

        Der Job wechselt zuerst auf ``initializing``. Die Datenquelle wird
        erst vom Worker nach erfolgreichem Kontextaufbau auf ``running``
        gesetzt, damit fehlerhafte Claims nicht als stabile Ingestion
        erscheinen.
        """
        with self._write(conn) as db:
            row = db.execute(
                """SELECT j.id FROM ingestion_jobs j
                   JOIN datasources d ON d.id = j.datasource_id
                   WHERE j.status = 'pending'
                   AND d.is_active = true
                   ORDER BY j.queued_at ASC, j.id ASC LIMIT 1
                   FOR UPDATE OF j SKIP LOCKED""",
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE ingestion_jobs SET status = 'initializing', "
                "started_at = NOW() WHERE id = %s",
                (row["id"],),
            )
            claimed = db.execute(
                "SELECT * FROM ingestion_jobs WHERE id = %s",
                (row["id"],),
            ).fetchone()
            return dict(claimed) if claimed else None

    def cleanup_old(self, days: int = 30, *, conn=None) -> int:
        """Loescht abgeschlossene Jobs aelter als N Tage."""
        with self._write(conn) as db:
            db.execute(
                """DELETE FROM ingestion_file_logs WHERE job_id IN (
                       SELECT id FROM ingestion_jobs
                       WHERE status IN ('success', 'error', 'cancelled')
                       AND finished_at < NOW() - make_interval(days => %s)
                   )""",
                (days,),
            )
            cursor = db.execute(
                """DELETE FROM ingestion_jobs
                   WHERE status IN ('success', 'error', 'cancelled')
                   AND finished_at < NOW() - make_interval(days => %s)""",
                (days,),
            )
            return cursor.rowcount


class FileLogRepository(BaseRepository):
    table = "ingestion_file_logs"
    default_sort = "file_name ASC"

    SORT_WHITELIST = {
        "file_name",
        "file_size",
        "duration_ms",
        "chunk_count",
        "status",
        "parser",
    }

    def list_by_job(
        self,
        job_id: int,
        status: str | None = None,
        search: str | None = None,
        sort: str = "file_name",
        direction: str = "asc",
        limit: int = 50,
        offset: int = 0,
        *,
        conn=None,
    ) -> tuple[list[dict], int]:
        """File-Logs eines Jobs mit Filter und Sortierung."""
        if sort not in self.SORT_WHITELIST:
            sort = "file_name"
        if direction not in ("asc", "desc"):
            direction = "asc"
        filters: dict = {"job_id": job_id}
        if status:
            filters["status"] = status
        if search:
            filters["file_name__like"] = search
        return self.list_filtered(
            filters,
            sort=f"{sort} {direction}",
            limit=limit,
            offset=offset,
            conn=conn,
        )

    def aggregate_by_status(self, job_id: int, *, conn=None) -> dict:
        """Zusammenfassung: {status: {count, chunks, avg_ms}}."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT status, COUNT(*) AS count,
                          SUM(chunk_count) AS chunks,
                          AVG(duration_ms) AS avg_ms
                   FROM ingestion_file_logs WHERE job_id = %s
                   GROUP BY status""",
                (job_id,),
            ).fetchall()
            return {r["status"]: dict(r) for r in rows}
