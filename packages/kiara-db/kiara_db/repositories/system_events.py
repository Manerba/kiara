"""Repository fuer `system_events` — strukturierte Security-Events.

Wird von der `@admin_function` `system.events.list` und dem CSV-Export-Router
gleichermassen verwendet (Shared-Repository-Regel aus TODO V18 Abschnitt 4).
Filter-Vertrag ist mit `_coerce_system_event_filters` vereinheitlicht, damit
Frontend, AdminFunction-Schema und CSV-Endpoint identische Parameter nutzen.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from kiara_db.connection import db_query, db_transaction

_SORT_ALLOWLIST = ("ts", "level", "event_type", "reason", "user_name", "ip")
_SEVERITY_ORDER = {"INFO": 10, "WARNING": 20, "ERROR": 30, "CRITICAL": 40}


def _split_csv_list(value: Any) -> list[str]:
    """Akzeptiert Liste, wiederholte Query-Keys oder komma-separierten String."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [part.strip() for part in str(value).split(",")]
    return [item for item in items if item]


def _coerce_datetime(value: Any) -> Any:
    """Laesst datetime durch, parst ISO-Strings. None/ungueltig → None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _coerce_system_event_filters(params: dict | None) -> dict:
    """Normalisiert oeffentliche Filter-Parameter in Repository-Keys.

    Akzeptiert `event_types`/`exclude_event_types` als Liste, wiederholte
    Query-Keys oder komma-separierten String. Leere Strings/Listen werden
    als "kein Filter" interpretiert. Unbekannte Keys werden ignoriert.
    """
    if not isinstance(params, dict):
        return {}

    out: dict[str, Any] = {}
    since = _coerce_datetime(params.get("since"))
    if since is not None:
        out["since"] = since
    until = _coerce_datetime(params.get("until"))
    if until is not None:
        out["until"] = until

    min_level = params.get("min_level")
    if isinstance(min_level, str):
        level = min_level.strip().upper()
        if level in _SEVERITY_ORDER:
            out["min_level"] = level

    event_types = _split_csv_list(params.get("event_types"))
    if event_types:
        out["event_types"] = event_types
    exclude_event_types = _split_csv_list(params.get("exclude_event_types"))
    if exclude_event_types:
        out["exclude_event_types"] = exclude_event_types

    reason = params.get("reason")
    if isinstance(reason, str) and reason.strip():
        out["reason"] = reason.strip()

    search = params.get("search")
    if isinstance(search, str) and search.strip():
        out["search"] = search.strip()

    return out


def _allowed_levels(min_level: str) -> list[str]:
    threshold = _SEVERITY_ORDER.get(min_level.upper(), 10)
    return [name for name, rank in _SEVERITY_ORDER.items() if rank >= threshold]


class SystemEventRepository:
    """Eigenes Repository (nicht BaseRepository, weil Filter-Semantik speziell).

    Alle Methoden akzeptieren `conn=None` fuer Cross-Table-Transaktionen.
    SQL-Injection-Schutz: parameterisierte Queries + Sort-Allowlist.
    """

    table = "system_events"

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

    def _build_where(self, filters: dict) -> tuple[str, list[Any]]:
        """Erzeugt WHERE-Clause + Param-Liste.

        NULL-Guardrail: `exclude_event_types` wird NULL-sicher uebersetzt,
        damit unparsbare Events (`event_type IS NULL`) nicht verschwinden.
        """
        clauses: list[str] = []
        params: list[Any] = []

        since = filters.get("since")
        if since is not None:
            clauses.append("ts >= %s")
            params.append(since)
        until = filters.get("until")
        if until is not None:
            clauses.append("ts <= %s")
            params.append(until)

        min_level = filters.get("min_level")
        if isinstance(min_level, str):
            allowed = _allowed_levels(min_level)
            if allowed:
                clauses.append("level = ANY(%s)")
                params.append(allowed)

        event_types = filters.get("event_types")
        if event_types:
            clauses.append("event_type = ANY(%s)")
            params.append(list(event_types))

        exclude_event_types = filters.get("exclude_event_types")
        if exclude_event_types:
            clauses.append("(event_type IS NULL OR event_type <> ALL(%s))")
            params.append(list(exclude_event_types))

        reason = filters.get("reason")
        if isinstance(reason, str) and reason:
            clauses.append("reason = %s")
            params.append(reason)

        search = filters.get("search")
        if isinstance(search, str) and search:
            clauses.append("message ILIKE %s")
            params.append(f"%{search}%")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        sort_field: str = "ts",
        sort_dir: str = "desc",
        limit_cap: int = 500,
        conn=None,
        **filters,
    ) -> list[dict]:
        """Liefert gefilterte Events. `**filters` steht am Ende (Python-Signatur-Regel)."""
        if sort_field not in _SORT_ALLOWLIST:
            sort_field = "ts"
        direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
        effective_limit = max(1, min(int(limit), int(limit_cap)))
        effective_offset = max(0, int(offset))

        where, params = self._build_where(filters)
        sql = (
            f"SELECT * FROM {self.table}{where} "
            f"ORDER BY {sort_field} {direction} LIMIT %s OFFSET %s"
        )
        params_with_paging = [*params, effective_limit, effective_offset]
        with self._read(conn) as db:
            rows = db.execute(sql, params_with_paging).fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, event_id: int, *, conn=None) -> dict | None:
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT * FROM {self.table} WHERE id = %s",
                (event_id,),
            ).fetchone()
            return dict(row) if row else None

    def count(self, *, conn=None, **filters) -> int:
        where, params = self._build_where(filters)
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT COUNT(*) AS c FROM {self.table}{where}",
                params,
            ).fetchone()
            return int(row["c"]) if row else 0

    def distinct_reasons(self, since=None, *, conn=None) -> list[str]:
        clauses = ["reason IS NOT NULL"]
        params: list[Any] = []
        since_dt = _coerce_datetime(since)
        if since_dt is not None:
            clauses.append("ts >= %s")
            params.append(since_dt)
        sql = (
            f"SELECT DISTINCT reason FROM {self.table} "
            f"WHERE {' AND '.join(clauses)} ORDER BY reason"
        )
        with self._read(conn) as db:
            rows = db.execute(sql, params).fetchall()
            return [r["reason"] for r in rows if r["reason"]]

    def distinct_event_types(self, since=None, *, conn=None) -> list[str]:
        clauses = ["event_type IS NOT NULL"]
        params: list[Any] = []
        since_dt = _coerce_datetime(since)
        if since_dt is not None:
            clauses.append("ts >= %s")
            params.append(since_dt)
        sql = (
            f"SELECT DISTINCT event_type FROM {self.table} "
            f"WHERE {' AND '.join(clauses)} ORDER BY event_type"
        )
        with self._read(conn) as db:
            rows = db.execute(sql, params).fetchall()
            return [r["event_type"] for r in rows if r["event_type"]]

    def count_older_than(self, days: int, *, conn=None) -> int:
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT COUNT(*) AS c FROM {self.table} "
                f"WHERE ts < NOW() - (%s * INTERVAL '1 day')",
                (int(days),),
            ).fetchone()
            return int(row["c"]) if row else 0

    def cleanup_older_than(self, days: int, *, conn=None) -> int:
        with self._write(conn) as db:
            result = db.execute(
                f"DELETE FROM {self.table} WHERE ts < NOW() - (%s * INTERVAL '1 day')",
                (int(days),),
            )
            rowcount = getattr(result, "rowcount", None)
            if rowcount is None:
                # Fallback: SELECT COUNT vor dem Delete ist zu aufwaendig,
                # manche DB-Wrapper geben nach DELETE keine rowcount zurueck.
                return 0
            return int(rowcount)
