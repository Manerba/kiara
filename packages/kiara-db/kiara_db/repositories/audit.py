"""Audit-Repositories (Query-Log, Cloud-Audit, Anonymisierungs-Audit).

Ersetzt SQL in: web/api_compliance.py (25), web/api_chat.py (Audit-Insert),
                cloud/audit.py
"""

from __future__ import annotations

from kiara_db.base import BaseRepository


class _AuditTrailMixin:
    """Tamper-Resistance: Audit-Eintraege duerfen nie geloescht werden.

    Compliance/Forensik erfordert append-only Audit-Trails. Repositories,
    die diesen Mixin einbinden, blockieren delete() und delete_where()
    aus BaseRepository.
    """

    def delete(self, *args, **kwargs):
        raise NotImplementedError(
            "Audit-Eintraege duerfen nicht geloescht werden (Tamper-Resistance)",
        )

    def delete_where(self, *args, **kwargs):
        raise NotImplementedError(
            "Audit-Eintraege duerfen nicht geloescht werden (Tamper-Resistance)",
        )


class QueryLogRepository(_AuditTrailMixin, BaseRepository):
    table = "query_log"
    default_sort = "created_at DESC"

    SORT_MAP = {
        "id": "q.id",
        "created_at": "q.created_at",
        "username": "u.username",
        "chain_type": "q.chain_type",
        "routed_to": "q.routed_to",
        "duration_ms": "q.duration_ms",
    }

    def list_with_joins(
        self,
        user_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        routed_to: str | None = None,
        chain_type: str | None = None,
        search_text: str | None = None,
        sort_field: str = "created_at",
        sort_dir: str = "desc",
        page: int = 1,
        page_size: int = 50,
        *,
        conn=None,
    ) -> dict:
        """Paginierte Audit-Liste mit User/Chat-JOINs.

        Nutzt eigene WHERE-Logik statt _build_where(), da Table-Prefixes
        (q.*, u.*) und OR-Bedingungen benoetigt werden.

        date_from/date_to werden als Datums-Strings im Format YYYY-MM-DD
        erwartet (analog `<input type="date">`). date_to wird als
        Tagesende-inklusiv interpretiert (`%s::date + interval '1 day'`),
        date_from als untere Grenze inklusiv. Zeitstempel mit Uhrzeit
        werden bei date_to auf Tagesgrenze getrimmt.
        """
        conditions = []
        params: list = []

        if user_id is not None:
            conditions.append("q.user_id = %s")
            params.append(user_id)
        if date_from:
            conditions.append("q.created_at >= %s")
            params.append(date_from)
        if date_to:
            conditions.append("q.created_at < %s::date + interval '1 day'")
            params.append(date_to)
        if routed_to:
            conditions.append("q.routed_to = %s")
            params.append(routed_to)
        if chain_type:
            conditions.append("q.chain_type = %s")
            params.append(chain_type)
        if search_text:
            conditions.append("(q.question_excerpt ILIKE %s OR q.answer_excerpt ILIKE %s)")
            like = f"%{search_text}%"
            params.extend([like, like])

        where = ""
        if conditions:
            where = " WHERE " + " AND ".join(conditions)

        page = max(1, page)
        page_size = max(1, page_size)
        sql_sort = self.SORT_MAP.get(sort_field, "q.created_at")
        sql_dir = "ASC" if sort_dir.lower() == "asc" else "DESC"
        offset = (page - 1) * page_size

        with self._read(conn) as db:
            total = db.execute(
                f"SELECT COUNT(*) AS count FROM query_log q"
                f" LEFT JOIN users u ON q.user_id = u.id{where}",
                params,
            ).fetchone()["count"]

            rows = db.execute(
                f"""SELECT q.id, q.user_id, q.question_hash,
                           q.question_excerpt, q.answer_excerpt,
                           q.routed_to, q.model_used, q.sources_count,
                           q.tools_used, q.chunks_retrieved,
                           q.entities_anonymized, q.duration_ms,
                           q.chat_id, q.chain_type, q.trace_id,
                           q.created_at,
                           u.username, c.title AS chat_title
                    FROM query_log q
                    LEFT JOIN users u ON q.user_id = u.id
                    LEFT JOIN chats c ON q.chat_id = c.id
                    {where}
                    ORDER BY {sql_sort} {sql_dir}
                    LIMIT %s OFFSET %s""",
                [*params, page_size, offset],
            ).fetchall()

        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return {
            "data": [dict(r) for r in rows],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
            },
        }

    def get_detail_with_joins(self, audit_id: int, *, conn=None) -> dict | None:
        """Einzelner Audit-Eintrag mit User- und Chat-JOINs."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT q.*, u.username, c.title AS chat_title
                   FROM query_log q
                   LEFT JOIN users u ON q.user_id = u.id
                   LEFT JOIN chats c ON q.chat_id = c.id
                   WHERE q.id = %s""",
                (audit_id,),
            ).fetchone()
        return dict(row) if row else None

    def count_since(self, since: str, *, routed_to: str | None = None, conn=None) -> int:
        """Anzahl der Queries seit einem Datum, optional nach routed_to gefiltert."""
        if routed_to:
            sql = (
                "SELECT COUNT(*) AS count FROM query_log WHERE created_at >= %s AND routed_to = %s"
            )
            params = [since, routed_to]
        else:
            sql = "SELECT COUNT(*) AS count FROM query_log WHERE created_at >= %s"
            params = [since]
        with self._read(conn) as db:
            row = db.execute(sql, params).fetchone()
            return row["count"] if row else 0

    def count_distinct_users_since(self, since: str, *, conn=None) -> int:
        """Anzahl der eindeutigen User seit einem Datum."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT COUNT(DISTINCT user_id) AS count FROM query_log WHERE created_at >= %s",
                (since,),
            ).fetchone()
            return row["count"] if row else 0


class CloudAuditLogRepository(_AuditTrailMixin, BaseRepository):
    table = "cloud_audit_log"
    default_sort = "created_at DESC"

    def find_matching_entry(
        self,
        correlation_id: str,
        *,
        conn=None,
    ) -> dict | None:
        """Findet den Cloud-Audit-Eintrag ueber die Korrelations-ID."""
        sql = (
            "SELECT question_anonymized, answer_anonymized,"
            "       provider, model, tokens_in, tokens_out,"
            "       entities_anonymized AS cloud_entities, context_sources"
            " FROM cloud_audit_log"
            " WHERE correlation_id = %s"
            " LIMIT 1"
        )
        with self._read(conn) as db:
            row = db.execute(sql, (correlation_id,)).fetchone()
        return dict(row) if row else None

    def list_with_user(
        self,
        user_id: int | None = None,
        date_from: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        conn=None,
    ) -> list[dict]:
        """Cloud-Audit-Log mit User-JOINs (fuer cloud/audit.py get_log)."""
        conditions: list[str] = []
        params: list = []

        if user_id is not None:
            conditions.append("c.user_id = %s")
            params.append(user_id)
        if date_from:
            conditions.append("c.created_at >= %s")
            params.append(date_from)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        params.extend([limit, offset])

        with self._read(conn) as db:
            rows = db.execute(
                f"""SELECT c.*, u.username
                    FROM cloud_audit_log c
                    LEFT JOIN users u ON c.user_id = u.id
                    {where}
                    ORDER BY c.created_at DESC
                    LIMIT %s OFFSET %s""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_aggregate_stats(self, since: str, *, conn=None) -> dict | None:
        """Aggregierte Statistiken seit einem Datum."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT COUNT(*) AS request_count,
                          COALESCE(SUM(tokens_in), 0) AS total_tokens_in,
                          COALESCE(SUM(tokens_out), 0) AS total_tokens_out,
                          COALESCE(SUM(entities_anonymized), 0) AS total_entities,
                          COALESCE(AVG(duration_ms), 0) AS avg_duration_ms
                   FROM cloud_audit_log
                   WHERE created_at >= %s""",
                (since,),
            ).fetchone()
        return dict(row) if row else None

    def get_daily_counts(self, since: str, *, conn=None) -> list[dict]:
        """Requests pro Tag seit einem Datum."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DATE(created_at) AS day, COUNT(*) AS count
                   FROM cloud_audit_log
                   WHERE created_at >= %s
                   GROUP BY DATE(created_at)
                   ORDER BY day""",
                (since,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_model(self, since: str, *, conn=None) -> list[dict]:
        """Aufschluesselung nach Provider/Modell seit einem Datum."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT provider, model,
                          COUNT(*) AS count,
                          SUM(tokens_in) AS tokens_in,
                          SUM(tokens_out) AS tokens_out
                   FROM cloud_audit_log
                   WHERE created_at >= %s
                   GROUP BY provider, model
                   ORDER BY count DESC, provider, model""",
                (since,),
            ).fetchall()
        return [dict(r) for r in rows]


class AuditAnonymizationRepository(_AuditTrailMixin, BaseRepository):
    table = "audit_anonymization"
    default_sort = "timestamp DESC"

    def total_entities(self, *, conn=None) -> int:
        """Gesamtzahl anonymisierter Entitaeten."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT COALESCE(SUM(entity_count), 0) AS total FROM audit_anonymization"
            ).fetchone()
            return row["total"] if row else 0

    def all_entity_types(self, *, conn=None) -> list[dict]:
        """Alle entity_types-JSONs (nicht-leer) fuer Kategorie-Aufschluesselung."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT entity_types FROM audit_anonymization WHERE entity_types != '{}'"
            ).fetchall()
            return [dict(r) for r in rows]

    def last_timestamp(self, *, conn=None) -> str | None:
        """Zeitpunkt der letzten Anonymisierung."""
        with self._read(conn) as db:
            row = db.execute("SELECT MAX(timestamp) AS last_ts FROM audit_anonymization").fetchone()
            return row["last_ts"] if row and row["last_ts"] else None

    def stats_since(self, since: str, *, conn=None) -> dict:
        """Aggregierte Statistik seit einem Datum (total + request count)."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT COALESCE(SUM(entity_count), 0) AS total_30d,
                          COUNT(*) AS requests_30d
                   FROM audit_anonymization
                   WHERE timestamp >= %s""",
                (since,),
            ).fetchone()
            return dict(row) if row else {"total_30d": 0, "requests_30d": 0}

    def daily_since(self, since: str, *, conn=None) -> list[dict]:
        """Taegliche Anonymisierungs-Statistik seit einem Datum."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DATE(timestamp) AS day,
                          SUM(entity_count) AS entities,
                          COUNT(*) AS requests
                   FROM audit_anonymization
                   WHERE timestamp >= %s
                   GROUP BY DATE(timestamp)
                   ORDER BY day""",
                (since,),
            ).fetchall()
            return [dict(r) for r in rows]

    def by_route(self, *, conn=None) -> list[dict]:
        """Route-Verteilung der Anonymisierung."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT route, COUNT(*) AS count,
                          SUM(entity_count) AS entities
                   FROM audit_anonymization
                   GROUP BY route
                   ORDER BY count DESC
                   LIMIT 100"""
            ).fetchall()
            return [dict(r) for r in rows]


class PromptDefenseRepository(BaseRepository):
    """CRUD fuer prompt_defense_events."""

    table = "prompt_defense_events"
    default_sort = "timestamp DESC"

    def create(
        self,
        *,
        user_id,
        source_type,
        matched_patterns,
        removed_snippets,
        chars_removed,
        original_length,
        severity,
        question_hash=None,
        chat_id=None,
        document_source=None,
    ):
        with self._write() as db:
            db.execute(
                """INSERT INTO prompt_defense_events
                   (user_id, source_type, matched_patterns, removed_snippets,
                    chars_removed, original_length, severity, question_hash,
                    chat_id, document_source)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    user_id,
                    source_type,
                    matched_patterns,
                    removed_snippets,
                    chars_removed,
                    original_length,
                    severity,
                    question_hash,
                    chat_id,
                    document_source,
                ),
            )

    def list_filtered(
        self, *, page=1, page_size=50, severity=None, source_type=None, date_from=None, date_to=None
    ):
        """Paginierte Abfrage mit optionalen Filtern."""
        page = max(1, page)
        page_size = max(1, page_size)
        conditions = []
        params = []
        if severity:
            conditions.append("severity = %s")
            params.append(severity)
        if source_type:
            conditions.append("source_type = %s")
            params.append(source_type)
        if date_from:
            conditions.append("timestamp >= %s")
            params.append(date_from)
        if date_to:
            conditions.append("timestamp < %s::date + interval '1 day'")
            params.append(date_to)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        offset = (page - 1) * page_size

        with self._read() as db:
            count_row = db.execute(
                f"SELECT COUNT(*) AS total FROM prompt_defense_events{where}",
                params,
            ).fetchone()
            total = count_row["total"] if count_row else 0

            rows = db.execute(
                f"""SELECT * FROM prompt_defense_events{where}
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s""",
                [*params, page_size, offset],
            ).fetchall()

        return {
            "data": [dict(r) for r in rows],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": (total + page_size - 1) // page_size if page_size else 1,
            },
        }

    def get_stats(self, days=30):
        """Aggregierte Statistiken: Events pro Tag, haeufigste Patterns, Severity-Verteilung."""
        days = int(days)
        with self._read() as db:
            row = db.execute(
                """SELECT COUNT(*) AS total_30d,
                           SUM(CASE WHEN severity = 'hoch' THEN 1 ELSE 0 END) AS high_count
                    FROM prompt_defense_events
                    WHERE timestamp >= NOW() - make_interval(days => %s)""",
                (days,),
            ).fetchone()
            total_30d = row["total_30d"] if row else 0
            high_count = row["high_count"] if row else 0

            source_row = db.execute(
                """SELECT source_type, COUNT(*) AS cnt
                    FROM prompt_defense_events
                    WHERE timestamp >= NOW() - make_interval(days => %s)
                    GROUP BY source_type
                    ORDER BY cnt DESC
                    LIMIT 1""",
                (days,),
            ).fetchone()
            top_source = source_row["source_type"] if source_row else None

            # Top-Pattern aus JSON-Array aggregieren
            pattern_row = db.execute(
                """SELECT val AS pattern, COUNT(*) AS cnt
                    FROM prompt_defense_events,
                         json_array_elements_text(matched_patterns::json) AS val
                    WHERE timestamp >= NOW() - make_interval(days => %s)
                      AND val != 'invisible_chars'
                    GROUP BY val
                    ORDER BY cnt DESC
                    LIMIT 1""",
                (days,),
            ).fetchone()
            top_pattern = pattern_row["pattern"] if pattern_row else None

        return {
            "total_30d": total_30d,
            "high_count": high_count,
            "top_source": top_source,
            "top_pattern": top_pattern,
        }
