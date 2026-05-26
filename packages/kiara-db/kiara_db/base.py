"""BaseRepository — gemeinsame CRUD-Operationen fuer alle Repositories.

Jede Subklasse setzt mindestens ``table``. Optional: ``pk``, ``default_sort``.

Alle Methoden akzeptieren einen optionalen ``conn``-Parameter (Option B aus
dem Repository-Pattern-Plan). Wird ``conn`` uebergeben, wird die bestehende
Connection/Transaction genutzt — ideal fuer Cross-Table-Transaktionen.
Wird ``conn`` weggelassen, oeffnet die Methode eine eigene Connection.

Beispiel Cross-Table-Transaction::

    with db_transaction() as db:
        job_repo.update(job_id, status="running", conn=db)
        ds_repo.update(ds_id, ingestion_status="running", conn=db)
        # Beide Updates in einer Transaktion

Beispiel Standalone::

    chat_repo.get_by_id(42)  # Oeffnet eigene Connection, schliesst sie danach
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any

from kiara_db.connection import db_query, db_transaction

logger = logging.getLogger(__name__)

# Regex: ein oder mehrere komma-getrennte Spalten, optional mit ASC/DESC
_SAFE_SORT_RE = re.compile(
    r"^[a-zA-Z_]\w*(?:\s+(?:ASC|DESC))?(?:,\s*[a-zA-Z_]\w*(?:\s+(?:ASC|DESC))?)*$",
    re.IGNORECASE,
)

# Regex: gueltiger SQL-Spaltenname (Defense-in-Depth gegen SQL-Injection in Filter-Keys)
_SAFE_COLUMN_RE = re.compile(r"^[a-zA-Z_]\w*$")
_UNSAFE_RAW_SQL_RE = re.compile(
    r"(;|--|/\*|\*/|\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|"
    r"GRANT|REVOKE|CALL|COPY|DO|EXECUTE|PG_SLEEP)\b)",
    re.IGNORECASE,
)


def _validate_sort(order: str) -> str:
    """Prueft ORDER-BY-Klausel gegen SQL-Injection (Defense-in-Depth)."""
    cleaned = order.strip()
    if not _SAFE_SORT_RE.match(cleaned):
        raise ValueError(f"Ungueltige ORDER-BY-Klausel: {order!r}")
    return cleaned


def _validate_column(col: str) -> str:
    """Prueft Spaltennamen gegen SQL-Injection (Defense-in-Depth)."""
    if not _SAFE_COLUMN_RE.match(col):
        raise ValueError(f"Ungueltiger Spaltenname: {col!r}")
    return col


def _validate_raw_sql(expr: str) -> str:
    """Erlaubt nur statische SQL-Ausdruecke ohne Statements/Kommentare."""
    cleaned = expr.strip()
    if not cleaned or _UNSAFE_RAW_SQL_RE.search(cleaned):
        raise ValueError(f"Ungueltiger RawSQL-Ausdruck: {expr!r}")
    return cleaned


# ---------------------------------------------------------------------------
# RawSQL-Marker
# ---------------------------------------------------------------------------


class RawSQL:
    """Marker fuer Raw-SQL-Ausdruecke, die NICHT als Parameter escaped werden.

    Beispiel::

        repo.update(42, updated_at=RawSQL("NOW()"))
        # -> SET updated_at = NOW()   (statt SET updated_at = %s mit 'NOW()')

    SICHERHEITSHINWEIS: Niemals User-Input in RawSQL() verwenden!
    RawSQL-Ausdruecke werden NICHT escaped und sind ein SQL-Injection-Vektor.
    Nur fuer statische SQL-Ausdruecke wie NOW(), COALESCE(...), etc.
    """

    __slots__ = ("expr",)

    def __init__(self, expr: str):
        self.expr = _validate_raw_sql(expr)

    def __repr__(self) -> str:
        return f"RawSQL({self.expr!r})"


# ---------------------------------------------------------------------------
# BaseRepository
# ---------------------------------------------------------------------------


class BaseRepository:
    """Basis fuer alle Repositories mit einer Haupttabelle.

    Repositories die mehrere Tabellen verwalten (PermissionRepo, ConfigRepo,
    CacheRepo, WopiRepo, FileHashRepo) erben NICHT von BaseRepository,
    sondern implementieren eigene Methoden mit explizitem SQL.

    **Option B — externe Connection:**
    Alle Methoden akzeptieren ``conn=None``. Bei ``None`` wird eine eigene
    Connection geoeffnet. Bei uebergebener Connection wird diese genutzt,
    ohne sie zu committen/schliessen (das macht der Aufrufer).
    """

    table: str = ""
    pk: str = "id"
    default_sort: str = "id DESC"

    # =====================================================================
    # Connection-Handling (Option B)
    # =====================================================================

    @contextmanager
    def _read(self, conn=None):
        """Liefert eine Read-Connection. Extern oder eigene."""
        if conn is not None:
            yield conn
        else:
            with db_query() as db:
                yield db

    @contextmanager
    def _write(self, conn=None):
        """Liefert eine Write-Connection mit Transaction. Extern oder eigene."""
        if conn is not None:
            # Externe Connection: kein Commit/Rollback/Close — Aufrufer steuert
            yield conn
        else:
            with db_transaction() as db:
                yield db

    # =====================================================================
    # Read-Operationen
    # =====================================================================

    def get_by_id(self, row_id: int | str, *, conn=None) -> dict | None:
        """Einzelne Zeile anhand des Primaerschluessels."""
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT * FROM {self.table} WHERE {self.pk} = %s",
                (row_id,),
            ).fetchone()
            return dict(row) if row else None

    def exists(self, row_id: int | str, *, conn=None) -> bool:
        """Prueft ob eine Zeile mit der ID existiert."""
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT 1 FROM {self.table} WHERE {self.pk} = %s LIMIT 1",
                (row_id,),
            ).fetchone()
            return row is not None

    def list_all(
        self,
        sort: str | None = None,
        limit: int = 10000,
        *,
        conn=None,
    ) -> list[dict]:
        """Alle Zeilen der Tabelle. Limit als Sicherheitsnetz.

        ``sort`` wird gegen ein sicheres Pattern validiert (Defense-in-Depth).
        Aufrufer SOLLTE den Wert zusaetzlich gegen eine Whitelist pruefen.
        """
        order = _validate_sort(sort or self.default_sort)
        with self._read(conn) as db:
            rows = db.execute(
                f"SELECT * FROM {self.table} ORDER BY {order} LIMIT %s",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_filtered(
        self,
        filters: dict[str, Any] | None = None,
        sort: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        conn=None,
    ) -> tuple[list[dict], int]:
        """Gefilterte + paginierte Liste. Gibt ``(rows, total_count)`` zurueck.

        ``sort`` wird gegen ein sicheres Pattern validiert (Defense-in-Depth).
        Aufrufer SOLLTE den Wert zusaetzlich gegen eine Whitelist pruefen.
        """
        where, params = self._build_where(filters)
        order = _validate_sort(sort or self.default_sort)

        with self._read(conn) as db:
            total = db.execute(
                f"SELECT COUNT(*) AS count FROM {self.table}{where}",
                params,
            ).fetchone()["count"]

            rows = db.execute(
                f"SELECT * FROM {self.table}{where} ORDER BY {order} LIMIT %s OFFSET %s",
                [*params, limit, offset],
            ).fetchall()

            return [dict(r) for r in rows], total

    def count(self, filters: dict[str, Any] | None = None, *, conn=None) -> int:
        """Anzahl der Zeilen (optional gefiltert)."""
        where, params = self._build_where(filters)
        with self._read(conn) as db:
            return db.execute(
                f"SELECT COUNT(*) AS count FROM {self.table}{where}",
                params,
            ).fetchone()["count"]

    # =====================================================================
    # Write-Operationen
    # =====================================================================

    def create(self, *, conn=None, **fields) -> int | str:
        """INSERT und gibt die neue ID zurueck (int bei SERIAL, str bei TEXT-PK).

        Unterstuetzt RawSQL-Werte::

            repo.create(name="Test", created_at=RawSQL("NOW()"))
        """
        cols, placeholders, params = self._split_fields(fields)
        col_str = ", ".join(cols)
        ph_str = ", ".join(placeholders)

        with self._write(conn) as db:
            cursor = db.execute(
                f"INSERT INTO {self.table} ({col_str}) VALUES ({ph_str}) RETURNING {self.pk}",
                params,
            )
            return cursor.fetchone()[self.pk]

    def create_many(self, rows: list[dict], *, conn=None) -> int:
        """Batch-INSERT. Gibt die Anzahl eingefuegter Zeilen zurueck.

        Alle Dicts muessen dieselben Keys haben. RawSQL wird NICHT unterstuetzt
        (executemany kann keine gemischten Formate).
        """
        if not rows:
            return 0
        cols = [_validate_column(c) for c in rows[0]]
        expected_keys = set(rows[0].keys())
        for row in rows[1:]:
            if set(row.keys()) != expected_keys:
                raise ValueError("create_many() erfordert identische Spalten pro Row")
        col_str = ", ".join(cols)
        ph_str = ", ".join(["%s"] * len(cols))

        with self._write(conn) as db:
            db.executemany(
                f"INSERT INTO {self.table} ({col_str}) VALUES ({ph_str})",
                [[r[c] for c in cols] for r in rows],
            )
            return len(rows)

    def update(self, row_id: int | str, *, conn=None, **fields) -> bool:
        """UPDATE und gibt True zurueck wenn Zeile existierte.

        Unterstuetzt RawSQL-Werte::

            repo.update(42, updated_at=RawSQL("NOW()"))
        """
        if not fields:
            return False
        set_parts = []
        params = []
        for k, v in fields.items():
            col = _validate_column(k)
            if isinstance(v, RawSQL):
                set_parts.append(f"{col} = {v.expr}")
            else:
                set_parts.append(f"{col} = %s")
                params.append(v)

        set_clause = ", ".join(set_parts)
        params.append(row_id)

        with self._write(conn) as db:
            cursor = db.execute(
                f"UPDATE {self.table} SET {set_clause} WHERE {self.pk} = %s",
                params,
            )
            return cursor.rowcount > 0

    def delete(self, row_id: int | str, *, conn=None) -> bool:
        """DELETE und gibt True zurueck wenn Zeile existierte."""
        with self._write(conn) as db:
            cursor = db.execute(
                f"DELETE FROM {self.table} WHERE {self.pk} = %s",
                (row_id,),
            )
            return cursor.rowcount > 0

    def update_where(
        self,
        filters: dict[str, Any],
        *,
        conn=None,
        **fields,
    ) -> int:
        """UPDATE mit beliebigen WHERE-Bedingungen. Gibt Anzahl betroffener Zeilen zurueck.

        Atomare Operation — ideal fuer Check-Then-Act ohne Race-Conditions::

            # Statt: row = repo.get(...); repo.update(row["id"], status="x")
            count = repo.update_where({"action_id": aid, "status": "pending"}, status="confirmed")

        Erfordert mindestens einen Filter (Schutz vor versehentlichem UPDATE ALL).
        """
        if not filters:
            raise ValueError("update_where() erfordert mindestens einen Filter")
        if not fields:
            return 0

        set_parts = []
        set_params: list = []
        for k, v in fields.items():
            col = _validate_column(k)
            if isinstance(v, RawSQL):
                set_parts.append(f"{col} = {v.expr}")
            else:
                set_parts.append(f"{col} = %s")
                set_params.append(v)

        where, where_params = self._build_where(filters)

        with self._write(conn) as db:
            cursor = db.execute(
                f"UPDATE {self.table} SET {', '.join(set_parts)}{where}",
                set_params + where_params,
            )
            return cursor.rowcount

    def delete_where(
        self,
        filters: dict[str, Any],
        *,
        conn=None,
    ) -> int:
        """DELETE mit beliebigen Filtern. Gibt Anzahl geloeschter Zeilen zurueck.

        Erfordert mindestens einen Filter (Schutz vor versehentlichem DELETE ALL).
        """
        if not filters:
            raise ValueError("delete_where() erfordert mindestens einen Filter")
        where, params = self._build_where(filters)

        with self._write(conn) as db:
            cursor = db.execute(
                f"DELETE FROM {self.table}{where}",
                params,
            )
            return cursor.rowcount

    def upsert(self, conflict_columns: list[str], *, conn=None, **fields) -> None:
        """INSERT ... ON CONFLICT DO UPDATE.

        Unterstuetzt RawSQL-Werte (konsistent mit create/update).
        """
        cols, placeholders, params = self._split_fields(fields)
        conflict_columns = [_validate_column(c) for c in conflict_columns]

        update_cols = [c for c in cols if c not in conflict_columns]
        update_clause = ", ".join(f"{k} = EXCLUDED.{k}" for k in update_cols)
        conflict = ", ".join(conflict_columns)

        col_str = ", ".join(cols)
        ph_str = ", ".join(placeholders)
        sql = f"INSERT INTO {self.table} ({col_str}) VALUES ({ph_str})"
        if update_clause:
            sql += f" ON CONFLICT ({conflict}) DO UPDATE SET {update_clause}"
        else:
            sql += f" ON CONFLICT ({conflict}) DO NOTHING"

        with self._write(conn) as db:
            db.execute(sql, params)

    # =====================================================================
    # Hilfsmethoden
    # =====================================================================

    @staticmethod
    def _split_fields(fields: dict[str, Any]) -> tuple[list[str], list[str], list]:
        """Trennt Fields in Spalten, Placeholders und Parameter.

        Behandelt RawSQL-Werte: diese werden als SQL-Ausdruck eingesetzt
        statt als parametrisierter Wert.
        """
        cols = []
        placeholders = []
        params = []
        for k, v in fields.items():
            col = _validate_column(k)
            cols.append(col)
            if isinstance(v, RawSQL):
                placeholders.append(v.expr)
            else:
                placeholders.append("%s")
                params.append(v)
        return cols, placeholders, params

    @staticmethod
    def _build_where(
        filters: dict[str, Any] | None,
    ) -> tuple[str, list]:
        """Baut WHERE-Klausel aus Filter-Dict.

        Unterstuetzte Filter-Formate::

            {"status": "active"}              -> status = %s
            {"status": None}                  -> status IS NULL
            {"status__ne": None}              -> status IS NOT NULL
            {"status__in": ["a", "b"]}        -> status IN (%s, %s)
            {"status__in": []}                -> FALSE (leere Liste)
            {"name__like": "test"}            -> name ILIKE %s  (mit %...%)
            {"created_at__gte": "2026-01-01"} -> created_at >= %s
            {"created_at__lte": "2026-12-31"} -> created_at <= %s
            {"status__ne": "deleted"}         -> status != %s

        ACHTUNG: Filter-Dict-Keys MUESSEN aus Code stammen (nie aus User-Input).
        Spaltennamen werden zusaetzlich gegen ein striktes Pattern validiert
        (Defense-in-Depth, ValueError bei verdaechtigen Mustern).
        Werte sind parametrisiert (%s) und damit sicher.
        """
        if not filters:
            return "", []

        conditions = []
        params = []

        for key, value in filters.items():
            if "__" in key:
                col, op = key.rsplit("__", 1)
            else:
                col, op = key, "eq"

            _validate_column(col)

            if op == "eq":
                if value is None:
                    conditions.append(f"{col} IS NULL")
                else:
                    conditions.append(f"{col} = %s")
                    params.append(value)
            elif op == "ne":
                if value is None:
                    conditions.append(f"{col} IS NOT NULL")
                else:
                    conditions.append(f"{col} != %s")
                    params.append(value)
            elif op == "in":
                if not value:
                    conditions.append("FALSE")
                else:
                    ph = ", ".join(["%s"] * len(value))
                    conditions.append(f"{col} IN ({ph})")
                    params.extend(value)
            elif op == "like":
                conditions.append(f"{col} ILIKE %s")
                params.append(f"%{value}%")
            elif op == "gte":
                conditions.append(f"{col} >= %s")
                params.append(value)
            elif op == "lte":
                conditions.append(f"{col} <= %s")
                params.append(value)
            else:
                raise ValueError(f"Unknown filter operator: '{op}' in key '{key}'")

        where = " WHERE " + " AND ".join(conditions)
        return where, params
