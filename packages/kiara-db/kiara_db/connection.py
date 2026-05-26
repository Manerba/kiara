"""Connection-Management fuer kiara-db.

Stellt Connection-Factory, Context-Manager und DSN-Verwaltung bereit.
Backend: PostgreSQL (psycopg2 mit RealDictCursor).

DSN-Initialisierung:
    - Direkt: ``init_db(dsn)`` beim Start aufrufen
    - Lazy: ``register_dsn_provider(callback)`` fuer Host-App-Integration
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modul-Level Singleton
# ---------------------------------------------------------------------------

_dsn: str | None = None
_dsn_provider: Callable[[], str | None] | None = None
_statement_timeout_ms: int = 30_000


# =========================================================================
# ConnectionWrapper
# =========================================================================


class ResultProxy:
    """Cursor-Ergebnis mit zentral verwaltetem Cursor-Lifecycle.

    `ConnectionWrapper.execute()` gibt historisch ein Cursor-aehnliches Objekt
    zurueck. Der Proxy erhaelt diese API fuer `fetchone()`, `fetchall()`,
    `fetchmany()`, Iteration und `rowcount`, schliesst den psycopg2-Cursor aber
    direkt nach der Datenuebernahme.
    """

    def __init__(self, cursor):
        self._cursor = cursor
        self._rows = None
        self._index = 0
        self.arraysize = getattr(cursor, "arraysize", 1)
        self.description = getattr(cursor, "description", None)
        self.query = getattr(cursor, "query", None)
        self.statusmessage = getattr(cursor, "statusmessage", None)
        self.rowcount = getattr(cursor, "rowcount", -1)
        self._closed = False
        try:
            if self.description is not None:
                self._rows = list(cursor.fetchall())
                self.rowcount = getattr(cursor, "rowcount", self.rowcount)
        finally:
            self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None

    def _require_rows(self):
        if self._rows is None:
            raise RuntimeError("Query liefert keine Ergebniszeilen")
        return self._rows

    def fetchone(self):
        rows = self._require_rows()
        if self._index >= len(rows):
            return None
        row = rows[self._index]
        self._index += 1
        return row

    def fetchall(self):
        rows = self._require_rows()
        remaining = rows[self._index :]
        self._index = len(rows)
        return list(remaining)

    def fetchmany(self, size=None):
        rows = self._require_rows()
        if size is None:
            size = self.arraysize
        start = self._index
        end = min(len(rows), start + max(0, int(size)))
        self._index = end
        return list(rows[start:end])

    def __iter__(self):
        self._require_rows()
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class ConnectionWrapper:
    """Proxy um psycopg2 Connection mit einheitlichem Interface.

    Stellt execute(), executemany(), executescript(), commit(), rollback(),
    close() und cursor() bereit.  Erstellt intern Cursors (psycopg2
    Connection hat kein execute()/executemany()).

    Alle SQL-Statements verwenden native PostgreSQL-Syntax (%s Placeholder,
    NOW(), SERIAL PRIMARY KEY, COALESCE, ILIKE).
    """

    def __init__(self, conn):
        self._conn = conn

    # -- Query-Interface ------------------------------------------------

    def execute(self, sql, params=None):
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params or ())
            return ResultProxy(cursor)
        except Exception:
            try:
                cursor.close()
            except Exception as close_exc:
                logger.warning(
                    "Cursor-Close nach execute-Fehler fehlgeschlagen: %s",
                    close_exc,
                    exc_info=True,
                )
            raise

    def executemany(self, sql, params_list):
        cursor = self._conn.cursor()
        try:
            cursor.executemany(sql, params_list)
        finally:
            cursor.close()
        return None

    def executescript(self, sql_script):
        """Multi-Statement-Ausfuehrung.

        Splittet auf ';', fuehrt einzeln aus.
        BEGIN/COMMIT/ROLLBACK werden uebersprungen
        (Transaktionen via Connection/Context-Manager).
        """
        cursor = self._conn.cursor()
        try:
            for statement in sql_script.split(";"):
                statement = statement.strip()
                if not statement:
                    continue
                upper = statement.upper().lstrip()
                if upper.startswith(("BEGIN", "COMMIT", "ROLLBACK")):
                    continue
                cursor.execute(statement, ())
        finally:
            cursor.close()
        return None

    # -- Transaction-/Lifecycle-Interface -------------------------------

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def cursor(self):
        return self._conn.cursor()

    def __getattr__(self, name):
        return getattr(self._conn, name)


# =========================================================================
# DSN-Verwaltung
# =========================================================================


def register_dsn_provider(provider: Callable[[], str | None] | None) -> None:
    """Registriert optionalen Lazy-Init-Hook fuer Host-Applikationen.

    Der Provider wird aufgerufen wenn noch kein DSN gesetzt ist und eine
    DB-Verbindung benoetigt wird. Dies ermoeglicht Lazy-Init ohne dass
    die Library die Host-App importieren muss.
    """
    global _dsn_provider
    _dsn_provider = provider


def init_db(dsn: str, force: bool = False) -> None:
    """Setzt den DSN fuer alle DB-Verbindungen.

    Ueberschreibt keinen Test-DSN, ausser force=True.
    """
    if not dsn:
        raise ValueError("DSN darf nicht leer sein")
    global _dsn
    if not force and _dsn is not None and "kiara_test" in _dsn:
        return
    _dsn = dsn


def set_statement_timeout_ms(timeout_ms: int | None) -> None:
    """Setzt den PostgreSQL statement_timeout fuer neu geoeffnete Sessions.

    ``0`` deaktiviert den serverseitigen Timeout explizit.
    """
    global _statement_timeout_ms
    if timeout_ms is None:
        timeout_ms = 30_000
    timeout_ms = int(timeout_ms)
    if timeout_ms < 0:
        raise ValueError("statement_timeout_ms darf nicht negativ sein")
    _statement_timeout_ms = timeout_ms


def _get_dsn() -> str:
    """Gibt den DSN zurueck. Nutzt optionalen Provider-Hook fuer Lazy-Init."""
    global _dsn
    if _dsn is None and _dsn_provider is not None:
        provided = _dsn_provider()
        if provided:
            init_db(provided)
    if _dsn is None:
        raise RuntimeError(
            "kiara_db nicht initialisiert — init_db(dsn) "
            "muss vor dem ersten DB-Zugriff aufgerufen werden."
        )
    return _dsn


# ---------------------------------------------------------------------------
# Connection Factory
# ---------------------------------------------------------------------------


def get_connection() -> ConnectionWrapper:
    """Erstellt eine DB-Verbindung zur PostgreSQL-Datenbank."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(_get_dsn(), cursor_factory=RealDictCursor, connect_timeout=5)
    with conn.cursor() as cursor:
        cursor.execute("SET statement_timeout = %s", (_statement_timeout_ms,))
    return ConnectionWrapper(conn)


def get_db() -> ConnectionWrapper:
    """Liefert eine Verbindung zur Datenbank."""
    return get_connection()


# ---------------------------------------------------------------------------
# Context-Manager
# ---------------------------------------------------------------------------


@contextmanager
def connect_query() -> Generator[ConnectionWrapper, None, None]:
    """Context-Manager fuer Read-Only-Queries."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def connect_transaction() -> Generator[ConnectionWrapper, None, None]:
    """Context-Manager fuer Transaktionen.

    Committet bei Erfolg, Rollback bei Exception. Rollback-Fehler werden
    geloggt, die urspruengliche Exception wird ueber das aeussere `raise`
    weitergereicht und nicht verdeckt.
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception as rollback_exc:
            logger.warning(
                "Rollback in connect_transaction fehlgeschlagen: %s",
                rollback_exc,
                exc_info=True,
            )
        raise
    finally:
        conn.close()


@contextmanager
def db_query() -> Generator[ConnectionWrapper, None, None]:
    """Context-Manager fuer Read-Only-Queries."""
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def db_transaction() -> Generator[ConnectionWrapper, None, None]:
    """Context-Manager fuer Transaktionen.

    Committet bei Erfolg, Rollback bei Exception. Rollback-Fehler werden
    geloggt, die urspruengliche Exception wird ueber das aeussere `raise`
    weitergereicht und nicht verdeckt.
    """
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception as rollback_exc:
            logger.warning(
                "Rollback in db_transaction fehlgeschlagen: %s",
                rollback_exc,
                exc_info=True,
            )
        raise
    finally:
        conn.close()
