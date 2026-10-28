"""Connection-Management fuer kiara-db.

Stellt Connection-Factory, Context-Manager und DSN-Verwaltung bereit.
Backend: PostgreSQL (psycopg2 mit RealDictCursor).

DSN-Initialisierung:
    - Direkt: ``init_db(dsn)`` beim Start aufrufen
    - Lazy: ``register_dsn_provider(callback)`` fuer Host-App-Integration
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)

from kiara_db._pool import PoolClosedError, PoolTimeoutError, _ConnectionPool  # noqa: F401

# ---------------------------------------------------------------------------
# Modul-Level Singleton
# ---------------------------------------------------------------------------

_dsn: str | None = None
_dsn_provider: Callable[[], str | None] | None = None
_statement_timeout_ms: int = 30_000
_pool_max_size = 16
_pool_wait_seconds = 5.0
_pool: _ConnectionPool | None = None
_all_pools: weakref.WeakSet = weakref.WeakSet()
_manager_lock = threading.RLock()
_shutdown = False


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

    def __init__(self, conn, *, release=None):
        self._conn = conn
        self._release = release
        self._pid = os.getpid()
        self._discard = False
        # Do not retain closed cursors (and their SQL) for an entire lease.
        # Live escaped cursors remain observable and are closed on return.
        self._cursors = weakref.WeakSet()

    def _connection(self):
        if self._conn is None or self._pid != os.getpid():
            from psycopg2 import InterfaceError

            raise InterfaceError("DB-Verbindung ist geschlossen oder gehoert einem anderen Prozess")
        return self._conn

    @property
    def closed(self):
        return self._conn is None or self._pid != os.getpid() or self._conn.closed

    @property
    def autocommit(self):
        return self._connection().autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._connection().autocommit = value

    # -- Query-Interface ------------------------------------------------

    def execute(self, sql, params=None):
        cursor = self.cursor()
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
        cursor = self.cursor()
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
        cursor = self.cursor()
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
        try:
            self._connection().commit()
        except BaseException:
            self._discard = True
            raise

    def rollback(self):
        try:
            self._connection().rollback()
        except BaseException:
            self._discard = True
            raise

    def close(self):
        conn, self._conn = self._conn, None
        if conn is None or self._pid != os.getpid():
            return
        for cursor in self._cursors:
            try:
                cursor.close()
            except Exception:
                self._discard = True
        self._cursors.clear()
        if self._release is None:
            conn.close()
        else:
            self._release(conn, discard=self._discard)

    def __del__(self):
        # Direct get_connection() callers previously relied on libpq cleanup
        # on garbage collection. An abandoned lease must not exhaust the pool.
        if self.__dict__.get("_conn") is not None:
            try:
                self.close()
            except Exception:
                logger.warning("DB-Lease-Cleanup fehlgeschlagen", exc_info=True)

    def cursor(self, *args, **kwargs):
        cursor = self._connection().cursor(*args, **kwargs)
        self._cursors.add(cursor)
        return cursor

    def __getattr__(self, name):
        return getattr(self._connection(), name)


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
    global _dsn, _pool, _shutdown
    with _manager_lock:
        if not force and _dsn is not None and "kiara_test" in _dsn:
            return
        if _dsn != dsn or _shutdown:
            if _pool is not None:
                _pool.close()
            _pool = None
        _dsn = dsn
        _shutdown = False


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


def configure_pool(*, max_size: int = 16, wait_seconds: float = 5.0) -> None:
    import math

    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
        raise ValueError("pool_max_size muss eine positive Ganzzahl sein")
    if not math.isfinite(wait_seconds) or wait_seconds <= 0:
        raise ValueError("pool_wait_seconds muss positiv und endlich sein")
    global _pool_max_size, _pool_wait_seconds
    with _manager_lock:
        _pool_max_size, _pool_wait_seconds = max_size, float(wait_seconds)


def close_pool() -> None:
    """Prozess-Shutdown: keine neuen Leases; ausgeliehene Sessions auslaufen lassen.

    init_db() erlaubt einen expliziten Neustart. atexit schliesst den Pool auch
    fuer CLI-Verbraucher ohne eigenen Lifespan-Hook.
    """
    global _shutdown
    with _manager_lock:
        _shutdown = True
        for pool in list(_all_pools):
            pool.close()


def pool_stats() -> dict:
    with _manager_lock:
        return _pool.stats() if _pool is not None else {"pid": os.getpid(), "size": 0}


def _after_fork() -> None:
    global _pool, _manager_lock, _all_pools
    for pool in list(_all_pools):
        pool.detach_after_fork()
    _pool = None
    _all_pools = weakref.WeakSet()
    _manager_lock = threading.RLock()


atexit.register(close_pool)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def get_connection() -> ConnectionWrapper:
    """Leiht eine exklusive Session; close() bereinigt sie und gibt sie zurueck."""
    global _pool
    dsn = _get_dsn()
    with _manager_lock:
        if _shutdown:
            raise PoolClosedError("DB-Pool ist heruntergefahren; init_db() erforderlich")
        settings = (dsn, _pool_max_size, _pool_wait_seconds, _statement_timeout_ms)
        if _pool is None or _pool.settings != settings:
            if _pool is not None:
                _pool.close()
            _pool = _ConnectionPool(*settings)
            _all_pools.add(_pool)
        pool = _pool
    conn = pool.acquire()
    return ConnectionWrapper(conn, release=pool.release)


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
