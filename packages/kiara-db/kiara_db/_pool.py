"""Bounded, process-owned leases; never share a transaction between callers."""

import logging
import os
import threading
import time
from collections import Counter

logger = logging.getLogger(__name__)


class PoolClosedError(RuntimeError):
    pass


class PoolTimeoutError(TimeoutError):
    pass


class _ConnectionPool:
    def __init__(self, dsn, max_size, wait_seconds, statement_timeout_ms):
        self.settings = (dsn, max_size, wait_seconds, statement_timeout_ms)
        self.pid = os.getpid()
        self._condition = threading.Condition()
        self._idle = []
        self._leased = set()
        self._total = 0
        self._closed = False
        self._counts = Counter()

    def acquire(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor

        dsn, limit, wait_seconds, timeout_ms = self.settings
        deadline = time.monotonic() + wait_seconds
        for attempt in range(2):
            with self._condition:
                while True:
                    if self._closed or self.pid != os.getpid():
                        raise PoolClosedError("DB-Pool ist geschlossen")
                    if self._idle:
                        conn = self._idle.pop()
                        self._leased.add(conn)
                        self._counts["reused"] += 1
                        break
                    if self._total < limit:
                        self._total += 1  # includes connection setup outside the lock
                        conn = None
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._counts["wait_timeouts"] += 1
                        raise PoolTimeoutError("DB-Pool-Kapazitaet erschoepft")
                    self._condition.wait(remaining)
            if conn is None:
                try:
                    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor, connect_timeout=5)
                except BaseException:
                    with self._condition:
                        self._total -= 1
                        self._condition.notify_all()
                    raise
                with self._condition:
                    self._leased.add(conn)
                    self._counts["connections_created"] += 1
            try:
                # Also a liveness check for an idle connection terminated by PG.
                # Autocommit leaves no transaction/lock behind at checkout.
                conn.autocommit = True
                with conn.cursor() as cursor:
                    cursor.execute("SET statement_timeout = %s", (timeout_ms,))
                conn.autocommit = False
                with self._condition:
                    if self._closed:
                        raise PoolClosedError("DB-Pool waehrend Verbindungsaufbau geschlossen")
                    self._counts["checkouts"] += 1
                return conn
            except BaseException as exc:
                self.release(conn, discard=True)
                if attempt or not isinstance(
                    exc, (psycopg2.OperationalError, psycopg2.InterfaceError)
                ):
                    raise
        raise AssertionError("unreachable")

    def release(self, conn, *, discard=False):
        if self.pid != os.getpid():
            return
        # Keep capacity reserved throughout reset: another borrower cannot see
        # this connection until transactions, locks AND session state are gone.
        reusable = not discard and not conn.closed and not self._closed
        if reusable:
            try:
                conn.reset()  # rollback + driver transaction characteristics
                conn.autocommit = True
                with conn.cursor() as cursor:
                    cursor.execute("DISCARD ALL")  # temp tables, LISTEN, locks, prepared statements
                conn.autocommit = False
                conn.notices.clear()
                conn.notifies.clear()
            except Exception:
                reusable = False
                logger.warning(
                    "DB-Session-Reset fehlgeschlagen; Verbindung verworfen", exc_info=True
                )
        with self._condition:
            self._leased.discard(conn)
            if reusable and not self._closed:
                self._idle.append(conn)
            else:
                try:
                    conn.close()
                except Exception:
                    logger.warning(
                        "Defekte DB-Verbindung konnte nicht geschlossen werden", exc_info=True
                    )
                finally:
                    self._total -= 1
                    self._counts["discarded"] += 1
            self._condition.notify_all()

    def close(self):
        """Reject new leases; active users finish and discard on return."""
        with self._condition:
            self._closed = True
            for conn in self._idle:
                conn.close()
            self._total -= len(self._idle)
            self._idle.clear()
            self._condition.notify_all()

    def detach_after_fork(self):
        """Discard the child's pool snapshot without entering inherited locks.

        Explicit psycopg close() can wait forever on a mutex held by a vanished
        thread. On our POSIX/GCC psycopg2 build, connection_dealloc instead
        checks procpid and skips conn_close in a foreign process. Drop Python
        references through that path; never close/reset inherited libpq state.
        See PROCESS_MODEL.md for the scope and native-memory lifetime.
        """
        for conn in [*self._idle, *self._leased]:
            try:
                os.close(conn.fileno())
            except Exception:
                pass
        self._closed = True
        self._idle.clear()
        self._leased.clear()
        self._total = 0
        # Retained wrappers may still refer to this retired pool. Keep it
        # unusable (original pid), without retaining another thread's lock.
        self._condition = threading.Condition()

    def stats(self):
        with self._condition:
            return dict(
                self._counts,
                pid=self.pid,
                size=self._total,
                idle=len(self._idle),
                borrowed=len(self._leased),
                capacity=self.settings[1],
                closed=self._closed,
            )
