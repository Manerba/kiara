"""Bounded real-PG fork probe, isolated so children can exit Python normally.

Only kiara_test; no schema reset and no production backend access. Invoked by
test_db_pool.py with the package path of the module under test.
"""

import atexit
import errno
import faulthandler
import gc
import json
import os
import signal
import sys
import threading
import time
import weakref

import psycopg2
from kiara_db import connection as db

DSN = "postgresql://kiara:kiara@localhost:5432/kiara_test"
_shutdown_refs = []


def backend_pid(conn):
    return conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]


def main(mode, rotate):
    db.init_db(DSN, force=True)
    db.configure_pool(max_size=4, wait_seconds=0.5)
    db.set_statement_timeout_ms(3000)
    active = db.get_connection()
    parent_pid = backend_pid(active)
    spare = db.get_connection()
    spare_pid = backend_pid(spare)
    named_cursor = spare.cursor(name="p50_inherited_cursor")
    named_cursor.execute("SELECT 42 AS value")
    spare_ref, cursor_ref = weakref.ref(spare._conn), weakref.ref(named_cursor)
    spare_fd = spare.fileno()
    if rotate:
        # The active query and spare lease now belong to a retired pool.
        db.configure_pool(max_size=5, wait_seconds=0.5)
    errors = []

    def query():
        try:
            active.execute("SELECT pg_sleep(0.8)")
        except BaseException as exc:
            errors.append(repr(exc))

    worker = threading.Thread(target=query, daemon=True)
    child = None
    read_fd, write_fd = os.pipe()
    root_pid = os.getpid()

    def mark_shutdown():
        if os.getpid() != root_pid:
            os.write(write_fd, b"\natexit-reached")

    atexit.register(mark_shutdown)
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    try:
        worker.start()
        with db.db_query() as observer:
            observer.autocommit = True
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                row = observer.execute(
                    "SELECT wait_event FROM pg_stat_activity WHERE pid=%s", (parent_pid,)
                ).fetchone()
                if row and row["wait_event"] == "PgSleep":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Sibling query was not observed in PgSleep")
            idle_ref = weakref.ref(observer._conn)
            inherited_fds = [active.fileno(), spare_fd, observer.fileno()]
        started = time.monotonic()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            for fd in inherited_fds:
                try:
                    os.fstat(fd)
                except OSError as exc:
                    assert exc.errno == errno.EBADF
                else:
                    raise AssertionError("Inherited PG socket remains open in child")
            for wrapper in (active, spare):
                assert wrapper.closed
                try:
                    wrapper.execute("SELECT 1")
                except psycopg2.InterfaceError:
                    pass
                else:
                    raise AssertionError("Inherited wrapper reused its parent's session")
            del wrapper
            # Reusing a detached descriptor must survive later driver GC.
            sentinel = os.open(os.devnull, os.O_RDONLY)
            if sentinel != spare_fd:
                os.dup2(sentinel, spare_fd)
                os.close(sentinel)
            active.close()
            if mode == "gc":
                spare.close()
                del named_cursor, spare
                gc.collect()
                assert spare_ref() is None
                assert cursor_ref() is None
                assert idle_ref() is None
                os.fstat(spare_fd)
            else:
                # Explicitly retain inherited objects past atexit, exercising
                # __del__ and psycopg deallocation during normal shutdown.
                _shutdown_refs.extend([spare, named_cursor])

            with db.db_query() as fresh:
                child_pid = backend_pid(fresh)
                assert child_pid not in (parent_pid, spare_pid)
                assert (
                    fresh.execute("SELECT current_database() AS db").fetchone()["db"]
                    == "kiara_test"
                )
                fresh.execute("CREATE TEMP TABLE p50_child_only (value integer)")
                fresh.commit()
                raw = fresh._conn
            with db.db_query() as fresh:
                assert backend_pid(fresh) == child_pid
                assert (
                    fresh.execute(
                        "SELECT to_regclass('pg_temp.p50_child_only') AS name"
                    ).fetchone()["name"]
                    is None
                )
            assert db.pool_stats()["pid"] == os.getpid()
            assert db.pool_stats()["connections_created"] == 1
            if mode == "gc":
                db.close_pool()
                assert raw.closed
                gc.collect()
                os.fstat(spare_fd)
            else:
                _shutdown_refs.append(db.get_connection())
            os.write(write_fd, json.dumps({"child_backend_pid": child_pid}).encode())
            return 0  # Deliberately normal interpreter shutdown; no os._exit.

        os.close(write_fd)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            exited, status = os.waitpid(child, os.WNOHANG)
            if exited:
                child = None  # Reaped: cleanup must not signal this PID again.
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Child exceeded 4s including GC and interpreter shutdown")
        assert os.waitstatus_to_exitcode(status) == 0
        payload, marker = os.read(read_fd, 4096).split(b"\n")
        assert marker == b"atexit-reached"
        child_pid = json.loads(payload)["child_backend_pid"]
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert not errors
        assert backend_pid(active) == parent_pid
        assert backend_pid(spare) == spare_pid
        assert named_cursor.fetchone()["value"] == 42
        with db.db_query() as observer:
            assert (
                observer.execute(
                    "SELECT pid FROM pg_stat_activity WHERE pid=%s", (child_pid,)
                ).fetchall()
                == []
            )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "rotated_pool": rotate,
                    "observed_wait_event": "PgSleep",
                    "normal_child_exit": True,
                    "parent_session_intact": True,
                    "child_session_released": True,
                    "child_seconds": time.monotonic() - started,
                    "module": db.__file__,
                    "driver": psycopg2.__version__,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if child != 0:
            if child is not None:
                os.kill(child, signal.SIGUSR1)
                time.sleep(0.05)
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
            worker.join(timeout=3)
            named_cursor.close()
            spare.close()
            active.close()
            db.close_pool()
            os.close(read_fd)
            # Already closed after a successful fork in the parent.
            if child is None:
                try:
                    os.close(write_fd)
                except OSError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] == "rotate"))
