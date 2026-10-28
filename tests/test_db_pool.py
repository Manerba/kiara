"""Real PostgreSQL tests for lease ownership and session isolation."""

import gc
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg2
import pytest
from kiara_db import connection as db

DSN = "postgresql://kiara:kiara@localhost:5432/kiara_test"


@pytest.fixture(autouse=True)
def isolated_pool():
    db.close_pool()
    db.configure_pool(max_size=4, wait_seconds=0.2)
    db.set_statement_timeout_ms(30_000)
    db.init_db(DSN, force=True)
    yield
    db.close_pool()
    db.configure_pool()
    db.set_statement_timeout_ms(30_000)
    db.init_db(DSN, force=True)


def pid(conn):
    return conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]


def test_closed_cursor_objects_are_released_during_open_lease(monkeypatch):
    references = []
    with db.db_query() as conn:
        original_cursor = conn.cursor

        def observe_cursor(*args, **kwargs):
            cursor = original_cursor(*args, **kwargs)
            references.append(weakref.ref(cursor))
            return cursor

        monkeypatch.setattr(conn, "cursor", observe_cursor)
        for number in range(1000):
            assert conn.execute("SELECT %s AS value", (number,)).fetchone()["value"] == number
        conn.executemany("SELECT %s", [(1,), (2,)])
        conn.executescript("SELECT 1; SELECT 2")
        with pytest.raises(psycopg2.errors.DivisionByZero):
            conn.execute("SELECT 1 / 0")
        conn.rollback()
        closed = conn.cursor()
        closed.close()
        del closed
        gc.collect()
        assert all(reference() is None for reference in references)
        assert not conn.closed

        # An escaped, still-open raw cursor must be closed before pool reuse.
        outstanding = conn.cursor()
        outstanding.execute("SELECT 42")
        assert not outstanding.closed
    assert outstanding.closed


def test_reuse_and_nested_transactions_are_independent():
    with db.db_transaction() as outer:
        outer_pid = pid(outer)
        outer.execute("CREATE TABLE p50_pool_probe (value integer)")
    try:
        with db.db_transaction() as outer:
            assert pid(outer) == outer_pid
            outer.execute("INSERT INTO p50_pool_probe VALUES (1)")
            with db.db_query() as inner:
                assert pid(inner) != outer_pid
                assert inner.execute("SELECT * FROM p50_pool_probe").fetchall() == []
            with pytest.raises(ValueError):
                with db.db_transaction() as inner:
                    inner.execute("INSERT INTO p50_pool_probe VALUES (2)")
                    raise ValueError("rollback inner")
        with db.db_query() as conn:
            assert conn.execute("SELECT value FROM p50_pool_probe").fetchall() == [{"value": 1}]
        assert db.pool_stats()["connections_created"] == 2
    finally:
        with db.db_transaction() as conn:
            conn.execute("DROP TABLE p50_pool_probe")


def test_session_state_and_escaped_cursors_do_not_survive_return():
    first = db.get_connection()
    first_pid = pid(first)
    first.execute("CREATE TEMP TABLE p50_session (value integer)")
    first.execute("PREPARE p50_statement AS SELECT 1")
    first.execute("LISTEN p50_channel")
    first.execute("SELECT pg_advisory_lock(506602)")
    first.execute("SET search_path = pg_catalog")
    first.execute("SET statement_timeout = 999")
    first.commit()
    cursor = first.cursor()
    cursor.execute("SELECT 1")
    first.rollback()
    first.set_session(isolation_level="SERIALIZABLE", readonly=True, deferrable=True)
    first.close()
    assert cursor.closed
    with pytest.raises(psycopg2.InterfaceError):
        first.execute("SELECT 1")
    with db.db_query() as second:
        assert pid(second) == first_pid
        assert (
            second.execute("SELECT to_regclass('pg_temp.p50_session') AS name").fetchone()["name"]
            is None
        )
        assert second.execute("SELECT * FROM pg_prepared_statements").fetchall() == []
        assert second.execute("SELECT * FROM pg_listening_channels()").fetchall() == []
        assert (
            second.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"]
            == "off"
        )
        assert (
            second.execute("SHOW transaction_isolation").fetchone()["transaction_isolation"]
            == "read committed"
        )
        assert second.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "30s"
        assert second.execute("SHOW search_path").fetchone()["search_path"] != "pg_catalog"
        with db.db_query() as observer:
            assert observer.execute("SELECT pg_try_advisory_lock(506602) AS ok").fetchone()["ok"]


def test_query_context_rolls_back_even_without_exception():
    with db.db_query() as conn:
        conn.execute("CREATE TABLE p50_uncommitted(value integer)")
    with db.db_query() as conn:
        assert (
            conn.execute("SELECT to_regclass('p50_uncommitted') AS name").fetchone()["name"] is None
        )


def test_commit_failure_discards_connection():
    with pytest.raises(psycopg2.IntegrityError):
        with db.db_transaction() as conn:
            old_pid = pid(conn)
            conn.execute(
                "CREATE TEMP TABLE p50_deferred (value integer UNIQUE DEFERRABLE INITIALLY DEFERRED)"
            )
            conn.execute("INSERT INTO p50_deferred VALUES (1),(1)")
    with db.db_query() as conn:
        assert pid(conn) != old_pid
    assert db.pool_stats()["discarded"] == 1


def test_dead_idle_connection_reconnects_and_timeout_does_not_leak():
    with db.db_query() as conn:
        old_pid = pid(conn)
    with psycopg2.connect(DSN) as observer:
        with observer.cursor() as cursor:
            cursor.execute("SELECT pg_terminate_backend(%s)", (old_pid,))
    with db.db_query() as conn:
        assert pid(conn) != old_pid
        conn.execute("SET LOCAL statement_timeout = 10")
        with pytest.raises(psycopg2.errors.QueryCanceled):
            conn.execute("SELECT pg_sleep(0.1)")
    with db.db_query() as conn:
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "30s"


def test_capacity_wait_and_nested_exhaustion_are_bounded():
    db.configure_pool(max_size=1, wait_seconds=0.05)
    first = db.get_connection()
    with pytest.raises(db.PoolTimeoutError):
        db.get_connection()
    started = threading.Event()

    def borrow():
        started.set()
        with db.db_query() as conn:
            return pid(conn)

    first_pid = pid(first)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(borrow)
        assert started.wait(1)
        first.close()
        assert future.result(timeout=2) == first_pid
    assert db.pool_stats()["size"] == 1


def test_parallel_users_get_distinct_sessions():
    barrier = threading.Barrier(4)

    def borrow(_):
        with db.db_transaction() as conn:
            backend_pid = pid(conn)
            barrier.wait(timeout=3)
            return backend_pid

    with ThreadPoolExecutor(max_workers=4) as executor:
        pids = list(executor.map(borrow, range(4)))
    assert len(set(pids)) == 4
    assert db.pool_stats()["borrowed"] == 0


def test_shutdown_and_dsn_rotation_do_not_close_active_transaction():
    old = db.get_connection()
    old_pid = pid(old)
    db.init_db(DSN + "?application_name=p50-rotated", force=True)
    with db.db_query() as current:
        assert pid(current) != old_pid
        assert (
            current.execute("SHOW application_name").fetchone()["application_name"] == "p50-rotated"
        )
    assert pid(old) == old_pid
    old_raw = old._conn
    old.close()
    assert old_raw.closed
    active = db.get_connection()
    db.close_pool()
    assert pid(active)
    with pytest.raises(db.PoolClosedError):
        db.get_connection()
    active.close()
    db.init_db(DSN, force=True)
    with db.db_query() as conn:
        assert pid(conn)


def test_test_database_initialization_remains_protected():
    import kiara.shared.db as shared

    shared.init_db_backend({"database": {"postgresql": {"dsn": "postgresql://never-use"}}})
    db.init_db("postgresql://never-use")
    with db.db_query() as conn:
        assert conn.execute("SELECT current_database() AS name").fetchone()["name"] == "kiara_test"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork boundary")
@pytest.mark.parametrize("mode", ["gc", "shutdown"])
@pytest.mark.parametrize("rotation", ["current", "rotate"])
def test_active_query_fork_and_later_child_cleanup_are_bounded(mode, rotation):
    # A fresh interpreter permits normal child shutdown without pytest catching
    # SystemExit. Resolve imports from the actual package under test, including
    # when this suite is run against installed packages after deployment.
    env = dict(os.environ, PYTHONPATH=str(Path(db.__file__).resolve().parent.parent))
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).parent / "helpers/db_fork_lifecycle.py"),
            mode,
            rotation,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        # Only this probe's private process group, including its own fork child.
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=3)
        pytest.fail(f"Fork lifecycle probe timed out: {stdout}\n{stderr}")
    assert process.returncode == 0, stdout + stderr
    report = json.loads(stdout)
    assert report["normal_child_exit"]
    assert report["parent_session_intact"]
    assert report["child_session_released"]
    assert report["observed_wait_event"] == "PgSleep"
    assert report["child_seconds"] < 4
    assert Path(report["module"]).resolve() == Path(db.__file__).resolve()
    print(stdout.strip())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork contract")
def test_fork_cannot_reuse_parent_session():
    parent = db.get_connection()
    parent_pid = pid(parent)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(read_fd)
            with pytest.raises(psycopg2.InterfaceError):
                parent.execute("SELECT 1")
            with db.db_query() as conn:
                os.write(write_fd, str(pid(conn)).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], 3)
        assert ready, "Child did not return from the fork hook"
        child_pid = int(os.read(read_fd, 64))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            exited, status = os.waitpid(child, os.WNOHANG)
            if exited:
                child = None
                break
            time.sleep(0.01)
        else:
            pytest.fail("Child did not exit after its query")
        assert os.waitstatus_to_exitcode(status) == 0
        assert child_pid != parent_pid
        assert pid(parent) == parent_pid
    finally:
        if child is not None:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        os.close(read_fd)
        parent.close()


def test_connection_setup_failure_returns_reserved_capacity(monkeypatch):
    connect = psycopg2.connect
    monkeypatch.setattr(
        psycopg2,
        "connect",
        lambda *a, **k: (_ for _ in ()).throw(psycopg2.OperationalError("offline")),
    )
    with pytest.raises(psycopg2.OperationalError):
        db.get_connection()
    assert db.pool_stats()["size"] == 0
    monkeypatch.setattr(psycopg2, "connect", connect)
    with db.db_query() as conn:
        assert pid(conn)


@pytest.mark.parametrize("failure", ["rollback", "reset"])
def test_failed_cleanup_discards_and_preserves_original_error(monkeypatch, failure):
    from unittest.mock import MagicMock

    raw = MagicMock()
    raw.closed = False
    raw.rollback.side_effect = RuntimeError("rollback failed") if failure == "rollback" else None
    raw.reset.side_effect = RuntimeError("reset failed") if failure == "reset" else None
    monkeypatch.setattr(psycopg2, "connect", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="original"):
        with db.db_transaction():
            raise ValueError("original")
    raw.close.assert_called_once()
    assert db.pool_stats()["size"] == 0
    assert db.pool_stats()["borrowed"] == 0
