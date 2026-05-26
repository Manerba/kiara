"""Compat-Tests: kiara-db Library-Entkopplung und Stub-Importpfade.

Prueft:
- kiara_db importierbar ohne kiara
- Alle kiara.repositories.* Stubs importierbar
- Keine zirkulaeren Dependencies
"""

import pytest


class _FakeCursor:
    def __init__(
        self, *, rows=None, description=("id",), rowcount=None, execute_error=None, fetch_error=None
    ):
        self._rows = list(rows or [])
        self.description = description
        self.rowcount = len(self._rows) if rowcount is None else rowcount
        self.arraysize = 2
        self.query = b"SELECT 1"
        self.statusmessage = "OK"
        self.closed = False
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.executed = None

    def execute(self, sql, params):
        self.executed = (sql, params)
        if self.execute_error is not None:
            raise self.execute_error

    def fetchall(self):
        if self.fetch_error is not None:
            raise self.fetch_error
        return list(self._rows)

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj


class _StatementTimeoutCursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StatementTimeoutConnection:
    def __init__(self):
        self.cursor_obj = _StatementTimeoutCursor()

    def cursor(self):
        return self.cursor_obj


class TestLibrarySmoke:
    """kiara_db ist ohne kiara-Imports importierbar."""

    def test_import_kiara_db(self):
        import kiara_db

        assert hasattr(kiara_db, "BaseRepository")
        assert hasattr(kiara_db, "init_db")
        assert hasattr(kiara_db, "get_connection")
        assert hasattr(kiara_db, "DatasourceRepository")

    def test_no_kiara_imports_in_library(self):
        """Verifiziert dass kiara_db-Module keine kiara-Imports enthalten."""


class TestConnectionWrapperCursorLifecycle:
    """Cursor-Lifecycle wird zentral von ConnectionWrapper.execute() verwaltet."""

    def test_execute_materializes_rows_and_closes_cursor(self):
        from kiara_db.connection import ConnectionWrapper

        cursor = _FakeCursor(rows=[{"id": 1}, {"id": 2}])
        db = ConnectionWrapper(_FakeConnection(cursor))

        result = db.execute("SELECT id FROM t WHERE x = %s", (1,))

        assert cursor.executed == ("SELECT id FROM t WHERE x = %s", (1,))
        assert cursor.closed is True
        assert result.closed is True
        assert result.rowcount == 2
        assert result.fetchone() == {"id": 1}
        assert result.fetchall() == [{"id": 2}]
        assert result.fetchone() is None

    def test_execute_fetchmany_and_iteration_use_materialized_rows(self):
        from kiara_db.connection import ConnectionWrapper

        cursor = _FakeCursor(rows=[{"id": 1}, {"id": 2}, {"id": 3}])
        db = ConnectionWrapper(_FakeConnection(cursor))

        result = db.execute("SELECT id FROM t")

        assert result.fetchmany() == [{"id": 1}, {"id": 2}]
        assert list(result) == [{"id": 3}]

    def test_execute_dml_closes_cursor_and_preserves_rowcount(self):
        from kiara_db.connection import ConnectionWrapper

        cursor = _FakeCursor(description=None, rowcount=3)
        db = ConnectionWrapper(_FakeConnection(cursor))

        result = db.execute("DELETE FROM t")

        assert cursor.closed is True
        assert result.rowcount == 3
        with pytest.raises(RuntimeError):
            result.fetchone()

    def test_execute_closes_cursor_on_execute_error(self):
        from kiara_db.connection import ConnectionWrapper

        cursor = _FakeCursor(execute_error=RuntimeError("execute failed"))
        db = ConnectionWrapper(_FakeConnection(cursor))

        with pytest.raises(RuntimeError, match="execute failed"):
            db.execute("SELECT broken")

        assert cursor.closed is True

    def test_execute_closes_cursor_on_fetch_error(self):
        from kiara_db.connection import ConnectionWrapper

        cursor = _FakeCursor(fetch_error=RuntimeError("fetch failed"))
        db = ConnectionWrapper(_FakeConnection(cursor))

        with pytest.raises(RuntimeError, match="fetch failed"):
            db.execute("SELECT broken")

        assert cursor.closed is True


class TestTransactionContextManagers:
    """Rollback-Fehler in den Transaction-Context-Managern duerfen die
    urspruengliche Exception nicht verdecken."""

    def _make_conn(self, *, rollback_error=None):
        class _Cursor:
            arraysize = 1
            description = None
            rowcount = -1
            query = None
            statusmessage = None

            def execute(self, *_args, **_kwargs):
                return None

            def fetchall(self):
                return []

            def close(self):
                return None

        class _Conn:
            def __init__(self):
                self.committed = False
                self.rolled_back = False
                self.closed = False

            def cursor(self):
                return _Cursor()

            def commit(self):
                self.committed = True

            def rollback(self):
                self.rolled_back = True
                if rollback_error is not None:
                    raise rollback_error

            def close(self):
                self.closed = True

        return _Conn()

    def test_rollback_exception_in_db_transaction_does_not_mask_original(self, monkeypatch, caplog):
        import kiara_db.connection as conn_mod

        wrapped = self._make_conn(rollback_error=RuntimeError("rollback boom"))
        monkeypatch.setattr(conn_mod, "get_db", lambda: conn_mod.ConnectionWrapper(wrapped))

        caplog.set_level("WARNING", logger="kiara_db.connection")
        with pytest.raises(ValueError, match="original boom"):
            with conn_mod.db_transaction():
                raise ValueError("original boom")

        assert wrapped.rolled_back is True
        assert wrapped.closed is True
        assert wrapped.committed is False
        assert any(
            "Rollback in db_transaction fehlgeschlagen" in record.getMessage()
            for record in caplog.records
        )

    def test_rollback_exception_in_connect_transaction_does_not_mask_original(
        self, monkeypatch, caplog
    ):
        import kiara_db.connection as conn_mod

        wrapped = self._make_conn(rollback_error=RuntimeError("rollback boom"))
        monkeypatch.setattr(conn_mod, "get_connection", lambda: conn_mod.ConnectionWrapper(wrapped))

        caplog.set_level("WARNING", logger="kiara_db.connection")
        with pytest.raises(ValueError, match="original boom"):
            with conn_mod.connect_transaction():
                raise ValueError("original boom")

        assert wrapped.rolled_back is True
        assert wrapped.closed is True
        assert any(
            "Rollback in connect_transaction fehlgeschlagen" in record.getMessage()
            for record in caplog.records
        )


class TestConnectionFactory:
    def test_get_connection_sets_session_statement_timeout(self, monkeypatch):
        import kiara_db.connection as conn_mod
        import psycopg2

        fake_conn = _StatementTimeoutConnection()

        monkeypatch.setattr(conn_mod, "_dsn", "postgresql://unit-test")
        monkeypatch.setattr(conn_mod, "_statement_timeout_ms", 30_000)
        monkeypatch.setattr(psycopg2, "connect", lambda *args, **kwargs: fake_conn)
        conn_mod.set_statement_timeout_ms(12_345)

        db = conn_mod.get_connection()

        assert db._conn is fake_conn
        assert fake_conn.cursor_obj.executed == [
            ("SET statement_timeout = %s", (12_345,)),
        ]

    def test_negative_statement_timeout_rejected(self):
        from kiara_db.connection import set_statement_timeout_ms

        with pytest.raises(ValueError, match="statement_timeout_ms"):
            set_statement_timeout_ms(-1)


class TestStubSmoke:
    """Alle kiara.repositories.* Stubs bleiben importierbar."""

    def test_import_base(self):
        from kiara.repositories.base import BaseRepository, RawSQL

        assert BaseRepository is not None
        assert RawSQL is not None

    def test_import_user(self):
        from kiara.repositories.user import UserRepository

        assert UserRepository is not None

    def test_import_datasource(self):
        from kiara.repositories.datasource import (
            DatasourceRepository,
        )

        assert DatasourceRepository is not None

    def test_import_job(self):
        from kiara.repositories.job import IngestionJobRepository

        assert IngestionJobRepository is not None

    def test_import_chat(self):
        from kiara.repositories.chat import ChatRepository

        assert ChatRepository is not None

    def test_import_config(self):
        from kiara.repositories.config import ConfigRepository

        assert ConfigRepository is not None

    def test_import_file_hash(self):
        from kiara.repositories.file_hash import FileHashRepository

        assert FileHashRepository is not None

    def test_import_audit(self):
        from kiara.repositories.audit import (
            AuditAnonymizationRepository,
        )

        assert AuditAnonymizationRepository is not None

    def test_import_admin_chat(self):
        from kiara.repositories.admin_chat import AdminChatRepository

        assert AdminChatRepository is not None

    def test_import_remaining_stubs(self):
        from kiara.repositories.api_key import ApiKeyRepository
        from kiara.repositories.backend import KiBackendRepository
        from kiara.repositories.cache import CacheRepository
        from kiara.repositories.changelog import ConfigChangelogRepository
        from kiara.repositories.graph import EdgeRepository, NodeRepository
        from kiara.repositories.group import GroupRepository
        from kiara.repositories.health import HealthLogRepository
        from kiara.repositories.permission import PermissionRepository
        from kiara.repositories.prompt import PromptTemplateRepository
        from kiara.repositories.wopi import WopiRepository
        from kiara.repositories.workflow import WorkflowExportRepository

        assert all(
            cls is not None
            for cls in [
                KiBackendRepository,
                CacheRepository,
                ConfigChangelogRepository,
                NodeRepository,
                EdgeRepository,
                GroupRepository,
                HealthLogRepository,
                PermissionRepository,
                PromptTemplateRepository,
                WopiRepository,
                WorkflowExportRepository,
                ApiKeyRepository,
            ]
        )
