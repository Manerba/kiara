"""Tests fuer Sprint 09: Tools Admin-Funktionen.

Testet tools.list, .permissions, .permissions.update, .test, .usage,
.connector_types, .erp.config.read/update, .erp.test_connection,
.erp.table_schema, .connections.list/create/update/delete/test,
.assignments, .assignments.update — 22 Unit-Tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiara.shared.constants import PASSWORD_MASK


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# tools.list
# ---------------------------------------------------------------------------


class TestToolsList:
    @patch("kiara.web.api_tools._get_registry")
    def test_tools_list(self, mock_reg):
        """Test 1: Tool-Liste mit Details."""
        defn = MagicMock()
        defn.name = "calculate"
        defn.description = "Berechnung"
        defn.requires_confirmation = False
        defn.required_permissions = []
        defn.parameters = {"type": "object", "properties": {}}
        defn.category = "builtin"
        mock_reg.return_value.list_tools.return_value = [defn]

        from kiara.admin.functions.tools import tools_list

        result = _run(tools_list(MOCK_USER))

        assert result.success is True
        assert len(result.data["tools"]) == 1
        assert result.data["tools"][0]["name"] == "calculate"


# ---------------------------------------------------------------------------
# tools.permissions
# ---------------------------------------------------------------------------


class TestToolsPermissions:
    @patch("kiara.config.load_config")
    def test_permissions_read(self, mock_config):
        """Test 2: Berechtigungsmatrix lesen."""
        mock_config.return_value = {
            "tools": {"permissions": {"Admins": ["calculate", "search"], "Users": ["search"]}},
        }

        from kiara.admin.functions.tools import tools_permissions

        result = _run(tools_permissions(MOCK_USER))

        assert result.success is True
        assert "Admins" in result.data["permissions"]
        assert len(result.data["permissions"]["Admins"]) == 2


class TestToolsPermissionsUpdate:
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_permissions_update(self, mock_config, mock_save):
        """Test 3: Berechtigungen speichern."""
        mock_config.return_value = {"tools": {"permissions": {}}}

        from kiara.admin.functions.tools import tools_permissions_update

        result = _run(
            tools_permissions_update(
                MOCK_USER,
                group="ERP-Leser",
                tools=["erp_order_status", "erp_stock_level"],
            )
        )

        assert result.success is True
        saved = mock_save.call_args[0][0]
        assert saved["tools"]["permissions"]["ERP-Leser"] == ["erp_order_status", "erp_stock_level"]


# ---------------------------------------------------------------------------
# tools.test
# ---------------------------------------------------------------------------


class TestToolsTest:
    @patch("kiara.web.api_tools._get_registry")
    def test_tool_not_found(self, mock_reg):
        """Test 4: Nicht existentes Tool → fail."""
        mock_reg.return_value.get_tool.return_value = None

        from kiara.admin.functions.tools import tools_test

        result = _run(tools_test(MOCK_USER, name="nonexistent"))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.web.api_tools._get_registry")
    def test_tool_success(self, mock_reg):
        """Test 5: Tool-Ausfuehrung erfolgreich."""
        mock_tool = MagicMock()
        mock_tool.validate_params.return_value = (True, "")
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"answer": 42}
        mock_result.error = None
        mock_result.display_text = "42"
        mock_tool.execute = AsyncMock(return_value=mock_result)
        mock_reg.return_value.get_tool.return_value = mock_tool

        from kiara.admin.functions.tools import tools_test

        result = _run(tools_test(MOCK_USER, name="calculate", params={"expr": "6*7"}))

        assert result.success is True
        assert result.data["success"] is True
        assert result.data["data"] == {"answer": 42}


@pytest.fixture
def document_test_runtime(monkeypatch):
    from kiara.tools.builtin_tools import SearchDocumentsTool
    from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool
    from kiara.tools.registry import ToolRegistry
    from kiara.web import api_tools

    config = {"tools": {"test_session_ttl_secs": 60, "test_max_sessions": 2}}
    chunks = [
        {
            "text": "Fassung eins",
            "metadata": {
                "source": "test.txt",
                "chunk_index": 0,
                "chunk_total": 1,
                "ds_ids": [7],
                "modified": "2026-10-01T12:00:00+00:00",
                "size_bytes": 100,
                "parser": "standard",
            },
        }
    ]
    retriever = MagicMock()
    retriever._store.get_collection_name.return_value = "kb_test"
    retriever.retrieve.side_effect = lambda **kwargs: chunks
    retriever.load_document_range.side_effect = lambda *args, **kwargs: chunks
    references = DocumentReferences("kb_test")
    registry = ToolRegistry()
    registry.register(SearchDocumentsTool(retriever, None, config, document_references=references))
    registry.register(ReadDocumentTool(retriever, config, references))
    monkeypatch.setattr(api_tools, "_get_registry", lambda: registry)
    monkeypatch.setattr(api_tools, "load_config", lambda: config)
    monkeypatch.setattr(api_tools, "_document_test_sessions", {})
    return api_tools, registry, config, chunks


def test_document_tests_isolate_users_and_allow_fresh_versions(document_test_runtime):
    from kiara.admin.functions.tools import tools_test

    api, registry, _, chunks = document_test_runtime
    first = _run(tools_test(MOCK_USER, "search_documents", {"query": "Dokumentfassung"}))
    assert first.success, first.message
    session = first.data["session_id"]
    reference = first.data["data"]["results"][0]["metadata"]["document_ref"]
    read_params = {"document_id": reference}
    assert _run(tools_test(MOCK_USER, "read_document", read_params, session_id=session)).data[
        "success"
    ]
    other_user = {"id": 2, "username": "other-admin"}
    denied = _run(tools_test(other_user, "read_document", read_params, session_id=session))
    assert not denied.success and "Testsitzung" in denied.message
    assert not _run(tools_test(MOCK_USER, "read_document", read_params)).success

    chunks[0] = {
        **chunks[0],
        "text": "Fassung zwei",
        "metadata": {**chunks[0]["metadata"], "modified": "2026-10-02T12:00:00+00:00"},
    }
    stale = _run(
        tools_test(MOCK_USER, "search_documents", {"query": "Dokumentfassung"}, session_id=session)
    )
    assert not stale.success and "Dokumentversion" in stale.message
    fresh = _run(tools_test(MOCK_USER, "search_documents", {"query": "Dokumentfassung"}))
    assert fresh.success and fresh.data["session_id"] != session
    new_reference = fresh.data["data"]["results"][0]["metadata"]["document_ref"]
    latest = _run(
        tools_test(
            MOCK_USER,
            "read_document",
            {"document_id": new_reference},
            session_id=fresh.data["session_id"],
        )
    )
    assert latest.data["success"] and latest.data["data"]["results"][0]["text"] == "Fassung zwei"
    assert not registry.get_tool("read_document")._references.resolve(reference)
    assert len(api._document_test_sessions) == 2


def test_document_test_sessions_expire_and_are_bounded(document_test_runtime, monkeypatch):
    from kiara.admin.functions.tools import tools_test

    api, registry, _, _ = document_test_runtime
    clock = [1.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    sessions = []
    for _ in range(3):
        result = _run(tools_test(MOCK_USER, "search_documents", {"query": "Dokumentfassung"}))
        sessions.append(result.data["session_id"])
        clock[0] += 1
    assert len(api._document_test_sessions) == 2
    assert sessions[0] not in api._document_test_sessions
    clock[0] = 100
    expired = _run(
        tools_test(
            MOCK_USER, "search_documents", {"query": "Dokumentfassung"}, session_id=sessions[-1]
        )
    )
    assert not expired.success and "abgelaufen" in expired.message
    assert not api._document_test_sessions
    fresh = _run(tools_test(MOCK_USER, "search_documents", {"query": "Dokumentfassung"}))
    assert fresh.success
    # Ein Config-/Registry-Wechsel beendet auch noch nicht abgelaufene Sitzungen.
    replacement = MagicMock()
    replacement.get_tool.side_effect = registry.get_tool
    monkeypatch.setattr(api, "_get_registry", lambda: replacement)
    invalidated = _run(
        tools_test(
            MOCK_USER,
            "search_documents",
            {"query": "Dokumentfassung"},
            session_id=fresh.data["session_id"],
        )
    )
    assert not invalidated.success


# ---------------------------------------------------------------------------
# tools.usage
# ---------------------------------------------------------------------------


class TestToolsUsage:
    @patch("kiara.shared.db.db_query")
    def test_usage_empty(self, mock_db):
        """Test 6: Leeres Query-Log → leere Liste."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.execute.return_value.fetchall.return_value = []
        mock_db.return_value = mock_ctx

        from kiara.admin.functions.tools import tools_usage

        result = _run(tools_usage(MOCK_USER))

        assert result.success is True
        assert result.data["tools"] == []

    @patch("kiara.shared.db.db_query")
    def test_usage_aggregated(self, mock_db):
        """Test 7: Tool-Nutzung korrekt aggregiert."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.execute.return_value.fetchall.return_value = [
            {"tools_used": '["search", "calculate"]', "created_at": "2026-03-28 12:00"},
            {"tools_used": '["search"]', "created_at": "2026-03-27 10:00"},
        ]
        mock_db.return_value = mock_ctx

        from kiara.admin.functions.tools import tools_usage

        result = _run(tools_usage(MOCK_USER))

        assert result.success is True
        tools = result.data["tools"]
        assert len(tools) == 2
        assert tools[0]["name"] == "search"  # haeufigster zuerst
        assert tools[0]["usage_count"] == 2


# ---------------------------------------------------------------------------
# tools.erp.config.read
# ---------------------------------------------------------------------------


class TestToolsErpConfigRead:
    @patch("kiara.config.load_config")
    def test_erp_config_read_with_dsn(self, mock_config):
        """Test 8: SQL-DSN wird maskiert."""
        mock_config.return_value = {
            "tools": {
                "erp": {
                    "backend": "sql",
                    "sql_dsn": "mssql://server/db",
                    "sql_driver": "mssql",
                    "mapping": {"orders": "Auftraege"},
                }
            },
        }

        from kiara.admin.functions.tools import tools_erp_config_read

        result = _run(tools_erp_config_read(MOCK_USER))

        assert result.success is True
        assert result.data["sql_dsn"] == PASSWORD_MASK
        assert result.data["backend"] == "sql"

    @patch("kiara.config.load_config")
    def test_erp_config_read_no_dsn(self, mock_config):
        """Test 9: Kein DSN → leerer String."""
        mock_config.return_value = {"tools": {"erp": {"backend": "mock"}}}

        from kiara.admin.functions.tools import tools_erp_config_read

        result = _run(tools_erp_config_read(MOCK_USER))

        assert result.data["sql_dsn"] == ""


# ---------------------------------------------------------------------------
# tools.erp.config.update
# ---------------------------------------------------------------------------


class TestToolsErpConfigUpdate:
    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt", side_effect=lambda v: v)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_erp_config_update_mask(self, mock_config, mock_save, _mock_enc, _mock_is_enc):
        """Test 10: PASSWORD_MASK → bestehendes DSN beibehalten."""
        mock_config.return_value = {
            "tools": {"erp": {"sql_dsn": "mssql://existing"}},
        }

        from kiara.admin.functions.tools import tools_erp_config_update

        result = _run(
            tools_erp_config_update(
                MOCK_USER,
                backend="sql",
                sql_dsn=PASSWORD_MASK,
            )
        )

        assert result.success is True
        saved = mock_save.call_args[0][0]
        # PASSWORD_MASK → kein Update des DSN-Felds
        assert saved["tools"]["erp"]["sql_dsn"] == "mssql://existing"

    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.shared.crypto.encrypt", side_effect=lambda v: v)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_erp_config_update_new_dsn(self, mock_config, mock_save, _mock_enc):
        """Test 11: Neuer DSN-Wert wird gespeichert."""
        mock_config.return_value = {"tools": {"erp": {}}}

        from kiara.admin.functions.tools import tools_erp_config_update

        result = _run(
            tools_erp_config_update(
                MOCK_USER,
                backend="sql",
                sql_dsn="mssql://newserver/db",
            )
        )

        assert result.success is True
        saved = mock_save.call_args[0][0]
        assert saved["tools"]["erp"]["sql_dsn"] == "mssql://newserver/db"

    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.shared.crypto.encrypt", side_effect=lambda v: v)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_erp_config_update_preserves_subtable_mapping_keys(
        self,
        mock_config,
        mock_save,
        _mock_enc,
    ):
        """Test 12: positions_/operations_-Keys bleiben im Mapping erhalten."""
        mapping = {
            "orders": {
                "table": "orders",
                "id_column": "order_no",
                "columns": {"order_id": "order_no"},
                "positions_table": "positions",
                "positions_fk": "order_no",
                "positions_columns": {"article_id": "article_no"},
            },
            "production_orders": {
                "table": "production_orders",
                "id_column": "fa_no",
                "columns": {"production_order_id": "fa_no"},
                "operations_table": "operations",
                "operations_fk": "fa_no",
                "operations_columns": {"name": "op_name"},
            },
        }
        mock_config.return_value = {"tools": {"erp": {}}}

        from kiara.admin.functions.tools import tools_erp_config_update

        result = _run(
            tools_erp_config_update(
                MOCK_USER,
                backend="sql",
                sql_dsn="mssql://newserver/db",
                mapping=mapping,
            )
        )

        assert result.success is True
        saved_mapping = mock_save.call_args[0][0]["tools"]["erp"]["mapping"]
        assert saved_mapping["orders"]["positions_table"] == "positions"
        assert saved_mapping["orders"]["positions_fk"] == "order_no"
        assert saved_mapping["orders"]["positions_columns"] == {"article_id": "article_no"}
        assert saved_mapping["production_orders"]["operations_table"] == "operations"
        assert saved_mapping["production_orders"]["operations_fk"] == "fa_no"
        assert saved_mapping["production_orders"]["operations_columns"] == {"name": "op_name"}


# ---------------------------------------------------------------------------
# tools.erp.test_connection
# ---------------------------------------------------------------------------


class TestToolsErpTestConnection:
    @patch("kiara.config.load_config")
    def test_erp_test_not_sql(self, mock_config):
        """Test 12: Backend ist nicht SQL → fail."""
        mock_config.return_value = {"tools": {"erp": {"backend": "mock"}}}

        from kiara.admin.functions.tools import tools_erp_test_connection

        result = _run(tools_erp_test_connection(MOCK_USER))

        assert result.success is False
        assert "nicht SQL" in result.message

    @patch("kiara.connectors.sql_connector.SQLConnector.test_connection")
    @patch("kiara.config.load_config")
    def test_erp_test_success(self, mock_config, mock_test):
        """Test 13: SQL-Verbindungstest erfolgreich."""
        mock_config.return_value = {
            "tools": {"erp": {"backend": "sql", "sql_dsn": "mssql://ok", "sql_driver": "auto"}},
        }
        mock_test.return_value = {"success": True, "tables": ["Auftraege"], "driver": "mssql"}

        from kiara.admin.functions.tools import tools_erp_test_connection

        result = _run(tools_erp_test_connection(MOCK_USER))

        assert result.success is True
        assert result.data["success"] is True


# ---------------------------------------------------------------------------
# tools.erp.table_schema
# ---------------------------------------------------------------------------


class TestToolsErpTableSchema:
    @patch("kiara.config.load_config")
    def test_table_schema_no_dsn(self, mock_config):
        """Test 14: Kein DSN → Fehlermeldung in data."""
        mock_config.return_value = {"tools": {"erp": {}}}

        from kiara.admin.functions.tools import tools_erp_table_schema

        result = _run(tools_erp_table_schema(MOCK_USER, table_name="orders"))

        assert result.success is True
        assert result.data["columns"] == []
        assert "Kein SQL-DSN" in result.data["error"]


# ---------------------------------------------------------------------------
# tools.connections.list
# ---------------------------------------------------------------------------


class TestToolsConnectionsList:
    @patch("kiara.web.api_tools._get_connector_fields")
    @patch("kiara.web.api_tools._mask_connection_config")
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_connections_list_masked(self, mock_reg, mock_mask, mock_fields):
        """Test 15: Verbindungsliste mit maskierten Passwoertern."""
        conn = MagicMock()
        conn.id = "conn-abc"
        conn.name = "Prod-DB"
        conn.connector_type = "sql"
        conn.config = {"host": "db.local", "password": "secret"}
        conn.enabled = True
        mock_reg.return_value.list_connections.return_value = [conn]
        mock_fields.return_value = [{"name": "password", "type": "password"}]
        mock_mask.return_value = {"host": "db.local", "password": PASSWORD_MASK}

        from kiara.admin.functions.tools import tools_connections_list

        result = _run(tools_connections_list(MOCK_USER))

        assert result.success is True
        assert len(result.data["connections"]) == 1
        assert result.data["connections"][0]["config"]["password"] == PASSWORD_MASK


# ---------------------------------------------------------------------------
# tools.connections.create
# ---------------------------------------------------------------------------


class TestToolsConnectionsCreate:
    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    @patch("kiara.web.api_tools._encrypt_connection_config")
    @patch("kiara.web.api_tools._get_connector_fields")
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_connections_create(self, mock_reg, mock_fields, mock_encrypt, mock_config, mock_save):
        """Test 16: Verbindung erstellen."""
        mock_fields.return_value = []
        mock_encrypt.return_value = {"host": "db.local"}
        mock_reg.return_value.save_to_config.return_value = {"connections": []}
        mock_config.return_value = {}

        from kiara.admin.functions.tools import tools_connections_create

        result = _run(
            tools_connections_create(
                MOCK_USER,
                name="Test-DB",
                type="sql",
                config={"host": "db.local"},
            )
        )

        assert result.success is True
        assert result.data["id"].startswith("conn-")
        assert "angelegt" in result.data["message"]


# ---------------------------------------------------------------------------
# tools.connections.update
# ---------------------------------------------------------------------------


class TestToolsConnectionsUpdate:
    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    @patch("kiara.web.api_tools._encrypt_connection_config")
    @patch("kiara.web.api_tools._get_connector_fields")
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_connections_update_not_found(
        self, mock_reg, mock_fields, mock_encrypt, mock_config, mock_save
    ):
        """Test 17: Nicht existente Verbindung → fail."""
        mock_reg.return_value.list_connections.return_value = []

        from kiara.admin.functions.tools import tools_connections_update

        result = _run(
            tools_connections_update(
                MOCK_USER,
                connection_id="conn-xxx",
                name="X",
                type="sql",
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# tools.connections.delete
# ---------------------------------------------------------------------------


class TestToolsConnectionsDelete:
    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_connections_delete(self, mock_reg, mock_config, mock_save):
        """Test 18: Verbindung loeschen."""
        mock_reg.return_value.save_to_config.return_value = {
            "connections": [],
            "tool_assignments": {},
        }
        mock_config.return_value = {}

        from kiara.admin.functions.tools import tools_connections_delete

        result = _run(tools_connections_delete(MOCK_USER, connection_id="conn-abc"))

        assert result.success is True
        mock_reg.return_value.remove_connection.assert_called_once_with("conn-abc")


# ---------------------------------------------------------------------------
# tools.connections.test
# ---------------------------------------------------------------------------


class TestToolsConnectionsTest:
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_connections_test_not_found(self, mock_reg):
        """Test 19: Nicht existente Verbindung → fail."""
        mock_reg.return_value.get_connector.side_effect = KeyError("not found")

        from kiara.admin.functions.tools import tools_connections_test

        result = _run(tools_connections_test(MOCK_USER, connection_id="conn-xxx"))

        assert result.success is False


# ---------------------------------------------------------------------------
# tools.assignments
# ---------------------------------------------------------------------------


class TestToolsAssignments:
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_assignments_read(self, mock_reg):
        """Test 20: Zuordnungen lesen."""
        mock_reg.return_value.get_tool_assignments.return_value = {
            "erp_order_status": "conn-abc",
        }

        from kiara.admin.functions.tools import tools_assignments

        result = _run(tools_assignments(MOCK_USER))

        assert result.success is True
        assert result.data["assignments"]["erp_order_status"] == "conn-abc"


class TestToolsAssignmentsUpdate:
    @patch("kiara.web.api_tools._connection_registry", None)
    @patch("kiara.web.api_tools._registry", None)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_assignments_update(self, mock_reg, mock_config, mock_save):
        """Test 21: Zuordnungen speichern."""
        mock_reg.return_value._tool_assignments = {}
        mock_reg.return_value.save_to_config.return_value = {
            "tool_assignments": {"search": "conn-abc"},
        }
        mock_config.return_value = {}

        from kiara.admin.functions.tools import tools_assignments_update

        result = _run(
            tools_assignments_update(
                MOCK_USER,
                assignments={"search": "conn-abc"},
            )
        )

        assert result.success is True


# ---------------------------------------------------------------------------
# tools.connector_types
# ---------------------------------------------------------------------------


class TestToolsConnectorTypes:
    @patch("kiara.web.api_tools._get_connection_registry")
    def test_connector_types(self, mock_reg):
        """Test 22: Konnektortypen auflisten."""
        mock_reg.return_value.list_connector_types.return_value = [
            {"type_id": "sql", "name": "SQL-Datenbank"},
        ]

        from kiara.admin.functions.tools import tools_connector_types

        result = _run(tools_connector_types(MOCK_USER))

        assert result.success is True
        assert len(result.data["types"]) == 1
