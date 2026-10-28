"""Tests fuer SQL-Connector Edge-Cases (Phase 6 Erweiterung)."""

from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors.sql_connector import SQLConnector


class TestUnknownDriverRaises:
    """Unbekannter Treiber loest ConnectorError aus."""

    def test_unknown_driver_raises(self):
        """SQLConnector mit driver='oracle' -> ConnectorError."""
        from kiara.exceptions import ConnectorError

        connector = SQLConnector("some_dsn", driver="oracle")
        with pytest.raises(ConnectorError, match="Unbekannter SQL-Treiber"):
            connector._connect()


class TestSqlSecretRedaction:
    """SQL-Connector darf DSN-Secrets nicht in Fehlertexte schreiben."""

    def test_mssql_url_error_redacts_password(self):
        from kiara.exceptions import ConnectorError

        connector = SQLConnector(
            "postgresql://user:supersecret@example.local:5432/db",
            driver="mssql",
        )

        with pytest.raises(ConnectorError) as exc_info:
            connector._connect()

        msg = str(exc_info.value)
        assert "supersecret" not in msg
        assert "user:***@" in msg

    def test_test_connection_redacts_driver_error(self):
        dsn = "DRIVER={ODBC};SERVER=srv;UID=user;PWD=supersecret;DATABASE=db"
        connector = SQLConnector(dsn, driver="mssql")

        with patch.object(
            connector,
            "_connect",
            side_effect=RuntimeError(
                "Login failed for DRIVER={ODBC};UID=user;PWD=supersecret;DATABASE=db"
            ),
        ):
            result = connector.test_connection()

        assert result["success"] is False
        assert "supersecret" not in result["message"]
        assert "PWD=***" in result["message"]


class TestExecuteQueryEmptyResult:
    """Query ohne Ergebniszeilen — mock-basiert (kein SQLite)."""

    def test_execute_query_empty_result(self):
        """Query liefert 0 Zeilen -> leere rows-Liste, row_count=0."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",), ("value",)]
        mock_cursor.fetchmany.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        connector = SQLConnector(
            "DRIVER={ODBC Driver 18};SERVER=srv;DATABASE=db",
            driver="mssql",
        )
        with patch.object(connector, "_connect", return_value=mock_conn):
            result = connector.execute_query("SELECT * FROM test_table")
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"] == []
        assert result["row_count"] == 0


class TestGetTableSchemaEmptyTable:
    """Leere Tabelle -> Schema wird trotzdem zurueckgegeben — mock-basiert."""

    def test_get_table_schema_empty_table(self):
        """Tabelle hat Spalten aber keine Daten -> Schema korrekt."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Simulate INFORMATION_SCHEMA.COLUMNS result for MSSQL
        mock_cursor.fetchall.return_value = [
            ("id", "int", "NO"),
            ("name", "nvarchar", "YES"),
            ("value", "float", "YES"),
        ]
        mock_conn.cursor.return_value = mock_cursor

        connector = SQLConnector(
            "DRIVER={ODBC Driver 18};SERVER=srv;DATABASE=db",
            driver="mssql",
        )
        with patch.object(connector, "_connect", return_value=mock_conn):
            schema = connector.get_table_schema("test_table")
        assert len(schema) == 3
        col_names = [c["name"] for c in schema]
        assert "id" in col_names
        assert "name" in col_names
        assert "value" in col_names
        for col in schema:
            assert "type" in col
            assert "nullable" in col


class TestTestConnectionSuccessReturnsTables:
    """test_connection() gibt Tabellenliste zurueck — mock-basiert."""

    def test_test_connection_success_returns_tables(self):
        """Erfolgreicher Test -> success=True, tables-Liste, driver."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # _list_tables for MSSQL reads from INFORMATION_SCHEMA.TABLES
        mock_cursor.fetchall.return_value = [("test_table",)]
        mock_conn.cursor.return_value = mock_cursor

        connector = SQLConnector(
            "DRIVER={ODBC Driver 18};SERVER=srv;DATABASE=db",
            driver="mssql",
        )
        with patch.object(connector, "_connect", return_value=mock_conn):
            result = connector.test_connection()
        assert result["success"] is True
        assert "test_table" in result["tables"]
        assert result["driver"] == "mssql"


class TestConnectionContextManager:
    """Context-Manager schliesst Verbindung — mock-basiert."""

    def test_connection_context_manager(self):
        """with connector.connection() as conn -> conn.close() wird aufgerufen."""
        mock_conn = MagicMock()

        connector = SQLConnector(
            "DRIVER={ODBC Driver 18};SERVER=srv;DATABASE=db",
            driver="mssql",
        )
        with patch.object(connector, "_connect", return_value=mock_conn):
            with connector.connection() as conn:
                assert conn._raw_conn is mock_conn
        mock_conn.close.assert_called_once()


class TestMSSQLConnectMocked:
    """MSSQL-Verbindung mit Mock (kein echter Server noetig)."""

    def test_mssql_connect_mocked(self):
        """Mock pyodbc.connect() -> wird mit DSN aufgerufen."""
        mock_pyodbc = MagicMock()
        mock_conn = MagicMock()
        mock_pyodbc.connect.return_value = mock_conn

        connector = SQLConnector(
            "DRIVER={ODBC Driver 18};SERVER=srv;DATABASE=db",
            driver="mssql",
        )

        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        with patch.dict("sys.modules", {"pyodbc": mock_pyodbc}):
            result = connector._connect()
            mock_pyodbc.connect.assert_called_once_with(
                "DRIVER={ODBC Driver 18};SERVER=srv;DATABASE=db",
                timeout=CONNECTOR_CONNECT_TIMEOUT,
            )
            assert result == mock_conn


class TestPostgreSQLConnectMocked:
    """PostgreSQL-Verbindung mit Mock."""

    def test_postgresql_connect_mocked(self):
        """Mock psycopg2.connect() -> wird mit DSN aufgerufen."""
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn

        connector = SQLConnector(
            "postgresql://user:pass@host:5432/dbname",
            driver="postgresql",
        )

        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            result = connector._connect()
            mock_psycopg2.connect.assert_called_once_with(
                "postgresql://user:pass@host:5432/dbname",
                connect_timeout=CONNECTOR_CONNECT_TIMEOUT,
            )
            assert result == mock_conn


class TestMySQLDSNWithoutPort:
    """MySQL-DSN ohne Port -> Fallback-Verhalten (dokumentierter Bug)."""

    def test_mysql_dsn_without_port(self):
        """MySQL-DSN ohne Port → urlparse extrahiert host/user/pass/db korrekt,
        Port faellt auf Default 3306 zurueck.
        """
        mock_pymysql = MagicMock()
        mock_conn = MagicMock()
        mock_pymysql.connect.return_value = mock_conn

        dsn = "mysql://user:pass@host/db"
        connector = SQLConnector(dsn, driver="mysql")

        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT, SQL_STATEMENT_TIMEOUT

        with patch.dict("sys.modules", {"pymysql": mock_pymysql}):
            result = connector._connect()
            # urlparse extrahiert Felder korrekt, Port Default 3306
            mock_pymysql.connect.assert_called_once_with(
                host="host",
                port=3306,
                user="user",
                password="pass",
                database="db",
                charset="utf8mb4",
                connect_timeout=CONNECTOR_CONNECT_TIMEOUT,
                read_timeout=SQL_STATEMENT_TIMEOUT,
                write_timeout=SQL_STATEMENT_TIMEOUT,
            )
            mock_conn.cursor.return_value.execute.assert_called_once_with(
                f"SET SESSION MAX_EXECUTION_TIME = {SQL_STATEMENT_TIMEOUT * 1000}"
            )
            assert result == mock_conn
