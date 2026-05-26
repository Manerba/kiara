"""Tests fuer den SQL-Connector (Phase 6)."""

import pytest

from kiara.connectors.sql_connector import SQLConnector
from kiara.exceptions import ConnectorError


class TestDetectDriver:
    """DSN-Erkennung fuer verschiedene Datenbank-Typen."""

    def test_detect_driver_mssql_odbc(self):
        dsn = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=srv;DATABASE=erp"
        assert SQLConnector._detect_driver(dsn) == "mssql"

    def test_detect_driver_mssql_keyword(self):
        assert SQLConnector._detect_driver("mssql://server/db") == "mssql"

    def test_detect_driver_postgresql(self):
        dsn = "postgresql://user:pass@localhost:5432/dbname"
        assert SQLConnector._detect_driver(dsn) == "postgresql"

    def test_detect_driver_mysql(self):
        dsn = "mysql://user:pass@localhost:3306/dbname"
        assert SQLConnector._detect_driver(dsn) == "mysql"

    def test_detect_driver_db_extension_falls_through_to_mssql(self):
        """.db/.sqlite DSNs fallen jetzt auf mssql-Default durch."""
        assert SQLConnector._detect_driver("/tmp/test.db") == "mssql"
        assert SQLConnector._detect_driver("/tmp/data.sqlite") == "mssql"

    def test_detect_driver_default_mssql(self):
        """Unbekannte DSN-Strings -> Default mssql."""
        assert SQLConnector._detect_driver("some-unknown-string") == "mssql"


class TestSQLiteDriverRemoved:
    """SQLite-Treiber wurde entfernt — Regression-Tests."""

    def test_sqlite_driver_not_supported(self):
        """driver='sqlite' loest ConnectorError aus."""
        connector = SQLConnector("test.db", driver="sqlite")
        with pytest.raises(ConnectorError, match="Unbekannter SQL-Treiber"):
            connector._connect()
