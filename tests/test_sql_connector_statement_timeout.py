"""Tests fuer treiberseitige SQL-Statement-Timeouts."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kiara.connectors.sql_connector import SQLConnector
from kiara.exceptions import StatementTimeout


def test_postgresql_sets_session_statement_timeout(monkeypatch):
    cursor = MagicMock()
    conn = MagicMock()
    conn.autocommit = False
    conn.cursor.return_value = cursor
    psycopg2 = SimpleNamespace(connect=MagicMock(return_value=conn))
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)

    connector = SQLConnector(
        "postgresql://user:pass@host/db",
        "postgresql",
        statement_timeout=12,
    )
    raw = connector._connect()

    assert raw is conn
    cursor.execute.assert_called_once_with("SET statement_timeout = 12000")
    assert conn.autocommit is False


def test_mssql_sets_cursor_timeout(monkeypatch):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    pyodbc = SimpleNamespace(connect=MagicMock(return_value=conn))
    monkeypatch.setitem(sys.modules, "pyodbc", pyodbc)

    connector = SQLConnector("DRIVER={ODBC};SERVER=db", "mssql", statement_timeout=7)
    with connector.connection() as wrapped:
        cur = wrapped.cursor()

    assert cur is cursor
    assert cursor.timeout == 7
    assert conn.timeout == 7
    conn.close.assert_called_once()


def test_mysql_sets_session_timeout_and_socket_timeouts(monkeypatch):
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    pymysql = SimpleNamespace(connect=MagicMock(return_value=conn))
    monkeypatch.setitem(sys.modules, "pymysql", pymysql)

    connector = SQLConnector(
        "mysql://user:pass@db.example/erp",
        "mysql",
        statement_timeout=9,
    )
    raw = connector._connect()

    assert raw is conn
    kwargs = pymysql.connect.call_args.kwargs
    assert kwargs["read_timeout"] == 9
    assert kwargs["write_timeout"] == 9
    cursor.execute.assert_called_once_with("SET SESSION MAX_EXECUTION_TIME = 9000")


def test_statement_timeout_exception_rolls_back_and_is_mapped(monkeypatch):
    conn = MagicMock()
    pyodbc = SimpleNamespace(connect=MagicMock(return_value=conn))
    monkeypatch.setitem(sys.modules, "pyodbc", pyodbc)

    connector = SQLConnector("DRIVER={ODBC};SERVER=db", "mssql", statement_timeout=5)

    with pytest.raises(StatementTimeout, match="5s"), connector.connection():
        raise RuntimeError("[HYT00] Query timeout expired")

    conn.rollback.assert_called_once()
    conn.close.assert_called_once()
