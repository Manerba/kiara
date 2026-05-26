"""SQL-Datenbankkonnektor fuer Ingestion und ERP-Abfragen."""

import logging
import math
import re
from contextlib import closing, contextmanager
from urllib.parse import quote, unquote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)


class _StatementTimeoutConnection:
    """Connection-Proxy, der pro Cursor den DB-API-Timeout setzt."""

    def __init__(self, raw_conn, driver: str, timeout_seconds: float) -> None:
        self._raw_conn = raw_conn
        self._driver = driver
        self._timeout_seconds = timeout_seconds

    def cursor(self, *args, **kwargs):
        cur = self._raw_conn.cursor(*args, **kwargs)
        if self._driver == "mssql":
            timeout_s = math.ceil(self._timeout_seconds)
            try:
                cur.timeout = timeout_s
            except Exception as exc:
                from kiara.exceptions import ConnectorError

                raise ConnectorError("MSSQL-Cursor unterstuetzt keinen Statement-Timeout.") from exc
        return cur

    def __getattr__(self, name: str):
        return getattr(self._raw_conn, name)


class SQLConnector:
    """Generischer SQL-Konnektor. Unterstuetzt MSSQL, PostgreSQL, MySQL.

    Nutzt DB-API 2.0 (PEP 249) — funktioniert mit pyodbc, psycopg2,
    pymysql etc.
    """

    def __init__(
        self,
        dsn: str,
        driver: str = "auto",
        statement_timeout: float | int | None = None,
    ):
        """
        Args:
            dsn: Verbindungsstring.
                 MSSQL: "DRIVER={ODBC Driver 18 for SQL Server};SERVER=srv;DATABASE=erp;UID=user;PWD=pass"
                 PostgreSQL: "postgresql://user:pass@host:5432/dbname"
                 MySQL: "mysql://user:pass@host:3306/dbname"
            driver: "auto" (erkennt aus DSN), "mssql", "postgresql", "mysql"
            statement_timeout: Timeout in Sekunden fuer einzelne Statements.
        """
        from kiara.shared.constants import SQL_STATEMENT_TIMEOUT

        self._dsn = dsn
        self._driver = driver if driver != "auto" else self._detect_driver(dsn)
        timeout = SQL_STATEMENT_TIMEOUT if statement_timeout is None else statement_timeout
        self._statement_timeout = float(timeout)
        if self._statement_timeout <= 0:
            from kiara.exceptions import ConnectorError

            raise ConnectorError("SQL-Statement-Timeout muss groesser als 0 sein.")

    @staticmethod
    def _detect_driver(dsn: str) -> str:
        """Erkennt den Treiber aus dem DSN-String."""
        dsn_lower = dsn.lower()
        if "driver=" in dsn_lower or "mssql" in dsn_lower or "sqlserver" in dsn_lower:
            return "mssql"
        if dsn_lower.startswith("postgresql"):
            return "postgresql"
        if dsn_lower.startswith("mysql"):
            return "mysql"
        return "mssql"  # Haeufigster On-Premise-Datenbanktyp

    @staticmethod
    def _redact_secret_text(text: str) -> str:
        """Entfernt Passwoerter aus SQL-DSNs und Treiber-Fehlermeldungen."""
        redacted = str(text)
        redacted = re.sub(
            r"(?i)\b(PWD|PASSWORD)\s*=\s*(?:\{[^}]*\}|[^;]*)",
            lambda m: f"{m.group(1)}=***",
            redacted,
        )
        redacted = re.sub(
            r"([a-z][a-z0-9+.-]*://)([^:@/\s]+):([^@/\s]+)@",
            r"\1\2:***@",
            redacted,
            flags=re.IGNORECASE,
        )
        return redacted

    @classmethod
    def _redact_dsn(cls, dsn: str, max_len: int = 80) -> str:
        """Redigiert einen DSN fuer Fehlermeldungen und begrenzt die Laenge."""
        redacted = cls._redact_secret_text(dsn)
        try:
            parsed = urlsplit(dsn)
            if parsed.scheme and parsed.netloc and parsed.password:
                username = quote(unquote(parsed.username or ""), safe="")
                host = parsed.hostname or ""
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                port = f":{parsed.port}" if parsed.port else ""
                netloc = f"{username}:***@{host}{port}" if username else f"***@{host}{port}"
                redacted = urlunsplit(
                    (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
                )
        except Exception:
            pass
        return redacted if len(redacted) <= max_len else redacted[:max_len] + "..."

    @contextmanager
    def connection(self):
        """Context-Manager fuer DB-Verbindung."""
        raw_conn = self._connect()
        conn = _StatementTimeoutConnection(
            raw_conn,
            self._driver,
            self._statement_timeout,
        )
        try:
            yield conn
        except Exception as exc:
            self._handle_connection_exception(raw_conn, exc)
        finally:
            try:
                raw_conn.close()
            except Exception:
                logger.debug("SQL-Connection konnte nicht geschlossen werden.", exc_info=True)

    def _connect(self):
        """Stellt Verbindung her basierend auf Treiber-Typ."""
        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        conn = None
        if self._driver == "mssql":
            if re.match(r"^[a-z]+://", self._dsn, re.IGNORECASE):
                from kiara.exceptions import ConnectorError

                safe_dsn = self._redact_dsn(self._dsn)
                raise ConnectorError(
                    f"MSSQL erfordert einen ODBC Connection String, kein URL-Format. "
                    f"Erhalten: {safe_dsn} — "
                    f"Beispiel: DRIVER={{ODBC Driver 18 for SQL Server}};"
                    f"SERVER=srv;DATABASE=db;UID=user;PWD=pass"
                )
            import pyodbc

            conn = pyodbc.connect(self._dsn, timeout=CONNECTOR_CONNECT_TIMEOUT)
        elif self._driver == "postgresql":
            import psycopg2

            conn = psycopg2.connect(self._dsn, connect_timeout=CONNECTOR_CONNECT_TIMEOUT)
        elif self._driver == "mysql":
            from urllib.parse import unquote, urlparse

            import pymysql

            parsed = urlparse(self._dsn)
            timeout_s = math.ceil(self._statement_timeout)
            conn = pymysql.connect(
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                user=unquote(parsed.username or ""),
                password=unquote(parsed.password or ""),
                database=unquote(parsed.path.lstrip("/")) or None,
                charset="utf8mb4",
                connect_timeout=CONNECTOR_CONNECT_TIMEOUT,
                read_timeout=timeout_s,
                write_timeout=timeout_s,
            )
        else:
            from kiara.exceptions import ConnectorError

            raise ConnectorError(f"Unbekannter SQL-Treiber: {self._driver}")
        try:
            self._apply_session_statement_timeout(conn)
        except Exception as exc:
            logger.error(
                "SQL-Statement-Timeout-Setup fehlgeschlagen (driver=%s, dsn=%s): %s",
                self._driver,
                self._redact_dsn(self._dsn),
                self._redact_secret_text(str(exc)),
            )
            try:
                conn.close()
            except Exception:
                pass
            raise
        return conn

    def _apply_session_statement_timeout(self, conn) -> None:
        """Setzt server-/sessionseitige Statement-Timeouts fuer SQL-Reads."""
        timeout_ms = math.ceil(self._statement_timeout * 1000)
        if self._driver == "postgresql":
            self._execute_connection_setup(
                conn,
                f"SET statement_timeout = {timeout_ms}",
                autocommit=True,
            )
        elif self._driver == "mysql":
            self._execute_connection_setup(
                conn,
                f"SET SESSION MAX_EXECUTION_TIME = {timeout_ms}",
            )
        elif self._driver == "mssql":
            try:
                conn.timeout = math.ceil(self._statement_timeout)
            except Exception as exc:
                from kiara.exceptions import ConnectorError

                raise ConnectorError(
                    "MSSQL-Connection unterstuetzt keinen Statement-Timeout."
                ) from exc

    @staticmethod
    def _execute_connection_setup(conn, sql: str, *, autocommit: bool = False) -> None:
        missing = object()
        old_autocommit = missing
        if autocommit:
            try:
                old_autocommit = conn.autocommit
                conn.autocommit = True
            except Exception:
                old_autocommit = missing
        cur = conn.cursor()
        try:
            cur.execute(sql)
        finally:
            close = getattr(cur, "close", None)
            if callable(close):
                close()
            if autocommit and old_autocommit is not missing:
                try:
                    conn.autocommit = old_autocommit
                except Exception:
                    logger.debug(
                        "SQL-Autocommit konnte nicht wiederhergestellt werden.", exc_info=True
                    )

    def _handle_connection_exception(self, raw_conn, exc: Exception) -> None:
        try:
            raw_conn.rollback()
        except Exception:
            pass
        if self._is_statement_timeout_error(exc):
            from kiara.exceptions import StatementTimeout

            raise StatementTimeout(
                f"SQL-Statement nach {self._statement_timeout:g}s abgebrochen."
            ) from exc
        raise exc

    def _is_statement_timeout_error(self, exc: Exception) -> bool:
        code = getattr(exc, "pgcode", None)
        if code == "57014":
            return True
        name = exc.__class__.__name__.lower()
        if "querycanceled" in name or "querycancelled" in name:
            return True
        args = getattr(exc, "args", ())
        if args:
            first = args[0]
            if first in (1317, 3024):
                return True
        text = str(exc).lower()
        if "statement timeout" in text or "max_execution_time" in text:
            return True
        if "hyt00" in text or "hyt01" in text:
            return True
        return bool("query timeout" in text or "timeout expired" in text)

    def test_connection(self) -> dict:
        """Verbindungstest. Gibt Tabellenliste zurueck."""
        try:
            with self.connection() as conn, closing(conn.cursor()) as cur:
                tables = self._list_tables(cur)
                return {"success": True, "tables": tables, "driver": self._driver}
        except Exception as exc:
            return {
                "success": False,
                "message": self._redact_secret_text(str(exc)),
                "driver": self._driver,
            }

    def _list_tables(self, cursor) -> list[str]:
        """Listet Tabellen/Views auf (treiberspezifisch)."""
        if self._driver == "mssql":
            cursor.execute(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_TYPE IN ('BASE TABLE', 'VIEW') ORDER BY TABLE_NAME"
            )
        elif self._driver == "postgresql":
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        elif self._driver == "mysql":
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() ORDER BY table_name"
            )
        else:
            return []
        return [row[0] for row in cursor.fetchall()]

    def execute_query(self, query: str, params: tuple = None, max_rows: int | None = None) -> dict:
        """Fuehrt eine parametrisierte Query aus.

        Returns:
            {"columns": [str], "rows": [tuple], "row_count": int}
        """
        from kiara.shared.constants import SQL_QUERY_MAX_ROWS_DEFAULT

        if max_rows is None:
            max_rows = SQL_QUERY_MAX_ROWS_DEFAULT
        with self.connection() as conn, closing(conn.cursor()) as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }

    def get_table_schema(self, table_name: str) -> list[dict]:
        """Gibt die Spalten einer Tabelle zurueck.

        Returns:
            [{"name": str, "type": str, "nullable": bool}]
        """
        with self.connection() as conn, closing(conn.cursor()) as cur:
            if self._driver == "mssql":
                cur.execute(
                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
                    "FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
                    (table_name,),
                )
            elif self._driver in ("postgresql", "mysql"):
                cur.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_name = %s ORDER BY ordinal_position",
                    (table_name,),
                )
            return [{"name": r[0], "type": r[1], "nullable": r[2] == "YES"} for r in cur.fetchall()]
