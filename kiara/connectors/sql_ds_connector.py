"""SQLDatasourceConnector — SQL-Datenbank als Datenquelle."""

import logging
import os

from kiara.config import DEFAULT_PATHS
from kiara.connectors.base import ConnectorBase
from kiara.connectors.registry import register
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


@register
class SQLDatasourceConnector(ConnectorBase):
    """Konnektor fuer SQL-Datenbanken als Datenquelle.

    Fuehrt eine SQL-Query gegen eine Datenbank aus und exportiert die
    Ergebnisse als Textdateien in den lokalen SQL-Cache. Unterstuetzt
    MSSQL, PostgreSQL und MySQL (Auto-Erkennung via DSN).

    Config-Felder:
        dsn:                Verbindungsstring (verschluesselt gespeichert)
        driver:             Treiber (auto/mssql/postgresql/mysql)
        query:              SQL-Query zur Datenextraktion
        text_columns:       Kommaseparierte Spalten fuer Textinhalt
        id_column:          Primaerschluessel-Spalte (default "id")
        timestamp_column:   Zeitstempel-Spalte fuer inkrementelle Updates
        text_template:      Template fuer die Textformatierung (z.B. "{name}: {beschreibung}")
    """

    name = ConnectorType.SQL
    label = "SQL-Datenbank"
    badge_class = "success"

    @staticmethod
    def config_schema() -> dict:
        return {
            "dsn": {
                "type": "str",
                "default": "",
                "label": "Verbindungsstring (DSN)",
                "placeholder": "postgresql://user:pass@host:5432/dbname",
                "required": True,
                "secret": True,
            },
            "driver": {
                "type": "str",
                "default": "auto",
                "label": "Treiber",
                "options": [
                    {"value": "auto", "label": "Auto-Erkennung"},
                    {"value": "mssql", "label": "MSSQL"},
                    {"value": "postgresql", "label": "PostgreSQL"},
                    {"value": "mysql", "label": "MySQL"},
                ],
            },
            "query": {
                "type": "str",
                "default": "",
                "label": "SQL-Query",
                "placeholder": "SELECT * FROM protokolle",
                "multiline": True,
            },
            "text_columns": {
                "type": "str",
                "default": "",
                "label": "Text-Spalten",
                "placeholder": "beschreibung, massnahme",
            },
            "id_column": {
                "type": "str",
                "default": "id",
                "label": "ID-Spalte",
            },
            "timestamp_column": {
                "type": "str",
                "default": "",
                "label": "Zeitstempel-Spalte",
                "placeholder": "updated_at",
            },
            "text_template": {
                "type": "str",
                "default": "",
                "label": "Text-Template",
                "placeholder": "{name}: {beschreibung}",
            },
        }

    @staticmethod
    def source_display(config: dict, row: dict) -> str:
        dsn = config.get("dsn", "")
        query = config.get("query", "")
        if dsn:
            driver = config.get("driver", "auto")
            short_query = query[:40] + "..." if len(query) > 40 else query
            return f"[{driver}] {short_query}" if short_query else f"[{driver}]"
        return ""

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        return os.path.join(DEFAULT_PATHS["sql_cache"], f"ds{row['id']}")

    @staticmethod
    def _build_scanner_config(config: dict) -> dict:
        """Baut das Scanner-Config-Dict aus dem Connector-Config."""
        return {
            "sql_dsn": config.get("dsn", ""),
            "sql_driver": config.get("driver", "auto"),
            "sql_query": config.get("query", ""),
            "sql_text_columns": config.get("text_columns", ""),
            "sql_id_column": config.get("id_column", "id"),
            "sql_timestamp_column": config.get("timestamp_column", ""),
            "sql_text_template": config.get("text_template", ""),
        }

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        from kiara.ingestion.sql_scanner import SQLScanner

        scanner = SQLScanner(SQLDatasourceConnector._build_scanner_config(config))
        result = scanner.test_connection()
        if result["success"]:
            tables = result.get("tables", [])
            return {
                "success": True,
                "message": f"Verbindung erfolgreich ({result['driver']}) — {len(tables)} Tabellen",
            }
        return result

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        from kiara.ingestion.sql_scanner import SQLScanner

        scanner = SQLScanner(SQLDatasourceConnector._build_scanner_config(config))
        path = os.path.join(DEFAULT_PATHS["sql_cache"], f"ds{row['id']}")
        os.makedirs(path, exist_ok=True)
        try:
            scanner.scan(path)
            return path
        except Exception as exc:
            from kiara.connectors.sql_connector import SQLConnector

            logger.error(
                "SQL scan failed for ds%s: %s",
                row["id"],
                SQLConnector._redact_secret_text(str(exc)),
            )
            return None
