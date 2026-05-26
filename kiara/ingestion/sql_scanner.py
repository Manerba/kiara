"""SQL-Scanner fuer die Ingestion-Pipeline."""

import hashlib
import logging
import os
import re

from kiara.connectors.sql_connector import SQLConnector

logger = logging.getLogger(__name__)


class SQLScanner:
    """Scannt SQL-Tabellen/Views und schreibt Zeilen als Textdateien."""

    def __init__(self, config: dict):
        """
        Config-Keys (aus Datasource-Spalten):
            sql_dsn: str - Verbindungsstring
            sql_driver: str - "auto" | "mssql" | "postgresql" | "mysql"
            sql_query: str - SELECT-Statement oder View-Name
            sql_text_columns: str - Kommaseparierte Spaltennamen fuer den Text
            sql_id_column: str - Primaerschluessel-Spalte
            sql_timestamp_column: str - Zeitstempel-Spalte fuer inkrementelle Updates (optional)
            sql_text_template: str - Jinja-aehnliches Template fuer den Text (optional)
        """
        self._dsn = config.get("sql_dsn", "")
        self._driver = config.get("sql_driver", "auto")
        self._query = config.get("sql_query", "")
        self._text_columns = [
            c.strip() for c in config.get("sql_text_columns", "").split(",") if c.strip()
        ]
        for col in self._text_columns:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
                raise ValueError(f"Ungueltiger Spaltenname fuer sql_text_columns: {col!r}")
        self._id_column = config.get("sql_id_column", "id")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self._id_column):
            raise ValueError(f"Ungueltiger Spaltenname fuer sql_id_column: {self._id_column!r}")
        self._timestamp_column = config.get("sql_timestamp_column", "")
        if self._timestamp_column and not re.match(
            r"^[a-zA-Z_][a-zA-Z0-9_]*$", self._timestamp_column
        ):
            raise ValueError(
                f"Ungueltiger Spaltenname fuer sql_timestamp_column: {self._timestamp_column!r}"
            )
        self._text_template = config.get("sql_text_template", "")

    def test_connection(self) -> dict:
        """Verbindungstest mit Tabellenvorschau."""
        connector = SQLConnector(self._dsn, self._driver)
        return connector.test_connection()

    def scan(self, output_dir: str, since_timestamp: str | None = None) -> list[str]:
        """Fuehrt Query aus und schreibt Zeilen als Textdateien.

        Jede Zeile wird zu einer .txt-Datei im output_dir:
            [Spalte1]: Wert1
            [Spalte2]: Wert2
            ...

        Args:
            output_dir: Zielverzeichnis fuer die Textdateien.
            since_timestamp: Wenn gesetzt, nur Zeilen nach diesem Zeitstempel
                            (fuer inkrementelle Updates).

        Returns:
            Liste der geschriebenen Dateipfade.
        """
        connector = SQLConnector(self._dsn, self._driver)
        os.makedirs(output_dir, exist_ok=True)

        # Query vorbereiten
        query = self._query
        params = None

        # Wenn Query nur ein Tabellenname ist, SELECT * daraus machen
        if not query.strip().upper().startswith("SELECT"):
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$", query.strip()):
                raise ValueError(f"Ungueltiger Tabellenname: {query!r}")
            query = f"SELECT * FROM {query}"

        # Inkrementeller Filter
        if since_timestamp and self._timestamp_column:
            # Placeholder ist treiberabhaengig: ? fuer pyodbc, %s fuer psycopg2/pymysql
            placeholder = "?" if connector._driver == "mssql" else "%s"
            query = self._wrap_incremental_query(
                query,
                self._timestamp_column,
                placeholder,
                connector._driver,
            )
            params = (since_timestamp,)

        result = connector.execute_query(query, params)
        columns = result["columns"]
        rows = result["rows"]

        logger.info("SQL-Scan: %d Zeilen aus Query", len(rows))

        files = []
        for row in rows:
            row_dict = dict(zip(columns, row, strict=False))

            # ID fuer Dateinamen
            row_id = str(row_dict.get(self._id_column, ""))
            if not row_id:
                # Hash ueber sortierte (column, value)-Paare: stabil gegenueber Spalten-Reorder im Query.
                serialized = "|".join(f"{k}={row_dict[k]}" for k in sorted(row_dict))
                row_id = hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]

            # Text generieren
            if self._text_template:
                text = self._render_template(self._text_template, row_dict)
            elif self._text_columns:
                text = self._columns_to_text(row_dict, self._text_columns)
            else:
                text = self._row_to_text(row_dict, columns)

            if not text.strip():
                continue

            # Datei schreiben. Der Hash-Suffix verhindert Kollisionen zwischen
            # unterschiedlich normalisierten Row-IDs wie "a/b" und "a:b".
            safe_id = self._row_filename_id(row_id)
            filepath = os.path.join(output_dir, f"row_{safe_id}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            files.append(filepath)

        # Bei Vollscan: verwaiste row_*.txt entfernen (geloeschte DB-Zeilen)
        if not since_timestamp:
            current_files = set(files)
            for existing in os.listdir(output_dir):
                if existing.startswith("row_") and existing.endswith(".txt"):
                    full_path = os.path.join(output_dir, existing)
                    if full_path not in current_files:
                        os.remove(full_path)
                        logger.debug("SQL-Scan: Verwaiste Datei entfernt: %s", existing)

        return files

    @staticmethod
    def _quote_identifier(driver: str, identifier: str) -> str:
        # Defense-in-depth: Constructor-Regex blockiert bereits gefaehrliche
        # Zeichen, das Quoting haertet zusaetzlich gegen Reserved Keywords
        # und schuetzt vor zukuenftigen Lockerungen der Validierung.
        if driver == "mssql":
            return "[" + identifier.replace("]", "]]") + "]"
        if driver == "mysql":
            return "`" + identifier.replace("`", "``") + "`"
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _wrap_incremental_query(
        query: str, timestamp_column: str, placeholder: str, driver: str
    ) -> str:
        """Kapselt die konfigurierte Query fuer inkrementelle Filterung."""
        base_query = query.strip().rstrip(";").strip()
        quoted_col = SQLScanner._quote_identifier(driver, timestamp_column)
        return (
            f"SELECT * FROM ({base_query}) AS kiara_sql_scan "
            f"WHERE kiara_sql_scan.{quoted_col} > {placeholder}"
        )

    @staticmethod
    def _row_filename_id(row_id: str) -> str:
        """Erzeugt einen stabilen, kollisionsarmen Dateinamen-Teil fuer Row-IDs."""
        safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in row_id).strip("_")
        safe_prefix = re.sub(r"_+", "_", safe_prefix)[:80] or "id"
        digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:16]
        return f"{safe_prefix}_{digest}"

    @staticmethod
    def _columns_to_text(row: dict, columns: list[str]) -> str:
        """Erzeugt Text aus ausgewaehlten Spalten."""
        lines = []
        for col in columns:
            value = row.get(col, "")
            if value is not None and str(value).strip():
                lines.append(f"{col}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _row_to_text(row: dict, columns: list[str]) -> str:
        """Erzeugt Text aus allen Spalten."""
        lines = []
        for col in columns:
            value = row.get(col, "")
            if value is not None and str(value).strip():
                lines.append(f"{col}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _render_template(template: str, row: dict) -> str:
        """Einfache Template-Engine: {spaltenname} wird ersetzt."""

        def replacer(match):
            key = match.group(1)
            return str(row.get(key, ""))

        return re.sub(r"\{(\w+)\}", replacer, template)
