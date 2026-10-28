"""Tests fuer den SQL-Scanner (Phase 6)."""

import os
from unittest.mock import MagicMock, patch

from kiara.ingestion.sql_scanner import SQLScanner

# -- Hilfsfunktion: SQLConnector-Mock erzeugen --


def _mock_connector(rows, columns, tables=None):
    """Erzeugt einen Mock fuer SQLConnector mit vordefinierten Ergebnissen.

    Args:
        rows: Liste von Tupeln (Zeilen).
        columns: Liste von Spaltennamen.
        tables: Tabellenliste fuer test_connection() (optional).
    """
    mock = MagicMock()
    mock.execute_query.return_value = {"columns": columns, "rows": rows}
    if tables is not None:
        mock.test_connection.return_value = {"success": True, "tables": tables}
    return mock


class TestScanCreatesTextFiles:
    """SQL-Scan erzeugt Textdateien aus Datenbankzeilen."""

    COLUMNS = ["id", "titel", "beschreibung", "massnahme", "erstellt_am"]
    ROWS = [
        (1, "Pruefung Maschine A", "Vibration erhoeht", "Lager tauschen", "2024-01-15"),
        (2, "Kalibrierung Sensor", "Abweichung 0.5%", "Neukalibrierung", "2024-02-20"),
        (3, "Wartung Foerderband", "Verschleiss Riemen", "Riemen erneuern", "2024-03-10"),
    ]

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_creates_text_files(self, MockConnCls, tmp_path):
        MockConnCls.return_value = _mock_connector(self.ROWS, self.COLUMNS)
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle",
                "sql_id_column": "id",
            }
        )

        files = scanner.scan(output_dir)
        assert len(files) == 3
        for f in files:
            assert os.path.exists(f)
            content = open(f, encoding="utf-8").read()
            assert "titel:" in content or "id:" in content

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_with_text_columns(self, MockConnCls, tmp_path):
        MockConnCls.return_value = _mock_connector(self.ROWS, self.COLUMNS)
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle",
                "sql_text_columns": "titel, beschreibung, massnahme",
                "sql_id_column": "id",
            }
        )

        files = scanner.scan(output_dir)
        assert len(files) == 3
        content = open(files[0], encoding="utf-8").read()
        assert "titel:" in content
        assert "beschreibung:" in content
        assert "massnahme:" in content
        # ID-Spalte sollte NICHT im Text sein (nur gemappte Spalten)
        assert "id:" not in content

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_table_name_only(self, MockConnCls, tmp_path):
        """Wenn nur ein Tabellenname (kein SELECT) -> SELECT * FROM tabelle."""
        MockConnCls.return_value = _mock_connector(self.ROWS, self.COLUMNS)
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "protokolle",
                "sql_id_column": "id",
            }
        )

        files = scanner.scan(output_dir)
        assert len(files) == 3


class TestColumnsToText:
    """Spalten-Mapping zu Text."""

    def test_columns_to_text(self):
        row = {"titel": "Test", "beschreibung": "Details", "empty": "", "none_val": None}
        result = SQLScanner._columns_to_text(row, ["titel", "beschreibung", "empty", "none_val"])
        assert "titel: Test" in result
        assert "beschreibung: Details" in result
        assert "empty" not in result
        assert "none_val" not in result

    def test_columns_to_text_missing_column(self):
        row = {"titel": "Test"}
        result = SQLScanner._columns_to_text(row, ["titel", "nonexistent"])
        assert "titel: Test" in result


class TestRenderTemplate:
    """Template-Rendering mit {spaltenname}."""

    def test_render_template(self):
        template = "Protokoll: {titel}\nBeschreibung: {beschreibung}\nDatum: {erstellt_am}"
        row = {"titel": "Pruefung", "beschreibung": "Alles OK", "erstellt_am": "2024-01-01"}
        result = SQLScanner._render_template(template, row)
        assert result == "Protokoll: Pruefung\nBeschreibung: Alles OK\nDatum: 2024-01-01"

    def test_render_template_missing_key(self):
        template = "Name: {name}, Wert: {missing}"
        row = {"name": "Test"}
        result = SQLScanner._render_template(template, row)
        assert "Name: Test" in result
        assert "Wert: " in result  # missing key -> leerer String

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_with_template(self, MockConnCls, tmp_path):
        """Scan mit Text-Template statt Spalten-Mapping."""
        MockConnCls.return_value = _mock_connector(
            rows=[(1, "Widget", "aktiv")],
            columns=["id", "name", "status"],
        )
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM items",
                "sql_id_column": "id",
                "sql_text_template": "Artikel: {name} (Status: {status})",
            }
        )

        files = scanner.scan(output_dir)
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert content == "Artikel: Widget (Status: aktiv)"


class TestIncrementalWithTimestamp:
    """Inkrementelle Abfrage mit Zeitstempel-Filter."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_incremental_where_appended(self, MockConnCls, tmp_path):
        # Mock gibt nur die Zeile zurueck die der Scanner nach Timestamp-Filter erhalten wuerde
        MockConnCls.return_value = _mock_connector(
            rows=[(2, "neu", "2024-06-01")],
            columns=["id", "msg", "updated_at"],
        )
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM logs",
                "sql_id_column": "id",
                "sql_timestamp_column": "updated_at",
            }
        )

        files = scanner.scan(output_dir, since_timestamp="2024-03-01")
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert "neu" in content

        # Pruefen, dass die Query den Timestamp-Filter enthaelt
        # postgresql-Treiber -> Platzhalter %s (nicht ?)
        call_args = MockConnCls.return_value.execute_query.call_args
        query_used = call_args[0][0]
        assert query_used == (
            "SELECT * FROM (SELECT * FROM logs) AS kiara_sql_scan "
            'WHERE kiara_sql_scan."updated_at" > %s'
        )
        assert call_args[0][1] == ("2024-03-01",)

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_incremental_with_existing_where(self, MockConnCls, tmp_path):
        """WHERE-Clause bleibt in der gekapselten Query erhalten."""
        MockConnCls.return_value = _mock_connector(
            rows=[(2, "neu A", "A", "2024-06-01")],
            columns=["id", "msg", "category", "updated_at"],
        )
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM logs WHERE category = 'A'",
                "sql_id_column": "id",
                "sql_timestamp_column": "updated_at",
            }
        )

        files = scanner.scan(output_dir, since_timestamp="2024-03-01")
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert "neu A" in content

        # Pruefen, dass AND (nicht WHERE) angehaengt wurde
        # postgresql-Treiber -> Platzhalter %s (nicht ?)
        call_args = MockConnCls.return_value.execute_query.call_args
        query_used = call_args[0][0]
        assert query_used == (
            "SELECT * FROM (SELECT * FROM logs WHERE category = 'A') "
            'AS kiara_sql_scan WHERE kiara_sql_scan."updated_at" > %s'
        )

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_incremental_wraps_order_by_and_strips_semicolon(self, MockConnCls, tmp_path):
        """ORDER BY/Semikolon bleiben in der Subquery gueltig."""
        MockConnCls.return_value = _mock_connector(
            rows=[(2, "neu", "2024-06-01")],
            columns=["id", "msg", "updated_at"],
        )
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM logs ORDER BY id DESC;",
                "sql_id_column": "id",
                "sql_timestamp_column": "updated_at",
            }
        )

        scanner.scan(output_dir, since_timestamp="2024-03-01")

        query_used = MockConnCls.return_value.execute_query.call_args[0][0]
        assert query_used == (
            "SELECT * FROM (SELECT * FROM logs ORDER BY id DESC) "
            'AS kiara_sql_scan WHERE kiara_sql_scan."updated_at" > %s'
        )


class TestRowFilenames:
    """SQL-Row-Dateinamen sind stabil und kollisionsarm."""

    def test_row_filename_id_uses_hash_suffix(self):
        a = SQLScanner._row_filename_id("a/b")
        b = SQLScanner._row_filename_id("a:b")

        assert a.startswith("a_b_")
        assert b.startswith("a_b_")
        assert a != b

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_keeps_colliding_normalized_ids(self, MockConnCls, tmp_path):
        MockConnCls.return_value = _mock_connector(
            rows=[("a/b", "slash"), ("a:b", "colon")],
            columns=["id", "msg"],
        )
        output_dir = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM logs",
                "sql_id_column": "id",
            }
        )

        files = scanner.scan(output_dir)

        assert len(files) == 2
        assert len({os.path.basename(path) for path in files}) == 2


class TestConnectionTest:
    """Verbindungstest delegiert an SQLConnector."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_test_connection(self, MockConnCls):
        MockConnCls.return_value = _mock_connector(
            rows=[],
            columns=[],
            tables=["demo", "other_table"],
        )
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM demo",
            }
        )
        result = scanner.test_connection()
        assert result["success"] is True
        assert "demo" in result["tables"]
