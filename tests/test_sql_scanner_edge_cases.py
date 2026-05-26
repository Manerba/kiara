"""Tests fuer SQL-Scanner Edge-Cases (Phase 6 Erweiterung)."""

import os
from unittest.mock import MagicMock, patch

from kiara.ingestion.sql_scanner import SQLScanner

# -- Hilfsfunktion: SQLConnector-Mock erzeugen --


def _mock_connector(rows, columns):
    """Erzeugt einen Mock fuer SQLConnector mit vordefinierten Ergebnissen."""
    mock = MagicMock()
    mock.execute_query.return_value = {"columns": columns, "rows": rows}
    return mock


# -- Vordefinierte Testdaten (entsprechen der ehemaligen sample_db) --

SAMPLE_COLUMNS = ["id", "name", "description", "count", "updated_at"]
SAMPLE_ROWS = [
    (1, "Widget A", "Beschreibung A", 100, "2024-01-15"),
    (2, "Widget B", "Beschreibung B", 200, "2024-02-01"),
    (3, "Widget C", "", 0, "2024-03-01"),
]


class TestScanAutoSelect:
    """Tabellen-Name statt SELECT -> automatisches Wrapping."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_auto_select(self, MockConnCls, tmp_path):
        """sql_query nur Tabellenname -> wird zu SELECT * FROM ..."""
        MockConnCls.return_value = _mock_connector(SAMPLE_ROWS, SAMPLE_COLUMNS)
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "protokolle",  # Kein SELECT
                "sql_id_column": "id",
            }
        )
        files = scanner.scan(output)
        assert len(files) == 3


class TestScanEmptyResult:
    """Query liefert 0 Zeilen -> leere Dateiliste."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_empty_result(self, MockConnCls, tmp_path):
        """Leere Tabelle -> keine Dateien, kein Fehler."""
        MockConnCls.return_value = _mock_connector(rows=[], columns=["id", "text"])
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM leer",
                "sql_id_column": "id",
            }
        )
        files = scanner.scan(output)
        assert files == []


class TestScanWithTextTemplate:
    """Template-basierte Text-Erzeugung."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_with_text_template(self, MockConnCls, tmp_path):
        """Template '{name} hat {count} Stueck' -> korrekte Substitution."""
        # Nur Zeile 1 zurueckgeben (wie WHERE id = 1)
        MockConnCls.return_value = _mock_connector(
            rows=[SAMPLE_ROWS[0]],
            columns=SAMPLE_COLUMNS,
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle WHERE id = 1",
                "sql_id_column": "id",
                "sql_text_template": "{name} hat {count} Stueck",
            }
        )
        files = scanner.scan(output)
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert content == "Widget A hat 100 Stueck"


class TestScanTemplateMissingKey:
    """Template mit unbekanntem Platzhalter -> leerer String."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_template_missing_key(self, MockConnCls, tmp_path):
        """Template '{missing_key}' -> wird zu leerem String."""
        MockConnCls.return_value = _mock_connector(
            rows=[SAMPLE_ROWS[0]],
            columns=SAMPLE_COLUMNS,
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle WHERE id = 1",
                "sql_id_column": "id",
                "sql_text_template": "Prefix: {missing_key} Suffix",
            }
        )
        files = scanner.scan(output)
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert content == "Prefix:  Suffix"


class TestScanRowWithoutIdColumn:
    """Kein passender id_column -> MD5-Hash als Dateiname."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_row_without_id_column(self, MockConnCls, tmp_path):
        """sql_id_column zeigt auf nicht existierende Spalte -> MD5-Hash."""
        MockConnCls.return_value = _mock_connector(
            rows=[SAMPLE_ROWS[0]],
            columns=SAMPLE_COLUMNS,
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle WHERE id = 1",
                "sql_id_column": "nonexistent_column",
            }
        )
        files = scanner.scan(output)
        assert len(files) == 1
        filename = os.path.basename(files[0])
        # Dateiname hat Format row_<md5hash>_<sha256suffix>.txt
        assert filename.startswith("row_")
        name_part = filename[4:-4]  # "row_" (4) und ".txt" (4) entfernen
        prefix, suffix = name_part.split("_")
        assert len(prefix) == 12
        assert len(suffix) == 16


class TestScanSkipsEmptyText:
    """Zeile mit leeren Text-Spalten -> keine Datei."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_skips_empty_text(self, MockConnCls, tmp_path):
        """Alle Text-Spalten leer -> Datei wird nicht geschrieben."""
        MockConnCls.return_value = _mock_connector(
            rows=[
                (1, "", ""),
                (2, None, None),
                (3, "Inhalt", "Beschreibung"),
            ],
            columns=["id", "text1", "text2"],
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM daten",
                "sql_id_column": "id",
                "sql_text_columns": "text1,text2",
            }
        )
        files = scanner.scan(output)
        # Nur Zeile 3 hat Inhalt
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert "Inhalt" in content


class TestScanWithSelectedColumns:
    """Nur ausgewaehlte Spalten im Text."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_with_selected_columns(self, MockConnCls, tmp_path):
        """sql_text_columns='name,description' -> nur diese im Text."""
        MockConnCls.return_value = _mock_connector(
            rows=[SAMPLE_ROWS[0]],
            columns=SAMPLE_COLUMNS,
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle WHERE id = 1",
                "sql_id_column": "id",
                "sql_text_columns": "name,description",
            }
        )
        files = scanner.scan(output)
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert "name: Widget A" in content
        assert "description: Beschreibung A" in content
        # count und updated_at sollen NICHT im Text sein
        assert "count:" not in content
        assert "updated_at:" not in content


class TestScanIncrementalAppendsWhere:
    """Inkrementeller Scan haengt WHERE-Clause an."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_incremental_appends_where(self, MockConnCls, tmp_path):
        """since_timestamp + timestamp_column -> WHERE ... > ? angehaengt."""
        # Nur Zeilen nach 2024-01-20 -> Zeile 2 und 3
        MockConnCls.return_value = _mock_connector(
            rows=[SAMPLE_ROWS[1], SAMPLE_ROWS[2]],
            columns=SAMPLE_COLUMNS,
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle",
                "sql_id_column": "id",
                "sql_timestamp_column": "updated_at",
            }
        )
        files = scanner.scan(output, since_timestamp="2024-01-20")
        assert len(files) >= 1
        # Pruefen dass Zeile 1 (2024-01-15) nicht enthalten ist
        all_content = ""
        for f in files:
            all_content += open(f, encoding="utf-8").read()
        assert "Widget A" not in all_content  # Zeile 1 war vor dem Datum


class TestScanIncrementalWithExistingWhere:
    """Inkrementeller Scan mit bestehender WHERE-Clause."""

    @patch("kiara.ingestion.sql_scanner.SQLConnector")
    def test_scan_incremental_with_existing_where(self, MockConnCls, tmp_path):
        """Query hat bereits WHERE -> AND timestamp > ? wird angehaengt."""
        # Nur Zeile 2: count=200, Datum 2024-02-01 -> drin
        MockConnCls.return_value = _mock_connector(
            rows=[SAMPLE_ROWS[1]],
            columns=SAMPLE_COLUMNS,
        )
        output = str(tmp_path / "output")
        scanner = SQLScanner(
            {
                "sql_dsn": "mock://db",
                "sql_driver": "postgresql",
                "sql_query": "SELECT * FROM protokolle WHERE count > 0",
                "sql_id_column": "id",
                "sql_timestamp_column": "updated_at",
            }
        )
        # Query hat WHERE count > 0 -> AND updated_at > '2024-01-20'
        files = scanner.scan(output, since_timestamp="2024-01-20")
        # Zeile 1: count=100 aber Datum zu alt -> raus
        # Zeile 2: count=200, Datum 2024-02-01 -> drin
        # Zeile 3: count=0 -> raus (count > 0 Filter)
        assert len(files) == 1
        content = open(files[0], encoding="utf-8").read()
        assert "Widget B" in content
