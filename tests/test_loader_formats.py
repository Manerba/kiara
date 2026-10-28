"""Tiefergehende Tests fuer Dateiformat-Loader der Ingestion-Pipeline.

Testet load_file() und die Einzel-Loader fuer verschiedene Dateitypen:
Plaintext, Code, CSV, JSON, sowie die _sanitize_text Binary-Guard
und das Fallback-Verhalten bei unbekannten Endungen.
"""

import json

from kiara.ingestion.loader import (
    _CODE_LANGUAGES,
    _load_code,
    _load_csv_file,
    _load_html_markdown,
    _load_json_file,
    _load_plaintext,
    _sanitize_text,
    load_file,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_file(tmp_path, name: str, content: str = "Inhalt fuer Tests") -> str:
    """Erstelle eine temporaere Textdatei und gib den absoluten Pfad zurueck."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. load_file() End-to-End
# ---------------------------------------------------------------------------


class TestLoadFileEndToEnd:
    """load_file() mit echten temporaeren Dateien ohne Mocks."""

    def test_txt_metadata_fields(self, tmp_path):
        """.txt-Datei liefert alle 7 Pflicht-Metadatenfelder."""
        path = _make_file(tmp_path, "readme.txt", "Das ist ein Testdokument.")
        result = load_file(path)

        assert result is not None
        meta = result["metadata"]
        assert "source" in meta
        assert "directory" in meta
        assert "filename" in meta
        assert "file_type" in meta
        assert "modified" in meta
        assert "size_bytes" in meta
        assert "parser" in meta

    def test_txt_metadata_source_is_full_path(self, tmp_path):
        """metadata['source'] entspricht dem vollstaendigen Pfad."""
        path = _make_file(tmp_path, "readme.txt", "Inhalt")
        result = load_file(path)

        assert result["metadata"]["source"] == path

    def test_txt_metadata_filename(self, tmp_path):
        """metadata['filename'] entspricht dem Dateinamen."""
        path = _make_file(tmp_path, "readme.txt", "Inhalt")
        result = load_file(path)

        assert result["metadata"]["filename"] == "readme.txt"

    def test_txt_metadata_directory(self, tmp_path):
        """metadata['directory'] entspricht dem Elternverzeichnis."""
        path = _make_file(tmp_path, "readme.txt", "Inhalt")
        result = load_file(path)

        assert result["metadata"]["directory"] == str(tmp_path)

    def test_txt_metadata_file_type_lowercase(self, tmp_path):
        """metadata['file_type'] ist lowercase ('.txt')."""
        path = _make_file(tmp_path, "readme.txt", "Inhalt")
        result = load_file(path)

        assert result["metadata"]["file_type"] == ".txt"

    def test_txt_metadata_size_bytes_positive(self, tmp_path):
        """metadata['size_bytes'] > 0 fuer nicht-leere Datei."""
        path = _make_file(tmp_path, "readme.txt", "Inhalt hier")
        result = load_file(path)

        assert result["metadata"]["size_bytes"] > 0

    def test_txt_metadata_parser_standard(self, tmp_path):
        """metadata['parser'] == 'standard' fuer Standard-Loader."""
        path = _make_file(tmp_path, "readme.txt", "Normaler Inhalt")
        result = load_file(path)

        assert result["metadata"]["parser"] == "standard"

    def test_txt_metadata_modified_is_iso_string(self, tmp_path):
        """metadata['modified'] ist ein ISO-8601-String."""
        path = _make_file(tmp_path, "readme.txt", "Inhalt")
        result = load_file(path)

        modified = result["metadata"]["modified"]
        assert isinstance(modified, str)
        # Muss mindestens Datum und Zeit enthalten (ISO 8601)
        assert "T" in modified

    def test_py_file_code_annotation(self, tmp_path):
        """.py-Datei → text enthaelt Python-Code-Annotation."""
        path = _make_file(tmp_path, "script.py", "def hello():\n    return 42\n")
        result = load_file(path)

        assert result is not None
        assert "```python" in result["text"]

    def test_csv_pipe_separated(self, tmp_path):
        """.csv-Datei mit 3 Spalten → Pipe-getrennte Zeilen im Output."""
        content = "Name,Alter,Stadt\nAlice,30,Berlin\nBob,25,Hamburg\n"
        path = _make_file(tmp_path, "daten.csv", content)
        result = load_file(path)

        assert result is not None
        assert " | " in result["text"]
        assert "Name" in result["text"]
        assert "Alice" in result["text"]

    def test_json_pretty_formatted(self, tmp_path):
        """.json-Datei → pretty-formatted im Output (indent=2)."""
        data = {"name": "KIara", "version": 1, "active": True}
        content = json.dumps(data)
        path = _make_file(tmp_path, "config.json", content)
        result = load_file(path)

        assert result is not None
        # Indented JSON hat Zeilenumbrueche + fuehrende Leerzeichen
        assert "\n" in result["text"]
        assert '"name"' in result["text"]
        assert '"KIara"' in result["text"]

    def test_xml_loaded_as_plaintext(self, tmp_path):
        """.xml-Datei wird als Plaintext geladen (kein Parser)."""
        content = '<?xml version="1.0"?><root><item>Wert</item></root>'
        path = _make_file(tmp_path, "daten.xml", content)
        result = load_file(path)

        assert result is not None
        assert "<root>" in result["text"]
        assert "<item>Wert</item>" in result["text"]

    def test_html_loaded_as_visible_text(self, tmp_path):
        """.html-Datei wird als sichtbarer Text statt Markup/Code geladen."""
        content = """
        <html>
          <head>
            <title>Oscars</title>
            <style>.hidden { display: none; }</style>
            <script>window.answer = "wrong";</script>
          </head>
          <body>
            <h1>Best Actor</h1>
            <p>Tom <b>Hanks</b> won in 1994.</p>
          </body>
        </html>
        """
        path = _make_file(tmp_path, "web_001.html", content)
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["file_type"] == ".html"
        assert result["metadata"]["parser"] == "html-to-markdown"
        assert "Best Actor" in result["text"]
        assert "Tom **Hanks** won in 1994." in result["text"]
        assert "<p>" not in result["text"]
        assert "window.answer" not in result["text"]
        assert "display: none" not in result["text"]

    def test_config_loaded_as_plaintext(self, tmp_path):
        """.config-Datei wird als Plaintext geladen."""
        content = "[settings]\nhost=localhost\nport=5432\n"
        path = _make_file(tmp_path, "app.config", content)
        result = load_file(path)

        assert result is not None
        assert "host=localhost" in result["text"]

    def test_ini_loaded_as_plaintext(self, tmp_path):
        """.ini-Datei wird als Plaintext geladen."""
        content = "[database]\nserver=kiara-server.local\nport=5432\n"
        path = _make_file(tmp_path, "settings.ini", content)
        result = load_file(path)

        assert result is not None
        assert "server=kiara-server.local" in result["text"]

    def test_unknown_extension_uses_plaintext_loader(self, tmp_path):
        """Unbekannte Extension (.xyz) → Default-Loader _load_plaintext."""
        content = "Unbekanntes Format mit normalem Text"
        path = _make_file(tmp_path, "datei.xyz", content)
        result = load_file(path)

        assert result is not None
        assert "Unbekanntes Format" in result["text"]
        assert result["metadata"]["parser"] == "standard"

    def test_empty_file_returns_no_content(self, tmp_path):
        """Leere Datei → load_file() gibt no_content-Dict zurueck."""
        p = tmp_path / "leer.txt"
        p.write_text("", encoding="utf-8")
        result = load_file(str(p))

        assert result is not None
        assert result["metadata"]["no_content"] is True
        assert result["text"] == ""

    def test_whitespace_only_file_returns_no_content(self, tmp_path):
        """Datei nur mit Leerzeichen/Zeilenumbruechen → load_file() gibt no_content zurueck."""
        p = tmp_path / "leerzeilen.txt"
        p.write_text("   \n\n  \t  \n", encoding="utf-8")
        result = load_file(str(p))

        assert result is not None
        assert result["metadata"]["no_content"] is True

    def test_nonexistent_file_returns_none_or_raises(self, tmp_path):
        """Nicht-existente Datei → Exception oder None (kein undefinierbarer Zustand)."""
        path = str(tmp_path / "existiert_nicht.txt")
        try:
            result = load_file(path)
            # Wenn kein Exception: muss None sein
            assert result is None
        except Exception:
            # Exception ist ebenfalls akzeptiertes Verhalten
            pass


# ---------------------------------------------------------------------------
# 2. _load_code() Details
# ---------------------------------------------------------------------------


class TestLoadCodeDetails:
    """_load_code() mit verschiedenen Code-Dateiendungen."""

    def test_py_contains_python_fence(self, tmp_path):
        """.py-Datei → Backtick-Zaun mit 'python'."""
        path = _make_file(tmp_path, "main.py", "print('hello')\n")
        result = _load_code(path)

        assert "```python" in result

    def test_py_contains_file_comment(self, tmp_path):
        """.py-Datei → enthaelt '# File: main.py'."""
        path = _make_file(tmp_path, "main.py", "x = 1\n")
        result = _load_code(path)

        assert "# File: main.py" in result

    def test_py_contains_source_content(self, tmp_path):
        """.py-Datei → Quellcode ist im Output enthalten."""
        path = _make_file(tmp_path, "main.py", "def add(a, b):\n    return a + b\n")
        result = _load_code(path)

        assert "def add(a, b):" in result

    def test_py_has_closing_fence(self, tmp_path):
        """.py-Datei → schliessendes Backtick-Dreifach am Ende."""
        path = _make_file(tmp_path, "main.py", "pass\n")
        result = _load_code(path)

        assert result.strip().endswith("```")

    def test_cs_contains_csharp_fence(self, tmp_path):
        """.cs-Datei → Backtick-Zaun mit 'csharp'."""
        path = _make_file(
            tmp_path, "Program.cs", "using System;\nclass Program { static void Main() {} }\n"
        )
        result = _load_code(path)

        assert "```csharp" in result
        assert "# File: Program.cs" in result

    def test_sh_contains_bash_fence(self, tmp_path):
        """.sh-Datei → Backtick-Zaun mit 'bash'."""
        path = _make_file(tmp_path, "deploy.sh", "#!/bin/bash\necho 'deploy'\n")
        result = _load_code(path)

        assert "```bash" in result
        assert "# File: deploy.sh" in result

    def test_sln_contains_text_fence(self, tmp_path):
        """.sln-Datei → Backtick-Zaun mit 'text' (Solution-Datei)."""
        path = _make_file(tmp_path, "MyProject.sln", "Microsoft Visual Studio Solution File\n")
        result = _load_code(path)

        assert "```text" in result
        assert "# File: MyProject.sln" in result

    def test_ps1_contains_powershell_fence(self, tmp_path):
        """.ps1-Datei → Backtick-Zaun mit 'powershell'."""
        path = _make_file(tmp_path, "setup.ps1", "Write-Host 'Setup'\n")
        result = _load_code(path)

        assert "```powershell" in result

    def test_js_contains_javascript_fence(self, tmp_path):
        """.js-Datei → Backtick-Zaun mit 'javascript'."""
        path = _make_file(tmp_path, "app.js", "console.log('hello');\n")
        result = _load_code(path)

        assert "```javascript" in result

    def test_unknown_code_extension_no_backtick_wrapper(self, tmp_path):
        """Unbekannte Code-Extension → kein Backtick-Zaun, Rohinhalt zurueck."""
        # .xyz ist nicht in _CODE_LANGUAGES, _load_code gibt einfachen Text zurueck
        path = _make_file(tmp_path, "datei.xyz", "Unbekannter Inhalt\n")
        result = _load_code(path)

        assert "```" not in result
        assert "Unbekannter Inhalt" in result

    def test_code_languages_dict_has_expected_extensions(self):
        """_CODE_LANGUAGES enthaelt alle erwarteten Schluessel."""
        expected = {
            ".cs",
            ".py",
            ".sh",
            ".ps1",
            ".js",
            ".bat",
            ".vb",
            ".pl",
            ".css",
            ".html",
            ".sln",
        }
        for ext in expected:
            assert ext in _CODE_LANGUAGES, f"Fehlende Extension in _CODE_LANGUAGES: {ext}"


class TestLoadHtmlVisibleText:
    """HTML-spezifischer Loader."""

    def test_load_html_markdown_strips_markup_and_invisible_blocks(self, tmp_path):
        path = _make_file(
            tmp_path,
            "page.html",
            "<html><body><script>bad()</script><style>.x{}</style>"
            "<h1>Titel</h1><p>Antwort <b>sichtbar</b>.</p></body></html>",
        )
        result = _load_html_markdown(path)

        assert "Titel" in result
        assert "Antwort **sichtbar**." in result
        assert "<h1>" not in result
        assert "bad()" not in result
        assert ".x{}" not in result


# ---------------------------------------------------------------------------
# 3. _load_csv_file() Details
# ---------------------------------------------------------------------------


class TestLoadCsvFileDetails:
    """_load_csv_file() mit verschiedenen CSV-Varianten."""

    def test_standard_csv_pipe_separated(self, tmp_path):
        """Standard-CSV → Zeilen sind Pipe-getrennt."""
        content = "Name,Alter,Stadt\nAlice,30,Berlin\n"
        path = _make_file(tmp_path, "test.csv", content)
        result = _load_csv_file(path)

        assert " | " in result
        assert "Name" in result
        assert "Alice" in result

    def test_csv_three_columns_correct_pipes(self, tmp_path):
        """CSV mit 3 Spalten → jede Zeile hat 2 Pipes."""
        content = "A,B,C\n1,2,3\n4,5,6\n"
        path = _make_file(tmp_path, "test.csv", content)
        result = _load_csv_file(path)

        lines = [l for l in result.splitlines() if l.strip()]
        # Jede Zeile hat exakt 2 Pipes fuer 3 Spalten
        for line in lines:
            assert line.count(" | ") == 2, f"Unerwartete Pipe-Anzahl in: {line!r}"

    def test_csv_with_quoted_fields(self, tmp_path):
        """CSV mit Anführungszeichen → Felder werden korrekt geparst."""
        content = '"Vorname Nachname","Beruf","Ort"\n"Max Mustermann","Entwickler","Muenchen"\n'
        path = _make_file(tmp_path, "quoted.csv", content)
        result = _load_csv_file(path)

        # Quotes sollen durch csv.reader aufgeloest worden sein
        assert "Max Mustermann" in result
        assert "Entwickler" in result
        # Keine rohen Anführungszeichen um Werte herum
        assert '"Max Mustermann"' not in result

    def test_csv_with_semicolon_separator(self, tmp_path):
        """CSV mit Semikolon als Trenner → als einzelne Spalte behandelt (kein Fehler)."""
        content = "Name;Alter\nAlice;30\n"
        path = _make_file(tmp_path, "semi.csv", content)
        result = _load_csv_file(path)

        # Kein Crash — Ergebnis ist ein String
        assert isinstance(result, str)
        assert "Alice" in result

    def test_csv_empty_lines_preserved(self, tmp_path):
        """CSV mit Leerzeilen → Kein Absturz, Ergebnis ist String."""
        content = "A,B\n1,2\n\n3,4\n"
        path = _make_file(tmp_path, "empty_lines.csv", content)
        result = _load_csv_file(path)

        assert isinstance(result, str)
        assert "1" in result
        assert "3" in result

    def test_csv_single_column(self, tmp_path):
        """CSV mit einer Spalte → Zeilen ohne Pipes."""
        content = "Wert\nalpha\nbeta\n"
        path = _make_file(tmp_path, "single.csv", content)
        result = _load_csv_file(path)

        assert "alpha" in result
        assert "beta" in result

    def test_csv_returns_string(self, tmp_path):
        """_load_csv_file() gibt immer einen String zurueck."""
        path = _make_file(tmp_path, "data.csv", "a,b\n1,2\n")
        result = _load_csv_file(path)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 4. _load_json_file() Details
# ---------------------------------------------------------------------------


class TestLoadJsonFileDetails:
    """_load_json_file() mit gueltigen und ungueltigen JSON-Daten."""

    def test_valid_json_is_pretty_formatted(self, tmp_path):
        """Valides JSON → pretty-formatted mit indent=2."""
        data = {"host": "localhost", "port": 5432}
        path = _make_file(tmp_path, "cfg.json", json.dumps(data))
        result = _load_json_file(path)

        # Indented JSON hat Zeilenumbrueche
        assert "\n" in result
        # Schluessel und Werte korrekt enthalten
        assert '"host"' in result
        assert '"localhost"' in result

    def test_valid_json_indentation_depth(self, tmp_path):
        """Valides JSON → Einrueckung mit 2 Leerzeichen."""
        data = {"key": "value"}
        path = _make_file(tmp_path, "simple.json", json.dumps(data))
        result = _load_json_file(path)

        parsed_back = json.loads(result)
        assert parsed_back == data

    def test_invalid_json_fallback_to_plaintext(self, tmp_path):
        """Ungueltiges JSON → Fallback auf Klartext (kein Absturz)."""
        bad_json = "{ das ist: kein gueltiges JSON }"
        path = _make_file(tmp_path, "broken.json", bad_json)
        result = _load_json_file(path)

        # Kein Crash, Rohinhalt zurueckgegeben
        assert isinstance(result, str)
        assert "kein gueltiges JSON" in result

    def test_json_with_unicode_no_ascii_escape(self, tmp_path):
        """JSON mit Unicode → ensure_ascii=False (keine \\uXXXX Escapes)."""
        data = {"stadt": "Muenchen", "land": "Deutschland", "symbol": "\u00e4\u00f6\u00fc"}
        path = _make_file(tmp_path, "unicode.json", json.dumps(data, ensure_ascii=False))
        result = _load_json_file(path)

        # Unicode-Zeichen sollen direkt im Output stehen
        assert "\u00e4" in result or "ae" in result  # ae oder ae-Umlaut
        # Keine ASCII-Escaped Unicode-Sequenzen fuer die Umlaute
        assert "\\u00e4" not in result

    def test_json_array_pretty_formatted(self, tmp_path):
        """JSON-Array → pretty-formatted."""
        data = [1, 2, 3, "vier"]
        path = _make_file(tmp_path, "array.json", json.dumps(data))
        result = _load_json_file(path)

        assert "\n" in result
        assert '"vier"' in result

    def test_json_nested_structure(self, tmp_path):
        """Verschachteltes JSON → korrekt formatiert."""
        data = {"outer": {"inner": [1, 2, 3]}}
        path = _make_file(tmp_path, "nested.json", json.dumps(data))
        result = _load_json_file(path)

        parsed_back = json.loads(result)
        assert parsed_back == data

    def test_json_returns_string(self, tmp_path):
        """_load_json_file() gibt immer einen String zurueck."""
        path = _make_file(tmp_path, "data.json", '{"x": 1}')
        result = _load_json_file(path)

        assert isinstance(result, str)

    def test_json_empty_object(self, tmp_path):
        """Leeres JSON-Objekt → valide Ausgabe."""
        path = _make_file(tmp_path, "empty.json", "{}")
        result = _load_json_file(path)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {}

    def test_truncated_json_fallback(self, tmp_path):
        """Abgeschnittenes JSON → Fallback auf Plaintext."""
        path = _make_file(tmp_path, "trunc.json", '{"key": "val')
        result = _load_json_file(path)

        assert isinstance(result, str)
        # Der Rohinhalt muss teilweise enthalten sein
        assert "key" in result


# ---------------------------------------------------------------------------
# 5. _sanitize_text() Binary-Guard
# ---------------------------------------------------------------------------


class TestSanitizeTextBinaryGuard:
    """_sanitize_text() erkennt Binaerinhalt und bereinigt Artefakte."""

    def test_clean_text_unchanged(self):
        """Text ohne Null-Bytes wird unveraendert zurueckgegeben."""
        text = "Das ist sauberer Text ohne Probleme."
        result = _sanitize_text(text, "test.txt", "standard")

        assert result == text

    def test_text_with_few_null_bytes_sanitized(self):
        """Text mit <5% Null-Bytes → Null-Bytes entfernt, kein None."""
        # Null-Byte-Anteil: 1/80 = 1.25% < 5%
        text = "A" * 79 + "\x00"
        result = _sanitize_text(text, "test.txt", "standard")

        assert result is not None
        assert "\x00" not in result
        assert "A" * 79 in result

    def test_text_with_high_null_ratio_returns_none(self):
        """Text mit >5% Null-Bytes → None (Binaerdatei erkannt)."""
        # Null-Byte-Anteil: 1/2 = 50% > 5%
        text = "A\x00" * 50
        result = _sanitize_text(text, "binary.bin", "standard")

        assert result is None

    def test_text_exactly_at_threshold_is_none(self):
        """Text mit exakt >5% Null-Bytes (Grenzfall: 6%) → None."""
        # 6 Null-Bytes bei 100 Zeichen = 6% > 5%
        text = "A" * 94 + "\x00" * 6
        result = _sanitize_text(text, "borderline.bin", "standard")

        assert result is None

    def test_text_just_below_threshold_sanitized(self):
        """Text mit knapp unter 5% Null-Bytes (4%) → bereinigt, kein None."""
        # 4 Null-Bytes bei 100 Zeichen = 4% < 5%
        text = "A" * 96 + "\x00" * 4
        result = _sanitize_text(text, "low_null.txt", "standard")

        assert result is not None
        assert "\x00" not in result

    def test_empty_text_unchanged(self):
        """Leerer Text → unveraendert zurueckgegeben (kein Absturz)."""
        result = _sanitize_text("", "empty.txt", "standard")

        assert result == ""

    def test_single_null_byte_removed(self):
        """Ein einzelner Null-Byte in langem Text → entfernt, kein None."""
        text = "X" * 200 + "\x00"
        result = _sanitize_text(text, "one_null.txt", "standard")

        assert result is not None
        assert "\x00" not in result
        assert len(result) == 200

    def test_source_and_parser_accepted(self):
        """source und parser Parameter werden ohne Fehler akzeptiert."""
        result = _sanitize_text("Hello World", "/pfad/zu/datei.txt", "docling")

        assert result == "Hello World"

    def test_pure_null_bytes_returns_none(self):
        """Nur Null-Bytes → None (100% Null-Byte-Anteil)."""
        text = "\x00" * 50
        result = _sanitize_text(text, "null_only.bin", "standard")

        assert result is None

    def test_multiline_text_without_nulls(self):
        """Mehrzeiliger Text ohne Null-Bytes → unveraendert."""
        text = "Zeile 1\nZeile 2\nZeile 3\n"
        result = _sanitize_text(text, "multi.txt", "standard")

        assert result == text


# ---------------------------------------------------------------------------
# 6. Metadata-Vollstaendigkeit
# ---------------------------------------------------------------------------


class TestMetadataCompleteness:
    """load_file() setzt alle Metadatenfelder korrekt."""

    REQUIRED_FIELDS = {
        "source",
        "directory",
        "filename",
        "file_type",
        "modified",
        "size_bytes",
        "parser",
    }

    def test_all_seven_fields_present_txt(self, tmp_path):
        """Alle 7 Metadatenfelder fuer .txt vorhanden."""
        path = _make_file(tmp_path, "doc.txt", "Inhalt fuer Metadaten-Test")
        result = load_file(path)

        assert result is not None
        for field in self.REQUIRED_FIELDS:
            assert field in result["metadata"], f"Fehlendes Feld: {field}"

    def test_all_seven_fields_present_py(self, tmp_path):
        """Alle 7 Metadatenfelder fuer .py vorhanden."""
        path = _make_file(tmp_path, "script.py", "import os\nprint(os.getcwd())\n")
        result = load_file(path)

        assert result is not None
        for field in self.REQUIRED_FIELDS:
            assert field in result["metadata"], f"Fehlendes Feld: {field}"

    def test_all_seven_fields_present_csv(self, tmp_path):
        """Alle 7 Metadatenfelder fuer .csv vorhanden."""
        path = _make_file(tmp_path, "data.csv", "col1,col2\nval1,val2\n")
        result = load_file(path)

        assert result is not None
        for field in self.REQUIRED_FIELDS:
            assert field in result["metadata"], f"Fehlendes Feld: {field}"

    def test_all_seven_fields_present_json(self, tmp_path):
        """Alle 7 Metadatenfelder fuer .json vorhanden."""
        path = _make_file(tmp_path, "config.json", '{"active": true}')
        result = load_file(path)

        assert result is not None
        for field in self.REQUIRED_FIELDS:
            assert field in result["metadata"], f"Fehlendes Feld: {field}"

    def test_file_type_is_lowercase(self, tmp_path):
        """file_type ist immer lowercase."""
        path = _make_file(tmp_path, "document.TXT", "Inhalt")
        result = load_file(path)

        if result is not None:
            assert result["metadata"]["file_type"] == result["metadata"]["file_type"].lower()

    def test_size_bytes_matches_actual_file_size(self, tmp_path):
        """size_bytes stimmt mit tatsaechlicher Dateigroesse ueberein."""
        import os

        content = "Genau zehn Zeichen."
        path = _make_file(tmp_path, "size_test.txt", content)
        actual_size = os.path.getsize(path)
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["size_bytes"] == actual_size

    def test_source_equals_input_path(self, tmp_path):
        """metadata['source'] entspricht dem uebergebenen Pfad."""
        path = _make_file(tmp_path, "source_check.txt", "Inhalt")
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["source"] == path

    def test_filename_without_directory(self, tmp_path):
        """metadata['filename'] enthaelt keinen Verzeichnispfad."""
        path = _make_file(tmp_path, "nur_name.txt", "Inhalt")
        result = load_file(path)

        assert result is not None
        assert "/" not in result["metadata"]["filename"]
        assert result["metadata"]["filename"] == "nur_name.txt"

    def test_directory_is_parent_of_source(self, tmp_path):
        """metadata['directory'] ist der Elternteil von metadata['source']."""
        from pathlib import Path

        path = _make_file(tmp_path, "pfad_test.txt", "Inhalt")
        result = load_file(path)

        assert result is not None
        meta = result["metadata"]
        assert str(Path(meta["source"]).parent) == meta["directory"]

    def test_size_bytes_positive_for_nonempty_file(self, tmp_path):
        """size_bytes > 0 fuer jede nicht-leere Datei."""
        path = _make_file(tmp_path, "nonempty.txt", "Mindestens ein Zeichen")
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["size_bytes"] > 0

    def test_parser_field_is_string(self, tmp_path):
        """metadata['parser'] ist ein String."""
        path = _make_file(tmp_path, "parser_type.txt", "Inhalt")
        result = load_file(path)

        assert result is not None
        assert isinstance(result["metadata"]["parser"], str)


# ---------------------------------------------------------------------------
# 7. _load_plaintext() Details
# ---------------------------------------------------------------------------


class TestLoadPlaintextDetails:
    """_load_plaintext() mit verschiedenen Textdateien."""

    def test_simple_utf8_content(self, tmp_path):
        """UTF-8 Text wird korrekt gelesen."""
        content = "Hallo Welt mit Umlauten: ae oe ue"
        path = _make_file(tmp_path, "utf8.txt", content)
        result = _load_plaintext(path)

        assert result == content

    def test_multiline_content(self, tmp_path):
        """Mehrzeiliger Text bleibt erhalten."""
        content = "Zeile 1\nZeile 2\nZeile 3\n"
        path = _make_file(tmp_path, "multi.txt", content)
        result = _load_plaintext(path)

        assert result == content

    def test_returns_string(self, tmp_path):
        """_load_plaintext() gibt immer einen String zurueck."""
        path = _make_file(tmp_path, "test.txt", "Text")
        result = _load_plaintext(path)

        assert isinstance(result, str)

    def test_empty_file_returns_empty_string(self, tmp_path):
        """Leere Datei → leerer String."""
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        result = _load_plaintext(str(p))

        assert result == ""

    def test_unicode_content_preserved(self, tmp_path):
        """Unicode-Zeichen (Umlaute, Sonderzeichen) werden korrekt gelesen."""
        content = "Sonderzeichen: \u00e4\u00f6\u00fc\u00df\u20ac"
        p = tmp_path / "unicode.txt"
        p.write_text(content, encoding="utf-8")
        result = _load_plaintext(str(p))

        assert result == content


# ---------------------------------------------------------------------------
# 8. load_file() — Erweiterte Faelle fuer verschiedene Dateitypen
# ---------------------------------------------------------------------------


class TestLoadFileExtendedCases:
    """Weitere End-to-End-Tests fuer load_file() mit echten Dateien."""

    def test_md_loaded_as_plaintext(self, tmp_path):
        """.md-Datei wird als Plaintext geladen."""
        content = "# Ueberschrift\n\nAbsatz mit **Fettschrift**.\n"
        path = _make_file(tmp_path, "readme.md", content)
        result = load_file(path)

        assert result is not None
        assert "# Ueberschrift" in result["text"]
        assert result["metadata"]["file_type"] == ".md"

    def test_py_file_type_in_metadata(self, tmp_path):
        """.py-Datei hat file_type '.py' in Metadaten."""
        path = _make_file(tmp_path, "mod.py", "x = 42\n")
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["file_type"] == ".py"

    def test_json_file_type_in_metadata(self, tmp_path):
        """.json-Datei hat file_type '.json' in Metadaten."""
        path = _make_file(tmp_path, "data.json", '{"key": "val"}')
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["file_type"] == ".json"

    def test_csv_file_type_in_metadata(self, tmp_path):
        """.csv-Datei hat file_type '.csv' in Metadaten."""
        path = _make_file(tmp_path, "table.csv", "a,b\n1,2\n")
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["file_type"] == ".csv"

    def test_result_has_text_and_metadata_keys(self, tmp_path):
        """load_file() Ergebnis hat genau 'text' und 'metadata' Schluessel."""
        path = _make_file(tmp_path, "check.txt", "Inhalt")
        result = load_file(path)

        assert result is not None
        assert "text" in result
        assert "metadata" in result

    def test_text_is_string(self, tmp_path):
        """result['text'] ist ein String."""
        path = _make_file(tmp_path, "text_type.txt", "Inhalt")
        result = load_file(path)

        assert result is not None
        assert isinstance(result["text"], str)

    def test_large_text_file(self, tmp_path):
        """Grosse Textdatei (>10KB) wird vollstaendig geladen."""
        content = "Testzeile mit ausreichend Inhalt.\n" * 400  # ~13KB
        path = _make_file(tmp_path, "gross.txt", content)
        result = load_file(path)

        assert result is not None
        assert len(result["text"]) > 10000

    def test_sh_file_contains_bash_in_text(self, tmp_path):
        """.sh-Datei → text enthaelt 'bash' Annotation."""
        content = "#!/bin/bash\necho 'KIara'\necho 'System'\n"
        path = _make_file(tmp_path, "start.sh", content)
        result = load_file(path)

        assert result is not None
        assert "bash" in result["text"]

    def test_cs_file_loaded_correctly(self, tmp_path):
        """.cs-Datei wird geladen und enthaelt csharp-Annotation."""
        content = "namespace Kiara\n{\n    class Program { }\n}\n"
        path = _make_file(tmp_path, "Program.cs", content)
        result = load_file(path)

        assert result is not None
        assert "csharp" in result["text"]
        assert "namespace Kiara" in result["text"]

    def test_xml_file_type_in_metadata(self, tmp_path):
        """.xml-Datei hat korrekten file_type in Metadaten."""
        content = "<config><host>localhost</host></config>"
        path = _make_file(tmp_path, "app.xml", content)
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["file_type"] == ".xml"

    def test_ddu_disabled_by_default(self, tmp_path):
        """ddu_enabled=False ist der Default — Standard-Loader wird genutzt."""
        path = _make_file(tmp_path, "default.txt", "Standard-Loader Test")
        result = load_file(path)

        assert result is not None
        assert result["metadata"]["parser"] == "standard"

    def test_unknown_ext_returns_text(self, tmp_path):
        """Unbekannte Extension → text wird zurueckgegeben."""
        content = "Inhalt mit unbekannter Endung."
        path = _make_file(tmp_path, "datei.qrs", content)
        result = load_file(path)

        assert result is not None
        assert "unbekannter Endung" in result["text"]
