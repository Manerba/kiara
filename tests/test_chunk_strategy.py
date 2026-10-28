"""Tests fuer die automatische Chunk-Groessen-Erkennung (Phase 7)."""

from unittest.mock import patch

from kiara.ingestion import chunker as chunker_mod
from kiara.ingestion.chunk_strategy import _IMAGE_EXTENSIONS, STRATEGIES, detect_chunk_params

# ── Stufe 1: Dateityp-Heuristik ──────────────────────────────────────────────


def test_csv_detected_as_tabular():
    """CSV-Dateien werden als tabellarisch erkannt."""
    meta = {"file_type": ".csv", "source": "/data/test.csv"}
    size, overlap = detect_chunk_params("col1,col2\nval1,val2", meta)
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


def test_xlsx_detected_as_tabular():
    """Excel-Dateien werden als tabellarisch erkannt."""
    meta = {"file_type": ".xlsx", "source": "/data/test.xlsx"}
    size, overlap = detect_chunk_params("Spalte A\tSpalte B\n1\t2", meta)
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


def test_python_detected_as_code():
    """Python-Dateien werden als Code erkannt."""
    meta = {"file_type": ".py", "source": "/src/main.py"}
    text = "def hello():\n    print('world')\n\nclass Foo:\n    pass"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["code"]
    assert meta["chunk_strategy"] == "code"


def test_html_visible_text_detected_as_prose():
    """HTML nach sichtbarer Text-Extraktion wird als Fliesstext gechunked."""
    meta = {
        "file_type": ".html",
        "parser": "html-to-markdown",
        "content_format": "markdown",
        "source_mime_type": "text/html",
        "source": "/web/page.html",
    }
    text = (
        "This article describes the 1994 Academy Award for Best Actor and gives "
        "the relevant background for the ceremony.\n" * 10
    )
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_html_visible_text_short_lines_still_prose():
    """HTML-visible-text darf durch Listen/Tabellenreste nicht zu tabular kippen."""
    meta = {
        "file_type": ".html",
        "parser": "html-to-markdown",
        "content_format": "markdown",
        "source_mime_type": "text/html",
        "source": "/web/awards.html",
    }
    text = "\n".join(
        [
            "Filmsite's Greatest Films",
            "of 1994",
            "",
            "Best Picture",
            "",
            "FORREST GUMP",
            "",
            "Actor:",
            "TOM HANKS in Forrest Gump",
            "Morgan Freeman in The Shawshank Redemption",
            "John Travolta in Pulp Fiction",
        ]
    )
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_raw_html_still_detected_as_code():
    """HTML ohne sichtbaren Text-Parser bleibt Code/Markup."""
    meta = {"file_type": ".html", "parser": "standard", "source": "/web/page.html"}
    text = "<html><body><p>Text</p></body></html>"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["code"]
    assert meta["chunk_strategy"] == "code"


def test_docx_detected_as_prose():
    """DOCX-Dateien werden als Fliesstext erkannt (bei langem Text ohne Tabellen)."""
    meta = {"file_type": ".docx", "source": "/docs/bericht.docx"}
    text = (
        "Dies ist ein langer Absatz in einem Dokument, der beschreibt wie das System funktioniert. "
        "Er enthaelt keine Tabellen oder Key-Value-Paare, sondern durchgehenden Fliesstext.\n" * 10
    )
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_pdf_detected_as_prose():
    """PDF-Dateien werden als Fliesstext erkannt."""
    meta = {"file_type": ".pdf", "source": "/docs/handbuch.pdf"}
    text = (
        "Dieses Dokument beschreibt die wesentlichen Aspekte der Implementierung und Konfiguration "
        "des Systems im Detail. Es richtet sich an Administratoren und Entwickler gleichermassen.\n"
        * 10
    )
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_imap_cache_detected_as_email():
    """TXT-Dateien im mail-cache werden als E-Mail erkannt."""
    meta = {"file_type": ".txt", "source": "/var/lib/kiara/mail-cache/ds5/inbox/msg001.txt"}
    text = "From: user@firma.de\nTo: admin@firma.de\nSubject: Test\n\nHallo, das ist eine Testnachricht."
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["email"]
    assert meta["chunk_strategy"] == "email"


def test_sql_cache_detected_as_tabular():
    """TXT-Dateien im sql-cache werden als tabellarisch erkannt."""
    meta = {"file_type": ".txt", "source": "/var/lib/kiara/sql-cache/ds7/query_001.txt"}
    text = "id: 1\ntitel: Testticket\nbeschreibung: Drucker defekt"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


# ── Stufe 2: Textanalyse-Korrektur ──────────────────────────────────────────


def test_docx_with_tables_becomes_tabular():
    """DOCX mit >40% Pipe-Zeilen wird als tabellarisch erkannt."""
    meta = {"file_type": ".docx", "source": "/docs/tabelle.docx"}
    lines = []
    for i in range(6):
        lines.append(f"| Spalte A | Spalte B | Wert {i} |")
    for i in range(4):
        lines.append(f"Normaler Text Zeile {i}")
    text = "\n".join(lines)
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


def test_docx_with_key_value_becomes_structured():
    """DOCX mit >50% Key-Value-Zeilen wird als strukturiert erkannt."""
    meta = {"file_type": ".docx", "source": "/docs/formular.docx"}
    lines = []
    for i in range(6):
        lines.append(
            f"Feldname {i}: Wert {i} mit etwas mehr Text hier drin damit die Zeilen lang genug sind"
        )
    for i in range(4):
        lines.append(
            f"Normaler Fliesstext in Zeile {i} der lang genug ist damit avg_line_len ueber 50 bleibt fuer den Test."
        )
    text = "\n".join(lines)
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["structured"]
    assert meta["chunk_strategy"] == "structured"


def test_short_lines_detected_as_structured():
    """Kurze Zeilen (avg <50) mit KV-Ratio >30% werden als strukturiert erkannt."""
    meta = {"file_type": ".txt", "source": "/data/config.txt"}
    lines = [
        "Name: Max Muster",
        "Alter: 42",
        "Stadt: Berlin",
        "Rolle: Admin",
        "Status: aktiv",
        "Abteilung: IT",
        "Leer",
        "Auch leer",
    ]
    text = "\n".join(lines)
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["structured"]
    assert meta["chunk_strategy"] == "structured"


def test_code_not_overridden_by_text_analysis():
    """Code bleibt Code, auch wenn Text kurze Zeilen hat."""
    meta = {"file_type": ".py", "source": "/src/script.py"}
    text = "x = 1\ny = 2\nz = 3\na = 4\nb = 5\nc = 6\nd = 7\ne = 8"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["code"]
    assert meta["chunk_strategy"] == "code"


def test_pure_prose_stays_prose():
    """DOCX mit langen Absaetzen ohne Pipes/KVs bleibt Fliesstext."""
    meta = {"file_type": ".docx", "source": "/docs/aufsatz.docx"}
    text = "\n".join(
        [
            "Dies ist ein langer Absatz ohne jegliche Tabellenstrukturen oder Key-Value-Paare in diesem Text.",
            "Ein weiterer Absatz der rein narrativ geschrieben ist und viele Woerter enthaelt ohne Pipes oder Doppelpunkte am Anfang.",
            "Der dritte Absatz beschreibt weitere Details zum Thema und ist ebenfalls durchgehender Fliesstext ohne besondere Struktur.",
            "Noch ein Absatz mit genuegend Text um die durchschnittliche Zeilenlaenge ueber 50 zu halten.",
            "Und ein fuenfter Absatz der ebenfalls lang genug ist um als normaler Fliesstext durchzugehen.",
        ]
    )
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


# ── Edge Cases ───────────────────────────────────────────────────────────────


def test_empty_text_returns_default():
    """Leerer Text liefert Default-Strategie."""
    meta = {"file_type": ".docx", "source": "/docs/leer.docx"}
    size, overlap = detect_chunk_params("", meta)
    assert (size, overlap) == STRATEGIES["default"]
    assert meta["chunk_strategy"] == "default"


def test_single_line_text():
    """Einzelne Zeile verursacht keine Division-by-Zero."""
    meta = {"file_type": ".txt", "source": "/data/einzeiler.txt"}
    size, overlap = detect_chunk_params(
        "Nur eine einzelne Zeile mit genug Text um nicht als kurz zu gelten in der Analyse.", meta
    )
    # Sollte nicht crashen
    assert size > 0
    assert overlap >= 0
    assert "chunk_strategy" in meta


def test_unknown_extension_uses_text_analysis():
    """Unbekannte Extension (.xyz) — Textanalyse entscheidet."""
    meta = {"file_type": ".xyz", "source": "/data/datei.xyz"}
    # Text mit vielen Pipe-Zeilen → tabular
    text = "\n".join([f"| Col1 | Col2 | Val{i} |" for i in range(10)])
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


# ── Integration ──────────────────────────────────────────────────────────────


def test_strategy_in_metadata():
    """Nach detect_chunk_params() ist chunk_strategy in den Metadaten gesetzt."""
    meta = {"file_type": ".pdf", "source": "/docs/test.pdf"}
    text = (
        "Ein normaler Text der lang genug ist fuer einen sinnvollen Test der Strategie-Erkennung.\n"
        * 5
    )
    detect_chunk_params(text, meta)
    assert "chunk_strategy" in meta
    assert meta["chunk_strategy"] in STRATEGIES


def test_strategies_dict_completeness():
    """Alle definierten Strategien haben gueltige Tupel."""
    for name, (size, overlap) in STRATEGIES.items():
        assert isinstance(size, int) and size > 0, f"Ungueltige chunk_size fuer {name}"
        assert isinstance(overlap, int) and overlap >= 0, f"Ungueltige chunk_overlap fuer {name}"
        assert overlap < size, f"Overlap >= size fuer {name}"


# ── Erweiterte Tests (Phase 7 Prompt) ────────────────────────────────────────


def test_txt_without_cache_prefix_uses_text_analysis():
    """.txt-Datei ohne mail-cache/sql-cache Prefix → Textanalyse greift."""
    meta = {"file_type": ".txt", "source": "/data/config.txt"}
    # Text mit ueberwiegend Key-Value-Zeilen → structured via Textanalyse
    text = "\n".join([f"Param{i}: Wert{i}" for i in range(10)])
    size, overlap = detect_chunk_params(text, meta)
    # .txt ist in _PROSE_EXTENSIONS → Basis "prose"
    # Textanalyse: alle 10 Zeilen sind KV → kv_ratio=1.0 > 0.5 → "structured"
    assert meta["chunk_strategy"] == "structured"
    assert (size, overlap) == STRATEGIES["structured"]


def test_json_always_tabular_regardless_of_content():
    """.json-Datei ist immer tabellarisch (Extension in _TABULAR_EXTENSIONS)."""
    meta = {"file_type": ".json", "source": "/data/export.json"}
    # Auch bei Fliesstext-aehnlichem Inhalt bleibt .json tabellarisch
    text = "Dies ist ein langer Fliesstext der in einer JSON-Datei steht.\n" * 10
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


def test_very_long_text_no_timeout():
    """Sehr langer Text (>10.000 Zeilen) wird performant verarbeitet."""
    import time

    meta = {"file_type": ".txt", "source": "/data/grosses_log.txt"}
    text = "\n".join([f"Zeile {i}: Logmeldung fuer Performance-Test" for i in range(15000)])
    start = time.time()
    size, overlap = detect_chunk_params(text, meta)
    duration = time.time() - start
    assert duration < 2.0, f"Verarbeitung dauerte {duration:.2f}s (>2s)"
    assert size > 0
    assert "chunk_strategy" in meta


def test_mixed_signals_pipes_win_over_kv():
    """Bei 50% Pipes + 50% KV gewinnt pipe_ratio (wird zuerst geprueft)."""
    meta = {"file_type": ".docx", "source": "/docs/mixed.docx"}
    lines = []
    for i in range(5):
        lines.append(f"| Spalte A | Spalte B | Wert {i} |")
    for i in range(5):
        lines.append(
            f"Feldname {i}: Wert {i} mit etwas mehr Text hier drin damit Zeilen laenger sind"
        )
    text = "\n".join(lines)
    size, overlap = detect_chunk_params(text, meta)
    # pipe_ratio = 0.5 > 0.4 → tabular (wird vor kv_ratio geprueft)
    assert meta["chunk_strategy"] == "tabular"
    assert (size, overlap) == STRATEGIES["tabular"]


def test_metadata_without_source_no_keyerror():
    """Metadata ohne 'source'-Key verursacht keinen KeyError."""
    meta = {"file_type": ".txt"}  # Kein "source"-Key
    text = (
        "Testtext fuer Metadata ohne source-Key mit genuegend Laenge fuer sinnvolle Analyse.\n" * 5
    )
    # Darf nicht crashen
    size, overlap = detect_chunk_params(text, meta)
    assert size > 0
    assert overlap >= 0
    assert "chunk_strategy" in meta


def test_strategies_dict_has_expected_keys():
    """STRATEGIES enthaelt alle erwarteten Strategie-Namen."""
    expected = {"prose", "tabular", "code", "structured", "email", "default"}
    assert set(STRATEGIES.keys()) == expected


def test_chunk_document_uses_text_splitter_for_html_visible_text():
    """Sichtbarer HTML-Text behaelt .html-Metadaten, nutzt aber Text-Splitting."""
    meta = {
        "file_type": ".html",
        "parser": "html-to-markdown",
        "content_format": "markdown",
        "source_mime_type": "text/html",
        "source": "/web/page.html",
    }
    seen = {}
    original = chunker_mod.get_splitter

    def spy_get_splitter(file_type, *args, **kwargs):
        seen["file_type"] = file_type
        return original(file_type, *args, **kwargs)

    with patch("kiara.ingestion.chunker.get_splitter", side_effect=spy_get_splitter):
        chunks = chunker_mod.chunk_document(
            "A long visible HTML paragraph. " * 100,
            meta,
            chunk_size=200,
            chunk_overlap=20,
        )

    assert seen["file_type"] == ".md"
    assert chunks
    assert chunks[0]["metadata"]["file_type"] == ".html"
    assert chunks[0]["metadata"]["parser"] == "html-to-markdown"


# ── CAD-Dateityp-Tests (Phase 8) ─────────────────────────────────────────────


def test_dxf_detected_as_structured():
    """.dxf-Datei wird als structured erkannt."""
    meta = {"file_type": ".dxf", "source": "/cad/zeichnung.dxf"}
    text = "=== CAD-ZEICHNUNG: zeichnung.dxf ===\nARTIKELNR: 12345"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["structured"]
    assert meta["chunk_strategy"] == "structured"


def test_step_detected_as_structured():
    """.step-Datei wird als structured erkannt."""
    meta = {"file_type": ".step", "source": "/cad/baugruppe.step"}
    text = "=== CAD-MODELL: baugruppe.step ===\nEinzelteil"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["structured"]
    assert meta["chunk_strategy"] == "structured"


def test_igs_detected_as_structured():
    """.igs-Datei wird als structured erkannt."""
    meta = {"file_type": ".igs", "source": "/cad/teil.igs"}
    text = "=== CAD-MODELL: teil.igs ===\nBounding-Box: 100 x 50 x 25 mm"
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["structured"]
    assert meta["chunk_strategy"] == "structured"


def test_cad_kv_output_stays_structured():
    """CAD-Output mit Key-Value-Paaren bleibt structured (Stage 2 ueberschreibt nicht)."""
    meta = {"file_type": ".dxf", "source": "/cad/detail.dxf"}
    # CAD-Output hat viele Key-Value-Zeilen — darf trotzdem nicht von
    # Textanalyse ueberschrieben werden, weil CAD in _CAD_EXTENSIONS ist
    text = "\n".join(
        [
            "=== CAD-ZEICHNUNG: detail.dxf ===",
            "--- SCHRIFTFELD ---",
            "ARTIKELNR: 99887-002",
            "MATERIAL: S235JR",
            "GEWICHT: 12.5 kg",
            "DATUM: 2025-01-15",
            "BEARBEITER: Max",
            "--- BEMASSUNGEN ---",
            "150.00",
            "75.50",
        ]
    )
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["structured"]
    assert meta["chunk_strategy"] == "structured"


# ── Bild-Extension-Tests (Phase 9 DDU) ───────────────────────────────────────


def test_png_detected_as_prose():
    """.png-Datei → detect_chunk_params() liefert 'prose' (768, 150)."""
    meta = {"file_type": ".png", "source": "/scans/document.png"}
    text = "OCR-erkannter Text aus einem gescannten Dokument mit mehreren Absaetzen.\n" * 5
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_jpg_detected_as_prose():
    """.jpg-Datei → detect_chunk_params() liefert 'prose' (768, 150)."""
    meta = {"file_type": ".jpg", "source": "/scans/photo.jpg"}
    text = "Text aus einer fotografierten Seite mit OCR-Erkennung und langem Fliesstext.\n" * 5
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_tiff_detected_as_prose():
    """.tiff-Datei → detect_chunk_params() liefert 'prose' (768, 150)."""
    meta = {"file_type": ".tiff", "source": "/scans/scan.tiff"}
    text = "Gescanntes Dokument mit OCR-Text der aus einem TIFF-Bild extrahiert wurde.\n" * 5
    size, overlap = detect_chunk_params(text, meta)
    assert (size, overlap) == STRATEGIES["prose"]
    assert meta["chunk_strategy"] == "prose"


def test_image_with_tabular_ocr_corrected():
    """Bild mit tabellarischem OCR-Text (pipe_ratio > 0.4) → Stufe 2 korrigiert zu 'tabular'."""
    meta = {"file_type": ".png", "source": "/scans/tabelle.png"}
    lines = []
    for i in range(6):
        lines.append(f"| Spalte A | Spalte B | Wert {i} |")
    for i in range(4):
        lines.append(f"Normaler Text Zeile {i}")
    text = "\n".join(lines)
    size, overlap = detect_chunk_params(text, meta)
    # Stufe 1: .png → "prose"
    # Stufe 2: pipe_ratio > 0.4 → korrigiert zu "tabular"
    assert (size, overlap) == STRATEGIES["tabular"]
    assert meta["chunk_strategy"] == "tabular"


def test_image_extensions_set():
    """_IMAGE_EXTENSIONS enthaelt genau die erwarteten Extensions."""
    expected = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}
    assert expected == _IMAGE_EXTENSIONS
