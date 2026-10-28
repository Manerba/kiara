"""Tests fuer DXF-Loader und Hilfsfunktionen (Phase 8)."""

import pytest

ezdxf = pytest.importorskip("ezdxf", reason="ezdxf nicht installiert")

from kiara.ingestion.loader import (
    _dxf_clean_mtext,
    _dxf_extract_bom,
    _dxf_extract_dimensions,
    _dxf_extract_leaders,
    _dxf_extract_texts_by_layer,
    _dxf_extract_title_block,
    _load_dxf,
    load_file,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_dxf(tmp_path):
    """Erzeugt eine leere DXF-Datei ohne Entities."""
    path = tmp_path / "empty.dxf"
    doc = ezdxf.new("R2010")
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def dxf_with_title_block(tmp_path):
    """Erzeugt eine DXF-Datei mit Schriftfeld-Block."""
    path = tmp_path / "title_block.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    block = doc.blocks.new("SCHRIFTFELD")
    block.add_attdef("ARTIKELNR", insert=(0, 0), dxfattribs={"prompt": "Artikelnr"})
    block.add_attdef("MATERIAL", insert=(0, -5), dxfattribs={"prompt": "Material"})
    block.add_attdef("GEWICHT", insert=(0, -10), dxfattribs={"prompt": "Gewicht"})
    block.add_attdef("DATUM", insert=(0, -15), dxfattribs={"prompt": "Datum"})
    insert = msp.add_blockref("SCHRIFTFELD", insert=(0, 0))
    insert.add_auto_attribs(
        {
            "ARTIKELNR": "99887-002",
            "MATERIAL": "S235JR",
            "GEWICHT": "12.5 kg",
            "DATUM": "2025-01-15",
        }
    )
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def dxf_with_bom(tmp_path):
    """Erzeugt DXF mit Stueckliste (3 Positionen)."""
    path = tmp_path / "bom.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    block = doc.blocks.new("BOM_ROW")
    block.add_attdef("POS", insert=(0, 0))
    block.add_attdef("MENGE", insert=(10, 0))
    block.add_attdef("BEZEICHNUNG", insert=(20, 0))
    for i, (pos, menge, bez) in enumerate(
        [
            ("1", "4", "Sechskantschraube M8x30"),
            ("2", "2", "Flanschlager UCP205"),
            ("3", "1", "Welle d=25 L=400"),
        ]
    ):
        ins = msp.add_blockref("BOM_ROW", insert=(0, i * -10))
        ins.add_auto_attribs({"POS": pos, "MENGE": menge, "BEZEICHNUNG": bez})
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def dxf_with_texts(tmp_path):
    """Erzeugt DXF mit TEXT/MTEXT-Entities auf verschiedenen Layern."""
    path = tmp_path / "texts.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # TEXT auf Layer "0" an verschiedenen Y-Positionen
    msp.add_text("Oberer Text", dxfattribs={"layer": "0", "insert": (0, 100)})
    msp.add_text("Unterer Text", dxfattribs={"layer": "0", "insert": (0, 10)})
    # TEXT auf anderem Layer
    msp.add_text("Hinweis Layer2", dxfattribs={"layer": "HINWEISE", "insert": (0, 50)})
    # MTEXT
    msp.add_mtext("MTEXT Inhalt", dxfattribs={"layer": "0", "insert": (0, 200)})
    doc.saveas(str(path))
    return str(path)


@pytest.fixture
def dxf_with_dimensions(tmp_path):
    """Erzeugt DXF mit DIMENSION-Entities."""
    path = tmp_path / "dims.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # Lineare Bemassung
    msp.add_linear_dim(base=(0, -20), p1=(0, 0), p2=(150, 0))
    doc.saveas(str(path))
    return str(path)


# ── 1.1 _dxf_clean_mtext() ───────────────────────────────────────────────────


class TestDxfCleanMtext:
    """Tests fuer _dxf_clean_mtext() — reine String-Funktion."""

    def test_plain_text_unchanged(self):
        """Einfacher Text ohne Formatierung bleibt unveraendert."""
        assert _dxf_clean_mtext("Hallo Welt") == "Hallo Welt"

    def test_font_format_codes_removed(self):
        """MTEXT mit \\fArial|b0|i0; Formatcodes werden entfernt."""
        raw = r"\fArial|b0|i0;Technische Zeichnung"
        result = _dxf_clean_mtext(raw)
        assert result == "Technische Zeichnung"

    def test_newline_code(self):
        """\\P wird zu echtem Zeilenumbruch."""
        raw = r"Zeile 1\PZeile 2"
        result = _dxf_clean_mtext(raw)
        assert result == "Zeile 1\nZeile 2"

    def test_nbsp_code(self):
        """\\~ wird zu normalem Leerzeichen."""
        raw = r"Wort\~Wort"
        result = _dxf_clean_mtext(raw)
        assert result == "Wort Wort"

    def test_nested_braces_removed(self):
        """Geschweifte Klammern werden entfernt."""
        raw = "{Formatiert} Text"
        result = _dxf_clean_mtext(raw)
        assert result == "Formatiert Text"

    def test_empty_string(self):
        """Leerer String bleibt leer."""
        assert _dxf_clean_mtext("") == ""

    def test_all_format_codes_combined(self):
        """Alle bekannten Formatcodes gleichzeitig."""
        raw = r"\fArial|b1|i0;{Titel}\P\~\H2.5;Klein"
        result = _dxf_clean_mtext(raw)
        assert "Titel" in result
        assert "Klein" in result
        assert "\\" not in result
        assert "{" not in result


# ── 1.2 _dxf_extract_title_block() ───────────────────────────────────────────


class TestDxfExtractTitleBlock:
    """Tests fuer _dxf_extract_title_block()."""

    def test_extract_artikelnr(self, dxf_with_title_block):
        """INSERT mit ATTRIB 'ARTIKELNR' wird korrekt extrahiert."""
        doc = ezdxf.readfile(dxf_with_title_block)
        result = _dxf_extract_title_block(doc)
        tags = {tag: val for tag, val in result}
        assert "ARTIKELNR" in tags
        assert tags["ARTIKELNR"] == "99887-002"

    def test_multiple_title_tags(self, dxf_with_title_block):
        """Mehrere Schriftfeld-Tags werden alle gefunden."""
        doc = ezdxf.readfile(dxf_with_title_block)
        result = _dxf_extract_title_block(doc)
        tags = {tag for tag, _ in result}
        assert {"ARTIKELNR", "MATERIAL", "GEWICHT", "DATUM"} <= tags

    def test_bom_block_not_recognized_as_title(self, dxf_with_bom):
        """BOM-Block mit POS/MENGE-Tags wird NICHT als Schriftfeld erkannt."""
        doc = ezdxf.readfile(dxf_with_bom)
        result = _dxf_extract_title_block(doc)
        tags = {tag for tag, _ in result}
        assert "POS" not in tags
        assert "MENGE" not in tags

    def test_empty_dxf_no_inserts(self, empty_dxf):
        """DXF ohne INSERT-Entities liefert leere Liste."""
        doc = ezdxf.readfile(empty_dxf)
        result = _dxf_extract_title_block(doc)
        assert result == []

    def test_deduplication(self, tmp_path):
        """Duplikat-Tags werden per seen-Set dedupliziert."""
        path = tmp_path / "dupl.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        # Zwei Bloecke mit gleichem ARTIKELNR-Tag
        for i, block_name in enumerate(["BLK_A", "BLK_B"]):
            block = doc.blocks.new(block_name)
            block.add_attdef("ARTIKELNR", insert=(0, 0))
            ins = msp.add_blockref(block_name, insert=(i * 100, 0))
            ins.add_auto_attribs({"ARTIKELNR": f"ART-{i}"})
        doc.saveas(str(path))

        doc2 = ezdxf.readfile(str(path))
        result = _dxf_extract_title_block(doc2)
        artikelnr_tags = [tag for tag, _ in result if tag == "ARTIKELNR"]
        assert len(artikelnr_tags) == 1  # Nur einmal

    def test_english_tags_recognized(self, tmp_path):
        """Englische Tags (DRAWN_BY, MATERIAL, REVISION) werden erkannt."""
        path = tmp_path / "english.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        block = doc.blocks.new("TITLE")
        block.add_attdef("DRAWN_BY", insert=(0, 0))
        block.add_attdef("MATERIAL", insert=(0, -5))
        block.add_attdef("REVISION", insert=(0, -10))
        ins = msp.add_blockref("TITLE", insert=(0, 0))
        ins.add_auto_attribs(
            {
                "DRAWN_BY": "Max Mustermann",
                "MATERIAL": "Aluminium",
                "REVISION": "B",
            }
        )
        doc.saveas(str(path))

        doc2 = ezdxf.readfile(str(path))
        result = _dxf_extract_title_block(doc2)
        tags = {tag for tag, _ in result}
        assert "DRAWN_BY" in tags
        assert "MATERIAL" in tags
        assert "REVISION" in tags


# ── 1.3 _dxf_extract_bom() ───────────────────────────────────────────────────


class TestDxfExtractBom:
    """Tests fuer _dxf_extract_bom()."""

    def test_three_insert_blocks_detected(self, dxf_with_bom):
        """3 INSERT-Bloecke gleichen Namens mit POS/MENGE/BEZEICHNUNG → Stueckliste."""
        doc = ezdxf.readfile(dxf_with_bom)
        result = _dxf_extract_bom(doc)
        assert len(result) == 3
        assert result[0]["POS"] == "1"
        assert result[0]["BEZEICHNUNG"] == "Sechskantschraube M8x30"

    def test_empty_dxf_no_bom(self, empty_dxf):
        """DXF ohne BOM-relevante Bloecke liefert leere Liste."""
        doc = ezdxf.readfile(empty_dxf)
        result = _dxf_extract_bom(doc)
        assert result == []

    def test_single_insert_not_enough(self, tmp_path):
        """Nur 1 INSERT-Block ist zu wenig fuer BOM-Erkennung."""
        path = tmp_path / "single.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        block = doc.blocks.new("BOM_ROW")
        block.add_attdef("POS", insert=(0, 0))
        block.add_attdef("MENGE", insert=(10, 0))
        ins = msp.add_blockref("BOM_ROW", insert=(0, 0))
        ins.add_auto_attribs({"POS": "1", "MENGE": "2"})
        doc.saveas(str(path))

        doc2 = ezdxf.readfile(str(path))
        result = _dxf_extract_bom(doc2)
        assert result == []


# ── 1.4 _dxf_extract_texts_by_layer() ────────────────────────────────────────


class TestDxfExtractTextsByLayer:
    """Tests fuer _dxf_extract_texts_by_layer()."""

    def test_texts_grouped_by_layer(self, dxf_with_texts):
        """TEXT-Entities werden nach Layer gruppiert."""
        doc = ezdxf.readfile(dxf_with_texts)
        result = _dxf_extract_texts_by_layer(doc)
        assert "0" in result
        assert "HINWEISE" in result

    def test_mtext_uses_plain_text(self, dxf_with_texts):
        """MTEXT plain_text() wird bevorzugt."""
        doc = ezdxf.readfile(dxf_with_texts)
        result = _dxf_extract_texts_by_layer(doc)
        # MTEXT "MTEXT Inhalt" auf Layer "0"
        assert "MTEXT Inhalt" in result["0"]

    def test_sorted_by_y_descending(self, dxf_with_texts):
        """Sortierung nach Y-Koordinate (absteigend = oben→unten)."""
        doc = ezdxf.readfile(dxf_with_texts)
        result = _dxf_extract_texts_by_layer(doc)
        layer0 = result["0"]
        # Y=200 (MTEXT) > Y=100 (Oberer) > Y=10 (Unterer)
        assert layer0.index("MTEXT Inhalt") < layer0.index("Oberer Text")
        assert layer0.index("Oberer Text") < layer0.index("Unterer Text")

    def test_texts_on_different_layers(self, dxf_with_texts):
        """Texte auf verschiedenen Layern werden korrekt getrennt."""
        doc = ezdxf.readfile(dxf_with_texts)
        result = _dxf_extract_texts_by_layer(doc)
        assert "Hinweis Layer2" in result["HINWEISE"]
        assert "Hinweis Layer2" not in result["0"]

    def test_empty_dxf_returns_empty_dict(self, empty_dxf):
        """DXF ohne TEXT/MTEXT liefert leeres Dict."""
        doc = ezdxf.readfile(empty_dxf)
        result = _dxf_extract_texts_by_layer(doc)
        assert result == {}


# ── 1.5 _dxf_extract_dimensions() ────────────────────────────────────────────


class TestDxfExtractDimensions:
    """Tests fuer _dxf_extract_dimensions()."""

    def test_linear_dimension_extracted(self, dxf_with_dimensions):
        """Lineare Bemassung wird extrahiert."""
        doc = ezdxf.readfile(dxf_with_dimensions)
        result = _dxf_extract_dimensions(doc)
        assert len(result) >= 1
        # Bemassung sollte ca. 150.00 sein (Distanz 0,0 zu 150,0)
        found_150 = any("150" in d for d in result)
        assert found_150, f"Erwartet '150' in {result}"

    def test_empty_dxf_no_dimensions(self, empty_dxf):
        """DXF ohne Bemassungen liefert leere Liste."""
        doc = ezdxf.readfile(empty_dxf)
        result = _dxf_extract_dimensions(doc)
        assert result == []

    def test_dimension_placeholder_inside_text_replaced(self):
        """Texte wie 'R<>' oder '<> mm' werden mit dem Messwert ersetzt."""

        class _Dxf:
            text = "R<> mm"

        class _Dim:
            dxf = _Dxf()
            measurement = 12.345

            def dxftype(self):
                return "DIMENSION"

        class _Doc:
            layouts = []

            def modelspace(self):
                return [_Dim()]

        result = _dxf_extract_dimensions(_Doc())

        assert result == ["R12.35 mm"]


# ── 1.6 _dxf_extract_leaders() ───────────────────────────────────────────────


class TestDxfExtractLeaders:
    """Tests fuer _dxf_extract_leaders()."""

    def test_empty_dxf_no_leaders(self, empty_dxf):
        """DXF ohne Leader liefert leere Liste."""
        doc = ezdxf.readfile(empty_dxf)
        result = _dxf_extract_leaders(doc)
        assert result == []


# ── 2. DXF-Loader Integrationstests ──────────────────────────────────────────


class TestLoadDxf:
    """Integrationstests fuer _load_dxf() und load_file()."""

    def test_full_dxf_has_header(self, dxf_with_title_block):
        """_load_dxf() Output hat CAD-ZEICHNUNG Header."""
        result = _load_dxf(dxf_with_title_block)
        assert "=== CAD-ZEICHNUNG:" in result

    def test_full_dxf_has_schriftfeld(self, dxf_with_title_block):
        """_load_dxf() Output hat SCHRIFTFELD Subheader."""
        result = _load_dxf(dxf_with_title_block)
        assert "--- SCHRIFTFELD ---" in result
        assert "ARTIKELNR: 99887-002" in result

    def test_empty_dxf_no_crash(self, empty_dxf):
        """Leere DXF gibt Header zurueck, kein Crash."""
        result = _load_dxf(empty_dxf)
        assert result is not None
        assert "=== CAD-ZEICHNUNG:" in result

    def test_bom_only_dxf(self, dxf_with_bom):
        """DXF mit BOM (kein Schriftfeld) hat STUECKLISTE Sektion."""
        result = _load_dxf(dxf_with_bom)
        assert "--- STUECKLISTE ---" in result
        assert "Sechskantschraube M8x30" in result
        # Kein Schriftfeld (BOM-Bloecke werden gefiltert)
        assert "--- SCHRIFTFELD ---" not in result

    def test_texts_dxf(self, dxf_with_texts):
        """DXF mit Annotationen hat ANNOTATIONEN Sektion."""
        result = _load_dxf(dxf_with_texts)
        assert "--- ANNOTATIONEN" in result

    def test_load_file_routes_to_dxf(self, dxf_with_title_block):
        """load_file() mit .dxf-Pfad routet korrekt zu _load_dxf."""
        result = load_file(dxf_with_title_block)
        assert result is not None
        assert "=== CAD-ZEICHNUNG:" in result["text"]
        assert result["metadata"]["file_type"] == ".dxf"


# ── 5. Edge Cases & Robustheit ────────────────────────────────────────────────


class TestDxfEdgeCases:
    """Edge Cases und Robustheits-Tests."""

    def test_special_chars_in_filename(self, tmp_path):
        """DXF mit Sonderzeichen im Dateinamen (Umlaute, Leerzeichen)."""
        path = tmp_path / "Zeichnung Übersicht (Rev-B).dxf"
        doc = ezdxf.new("R2010")
        doc.saveas(str(path))
        result = _load_dxf(str(path))
        assert result is not None
        assert "Zeichnung Übersicht (Rev-B).dxf" in result

    def test_corrupt_dxf_returns_none(self, tmp_path):
        """Korrupte DXF-Datei → load_file() gibt None zurueck (_load_dxf liefert None)."""
        path = tmp_path / "corrupt.dxf"
        path.write_text("DIES IST KEINE GUELTIGE DXF DATEI")
        result = load_file(str(path))
        # _load_dxf gibt None zurueck bei korrupten Dateien (readfile() Exception)
        # → load_file erkennt None und liefert None (Datei nicht verarbeitbar)
        assert result is None

    def test_many_text_entities(self, tmp_path):
        """DXF mit 1000+ TEXT-Entities → kein Timeout."""
        import time

        path = tmp_path / "many_texts.dxf"
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        for i in range(1200):
            msp.add_text(f"Text Nr. {i}", dxfattribs={"insert": (0, i)})
        doc.saveas(str(path))

        start = time.time()
        result = _load_dxf(str(path))
        duration = time.time() - start
        assert result is not None
        assert duration < 5.0, f"Verarbeitung dauerte {duration:.2f}s"
        assert "Text Nr. 0" in result
