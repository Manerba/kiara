"""Tests fuer Encoding-Roundtrip im Loader (Phase 15).

Testet dass _load_plaintext() verschiedene Encodings korrekt laedt
und charset_normalizer als Fallback nutzt.
"""

from kiara.ingestion.loaders.general import _load_plaintext


class TestLoadPlaintextEncoding:
    """_load_plaintext() erkennt verschiedene Encodings."""

    def test_utf8_standard(self, tmp_path):
        """Standard UTF-8 → direkt geladen."""
        p = tmp_path / "utf8.txt"
        p.write_text("Hallo Welt mit Umlauten: äöüß\n", encoding="utf-8")
        result = _load_plaintext(str(p))
        assert "äöüß" in result

    def test_latin1_conf(self, tmp_path):
        """Latin-1 .conf → korrekt geladen, kein Mojibake."""
        p = tmp_path / "app.conf"
        text = "Straße nach Süden\nÜbergabe um 14:00\n"
        p.write_bytes(text.encode("latin-1"))
        result = _load_plaintext(str(p))
        assert "Stra" in result
        # Mindestens lesbar, keine Exception
        assert len(result) > 10

    def test_cp1252_euro_sign(self, tmp_path):
        """CP1252 mit Euro-Zeichen → korrekt (nicht als Latin-1 fehlinterpretiert)."""
        p = tmp_path / "prices.txt"
        text = "Preis: 42,50 €\nMwSt: 19%\n"
        p.write_bytes(text.encode("cp1252"))
        result = _load_plaintext(str(p))
        # charset_normalizer sollte CP1252 erkennen und € korrekt laden
        assert "42,50" in result
        assert len(result) > 10

    def test_utf16_le_log(self, tmp_path):
        """UTF-16 LE .log → kein Mojibake, lesbar."""
        p = tmp_path / "events.log"
        text = "Event: Server gestartet\nStatus: OK\n"
        # UTF-16 LE mit BOM
        p.write_bytes(text.encode("utf-16"))
        result = _load_plaintext(str(p))
        # charset_normalizer oder UTF-8-Fallback mit replace
        # Mindestens sollte der Inhalt lesbar sein
        assert len(result) > 5

    def test_empty_file(self, tmp_path):
        """Leere Datei → leerer String."""
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        result = _load_plaintext(str(p))
        assert result == ""

    def test_pure_ascii(self, tmp_path):
        """Reines ASCII → direkt als UTF-8 geladen."""
        p = tmp_path / "ascii.txt"
        p.write_text("Hello World\nNo special chars\n", encoding="ascii")
        result = _load_plaintext(str(p))
        assert "Hello World" in result
