"""Tests fuer kiara.ingestion.content_detect (Phase 15).

Testet die content-basierte Dateierkennung: Text vs. Binaer,
verschiedene Encodings, Grenzwerte, I/O-Fehler.
"""

import os

import pytest

pytest.importorskip("charset_normalizer", reason="charset_normalizer nicht installiert")

from kiara.ingestion.content_detect import detect_content, is_text_file


class TestDetectContentText:
    """Textdateien werden korrekt erkannt."""

    def test_utf8_text_file(self, tmp_path):
        """UTF-8-Textdatei mit .xyz-Endung → is_text=True."""
        p = tmp_path / "readme.xyz"
        p.write_text("Dies ist eine Textdatei mit Umlauten: äöü\n", encoding="utf-8")
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc is not None
        assert conf > 0.5

    def test_latin1_text_file(self, tmp_path):
        """Latin-1-Textdatei → is_text=True, Encoding erkannt."""
        p = tmp_path / "config.conf"
        p.write_bytes("Stra\xdfe nach S\xfcden\n".encode("latin-1"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc is not None

    def test_utf16_le_with_bom(self, tmp_path):
        """UTF-16 LE mit BOM → is_text=True (nicht als binaer erkannt!)."""
        p = tmp_path / "log.log"
        text = "Server gestartet um 08:15\nKeine Fehler gefunden\n"
        p.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert "utf-16" in (enc or "").lower()

    def test_utf16_be_with_bom(self, tmp_path):
        """UTF-16 BE mit BOM → is_text=True."""
        p = tmp_path / "data.dat"
        text = "Hello World\nLine 2\n"
        p.write_bytes(b"\xfe\xff" + text.encode("utf-16-be"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert "utf-16" in (enc or "").lower()

    def test_utf8_no_bom(self, tmp_path):
        """UTF-8 ohne BOM → is_text=True."""
        p = tmp_path / "notes.txt"
        p.write_text("Einfacher Text ohne besondere Zeichen\n", encoding="utf-8")
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True

    def test_is_text_file_wrapper(self, tmp_path):
        """Wrapper is_text_file() gibt bool zurueck."""
        p = tmp_path / "test.log"
        p.write_text("Log-Eintrag\n", encoding="utf-8")
        assert is_text_file(str(p)) is True


class TestDetectContentBinary:
    """Binaerdateien werden korrekt abgelehnt."""

    def test_binary_file(self, tmp_path):
        """Datei mit vielen Non-Printable-Bytes → is_text=False."""
        p = tmp_path / "data.bin"
        # 50% Non-Printable Bytes
        p.write_bytes(bytes(range(256)) * 4)
        is_text, _, _ = detect_content(str(p))
        assert is_text is False

    def test_exe_like_binary(self, tmp_path):
        """EXE-aehnliche Datei → is_text=False."""
        p = tmp_path / "app.exe"
        p.write_bytes(b"\x4d\x5a" + b"\x00" * 100 + b"\x90" * 100)
        is_text, _, _ = detect_content(str(p))
        assert is_text is False

    def test_is_text_file_binary(self, tmp_path):
        """Wrapper is_text_file() gibt False fuer Binaer zurueck."""
        p = tmp_path / "data.bin"
        p.write_bytes(bytes(range(256)) * 4)
        assert is_text_file(str(p)) is False

    def test_utf16_without_bom_rejected(self, tmp_path):
        """Binaerdaten die als UTF-16 ohne BOM erkannt werden → is_text=False.

        charset_normalizer interpretiert Bytefolgen wie PNG-Header als UTF-16,
        weil die \\x00-Bytes mit Header-Bytes in CJK-Codepoints decoden, die
        isprintable() nicht ablehnt. Echte UTF-16-Dateien tragen praktisch
        immer ein BOM und werden vom BOM-Pfad behandelt.
        """
        p = tmp_path / "image.png"
        # PNG-Signatur + IHDR + Whitespace-aehnliche Bytes,
        # die mit \\x00 zu CJK/Latin-Codepoints decoden
        png = (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\rIHDR"
            + b"\x00\x01\x00\x01\x08\x06\x00\x00\x00"
            + b"abcdefghij" * 200
        )
        p.write_bytes(png)
        is_text, _, _ = detect_content(str(p))
        assert is_text is False


class TestDetectContentEdgeCases:
    """Grenzfaelle und Fehlerbehandlung."""

    def test_empty_file(self, tmp_path):
        """Leere Datei → is_text=True mit leerem UTF-8-Text."""
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc == "utf-8"
        assert conf == 1.0

    def test_nonexistent_file(self, tmp_path):
        """Nicht-existente Datei → is_text=False, confidence=0.0."""
        p = tmp_path / "nope.txt"
        is_text, enc, conf = detect_content(str(p))
        assert is_text is False
        assert conf == 0.0

    def test_threshold_below(self, tmp_path):
        """4.9% Non-Printable → is_text=True (unter 5% Schwelle)."""
        # 1000 Zeichen: 49 Non-Printable, 951 Printable
        printable = b"A" * 951
        non_printable = b"\x01" * 49
        p = tmp_path / "almost.dat"
        p.write_bytes(printable + non_printable)
        is_text, _, _ = detect_content(str(p))
        assert is_text is True

    def test_threshold_above(self, tmp_path):
        """5.1% Non-Printable → is_text=False (ueber 5% Schwelle)."""
        # 1000 Zeichen: 51 Non-Printable, 949 Printable
        printable = b"A" * 949
        non_printable = b"\x01" * 51
        p = tmp_path / "toomuch.dat"
        p.write_bytes(printable + non_printable)
        is_text, _, _ = detect_content(str(p))
        assert is_text is False

    @pytest.mark.skipif(os.getuid() == 0, reason="root ignoriert Dateiberechtigungen")
    def test_permission_error(self, tmp_path):
        """Datei ohne Leserechte → is_text=False."""
        p = tmp_path / "noaccess.txt"
        p.write_text("geheim")
        os.chmod(str(p), 0o000)
        try:
            is_text, _, conf = detect_content(str(p))
            assert is_text is False
            assert conf == 0.0
        finally:
            os.chmod(str(p), 0o644)
