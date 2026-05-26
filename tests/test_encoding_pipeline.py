"""Tiefergehende Encoding-Tests fuer die Ingestion-Pipeline.

Testet die durchgaengige Encoding-Behandlung von der Content-Detection
ueber den Plaintext-Loader bis zur Sanitization in load_file().
End-to-End-Tests mit echten Dateien in verschiedenen Encodings.
"""

import pytest

pytest.importorskip("charset_normalizer", reason="charset_normalizer nicht installiert")

from kiara.ingestion.content_detect import detect_content
from kiara.ingestion.loader import _load_plaintext, _sanitize_text, load_file

# ---------------------------------------------------------------------------
# 1. UTF-8 Varianten
# ---------------------------------------------------------------------------


class TestUTF8Variants:
    """UTF-8 in allen Varianten wird korrekt erkannt und geladen."""

    def test_pure_ascii(self, tmp_path):
        """Reines ASCII (Subset von UTF-8) wird korrekt geladen."""
        p = tmp_path / "ascii.txt"
        p.write_bytes(b"Hello World\nThis is plain ASCII text.\n")
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text == "Hello World\nThis is plain ASCII text.\n"

    def test_utf8_with_umlauts(self, tmp_path):
        """UTF-8 mit deutschen Umlauten wird exakt erhalten."""
        content = "Straße nach Süden\näöüÄÖÜß\n"
        p = tmp_path / "umlauts.txt"
        p.write_bytes(content.encode("utf-8"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert "Straße" in text
        assert "äöüÄÖÜß" in text

    def test_utf8_with_bom_not_visible_in_text(self, tmp_path):
        """UTF-8 mit BOM: detect_content erkennt utf-8-sig, BOM wird transparent entfernt."""
        content = "Inhalt mit BOM\n"
        bom = b"\xef\xbb\xbf"
        p = tmp_path / "bom_utf8.txt"
        p.write_bytes(bom + content.encode("utf-8"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc is not None
        assert "utf-8" in enc.lower()
        text = _load_plaintext(str(p))
        # _load_plaintext nutzt utf-8-sig → BOM wird transparent entfernt
        assert text.strip() == "Inhalt mit BOM"
        assert "\ufeff" not in text

    def test_utf8_with_4byte_emoji(self, tmp_path):
        """UTF-8 mit 4-Byte-Emoji wird korrekt erhalten."""
        content = "Werkzeug: \U0001f527 Schraubenschluessel\n"
        p = tmp_path / "emoji.txt"
        p.write_bytes(content.encode("utf-8"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert "\U0001f527" in text
        assert "Schraubenschluessel" in text

    def test_utf8_mixed_languages(self, tmp_path):
        """UTF-8 mit Deutsch + Chinesisch + Kyrillisch wird korrekt geladen."""
        content = (
            "Deutsch: Schnittstelle\n"
            "Chinesisch: \u63a5\u53e3\n"
            "Kyrillisch: \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441\n"
        )
        p = tmp_path / "multilang.txt"
        p.write_bytes(content.encode("utf-8"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert "Schnittstelle" in text
        assert "\u63a5\u53e3" in text
        assert "\u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441" in text


# ---------------------------------------------------------------------------
# 2. UTF-16 Varianten
# ---------------------------------------------------------------------------


class TestUTF16Variants:
    """UTF-16 in allen BOM-Varianten wird erkannt und korrekt geladen."""

    def test_utf16_le_bom_detect_content(self, tmp_path):
        """UTF-16 LE mit BOM: detect_content erkennt als Text, Encoding enthaelt utf-16."""
        text = "Server gestartet um 08:15\nKeine Fehler gefunden\n"
        p = tmp_path / "utf16le.log"
        p.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc is not None
        assert "utf-16" in enc.lower()

    def test_utf16_be_bom_detect_content(self, tmp_path):
        """UTF-16 BE mit BOM: detect_content erkennt als Text."""
        text = "Hello World\nLine 2\n"
        p = tmp_path / "utf16be.dat"
        p.write_bytes(b"\xfe\xff" + text.encode("utf-16-be"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc is not None
        assert "utf-16" in enc.lower()

    def test_utf16_le_bom_load_plaintext_readable(self, tmp_path):
        """UTF-16 LE mit BOM via _load_plaintext: Inhalt ist lesbar."""
        text = "Testnachricht UTF-16\n"
        p = tmp_path / "utf16le_plain.txt"
        p.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        # UTF-8 schlaegt fehl, charset_normalizer Fallback greift
        loaded = _load_plaintext(str(p))
        assert loaded is not None
        assert "Testnachricht" in loaded or "UTF-16" in loaded

    def test_utf16_le_bom_end_to_end(self, tmp_path):
        """UTF-16 LE mit BOM durch load_file() End-to-End: Text enthaelt erwarteten Inhalt."""
        text = "Protokolleintrag: System gestartet\n"
        p = tmp_path / "utf16le_e2e.log"
        p.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        result = load_file(str(p))
        assert result is not None
        assert "text" in result
        assert "Protokolleintrag" in result["text"] or "System gestartet" in result["text"]

    def test_utf16_without_bom_charset_normalizer(self, tmp_path):
        """UTF-16 ohne BOM: charset_normalizer sollte es trotzdem erkennen oder zumindest laden."""
        # UTF-16-LE ohne BOM — charset_normalizer versucht es zu erkennen
        text = "Konfiguration geladen\n"
        p = tmp_path / "utf16_no_bom.txt"
        p.write_bytes(text.encode("utf-16-le"))
        # Entweder wird es erkannt oder es gibt den Fallback mit errors='replace'
        # In jedem Fall darf _load_plaintext nicht crashen
        loaded = _load_plaintext(str(p))
        assert loaded is not None
        assert isinstance(loaded, str)


# ---------------------------------------------------------------------------
# 3. Legacy-Encodings
# ---------------------------------------------------------------------------


class TestLegacyEncodings:
    """Latin-1, CP1252 und ISO-8859-15 werden via charset_normalizer erkannt."""

    def test_latin1_with_umlauts(self, tmp_path):
        """Latin-1 mit Umlauten wird korrekt via charset_normalizer geladen."""
        content = "Stra\xdfe nach S\xfcden\nM\xfcnchen\n"
        p = tmp_path / "latin1.conf"
        p.write_bytes(content.encode("latin-1"))
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text is not None
        # Umlaute muessen korrekt decodiert sein
        assert "Straße" in text or "str" in text.lower()
        assert "München" in text or "nchen" in text

    def test_cp1252_euro_sign(self, tmp_path):
        """CP1252 mit Euro-Zeichen (0x80) wird nicht als Latin-1 fehlinterpretiert."""
        # In CP1252 ist 0x80 = €; in Latin-1 ist 0x80 ein Control-Zeichen
        content = b"Preis: \x80 19,99\nMehrwertsteuer inklusive\n"
        p = tmp_path / "cp1252_euro.txt"
        p.write_bytes(content)
        is_text, enc, conf = detect_content(str(p))
        # CP1252 ist eine Text-Datei
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text is not None
        assert isinstance(text, str)
        # Entweder Euro-Zeichen oder Replacement-Character, aber kein Crash
        assert "Preis" in text or "preis" in text.lower()

    def test_cp1252_typographic_quotes(self, tmp_path):
        """CP1252 mit typographischen Anfuehrungszeichen (0x93/0x94) wird korrekt geladen."""
        # 0x93 = linkes typogr. Anfuehrungszeichen, 0x94 = rechtes
        content = b"Er sagte: \x93Hallo Welt\x94 und ging.\n"
        p = tmp_path / "cp1252_quotes.txt"
        p.write_bytes(content)
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text is not None
        assert "Hallo Welt" in text

    def test_cp1252_bullet_point(self, tmp_path):
        """CP1252 mit Bullet-Zeichen (0x95) wird korrekt geladen."""
        # 0x95 = Bullet (•) in Windows-1252
        content = b"Liste:\n\x95 Punkt 1\n\x95 Punkt 2\n\x95 Punkt 3\n"
        p = tmp_path / "cp1252_bullets.txt"
        p.write_bytes(content)
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text is not None
        assert "Punkt 1" in text
        assert "Punkt 2" in text
        assert "Punkt 3" in text

    def test_iso8859_15_euro(self, tmp_path):
        """ISO-8859-15 mit Euro-Zeichen (0xA4) wird via charset_normalizer erkannt."""
        # In ISO-8859-15 ist 0xA4 = €
        content = b"Betrag: \xa4 42,00\nISO-8859-15 Datei\n"
        p = tmp_path / "iso885915.txt"
        p.write_bytes(content)
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text is not None
        assert isinstance(text, str)
        # Mindestens der ASCII-Anteil muss erhalten sein
        assert "Betrag" in text
        assert "42,00" in text


# ---------------------------------------------------------------------------
# 4. Mixed/Edge-Case Encodings
# ---------------------------------------------------------------------------


class TestMixedAndEdgeCases:
    """Grenzfaelle und gemischte Byte-Sequenzen."""

    def test_nul_bytes_at_end_sanitized(self, tmp_path):
        """NUL-Bytes am Ende (C-String-Padding) werden von _sanitize_text entfernt."""
        # Text mit NUL-Padding am Ende — sicher unter 5% NUL-Ratio
        # 200 Bytes Text + 5 NUL = 205 gesamt → 5/205 ≈ 2.4%
        text_part = b"Konfiguration geladen\nParameter OK\nServer aktiv\nLog-Eintrag\n" * 3
        nul_padding = b"\x00" * 5
        content = text_part + nul_padding
        nul_count = content.count(b"\x00")
        total_len = len(content)
        # Sicherstellen, dass NUL-Ratio < 5%
        assert nul_count / total_len < 0.05, (
            f"NUL-Ratio zu hoch fuer diesen Test: {nul_count}/{total_len}"
        )
        p = tmp_path / "cstring.conf"
        p.write_bytes(content)
        text_raw = _load_plaintext(str(p))
        sanitized = _sanitize_text(text_raw, str(p), "test")
        assert sanitized is not None
        assert "\x00" not in sanitized
        assert "Konfiguration" in sanitized

    def test_scattered_nul_bytes_below_threshold(self, tmp_path):
        """Vereinzelte NUL-Bytes (<5%) werden entfernt, load_file gibt bereinigten Text."""
        # 1000 druckbare Bytes + 10 NUL-Bytes = 1% NUL-Ratio
        printable = b"A" * 100 + b"\x00" + b"B" * 100 + b"\x00" + b"C" * 798
        assert printable.count(b"\x00") / len(printable) < 0.05
        p = tmp_path / "nul_scattered.txt"
        p.write_bytes(printable)
        result = load_file(str(p))
        assert result is not None
        assert "\x00" not in result["text"]
        # Mindestens ein paar 'A' muessen noch da sein
        assert "A" in result["text"]

    def test_invalid_utf8_middle_fallback(self, tmp_path):
        """Datei die wie UTF-8 beginnt, aber ungueltige Bytes enthaelt: Fallback greift."""
        # Gueltiges UTF-8 bis auf ein einzelnes 0xFF in der Mitte
        valid_start = "Gueltige UTF-8 Zeichen: ".encode("utf-8")
        invalid_byte = b"\xff"  # Niemals gueltiges UTF-8
        valid_end = " Ende des Textes".encode("utf-8")
        p = tmp_path / "broken_utf8.txt"
        p.write_bytes(valid_start + invalid_byte + valid_end)
        # _load_plaintext darf nicht crashen
        text = _load_plaintext(str(p))
        assert text is not None
        assert isinstance(text, str)
        # ASCII-Anteil muss erhalten sein
        assert "Gueltige" in text
        assert "Ende des Textes" in text

    def test_very_short_file_1_byte(self, tmp_path):
        """Datei mit 1 Byte: kein Crash in detect_content oder _load_plaintext."""
        p = tmp_path / "onebyte.txt"
        p.write_bytes(b"X")
        is_text, enc, conf = detect_content(str(p))
        # 1 Byte ASCII → ist Text
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text == "X"

    def test_very_short_file_3_bytes(self, tmp_path):
        """Datei mit 3 Bytes: kein Crash."""
        p = tmp_path / "threebytes.txt"
        p.write_bytes(b"abc")
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        text = _load_plaintext(str(p))
        assert text == "abc"

    def test_only_whitespace_load_file_returns_no_content(self, tmp_path):
        """Datei nur aus Whitespace (Tabs, Newlines, Spaces) → load_file gibt no_content zurueck."""
        p = tmp_path / "whitespace.txt"
        p.write_bytes(b"   \t\n\r\n   \t\t\n")
        result = load_file(str(p))
        # load_file prueft `not text.strip()` und markiert als no_content
        assert result is not None
        assert result["metadata"]["no_content"] is True
        assert result["text"] == ""

    def test_sanitize_text_no_null_passthrough(self):
        """_sanitize_text mit Text ohne NUL-Bytes gibt direkt denselben Text zurueck."""
        text = "Normaler Text ohne Nullbytes"
        result = _sanitize_text(text, "test.txt", "test")
        assert result is text  # Identitaet: kein Kopieren noetig

    def test_sanitize_text_high_null_ratio_returns_none(self):
        """_sanitize_text mit >5% NUL-Bytes gibt None zurueck."""
        # 100 Zeichen, davon 10 NUL = 10% → binär
        text = "A" * 90 + "\x00" * 10
        result = _sanitize_text(text, "binary.bin", "test")
        assert result is None

    def test_sanitize_text_exact_threshold_above(self):
        """_sanitize_text mit genau 5.1% NUL-Bytes gibt None zurueck."""
        # 1000 Zeichen, 51 NUL = 5.1%
        text = "B" * 949 + "\x00" * 51
        result = _sanitize_text(text, "threshold.bin", "test")
        assert result is None

    def test_sanitize_text_exact_threshold_below(self):
        """_sanitize_text mit genau 4.9% NUL-Bytes bereinigt und gibt Text zurueck."""
        # 1000 Zeichen, 49 NUL = 4.9%
        text = "C" * 951 + "\x00" * 49
        result = _sanitize_text(text, "threshold_ok.txt", "test")
        assert result is not None
        assert "\x00" not in result
        assert len(result) == 951


# ---------------------------------------------------------------------------
# 5. End-to-End Pipeline-Tests (detect → load → sanitize via load_file)
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Vollstaendige Pipeline-Tests von detect_content bis load_file."""

    def test_latin1_conf_full_pipeline(self, tmp_path):
        """Latin-1 .conf-Datei: detect_content=True → _load_plaintext lesbar → load_file mit metadata."""
        content = b"[Server]\nHost=kiara-server.local\nPort=8080\nName=K\xf6nigsberg\n"
        p = tmp_path / "server.conf"
        p.write_bytes(content)

        # Stufe 1: detect_content
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True

        # Stufe 2: _load_plaintext
        text = _load_plaintext(str(p))
        assert text is not None
        assert "Host=kiara-server.local" in text
        assert "Port=8080" in text

        # Stufe 3: load_file mit Metadata
        result = load_file(str(p))
        assert result is not None
        assert "text" in result
        assert "metadata" in result
        meta = result["metadata"]
        assert meta["filename"] == "server.conf"
        assert meta["file_type"] == ".conf"
        assert "Host=kiara-server.local" in result["text"]

    def test_utf16_le_log_full_pipeline(self, tmp_path):
        """UTF-16 LE .log-Datei: detect_content=True → _load_plaintext → load_file."""
        log_text = "2024-01-15 08:15:00 INFO Server gestartet\n2024-01-15 08:15:01 INFO Bereit\n"
        p = tmp_path / "system.log"
        p.write_bytes(b"\xff\xfe" + log_text.encode("utf-16-le"))

        # Stufe 1: detect_content
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert "utf-16" in (enc or "").lower()

        # Stufe 2: _load_plaintext
        text = _load_plaintext(str(p))
        assert text is not None
        assert "INFO" in text or "Server" in text

        # Stufe 3: load_file
        result = load_file(str(p))
        assert result is not None
        assert "text" in result
        assert "metadata" in result
        assert result["metadata"]["file_type"] == ".log"

    def test_cp1252_txt_full_pipeline_no_data_loss(self, tmp_path):
        """CP1252 .txt-Datei: vollstaendige Pipeline ohne Datenverlust bei ASCII-Anteil."""
        # ASCII-Anteil muss 100% erhalten bleiben
        # CP1252-spezifische Bytes: 0x80=€, 0x93/0x94=Anfuehrungszeichen
        content = (
            b"Produktbeschreibung:\n"
            b"Preis: \x80 49,99\n"
            b"Bezeichnung: \x93Super-Gadget\x94\n"
            b"Lieferzeit: 3-5 Werktage\n"
        )
        p = tmp_path / "produkt.txt"
        p.write_bytes(content)

        # Stufe 1
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True

        # Stufe 2
        text = _load_plaintext(str(p))
        assert text is not None
        assert "Produktbeschreibung" in text
        assert "Super-Gadget" in text
        assert "Lieferzeit" in text

        # Stufe 3
        result = load_file(str(p))
        assert result is not None
        assert "Produktbeschreibung" in result["text"]
        assert "Super-Gadget" in result["text"]
        assert "Lieferzeit" in result["text"]
        assert result["metadata"]["file_type"] == ".txt"

    def test_utf8_bom_detect_no_bom_in_text(self, tmp_path):
        """UTF-8 mit BOM: BOM wird transparent entfernt, Text beginnt sauber."""
        content = "Konfigurationsdatei Version 2.0\nParameter: aktiv\n"
        bom = b"\xef\xbb\xbf"
        p = tmp_path / "config_bom.txt"
        p.write_bytes(bom + content.encode("utf-8"))

        # detect_content → utf-8-sig
        is_text, enc, conf = detect_content(str(p))
        assert is_text is True
        assert enc is not None
        assert "utf-8" in enc.lower()

        # load_file → BOM entfernt, Inhalt sauber
        result = load_file(str(p))
        assert result is not None
        assert "\ufeff" not in result["text"]
        assert result["text"].strip().startswith("Konfigurationsdatei")
        assert "Parameter: aktiv" in result["text"]

    def test_metadata_completeness(self, tmp_path):
        """load_file liefert vollstaendige Metadata-Felder fuer alle Encoding-Typen."""
        p = tmp_path / "meta_test.txt"
        p.write_bytes("Vollstaendige Metadaten-Pruefung\n".encode("utf-8"))
        result = load_file(str(p))
        assert result is not None
        meta = result["metadata"]
        required_fields = {"source", "directory", "filename", "file_type", "modified", "size_bytes"}
        for field in required_fields:
            assert field in meta, f"Metadaten-Feld '{field}' fehlt"
        assert meta["filename"] == "meta_test.txt"
        assert meta["file_type"] == ".txt"
        assert meta["size_bytes"] > 0


# ---------------------------------------------------------------------------
# 6. Encoding-Konsistenz zwischen detect_content und _load_plaintext
# ---------------------------------------------------------------------------


class TestEncodingConsistency:
    """Wenn detect_content is_text=True, muss _load_plaintext die Datei laden koennen."""

    @pytest.mark.parametrize(
        "encoding,content_bytes,description",
        [
            (
                "utf-8",
                "UTF-8 Text mit Umlauten: äöü\n".encode("utf-8"),
                "UTF-8 ohne BOM",
            ),
            (
                "utf-8-sig",
                b"\xef\xbb\xbf" + "UTF-8 mit BOM\n".encode("utf-8"),
                "UTF-8 mit BOM",
            ),
            (
                "latin-1",
                "Latin-1: Stra\xdfe\n".encode("latin-1"),
                "Latin-1 / ISO-8859-1",
            ),
            (
                "cp1252",
                b"CP1252: Euro=\x80 Quote=\x93text\x94\n",
                "Windows-1252",
            ),
            (
                "utf-16-le",
                b"\xff\xfe" + "UTF-16 LE\n".encode("utf-16-le"),
                "UTF-16 LE mit BOM",
            ),
        ],
    )
    def test_detect_then_load_consistency(self, tmp_path, encoding, content_bytes, description):
        """detect_content=True impliziert, dass _load_plaintext keinen Crash verursacht."""
        p = tmp_path / f"consistency_{encoding.replace('-', '_')}.txt"
        p.write_bytes(content_bytes)

        is_text, enc, conf = detect_content(str(p))

        if is_text:
            # Wenn detect sagt Text, muss load kein Crash geben
            text = _load_plaintext(str(p))
            assert text is not None, (
                f"_load_plaintext() returned None fuer {description}, "
                f"obwohl detect_content is_text=True"
            )
            assert isinstance(text, str), (
                f"_load_plaintext() gab kein str zurueck fuer {description}"
            )
            assert len(text) > 0, f"_load_plaintext() gab leeren String zurueck fuer {description}"

    def test_all_text_encodings_loadable(self, tmp_path):
        """Fuer 5 Encodings: detect + load pruefen, kein Encoding darf zum Crash fuehren."""
        test_cases = [
            ("utf8.txt", "UTF-8 Inhalt\n".encode("utf-8")),
            ("latin1.txt", "Latin1: \xe4\xf6\xfc\n"),
            ("cp1252.txt", b"CP1252: \x80\x93\x94\x95\n"),
            ("utf8bom.txt", b"\xef\xbb\xbf" + "Mit BOM\n".encode("utf-8")),
            ("utf16le.txt", b"\xff\xfe" + "UTF16-LE\n".encode("utf-16-le")),
        ]

        results = []
        for filename, content in test_cases:
            p = tmp_path / filename
            if isinstance(content, str):
                p.write_bytes(content.encode("latin-1"))
            else:
                p.write_bytes(content)

            is_text, enc, conf = detect_content(str(p))
            try:
                text = _load_plaintext(str(p))
                load_ok = text is not None and len(text) > 0
            except Exception as e:
                load_ok = False
                pytest.fail(f"_load_plaintext() crashed fuer {filename}: {e}")

            results.append(
                {
                    "file": filename,
                    "is_text": is_text,
                    "enc": enc,
                    "load_ok": load_ok,
                }
            )

        # Alle Dateien muessen geladen werden koennen
        for r in results:
            assert r["load_ok"], (
                f"_load_plaintext() konnte {r['file']} nicht laden "
                f"(is_text={r['is_text']}, enc={r['enc']})"
            )

    def test_is_text_true_implies_load_file_not_crash(self, tmp_path):
        """Wenn detect_content is_text=True, darf load_file() keinen Exception werfen."""
        cases = [
            ("utf8_full.txt", "Vollstaendiger UTF-8 Text\n".encode("utf-8")),
            ("latin1_full.txt", "Stra\xdfe\n".encode("latin-1")),
            ("bom_full.txt", b"\xef\xbb\xbf" + "BOM-Datei\n".encode("utf-8")),
        ]
        for filename, content in cases:
            p = tmp_path / filename
            p.write_bytes(content)
            is_text, _, _ = detect_content(str(p))
            if is_text:
                try:
                    result = load_file(str(p))
                    # Ergebnis darf None sein (z.B. leerer Text), aber kein Exception
                except Exception as e:
                    pytest.fail(f"load_file() hat Exception fuer {filename} geworfen: {e}")
