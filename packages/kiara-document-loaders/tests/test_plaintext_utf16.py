"""UTF-16/UTF-32-Textdateien duerfen nicht als Binaerdaten enden."""

import codecs

import pytest
from kiara_document_loaders import _load_plaintext

TEXT = "Treiber-Lizenz Größe 42 µm – Zeile 1\r\nZeile 2: ÄÖÜ äöü ß\r\n" * 20
GREEK = "Άδεια χρήσης λογισμικού\r\n" * 20
# Treiber-/Manifest-Dateien sind meist reines ASCII; charset_normalizer rät hier falsch.
ASCII = '<?xml version="1.0"?>\r\n<Package name="driver" version="1.0.2"/>\r\n' * 30


def _write(tmp_path, data: bytes):
    path = tmp_path / "datei.txt"
    path.write_bytes(data)
    return str(path)


@pytest.mark.parametrize(
    ("bom", "encoding"),
    [
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
    ],
)
def test_bom_files_are_decoded(tmp_path, bom, encoding):
    text = _load_plaintext(_write(tmp_path, bom + TEXT.encode(encoding)))

    assert text == TEXT
    assert "\x00" not in text
    assert not text.startswith("﻿")


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_utf16_without_bom_is_decoded(tmp_path, encoding):
    text = _load_plaintext(_write(tmp_path, TEXT.encode(encoding)))

    assert text == TEXT


@pytest.mark.parametrize(
    ("bom", "encoding"),
    [
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (b"", "utf-16-le"),
        (b"", "utf-16-be"),
    ],
)
def test_ascii_only_utf16_is_decoded(tmp_path, bom, encoding):
    text = _load_plaintext(_write(tmp_path, bom + ASCII.encode(encoding)))

    assert text == ASCII


def test_utf16_greek_with_bom(tmp_path):
    data = codecs.BOM_UTF16_LE + GREEK.encode("utf-16-le")

    assert _load_plaintext(_write(tmp_path, data)) == GREEK


def test_utf16_with_odd_trailing_byte_is_decoded(tmp_path):
    data = codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le") + b"\x0a"

    text = _load_plaintext(_write(tmp_path, data))

    assert text == TEXT


def test_utf16_html_and_xml_content(tmp_path):
    xml = '<?xml version="1.0" encoding="UTF-16"?>\r\n<a><b>Größe</b></a>\r\n'

    text = _load_plaintext(_write(tmp_path, codecs.BOM_UTF16_LE + xml.encode("utf-16-le")))

    assert text == xml


def test_real_binary_with_nuls_is_not_decoded_as_utf16(tmp_path):
    # Druckerdaten: NUL und ESC-Sequenzen an beiden Byte-Positionen
    data = b"\x00\x00\x00\x00\x00" + b"   \x1b$\x00\x00\x1bc\x1b[4w\r\n" * 50

    text = _load_plaintext(_write(tmp_path, data))

    assert text.count("\x00") / len(text) > 0.05


def test_utf8_and_cp1252_unchanged(tmp_path):
    assert _load_plaintext(_write(tmp_path, "Größe\n".encode("utf-8"))) == "Größe\n"
    assert _load_plaintext(_write(tmp_path, "Preis 5 € – netto\n".encode("cp1252"))) == (
        "Preis 5 € – netto\n"
    )
    assert _load_plaintext(_write(tmp_path, codecs.BOM_UTF8 + b"abc")) == "abc"
    assert _load_plaintext(_write(tmp_path, b"")) == ""
