"""Inhaltsrouting mit echten Dokumenten und ohne GPU/Produktivdatenbank."""

import io
import struct
import zipfile
from unittest.mock import Mock

import httpx
import pytest
from kiara_document_loaders.content_detect import detect_file_format
from kiara_ingestion_core.models import FileLogEntry, ProcessingContext
from PIL import Image

from kiara.ingestion import ddu_client, loader, runtime

BACKEND = {"url": "http://ddu.test:5001", "timeout": 2}


def _http(monkeypatch, status=200, body=None):
    post = Mock(
        return_value=httpx.Response(
            status,
            json=body or {"document": {"md_content": "Gelesener Inhalt"}},
            request=httpx.Request("POST", BACKEND["url"] + "/v1/convert/file"),
        )
    )
    monkeypatch.setattr(ddu_client.httpx, "post", post)
    return post


def _process(path, **options):
    log = FileLogEntry(str(path), path.name, path.stat().st_size, path.suffix)
    stats = {"errors": 0}
    result = runtime._process_single_file(
        str(path),
        {"ingestion": {"chunk_size": 100, "chunk_overlap": 0}},
        {},
        1,
        ProcessingContext(**options),
        stats,
        file_log=log,
    )
    return result, log, stats


@pytest.mark.parametrize("name", ["tabelle.xls", "tabelle.txt", "tabelle.xlsx"])
def test_real_xlsx_is_read_independently_of_filename(tmp_path, name):
    from openpyxl import Workbook

    book = Workbook()
    book.active.append(["Artikel", "Menge"])
    book.active.append(["Schraube", 42])
    data = io.BytesIO()
    book.save(data)
    path = tmp_path / name
    path.write_bytes(data.getvalue())
    result = loader.load_file(str(path))
    assert "Schraube | 42" in result["text"]
    assert result["metadata"]["file_type"] == ".xlsx"
    assert result["metadata"]["source_extension"] == path.suffix
    assert path.read_bytes() == data.getvalue()


def test_docx_named_xlsx_uses_word_loader(tmp_path):
    from docx import Document

    document = Document()
    document.add_paragraph("Technische Beschreibung")
    path = tmp_path / "dokument.xlsx"
    document.save(path)
    result = loader.load_file(str(path))
    assert result["text"] == "Technische Beschreibung"
    assert result["metadata"]["detected_format"] == ".docx"


def test_html_named_odt_is_converted_without_docling(tmp_path, monkeypatch):
    path = tmp_path / "anleitung.odt"
    path.write_text("<!DOCTYPE html><html><body><h1>Anleitung</h1><p>Inhalt</p></body></html>")
    post = _http(monkeypatch)
    result = loader.load_file(str(path), ddu_enabled=True, ddu_backends=[BACKEND])
    assert "# Anleitung" in result["text"]
    assert result["metadata"]["file_type"] == ".html"
    post.assert_not_called()


@pytest.mark.parametrize("codec", ["utf-8-sig", "utf-16", "utf-32"])
def test_wide_html_with_wrong_extension(tmp_path, codec):
    path = tmp_path / "text.xlsx"
    path.write_bytes("<html><body><p>Größe</p></body></html>".encode(codec))
    result = loader.load_file(str(path))
    assert "Größe" in result["text"]
    assert result["metadata"]["file_type"] == ".html"


def test_xml_without_declaration_is_identified(tmp_path):
    path = tmp_path / "dokument.odt"
    path.write_text('<document value="42"/>')
    assert detect_file_format(str(path)).extension == ".xml"


@pytest.mark.parametrize(
    "fragment",
    [
        "<title>Titel</title><p>Text</p>",
        '<meta charset="utf-8"><span>Text</span>',
        "<ul><li>Punkt</li></ul>",
    ],
)
def test_html_fragment_with_html_extension_stays_html(tmp_path, fragment):
    path = tmp_path / "seite.html"
    path.write_text(fragment)
    assert detect_file_format(str(path)).extension == ".html"


def test_xml_declaration_wins_over_html_extension(tmp_path):
    path = tmp_path / "feed.html"
    path.write_text('<?xml version="1.0"?><rss version="2.0"/>')
    assert detect_file_format(str(path)).extension == ".xml"


def test_utf8_sample_boundary_does_not_turn_text_into_binary(tmp_path):
    path = tmp_path / "text.odt"
    path.write_bytes(b"\xef\xbb\xbf" + b"a" * 8186 + "Größe".encode())
    result = loader.load_file(str(path))
    assert result["text"].endswith("Größe")
    assert result["metadata"]["detected_format"] == ".txt"


def test_native_odt_is_identified_and_loaded(tmp_path):
    from odf.opendocument import OpenDocumentText
    from odf.text import P

    document = OpenDocumentText()
    document.text.addElement(P(text="Betriebsanleitung"))
    path = tmp_path / "anleitung.txt"
    document.save(str(path), addsuffix=False)
    result = loader.load_file(str(path))
    assert result["text"] == "Betriebsanleitung"
    assert result["metadata"]["detected_format"] == ".odt"


def test_xlsx_failure_closes_file_handle(tmp_path, monkeypatch):
    import builtins

    from kiara_document_loaders.general import _load_xlsx

    path = tmp_path / "defekt.xlsx"
    path.write_bytes(b"PK\x03\x04abgeschnitten")
    stream = builtins.open(path, "rb")
    monkeypatch.setattr(
        "kiara_document_loaders.general.open", lambda *a, **k: stream, raising=False
    )
    with pytest.raises(zipfile.BadZipFile):
        _load_xlsx(str(path))
    assert stream.closed


def test_pdf_upload_uses_detected_mime_and_suffix(tmp_path, monkeypatch):
    path = tmp_path / "dokument.png"
    original = b"%PDF-1.4\nsynthetic PDF header, HTTP mocked"
    path.write_bytes(original)
    post = _http(monkeypatch)
    result = loader.load_file(
        str(path), ddu_enabled=True, ddu_backends=[BACKEND], ddu_ocr_images=False
    )
    assert result["text"] == "Gelesener Inhalt"
    filename, _, mime = post.call_args.kwargs["files"][0][1]
    assert filename == "dokument.pdf"
    assert mime == "application/pdf"
    assert path.read_bytes() == original


def test_tiny_image_named_pdf_obeys_image_preflight(tmp_path, monkeypatch):
    path = tmp_path / "bild.pdf"
    Image.new("RGB", (1, 1), "white").save(path, format="PNG")
    post = _http(monkeypatch)
    _, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert log.status == "skipped" and not log.empty_content_confirmed
    assert stats["ddu_images_skipped"] == 1
    post.assert_not_called()


def test_ocr_disabled_applies_to_detected_image(tmp_path, monkeypatch):
    path = tmp_path / "bild.pdf"
    Image.new("RGB", (8, 8), "white").save(path, format="PNG")
    post = _http(monkeypatch)
    loader.load_file(str(path), ddu_enabled=True, ddu_backends=[BACKEND], ddu_ocr_images=False)
    post.assert_not_called()


def test_unsupported_archive_skips_without_empty_completion(tmp_path):
    path = tmp_path / "archiv.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inhalt.txt", "Archiv ist kein Word-Dokument")
    _, log, stats = _process(path)
    assert log.status == "skipped" and "ZIP" in log.error_message
    assert not log.empty_content_confirmed
    assert stats == {"errors": 0, "unsupported_formats_skipped": 1}


def test_broken_container_is_error_with_reason_not_skip(tmp_path):
    path = tmp_path / "defekt.odt"
    path.write_bytes(b"PK\x03\x04abgeschnitten")
    _, log, stats = _process(path)
    assert log.status == "error" and "zip" in log.error_message.lower()
    assert not log.empty_content_confirmed
    assert stats["errors"] == 1


def test_loader_exception_reaches_file_log(tmp_path):
    path = tmp_path / "defekt.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        archive.writestr("word/document.xml", "<document/>")
    _, log, stats = _process(path)
    assert log.status == "error"
    assert "DOCX-Loader" in log.error_message and "KeyError" in log.error_message
    assert stats["errors"] == 1


def test_fallback_failure_preserves_loader_and_backend_cause(tmp_path, monkeypatch):
    path = tmp_path / "defekt.pdf"
    path.write_bytes(b"%PDF-1.4\nabgeschnitten")
    post = _http(monkeypatch, 503, {"detail": "backend unavailable"})
    _, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert log.status == "error" and log.is_infra_error
    assert "PDF-Loader" in log.error_message
    assert "HTTP 503" in log.error_message
    assert stats["errors"] == 1
    post.assert_called_once()


def test_task_missing_404_stays_document_error(tmp_path, monkeypatch):
    path = tmp_path / "dokument.pdf"
    path.write_bytes(b"%PDF-1.4\nplaceholder")
    _http(
        monkeypatch, 404, {"detail": "Task result not found. Please wait for a completion status."}
    )
    _, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert log.status == "error" and not log.is_infra_error
    assert stats["ddu_document_errors"] == 1


def test_unknown_binary_never_becomes_plaintext(tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes(b"\x00\xff\x00\x01" * 100)
    _, log, stats = _process(path)
    assert log.status == "error" and not log.empty_content_confirmed
    assert "Binaeres Dateiformat" in log.error_message
    assert stats["errors"] == 1


def _ole_container(*stream_names):
    """Minimaler gueltiger CFB-Container, nur fuer die Stream-Erkennung."""
    free, end, fat = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD
    header = bytearray(512)
    header[:8] = bytes.fromhex("d0cf11e0a1b11ae1")
    struct.pack_into("<HHHHH", header, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<IIIIIIIII", header, 40, 0, 1, 1, 0, 4096, end, 0, end, 0)
    struct.pack_into("<109I", header, 76, 0, *([free] * 108))
    fat_sector = struct.pack("<128I", fat, end, *([free] * 126))

    def entry(name, kind, child=free, right=free):
        data = bytearray(128)
        encoded = (name + "\0").encode("utf-16-le")
        data[: len(encoded)] = encoded
        struct.pack_into("<HBBIII", data, 64, len(encoded), kind, 1, free, right, child)
        struct.pack_into("<IQ", data, 116, end, 0)
        return data

    assert 1 <= len(stream_names) <= 3
    directory = entry("Root Entry", 5, 1)
    for index, name in enumerate(stream_names, 1):
        directory += entry(name, 2, right=index + 1 if index < len(stream_names) else free)
    directory += bytes(512 - len(directory))
    return header + fat_sector + directory


def test_encrypted_office_is_not_misidentified_as_legacy_xls(tmp_path, monkeypatch):
    pytest.importorskip("olefile")
    path = tmp_path / "Liste1.xlsx"
    path.write_bytes(_ole_container("EncryptionInfo", "EncryptedPackage"))
    post = _http(monkeypatch)
    assert detect_file_format(str(path)).extension == ".encrypted-office"
    _, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert log.status == "skipped" and not log.empty_content_confirmed
    assert "Verschluesseltes Office-Dokument" in log.error_message
    assert "Kennwort" in log.error_message
    assert stats["errors"] == 0
    post.assert_not_called()


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ("Workbook", ".xls"),
        ("Book", ".xls"),
        ("WordDocument", ".doc"),
        ("PowerPoint Document", ".ppt"),
    ],
)
def test_ole_directory_controls_format(tmp_path, stream, expected):
    pytest.importorskip("olefile", reason="Neue Laufzeitabhaengigkeit erforderlich")
    path = tmp_path / "falscher-name.xlsx"
    path.write_bytes(_ole_container(stream))
    assert detect_file_format(str(path)).extension == expected


@pytest.mark.parametrize("stream", ["WordDocument", "PowerPoint Document"])
def test_legacy_office_is_skipped_with_visible_reason(tmp_path, monkeypatch, stream):
    pytest.importorskip("olefile", reason="Neue Laufzeitabhaengigkeit erforderlich")
    path = tmp_path / "alt.xlsx"
    path.write_bytes(_ole_container(stream))
    post = _http(monkeypatch)
    _, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert log.status == "skipped" and ".xlsx" in log.error_message
    assert not log.empty_content_confirmed
    post.assert_not_called()


def test_biff2_xls_named_xlsx_uses_real_xlrd(tmp_path, monkeypatch):
    pytest.importorskip("xlrd", reason="Neue Laufzeitabhaengigkeit erforderlich")

    def record(op, data):
        return struct.pack("<HH", op, len(data)) + data

    data = record(0x0009, struct.pack("<HH", 2, 0x10))
    data += record(0x0004, struct.pack("<HH", 0, 0) + bytes(3) + b"\x07Artikel")
    data += record(0x0003, struct.pack("<HH", 1, 0) + bytes(3) + struct.pack("<d", 42))
    data += record(0x000A, b"")
    path = tmp_path / "Liste1.xlsx"
    path.write_bytes(data)
    post = _http(monkeypatch)
    result = loader.load_file(str(path), ddu_enabled=True, ddu_backends=[BACKEND])
    assert "Artikel" in result["text"] and "42" in result["text"]
    assert result["metadata"]["detected_format"] == ".xls"
    post.assert_not_called()


# Hausaenderung 18.11.2026: Inhaltserkennung ohne Fehlgriffe, Standard-Loader nach Docling-Absage


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("bmw.txt", b"BMW-Teileliste\nZeile 2\n"),
        ("notiz.txt", b"P1 Projektliste\n# Kommentar\n"),
        ("p6.txt", b"P6\nkein Bild\n"),
    ],
)
def test_text_resembling_image_headers_stays_text(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    detected = detect_file_format(str(path))
    assert detected.extension == ".txt"
    result, log, stats = _process(path)
    assert log.status == "success" and stats["errors"] == 0
    assert result[0] and content.decode().split()[0] in result[0][0]


def test_real_bmp_is_still_detected(tmp_path):
    path = tmp_path / "bild.dat"
    Image.new("RGB", (4, 4)).save(path, format="BMP")
    assert detect_file_format(str(path)).extension == ".bmp"


def _text_pdf(path, text):
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, text)
    pdf.save()


def test_docling_document_error_falls_back_to_standard_loader(tmp_path, monkeypatch):
    path = tmp_path / "plan.pdf"
    _text_pdf(path, "Leistungsbeschreibung Gebaeudeautomation")
    post = _http(
        monkeypatch, 404, {"detail": "Task result not found. Please wait for a completion status."}
    )
    result, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND, BACKEND])
    post.assert_called_once()
    assert log.status == "success" and log.parser_fallback
    assert log.error_phase == "ddu_fallback" and "HTTP 404" in log.error_message
    assert stats["errors"] == 0 and stats["ddu_fallback_files"] == 1
    assert any("Gebaeudeautomation" in text for text in result[0])
    assert all("ddu_error" not in meta for meta in result[1])


def test_docling_infrastructure_error_still_fails_file(tmp_path, monkeypatch):
    path = tmp_path / "plan.pdf"
    _text_pdf(path, "Leistungsbeschreibung")
    _http(monkeypatch, 503, {"detail": "backend unavailable"})
    _, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert log.status == "error" and log.is_infra_error and stats["errors"] == 1
