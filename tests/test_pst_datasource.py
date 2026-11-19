"""PST-Datenquelle: RTF-Entkapselung, Ablage/Verfuegbarkeit, Scanner-Ausgabe."""

import json
import os
from datetime import datetime, timezone

import pytest

from kiara.config import DEFAULT_PATHS
from kiara.connectors import registry
from kiara.connectors.pst_connector import (
    PSTConnector,
    list_pst_files,
    resolve_pst_path,
)
from kiara.ingestion import pst_scanner
from kiara.ingestion.pst_reader import (
    PstAttachment,
    PstItem,
    codepage_to_charset,
    decapsulate_rtf,
    item_kind,
)
from kiara.ingestion.pst_scanner import PSTScanner


@pytest.fixture
def dropzone(tmp_path, monkeypatch):
    zone = tmp_path / "pst"
    zone.mkdir()
    monkeypatch.setitem(DEFAULT_PATHS, "pst_dropzone", str(zone))
    return zone


# --- Reader-Hilfen ---------------------------------------------------------------


def test_decapsulate_rtf_html_keeps_tags_and_drops_htmlrtf():
    rtf = (
        rb"{\rtf1\ansi\ansicpg1252\fromhtml1 {\fonttbl{\f0 Arial;}}"
        rb"{\*\htmltag19 <html><body>}"
        rb"\htmlrtf {\b nur RTF}\htmlrtf0 "
        rb"Gr\'fc\'dfe {\*\htmltag84 &nbsp;}\u8364? 5"
        rb"{\*\htmltag0 \par }"
        rb"{\*\htmltag27 </body></html>}}"
    )
    mime, text = decapsulate_rtf(rtf)
    assert mime == "text/html"
    assert text.startswith("<html><body>")
    assert "nur RTF" not in text
    assert "Grüße &nbsp;€ 5" in text
    assert text.rstrip().endswith("</body></html>")


def test_decapsulate_rtf_plain_text():
    rtf = rb"{\rtf1\ansi\ansicpg1252\fromtext {\fonttbl{\f0 Arial;}}Zeile 1\par Zeile\tab 2\par}"
    mime, text = decapsulate_rtf(rtf)
    assert mime == "text/plain"
    assert text == "Zeile 1\nZeile\t2"


@pytest.mark.parametrize(
    ("codepage", "expected"),
    [(28591, "iso8859-1"), (65001, "utf-8"), (1252, "cp1252"), (None, "utf-8"), (99999, "utf-8")],
)
def test_codepage_to_charset(codepage, expected):
    assert codepage_to_charset(codepage) == expected


@pytest.mark.parametrize(
    ("message_class", "kind"),
    [
        ("IPM.Note", "mail"),
        ("IPM.Note.SMIME", "mail"),
        ("IPM.Schedule.Meeting.Request", "mail"),
        ("REPORT.IPM.Note.NDR", "mail"),
        ("IPM.Appointment", "appointment"),
        ("IPM.Contact", None),
        ("IPM.Task", None),
        ("", None),
    ],
)
def test_item_kind(message_class, kind):
    assert item_kind(message_class) == kind


# --- Ablage und Verfuegbarkeit ------------------------------------------------------


@pytest.mark.parametrize("name", ["", "../x.pst", "a/b.pst", "a\\b.pst", ".x.pst", "x.txt"])
def test_resolve_pst_path_rejects_unsafe_names(dropzone, name):
    assert resolve_pst_path({"pst_file": name}) is None


def test_list_pst_files_and_schema_options(dropzone):
    (dropzone / "b.PST").write_bytes(b"x" * 10)
    (dropzone / "a.pst").write_bytes(b"x")
    (dropzone / "notiz.txt").write_text("x")
    (dropzone / ".versteckt.pst").write_bytes(b"x")
    (dropzone / "ordner.pst").mkdir()

    assert [f["name"] for f in list_pst_files()] == ["a.pst", "b.PST"]
    options = PSTConnector.config_schema()["pst_file"]["options"]
    assert [o["value"] for o in options] == ["", "a.pst", "b.PST"]


def test_list_pst_files_without_dropzone(tmp_path, monkeypatch):
    monkeypatch.setitem(DEFAULT_PATHS, "pst_dropzone", str(tmp_path / "fehlt"))
    assert list_pst_files() == []


def test_source_unavailable_reason(dropzone):
    config = {"pst_file": "fm.pst"}
    assert "nicht mehr in der Ablage" in PSTConnector.source_unavailable_reason(config, {})
    (dropzone / "fm.pst").write_bytes(b"x")
    assert PSTConnector.source_unavailable_reason(config, {}) is None
    assert PSTConnector.source_unavailable_reason({}, {}) == "Keine PST-Datei ausgewählt"


def test_registry_source_unavailable_reason(dropzone):
    row = {"id": 1, "type": "pst", "connector_config": json.dumps({"pst_file": "fm.pst"})}
    assert registry.source_unavailable_reason(row)
    # Admin-Cache liefert connector_config als Dict
    assert registry.source_unavailable_reason({**row, "connector_config": {"pst_file": "fm.pst"}})
    (dropzone / "fm.pst").write_bytes(b"x")
    assert registry.source_unavailable_reason(row) is None
    assert registry.source_unavailable_reason({"id": 2, "type": "local"}) is None
    assert registry.source_unavailable_reason({"id": 3, "type": "gibtsnicht"}) is None


# --- Scanner -------------------------------------------------------------------------


def _item(identifier, subject, *, kind="mail", attachments=(), start=None):
    return PstItem(
        kind=kind,
        identifier=identifier,
        message_class="IPM.Appointment" if kind == "appointment" else "IPM.Note",
        subject=subject,
        sender="Absender <a@example.org>",
        to="Empfaenger <e@example.org>",
        cc="",
        date=datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc),
        body=f"Text {subject}".encode(),
        body_mime="text/plain",
        body_charset="utf-8",
        start=start,
        end=start,
        location="Raum 1" if start else "",
        attachments=list(attachments),
    )


class _FakeFolder:
    def __init__(self, items):
        self.items = items

    @property
    def number_of_sub_messages(self):
        return len(self.items)

    def get_sub_message(self, index):
        return self.items[index]


class _FakePst:
    folders: list = []

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass

    def iter_folders(self):
        return list(self.folders)

    def count_messages(self, folders):
        return sum(f.number_of_sub_messages for _p, f in folders)

    def read_item(self, message, *, with_attachments, max_attachment_bytes):
        return message


@pytest.fixture
def fake_pst(dropzone, monkeypatch):
    pst = dropzone / "test.pst"
    pst.write_bytes(b"pst")
    monkeypatch.setattr(pst_scanner, "PstFile", _FakePst)
    start = datetime(2026, 4, 9, 7, 30, tzinfo=timezone.utc)
    _FakePst.folders = [
        (
            "Posteingang",
            _FakeFolder([_item(1, "Hallo", attachments=[PstAttachment("a.pdf", b"%PDF")])]),
        ),
        ("Kalender", _FakeFolder([_item(2, "Termin X", kind="appointment", start=start)])),
    ]
    return pst


def _files(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _dirs, files in os.walk(root)
        for f in files
        if not f.startswith(".")
    )


def test_scan_writes_mails_appointments_and_attachments(fake_pst, tmp_path):
    out = tmp_path / "cache"
    scanner = PSTScanner({"pst_path": str(fake_pst), "folders": "*"})
    written = scanner.scan(str(out))

    files = _files(out)
    assert len(written) == len(files) == 3
    mail = next(f for f in files if f.endswith("Hallo.txt"))
    text = (out / mail).read_text()
    assert text.startswith("Betreff: Hallo\nVon: Absender <a@example.org>\nAn: Empfaenger")
    assert "Ordner: Posteingang\n---\nText Hallo" in text
    assert any("/attachments/" in f and f.endswith("_1_a.pdf") for f in files)
    termin = (out / next(f for f in files if f.endswith("Termin X.txt"))).read_text()
    assert termin.startswith("Termin: Termin X\nBeginn: ")
    assert "Ort: Raum 1" in termin and "Ordner: Kalender" in termin
    assert scanner.is_current(str(out))


def test_scan_skips_unchanged_and_removes_stale_files(fake_pst, tmp_path):
    out = tmp_path / "cache"
    PSTScanner({"pst_path": str(fake_pst)}).scan(str(out))
    assert PSTScanner({"pst_path": str(fake_pst)}).scan(str(out)) == []

    # Kalender abgewaehlt -> Optionen geaendert, Terminexport verschwindet
    PSTScanner({"pst_path": str(fake_pst), "include_calendar": False}).scan(str(out))
    files = _files(out)
    assert len(files) == 2
    assert not any(f.endswith("Termin X.txt") for f in files)


def test_scan_folder_selection(fake_pst, tmp_path):
    out = tmp_path / "cache"
    PSTScanner({"pst_path": str(fake_pst), "folders": "Kalender"}).scan(str(out))
    files = _files(out)
    assert len(files) == 1
    assert files[0].endswith("Termin X.txt")


def test_scan_cancel(fake_pst, tmp_path):
    with pytest.raises(InterruptedError):
        PSTScanner({"pst_path": str(fake_pst)}).scan(
            str(tmp_path / "cache"), control_callback=lambda: "cancel"
        )


def test_badge_labels_short_names():
    from kiara.connectors.smb_connector import SMBConnector

    assert PSTConnector.badge_label == "PST"
    assert SMBConnector.badge_label == "SMB"
