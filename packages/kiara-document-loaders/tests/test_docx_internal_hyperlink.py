"""DOCX mit Sprungmarken-Relationship (Target="#...") ohne TargetMode="External"."""

import io
import zipfile

import pytest

pytest.importorskip("docx", reason="python-docx nicht installiert")

from docx import Document
from kiara_document_loaders import _load_docx
from kiara_document_loaders.general import _repair_internal_hyperlink_targets

HYPERLINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
RELS = "word/_rels/document.xml.rels"


def _make_docx(tmp_path, extra_relationships: str = "") -> str:
    doc = Document()
    doc.add_paragraph("Backupplan")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "VM"
    table.rows[0].cells[1].text = "Parity"
    buf = io.BytesIO()
    doc.save(buf)

    out = tmp_path / "plan.docx"
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            blob = src.read(info.filename)
            if info.filename == RELS:
                blob = blob.replace(
                    b"</Relationships>", extra_relationships.encode() + b"</Relationships>"
                )
            dst.writestr(info, blob)
    return str(out)


def test_internal_anchor_relationship_is_loaded(tmp_path):
    path = _make_docx(
        tmp_path,
        f'<Relationship Id="rId99" Type="{HYPERLINK}" Target="#Wiederherstellung eines PVE-Host|outline"/>',
    )

    text = _load_docx(path)

    assert "Backupplan" in text
    assert "VM | Parity" in text


def test_external_hyperlink_is_left_alone(tmp_path):
    rel = (
        f'<Relationship Id="rId98" Type="{HYPERLINK}" '
        'Target="https://example.org/doku" TargetMode="External"/>'
    )
    path = _make_docx(tmp_path, rel)
    data = open(path, "rb").read()

    repaired = _repair_internal_hyperlink_targets(data)

    with zipfile.ZipFile(io.BytesIO(repaired)) as z:
        rels = z.read(RELS)
    assert b'Target="https://example.org/doku" TargetMode="External"' in rels
    assert rels.count(b'TargetMode="External"') == 1


def test_repair_marks_only_unmarked_anchor_targets(tmp_path):
    rels = (
        f'<Relationship Id="rId97" Type="{HYPERLINK}" Target="#a"/>'
        f'<Relationship Id="rId96" Type="{HYPERLINK}" Target="#b" TargetMode="External"/>'
    )
    path = _make_docx(tmp_path, rels)

    repaired = _repair_internal_hyperlink_targets(open(path, "rb").read())

    with zipfile.ZipFile(io.BytesIO(repaired)) as z:
        content = z.read(RELS)
    assert b'Target="#a" TargetMode="External"/>' in content
    assert content.count(b'TargetMode="External"') == 2


def test_regular_docx_is_unchanged(tmp_path):
    path = _make_docx(tmp_path)

    assert "Backupplan" in _load_docx(path)


def test_unrelated_keyerror_is_still_raised(tmp_path):
    path = _make_docx(
        tmp_path,
        f'<Relationship Id="rId95" Type="{HYPERLINK.replace("hyperlink", "image")}" Target="media/fehlt.png"/>',
    )

    with pytest.raises(KeyError):
        _load_docx(path)
