"""Lazy extraction must preserve the eager reader's content and gap detection."""

import zipfile
from pathlib import Path

import pytest

from kiara.cad import reader_worker as worker
from kiara.cad.adapter import CadReadError, load_pkg

FIXTURE = Path(__file__).parent / "data/cad/neutral-plate.pkg"


def eager(path):
    reader = worker.pkgleser.Leser(mit_punkten=False)
    model = reader.lesen(path)
    records, issues = {}, []
    for identity, segments in reader.index.items():
        record, findings = worker._record(segments, reader.index)
        records[identity] = record
        issues.extend(findings)
        if len(record["sources"]) > 1:
            issues.append({"code": "ambiguous_part_identity", "native_id": identity})
        if (record["materials"] or record["labels"]) and identity not in model["teile"]:
            issues.append({"code": "unmapped_content_metadata", "native_id": identity})
    return dict(source_hash=worker._digest(path), model=model, records=records, issues=issues)


def _without_structure_ids(nodes):
    for node in nodes:
        node.pop("struktur_id", None)
        _without_structure_ids(node.get("kinder", []))


def lazy(path):
    raw = worker.extract(path)
    # Kiara-Zusätze gegenüber dem Vendor-Leser getrennt prüfen.
    assert raw["container_inventory"]["mengenrolle"] == "ungeklaert"
    _without_structure_ids(raw["model"]["objekte"])
    skip = {"identical_copies", "container_inventory"}
    return {k: v for k, v in raw.items() if k not in skip}, raw["identical_copies"]


def test_complete_multistream_content_equals_eager_reader():
    # Three object streams, shared IDs across segments, parts and assembly roots.
    expected = eager(FIXTURE)
    assert len(expected["model"]["objekte"]) == 3
    assert not expected["issues"]
    assert lazy(FIXTURE) == (expected, 0)


def test_unused_unknown_annotation_is_still_examined(tmp_path):
    from objstrom import Token, kodieren

    path = tmp_path / "unused-label.pkg"
    tokens = [
        ("q", None),
        ("8021", None),
        ("D", 1),
        ("s", (1, "part_struct")),
        ("z", "part_struct"),
        ("ID", "7:ffff"),
        ("8022", None),
        ("s", (2, "A3D_LBL_TAG")),
        ("t", (18, "A3D_LBL_TAG")),
        ("u", None),
        ("r", None),
    ]
    unused = kodieren([Token(0, kind, value) for kind, value in tokens])
    with zipfile.ZipFile(FIXTURE) as source, zipfile.ZipFile(path, "w") as target:
        for info in source.infolist():
            data = source.read(info)
            if info.filename == "0":
                pos = worker.pkgleser.segmentkoepfe(data)[-1][0]
                data = data[:pos] + unused + data[pos:]
            target.writestr(info.filename, data)
    raw, copies = lazy(path)
    assert (raw, copies) == (eager(path), 0)
    assert any(x["code"] == "annotation_form_unknown" for x in raw["issues"])
    with pytest.raises(CadReadError, match="annotation_form_unknown"):
        load_pkg(path)


def test_byte_identical_stream_copies_are_one_content(tmp_path):
    path = tmp_path / "duplicate.pkg"
    with zipfile.ZipFile(FIXTURE) as source, zipfile.ZipFile(path, "w") as target:
        for info in source.infolist():
            target.writestr(info.filename, source.read(info))
        with pytest.warns(UserWarning, match="Duplicate name"):
            target.writestr("2", source.read("2"))
    raw, copies = lazy(path)
    expected = eager(path)
    assert raw["model"] == expected["model"]
    assert raw["model"]["warnungen"]
    # Gleiche IDs als byte-gleiche Kopie in zwei Strömen: ein Inhalt, keine Mehrdeutigkeit.
    assert copies > 0
    assert any(x["code"] == "ambiguous_part_identity" for x in expected["issues"])
    assert not any(x["code"] == "ambiguous_part_identity" for x in raw["issues"])
    assert all(len(r["sources"]) <= 1 for r in raw["records"].values())


class _Directory:
    def __init__(self, data, heads):
        self.b, self.koepfe = data, heads


class _Segment:
    def __init__(self, start):
        self.start = start


def test_only_byte_identical_segments_are_collapsed():
    index = worker._SegmentIndex()
    a = _Directory(b"xxABCDyy", [(2, 6, "part_struct", "1:a")])
    b = _Directory(b"ABCD", [(0, 4, "part_struct", "1:a")])
    c = _Directory(b"ABCE", [(0, 4, "part_struct", "1:a")])
    for d in (a, b, c):
        for start, end, _cls, _sid in d.koepfe:
            index.ends[(id(d), start)] = end
    sa, sb, sc = _Segment(2), _Segment(0), _Segment(0)
    assert index._distinct([(a, sa), (b, sb)]) == [sa]
    assert index._distinct([(a, sa), (b, sb), (c, sc)]) == [sa, sc]
