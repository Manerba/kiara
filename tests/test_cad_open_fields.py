"""Modus open_fields: Feldform-Befunde veröffentlichen, aber sichtbar markieren."""

import pytest

from kiara.cad.adapter import CadReadError, normalize

SOURCE = {"entry": "0", "segment": 10, "object": 4}


def raw(issues):
    return {
        "source_hash": "a" * 64,
        "issues": issues,
        "records": {
            "part": {
                "labels": [
                    {"source": SOURCE, "data": {"form_erkannt": False, "grund": "Konstante",
                                                "label_roh": [{"feld": 5, "art": "B", "wert": 3}]}}
                ]
            }
        },
        "model": {
            "teile": {"part": {"topologie": {"body": 1}}},
            "objekte": [{"typ": "teil", "teil_id": "part", "name": "Neutral", "id": "occ"}],
            "warnungen": [],
            "einheiten": {"laenge": "mm"},
            "lage_konvention": "p_global = p · M + t",
        },
    }


def test_strict_mode_still_rejects_unknown_field_forms():
    with pytest.raises(CadReadError, match="annotation_form_unknown"):
        normalize(raw([{"code": "annotation_form_unknown", "source": SOURCE}]), "n.pkg")


def test_open_fields_publishes_marked_snapshot():
    snap = normalize(
        raw([{"code": "annotation_form_unknown", "source": SOURCE}]), "n.pkg", open_fields=True
    )
    assert snap["verification_complete"] is False
    assert snap["open_findings"] == [{"code": "annotation_form_unknown", "source": SOURCE}]
    (label,) = snap["contents"][0]["labels"]
    assert label["type"] == "unread" and label["text"] == [] and label["reason"] == "Konstante"


def test_open_fields_keeps_structural_findings_blocking():
    issues = [{"code": "annotation_form_unknown", "source": SOURCE},
              {"code": "ambiguous_part_identity", "native_id": "part"}]
    with pytest.raises(CadReadError, match="ambiguous_part_identity"):
        normalize(raw(issues), "n.pkg", open_fields=True)


def test_clean_model_is_verified():
    r = raw([])
    r["records"]["part"]["labels"] = []
    snap = normalize(r, "n.pkg")
    assert snap["verification_complete"] is True and snap["open_findings"] == []


def test_unknown_fields_switch_defaults_to_skip_and_validates():
    from kiara.cad.ingestion import unknown_fields_audit

    assert unknown_fields_audit({}) is False
    assert unknown_fields_audit({"ingestion": {"cad_unknown_fields": "skip"}}) is False
    assert unknown_fields_audit({"ingestion": {"cad_unknown_fields": "audit"}}) is True
    with pytest.raises(ValueError):
        unknown_fields_audit({"ingestion": {"cad_unknown_fields": "egal"}})


def test_audit_publication_has_own_processing_version():
    from kiara.cad.ingestion import _audit_version, _version

    assert _audit_version("a" * 64) != _version("a" * 64)
