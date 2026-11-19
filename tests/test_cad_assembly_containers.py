"""Baugruppentexte, Containerinhalte und nichtendliche Quader im Snapshot (cad-pkg-reverse 9ebaaa3)."""

import json

from kiara.cad import reader_worker as worker
from kiara.cad.adapter import normalize


def _label(text, hidden=None):
    label = {
        "source": {"entry": "0", "segment": 1, "object": 7},
        "data": {
            "form_erkannt": True,
            "annotationstyp": "text",
            "belegt": {"text_liste": [text]},
            "kette": {"inhalt": {"objekt": 3}},
            "felder": {"text_liste": [{"feld": 1}]},
        },
    }
    if hidden is not None:
        label["data"]["belegt"]["individuell_ausgeblendet"] = hidden
        label["data"]["felder"]["individuell_ausgeblendet"] = {"feld": 6}
    return label


def _raw(quader, inventory):
    return {
        "source_hash": "0" * 64,
        "model": {
            "einheiten": {"laenge": "mm"},
            "lage_konvention": "p_global = p · M + t",
            "warnungen": [],
            "teile": {"1:1": {"quader": quader, "topologie": {"body": [1]}}},
            "objekte": [
                {
                    "typ": "baugruppe",
                    "id": "1:9",
                    "name": "BG",
                    "ort": ["0", 10, 1],
                    "struktur": ["0", 20, 2],
                    "struktur_id": "1:8",
                    "kinder": [
                        {"typ": "teil", "id": "1:2", "name": "T", "ort": ["0", 10, 4], "teil_id": "1:1"}
                    ],
                }
            ],
        },
        "records": {
            "1:1": {"sources": [{"entry": "0", "segment": 30, "object": 1}],
                    "labels": [_label("Featuretext", hidden=True)]},
            "1:8": {"labels": [_label("Baugruppentext")], "db_flags": []},
            "1:5": {"labels": [_label("Containertext")], "materials": []},
            "1:7": {"labels": [_label("Verschachtelt")]},
        },
        "issues": [],
        "container_inventory": inventory,
    }


INVENTORY = {
    "schema": 1,
    "mengenrolle": "ungeklaert",
    "zusaetzliche_teilinhalte": ["1:5"],
    "unbekannte_verknuepfungen": [{"grund": "Teilkindklasse_unbelegt"}],
    "beziehungen": [
        {
            "eltern_teil_id": "1:1",
            "container_anker": {"objekt": 11},
            "inhalte": [
                {"zuordnung": "expliziter_verweis", "teil_id": "1:5", "name": "C",
                 "anker": {"objekt": 12}, "inhalt_anker": {"objekt": 13}, "lage_lokal": None},
                {"zuordnung": "ungeklaert", "teil_id": "1:6", "anker": {"objekt": 14}},
            ],
        }
    ],
    "baugruppencontainer": {
        "zusaetzliche_teilinhalte": ["1:7"],
        "beziehungen": [{
            "besitzer_anker": {"objekt": 20},
            "inhalte": [{
                "beziehung": "container_inhaltscontainer", "zuordnung": "expliziter_verweis",
                "inhalt_anker": {"objekt": 21},
                "inhalte": [{"beziehung": "container_teilinhalt", "zuordnung": "expliziter_verweis",
                             "inhalt_id": "1:7", "name": "N", "anker": {"objekt": 22},
                             "inhalt_anker": {"objekt": 23}}],
            }],
        }],
    },
}


def test_assembly_labels_and_container_contents_without_occurrences():
    snap = normalize(_raw({"min": [0.0] * 3, "max": [1.0] * 3}, INVENTORY), "x.pkg")
    by_kind = {}
    for content in snap["contents"]:
        by_kind.setdefault(content["kind"], []).append(content)
    assembly = by_kind["assembly"][0]
    assert [label["text"] for label in assembly["labels"]] == [["Baugruppentext"]]
    assert assembly["data"]["structure_id"] == "1:8"
    container = by_kind["container_part"]
    # Nur explizit verknüpfte Inhalte (direkt und verschachtelt), ohne Vorkommen und Stückzahl.
    assert sorted(c["native_id"] for c in container) == ["1:5", "1:7"]
    nested = next(c for c in container if c["native_id"] == "1:7")
    assert nested["data"]["containers"][0]["container_anchor"] == {"objekt": 21}
    part = next(c for c in snap["contents"] if c["kind"] == "part")
    assert part["labels"][0]["hidden"] is True and part["labels"][0]["hidden_field"] == {"feld": 6}
    direct = next(c for c in container if c["native_id"] == "1:5")
    assert direct["labels"][0]["text"] == ["Containertext"]
    assert direct["data"]["quantity_role"] == "ungeklaert"
    ids = {c["object_id"] for c in container}
    assert all(o["object_id"] not in ids for o in snap["occurrences"])
    assert snap["content_containers"]["unbekannte_verknuepfungen"]
    assert snap["verification_complete"]


def test_non_finite_box_keeps_raw_marker_and_status():
    box = {"min": [float("-inf")] * 3, "max": [float("inf")] * 3}
    raw = json.loads(json.dumps(worker._finite(_raw(box, INVENTORY)), allow_nan=False))
    snap = normalize(raw, "x.pkg")
    part = next(c for c in snap["contents"] if c["kind"] == "part")
    prop = next(p for p in part["properties"] if p["name"] == "quader")
    assert prop["status"] == "not_finite"
    assert prop["value"]["min"][0] == {"nicht_endliche_zahl": "-inf"}
