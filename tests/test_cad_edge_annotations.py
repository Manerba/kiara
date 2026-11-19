"""Verified ordered edge references survive the ingestion boundary."""

import json
from pathlib import Path

import pytest

from kiara.cad import reader_worker as worker
from kiara.cad.adapter import normalize


def segment():
    from objstrom import Token

    data = json.loads((Path(__file__).parent / "data/cad/neutral-edge-text-set.json").read_text())[
        "segment"
    ]
    objects = data.pop("objekte")
    seg = worker.pkgleser.Segment(**data)
    for number, obj in objects.items():
        fields = obj.pop("felder")
        seg.objekte[int(number)] = worker.pkgleser.Objekt(
            **obj, felder=[Token(**field) for field in fields]
        )
    return seg


def test_all_labels_and_both_ordered_references_reach_snapshot():
    seg = segment()
    labels = []
    for number in (4, 104, 204):
        decoded = worker.beschriftungen.decode(seg, seg.objekte[number])
        assert decoded["form_erkannt"]
        labels.append(
            {
                "source": {"entry": seg.eintrag, "segment": seg.start, "object": number},
                "data": decoded,
            }
        )
    raw = {
        "source_hash": "a" * 64,
        "issues": [],
        "records": {"part": {"labels": labels}},
        "model": {
            "teile": {"part": {"topologie": {"body": 1}}},
            "objekte": [{"typ": "teil", "teil_id": "part", "name": "Neutral", "id": "occ"}],
            "warnungen": [],
            "einheiten": {"laenge": "mm"},
            "lage_konvention": "p_global = p · M + t",
        },
    }
    result = normalize(raw, "neutral.pkg")["contents"][0]["labels"]
    assert [item["text"] for item in result] == [
        ["NEUTRAL TEXT 1"],
        ["NEUTRAL TEXT 2"],
        ["NEUTRAL TEXT 3"],
    ]
    assert [item["reference_graph"]["referenzanzahl"] for item in result] == [1, 1, 2]
    references = result[2]["reference_graph"]["referenzen"]
    assert [{k: r[k] for k in ("referenz", "element", "klasse")} for r in references] == [
        {"referenz": 212, "element": 214, "klasse": "edge"},
        {"referenz": 400, "element": 401, "klasse": "edge"},
    ]
    # Seit Reader b709d3a: Referenzposition und normierter Parameter je Referenz.
    assert all(
        {"position_lokal_mm", "parameter_normiert", "parameter_geometrisch_geprueft"} <= set(r)
        for r in references
    )
    assert len(result[2]["reference_fields"]["referenzen"]) == 2
    assert result[2]["source"]["object"] == 204


def test_bad_second_reference_invalidates_every_label_in_shared_set():
    seg = segment()
    # The second reference must point at an edge owned by this part.
    seg.objekte[401].klasse = "vertex"
    for number in (4, 104, 204):
        assert not worker.beschriftungen.decode(seg, seg.objekte[number])["form_erkannt"]


@pytest.mark.parametrize(
    "reference_fields",
    [{}, {"referenzanzahl": {"offset": 12}}, {"referenzen": [{"offset": 14}]}],
)
def test_feature_text_keeps_graph_without_assuming_edge_reference_fields(reference_fields):
    graph = {"cf_ref": 12, "feature_flaechen": [21, 22], "set_beschriftungen": 1}
    source = {"entry": "0", "segment": 100, "object": 4}
    text_fields = [{"offset": 8}]
    raw = {
        "source_hash": "a" * 64,
        "issues": [],
        "records": {
            "part": {
                "labels": [
                    {
                        "source": source,
                        "data": {
                            "annotationstyp": "feature_text",
                            "belegt": {"text_liste": ["Neutral feature"]},
                            "kette": {"inhalt": 9},
                            "felder": {"text_liste": text_fields, **reference_fields},
                            "referenzgraph": graph,
                        },
                    }
                ],
            },
        },
        "model": {
            "teile": {"part": {"topologie": {"body": 1}}},
            "objekte": [{"typ": "teil", "teil_id": "part", "name": "Neutral", "id": "occ"}],
            "warnungen": [],
            "einheiten": {"laenge": "mm"},
            "lage_konvention": "p_global = p · M + t",
        },
    }
    snapshot = normalize(raw, "neutral.pkg")
    assert snapshot["contents"][0]["labels"] == [
        {
            "type": "feature_text",
            "text": ["Neutral feature"],
            "source": source,
            "owner": 9,
            "fields": text_fields,
            "reference_graph": graph,
            "reference_fields": reference_fields,
        }
    ]
    assert snapshot["structure_complete"]
