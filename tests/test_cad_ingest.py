"""Independent quantities, publication, permissions and actual PKG extraction."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.cad import repository
from kiara.cad.adapter import CadReadError, chunks, document, normalize
from kiara.ingestion.processing_revision import processing_file_hash, stamp_document_versions


def _raw():
    part = {"typ": "teil", "name": "Schraube", "teil_id": "p", "id": "p1", "ort": ["0", 20, 1]}
    child = {
        "typ": "baugruppe",
        "name": "Modul",
        "id": "a1",
        "ort": ["0", 10, 1],
        "struktur": ["0", 30, 1],
        "kinder": [deepcopy(part), deepcopy(part)],
    }
    # Two occurrences of one shared subassembly; native child IDs repeat.
    root = {
        "typ": "baugruppe",
        "name": "Maschine",
        "id": "root",
        "ort": ["0", 1, 1],
        "struktur": ["0", 1, 2],
        "kinder": [deepcopy(child), deepcopy(child), deepcopy(part)],
    }
    wp = {
        "typ": "arbeitsebene",
        "name": "Zeichnung",
        "id": "wp",
        "ort": ["0", 90, 1],
        "kinder": [deepcopy(part)],
    }
    return {
        "source_hash": "a" * 64,
        "issues": [],
        "records": {"p": {"sources": [{"entry": "0", "segment": 5, "object": 1}]}},
        "model": {
            "teile": {
                "p": {
                    "topologie": {"body": 1},
                    "dichte_g_cm3": 7.85,
                    "attribute": {"Produkt": {"werte": [["Sachnummer", "0012"]]}},
                }
            },
            "objekte": [root, wp],
            "warnungen": [],
            "einheiten": {"laenge": "mm"},
            "lage_konvention": "p_global = p · M + t",
        },
    }


@pytest.fixture
def snapshot():
    return normalize(_raw(), "neutral.pkg")


def test_real_pkg_child_reader_and_loader():
    from kiara.ingestion.loader import load_file

    path = Path(__file__).parent / "data/cad/neutral-plate.pkg"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = load_file(
        str(path), ddu_enabled=True, ddu_backends=[{"url": "http://must-not-be-called.invalid"}]
    )
    snapshot = doc["cad_snapshot"]
    assert snapshot["structure_complete"]
    assert sum(c["has_body"] for c in snapshot["contents"]) == 1
    assert snapshot["source_hash"] == before == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "eckpunkte" not in json.dumps(snapshot)


def test_shared_children_have_distinct_paths_and_one_content(snapshot):
    parts = [o for o in snapshot["occurrences"] if o["physical"] and o["kind"] == "part"]
    assert len(parts) == 5  # two modules * two screws + one direct screw
    assert len({p["occurrence_id"] for p in parts}) == 5
    assert len({p["object_id"] for p in parts}) == 1
    assert len({p["parent_id"] for p in parts}) == 3
    doc = document(snapshot, {"source": "/neutral.pkg", "file_type": ".pkg"})
    assert len([s for s in doc["cad_sections"] if s["kind"] == "part"]) == 1
    assert "0012" in doc["text"]
    result = chunks(doc, 100)
    assert all(doc["text"][c["char_start"] : c["char_end"]] == c["text"] for c in result)
    assert all(
        any(s["start"] <= c["char_start"] < c["char_end"] <= s["end"] for s in doc["cad_sections"])
        for c in result
    )


def test_new_geometry_or_reader_revision_changes_publication(snapshot, monkeypatch):
    from kiara import cad

    def version(s):
        cs = chunks(document(s, {"source": "/neutral.pkg"}), 1000)
        stamp_document_versions([c["text"] for c in cs], [c["metadata"] for c in cs])
        return cs[0]["metadata"]["processing_version"]

    changed = deepcopy(snapshot)
    changed["source_hash"] = "b" * 64
    assert version(changed) != version(snapshot)
    config = {"ingestion": {"chunk_size": 1000}}
    old = processing_file_hash(config, "x.pkg", "a" * 64)
    monkeypatch.setattr(cad, "ADAPTER_REVISION", "new-reader")
    assert processing_file_hash(config, "x.pkg", "a" * 64) != old
    assert processing_file_hash(config, "x.pdf", "a" * 64) == "a" * 64


def test_unknown_relevant_shape_stops_extraction():
    raw = _raw()
    raw["issues"] = [{"code": "material_form_unknown", "source": {"object": 23}}]
    with pytest.raises(CadReadError, match="ungeklärte"):
        normalize(raw, "unknown.pkg")


@pytest.fixture
def stored(test_db, snapshot):
    repository.migrate(test_db)
    ds = test_db.execute(
        "INSERT INTO datasources(name,type,path) VALUES('CAD test','local','/cad') RETURNING id"
    ).fetchone()["id"]
    test_db.execute(
        "INSERT INTO file_hashes(datasource_id,embedding_model,file_path,file_hash) VALUES(%s,'test-model','/cad/neutral.pkg','h')",
        (ds,),
    )
    test_db.execute(
        "INSERT INTO document_publications(collection_name,source,generation,versions,completed_at) VALUES('kb_test-model','/cad/neutral.pkg','g',%s::jsonb,now())",
        (json.dumps({"cad-pkg": "v1"}),),
    )
    repository.publish(test_db, "kb_test-model", "/cad/neutral.pkg", "test-model", "v1", snapshot)
    test_db.commit()
    model_id = test_db.execute(
        "SELECT id FROM cad_models WHERE collection_name='kb_test-model'"
    ).fetchone()["id"]
    return {"db": test_db, "ds": ds, "model_id": model_id, "snapshot": snapshot}


def _query(case, **params):
    return repository.query(
        params, allowed_datasource_ids=[case["ds"]], embedding_model="test-model"
    )


def test_exact_queries_shared_bom_numeric_filters_and_acl(stored):
    found = _query(stored, operation="search", property_name="Sachnummer", property_value="0012")
    assert len(found["items"]) == 1
    assert not _query(stored, operation="search", property_name="Sachnummer", property_value="12")[
        "items"
    ]
    obj = found["items"][0]["object_id"]
    assert (
        len(
            _query(stored, operation="where_used", model_id=stored["model_id"], object_id=obj)[
                "items"
            ]
        )
        == 5
    )
    bom = _query(stored, operation="bom", model_id=stored["model_id"])
    assert [(r["object_id"], r["quantity"]) for r in bom["items"]] == [(obj, 5)]
    assert _query(stored, operation="search", numeric_field="dichte_g_cm3", minimum=7)["items"]
    assert not _query(stored, operation="search", numeric_field="dichte_g_cm3", minimum=8)["items"]
    assert (
        repository.query(
            {"operation": "search"}, allowed_datasource_ids=[], embedding_model="test-model"
        )["items"]
        == []
    )
    with pytest.raises(LookupError):
        repository.query(
            {"operation": "bom", "model_id": stored["model_id"]},
            allowed_datasource_ids=[stored["ds"] + 1000],
            embedding_model="test-model",
        )
    # SQL visibility is independent of the selected embedding model.
    assert repository.query(
        {"operation": "search"},
        allowed_datasource_ids=[stored["ds"]],
        embedding_model="other-model",
    )["items"]


def test_pending_failed_and_replaced_snapshots_are_not_mixed(stored):
    db = stored["db"]
    db.execute(
        "UPDATE document_publications SET completed_at=NULL WHERE collection_name='kb_test-model'"
    )
    db.commit()
    assert _query(stored, operation="search")["items"]
    changed = deepcopy(stored["snapshot"])
    changed["source_hash"] = "b" * 64
    repository.publish(db, "kb_test-model", "/cad/neutral.pkg", "test-model", "v2", changed)
    db.rollback()
    db.execute(
        "UPDATE document_publications SET completed_at=now() WHERE collection_name='kb_test-model'"
    )
    db.commit()
    assert _query(stored, operation="search")["items"][0]["source_hash"] == "a" * 64
    repository.publish(db, "kb_test-model", "/cad/neutral.pkg", "test-model", "v2", changed)
    db.execute(
        "UPDATE document_publications SET versions=%s::jsonb WHERE collection_name='kb_test-model'",
        (json.dumps({"cad-pkg": "v2"}),),
    )
    db.commit()
    assert _query(stored, operation="search")["items"][0]["source_hash"] == "b" * 64
    with pytest.raises(ValueError, match="Dateiversion"):
        _query(stored, operation="bom", model_id=stored["model_id"], source_hash="a" * 64)
    # Vector cleanup cannot revoke the independently retained SQL membership.
    db.execute("DELETE FROM file_hashes WHERE datasource_id=%s", (stored["ds"],))
    db.commit()
    assert _query(stored, operation="search")["items"]
    # Explicit SQL membership deletion does revoke visibility.
    db.execute("DELETE FROM cad_model_owners WHERE datasource_id=%s", (stored["ds"],))
    db.commit()
    assert not _query(stored, operation="search")["items"]
    repository.cleanup(db, "kb_test-model", ["/cad/neutral.pkg"])
    db.commit()
    assert (
        db.execute(
            "SELECT count(*) AS n FROM cad_occurrences WHERE model_id=%s", (stored["model_id"],)
        ).fetchone()["n"]
        == 0
    )


def test_numeric_unit_conversion_and_exclusive_bounds(stored):
    params = {
        "operation": "search",
        "numeric_field": "dichte_g_cm3",
        "numeric_unit": "kg/m3",
        "minimum": 7850,
    }
    assert _query(stored, **params)["items"]
    assert not _query(stored, **params, minimum_exclusive=True)["items"]
    assert _query(stored, operation="search", numeric_field="dichte_g_cm3", maximum=7.85)["items"]
    assert not _query(
        stored,
        operation="search",
        numeric_field="dichte_g_cm3",
        maximum=7.85,
        maximum_exclusive=True,
    )["items"]
    assert repository.validate(repository.validate(params)) == repository.validate(params)
    with pytest.raises(ValueError, match="Einheit"):
        _query(stored, **{**params, "numeric_unit": "mm"})


def test_subassembly_bom_and_occurrence_pagination(stored):
    assemblies = _query(stored, operation="search", query="Modul")["items"]
    assert len(assemblies) == 1
    occurrences = _query(
        stored,
        operation="where_used",
        model_id=stored["model_id"],
        object_id=assemblies[0]["object_id"],
        limit=1,
    )
    assert occurrences["has_more"]
    second = _query(
        stored,
        operation="where_used",
        model_id=stored["model_id"],
        object_id=assemblies[0]["object_id"],
        limit=1,
        offset=1,
    )
    assert not second["has_more"]
    assert second["items"][0]["occurrence_id"] != occurrences["items"][0]["occurrence_id"]
    bom = _query(
        stored,
        operation="bom",
        model_id=stored["model_id"],
        occurrence_id=occurrences["items"][0]["occurrence_id"],
    )
    assert [r["quantity"] for r in bom["items"]] == [2]
    root = _query(stored, operation="search", query="Maschine")["items"][0]
    root_occ = _query(
        stored, operation="where_used", model_id=stored["model_id"], object_id=root["object_id"]
    )["items"][0]
    direct = _query(
        stored,
        operation="bom",
        model_id=stored["model_id"],
        occurrence_id=root_occ["occurrence_id"],
        recursive=False,
    )
    assert sorted((r["kind"], r["quantity"]) for r in direct["items"]) == [
        ("assembly", 2),
        ("part", 1),
    ]


def test_incomplete_model_cannot_claim_bom(stored):
    stored["db"].execute(
        "UPDATE cad_models SET structure_complete=false WHERE id=%s", (stored["model_id"],)
    )
    stored["db"].commit()
    with pytest.raises(ValueError, match="Unvollständig"):
        _query(stored, operation="bom", model_id=stored["model_id"])


def test_chat_tool_applies_datasource_scope(stored):
    from kiara.tools.cad import CadQueryTool

    tool = CadQueryTool({}, [stored["ds"]], "test-model")
    result = asyncio.run(tool.execute({"operation": "bom", "model_id": stored["model_id"]}))
    assert result.success and result.data["items"][0]["quantity"] == 5
    assert result.data["sources"][0]["source"] == "/cad/neutral.pkg"
    result = asyncio.run(tool.execute({"operation": "search", "datasource_id": stored["ds"] + 1}))
    assert not result.success


def test_regular_pipeline_publishes_both_targets_with_source_provenance(
    whole_pipeline, tmp_path, monkeypatch
):
    case = whole_pipeline
    pkg = tmp_path / "part.pkg"
    pkg.write_bytes((Path(__file__).parent / "data/cad/neutral-plate.pkg").read_bytes())
    case.store._config["datasource"]["file_types"] = [".pkg"]
    result = case.run_whole(case.ds, chunk_size=1000)
    assert result["processed_files"] == 1 and result["errors"] == 0
    case.enrich.assert_not_called()
    results = repository.query(
        {"operation": "search"},
        allowed_datasource_ids=[case.ds],
        embedding_model="test-model",
        collection_name="kb_test",
    )
    assert results["items"]
    vectors = case.store.get_documents_by_source(str(pkg))
    assert vectors["ids"]
    versions = {m["processing_version"] for m in vectors["metadatas"]}
    sql_version = results["items"][0]["processing_version"]
    assert len(versions) == 1
    assert results["items"][0]["source_hash"] == hashlib.sha256(pkg.read_bytes()).hexdigest()
    assert all(
        m["cad_source_hash"] == hashlib.sha256(pkg.read_bytes()).hexdigest()
        for m in vectors["metadatas"]
    )
    repeat = case.run_whole(case.ds, chunk_size=1000)
    assert repeat["processed_files"] == 0 and repeat["errors"] == 0
    from kiara import cad

    monkeypatch.setattr(cad, "ADAPTER_REVISION", "test-upgrade")
    upgraded = case.run_whole(case.ds, chunk_size=1000)
    assert upgraded["processed_files"] == 1 and upgraded["errors"] == 0
    changed = repository.query(
        {"operation": "search"},
        allowed_datasource_ids=[case.ds],
        embedding_model="test-model",
        collection_name="kb_test",
    )["items"][0]
    assert changed["processing_version"] != sql_version
    assert {
        m["processing_version"] for m in case.store.get_documents_by_source(str(pkg))["metadatas"]
    } != versions


def test_failed_sql_publication_retains_previous_snapshot(stored):
    from kiara.shared.db import db_transaction

    changed = deepcopy(stored["snapshot"])
    changed["source_hash"] = "b" * 64
    with pytest.raises(RuntimeError, match="rollback"), db_transaction() as conn:
        repository.publish(conn, "kb_test-model", "/cad/neutral.pkg", "test-model", "v2", changed)
        raise RuntimeError("rollback")
    item = _query(stored, operation="search")["items"][0]
    assert item["processing_version"] == "v1" and item["source_hash"] == "a" * 64


def test_mcp_anonymizes_all_property_text_and_preserves_generated_ids(stored, monkeypatch):
    from types import SimpleNamespace

    from test_mcp_anonymization import NAME, _config

    from kiara.services.mcp_anonymization import McpOutputAnonymizer
    from kiara.services.retrieval_only import RetrievalOnlyError, RetrievalOnlyService

    db = stored["db"]
    row = db.execute(
        "SELECT object_id,record FROM cad_contents WHERE model_id=%s AND kind='part'",
        (stored["model_id"],),
    ).fetchone()
    record = row["record"]
    record["names"] = [NAME]
    record["properties"] = [
        {
            "namespace": NAME,
            "name": NAME,
            "value": NAME,
            "status": "stored",
            "unit": None,
            "source": None,
        }
    ]
    db.execute(
        "UPDATE cad_contents SET names=%s,properties=%s::jsonb,record=%s::jsonb WHERE model_id=%s AND object_id=%s",
        (
            [NAME],
            json.dumps(record["properties"]),
            json.dumps(record),
            stored["model_id"],
            row["object_id"],
        ),
    )
    db.commit()
    audits = []
    monkeypatch.setattr(McpOutputAnonymizer, "audit", lambda self, *args: audits.append(args))

    def service(level="anonymize", entries=True):
        return RetrievalOnlyService(
            _config(entries=entries),
            embedding_model="test-model",
            restriction_levels={stored["ds"]: level},
            permission_repo=SimpleNamespace(get_public_datasource_ids=lambda: []),
            datasource_repo=SimpleNamespace(get_active=lambda: [{"id": stored["ds"]}]),
        )

    user = {"id": 77, "datasource_ids": [stored["ds"]]}
    params = {"operation": "object", "model_id": stored["model_id"], "object_id": row["object_id"]}
    output = service().cad_query(user, params)
    assert NAME not in json.dumps(output, ensure_ascii=False)
    assert output["items"][0]["object_id"] == row["object_id"]
    assert output["items"][0]["source_hash"] == "a" * 64
    assert audits and audits[-1][1] == "cad_query"
    assert NAME in json.dumps(service("public").cad_query(user, params), ensure_ascii=False)
    assert service("blocked").cad_query(user, {"operation": "search"})["items"] == []
    with pytest.raises(RetrievalOnlyError) as exc:
        service(entries=False).cad_query(user, params)
    assert exc.value.http_status == 503


def test_vendored_reader_matches_recorded_source():
    from kiara import cad

    root = Path(cad.__file__).parent / "_vendor"
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["commit"] == cad.READER_REVISION
    for path, expected in provenance["files"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected


def test_cad_rest_requires_auth_and_passes_authenticated_scope(monkeypatch):
    from fastapi.testclient import TestClient
    from test_mcp_retrieval import _app_config

    from kiara.mcp_server import _create_app
    from kiara.web.oidc_auth import OIDCRetrievalAuthenticator

    app = _create_app(_app_config())
    with TestClient(app, raise_server_exceptions=False) as client:
        assert (
            client.post("/api/retrieval/cad_query", json={"operation": "search"}).status_code == 401
        )
    user = {"id": 17, "is_admin": False, "datasource_ids": [3]}
    calls = []
    monkeypatch.setattr(OIDCRetrievalAuthenticator, "authenticate", lambda *a, **k: user)

    def query(actual_user, params):
        calls.append((actual_user, params))
        return {"ok": True, "items": []}

    monkeypatch.setattr("kiara.mcp_server._run_cad_query", query)
    params = {
        "operation": "search",
        "numeric_field": "dichte_g_cm3",
        "minimum": 7,
        "minimum_exclusive": True,
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/retrieval/cad_query",
            json=params,
            headers={"Authorization": "Bearer jwt.header.payload"},
        )
    assert response.status_code == 200 and response.json()["ok"]
    assert calls == [(user, params)]
