"""Real SQL and in-memory vector integration for independent PKG targets."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_cad_ingest import snapshot, stored  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.cad import repository
from kiara.ingestion import pipeline


def prepare(case, tmp_path, *, mixed=False):
    path = tmp_path / "part.pkg"
    path.write_bytes((Path(__file__).parent / "data/cad/neutral-plate.pkg").read_bytes())
    case.store._config["datasource"]["file_types"] = [".pkg", ".txt"] if mixed else [".pkg"]
    return path


def run(case, vector, relational, *, incremental=True):
    return pipeline.run_ingestion(
        case.store._config,
        datasource_id=case.ds,
        incremental=incremental,
        embedding_model="test-model",
        ds_overrides={
            "cad_vector_enabled": vector,
            "cad_relational_enabled": relational,
            "chunk_size": 1000,
            "chunk_overlap": 0,
        },
    )


def sql(case):
    return repository.query({"operation": "search"}, allowed_datasource_ids=[case.ds])["items"]


@pytest.mark.parametrize(
    "vector,relational", [(True, True), (True, False), (False, True), (False, False)]
)
def test_four_targets(whole_pipeline, tmp_path, monkeypatch, vector, relational):
    case = whole_pipeline
    path = prepare(case, tmp_path)
    if not vector:
        monkeypatch.setattr(
            pipeline, "VectorStore", Mock(side_effect=AssertionError("vector backend called"))
        )
        monkeypatch.setattr(
            pipeline, "get_embedding_model", Mock(side_effect=AssertionError("embedding requested"))
        )
    result = run(case, vector, relational)
    assert result["errors"] == 0
    assert bool(sql(case)) is relational
    assert bool(case.store.get_documents_by_source(str(path))["ids"]) is vector
    if vector or relational:
        assert result["processed_files"] == 1
    else:
        assert result["skipped_files"] == 1
    assert run(case, vector, relational)["processed_files"] == 0


def test_switches_preserve_disabled_target_and_catch_up_unchanged_file(
    whole_pipeline, tmp_path, monkeypatch
):
    from kiara import cad

    case = whole_pipeline
    path = prepare(case, tmp_path)
    run(case, True, True)
    old_sql = sql(case)
    old_vectors = deepcopy(case.store.get_documents_by_source(str(path)))
    old_hashes = case.hashes.load(case.ds, "test-model")
    monkeypatch.setattr(cad, "ADAPTER_REVISION", "changed")
    # Even a full SQL-only import must not clear vectors or their completion hashes.
    run(case, False, True, incremental=False)
    assert sql(case)[0]["processing_version"] != old_sql[0]["processing_version"]
    assert case.store.get_documents_by_source(str(path)) == old_vectors
    assert case.hashes.load(case.ds, "test-model") == old_hashes
    current_sql = sql(case)
    run(case, True, False, incremental=False)
    assert sql(case) == current_sql
    assert case.store.get_documents_by_source(str(path)) != old_vectors
    before = deepcopy(case.store.get_documents_by_source(str(path)))
    run(case, False, False, incremental=False)
    assert sql(case) == current_sql
    assert case.store.get_documents_by_source(str(path)) == before
    # Re-enabling SQL populates a previously missing snapshot despite unchanged vector hash.
    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        conn.execute("DELETE FROM cad_model_owners WHERE datasource_id=%s", (case.ds,))
    assert run(case, True, True)["cad_relational_files"] == 1
    assert sql(case)
    assert case.store.get_documents_by_source(str(path)) == before


def test_mixed_source_full_import_keeps_disabled_pkg_vectors(whole_pipeline, tmp_path):
    case = whole_pipeline
    path = prepare(case, tmp_path, mixed=True)
    Path(case.path).write_text("A useful ordinary document with its own contents. " * 30)
    run(case, True, True)
    before = deepcopy(case.store.get_documents_by_source(str(path)))
    digest = case.hashes.load(case.ds, "test-model")[str(path)]
    result = run(case, False, False, incremental=False)
    assert result["errors"] == 0 and result["processed_files"] == 1
    assert case.store.get_documents_by_source(str(path)) == before
    assert case.hashes.load(case.ds, "test-model")[str(path)] == digest
    path.unlink()
    run(case, False, False)
    assert case.store.get_documents_by_source(str(path)) == before
    assert sql(case)


def test_shared_sql_source_preserves_other_datasource_version(stored, test_db):
    ds2 = test_db.execute(
        "INSERT INTO datasources(name,type) VALUES('other','local') RETURNING id"
    ).fetchone()["id"]
    test_db.execute(
        "INSERT INTO cad_model_owners VALUES(%s,'/cad/neutral.pkg',%s)", (ds2, stored["model_id"])
    )
    changed = deepcopy(stored["snapshot"])
    changed["source_hash"] = "b" * 64
    repository.publish(
        test_db, "", "/cad/neutral.pkg", "", "sql-v2", changed, datasource_id=stored["ds"]
    )
    test_db.commit()
    first = repository.query({"operation": "search"}, allowed_datasource_ids=[stored["ds"]])
    second = repository.query({"operation": "search"}, allowed_datasource_ids=[ds2])
    assert first["items"][0]["source_hash"] == "b" * 64
    assert second["items"][0]["source_hash"] == "a" * 64
    # Removing the old vector publication must not cascade into retained SQL snapshots.
    test_db.execute("DELETE FROM document_publications WHERE source='/cad/neutral.pkg'")
    test_db.commit()
    assert repository.query({"operation": "search"}, allowed_datasource_ids=[ds2])["items"]


def test_sql_job_lifecycle_without_vector_services(test_db, tmp_path, monkeypatch):
    from kiara.cad.jobs import execute
    from kiara.config import load_config
    from kiara.retrieval import index_compatibility as ic
    from kiara.web import cache_refresh
    from kiara.web import ingestion_worker as worker_module

    path = tmp_path / "part.pkg"
    path.write_bytes((Path(__file__).parent / "data/cad/neutral-plate.pkg").read_bytes())
    ds = test_db.execute(
        "INSERT INTO datasources(name,type,connector_config,ds_file_types,ds_cad_vector_enabled) "
        "VALUES('SQL only','local',%s,'[\".pkg\"]',false) RETURNING id",
        (json.dumps({"path": str(tmp_path)}),),
    ).fetchone()["id"]
    test_db.commit()
    config = load_config()
    monkeypatch.setattr("kiara.config.load_config", lambda: deepcopy(config))
    monkeypatch.setattr(
        ic, "guard_job_contract", Mock(side_effect=AssertionError("vector contract called"))
    )
    monkeypatch.setattr(ic, "guard_index", Mock(side_effect=AssertionError("vector index called")))
    monkeypatch.setattr(
        pipeline, "VectorStore", Mock(side_effect=AssertionError("vector store called"))
    )
    for name in ("refresh_datasource_cache", "refresh_jobs_cache"):
        monkeypatch.setattr(cache_refresh, name, Mock())
    job_id = worker_module.enqueue_job(ds, embedding_model="")
    worker = worker_module.IngestionWorker()
    row = worker._fetch_next_job()
    assert row["id"] == job_id and row["embedding_model"] == ""
    execute(worker, row, lambda: None)
    test_db.commit()
    result = test_db.execute("SELECT * FROM ingestion_jobs WHERE id=%s", (job_id,)).fetchone()
    assert result["status"] == "success", result["error_message"]
    assert result["stats_json"]["cad_relational_files"] == 1
    assert repository.query({"operation": "search"}, allowed_datasource_ids=[ds])["items"]
    assert not test_db.execute("SELECT * FROM file_hashes WHERE datasource_id=%s", (ds,)).fetchone()
    assert not test_db.execute(
        "SELECT * FROM datasource_embeddings WHERE datasource_id=%s", (ds,)
    ).fetchone()
    assert worker._recover_single_job(None, {**dict(result), "requested_action": None})


@pytest.mark.asyncio
async def test_settings_roundtrip_and_defaults(test_db, monkeypatch):
    from kiara.admin.functions.datasources import datasource_update
    from kiara.admin.services.datasource_service import create_datasource
    from kiara.ingestion.datasource_runtime import build_datasource_overrides
    from kiara.repositories.datasource import DatasourceRepository
    from kiara.web.api_datasource import DatasourceRequest

    monkeypatch.setattr("kiara.web.cache_refresh.refresh_datasource_cache", Mock())
    defaults = DatasourceRequest()
    assert defaults.ds_cad_vector_enabled and defaults.ds_cad_relational_enabled
    created = await create_datasource(
        name="CAD selection", ds_type="local", connector_config={"path": "/tmp"}
    )
    ds_id = created["id"]
    for vector, relational in ((False, True), (True, False), (False, False), (True, True)):
        result = await datasource_update(
            {}, ds_id, ds_cad_vector_enabled=vector, ds_cad_relational_enabled=relational
        )
        assert result.success
        ds = DatasourceRepository().get_by_id(ds_id)
        assert ds["ds_cad_vector_enabled"] is vector
        assert ds["ds_cad_relational_enabled"] is relational
        overrides = build_datasource_overrides(ds)
        assert overrides["cad_vector_enabled"] is vector
        assert overrides["cad_relational_enabled"] is relational


def test_migration_backfills_existing_sql_and_is_idempotent(stored, test_db):
    test_db.execute("DELETE FROM cad_model_owners")
    test_db.execute(
        "ALTER TABLE cad_models ADD CONSTRAINT cad_models_collection_name_source_fkey "
        "FOREIGN KEY(collection_name,source) REFERENCES document_publications(collection_name,source) ON DELETE CASCADE"
    )
    repository.migrate(test_db)
    repository.migrate(test_db)
    test_db.commit()
    found = repository.query({"operation": "search"}, allowed_datasource_ids=[stored["ds"]])
    assert found["items"][0]["source_hash"] == "a" * 64
    test_db.execute("DELETE FROM document_publications")
    test_db.commit()
    assert repository.query({"operation": "search"}, allowed_datasource_ids=[stored["ds"]])["items"]


def test_unknown_reader_field_skips_file_and_preserves_sql_and_vectors(
    whole_pipeline, tmp_path, monkeypatch
):
    # Nutzerentscheidung 04.11.2026: PKG-Lesefehler werden protokolliert und die
    # Datei uebersprungen; der Job laeuft weiter, der Bestand bleibt erhalten.
    from kiara.cad.adapter import CadReadError

    case = whole_pipeline
    path = prepare(case, tmp_path)
    run(case, True, True)
    old_sql = sql(case)
    old_vectors = deepcopy(case.store.get_documents_by_source(str(path)))
    monkeypatch.setattr(
        "kiara.cad.adapter.load_pkg", Mock(side_effect=CadReadError("ungeklärtes Testfeld"))
    )
    result = run(case, True, True, incremental=False)
    assert result["errors"] == 1 and "abort_reason" not in result
    assert result["cad_relational_errors"] == 1
    assert result["cad_error_policy"] == "skip_file"
    assert sql(case) == old_sql
    assert case.store.get_documents_by_source(str(path)) == old_vectors


def test_removed_file_only_cleans_selected_sql_target(whole_pipeline, tmp_path):
    case = whole_pipeline
    path = prepare(case, tmp_path)
    run(case, True, True)
    before = deepcopy(case.store.get_documents_by_source(str(path)))
    path.unlink()
    run(case, False, False)
    assert sql(case)
    run(case, False, True)
    assert not sql(case)
    assert case.store.get_documents_by_source(str(path)) == before


def test_real_pilot_relational_only(test_db, tmp_path, monkeypatch):
    import os
    import shutil
    import time

    from kiara.config import load_config

    source = os.getenv("CAD_PILOT_PKG")
    if not source:
        pytest.skip("Explicit CAD_PILOT_PKG required")
    pkg = tmp_path / "pilot.pkg"
    shutil.copyfile(source, pkg)
    ds = test_db.execute(
        "INSERT INTO datasources(name,type) VALUES('CAD pilot SQL','local') RETURNING id"
    ).fetchone()["id"]
    test_db.commit()
    config = load_config()
    config["datasource"]["path"] = str(tmp_path)
    config["datasource"]["file_types"] = [".pkg"]
    monkeypatch.setattr(
        pipeline, "VectorStore", Mock(side_effect=AssertionError("vector store called"))
    )
    monkeypatch.setattr(
        pipeline, "get_embedding_model", Mock(side_effect=AssertionError("embedding called"))
    )
    started = time.monotonic()
    result = pipeline.run_ingestion(
        config,
        datasource_id=ds,
        incremental=True,
        ds_overrides={"cad_vector_enabled": False, "cad_relational_enabled": True},
    )
    assert result["errors"] == 0 and result["cad_relational_files"] == 1

    def count(query):
        return test_db.execute(query).fetchone()["n"]

    assert (
        count(
            "SELECT count(*) n FROM cad_contents WHERE kind='part' AND (record->>'has_body')::boolean"
        )
        == 341
    )
    assert (
        count(
            "SELECT count(*) n FROM cad_occurrences o JOIN cad_contents c USING(model_id,object_id) WHERE o.physical AND c.kind='part'"
        )
        == 8730
    )
    thickness = repository.query(
        {"operation": "search", "numeric_field": "blechdicke_mm", "minimum": 2, "maximum": 2},
        allowed_datasource_ids=[ds],
    )
    assert len(thickness["items"]) == 25
    repeat = pipeline.run_ingestion(
        config,
        datasource_id=ds,
        incremental=True,
        ds_overrides={"cad_vector_enabled": False, "cad_relational_enabled": True},
    )
    assert repeat["cad_relational_files"] == 0 and repeat["cad_relational_skipped"] == 1
    print(
        json.dumps(
            {
                "sql_only_seconds": round(time.monotonic() - started, 2),
                "parts": 341,
                "part_occurrences": 8730,
                "thickness_2mm": 25,
                "repeat_skipped": True,
            }
        )
    )


def test_uppercase_pkg_selection_is_processed(whole_pipeline, tmp_path):
    case = whole_pipeline
    prepare(case, tmp_path)
    case.store._config["datasource"]["file_types"] = [".PKG"]
    assert run(case, False, True)["cad_relational_files"] == 1
