"""A rejected PKG must not stop other files or erase its published version."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.cad import repository
from kiara.ingestion import pipeline

PKG = Path(__file__).parent / "data/cad/neutral-plate.pkg"


def run(case, *, vector, mixed=False, logs=None, incremental=False):
    case.store._config["datasource"]["file_types"] = [".pkg", ".txt"] if mixed else [".pkg"]
    return pipeline.run_ingestion(
        case.store._config,
        datasource_id=case.ds,
        embedding_model="test-model",
        incremental=incremental,
        file_log_callback=logs.append if logs is not None else None,
        ds_overrides={
            "cad_vector_enabled": vector,
            "cad_relational_enabled": True,
            "chunk_size": 1000,
            "chunk_overlap": 0,
        },
    )


@pytest.mark.parametrize("vector,mixed", [(False, False), (False, True), (True, True)])
def test_bad_pkg_is_logged_once_good_files_continue_and_old_data_survives(
    whole_pipeline, test_db, tmp_path, vector, mixed
):
    case = whole_pipeline
    bad = tmp_path / "0_bad.pkg"
    bad.write_bytes(PKG.read_bytes())
    assert run(case, vector=True)["errors"] == 0
    before = deepcopy(case.store.get_documents_by_source(str(bad)))
    before_hash = case.hashes.load(case.ds, "test-model")[str(bad)]
    before_sql = test_db.execute(
        "SELECT m.source_hash,m.processing_version FROM cad_models m JOIN cad_model_owners o "
        "ON o.model_id=m.id WHERE o.datasource_id=%s AND o.source=%s",
        (case.ds, str(bad)),
    ).fetchone()
    bad.write_bytes(b"not a valid PKG")
    good = tmp_path / "1_good.pkg"
    good.write_bytes(PKG.read_bytes())
    if mixed:
        Path(case.path).write_text("An ordinary document after a rejected CAD file. " * 30)
    logs = []
    result = run(case, vector=vector, mixed=mixed, logs=logs)
    assert result["errors"] == result["cad_relational_errors"] == 1
    assert not result.get("abort_reason")
    assert result["cad_relational_files"] == 1
    errors = [entry for entry in logs if entry.status == "error"]
    assert len(errors) == 1 and errors[0].file_path == str(bad)
    assert errors[0].error_phase == "cad_relational" and errors[0].error_message
    test_db.commit()
    after_sql = test_db.execute(
        "SELECT m.source_hash,m.processing_version FROM cad_models m JOIN cad_model_owners o "
        "ON o.model_id=m.id WHERE o.datasource_id=%s AND o.source=%s",
        (case.ds, str(bad)),
    ).fetchone()
    assert after_sql == before_sql
    assert case.store.get_documents_by_source(str(bad)) == before
    assert case.hashes.load(case.ds, "test-model")[str(bad)] == before_hash
    assert test_db.execute(
        "SELECT 1 FROM cad_model_owners WHERE datasource_id=%s AND source=%s",
        (case.ds, str(good)),
    ).fetchone()
    if mixed:
        assert case.store.get_documents_by_source(str(case.path))["ids"]
    if vector:
        assert case.store.get_documents_by_source(str(good))["ids"]


def test_cancel_is_still_honored_after_file_error(whole_pipeline, tmp_path):
    case = whole_pipeline
    (tmp_path / "0_bad.pkg").write_bytes(b"broken PKG")
    (tmp_path / "1_good.pkg").write_bytes(PKG.read_bytes())
    case.store._config["datasource"]["file_types"] = [".pkg"]
    logs = []
    result = pipeline.run_ingestion(
        case.store._config,
        datasource_id=case.ds,
        embedding_model="test-model",
        ds_overrides={"cad_vector_enabled": False, "cad_relational_enabled": True},
        file_log_callback=logs.append,
        control_callback=lambda: "cancel" if logs else None,
    )
    assert result["errors"] == 1 and result["cancelled"]
    assert result["processed_files"] == 0 and len(logs) == 1


def test_sql_only_job_finishes_with_recorded_file_errors(test_db, tmp_path, monkeypatch):
    from kiara.cad.jobs import execute
    from kiara.config import load_config
    from kiara.web import cache_refresh
    from kiara.web import ingestion_worker as worker_module

    (tmp_path / "0_bad.pkg").write_bytes(b"broken PKG")
    (tmp_path / "1_good.pkg").write_bytes(PKG.read_bytes())
    ds = test_db.execute(
        "INSERT INTO datasources(name,type,connector_config,ds_file_types,ds_cad_vector_enabled) "
        "VALUES('CAD error continuation','local',%s,'[\".pkg\"]',false) RETURNING id",
        (json.dumps({"path": str(tmp_path)}),),
    ).fetchone()["id"]
    test_db.commit()
    config = load_config()
    monkeypatch.setattr("kiara.config.load_config", lambda: deepcopy(config))
    monkeypatch.setattr(pipeline, "VectorStore", Mock(side_effect=AssertionError("vector access")))
    for name in ("refresh_datasource_cache", "refresh_jobs_cache"):
        monkeypatch.setattr(cache_refresh, name, Mock())
    job_id = worker_module.enqueue_job(ds, embedding_model="")
    worker = worker_module.IngestionWorker()
    row = worker._fetch_next_job()
    assert row["id"] == job_id
    execute(worker, row, lambda: None)
    test_db.commit()
    result = test_db.execute("SELECT * FROM ingestion_jobs WHERE id=%s", (job_id,)).fetchone()
    assert result["status"] == "success", result["error_message"]
    assert result["stats_json"]["errors"] == 1
    assert result["stats_json"]["cad_relational_files"] == 1
    errors = test_db.execute(
        "SELECT * FROM ingestion_file_logs WHERE job_id=%s AND status='error'",
        (job_id,),
    ).fetchall()
    assert len(errors) == 1 and errors[0]["file_name"] == "0_bad.pkg"
    assert repository.query({"operation": "search"}, allowed_datasource_ids=[ds])["items"]
