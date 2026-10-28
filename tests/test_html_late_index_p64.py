"""P64: selected Dense/Late processing must match index, request, cache and job.

These targeted component regressions use real kiara_test PostgreSQL for stored
bindings and job snapshots. HTTP/Qdrant doubles are unit boundaries only; they
do not replace the separate real KIron integration acceptance.
"""

import copy
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from test_index_compatibility_s4 import _config, _persist_contract, _resolved

from kiara.retrieval import embedding
from kiara.retrieval import index_compatibility as contract
from kiara.retrieval.index_compatibility import IndexCompatibilityError, IndexVectorKind
from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore


def _settings(late):
    config = copy.deepcopy(_config())
    config["enrichment"] = {"late_chunking": late}
    config["ollama"] = {"embedding_model": "model"}
    return config


def _resolved_for(target):
    late = target.vector_kind is IndexVectorKind.LATE_CHUNKING
    return replace(
        _resolved(target),
        profile_id="profile-late-v1" if late else "profile-dense-v1",
        profile_endpoint="/api/embed_late" if late else "/api/embed",
        profile_supported_roles=("search_document", "search_query"),
        profile_role_sensitive=True,
        profile_formatting_owner="server",
    )


@pytest.fixture
def raw_store(monkeypatch):
    """Only remote Qdrant and dimension discovery are replaced, never PG guards."""
    raw = MagicMock()
    monkeypatch.setattr("kiara.retrieval.vectorstore_qdrant.QdrantClient", lambda **_: raw)
    monkeypatch.setattr(QdrantVectorStore, "_ensure_collection", lambda *_a, **_kw: None)
    monkeypatch.setattr(QdrantVectorStore, "_detect_source_key_ready", lambda *_: False)
    monkeypatch.setattr(embedding, "get_embedding_dimension", lambda *_: 3)
    return raw


@pytest.fixture
def observed_guard(monkeypatch):
    """Observe actual target choice while providing a matching explicit profile."""
    targets = []

    @contextmanager
    def guard(_config, target, *, operation, **kwargs):
        targets.append(target)
        yield SimpleNamespace(
            target=target,
            resolved=_resolved_for(target),
            operation=operation,
            fence_version=7,
            validation_scope=contract.IndexValidationScope.LIVE_PROFILE,
            write=kwargs.get("write", False),
            require_configured_default=kwargs.get("require_configured_default", False),
        )

    monkeypatch.setattr(contract, "guard_index_for_active_job", guard)
    monkeypatch.setattr(contract, "guard_index", guard)
    return targets


@pytest.mark.parametrize("late", [False, True])
def test_store_first_write_selects_primary_processing_target(raw_store, observed_guard, late):
    store = QdrantVectorStore(_settings(late), embedding_model="model", intent="write")
    expected = IndexVectorKind.LATE_CHUNKING if late else IndexVectorKind.DENSE
    assert observed_guard
    assert all(target.vector_kind is expected for target in observed_guard)
    assert store.get_collection_name() == "kb_model"


@pytest.mark.parametrize("late", [False, True])
def test_query_uses_selected_profile_endpoint_role_and_exact_question(
    monkeypatch, observed_guard, late
):
    requests = []

    def post(url, *, json, **_kwargs):
        requests.append((url, copy.deepcopy(json)))
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3]]},
            request=httpx.Request(
                "POST", str(url) if str(url).startswith("http") else "http://kiron" + str(url)
            ),
        )

    monkeypatch.setattr(
        embedding, "_resolve_backend", lambda *_: {"provider": "ollama", "url": "http://kiron"}
    )
    monkeypatch.setattr(embedding, "get_embedding_dimension", lambda *_: 3)
    monkeypatch.setattr(embedding, "_get_ollama_http_client", lambda *_: SimpleNamespace(post=post))
    monkeypatch.setattr(embedding.httpx, "post", post)
    question = "Welche Spannung gilt für V42?"
    vector = embedding.embed_query(_settings(late), "model", question)
    assert vector == [0.1, 0.2, 0.3]
    assert len(requests) == 1
    url, payload = requests[0]
    assert payload["input_type"] == "search_query"
    assert payload["model"] == "model"
    if late:
        assert url.endswith("/api/embed_late")
        assert payload["document"] == question
        assert payload["chunks"] == [{"text": question, "char_start": 0, "char_end": len(question)}]
        assert observed_guard[-1].vector_kind is IndexVectorKind.LATE_CHUNKING
    else:
        assert url.endswith("/api/embed")
        assert payload["input"] == [question]
        assert observed_guard[-1].vector_kind is IndexVectorKind.DENSE


@pytest.mark.parametrize("late", [False, True])
def test_query_cache_identity_uses_selected_primary_profile(monkeypatch, observed_guard, late):
    from kiara.cache.query_cache import QueryCache

    monkeypatch.setattr("kiara.config.get_embedding_model", lambda _: "model")
    cache = QueryCache(_settings(late))
    with cache._contract_guard("cache_read") as identities:
        assert len(identities) == 1
        identity = identities[0]
        assert identity["profile_kind"] == ("late_chunking" if late else "dense")
        assert identity["endpoint"] == ("/api/embed_late" if late else "/api/embed")
        assert identity["input_type"] == "search_query"
        assert identity["logical_index_key"] == observed_guard[-1].logical_index_key


@pytest.fixture
def local_contracts(test_db, monkeypatch):
    """Keep actual PG state/fences, replacing only remote discovery and count."""
    assert test_db.execute("SELECT current_database() AS name").fetchone()["name"] == "kiara_test"
    monkeypatch.setattr(contract, "_collection_state", lambda *_: (False, 0))
    monkeypatch.setattr(
        contract,
        "resolve_current_index_contract",
        lambda _conn, _cfg, target, **_kwargs: _resolved_for(target),
    )
    return test_db


@pytest.mark.parametrize("late", [False, True])
def test_empty_first_write_persists_selected_primary_binding(local_contracts, raw_store, late):
    store = QdrantVectorStore(_settings(late), embedding_model="model", intent="write")
    row = local_contracts.execute(
        "SELECT vector_kind, profile_kind, profile_endpoint, evidence_type FROM index_contracts WHERE collection_name = %s",
        (store.get_collection_name(),),
    ).fetchone()
    assert row is not None
    expected_kind = "late_chunking" if late else "dense"
    assert row["vector_kind"] == expected_kind
    assert row["profile_kind"] == expected_kind
    assert row["profile_endpoint"] == ("/api/embed_late" if late else "/api/embed")
    assert row["evidence_type"] == "empty_first_write"
    raw_store.upsert.assert_not_called()


def test_late_config_cannot_redeclare_existing_dense_physical_binding(local_contracts, raw_store):
    config = _settings(True)
    dense = contract.dense_index_target(config, "model")
    _persist_contract(local_contracts, dense, _resolved_for(dense))
    local_contracts.commit()
    with pytest.raises(IndexCompatibilityError) as caught:
        QdrantVectorStore(config, embedding_model="model", intent="write")
    assert caught.value.reason_code == "index_physical_binding_conflict"
    row = local_contracts.execute("SELECT vector_kind FROM index_contracts").fetchall()
    assert [item["vector_kind"] for item in row] == ["dense"]
    raw_store.upsert.assert_not_called()
    raw_store.create_collection.assert_not_called()


def test_maintenance_uses_stored_late_binding_when_current_config_is_dense(
    local_contracts, raw_store, monkeypatch
):
    config = _settings(False)
    late = contract.late_index_target(config, "model")
    _persist_contract(local_contracts, late, _resolved_for(late))
    local_contracts.commit()
    # This is an existing index. The shared first-write fixture otherwise
    # models a missing collection, which maintenance must correctly reject.
    monkeypatch.setattr(contract, "_collection_state", lambda *_: (True, 1))
    discovery = MagicMock(
        side_effect=AssertionError("maintenance must not discover a live profile")
    )
    dimension = MagicMock(
        side_effect=AssertionError("maintenance must not compute embedding dimensions")
    )
    monkeypatch.setattr(contract, "resolve_current_index_contract", discovery)
    monkeypatch.setattr(embedding, "get_embedding_dimension", dimension)
    store = QdrantVectorStore(config, embedding_model="model", intent="maintenance")
    assert store._index_target.vector_kind is IndexVectorKind.LATE_CHUNKING
    assert store.get_collection_name() == "kb_model"
    discovery.assert_not_called()
    dimension.assert_not_called()
    raw_store.upsert.assert_not_called()
    raw_store.create_collection.assert_not_called()


def test_active_late_job_snapshot_mismatch_blocks_before_upsert(local_contracts, monkeypatch):
    from kiara.ingestion.runtime import set_current_job_id

    config = _settings(True)
    target = contract.late_index_target(config, "model")
    resolved = _resolved_for(target)
    _persist_contract(local_contracts, target, resolved)
    ds_id = local_contracts.execute(
        "INSERT INTO datasources (name,type,is_active) VALUES ('p64-late-job','local',true) RETURNING id"
    ).fetchone()["id"]
    job_id = local_contracts.execute(
        "INSERT INTO ingestion_jobs (datasource_id,embedding_model,contract_logical_index_key,"
        "contract_index_compatibility_id,contract_query_compatibility_id,contract_canonical_model_id,"
        "contract_profile_id,contract_profile_kind,contract_profile_endpoint,contract_snapshot_sha256,"
        "contract_fence_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1) RETURNING id",
        (
            ds_id,
            "model",
            target.logical_index_key,
            resolved.index_compatibility_id,
            resolved.query_compatibility_id,
            resolved.canonical_model_id,
            "wrong-profile",
            resolved.profile_kind,
            resolved.profile_endpoint,
            resolved.contract_snapshot_sha256,
        ),
    ).fetchone()["id"]
    local_contracts.commit()
    raw = MagicMock()
    monkeypatch.setattr(contract, "raw_qdrant_client", lambda _: raw)
    monkeypatch.setattr(contract, "_collection_state", lambda *_: (True, 1))
    store = object.__new__(QdrantVectorStore)
    store._config = config
    store._index_target = target
    store._client = raw
    store._collection = target.collection_name
    store._require_configured_default = False
    store._intent = "write"
    set_current_job_id(job_id)
    try:
        with pytest.raises(IndexCompatibilityError) as caught:
            store.add_with_embeddings(["V42"], [{"source": "a.txt"}], ["a" * 32], [[0.1, 0.2, 0.3]])
        assert caught.value.reason_code == "index_job_contract_mismatch"
    finally:
        set_current_job_id(None)
    raw.upsert.assert_not_called()


@pytest.mark.parametrize("eligible", [False, True])
def test_selected_late_cannot_write_dense_or_complete_without_late_result(eligible):
    from test_pipeline_decomposition import _make_file_log, _make_ictx

    from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

    ictx = _make_ictx(
        config=_settings(True),
        file_hashes={"/late.txt": ("hash", 3)},
        last_full_text="V42",
        last_original_chunks=["V42"],
        last_original_spans=[(0, 3)],
    )
    with (
        patch("kiara.ingestion.runtime._late_chunking_file_eligible", return_value=eligible),
        patch("kiara.ingestion.runtime.embed_texts_late", return_value=None) as late,
        patch("kiara.ingestion.runtime._write_embeddings", return_value=["id1"]) as writes,
        patch("kiara.ingestion.runtime._write_to_elasticsearch") as elasticsearch,
        patch("kiara.ingestion.runtime._persist_graph_and_associations"),
        patch("kiara.ingestion.runtime._mark_file_processed") as completed,
        patch("kiara.ingestion.runtime._persist_hashes_checkpoint"),
    ):
        action, _ = _handle_embedding_phase(
            ictx,
            "/late.txt",
            ["V42"],
            [{"source": "/late.txt"}],
            ["id1"],
            _make_file_log("/late.txt"),
            0,
            time.monotonic(),
        )
    assert action in {_FileAction.SKIP, _FileAction.ABORT}
    assert ictx.stats["processed_files"] == 0
    assert ictx.stats["errors"] > 0 or ictx.stats.get("abort_reason")
    writes.assert_not_called()
    elasticsearch.assert_not_called()
    completed.assert_not_called()
    assert late.call_count == int(eligible)


@pytest.mark.parametrize(
    ("kind", "expected_default_fence"),
    [
        (IndexVectorKind.DENSE, True),
        (IndexVectorKind.LATE_CHUNKING, True),
        (IndexVectorKind.MULTI_VECTOR, False),
    ],
)
def test_recovery_preserves_default_model_fence_for_primary_indices(
    monkeypatch, kind, expected_default_fence
):
    from test_pipeline_decomposition import _make_ictx

    from kiara.ingestion import runtime

    ictx = _make_ictx(store=SimpleNamespace(_require_configured_default=True))
    error = SimpleNamespace(target=SimpleNamespace(vector_kind=kind))
    recovery = MagicMock(return_value=True)
    monkeypatch.setattr(runtime, "_wait_for_index_contract_target_recovery", recovery)

    assert runtime._wait_for_index_contract_recovery(ictx, error) is True
    recovery.assert_called_once_with(
        ictx.config,
        error,
        control_callback=ictx.control_callback,
        status_callback=ictx.status_callback,
        stats=ictx.stats,
        require_configured_default=expected_default_fence,
    )


@pytest.fixture
def enqueued_late_job(local_contracts, monkeypatch):
    from kiara.web import ingestion_worker

    config = _settings(True)
    target = contract.late_index_target(config, "model")
    ds_id = local_contracts.execute(
        "INSERT INTO datasources (name,type,is_active) "
        "VALUES ('p64-late-enqueue','local',true) RETURNING id"
    ).fetchone()["id"]
    local_contracts.commit()
    monkeypatch.setattr("kiara.config.load_config", lambda: config)
    monkeypatch.setattr(contract, "raw_qdrant_client", lambda _: MagicMock())
    monkeypatch.setattr(ingestion_worker, "safe_refresh", lambda *_args, **_kwargs: None)
    job_id = ingestion_worker.enqueue_job(ds_id, embedding_model="model")
    job = dict(
        local_contracts.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
    )
    return target, _resolved_for(target), job


def test_enqueue_persists_selected_late_profile_snapshot(local_contracts, enqueued_late_job):
    target, resolved, job = enqueued_late_job
    assert job["embedding_model"] == "model"
    assert job["contract_logical_index_key"] == target.logical_index_key
    assert job["contract_profile_kind"] == "late_chunking"
    assert job["contract_profile_endpoint"] == "/api/embed_late"
    assert job["contract_profile_id"] == "profile-late-v1"
    assert job["contract_canonical_model_id"] == resolved.canonical_model_id
    assert job["contract_index_compatibility_id"] == resolved.index_compatibility_id
    assert job["contract_query_compatibility_id"] == resolved.query_compatibility_id
    assert job["contract_snapshot_sha256"] == resolved.contract_snapshot_sha256
    binding = local_contracts.execute(
        "SELECT vector_kind, fence_version FROM index_contracts WHERE logical_index_key = %s",
        (target.logical_index_key,),
    ).fetchone()
    assert binding["vector_kind"] == "late_chunking"
    assert job["contract_fence_version"] == binding["fence_version"] >= 1


def test_mode_change_after_late_enqueue_blocks_before_store_access(enqueued_late_job, raw_store):
    from kiara.ingestion.runtime import set_current_job_id

    _, _, job = enqueued_late_job
    set_current_job_id(job["id"])
    try:
        with pytest.raises(IndexCompatibilityError) as caught:
            QdrantVectorStore(_settings(False), embedding_model="model", intent="write")
        assert caught.value.reason_code == "index_job_target_mismatch"
    finally:
        set_current_job_id(None)
    raw_store.get_collection.assert_not_called()
    raw_store.create_collection.assert_not_called()
    raw_store.upsert.assert_not_called()
