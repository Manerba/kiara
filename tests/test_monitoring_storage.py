"""Storage accounting must distinguish missing measurements from empty stores."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from kiara.monitoring import storage


def test_postgresql_table_failure_keeps_measured_database_size():
    connection = MagicMock()
    connection.execute.side_effect = [
        MagicMock(fetchone=lambda: {"size_bytes": 1000}),
        RuntimeError("detail query failed"),
    ]
    with patch.object(storage, "db_query") as query:
        query.return_value.__enter__.return_value = connection
        result = storage._postgresql_sizes()
    assert result == {"status": "ok", "size_bytes": 1000}


@pytest.mark.parametrize("value", [None, -1, True, "1000"])
def test_invalid_database_size_is_unavailable(value):
    with patch.object(storage, "db_query") as query:
        query.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {
            "size_bytes": value
        }
        result = storage._postgresql_sizes()
    assert result["status"] == "unavailable"
    assert result["size_bytes"] is None


def test_qdrant_autodetects_storage_and_honors_explicit_override(tmp_path, monkeypatch):
    auto = tmp_path / "auto"
    (auto / "collections").mkdir(parents=True)
    custom = tmp_path / "custom"
    (custom / "collections").mkdir(parents=True)
    monkeypatch.setattr(storage, "_QDRANT_PATHS", (str(auto),))
    config = {"vectordb": {"qdrant_url": "http://localhost:6333"}}
    assert storage._qdrant_path(config) == auto
    config["monitoring"] = {"qdrant_data_path": str(custom)}
    assert storage._qdrant_path(config) == custom
    config["monitoring"]["qdrant_data_path"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        storage._qdrant_path(config)


def test_qdrant_remote_server_does_not_measure_unrelated_local_storage(tmp_path, monkeypatch):
    (tmp_path / "collections").mkdir()
    monkeypatch.setattr(storage, "_QDRANT_PATHS", (str(tmp_path),))
    with pytest.raises(ValueError, match="externes Qdrant"):
        storage._qdrant_path({"vectordb": {"qdrant_url": "https://qdrant.example:6333"}})


def test_qdrant_autodetect_skips_unreadable_alternative(tmp_path, monkeypatch):
    from pathlib import Path

    (tmp_path / "collections").mkdir()
    inaccessible = tmp_path / "docker-volumes"
    original = Path.is_dir

    def is_dir(path):
        if path == inaccessible / "collections":
            raise PermissionError("docker volume parent is root-only")
        return original(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    monkeypatch.setattr(storage, "_QDRANT_PATHS", (str(inaccessible), str(tmp_path)))
    assert storage._qdrant_path({}) == tmp_path


def test_qdrant_ambiguous_paths_require_config(tmp_path, monkeypatch):
    paths = [tmp_path / "first", tmp_path / "second"]
    for path in paths:
        (path / "collections").mkdir(parents=True)
    monkeypatch.setattr(storage, "_QDRANT_PATHS", tuple(map(str, paths)))
    with pytest.raises(ValueError, match="nicht eindeutig"):
        storage._qdrant_path({})


def test_directory_sizes_count_allocated_blocks_and_hardlinks_once(tmp_path):
    collection = tmp_path / "collections" / "kb_model"
    collection.mkdir(parents=True)
    sparse = collection / "vectors"
    with sparse.open("wb") as output:
        output.seek(10_000_000)
        output.write(b"vector")
    (collection / "link").hardlink_to(sparse)
    total, collections = storage._directory_sizes(tmp_path)
    expected = sum(
        path.stat().st_blocks * 512
        for path in [tmp_path, tmp_path / "collections", collection, sparse]
    )
    assert total == expected
    assert collections == {
        "kb_model": (collection.stat().st_blocks + sparse.stat().st_blocks) * 512
    }
    assert total < sparse.stat().st_size


def test_directory_symlinks_do_not_silently_undercount(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolische Links"):
        storage._directory_sizes(tmp_path)


@pytest.fixture
def qdrant_metadata():
    with (
        patch(
            "kiara.retrieval.vectorstore.VectorStore.list_collections",
            return_value=[
                {"name": "kb_first", "model": "first", "chunks": 1},
                {"name": "kb_second", "model": "second", "chunks": 2},
            ],
        ) as dense,
        patch(
            "kiara.retrieval.colbert.ColBERTStore.list_colbert_collections",
            return_value=[
                {"name": "colbert_model", "model": "model", "kind": "colbert", "chunks": 1},
            ],
        ),
    ):
        yield dense


def test_qdrant_reports_all_models_and_colbert_without_double_counting(
    tmp_path, monkeypatch, qdrant_metadata
):
    for name in ["kb_first", "kb_second", "colbert_model"]:
        path = tmp_path / "collections" / name
        path.mkdir(parents=True)
        (path / "data").write_bytes(b"payload")
    config = {"monitoring": {"qdrant_data_path": str(tmp_path)}}
    result = storage._qdrant_sizes(config)
    assert {row["name"] for row in result["collections"]} == {
        "kb_first",
        "kb_second",
        "colbert_model",
    }
    assert result["size_bytes"] == storage._directory_sizes(tmp_path)[0]
    assert all(row["size_bytes"] > 0 for row in result["collections"])
    assert (
        next(row for row in result["collections"] if row["kind"] == "colbert")["model"] == "model"
    )
    assert qdrant_metadata.call_count == 1


def test_unreadable_qdrant_reports_unknown_sizes_including_collections(
    monkeypatch, qdrant_metadata
):
    monkeypatch.setattr(storage, "_qdrant_path", MagicMock(side_effect=PermissionError))
    result = storage._qdrant_sizes({})
    assert result["status"] == "unavailable"
    assert result["size_bytes"] is None
    assert len(result["collections"]) == 3
    assert all(row["size_bytes"] is None for row in result["collections"])


def test_disappearing_qdrant_segment_retries_once(monkeypatch, qdrant_metadata, tmp_path):
    monkeypatch.setattr(storage, "_qdrant_path", lambda _: tmp_path)
    scan = MagicMock(side_effect=[FileNotFoundError, (100, {"kb_first": 50})])
    monkeypatch.setattr(storage, "_directory_sizes", scan)
    assert storage._qdrant_sizes({})["size_bytes"] == 100
    assert scan.call_count == 2


@pytest.fixture
def es_storage_inventory(monkeypatch, s4_guard_passthrough):
    from kiara.retrieval.elasticsearch_indices import (
        DatasourceIndexAvailability,
        DatasourceIndexTarget,
    )

    target = DatasourceIndexTarget(1, "custom_ds_1", 2)
    monkeypatch.setattr(
        "kiara.retrieval.elasticsearch_inventory.inventory",
        lambda config: ({"counts_match": True}, DatasourceIndexAvailability((target,), (), {1: 2})),
    )
    return {"retrieval": {"hybrid_search": "elasticsearch", "elasticsearch_index": "custom"}}


def test_elasticsearch_counts_replicas_and_translog_and_scopes_configured_index(
    es_storage_inventory,
):
    with patch("kiara.retrieval.elasticsearch_service._get_client") as get_client:
        stats = get_client.return_value.options.return_value.indices.stats
        stats.return_value = {
            "_shards": {"total": 2, "successful": 2, "failed": 0},
            "indices": {
                "custom_ds_1": {
                    "primaries": {"store": {"size_in_bytes": 50}, "translog": {"size_in_bytes": 5}},
                    "total": {"store": {"size_in_bytes": 100}, "translog": {"size_in_bytes": 10}},
                }
            },
        }
        result = storage._elasticsearch_sizes(es_storage_inventory)
        assert result["size_bytes"] == 110
        assert result["primary_store_bytes"] == 50
        assert result["primary_translog_bytes"] == 5
        assert not result["retained_common_index_included"]
        stats.assert_called_once_with(
            index=["custom_ds_1"], metric="store,translog", level="indices", expand_wildcards="none"
        )


@pytest.mark.parametrize(
    "response",
    [
        {"_shards": {"failed": 1}},
        {"_shards": {"total": 2, "successful": 1, "failed": 0}, "indices": {}},
        {"_shards": {"total": 1, "successful": 1, "failed": 0}, "indices": {}},
        {"_shards": {"total": 1, "successful": 1, "failed": 0}, "indices": {"custom_ds_1": {}}},
    ],
)
def test_incomplete_elasticsearch_response_is_unavailable(response, es_storage_inventory):
    with patch("kiara.retrieval.elasticsearch_service._get_client") as client:
        client.return_value.options.return_value.indices.stats.return_value = response
        result = storage._elasticsearch_sizes(es_storage_inventory)
        client.return_value.options.return_value.indices.stats.assert_called_once()
    assert result["status"] == "unavailable" and result["size_bytes"] is None


@pytest.mark.parametrize("failed", [None, "postgresql", "qdrant", "elasticsearch"])
def test_total_requires_every_measurement_and_never_adds_details(monkeypatch, failed):
    for name in ["postgresql", "qdrant", "elasticsearch"]:
        result = (
            storage._unavailable("error")
            if name == failed
            else {
                "status": "ok",
                "size_bytes": 100,
                "tables": [{"size_bytes": 50}],
                "collections": [{"size_bytes": 50}],
            }
        )
        monkeypatch.setattr(storage, "_" + name + "_sizes", lambda *_, result=result: result)
    result = storage._collect_sizes({})
    assert result["total_bytes"] == (None if failed else 300)


def test_cache_is_nonblocking_single_refresh_and_returns_copies(monkeypatch):
    monkeypatch.setattr(storage, "_snapshot", {})
    monkeypatch.setattr(storage, "_refreshing", False)
    monkeypatch.setattr(storage, "_refreshed_at", 0)
    monkeypatch.setattr(storage.time, "monotonic", lambda: 100)
    with patch.object(storage.threading, "Thread") as thread:
        assert storage.get_storage_sizes()["loading"] is True
        assert storage.get_storage_sizes()["loading"] is True
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()
        monkeypatch.setattr(storage, "_snapshot", {"total_bytes": 10, "qdrant": {"size_bytes": 10}})
        monkeypatch.setattr(storage, "_refreshing", False)
        monkeypatch.setattr(storage, "_refreshed_at", 100)
        snapshot = storage.get_storage_sizes()
        snapshot["qdrant"]["size_bytes"] = 0
        assert storage.get_storage_sizes()["qdrant"]["size_bytes"] == 10
        thread.assert_called_once()
        monkeypatch.setattr(storage.time, "monotonic", lambda: 160)
        storage.get_storage_sizes()
        assert thread.call_count == 2


def test_storage_admin_function_preserves_unavailable_values():
    from kiara.admin.functions.monitoring import system_storage

    with patch.object(storage, "get_storage_sizes", return_value={"total_bytes": None}) as get:
        result = asyncio.run(system_storage({"id": 1, "username": "admin"}))
    assert result.success
    assert result.data == {"total_bytes": None}
    get.assert_called_once()


def test_storage_http_route_requires_admin_and_returns_raw_snapshot():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kiara.admin.functions import monitoring  # noqa: F401
    from kiara.admin.registry import admin_registry
    from kiara.admin.router import _create_endpoint
    from kiara.web.auth import get_admin_user

    function = admin_registry.get("system.storage")
    app = FastAPI()
    app.add_api_route(function.http_path, _create_endpoint(function), methods=["GET"])
    snapshot = {"total_bytes": None, "qdrant": {"status": "ok", "size_bytes": 0}}
    with (
        TestClient(app) as client,
        patch.object(storage, "get_storage_sizes", return_value=snapshot),
    ):
        assert client.get("/api/monitoring/storage").status_code in {401, 403}
        app.dependency_overrides[get_admin_user] = lambda: {"id": 1, "username": "admin"}
        response = client.get("/api/monitoring/storage")
    assert response.status_code == 200
    assert response.json() == snapshot
