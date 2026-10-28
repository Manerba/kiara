"""HTML inventory remains cancellable without touching the production stores."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kiara.ingestion._file_utils import _CancelledError


def _store():
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    store = object.__new__(QdrantVectorStore)
    store._client = Mock()
    store._collection = "test"
    store._use_source_key = True
    store._index_contract_guard = lambda **_kwargs: nullcontext()
    return store


def test_source_inventory_checks_control_before_first_remote_read():
    store = _store()
    stop = Mock(side_effect=_CancelledError)
    with pytest.raises(_CancelledError):
        store.list_sources_for_datasource(12, batch_callback=stop)
    store._client.scroll.assert_not_called()
    stop.assert_called_once_with(0)


def test_source_inventory_stops_between_remote_pages_without_partial_result():
    store = _store()
    point = SimpleNamespace(payload={"source": "/data/a.html"})
    store._client.scroll.side_effect = [([point], "next"), ([point], None)]
    scanned = []

    def control(count):
        scanned.append(count)
        if count == 1:
            raise _CancelledError

    with pytest.raises(_CancelledError):
        store.list_sources_for_datasource(12, batch_callback=control)
    assert store._client.scroll.call_count == 1
    assert scanned == [0, 1]


def test_source_inventory_keeps_original_result_and_reports_final_progress():
    store = _store()
    point = SimpleNamespace(payload={"source": "/data/a.html"})
    store._client.scroll.side_effect = [([point], "next"), ([point], None)]
    scanned = []
    result = store.list_sources_for_datasource(12, batch_callback=scanned.append)
    assert result == [{"source": "/data/a.html", "file_type": "", "chunk_count": 2, "modified": ""}]
    assert scanned == [0, 1, 2]


@pytest.fixture
def inventory(monkeypatch, tmp_path):
    import kiara.connectors.registry as registry
    import kiara.shared.db as db_module
    from kiara.ingestion._file_utils import ChunkDatasources, HashStore

    connector = SimpleNamespace(effective_path=lambda *_args: str(tmp_path))
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "get", lambda _type: connector)
    monkeypatch.setattr("kiara.ingestion.datasource_runtime.load_connector_config", lambda *_: {})
    monkeypatch.setattr(HashStore, "load", lambda *_: {})
    chunk_ids = [f"chunk-{i}" for i in range(501)]
    monkeypatch.setattr(ChunkDatasources, "get_chunk_ids", lambda *_: chunk_ids)
    store = Mock()
    store.list_sources_for_datasource.return_value = []
    store.get_documents_by_ids.side_effect = lambda ids, **_: {
        "ids": ids,
        "metadatas": [{"source": str(tmp_path / "a.html")} for _ in ids],
    }
    factory = Mock(return_value=store)
    monkeypatch.setattr("kiara.retrieval.vectorstore.VectorStore", factory)
    db = Mock()
    db.execute.return_value.fetchone.return_value = {"id": 12, "type": "local"}
    db.execute.return_value.fetchall.return_value = [{"embedding_model": "nomic-embed-text"}]

    @contextmanager
    def read():
        yield db

    monkeypatch.setattr(db_module, "db_query", read)
    return store, factory, db


@pytest.mark.parametrize("signal", ["cancel", "shutdown"])
def test_preview_cancel_propagates_before_any_datasource_read(inventory, signal):
    from kiara.ingestion.html_migration import preview_migration

    store, factory, db = inventory
    with pytest.raises(_CancelledError):
        preview_migration({}, [12], control_callback=lambda: signal)
    db.execute.assert_not_called()
    factory.assert_not_called()
    store.get_documents_by_ids.assert_not_called()


def test_preview_passes_source_control_and_never_reports_cancel_as_unresolved(inventory):
    from kiara.ingestion.html_migration import preview_migration

    store, _, _ = inventory
    cancelled = False
    progress = Mock()

    def list_sources(_ds, *, batch_callback):
        nonlocal cancelled
        batch_callback(0)
        cancelled = True
        batch_callback(1000)

    store.list_sources_for_datasource.side_effect = list_sources
    with pytest.raises(_CancelledError):
        preview_migration(
            {},
            [12],
            control_callback=lambda: "cancel" if cancelled else None,
            progress_callback=progress,
        )
    store.get_documents_by_ids.assert_not_called()
    progress.assert_called_once_with(0, 0)


def test_preview_checks_control_between_id_batches(inventory):
    from kiara.ingestion.html_migration import preview_migration

    store, _, _ = inventory
    progress = []
    cancelled = False

    def updated(done, total):
        nonlocal cancelled
        progress.append((done, total))
        cancelled = done == 500

    with pytest.raises(_CancelledError):
        preview_migration(
            {},
            [12],
            control_callback=lambda: "cancel" if cancelled else None,
            progress_callback=updated,
        )
    assert store.get_documents_by_ids.call_count == 1
    assert progress[-1] == (500, 501)


def test_preview_without_callbacks_keeps_existing_store_signature(inventory):
    from kiara.ingestion.html_migration import preview_migration

    store, _, _ = inventory
    result = preview_migration({}, [12])
    assert result["unresolved"] == []
    store.list_sources_for_datasource.assert_called_once_with(12)
