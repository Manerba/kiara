"""Tests fuer _run_repair_rebuild (Collection-Rebuild via Snapshot/Restore)."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

# ---------------------------------------------------------------------------
# Helpers: Reset des globalen _task-State vor jedem Test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_task():
    """Setzt den globalen _task-State vor jedem Test zurueck."""
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        svc._task.update(
            {
                "running": True,
                "action": "repair_rebuild",
                "progress": None,
                "result": None,
            }
        )
    yield
    with svc._lock:
        svc._task.update(
            {
                "running": False,
                "action": None,
                "progress": None,
                "result": None,
            }
        )


def _get_result():
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        return dict(svc._task.get("result") or {})


def _is_running():
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        return svc._task["running"]


# ---------------------------------------------------------------------------
# Mock-Helpers
# ---------------------------------------------------------------------------


def _make_collection_info(
    segments_count=1, points_count=50000, status_value="green", optimizer_status="ok"
):
    info = MagicMock()
    info.segments_count = segments_count
    info.points_count = points_count
    info.status.value = status_value
    info.optimizer_status.status = optimizer_status
    return info


def _make_snapshot_info(name="snap-001.snapshot"):
    snap = MagicMock()
    snap.name = name
    return snap


# Patching der Quell-Module (lazy imports in _run_repair_rebuild)
_P_CONFIG = "kiara.config.load_config"
_P_VS = "kiara.retrieval.vectorstore.VectorStore"
_P_QVS = "kiara.retrieval.vectorstore_qdrant.QdrantVectorStore"
_P_QC = "qdrant_client.QdrantClient"
_P_SP = "qdrant_client.models.SnapshotPriority"


# ---------------------------------------------------------------------------
# Test 1: Keine fragmentierten Collections
# ---------------------------------------------------------------------------


def test_repair_rebuild_no_fragmented():
    """Keine Collection >5 Segmente → sofort 'success'."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 50000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value
        client.get_collection.return_value = _make_collection_info(segments_count=3)

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "success"
        assert "Keine fragmentierten" in result["message"]
        assert not _is_running()


# ---------------------------------------------------------------------------
# Test 2: Happy Path — 1 Collection erfolgreich rebuilt
# ---------------------------------------------------------------------------


def test_repair_rebuild_happy_path():
    """1 fragmentierte Collection, Snapshot + Drop + Restore erfolgreich."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
        patch(_P_QVS) as mock_qvs,
        patch("os.path.isfile", return_value=True),
        patch("os.path.getsize", return_value=1024 * 1024 * 100),
        patch("time.sleep"),
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 50000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value

        info_before = _make_collection_info(segments_count=10)
        info_after = _make_collection_info(segments_count=2, points_count=50000)
        client.get_collection.side_effect = [info_before, info_after, info_after]

        client.create_snapshot.return_value = _make_snapshot_info("snap-001.snapshot")
        mock_qvs._PAYLOAD_INDEXES = {"source": "keyword", "ds_ids": "integer"}

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "success"
        assert "Rebuild" in result["message"]
        assert result["rebuilt"][0]["segments_before"] == 10
        assert result["rebuilt"][0]["segments_after"] == 2

        client.create_snapshot.assert_called_once()
        client.delete_collection.assert_called_once_with(collection_name="kb_nomic-embed-text")
        client.recover_snapshot.assert_called_once()
        client.delete_snapshot.assert_called_once()
        assert not _is_running()


# ---------------------------------------------------------------------------
# Test 3: Snapshot-Datei nicht gefunden → RuntimeError, kein Drop
# ---------------------------------------------------------------------------


def test_repair_rebuild_snapshot_missing():
    """Snapshot-Datei existiert nicht → RuntimeError, kein Drop."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
        patch("os.path.isfile", return_value=False),
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 50000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value
        client.get_collection.return_value = _make_collection_info(segments_count=10)
        client.create_snapshot.return_value = _make_snapshot_info()

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "error"
        assert "nicht gefunden" in result["error"]
        client.delete_collection.assert_not_called()
        assert not _is_running()


# ---------------------------------------------------------------------------
# Test 4: Snapshot 0 Bytes → RuntimeError, kein Drop
# ---------------------------------------------------------------------------


def test_repair_rebuild_snapshot_empty():
    """Snapshot 0 Bytes → RuntimeError, kein Drop."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
        patch("os.path.isfile", return_value=True),
        patch("os.path.getsize", return_value=0),
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 50000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value
        client.get_collection.return_value = _make_collection_info(segments_count=10)
        client.create_snapshot.return_value = _make_snapshot_info()

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "error"
        assert "leer (0 Bytes)" in result["error"]
        client.delete_collection.assert_not_called()
        assert not _is_running()


# ---------------------------------------------------------------------------
# Test 5: Restore schlaegt fehl → error-Result
# ---------------------------------------------------------------------------


def test_repair_rebuild_restore_fails():
    """recover_snapshot wirft Exception → error-Result."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
        patch("os.path.isfile", return_value=True),
        patch("os.path.getsize", return_value=1024 * 1024),
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 50000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value
        client.get_collection.return_value = _make_collection_info(segments_count=10)
        client.create_snapshot.return_value = _make_snapshot_info()
        client.recover_snapshot.side_effect = RuntimeError("Qdrant restore failed")

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "error"
        assert "estore" in result["error"]  # "Restore" oder "restore"
        # Bei Restore-Fehler wird der Snapshot bewusst ERHALTEN (nicht geloescht)
        # fuer manuelles Recovery — delete_snapshot wird nicht aufgerufen
        client.delete_snapshot.assert_not_called()
        assert not _is_running()


# ---------------------------------------------------------------------------
# Test 6: Payload-Indexes werden nach Restore erstellt
# ---------------------------------------------------------------------------


def test_repair_rebuild_payload_indexes():
    """Nach Restore werden alle _PAYLOAD_INDEXES erstellt."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
        patch(_P_QVS) as mock_qvs,
        patch("os.path.isfile", return_value=True),
        patch("os.path.getsize", return_value=1024 * 1024 * 100),
        patch("time.sleep"),
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 50000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value
        info_before = _make_collection_info(segments_count=10)
        info_after = _make_collection_info(segments_count=2, points_count=50000)
        client.get_collection.side_effect = [info_before, info_after, info_after]
        client.create_snapshot.return_value = _make_snapshot_info()

        mock_qvs._PAYLOAD_INDEXES = {
            "source": "keyword",
            "source_key": "keyword",
            "directory": "text",
            "file_type": "keyword",
            "ds_ids": "integer",
            "modified": "datetime",
        }

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "success"

        index_calls = client.create_payload_index.call_args_list
        created_fields = set()
        for c in index_calls:
            # Beide Aufruf-Varianten abdecken (positional/keyword)
            if c.kwargs.get("field_name"):
                created_fields.add(c.kwargs["field_name"])
            elif len(c.args) >= 2:
                created_fields.add(c.args[1])
        assert created_fields == {
            "source",
            "source_key",
            "directory",
            "file_type",
            "ds_ids",
            "modified",
        }


# ---------------------------------------------------------------------------
# Test 7: Collection unter Schwellwert wird NICHT rebuilt
# ---------------------------------------------------------------------------


def test_repair_rebuild_threshold():
    """Collection mit 5000 Chunks + 10 Segmenten → NICHT rebuilt (< 10000)."""
    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_QC) as mock_qc_cls,
    ):
        mock_vs.list_collections.return_value = [
            {"name": "kb_nomic-embed-text", "chunks": 5000, "model": "nomic-embed-text"},
        ]
        client = mock_qc_cls.return_value

        from kiara.admin.services.integrity_service import _run_repair_rebuild

        _run_repair_rebuild()

        result = _get_result()
        assert result["status"] == "success"
        assert "Keine fragmentierten" in result["message"]
        client.get_collection.assert_not_called()
        assert not _is_running()
