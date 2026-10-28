"""Tests fuer den IngestionWorker und die Job-Queue.

Testet:
- enqueue_job: Duplikat-Schutz, Status-Uebergang
- Worker: Stale-Recovery, Poll-Loop
- Cancel/Pause/Resume Logik
- build_datasource_overrides Kompatibilitaet

Alle Tests laufen gegen die geteilte PG-Datenbank (kiara_test).
Datasource-Rows werden per RETURNING id eingefuegt, um FK-Konflikte zu vermeiden.
"""

import asyncio
import json
import tempfile
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from kiara.repositories.datasource import DatasourceRepository
from kiara.shared.db import get_connection
from kiara.web.ingestion_worker import IngestionWorker

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def job_db():
    """PostgreSQL-DB mit ingestion_jobs + datasources — saubere Testdaten."""
    conn = get_connection()
    # Schema sicherstellen (andere Tests koennen datasources droppen+recreaten)
    try:
        conn.execute("SELECT ingestion_status FROM datasources LIMIT 0")
    except Exception:
        conn.rollback()
        conn.execute("DROP TABLE IF EXISTS ingestion_file_logs CASCADE")
        conn.execute("DROP TABLE IF EXISTS ingestion_jobs CASCADE")
        conn.execute("DROP TABLE IF EXISTS datasource_embeddings CASCADE")
        conn.execute("DROP TABLE IF EXISTS datasources CASCADE")
        from kiara.web.schema import create_tables

        create_tables(conn)
        conn.commit()
    # Andere isolierte API-Tests ersetzen ingestion_jobs absichtlich durch
    # ein Minimalschema. S4 muss seine additiven Job-Snapshot-Spalten daher
    # vor jedem Worker-Test idempotent wiederherstellen.
    from kiara.web.index_contract_migration import migrate_index_contract_schema

    migrate_index_contract_schema(conn)
    conn.commit()
    # Bestehende Testdaten aufraeumen (FK-Reihenfolge beachten)
    conn.execute(
        "DELETE FROM ingestion_jobs WHERE datasource_id IN "
        "(SELECT id FROM datasources WHERE name IN ('Test DS', 'Inaktive DS'))"
    )
    conn.execute(
        "DELETE FROM datasource_embeddings WHERE datasource_id IN "
        "(SELECT id FROM datasources WHERE name IN ('Test DS', 'Inaktive DS'))"
    )
    conn.execute("DELETE FROM datasources WHERE name IN ('Test DS', 'Inaktive DS')")
    conn.commit()

    # Testdaten einfuegen und IDs merken
    cur1 = conn.execute(
        "INSERT INTO datasources (name, type, path, is_active) "
        "VALUES ('Test DS', 'local', '/tmp/test', true) RETURNING id"
    )
    ds1_id = cur1.fetchone()["id"]

    cur2 = conn.execute(
        "INSERT INTO datasources (name, type, path, is_active) "
        "VALUES ('Inaktive DS', 'local', '/tmp/test2', false) RETURNING id"
    )
    ds2_id = cur2.fetchone()["id"]
    conn.commit()
    conn.close()
    return {"ds1_id": ds1_id, "ds2_id": ds2_id}


@pytest.fixture
def mock_db(job_db, monkeypatch):
    """Patcht db_query/db_transaction auf PostgreSQL-Verbindungen."""
    from contextlib import contextmanager

    import kiara.shared.db as shared_db
    import kiara.web.db_utils as db_utils

    @contextmanager
    def fake_db_query():
        conn = get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def fake_db_transaction():
        conn = get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_utils, "db_query", fake_db_query)
    monkeypatch.setattr(db_utils, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(shared_db, "db_query", fake_db_query)
    monkeypatch.setattr(shared_db, "db_transaction", fake_db_transaction)
    # Patch am Verwendungsort (ingestion_worker importiert direkt)
    import kiara.web.ingestion_worker as worker_mod

    monkeypatch.setattr(worker_mod, "db_query", fake_db_query)
    monkeypatch.setattr(worker_mod, "db_transaction", fake_db_transaction)

    return job_db


# ---------------------------------------------------------------------------
# enqueue_job Tests
# ---------------------------------------------------------------------------


class TestEnqueueJob:
    def test_enqueue_creates_pending_job(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job

        job_id = enqueue_job(
            ds_id, job_type="incremental", triggered_by="manual", embedding_model="nomic-embed-text"
        )
        assert job_id > 0

        conn = get_connection()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
        conn.close()
        assert row["status"] == "pending"
        assert row["datasource_id"] == ds_id
        assert row["job_type"] == "incremental"
        assert row["embedding_model"] == "nomic-embed-text"

    def test_enqueue_rejects_empty_model(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job

        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            enqueue_job(ds_id, embedding_model="")

    def test_enqueue_rejects_duplicate(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job

        enqueue_job(ds_id, embedding_model="nomic-embed-text")
        with pytest.raises(ValueError, match="bereits einen aktiven Job"):
            enqueue_job(ds_id, embedding_model="nomic-embed-text")

    def test_enqueue_rejects_inactive_datasource(self, mock_db):
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        conn.execute("UPDATE datasources SET is_active = false WHERE id = %s", (ds_id,))
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import enqueue_job

        with pytest.raises(ValueError, match="deaktiviert"):
            enqueue_job(ds_id, embedding_model="nomic-embed-text")

    def test_enqueue_allows_after_completion(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job

        job_id = enqueue_job(ds_id, embedding_model="nomic-embed-text")

        conn = get_connection()
        conn.execute("UPDATE ingestion_jobs SET status = 'success' WHERE id = %s", (job_id,))
        conn.commit()
        conn.close()

        job_id2 = enqueue_job(ds_id, embedding_model="nomic-embed-text")
        assert job_id2 > job_id

    def test_enqueue_full_reingest(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job

        job_id = enqueue_job(
            ds_id, job_type="full", triggered_by="api", embedding_model="nomic-embed-text"
        )

        conn = get_connection()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
        conn.close()
        assert row["job_type"] == "full"
        assert row["triggered_by"] == "api"


class TestContractDiscoveryRecovery:
    def test_transient_claim_failure_leaves_job_pending(self, mock_db):
        from contextlib import contextmanager

        from kiara.retrieval.index_compatibility import (
            IndexContractUnavailableError,
            IndexVectorKind,
            make_index_target,
        )
        from kiara.web.ingestion_worker import enqueue_job

        ds_id = mock_db["ds1_id"]
        job_id = enqueue_job(ds_id, embedding_model="nomic-embed-text")
        conn = get_connection()
        before = conn.execute(
            "SELECT finished_at, error_message FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        conn.close()
        target = make_index_target(
            {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
            collection_name="kb_nomic-embed-text",
            vector_kind=IndexVectorKind.DENSE,
            requested_model="nomic-embed-text",
        )
        unavailable = IndexContractUnavailableError(
            "index_current_profile_not_verified",
            "job_claim",
            target,
            "contract_discovery_request_failed",
        )

        @contextmanager
        def unavailable_guard(*_args, **_kwargs):
            raise unavailable
            yield  # pragma: no cover

        worker = IngestionWorker()
        with patch(
            "kiara.retrieval.index_compatibility.guard_job_contract",
            unavailable_guard,
        ):
            assert worker._fetch_next_job() is None

        conn = get_connection()
        row = conn.execute(
            "SELECT status, finished_at, error_message FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        conn.close()
        assert row["status"] == "pending"
        assert row["finished_at"] == before["finished_at"]
        assert row["error_message"] == before["error_message"]


# ---------------------------------------------------------------------------
# Worker Cancel/Pause Tests
# ---------------------------------------------------------------------------


class TestWorkerControl:
    def test_control_read_does_not_wait_for_job_row_lock(self, mock_db):
        """Vector-Guard-Lock darf den verschachtelten Control-Read nicht blockieren."""
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        job_id = conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, requested_action) "
            "VALUES (%s, 'running', NOW(), 'cancel') RETURNING id",
            (ds_id,),
        ).fetchone()["id"]
        conn.commit()

        worker = IngestionWorker()
        worker._stop = threading.Event()
        check_control = worker._build_check_control(job_id)

        # Simuliert guard_index_for_active_job(): Die aeussere Transaktion
        # haelt den Job-Datensatz, waehrend der Control-Callback ihn liest.
        conn.execute(
            "SELECT id FROM ingestion_jobs WHERE id = %s FOR UPDATE",
            (job_id,),
        )

        result = []
        reader = threading.Thread(target=lambda: result.append(check_control()))
        completed_while_locked = False
        try:
            reader.start()
            reader.join(timeout=1.0)
            completed_while_locked = not reader.is_alive()
        finally:
            conn.rollback()
            conn.close()
            reader.join(timeout=2.0)

        assert completed_while_locked is True
        assert result == ["cancel"]

    def test_request_cancel_pending_job(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job, request_cancel

        job_id = enqueue_job(ds_id, embedding_model="nomic-embed-text")

        result = request_cancel(job_id)
        assert result is True

        conn = get_connection()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
        conn.close()
        assert row["status"] == "cancelled"

    def test_request_cancel_running_job(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import request_cancel

        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) "
            "VALUES (%s, 'running', NOW()) RETURNING id",
            (ds_id,),
        )
        job_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        result = request_cancel(job_id)
        assert result is True

        conn = get_connection()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
        conn.close()
        assert row["requested_action"] == "cancel"

    def test_request_cancel_stale_running_job_finalizes_cancelled(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import request_cancel

        stale_started_at = datetime.now() - timedelta(hours=1)
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at, stats_json) "
            "VALUES (%s, 'running', %s, '{}'::jsonb) RETURNING id",
            (ds_id, stale_started_at),
        )
        job_id = cur.fetchone()["id"]
        conn.execute(
            "UPDATE datasources SET ingestion_status = 'running' WHERE id = %s",
            (ds_id,),
        )
        conn.commit()
        conn.close()

        result = request_cancel(job_id)
        assert result is True

        conn = get_connection()
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
        ds_row = conn.execute(
            "SELECT ingestion_status FROM datasources WHERE id = %s", (ds_id,)
        ).fetchone()
        conn.close()

        assert row["status"] == "cancelled"
        assert row["requested_action"] is None
        assert row["finished_at"] is not None
        stats_json = row["stats_json"]
        if isinstance(stats_json, str):
            stats_json = json.loads(stats_json)
        assert "stale running Job" in stats_json["abort_reason"]
        assert ds_row["ingestion_status"] == "idle"

    def test_request_cancel_nonexistent_returns_false(self, mock_db):
        from kiara.web.ingestion_worker import request_cancel

        assert request_cancel(9999) is False

    def test_request_pause_running_job(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import is_job_paused, request_pause

        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) "
            "VALUES (%s, 'running', NOW()) RETURNING id",
            (ds_id,),
        )
        job_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        assert request_pause(job_id) is True
        assert is_job_paused(job_id) is True

    def test_request_resume(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import is_job_paused, request_pause, request_resume

        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) "
            "VALUES (%s, 'running', NOW()) RETURNING id",
            (ds_id,),
        )
        job_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        request_pause(job_id)
        assert is_job_paused(job_id) is True

        assert request_resume(job_id) is True
        assert is_job_paused(job_id) is False

    def test_request_pause_nonrunning_returns_false(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.web.ingestion_worker import enqueue_job, request_pause

        job_id = enqueue_job(ds_id, embedding_model="nomic-embed-text")
        assert request_pause(job_id) is False


class TestParallelScan:
    def test_parallel_scan_none_path_sets_scan_error(self):
        from kiara.web.ingestion_worker import IngestionWorker

        class FailingConnector:
            @staticmethod
            def prepare_and_scan(*args, **kwargs):
                return None

        worker = IngestionWorker()
        worker._set_job_phase = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            scan_info = worker._start_parallel_scan(
                job_id=77,
                config={"datasource": {}},
                connector_cls=FailingConnector,
                conn_config={},
                ds_row={"id": 5},
                cache_path=tmpdir,
                check_control=lambda: None,
            )
            assert scan_info["scan_complete"].wait(timeout=2.0)

        assert isinstance(scan_info["scan_error"][0], RuntimeError)
        assert "keinen Cache-Pfad" in str(scan_info["scan_error"][0])


class TestDatasourceDeleteJobGuards:
    def test_delete_datasource_blocks_pending_job_before_soft_delete(self, mock_db):
        ds_id = mock_db["ds1_id"]
        from kiara.admin.services.datasource_service import delete_datasource_full
        from kiara.web.ingestion_worker import enqueue_job

        enqueue_job(ds_id, embedding_model="nomic-embed-text")

        with pytest.raises(ValueError, match="aktiven Ingestion-Job"):
            asyncio.run(delete_datasource_full(ds_id))

        conn = get_connection()
        row = conn.execute(
            "SELECT is_active, name FROM datasources WHERE id = %s",
            (ds_id,),
        ).fetchone()
        conn.close()

        assert row["is_active"] is True
        assert not row["name"].startswith("[DELETING]")


# ---------------------------------------------------------------------------
# Backend-Warmup
# ---------------------------------------------------------------------------


class TestBackendWarmup:
    def test_failed_warmup_stats_survive_error_finalize(self):
        from kiara.backend_warmup import WarmupResult

        worker = IngestionWorker()
        worker._set_job_phase = MagicMock()
        worker._set_warmup_stats = MagicMock()
        worker._finalize_error = MagicMock()

        warmup_detail = {
            "category": "embedding",
            "model": "nomic-embed-text",
            "url": "http://gpu-server.local:11434",
            "ok": False,
            "error": "GPU-Operation blockiert (status code: 503)",
            "duration_s": 27.7,
        }
        warmup_result = WarmupResult(
            success=False,
            duration_s=27.7,
            details=[warmup_detail],
        )

        with patch("kiara.backend_warmup.build_warmup_tasks", return_value=[MagicMock()]):
            with patch("kiara.backend_warmup.BackendWarmer") as warmer_cls:
                with patch("kiara.web.ingestion_worker.time.time", return_value=1028):
                    warmer_cls.return_value.warm.return_value = warmup_result
                    result = worker._run_warmup(
                        379,
                        28,
                        {"id": 28},
                        "nomic-embed-text",
                        {},
                        1000,
                        lambda: None,
                    )

        assert result is None
        worker._set_warmup_stats.assert_called_once_with(379, warmup_result)
        args = worker._finalize_error.call_args.args
        kwargs = worker._finalize_error.call_args.kwargs
        assert args[:4] == (
            379,
            28,
            "Backend-Warmup fehlgeschlagen: "
            "embedding(nomic-embed-text): GPU-Operation blockiert (status code: 503)",
            28,
        )
        assert kwargs["embedding_model"] == "nomic-embed-text"
        assert kwargs["stats"]["warmup"]["success"] is False
        assert kwargs["stats"]["warmup"]["duration_s"] == 27.7
        assert kwargs["stats"]["warmup"]["details"] == [warmup_detail]


class TestIndexCompatibilityAbort:
    def test_worker_finalizes_pipeline_contract_abort_as_error(self, mock_db):
        import threading

        from kiara.web.ingestion_worker import enqueue_job

        ds_id = mock_db["ds1_id"]
        job_id = enqueue_job(ds_id, embedding_model="guarded-model")
        diagnostic = {
            "error": "index_compatibility_blocked",
            "reason_code": "index_contract_unverified",
            "operation": "document_embedding",
            "current_state": "unverified",
        }
        stats = {
            "errors": 1,
            "processed_files": 0,
            "abort_reason": (
                "Index-Kompatibilitaetsguard: index_contract_unverified "
                "(stored contract is unverified)"
            ),
            "index_compatibility_error": diagnostic,
        }
        worker = IngestionWorker()
        worker._stop = threading.Event()
        worker._finalize_success = MagicMock()

        with patch("kiara.web.ingestion_worker.safe_refresh"):
            worker._resolve_job_outcome(
                job_id,
                ds_id,
                None,
                "guarded-model",
                stats,
                3,
                lambda: None,
            )

        conn = get_connection()
        job = conn.execute(
            "SELECT status, error_message, stats_json FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        datasource = conn.execute(
            "SELECT ingestion_status FROM datasources WHERE id = %s",
            (ds_id,),
        ).fetchone()
        conn.close()
        stored_stats = job["stats_json"]
        if isinstance(stored_stats, str):
            stored_stats = json.loads(stored_stats)

        assert job["status"] == "error"
        assert "index_contract_unverified" in job["error_message"]
        assert datasource["ingestion_status"] == "error"
        assert stored_stats["index_compatibility_error"] == diagnostic
        worker._finalize_success.assert_not_called()

    def test_model_switch_cleanup_guard_aborts_pipeline_and_worker(
        self,
        mock_db,
        tmp_path,
    ):
        """Der vormals nicht-kritische Abschluss-Catch darf nie Success liefern."""
        import threading

        from kiara.ingestion.pipeline import run_ingestion
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexVectorKind,
            make_index_target,
        )
        from kiara.web.ingestion_worker import enqueue_job

        ds_id = mock_db["ds1_id"]
        job_id = enqueue_job(ds_id, embedding_model="guarded-model")
        config = {
            "datasource": {"path": str(tmp_path)},
            "vectordb": {
                "path": str(tmp_path / "vectors"),
                "qdrant_url": "http://qdrant.test:6333",
            },
            "ingestion": {"batch_size": 10},
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-new"}},
        }
        target = make_index_target(
            config,
            collection_name="colbert_guarded",
            vector_kind=IndexVectorKind.MULTI_VECTOR,
            requested_model="colbert-old",
        )
        error = IndexCompatibilityError(
            "index_contract_mismatch",
            "colbert_model_switch_cleanup",
            target,
            "injected model-switch compatibility failure",
            current_state="mismatch",
        )
        hash_store = MagicMock()
        vector_store = MagicMock()
        chunk_ds = MagicMock()

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=vector_store),
            patch("kiara.ingestion.pipeline.ChunkDatasources", return_value=chunk_ds),
            patch("kiara.ingestion.pipeline._check_ddu_preflight"),
            patch(
                "kiara.ingestion.pipeline._load_and_diff_files",
                return_value=([], [], {}, [], hash_store, {}),
            ),
            patch(
                "kiara.ingestion.pipeline._cleanup_colbert_model_switch_if_ready",
                side_effect=error,
            ) as cleanup,
        ):
            stats = run_ingestion(
                config,
                datasource_id=ds_id,
                embedding_model="guarded-model",
            )

        assert cleanup.call_count == 1
        assert stats["index_compatibility_error"] == error.to_dict()
        assert stats["error_details"][0]["reason_code"] == error.reason_code
        assert stats["error_details"][0]["consumer"] == "colbert_model_switch_cleanup"
        hash_store.set_hash_mode.assert_not_called()
        vector_store.add.assert_not_called()
        vector_store.delete_by_ids.assert_not_called()
        vector_store.update_ds_ids.assert_not_called()

        worker = IngestionWorker()
        worker._stop = threading.Event()
        worker._finalize_success = MagicMock()
        with patch("kiara.web.ingestion_worker.safe_refresh"):
            worker._resolve_job_outcome(
                job_id,
                ds_id,
                config,
                "guarded-model",
                stats,
                1,
                lambda: None,
            )

        conn = get_connection()
        job = conn.execute(
            "SELECT status, error_message, stats_json FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        conn.close()
        stored_stats = job["stats_json"]
        if isinstance(stored_stats, str):
            stored_stats = json.loads(stored_stats)

        assert job["status"] == "error"
        assert error.reason_code in job["error_message"]
        assert stored_stats["index_compatibility_error"] == error.to_dict()
        assert stored_stats["error_details"][0]["reason_code"] == error.reason_code
        worker._finalize_success.assert_not_called()

    def test_colbert_contract_abort_stops_next_file_and_finalizes_job_error(
        self,
        mock_db,
        tmp_path,
    ):
        """Pipeline+Worker: ColBERT-Guardfehler kann nie in Success enden."""
        import threading

        from kiara.ingestion.pipeline import run_ingestion
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexVectorKind,
            make_index_target,
        )
        from kiara.web.ingestion_worker import enqueue_job

        ds_id = mock_db["ds1_id"]
        job_id = enqueue_job(ds_id, embedding_model="guarded-model")
        files = ["/data/first.txt", "/data/second.txt"]
        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt"],
                "exclude_patterns": [],
            },
            "vectordb": {
                "path": str(tmp_path / "vectors"),
                "qdrant_url": "http://qdrant.test:6333",
            },
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-test"}},
            "graph": {"enabled": False},
        }
        target = make_index_target(
            config,
            collection_name="colbert_guarded",
            vector_kind=IndexVectorKind.MULTI_VECTOR,
            requested_model="colbert-test",
        )
        error = IndexCompatibilityError(
            "index_contract_unverified",
            "colbert_upsert",
            target,
            "injected ColBERT contract failure",
            current_state="unverified",
        )
        hash_store = MagicMock()
        store = MagicMock()
        chunk_ds = MagicMock()
        chunk_ds.chunks_exist.return_value = set()

        with (
            patch(
                "kiara.ingestion.pipeline._load_and_diff_files",
                return_value=(
                    files,
                    files,
                    {path: (f"hash-{index}", 100) for index, path in enumerate(files)},
                    [],
                    hash_store,
                    {},
                ),
            ),
            patch("kiara.ingestion.pipeline.VectorStore", return_value=store),
            patch("kiara.ingestion.pipeline.ChunkDatasources", return_value=chunk_ds),
            patch(
                "kiara.ingestion.pipeline.get_embedding_model",
                return_value="guarded-model",
            ),
            patch(
                "kiara.ingestion.runtime._process_single_file",
                return_value=(
                    ["chunk"],
                    [{"source": files[0]}],
                    ["id1"],
                    "full text",
                    [(0, 5)],
                ),
            ) as process_file,
            patch("kiara.ingestion.runtime._write_to_elasticsearch"),
            patch("kiara.ingestion.runtime._persist_graph_and_associations"),
            patch(
                "kiara.ingestion.runtime._sync_colbert_chunks",
                side_effect=error,
            ) as colbert_sync,
        ):
            stats = run_ingestion(
                config,
                datasource_id=ds_id,
                ds_overrides={"colbert_enabled": True},
                embedding_model="guarded-model",
            )

        assert stats["index_compatibility_error"] == error.to_dict()
        assert "index_contract_unverified" in stats["abort_reason"]
        assert process_file.call_count == 1
        assert colbert_sync.call_count == 1
        assert store.add.call_count == 1

        worker = IngestionWorker()
        worker._stop = threading.Event()
        worker._finalize_success = MagicMock()
        with patch("kiara.web.ingestion_worker.safe_refresh"):
            worker._resolve_job_outcome(
                job_id,
                ds_id,
                config,
                "guarded-model",
                stats,
                3,
                lambda: None,
            )

        conn = get_connection()
        job = conn.execute(
            "SELECT status, stats_json FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        conn.close()
        stored_stats = job["stats_json"]
        if isinstance(stored_stats, str):
            stored_stats = json.loads(stored_stats)
        assert job["status"] == "error"
        assert stored_stats["index_compatibility_error"] == error.to_dict()
        worker._finalize_success.assert_not_called()


# ---------------------------------------------------------------------------
# Stale Recovery
# ---------------------------------------------------------------------------


class TestStaleRecovery:
    def test_recover_marks_running_as_error(self, mock_db):
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, 'running')",
            (ds_id,),
        )
        conn.execute(
            "UPDATE datasources SET ingestion_status = 'running' WHERE id = %s",
            (ds_id,),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND status = 'error'",
            (ds_id,),
        ).fetchone()
        ds = conn.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
        conn.close()

        assert job is not None
        assert "Restart" in job["error_message"]
        assert ds["ingestion_status"] == "idle"


# ---------------------------------------------------------------------------
# build_datasource_overrides
# ---------------------------------------------------------------------------


def _sql_col_type(v):
    """SQL-Spaltentyp fuer einen Python-Wert. bool vor int pruefen,
    weil bool Subclass von int ist (isinstance(False, int) == True).
    """
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return "INTEGER"
    return "TEXT"


class TestBuildOverrides:
    def _make_row(self, **kwargs):
        defaults = {
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": "auto",
            "ds_ddu_enabled": False,
            "ds_ddu_ocr_images": 1,
            "ds_vlm_enabled": False,
            "ds_index_attachments": 1,
            "ds_accept_all_text": 0,
            "ds_contextual_retrieval": 0,
            "ds_enrichment_prompt_id": None,
            "ds_enrichment_backend_id": None,
            "restriction_level": "anonymize",
        }
        defaults.update(kwargs)
        col_defs = ", ".join(f"{k} {_sql_col_type(v)}" for k, v in defaults.items())
        placeholders = ", ".join("%s" for _ in defaults)
        col_names = ", ".join(defaults.keys())
        conn = get_connection()
        conn.execute("DROP TABLE IF EXISTS _test_make_row CASCADE")
        conn.execute(f"CREATE TABLE _test_make_row ({col_defs})")
        conn.execute(
            f"INSERT INTO _test_make_row ({col_names}) VALUES ({placeholders})",
            tuple(defaults.values()),
        )
        row = conn.execute("SELECT * FROM _test_make_row").fetchone()
        conn.execute("DROP TABLE IF EXISTS _test_make_row CASCADE")
        conn.commit()
        conn.close()
        return row

    def test_defaults_no_feature_overrides(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row()
        overrides = build_datasource_overrides(row)
        # restriction_level wird immer gesetzt, aber keine Feature-Overrides
        assert overrides is not None
        assert "chunk_size" not in overrides
        assert "ddu_enabled" not in overrides
        assert "vlm_enabled" not in overrides
        assert "index_attachments" not in overrides
        assert "accept_all_text" not in overrides
        assert "contextual_retrieval" not in overrides

    def test_ddu_enabled(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(ds_ddu_enabled=True)
        overrides = build_datasource_overrides(row)
        assert overrides is not None
        assert overrides["ddu_enabled"] is True

    def test_chunk_strategy(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(ds_chunk_strategy="prose")
        overrides = build_datasource_overrides(row)
        assert overrides["chunk_strategy"] == "prose"

    def test_custom_chunk_size(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(ds_chunk_size=500, ds_chunk_overlap=100)
        overrides = build_datasource_overrides(row)
        assert overrides["chunk_size"] == 500
        assert overrides["chunk_overlap"] == 100

    def test_index_attachments_disabled(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(ds_index_attachments=0)
        overrides = build_datasource_overrides(row)
        assert overrides is not None
        assert overrides["index_attachments"] is False

    def test_index_attachments_enabled_default(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(ds_index_attachments=1)
        overrides = build_datasource_overrides(row)
        # restriction_level wird immer gesetzt, aber index_attachments nicht
        assert overrides is not None
        assert "index_attachments" not in overrides

    def test_index_attachments_missing_column(self):
        """Alte DB-Rows ohne ds_index_attachments → Fallback auf aktiv."""
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        # Row ohne ds_index_attachments Spalte (aber mit anderen neuen Feldern)
        defaults = {
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": "auto",
            "ds_ddu_enabled": False,
            "ds_ddu_ocr_images": 1,
            "ds_vlm_enabled": False,
            "ds_accept_all_text": 0,
            "ds_contextual_retrieval": 0,
            "ds_enrichment_prompt_id": None,
            "ds_enrichment_backend_id": None,
            "restriction_level": "anonymize",
        }
        col_defs = ", ".join(f"{k} {_sql_col_type(v)}" for k, v in defaults.items())
        placeholders = ", ".join("%s" for _ in defaults)
        col_names = ", ".join(defaults.keys())
        conn = get_connection()
        conn.execute("DROP TABLE IF EXISTS _test_missing_col CASCADE")
        conn.execute(f"CREATE TABLE _test_missing_col ({col_defs})")
        conn.execute(
            f"INSERT INTO _test_missing_col ({col_names}) VALUES ({placeholders})",
            tuple(defaults.values()),
        )
        row = conn.execute("SELECT * FROM _test_missing_col").fetchone()
        conn.execute("DROP TABLE IF EXISTS _test_missing_col CASCADE")
        conn.commit()
        conn.close()

        overrides = build_datasource_overrides(row)
        # restriction_level wird immer gesetzt, aber index_attachments nicht
        assert overrides is not None
        assert "index_attachments" not in overrides


# ---------------------------------------------------------------------------
# _load_job_context — Regressionstest fuer load_connector_config()-Aufruferpfad
# ---------------------------------------------------------------------------


class TestLoadJobContext:
    """Verifiziert, dass _load_job_context() den neuen load_connector_config() nutzt."""

    @patch("kiara.web.ingestion_worker.load_connector_config")
    @patch("kiara.config.load_config", return_value={"key": "val"})
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch.object(DatasourceRepository, "get_by_id")
    def test_load_job_context_uses_load_connector_config_for_manual_jobs(
        self, mock_get_ds, mock_ensure, mock_get_conn, mock_load_cfg, mock_load_cc
    ):
        ds_row = {"id": 1, "type": "local", "name": "Test"}
        mock_get_ds.return_value = ds_row

        connector_cls = MagicMock()
        connector_cls.SCAN_CACHE_MAX_AGE = 300
        connector_cls.effective_path.return_value = "/tmp/cache"
        mock_get_conn.return_value = connector_cls
        mock_load_cc.return_value = {"path": "/data"}

        job_row = {
            "triggered_by": "manual",
            "embedding_model": "nomic-embed-text",
            "job_type": "incremental",
            "stats_json": {},
        }

        worker = IngestionWorker()
        with patch("os.path.exists", return_value=True), patch("os.remove") as mock_remove:
            ctx = worker._load_job_context(job_row, 1, 99)

        mock_load_cc.assert_called_once_with(ds_row, connector_cls)
        assert ctx["conn_config"] == {"path": "/data"}
        connector_cls.effective_path.assert_called_once_with({"path": "/data"}, dict(ds_row))
        mock_remove.assert_called_once()

    @patch("kiara.web.ingestion_worker.load_connector_config")
    @patch("kiara.config.load_config", return_value={"key": "val"})
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch.object(DatasourceRepository, "get_by_id")
    def test_load_job_context_skips_scan_cache_invalidation_for_file_connectors(
        self, mock_get_ds, mock_ensure, mock_get_conn, mock_load_cfg, mock_load_cc
    ):
        """SMB/Local/WebDAV haben SCAN_CACHE_MAX_AGE=0 — kein os.remove auf Mountpoint.

        Regression: Frueher pruefte der Code nur ``hasattr(cls, "SCAN_CACHE_MAX_AGE")``,
        was wegen des Default-Werts in ConnectorBase IMMER True war. Damit wurde
        fuer SMB ``os.remove("/mnt/kiara-ds-X/.../.scan_done")`` aufgerufen, was
        bei nicht beschreibbarem Mountpoint einen PermissionError warf, der den
        gesamten Job mit Errno 13 abbrach.
        """
        ds_row = {"id": 14, "type": "smb", "name": "SMB-DS"}
        mock_get_ds.return_value = ds_row

        connector_cls = MagicMock()
        connector_cls.SCAN_CACHE_MAX_AGE = 0  # SMB/Local/WebDAV haben keinen Scan-Cache
        connector_cls.effective_path.return_value = "/mnt/kiara-ds-14"
        mock_get_conn.return_value = connector_cls
        mock_load_cc.return_value = {"server": "fileserver", "share": "data"}

        worker = IngestionWorker()
        with patch("os.remove") as mock_remove:
            worker._load_job_context(
                {
                    "triggered_by": "manual",
                    "embedding_model": "nomic-embed-text",
                    "job_type": "incremental",
                    "stats_json": {},
                },
                14,
                99,
            )

        mock_remove.assert_not_called()
        connector_cls.effective_path.assert_not_called()

    @patch("kiara.web.ingestion_worker.load_connector_config")
    @patch("kiara.config.load_config", return_value={"key": "val"})
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch.object(DatasourceRepository, "get_by_id")
    def test_load_job_context_swallows_permission_error_on_scan_cache_remove(
        self, mock_get_ds, mock_ensure, mock_get_conn, mock_load_cfg, mock_load_cc
    ):
        """Defensive: PermissionError bei os.remove darf den Job nicht abbrechen."""
        ds_row = {"id": 5, "type": "imap", "name": "Mail-DS"}
        mock_get_ds.return_value = ds_row

        connector_cls = MagicMock()
        connector_cls.SCAN_CACHE_MAX_AGE = 86400
        connector_cls.effective_path.return_value = "/var/lib/kiara/mail_cache/ds5"
        mock_get_conn.return_value = connector_cls
        mock_load_cc.return_value = {"server": "imap.example.com"}

        worker = IngestionWorker()
        with patch("os.remove", side_effect=PermissionError(13, "Permission denied")):
            ctx = worker._load_job_context(
                {
                    "triggered_by": "manual",
                    "embedding_model": "nomic-embed-text",
                    "job_type": "incremental",
                    "stats_json": {},
                },
                5,
                99,
            )

        assert ctx["connector_cls"] is connector_cls


# ---------------------------------------------------------------------------
# Luecke 3: Worker-Marker _kiara_enforce_persisted_colbert_model
# ---------------------------------------------------------------------------


class TestWorkerColbertMarker:
    """Worker-Entry-Point setzt _kiara_enforce_persisted_colbert_model = True."""

    @patch("kiara.web.ingestion_worker.load_connector_config")
    @patch(
        "kiara.config.load_config",
        return_value={
            "key": "val",
            "retrieval": {"colbert": {"enabled": True}},
        },
    )
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch.object(DatasourceRepository, "get_by_id")
    def test_load_job_context_sets_colbert_marker(
        self, mock_get_ds, mock_ensure, mock_get_conn, mock_load_cfg, mock_load_cc
    ):
        ds_row = {"id": 1, "type": "local", "name": "Test"}
        mock_get_ds.return_value = ds_row
        connector_cls = MagicMock()
        connector_cls.SCAN_CACHE_MAX_AGE = 300
        connector_cls.effective_path.return_value = "/tmp/cache"
        mock_get_conn.return_value = connector_cls
        mock_load_cc.return_value = {"path": "/data"}

        worker = IngestionWorker()
        with patch("os.path.exists", return_value=True), patch("os.remove"):
            ctx = worker._load_job_context(
                {
                    "triggered_by": "manual",
                    "embedding_model": "nomic-embed-text",
                    "job_type": "incremental",
                    "stats_json": {},
                },
                1,
                99,
            )

        assert ctx["config"].get("_kiara_enforce_persisted_colbert_model") is True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_worker_returns_same_instance(self):
        import kiara.web.ingestion_worker as mod

        mod._worker = None
        w1 = mod.get_worker()
        w2 = mod.get_worker()
        assert w1 is w2
        mod._worker = None
