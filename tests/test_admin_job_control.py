"""Tests fuer Sprint 05: Job-Steuerung + Schedule Admin-Funktionen.

Testet jobs.start, .start_full, .cancel, .pause, .resume, .cleanup_history,
.delete, .schedule.set, .schedule.delete — 15 Unit-Tests mit Mocking.
"""

import asyncio
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# jobs.start Tests
# ---------------------------------------------------------------------------


class TestJobsStart:
    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.web.ingestion_worker.enqueue_job")
    @patch("kiara.config.load_config")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_start_incremental(self, mock_get, mock_config, mock_enqueue, mock_refresh):
        """Test 17: enqueue_job mit mode='incremental'."""
        mock_get.return_value = {"id": 5, "is_active": True, "name": "DS5"}
        mock_config.return_value = {"embedding_model": "nomic-embed-text"}
        mock_enqueue.return_value = 100

        from kiara.admin.functions.jobs import jobs_start

        result = _run(jobs_start(MOCK_USER, ds_id=5))

        assert result.success is True
        assert result.data["job_id"] == 100
        assert result.data["mode"] == "incremental"
        mock_enqueue.assert_called_once_with(
            5,
            job_type="incremental",
            triggered_by="chat",
            embedding_model="nomic-embed-text",
        )

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_start_inactive_ds_fails(self, mock_get):
        """Test 18: Fehler bei is_active=False."""
        mock_get.return_value = {"id": 5, "is_active": False, "name": "DS5"}

        from kiara.admin.functions.jobs import jobs_start

        result = _run(jobs_start(MOCK_USER, ds_id=5))

        assert result.success is False
        assert "deaktiviert" in result.message

    @patch("kiara.web.ingestion_worker.enqueue_job")
    @patch("kiara.config.load_config")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_start_duplicate_job_fails(self, mock_get, mock_config, mock_enqueue):
        """Test 19: ValueError bei bereits aktivem Job → AdminResult.fail."""
        mock_get.return_value = {"id": 5, "is_active": True, "name": "DS5"}
        mock_config.return_value = {"embedding_model": "nomic-embed-text"}
        mock_enqueue.side_effect = ValueError("DS 5 hat bereits einen aktiven Job (#99)")

        from kiara.admin.functions.jobs import jobs_start

        result = _run(jobs_start(MOCK_USER, ds_id=5))

        assert result.success is False
        assert "aktiven Job" in result.message


# ---------------------------------------------------------------------------
# jobs.start_full Tests
# ---------------------------------------------------------------------------


class TestJobsStartFull:
    def test_start_full_requires_confirmation(self):
        """Test 20: requires_confirmation=True wird geprueft."""
        from kiara.admin.registry import admin_registry

        func = admin_registry.get("jobs.start_full")
        assert func is not None
        assert func.requires_confirmation is True


# ---------------------------------------------------------------------------
# jobs.cancel Tests
# ---------------------------------------------------------------------------


class TestJobsCancel:
    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.web.ingestion_worker.request_cancel")
    def test_cancel_running_job(self, mock_cancel, mock_refresh):
        """Test 21: request_cancel aufgerufen + Cache refresh."""
        mock_cancel.return_value = True

        from kiara.admin.functions.jobs import jobs_cancel

        result = _run(jobs_cancel(MOCK_USER, job_id=42))

        assert result.success is True
        assert result.data["status"] == "cancelling"
        mock_cancel.assert_called_once_with(42)
        mock_refresh.assert_called_once()

    @patch("kiara.web.ingestion_worker.request_cancel")
    def test_cancel_nonrunning_fails(self, mock_cancel):
        """Test 22: request_cancel returns False → Fehler."""
        mock_cancel.return_value = False

        from kiara.admin.functions.jobs import jobs_cancel

        result = _run(jobs_cancel(MOCK_USER, job_id=42))

        assert result.success is False
        assert "kann nicht abgebrochen werden" in result.message


# ---------------------------------------------------------------------------
# jobs.pause Tests
# ---------------------------------------------------------------------------


class TestJobsPause:
    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.web.ingestion_worker.request_pause")
    @patch("kiara.web.ingestion_worker.is_job_paused")
    @patch("kiara.web.ingestion_worker.get_running_job_id")
    def test_pause_running_job(self, mock_running, mock_paused, mock_pause, mock_refresh):
        """Test 23: request_pause aufgerufen."""
        mock_running.return_value = 42
        mock_paused.return_value = False
        mock_pause.return_value = True

        from kiara.admin.functions.jobs import jobs_pause

        result = _run(jobs_pause(MOCK_USER, job_id=42))

        assert result.success is True
        assert result.data["paused"] is True
        mock_pause.assert_called_once_with(42)

    @patch("kiara.web.ingestion_worker.is_job_paused")
    @patch("kiara.web.ingestion_worker.get_running_job_id")
    def test_pause_already_paused_fails(self, mock_running, mock_paused):
        """Test 24: Fehler wenn is_job_paused=True."""
        mock_running.return_value = 42
        mock_paused.return_value = True

        from kiara.admin.functions.jobs import jobs_pause

        result = _run(jobs_pause(MOCK_USER, job_id=42))

        assert result.success is False
        assert "bereits pausiert" in result.message


# ---------------------------------------------------------------------------
# jobs.resume Tests
# ---------------------------------------------------------------------------


class TestJobsResume:
    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.web.ingestion_worker.request_resume")
    @patch("kiara.web.ingestion_worker.is_job_paused")
    def test_resume_paused_job(self, mock_paused, mock_resume, mock_refresh):
        """Test 25: request_resume aufgerufen."""
        mock_paused.return_value = True
        mock_resume.return_value = True

        from kiara.admin.functions.jobs import jobs_resume

        result = _run(jobs_resume(MOCK_USER, job_id=42))

        assert result.success is True
        assert result.data["paused"] is False
        mock_resume.assert_called_once_with(42)

    @patch("kiara.web.ingestion_worker.is_job_paused")
    def test_resume_not_paused_fails(self, mock_paused):
        """Test 26: Fehler wenn is_job_paused=False."""
        mock_paused.return_value = False

        from kiara.admin.functions.jobs import jobs_resume

        result = _run(jobs_resume(MOCK_USER, job_id=42))

        assert result.success is False
        assert "nicht pausiert" in result.message


# ---------------------------------------------------------------------------
# jobs.delete Tests
# ---------------------------------------------------------------------------


class TestJobsDelete:
    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.shared.db.db_transaction")
    @patch("kiara.repositories.job.IngestionJobRepository.get_by_id")
    def test_delete_completed_job(self, mock_get, mock_db_tx, mock_refresh):
        """Test 27: File-Logs + Job geloescht."""
        mock_get.return_value = {"id": 42, "status": "completed"}
        mock_conn = MagicMock()
        mock_db_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.admin.functions.jobs import jobs_delete

        result = _run(jobs_delete(MOCK_USER, job_id=42))

        assert result.success is True
        assert mock_conn.execute.call_count == 2  # file_logs + jobs
        mock_refresh.assert_called_once()

    @patch("kiara.repositories.job.IngestionJobRepository.get_by_id")
    def test_delete_running_job_fails(self, mock_get):
        """Test 28: Fehler bei status='running'."""
        mock_get.return_value = {"id": 42, "status": "running"}

        from kiara.admin.functions.jobs import jobs_delete

        result = _run(jobs_delete(MOCK_USER, job_id=42))

        assert result.success is False
        assert "noch aktiv" in result.message


# ---------------------------------------------------------------------------
# jobs.cleanup_history Tests
# ---------------------------------------------------------------------------


class TestJobsCleanupHistory:
    @patch("kiara.repositories.job.IngestionJobRepository.cleanup_old")
    def test_cleanup_history(self, mock_cleanup):
        """Test 29: cleanup_old(days=30) aufgerufen."""
        mock_cleanup.return_value = 5

        from kiara.admin.functions.jobs import jobs_cleanup_history

        result = _run(jobs_cleanup_history(MOCK_USER, days=30))

        assert result.success is True
        assert result.data["deleted"] == 5
        mock_cleanup.assert_called_once_with(days=30)


# ---------------------------------------------------------------------------
# Schedule Tests
# ---------------------------------------------------------------------------


class TestSchedule:
    @patch("kiara.web.cron_manager.write_schedule")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_schedule_set(self, mock_get, mock_write):
        """Test 30: write_schedule aufgerufen."""
        mock_get.return_value = {"id": 5, "name": "DS5"}

        from kiara.admin.functions.jobs import jobs_schedule_set

        result = _run(jobs_schedule_set(MOCK_USER, ds_id=5, cron_expr="0 2 * * *"))

        assert result.success is True
        assert result.data["cron_expr"] == "0 2 * * *"
        mock_write.assert_called_once_with(5, "DS5", "0 2 * * *")

    @patch("kiara.repositories.datasource.DatasourceRepository.exists")
    def test_schedule_delete_nonexistent_ds(self, mock_exists):
        """Test 31: Fehler bei nicht existierender DS."""
        mock_exists.return_value = False

        from kiara.admin.functions.jobs import jobs_schedule_delete

        result = _run(jobs_schedule_delete(MOCK_USER, ds_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ===========================================================================
# Erweiterte Tests: Edge-Cases + Failure-Pfade
# ===========================================================================


class TestJobsStartFullFunctional:
    """Funktionale Tests fuer jobs.start_full (nicht nur Decorator-Check)."""

    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.web.ingestion_worker.enqueue_job")
    @patch("kiara.config.load_config")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_start_full_success(self, mock_get, mock_config, mock_enqueue, mock_refresh):
        """start_full ruft enqueue_job mit job_type='full' auf."""
        mock_get.return_value = {"id": 5, "is_active": True, "name": "DS5"}
        mock_config.return_value = {"embedding_model": "nomic-embed-text"}
        mock_enqueue.return_value = 200

        from kiara.admin.functions.jobs import jobs_start_full

        result = _run(jobs_start_full(MOCK_USER, ds_id=5))

        assert result.success is True
        assert result.data["job_id"] == 200
        assert result.data["mode"] == "full"
        mock_enqueue.assert_called_once_with(
            5,
            job_type="full",
            triggered_by="chat",
            embedding_model="nomic-embed-text",
        )

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_start_full_nonexistent_ds_fails(self, mock_get):
        """DS nicht gefunden → AdminResult.fail."""
        mock_get.return_value = None

        from kiara.admin.functions.jobs import jobs_start_full

        result = _run(jobs_start_full(MOCK_USER, ds_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_start_full_inactive_ds_fails(self, mock_get):
        """Deaktivierte DS → AdminResult.fail."""
        mock_get.return_value = {"id": 5, "is_active": False, "name": "DS5"}

        from kiara.admin.functions.jobs import jobs_start_full

        result = _run(jobs_start_full(MOCK_USER, ds_id=5))

        assert result.success is False
        assert "deaktiviert" in result.message


class TestJobsPauseResumeEdgeCases:
    """Edge-Cases fuer pause/resume request-failure."""

    @patch("kiara.web.ingestion_worker.request_pause")
    @patch("kiara.web.ingestion_worker.is_job_paused")
    @patch("kiara.web.ingestion_worker.get_running_job_id")
    def test_pause_request_fails(self, mock_running, mock_paused, mock_pause):
        """request_pause() returns False → AdminResult.fail."""
        mock_running.return_value = 42
        mock_paused.return_value = False
        mock_pause.return_value = False

        from kiara.admin.functions.jobs import jobs_pause

        result = _run(jobs_pause(MOCK_USER, job_id=42))

        assert result.success is False
        assert "konnte nicht pausiert werden" in result.message

    @patch("kiara.web.ingestion_worker.get_running_job_id")
    def test_pause_job_not_running(self, mock_running):
        """Job laeuft nicht → Fehlermeldung mit aktuellem Job."""
        mock_running.return_value = 99  # anderer Job laeuft

        from kiara.admin.functions.jobs import jobs_pause

        result = _run(jobs_pause(MOCK_USER, job_id=42))

        assert result.success is False
        assert "läuft nicht" in result.message
        assert "99" in result.message

    @patch("kiara.web.ingestion_worker.get_running_job_id")
    def test_pause_no_job_running(self, mock_running):
        """Kein Job laeuft → Fehlermeldung mit 'keiner'."""
        mock_running.return_value = None

        from kiara.admin.functions.jobs import jobs_pause

        result = _run(jobs_pause(MOCK_USER, job_id=42))

        assert result.success is False
        assert "keiner" in result.message

    @patch("kiara.web.ingestion_worker.request_resume")
    @patch("kiara.web.ingestion_worker.is_job_paused")
    def test_resume_request_fails(self, mock_paused, mock_resume):
        """request_resume() returns False → AdminResult.fail."""
        mock_paused.return_value = True
        mock_resume.return_value = False

        from kiara.admin.functions.jobs import jobs_resume

        result = _run(jobs_resume(MOCK_USER, job_id=42))

        assert result.success is False
        assert "konnte nicht fortgesetzt werden" in result.message


class TestJobsDeleteEdgeCases:
    @patch("kiara.repositories.job.IngestionJobRepository.get_by_id")
    def test_delete_nonexistent_job_fails(self, mock_get):
        """Job nicht gefunden → AdminResult.fail."""
        mock_get.return_value = None

        from kiara.admin.functions.jobs import jobs_delete

        result = _run(jobs_delete(MOCK_USER, job_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.repositories.job.IngestionJobRepository.get_by_id")
    def test_delete_pending_job_fails(self, mock_get):
        """Status 'pending' → kann nicht geloescht werden."""
        mock_get.return_value = {"id": 42, "status": "pending"}

        from kiara.admin.functions.jobs import jobs_delete

        result = _run(jobs_delete(MOCK_USER, job_id=42))

        assert result.success is False
        assert "noch aktiv" in result.message

    @patch("kiara.repositories.job.IngestionJobRepository.get_by_id")
    def test_delete_paused_job_fails(self, mock_get):
        """Status 'paused' → kann nicht geloescht werden."""
        mock_get.return_value = {"id": 42, "status": "paused"}

        from kiara.admin.functions.jobs import jobs_delete

        result = _run(jobs_delete(MOCK_USER, job_id=42))

        assert result.success is False
        assert "noch aktiv" in result.message

    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.shared.db.db_transaction")
    @patch("kiara.repositories.job.IngestionJobRepository.get_by_id")
    def test_delete_error_status_succeeds(self, mock_get, mock_db_tx, mock_refresh):
        """Status 'error' → Loeschung erfolgreich."""
        mock_get.return_value = {"id": 42, "status": "error"}
        mock_conn = MagicMock()
        mock_db_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.admin.functions.jobs import jobs_delete

        result = _run(jobs_delete(MOCK_USER, job_id=42))

        assert result.success is True


class TestScheduleEdgeCases:
    @patch("kiara.web.cron_manager.write_schedule")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_schedule_set_invalid_cron_fails(self, mock_get, mock_write):
        """Ungueltiger Cron-Ausdruck → ValueError → AdminResult.fail."""
        mock_get.return_value = {"id": 5, "name": "DS5"}
        mock_write.side_effect = ValueError("Ungueltiger Cron-Ausdruck: invalid")

        from kiara.admin.functions.jobs import jobs_schedule_set

        result = _run(jobs_schedule_set(MOCK_USER, ds_id=5, cron_expr="invalid"))

        assert result.success is False
        assert "Ungueltig" in result.message

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_schedule_set_nonexistent_ds_fails(self, mock_get):
        """DS nicht gefunden → AdminResult.fail."""
        mock_get.return_value = None

        from kiara.admin.functions.jobs import jobs_schedule_set

        result = _run(jobs_schedule_set(MOCK_USER, ds_id=999, cron_expr="0 2 * * *"))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.web.cron_manager.remove_schedule")
    @patch("kiara.repositories.datasource.DatasourceRepository.exists")
    def test_schedule_delete_success(self, mock_exists, mock_remove):
        """Erfolgreiche Schedule-Loeschung."""
        mock_exists.return_value = True

        from kiara.admin.functions.jobs import jobs_schedule_delete

        result = _run(jobs_schedule_delete(MOCK_USER, ds_id=5))

        assert result.success is True
        assert result.data["message"] == "Schedule entfernt"
        mock_remove.assert_called_once_with(5)
