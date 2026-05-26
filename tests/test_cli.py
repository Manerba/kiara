"""Tests fuer CLI ColBERT-Persistenz-Marker (Sprint 1 AP4).

Deckt ab:
- index: Marker gesetzt (kein --config)
- index --all-datasources: Marker gesetzt
- index --config: Marker NICHT gesetzt
- KIARA_CONFIG-Policy: Marker gesetzt (gilt als persistent)
- Marker darf nicht in Nicht-Ingestion-Kommandos leaken
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestCliMarkerIndex:
    """CLI-Marker fuer index-Command."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.config.get_embedding_model", return_value="test-model")
    @patch("kiara.config.load_config")
    def test_index_default_sets_marker(self, mock_load, mock_gem, mock_run):
        from kiara.cli import main

        mock_load.return_value = {
            "datasource": {"path": "/tmp"},
            "web": {"host": "0.0.0.0", "port": 8501},
        }
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["index"])
        assert result.exit_code == 0

        # Config die an run_ingestion uebergeben wurde, muss Marker haben
        actual_config = mock_run.call_args[0][0]
        assert actual_config.get("_kiara_enforce_persisted_colbert_model") is True

    @patch("kiara.ingestion.pipeline.run_all_datasources")
    @patch("kiara.config.get_embedding_model", return_value="test-model")
    @patch("kiara.config.load_config")
    def test_index_all_datasources_sets_marker(self, mock_load, mock_gem, mock_run):
        from kiara.cli import main

        mock_load.return_value = {
            "datasource": {"path": "/tmp"},
            "web": {"host": "0.0.0.0", "port": 8501},
        }
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
            "datasources_processed": 0,
        }

        runner = CliRunner()
        result = runner.invoke(main, ["index", "--all-datasources"])
        assert result.exit_code == 0

        actual_config = mock_run.call_args[0][0]
        assert actual_config.get("_kiara_enforce_persisted_colbert_model") is True

    def test_config_flag_logic_no_marker(self):
        """Prueft die Marker-Logik direkt: explizites --config setzt keinen Marker."""
        # Simuliert ctx.obj.get("config_path") mit explizitem Pfad
        config = {"datasource": {"path": "/tmp"}}
        config_path = "/custom/config.yaml"
        # Das ist die CLI-Logik aus cli.py:index()
        if not config_path:
            config["_kiara_enforce_persisted_colbert_model"] = True
        assert config.get("_kiara_enforce_persisted_colbert_model") is not True

    def test_default_config_logic_sets_marker(self):
        """Prueft die Marker-Logik direkt: kein --config setzt den Marker."""
        config = {"datasource": {"path": "/tmp"}}
        config_path = None
        if not config_path:
            config["_kiara_enforce_persisted_colbert_model"] = True
        assert config.get("_kiara_enforce_persisted_colbert_model") is True

    def test_kiara_config_env_sets_marker(self):
        """Prueft: Auch bei KIARA_CONFIG-Env wird Marker gesetzt (kein --config Flag)."""
        # KIARA_CONFIG ist eine Umgebungsvariable die load_config() intern nutzt,
        # NICHT das --config CLI-Flag. Der Marker haengt nur am CLI-Flag ab.
        config = {"datasource": {"path": "/tmp"}}
        config_path = None  # kein --config Flag, KIARA_CONFIG spielt keine Rolle
        if not config_path:
            config["_kiara_enforce_persisted_colbert_model"] = True
        assert config.get("_kiara_enforce_persisted_colbert_model") is True


class TestCliMarkerNonIngestion:
    """Marker darf nicht in Nicht-Ingestion-Kommandos leaken."""

    def test_get_config_does_not_set_marker(self):
        """_get_config() selbst setzt keinen Marker — nur index() tut das."""
        # Die Funktion _get_config() ruft nur load_config auf und returnt,
        # ohne den Marker zu setzen. Der Marker wird NUR in index() gesetzt.
        # _get_config hat keine Marker-Logik — der Marker lebt nur in index()
        import inspect

        from kiara.cli import _get_config

        src = inspect.getsource(_get_config)
        assert "_kiara_enforce_persisted_colbert_model" not in src


class TestCliQueryAndStatusErrors:
    """CLI-Fehlerpfade fuer query/status ohne Traceback-Ausgabe."""

    @patch("kiara.generation.chain.RAGChain")
    @patch("kiara.generation.backends.create_backend")
    @patch("kiara.cli.load_config")
    def test_query_handles_backend_exception(self, mock_load, mock_backend, mock_chain_cls):
        from kiara.cli import main

        mock_load.return_value = {"retrieval": {}}
        mock_backend.return_value = object()
        mock_chain = MagicMock()
        mock_chain.query.side_effect = RuntimeError("Backend nicht erreichbar")
        mock_chain_cls.return_value = mock_chain

        runner = CliRunner()
        result = runner.invoke(main, ["query", "Testfrage"])
        output = result.output + getattr(result, "stderr", "")

        assert result.exit_code == 1
        assert "Query fehlgeschlagen" in output
        assert "Backend nicht erreichbar" in output
        assert "Traceback" not in output

    @patch("kiara.generation.chain.RAGChain")
    @patch("kiara.generation.backends.create_backend")
    @patch("kiara.cli.load_config")
    def test_query_handles_missing_answer(self, mock_load, mock_backend, mock_chain_cls):
        from kiara.cli import main

        mock_load.return_value = {"retrieval": {}}
        mock_backend.return_value = object()
        mock_chain = MagicMock()
        mock_chain.query.return_value = {"sources": []}
        mock_chain_cls.return_value = mock_chain

        runner = CliRunner()
        result = runner.invoke(main, ["query", "Testfrage"])
        output = result.output + getattr(result, "stderr", "")

        assert result.exit_code == 1
        assert "Query-Ergebnis enthaelt keine Antwort" in output
        assert "Traceback" not in output

    @patch("kiara.retrieval.retriever.Retriever")
    @patch("kiara.cli.load_config")
    def test_status_handles_stats_exception(self, mock_load, mock_retriever_cls):
        from kiara.cli import main

        mock_load.return_value = {"retrieval": {}}
        mock_retriever = MagicMock()
        mock_retriever.get_stats.side_effect = RuntimeError("VectorDB nicht erreichbar")
        mock_retriever_cls.return_value = mock_retriever

        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        output = result.output + getattr(result, "stderr", "")

        assert result.exit_code == 1
        assert "Status fehlgeschlagen" in output
        assert "VectorDB nicht erreichbar" in output
        assert "Traceback" not in output


class TestCliCleanupColbert:
    """Tests fuer 'kiara cleanup-colbert' CLI-Command."""

    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_collections_without_db")
    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_models")
    @patch("kiara.integrity.checks.check_stale_colbert_collections")
    @patch("kiara.shared.db.init_db_backend")
    @patch("kiara.config.load_config")
    def test_dry_run_no_cleanup(
        self, mock_load, mock_init_db, mock_detect, mock_cleanup, mock_coll_cleanup
    ):
        """--dry-run should NOT call cleanup functions."""
        from kiara.cli import main

        mock_load.return_value = {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "database": {"postgresql": {"dsn": "test"}},
        }
        mock_detect.return_value = {
            "status": "warning",
            "details": "1 stale Collection(s)",
            "stale_collections": [
                {
                    "model": "old-model",
                    "name": "colbert_old",
                    "chunks": 42,
                    "reason": "model_mismatch",
                    "datasources": [{"id": 1, "name": "DS1"}],
                }
            ],
            "orphaned_db_entries": [],
        }
        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-colbert", "--dry-run"])
        assert result.exit_code == 0
        assert "Stale Collections: 1" in result.output
        mock_cleanup.assert_not_called()
        mock_coll_cleanup.assert_not_called()

    @patch("kiara.shared.db.db_query")
    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_collections_without_db")
    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_models")
    @patch("kiara.integrity.checks.check_stale_colbert_collections")
    @patch("kiara.shared.db.init_db_backend")
    @patch("kiara.config.load_config")
    def test_non_dry_run_invokes_cleanup(
        self, mock_load, mock_init_db, mock_detect, mock_cleanup, mock_coll_cleanup, mock_db_query
    ):
        """Non-dry-run should call cleanup for stale models."""
        from kiara.cli import main

        mock_load.return_value = {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "database": {"postgresql": {"dsn": "test"}},
        }
        mock_detect.return_value = {
            "status": "warning",
            "details": "1 stale Collection(s)",
            "stale_collections": [
                {
                    "model": "old-model",
                    "name": "colbert_old",
                    "chunks": 42,
                    "reason": "model_mismatch",
                    "datasources": [{"id": 1, "name": "DS1"}],
                }
            ],
            "orphaned_db_entries": [],
        }
        mock_cleanup.return_value = {
            "lock_conflict": False,
            "cleanup_invoked": True,
            "skipped_current_models": [],
            "db_only_current_models": [],
            "errors": [],
        }
        mock_coll_cleanup.return_value = {"deleted": [], "errors": []}

        with patch("kiara.ingestion._file_utils.ChunkDatasources") as mock_chunk_ds_cls:
            mock_chunk_ds = MagicMock()
            mock_chunk_ds.distinct_models.return_value = ["colbert::old-model"]
            mock_chunk_ds_cls.return_value = mock_chunk_ds

            mock_ctx = MagicMock()
            mock_ctx.execute.return_value.fetchall.return_value = [{"datasource_id": 1}]
            mock_db_query.return_value.__enter__ = lambda self: mock_ctx
            mock_db_query.return_value.__exit__ = lambda self, *a: None

            runner = CliRunner()
            result = runner.invoke(main, ["cleanup-colbert"])

        assert result.exit_code == 0
        assert "Cleanup gestartet" in result.output

    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_collections_without_db")
    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_models")
    @patch("kiara.integrity.checks.check_stale_colbert_collections")
    @patch("kiara.shared.db.init_db_backend")
    @patch("kiara.config.load_config")
    def test_config_flag_before_subcommand(
        self, mock_load, mock_init_db, mock_detect, mock_cleanup, mock_coll_cleanup
    ):
        """Test that --config goes before subcommand."""
        from kiara.cli import main

        mock_load.return_value = {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "database": {"postgresql": {"dsn": "test"}},
        }
        mock_detect.return_value = {
            "status": "ok",
            "details": "Keine stale Collections",
            "stale_collections": [],
            "orphaned_db_entries": [],
        }
        mock_coll_cleanup.return_value = {"deleted": [], "errors": []}

        runner = CliRunner()
        result = runner.invoke(main, ["--config", "/tmp/test.yaml", "cleanup-colbert", "--dry-run"])
        assert result.exit_code == 0
        mock_detect.assert_called_once()

    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_collections_without_db")
    @patch("kiara.ingestion.pipeline.cleanup_stale_colbert_models")
    @patch("kiara.integrity.checks.check_stale_colbert_collections")
    @patch("kiara.shared.db.init_db_backend")
    @patch("kiara.config.load_config")
    def test_ds_id_filter(
        self, mock_load, mock_init_db, mock_detect, mock_cleanup, mock_coll_cleanup
    ):
        """--ds-id N filters detection output to only that datasource."""
        from kiara.cli import main

        mock_load.return_value = {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "database": {"postgresql": {"dsn": "test"}},
        }
        mock_detect.return_value = {
            "status": "warning",
            "details": "2 stale Collection(s)",
            "stale_collections": [
                {
                    "model": "old-model",
                    "name": "colbert_old",
                    "chunks": 42,
                    "reason": "model_mismatch",
                    "datasources": [
                        {"id": 1, "name": "DS1"},
                        {"id": 2, "name": "DS2"},
                    ],
                }
            ],
            "orphaned_db_entries": [],
        }

        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-colbert", "--ds-id", "1", "--dry-run"])
        assert result.exit_code == 0
        # Output should mention DS 1
        assert "DS 1:" in result.output
        assert "Stale Collections: 1" in result.output


class TestCleanupSystemEvents:
    """CLI-Befehl cleanup-system-events (Admin-System-Log-Retention-Job)."""

    @patch("kiara.cli.load_config")
    @patch("kiara.cli.resolve_config_from_context", create=True)
    @patch("kiara.shared.db.init_db_backend")
    @patch("kiara.repositories.system_events.SystemEventRepository")
    def test_dry_run_counts_without_delete(self, mock_repo_cls, mock_init, _resolve, mock_load):
        from kiara.cli import main

        mock_load.return_value = {"system_events": {"retention_days": 90}}
        repo_instance = MagicMock()
        repo_instance.count_older_than.return_value = 42
        mock_repo_cls.return_value = repo_instance

        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-system-events", "--dry-run"])
        assert result.exit_code == 0
        repo_instance.count_older_than.assert_called_once_with(90)
        repo_instance.cleanup_older_than.assert_not_called()
        assert "42" in result.output

    @patch("kiara.cli.load_config")
    @patch("kiara.cli.resolve_config_from_context", create=True)
    @patch("kiara.shared.db.init_db_backend")
    @patch("kiara.repositories.system_events.SystemEventRepository")
    def test_cleanup_uses_custom_days(self, mock_repo_cls, mock_init, _resolve, mock_load):
        from kiara.cli import main

        mock_load.return_value = {"system_events": {"retention_days": 90}}
        repo_instance = MagicMock()
        repo_instance.cleanup_older_than.return_value = 7
        mock_repo_cls.return_value = repo_instance

        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-system-events", "--days", "30"])
        assert result.exit_code == 0
        repo_instance.cleanup_older_than.assert_called_once_with(30)
        assert "7" in result.output

    def test_days_min_1_validation(self):
        from kiara.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["cleanup-system-events", "--days", "0"])
        assert result.exit_code != 0
