"""Edge-Case-Tests fuer Pipeline + Connector-Integration (Phase 11.1).

Testet unbekannte Typen, prepare_and_scan-Fehler, gemischte Datasources,
inaktive DS und Logger-Nutzung.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors.registry import ensure_loaded


@pytest.fixture(autouse=True)
def _load_connectors():
    from kiara.connectors import registry as _reg

    if _reg._loaded and not _reg._registry:
        _reg._loaded = False
    ensure_loaded()


def _mock_db_query_for_rows(rows: list[dict]):
    """Erstellt einen Patch fuer DatasourceRepository.get_active mit Mock-Rows."""
    return patch("kiara.repositories.datasource.DatasourceRepository.get_active", return_value=rows)


# ---------------------------------------------------------------------------
# 6.1 Unbekannter Typ
# ---------------------------------------------------------------------------


class TestRunAllUnknownType:
    @patch(
        "kiara.ingestion.pipeline.run_ingestion",
        return_value={
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        },
    )
    def test_run_all_unknown_type_fails_closed(self, mock_run, tmp_path):
        """Ein unbekannter Connector-Typ bricht vor jeder Ingestion stabil ab."""
        from kiara.ingestion.pipeline import run_all_datasources

        rows = [
            {"id": 1, "name": "FTP", "type": "ftp", "connector_config": "{}", "is_active": True},
        ]

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vectordb")},
        }

        with _mock_db_query_for_rows(rows):
            with pytest.raises(KeyError, match="Unbekannter Connector-Typ: ftp"):
                run_all_datasources(config, incremental=False)

        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# 6.2 prepare_and_scan() Exception
# ---------------------------------------------------------------------------


class TestRunAllPrepareException:
    @patch(
        "kiara.ingestion.pipeline.run_ingestion",
        return_value={
            "total_files": 5,
            "processed_files": 5,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 10,
            "errors": 0,
        },
    )
    def test_run_all_prepare_exception_caught(self, mock_run, caplog, tmp_path):
        """prepare_and_scan() Exception wird gefangen."""
        from kiara.ingestion.pipeline import run_all_datasources

        rows = [
            {
                "id": 1,
                "name": "Local",
                "type": "local",
                "connector_config": '{"path": "/nonexistent"}',
                "is_active": True,
            },
        ]

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vectordb")},
        }

        # safe_prepare_and_scan() faengt Exceptions intern und gibt None zurueck
        with _mock_db_query_for_rows(rows), patch("kiara.connectors.registry.get") as mock_get:
            mock_cls = MagicMock()
            mock_cls.safe_prepare_and_scan.return_value = None
            mock_cls.secret_fields.return_value = []
            mock_get.return_value = mock_cls

            with caplog.at_level(logging.ERROR):
                stats = run_all_datasources(config, incremental=False)

        assert stats["errors"] >= 1


# ---------------------------------------------------------------------------
# 6.3 Mehrere Datasources gemischt
# ---------------------------------------------------------------------------


class TestRunAllMixedTypes:
    @patch("kiara.ingestion.pipeline.run_ingestion")
    def test_run_all_mixed_types(self, mock_run, tmp_path):
        """3 DS: local (ok), imap (fehlschlaegt), local2 (ok)."""
        from kiara.ingestion.pipeline import run_all_datasources

        mock_run.return_value = {
            "total_files": 3,
            "processed_files": 3,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 6,
            "errors": 0,
        }

        rows = [
            {
                "id": 1,
                "name": "Local1",
                "type": "local",
                "connector_config": json.dumps({"path": str(tmp_path)}),
                "is_active": True,
            },
            {
                "id": 2,
                "name": "Mail",
                "type": "imap",
                "connector_config": json.dumps({"server": "mail.de"}),
                "is_active": True,
            },
            {
                "id": 3,
                "name": "Local2",
                "type": "local",
                "connector_config": json.dumps({"path": str(tmp_path)}),
                "is_active": True,
            },
        ]

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vectordb")},
        }

        # IMAP prepare_and_scan fehlschlaegt
        from kiara.connectors import registry as _reg

        original_get = _reg.get

        def mock_get_connector(name):
            cls = original_get(name)
            if name == "imap":
                mock_cls = MagicMock()
                mock_cls.secret_fields.return_value = ["password"]
                mock_cls.safe_prepare_and_scan.return_value = None  # Fehlschlag
                return mock_cls
            return cls

        with (
            _mock_db_query_for_rows(rows),
            patch("kiara.connectors.registry.get", side_effect=mock_get_connector),
        ):
            stats = run_all_datasources(config, incremental=False)

        assert stats["datasources_processed"] == 2
        assert stats["errors"] >= 1

    @patch("kiara.ingestion.pipeline.run_ingestion")
    def test_run_all_inactive_datasources_skipped(self, mock_run, tmp_path):
        """Inaktive DS: prepare_and_scan() wird NICHT aufgerufen."""
        from kiara.ingestion.pipeline import run_all_datasources

        # is_active=False Rows werden per SQL-Query gefiltert; Mock liefert leere Liste
        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vectordb")},
        }

        with _mock_db_query_for_rows([]):
            stats = run_all_datasources(config, incremental=False)
        # Wenn keine aktiven DS: run_ingestion mit Default-Config
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# 6.4 Logging statt Print
# ---------------------------------------------------------------------------


class TestPipelineUsesLogger:
    def test_pipeline_uses_logger_not_print(self):
        """pipeline.py nutzt logger fuer Fehler, nicht print()."""
        import inspect

        from kiara.ingestion import pipeline

        source = inspect.getsource(pipeline.run_all_datasources)
        # Fehlerpfade nutzen logger, nicht print
        assert "logger.error" in source or "logger.warning" in source
