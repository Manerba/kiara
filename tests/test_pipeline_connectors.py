"""Tests fuer Pipeline Konnektor-Integration (Phase 11.1 Refactoring).

Testet run_all_datasources() mit dem neuen Registry-basierten Dispatch
statt der alten if/elif-Ketten. Rows haben jetzt connector_config JSON.
"""

import json
from unittest.mock import MagicMock, patch

from kiara.ingestion.pipeline import run_all_datasources


def _make_row(data: dict):
    """Erzeugt ein dict-aehnliches Row-Objekt."""
    return data


def _mock_db_query(rows):
    """Erzeugt einen Patch fuer DatasourceRepository.get_active mit den gegebenen Rows."""
    return patch("kiara.repositories.datasource.DatasourceRepository.get_active", return_value=rows)


def _base_config(tmp_path):
    """Erzeugt eine Basis-Konfiguration."""
    return {
        "datasource": {
            "path": str(tmp_path / "data"),
            "file_types": [".txt"],
            "exclude_patterns": [],
        },
        "vectordb": {"path": str(tmp_path / "vectordb")},
        "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
        "ollama": {"url": "http://localhost:11434", "embedding_model": "test"},
    }


def _imap_row(**overrides):
    """Standard-IMAP-Datasource-Row mit connector_config JSON."""
    cc = {
        "server": "mail.firma.de",
        "port": 993,
        "ssl": True,
        "username": "user@firma.de",
        "password": "geheim",
        "folders": "INBOX",
        "since_days": 90,
        "download_attachments": True,
        "max_attachment_mb": 25,
    }
    row = {
        "id": 1,
        "name": "Firmenmail",
        "type": "imap",
        "is_active": True,
        "path": "",
        "connector_config": json.dumps(cc),
        "ds_chunk_strategy": "auto",
        "ds_chunk_size": None,
        "ds_chunk_overlap": None,
        "ds_file_types": None,
        "ds_exclude_patterns": None,
        "ds_ddu_enabled": False,
        "ds_vlm_enabled": False,
    }
    row.update(overrides)
    return row


def _sql_row(**overrides):
    """Standard-SQL-Datasource-Row mit connector_config JSON."""
    cc = {
        "dsn": "postgresql://localhost:5432/erp",
        "driver": "postgresql",
        "query": "SELECT * FROM auftraege",
        "text_columns": "",
        "id_column": "id",
        "timestamp_column": "",
        "text_template": "",
    }
    row = {
        "id": 2,
        "name": "ERP-Daten",
        "type": "sql",
        "is_active": True,
        "path": "",
        "connector_config": json.dumps(cc),
        "ds_chunk_strategy": "auto",
        "ds_chunk_size": None,
        "ds_chunk_overlap": None,
        "ds_file_types": None,
        "ds_exclude_patterns": None,
        "ds_ddu_enabled": False,
        "ds_vlm_enabled": False,
    }
    row.update(overrides)
    return row


def _local_row(**overrides):
    """Standard-Local-Datasource-Row mit connector_config JSON."""
    cc = {"path": "/tmp/data"}
    row = {
        "id": 10,
        "name": "Lokale Daten",
        "type": "local",
        "is_active": True,
        "path": "",
        "connector_config": json.dumps(cc),
        "ds_chunk_strategy": "auto",
        "ds_chunk_size": None,
        "ds_chunk_overlap": None,
        "ds_file_types": None,
        "ds_exclude_patterns": None,
        "ds_ddu_enabled": False,
        "ds_vlm_enabled": False,
    }
    row.update(overrides)
    return row


class TestRunAllDatasourcesIMAP:
    """IMAP-Datasource -> safe_prepare_and_scan() + run_ingestion()."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_run_all_datasources_imap(
        self, mock_is_enc, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """IMAP-DS -> safe_prepare_and_scan() aufgerufen, danach run_ingestion()."""
        config = _base_config(tmp_path)
        mock_cls = MagicMock()
        mock_cls.safe_prepare_and_scan.return_value = "/var/lib/kiara/mail-cache/ds1"
        mock_cls.secret_fields.return_value = ["password"]
        mock_cls.HASH_MODE = "content"
        mock_get_conn.return_value = mock_cls
        mock_run.return_value = {
            "total_files": 1,
            "processed_files": 1,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 5,
            "errors": 0,
        }

        with _mock_db_query([_imap_row()]):
            result = run_all_datasources(config, incremental=False)

        mock_cls.safe_prepare_and_scan.assert_called_once()
        mock_run.assert_called_once()
        assert result["datasources_processed"] == 1


class TestRunAllDatasourcesSQL:
    """SQL-Datasource -> safe_prepare_and_scan() + run_ingestion()."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_run_all_datasources_sql(
        self, mock_is_enc, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """SQL-DS -> safe_prepare_and_scan() aufgerufen, danach run_ingestion()."""
        config = _base_config(tmp_path)
        mock_cls = MagicMock()
        mock_cls.safe_prepare_and_scan.return_value = "/var/lib/kiara/sql-cache/ds2"
        mock_cls.secret_fields.return_value = ["dsn"]
        mock_cls.HASH_MODE = "mtime"
        mock_get_conn.return_value = mock_cls
        mock_run.return_value = {
            "total_files": 1,
            "processed_files": 1,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 3,
            "errors": 0,
        }

        with _mock_db_query([_sql_row()]):
            result = run_all_datasources(config, incremental=False)

        mock_cls.safe_prepare_and_scan.assert_called_once()
        mock_run.assert_called_once()
        assert result["datasources_processed"] == 1


class TestRunAllDatasourcesPrepareFails:
    """safe_prepare_and_scan() gibt None zurueck -> errors += 1, naechste DS weiter."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_imap_prepare_fails_local_continues(
        self, mock_is_enc, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """IMAP safe_prepare_and_scan None -> errors += 1, lokale DS wird verarbeitet."""
        config = _base_config(tmp_path)
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }
        call_count = [0]

        def side_effect(ds_type):
            call_count[0] += 1
            cls = MagicMock()
            cls.secret_fields.return_value = []
            cls.HASH_MODE = "mtime"
            if ds_type == "imap":
                cls.safe_prepare_and_scan.return_value = None
                cls.secret_fields.return_value = ["password"]
                cls.HASH_MODE = "content"
            else:
                cls.safe_prepare_and_scan.return_value = str(tmp_path / "data")
            return cls

        mock_get_conn.side_effect = side_effect

        with _mock_db_query(
            [
                _imap_row(id=1),
                _local_row(
                    id=2, **{"connector_config": json.dumps({"path": str(tmp_path / "data")})}
                ),
            ]
        ):
            result = run_all_datasources(config, incremental=False)

        assert result["errors"] >= 1  # IMAP-Fehler gezaehlt
        mock_run.assert_called_once()  # Lokale DS wurde verarbeitet


class TestRunAllDatasourcesDecryption:
    """Verschluesselte Secrets in connector_config werden entschluesselt."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.decrypt", return_value="entschluesselt")
    @patch("kiara.shared.crypto.is_encrypted", return_value=True)
    def test_imap_password_decrypted(
        self, mock_is_enc, mock_decrypt, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """Verschluesseltes Passwort -> decrypt() vor safe_prepare_and_scan()."""
        cc = {
            "server": "mail.firma.de",
            "password": "ENC:abc123",
        }

        config = _base_config(tmp_path)
        mock_cls = MagicMock()
        mock_cls.safe_prepare_and_scan.return_value = "/var/lib/kiara/mail-cache/ds1"
        mock_cls.secret_fields.return_value = ["password"]
        mock_cls.HASH_MODE = "content"
        mock_get_conn.return_value = mock_cls
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        with _mock_db_query([_imap_row(connector_config=json.dumps(cc))]):
            run_all_datasources(config, incremental=False)

        mock_decrypt.assert_called_with("ENC:abc123")

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.decrypt", return_value="postgresql://localhost:5432/erp")
    @patch("kiara.shared.crypto.is_encrypted", return_value=True)
    def test_sql_dsn_decrypted(
        self, mock_is_enc, mock_decrypt, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """Verschluesselter DSN -> decrypt() vor safe_prepare_and_scan()."""
        cc = {
            "dsn": "ENC:xyz789",
            "driver": "postgresql",
            "query": "SELECT 1",
        }

        config = _base_config(tmp_path)
        mock_cls = MagicMock()
        mock_cls.safe_prepare_and_scan.return_value = "/var/lib/kiara/sql-cache/ds2"
        mock_cls.secret_fields.return_value = ["dsn"]
        mock_cls.HASH_MODE = "mtime"
        mock_get_conn.return_value = mock_cls
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        with _mock_db_query([_sql_row(connector_config=json.dumps(cc))]):
            run_all_datasources(config, incremental=False)

        mock_decrypt.assert_called_with("ENC:xyz789")


class TestRunAllDatasourcesOverrides:
    """Per-DS Overrides (chunk_strategy, ddu_enabled) werden korrekt weitergereicht."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_chunk_strategy_override(
        self, mock_is_enc, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """ds_chunk_strategy wird als Override an run_ingestion() weitergegeben."""
        row = _local_row(ds_chunk_strategy="prose")

        config = _base_config(tmp_path)
        mock_cls = MagicMock()
        mock_cls.safe_prepare_and_scan.return_value = "/tmp/data"
        mock_cls.secret_fields.return_value = []
        mock_cls.HASH_MODE = "mtime"
        mock_get_conn.return_value = mock_cls
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        with _mock_db_query([row]):
            run_all_datasources(config, incremental=False)

        call_kwargs = mock_run.call_args
        ds_overrides = call_kwargs.kwargs.get("ds_overrides") or call_kwargs[1].get("ds_overrides")
        assert ds_overrides is not None
        assert ds_overrides["chunk_strategy"] == "prose"

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.connectors.registry.ensure_loaded")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_run_all_datasources_forwards_colbert_enabled(
        self, mock_is_enc, mock_get_conn, mock_ensure, mock_run, tmp_path
    ):
        """ds_colbert_enabled=True → colbert_enabled=True in ds_overrides."""
        row = _local_row(
            ds_colbert_enabled=True,
            ds_index_attachments=1,
            ds_accept_all_text=0,
            ds_contextual_retrieval=0,
            ds_enrichment_prompt_id=None,
            ds_enrichment_backend_id=None,
            restriction_level="anonymize",
            ds_ddu_ocr_images=1,
        )

        config = _base_config(tmp_path)
        mock_cls = MagicMock()
        mock_cls.safe_prepare_and_scan.return_value = "/tmp/data"
        mock_cls.secret_fields.return_value = []
        mock_cls.HASH_MODE = "mtime"
        mock_get_conn.return_value = mock_cls
        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        with _mock_db_query([row]):
            run_all_datasources(config, incremental=False)

        call_kwargs = mock_run.call_args
        ds_overrides = call_kwargs.kwargs.get("ds_overrides") or call_kwargs[1].get("ds_overrides")
        assert ds_overrides is not None
        assert ds_overrides["colbert_enabled"] is True
