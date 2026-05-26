"""Integrations-Tests fuer hash_mode in der Ingestion-Pipeline.

Abdeckung:
- _compute_file_diff mit content-Mode: mtime-Aenderung wird ignoriert
- _load_and_diff_files: automatische Migration bei Mode-Wechsel
- run_ingestion: Early-Return persistiert hash_mode
- run_all_datasources: liest HASH_MODE vom Connector
- Connector-Klassen: HASH_MODE-Attribute
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from kiara.ingestion._file_utils import (
    HashStore,
    compute_file_hash,
)


@pytest.fixture()
def pipeline_tmp(tmp_path):
    """Erzeugt temporaere Verzeichnisstruktur fuer Pipeline-Tests."""
    src_dir = tmp_path / "data"
    src_dir.mkdir()
    db_dir = tmp_path / "vectordb"
    db_dir.mkdir()

    # Test-Dateien anlegen
    for i in range(3):
        (src_dir / f"mail_{i}.eml").write_bytes(
            f"From: user{i}@test.de\nSubject: Test {i}\n\nBody {i}".encode()
        )

    config = {
        "datasource": {
            "path": str(src_dir),
            "file_types": [".eml"],
            "exclude_patterns": [],
        },
        "ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 100,
        },
        "vectordb": {
            "path": str(db_dir),
        },
    }
    return config, src_dir, db_dir


# ── _compute_file_diff ───────────────────────────────────────────────────────


class TestComputeFileDiffContentMode:
    def test_mtime_change_ignored_in_content_mode(self, pipeline_tmp):
        """Content-Mode ignoriert mtime-Aenderungen bei unveraendertem Inhalt."""
        from kiara.ingestion.pipeline import _compute_file_diff

        config, src_dir, _ = pipeline_tmp
        files = sorted(str(p) for p in src_dir.glob("*.eml"))

        # Erstmal Hashes mit content-Mode berechnen
        stored = {fp: compute_file_hash(fp, "content") for fp in files}

        # mtime aller Dateien aendern (simuliert IMAP-Resync)
        for fp in files:
            stat = os.stat(fp)
            os.utime(fp, (stat.st_atime, stat.st_mtime + 7200))

        # Content-Mode: keine Dateien zu verarbeiten
        to_process, _, deleted = _compute_file_diff(
            files,
            True,
            stored,
            hash_mode="content",
        )
        assert to_process == []
        assert deleted == []

    def test_mtime_change_detected_in_mtime_mode(self, pipeline_tmp):
        """mtime-Mode erkennt mtime-Aenderungen als Aenderung."""
        from kiara.ingestion.pipeline import _compute_file_diff

        config, src_dir, _ = pipeline_tmp
        files = sorted(str(p) for p in src_dir.glob("*.eml"))

        stored = {fp: compute_file_hash(fp, "mtime") for fp in files}

        # mtime aendern
        for fp in files:
            stat = os.stat(fp)
            os.utime(fp, (stat.st_atime, stat.st_mtime + 7200))

        to_process, _, _ = _compute_file_diff(
            files,
            True,
            stored,
            hash_mode="mtime",
        )
        assert len(to_process) == 3  # Alle als geaendert erkannt


# ── _load_and_diff_files: auto migration ────────────────────────────────────


class TestLoadAndDiffAutoMigration:
    @patch("kiara.ingestion.pipeline.VectorStore")
    def test_auto_migration_on_mode_change(self, mock_vs_cls, pipeline_tmp):
        """Mode-Wechsel mtime → content loest automatische Migration aus."""
        config, src_dir, db_dir = pipeline_tmp
        files = sorted(str(p) for p in src_dir.glob("*.eml"))

        mock_store = MagicMock()
        mock_vs_cls.return_value = mock_store

        # HashStore mit mtime-Hashes vorbefuellen
        hs = HashStore()
        mtime_hashes = {fp: (compute_file_hash(fp, "mtime"), os.path.getsize(fp)) for fp in files}
        hs.upsert_many(1, "nomic", mtime_hashes)
        hs.set_hash_mode(1, "nomic", "mtime")

        # mtime aller Dateien aendern (simuliert IMAP-Resync)
        for fp in files:
            stat = os.stat(fp)
            os.utime(fp, (stat.st_atime, stat.st_mtime + 7200))

        from kiara.ingestion.pipeline import _load_and_diff_files

        result = _load_and_diff_files(
            config,
            mock_store,
            True,
            1,
            "nomic",
            hash_mode="content",
        )
        _, files_to_process, _, _, hash_store, _ = result

        # Nach Migration: content-Hashes sind stabil → keine Dateien zu verarbeiten
        assert files_to_process == []
        assert hash_store.get_hash_mode(1, "nomic") == "content"

    def test_full_reingest_marks_known_files_for_cleanup(self, pipeline_tmp):
        """Full-Reingest bereinigt alte Chunks fuer bekannte Dateien."""
        config, src_dir, _ = pipeline_tmp
        files = sorted(str(p) for p in src_dir.glob("*.eml"))
        deleted_path = str(src_dir / "deleted.eml")
        ds_id = 1087
        model = "nomic"

        hs = HashStore()
        hs.delete_datasource(ds_id)
        stored = {fp: (compute_file_hash(fp, "mtime"), os.path.getsize(fp)) for fp in files[:2]}
        stored[deleted_path] = ("old-hash", 123)
        hs.upsert_many(ds_id, model, stored)

        mock_store = MagicMock()
        from kiara.ingestion.pipeline import _load_and_diff_files

        with patch("kiara.ingestion.pipeline._cleanup_stale_chunks") as mock_cleanup:
            result = _load_and_diff_files(
                config,
                mock_store,
                False,
                ds_id,
                model,
                hash_mode="mtime",
            )

        _, files_to_process, _, files_to_delete, _, _ = result
        assert files_to_process == files
        assert set(files_to_delete) == set([*files[:2], deleted_path])
        mock_cleanup.assert_called_once()
        assert mock_cleanup.call_args.args[-1] == files_to_delete

        hs.delete_datasource(ds_id)


# ── run_ingestion: Early-Return persistiert hash_mode ────────────────────────


class TestRunIngestionEarlyReturn:
    @patch("kiara.ingestion.pipeline.VectorStore")
    def test_early_return_persists_hash_mode(self, mock_vs_cls, pipeline_tmp):
        """run_ingestion persistiert hash_mode auch wenn keine Dateien zu verarbeiten."""
        config, src_dir, db_dir = pipeline_tmp
        files = sorted(str(p) for p in src_dir.glob("*.eml"))

        mock_store = MagicMock()
        mock_vs_cls.return_value = mock_store

        # Alle Hashes vorher speichern → "no files to process"
        hs = HashStore()
        content_hashes = {
            fp: (compute_file_hash(fp, "content"), os.path.getsize(fp)) for fp in files
        }
        hs.upsert_many(1, "nomic", content_hashes)
        hs.set_hash_mode(1, "nomic", "content")

        from kiara.ingestion.pipeline import run_ingestion

        stats = run_ingestion(
            config,
            incremental=True,
            datasource_id=1,
            embedding_model="nomic",
            hash_mode="content",
        )

        assert stats["processed_files"] == 0

        # hash_mode muss persistiert sein
        hs2 = HashStore()
        assert hs2.get_hash_mode(1, "nomic") == "content"


# ── run_all_datasources: HASH_MODE vom Connector ────────────────────────────


class TestRunAllDatasourcesHashMode:
    @patch("kiara.ingestion.pipeline.run_ingestion")
    def test_passes_hash_mode_from_connector(self, mock_run):
        """run_all_datasources liest HASH_MODE vom Connector und reicht ihn durch."""

        mock_run.return_value = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        # Mock: Connector mit HASH_MODE = "content"
        mock_connector = MagicMock()
        mock_connector.HASH_MODE = "content"
        mock_connector.safe_prepare_and_scan.return_value = "/tmp/fake"
        mock_connector.secret_fields.return_value = []

        from kiara.ingestion.pipeline import run_all_datasources

        # Fake DB-Row fuer eine aktive Datasource
        mock_row = {
            "id": 15,
            "type": "imap",
            "name": "Test-Mail",
            "is_active": True,
            "connector_config": "{}",
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": None,
            "ds_ddu_enabled": False,
            "ds_vlm_enabled": False,
            "ds_index_attachments": 1,
            "ds_accept_all_text": 0,
        }

        with (
            patch(
                "kiara.repositories.datasource.DatasourceRepository.get_active",
                return_value=[mock_row],
            ),
            patch("kiara.connectors.registry.get") as mock_get,
            patch("kiara.connectors.registry.ensure_loaded"),
            patch("kiara.ingestion.pipeline.load_config") as mock_cfg,
        ):
            mock_get.return_value = mock_connector
            mock_cfg.return_value = {
                "datasource": {"path": "/tmp/fake", "file_types": [".eml"], "exclude_patterns": []},
                "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 100},
                "vectordb": {"path": "/tmp/vecdb"},
            }

            run_all_datasources(incremental=True)

        # Pruefen ob hash_mode="content" durchgereicht wurde
        assert mock_run.called
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("hash_mode") == "content"


# ── Connector-Klassen: HASH_MODE ────────────────────────────────────────────


class TestConnectorHashModes:
    def test_connector_base_default(self):
        """ConnectorBase Default ist 'mtime'."""
        from kiara.connectors.base import ConnectorBase

        assert ConnectorBase.HASH_MODE == "mtime"

    def test_imap_connector_content(self):
        """IMAPConnector hat HASH_MODE = 'content'."""
        from kiara.connectors.imap_connector import IMAPConnector

        assert IMAPConnector.HASH_MODE == "content"

    def test_exchange_connector_content(self):
        """ExchangeOnlineConnector hat HASH_MODE = 'content'."""
        from kiara.connectors.exchange_connector import ExchangeOnlineConnector

        assert ExchangeOnlineConnector.HASH_MODE == "content"

    def test_local_connector_mtime(self):
        """LocalConnector erbt Default 'mtime'."""
        from kiara.connectors.local_connector import LocalConnector

        assert LocalConnector.HASH_MODE == "mtime"


# ── Heal-Pfad: Legacy-Zeilen ohne file_size ────────────────────────────────


class TestHealMissingSizeInPipeline:
    def test_load_and_diff_heals_missing_file_size(self, pipeline_tmp):
        """_load_and_diff_files heilt Legacy-Zeilen mit file_size IS NULL."""
        config, src_dir, db_dir = pipeline_tmp
        files = sorted(str(p) for p in src_dir.glob("*.eml"))

        hs = HashStore()
        ds_id = 9350
        model = "nomic"
        hs.delete_datasource(ds_id)

        # Hashes MIT file_size einfuegen (normaler Pfad)
        normal_hashes = {fp: (compute_file_hash(fp, "mtime"), os.path.getsize(fp)) for fp in files}
        hs.upsert_many(ds_id, model, normal_hashes)
        hs.set_hash_mode(ds_id, model, "mtime")

        # Eine Zeile manuell auf file_size=NULL setzen (simuliert Legacy-Daten)
        target_fp = files[0]
        from kiara.shared.db import db_query, db_transaction

        with db_transaction() as conn:
            conn.execute(
                "UPDATE file_hashes SET file_size = NULL "
                "WHERE datasource_id = %s AND embedding_model = %s AND file_path = %s",
                (ds_id, model, target_fp),
            )

        # Pruefen: file_size ist NULL
        with db_query() as conn:
            row = conn.execute(
                "SELECT file_size FROM file_hashes "
                "WHERE datasource_id = %s AND embedding_model = %s AND file_path = %s",
                (ds_id, model, target_fp),
            ).fetchone()
        assert row["file_size"] is None

        # _load_and_diff_files ausfuehren — heilt die NULL-Zeile
        from unittest.mock import MagicMock

        mock_store = MagicMock()

        from kiara.ingestion.pipeline import _load_and_diff_files

        _load_and_diff_files(
            config,
            mock_store,
            True,
            ds_id,
            model,
            hash_mode="mtime",
        )

        # file_size muss jetzt gesetzt sein
        with db_query() as conn:
            row = conn.execute(
                "SELECT file_size FROM file_hashes "
                "WHERE datasource_id = %s AND embedding_model = %s AND file_path = %s",
                (ds_id, model, target_fp),
            ).fetchone()
        assert row["file_size"] is not None
        assert row["file_size"] == os.path.getsize(target_fp)

        # Aufraeumen
        hs.delete_datasource(ds_id)
