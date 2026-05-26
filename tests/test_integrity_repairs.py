"""Tests fuer Integrity-Repair ES-Sync.

Prueft, dass repair_ds_ids und repair_reconstruct neben Qdrant auch ES aktualisieren.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_task():
    """Setzt den globalen _task-State vor jedem Test zurueck."""
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        svc._task.update(
            {
                "running": True,
                "action": None,
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


def _config():
    return {
        "retrieval": {
            "hybrid_search": "elasticsearch",
            "elasticsearch_url": "http://localhost:9200",
        }
    }


# ---------------------------------------------------------------------------
# repair_ds_ids + ES-Sync
# ---------------------------------------------------------------------------


class TestRepairDsIdsEsSync:
    """repair_ds_ids aktualisiert Qdrant UND ES."""

    @patch("kiara.retrieval.elasticsearch_service.update_ds_ids")
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_ds_ids_sync_calls_es(self, mock_es_configured, mock_es_update):
        """repair_ds_ids ruft es_svc.update_ds_ids() fuer jede Batch auf."""
        from kiara.admin.services.integrity_service import _run_repair_ds_ids

        store = MagicMock()
        store.count_total.return_value = 3

        chunk_ds = MagicMock()
        batch1 = {"c1": [1, 2], "c2": [2]}
        batch2 = {"c3": [1]}
        chunk_ds.get_all_ds_ids_grouped.return_value = [batch1, batch2]

        config = _config()
        with (
            patch("kiara.config.load_config", return_value=config),
            patch("kiara.config.get_embedding_model", return_value="model1"),
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs_cls,
            patch("kiara.ingestion._file_utils.ChunkDatasources", return_value=chunk_ds),
        ):
            # list_collections() liefert eine Collection damit non_empty nicht leer ist
            mock_vs_cls.list_collections.return_value = [
                {"name": "kb_model1", "chunks": 3, "model": "model1"}
            ]
            # VectorStore(config, embedding_model=...) gibt den store-Mock zurueck
            mock_vs_cls.return_value = store
            _run_repair_ds_ids()

        result = _get_result()
        assert result["status"] == "success"
        assert result["updated"] == 3

        # Qdrant
        assert store.update_ds_ids.call_count == 2
        store.update_ds_ids.assert_any_call(batch1)
        store.update_ds_ids.assert_any_call(batch2)

        # ES
        assert mock_es_update.call_count == 2
        mock_es_update.assert_any_call(config, batch1)
        mock_es_update.assert_any_call(config, batch2)

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=False)
    def test_ds_ids_no_es_when_disabled(self, mock_es_configured):
        """Wenn ES nicht konfiguriert, kein ES-Call."""
        from kiara.admin.services.integrity_service import _run_repair_ds_ids

        store = MagicMock()
        store.count_total.return_value = 1

        chunk_ds = MagicMock()
        chunk_ds.get_all_ds_ids_grouped.return_value = [{"c1": [1]}]

        with (
            patch("kiara.config.load_config", return_value=_config()),
            patch("kiara.config.get_embedding_model", return_value="m1"),
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs_cls2,
            patch("kiara.ingestion._file_utils.ChunkDatasources", return_value=chunk_ds),
            patch("kiara.retrieval.elasticsearch_service.update_ds_ids") as mock_es_update,
        ):
            mock_vs_cls2.list_collections.return_value = [
                {"name": "kb_m1", "chunks": 1, "model": "m1"}
            ]
            mock_vs_cls2.return_value = store
            _run_repair_ds_ids()

        result = _get_result()
        assert result["status"] == "success"
        mock_es_update.assert_not_called()


# ---------------------------------------------------------------------------
# repair_reconstruct + ES-Sync
# ---------------------------------------------------------------------------


class TestRepairReconstructEsSync:
    """repair_reconstruct aktualisiert ds_ids in Qdrant UND ES."""

    @patch("kiara.retrieval.elasticsearch_service.update_ds_ids")
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_reconstruct_calls_es(self, mock_es_configured, mock_es_update):
        """repair_reconstruct ruft es_svc.update_ds_ids() im ds_ids-Update-Loop."""
        from contextlib import contextmanager

        from kiara.admin.services.integrity_service import _run_repair_reconstruct

        # DB-Mocks
        call_count = [0]

        @contextmanager
        def fake_db_query():
            conn = MagicMock()
            call_count[0] += 1
            if call_count[0] == 1:
                # Erste Abfrage: Datasources
                conn.execute.return_value.fetchall.return_value = [
                    {"id": 1, "name": "DS1", "path": "/data", "is_active": True}
                ]
            else:
                # Folgeabfragen: count_rows
                conn.execute.return_value.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "model1", "cnt": 1, "file_est": 1}
                ]
            yield conn

        @contextmanager
        def fake_db_tx():
            conn = MagicMock()
            yield conn

        # VectorStore: 1 Collection, 1 Chunk
        store = MagicMock()
        store.scroll_all.return_value = [[("c1", {"source": "/data/file.txt"})]]

        # ChunkDatasources
        chunk_ds = MagicMock()
        batch_map = {"c1": [1]}
        chunk_ds.get_all_ds_ids_grouped.return_value = [batch_map]

        config = _config()
        with (
            patch("kiara.config.load_config", return_value=config),
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs_cls,
            patch("kiara.shared.db.db_query", side_effect=fake_db_query),
            patch("kiara.shared.db.db_transaction", side_effect=fake_db_tx),
            patch("kiara.web.api_datasource._effective_path", return_value="/data"),
            patch("kiara.repositories.datasource.DatasourceEmbeddingRepository"),
            patch("kiara.ingestion._file_utils.ChunkDatasources", return_value=chunk_ds),
        ):
            mock_vs_cls.list_collections.return_value = [
                {"name": "kb_model1", "model": "model1", "chunks": 1}
            ]
            mock_vs_cls.return_value = store
            _run_repair_reconstruct()

        result = _get_result()
        assert result["status"] == "success"

        # ES muss aufgerufen worden sein
        mock_es_update.assert_called()
        mock_es_update.assert_any_call(config, batch_map)


class TestRecalcDatasourceEmbeddings:
    """Statistik-Rebuild ersetzt datasource_embeddings atomar aus chunk_datasources."""

    def test_recalc_deletes_stale_datasource_embedding_rows(self):
        from kiara.admin.services.integrity_service import _recalc_datasource_embeddings
        from kiara.shared.db import get_connection

        conn = get_connection()
        ds_ids = []
        try:
            row1 = conn.execute(
                "INSERT INTO datasources (name, type, path, is_active) "
                "VALUES ('Recalc DS 1', 'local', '/tmp/recalc1', true) RETURNING id"
            ).fetchone()
            row2 = conn.execute(
                "INSERT INTO datasources (name, type, path, is_active) "
                "VALUES ('Recalc DS 2', 'local', '/tmp/recalc2', true) RETURNING id"
            ).fetchone()
            ds1 = row1["id"]
            ds2 = row2["id"]
            ds_ids.extend([ds1, ds2])
            conn.execute(
                "INSERT INTO chunk_datasources "
                "(chunk_id, datasource_id, embedding_model) "
                "VALUES ('recalc-c1', %s, 'model-a'), ('recalc-c2', %s, 'model-a')",
                (ds1, ds1),
            )
            # Hinweis: chunk_datasources_datasource_id_fkey verhindert seit
            # 0.0.1+423 orphan-Junction-Rows mit nicht existierender DS —
            # entsprechender Pfad in _recalc_datasource_embeddings ist defensiv,
            # kann hier aber nicht mehr provoziert werden.
            conn.execute(
                "INSERT INTO datasource_embeddings "
                "(datasource_id, embedding_model, chunk_count) "
                "VALUES (%s, 'model-a', 99), (%s, 'model-stale', 7)",
                (ds1, ds2),
            )
            conn.commit()
        finally:
            conn.close()

        try:
            _recalc_datasource_embeddings({})

            conn = get_connection()
            try:
                row = conn.execute(
                    "SELECT chunk_count FROM datasource_embeddings "
                    "WHERE datasource_id = %s AND embedding_model = 'model-a'",
                    (ds1,),
                ).fetchone()
                stale = conn.execute(
                    "SELECT 1 FROM datasource_embeddings "
                    "WHERE datasource_id = %s AND embedding_model = 'model-stale'",
                    (ds2,),
                ).fetchone()
            finally:
                conn.close()
            assert row["chunk_count"] == 2
            assert stale is None
        finally:
            conn = get_connection()
            try:
                conn.execute(
                    "DELETE FROM chunk_datasources WHERE datasource_id = ANY(%s)",
                    (ds_ids,),
                )
                conn.execute(
                    "DELETE FROM datasource_embeddings WHERE datasource_id = ANY(%s)",
                    (ds_ids,),
                )
                conn.execute("DELETE FROM datasources WHERE id = ANY(%s)", (ds_ids,))
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Static assertions for tab_integrity.js stale ColBERT UI elements
# ---------------------------------------------------------------------------


class TestTabIntegrityJsStaleColbert:
    """Static assertions for tab_integrity.js stale ColBERT UI elements."""

    @pytest.fixture(autouse=True)
    def _load_js(self):
        with open("kiara/web/static/js/tab_integrity.js") as f:
            self.js_content = f.read()

    def test_check_defs_includes_stale_colbert(self):
        assert "stale_colbert" in self.js_content
        assert "'stale_colbert'" in self.js_content

    def test_repair_config_includes_cleanup_colbert(self):
        assert "repair_cleanup_colbert" in self.js_content
        assert "/api/admin/integrity/cleanup/colbert" in self.js_content

    def test_repair_labels_includes_cleanup_colbert(self):
        assert "repair_cleanup_colbert:" in self.js_content
        assert "'ColBERT-Cleanup'" in self.js_content

    def test_render_stale_colbert_detail_exists(self):
        assert "renderStaleColbertDetail" in self.js_content

    def test_render_check_detail_case_stale_colbert(self):
        assert "case 'stale_colbert':" in self.js_content

    def test_render_repair_buttons_stale_colbert(self):
        assert "stale_collections" in self.js_content
        assert "orphaned_db_entries" in self.js_content

    def test_quick_check_keyspace_field(self):
        assert "chunk_ds_keyspace_distinct_chunks" in self.js_content

    def test_layer_counts_dense_colbert_rows(self):
        assert "dense_chunk_ds_total_rows" in self.js_content
        assert "colbert_chunk_ds_total_rows" in self.js_content
        assert "dense_ds_embeddings_total" in self.js_content
        assert "colbert_ds_embeddings_total" in self.js_content

    def test_escape_html_for_stale_values(self):
        # renderStaleColbertDetail uses escapeHtml for dynamic values
        assert "escapeHtml(c.name)" in self.js_content
        assert "escapeHtml(c.model)" in self.js_content
        assert "escapeHtml(c.reason)" in self.js_content

    def test_layer_counts_order(self):
        # Verify ordering: Qdrant -> ES(optional) -> chunk_ds_count -> keyspace -> total_rows -> dense/colbert -> ds_embeddings -> dense/colbert -> file_hashes
        idx_keyspace = self.js_content.index("chunk_ds_keyspace_distinct_chunks")
        idx_total_rows = self.js_content.index("chunk_ds_total_rows", idx_keyspace)
        idx_dense = self.js_content.index("dense_chunk_ds_total_rows", idx_total_rows)
        idx_file = self.js_content.index("file_hashes_count", idx_dense)
        assert idx_keyspace < idx_total_rows < idx_dense < idx_file


# =====================================================================
# Issue #1058 — Integrity-Repair vs. laufende Ingestion (Race-Vermeidung)
# =====================================================================


class TestStartRepairIngestionGuard:
    """start_repair muss laufende Ingestion-Jobs erkennen und ablehnen."""

    @pytest.fixture(autouse=True)
    def _reset_idle(self):
        """Fuer diese Klasse muss _task auf running=False stehen, sonst greift der spaetere Lock-Check."""
        from kiara.admin.services import integrity_service as svc

        with svc._lock:
            svc._task.update({"running": False, "action": None, "progress": None, "result": None})
        yield
        with svc._lock:
            svc._task.update({"running": False, "action": None, "progress": None, "result": None})

    def test_start_repair_blocked_when_ingestion_running(self):
        """Wenn ein Ingestion-Job aktiv ist, wirft start_repair RuntimeError BEVOR der lokale Lock greift."""
        from kiara.admin.services.integrity_service import start_repair

        with patch("kiara.web.ingestion_worker.get_running_job_id", return_value=42):
            with pytest.raises(RuntimeError, match="Ingestion-Job #42 laeuft"):
                start_repair("repair_orphans")

    def test_start_repair_proceeds_when_no_ingestion(self):
        """Ohne aktiven Ingestion-Job laeuft start_repair durch (Thread wird gestartet)."""
        from kiara.admin.services import integrity_service as svc

        with patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None):
            # _run_repair_orphans selbst mocken, damit kein echter Repair laeuft
            with patch.object(svc, "_run_repair_orphans", lambda: None):
                svc.start_repair("repair_orphans")
        # Nach Aufruf: _task hat action gesetzt
        with svc._lock:
            assert svc._task["action"] == "repair_orphans"

    def test_start_repair_unknown_action_still_validated_first(self):
        """Unbekannte Actions werden weiterhin VOR dem Ingestion-Check abgelehnt (ValueError)."""
        from kiara.admin.services.integrity_service import start_repair

        # Auch wenn Ingestion laeuft: ValueError fuer unknown action geht zuerst
        with patch("kiara.web.ingestion_worker.get_running_job_id", return_value=99):
            with pytest.raises(ValueError, match="Unbekannte Repair-Aktion"):
                start_repair("unknown_action")
