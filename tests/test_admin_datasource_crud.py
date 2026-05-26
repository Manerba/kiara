"""Tests fuer Sprint 05: Datenquellen-CRUD Admin-Funktionen.

Testet datasource.create, .update, .delete, .test_connection, .browse,
.assign_groups, .assign_users — 16 Unit-Tests mit Mocking.
"""

import asyncio
import inspect
import json
from unittest.mock import ANY, MagicMock, patch

import pytest

from kiara.shared.constants import PASSWORD_MASK

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


def _mock_connector_cls(secret_fields=None):
    """Erstellt eine Mock-Connector-Klasse."""
    cls = MagicMock()
    cls.secret_fields.return_value = secret_fields or []
    cls.config_schema.return_value = {"type": "object", "properties": {}}
    cls.default_config.return_value = {}
    cls.supports_browse = False
    cls.name = "local"
    cls.label = "Lokal"
    return cls


# ---------------------------------------------------------------------------
# datasource.create Tests
# ---------------------------------------------------------------------------


class TestDatasourceCreate:
    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.create")
    @patch("kiara.config.load_config")
    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_create_local_datasource(
        self, mock_ensure, mock_get_connector, mock_encrypt, mock_config, mock_create, mock_refresh
    ):
        """Test 1: create mit type=local, verify DB-Eintrag."""
        mock_get_connector.return_value = _mock_connector_cls()
        mock_config.return_value = {"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}}
        mock_create.return_value = 42

        from kiara.admin.functions.datasources import datasource_create

        result = _run(
            datasource_create(
                MOCK_USER,
                name="TestDS",
                type="local",
                connector_config={"path": "/tmp/test"},
            )
        )

        assert result.success is True
        assert result.data["id"] == 42
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["name"] == "TestDS"
        assert call_kwargs["type"] == "local"
        assert call_kwargs["is_active"]
        assert call_kwargs["restriction_level"] == "anonymize"  # Default

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.create")
    @patch("kiara.config.load_config")
    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_create_webdav_password_encrypted(
        self, mock_ensure, mock_get_connector, mock_encrypt, mock_config, mock_create, mock_refresh
    ):
        """Test 2: Passwort in connector_config ist nach create verschluesselt.
        create_datasource ruft encrypt_secret_fields(), das intern kiara.shared.crypto.encrypt
        verwendet — daher wird dort gepatcht (nicht mehr im datasource_service-Namespace)."""
        mock_get_connector.return_value = _mock_connector_cls(secret_fields=["password"])
        mock_encrypt.return_value = "ENCRYPTED_PW"
        mock_config.return_value = {"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}}
        mock_create.return_value = 43

        from kiara.admin.functions.datasources import datasource_create

        result = _run(
            datasource_create(
                MOCK_USER,
                name="WebDAV-Test",
                type="webdav",
                connector_config={"url": "https://example.com", "password": "geheim"},
            )
        )

        assert result.success is True
        call_kwargs = mock_create.call_args[1]
        saved_config = json.loads(call_kwargs["connector_config"])
        assert saved_config["password"] == "ENCRYPTED_PW"
        mock_encrypt.assert_called_once_with("geheim")

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_create_unknown_type_fails(self, mock_ensure, mock_get_connector):
        """Test 3: ValueError bei unbekanntem Konnektor-Typ."""
        mock_get_connector.return_value = None

        from kiara.admin.functions.datasources import datasource_create

        result = _run(
            datasource_create(
                MOCK_USER,
                name="Bad",
                type="ftp",
                connector_config={},
            )
        )

        assert result.success is False
        assert "Unbekannter Konnektor-Typ" in result.message


# ---------------------------------------------------------------------------
# datasource.update Tests
# ---------------------------------------------------------------------------


class TestDatasourceUpdate:
    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_partial_fields(self, mock_get, mock_update, mock_refresh):
        """Test 4: Nur uebergebene Felder werden geaendert."""
        mock_get.return_value = {"id": 1, "name": "Alt", "type": "local", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(datasource_update(MOCK_USER, ds_id=1, name="Neu"))

        assert result.success is True
        mock_update.assert_called_once_with(1, name="Neu")

    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_update_password_mask_preserved(
        self, mock_ensure, mock_get_connector, mock_get, mock_update, mock_refresh, mock_encrypt
    ):
        """Test 5: PASSWORD_MASK → bestehender verschluesselter Wert bleibt."""
        mock_get_connector.return_value = _mock_connector_cls(secret_fields=["password"])
        mock_get.return_value = {
            "id": 5,
            "name": "DS5",
            "type": "webdav",
            "connector_config": json.dumps({"url": "https://x.de", "password": "ENC_OLD"}),
        }

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=5,
                connector_config={"url": "https://y.de", "password": PASSWORD_MASK},
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        saved = json.loads(call_kwargs["connector_config"])
        assert saved["password"] == "ENC_OLD"  # Bestehender Wert beibehalten
        mock_encrypt.assert_not_called()  # Kein Re-Encrypt

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_nonexistent_ds_fails(self, mock_get):
        """Test 6: AdminResult.fail bei nicht existierender DS."""
        mock_get.return_value = None

        from kiara.admin.functions.datasources import datasource_update

        result = _run(datasource_update(MOCK_USER, ds_id=999, name="Nix"))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_integer_ds_fields_get_int_cast(self, mock_get, mock_update, mock_refresh):
        """Regression: integer-Spalten erhalten int-Cast aus boolean-Frontend-Wert.

        Schema-Inkonsistenz: ds_ddu_ocr_images, ds_accept_all_text,
        ds_index_attachments und ds_contextual_retrieval sind in PostgreSQL
        ``integer DEFAULT 0/1``. Das JS-Frontend sendet aber boolean
        (Checkbox.checked). Vor dem Fix schlug der UPDATE mit
        ``Spalte ds_ddu_ocr_images hat Typ integer, aber Ausdruck hat Typ boolean``
        fehl. Der Cast in datasource_update() heilt das.
        """
        mock_get.return_value = {"id": 1, "name": "DS1", "type": "smb", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=1,
                ds_ddu_ocr_images=True,
                ds_accept_all_text=False,
                ds_index_attachments=True,
                ds_contextual_retrieval=False,
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["ds_ddu_ocr_images"] == 1
        assert call_kwargs["ds_accept_all_text"] == 0
        assert call_kwargs["ds_index_attachments"] == 1
        assert call_kwargs["ds_contextual_retrieval"] == 0
        # Strikt int, nicht bool (PostgreSQL behandelt isinstance(True, int) zwar
        # als True, aber die Treiber-Schicht sendet bool als 'true'/'false')
        for col in (
            "ds_ddu_ocr_images",
            "ds_accept_all_text",
            "ds_index_attachments",
            "ds_contextual_retrieval",
        ):
            assert type(call_kwargs[col]) is int, (
                f"{col} ist {type(call_kwargs[col]).__name__}, erwartet int"
            )

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_boolean_ds_fields_stay_bool(self, mock_get, mock_update, mock_refresh):
        """Echte boolean-Spalten (ds_ddu_enabled, ds_vlm_enabled, ds_colbert_enabled)
        werden NICHT gecastet — sie bleiben bool, weil PostgreSQL boolean erwartet."""
        mock_get.return_value = {"id": 1, "name": "DS1", "type": "smb", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=1,
                ds_ddu_enabled=True,
                ds_vlm_enabled=False,
                ds_colbert_enabled=True,
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["ds_ddu_enabled"] is True
        assert call_kwargs["ds_vlm_enabled"] is False
        assert call_kwargs["ds_colbert_enabled"] is True


# ---------------------------------------------------------------------------
# datasource.delete Tests
# ---------------------------------------------------------------------------


class TestDatasourceDelete:
    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_cascade_order(self, mock_config, mock_get, mock_cascade, mock_cache_delete):
        """Test 7: 7-Schritte-Cleanup (Mock VectorStore)."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/test_vdb"},
            "embedding_model": "nomic-embed-text",
        }
        mock_get.return_value = {"id": 10, "name": "TestDS", "type": "local"}

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch("kiara.ingestion._file_utils.ChunkDatasources") as mock_cds,
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch"),
            patch("kiara.repositories.file_hash.FileHashRepository") as mock_fh,
            patch("kiara.ingestion._file_utils.HashStore"),
            patch("kiara.retrieval.colbert.ColBERTStore.list_colbert_collections", return_value=[]),
        ):
            mock_vs.list_collections.return_value = [{"model": "test-model"}]
            store_instance = MagicMock()
            store_instance.get_collection_name.return_value = "kb_test-model"
            mock_vs.return_value = store_instance
            chunk_ds_instance = MagicMock()
            chunk_ds_instance.preview_datasource_removal.return_value = (
                ["c1"],
                {"c2": [2, 3], "c3": [3]},
            )
            mock_cds.return_value = chunk_ds_instance
            mock_fh_instance = MagicMock()
            mock_fh_instance.get_all_hashes.return_value = {}
            mock_fh.return_value = mock_fh_instance

            from kiara.admin.services.datasource_service import delete_datasource_full

            result = _run(delete_datasource_full(10))

        assert result["message"] == "Datenquelle geloescht"
        # Phase 1: Preview (read-only)
        chunk_ds_instance.preview_datasource_removal.assert_called_once_with(10, "test-model")
        # Phase 2: Qdrant-Cleanup
        store_instance.delete_by_ids.assert_called_once_with(["c1"])
        store_instance.update_ds_ids.assert_called_once_with({"c2": [2, 3], "c3": [3]}, wait=True)
        # Phase 3: DB-Cleanup laeuft erst im finalen atomaren Cascade-Block
        chunk_ds_instance.delete_datasource_entries.assert_not_called()
        chunk_ds_instance.preview_graph_orphan_chunk_ids_after_datasource_removal.assert_called_once_with(
            10, conn=ANY
        )
        # Negative Asserts: alte atomare Methode darf NICHT aufgerufen werden
        chunk_ds_instance.remove_datasource_full.assert_not_called()
        chunk_ds_instance.get_chunk_ids.assert_not_called()
        chunk_ds_instance.remove_datasource.assert_not_called()
        chunk_ds_instance.get_datasource_ids_for_chunks.assert_not_called()
        mock_cascade.assert_called_once_with(10, conn=ANY)

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_nonexistent_ds_fails(self, mock_config, mock_get):
        """Test 8: ValueError bei nicht existierender DS."""
        mock_config.return_value = {"vectordb": {"path": "/tmp"}}
        mock_get.return_value = None

        from kiara.admin.functions.datasources import datasource_delete

        result = _run(datasource_delete(MOCK_USER, ds_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# datasource.assign_groups / assign_users Tests
# ---------------------------------------------------------------------------


class TestDatasourceAssign:
    @patch("kiara.repositories.permission.PermissionRepository.set_datasource_groups")
    @patch("kiara.repositories.datasource.DatasourceRepository.exists")
    def test_assign_groups_replace_all(self, mock_exists, mock_set_groups):
        """Test 9: Bestehende Zuordnungen werden ersetzt."""
        mock_exists.return_value = True

        from kiara.admin.functions.datasources import datasource_assign_groups

        result = _run(datasource_assign_groups(MOCK_USER, ds_id=5, group_ids=[1, 3]))

        assert result.success is True
        mock_set_groups.assert_called_once_with(5, [1, 3])

    @patch("kiara.repositories.permission.PermissionRepository.set_datasource_users")
    @patch("kiara.repositories.datasource.DatasourceRepository.exists")
    def test_assign_users_empty_list(self, mock_exists, mock_set_users):
        """Test 10: Leere Liste entfernt alle Zuordnungen."""
        mock_exists.return_value = True

        from kiara.admin.functions.datasources import datasource_assign_users

        result = _run(datasource_assign_users(MOCK_USER, ds_id=8, user_ids=[]))

        assert result.success is True
        mock_set_users.assert_called_once_with(8, [])


# ---------------------------------------------------------------------------
# datasource.test_connection Tests
# ---------------------------------------------------------------------------


class TestDatasourceTestConnection:
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_test_connection_success(self, mock_ensure, mock_get_connector):
        """Test 11: safe_test_connection mit Mock-Connector."""
        mock_cls = _mock_connector_cls()
        mock_cls.safe_test_connection.return_value = {"success": True, "message": "OK"}
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_test_connection

        result = _run(
            datasource_test_connection(
                MOCK_USER,
                type="local",
                config={"path": "/tmp"},
            )
        )

        assert result.success is True
        assert result.data["success"] is True

    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_test_connection_with_ds_id_resolves_mask(
        self, mock_ensure, mock_get_connector, mock_get, mock_encrypt
    ):
        """Test 12: PASSWORD_MASK wird aus bestehender DS aufgeloest."""
        mock_cls = _mock_connector_cls(secret_fields=["password"])
        # safe_test_connection mutiert das Dict im finally-Block (loescht Klartext-PW
        # aus Speicher) — daher den Wert beim Aufruf wegspeichern, nicht spaeter
        # ueber call_args lesen.
        captured = {}

        def _capture(config, row):
            captured.update(config)
            return {"success": True}

        mock_cls.safe_test_connection.side_effect = _capture
        mock_get_connector.return_value = mock_cls

        mock_get.return_value = {
            "id": 5,
            "connector_config": json.dumps({"password": "ENC_VAL"}),
        }

        from kiara.admin.functions.datasources import datasource_test_connection

        result = _run(
            datasource_test_connection(
                MOCK_USER,
                type="webdav",
                config={"url": "https://x.de", "password": PASSWORD_MASK},
                ds_id=5,
            )
        )

        assert result.success is True
        assert captured["password"] == "ENC_VAL"

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_test_connection_rejects_mask_from_other_connector_type(
        self,
        mock_ensure,
        mock_get_connector,
        mock_get,
    ):
        """PASSWORD_MASK darf nicht aus einer DS anderen Typs stammen."""
        mock_cls = _mock_connector_cls(secret_fields=["password"])
        mock_get_connector.return_value = mock_cls
        mock_get.return_value = {
            "id": 5,
            "type": "imap",
            "connector_config": json.dumps({"password": "ENC_VAL"}),
        }

        from kiara.admin.functions.datasources import datasource_test_connection

        result = _run(
            datasource_test_connection(
                MOCK_USER,
                type="webdav",
                config={"url": "https://x.de", "password": PASSWORD_MASK},
                ds_id=5,
            )
        )

        assert result.success is False
        assert "Konnektor-Typ" in result.message
        mock_cls.safe_test_connection.assert_not_called()


# ---------------------------------------------------------------------------
# datasource.browse Tests
# ---------------------------------------------------------------------------


class TestDatasourceBrowse:
    def test_browse_local_directory(self, tmp_path):
        """Test 13: Lokales Verzeichnis listen, Eintraege mit is_dir."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file.txt").write_text("test")

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="local",
                config={"path": str(tmp_path)},
            )
        )

        assert result.success is True
        entries = result.data["entries"]
        names = [e["name"] for e in entries]
        assert "subdir" in names
        assert "file.txt" in names
        subdir_entry = next(e for e in entries if e["name"] == "subdir")
        assert subdir_entry["is_dir"] is True

    def test_browse_local_nonexistent_fails(self):
        """Test 14: Nicht existierendes Verzeichnis → Fehler."""
        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="local",
                config={"path": "/nonexistent/path/xyz"},
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.connectors.registry.get")
    def test_browse_imap_list_folders(self, mock_get_connector):
        """Test 15: IMAP connector.list_folders mit Mock."""
        mock_cls = MagicMock()
        mock_cls.secret_fields.return_value = []
        mock_cls.list_folders.return_value = ["INBOX", "Sent", "Trash"]
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="imap",
                config={"host": "mail.example.com"},
            )
        )

        assert result.success is True
        assert result.data["folders"] == ["INBOX", "Sent", "Trash"]

    @patch("kiara.connectors.registry.get")
    def test_browse_exchange_online_list_folders(self, mock_get_connector):
        """Exchange Online nutzt weiter list_folders."""
        mock_cls = MagicMock()
        mock_cls.secret_fields.return_value = []
        mock_cls.list_folders.return_value = ["INBOX"]
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="exchange_online",
                config={"username": "mail@example.com"},
            )
        )

        assert result.success is True
        assert result.data["folders"] == ["INBOX"]

    def test_browse_unsupported_type_fails(self):
        """Test 16: Unbekannter Typ → Fehler."""
        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="ftp",
                config={},
            )
        )

        assert result.success is False
        assert "nicht unterstützt" in result.message


# ===========================================================================
# Erweiterte Tests: Edge-Cases + Failure-Pfade
# ===========================================================================


class TestResolveMaskedConfig:
    """Tests fuer resolve_masked_config() — alle 3 Branches."""

    def test_plaintext_password_gets_encrypted(self):
        """Neues Klartext-Passwort wird mit encrypt() verschluesselt."""
        mock_cls = _mock_connector_cls(secret_fields=["password"])

        with patch("kiara.admin.services.datasource_service.encrypt") as mock_encrypt:
            mock_encrypt.return_value = "ENC_NEW"
            from kiara.admin.services.datasource_service import resolve_masked_config

            result = resolve_masked_config(
                {"url": "https://x.de", "password": "neues_pw"},
                {"url": "https://x.de", "password": "ENC_OLD"},
                mock_cls,
            )

        assert result["password"] == "ENC_NEW"
        mock_encrypt.assert_called_once_with("neues_pw")

    def test_empty_password_not_encrypted(self):
        """Leeres Passwort-Feld wird nicht verschluesselt."""
        mock_cls = _mock_connector_cls(secret_fields=["password"])

        with patch("kiara.admin.services.datasource_service.encrypt") as mock_encrypt:
            from kiara.admin.services.datasource_service import resolve_masked_config

            result = resolve_masked_config(
                {"password": ""},
                {"password": "ENC_OLD"},
                mock_cls,
            )

        assert result["password"] == ""
        mock_encrypt.assert_not_called()

    def test_multiple_secret_fields_mixed(self):
        """Mix aus MASK + Klartext bei mehreren Secret-Feldern."""
        mock_cls = _mock_connector_cls(secret_fields=["password", "api_key"])

        with patch("kiara.admin.services.datasource_service.encrypt") as mock_encrypt:
            mock_encrypt.return_value = "ENC_KEY"
            from kiara.admin.services.datasource_service import resolve_masked_config

            result = resolve_masked_config(
                {"password": PASSWORD_MASK, "api_key": "new_key_123"},
                {"password": "ENC_OLD_PW", "api_key": "ENC_OLD_KEY"},
                mock_cls,
            )

        assert result["password"] == "ENC_OLD_PW"  # MASK → beibehalten
        assert result["api_key"] == "ENC_KEY"  # Klartext → verschluesselt
        mock_encrypt.assert_called_once_with("new_key_123")


class TestDeleteDatasourceExceptionHandling:
    """Tests fuer delete_datasource_full() — Exception-Toleranz der 7 Schritte."""

    def test_es_file_hash_snapshot_is_taken_inside_delete_lock(self):
        """ES-Cleanup nutzt keinen spaeteren file_hashes-Read ausserhalb des Delete-Locks."""
        from kiara.admin.services.datasource_service import delete_datasource_full

        src = inspect.getsource(delete_datasource_full)
        snapshot_idx = src.index("SELECT DISTINCT file_path FROM file_hashes")
        external_cleanup_idx = src.index("# 1. Qdrant-Cleanup")
        es_cleanup_idx = src.index("# 2. Elasticsearch")

        assert snapshot_idx < external_cleanup_idx
        es_block = src[es_cleanup_idx : src.index("# 3. DB-Cascade")]
        assert "db_query" not in es_block

    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_survives_qdrant_exception(
        self, mock_config, mock_get, mock_cascade, mock_cache
    ):
        """Qdrant-Fehler ist non-fatal — Loeschung laeuft weiter."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
        }
        mock_get.return_value = {"id": 10, "name": "DS", "type": "local"}

        with patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs:
            mock_vs.list_collections.side_effect = Exception("Qdrant down")

            from kiara.admin.services.datasource_service import delete_datasource_full

            result = _run(delete_datasource_full(10))

        assert result["message"] == "Datenquelle geloescht"
        mock_cascade.assert_called_once_with(10, conn=ANY)  # DB-Cascade trotzdem ausgefuehrt

    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_survives_hashstore_exception(
        self, mock_config, mock_get, mock_cascade, mock_cache
    ):
        """HashStore-Fehler ist non-fatal."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
        }
        mock_get.return_value = {"id": 10, "name": "DS", "type": "local"}

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch("kiara.ingestion._file_utils.ChunkDatasources"),
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch"),
            patch("kiara.repositories.file_hash.FileHashRepository"),
            patch("kiara.ingestion._file_utils.HashStore") as mock_hs,
        ):
            mock_vs.list_collections.return_value = []
            mock_hs.side_effect = Exception("HashStore kaputt")

            from kiara.admin.services.datasource_service import delete_datasource_full

            result = _run(delete_datasource_full(10))

        assert result["message"] == "Datenquelle geloescht"
        mock_cascade.assert_called_once()

    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.web.cron_manager.remove_schedule")
    @patch("kiara.shared.mount.unmount")
    @patch("kiara.connectors.registry.auto_mount_point")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_smb_with_mount_and_schedule_exceptions(
        self,
        mock_config,
        mock_get,
        mock_cascade,
        mock_mount_point,
        mock_unmount,
        mock_remove_schedule,
        mock_cache,
    ):
        """Mount-Unmount und Schedule-Fehler sind non-fatal bei SMB-DS."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
        }
        mock_get.return_value = {"id": 10, "name": "DS", "type": "smb"}
        mock_mount_point.return_value = "/mnt/kiara/ds_10"
        mock_unmount.side_effect = Exception("umount fehlgeschlagen")
        mock_remove_schedule.side_effect = Exception("Schedule nicht gefunden")

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch("kiara.ingestion._file_utils.ChunkDatasources"),
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch"),
            patch("kiara.repositories.file_hash.FileHashRepository"),
            patch("kiara.ingestion._file_utils.HashStore"),
        ):
            mock_vs.list_collections.return_value = []

            from kiara.admin.services.datasource_service import delete_datasource_full

            result = _run(delete_datasource_full(10))

        assert result["message"] == "Datenquelle geloescht"
        mock_unmount.assert_called_once()
        mock_cascade.assert_called_once()

    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_local_ds_skips_mount(self, mock_config, mock_get, mock_cascade, mock_cache):
        """Lokale DS ueberspringt Mount-Unmount (type != smb/webdav)."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
        }
        mock_get.return_value = {"id": 10, "name": "DS", "type": "local"}

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch("kiara.ingestion._file_utils.ChunkDatasources"),
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch"),
            patch("kiara.repositories.file_hash.FileHashRepository"),
            patch("kiara.ingestion._file_utils.HashStore"),
            patch("kiara.web.cron_manager.remove_schedule"),
            patch("kiara.shared.mount.unmount") as mock_unmount,
        ):
            mock_vs.list_collections.return_value = []

            from kiara.admin.services.datasource_service import delete_datasource_full

            result = _run(delete_datasource_full(10))

        mock_unmount.assert_not_called()


class TestDatasourceUpdateEdgeCases:
    """Zusaetzliche Edge-Cases fuer datasource.update."""

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_no_changes_fails(self, mock_get):
        """Nur ds_id ohne Aenderungen → 'Keine Aenderungen angegeben'."""
        mock_get.return_value = {"id": 1, "name": "DS", "type": "local", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(datasource_update(MOCK_USER, ds_id=1))

        assert result.success is False
        assert "Keine" in result.message

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_boolean_to_int_conversion(self, mock_get, mock_update, mock_refresh):
        """Boolean-Werte werden zu int konvertiert (0/1)."""
        mock_get.return_value = {"id": 1, "name": "DS", "type": "local", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=1,
                ds_ddu_enabled=True,
                ds_vlm_enabled=False,
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["ds_ddu_enabled"]
        assert not call_kwargs["ds_vlm_enabled"]

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_restriction_level(self, mock_get, mock_update, mock_refresh):
        """restriction_level wird als ds-Feld korrekt durchgereicht."""
        mock_get.return_value = {"id": 1, "name": "DS", "type": "local", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=1,
                restriction_level="blocked",
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["restriction_level"] == "blocked"

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_list_to_json(self, mock_get, mock_update, mock_refresh):
        """Listen werden als JSON serialisiert."""
        mock_get.return_value = {"id": 1, "name": "DS", "type": "local", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=1,
                ds_exclude_patterns=["*.tmp", "*.bak", "~*"],
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["ds_exclude_patterns"] == '["*.tmp", "*.bak", "~*"]'

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.update")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_update_empty_list_to_empty_string(self, mock_get, mock_update, mock_refresh):
        """Leere Listen werden als leerer String gespeichert (nicht '[]')."""
        mock_get.return_value = {"id": 1, "name": "DS", "type": "local", "connector_config": "{}"}

        from kiara.admin.functions.datasources import datasource_update

        result = _run(
            datasource_update(
                MOCK_USER,
                ds_id=1,
                ds_file_types=[],
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["ds_file_types"] == ""


class TestDatasourceTestConnectionEdgeCases:
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_test_connection_failure_returns_fail(self, mock_ensure, mock_get_connector):
        """safe_test_connection mit success=False → AdminResult.fail."""
        mock_cls = _mock_connector_cls()
        mock_cls.safe_test_connection.return_value = {
            "success": False,
            "message": "Connection refused",
        }
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_test_connection

        result = _run(
            datasource_test_connection(
                MOCK_USER,
                type="webdav",
                config={"url": "https://dead.host"},
            )
        )

        assert result.success is False
        assert "Connection refused" in result.message

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_test_connection_unknown_type(self, mock_ensure, mock_get_connector):
        """Unbekannter Konnektor-Typ → AdminResult.fail."""
        mock_get_connector.return_value = None

        from kiara.admin.functions.datasources import datasource_test_connection

        result = _run(
            datasource_test_connection(
                MOCK_USER,
                type="ftp",
                config={},
            )
        )

        assert result.success is False
        assert "Unbekannter Konnektor-Typ" in result.message


class TestDatasourceBrowseEdgeCases:
    def test_browse_local_with_explicit_path(self, tmp_path):
        """path-Parameter wird relativ zu config.path gebrowst."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "inner.txt").write_text("hi")

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="local",
                config={"path": str(tmp_path)},
                path="sub",
            )
        )

        assert result.success is True
        assert result.data["path"] == "sub"
        assert result.data["parent"] == ""
        assert any(e["name"] == "inner.txt" for e in result.data["entries"])

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_smb_uses_connector_capability(self, mock_ensure, mock_get_connector):
        """SMB-Browse delegiert an list_directory."""
        mock_cls = _mock_connector_cls()
        mock_cls.supports_browse = True
        mock_cls.list_directory.return_value = {"entries": [{"name": "A", "is_dir": True}]}
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"server": "fileserver"},
                path="a\\b",
            )
        )

        assert result.success is True
        assert result.data["path"] == "a/b"
        assert result.data["parent"] == "a"
        assert result.data["entries"] == [{"name": "A", "is_dir": True}]
        mock_cls.list_directory.assert_called_once_with(
            {"server": "fileserver"},
            "a/b",
            directories_only=False,
        )

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_smb_connector_error_fails(self, mock_ensure, mock_get_connector):
        from kiara.connectors.browse import ConnectorBrowseError

        mock_cls = _mock_connector_cls()
        mock_cls.supports_browse = True
        mock_cls.list_directory.side_effect = ConnectorBrowseError(
            "Remote kaputt", 502, "bad_response"
        )
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"server": "fileserver"},
            )
        )

        assert result.success is False
        assert "Remote kaputt" in result.message

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_webdav_uses_connector_capability(self, mock_ensure, mock_get_connector):
        """WebDAV-Browse delegiert an list_directory."""
        mock_cls = _mock_connector_cls()
        mock_cls.supports_browse = True
        mock_cls.list_directory.return_value = {"entries": [{"name": "Docs", "is_dir": True}]}
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="webdav",
                config={"url": "https://x.de"},
            )
        )

        assert result.success is True
        assert result.data["entries"] == [{"name": "Docs", "is_dir": True}]
        mock_cls.list_directory.assert_called_once_with(
            {"url": "https://x.de"},
            "",
            directories_only=False,
        )

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_password_mask_without_ds_id_fails(self, mock_ensure, mock_get_connector):
        mock_cls = _mock_connector_cls(secret_fields=["password"])
        mock_cls.supports_browse = True
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"password": PASSWORD_MASK},
            )
        )

        assert result.success is False
        assert "Passwort erforderlich" in result.message
        mock_cls.list_directory.assert_not_called()

    @patch("kiara.admin.functions.datasources.DatasourceRepository")
    @patch("kiara.shared.crypto.decrypt", return_value="clear-secret")
    @patch("kiara.shared.crypto.is_encrypted", return_value=True)
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_resolves_masked_password_with_ds_id(
        self,
        mock_ensure,
        mock_get_connector,
        mock_is_encrypted,
        mock_decrypt,
        mock_repo_cls,
    ):
        mock_cls = _mock_connector_cls(secret_fields=["password"])
        mock_cls.supports_browse = True
        mock_cls.list_directory.return_value = {"entries": []}
        mock_get_connector.return_value = mock_cls
        mock_repo_cls.return_value.get_by_id.return_value = {
            "id": 42,
            "type": "smb",
            "connector_config": json.dumps({"server": "files", "password": "ENC_SECRET"}),
        }

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"server": "files", "password": PASSWORD_MASK},
                ds_id=42,
            )
        )

        assert result.success is True
        effective_config = mock_cls.list_directory.call_args.args[0]
        assert effective_config["password"] == "clear-secret"

    @patch("kiara.admin.functions.datasources.DatasourceRepository")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_masked_missing_secret_fails(
        self,
        mock_ensure,
        mock_get_connector,
        mock_repo_cls,
    ):
        mock_cls = _mock_connector_cls(secret_fields=["password"])
        mock_cls.supports_browse = True
        mock_get_connector.return_value = mock_cls
        mock_repo_cls.return_value.get_by_id.return_value = {
            "id": 42,
            "type": "smb",
            "connector_config": json.dumps({"server": "files"}),
        }

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"server": "files", "password": PASSWORD_MASK},
                ds_id=42,
            )
        )

        assert result.success is False
        assert "Passwort erforderlich" in result.message
        mock_cls.list_directory.assert_not_called()

    @patch("kiara.admin.functions.datasources.DatasourceRepository")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_ds_id_wrong_type_fails_before_connector_call(
        self,
        mock_ensure,
        mock_get_connector,
        mock_repo_cls,
    ):
        mock_cls = _mock_connector_cls()
        mock_cls.supports_browse = True
        mock_get_connector.return_value = mock_cls
        mock_repo_cls.return_value.get_by_id.return_value = {
            "id": 42,
            "type": "webdav",
            "connector_config": "{}",
        }

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"server": "files"},
                ds_id=42,
            )
        )

        assert result.success is False
        assert "Konnektor-Typ" in result.message
        mock_cls.list_directory.assert_not_called()

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_fail_closed_supports_browse_must_be_true(self, mock_ensure, mock_get_connector):
        mock_cls = _mock_connector_cls()
        mock_cls.supports_browse = "yes"
        mock_get_connector.return_value = mock_cls

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={"server": "files"},
            )
        )

        assert result.success is False
        assert "nicht unterstützt" in result.message
        mock_cls.list_directory.assert_not_called()

    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_legacy_exchange_alias_fails(self, mock_ensure, mock_get_connector):
        mock_get_connector.return_value = None

        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="exchange",
                config={},
            )
        )

        assert result.success is False
        assert "nicht unterstützt" in result.message

    def test_browse_admin_function_schema_accepts_ds_id(self):
        import kiara.admin.functions.datasources  # noqa: F401
        from kiara.admin.function_tool import AdminFunctionTool
        from kiara.admin.registry import admin_registry

        func = admin_registry.get("datasource.browse")
        schema = func.parameters
        ds_id_schema = schema["properties"]["ds_id"]
        assert ds_id_schema["type"] == "integer"
        assert ds_id_schema["minimum"] == 1
        assert "ds_id" not in schema["required"]

        ok, msg = AdminFunctionTool(func).validate_params(
            {
                "type": "smb",
                "config": {},
                "ds_id": 42,
            }
        )
        assert ok is True, msg

        ok, msg = AdminFunctionTool(func).validate_params(
            {
                "type": "smb",
                "config": {},
                "ds_id": 0,
            }
        )
        assert ok is False
        assert "Minimalwert" in msg

    @patch("kiara.connectors.registry.ensure_loaded")
    def test_browse_direct_ds_id_zero_fails_before_lookup(self, mock_ensure):
        from kiara.admin.functions.datasources import datasource_browse

        result = _run(
            datasource_browse(
                MOCK_USER,
                type="smb",
                config={},
                ds_id=0,
            )
        )

        assert result.success is False
        mock_ensure.assert_not_called()

    def test_admin_function_metadata_uses_current_connector_type_tokens(self):
        import kiara.admin.functions.datasources  # noqa: F401
        from kiara.admin.registry import admin_registry

        create = admin_registry.get("datasource.create")
        browse = admin_registry.get("datasource.browse")
        connector_types = admin_registry.get("datasource.connector_types")

        assert "exchange_online" in create.parameters["properties"]["type"]["enum"]
        assert "exchange" not in create.parameters["properties"]["type"]["enum"]
        assert "sqlds" not in create.parameters["properties"]["type"]["enum"]
        assert "exchange_online" in browse.parameters["properties"]["type"]["description"]
        assert "exchange_online" in connector_types.help
        assert "sqlds" not in connector_types.help


# ---------------------------------------------------------------------------
# ChunkDatasources.remove_datasource_full() Tests
# ---------------------------------------------------------------------------


class TestRemoveDatasourceFull:
    """Unit-Tests fuer ChunkDatasources.remove_datasource_full()."""

    def test_remove_datasource_full_basic(self):
        """remove_datasource_full gibt korrekte Orphans und ds_ids_map zurueck."""
        from kiara.ingestion._file_utils import ChunkDatasources

        with patch("kiara_ingestion_core.index_state.db_transaction") as mock_tx:
            mock_conn = MagicMock()
            mock_tx.return_value.__enter__.return_value = mock_conn

            # Setup: Chunks A+B gehoeren zu DS 1.
            # Nach DELETE: Chunk B hat noch DS 2, Chunk A ist Orphan.
            select_chunks = MagicMock()
            select_chunks.fetchall.return_value = [
                {"chunk_id": "a"},
                {"chunk_id": "b"},
            ]
            delete_result = MagicMock()
            select_remaining = MagicMock()
            select_remaining.fetchall.return_value = [
                {"chunk_id": "b", "datasource_id": 2},
            ]
            mock_conn.execute.side_effect = [
                select_chunks,
                delete_result,
                select_remaining,
            ]

            chunk_ds = ChunkDatasources()
            orphans, ds_map = chunk_ds.remove_datasource_full(1, "test-model")

            assert orphans == ["a"]
            assert ds_map == {"b": [2]}

    def test_remove_datasource_full_propagates_exception(self):
        """Exception in DB-Transaktion wird nicht verschluckt (implizites Rollback)."""
        from kiara.ingestion._file_utils import ChunkDatasources

        with patch("kiara_ingestion_core.index_state.db_transaction") as mock_tx:
            mock_conn = MagicMock()
            mock_tx.return_value.__enter__.return_value = mock_conn

            # SELECT OK, DELETE wirft Exception
            select_chunks = MagicMock()
            select_chunks.fetchall.return_value = [{"chunk_id": "a"}]
            mock_conn.execute.side_effect = [
                select_chunks,
                Exception("DB write error"),
            ]

            chunk_ds = ChunkDatasources()
            with pytest.raises(Exception, match="DB write error"):
                chunk_ds.remove_datasource_full(1, "test-model")

    def test_remove_datasource_full_empty(self):
        """DS ohne Chunks: leere Listen zurueck, kein DELETE."""
        from kiara.ingestion._file_utils import ChunkDatasources

        with patch("kiara_ingestion_core.index_state.db_transaction") as mock_tx:
            mock_conn = MagicMock()
            mock_tx.return_value.__enter__.return_value = mock_conn

            select_empty = MagicMock()
            select_empty.fetchall.return_value = []
            mock_conn.execute.return_value = select_empty

            chunk_ds = ChunkDatasources()
            orphans, ds_map = chunk_ds.remove_datasource_full(999, "test-model")

            assert orphans == []
            assert ds_map == {}
            # Nur 1 SELECT, kein DELETE
            mock_conn.execute.assert_called_once()
