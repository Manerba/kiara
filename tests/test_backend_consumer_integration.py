"""Integration-Tests: Consumer-Module mit dem Backend-System.

Prueft, dass VectorStoreBase, Pipeline, api_datasource_features
und andere Consumer korrekt mit get_*_backend() aus config.py arbeiten.
"""

import sys
import types

# ldap3 ist eine optionale Abhaengigkeit und steht in der Test-Umgebung nicht
# zur Verfuegung. Stub-Modul vorab registrieren, damit der transitive Import-Pfad
# admin_app -> api_health -> health.checker -> checks_services -> shared.ldap
# nicht abbricht.
if "ldap3" not in sys.modules:
    _ldap3_stub = types.ModuleType("ldap3")
    _ldap3_stub.Server = object
    _ldap3_stub.Connection = object
    _ldap3_stub.NONE = None
    _ldap3_stub.SUBTREE = None
    _ldap3_stub.Tls = object
    sys.modules["ldap3"] = _ldap3_stub

from unittest.mock import MagicMock, patch

import pytest

import kiara.web.api_datasource_features as _vlm_features


@pytest.fixture(autouse=True)
def _reset_vlm_cache():
    """VLM-Cache vor jedem Test zuruecksetzen, damit keine Cache-Leaks auftreten."""
    _vlm_features._vlm_cache_result = None
    _vlm_features._vlm_cache_ts = 0.0
    yield
    _vlm_features._vlm_cache_result = None
    _vlm_features._vlm_cache_ts = 0.0


def _base_config(**overrides):
    """Minimale YAML-Config fuer Tests."""
    config = {
        "ollama": {
            "url": "http://yaml-default:11434",
            "llm_url": "http://yaml-llm:11434",
            "embedding_url": "http://yaml-embed:11434",
            "embedding_model": "nomic-embed-text",
            "llm_model": "qwen3:8b",
            "vlm_model": "",
        },
        "cloud": {
            "enabled": False,
            "provider": "anthropic",
            "api_key": "",
            "model": "claude-sonnet-4-6",
            "base_url": "",
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v
    return config


# ===================================================================
# A) VectorStoreBase (retrieval/vectorstore_base.py)
# ===================================================================


class TestVectorStoreBaseBackend:
    def test_embedding_model_from_db(self):
        """VectorStoreBase holt Embedding-Modell via get_embedding_model()."""
        config = _base_config()
        db_backend = {
            "url": "http://db-emb:11434",
            "model": "mxbai-embed-large",
            "id": 1,
            "category": "embedding",
            "provider": "ollama",
        }
        with (
            patch("kiara.backend_selector.get_next_backend", return_value=db_backend),
            patch(
                "kiara.retrieval.vectorstore_base.get_embedding_model",
                return_value="mxbai-embed-large",
            ),
        ):
            from kiara.retrieval.vectorstore_base import VectorStoreBase

            # VectorStoreBase ist abstrakt → testen wir nur __init__
            class DummyStore(VectorStoreBase):
                def add(self, *a, **kw):
                    pass

                def query(self, *a, **kw):
                    return []

                def delete_by_sources(self, *a):
                    pass

                def delete_by_ids(self, *a):
                    pass

                def get_ids_by_sources(self, *a):
                    return []

                def count_total(self):
                    return 0

                def query_by_ids(self, *a, **kw):
                    return []

                def scroll_all(self, *a, **kw):
                    return iter([])

                def update_ds_ids(self, *a, **kw):
                    pass

                def remove_payload_field(self, *a):
                    pass

                def get_stats(self):
                    return {}

                def get_directories(self):
                    return []

                def get_file_types(self):
                    return []

                def add_with_embeddings(self, *a, **kw):
                    pass

                def get_collection_name(self):
                    return "test"

                def get_all_documents(self, include=None):
                    return {"ids": [], "documents": [], "metadatas": []}

                def get_documents_by_ids(self, ids, include=None):
                    return {"ids": [], "documents": [], "metadatas": []}

                def search(self, query_text="", k=10, offset=0, where=None):
                    return []

                def count_filtered(self, where=None):
                    return 0

                @classmethod
                def list_collections(cls, config):
                    return []

                def list_sources_for_datasource(self, ds_id, batch_size=1000):
                    return []

            store = DummyStore(config)
            assert store._embed_model == "mxbai-embed-large"

    def test_explicit_model_overrides_db(self):
        """Explizit uebergebenes Embedding-Model hat Vorrang."""
        config = _base_config()
        db_backend = {
            "url": "http://db-emb:11434",
            "model": "db-model",
            "id": 1,
            "category": "embedding",
            "provider": "ollama",
        }
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            from kiara.retrieval.vectorstore_base import VectorStoreBase

            class DummyStore(VectorStoreBase):
                def add(self, *a, **kw):
                    pass

                def query(self, *a, **kw):
                    return []

                def delete_by_sources(self, *a):
                    pass

                def delete_by_ids(self, *a):
                    pass

                def get_ids_by_sources(self, *a):
                    return []

                def count_total(self):
                    return 0

                def query_by_ids(self, *a, **kw):
                    return []

                def scroll_all(self, *a, **kw):
                    return iter([])

                def update_ds_ids(self, *a, **kw):
                    pass

                def remove_payload_field(self, *a):
                    pass

                def get_stats(self):
                    return {}

                def get_directories(self):
                    return []

                def get_file_types(self):
                    return []

                def add_with_embeddings(self, *a, **kw):
                    pass

                def get_collection_name(self):
                    return "test"

                def get_all_documents(self, include=None):
                    return {"ids": [], "documents": [], "metadatas": []}

                def get_documents_by_ids(self, ids, include=None):
                    return {"ids": [], "documents": [], "metadatas": []}

                def search(self, query_text="", k=10, offset=0, where=None):
                    return []

                def count_filtered(self, where=None):
                    return 0

                @classmethod
                def list_collections(cls, config):
                    return []

                def list_sources_for_datasource(self, ds_id, batch_size=1000):
                    return []

            store = DummyStore(config, embedding_model="override-model")
            assert store._embed_model == "override-model"


# ===================================================================
# C) VLM-Check (web/api_datasource_features.py)
# ===================================================================


class TestVlmAvailableBackend:
    def test_vlm_available_uses_db_backend(self):
        """vlm_available() nutzt get_vlm_backend() aus config.py."""
        from fastapi.testclient import TestClient

        from kiara.web.admin_app import app
        from kiara.web.auth import get_admin_user

        user = {"id": 1, "username": "admin", "is_admin": True}
        app.dependency_overrides[get_admin_user] = lambda: user

        try:
            db_backend = {
                "url": "http://db-vlm:11434",
                "model": "llava:7b",
                "id": 1,
                "category": "vlm",
                "provider": "ollama",
            }
            mock_ollama = MagicMock()
            mock_ollama.list.return_value = {"models": [{"name": "llava:7b"}, {"name": "qwen3:8b"}]}

            with (
                patch("kiara.backend_selector.get_next_backend", return_value=db_backend),
                patch("kiara.web.api_datasource_features.load_config", return_value=_base_config()),
                patch("ollama.Client", return_value=mock_ollama),
                TestClient(app) as client,
            ):
                resp = client.get("/api/admin/vlm-available")

            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is True
            assert data["model"] == "llava:7b"
        finally:
            app.dependency_overrides.pop(get_admin_user, None)

    def test_vlm_not_configured_returns_unavailable(self):
        """Kein VLM-Backend → available: false."""
        from fastapi.testclient import TestClient

        from kiara.web.admin_app import app
        from kiara.web.auth import get_admin_user

        user = {"id": 1, "username": "admin", "is_admin": True}
        app.dependency_overrides[get_admin_user] = lambda: user

        try:
            with (
                patch("kiara.backend_selector.get_next_backend", return_value=None),
                patch("kiara.web.api_datasource_features.load_config", return_value=_base_config()),
                TestClient(app) as client,
            ):
                resp = client.get("/api/admin/vlm-available")

            assert resp.status_code == 200
            assert resp.json()["available"] is False
        finally:
            app.dependency_overrides.pop(get_admin_user, None)


# ===================================================================
# D) Cloud-Backend in Config-Bridge
# ===================================================================


class TestCloudBackendConsumer:
    def test_cloud_backend_returns_all_fields(self):
        """get_cloud_backend() liefert provider, api_key, model, base_url."""
        from kiara.config import get_cloud_backend

        db_backend = {
            "id": 1,
            "category": "cloud",
            "provider": "openai",
            "url": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "api_key": "sk-openai-key",
        }
        config = _base_config()
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            result = get_cloud_backend(config)
        assert result == {
            "provider": "openai",
            "api_key": "sk-openai-key",
            "model": "gpt-4",
            "base_url": "https://api.openai.com/v1",
        }

    def test_cloud_none_when_db_empty_and_yaml_disabled(self):
        """None wenn weder DB noch YAML Cloud-Backend hat."""
        from kiara.config import get_cloud_backend

        config = _base_config()  # cloud.enabled = False
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            result = get_cloud_backend(config)
        assert result is None
