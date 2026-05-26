"""API-Tests fuer VLM-Integration (Phase 10).

Testet /api/admin/vlm-available Endpoint, Modellname-Matching (G1-Fix)
und Datasource CRUD mit ds_vlm_enabled.
Nutzt FastAPI TestClient mit PostgreSQL-DB und gemockter Auth.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kiara.web.api_datasource_features as _vlm_features
from kiara.shared.db import get_connection
from kiara.web.api_datasource import router
from kiara.web.auth import get_admin_user
from tests.admin_confirm import confirmed_request


@pytest.fixture(autouse=True)
def _reset_vlm_cache():
    """VLM-Cache vor jedem Test zuruecksetzen, damit keine Cache-Leaks auftreten."""
    _vlm_features._vlm_cache_result = None
    _vlm_features._vlm_cache_ts = 0.0
    yield
    _vlm_features._vlm_cache_result = None
    _vlm_features._vlm_cache_ts = 0.0


# -- DB-Schema ----------------------------------------------------------------

_FULL_DS_SCHEMA = """
    CREATE TABLE datasources (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'local',
        path TEXT DEFAULT '',
        is_active BOOLEAN DEFAULT true,
        created_at TEXT DEFAULT (NOW()),
        updated_at TEXT DEFAULT (NOW()),
        webdav_url TEXT DEFAULT '',
        webdav_username TEXT DEFAULT '',
        webdav_password TEXT DEFAULT '',
        webdav_base_path TEXT DEFAULT '/',
        smb_server TEXT DEFAULT '',
        smb_share TEXT DEFAULT '',
        smb_username TEXT DEFAULT '',
        smb_password TEXT DEFAULT '',
        smb_domain TEXT DEFAULT '',
        smb_mount_point TEXT DEFAULT '',
        imap_server TEXT DEFAULT '',
        imap_port INTEGER DEFAULT 993,
        imap_ssl INTEGER DEFAULT 1,
        imap_username TEXT DEFAULT '',
        imap_password TEXT DEFAULT '',
        imap_folders TEXT DEFAULT 'INBOX',
        imap_since_days INTEGER DEFAULT 90,
        imap_download_attachments INTEGER DEFAULT 1,
        imap_max_attachment_mb INTEGER DEFAULT 25,
        sql_dsn TEXT DEFAULT '',
        sql_driver TEXT DEFAULT 'auto',
        sql_query TEXT DEFAULT '',
        sql_text_columns TEXT DEFAULT '',
        sql_id_column TEXT DEFAULT 'id',
        sql_timestamp_column TEXT DEFAULT '',
        sql_text_template TEXT DEFAULT '',
        connector_config TEXT DEFAULT '{}',
        ds_chunk_strategy TEXT DEFAULT 'auto',
        ds_chunk_size INTEGER DEFAULT NULL,
        ds_chunk_overlap INTEGER DEFAULT NULL,
        ds_file_types TEXT DEFAULT NULL,
        ds_exclude_patterns TEXT DEFAULT NULL,
        ds_ddu_enabled BOOLEAN DEFAULT false,
        ds_ddu_ocr_images INTEGER DEFAULT 1,
        ds_vlm_enabled BOOLEAN DEFAULT false,
        ds_accept_all_text INTEGER DEFAULT 0,
        ds_index_attachments INTEGER DEFAULT 1,
        sync_interval TEXT DEFAULT 'global',
        restriction_level TEXT DEFAULT 'anonymize',
        ds_contextual_retrieval INTEGER DEFAULT 0,
        ds_enrichment_prompt_id INTEGER DEFAULT NULL,
        ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false,
        ingestion_status TEXT DEFAULT 'idle',
        last_ingested_at TEXT DEFAULT NULL,
        last_error TEXT DEFAULT '',
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        processed_files INTEGER DEFAULT 0
    );

    CREATE TABLE datasource_groups (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        group_id INTEGER NOT NULL
    );

    CREATE TABLE datasource_users (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL
    );

    CREATE TABLE datasource_embeddings (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        last_ingested_at TEXT DEFAULT NULL,
        UNIQUE(datasource_id, embedding_model)
    );

    CREATE TABLE ingestion_jobs (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        job_type TEXT NOT NULL DEFAULT 'incremental',
        triggered_by TEXT NOT NULL DEFAULT 'manual',
        embedding_model TEXT DEFAULT '',
        requested_action TEXT DEFAULT NULL,
        progress_processed INTEGER DEFAULT 0,
        progress_total INTEGER DEFAULT 0,
        queued_at TEXT DEFAULT (NOW()),
        started_at TEXT DEFAULT NULL,
        finished_at TEXT DEFAULT NULL,
        stats_json TEXT DEFAULT '{}',
        error_message TEXT DEFAULT '',
        resumed_from_job INTEGER DEFAULT NULL,
        FOREIGN KEY (datasource_id) REFERENCES datasources(id) ON DELETE CASCADE
    );

    CREATE TABLE cache_datasources (
        datasource_id INTEGER PRIMARY KEY,
        payload_json TEXT NOT NULL,
        updated_at TEXT DEFAULT (NOW())
    );
"""


def _setup_vlm_db():
    """Erstellt DB-Schema fuer VLM-Tests und gibt Connection zurueck."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS cache_datasources CASCADE")
    conn.execute("DROP TABLE IF EXISTS ingestion_jobs CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_embeddings CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasources CASCADE")
    conn.executescript(_FULL_DS_SCHEMA)
    conn.commit()
    return conn


# -- Fixture ------------------------------------------------------------------


@pytest.fixture
def client():
    """API TestClient mit PostgreSQL-DB und gemockten Abhaengigkeiten."""
    setup_conn = _setup_vlm_db()

    from kiara.admin.registry import admin_registry, load_admin_functions
    from kiara.admin.router import generate_admin_routes

    load_admin_functions()

    app = FastAPI()
    app.include_router(router)
    generate_admin_routes(app, admin_registry)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    _cfg = {
        "datasource": {"path": "/tmp"},
        "ollama": {"url": "http://localhost:11434", "vlm_model": "qwen2.5-vl:7b"},
        "ingestion": {"chunk_size": 1000, "chunk_overlap": 200},
    }
    with (
        patch("kiara.web.api_datasource.encrypt", side_effect=lambda x: x),
        patch("kiara.shared.crypto.encrypt", side_effect=lambda x: x),
        patch("kiara.web.api_datasource.load_config", return_value=_cfg),
        patch("kiara.config.load_config", return_value=_cfg),
        patch("kiara.web.api_datasource_features.load_config", return_value=_cfg),
        patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None),
        patch("kiara.web.ingestion_worker.is_job_paused", return_value=False),
        patch("kiara.retrieval.vectorstore.collection_name_for_model", return_value="kb_test"),
        TestClient(app) as c,
    ):
        yield c

    setup_conn.rollback()
    setup_conn.close()


@pytest.fixture
def client_no_vlm_model():
    """API TestClient OHNE vlm_model in Config."""
    setup_conn = _setup_vlm_db()

    app = FastAPI()
    app.include_router(router)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    _cfg = {
        "datasource": {"path": "/tmp"},
        "ollama": {"url": "http://localhost:11434", "vlm_model": ""},
    }
    with (
        patch("kiara.web.api_datasource.encrypt", side_effect=lambda x: x),
        patch("kiara.web.api_datasource.load_config", return_value=_cfg),
        patch("kiara.web.api_datasource_features.load_config", return_value=_cfg),
        patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None),
        patch("kiara.web.ingestion_worker.is_job_paused", return_value=False),
        patch("kiara.retrieval.vectorstore.collection_name_for_model", return_value="kb_test"),
        TestClient(app) as c,
    ):
        yield c

    setup_conn.rollback()
    setup_conn.close()


@pytest.fixture
def client_no_auth():
    """API TestClient OHNE Auth-Override (anonymer Zugriff)."""
    setup_conn = _setup_vlm_db()

    app = FastAPI()
    app.include_router(router)

    # KEIN dependency override

    _cfg = {
        "datasource": {"path": "/tmp"},
        "ollama": {"url": "http://localhost:11434", "vlm_model": "qwen2.5-vl:7b"},
    }
    with (
        patch("kiara.web.api_datasource.encrypt", side_effect=lambda x: x),
        patch("kiara.web.api_datasource.load_config", return_value=_cfg),
        patch("kiara.web.api_datasource_features.load_config", return_value=_cfg),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    setup_conn.rollback()
    setup_conn.close()


def _create_ds(client, name="Test DS", vlm_enabled=False, ddu_enabled=False, **extra):
    """Hilfsfunktion: Erstellt eine Datasource via POST."""
    payload = {
        "name": name,
        "type": "local",
        "connector_config": {"path": "/tmp/test"},
        "ds_vlm_enabled": vlm_enabled,
        "ds_ddu_enabled": ddu_enabled,
        **extra,
    }
    resp = client.post("/api/admin/datasources", json=payload)
    assert resp.status_code == 200
    return resp.json()["id"]


# -- 3.1 /api/admin/vlm-available Endpoint -----------------------------------


def test_vlm_available_model_configured_and_present(client):
    """config vlm_model='qwen2.5-vl:7b', Ollama-Mock listet das Modell -> available=True."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "qwen2.5-vl:7b"}]}
    mock_ollama.Client.return_value = mock_client

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["model"] == "qwen2.5-vl:7b"


def test_vlm_available_model_not_configured(client_no_vlm_model):
    """Kein VLM-Backend konfiguriert -> available=False, 'Kein VLM-Modell konfiguriert'."""
    with patch("kiara.config.get_vlm_backend", return_value=None):
        resp = client_no_vlm_model.get("/api/admin/vlm-available")

    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "Kein VLM-Modell konfiguriert" in data["message"]


def test_vlm_available_model_not_on_ollama(client):
    """VLM-Backend konfiguriert, Ollama-Mock listet nur andere Modelle -> available=False."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.list.return_value = {
        "models": [{"name": "llama3:8b"}, {"name": "nomic-embed-text:latest"}]
    }
    mock_ollama.Client.return_value = mock_client

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "nicht auf Ollama gefunden" in data["message"]


def test_vlm_available_ollama_unreachable(client):
    """Ollama-Client wirft ConnectionError -> available=False, 'Ollama nicht erreichbar'."""
    mock_ollama = MagicMock()
    mock_ollama.Client.side_effect = ConnectionError("Connection refused")

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "Ollama nicht erreichbar" in data["message"]


def test_vlm_available_requires_auth(client_no_auth):
    """Kein Auth -> 401 oder 403."""
    resp = client_no_auth.get("/api/admin/vlm-available")
    assert resp.status_code in (401, 403)


# -- 3.2 /api/admin/vlm-available — Modellname-Matching (G1-Fix) -------------


def test_vlm_match_exact(client):
    """config='qwen2.5-vl:7b', Ollama hat 'qwen2.5-vl:7b' -> available=True."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "qwen2.5-vl:7b"}]}
    mock_ollama.Client.return_value = mock_client

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.json()["available"] is True


def test_vlm_match_basename_without_tag(client):
    """config='qwen2.5-vl' (kein Tag), Ollama hat 'qwen2.5-vl:latest' -> available=True."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "qwen2.5-vl:latest"}]}
    mock_ollama.Client.return_value = mock_client

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.json()["available"] is True


def test_vlm_no_match_different_tag(client):
    """config='qwen2.5-vl:7b', Ollama hat 'qwen2.5-vl:72b' -> available=False."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "qwen2.5-vl:72b"}]}
    mock_ollama.Client.return_value = mock_client

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.json()["available"] is False


def test_vlm_no_match_substring(client):
    """config='qwen2.5-vl:7b', Ollama hat 'qwen2.5-vl:7b-q4_0' -> available=False."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "qwen2.5-vl:7b-q4_0"}]}
    mock_ollama.Client.return_value = mock_client

    vlm_backend = {"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"}
    with (
        patch("kiara.config.get_vlm_backend", return_value=vlm_backend),
        patch.dict("sys.modules", {"ollama": mock_ollama}),
    ):
        resp = client.get("/api/admin/vlm-available")

    assert resp.json()["available"] is False


# -- 3.3 Datasource CRUD — ds_vlm_enabled ------------------------------------


def test_post_ds_with_vlm_enabled_true(client):
    """POST mit ds_vlm_enabled=true -> DB hat INTEGER 1."""
    ds_id = _create_ds(client, vlm_enabled=True)

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert ds["ds_vlm_enabled"]


def test_post_ds_with_vlm_enabled_false(client):
    """POST mit ds_vlm_enabled=false -> DB hat INTEGER 0."""
    ds_id = _create_ds(client, vlm_enabled=False)

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert not ds["ds_vlm_enabled"]


def test_put_ds_vlm_enabled_toggle(client):
    """PUT ds_vlm_enabled=false auf eine DS mit vlm=true -> DB hat 0."""
    ds_id = _create_ds(client, vlm_enabled=True)

    resp = confirmed_request(
        client,
        "PUT",
        f"/api/admin/datasources/{ds_id}",
        json={
            "name": "Test DS",
            "type": "local",
            "connector_config": {"path": "/tmp/test"},
            "ds_vlm_enabled": False,
        },
    )
    assert resp.status_code == 200

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert not ds["ds_vlm_enabled"]


def test_get_ds_returns_vlm_enabled(client):
    """GET liefert ds_vlm_enabled korrekt zurueck (0 und 1)."""
    _create_ds(client, name="VLM On", vlm_enabled=True)
    _create_ds(client, name="VLM Off", vlm_enabled=False)

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2

    by_name = {d["name"]: d for d in data}
    assert by_name["VLM On"]["ds_vlm_enabled"]
    assert not by_name["VLM Off"]["ds_vlm_enabled"]


def test_default_ds_vlm_enabled_is_zero(client):
    """POST ohne explizites ds_vlm_enabled -> DB hat 0."""
    resp = client.post(
        "/api/admin/datasources",
        json={
            "name": "Default DS",
            "type": "local",
            "connector_config": {"path": "/tmp/test"},
        },
    )
    assert resp.status_code == 200
    ds_id = resp.json()["id"]

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert not ds["ds_vlm_enabled"]


def test_post_ds_with_both_ddu_and_vlm(client):
    """POST mit ds_ddu_enabled=true UND ds_vlm_enabled=true -> beide in DB korrekt."""
    ds_id = _create_ds(client, ddu_enabled=True, vlm_enabled=True)

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert ds["ds_ddu_enabled"]
    assert ds["ds_vlm_enabled"]
