"""API-Tests fuer DDU-Integration (Phase 9).

Testet /api/admin/ddu-available Endpoint und Datasource CRUD mit ds_ddu_enabled.
Nutzt FastAPI TestClient mit PostgreSQL-DB und gemockter Auth.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.api_datasource import router
from kiara.web.auth import get_admin_user
from tests.admin_confirm import confirmed_request

# ── Helpers ──────────────────────────────────────────────────────────────────

_LOCAL_BACKEND = {
    "url": "",
    "is_local": True,
    "model": "docling",
    "table_mode": "accurate",
    "do_ocr": True,
    "ocr_lang": ["de", "en"],
    "timeout": 120.0,
    "concurrency_limit": None,
}
_REMOTE_BACKEND = {
    "url": "http://ddu:5001",
    "is_local": False,
    "model": "docling",
    "table_mode": "accurate",
    "do_ocr": True,
    "ocr_lang": ["de", "en"],
    "timeout": 120.0,
    "concurrency_limit": 2,
}
_REMOTE_BACKEND_2 = {
    **_REMOTE_BACKEND,
    "url": "http://ddu2:5001",
}


# ── DB-Schema ────────────────────────────────────────────────────────────────

_FULL_DS_SCHEMA = """
    DROP TABLE IF EXISTS cache_jobs CASCADE;
    DROP TABLE IF EXISTS cache_datasources CASCADE;
    DROP TABLE IF EXISTS ingestion_jobs CASCADE;
    DROP TABLE IF EXISTS datasource_embeddings CASCADE;
    DROP TABLE IF EXISTS datasource_users CASCADE;
    DROP TABLE IF EXISTS datasource_groups CASCADE;
    DROP TABLE IF EXISTS datasources CASCADE;

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
        ds_index_attachments INTEGER DEFAULT 1,
        ds_accept_all_text INTEGER DEFAULT 0,
        sync_interval TEXT DEFAULT 'global',
        ingestion_status TEXT DEFAULT 'idle',
        last_ingested_at TEXT DEFAULT NULL,
        last_error TEXT DEFAULT '',
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        processed_files INTEGER DEFAULT 0,
        restriction_level TEXT DEFAULT 'anonymize' NOT NULL,
        ds_contextual_retrieval INTEGER DEFAULT 0,
        ds_enrichment_prompt_id INTEGER DEFAULT NULL,
        ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false,
        progress_total INTEGER DEFAULT 0,
        next_sync_at TEXT DEFAULT NULL
    );

    CREATE TABLE datasource_groups (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        group_id INTEGER NOT NULL
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

    CREATE TABLE datasource_users (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL
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
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (NOW())
    );

    CREATE TABLE cache_jobs (
        job_id        INTEGER PRIMARY KEY,
        datasource_id INTEGER DEFAULT 0,
        status        TEXT DEFAULT '',
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (NOW())
    );
    CREATE INDEX IF NOT EXISTS idx_cache_jobs_status ON cache_jobs(status);
"""


# ── Fixture ──────────────────────────────────────────────────────────────────


def _ensure_schema():
    """Stellt sicher, dass alle benoetigten Tabellen und Spalten existieren.

    Erstellt das vollstaendige Schema immer neu, um Schema-Konflikte zu vermeiden.
    """
    conn = get_connection()
    conn.executescript(_FULL_DS_SCHEMA)
    conn.commit()
    conn.close()


@pytest.fixture
def client():
    """API TestClient mit PostgreSQL-DB und gemockten Abhaengigkeiten."""
    _ensure_schema()

    from kiara.admin.registry import admin_registry, load_admin_functions
    from kiara.admin.router import generate_admin_routes

    load_admin_functions()

    app = FastAPI()
    app.include_router(router)
    generate_admin_routes(app, admin_registry)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with (
        patch("kiara.web.api_datasource.encrypt", side_effect=lambda x: x, create=True),
        patch("kiara.shared.crypto.encrypt", side_effect=lambda x: x),
        patch(
            "kiara.web.api_datasource.load_config",
            return_value={"datasource": {"path": "/tmp"}},
            create=True,
        ),
        patch(
            "kiara.config.load_config",
            return_value={
                "datasource": {"path": "/tmp"},
                "ingestion": {"chunk_size": 1000, "chunk_overlap": 200},
            },
        ),
        patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None),
        patch("kiara.web.ingestion_worker.is_job_paused", return_value=False),
        patch(
            "kiara.retrieval.vectorstore.collection_name_for_model",
            return_value="kb_test",
            create=True,
        ),
        patch("kiara.connectors.registry.ensure_loaded"),
        patch("kiara.connectors.registry.get") as mock_get_connector,
    ):
        mock_cls = MagicMock()
        mock_cls.secret_fields.return_value = []
        mock_get_connector.return_value = mock_cls
        with TestClient(app) as c:
            yield c

    # Cleanup: Testdaten entfernen (Tabellen bleiben bestehen fuer andere Tests)
    cleanup_conn = get_connection()
    try:
        cleanup_conn.execute("DELETE FROM cache_jobs")
        cleanup_conn.execute("DELETE FROM cache_datasources")
        cleanup_conn.execute("DELETE FROM ingestion_jobs")
        cleanup_conn.execute("DELETE FROM datasource_embeddings")
        cleanup_conn.execute("DELETE FROM datasource_users")
        cleanup_conn.execute("DELETE FROM datasource_groups")
        cleanup_conn.execute("DELETE FROM datasources")
        cleanup_conn.commit()
    finally:
        cleanup_conn.close()


@pytest.fixture
def client_no_auth():
    """API TestClient OHNE Auth-Override (anonymer Zugriff)."""
    _ensure_schema()

    app = FastAPI()
    app.include_router(router)

    # KEIN dependency override → get_admin_user wird normal aufgerufen

    with (
        patch("kiara.web.api_datasource.encrypt", side_effect=lambda x: x, create=True),
        patch(
            "kiara.web.api_datasource.load_config",
            return_value={"datasource": {"path": "/tmp"}},
            create=True,
        ),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    # Cleanup: Testdaten entfernen (Tabellen bleiben bestehen fuer andere Tests)
    cleanup_conn = get_connection()
    try:
        cleanup_conn.execute("DELETE FROM cache_jobs")
        cleanup_conn.execute("DELETE FROM cache_datasources")
        cleanup_conn.execute("DELETE FROM ingestion_jobs")
        cleanup_conn.execute("DELETE FROM datasource_embeddings")
        cleanup_conn.execute("DELETE FROM datasource_users")
        cleanup_conn.execute("DELETE FROM datasource_groups")
        cleanup_conn.execute("DELETE FROM datasources")
        cleanup_conn.commit()
    finally:
        cleanup_conn.close()


def _create_ds(client, name="Test DS", ddu_enabled=False, **extra):
    """Hilfsfunktion: Erstellt eine Datasource via POST."""
    payload = {
        "name": name,
        "type": "local",
        "connector_config": {"path": "/tmp/test"},
        "ds_ddu_enabled": ddu_enabled,
        **extra,
    }
    resp = client.post("/api/admin/datasources", json=payload)
    assert resp.status_code == 200
    return resp.json()["id"]


# ── 4.1 /api/admin/ddu-available Endpoint ────────────────────────────────────


def test_ddu_available_ignores_local_backend(client):
    """Lokales Backend ohne URL wird remote-only ignoriert."""
    from kiara.web.api_datasource_features import invalidate_ddu_cache

    invalidate_ddu_cache()
    with (
        patch("kiara.config.get_ddu_backends", return_value=[_LOCAL_BACKEND]),
        patch("kiara.web.api_datasource_features.load_config", return_value={}),
    ):
        resp = client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "nicht erreichbar" in data["message"]


def test_ddu_available_without_backend(client):
    """Keine Backends → {'available': False, 'message': '...'}."""
    from kiara.web.api_datasource_features import invalidate_ddu_cache

    invalidate_ddu_cache()
    with (
        patch("kiara.config.get_ddu_backends", return_value=[]),
        patch("kiara.web.api_datasource_features.load_config", return_value={}),
    ):
        resp = client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "message" in data


def test_ddu_available_requires_auth(client_no_auth):
    """Endpoint erfordert Admin-Auth (kein anonymer Zugriff)."""
    resp = client_no_auth.get("/api/admin/ddu-available")
    # Ohne Auth sollte 401 oder 403 zurueckgegeben werden
    assert resp.status_code in (401, 403)


# ── 4.2 Datasource CRUD — ds_ddu_enabled ─────────────────────────────────────


def test_post_ds_with_ddu_enabled_true(client):
    """POST /api/admin/datasources mit ds_ddu_enabled=true → DB hat INTEGER 1."""
    ds_id = _create_ds(client, ddu_enabled=True)

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert ds["ds_ddu_enabled"]


def test_put_ds_ddu_enabled_false(client):
    """PUT /api/admin/datasources/{id} mit ds_ddu_enabled=false → DB hat INTEGER 0."""
    ds_id = _create_ds(client, ddu_enabled=True)

    resp = confirmed_request(
        client,
        "PUT",
        f"/api/admin/datasources/{ds_id}",
        json={
            "name": "Test DS",
            "type": "local",
            "connector_config": {"path": "/tmp/test"},
            "ds_ddu_enabled": False,
        },
    )
    assert resp.status_code == 200

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = next(d for d in data if d["id"] == ds_id)
    assert not ds["ds_ddu_enabled"]


def test_get_ds_returns_ddu_enabled(client):
    """GET /api/admin/datasources liefert ds_ddu_enabled korrekt zurueck."""
    _create_ds(client, name="DDU On", ddu_enabled=True)
    _create_ds(client, name="DDU Off", ddu_enabled=False)

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2

    by_name = {d["name"]: d for d in data}
    assert by_name["DDU On"]["ds_ddu_enabled"]
    assert not by_name["DDU Off"]["ds_ddu_enabled"]


def test_default_ds_ddu_enabled_is_zero(client):
    """Default-Wert: neue Datasource ohne ds_ddu_enabled → DB hat 0."""
    # Erstelle DS ohne explizites ds_ddu_enabled
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
    assert not ds["ds_ddu_enabled"]


# ── 4.1b /api/admin/ddu-available — Remote + Cache ──────────────────────────


@pytest.fixture
def features_client():
    """TestClient mit Feature-Router, isoliert vom Haupt-Router."""
    from kiara.web.api_datasource_features import invalidate_ddu_cache
    from kiara.web.api_datasource_features import router as feat_router

    invalidate_ddu_cache()

    app = FastAPI()
    app.include_router(feat_router)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with TestClient(app) as c:
        yield c

    invalidate_ddu_cache()


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote",
    new_callable=AsyncMock,
    return_value={"available": True, "mode": "remote", "url": "http://ddu:5001"},
)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_remote_success(mock_cfg, mock_be, mock_remote, features_client):
    """Remote erreichbar → {'available': True, 'mode': 'remote', 'url': ...}."""
    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["mode"] == "remote"
    assert data["url"] == "http://ddu:5001"


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote", new_callable=AsyncMock, return_value=None
)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND, _LOCAL_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_remote_unreachable_ignores_local_backend(
    mock_cfg, mock_be, mock_remote, features_client
):
    """Remote down + lokales Backend → unavailable, kein lokaler Fallback."""
    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "nicht erreichbar" in data["message"]


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote", new_callable=AsyncMock, return_value=None
)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_remote_unreachable_no_local(mock_cfg, mock_be, mock_remote, features_client):
    """Remote down + kein lokales Backend → {'available': False, 'message': '...'}."""
    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "nicht erreichbar" in data["message"]


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote",
    new_callable=AsyncMock,
    return_value={"available": True, "mode": "remote", "url": "http://ddu:5001"},
)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_cache_hit(mock_cfg, mock_be, mock_remote, features_client):
    """Zweiter Aufruf innerhalb TTL → _check_ddu_remote nur 1x aufgerufen."""
    features_client.get("/api/admin/ddu-available")
    features_client.get("/api/admin/ddu-available")

    assert mock_remote.call_count == 1


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote",
    new_callable=AsyncMock,
    return_value={"available": True, "mode": "remote", "url": "http://ddu:5001"},
)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_cache_invalidation(mock_cfg, mock_be, mock_remote, features_client):
    """invalidate_ddu_cache() → zweiter Check erzwungen."""
    from kiara.web.api_datasource_features import invalidate_ddu_cache

    features_client.get("/api/admin/ddu-available")
    assert mock_remote.call_count == 1

    invalidate_ddu_cache()
    features_client.get("/api/admin/ddu-available")
    assert mock_remote.call_count == 2


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote", new_callable=AsyncMock, return_value=None
)
@patch("kiara.config.get_ddu_backends", return_value=[])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_no_backend_no_local(mock_cfg, mock_be, mock_remote, features_client):
    """Kein Backend → {'available': False, 'message': 'Kein aktives DDU-Backend...'}."""
    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "Kein aktives DDU-Backend" in data["message"]


@patch("kiara.web.api_datasource_features._check_ddu_remote", new_callable=AsyncMock)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND, _REMOTE_BACKEND_2])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_failover_flag(mock_cfg, mock_be, mock_remote, features_client):
    """Erstes Remote down, zweites erreichbar → Response enthaelt 'failover': True."""
    mock_remote.side_effect = [
        None,
        {"available": True, "mode": "remote", "url": "http://ddu2:5001"},
    ]

    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["mode"] == "remote"
    assert data["url"] == "http://ddu2:5001"
    assert data.get("failover") is True


@patch(
    "kiara.web.api_datasource_features._check_ddu_remote", new_callable=AsyncMock, return_value=None
)
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND, _LOCAL_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_local_backend_no_failover_flag(
    mock_cfg, mock_be, mock_remote, features_client
):
    """Remote down + lokales Backend → unavailable, kein failover-Flag."""
    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "failover" not in data or data.get("failover") is not True


# ── 4.1c /api/admin/ddu-available — IP-Pinning + Probe-URL ───────────────────


def _patch_dns_for_test(monkeypatch, host: str, ips: list[str]):
    import socket as _sock

    def fake(h, port, *args, **kwargs):
        if h == host:
            infos = []
            for ip in ips:
                fam = _sock.AF_INET6 if ":" in ip else _sock.AF_INET
                sa = (ip, port, 0, 0) if fam == _sock.AF_INET6 else (ip, port)
                infos.append((fam, _sock.SOCK_STREAM, _sock.IPPROTO_TCP, "", sa))
            return infos
        raise _sock.gaierror(f"unknown host: {h}")

    monkeypatch.setattr(_sock, "getaddrinfo", fake)


def test_ddu_probe_url_blocks_loopback(features_client):
    """?url=http://127.0.0.1:5001 → 200, available=false (Hard-Block)."""
    resp = features_client.get("/api/admin/ddu-available", params={"url": "http://127.0.0.1:5001"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


@pytest.mark.parametrize(
    "url",
    [
        "http://example.de:bad/",
        "http://user:pw@example.de:bad/",
        "http://[::1/",
        "http://[::1]:bad/",
        "not a url",
    ],
)
def test_ddu_probe_url_malformed_returns_200(features_client, url):
    """Kaputte Probe-URLs muessen fail-closed mit 200 + available=false enden,
    nicht als 500 nach aussen schlagen.
    """
    resp = features_client.get("/api/admin/ddu-available", params={"url": url})
    assert resp.status_code == 200, (
        f"URL {url!r} hat 500 erzeugt — Pinning haette PinningError werfen sollen"
    )
    data = resp.json()
    assert data["available"] is False


def test_ddu_probe_malformed_ipv6_with_credentials_no_leak(features_client, caplog):
    """Malformed IPv6 + Credentials: kein 500, keine Credentials im Log."""
    caplog.set_level("DEBUG")
    resp = features_client.get(
        "/api/admin/ddu-available",
        params={"url": "http://user:supersecret@[::1/"},
    )
    assert resp.status_code == 200
    assert resp.json()["available"] is False
    # In den Pinning- und DDU-Loggern darf das Secret NICHT auftauchen
    relevant = [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith(("kiara.shared.http_pinning", "kiara.web.api_datasource_features"))
    ]
    log_text = " ".join(relevant)
    assert "supersecret" not in log_text
    assert "user:supersecret" not in log_text


def test_ddu_probe_url_blocks_ftp_scheme(features_client):
    """?url=ftp://example.de → 200, available=false (Scheme)."""
    resp = features_client.get("/api/admin/ddu-available", params={"url": "ftp://example.de"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


def test_ddu_probe_url_redacts_credentials(features_client, caplog):
    """?url=http://user:pw@example.de → 200, Pinning- und DDU-Log redacted."""
    caplog.set_level("DEBUG")
    resp = features_client.get(
        "/api/admin/ddu-available",
        params={"url": "http://user:secret@example.de"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    # Nur die relevanten Logger (Pinning + DDU); httpx-Access-Log enthaelt
    # die Original-URL als URL-encoded Query, das ist nicht unser Log.
    relevant = [
        r.getMessage()
        for r in caplog.records
        if r.name.startswith(("kiara.shared.http_pinning", "kiara.web.api_datasource_features"))
    ]
    log_text = " ".join(relevant)
    assert "secret" not in log_text
    assert "user:secret" not in log_text
    assert "***@" in log_text


@patch(
    "kiara.config.get_ddu_backends",
    return_value=[
        {
            "url": "http://gpu-server.local:5001",
            "is_local": False,
            "model": "docling",
            "table_mode": "accurate",
            "do_ocr": True,
            "ocr_lang": ["de", "en"],
            "timeout": 120.0,
            "concurrency_limit": None,
        }
    ],
)
@patch(
    "kiara.web.api_datasource_features.load_config",
    return_value={"ddu": {"internal_cidrs": ["10.0.0.0/8"]}},
)
def test_ddu_internal_backend_reachable_with_allowlist(
    mock_cfg, mock_be, features_client, monkeypatch
):
    """Internes Backend mit ddu.internal_cidrs erreichbar."""
    _patch_dns_for_test(monkeypatch, "gpu-server.local", ["gpu-server.local"])

    captured = []

    def fake_handle(self, request):
        captured.append(request)
        return httpx.Response(200, request=request)

    async def fake_async(self, request):
        captured.append(request)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_async)

    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["mode"] == "remote"


@patch(
    "kiara.config.get_ddu_backends",
    return_value=[
        {
            "url": "http://gpu-server.local:5001",
            "is_local": False,
            "model": "docling",
            "table_mode": "accurate",
            "do_ocr": True,
            "ocr_lang": ["de", "en"],
            "timeout": 120.0,
            "concurrency_limit": None,
        }
    ],
)
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_internal_backend_blocked_without_allowlist(
    mock_cfg, mock_be, features_client, monkeypatch
):
    """Internes Backend ohne ddu.internal_cidrs nicht erreichbar."""
    _patch_dns_for_test(monkeypatch, "gpu-server.local", ["gpu-server.local"])

    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


@patch("kiara.config.get_ddu_backends", return_value=[_LOCAL_BACKEND, _REMOTE_BACKEND])
@patch(
    "kiara.web.api_datasource_features.load_config",
    return_value={"ddu": {"internal_cidrs": ["127.0.0.0/8"]}},
)
def test_ddu_invalid_cidr_validated_before_backend_loop(mock_cfg, mock_be, features_client, caplog):
    """validate_allowed_private_cidrs laeuft VOR der Backend-Schleife.

    Backend ohne URL zuerst + ungueltige CIDR → trotzdem fail-closed.
    """
    caplog.set_level("WARNING", logger="kiara.web.api_datasource_features")

    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    assert "ungueltige ddu.internal_cidrs" in data["message"]
    # Genau ein WARNING im DDU-Logger
    warns = [
        r
        for r in caplog.records
        if r.name == "kiara.web.api_datasource_features" and r.levelname == "WARNING"
    ]
    assert len(warns) == 1


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={"ddu": "broken"})
def test_ddu_non_dict_ddu_section_is_pinning_config_error(
    mock_cfg, mock_be, features_client, caplog
):
    """ddu: 'broken' (kein Dict) → PinningConfigError → fail-closed, 1 WARNING."""
    caplog.set_level("WARNING", logger="kiara.web.api_datasource_features")

    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    warns = [
        r
        for r in caplog.records
        if r.name == "kiara.web.api_datasource_features" and r.levelname == "WARNING"
    ]
    assert len(warns) == 1


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch(
    "kiara.web.api_datasource_features.load_config",
    return_value={"ddu": {"internal_cidrs": ["abc"]}},
)
def test_ddu_unparseable_cidr_fail_closed(mock_cfg, mock_be, features_client, caplog):
    """ddu.internal_cidrs=['abc'] → PinningConfigError → fail-closed."""
    caplog.set_level("WARNING")

    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False
    feature_warns = [
        r
        for r in caplog.records
        if r.name == "kiara.web.api_datasource_features" and r.levelname == "WARNING"
    ]
    assert len(feature_warns) == 1
    pin_warns = [
        r
        for r in caplog.records
        if r.name == "kiara.shared.http_pinning" and r.levelname == "WARNING"
    ]
    # Helper darf PinningConfigError NICHT loggen.
    assert len(pin_warns) == 0


def test_ddu_pinning_config_error_not_cached(features_client):
    """Cache-Pollution: ungueltig → korrigiert → kein Lesen aus Fehler-Cache."""
    # 1. Aufruf: ungueltige Config
    with (
        patch(
            "kiara.web.api_datasource_features.load_config",
            return_value={"ddu": {"internal_cidrs": ["127.0.0.0/8"]}},
        ),
        patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND]),
    ):
        resp1 = features_client.get("/api/admin/ddu-available")
    assert resp1.json()["available"] is False
    # 2. Aufruf: korrigierte Config + Remote-Mock
    with (
        patch(
            "kiara.web.api_datasource_features._check_ddu_remote",
            new_callable=AsyncMock,
            return_value={"available": True, "mode": "remote", "url": "http://ddu:5001"},
        ),
        patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND]),
        patch(
            "kiara.web.api_datasource_features.load_config",
            return_value={"ddu": {"internal_cidrs": ["10.0.0.0/8"]}},
        ),
    ):
        resp2 = features_client.get("/api/admin/ddu-available")
    # Wenn das Fehler-Resultat gecached gewesen waere, kaeme jetzt false zurueck.
    assert resp2.json()["available"] is True


@pytest.mark.asyncio
async def test_check_ddu_remote_does_not_swallow_value_error(monkeypatch):
    """_check_ddu_remote schluckt echte ValueError nicht.

    Programmierfehler im Helper (z. B. unerlaubte kwargs) muessen
    propagieren.
    """
    from kiara.web.api_datasource_features import _check_ddu_remote

    async def boom(*args, **kwargs):
        raise ValueError("programmierfehler")

    monkeypatch.setattr("kiara.web.api_datasource_features.pinned_request_async", boom)

    with pytest.raises(ValueError):
        await _check_ddu_remote({"url": "http://ddu:5001"}, allowed_private_cidrs=None)


@pytest.mark.asyncio
async def test_check_ddu_remote_call_args_probe_path(monkeypatch):
    """Probe-Pfad ruft _check_ddu_remote mit allowed_private_cidrs=None."""
    seen_kwargs: list = []

    async def fake_pinned(method, url, **kwargs):
        seen_kwargs.append(kwargs)
        return httpx.Response(200, request=httpx.Request(method, url))

    # _check_ddu_remote selbst nutzen, nicht den Endpoint
    monkeypatch.setattr("kiara.web.api_datasource_features.pinned_request_async", fake_pinned)
    from kiara.web.api_datasource_features import _check_ddu_remote

    await _check_ddu_remote({"url": "http://ddu:5001"}, allowed_private_cidrs=None)
    assert seen_kwargs[0].get("allowed_private_cidrs") is None


@pytest.mark.asyncio
async def test_check_ddu_remote_call_args_with_cidrs(monkeypatch):
    """Backend-Iteration ruft _check_ddu_remote mit Config-CIDRs auf."""
    seen_kwargs: list = []

    async def fake_pinned(method, url, **kwargs):
        seen_kwargs.append(kwargs)
        return httpx.Response(200, request=httpx.Request(method, url))

    monkeypatch.setattr("kiara.web.api_datasource_features.pinned_request_async", fake_pinned)
    from kiara.web.api_datasource_features import _check_ddu_remote

    cidrs = ["10.0.0.0/8"]
    await _check_ddu_remote({"url": "http://ddu:5001"}, allowed_private_cidrs=cidrs)
    assert seen_kwargs[0].get("allowed_private_cidrs") == cidrs
