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



# ── Fixture ──────────────────────────────────────────────────────────────────


def _ensure_schema():
    """Echtes KIara-Schema (schema.sql + Migrationen), Testdaten geleert.

    Frueher legte der Test eine eigene Kopie der Datasource-Tabellen per
    DROP ... CASCADE an; ihr fehlten neuere Spalten (z. B. CAD-Zielschalter),
    und das CASCADE beschaedigte Fremdschluessel anderer Tests.
    """
    from db_helpers import reset_test_data

    reset_test_data()


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
    assert "DDU-Backend" in data["message"]


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


# ── 4.1b /api/admin/ddu-available — Konfigurationspruefung + Probe ───────────
# Hausaenderung 30.10.2026: ohne ``url`` keine Netzwerkprobe, kein IP-Pinning.


@pytest.fixture
def features_client():
    """TestClient mit Feature-Router, isoliert vom Haupt-Router."""
    from kiara.web.api_datasource_features import router as feat_router

    app = FastAPI()
    app.include_router(feat_router)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with TestClient(app) as c:
        yield c


@patch("kiara.ingestion.ddu_client.check_ddu_health")
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_configured_without_network_probe(
    mock_cfg, mock_be, mock_health, features_client
):
    """Aktives Remote-Backend → available, aber keine Netzwerkanfrage (GPU-Weckeffekt)."""
    resp = features_client.get("/api/admin/ddu-available")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"available": True, "mode": "remote", "url": "http://ddu:5001"}
    mock_health.assert_not_called()


@patch("kiara.config.get_ddu_backends", return_value=[_LOCAL_BACKEND])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_local_backend_only_is_unavailable(mock_cfg, mock_be, features_client):
    """Nur lokales Backend ohne URL → unavailable (Docling ist remote-only)."""
    data = features_client.get("/api/admin/ddu-available").json()
    assert data["available"] is False
    assert "Kein aktives DDU-Backend" in data["message"]


@patch("kiara.config.get_ddu_backends", return_value=[])
@patch("kiara.web.api_datasource_features.load_config", return_value={})
def test_ddu_available_no_backend(mock_cfg, mock_be, features_client):
    data = features_client.get("/api/admin/ddu-available").json()
    assert data["available"] is False
    assert "Kein aktives DDU-Backend" in data["message"]


@pytest.mark.parametrize("url", ["http://10.20.30.40:5001", "https://ddu.example:5001/"])
def test_ddu_probe_url_accepts_private_and_public_targets(features_client, url):
    """Probe nutzt den Ingest-Check; auch private Adressen sind erlaubt."""
    with patch("kiara.ingestion.ddu_client.check_ddu_health", return_value=(True, "")) as m:
        data = features_client.get("/api/admin/ddu-available", params={"url": url}).json()
    assert data == {"available": True, "mode": "remote", "url": url}
    m.assert_called_once_with([{"url": url}])


def test_ddu_probe_url_failure_names_the_reason(features_client):
    """Fehlschlag der Probe → Ursache aus dem Health-Check steht in der Meldung."""
    err = "http://10.20.30.40:5001/health: Verbindung verweigert"
    with patch("kiara.ingestion.ddu_client.check_ddu_health", return_value=(False, err)):
        data = features_client.get(
            "/api/admin/ddu-available", params={"url": "http://10.20.30.40:5001"}
        ).json()
    assert data["available"] is False
    assert "Verbindung verweigert" in data["message"]


@pytest.mark.parametrize("url", ["ftp://ddu:5001", "file:///etc/passwd", "ddu:5001"])
def test_ddu_probe_url_rejects_non_http_scheme(features_client, url):
    with patch("kiara.ingestion.ddu_client.check_ddu_health") as m:
        data = features_client.get("/api/admin/ddu-available", params={"url": url}).json()
    assert data["available"] is False
    assert "http://" in data["message"]
    m.assert_not_called()
