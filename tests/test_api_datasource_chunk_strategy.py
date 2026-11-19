"""API-Roundtrip-Tests fuer ds_chunk_strategy (Phase 7 + Phase 11.1 Refactoring).

Testet POST, PUT, GET der Datasource-API bezueglich Chunk-Strategie:
  - POST speichert ds_chunk_strategy korrekt
  - PUT aendert ds_chunk_strategy
  - GET liefert ds_chunk_strategy zurueck
  - Ungueltiger Wert wird trotzdem gespeichert (keine Server-Validierung)

Phase 11.1: connector_config JSON-Spalte statt Einzelspalten.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.web.api_datasource import router
from kiara.web.auth import get_admin_user
from tests.admin_confirm import confirmed_request

# ---------------------------------------------------------------------------
# DB-Schema (Phase 11.1: connector_config statt 30+ Einzelspalten)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """API TestClient mit PostgreSQL-DB und gemockten Abhaengigkeiten."""
    # Echtes KIara-Schema statt eigener Tabellenkopie (fehlende Spalten).
    from db_helpers import reset_test_data

    reset_test_data()

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
        patch("kiara.shared.crypto.encrypt", side_effect=lambda x: x),
        patch("kiara.admin.services.datasource_service.encrypt", side_effect=lambda x: x),
        patch("kiara.shared.crypto.is_encrypted", return_value=False),
        patch(
            "kiara.config.load_config",
            return_value={"datasource": {"path": "/tmp"}, "chunk_size": 1000, "chunk_overlap": 200},
        ),
        patch(
            "kiara.admin.services.datasource_service.load_config",
            return_value={"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}},
        ),
        patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None),
        patch("kiara.web.ingestion_worker.is_job_paused", return_value=False),
    ):
        with TestClient(app) as c:
            yield c

    # Cleanup + Schema wiederherstellen
    from db_helpers import restore_test_schema

    restore_test_schema()


def _create_ds(client, name="Test DS", chunk_strategy="auto", **extra):
    """Hilfsfunktion: Erstellt eine Datasource via POST."""
    payload = {
        "name": name,
        "type": "local",
        "connector_config": {"path": "/tmp/test"},
        "ds_chunk_strategy": chunk_strategy,
        **extra,
    }
    resp = client.post("/api/admin/datasources", json=payload)
    assert resp.status_code == 200, f"POST failed: {resp.json()}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_stores_chunk_strategy(client):
    """POST /api/admin/datasources mit ds_chunk_strategy='prose' → DB speichert 'prose'."""
    ds_id = _create_ds(client, chunk_strategy="prose")

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == ds_id
    assert data[0]["ds_chunk_strategy"] == "prose"


def test_put_updates_chunk_strategy(client):
    """PUT /api/admin/datasources/{id} aendert ds_chunk_strategy von 'auto' auf 'code'."""
    ds_id = _create_ds(client, chunk_strategy="auto")

    resp = confirmed_request(
        client,
        "PUT",
        f"/api/admin/datasources/{ds_id}",
        json={
            "name": "Test DS",
            "type": "local",
            "connector_config": {"path": "/tmp/test"},
            "ds_chunk_strategy": "code",
        },
    )
    assert resp.status_code == 200

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    assert data[0]["ds_chunk_strategy"] == "code"


def test_get_returns_chunk_strategy(client):
    """GET /api/admin/datasources liefert ds_chunk_strategy korrekt zurueck."""
    _create_ds(client, name="DS Prose", chunk_strategy="prose")
    _create_ds(client, name="DS Tabular", chunk_strategy="tabular")

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2

    strategies = {d["name"]: d["ds_chunk_strategy"] for d in data}
    assert strategies["DS Prose"] == "prose"
    assert strategies["DS Tabular"] == "tabular"


def test_invalid_chunk_strategy_accepted(client):
    """Ungueltiger Wert 'invalid' wird ohne Fehler gespeichert (keine Validierung)."""
    ds_id = _create_ds(client, chunk_strategy="invalid")

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    assert data[0]["ds_chunk_strategy"] == "invalid"


def test_post_stores_connector_config(client):
    """POST speichert connector_config als JSON."""
    payload = {
        "name": "SMB Test",
        "type": "smb",
        "connector_config": {
            "server": "192.168.1.10",
            "share": "Daten",
            "username": "admin",
            "password": "pw",
        },
    }
    resp = client.post("/api/admin/datasources", json=payload)
    assert resp.status_code == 200, f"POST failed: {resp.text}"

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = data[0]
    # password should be masked in GET response
    assert ds["connector_config"]["server"] == "192.168.1.10"
    assert ds["connector_config"]["password"] == "********"


def test_connector_types_endpoint(client):
    """GET /api/admin/connector-types liefert alle registrierten Konnektoren."""
    resp = client.get("/api/admin/connector-types")
    assert resp.status_code == 200
    types = resp.json()["data"]
    names = [t["name"] for t in types]
    assert "local" in names
    assert "smb" in names
    assert "imap" in names
    assert "exchange_online" in names
    assert "sql" in names
    assert "webdav" in names
    # Jeder Typ hat ein Schema
    for t in types:
        assert "schema" in t
        assert "supports_browse" in t
        assert isinstance(t["schema"], dict)
