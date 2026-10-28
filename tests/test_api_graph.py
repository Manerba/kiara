"""Tests fuer GraphRAG Admin-API-Endpunkte."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.graph.store import GraphStore
from kiara.web.api_graph import router
from kiara.web.auth import get_admin_user

# --- Fixtures ---


@pytest.fixture
def graph_store():
    """Erstellt einen GraphStore (PostgreSQL, via conftest._init_pg_backend)."""
    store = GraphStore()
    store.clear()
    yield store
    store.clear()


@pytest.fixture
def app(graph_store):
    """Erstellt eine FastAPI-App mit gemockter Auth und echtem GraphStore."""
    test_app = FastAPI()
    test_app.include_router(router)

    # Override auth dependency
    async def mock_admin_user():
        return {"id": 1, "username": "admin", "is_admin": True}

    test_app.dependency_overrides[get_admin_user] = mock_admin_user
    return test_app


@pytest.fixture
def client(app, graph_store):
    """TestClient mit gemocktem _get_graph_store."""
    with patch("kiara.web.api_graph._get_graph_store", return_value=graph_store):
        with TestClient(app) as c:
            yield c


# --- Tests ---


def test_graph_stats_returns_json(client, graph_store):
    """GET /api/admin/graph/stats gibt JSON mit allen erwarteten Feldern zurueck."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH", source="erp_import")
    graph_store.add_node("supplier:alu", "supplier", "Alu-Form GmbH", source="extraction")
    graph_store.add_edge("supplier:alu", "customer:k-1001", "liefert")

    resp = client.get("/api/admin/graph/stats")
    assert resp.status_code == 200

    data = resp.json()
    assert "total_nodes" in data
    assert "total_edges" in data
    assert "nodes_by_type" in data
    assert "edges_by_relation" in data
    assert "sources" in data
    assert data["total_nodes"] == 2
    assert data["total_edges"] == 1
    assert data["nodes_by_type"]["customer"] == 1
    assert data["sources"]["erp_import"] == 1


def test_graph_stats_requires_auth():
    """Ohne Admin-Auth gibt stats 401 zurueck."""
    test_app = FastAPI()
    test_app.include_router(router)
    # Keine dependency_overrides -> echte Auth wird versucht

    with patch("kiara.web.api_graph._get_graph_store"):
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.get("/api/admin/graph/stats")
            assert resp.status_code in (401, 403)


def test_sync_erp_creates_nodes(client, graph_store):
    """POST /api/admin/graph/sync-erp gibt nodes_created > 0 zurueck."""
    mock_result = {"nodes_created": 5, "edges_created": 3, "duration_ms": 10.0}

    # ERPGraphSync wird lazy importiert in sync_erp(), daher an der Quelle patchen
    with patch("kiara.graph.erp_sync.ERPGraphSync") as MockSync:
        instance = MockSync.return_value
        instance.sync_all = AsyncMock(return_value=mock_result)

        resp = client.post("/api/admin/graph/sync-erp")
        assert resp.status_code == 200

        data = resp.json()
        assert data["nodes_created"] == 5
        assert data["edges_created"] == 3
        assert "message" in data


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (
            __import__(
                "kiara.graph.erp_sync", fromlist=["ERPSyncAlreadyRunningError"]
            ).ERPSyncAlreadyRunningError("busy"),
            409,
        ),
        (NotImplementedError("no bulk"), 501),
        (
            __import__("kiara.tools.erp_backend", fromlist=["ERPUpstreamError"]).ERPUpstreamError(
                "upstream"
            ),
            503,
        ),
        (RuntimeError("boom"), 500),
    ],
)
def test_sync_erp_error_mapping(client, graph_store, exc, status):
    with patch("kiara.graph.erp_sync.ERPGraphSync") as MockSync:
        instance = MockSync.return_value
        instance.sync_all = AsyncMock(side_effect=exc)

        resp = client.post("/api/admin/graph/sync-erp")

    assert resp.status_code == status
    assert isinstance(resp.json()["detail"], str)


def test_sync_erp_constructor_error_mapping(client, graph_store):
    from kiara.graph.erp_sync import ERPSyncConfigurationError

    with patch(
        "kiara.graph.erp_sync.ERPGraphSync",
        side_effect=ERPSyncConfigurationError("bad config"),
    ):
        resp = client.post("/api/admin/graph/sync-erp")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "bad config"


def test_sync_erp_assignment_error_mapping(client, graph_store):
    from kiara.tools.erp_connectors.registry import ERPAssignmentBrokenError

    with patch(
        "kiara.graph.erp_sync.ERPGraphSync",
        side_effect=ERPAssignmentBrokenError("broken assignment"),
    ):
        resp = client.post("/api/admin/graph/sync-erp")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "ERP-Graph-Sync ist nicht korrekt konfiguriert."


def test_sync_erp_disabled_by_config(client, graph_store):
    """POST /api/admin/graph/sync-erp gibt 503 zurueck wenn graph.erp_sync_enabled=False."""
    with (
        patch("kiara.web.api_graph.load_config") as mock_config,
        patch("kiara.graph.erp_sync.ERPGraphSync") as MockSync,
    ):
        mock_config.return_value = {"graph": {"erp_sync_enabled": False}}
        resp = client.post("/api/admin/graph/sync-erp")

    assert resp.status_code == 503
    assert "deaktiviert" in resp.json()["detail"].lower()
    MockSync.assert_not_called()


def test_clear_graph_deletes_all(client, graph_store):
    """POST /api/admin/graph/clear loescht alle Knoten."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_edge("n:a", "n:b", "betrifft")

    resp = client.post("/api/admin/graph/clear?confirm=clear_graph")
    assert resp.status_code == 200

    data = resp.json()
    assert "2" in data["message"] or "Knoten" in data["message"]

    stats = graph_store.get_stats()
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0


def test_search_nodes_by_query(client, graph_store):
    """GET /api/admin/graph/search?q=Mueller findet Knoten."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH")
    graph_store.add_node("supplier:schmidt", "supplier", "Schmidt AG")

    resp = client.get("/api/admin/graph/search", params={"q": "Mueller"})
    assert resp.status_code == 200

    data = resp.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["label"] == "Mueller GmbH"


def test_search_nodes_with_type_filter(client, graph_store):
    """GET /api/admin/graph/search?q=Mueller&entity_type=customer filtert nach Typ."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH")
    graph_store.add_node("supplier:mueller", "supplier", "Mueller Zulieferer")

    # Ohne Filter: beide
    resp = client.get("/api/admin/graph/search", params={"q": "Mueller"})
    assert len(resp.json()["nodes"]) == 2

    # Mit Typ-Filter: nur customer
    resp = client.get("/api/admin/graph/search", params={"q": "Mueller", "entity_type": "customer"})
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["entity_type"] == "customer"
