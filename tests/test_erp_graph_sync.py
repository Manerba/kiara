"""Tests fuer ERPGraphSync-Bulk-Ablauf."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from kiara.graph.erp_sync import (
    ERPGraphSync,
    ERPSyncAlreadyRunningError,
    ERPSyncConfigurationError,
)
from kiara.tools.erp_backend import ERPUpstreamError


class _FakeGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.clear_called = False

    def clear(self, source=None, *, conn=None):
        self.clear_called = True
        self.nodes.clear()
        self.edges.clear()
        return 0

    def add_node(
        self,
        node_id,
        entity_type,
        label,
        properties=None,
        source="extraction",
        conn=None,
        chunk_id=None,
    ):
        self.nodes.add(node_id)

    def add_edge(
        self,
        source_id,
        target_id,
        relation,
        properties=None,
        chunk_id=None,
        source="extraction",
        conn=None,
    ):
        self.edges.append((source_id, target_id, relation))

    def node_exists(self, node_id, *, conn=None):
        return node_id in self.nodes


class _FakeBackend:
    async def list_all_customers(self):
        return [{"customer_id": "K-1", "name": "Müller GmbH"}]

    async def list_all_articles(self):
        return [{"article_id": "ART-1", "description": "Teil", "supplier": "Müller Lieferant"}]

    async def list_all_orders(self):
        return [
            {
                "order_id": "A-1",
                "customer_id": "K-1",
                "positions": [{"article_id": "ART-1"}],
            }
        ]

    async def list_all_production_orders(self):
        return [{"production_order_id": "FA-1", "order_reference": "A-1"}]


class _FailingBackend(_FakeBackend):
    async def list_all_customers(self):
        raise ERPUpstreamError("boom")


@contextmanager
def _fake_db_transaction():
    yield object()


@pytest.mark.asyncio
async def test_sync_all_uses_bulk_and_normalizes_ids():
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_FakeBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
    ):
        result = await sync.sync_all()

    assert result["nodes_created"] == 5
    assert "customer:k-1" in graph.nodes
    assert "supplier:muller-lieferant" in graph.nodes
    assert ("order:a-1", "customer:k-1", "bestellt_bei") in graph.edges


@pytest.mark.asyncio
async def test_bulk_failure_happens_before_clear():
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_FailingBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
        pytest.raises(ERPUpstreamError),
    ):
        await sync.sync_all()

    assert graph.clear_called is False


@pytest.mark.asyncio
async def test_partial_assignments_fail_before_fetch():
    graph = _FakeGraph()
    config = {
        "tools": {
            "erp": {
                "backend": "mock",
                "tool_assignments": {"erp_order_status": "conn-1"},
            }
        }
    }
    sync = ERPGraphSync(config, graph)

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
        pytest.raises(ERPSyncConfigurationError),
    ):
        await sync.sync_all()

    assert graph.clear_called is False


# ---------------------------------------------------------------------------
# Issue #1043: V28-Testplan §5 — Lock, Idempotenz, Kollisionen, Optional-Skips
# ---------------------------------------------------------------------------


def test_acquire_sync_lock_conflict_raises():
    """#1043 Lock-Konflikt: pg_try_advisory_lock=false → ERPSyncAlreadyRunningError."""
    sync = ERPGraphSync({}, _FakeGraph())
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = {"acquired": False}
    with (
        patch("kiara.shared.db.get_db", return_value=mock_conn),
        pytest.raises(ERPSyncAlreadyRunningError),
    ):
        sync._acquire_sync_lock()
    # Connection wurde auch bei Lock-Konflikt geschlossen (kein Leak).
    # close() kann durch die try/except-Struktur mehrfach aufgerufen werden
    # (erst im Early-Return bei not acquired, dann nochmal im except-Block),
    # beides ist bei psycopg2 safe.
    assert mock_conn.close.called


def test_release_sync_lock_noop_on_none():
    """#1043 Lock-Release: None-Connection ist safe (kein AttributeError)."""
    # Darf nicht werfen
    ERPGraphSync._release_sync_lock(None)


def test_release_sync_lock_closes_connection():
    """#1043 Lock-Release: Unlock + Commit + Close auf derselben Connection."""
    mock_conn = MagicMock()
    ERPGraphSync._release_sync_lock(mock_conn)
    # Unlock wurde ausgeloest
    sql_calls = [c.args[0] for c in mock_conn.execute.call_args_list]
    assert any("pg_advisory_unlock" in s for s in sql_calls)
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_sync_all_idempotent():
    """#1043 Idempotenz: 2x sync_all → gleiche Nodes/Edges, kein Duplikat."""
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_FakeBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
    ):
        first = await sync.sync_all()
        first_nodes = set(graph.nodes)
        first_edges = set(graph.edges)
        second = await sync.sync_all()

    # Zweiter Lauf liefert identisches Ergebnis (gleiche Node-Menge + Edges).
    # _FakeGraph.clear() leert vor jedem Lauf — entscheidend ist, dass der
    # zweite Lauf dasselbe Endergebnis reproduziert.
    assert first["nodes_created"] == second["nodes_created"]
    assert set(graph.nodes) == first_nodes
    assert set(graph.edges) == first_edges


class _UnicodeBackend(_FakeBackend):
    async def list_all_customers(self):
        return [{"customer_id": "Kümmel-1", "name": "Küche GmbH"}]

    async def list_all_orders(self):
        return [
            {
                "order_id": "A-1",
                "customer_id": "Kümmel-1",
                "positions": [{"article_id": "ART-1"}],
            }
        ]


@pytest.mark.asyncio
async def test_sync_all_unicode_customer_id_normalized():
    """#1043 Unicode: Umlaute in customer_id → stabile Normalisierung."""
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_UnicodeBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
    ):
        await sync.sync_all()

    # Normalisierter Prefix + beliebige Darstellung des Body-Teils, wichtig:
    # Order referenziert den gleichen normalisierten customer-Knoten.
    customer_nodes = [n for n in graph.nodes if n.startswith("customer:")]
    assert len(customer_nodes) == 1
    cust_node = customer_nodes[0]
    order_edges = [e for e in graph.edges if e[2] == "bestellt_bei"]
    assert len(order_edges) == 1
    assert order_edges[0][1] == cust_node


class _CollisionBackend(_FakeBackend):
    async def list_all_customers(self):
        # Zwei Kunden, die nach Normalisierung auf denselben node_id abbilden.
        return [
            {"customer_id": "K-1", "name": "GmbH A"},
            {"customer_id": "k-1", "name": "GmbH B"},
        ]


@pytest.mark.asyncio
async def test_normalized_id_collision_blocks_before_clear():
    """#1043 Preflight: ID-Kollision fails-closed VOR clear(source=erp_import)."""
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_CollisionBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
        pytest.raises(ERPSyncConfigurationError, match="Kollision"),
    ):
        await sync.sync_all()

    # clear() darf NICHT gelaufen sein — Graph wird nicht vor Preflight geleert
    assert graph.clear_called is False


class _SupplierCollisionBackend(_FakeBackend):
    async def list_all_articles(self):
        # Zwei Artikel mit unterschiedlichen Supplier-Namen, die nach
        # Normalisierung denselben supplier-Node ergeben (case-insensitive).
        return [
            {"article_id": "ART-1", "description": "Teil A", "supplier": "Muster AG"},
            {"article_id": "ART-2", "description": "Teil B", "supplier": "muster-ag"},
        ]


@pytest.mark.asyncio
async def test_supplier_normalized_collision_blocks_before_clear():
    """#1043 Preflight: Zwei Supplier-Namen kollidieren nach Normalisierung → Fehler vor clear."""
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_SupplierCollisionBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
        pytest.raises(ERPSyncConfigurationError, match="Kollision"),
    ):
        await sync.sync_all()

    assert graph.clear_called is False


class _MissingRefsBackend(_FakeBackend):
    async def list_all_orders(self):
        # Order ohne customer_id + Position ohne article_id
        return [
            {
                "order_id": "A-2",
                "customer_id": None,
                "positions": [{"article_id": None}],
            }
        ]

    async def list_all_production_orders(self):
        # Production-Order ohne order_reference → produziert_fuer-Kante skippen
        return [{"production_order_id": "FA-2", "order_reference": None}]


@pytest.mark.asyncio
async def test_optional_references_skipped_gracefully():
    """#1043 Optionale Referenzen (customer_id/article_id/order_reference) fehlen
    → Sync laeuft durch, relevante Kanten werden uebersprungen, keine Exception."""
    graph = _FakeGraph()
    sync = ERPGraphSync({}, graph, erp_backend=_MissingRefsBackend())

    with (
        patch.object(sync, "_acquire_sync_lock", return_value=object()),
        patch.object(sync, "_release_sync_lock"),
        patch("kiara.shared.db.db_transaction", _fake_db_transaction),
    ):
        await sync.sync_all()

    # order-Knoten wurde erstellt (order_id selbst ist required)
    assert "order:a-2" in graph.nodes
    assert "production_order:fa-2" in graph.nodes
    # Aber keine bestellt_bei/enthaelt/produziert_fuer Kanten
    assert not any(e[2] == "bestellt_bei" for e in graph.edges)
    assert not any(e[2] == "enthaelt" and e[0] == "order:a-2" for e in graph.edges)
    assert not any(
        e[2] == "produziert_fuer" and e[0] == "production_order:fa-2" for e in graph.edges
    )
