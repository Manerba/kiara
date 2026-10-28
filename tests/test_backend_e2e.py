"""End-to-End Tests: Backend-Lifecycle durch alle Schichten.

Testet den vollstaendigen Fluss von API → DB → Cache → Config-Bridge.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kiara.backend_selector import (
    get_all_active,
    get_next_backend,
    invalidate_cache,
)
from kiara.web.admin_app import app
from tests.admin_confirm import confirmed_request


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_admin():
    user = {"id": 1, "username": "admin", "is_admin": True}
    from kiara.web.auth import get_admin_user

    app.dependency_overrides[get_admin_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture
def e2e_db(tmp_path):
    """DB mit ki_backends + reale Cache-Invalidierung (kein Mock)."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ki_backends (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL CHECK (category IN ('embedding', 'llm', 'cloud', 'vlm')),
            provider TEXT NOT NULL CHECK (provider IN ('ollama', 'anthropic', 'openai')),
            name TEXT NOT NULL,
            url TEXT DEFAULT '',
            base_url TEXT DEFAULT '',
            model TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            priority INTEGER DEFAULT 10,
            trusted BOOLEAN DEFAULT false,
            extra_config TEXT DEFAULT '{}',
            last_test_at TEXT DEFAULT NULL,
            last_test_ok INTEGER DEFAULT NULL,
            last_test_error TEXT DEFAULT '',
            created_at TEXT DEFAULT (NOW()),
            updated_at TEXT DEFAULT (NOW())
        );
    """)
    conn.commit()
    conn.close()

    @contextmanager
    def mock_db_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def mock_db_transaction():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # Patch BEIDE: API-Modul UND backend_selector (fuer echte Cache-Tests)
    with (
        patch("kiara_db.base.db_query", mock_db_query),
        patch("kiara_db.base.db_transaction", mock_db_transaction),
        patch("kiara.shared.db.db_query", mock_db_query),
    ):
        invalidate_cache()
        yield
        # Cleanup
        c = get_connection()
        c.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
        c.commit()
        c.close()
        invalidate_cache()


def _api_create(client, **overrides):
    """Backend ueber API erstellen."""
    data = {
        "category": "llm",
        "provider": "ollama",
        "name": "Test",
        "model": "qwen3:8b",
        "url": "http://server:11434",
    }
    data.update(overrides)
    return client.post(
        "/api/admin/ki/backends/", json=data, headers={"X-Requested-With": "XMLHttpRequest"}
    )


# ===================================================================
# A) CRUD → Cache → Config-Bridge
# ===================================================================


class TestCrudCacheFlow:
    def test_create_backend_visible_in_selector(self, client, mock_admin, e2e_db):
        """Neues Backend ist sofort im Backend-Selector verfuegbar."""
        resp = _api_create(client, category="llm", model="qwen3:8b", url="http://new-server:11434")
        assert resp.status_code == 200

        # Backend-Selector soll das neue Backend finden
        backend = get_next_backend("llm")
        assert backend is not None
        assert backend["url"] == "http://new-server:11434"
        assert backend["model"] == "qwen3:8b"

    def test_delete_last_backend_selector_returns_none(self, client, mock_admin, e2e_db):
        """Letztes Backend loeschen → Selector gibt None zurueck."""
        resp = _api_create(client, category="vlm", provider="ollama", model="llava:7b")
        bid = resp.json()["id"]

        # Loeschen
        client.delete(
            f"/api/admin/ki/backends/{bid}", headers={"X-Requested-With": "XMLHttpRequest"}
        )

        backend = get_next_backend("vlm")
        assert backend is None

    def test_update_url_propagates_to_selector(self, client, mock_admin, e2e_db):
        """URL-Aenderung ist im naechsten Selector-Aufruf sichtbar."""
        resp = _api_create(client, category="llm", url="http://old:11434")
        bid = resp.json()["id"]

        confirmed_request(
            client,
            "PUT",
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Updated",
                "model": "qwen3:8b",
                "url": "http://new:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        backend = get_next_backend("llm")
        assert backend["url"] == "http://new:11434"


# ===================================================================
# B) Failover-Szenarien
# ===================================================================


@patch("kiara.backend_selector._is_healthy_cached", return_value=True)
class TestFailoverE2E:
    def test_embedding_always_highest_priority(self, _health, client, mock_admin, e2e_db):
        """Embedding-Failover gibt immer das Backend mit niedrigstem Priority-Wert."""
        _api_create(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Prio 20",
            url="http://prio20:11434",
            priority=20,
        )
        _api_create(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Prio 5",
            url="http://prio5:11434",
            priority=5,
        )
        _api_create(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Prio 15",
            url="http://prio15:11434",
            priority=15,
        )

        # Immer Prio-5-Backend (niedrigste Zahl = hoechste Prioritaet)
        for _ in range(5):
            backend = get_next_backend("embedding")
            assert backend["url"] == "http://prio5:11434"

    def test_deactivate_primary_uses_next(self, _health, client, mock_admin, e2e_db):
        """Deaktivierung des primaeren Backends → naechstes wird verwendet."""
        resp1 = _api_create(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Primary",
            url="http://primary:11434",
            priority=10,
        )
        bid1 = resp1.json()["id"]
        _api_create(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Secondary",
            url="http://secondary:11434",
            priority=20,
        )

        # Primaeres deaktivieren
        client.patch(
            f"/api/admin/ki/backends/{bid1}/toggle",
            json={"is_active": False},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        backend = get_next_backend("embedding")
        assert backend["url"] == "http://secondary:11434"


# ===================================================================
# C) Round-Robin Szenarien
# ===================================================================


class TestRoundRobinE2E:
    def test_llm_round_robin_three_backends(self, client, mock_admin, e2e_db):
        """LLM Round-Robin rotiert durch 3 Backends."""
        _api_create(client, name="LLM-A", url="http://a:11434", priority=10)
        _api_create(client, name="LLM-B", url="http://b:11434", priority=20)
        _api_create(client, name="LLM-C", url="http://c:11434", priority=30)

        urls = [get_next_backend("llm")["url"] for _ in range(6)]
        assert urls == [
            "http://a:11434",
            "http://b:11434",
            "http://c:11434",
            "http://a:11434",
            "http://b:11434",
            "http://c:11434",
        ]

    def test_rr_after_backend_removed(self, client, mock_admin, e2e_db):
        """Nach Loeschen eines Backends rotiert RR ueber verbleibende."""
        resp_a = _api_create(client, name="LLM-A", url="http://a:11434", priority=10)
        _api_create(client, name="LLM-B", url="http://b:11434", priority=20)
        resp_c = _api_create(client, name="LLM-C", url="http://c:11434", priority=30)

        # Mittleres loeschen (B)
        # Erst A und B abrufen um RR-Stand zu setzen
        get_next_backend("llm")  # A
        get_next_backend("llm")  # B

        # B loeschen → nur noch A und C
        bid_a = resp_a.json()["id"]
        # Wir loeschen B (id = bid_a + 1 wegen AUTOINCREMENT)
        confirmed_request(
            client,
            "DELETE",
            f"/api/admin/ki/backends/{bid_a + 1}",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        # Nun nur noch A und C
        urls = [get_next_backend("llm")["url"] for _ in range(4)]
        # Nach Cache-Invalidierung startet RR neu
        assert set(urls) == {"http://a:11434", "http://c:11434"}


# ===================================================================
# D) Toggle-Lifecycle
# ===================================================================


class TestToggleLifecycle:
    def test_toggle_off_removes_from_selection(self, client, mock_admin, e2e_db):
        """Deaktiviertes Backend wird nicht mehr selektiert."""
        resp = _api_create(client, category="llm", url="http://only:11434")
        bid = resp.json()["id"]

        # Deaktivieren
        client.patch(
            f"/api/admin/ki/backends/{bid}/toggle",
            json={"is_active": False},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        assert get_next_backend("llm") is None

    def test_toggle_on_adds_back(self, client, mock_admin, e2e_db):
        """Reaktiviertes Backend ist wieder in der Selektion."""
        resp = _api_create(client, category="llm", url="http://toggled:11434")
        bid = resp.json()["id"]

        # Deaktivieren und wieder aktivieren
        client.patch(
            f"/api/admin/ki/backends/{bid}/toggle",
            json={"is_active": False},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert get_next_backend("llm") is None

        client.patch(
            f"/api/admin/ki/backends/{bid}/toggle",
            json={"is_active": True},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        backend = get_next_backend("llm")
        assert backend is not None
        assert backend["url"] == "http://toggled:11434"

    def test_multiple_backends_toggle_one(self, client, mock_admin, e2e_db):
        """Bei mehreren Backends: eines deaktivieren → Rest bleibt verfuegbar."""
        resp_a = _api_create(client, name="A", url="http://a:11434", priority=10)
        _api_create(client, name="B", url="http://b:11434", priority=20)

        bid_a = resp_a.json()["id"]
        client.patch(
            f"/api/admin/ki/backends/{bid_a}/toggle",
            json={"is_active": False},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        # Nur B sollte verfuegbar sein
        all_active = get_all_active("llm")
        assert len(all_active) == 1
        assert all_active[0]["url"] == "http://b:11434"
