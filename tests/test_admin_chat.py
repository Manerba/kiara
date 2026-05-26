"""Tests fuer Sprint 02 — Admin-Chat: Repositories, API-Endpunkte, AdminHelpTool.

Laeuft gegen PostgreSQL (kiara_test DB via conftest.py).
"""

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from kiara.repositories.admin_chat import (
    AdminChatMessageRepository,
    AdminChatRepository,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_repo():
    return AdminChatRepository()


@pytest.fixture
def msg_repo():
    return AdminChatMessageRepository()


@pytest.fixture
def user_id(test_db):
    """Erstellt einen Test-User und gibt dessen ID zurueck."""
    unique_name = f"admchat_{uuid.uuid4().hex[:8]}"
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (unique_name, "hash123"),
    )
    uid = cursor.fetchone()["id"]
    test_db.commit()
    return uid


@pytest.fixture
def other_user_id(test_db):
    """Zweiter User fuer Ownership-Tests."""
    unique_name = f"admchat_other_{uuid.uuid4().hex[:8]}"
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (unique_name, "hash456"),
    )
    uid = cursor.fetchone()["id"]
    test_db.commit()
    return uid


@pytest.fixture
def chat_id(test_db, chat_repo, user_id):
    """Erstellt einen Test-Chat und gibt dessen ID zurueck."""
    cid = chat_repo.create(conn=test_db, user_id=user_id)
    test_db.commit()
    return cid


# ---------------------------------------------------------------------------
# AdminChatRepository
# ---------------------------------------------------------------------------


class TestAdminChatCreate:
    def test_create_returns_id(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id)
        assert isinstance(chat_id, int)
        assert chat_id > 0

    def test_create_default_title(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id)
        chat = chat_repo.get_by_id(chat_id, conn=test_db)
        assert chat["title"] == "Neuer Chat"

    def test_create_default_status(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id)
        chat = chat_repo.get_by_id(chat_id, conn=test_db)
        assert chat["status"] == "active"

    def test_create_custom_title(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id, title="Mein Chat")
        chat = chat_repo.get_by_id(chat_id, conn=test_db)
        assert chat["title"] == "Mein Chat"


class TestAdminChatListByUser:
    @staticmethod
    def _cleanup_chats(test_db, uid):
        """Entfernt vorhandene admin_chats fuer den User (Altdaten)."""
        test_db.execute(
            "DELETE FROM admin_chat_messages WHERE chat_id IN "
            "(SELECT id FROM admin_chats WHERE user_id = %s)",
            (uid,),
        )
        test_db.execute("DELETE FROM admin_chats WHERE user_id = %s", (uid,))
        test_db.commit()

    def test_empty_list(self, test_db, chat_repo, user_id):
        self._cleanup_chats(test_db, user_id)
        result = chat_repo.list_by_user(user_id, conn=test_db)
        assert result == []

    def test_returns_own_chats(self, test_db, chat_repo, user_id):
        self._cleanup_chats(test_db, user_id)
        chat_repo.create(conn=test_db, user_id=user_id, title="Chat A")
        chat_repo.create(conn=test_db, user_id=user_id, title="Chat B")
        test_db.commit()
        result = chat_repo.list_by_user(user_id, conn=test_db)
        assert len(result) == 2

    def test_excludes_other_users(self, test_db, chat_repo, user_id, other_user_id):
        self._cleanup_chats(test_db, user_id)
        self._cleanup_chats(test_db, other_user_id)
        chat_repo.create(conn=test_db, user_id=user_id, title="Mein Chat")
        chat_repo.create(conn=test_db, user_id=other_user_id, title="Fremder Chat")
        test_db.commit()
        result = chat_repo.list_by_user(user_id, conn=test_db)
        assert len(result) == 1
        assert result[0]["title"] == "Mein Chat"

    def test_filter_by_status(self, test_db, chat_repo, user_id):
        self._cleanup_chats(test_db, user_id)
        cid1 = chat_repo.create(conn=test_db, user_id=user_id, title="Aktiv")
        cid2 = chat_repo.create(conn=test_db, user_id=user_id, title="Archiv")
        chat_repo.archive(cid2, conn=test_db)
        test_db.commit()
        active = chat_repo.list_by_user(user_id, status="active", conn=test_db)
        archived = chat_repo.list_by_user(user_id, status="archived", conn=test_db)
        assert len(active) == 1
        assert active[0]["title"] == "Aktiv"
        assert len(archived) == 1
        assert archived[0]["title"] == "Archiv"

    def test_message_count_subquery(self, test_db, chat_repo, msg_repo, user_id):
        cid = chat_repo.create(conn=test_db, user_id=user_id)
        msg_repo.add_message(cid, "user", "Frage 1", conn=test_db)
        msg_repo.add_message(cid, "assistant", "Antwort 1", conn=test_db)
        test_db.commit()
        result = chat_repo.list_by_user(user_id, conn=test_db)
        assert result[0]["message_count"] == 2


class TestAdminChatOwnership:
    def test_get_owned_returns_chat(self, test_db, chat_repo, user_id, chat_id):
        chat = chat_repo.get_owned(chat_id, user_id, conn=test_db)
        assert chat is not None
        assert chat["id"] == chat_id

    def test_get_owned_rejects_other_user(
        self, test_db, chat_repo, user_id, other_user_id, chat_id
    ):
        result = chat_repo.get_owned(chat_id, other_user_id, conn=test_db)
        assert result is None

    def test_get_owned_nonexistent(self, test_db, chat_repo, user_id):
        result = chat_repo.get_owned(99999, user_id, conn=test_db)
        assert result is None


class TestAdminChatArchiveRestore:
    def test_archive(self, test_db, chat_repo, user_id, chat_id):
        chat_repo.archive(chat_id, conn=test_db)
        chat = chat_repo.get_by_id(chat_id, conn=test_db)
        assert chat["status"] == "archived"

    def test_restore(self, test_db, chat_repo, user_id, chat_id):
        chat_repo.archive(chat_id, conn=test_db)
        chat_repo.restore(chat_id, conn=test_db)
        chat = chat_repo.get_by_id(chat_id, conn=test_db)
        assert chat["status"] == "active"


class TestAdminChatTouch:
    def test_touch_updates_timestamp(self, test_db, chat_repo, user_id, chat_id):
        before = chat_repo.get_by_id(chat_id, conn=test_db)
        # Kleine Verzoegerung erzwingen via DB
        test_db.execute("SELECT pg_sleep(0.05)")
        chat_repo.touch(chat_id, conn=test_db)
        after = chat_repo.get_by_id(chat_id, conn=test_db)
        assert after["updated_at"] >= before["updated_at"]


class TestAdminChatDelete:
    def test_delete_cascade(self, test_db, chat_repo, msg_repo, user_id, chat_id):
        """DELETE eines Chats loescht auch alle Messages (ON DELETE CASCADE)."""
        msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Antwort", conn=test_db)
        test_db.commit()
        chat_repo.delete(chat_id, conn=test_db)
        test_db.commit()
        msgs = msg_repo.list_by_chat(chat_id, conn=test_db)
        assert msgs == []


# ---------------------------------------------------------------------------
# AdminChatMessageRepository
# ---------------------------------------------------------------------------


class TestAdminChatMessageAdd:
    def test_add_returns_id(self, test_db, msg_repo, chat_id):
        mid = msg_repo.add_message(chat_id, "user", "Hallo", conn=test_db)
        assert isinstance(mid, int)
        assert mid > 0

    def test_add_stores_content(self, test_db, msg_repo, chat_id):
        mid = msg_repo.add_message(chat_id, "user", "Test-Frage", conn=test_db)
        msg = msg_repo.get_by_id(mid, conn=test_db)
        assert msg["role"] == "user"
        assert msg["content"] == "Test-Frage"
        assert msg["chat_id"] == chat_id

    def test_add_json_serialization(self, test_db, msg_repo, chat_id):
        """Sources, tool_calls und metadata werden als JSON gespeichert."""
        sources = [{"source": "test.pdf", "score": 0.9}]
        tool_calls = [{"tool": "search", "params": {"q": "test"}}]
        metadata = {"type": "greeting"}
        mid = msg_repo.add_message(
            chat_id,
            "assistant",
            "Antwort",
            sources=sources,
            tool_calls=tool_calls,
            metadata=metadata,
            conn=test_db,
        )
        msg = msg_repo.get_by_id(mid, conn=test_db)
        assert json.loads(msg["sources"]) == sources
        assert json.loads(msg["tool_calls"]) == tool_calls
        assert json.loads(msg["metadata"]) == metadata

    def test_add_null_optional_fields(self, test_db, msg_repo, chat_id):
        mid = msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        msg = msg_repo.get_by_id(mid, conn=test_db)
        assert msg["sources"] is None
        assert msg["tool_calls"] is None
        assert msg["metadata"] is None


class TestAdminChatMessageList:
    def test_list_by_chat_ordered(self, test_db, msg_repo, chat_id):
        msg_repo.add_message(chat_id, "user", "Frage 1", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Antwort 1", conn=test_db)
        msg_repo.add_message(chat_id, "user", "Frage 2", conn=test_db)
        test_db.commit()
        msgs = msg_repo.list_by_chat(chat_id, conn=test_db)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "Frage 1"
        assert msgs[1]["content"] == "Antwort 1"
        assert msgs[2]["content"] == "Frage 2"

    def test_list_empty_chat(self, test_db, msg_repo, chat_id):
        msgs = msg_repo.list_by_chat(chat_id, conn=test_db)
        assert msgs == []


class TestAdminChatMessageHistory:
    def test_load_history_chronological(self, test_db, msg_repo, chat_id):
        """load_history gibt Nachrichten in chronologischer Reihenfolge zurueck."""
        msg_repo.add_message(chat_id, "user", "Erste", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Zweite", conn=test_db)
        msg_repo.add_message(chat_id, "user", "Dritte", conn=test_db)
        test_db.commit()
        history = msg_repo.load_history(chat_id, conn=test_db)
        assert len(history) == 3
        assert history[0]["content"] == "Erste"
        assert history[2]["content"] == "Dritte"

    def test_load_history_excludes_system(self, test_db, msg_repo, chat_id):
        """system-Role wird aus der History ausgeschlossen."""
        msg_repo.add_message(chat_id, "system", "System-Prompt", conn=test_db)
        msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Antwort", conn=test_db)
        test_db.commit()
        history = msg_repo.load_history(chat_id, conn=test_db)
        assert len(history) == 2
        roles = [h["role"] for h in history]
        assert "system" not in roles

    def test_load_history_limit(self, test_db, msg_repo, chat_id):
        for i in range(10):
            msg_repo.add_message(chat_id, "user", f"Msg {i}", conn=test_db)
        test_db.commit()
        history = msg_repo.load_history(chat_id, limit=3, conn=test_db)
        assert len(history) == 3
        # Die letzten 3 (neuesten) Nachrichten
        assert history[0]["content"] == "Msg 7"
        assert history[2]["content"] == "Msg 9"

    def test_load_history_returns_role_content_metadata(self, test_db, msg_repo, chat_id):
        msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        test_db.commit()
        history = msg_repo.load_history(chat_id, conn=test_db)
        assert set(history[0].keys()) == {"role", "content", "metadata"}

    def test_load_history_includes_metadata(self, test_db, msg_repo, chat_id):
        """load_history liefert metadata (fuer Setup-Message-Filterung)."""
        msg_repo.add_message(
            chat_id,
            "assistant",
            "Setup-Text",
            metadata={"type": "setup_wizard"},
            conn=test_db,
        )
        msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        test_db.commit()
        history = msg_repo.load_history(chat_id, conn=test_db)
        assert len(history) == 2
        meta = json.loads(history[0]["metadata"])
        assert meta["type"] == "setup_wizard"
        assert history[1]["metadata"] is None


class TestAdminChatMessageDelete:
    def test_delete_by_chat(self, test_db, msg_repo, chat_id):
        msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Antwort", conn=test_db)
        test_db.commit()
        msg_repo.delete_by_chat(chat_id, conn=test_db)
        test_db.commit()
        msgs = msg_repo.list_by_chat(chat_id, conn=test_db)
        assert msgs == []


# ---------------------------------------------------------------------------
# AdminHelpTool
# ---------------------------------------------------------------------------


def _run(coro):
    """Hilfsfunktion: fuehrt eine async-Coroutine synchron aus."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# API-Endpunkte (FastAPI TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(test_db, user_id):
    """FastAPI TestClient mit Admin-Chat-Router und Auth-Override."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kiara.web.api_admin_chat import router
    from kiara.web.auth import get_admin_user

    # Vorhandene admin_chats des Test-Users entfernen (Altdaten aus frueheren Runs).
    # Cleanup auf test_db (gleiche Connection) um Deadlocks zu vermeiden.
    test_db.execute(
        "DELETE FROM admin_chat_messages WHERE chat_id IN "
        "(SELECT id FROM admin_chats WHERE user_id = %s)",
        (user_id,),
    )
    test_db.execute("DELETE FROM admin_chats WHERE user_id = %s", (user_id,))
    test_db.commit()

    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_admin_user] = lambda: {
        "id": user_id,
        "username": "testadmin",
        "vorname": "Test",
        "nachname": "Admin",
        "email": "test@test.de",
        "is_admin": True,
        "must_change_password": False,
        "group_ids": [],
        "datasource_ids": [],
        "_session_token": "test_token",
    }

    yield TestClient(app)

    # Cleanup nach dem Test: auf test_db Connection um Lock-Konflikte zu vermeiden
    test_db.execute(
        "DELETE FROM admin_chat_messages WHERE chat_id IN "
        "(SELECT id FROM admin_chats WHERE user_id = %s)",
        (user_id,),
    )
    test_db.execute("DELETE FROM admin_chats WHERE user_id = %s", (user_id,))
    test_db.commit()


class TestApiListChats:
    def test_empty_list(self, api_client):
        resp = api_client.get("/api/admin-chat/chats")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_create(self, api_client):
        api_client.post("/api/admin-chat/chats")
        resp = api_client.get("/api/admin-chat/chats")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestApiCreateChat:
    def test_create_returns_201(self, api_client):
        resp = api_client.post("/api/admin-chat/chats")
        assert resp.status_code == 201

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=True)
    def test_create_has_greeting_message(self, mock_backend, api_client):
        resp = api_client.post("/api/admin-chat/chats")
        data = resp.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1
        greeting = data["messages"][0]
        assert greeting["role"] == "assistant"
        assert "Hallo" in greeting["content"]
        meta = json.loads(greeting["metadata"])
        assert meta["type"] == "greeting"

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=True)
    def test_create_greeting_contains_kpis(self, mock_backend, api_client):
        resp = api_client.post("/api/admin-chat/chats")
        content = resp.json()["messages"][0]["content"]
        assert "KI-Backends" in content
        assert "Datenquellen" in content
        assert "VektorDB" in content

    def test_create_setup_wizard_without_backends(self, api_client):
        """Ohne aktive Backends: Setup-Wizard statt KPI-Greeting."""
        with patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=False):
            resp = api_client.post("/api/admin-chat/chats")
        data = resp.json()
        greeting = data["messages"][0]
        assert "Willkommen" in greeting["content"]
        meta = json.loads(greeting["metadata"])
        assert meta["type"] == "setup_wizard"

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=False)
    def test_create_stale_greeting_type_mismatch(self, mock_backend, api_client):
        """Client sendet veralteten greeting_type -> Server ignoriert Client-Text."""
        resp = api_client.post(
            "/api/admin-chat/chats",
            json={"greeting": "Hallo KPIs...", "greeting_type": "greeting"},
        )
        data = resp.json()
        greeting = data["messages"][0]
        # Server hat Setup-Wizard generiert, nicht den Client-Text
        assert "Willkommen" in greeting["content"]
        meta = json.loads(greeting["metadata"])
        assert meta["type"] == "setup_wizard"


class TestApiGetChat:
    def test_get_existing_chat(self, api_client):
        create_resp = api_client.post("/api/admin-chat/chats")
        chat_id = create_resp.json()["id"]
        resp = api_client.get(f"/api/admin-chat/chats/{chat_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == chat_id
        assert "messages" in resp.json()

    def test_get_nonexistent_returns_404(self, api_client):
        resp = api_client.get("/api/admin-chat/chats/99999")
        assert resp.status_code == 404


class TestApiUpdateChat:
    def test_update_title(self, api_client):
        create_resp = api_client.post("/api/admin-chat/chats")
        chat_id = create_resp.json()["id"]
        resp = api_client.put(
            f"/api/admin-chat/chats/{chat_id}",
            json={"title": "Neuer Titel"},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Neuer Titel"

    def test_update_status(self, api_client):
        create_resp = api_client.post("/api/admin-chat/chats")
        chat_id = create_resp.json()["id"]
        resp = api_client.put(
            f"/api/admin-chat/chats/{chat_id}",
            json={"status": "archived"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"

    def test_update_no_changes_returns_400(self, api_client):
        create_resp = api_client.post("/api/admin-chat/chats")
        chat_id = create_resp.json()["id"]
        resp = api_client.put(
            f"/api/admin-chat/chats/{chat_id}",
            json={},
        )
        assert resp.status_code == 400

    def test_update_nonexistent_returns_404(self, api_client):
        resp = api_client.put(
            "/api/admin-chat/chats/99999",
            json={"title": "Test"},
        )
        assert resp.status_code == 404


class TestApiDeleteChat:
    def test_delete_returns_204(self, api_client):
        create_resp = api_client.post("/api/admin-chat/chats")
        chat_id = create_resp.json()["id"]
        resp = api_client.delete(f"/api/admin-chat/chats/{chat_id}")
        assert resp.status_code == 204

    def test_delete_removes_chat(self, api_client):
        create_resp = api_client.post("/api/admin-chat/chats")
        chat_id = create_resp.json()["id"]
        api_client.delete(f"/api/admin-chat/chats/{chat_id}")
        resp = api_client.get(f"/api/admin-chat/chats/{chat_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, api_client):
        resp = api_client.delete("/api/admin-chat/chats/99999")
        assert resp.status_code == 404


class TestApiGreeting:
    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=True)
    def test_greeting_returns_kpi_type(self, mock_backend, api_client):
        resp = api_client.get("/api/admin-chat/greeting")
        assert resp.status_code == 200
        data = resp.json()
        assert "username" in data
        assert "greeting" in data
        assert data["type"] == "greeting"
        assert "Hallo" in data["greeting"]

    def test_greeting_returns_setup_type_without_backends(self, api_client):
        with patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=False):
            resp = api_client.get("/api/admin-chat/greeting")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "setup_wizard"
        assert "Willkommen" in data["greeting"]


class TestApiBackends:
    def test_backends_returns_list(self, api_client):
        resp = api_client.get("/api/admin-chat/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_backends_no_api_key(self, api_client):
        """API-Keys duerfen NICHT in der Antwort stehen."""
        resp = api_client.get("/api/admin-chat/backends")
        for b in resp.json():
            assert "api_key" not in b
            assert set(b.keys()) <= {"id", "name", "provider", "model"}


# ---------------------------------------------------------------------------
# _format_greeting()
# ---------------------------------------------------------------------------


class TestFormatGreeting:
    def test_format_with_running_job(self):
        from kiara.web.api_admin_chat import _format_greeting

        kpis = {
            "backends": {"total": 5, "active": 3},
            "datasources": {"total": 10, "active": 8},
            "vectordb": {"chunks_by_model": [{"model": "m1", "chunks": 1000}]},
            "jobs": {"running": {"datasource": "TestDS"}, "pending": 2},
        }
        text = _format_greeting("Max", kpis)
        assert "Hallo Max" in text
        assert "3/5" in text
        assert "8/10" in text
        assert "1000 Chunks" in text
        assert "1 laufend (TestDS)" in text

    def test_format_no_running_job(self):
        from kiara.web.api_admin_chat import _format_greeting

        kpis = {
            "backends": {"total": 2, "active": 2},
            "datasources": {"total": 0, "active": 0},
            "vectordb": {"chunks_by_model": []},
            "jobs": {"running": None, "pending": 0},
        }
        text = _format_greeting("Admin", kpis)
        assert "keine laufend" in text
        assert "0 Chunks" in text

    def test_format_multiple_models(self):
        from kiara.web.api_admin_chat import _format_greeting

        kpis = {
            "backends": {"total": 1, "active": 1},
            "datasources": {"total": 1, "active": 1},
            "vectordb": {
                "chunks_by_model": [
                    {"model": "m1", "chunks": 500},
                    {"model": "m2", "chunks": 300},
                ]
            },
            "jobs": {"running": None, "pending": 0},
        }
        text = _format_greeting("Test", kpis)
        assert "800 Chunks" in text


# ---------------------------------------------------------------------------
# Sprint 11a: Setup-Flow, History-Filter, Cloud-Fallback
# ---------------------------------------------------------------------------


class TestSetupGreetingTexts:
    def test_setup_greeting_text_content(self):
        from kiara.web.api_admin_chat import _setup_greeting_text

        text = _setup_greeting_text()
        assert "Willkommen" in text
        assert "Cloud-Backend" in text
        assert "Lokales Backend" in text
        assert "hilfe" in text

    def test_help_text_content(self):
        from kiara.web.api_admin_chat import _help_text

        text = _help_text()
        assert "Einrichtungshilfe" in text
        assert "Anthropic" in text
        assert "OpenAI" in text
        assert "Ollama" in text
        assert "abbrechen" in text
        assert "Kiron" in text

    def test_build_chat_greeting_no_backend(self):
        from kiara.web.api_admin_chat import _build_chat_greeting

        with patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=False):
            text, meta_type = _build_chat_greeting({"username": "Admin"})
        assert meta_type == "setup_wizard"
        assert "Willkommen" in text

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=True)
    def test_build_chat_greeting_with_backend(self, mock_backend):
        from kiara.web.api_admin_chat import _build_chat_greeting

        text, meta_type = _build_chat_greeting({"username": "Admin"})
        assert meta_type == "greeting"
        assert "Hallo" in text


class TestHandleSetupMode:
    @pytest.fixture(autouse=True)
    def _clean_setup_state(self):
        """Bereinigt den globalen Setup-State vor/nach jedem Test."""
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    def test_hilfe_returns_help_text(self):
        from kiara.web.api_admin_chat import _handle_setup_mode

        response, mask = asyncio.run(_handle_setup_mode(9001, "hilfe", {"id": 1}))
        assert "Einrichtungshilfe" in response
        assert mask is False

    def test_abbrechen_resets_to_greeting(self):
        from kiara.web.api_admin_chat import _handle_setup_mode

        response, mask = asyncio.run(_handle_setup_mode(9002, "abbrechen", {"id": 1}))
        assert "Willkommen" in response
        assert mask is False

    def test_cloud_anthropic_asks_for_key(self):
        from kiara.web.api_admin_chat import _handle_setup_mode

        response, mask = asyncio.run(_handle_setup_mode(9003, "cloud anthropic", {"id": 1}))
        assert "API-Key" in response
        assert mask is False

    def test_cloud_openai_asks_for_key(self):
        from kiara.web.api_admin_chat import _handle_setup_mode

        response, mask = asyncio.run(_handle_setup_mode(9004, "cloud openai", {"id": 1}))
        assert "API-Key" in response
        assert mask is False

    def test_lokal_without_url_shows_unknown(self):
        """'lokal' ohne URL wird nicht als Befehl erkannt (braucht 'lokal <URL>')."""
        from kiara.web.api_admin_chat import _handle_setup_mode

        response, mask = asyncio.run(_handle_setup_mode(9005, "lokal", {"id": 1}))
        assert "Einrichtungsmodus" in response
        assert mask is False

    def test_unknown_command_in_setup(self):
        from kiara.web.api_admin_chat import _handle_setup_mode

        response, mask = asyncio.run(_handle_setup_mode(9006, "was ist kiara?", {"id": 1}))
        assert "Einrichtungsmodus" in response
        assert mask is False


class TestSetupSseGenerator:
    def test_normal_message_saved_and_streamed(self, test_db, chat_id):
        from kiara.web.api_admin_chat import _setup_sse_generator

        events = list(_setup_sse_generator(chat_id, "hilfe", "Hilfe-Text"))
        # Sollte chunk + done enthalten
        assert any("chunk" in e for e in events)
        assert any("done" in e for e in events)
        # User-Nachricht in DB gespeichert
        from kiara.repositories.admin_chat import AdminChatMessageRepository

        msgs = AdminChatMessageRepository().list_by_chat(chat_id)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "hilfe"

    def test_masked_input_saves_placeholder(self, test_db, chat_id):
        from kiara.web.api_admin_chat import _setup_sse_generator

        events = list(
            _setup_sse_generator(
                chat_id,
                "sk-ant-GEHEIM",
                "Backend konfiguriert!",
                mask_input=True,
            )
        )
        # user_masked Event gesendet
        assert any("user_masked" in e for e in events)
        # DB enthaelt Platzhalter, nicht den Key
        from kiara.repositories.admin_chat import AdminChatMessageRepository

        msgs = AdminChatMessageRepository().list_by_chat(chat_id)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert user_msgs[0]["content"] == "(API-Key eingegeben)"


class TestHistoryFilterSetupMessages:
    def test_setup_messages_filtered_from_history_block(self, test_db, msg_repo, chat_id):
        """Setup-Messages und maskierte Keys werden aus dem LLM-History-Block gefiltert."""
        # Setup-Messages einfuegen
        msg_repo.add_message(
            chat_id,
            "assistant",
            "Willkommen...",
            metadata={"type": "setup_wizard"},
            conn=test_db,
        )
        msg_repo.add_message(
            chat_id,
            "user",
            "(API-Key eingegeben)",
            metadata={"type": "setup_input"},
            conn=test_db,
        )
        msg_repo.add_message(
            chat_id,
            "assistant",
            "Backend konfiguriert!",
            metadata={"type": "setup_response"},
            conn=test_db,
        )
        # Normale Nachrichten
        msg_repo.add_message(chat_id, "user", "Wie ist der Status?", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Alles OK.", conn=test_db)
        test_db.commit()

        from kiara.web.api_admin_chat import _SETUP_META_TYPES

        history = msg_repo.load_history(chat_id)
        lines = []
        for msg in history:
            if msg["content"] == "(API-Key eingegeben)":
                continue
            meta_raw = msg.get("metadata")
            if meta_raw:
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                if meta.get("type") in _SETUP_META_TYPES:
                    continue
            lines.append(msg["content"])

        # Nur die normalen Nachrichten sollten uebrig sein
        assert len(lines) == 2
        assert "Wie ist der Status?" in lines
        assert "Alles OK." in lines
        assert "(API-Key eingegeben)" not in lines


# ---------------------------------------------------------------------------
# Sprint 11a Ergaenzungen: State-Machine, Cleanup, Fallback, Edge-Cases
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Fuehrt eine Coroutine synchron aus (Python 3.12+)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSetupStateMachineCloudKey:
    """P0: awaiting_cloud_key State — Backend-Erstellung, Test, Aktivierung, Rollback."""

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cloud_key_success_activates_backend(self, mock_svc):
        """Erfolgreicher API-Key → Backend angelegt, getestet, aktiviert."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.create_backend = AsyncMock(return_value={"id": 42})
        mock_svc.test_backend = AsyncMock(return_value={"models": ["claude-sonnet-4-6"]})
        mock_svc.update_backend = AsyncMock()

        # State vorbereiten: wir warten auf den Key
        _setup_state[7001] = {
            "state": "awaiting_cloud_key",
            "provider": "anthropic",
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7001, "sk-ant-secret123", {"id": 1}))

        assert "konfiguriert" in response
        assert "claude-sonnet" in response
        assert mask is True  # API-Key soll maskiert werden
        mock_svc.create_backend.assert_awaited_once()
        mock_svc.update_backend.assert_awaited_once_with(42, is_active=True)
        assert 7001 not in _setup_state  # State aufgeraeumt

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cloud_key_failure_rolls_back(self, mock_svc):
        """Fehlgeschlagener Test → Backend wird geloescht (Rollback)."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.create_backend = AsyncMock(return_value={"id": 99})
        mock_svc.test_backend = AsyncMock(side_effect=ValueError("Unauthorized"))
        mock_svc.delete_backend = AsyncMock()

        _setup_state[7002] = {
            "state": "awaiting_cloud_key",
            "provider": "anthropic",
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7002, "sk-bad-key", {"id": 1}))

        assert "konnte nicht eingerichtet werden" in response
        assert mask is True  # Key bleibt maskiert auch bei Fehler
        mock_svc.delete_backend.assert_awaited_once_with(99)

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cloud_key_model_not_available(self, mock_svc):
        """Modell nicht im API-Key enthalten → Rollback."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.create_backend = AsyncMock(return_value={"id": 55})
        mock_svc.test_backend = AsyncMock(return_value={"models": ["gpt-3.5-turbo"]})
        mock_svc.delete_backend = AsyncMock()

        _setup_state[7003] = {
            "state": "awaiting_cloud_key",
            "provider": "anthropic",
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7003, "sk-limited", {"id": 1}))

        assert "konnte nicht eingerichtet werden" in response
        mock_svc.delete_backend.assert_awaited_once_with(55)


class TestSetupStateMachineModelChoice:
    """P0: awaiting_model_choice State — Ollama-Modellauswahl."""

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_select_model_by_number(self, mock_svc):
        """Numerische Auswahl '2' waehlt das zweite Modell."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.update_backend = AsyncMock()
        _setup_state[7010] = {
            "state": "awaiting_model_choice",
            "backend_id": 33,
            "models": ["qwen3:8b", "llama3.3:70b", "nomic-embed-text"],
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7010, "2", {"id": 1}))

        assert "llama3.3:70b" in response
        assert "bereit" in response
        assert mask is False
        mock_svc.update_backend.assert_awaited_once_with(33, model="llama3.3:70b", is_active=True)
        assert 7010 not in _setup_state

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_select_model_by_name(self, mock_svc):
        """Textuelle Auswahl per Modellname (case-insensitive)."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.update_backend = AsyncMock()
        _setup_state[7011] = {
            "state": "awaiting_model_choice",
            "backend_id": 34,
            "models": ["qwen3:8b", "llama3.3:70b"],
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7011, "qwen3:8b", {"id": 1}))

        assert "qwen3:8b" in response
        mock_svc.update_backend.assert_awaited_once_with(34, model="qwen3:8b", is_active=True)

    def test_invalid_model_choice_shows_options(self):
        """Ungueltige Auswahl zeigt die Optionsliste erneut."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        _setup_state[7012] = {
            "state": "awaiting_model_choice",
            "backend_id": 35,
            "models": ["qwen3:8b", "llama3.3:70b"],
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7012, "99", {"id": 1}))

        assert "Ungueltige" in response
        assert "qwen3:8b" in response
        assert "llama3.3:70b" in response
        assert 7012 in _setup_state  # State bleibt bestehen

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_activate_failure_returns_error(self, mock_svc):
        """Backend-Aktivierung schlaegt fehl → Fehlermeldung."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.update_backend = AsyncMock(side_effect=RuntimeError("DB locked"))
        _setup_state[7013] = {
            "state": "awaiting_model_choice",
            "backend_id": 36,
            "models": ["qwen3:8b"],
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(7013, "1", {"id": 1}))

        assert "Fehler" in response
        assert "DB locked" in response


class TestSetupLocalBackendFlow:
    """P0: Lokales Backend Setup via 'lokal http://...'."""

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_lokal_single_model_auto_activates(self, mock_svc):
        """Genau 1 Modell → wird direkt aktiviert ohne Rückfrage."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.create_backend = AsyncMock(return_value={"id": 70})
        mock_svc.test_backend = AsyncMock(return_value={"models": ["qwen3:8b"]})
        mock_svc.update_backend = AsyncMock()

        response, mask = _run_async(
            _handle_setup_mode(7020, "lokal http://gpu-server.local:11434", {"id": 1})
        )

        assert "konfiguriert" in response
        assert "qwen3:8b" in response
        mock_svc.update_backend.assert_awaited_once_with(70, model="qwen3:8b", is_active=True)
        assert 7020 not in _setup_state

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_lokal_multi_model_asks_for_choice(self, mock_svc):
        """Mehrere Modelle → Modellauswahl wird angeboten, Backend bleibt inaktiv."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.create_backend = AsyncMock(return_value={"id": 71})
        mock_svc.test_backend = AsyncMock(return_value={"models": ["qwen3:8b", "llama3.3:70b"]})

        response, mask = _run_async(
            _handle_setup_mode(7021, "lokal http://gpu-server.local:11434", {"id": 1})
        )

        assert "Verfuegbare Modelle" in response
        assert "[1]" in response
        assert "qwen3:8b" in response
        assert 7021 in _setup_state
        assert _setup_state[7021]["state"] == "awaiting_model_choice"
        assert _setup_state[7021]["backend_id"] == 71

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_lokal_no_models_rolls_back(self, mock_svc):
        """Keine Modelle gefunden → Rollback."""
        from kiara.web.api_admin_chat import _handle_setup_mode

        mock_svc.create_backend = AsyncMock(return_value={"id": 72})
        mock_svc.test_backend = AsyncMock(return_value={"models": []})
        mock_svc.delete_backend = AsyncMock()

        response, mask = _run_async(
            _handle_setup_mode(7022, "lokal http://gpu-server.local:11434", {"id": 1})
        )

        assert "konnte nicht eingerichtet werden" in response
        mock_svc.delete_backend.assert_awaited_once_with(72)

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_lokal_connection_failure_rolls_back(self, mock_svc):
        """Server nicht erreichbar → Rollback."""
        from kiara.web.api_admin_chat import _handle_setup_mode

        mock_svc.create_backend = AsyncMock(return_value={"id": 73})
        mock_svc.test_backend = AsyncMock(side_effect=ConnectionError("Connection refused"))
        mock_svc.delete_backend = AsyncMock()

        response, mask = _run_async(
            _handle_setup_mode(7023, "lokal http://bad-host:11434", {"id": 1})
        )

        assert "konnte nicht eingerichtet werden" in response
        mock_svc.delete_backend.assert_awaited_once_with(73)


class TestCleanupSetupState:
    """P1: _cleanup_setup_state loescht State + ggf. Backend."""

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cleanup_deletes_pending_backend(self, mock_svc):
        """Backend-ID im State → wird geloescht."""
        from kiara.web.api_admin_chat import _cleanup_setup_state, _setup_state

        mock_svc.delete_backend = AsyncMock()
        _setup_state[8001] = {"backend_id": 42, "state": "awaiting_model_choice", "ts": time.time()}

        _run_async(_cleanup_setup_state(8001))

        mock_svc.delete_backend.assert_awaited_once_with(42)
        assert 8001 not in _setup_state

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cleanup_without_backend_id(self, mock_svc):
        """Kein backend_id im State → nur State entfernen, kein delete_backend."""
        from kiara.web.api_admin_chat import _cleanup_setup_state, _setup_state

        _setup_state[8002] = {"state": "awaiting_cloud_key", "ts": time.time()}

        _run_async(_cleanup_setup_state(8002))

        mock_svc.delete_backend.assert_not_called()
        assert 8002 not in _setup_state

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cleanup_handles_delete_failure(self, mock_svc):
        """delete_backend schlaegt fehl → kein Crash, nur Warning."""
        from kiara.web.api_admin_chat import _cleanup_setup_state, _setup_state

        mock_svc.delete_backend = AsyncMock(side_effect=RuntimeError("not found"))
        _setup_state[8003] = {"backend_id": 99, "ts": time.time()}

        # Soll nicht crashen
        _run_async(_cleanup_setup_state(8003))
        assert 8003 not in _setup_state

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_abort_during_awaiting_cloud_key_cleans_up(self, mock_svc):
        """'abbrechen' waehrend awaiting_cloud_key → State + Backend aufgeraeumt."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.delete_backend = AsyncMock()
        _setup_state[8004] = {
            "state": "awaiting_cloud_key",
            "provider": "anthropic",
            "backend_id": 55,
            "ts": time.time(),
        }

        response, mask = _run_async(_handle_setup_mode(8004, "abbrechen", {"id": 1}))

        assert "Willkommen" in response
        assert 8004 not in _setup_state
        mock_svc.delete_backend.assert_awaited_once_with(55)


class TestSetupStateLocking:
    """Guards fuer den zentral gekapselten Setup-State."""

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    def test_state_helpers_copy_mutable_state(self):
        from kiara.web.api_admin_chat import (
            _get_setup_state,
            _pop_setup_state,
            _set_setup_state,
        )

        state = {"state": "awaiting_model_choice", "models": ["qwen3:8b"], "ts": time.time()}
        _run_async(_set_setup_state(8101, state))

        state["models"].append("llama3.3:70b")
        loaded = _run_async(_get_setup_state(8101))
        assert loaded["models"] == ["qwen3:8b"]

        loaded["models"].append("mutated-copy")
        loaded_again = _run_async(_get_setup_state(8101))
        assert loaded_again["models"] == ["qwen3:8b"]

        popped = _run_async(_pop_setup_state(8101))
        assert popped["models"] == ["qwen3:8b"]
        assert _run_async(_get_setup_state(8101)) == {}

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_cleanup_deletes_pending_backend_after_releasing_lock(self, mock_svc):
        from kiara.web.api_admin_chat import (
            _cleanup_setup_state,
            _setup_state,
            _setup_state_lock,
        )

        async def _delete_backend(backend_id):
            assert backend_id == 77
            assert not _setup_state_lock.locked()

        mock_svc.delete_backend = AsyncMock(side_effect=_delete_backend)
        _setup_state[8102] = {"backend_id": 77, "state": "awaiting_model_choice", "ts": time.time()}

        _run_async(_cleanup_setup_state(8102))

        mock_svc.delete_backend.assert_awaited_once_with(77)
        assert 8102 not in _setup_state


class TestSetupTTLCleanup:
    """P1: Stale Setup-States werden nach 30 Minuten aufgeraeumt."""

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        from kiara.web.api_admin_chat import _setup_state

        _setup_state.clear()
        yield
        _setup_state.clear()

    @patch("kiara.admin.services.ki_backend_service", create=True)
    def test_stale_states_cleaned_on_next_call(self, mock_svc):
        """States aelter als 30 Min werden beim naechsten Aufruf entfernt."""
        from kiara.web.api_admin_chat import _handle_setup_mode, _setup_state

        mock_svc.delete_backend = AsyncMock()

        # Staler State (>30 Min alt)
        _setup_state[9001] = {
            "state": "awaiting_cloud_key",
            "backend_id": 88,
            "ts": time.time() - 2000,  # 33 Minuten alt
        }
        # Frischer State (aktuell)
        _setup_state[9002] = {
            "state": "awaiting_cloud_key",
            "ts": time.time(),
        }

        # Neuer Aufruf fuer einen dritten Chat triggert Cleanup
        _run_async(_handle_setup_mode(9003, "hilfe", {"id": 1}))

        assert 9001 not in _setup_state  # staler State entfernt
        assert 9002 in _setup_state  # frischer State bleibt
        mock_svc.delete_backend.assert_awaited_once_with(88)


class TestHasActiveLlmBackendEdgeCases:
    """P2: _has_active_llm_backend Fehlerbehandlung."""

    def test_returns_false_on_exception(self):
        """Bei DB-Fehler → False (graceful degradation, kein Crash)."""
        from kiara.web.api_admin_chat import _has_active_llm_backend

        with patch(
            "kiara.repositories.backend.KiBackendRepository.list_by_category",
            side_effect=RuntimeError("DB unavailable"),
        ):
            result = _has_active_llm_backend()
        assert result is False


class TestCreateChatGreetingTypeMatch:
    """P2: create_chat uebernimmt Client-Text wenn greeting_type matched."""

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=False)
    def test_matching_type_uses_client_text(self, mock_backend, api_client):
        """Client sendet setup_wizard-Typ → Server uebernimmt Client-Text."""
        custom_text = "Mein eigener Setup-Text"
        resp = api_client.post(
            "/api/admin-chat/chats",
            json={"greeting": custom_text, "greeting_type": "setup_wizard"},
        )
        data = resp.json()
        greeting = data["messages"][0]
        assert greeting["content"] == custom_text
        meta = json.loads(greeting["metadata"])
        assert meta["type"] == "setup_wizard"

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=True)
    def test_matching_kpi_type_uses_client_text(self, mock_backend, api_client):
        """Client sendet greeting-Typ mit Backend aktiv → Client-Text."""
        custom_kpi = "Hallo! Meine KPIs..."
        resp = api_client.post(
            "/api/admin-chat/chats",
            json={"greeting": custom_kpi, "greeting_type": "greeting"},
        )
        data = resp.json()
        assert data["messages"][0]["content"] == custom_kpi

    @patch("kiara.web.api_admin_chat._has_active_llm_backend", return_value=True)
    def test_empty_body_uses_server_greeting(self, mock_backend, api_client):
        """Leerer Body → Server generiert KPI-Greeting."""
        resp = api_client.post("/api/admin-chat/chats")
        data = resp.json()
        assert "Hallo" in data["messages"][0]["content"]


# =====================================================================
# Issue #1057 — Admin-Chat darf Cloud-Backends nicht ohne Anonymizer verwenden
# =====================================================================


class TestAdminChatAnonymizerWiring:
    """Issue #1057: Stream-Generator muss Anonymizer erzeugen + an AdminAgentChain durchreichen."""

    def test_stream_admin_query_creates_anonymizer_and_passes_to_chain(self):
        import inspect

        from kiara.web.api_admin_chat import _stream_admin_query

        src = inspect.getsource(_stream_admin_query)
        # Anonymizer wird im selben Modul wie der User-Agent-Pfad erzeugt
        assert "_create_agent_anonymizer(" in src, (
            "Anonymizer-Aufruf fehlt — Admin-Chat wuerde Cloud-Backends ohne PII-Schutz verwenden"
        )
        # AdminAgentChain bekommt den Anonymizer als Keyword-Argument
        assert "anonymizer=anonymizer" in src, (
            "AdminAgentChain bekommt den Anonymizer nicht durchgereicht — Tool-Adapter laeuft ungeschuetzt"
        )
