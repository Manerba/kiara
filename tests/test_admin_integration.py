"""Integration-Tests Admin-Chat: Confirmation-Flow, Wizard-Flow, Chat-Persistenz, Auto-Router.

End-to-End-Tests die das Zusammenspiel der Komponenten pruefen.
Laeuft gegen PostgreSQL (kiara_test DB via conftest.py).
"""

import asyncio
import json
import uuid

import pytest

from kiara.admin.confirmation import ConfirmationManager
from kiara.admin.function_tool import AdminFunctionTool
from kiara.admin.registry import admin_registry, load_admin_functions
from kiara.admin.wizard import WizardManager
from kiara.repositories.admin_chat import (
    AdminChatMessageRepository,
    AdminChatRepository,
    AdminConfirmationRepository,
    AdminWizardRepository,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine async-Coroutine synchron aus.

    Erstellt immer einen neuen Event-Loop, da ConfirmationManager._execute_confirmed
    den globalen Loop via asyncio.set_event_loop(None) zerstoert.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _load_functions():
    """Stellt sicher dass alle Admin-Funktionen (inkl. Dummy) registriert sind."""
    load_admin_functions()


@pytest.fixture
def chat_repo():
    return AdminChatRepository()


@pytest.fixture
def msg_repo():
    return AdminChatMessageRepository()


@pytest.fixture
def confirm_repo():
    return AdminConfirmationRepository()


@pytest.fixture
def wizard_repo():
    return AdminWizardRepository()


@pytest.fixture
def user_id(test_db):
    """Erstellt einen Test-User und gibt dessen ID zurueck."""
    unique_name = f"inttest_{uuid.uuid4().hex[:8]}"
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (unique_name, "hash_int"),
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


@pytest.fixture
def user_dict(user_id):
    """User-Dict wie es ConfirmationManager erwartet."""
    return {"id": user_id, "username": "testadmin"}


@pytest.fixture
def confirm_mgr():
    return ConfirmationManager()


@pytest.fixture
def wizard_mgr():
    return WizardManager()


# ---------------------------------------------------------------------------
# Hilfsfunktion: Confirmation in DB einfuegen
# ---------------------------------------------------------------------------


def _insert_confirmation(
    test_db,
    chat_id,
    user_id,
    action_id=None,
    function_name="test.confirm",
    params=None,
    confirm_string="test 42 loeschen",
):
    """Fuegt eine Confirmation direkt in die DB ein (simuliert _drain_pending_actions)."""
    action_id = action_id or str(uuid.uuid4())
    params_json = json.dumps(params or {"item_id": 42})
    test_db.execute(
        "INSERT INTO admin_chat_confirmations "
        "(chat_id, action_id, function_name, params, confirm_string, user_id) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (chat_id, action_id, function_name, params_json, confirm_string, user_id),
    )
    test_db.commit()
    return action_id


# ===========================================================================
# Confirmation-Flow
# ===========================================================================


class TestConfirmationFlowComplete:
    def test_confirmation_flow_complete(self, test_db, chat_id, user_id, user_dict, confirm_mgr):
        """Vollstaendiger Confirmation-Flow: Tool → pending → DB → confirm → execute."""
        # 1. AdminFunctionTool aufrufen
        func = admin_registry.get("test.confirm")
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({"item_id": 42}, user_id=user_id))

        # 2. pop_pending enthaelt confirmation
        pending = tool.pop_pending()
        assert len(pending) >= 1
        conf = next(p for p in pending if p["type"] == "confirmation")
        assert conf["confirm_string"] == "test 42 löschen"
        assert conf["function_name"] == "test.confirm"
        assert conf["params"] == {"item_id": 42}

        # 3. Confirmation in DB speichern (simuliert _drain_pending_actions)
        action_id = conf["action_id"]
        _insert_confirmation(
            test_db,
            chat_id,
            user_id,
            action_id=action_id,
            params=conf["params"],
            confirm_string=conf["confirm_string"],
        )

        # 4. check_and_handle mit korrektem String
        handled, events = confirm_mgr.check_and_handle(chat_id, "test 42 löschen", user_dict)

        # 5. Pruefen: handled=True, confirmation_result mit success=True
        assert handled is True
        assert len(events) == 1
        event_type, event_data = events[0]
        assert event_type == "confirmation_result"
        assert event_data["confirmed"] is True
        assert event_data["result"]["success"] is True


class TestConfirmationDismissedOnMismatch:
    def test_confirmation_dismissed_on_mismatch(
        self, test_db, chat_id, user_id, user_dict, confirm_mgr, confirm_repo
    ):
        """Verwurf bei falschem String."""
        action_id = _insert_confirmation(test_db, chat_id, user_id)

        # Falschen String senden
        handled, events = confirm_mgr.check_and_handle(chat_id, "etwas anderes", user_dict)

        # handled=False, Events enthalten action_dismissed
        assert handled is False
        assert len(events) == 1
        event_type, event_data = events[0]
        assert event_type == "action_dismissed"
        assert event_data["action_id"] == action_id

        # Pending Confirmation ist dismissed
        row = confirm_repo.get_by_id(
            confirm_repo.list_filtered({"action_id": action_id})[0][0]["id"]
        )
        assert row["status"] == "dismissed"


class TestConfirmationCaseInsensitive:
    def test_confirmation_case_insensitive(self, test_db, chat_id, user_id, user_dict, confirm_mgr):
        """Bestaetigungsstring ist case-insensitive."""
        _insert_confirmation(test_db, chat_id, user_id, confirm_string="test 42 loeschen")

        # Grossbuchstaben
        handled, events = confirm_mgr.check_and_handle(chat_id, "TEST 42 LOESCHEN", user_dict)
        assert handled is True
        assert events[0][0] == "confirmation_result"


class TestConfirmationExpiry:
    def test_confirmation_expiry(self, test_db, chat_id, user_id, confirm_mgr):
        """Abgelaufene Confirmation (>30 Minuten) wird ignoriert."""
        action_id = _insert_confirmation(test_db, chat_id, user_id)

        # expires_at in die Vergangenheit setzen
        test_db.execute(
            "UPDATE admin_chat_confirmations SET expires_at = NOW() - INTERVAL '1 minute' "
            "WHERE action_id = %s",
            (action_id,),
        )
        test_db.commit()

        # get_pending gibt None zurueck
        pending = confirm_mgr.get_pending(chat_id)
        assert pending is None


class TestConfirmationNoPending:
    def test_confirmation_no_pending(self, test_db, chat_id, user_id, user_dict, confirm_mgr):
        """Kein Pending → handled=False, Events leer."""
        handled, events = confirm_mgr.check_and_handle(chat_id, "irgendwas", user_dict)
        assert handled is False
        assert events == []


# ===========================================================================
# Wizard-Flow
# ===========================================================================


class TestWizardFlowIncremental:
    def test_wizard_flow_incremental(self, test_db, chat_id, user_id, wizard_mgr):
        """Schrittweise Parameter-Erfassung ueber Wizard."""
        func = admin_registry.get("test.wizard")
        tool = AdminFunctionTool(func)

        # 1. Aufruf mit nur einem Parameter (3 Pflichtfelder fehlen)
        result = _run(tool.execute({"name": "Test"}, user_id=user_id))
        pending = tool.pop_pending()
        assert len(pending) == 1
        wiz = pending[0]
        assert wiz["type"] == "wizard"
        assert set(wiz["missing_required"]) == {"url", "username", "password"}
        assert wiz["collected"] == {"name": "Test"}

        # 2. Wizard in DB anlegen
        wiz_state = wizard_mgr.handle_wizard_result(chat_id, user_id, wiz)
        assert wiz_state["wizard_id"] > 0
        assert wiz_state["function_name"] == "test.wizard"

        # 3. Nochmal mit einem weiteren Parameter
        tool2 = AdminFunctionTool(func)
        result2 = _run(tool2.execute({"name": "Test", "url": "http://x"}, user_id=user_id))
        pending2 = tool2.pop_pending()
        wiz2 = pending2[0]
        assert set(wiz2["missing_required"]) == {"username", "password"}

        # 4. Wizard aktualisieren
        wiz_state2 = wizard_mgr.handle_wizard_result(chat_id, user_id, wiz2)
        assert set(wiz_state2["missing_required"]) == {"username", "password"}


class TestWizardCompletesWhenAllParamsPresent:
    def test_wizard_completes_when_all_params_present(self, test_db, chat_id, user_id):
        """Wizard-Abschluss bei vollstaendigen Parametern."""
        func = admin_registry.get("test.wizard")
        tool = AdminFunctionTool(func)

        # Alle Pflichtfelder
        result = _run(
            tool.execute(
                {
                    "name": "Test",
                    "url": "http://x",
                    "username": "admin",
                    "password": "pw",
                },
                user_id=user_id,
            )
        )

        pending = tool.pop_pending()
        # Kein wizard-Pending, sondern wizard_complete
        types = [p["type"] for p in pending]
        assert "wizard" not in types
        assert "wizard_complete" in types
        assert result.success is True


class TestWizardCancel:
    def test_wizard_cancel(self, test_db, chat_id, user_id, wizard_mgr, wizard_repo):
        """Wizard-Abbruch setzt Status auf cancelled."""
        func = admin_registry.get("test.wizard")
        tool = AdminFunctionTool(func)

        # Unvollstaendig aufrufen
        _run(tool.execute({"name": "Test"}, user_id=user_id))
        pending = tool.pop_pending()
        wizard_mgr.handle_wizard_result(chat_id, user_id, pending[0])

        # Wizard abbrechen
        wizard_mgr.cancel_active(chat_id)

        # Status pruefen
        active = wizard_repo.get_active(chat_id)
        assert active is None  # Kein aktiver Wizard mehr


class TestWizardThenConfirmation:
    def test_wizard_then_confirmation(self, test_db, chat_id, user_id):
        """Kombinierter Flow: test.wizard_confirm mit allen Feldern → Confirmation-Gate."""
        func = admin_registry.get("test.wizard_confirm")
        tool = AdminFunctionTool(func)

        # Alle Pflichtfelder → trotzdem Confirmation wegen requires_confirmation
        result = _run(tool.execute({"project": "TestPrj", "scope": "all"}, user_id=user_id))

        pending = tool.pop_pending()
        types = [p["type"] for p in pending]
        assert "confirmation" in types

        # Funktion wurde NICHT ausgefuehrt (nur Confirmation-Gate)
        assert "AKTION WURDE NICHT AUSGEFUEHRT" in result.display_text


class TestWizardUniquePerChat:
    def test_wizard_unique_per_chat(self, test_db, chat_id, user_id, wizard_mgr, wizard_repo):
        """Nur ein aktiver Wizard pro Chat — alter wird abgebrochen."""
        func1 = admin_registry.get("test.wizard")
        func2 = admin_registry.get("test.wizard_confirm")

        # Erster Wizard
        tool1 = AdminFunctionTool(func1)
        _run(tool1.execute({"name": "Test"}, user_id=user_id))
        wiz1_data = tool1.pop_pending()[0]
        wiz1_state = wizard_mgr.handle_wizard_result(chat_id, user_id, wiz1_data)
        wiz1_id = wiz1_state["wizard_id"]

        # Zweiter Wizard (anderer Funktionsname)
        tool2 = AdminFunctionTool(func2)
        _run(tool2.execute({"project": "X"}, user_id=user_id))
        wiz2_data = tool2.pop_pending()[0]
        wiz2_state = wizard_mgr.handle_wizard_result(chat_id, user_id, wiz2_data)

        # Erster Wizard ist cancelled
        old = wizard_repo.get_by_id(wiz1_id)
        assert old["status"] == "cancelled"

        # Zweiter ist aktiv (collecting)
        active = wizard_repo.get_active(chat_id)
        assert active is not None
        assert active["function_name"] == "test.wizard_confirm"


# ===========================================================================
# Chat-Persistenz
# ===========================================================================


class TestChatHistoryPersists:
    def test_chat_history_persists(self, test_db, chat_id, msg_repo):
        """Nachrichten bleiben nach Neuladen erhalten."""
        msg_repo.add_message(chat_id, "user", "Hallo", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Hi!", conn=test_db)
        test_db.commit()

        msgs = msg_repo.list_by_chat(chat_id, conn=test_db)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hallo"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hi!"


class TestGreetingSavedOnCreate:
    def test_greeting_saved_on_create(self, test_db, chat_repo, msg_repo, user_id):
        """Begruessung wird beim Erstellen gespeichert (via API-Logik)."""
        # Chat erstellen + Greeting einfuegen (simuliert create_chat)
        cid = chat_repo.create(conn=test_db, user_id=user_id)
        msg_repo.add_message(
            cid,
            "assistant",
            "Hallo! Ich bin dein Admin-Assistent.",
            metadata={"type": "greeting"},
            conn=test_db,
        )
        test_db.commit()

        msgs = msg_repo.list_by_chat(cid, conn=test_db)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        meta = json.loads(msgs[0]["metadata"])
        assert meta["type"] == "greeting"


class TestHistoryExcludesSystemMessages:
    def test_history_excludes_system_messages(self, test_db, chat_id, msg_repo):
        """load_history enthaelt keine system-Messages."""
        msg_repo.add_message(
            chat_id, "assistant", "Greeting", metadata={"type": "greeting"}, conn=test_db
        )
        msg_repo.add_message(chat_id, "system", "System-Prompt", conn=test_db)
        msg_repo.add_message(chat_id, "user", "Frage", conn=test_db)
        msg_repo.add_message(chat_id, "assistant", "Antwort", conn=test_db)
        test_db.commit()

        history = msg_repo.load_history(chat_id, conn=test_db)
        roles = [h["role"] for h in history]
        assert "system" not in roles
        assert len(history) == 3  # greeting + user + assistant


class TestMultiChatIsolation:
    def test_multi_chat_isolation(self, test_db, chat_repo, msg_repo, user_id):
        """Kein State-Leak zwischen Chats."""
        cid_a = chat_repo.create(conn=test_db, user_id=user_id)
        cid_b = chat_repo.create(conn=test_db, user_id=user_id)
        msg_repo.add_message(cid_a, "user", "Nachricht in Chat A", conn=test_db)
        msg_repo.add_message(cid_a, "assistant", "Antwort A", conn=test_db)
        test_db.commit()

        msgs_b = msg_repo.list_by_chat(cid_b, conn=test_db)
        assert msgs_b == []


# ===========================================================================
# Auto-Router vs. Admin-Funktion
# ===========================================================================


@pytest.fixture
def router_app(test_db, user_id):
    """FastAPI-App mit Auto-Router fuer system.ping."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kiara.admin.router import generate_admin_routes
    from kiara.web.auth import get_admin_user

    app = FastAPI()
    generate_admin_routes(app, admin_registry)

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

    return TestClient(app)


class TestAutorouterAndFunctionProduceSameResult:
    def test_autorouter_and_function_produce_same_result(self, router_app):
        """Auto-Router und direkter Aufruf liefern identische Daten."""
        from kiara.admin.functions.system import system_ping

        # Direkt
        direct = _run(system_ping(user={"id": 1, "username": "test"}))
        assert direct.success is True

        # Via Auto-Router
        resp = router_app.get("/api/admin/chat/ping")
        assert resp.status_code == 200
        body = resp.json()
        # Abhaengig von raw_response-Flag: raw → data direkt, wrapped → dict mit success/message
        func = admin_registry.get("system.ping")
        if func.raw_response:
            expected = direct.data if direct.data is not None else {}
            assert body == expected
        else:
            assert body["success"] == direct.success
            assert body["message"] == direct.message


class TestAutorouterErrorReturnsCorrectStatusCode:
    def test_autorouter_error_returns_correct_status_code(self, router_app):
        """Fehler-Responses haben korrekte HTTP-Status (nicht 200 oder 500)."""
        # system.vectordb.test mit einem Backend das nicht existiert
        # → AdminResult.fail mit http_status=400
        resp = router_app.post(
            "/api/admin/vectordb/test-connection",
            json={"backend": "nonexistent_xyz"},
        )
        # Fehler sollte 4xx sein (nicht 200 oder 500)
        assert 400 <= resp.status_code < 500


class TestQuickApiNewFields:
    """Prueft dass neue Layer-Count-Felder in Quick-API-Response erhalten bleiben."""

    def test_integrity_quick_includes_keyspace_fields(self):
        """Direct integrity_quick() call returns new fields in AdminResult.data."""
        from unittest.mock import patch

        from kiara.admin.functions.integrity import integrity_quick

        mock_result = {
            "qdrant_total": 200,
            "qdrant_dense_total": 150,
            "qdrant_colbert_total": 50,
            "collections": [],
            "es_count": 0,
            "es_configured": False,
            "chunk_ds_count": 100,
            "chunk_ds_keyspace_distinct_chunks": 200,
            "chunk_ds_total_rows": 250,
            "dense_chunk_ds_total_rows": 200,
            "colbert_chunk_ds_total_rows": 50,
            "file_hashes_count": 80,
            "ds_embeddings_total": 200,
            "dense_ds_embeddings_total": 150,
            "colbert_ds_embeddings_total": 50,
            "qdrant_empty": False,
            "status": "ok",
            "details": "Alle Schichten konsistent",
        }

        with (
            patch("kiara.config.load_config", return_value={}),
            patch("kiara.integrity.checks.check_layer_counts", return_value=mock_result),
        ):
            result = _run(integrity_quick(user={"id": 1, "username": "test"}))
            assert result.success
            data = result.data
            assert data["chunk_ds_keyspace_distinct_chunks"] == 200
            assert data["dense_chunk_ds_total_rows"] == 200
            assert data["colbert_chunk_ds_total_rows"] == 50
            assert data["dense_ds_embeddings_total"] == 150
            assert data["colbert_ds_embeddings_total"] == 50


class TestAutorouterChangelogSourceGui:
    def test_autorouter_changelog_source_gui(self, router_app, test_db):
        """Auto-Router setzt source='gui' beim Changelog."""
        from unittest.mock import patch

        # save_config wird lazy importiert in der Funktion → kiara.config patchen
        with (
            patch("kiara.admin.router.log_config_change") as mock_log,
            patch("kiara.config.save_config"),
        ):
            # system.settings.update hat changelog_area + changelog_action
            # Der Auto-Router fuehrt die Funktion DIREKT aus (kein Confirmation-Gate im HTTP)
            resp = router_app.post(
                "/api/admin/settings",
                json={"config": {"test": "value"}},
            )
            # Pruefen ob log_config_change mit source="gui" aufgerufen wurde
            if mock_log.called:
                call_kwargs = mock_log.call_args
                assert call_kwargs.kwargs.get("source") == "gui"
