"""Tests fuer Sprint 03: Bestaetigungs- & Wizard-System.

Testet AdminConfirmationRepository, AdminWizardRepository,
ConfirmationManager, WizardManager und AdminFunctionTool.
"""

import asyncio
import json
import uuid

import pytest

from kiara.admin.confirmation import ConfirmationManager
from kiara.admin.function import AdminFunction
from kiara.admin.function_tool import AdminFunctionTool
from kiara.admin.result import AdminResult
from kiara.admin.wizard import WizardManager
from kiara.repositories.admin_chat import (
    AdminChatRepository,
    AdminConfirmationRepository,
    AdminWizardRepository,
)

# --- Fixtures ---


@pytest.fixture
def confirm_repo():
    return AdminConfirmationRepository()


@pytest.fixture
def wizard_repo():
    return AdminWizardRepository()


@pytest.fixture
def chat_repo():
    return AdminChatRepository()


@pytest.fixture
def user_id(test_db):
    """Erstellt einen Test-User und gibt seine ID zurueck."""
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (f"test_cw_{uuid.uuid4().hex[:8]}", "hash"),
    )
    test_db.commit()
    return cursor.fetchone()["id"]


@pytest.fixture
def chat_id(test_db, chat_repo, user_id):
    """Erstellt einen Test-Chat und gibt seine ID zurueck."""
    cid = chat_repo.create(conn=test_db, user_id=user_id)
    test_db.commit()
    return cid


@pytest.fixture
def action_id():
    return str(uuid.uuid4())


def _make_func(
    name="test.func",
    requires_confirmation=False,
    confirm_template=None,
    required=None,
    properties=None,
    changelog_area=None,
    changelog_action=None,
    pk_param=None,
):
    """Erstellt eine AdminFunction fuer Tests."""
    if properties is None:
        properties = {"item_id": {"type": "integer", "description": "Test-ID"}}
    if required is None:
        required = list(properties.keys())

    async def _fn(user, **kwargs):
        return AdminResult.ok(f"OK: {kwargs}")

    return AdminFunction(
        name=name,
        category="test",
        description=f"Test-Funktion {name}",
        help="Hilfe",
        examples=[],
        parameters={"type": "object", "properties": properties, "required": required},
        returns="Test",
        requires_confirmation=requires_confirmation,
        confirm_template=confirm_template,
        pk_param=pk_param,
        changelog_area=changelog_area,
        changelog_action=changelog_action,
        http_method=None,
        http_path=None,
        fn=_fn,
    )


# =====================================================================
# Confirmation Repository Tests
# =====================================================================


class TestConfirmationRepoCreateAndGetPending:
    def test_create_and_get_pending(self, test_db, confirm_repo, chat_id, user_id, action_id):
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=action_id,
            function_name="test.delete",
            params='{"id": 42}',
            confirm_string="test 42 loeschen",
            user_id=user_id,
        )
        test_db.commit()
        pending = confirm_repo.get_pending(chat_id)
        assert pending is not None
        assert pending["action_id"] == action_id
        assert pending["status"] == "pending"
        assert pending["confirm_string"] == "test 42 loeschen"


class TestConfirmationRepoConfirm:
    def test_confirm_changes_status(self, test_db, confirm_repo, chat_id, user_id, action_id):
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=action_id,
            function_name="test.delete",
            params="{}",
            confirm_string="confirm",
            user_id=user_id,
        )
        test_db.commit()
        result = confirm_repo.confirm(action_id)
        assert result is True
        assert confirm_repo.get_pending(chat_id) is None


class TestConfirmationRepoDismiss:
    def test_dismiss_changes_status(self, test_db, confirm_repo, chat_id, user_id, action_id):
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=action_id,
            function_name="test.delete",
            params="{}",
            confirm_string="confirm",
            user_id=user_id,
        )
        test_db.commit()
        result = confirm_repo.dismiss(action_id)
        assert result is True
        assert confirm_repo.get_pending(chat_id) is None


class TestConfirmationRepoDismissAll:
    def test_dismiss_all_multiple(self, test_db, confirm_repo, chat_id, user_id):
        for i in range(3):
            confirm_repo.create(
                conn=test_db,
                chat_id=chat_id,
                action_id=str(uuid.uuid4()),
                function_name="test.delete",
                params="{}",
                confirm_string=f"confirm_{i}",
                user_id=user_id,
            )
        test_db.commit()
        count = confirm_repo.dismiss_all(chat_id)
        assert count == 3
        assert confirm_repo.get_pending(chat_id) is None


# =====================================================================
# Confirmation Manager Tests
# =====================================================================


class TestConfirmationManagerMatch:
    def test_match_case_insensitive(self, test_db, confirm_repo, chat_id, user_id):
        """User-Input matcht confirm_string (case-insensitive)."""
        aid = str(uuid.uuid4())
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=aid,
            function_name="test.confirm",
            params='{"item_id": 42}',
            confirm_string="test 42 loeschen",
            user_id=user_id,
        )
        test_db.commit()

        # Dummy-Funktion in Registry registrieren
        from kiara.admin.registry import admin_registry

        func = _make_func(
            "test.confirm", requires_confirmation=True, confirm_template="test {item_id} loeschen"
        )
        admin_registry.register(func)

        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_id, "  Test 42 Loeschen  ", {"id": user_id})
        assert handled is True
        assert len(events) == 1
        assert events[0][0] == "confirmation_result"
        assert events[0][1]["confirmed"] is True
        assert events[0][1]["result"]["success"] is True


class TestConfirmationManagerMismatch:
    def test_mismatch_dismisses(self, test_db, confirm_repo, chat_id, user_id):
        """User-Input matcht nicht -> dismissed, handled=False."""
        aid = str(uuid.uuid4())
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=aid,
            function_name="test.delete",
            params="{}",
            confirm_string="test 42 loeschen",
            user_id=user_id,
        )
        test_db.commit()

        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_id, "etwas anderes", {"id": user_id})
        assert handled is False
        assert len(events) == 1
        assert events[0][0] == "action_dismissed"


class TestConfirmationManagerExpiry:
    def test_expired_confirmation_ignored(self, test_db, confirm_repo, chat_id, user_id):
        """Confirmation mit abgelaufenem expires_at wird ignoriert."""
        aid = str(uuid.uuid4())
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=aid,
            function_name="test.delete",
            params="{}",
            confirm_string="test",
            user_id=user_id,
        )
        test_db.commit()
        # expires_at in die Vergangenheit setzen
        test_db.execute(
            "UPDATE admin_chat_confirmations SET expires_at = NOW() - INTERVAL '1 minute' "
            "WHERE action_id = %s",
            (aid,),
        )
        test_db.commit()

        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_id, "test", {"id": user_id})
        assert handled is False
        assert events == []


# =====================================================================
# AdminFunctionTool Tests
# =====================================================================


class TestAdminFunctionToolConfirmationGate:
    def test_confirmation_gate_does_not_execute(self):
        """AdminFunctionTool gibt confirmation statt auszufuehren."""
        func = _make_func(
            "test.gate", requires_confirmation=True, confirm_template="item {item_id} loeschen"
        )
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"item_id": 99}, user_id=1))
        finally:
            loop.close()

        assert result.success is True
        assert "erfordert eine Bestaetigung" in result.display_text
        pending = tool.pop_pending()
        assert len(pending) >= 1
        conf = [a for a in pending if a["type"] == "confirmation"]
        assert len(conf) == 1
        assert conf[0]["confirm_string"] == "item 99 loeschen"


class TestAdminFunctionToolDirectExecution:
    def test_direct_execution_no_confirmation(self):
        """Funktion ohne requires_confirmation wird direkt ausgefuehrt."""
        func = _make_func("test.direct", requires_confirmation=False)
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"item_id": 5}, user_id=1))
        finally:
            loop.close()

        assert result.success is True
        assert "OK" in result.display_text
        pending = tool.pop_pending()
        # Sollte wizard_complete haben (Datenhygiene)
        types = [a["type"] for a in pending]
        assert "confirmation" not in types


class TestAdminFunctionToolWizardGate:
    def test_missing_required_fields_trigger_wizard(self):
        """AdminFunctionTool bei fehlenden Pflichtfeldern: wizard-Daten im Seitenkanal."""
        func = _make_func(
            "test.wiz",
            properties={
                "name": {"type": "string", "description": "Name"},
                "url": {"type": "string", "description": "URL"},
                "extra": {"type": "string", "description": "Optional"},
            },
            required=["name", "url"],
        )
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"name": "test"}, user_id=1))
        finally:
            loop.close()

        assert result.success is True
        assert "Fehlende Pflichtfelder" in result.display_text
        pending = tool.pop_pending()
        wizards = [a for a in pending if a["type"] == "wizard"]
        assert len(wizards) == 1
        assert wizards[0]["collected"] == {"name": "test"}
        assert "url" in wizards[0]["missing_required"]
        assert "extra" in wizards[0]["optional"]


# =====================================================================
# Wizard Repository Tests
# =====================================================================


class TestWizardRepoCreateAndGetActive:
    def test_create_and_get_active(self, test_db, wizard_repo, chat_id, user_id):
        wid = wizard_repo.create(
            conn=test_db,
            chat_id=chat_id,
            user_id=user_id,
            function_name="test.create",
            collected_params='{"type": "webdav"}',
        )
        test_db.commit()
        active = wizard_repo.get_active(chat_id)
        assert active is not None
        assert active["id"] == wid
        assert active["status"] == "collecting"
        assert json.loads(active["collected_params"]) == {"type": "webdav"}


class TestWizardRepoUpdateParams:
    def test_update_params_merges(self, test_db, wizard_repo, chat_id, user_id):
        wid = wizard_repo.create(
            conn=test_db,
            chat_id=chat_id,
            user_id=user_id,
            function_name="test.create",
            collected_params='{"type": "webdav"}',
        )
        test_db.commit()
        wizard_repo.update_params(wid, {"url": "https://example.com", "type": "webdav"})
        row = wizard_repo.get_by_id(wid)
        params = json.loads(row["collected_params"])
        assert params == {"type": "webdav", "url": "https://example.com"}


class TestWizardRepoComplete:
    def test_complete_changes_status(self, test_db, wizard_repo, chat_id, user_id):
        wid = wizard_repo.create(
            conn=test_db,
            chat_id=chat_id,
            user_id=user_id,
            function_name="test.create",
            collected_params="{}",
        )
        test_db.commit()
        wizard_repo.complete(wid)
        row = wizard_repo.get_by_id(wid)
        assert row["status"] == "executed"
        assert wizard_repo.get_active(chat_id) is None


class TestWizardRepoCancel:
    def test_cancel_changes_status(self, test_db, wizard_repo, chat_id, user_id):
        wid = wizard_repo.create(
            conn=test_db,
            chat_id=chat_id,
            user_id=user_id,
            function_name="test.create",
            collected_params="{}",
        )
        test_db.commit()
        wizard_repo.cancel(wid)
        row = wizard_repo.get_by_id(wid)
        assert row["status"] == "cancelled"


class TestWizardRepoUniqueActive:
    def test_only_one_active_per_chat(self, test_db, wizard_repo, chat_id, user_id):
        """UNIQUE Index: Nur ein aktiver Wizard pro Chat."""
        wizard_repo.create(
            conn=test_db,
            chat_id=chat_id,
            user_id=user_id,
            function_name="test.create",
            collected_params="{}",
        )
        test_db.commit()
        with pytest.raises(Exception):
            wizard_repo.create(
                conn=test_db,
                chat_id=chat_id,
                user_id=user_id,
                function_name="test.other",
                collected_params="{}",
            )
            test_db.commit()


# =====================================================================
# Wizard Manager Tests
# =====================================================================


class TestWizardManagerCreateNew:
    def test_creates_new_wizard(self, test_db, chat_id, user_id):
        mgr = WizardManager()
        result = mgr.handle_wizard_result(
            chat_id,
            user_id,
            {
                "function_name": "test.create",
                "collected": {"type": "webdav"},
                "missing_required": ["name", "url"],
                "optional": ["enabled"],
                "parameter_help": {},
            },
        )
        assert result["wizard_id"] is not None
        assert result["function_name"] == "test.create"
        assert result["collected"] == {"type": "webdav"}
        assert "name" in result["missing_required"]


class TestWizardManagerUpdateExisting:
    def test_updates_existing_wizard(self, test_db, chat_id, user_id):
        mgr = WizardManager()
        r1 = mgr.handle_wizard_result(
            chat_id,
            user_id,
            {
                "function_name": "test.create",
                "collected": {"type": "webdav"},
                "missing_required": ["name", "url"],
                "optional": [],
                "parameter_help": {},
            },
        )
        r2 = mgr.handle_wizard_result(
            chat_id,
            user_id,
            {
                "function_name": "test.create",
                "collected": {"type": "webdav", "url": "https://x.com"},
                "missing_required": ["name"],
                "optional": [],
                "parameter_help": {},
            },
        )
        assert r2["wizard_id"] == r1["wizard_id"]
        assert r2["collected"] == {"type": "webdav", "url": "https://x.com"}


class TestWizardManagerReplaceDifferentFunction:
    def test_replaces_old_wizard(self, test_db, chat_id, user_id):
        mgr = WizardManager()
        r1 = mgr.handle_wizard_result(
            chat_id,
            user_id,
            {
                "function_name": "test.create",
                "collected": {"type": "webdav"},
                "missing_required": ["name"],
                "optional": [],
                "parameter_help": {},
            },
        )
        r2 = mgr.handle_wizard_result(
            chat_id,
            user_id,
            {
                "function_name": "test.other",
                "collected": {"mode": "full"},
                "missing_required": ["target"],
                "optional": [],
                "parameter_help": {},
            },
        )
        assert r2["wizard_id"] != r1["wizard_id"]
        assert r2["function_name"] == "test.other"
        # Alter Wizard sollte cancelled sein
        repo = AdminWizardRepository()
        old = repo.get_by_id(r1["wizard_id"])
        assert old["status"] == "cancelled"


# =====================================================================
# Integration Tests
# =====================================================================

# =====================================================================
# Edge-Case & Fehlerbehandlungs-Tests
# =====================================================================


class TestConfirmationRepoConfirmInvalidActionId:
    def test_confirm_nonexistent_returns_false(self, test_db, confirm_repo):
        """confirm() bei nicht-existierender action_id gibt False zurueck."""
        result = confirm_repo.confirm("nonexistent-uuid")
        assert result is False

    def test_dismiss_nonexistent_returns_false(self, test_db, confirm_repo):
        """dismiss() bei nicht-existierender action_id gibt False zurueck."""
        result = confirm_repo.dismiss("nonexistent-uuid")
        assert result is False


class TestConfirmationManagerNoPending:
    def test_no_pending_returns_unhandled(self, test_db, chat_id):
        """check_and_handle bei leerem Chat gibt (False, []) zurueck."""
        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_id, "anything", {"id": 1})
        assert handled is False
        assert events == []


class TestConfirmationManagerFunctionRemoved:
    def test_confirmed_but_function_gone(self, test_db, confirm_repo, chat_id, user_id):
        """Funktion nicht mehr in Registry nach Bestaetigung → Fehler-Event."""
        aid = str(uuid.uuid4())
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=aid,
            function_name="test.removed_function",
            params='{"x": 1}',
            confirm_string="remove",
            user_id=user_id,
        )
        test_db.commit()

        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_id, "remove", {"id": user_id})
        assert handled is True
        assert events[0][0] == "confirmation_result"
        assert events[0][1]["result"]["success"] is False
        assert "nicht mehr verfuegbar" in events[0][1]["result"]["message"]


class TestAdminFunctionToolConfirmTemplateKeyError:
    def test_keyerror_fallback_to_func_name(self):
        """confirm_template referenziert Parameter der nicht in params ist → Fallback auf func.name."""
        func = _make_func(
            "test.keyerr", requires_confirmation=True, confirm_template="delete {missing_param}"
        )
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"item_id": 1}, user_id=1))
        finally:
            loop.close()

        pending = tool.pop_pending()
        conf = next(a for a in pending if a["type"] == "confirmation")
        assert conf["confirm_string"] == "test.keyerr"


class TestAdminFunctionToolEmptyConfirmString:
    def test_empty_template_result_fallback(self):
        """confirm_template evaluiert zu leerem String → Fallback auf func.name."""
        func = _make_func(
            "test.empty_tpl",
            requires_confirmation=True,
            confirm_template="",
            properties={"item_id": {"type": "integer", "description": "ID"}},
        )
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(tool.execute({"item_id": 1}, user_id=1))
        finally:
            loop.close()

        pending = tool.pop_pending()
        conf = next(a for a in pending if a["type"] == "confirmation")
        assert conf["confirm_string"] == "test.empty_tpl"


class TestAdminFunctionToolExecutionException:
    def test_fn_exception_returns_error_result(self):
        """fn wirft Exception → ToolResult mit success=False und Fehlermeldung."""

        async def _failing_fn(user, item_id):
            raise RuntimeError("DB-Verbindung verloren")

        func = AdminFunction(
            name="test.failing",
            category="test",
            description="Failing",
            help="",
            examples=[],
            returns="",
            parameters={
                "type": "object",
                "properties": {"item_id": {"type": "integer", "description": "ID"}},
                "required": ["item_id"],
            },
            requires_confirmation=False,
            confirm_template=None,
            pk_param=None,
            changelog_area=None,
            changelog_action=None,
            http_method=None,
            http_path=None,
            fn=_failing_fn,
        )
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"item_id": 1}, user_id=1))
        finally:
            loop.close()

        assert result.success is False
        assert "DB-Verbindung verloren" in result.display_text


class TestAdminFunctionToolDefinition:
    def test_definition_maps_fields(self):
        """definition() generiert ToolDefinition korrekt aus AdminFunction."""
        func = _make_func("test.def", requires_confirmation=True)
        tool = AdminFunctionTool(func)
        defn = tool.definition()
        assert defn.name == "test.def"
        assert defn.requires_confirmation is True
        assert defn.category == "test"
        assert "properties" in defn.parameters


class TestWizardManagerCancelActive:
    def test_cancel_active_cancels_wizard(self, test_db, chat_id, user_id):
        """cancel_active() bricht den aktiven Wizard ab."""
        mgr = WizardManager()
        r = mgr.handle_wizard_result(
            chat_id,
            user_id,
            {
                "function_name": "test.cancel",
                "collected": {"a": 1},
                "missing_required": ["b"],
                "optional": [],
                "parameter_help": {},
            },
        )
        mgr.cancel_active(chat_id)
        repo = AdminWizardRepository()
        row = repo.get_by_id(r["wizard_id"])
        assert row["status"] == "cancelled"
        assert repo.get_active(chat_id) is None

    def test_cancel_active_on_empty_chat(self, test_db, chat_id):
        """cancel_active() bei keinem aktiven Wizard ist ein No-Op."""
        mgr = WizardManager()
        mgr.cancel_active(chat_id)  # Sollte keinen Fehler werfen


# =====================================================================
# Integration Tests (urspruenglich)
# =====================================================================


class TestAdminFunctionToolPopPending:
    def test_pop_pending_returns_and_clears(self):
        func = _make_func("test.pop", requires_confirmation=True, confirm_template="test {item_id}")
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(tool.execute({"item_id": 1}, user_id=1))
        finally:
            loop.close()

        actions = tool.pop_pending()
        assert len(actions) >= 1
        # Zweiter Aufruf muss leer sein
        assert tool.pop_pending() == []


class TestAdminFunctionToolSidebandAfterExecute:
    def test_sideband_confirmation_after_execute(self):
        """execute() bei requires_confirmation fuellt _pending, pop_pending() liefert es."""
        func = _make_func("test.sb", requires_confirmation=True, confirm_template="sb {item_id}")
        tool = AdminFunctionTool(func)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"item_id": 7}, user_id=1))
        finally:
            loop.close()

        assert result.success is True
        pending = tool.pop_pending()
        types = [a["type"] for a in pending]
        assert "confirmation" in types
        conf = next(a for a in pending if a["type"] == "confirmation")
        assert conf["confirm_string"] == "sb 7"
        assert conf["params"] == {"item_id": 7}


class TestConfirmationFlowE2E:
    def test_complete_confirmation_flow(self, test_db, confirm_repo, chat_id, user_id):
        """Kompletter Flow: Tool-Call -> Confirmation -> Bestaetigung -> Ausfuehrung."""
        # 1. AdminFunction erstellen und registrieren
        from kiara.admin.registry import admin_registry

        executed = {}

        async def _fn(user, item_id):
            executed["item_id"] = item_id
            return AdminResult.ok(f"Geloescht: {item_id}")

        func = AdminFunction(
            name="test.e2e_confirm",
            category="test",
            description="E2E-Test",
            help="",
            examples=[],
            parameters={
                "type": "object",
                "properties": {"item_id": {"type": "integer", "description": "ID"}},
                "required": ["item_id"],
            },
            returns="OK",
            requires_confirmation=True,
            confirm_template="e2e {item_id} loeschen",
            pk_param="item_id",
            changelog_area=None,
            changelog_action=None,
            http_method=None,
            http_path=None,
            fn=_fn,
        )
        admin_registry.register(func)

        # 2. Tool erstellen, ausfuehren -> Seitenkanal hat Confirmation
        tool = AdminFunctionTool(func)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(tool.execute({"item_id": 42}, user_id=user_id))
        finally:
            loop.close()
        assert "erfordert eine Bestaetigung" in result.display_text

        # 3. Seitenkanal auslesen und in DB speichern
        pending_actions = tool.pop_pending()
        conf_action = next(a for a in pending_actions if a["type"] == "confirmation")
        confirm_repo.create(
            conn=test_db,
            chat_id=chat_id,
            action_id=conf_action["action_id"],
            function_name=conf_action["function_name"],
            params=json.dumps(conf_action["params"]),
            confirm_string=conf_action["confirm_string"],
            user_id=user_id,
        )
        test_db.commit()

        # 4. User bestaetigt
        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_id, "e2e 42 loeschen", {"id": user_id})
        assert handled is True
        assert events[0][0] == "confirmation_result"
        assert events[0][1]["result"]["success"] is True
        assert events[0][1]["result"]["message"] == "Geloescht: 42"
        assert executed["item_id"] == 42


class TestWizardFlowE2E:
    def test_complete_wizard_flow(self, test_db, chat_id, user_id):
        """Kompletter Flow: Tool-Call -> Wizard -> Schrittweise -> Ausfuehrung."""
        func = _make_func(
            "test.e2e_wizard",
            properties={
                "name": {"type": "string", "description": "Name"},
                "url": {"type": "string", "description": "URL"},
                "username": {"type": "string", "description": "Benutzer"},
            },
            required=["name", "url", "username"],
        )
        tool = AdminFunctionTool(func)
        mgr = WizardManager()

        # Schritt 1: Nur name angegeben -> Wizard
        loop = asyncio.new_event_loop()
        try:
            r1 = loop.run_until_complete(tool.execute({"name": "TestDS"}, user_id=user_id))
        finally:
            loop.close()
        assert "Fehlende Pflichtfelder" in r1.display_text
        p1 = tool.pop_pending()
        wizard_action = next(a for a in p1 if a["type"] == "wizard")
        w1 = mgr.handle_wizard_result(chat_id, user_id, wizard_action)
        assert len(w1["missing_required"]) == 2

        # Schritt 2: url und user angegeben -> Ausfuehrung
        loop = asyncio.new_event_loop()
        try:
            r2 = loop.run_until_complete(
                tool.execute(
                    {"name": "TestDS", "url": "https://x.com", "username": "admin"}, user_id=user_id
                )
            )
        finally:
            loop.close()
        assert r2.success is True
        assert "OK" in r2.display_text


class TestWizardThenConfirmationCombined:
    def test_wizard_then_confirmation_flow(self, test_db, confirm_repo, chat_id, user_id):
        """Kombinierter Flow: Funktion mit requires_confirmation=True UND fehlenden
        Pflichtfeldern. Erst Wizard-Gate, dann Confirmation-Gate."""
        func = _make_func(
            "test.e2e_wc",
            requires_confirmation=True,
            confirm_template="{item_id} loeschen",
            properties={
                "item_id": {"type": "integer", "description": "ID"},
                "scope": {"type": "string", "description": "Bereich"},
            },
            required=["item_id", "scope"],
        )
        tool = AdminFunctionTool(func)

        # Schritt 1: Nur item_id -> Wizard-Gate (fehlende Felder)
        loop = asyncio.new_event_loop()
        try:
            r1 = loop.run_until_complete(tool.execute({"item_id": 10}, user_id=user_id))
        finally:
            loop.close()
        assert "Fehlende Pflichtfelder" in r1.display_text
        p1 = tool.pop_pending()
        types1 = [a["type"] for a in p1]
        assert "wizard" in types1
        assert "confirmation" not in types1

        # Schritt 2: Alle Felder -> Confirmation-Gate
        loop = asyncio.new_event_loop()
        try:
            r2 = loop.run_until_complete(
                tool.execute({"item_id": 10, "scope": "all"}, user_id=user_id)
            )
        finally:
            loop.close()
        assert "erfordert eine Bestaetigung" in r2.display_text
        p2 = tool.pop_pending()
        types2 = [a["type"] for a in p2]
        assert "confirmation" in types2
        conf = next(a for a in p2 if a["type"] == "confirmation")
        assert conf["confirm_string"] == "10 loeschen"
