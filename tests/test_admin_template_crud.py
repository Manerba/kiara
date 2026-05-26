"""Tests fuer Sprint 06+16.1: Template-CRUD + Auto-Router-Integration.

Testet ki.templates.create/update/delete + User-Chat-Guards,
Transaction-Safety, History-Integration, Delete-Guard — 25+ Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


def _mock_tx_context(row_data=None, history_check=None):
    """Erstellt einen Mock fuer db_transaction() mit SELECT FOR UPDATE.

    row_data: dict fuer die Template-Row (None = nicht gefunden)
    history_check: Ergebnis des History-Checks (None = keine History)
    """
    mock_tx = MagicMock()
    mock_cursor = MagicMock()

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        result = MagicMock()
        if call_count[0] == 1:
            # Erster Call: SELECT ... FOR UPDATE
            result.fetchone.return_value = row_data
        elif call_count[0] == 2 and history_check is not None:
            # Zweiter Call: History-Check (nur bei Delete)
            result.fetchone.return_value = history_check
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    mock_tx.execute = MagicMock(side_effect=side_effect)
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_tx)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_tx


# ---------------------------------------------------------------------------
# ki.templates.create
# ---------------------------------------------------------------------------


class TestKiTemplatesCreate:
    @patch("kiara.shared.db.db_query")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.create")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.get_by_name")
    @patch("kiara.web.api_ki._validate_template_content")
    def test_create_template(self, mock_validate, mock_get_name, mock_create, mock_db):
        """Test 46: validate chain_type + create."""
        mock_get_name.return_value = None
        mock_create.return_value = 10
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="test-template",
                content="Kontext: {context}\nFrage: {question}",
                chain_type="standard",
            )
        )

        assert result.success is True
        assert result.data["id"] == 10
        mock_validate.assert_called_once()

    def test_create_template_invalid_chain(self):
        """Test 47: ungueltiger chain_type → fail."""
        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="test",
                content="test",
                chain_type="invalid",
            )
        )
        assert result.success is False
        assert "chain_type" in result.message

    @patch("kiara.web.api_ki._validate_template_content")
    def test_create_template_invalid_content(self, mock_validate):
        """Test 48: standard ohne {context} → fail."""
        from fastapi import HTTPException

        mock_validate.side_effect = HTTPException(
            status_code=422,
            detail="Standard-Template muss die Platzhalter {context} und {question} enthalten",
        )

        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="test",
                content="kein kontext",
                chain_type="standard",
            )
        )
        assert result.success is False
        assert "{context}" in result.message

    @patch("kiara.web.api_ki._validate_template_content")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.get_by_name")
    def test_create_template_duplicate_name(self, mock_get_name, mock_validate):
        """Test 49: Name existiert → fail."""
        mock_get_name.return_value = {"id": 5, "name": "existing"}

        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="existing",
                content="test",
                chain_type="standard",
            )
        )
        assert result.success is False
        assert "existiert bereits" in result.message

    @patch("kiara.shared.db.db_query")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.create")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.get_by_name")
    @patch("kiara.web.api_ki._validate_template_content")
    def test_create_enrichment_template(self, mock_validate, mock_get_name, mock_create, mock_db):
        """Test 15: chain_type='enrichment' + show_in_chat=false → Erfolg."""
        mock_get_name.return_value = None
        mock_create.return_value = 20
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="test-enrichment",
                content="Test {chunk_text} template",
                chain_type="enrichment",
                show_in_chat=False,
            )
        )

        assert result.success is True
        assert result.data["id"] == 20

    def test_create_enrichment_show_in_chat_blocked(self):
        """Test 29: enrichment + show_in_chat=true → fail."""
        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="test",
                content="test {chunk_text}",
                chain_type="enrichment",
                show_in_chat=True,
            )
        )
        assert result.success is False
        assert "Chat-Modus" in result.message

    def test_create_cloud_show_in_chat_blocked(self):
        """Test 29b: is_cloud=true + show_in_chat=true → fail."""
        from kiara.admin.functions.ki import ki_templates_create

        result = _run(
            ki_templates_create(
                user=MOCK_USER,
                name="test",
                content="test {context} {question}",
                chain_type="standard",
                is_cloud=True,
                show_in_chat=True,
            )
        )
        assert result.success is False
        assert "Chat-Modus" in result.message


# ---------------------------------------------------------------------------
# ki.templates.update (Transaction-Safety + History)
# ---------------------------------------------------------------------------


class TestKiTemplatesUpdate:
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_template_partial(self, mock_tx_factory, mock_update):
        """Test 50: nur uebergebene Felder."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "standard",
            "is_system": False,
            "content": "old",
            "description": "old desc",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, mock_tx = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                label="Neues Label",
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["label"] == "Neues Label"
        assert "content" not in call_kwargs

    @patch("kiara.shared.db.db_transaction")
    def test_update_template_system_chain_locked(self, mock_tx_factory):
        """Test 51: System-Template chain_type → fail."""
        row = {
            "id": 1,
            "name": "default",
            "label": "Standard",
            "chain_type": "standard",
            "is_system": True,
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=1,
                chain_type="agent",
            )
        )

        assert result.success is False
        assert "System-Templates" in result.message

    @patch("kiara.web.api_ki._validate_template_content")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_template_validate_content(
        self, mock_tx_factory, mock_update, mock_save_history, mock_validate
    ):
        """Test 52: Content gegen chain_type validiert."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "standard",
            "is_system": False,
            "content": "old",
            "cloud_system_prompt": "",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                content="Kontext: {context}\nFrage: {question}",
            )
        )

        assert result.success is True
        mock_validate.assert_called_once_with(
            "Kontext: {context}\nFrage: {question}",
            "standard",
        )

    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_content_creates_history(self, mock_tx_factory, mock_update, mock_save_history):
        """Test 9: Content-Aenderung → History wird geschrieben."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "unchained",
            "is_system": False,
            "content": "alter Content",
            "cloud_system_prompt": "sys prompt",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, mock_tx = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                content="neuer Content",
            )
        )

        assert result.success is True
        mock_save_history.assert_called_once_with(
            3,
            "alter Content",
            cloud_system_prompt="sys prompt",
            changed_by="admin",
            conn=mock_tx,
        )

    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_same_content_no_history(self, mock_tx_factory, mock_update, mock_save_history):
        """Test 10: Content unveraendert → keine History."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "unchained",
            "is_system": False,
            "content": "gleicher Content",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                content="gleicher Content",
            )
        )

        assert result.success is True
        mock_save_history.assert_not_called()

    @patch("kiara.shared.db.db_transaction")
    def test_update_invalid_chain_type(self, mock_tx_factory):
        """Test 18: ungueltiger chain_type → AdminResult.fail."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "standard",
            "is_system": False,
            "content": "c",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                chain_type="invalid_type",
            )
        )

        assert result.success is False
        assert "Ungueltiger chain_type" in result.message

    @patch("kiara.shared.db.db_transaction")
    def test_update_custom_reserved_chain_type(self, mock_tx_factory):
        """Custom-Template mit reserviertem chain_type (rewriter) → fail."""
        row = {
            "id": 3,
            "name": "custom",
            "label": "Custom",
            "chain_type": "standard",
            "is_system": False,
            "content": "c",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                chain_type="rewriter",
            )
        )

        assert result.success is False
        assert "reserviert" in result.message

    @patch("kiara.shared.db.db_transaction")
    def test_update_show_in_chat_enrichment_blocked(self, mock_tx_factory):
        """Test 30: show_in_chat=true bei enrichment → fail."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "enrichment",
            "is_system": False,
            "content": "{chunk_text}",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                show_in_chat=True,
            )
        )

        assert result.success is False
        assert "Chat-Modus" in result.message

    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_self_heal_misconfigured_visibility(self, mock_tx_factory, mock_update):
        """Test 31: fehlkonfiguriertes enrichment mit show_in_chat=1 → wird geheilt."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "enrichment",
            "is_system": False,
            "content": "{chunk_text}",
            "is_cloud": False,
            "show_in_chat": 1,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                label="Neues Label",
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["show_in_chat"] == 0

    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_for_update_reread(self, mock_tx_factory, mock_update, mock_save_history):
        """Test 26: SELECT ... FOR UPDATE wird vor Validation/History ausgefuehrt."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "unchained",
            "is_system": False,
            "content": "old",
            "cloud_system_prompt": "",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, mock_tx = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                content="new",
            )
        )

        # Erster execute-Aufruf muss FOR UPDATE sein
        first_call = mock_tx.execute.call_args_list[0]
        assert "FOR UPDATE" in first_call[0][0]

    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_history_and_update_same_transaction(
        self, mock_tx_factory, mock_update, mock_save_history
    ):
        """Test 25: save_history + update in derselben Transaction."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "unchained",
            "is_system": False,
            "content": "old",
            "cloud_system_prompt": "",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, mock_tx = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                content="new",
            )
        )

        # Beide mit conn=tx aufgerufen
        assert mock_save_history.call_args[1]["conn"] is mock_tx
        assert mock_update.call_args[1]["conn"] is mock_tx


# ---------------------------------------------------------------------------
# ki.templates.delete (Transaction + Guards)
# ---------------------------------------------------------------------------


class TestKiTemplatesDelete:
    @patch("kiara.repositories.prompt.PromptTemplateRepository.delete")
    @patch("kiara.shared.db.db_transaction")
    def test_delete_template(self, mock_tx_factory, mock_delete):
        """Test 53: delete aufgerufen."""
        row = {"id": 5, "name": "custom", "label": "Custom", "is_system": False}
        mock_cm, mock_tx = _mock_tx_context(row, history_check=None)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_delete

        result = _run(ki_templates_delete(user=MOCK_USER, template_id=5))

        assert result.success is True
        mock_delete.assert_called_once_with(5, conn=mock_tx)

    @patch("kiara.shared.db.db_transaction")
    def test_delete_template_system(self, mock_tx_factory):
        """Test 16/54: System-Template → fail."""
        row = {"id": 1, "name": "default", "label": "Standard", "is_system": True}
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_delete

        result = _run(ki_templates_delete(user=MOCK_USER, template_id=1))

        assert result.success is False
        assert "System-Templates" in result.message

    @patch("kiara.shared.db.db_transaction")
    def test_delete_template_with_history(self, mock_tx_factory):
        """Test 32: Template mit History → fail."""
        row = {"id": 5, "name": "custom", "label": "Custom", "is_system": False}
        mock_cm, _ = _mock_tx_context(row, history_check={"1": 1})
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_delete

        result = _run(ki_templates_delete(user=MOCK_USER, template_id=5))

        assert result.success is False
        assert "Versionierte" in result.message


# ---------------------------------------------------------------------------
# Ergaenzende Edge-Case Tests
# ---------------------------------------------------------------------------


class TestKiTemplatesEdgeCases:
    @patch("kiara.shared.db.db_transaction")
    def test_update_template_no_changes(self, mock_tx_factory):
        """Keine Felder uebergeben → 'Keine Aenderungen'."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "standard",
            "is_system": False,
            "content": "c",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(ki_templates_update(user=MOCK_USER, template_id=3))

        assert result.success is True
        assert "Keine Aenderungen" in result.message

    @patch("kiara.shared.db.db_transaction")
    def test_update_template_not_found(self, mock_tx_factory):
        """Template nicht gefunden → fail."""
        mock_cm, _ = _mock_tx_context(None)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=999,
                label="Test",
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.shared.db.db_transaction")
    def test_delete_template_not_found(self, mock_tx_factory):
        """Template nicht gefunden → fail."""
        mock_cm, _ = _mock_tx_context(None)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_delete

        result = _run(ki_templates_delete(user=MOCK_USER, template_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.web.api_ki._validate_template_content")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_update_system_template_other_fields_allowed(
        self, mock_tx_factory, mock_update, mock_validate
    ):
        """System-Template: andere Felder aenderbar — nur chain_type gesperrt."""
        row = {
            "id": 1,
            "name": "system_standard",
            "label": "Standard",
            "chain_type": "standard",
            "is_system": True,
            "content": "Du bist ein Assistent.\n{context}\n{question}",
            "is_cloud": False,
            "show_in_chat": 1,
        }
        mock_cm, _ = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=1,
                label="Neues Label",
                description="Neue Beschreibung",
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["label"] == "Neues Label"
        assert call_kwargs["description"] == "Neue Beschreibung"


# ---------------------------------------------------------------------------
# Auto-Router-Integration (Unit-Tests mit Registry)
# ---------------------------------------------------------------------------


class TestAutoRouterIntegration:
    def test_post_create_user_via_autorouter(self):
        """Test 55: POST /api/admin/users → users.create registriert."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("users.create")
        assert func is not None
        assert func.http_method == "POST"
        assert func.http_path == "/api/admin/users"
        assert func.raw_response is True

    def test_delete_group_via_autorouter(self):
        """Test 56: DELETE → raw_response Format."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("groups.delete")
        assert func is not None
        assert func.http_method == "DELETE"
        assert func.raw_response is True
        assert func.requires_confirmation is True

    def test_patch_toggle_via_autorouter(self):
        """Test 57: PATCH Body-Parsing fuer ki.backends.toggle."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.backends.toggle")
        assert func is not None
        assert func.http_method == "PATCH"
        assert func.http_path == "/api/admin/ki/backends/{backend_id}/toggle"

    def test_autorouter_changelog_source_gui(self):
        """Test 58: Changelog-Felder bei CRUD-Funktionen gesetzt."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()

        func = admin_registry.get("users.create")
        assert func.changelog_area == "users"
        assert func.changelog_action == "create"

        func = admin_registry.get("ki.backends.delete")
        assert func.changelog_area == "ki_backends"
        assert func.changelog_action == "delete"

        func = admin_registry.get("ki.config.update")
        assert func.changelog_area == "ki"
        assert func.changelog_action == "update"
        assert func.http_method is None
