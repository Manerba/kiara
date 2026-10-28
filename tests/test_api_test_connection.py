"""Tests fuer POST /api/admin/datasources/test-connection.

Testet den Endpoint fuer ad-hoc Verbindungstests aus dem Create- und
Edit-Dialog: Typ-/Config-Validierung, PASSWORD_MASK-Aufloesung,
safe_test_connection-Delegation, synthetisches row-Dict.
"""

from unittest.mock import MagicMock, patch

import pytest

# test_datasource_connection wurde zu admin.functions.datasources migriert.
# Tests fuer die neue Funktion sind in test_admin_datasource_crud.py.
pytest.skip(
    "test_datasource_connection migriert zu admin.functions.datasources",
    allow_module_level=True,
)

from fastapi import HTTPException

from kiara.web.api_datasource import (
    ConnectorActionRequest,
)
from kiara.web.api_datasource import (
    test_datasource_connection as _test_connection_endpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_body(type="imap", config=None, ds_id=None):
    """Erzeugt ein ConnectorActionRequest-Objekt."""
    return ConnectorActionRequest(
        type=type,
        config=config if config is not None else {},
        ds_id=ds_id,
    )


def _make_connector_cls(
    secret_fields=None,
    test_result=None,
):
    """Erzeugt eine Mock-Connector-Klasse mit konfigurierbaren Rueckgaben."""
    cls = MagicMock()
    cls.name = "mock"
    cls.secret_fields.return_value = secret_fields or []
    cls.safe_test_connection.return_value = test_result or {
        "success": True,
        "message": "OK",
    }
    return cls


MOCK_USER = {"username": "admin", "is_admin": True}


# ---------------------------------------------------------------------------
# 1. Typ-Validierung
# ---------------------------------------------------------------------------


class TestTypeValidation:
    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector", return_value=None)
    def test_unknown_type_returns_400(self, mock_get, mock_load):
        """Unbekannter Konnektor-Typ -> HTTPException 400."""
        body = _make_body(type="ftp", config={})
        with pytest.raises(HTTPException) as exc_info:
            _test_connection_endpoint(body, user=MOCK_USER)
        assert exc_info.value.status_code == 400
        assert "ftp" in str(exc_info.value.detail)

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector", return_value=None)
    def test_empty_type_returns_400(self, mock_get, mock_load):
        """Leerer Typ -> HTTPException 400."""
        body = _make_body(type="", config={})
        with pytest.raises(HTTPException) as exc_info:
            _test_connection_endpoint(body, user=MOCK_USER)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# 2. Config-Validierung (Pydantic faengt invalid configs ab, diese Tests
#    pruefen den Defense-in-depth isinstance-Check im Endpoint)
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_config_not_dict_rejected_by_pydantic(self):
        """config ist kein Dict -> Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConnectorActionRequest(type="imap", config="not-a-dict")

    def test_config_none_rejected_by_pydantic(self):
        """config ist None -> Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConnectorActionRequest(type="imap", config=None)

    def test_config_missing_rejected_by_pydantic(self):
        """config fehlt -> Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConnectorActionRequest(type="imap")


# ---------------------------------------------------------------------------
# 3. Happy Path — Create-Dialog (kein ds_id)
# ---------------------------------------------------------------------------


class TestCreateDialogHappyPath:
    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_create_dialog_success(self, mock_get, mock_load):
        """Create-Dialog: Config direkt testen, kein DB-Zugriff."""
        connector_cls = _make_connector_cls(
            secret_fields=["password"],
            test_result={"success": True, "message": "Verbunden"},
        )
        mock_get.return_value = connector_cls

        config = {"server": "mail.de", "password": "klartext123"}
        body = _make_body(type="imap", config=config)

        result = _test_connection_endpoint(body, user=MOCK_USER)

        assert result["success"] is True
        assert result["message"] == "Verbunden"
        connector_cls.safe_test_connection.assert_called_once()
        # Klartext-Passwort wird direkt durchgereicht
        call_config = connector_cls.safe_test_connection.call_args[0][0]
        assert call_config["password"] == "klartext123"

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_create_dialog_no_secret_fields(self, mock_get, mock_load):
        """Konnektor ohne Secret-Felder (z.B. local): kein PASSWORD_MASK-Check."""
        connector_cls = _make_connector_cls(
            secret_fields=[],
            test_result={"success": True, "message": "Pfad erreichbar"},
        )
        mock_get.return_value = connector_cls

        body = _make_body(type="local", config={"path": "/data"})
        result = _test_connection_endpoint(body, user=MOCK_USER)
        assert result["success"] is True

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_create_dialog_connection_failure(self, mock_get, mock_load):
        """Create-Dialog: Konnektor meldet Fehler -> HTTPException 502."""
        from fastapi.exceptions import HTTPException

        connector_cls = _make_connector_cls(
            test_result={"success": False, "message": "Connection refused"},
        )
        mock_get.return_value = connector_cls

        body = _make_body(type="imap", config={"server": "bad.host"})
        with pytest.raises(HTTPException) as exc_info:
            _test_connection_endpoint(body, user=MOCK_USER)
        assert exc_info.value.status_code == 502
        assert "refused" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 4. Happy Path — Edit-Dialog (mit ds_id, PASSWORD_MASK)
# ---------------------------------------------------------------------------


class TestEditDialogPasswordMask:
    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    @patch("kiara.web.api_datasource._get_connector_config")
    @patch("kiara.web.api_datasource._decrypt_config")
    def test_password_mask_resolved_from_db(
        self,
        mock_decrypt,
        mock_get_cc,
        mock_get,
        mock_load,
    ):
        """Edit-Dialog: PASSWORD_MASK wird gegen gespeichertes Passwort aufgeloest."""
        connector_cls = _make_connector_cls(
            secret_fields=["password"],
            test_result={"success": True, "message": "OK"},
        )
        mock_get.return_value = connector_cls

        # DB gibt eine gespeicherte Row zurueck
        stored_row = {"id": 42, "type": "imap", "connector_config": "{}"}

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = stored_row
            mock_get_cc.return_value = {"server": "mail.de", "password": "gAAAAABenc"}
            mock_decrypt.return_value = {"server": "mail.de", "password": "geheim123"}

            body = _make_body(
                type="imap",
                config={"server": "neuer-host.de", "password": "********"},
                ds_id=42,
            )
            result = _test_connection_endpoint(body, user=MOCK_USER)

        assert result["success"] is True
        # safe_test_connection wurde mit aufgeloestem Passwort aufgerufen
        call_config = connector_cls.safe_test_connection.call_args[0][0]
        assert call_config["password"] == "geheim123"
        assert call_config["server"] == "neuer-host.de"

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    @patch("kiara.web.api_datasource._get_connector_config")
    @patch("kiara.web.api_datasource._decrypt_config")
    def test_new_password_in_edit_dialog_not_resolved(
        self,
        mock_decrypt,
        mock_get_cc,
        mock_get,
        mock_load,
    ):
        """Edit-Dialog: Neues Klartext-Passwort wird direkt verwendet (kein DB-Lookup)."""
        connector_cls = _make_connector_cls(
            secret_fields=["password"],
            test_result={"success": True, "message": "OK"},
        )
        mock_get.return_value = connector_cls

        # ds_id vorhanden, aber Passwort ist KEIN PASSWORD_MASK
        body = _make_body(
            type="imap",
            config={"server": "mail.de", "password": "neuesPasswort"},
            ds_id=42,
        )

        # DB wird trotzdem geladen (weil ds_id vorhanden), aber password wird nicht aufgeloest
        stored_row = {"id": 42, "type": "imap", "connector_config": "{}"}

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = stored_row
            mock_get_cc.return_value = {"password": "gAAAAABenc"}
            mock_decrypt.return_value = {"password": "altesPasswort"}

            result = _test_connection_endpoint(body, user=MOCK_USER)

        call_config = connector_cls.safe_test_connection.call_args[0][0]
        assert call_config["password"] == "neuesPasswort"


# ---------------------------------------------------------------------------
# 5. PASSWORD_MASK-Fehlerfaelle
# ---------------------------------------------------------------------------


class TestPasswordMaskErrors:
    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_password_mask_without_ds_id_returns_400(self, mock_get, mock_load):
        """Create-Dialog mit PASSWORD_MASK (sollte nicht passieren) -> 400."""
        connector_cls = _make_connector_cls(secret_fields=["password"])
        mock_get.return_value = connector_cls

        body = _make_body(
            type="imap",
            config={"server": "mail.de", "password": "********"},
        )
        with pytest.raises(HTTPException) as exc_info:
            _test_connection_endpoint(body, user=MOCK_USER)
        assert exc_info.value.status_code == 400
        assert "password" in str(exc_info.value.detail)

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_password_mask_ds_not_found_returns_400(
        self,
        mock_get,
        mock_load,
    ):
        """PASSWORD_MASK mit ds_id, aber DS existiert nicht mehr -> 400."""
        connector_cls = _make_connector_cls(secret_fields=["password"])
        mock_get.return_value = connector_cls

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = None
            body = _make_body(
                type="imap",
                config={"server": "mail.de", "password": "********"},
                ds_id=999,
            )
            with pytest.raises(HTTPException) as exc_info:
                _test_connection_endpoint(body, user=MOCK_USER)
        assert exc_info.value.status_code == 400
        assert "password" in str(exc_info.value.detail)

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    @patch("kiara.web.api_datasource._get_connector_config")
    @patch("kiara.web.api_datasource._decrypt_config")
    def test_password_mask_stored_field_empty_returns_400(
        self,
        mock_decrypt,
        mock_get_cc,
        mock_get,
        mock_load,
    ):
        """PASSWORD_MASK, aber gespeichertes Feld ist leer -> 400."""
        connector_cls = _make_connector_cls(secret_fields=["password"])
        mock_get.return_value = connector_cls

        stored_row = {"id": 42, "type": "imap", "connector_config": "{}"}

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = stored_row
            mock_get_cc.return_value = {"password": ""}
            mock_decrypt.return_value = {"password": ""}

            body = _make_body(
                type="imap",
                config={"server": "mail.de", "password": "********"},
                ds_id=42,
            )
            with pytest.raises(HTTPException) as exc_info:
                _test_connection_endpoint(body, user=MOCK_USER)
        assert exc_info.value.status_code == 400
        assert "leer" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# 6. Synthetisches row-Dict
# ---------------------------------------------------------------------------


class TestSyntheticRow:
    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_row_id_is_neu_for_create_dialog(self, mock_get, mock_load):
        """Create-Dialog: row['id'] == 'neu' (nicht None)."""
        connector_cls = _make_connector_cls()
        mock_get.return_value = connector_cls

        body = _make_body(type="imap", config={"server": "mail.de"})
        _test_connection_endpoint(body, user=MOCK_USER)

        call_row = connector_cls.safe_test_connection.call_args[0][1]
        assert call_row["id"] == "neu"
        assert call_row["type"] == "imap"
        assert call_row["name"] == "(Verbindungstest)"

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_row_id_is_ds_id_for_edit_dialog(self, mock_get, mock_load):
        """Edit-Dialog ohne PASSWORD_MASK: row['id'] == ds_id."""
        connector_cls = _make_connector_cls()
        mock_get.return_value = connector_cls

        # ds_id angegeben, aber kein PASSWORD_MASK -> kein DB-Zugriff noetig
        body = _make_body(
            type="imap",
            config={"server": "mail.de"},
            ds_id=42,
        )

        with patch("kiara.web.api_datasource.ds_repo") as mock_ds_repo:
            mock_ds_repo.get_by_id.return_value = None
            with patch("kiara.web.api_datasource._decrypt_config"):
                _test_connection_endpoint(body, user=MOCK_USER)

        call_row = connector_cls.safe_test_connection.call_args[0][1]
        assert call_row["id"] == 42


# ---------------------------------------------------------------------------
# 7. safe_test_connection-Delegation
# ---------------------------------------------------------------------------


class TestSafeTestConnectionDelegation:
    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_safe_test_connection_called_not_test_connection(
        self,
        mock_get,
        mock_load,
    ):
        """Endpoint ruft safe_test_connection auf, NICHT test_connection."""
        connector_cls = _make_connector_cls()
        mock_get.return_value = connector_cls

        body = _make_body(type="imap", config={"server": "x"})
        _test_connection_endpoint(body, user=MOCK_USER)

        connector_cls.safe_test_connection.assert_called_once()
        connector_cls.test_connection.assert_not_called()

    @patch("kiara.web.api_datasource.ensure_loaded")
    @patch("kiara.web.api_datasource.get_connector")
    def test_ensure_loaded_called_before_get_connector(
        self,
        mock_get,
        mock_load,
    ):
        """ensure_loaded() muss vor get_connector() aufgerufen werden."""
        mock_get.return_value = None  # Typ unbekannt -> 400

        body = _make_body(type="x", config={})
        with pytest.raises(HTTPException):
            _test_connection_endpoint(body, user=MOCK_USER)

        mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# 8. Endpoint ist sync (def, nicht async def)
# ---------------------------------------------------------------------------


class TestEndpointSync:
    def test_test_connection_is_sync(self):
        """test_datasource_connection ist eine sync-Funktion (def, nicht async def)."""
        import asyncio

        assert not asyncio.iscoroutinefunction(_test_connection_endpoint)

    def test_list_folders_is_sync(self):
        """list_datasource_folders ist eine sync-Funktion."""
        import asyncio

        from kiara.web.api_datasource import list_datasource_folders

        assert not asyncio.iscoroutinefunction(list_datasource_folders)

    def test_pydantic_body_validation(self):
        """Fehlender type-Key -> Pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConnectorActionRequest(config={})
