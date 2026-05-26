from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_editor_open_returns_signed_docsapi_config(tmp_path):
    from kiara.web.api_onlyoffice import editor_router
    from kiara.web.auth import get_chat_user

    ds_dir = tmp_path / "datasource"
    ds_dir.mkdir()
    doc = ds_dir / "document.docx"
    doc.write_bytes(b"PK\x03\x04")

    app = FastAPI()
    app.include_router(editor_router)
    app.dependency_overrides[get_chat_user] = lambda: {
        "id": 7,
        "username": "alice",
        "datasource_ids": [1],
    }
    client = TestClient(app, raise_server_exceptions=False)

    with (
        patch(
            "kiara.web.api_onlyoffice.load_config",
            return_value={"datasource": {"path": str(ds_dir)}},
        ),
        patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo,
        patch("kiara.web.api_onlyoffice.wopi_repo") as mock_wopi_repo,
        patch("kiara.web.api_onlyoffice._check_datasource_acl"),
    ):
        mock_config_repo.get.return_value = {
            "document_server_url": "https://onlyoffice.example",
            "jwt_secret": "secret",
            "wopi_external_base_url": "https://kiara.example",
            "proof_validation_enabled": True,
        }
        resp = client.get("/api/onlyoffice/open", params={"file_path": "document.docx"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["document_server_url"] == "https://onlyoffice.example"
    config = data["config"]
    decoded = pyjwt.decode(config["token"], "secret", algorithms=["HS256"])
    assert decoded["document"]["title"] == "document.docx"
    assert decoded["document"]["url"].startswith("https://kiara.example/api/wopi/files/")
    assert decoded["editorConfig"]["callbackUrl"].startswith(
        "https://kiara.example/api/onlyoffice/callback?"
    )
    mock_wopi_repo.create_token.assert_called_once()


def test_document_server_url_rejects_credentials_and_query():
    from kiara.web.api_onlyoffice import _normalize_document_server_url

    with pytest.raises(ValueError):
        _normalize_document_server_url("https://user:pass@onlyoffice.example")
    with pytest.raises(ValueError):
        _normalize_document_server_url("https://onlyoffice.example/health?x=1")


def test_document_server_request_rejects_loopback_target():
    from kiara.web.api_onlyoffice import _assert_document_server_request_allowed

    with pytest.raises(ValueError):
        _assert_document_server_request_allowed("http://127.0.0.1:8080")


def test_admin_onlyoffice_test_blocks_loopback_before_http_request():
    from kiara.web.api_onlyoffice import admin_router
    from kiara.web.auth import get_admin_user

    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_admin_user] = lambda: {
        "id": 1,
        "username": "admin",
        "is_admin": True,
    }
    client = TestClient(app, raise_server_exceptions=False)

    with (
        patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo,
        patch("httpx.AsyncClient") as mock_client,
    ):
        mock_config_repo.get.return_value = {
            "document_server_url": "http://127.0.0.1:8080",
            "jwt_secret": "secret",
        }
        resp = client.post("/api/admin/onlyoffice/test")

    assert resp.status_code == 400
    assert "nicht erlaubt" in resp.json()["detail"] or "Loopback" in resp.json()["detail"]
    mock_client.assert_not_called()
