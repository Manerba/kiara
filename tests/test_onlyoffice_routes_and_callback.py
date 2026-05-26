from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeDbTransaction:
    def __init__(self, events: list[str] | None = None):
        self.events = events

    def __enter__(self):
        if self.events is not None:
            self.events.append("tx_enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.events is not None:
            self.events.append("tx_exit")

    def execute(self, sql, params=None):
        if self.events is not None:
            self.events.append("advisory")
        assert "pg_advisory_xact_lock" in sql
        assert params


def test_onlyoffice_admin_router_does_not_expose_user_routes():
    from kiara.web.api_onlyoffice import admin_router

    app = FastAPI()
    app.include_router(admin_router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/onlyoffice/open", params={"file_path": "document.docx"})

    assert resp.status_code == 404


def test_onlyoffice_user_router_does_not_expose_admin_routes():
    from kiara.web.api_onlyoffice import callback_router, editor_router, wopi_router

    app = FastAPI()
    app.include_router(editor_router)
    app.include_router(wopi_router)
    app.include_router(callback_router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/api/admin/onlyoffice/config")

    assert resp.status_code == 404


def test_onlyoffice_callback_persists_downloaded_document(tmp_path):
    from kiara.web.api_onlyoffice import _sign_jwt, callback_router

    target = tmp_path / "document.docx"
    target.write_bytes(b"old")

    async def fake_download(download_url, oo_config):
        assert download_url == "http://onlyoffice/cache/document.docx"
        assert oo_config["document_server_url"] == "http://onlyoffice"
        return b"new"

    app = FastAPI()
    app.include_router(callback_router)
    client = TestClient(app, raise_server_exceptions=False)

    with (
        patch(
            "kiara.web.api_onlyoffice._validate_wopi_token",
            return_value={
                "user_id": 1,
                "username": "testuser",
                "file_id": "file123",
                "file_path": str(target),
            },
        ),
        patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo,
        patch(
            "kiara.web.api_onlyoffice._download_onlyoffice_callback_content",
            new=fake_download,
        ),
        patch(
            "kiara.web.api_onlyoffice.db_transaction",
            side_effect=lambda: _FakeDbTransaction(),
        ),
        patch(
            "kiara.web.api_onlyoffice._load_wopi_user_with_acl",
            return_value={"id": 1, "username": "testuser", "datasource_ids": []},
        ),
        patch(
            "kiara.web.api_onlyoffice._check_datasource_acl",
            return_value=None,
        ),
    ):
        mock_config_repo.get.return_value = {
            "document_server_url": "http://onlyoffice",
            "jwt_secret": "secret",
        }
        jwt_token = _sign_jwt(
            {
                "payload": {
                    "status": 2,
                    "url": "http://onlyoffice/cache/document.docx",
                    "key": "doc-key",
                }
            },
            "secret",
        )

        resp = client.post(
            "/api/onlyoffice/callback",
            params={"file_id": "file123", "access_token": "token123"},
            json={"status": 2, "url": "http://onlyoffice/cache/document.docx", "key": "doc-key"},
            headers={"Authorization": f"Bearer {jwt_token}"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"error": 0}
    assert target.read_bytes() == b"new"


def test_onlyoffice_callback_fails_closed_without_token(tmp_path):
    from kiara.web.api_onlyoffice import callback_router

    target = tmp_path / "document.docx"
    target.write_bytes(b"old")

    app = FastAPI()
    app.include_router(callback_router)
    client = TestClient(app, raise_server_exceptions=False)

    with patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo:
        mock_config_repo.get.return_value = {
            "document_server_url": "http://onlyoffice",
            "jwt_secret": "secret",
        }
        resp = client.post(
            "/api/onlyoffice/callback",
            params={"file_id": "file123"},
            json={"status": 2, "url": "http://onlyoffice/cache/document.docx", "key": "doc-key"},
        )

    assert resp.status_code == 401
    assert resp.json() == {"error": 1}
    assert target.read_bytes() == b"old"


@pytest.mark.asyncio
async def test_wopi_put_file_keeps_lock_check_and_write_in_one_transaction(tmp_path, monkeypatch):
    from kiara.web import api_onlyoffice as api

    target = tmp_path / "document.docx"
    target.write_bytes(b"")
    events: list[str] = []

    class FakeRequest:
        headers = {"content-length": "3"}

        async def stream(self):
            events.append("stream")
            yield b"new"

    class FakeWopiRepo:
        def get_lock(self, file_id, *, for_update=False, conn=None):
            events.append("get_lock")
            assert file_id == "file123"
            assert for_update is True
            assert conn is not None
            return None

    async def fake_proof(_request, _access_token):
        events.append("proof")

    def fake_write(full_path, content, allow_create=False):
        events.append("write")
        assert full_path == str(target)
        assert content == b"new"
        assert allow_create is True

    monkeypatch.setattr(
        api,
        "_validate_wopi_token",
        lambda _access_token, _file_id: {
            "user_id": 1,
            "username": "testuser",
            "file_id": "file123",
            "file_path": str(target),
        },
    )
    monkeypatch.setattr(api, "_validate_wopi_proof", fake_proof)
    monkeypatch.setattr(api, "wopi_repo", FakeWopiRepo())
    monkeypatch.setattr(api, "db_transaction", lambda: _FakeDbTransaction(events))
    monkeypatch.setattr(api, "_write_file_atomic", fake_write)
    monkeypatch.setattr(
        api,
        "_load_wopi_user_with_acl",
        lambda _user_id: {"id": 1, "username": "testuser", "datasource_ids": []},
    )
    monkeypatch.setattr(api, "_check_datasource_acl", lambda *_args, **_kwargs: None)

    response = await api.wopi_put_file("file123", FakeRequest(), access_token="token123")

    assert response.status_code == 200
    assert events == [
        "proof",
        "stream",
        "tx_enter",
        "advisory",
        "get_lock",
        "write",
        "tx_exit",
    ]


def test_onlyoffice_callback_rejects_invalid_jwt(tmp_path):
    from kiara.web.api_onlyoffice import callback_router

    target = tmp_path / "document.docx"
    target.write_bytes(b"old")

    app = FastAPI()
    app.include_router(callback_router)
    client = TestClient(app, raise_server_exceptions=False)

    with patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo:
        mock_config_repo.get.return_value = {
            "document_server_url": "http://onlyoffice",
            "jwt_secret": "secret",
        }
        resp = client.post(
            "/api/onlyoffice/callback",
            params={"file_id": "file123", "access_token": "token123"},
            json={"status": 2, "url": "http://onlyoffice/cache/document.docx", "key": "doc-key"},
            headers={"Authorization": "Bearer invalid"},
        )

    assert resp.status_code == 401
    assert resp.json() == {"error": 1}
    assert target.read_bytes() == b"old"


def test_onlyoffice_callback_fails_closed_when_download_fails(tmp_path):
    from kiara.web.api_onlyoffice import _sign_jwt, callback_router

    target = tmp_path / "document.docx"
    target.write_bytes(b"old")

    async def failing_download(download_url, oo_config):
        raise RuntimeError("download failed")

    app = FastAPI()
    app.include_router(callback_router)
    client = TestClient(app, raise_server_exceptions=False)

    with (
        patch(
            "kiara.web.api_onlyoffice._validate_wopi_token",
            return_value={
                "user_id": 1,
                "username": "testuser",
                "file_id": "file123",
                "file_path": str(target),
            },
        ),
        patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo,
        patch(
            "kiara.web.api_onlyoffice._download_onlyoffice_callback_content",
            new=failing_download,
        ),
    ):
        mock_config_repo.get.return_value = {
            "document_server_url": "http://onlyoffice",
            "jwt_secret": "secret",
        }
        jwt_token = _sign_jwt(
            {
                "payload": {
                    "status": 2,
                    "url": "http://onlyoffice/cache/document.docx",
                    "key": "doc-key",
                }
            },
            "secret",
        )

        resp = client.post(
            "/api/onlyoffice/callback",
            params={"file_id": "file123", "access_token": "token123"},
            json={"status": 2, "url": "http://onlyoffice/cache/document.docx", "key": "doc-key"},
            headers={"Authorization": f"Bearer {jwt_token}"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"error": 1}
    assert target.read_bytes() == b"old"
