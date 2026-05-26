"""Security-Regressions fuer Datasource-Browse-Endpunkte."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.enums import ConnectorType
from kiara.web.api_datasource import ShareBrowseRequest, WebDAVBrowseRequest
from kiara.web.api_datasource_browse import browse_share, browse_webdav


def _row(connector_type: str, config: dict, active: bool = True) -> dict:
    return {
        "id": 1,
        "type": connector_type,
        "is_active": active,
        "connector_config": json.dumps(config),
    }


def test_browse_share_rejects_masked_password_for_changed_target():
    stored = _row(
        ConnectorType.SMB,
        {
            "server": "files-a",
            "share": "docs",
            "username": "alice",
            "password": "stored-secret",
            "domain": "WORKGROUP",
        },
    )

    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id", return_value=stored):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_share(
                    ShareBrowseRequest(
                        ds_id=1,
                        server="files-b",
                        share="docs",
                        username="alice",
                        password=PASSWORD_MASK,
                        domain="WORKGROUP",
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 400


def test_browse_share_resolves_masked_password_only_for_same_target():
    stored = _row(
        ConnectorType.SMB,
        {
            "server": "files-a",
            "share": "docs",
            "username": "alice",
            "password": "stored-secret",
            "domain": "WORKGROUP",
        },
    )

    def _run(args, **kwargs):
        cred_path = Path(args[3])
        assert "password=stored-secret" in cred_path.read_text()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with (
        patch("kiara.web.api_datasource_browse.ds_repo.get_by_id", return_value=stored),
        patch("subprocess.run", side_effect=_run),
    ):
        result = asyncio.run(
            browse_share(
                ShareBrowseRequest(
                    ds_id=1,
                    server="files-a",
                    share="docs",
                    username="alice",
                    password=PASSWORD_MASK,
                    domain="WORKGROUP",
                ),
                user={"is_admin": True},
            )
        )

    assert result["entries"] == []


def test_browse_share_rejects_inactive_datasource_context():
    stored = _row(
        ConnectorType.SMB,
        {
            "server": "files-a",
            "share": "docs",
            "password": "stored-secret",
        },
        active=False,
    )

    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id", return_value=stored):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_share(
                    ShareBrowseRequest(
                        ds_id=1,
                        server="files-a",
                        share="docs",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 409


def test_browse_share_rejects_ds_id_zero_before_context_lookup():
    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id") as mock_get:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_share(
                    ShareBrowseRequest(
                        ds_id=0,
                        server="files-a",
                        share="docs",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 400
    mock_get.assert_not_called()


def test_browse_share_rejects_wrong_datasource_type_with_plaintext_password():
    stored = _row(
        ConnectorType.WEBDAV,
        {
            "url": "https://cloud.example/dav",
            "password": "stored-secret",
        },
    )

    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id", return_value=stored):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_share(
                    ShareBrowseRequest(
                        ds_id=1,
                        server="files-a",
                        share="docs",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 400


def test_browse_webdav_rejects_masked_password_for_changed_url():
    stored = _row(
        ConnectorType.WEBDAV,
        {
            "url": "https://cloud-a.example/dav",
            "username": "alice",
            "password": "stored-secret",
        },
    )

    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id", return_value=stored):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_webdav(
                    WebDAVBrowseRequest(
                        ds_id=1,
                        url="https://cloud-b.example/dav",
                        username="alice",
                        password=PASSWORD_MASK,
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 400


def test_browse_webdav_uses_tls_verification_default():
    xml = b'<?xml version="1.0"?><d:multistatus xmlns:d="DAV:"></d:multistatus>'
    mock_resp = MagicMock(status_code=207, content=xml)
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.request.return_value = mock_resp

    with patch("httpx.Client", return_value=mock_client) as mock_client_cls:
        result = asyncio.run(
            browse_webdav(
                WebDAVBrowseRequest(
                    url="https://cloud.example/dav",
                    username="alice",
                    password="typed-secret",
                ),
                user={"is_admin": True},
            )
        )

    assert result["entries"] == []
    assert mock_client_cls.call_args.kwargs.get("verify") is not False


def test_browse_share_returns_normalized_path_parent_and_remote_error():
    from kiara.connectors.browse import ConnectorBrowseError

    with patch(
        "kiara.web.api_datasource_browse.SMBConnector.list_directory",
        side_effect=ConnectorBrowseError("kaputt", 502, "bad_response"),
    ):
        result = asyncio.run(
            browse_share(
                ShareBrowseRequest(
                    server="files-a",
                    share="docs",
                    password="typed-secret",
                    path="a\\b",
                ),
                user={"is_admin": True},
            )
        )

    assert result == {"path": "a/b", "entries": [], "error": "kaputt"}


def test_browse_webdav_returns_normalized_path_parent():
    xml = b'<?xml version="1.0"?><d:multistatus xmlns:d="DAV:"></d:multistatus>'
    mock_resp = MagicMock(status_code=207, content=xml)
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.request.return_value = mock_resp

    with patch("httpx.Client", return_value=mock_client):
        result = asyncio.run(
            browse_webdav(
                WebDAVBrowseRequest(
                    url="https://cloud.example/dav",
                    username="alice",
                    password="typed-secret",
                    path="a\\b",
                ),
                user={"is_admin": True},
            )
        )

    assert result["path"] == "a/b"
    assert result["parent"] == "a"
    assert result["entries"] == []


def test_browse_webdav_rejects_ds_id_zero_before_context_lookup():
    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id") as mock_get:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_webdav(
                    WebDAVBrowseRequest(
                        ds_id=0,
                        url="https://cloud.example/dav",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 400
    mock_get.assert_not_called()


def test_browse_webdav_rejects_wrong_datasource_type_with_plaintext_password():
    stored = _row(
        ConnectorType.SMB,
        {
            "server": "files-a",
            "share": "docs",
            "password": "stored-secret",
        },
    )

    with patch("kiara.web.api_datasource_browse.ds_repo.get_by_id", return_value=stored):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                browse_webdav(
                    WebDAVBrowseRequest(
                        ds_id=1,
                        url="https://cloud.example/dav",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )

    assert exc_info.value.status_code == 400


def test_browse_picker_path_security_errors_are_http_errors():
    with pytest.raises(HTTPException) as smb_exc:
        asyncio.run(
            browse_share(
                ShareBrowseRequest(
                    server="files-a",
                    share="docs",
                    password="typed-secret",
                    path="../x",
                ),
                user={"is_admin": True},
            )
        )
    assert smb_exc.value.status_code == 403

    with pytest.raises(HTTPException) as webdav_exc:
        asyncio.run(
            browse_webdav(
                WebDAVBrowseRequest(
                    url="https://cloud.example/dav",
                    password="typed-secret",
                    path="bad\npath",
                ),
                user={"is_admin": True},
            )
        )
    assert webdav_exc.value.status_code == 400


def test_browse_picker_unexpected_connector_exceptions_propagate():
    with patch(
        "kiara.web.api_datasource_browse.SMBConnector.list_directory",
        side_effect=RuntimeError("programming bug"),
    ):
        with pytest.raises(RuntimeError, match="programming bug"):
            asyncio.run(
                browse_share(
                    ShareBrowseRequest(
                        server="files-a",
                        share="docs",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )

    with patch(
        "kiara.web.api_datasource_browse.WebDAVConnector.list_directory",
        side_effect=RuntimeError("programming bug"),
    ):
        with pytest.raises(RuntimeError, match="programming bug"):
            asyncio.run(
                browse_webdav(
                    WebDAVBrowseRequest(
                        url="https://cloud.example/dav",
                        password="typed-secret",
                    ),
                    user={"is_admin": True},
                )
            )
