"""Tests fuer zentrale Connector-Browse-Helper."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors.browse import (
    ConnectorBrowseError,
    browse_smbclient,
    browse_webdav_propfind,
    list_local_directory,
    normalize_browse_path,
    parent_path,
)

_ROOT = Path(__file__).resolve().parent.parent


def test_connector_browse_error_contract():
    exc = ConnectorBrowseError("kaputt", 502, "bad_response")
    assert exc.message == "kaputt"
    assert exc.status_code == 502
    assert exc.reason == "bad_response"
    assert str(exc) == "kaputt"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("", None),
        ("/", None),
        (".", None),
        ("a", ""),
        ("a/b", "a"),
        ("a/b/c", "a/b"),
    ],
)
def test_parent_path_table(path, expected):
    assert parent_path(path) == expected


def test_normalize_browse_path_blocks_traversal_and_absolute():
    assert normalize_browse_path(None) == ""
    assert normalize_browse_path(".") == ""
    assert normalize_browse_path("/") == ""
    assert normalize_browse_path("a\\b") == "a/b"

    with pytest.raises(ConnectorBrowseError) as traversal:
        normalize_browse_path("../x")
    assert traversal.value.status_code == 403
    assert traversal.value.reason == "forbidden"

    with pytest.raises(ConnectorBrowseError) as absolute:
        normalize_browse_path("/x")
    assert absolute.value.status_code == 403

    with pytest.raises(ConnectorBrowseError) as bad_input:
        normalize_browse_path("a\nb")
    assert bad_input.value.status_code == 400


def test_list_local_directory_shape_sorting_and_directories_only(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "A").mkdir()
    (tmp_path / "a.txt").write_text("a")

    result = list_local_directory(str(tmp_path), "", directories_only=False)
    names = [entry["name"] for entry in result["entries"]]
    assert names == ["A", "a.txt", "b.txt"]
    assert all("name" in entry and "is_dir" in entry for entry in result["entries"])
    assert all(
        "path" in entry and "modified" in entry and "size" in entry for entry in result["entries"]
    )

    dirs = list_local_directory(str(tmp_path), "", directories_only=True)
    assert [entry["name"] for entry in dirs["entries"]] == ["A"]


def test_list_local_directory_security_errors(tmp_path):
    with pytest.raises(ConnectorBrowseError) as traversal:
        list_local_directory(str(tmp_path), "../x")
    assert traversal.value.status_code == 403

    with pytest.raises(ConnectorBrowseError) as outside:
        list_local_directory(str(tmp_path), "/", allow_absolute=True)
    assert outside.value.status_code == 403

    with pytest.raises(ConnectorBrowseError) as missing:
        list_local_directory(str(tmp_path), "missing")
    assert missing.value.status_code == 404


def test_browse_smbclient_args_password_and_sub_path(tmp_path):
    config = {
        "server": "files",
        "share": "docs",
        "username": "alice",
        "password": "secret",
        "domain": "WORKGROUP",
        "sub_path": "Subdir",
    }

    def fake_run(args, **kwargs):
        cred_path = args[3]
        assert args[:3] == ["smbclient", "//files/docs", "--authentication-file"]
        assert args[4] == "-c"
        assert args[5] == 'cd "Subdir\\Sub2"; ls'
        assert "secret" not in " ".join(args)
        with open(cred_path) as fh:
            assert "password=secret" in fh.read()
        return SimpleNamespace(
            returncode=0,
            stdout="  Alpha                           D        0  Mon Apr 26 12:00:00 2026\n"
            "  z.txt                           A       12  Mon Apr 26 12:00:00 2026\n",
            stderr="",
        )

    with patch("kiara.connectors.browse.subprocess.run", side_effect=fake_run):
        result = browse_smbclient(config, "Sub2", directories_only=False)

    assert [entry["name"] for entry in result["entries"]] == ["Alpha", "z.txt"]
    assert result["entries"][0]["is_dir"] is True
    assert result["entries"][1]["size"] == 12


@pytest.mark.parametrize(
    "path", ['dir"; rm /tmp/x; "', "bad;path", "bad$path", "bad`path", "a|b", "a*b", "a?b"]
)
def test_browse_smbclient_blocks_command_injection(path):
    config = {"server": "files", "share": "docs", "password": "secret"}
    with pytest.raises(ConnectorBrowseError) as exc_info:
        browse_smbclient(config, path)
    assert exc_info.value.status_code == 403
    assert exc_info.value.reason == "forbidden"


@pytest.mark.parametrize(
    "name",
    [
        "Linux-KB (Embedding-Test)",
        "R&D",
        "Projekt [Final]",
        "v1.2+rc1",
        "user@host",
        "Andre's Notes",
        "Müller's Übung",
    ],
)
def test_browse_smbclient_allows_common_directory_chars(name):
    """Verzeichnisnamen mit haeufigen harmlosen Sonderzeichen muessen browsebar sein."""
    config = {"server": "files", "share": "docs", "password": "secret"}

    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["smb_cmd"] = args[5]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("kiara.connectors.browse.subprocess.run", side_effect=fake_run):
        browse_smbclient(config, name)

    assert captured["smb_cmd"] == f'cd "{name}"; ls'


def test_browse_smbclient_validates_inputs_before_subprocess():
    with pytest.raises(ConnectorBrowseError) as missing:
        browse_smbclient({"server": "", "share": "docs"}, "")
    assert missing.value.status_code == 400

    with pytest.raises(ConnectorBrowseError) as credentials:
        browse_smbclient({"server": "files", "share": "docs", "password": "a\nb"}, "")
    assert credentials.value.status_code == 400

    with pytest.raises(ConnectorBrowseError) as sub_path:
        browse_smbclient({"server": "files", "share": "docs", "sub_path": "../x"}, "")
    assert sub_path.value.status_code == 403


def test_browse_smbclient_maps_process_failure():
    with patch("kiara.connectors.browse.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=1, stdout="", stderr="NT_STATUS_ACCESS_DENIED"
        )
        with pytest.raises(ConnectorBrowseError) as exc_info:
            browse_smbclient({"server": "files", "share": "docs"}, "")
    assert exc_info.value.status_code == 502
    assert exc_info.value.reason == "bad_response"
    assert "NT_STATUS" in exc_info.value.message


def test_browse_webdav_builds_url_and_parses_directories():
    xml = b"""<?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/dav/Docs/Sub/</d:href>
        <d:propstat><d:prop><d:resourcetype><d:collection /></d:resourcetype></d:prop></d:propstat>
      </d:response>
      <d:response>
        <d:href>/dav/Docs/Sub/Alpha/</d:href>
        <d:propstat><d:prop><d:resourcetype><d:collection /></d:resourcetype></d:prop></d:propstat>
      </d:response>
      <d:response>
        <d:href>/dav/Docs/Sub/file.txt</d:href>
        <d:propstat><d:prop><d:getcontentlength>5</d:getcontentlength></d:prop></d:propstat>
      </d:response>
    </d:multistatus>"""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.request.return_value = MagicMock(status_code=207, content=xml)

    with patch("kiara.connectors.browse.httpx.Client", return_value=mock_client) as client_cls:
        result = browse_webdav_propfind(
            {
                "url": "https://cloud.example/dav/",
                "username": "alice",
                "password": "secret",
                "base_path": "/Docs",
            },
            "Sub",
            directories_only=True,
        )

    assert client_cls.call_args.kwargs.get("verify") is not False
    assert mock_client.request.call_args.args[:2] == (
        "PROPFIND",
        "https://cloud.example/dav/Docs/Sub/",
    )
    assert mock_client.request.call_args.kwargs["headers"]["Depth"] == "1"
    assert result["entries"] == [{"name": "Alpha", "is_dir": True}]


def test_browse_webdav_quotes_reserved_path_chars_and_blocks_encoded_traversal():
    xml = b'<?xml version="1.0"?><d:multistatus xmlns:d="DAV:"></d:multistatus>'
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.request.return_value = MagicMock(status_code=207, content=xml)

    with patch("kiara.connectors.browse.httpx.Client", return_value=mock_client):
        result = browse_webdav_propfind(
            {"url": "https://cloud.example/dav/", "base_path": "/Docs"},
            "A#B/Q?R/100%",
            directories_only=True,
        )

    assert result["entries"] == []
    assert mock_client.request.call_args.args[:2] == (
        "PROPFIND",
        "https://cloud.example/dav/Docs/A%23B/Q%3FR/100%25/",
    )

    with patch("kiara.connectors.browse.httpx.Client") as client_cls:
        with pytest.raises(ConnectorBrowseError) as exc_info:
            browse_webdav_propfind(
                {"url": "https://cloud.example/dav/", "base_path": "/Docs"},
                "%2e%2e/secret",
            )
    assert exc_info.value.status_code == 403
    client_cls.assert_not_called()


def test_browse_webdav_allows_identity_folder_name():
    xml = b"""<?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/dav/</d:href>
        <d:propstat><d:prop><d:resourcetype><d:collection /></d:resourcetype></d:prop></d:propstat>
      </d:response>
      <d:response>
        <d:href>/dav/Identity/</d:href>
        <d:propstat><d:prop><d:resourcetype><d:collection /></d:resourcetype></d:prop></d:propstat>
      </d:response>
    </d:multistatus>"""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.request.return_value = MagicMock(status_code=207, content=xml)

    with patch("kiara.connectors.browse.httpx.Client", return_value=mock_client):
        result = browse_webdav_propfind({"url": "https://cloud.example/dav"}, "")

    assert result["entries"] == [{"name": "Identity", "is_dir": True}]


def test_browse_webdav_rejects_bad_xml_and_large_response():
    with patch("kiara.connectors.browse.httpx.Client") as client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.request.return_value = MagicMock(status_code=207, content=b"<!DOCTYPE x><x />")
        client_cls.return_value = mock_client
        with pytest.raises(ConnectorBrowseError) as exc_info:
            browse_webdav_propfind({"url": "https://cloud.example/dav"}, "")
    assert exc_info.value.status_code == 502

    with patch("kiara.connectors.browse.httpx.Client") as client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.request.return_value = MagicMock(
            status_code=207,
            content=b'<?xml version="1.0"?><!ENTITY x "y"><x />',
        )
        client_cls.return_value = mock_client
        with pytest.raises(ConnectorBrowseError) as exc_info:
            browse_webdav_propfind({"url": "https://cloud.example/dav"}, "")
    assert exc_info.value.status_code == 502

    with patch("kiara.connectors.browse.httpx.Client") as client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.request.return_value = MagicMock(
            status_code=207, content=b"x" * (2 * 1024 * 1024 + 1)
        )
        client_cls.return_value = mock_client
        with pytest.raises(ConnectorBrowseError) as exc_info:
            browse_webdav_propfind({"url": "https://cloud.example/dav"}, "")
    assert exc_info.value.message == "WebDAV-Antwort zu gross"


def test_browse_webdav_validates_config_root_and_credentials():
    with pytest.raises(ConnectorBrowseError) as credentials:
        browse_webdav_propfind({"url": "https://cloud.example/dav", "username": "a\rb"}, "")
    assert credentials.value.status_code == 400

    with pytest.raises(ConnectorBrowseError) as base_path:
        browse_webdav_propfind({"url": "https://cloud.example/dav", "base_path": "../x"}, "")
    assert base_path.value.status_code == 403


def test_datasource_workflow_docs_use_current_connector_tokens():
    for rel in (
        "kiara/admin/wiki_content/workflows/setup-datasource.md",
        "kiara/admin/wiki_content/workflows/first-setup.md",
    ):
        text = (_ROOT / rel).read_text(encoding="utf-8")
        assert "exchange_online" in text
        assert "exchange, mssql, postgresql, mysql" not in text
        assert "**exchange**" not in text
        assert "**mssql**" not in text
        assert "**postgresql**" not in text
        assert "**mysql**" not in text
        assert "`sql`" in text
