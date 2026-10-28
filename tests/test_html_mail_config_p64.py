"""Archivkonfiguration bleibt vom Connector bis zur Ingestion wirksam."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kiara.config import DEFAULT_PATHS, get_mail_archive_config
from kiara.ingestion._file_utils import scan_files


def test_mail_archive_defaults_and_explicit_config(tmp_path):
    assert get_mail_archive_config({}) == {
        "path": DEFAULT_PATHS["mail_originals"],
        "max_body_bytes": 25 * 1024 * 1024,
    }
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 8192}
    config = {"ingestion": {"mail_archive": archive}}
    assert get_mail_archive_config(config) == archive


@pytest.mark.parametrize("block", [None, [], "archive"])
def test_mail_archive_rejects_non_mapping(block):
    with pytest.raises(ValueError, match="mail_archive.*Mapping"):
        get_mail_archive_config({"ingestion": {"mail_archive": block}})


@pytest.mark.parametrize("path", ["", "relative/originals", None, 17])
def test_mail_archive_rejects_non_absolute_path(path):
    with pytest.raises(ValueError, match="mail_archive.path"):
        get_mail_archive_config({"ingestion": {"mail_archive": {"path": path}}})


@pytest.mark.parametrize("limit", [0, -1, True, False, "8192", 1.5, None])
def test_mail_archive_rejects_invalid_body_limit(limit):
    with pytest.raises(ValueError, match="mail_archive.max_body_bytes"):
        get_mail_archive_config({"ingestion": {"mail_archive": {"max_body_bytes": limit}}})


@pytest.mark.parametrize("connector_kind", ["imap", "exchange"])
def test_connector_scanner_uses_supplied_runtime_archive(tmp_path, monkeypatch, connector_kind):
    import kiara.config as config_module
    import kiara.connectors.base as base
    import kiara.ingestion.imap_scanner as imap_module
    import kiara.ingestion.oauth2_imap_scanner as exchange_module
    from kiara.connectors.exchange_connector import ExchangeOnlineConnector
    from kiara.connectors.imap_connector import IMAPConnector

    connector = IMAPConnector if connector_kind == "imap" else ExchangeOnlineConnector
    scanner_module = imap_module if connector_kind == "imap" else exchange_module
    scanner_name = "IMAPScanner" if connector_kind == "imap" else "OAuth2IMAPScanner"
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 4096}
    cache = tmp_path / "cache"
    scanner = Mock()
    scanner_factory = Mock(return_value=scanner)
    monkeypatch.setattr(scanner_module, scanner_name, scanner_factory)
    monkeypatch.setattr(base, "mail_cache_path", lambda _ds_id: str(cache))
    monkeypatch.setattr(connector, "_check_scan_cache", Mock(return_value=False))
    monkeypatch.setattr(
        config_module, "load_config", Mock(side_effect=AssertionError("runtime config lost"))
    )
    progress, control = Mock(), Mock()

    result = connector.prepare_and_scan(
        {"username": "mail@example.test"},
        {"id": 42},
        {"ingestion": {"mail_archive": archive}},
        progress_callback=progress,
        control_callback=control,
    )

    assert result == str(cache)
    assert scanner_factory.call_args.args[0]["mail_archive"] == archive
    scanner.scan.assert_called_once_with(
        str(cache), progress_callback=progress, control_callback=control
    )


def test_config_reaches_txt_provenance_through_processing_context(tmp_path, monkeypatch):
    import kiara.config as config_module
    from kiara.ingestion.pipeline import _apply_ds_overrides
    from kiara.ingestion.runtime import _load_content

    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 2048}
    source = tmp_path / "cached-mail.txt"
    source.write_text("Auftrag AB-314: Ventil, 24 V.", encoding="utf-8")
    provenance = {
        "parser": "html-to-markdown",
        "content_format": "markdown",
        "source_mime_type": "text/html",
        "html_parser_revision": "fixture-revision",
    }
    read_metadata = Mock(return_value=provenance)
    monkeypatch.setitem(
        sys.modules,
        "kiara.ingestion.mail_archive",
        SimpleNamespace(read_mail_metadata=read_metadata),
    )
    monkeypatch.setattr(
        config_module, "load_config", Mock(side_effect=AssertionError("runtime config lost"))
    )
    config = {
        "ingestion": {"mail_archive": archive, "chunk_size": 1000},
        "datasource": {"file_types": [".txt"]},
    }
    context = _apply_ds_overrides(config, {"chunk_size": 640})

    doc, ddu_error = _load_content(str(source), context, {})

    assert context.mail_archive == archive
    assert config["ingestion"]["chunk_size"] == 640
    assert not ddu_error
    assert doc is not None
    assert doc["text"] == source.read_text(encoding="utf-8")
    for key, value in provenance.items():
        assert doc["metadata"][key] == value
    read_metadata.assert_called_once_with(str(source), doc["text"], archive)


def _scan_config(source, archive):
    return {
        "datasource": {"path": str(source), "file_types": [".html", ".txt", ".json"]},
        "ingestion": {"mail_archive": {"path": str(archive), "max_body_bytes": 4096}},
    }


def test_scan_excludes_nested_archive_but_keeps_similarly_named_directory(tmp_path):
    archive = tmp_path / "originals"
    archive.mkdir()
    (archive / "mail.html").write_text("<p>Archivoriginal</p>")
    (archive / "metadata.json").write_text('{"revision": "fixture"}')
    visible = tmp_path / "originals-not-archive"
    visible.mkdir()
    document = visible / "document.txt"
    document.write_text("Indexierbares Dokument")
    stats = {}

    assert scan_files(_scan_config(tmp_path, archive), file_stats=stats) == [str(document)]
    assert set(stats) == {str(document)}


@pytest.mark.parametrize("target", ["archive", "child", "symlink"])
def test_scan_rejects_datasource_inside_archive(tmp_path, target):
    archive = tmp_path / "originals"
    child = archive / "body"
    child.mkdir(parents=True)
    (child / "mail.html").write_text("<p>Archivoriginal</p>")
    alias = tmp_path / "source-alias"
    alias.symlink_to(archive, target_is_directory=True)
    source = {"archive": archive, "child": child, "symlink": alias}[target]

    with pytest.raises(ValueError, match="Mailoriginalarchiv"):
        scan_files(_scan_config(source, archive))


def test_scan_excludes_archive_file_symlinks(tmp_path):
    archive = tmp_path / "originals"
    archive.mkdir()
    original = archive / "mail.html"
    original.write_text("<p>Archivoriginal</p>")
    source = tmp_path / "source"
    source.mkdir()
    (source / "linked-mail.html").symlink_to(original)
    (source / "linked-originals").symlink_to(archive, target_is_directory=True)

    assert scan_files(_scan_config(source, archive)) == []
