"""Die Connectoranbindung nutzt den expliziten Bestandsabruf und Runtime-Konfiguration."""

import json
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("connector_kind", ["imap", "exchange"])
def test_reprocess_connector_passes_runtime_config_paths_and_callbacks(
    tmp_path, monkeypatch, connector_kind
):
    import kiara.config as config_module
    import kiara.connectors.base as base
    import kiara.ingestion.imap_scanner as imap_module
    import kiara.ingestion.oauth2_imap_scanner as exchange_module
    from kiara.connectors.exchange_connector import ExchangeOnlineConnector
    from kiara.connectors.imap_connector import IMAPConnector

    connector = IMAPConnector if connector_kind == "imap" else ExchangeOnlineConnector
    module = imap_module if connector_kind == "imap" else exchange_module
    scanner_name = "IMAPScanner" if connector_kind == "imap" else "OAuth2IMAPScanner"
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = cache / ".scan_done"
    marker.write_text("Unveraenderter Scanmarker")
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 4096}
    paths = [str(cache / "mail.txt")]
    progress, control = Mock(), Mock()
    expected = {"files": paths, "errors": {}, "cancelled": False}
    scanner = Mock()
    scanner.reprocess_cached_mails.return_value = expected
    scanner.scan.side_effect = AssertionError("Normaler Scan ist keine Migration")
    scanner_factory = Mock(return_value=scanner)
    monkeypatch.setattr(module, scanner_name, scanner_factory)
    monkeypatch.setattr(base, "mail_cache_path", lambda _ds_id: str(cache))
    check_scan_cache = Mock(side_effect=AssertionError("Scanmarker ist keine Migrationsgrenze"))
    monkeypatch.setattr(connector, "_check_scan_cache", check_scan_cache)
    monkeypatch.setattr(
        config_module, "load_config", Mock(side_effect=AssertionError("Runtime-Config verloren"))
    )

    result = connector.reprocess_cached(
        {"username": "mail@example.test", "folders": "INBOX"},
        {"id": 42},
        {"ingestion": {"mail_archive": archive}},
        cache_paths=paths,
        progress_callback=progress,
        control_callback=control,
    )

    assert result is expected
    assert scanner_factory.call_args.args[0]["mail_archive"] == archive
    scanner.reprocess_cached_mails.assert_called_once_with(
        str(cache), cache_paths=paths, progress_callback=progress, control_callback=control
    )
    scanner.scan.assert_not_called()
    check_scan_cache.assert_not_called()
    assert marker.read_text() == "Unveraenderter Scanmarker"


@pytest.mark.parametrize("signal", ["cancel", "shutdown"])
def test_migration_control_signal_stops_before_first_file(tmp_path, monkeypatch, signal):
    from kiara.ingestion.imap_scanner import IMAPScanner

    scanner = IMAPScanner({})
    folder_key = scanner._folder_key("INBOX")
    path = tmp_path / folder_key / "mail.txt"
    path.parent.mkdir()
    path.write_text("Unbearbeitet")
    registry = {folder_key: {"uidvalidity": "123", "uids": {"42": str(path.relative_to(tmp_path))}}}
    (tmp_path / ".uid_registry.json").write_text(json.dumps(registry))
    connect = Mock(side_effect=AssertionError("Abbruch muss vor Netzabruf erfolgen"))
    monkeypatch.setattr(scanner, "connect", connect)
    result = scanner.reprocess_cached_mails(str(tmp_path), control_callback=lambda: signal)
    assert result == {"files": [], "errors": {}, "cancelled": True}
    assert path.read_text() == "Unbearbeitet"
    connect.assert_not_called()
