"""HTML mail selection uses MIME originals, never guesses from cached TXT."""

import json
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import Mock

import pytest
from test_mail_migration_p64 import _raw, _setup

from kiara.ingestion._file_utils import compute_file_hash
from kiara.ingestion.imap_scanner import IMAPScanner
from kiara.ingestion.mail_archive import read_mail_metadata
from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner
from kiara.ingestion.processing_revision import processing_file_hash


def _prepare(scanner, output, path, settings, **kwargs):
    from kiara.ingestion.mail_migration import prepare_html_mails

    config = {"ingestion": {"mail_archive": settings, "chunk_size": 640, "chunk_overlap": 100}}
    return prepare_html_mails(scanner, str(output), [str(path)], config, **kwargs)


@pytest.mark.parametrize("scanner_class", [IMAPScanner, OAuth2IMAPScanner])
def test_old_html_mail_is_classified_once_and_preserves_headers(
    tmp_path, monkeypatch, scanner_class
):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch, scanner_class)
    registry_before = (output / ".uid_registry.json").read_bytes()
    marker_before = (output / ".scan_done").read_bytes()
    progress = []

    result = _prepare(
        scanner,
        output,
        path,
        settings,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert result["files"] == result["html_paths"] == [str(path)]
    assert result["errors"] == result["error_kinds"] == {}
    assert result["examined"] == 1
    assert not result["cancelled"]
    assert path.read_text().startswith("Betreff: Alt\nOrdner: INBOX\n---\n")
    assert "Ventil V42" in path.read_text()
    assert sum(call[0] == "fetch" for call in conn.calls) == 1
    assert conn.flags == {b"42": {"\\Flagged"}}
    assert (output / ".uid_registry.json").read_bytes() == registry_before
    assert (output / ".scan_done").read_bytes() == marker_before
    assert progress == [(1, 1)]
    assert read_mail_metadata(path, path.read_text(), settings)["source_mime_type"] == "text/html"


def _multipart_plain_with_html():
    msg = MIMEMultipart("alternative")
    msg.attach(MIMEText("Klartext <b>ist Klartext</b>", "plain", "utf-8"))
    msg.attach(MIMEText("<p>Andere HTML-Alternative</p>", "html", "utf-8"))
    return msg.as_bytes()


def _plain_with_html_attachment():
    msg = MIMEMultipart("mixed")
    msg.attach(MIMEText("Klartext", "plain", "utf-8"))
    attachment = MIMEText("<p>Anhang</p>", "html", "utf-8")
    attachment.add_header("Content-Disposition", "attachment", filename="attachment.html")
    msg.attach(attachment)
    return msg.as_bytes()


@pytest.mark.parametrize(
    "raw",
    [
        _raw("Klartext <html><b>keine HTML-Mail</b></html>", "plain"),
        _multipart_plain_with_html(),
        _plain_with_html_attachment(),
    ],
)
def test_plain_main_body_does_not_write_cache_or_archive(tmp_path, monkeypatch, raw):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch, messages={b"42": raw})
    before = path.read_bytes()
    result = _prepare(scanner, output, path, settings)
    assert result["files"] == result["html_paths"] == []
    assert result["skipped"]["plain"] == 1
    assert not result["errors"]
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()


def _attachment_only():
    msg = MIMEMultipart()
    attachment = MIMEApplication(b"attachment")
    attachment.add_header("Content-Disposition", "attachment", filename="test.pdf")
    msg.attach(attachment)
    return msg.as_bytes()


@pytest.mark.parametrize(
    "raw",
    [
        _raw(" \n", "plain"),
        _raw(" \n", "html"),
        _raw("<html><body><script>hidden()</script></body></html>"),
        _attachment_only(),
    ],
)
def test_empty_main_body_is_an_error_without_writes(tmp_path, monkeypatch, raw):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch, messages={b"42": raw})
    before = path.read_bytes()
    result = _prepare(scanner, output, path, settings)
    assert result["files"] == []
    assert result["error_kinds"] == {str(path): "empty_body"}
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()


@pytest.mark.parametrize("hash_mode", ["content", "mtime"])
def test_outdated_skip_requires_verified_archive_and_current_hash(tmp_path, monkeypatch, hash_mode):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    _prepare(scanner, output, path, settings)
    config = {"ingestion": {"mail_archive": settings, "chunk_size": 640, "chunk_overlap": 100}}
    stored = {
        str(path): processing_file_hash(config, str(path), compute_file_hash(str(path), hash_mode))
    }
    archive = next((tmp_path / "originals").rglob("*.json"))
    before = (
        path.read_bytes(),
        path.stat().st_mtime_ns,
        archive.read_bytes(),
        archive.stat().st_mtime_ns,
    )
    monkeypatch.setattr(
        scanner, "connect", lambda: pytest.fail("Verified archive needs no network")
    )
    conn.calls.clear()

    result = _prepare(scanner, output, path, settings, stored_hashes=stored, hash_mode=hash_mode)

    assert result["files"] == []
    assert result["html_paths"] == [str(path)]
    assert result["skipped"]["current"] == 1
    assert not result["errors"]
    assert (
        path.read_bytes(),
        path.stat().st_mtime_ns,
        archive.read_bytes(),
        archive.stat().st_mtime_ns,
    ) == before
    assert conn.calls == []


def test_missing_archive_never_uses_legacy_hash_to_skip(tmp_path, monkeypatch):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    stored = {str(path): compute_file_hash(str(path), "content")}
    result = _prepare(scanner, output, path, settings, stored_hashes=stored)
    assert result["files"] == [str(path)]
    assert any(call[0] == "fetch" for call in conn.calls)


@pytest.mark.parametrize("change", ["revision", "all", "damaged_cache"])
def test_verified_original_repairs_or_reprocesses_without_network(tmp_path, monkeypatch, change):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch)
    _prepare(scanner, output, path, settings)
    config = {"ingestion": {"mail_archive": settings, "chunk_size": 640, "chunk_overlap": 100}}
    stored = {
        str(path): processing_file_hash(config, str(path), compute_file_hash(str(path), "content"))
    }
    before = path.read_bytes()
    if change == "revision":
        import kiara_document_loaders.html as html

        original_revision = html.conversion_revision()
        monkeypatch.setattr(html, "conversion_revision", lambda: (*original_revision, "future"))
    elif change == "damaged_cache":
        path.write_bytes(b"abgebrochener Cacheersatz")
    monkeypatch.setattr(
        scanner, "connect", lambda: pytest.fail("Verified archive needs no network")
    )
    result = _prepare(
        scanner,
        output,
        path,
        settings,
        stored_hashes=stored,
        mode="all" if change == "all" else "outdated",
    )
    assert result["files"] == [str(path)]
    assert not result["errors"]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "fake_options, kind",
    [
        ({"messages": {}}, "missing_original"),
        ({"validity": b"999"}, "preparation"),
        ({"fetch_error": RuntimeError("network unavailable")}, "preparation"),
    ],
)
def test_missing_original_is_distinct_from_identity_and_network_errors(
    tmp_path, monkeypatch, fake_options, kind
):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch, **fake_options)
    before = path.read_bytes()
    result = _prepare(scanner, output, path, settings)
    assert result["files"] == []
    assert result["error_kinds"] == {str(path): kind}
    assert path.read_bytes() == before


@pytest.mark.parametrize("html_only", [False, True])
@pytest.mark.parametrize(
    "notification",
    [
        b"1 (FLAGS (\\Seen))",
        b"8 (FLAGS ())",
        b"2 (UID 43 FLAGS (\\Seen $Forwarded custom-keyword))",
        b"1 (FLAGS (\\Answered \\Flagged) UID 42)",
        b"2 (FLAGS (\\Seen) MODSEQ (18446744073709551615) UID 43)",
        b"2 (UID 43 MODSEQ (2) FLAGS (\\Seen))",
        b"2 (flags (\\Seen) uid 43)",
    ],
)
def test_valid_flag_notification_does_not_reject_requested_mail(
    tmp_path, monkeypatch, notification, html_only
):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    raw = _raw()
    monkeypatch.setattr(
        conn,
        "uid",
        lambda *_args: (
            "OK",
            [
                notification,
                (b"1 (UID 42 BODY[] {" + str(len(raw)).encode() + b"}", raw),
                b")",
                notification,
            ],
        ),
    )

    result = (
        _prepare(scanner, output, path, settings)
        if html_only
        else scanner.reprocess_cached_mails(str(output))
    )

    assert result["files"] == [str(path)]
    assert result["errors"] == {}
    assert "Ventil V42" in path.read_text()
    assert read_mail_metadata(path, path.read_text(), settings)["source_mime_type"] == "text/html"


@pytest.mark.parametrize(
    "notification",
    [
        b"1 (FLAGS (\\Seen)",
        b"1 (FLAGS (\\Seen) BODY[] NIL)",
        b"1 (UID 42)",
        b"1 (FLAGS (\\Seen) UID 42 UID 43)",
        b"1 (FLAGS () FLAGS (\\Seen))",
        b"0 (FLAGS ())",
        b"1 (FLAGS () UID 4294967296)",
        b"1 (FLAGS () MODSEQ (18446744073709551616))",
        b'1 (FLAGS ("quoted"))',
        b"1 (FLAGS (invalid\nflag))",
    ],
)
def test_malformed_notification_with_complete_mail_still_fails_closed(
    tmp_path, monkeypatch, notification
):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    raw = _raw()
    monkeypatch.setattr(
        conn, "uid", lambda *_args: ("OK", [(b"1 (UID 42 BODY[] {100}", raw), b")", notification])
    )
    before = path.read_bytes()

    result = _prepare(scanner, output, path, settings)

    assert result["error_kinds"] == {str(path): "preparation"}
    assert result["files"] == []
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()


def test_flags_without_requested_mail_remain_missing_original(tmp_path, monkeypatch):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(conn, "uid", lambda *_args: ("OK", [b"1 (UID 42 FLAGS (\\Seen))"]))
    before = path.read_bytes()

    result = _prepare(scanner, output, path, settings)

    assert result["error_kinds"] == {str(path): "missing_original"}
    assert result["files"] == []
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "response",
    [
        ("NO", [b"permission denied"]),
        ("BAD", [b"FETCH not accepted"]),
        ("OK", [(b"1 (UID 43 BODY[] {5}", b"wrong")]),
        ("OK", [(b"1 (BODY[] {5}", b"wrong")]),
        ("OK", [(b"1 (UID 42 BODY[] {0}", b"")]),
        ("OK", [b"1 (UID 42 BODY[] NIL)"]),
        ("OK", [(b"1 (UID 42 BODY[] {5}", b"first"), (b"1 (UID 42 BODY[] {6}", b"second")]),
        ("OK", [b"1 (FLAGS (\\Seen))", (b"1 (UID 43 BODY[] {5}", b"wrong")]),
        ("OK", [b"1 (FLAGS (\\Seen))", (b"1 (UID 42 BODY[] {0}", b"")]),
        ("OK", [b"1 (FLAGS (\\Seen))", (b"1 (BODY[] {5}", b"wrong")]),
        ("OK", [b"1 (FLAGS (\\Seen))", (b"1 (UID 42 UID 42 BODY[] {5}", b"wrong")]),
        (
            "OK",
            [
                b"1 (FLAGS (\\Seen))",
                (b"1 (UID 42 BODY[] {5}", b"first"),
                (b"1 (UID 42 BODY[] {6}", b"second"),
            ],
        ),
    ],
)
def test_fetch_rejection_or_ambiguous_response_is_not_a_missing_original(
    tmp_path, monkeypatch, response
):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(conn, "uid", lambda *_args: response)
    before = path.read_bytes()
    result = _prepare(scanner, output, path, settings)
    assert result["error_kinds"] == {str(path): "preparation"}
    assert result["files"] == []
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()


@pytest.mark.parametrize("stop_at", [1, 2, 3])
def test_cancel_before_entry_fetch_or_write_preserves_cache(tmp_path, monkeypatch, stop_at):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    before = path.read_bytes()
    calls = 0

    def control():
        nonlocal calls
        calls += 1
        return "cancel" if calls == stop_at else None

    result = _prepare(scanner, output, path, settings, control_callback=control)
    assert result["cancelled"]
    assert result["files"] == []
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()
    if stop_at < 3:
        assert conn.calls == []


def test_control_exception_propagates_before_write_and_closes_session(tmp_path, monkeypatch):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    before = path.read_bytes()
    calls = 0

    def control():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Jobsteuerung fehlgeschlagen")

    with pytest.raises(RuntimeError, match="Jobsteuerung"):
        _prepare(scanner, output, path, settings, control_callback=control)
    assert path.read_bytes() == before
    assert conn.calls[-1] == ("logout",)


def test_conversion_rejection_preserves_cache_and_originals(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch)
    before = path.read_bytes()

    def reject(_self):
        raise ValueError("DOM depth guard")

    from kiara.ingestion.imap_scanner import _MailBody

    monkeypatch.setattr(_MailBody, "convert", reject)
    result = _prepare(scanner, output, path, settings)
    assert result["files"] == []
    assert result["html_paths"] == [str(path)]
    assert result["error_kinds"] == {str(path): "preparation"}
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()


def test_plain_archived_mail_is_preserved_without_network(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(
        tmp_path, monkeypatch, messages={b"42": _raw("Plain <html>literal</html>", "plain")}
    )
    scanner.reprocess_cached_mails(str(output))
    archive = next((tmp_path / "originals").rglob("*.json"))
    before = path.read_bytes(), archive.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(scanner, "connect", lambda: pytest.fail("Plain archive needs no network"))
    result = _prepare(scanner, output, path, settings, mode="all")
    assert result["skipped"]["plain"] == 1
    assert not result["files"]
    assert (path.read_bytes(), archive.read_bytes(), path.stat().st_mtime_ns) == before


def test_archived_empty_html_body_is_not_reported_current(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(
        tmp_path, monkeypatch, messages={b"42": _raw("<script>hidden()</script>")}
    )
    scanner.reprocess_cached_mails(str(output))
    config = {"ingestion": {"mail_archive": settings, "chunk_size": 640, "chunk_overlap": 100}}
    stored = {
        str(path): processing_file_hash(config, str(path), compute_file_hash(str(path), "content"))
    }
    archive = next((tmp_path / "originals").rglob("*.json"))
    before = path.read_bytes(), archive.read_bytes()
    result = _prepare(scanner, output, path, settings, stored_hashes=stored)
    assert result["error_kinds"] == {str(path): "empty_body"}
    assert result["skipped"]["current"] == 0
    assert (path.read_bytes(), archive.read_bytes()) == before


@pytest.mark.parametrize(
    "bad_prefix", ["Unbekannter Cachetext", "Betreff: Alt\nOrdner: Sent\n---\nBody"]
)
def test_refetch_requires_existing_header_identity_before_write(tmp_path, monkeypatch, bad_prefix):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch)
    path.write_text(bad_prefix)
    result = _prepare(scanner, output, path, settings)
    assert result["error_kinds"] == {str(path): "preparation"}
    assert path.read_text() == bad_prefix
    assert not (tmp_path / "originals").exists()


def test_html_original_over_archive_limit_is_not_written(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch)
    settings["max_body_bytes"] = 4
    before = path.read_bytes()
    result = _prepare(scanner, output, path, settings)
    assert result["error_kinds"] == {str(path): "preparation"}
    assert path.read_bytes() == before
    assert not (tmp_path / "originals").exists()


def test_corrupt_original_is_not_refetched_or_overwritten(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch)
    _prepare(scanner, output, path, settings)
    archive = next((tmp_path / "originals").rglob("*.json"))
    record = json.loads(archive.read_text())
    record["body_hash"] = "corrupt"
    archive.write_text(json.dumps(record))
    before = path.read_bytes(), archive.read_bytes()
    monkeypatch.setattr(
        scanner, "connect", lambda: pytest.fail("Invalid identity must fail closed")
    )
    result = _prepare(scanner, output, path, settings, mode="all")
    assert result["error_kinds"] == {str(path): "preparation"}
    assert not result["files"]
    assert (path.read_bytes(), archive.read_bytes()) == before


@pytest.mark.parametrize("mutation", ["symlink", "unregistered", "ambiguous"])
def test_unknown_or_ambiguous_identity_never_fetches(tmp_path, monkeypatch, mutation):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    registry_path = output / ".uid_registry.json"
    registry = json.loads(registry_path.read_text())
    if mutation == "symlink":
        other = tmp_path / "other.txt"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    elif mutation == "unregistered":
        registry[scanner._folder_key("INBOX")]["uids"].clear()
    else:
        registry[scanner._folder_key("INBOX")]["uids"]["43"] = str(path.relative_to(output))
    registry_path.write_text(json.dumps(registry))
    before = path.read_bytes()
    result = _prepare(scanner, output, path, settings)
    assert result["error_kinds"] == {str(path): "preparation"}
    assert path.read_bytes() == before
    assert not conn.calls


def test_one_missing_original_does_not_block_another_html_mail(tmp_path, monkeypatch):
    from kiara.ingestion.mail_migration import prepare_html_mails

    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch, messages={b"43": _raw()})
    second = path.with_name("20010101_43_second.txt")
    second.write_bytes(path.read_bytes())
    registry_path = output / ".uid_registry.json"
    registry = json.loads(registry_path.read_text())
    registry[scanner._folder_key("INBOX")]["uids"]["43"] = str(second.relative_to(output))
    registry_path.write_text(json.dumps(registry))
    progress = []
    result = prepare_html_mails(
        scanner,
        str(output),
        [str(path), str(second)],
        {"ingestion": {"mail_archive": settings}},
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert result["files"] == [str(second)]
    assert result["error_kinds"] == {str(path): "missing_original"}
    assert result["examined"] == 2
    assert progress == [(1, 2), (2, 2)]


def test_progress_failure_propagates_after_preparation(tmp_path, monkeypatch):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="Jobsteuerung"):
        _prepare(
            scanner,
            output,
            path,
            settings,
            progress_callback=Mock(side_effect=RuntimeError("Jobsteuerung fehlgeschlagen")),
        )
    assert "Ventil V42" in path.read_text()
    assert conn.calls[-1] == ("logout",)


@pytest.mark.parametrize("connector_kind", ["imap", "exchange"])
def test_connector_html_preparation_preserves_runtime_and_callbacks(
    tmp_path, monkeypatch, connector_kind
):
    import kiara.connectors.base as base
    import kiara.ingestion.imap_scanner as imap_module
    import kiara.ingestion.mail_migration as migration
    import kiara.ingestion.oauth2_imap_scanner as exchange_module
    from kiara.connectors.exchange_connector import ExchangeOnlineConnector
    from kiara.connectors.imap_connector import IMAPConnector

    connector = IMAPConnector if connector_kind == "imap" else ExchangeOnlineConnector
    module = imap_module if connector_kind == "imap" else exchange_module
    scanner_name = "IMAPScanner" if connector_kind == "imap" else "OAuth2IMAPScanner"
    scanner = Mock()
    factory = Mock(return_value=scanner)
    monkeypatch.setattr(module, scanner_name, factory)
    monkeypatch.setattr(base, "mail_cache_path", lambda _id: str(tmp_path))
    prepare = Mock(return_value={"files": []})
    monkeypatch.setattr(migration, "prepare_html_mails", prepare)
    config = {
        "ingestion": {"mail_archive": {"path": str(tmp_path / "archive"), "max_body_bytes": 4096}}
    }
    paths = [str(tmp_path / "mail.txt")]
    progress, control = Mock(), Mock()
    options = {
        "config": config,
        "mode": "all",
        "stored_hashes": {paths[0]: "old"},
        "hash_mode": "content",
    }
    result = connector.reprocess_cached(
        {"username": "mail@example.test"},
        {"id": 42},
        config,
        cache_paths=paths,
        progress_callback=progress,
        control_callback=control,
        html_options=options,
    )
    assert result is prepare.return_value
    prepare.assert_called_once_with(
        scanner,
        str(tmp_path),
        paths,
        config,
        mode="all",
        stored_hashes={paths[0]: "old"},
        hash_mode="content",
        progress_callback=progress,
        control_callback=control,
    )
    assert options["config"] is config
    assert factory.call_args.args[0]["mail_archive"] == config["ingestion"]["mail_archive"]
    scanner.reprocess_cached_mails.assert_not_called()
    scanner.scan.assert_not_called()
