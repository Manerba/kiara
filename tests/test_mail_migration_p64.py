"""Mail-Bestandsaufbereitung: nur lokale Dateien und Fake-IMAP."""

import json
from email.mime.text import MIMEText
from pathlib import Path

import pytest

from kiara.ingestion.imap_scanner import IMAPScanner
from kiara.ingestion.mail_archive import read_mail_metadata
from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner


def _raw(text="<p>Ventil V42</p><p>nicht fuer Trinkwasser</p>", subtype="html"):
    msg = MIMEText(text, subtype, "utf-8")
    msg["Subject"] = "Neuer Betreff darf den Pfad nicht aendern"
    msg["Date"] = "Mon, 01 Jan 2001 10:00:00 +0100"
    return msg.as_bytes()


class FakeIMAP:
    def __init__(self, *, validity=b"123", messages=None, fetch_error=None):
        self.validity = validity
        self.messages = {b"42": _raw()} if messages is None else messages
        self.fetch_error = fetch_error
        self.calls = []
        self.flags = {b"42": {"\\Flagged"}}

    def select(self, folder, readonly=False):
        self.calls.append(("select", folder, readonly))
        assert readonly is True
        return "OK", [b"1"]

    def response(self, key):
        self.calls.append(("response", key))
        assert key == "UIDVALIDITY"
        return key, [self.validity]

    def uid(self, command, uids, query):
        self.calls.append((command, uids, query))
        assert command == "fetch"
        assert query == "(UID BODY.PEEK[])"
        if self.fetch_error:
            raise self.fetch_error
        return "OK", [
            (b"1 (UID " + uid + b" BODY[] {100}", self.messages[uid])
            for uid in uids.split(b",")
            if uid in self.messages
        ]

    def logout(self):
        self.calls.append(("logout",))


def _setup(tmp_path, monkeypatch, scanner_class=IMAPScanner, **fake_options):
    output = tmp_path / "cache"
    key = IMAPScanner._folder_key("INBOX")
    path = output / key / "20010101_42_alter-betreff.txt"
    path.parent.mkdir(parents=True)
    path.write_text("Betreff: Alt\nOrdner: INBOX\n---\nVerlustbehafteter Alttext")
    registry = {key: {"uidvalidity": "123", "uids": {"42": str(path.relative_to(output))}}}
    (output / ".uid_registry.json").write_text(json.dumps(registry))
    (output / ".scan_done").write_text("Alter abgeschlossener Scan")
    settings = {"path": str(tmp_path / "originals"), "max_body_bytes": 100000}
    scanner = scanner_class({"imap_since_days": 1, "mail_archive": settings})
    conn = FakeIMAP(**fake_options)
    monkeypatch.setattr(scanner, "connect", lambda: conn)
    return scanner, conn, output, path, settings


@pytest.mark.parametrize("scanner_class", [IMAPScanner, OAuth2IMAPScanner])
def test_known_old_uid_is_refetched_to_exact_existing_path_without_flag_changes(
    tmp_path, monkeypatch, scanner_class
):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch, scanner_class)
    registry_before = (output / ".uid_registry.json").read_bytes()
    marker_before = (output / ".scan_done").read_bytes()
    result = scanner.reprocess_cached_mails(str(output))
    assert result == {"files": [str(path)], "errors": {}, "cancelled": False}
    assert path.read_text().endswith("Ventil V42\n\nnicht fuer Trinkwasser\n")
    assert list(output.rglob("*.txt")) == [path]
    assert conn.flags == {b"42": {"\\Flagged"}}
    assert ("fetch", b"42", "(UID BODY.PEEK[])") in conn.calls
    assert (output / ".uid_registry.json").read_bytes() == registry_before
    assert (output / ".scan_done").read_bytes() == marker_before
    assert read_mail_metadata(path, path.read_text(), settings)["source_mime_type"] == "text/html"


@pytest.mark.parametrize("validity", [None, b"", b"999", b"not-a-validity"])
def test_uidvalidity_must_match_before_any_refetch(tmp_path, monkeypatch, validity):
    scanner, conn, output, path, _ = _setup(tmp_path, monkeypatch, validity=validity)
    previous = path.read_bytes()
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == []
    assert "UIDVALIDITY" in result["errors"][str(path)]
    assert path.read_bytes() == previous
    assert not any(call[0] == "fetch" for call in conn.calls)


@pytest.mark.parametrize("stored", [None, "", "123:124"])
def test_missing_or_invalid_recorded_uidvalidity_stays_open(tmp_path, monkeypatch, stored):
    scanner, conn, output, path, _ = _setup(tmp_path, monkeypatch)
    registry_path = output / ".uid_registry.json"
    registry = json.loads(registry_path.read_text())
    registry[scanner._folder_key("INBOX")]["uidvalidity"] = stored
    registry_path.write_text(json.dumps(registry))
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == []
    assert "UIDVALIDITY" in result["errors"][str(path)]
    assert not any(call[0] == "fetch" for call in conn.calls)


def test_reconnect_rechecks_uidvalidity_before_retry(tmp_path, monkeypatch):
    scanner, first, output, path, _ = _setup(
        tmp_path, monkeypatch, fetch_error=ConnectionResetError("Connection reset")
    )
    replacement = FakeIMAP(validity=b"999")
    monkeypatch.setattr(scanner, "_reconnect", lambda *_args, **_kwargs: replacement)
    previous = path.read_bytes()
    result = scanner.reprocess_cached_mails(str(output))
    assert "UIDVALIDITY" in result["errors"][str(path)]
    assert path.read_bytes() == previous
    assert any(call[0] == "fetch" for call in first.calls)
    assert not any(call[0] == "fetch" for call in replacement.calls)


def test_unregistered_or_escaping_paths_are_never_written(tmp_path, monkeypatch):
    scanner, conn, output, path, _ = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("Unveraendert")
    unknown = path.with_name("unregistered.txt")
    unknown.write_text("Auch unveraendert")
    result = scanner.reprocess_cached_mails(str(output), cache_paths=[str(outside), str(unknown)])
    assert set(result["errors"]) == {str(outside), str(unknown)}
    assert result["files"] == []
    assert outside.read_text() == "Unveraendert"
    assert unknown.read_text() == "Auch unveraendert"
    assert conn.calls == []


def test_registry_path_traversal_stays_open(tmp_path, monkeypatch):
    scanner, conn, output, _, _ = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("Unveraendert")
    registry = {
        scanner._folder_key("INBOX"): {"uidvalidity": "123", "uids": {"42": "../outside.txt"}}
    }
    (output / ".uid_registry.json").write_text(json.dumps(registry))
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == []
    assert len(result["errors"]) == 1
    assert outside.read_text() == "Unveraendert"
    assert conn.calls == []


def test_atomic_archive_commit_error_leaves_previous_mail_open(tmp_path, monkeypatch):
    scanner, _, output, path, _ = _setup(tmp_path, monkeypatch)
    previous = path.read_bytes()
    from kiara.ingestion import mail_archive

    original = mail_archive.os.replace

    def fail_archive(source, target):
        if Path(target).suffix == ".json":
            raise OSError("Archiv-Commitfehler")
        return original(source, target)

    monkeypatch.setattr(mail_archive.os, "replace", fail_archive)
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == []
    assert "Archiv-Commitfehler" in result["errors"][str(path)]
    assert path.read_bytes() == previous
    assert list(tmp_path.rglob("*.tmp")) == []


def test_missing_uid_does_not_invent_mime_or_report_success(tmp_path, monkeypatch):
    scanner, _, output, path, _ = _setup(tmp_path, monkeypatch, messages={})
    previous = path.read_bytes()
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == []
    assert str(path) in result["errors"]
    assert path.read_bytes() == previous
    assert not (tmp_path / "originals").exists()


def test_archive_repairs_interrupted_cache_without_network(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(tmp_path, monkeypatch)
    scanner.reprocess_cached_mails(str(output))
    original = path.read_bytes()
    path.write_bytes(b"abgebrochener Cacheersatz")

    def no_network():
        pytest.fail("Verifiziertes Original darf keinen Netzabruf benoetigen")

    monkeypatch.setattr(scanner, "connect", no_network)
    result = scanner.reprocess_cached_mails(str(output))
    assert result == {"files": [str(path)], "errors": {}, "cancelled": False}
    assert path.read_bytes() == original
    metadata = read_mail_metadata(path, path.read_text(), settings)
    assert not {"ingested", "completed", "embedding_model"} & metadata.keys()


def test_bad_mail_does_not_block_other_registry_uid(tmp_path, monkeypatch):
    scanner, conn, output, bad_path, _ = _setup(tmp_path, monkeypatch, messages={b"43": _raw()})
    good_path = bad_path.with_name("second-mail.txt")
    good_path.write_text(bad_path.read_text())
    registry_path = output / ".uid_registry.json"
    registry = json.loads(registry_path.read_text())
    registry[scanner._folder_key("INBOX")]["uids"]["43"] = str(good_path.relative_to(output))
    registry_path.write_text(json.dumps(registry))
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == [str(good_path)]
    assert set(result["errors"]) == {str(bad_path)}
    assert conn.calls[-1] == ("logout",)


def test_cancel_and_progress_callbacks_preserve_remaining_files(tmp_path, monkeypatch):
    scanner, _, output, path, _ = _setup(tmp_path, monkeypatch)
    second = path.with_name("second-mail.txt")
    second.write_text("Unbearbeitet")
    registry_path = output / ".uid_registry.json"
    registry = json.loads(registry_path.read_text())
    registry[scanner._folder_key("INBOX")]["uids"]["43"] = str(second.relative_to(output))
    registry_path.write_text(json.dumps(registry))
    progress = []
    controls = iter([None, "cancel"])
    result = scanner.reprocess_cached_mails(
        str(output),
        progress_callback=lambda done, total: progress.append((done, total)),
        control_callback=lambda: next(controls),
    )
    assert result == {"files": [str(path)], "errors": {}, "cancelled": True}
    assert progress == [(1, 2)]
    assert second.read_text() == "Unbearbeitet"


def test_callback_failure_propagates_and_closes_connection(tmp_path, monkeypatch):
    scanner, conn, output, _, _ = _setup(tmp_path, monkeypatch)

    def fail_callback(*_args):
        raise RuntimeError("Jobsteuerung fehlgeschlagen")

    with pytest.raises(RuntimeError, match="Jobsteuerung"):
        scanner.reprocess_cached_mails(str(output), progress_callback=fail_callback)
    assert conn.calls[-1] == ("logout",)


def test_registered_symlink_does_not_replace_target(tmp_path, monkeypatch):
    scanner, conn, output, path, _ = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("Unveraendert")
    path.unlink()
    path.symlink_to(outside)
    result = scanner.reprocess_cached_mails(str(output))
    assert str(path) in result["errors"]
    assert outside.read_text() == "Unveraendert"
    assert path.is_symlink()
    assert conn.calls == []


def test_plain_original_keeps_actual_mime_type(tmp_path, monkeypatch):
    scanner, _, output, path, settings = _setup(
        tmp_path, monkeypatch, messages={b"42": _raw("Klartext <b>bleibt</b>", "plain")}
    )
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == [str(path)]
    assert path.read_text().endswith("Klartext <b>bleibt</b>")
    assert read_mail_metadata(path, path.read_text(), settings)["source_mime_type"] == "text/plain"


def test_unusable_old_prefix_triggers_guarded_original_refetch(tmp_path, monkeypatch):
    scanner, conn, output, path, _ = _setup(tmp_path, monkeypatch)
    scanner.reprocess_cached_mails(str(output))
    archive = next((tmp_path / "originals").rglob("*.json"))
    record = json.loads(archive.read_text())
    record.pop("cache_prefix_base64")
    record.pop("cache_prefix_hash")
    archive.write_text(json.dumps(record))
    path.write_bytes(b"\xffAbbruch")
    conn.calls.clear()
    result = scanner.reprocess_cached_mails(str(output))
    assert result["files"] == [str(path)]
    assert ("response", "UIDVALIDITY") in conn.calls
    assert ("fetch", b"42", "(UID BODY.PEEK[])") in conn.calls


def test_corrupt_archived_original_stays_open(tmp_path, monkeypatch):
    scanner, conn, output, path, _ = _setup(tmp_path, monkeypatch)
    scanner.reprocess_cached_mails(str(output))
    archive = next((tmp_path / "originals").rglob("*.json"))
    record = json.loads(archive.read_text())
    record["body_hash"] = "0" * 64
    archive.write_text(json.dumps(record))
    previous = path.read_bytes()
    conn.calls.clear()
    result = scanner.reprocess_cached_mails(str(output))
    assert str(path) in result["errors"]
    assert path.read_bytes() == previous
    assert conn.calls == []


def test_retry_after_conversion_failure_processes_same_whole_file(tmp_path, monkeypatch):
    from kiara_document_loaders import html

    scanner, _, output, path, _ = _setup(tmp_path, monkeypatch)
    previous = path.read_bytes()
    original_converter = html.convert_html

    def fail_conversion(_source):
        raise ValueError("Konvertierung fehlgeschlagen")

    monkeypatch.setattr(html, "convert_html", fail_conversion)
    first = scanner.reprocess_cached_mails(str(output))
    assert first["files"] == []
    assert path.read_bytes() == previous
    monkeypatch.setattr(html, "convert_html", original_converter)
    second = scanner.reprocess_cached_mails(str(output))
    assert second == {"files": [str(path)], "errors": {}, "cancelled": False}


def test_corrupt_registry_aborts_instead_of_reporting_empty_success(tmp_path, monkeypatch):
    scanner, conn, output, _, _ = _setup(tmp_path, monkeypatch)
    (output / ".uid_registry.json").write_text("{")
    with pytest.raises(ValueError):
        scanner.reprocess_cached_mails(str(output))
    assert conn.calls == []


def _historical_setup(tmp_path, monkeypatch, folder="INBOX"):
    scanner, conn, output, path, settings = _setup(tmp_path, monkeypatch)
    scanner._folders = []  # Produktive Mailbox ist mit folders="*" konfiguriert.
    canonical = output / scanner._folder_key(folder) / path.name
    canonical.parent.mkdir(exist_ok=True)
    if canonical != path:
        path.rename(canonical)
    canonical.write_text(f"Betreff: Alt\nOrdner: {folder}\n---\nAlter Body")
    historical_key = folder.replace("/", "_")
    historical = output / historical_key / "20010101_alter-betreff.txt"
    historical.parent.mkdir()
    historical.write_bytes(canonical.read_bytes())
    registry = {
        key: {"uidvalidity": "123", "uids": {"42": str(cache.relative_to(output))}}
        for key, cache in ((historical_key, historical), (scanner._folder_key(folder), canonical))
    }
    (output / ".uid_registry.json").write_text(json.dumps(registry))
    return scanner, conn, output, historical, canonical, settings


@pytest.mark.parametrize("folder", ["INBOX", "Parent/Child"])
def test_historical_folder_uses_header_and_canonical_uid_evidence(tmp_path, monkeypatch, folder):
    scanner, conn, output, historical, canonical, _settings = _historical_setup(
        tmp_path, monkeypatch, folder
    )
    registry_before = (output / ".uid_registry.json").read_bytes()
    marker_before = (output / ".scan_done").read_bytes()
    canonical_before = canonical.read_bytes()
    result = scanner.reprocess_cached_mails(str(output), cache_paths=[str(historical)])
    assert result == {"files": [str(historical)], "errors": {}, "cancelled": False}
    assert historical.read_text().endswith("Ventil V42\n\nnicht fuer Trinkwasser\n")
    assert ("select", folder, True) in conn.calls
    assert ("fetch", b"42", "(UID BODY.PEEK[])") in conn.calls
    assert conn.flags == {b"42": {"\\Flagged"}}
    assert canonical.read_bytes() == canonical_before
    assert (output / ".uid_registry.json").read_bytes() == registry_before
    assert (output / ".scan_done").read_bytes() == marker_before


@pytest.mark.parametrize(
    "defect",
    [
        "missing_canonical",
        "different_uid",
        "different_validity",
        "missing_validity",
        "multiple_headers",
        "ambiguous_legacy",
        "ambiguous_canonical",
        "canonical_symlink",
    ],
)
def test_historical_folder_without_unique_original_evidence_stays_open(
    tmp_path, monkeypatch, defect
):
    scanner, conn, output, historical, canonical, _settings = _historical_setup(
        tmp_path, monkeypatch
    )
    registry_path = output / ".uid_registry.json"
    registry = json.loads(registry_path.read_text())
    current = registry[scanner._folder_key("INBOX")]
    if defect == "missing_canonical":
        del registry[scanner._folder_key("INBOX")]
    elif defect == "different_uid":
        current["uids"]["43"] = current["uids"].pop("42")
    elif defect == "different_validity":
        current["uidvalidity"] = "999"
    elif defect == "missing_validity":
        registry["INBOX"]["uidvalidity"] = None
    elif defect == "multiple_headers":
        historical.write_text("Ordner: INBOX\nOrdner: INBOX\n---\nAlter Body")
    elif defect == "ambiguous_legacy":
        registry["INBOX"]["uids"]["43"] = registry["INBOX"]["uids"]["42"]
    elif defect == "ambiguous_canonical":
        current["uids"]["43"] = current["uids"]["42"]
    elif defect == "canonical_symlink":
        external = tmp_path / "outside.txt"
        external.write_bytes(canonical.read_bytes())
        canonical.unlink()
        canonical.symlink_to(external)
    registry_path.write_text(json.dumps(registry))
    previous = historical.read_bytes()
    registry_before = registry_path.read_bytes()
    result = scanner.reprocess_cached_mails(str(output), cache_paths=[str(historical)])
    assert result["files"] == []
    assert str(historical) in result["errors"]
    assert historical.read_bytes() == previous
    assert registry_path.read_bytes() == registry_before
    assert conn.calls == []
