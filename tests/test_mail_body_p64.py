"""MIME-Hauptbody und Originalarchiv: verbindliche Regressionen fuer Projekt #64."""

import base64
import hashlib
import json
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kiara.ingestion.imap_scanner import IMAPScanner


def _html():
    return MIMEText("<p>Ventil V42</p><p>nicht fuer Trinkwasser</p>", "html", "utf-8")


def _multipart(*parts, subtype="mixed", **params):
    msg = MIMEMultipart(subtype, **params)
    for part in parts:
        msg.attach(part)
    return msg


@pytest.mark.parametrize("singlepart", [True, False])
def test_html_main_body_is_markdown_with_paragraphs(singlepart):
    msg = _html() if singlepart else _multipart(_html())
    result = IMAPScanner._extract_body(msg)
    assert result == "Ventil V42\n\nnicht fuer Trinkwasser\n"


def test_nonempty_plain_main_body_is_preferred_unchanged():
    plain = "  Ventil V42: nicht fuer Trinkwasser.\n"
    msg = _multipart(_html(), MIMEText(plain, "plain"), subtype="alternative")
    assert IMAPScanner._extract_body(msg) == plain


def test_whitespace_plain_alternative_does_not_hide_html():
    msg = _multipart(MIMEText(" \t\r\n", "plain"), _html(), subtype="alternative")
    assert IMAPScanner._extract_body(msg) == "Ventil V42\n\nnicht fuer Trinkwasser\n"


@pytest.mark.parametrize("marker", ["attachment", "filename", "inline_filename"])
def test_text_attachment_is_never_main_body(marker):
    attachment = MIMEText("FALSCH: Text aus Anhang", "plain")
    if marker == "attachment":
        attachment.add_header("Content-Disposition", "attachment")
    elif marker == "filename":
        attachment.set_param("name", "notiz.txt")
    else:
        attachment.add_header("Content-Disposition", "inline", filename="notiz.txt")
    msg = _multipart(attachment, _html())
    assert IMAPScanner._extract_body(msg) == "Ventil V42\n\nnicht fuer Trinkwasser\n"


def test_forwarded_message_subtree_is_never_main_body():
    forwarded = MIMEMessage(MIMEText("FALSCH: weitergeleitete Mail", "plain"))
    assert IMAPScanner._extract_body(_multipart(forwarded, _html())) == (
        "Ventil V42\n\nnicht fuer Trinkwasser\n"
    )


@pytest.mark.parametrize("explicit_start", [False, True])
def test_related_uses_only_root_body(explicit_start):
    root = _html()
    root["Content-ID"] = "<body-root>"
    resource = MIMEText("FALSCH: related Ressource", "plain")
    resource["Content-ID"] = "<text-resource>"
    parts = [resource, root] if explicit_start else [root, resource]
    params = {"start": "<body-root>"} if explicit_start else {}
    assert IMAPScanner._extract_body(_multipart(*parts, subtype="related", **params)) == (
        "Ventil V42\n\nnicht fuer Trinkwasser\n"
    )


def test_related_missing_explicit_root_fails_closed():
    msg = _multipart(_html(), subtype="related", start="<missing>")
    with pytest.raises(ValueError, match="related|[Rr]oot"):
        IMAPScanner._extract_body(msg)


def _scanner(tmp_path, **archive_overrides):
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 100000}
    archive.update(archive_overrides)
    return IMAPScanner({"imap_download_attachments": False, "mail_archive": archive}), archive


def _save(scanner, tmp_path, msg=None):
    msg = msg or _html()
    msg["Subject"] = "Ventil"
    msg["Date"] = "Mon, 01 Jan 2024 10:00:00 +0100"
    return Path(scanner._save_mail(msg.as_bytes(), "INBOX", str(tmp_path / "cache"), "42")[0])


def test_selected_original_and_provenance_are_outside_indexed_cache(tmp_path):
    scanner, settings = _scanner(tmp_path)
    msg = MIMEText("<p>Grüße: V42</p>", "html", "iso-8859-1")
    original = msg.get_payload(decode=True)
    cache_path = _save(scanner, tmp_path, msg)
    archives = list((tmp_path / "originals").rglob("*.json"))
    assert len(archives) == 1
    assert list((tmp_path / "cache").rglob("*.*")) == [cache_path]
    record = json.loads(archives[0].read_text())
    assert base64.b64decode(record["body_base64"]) == original
    assert record["mime_type"] == "text/html"
    assert record["charset"] == "iso-8859-1"
    assert record["body_hash"] == hashlib.sha256(original).hexdigest()
    assert record["body_hash"] != record["metadata"]["html_source_hash"]
    from kiara.ingestion.mail_archive import read_mail_metadata

    metadata = read_mail_metadata(str(cache_path), cache_path.read_text(), settings)
    assert metadata["parser"] == "html-to-markdown"
    assert metadata["content_format"] == "markdown"
    assert metadata["source_mime_type"] == "text/html"
    assert metadata["mail_body_hash"] == record["body_hash"]
    assert metadata["mail_cache_hash"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert not {"ingested", "completed", "embedding_model"} & metadata.keys()


def test_original_archive_keeps_only_latest_body_per_cache_path(tmp_path):
    scanner, settings = _scanner(tmp_path)
    cache_path = _save(scanner, tmp_path)
    cache_path_again = _save(scanner, tmp_path, MIMEText("neu", "plain", "utf-8"))
    assert cache_path_again == cache_path
    archives = list((tmp_path / "originals").rglob("*.json"))
    assert len(archives) == 1
    record = json.loads(archives[0].read_text())
    assert base64.b64decode(record["body_base64"]) == b"neu"
    from kiara.ingestion.mail_archive import read_mail_metadata

    assert read_mail_metadata(cache_path, cache_path.read_text(), settings)["content_format"] == (
        "text"
    )


@pytest.mark.parametrize("nested", [True, False])
def test_archive_in_indexed_cache_is_rejected_before_writes(tmp_path, nested):
    root = tmp_path / "cache"
    if nested:
        root = root / "raw"
    scanner, _ = _scanner(tmp_path, path=str(root))
    with pytest.raises(ValueError, match="[Aa]rchiv|archive"):
        _save(scanner, tmp_path)
    assert not (tmp_path / "cache").exists()


def test_original_body_limit_prevents_any_cache_write(tmp_path):
    scanner, _ = _scanner(tmp_path, max_body_bytes=8)
    with pytest.raises(ValueError, match="[Ll]imit|[Gg]ro|bytes"):
        _save(scanner, tmp_path)
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "originals").exists()


def test_present_archive_with_inconsistent_cache_fails_closed(tmp_path):
    scanner, settings = _scanner(tmp_path)
    cache_path = _save(scanner, tmp_path)
    from kiara.ingestion.mail_archive import read_mail_metadata

    with pytest.raises(ValueError, match="[Hh]ash|[Pp]rovenienz|[Cc]ache"):
        read_mail_metadata(cache_path, cache_path.read_text() + "manipuliert", settings)


def test_absent_archive_is_normal_for_legacy_and_other_text_files(tmp_path):
    from kiara.ingestion.mail_archive import read_mail_metadata

    _, settings = _scanner(tmp_path)
    assert read_mail_metadata(tmp_path / "ordinary.txt", "normaler Text", settings) == {}


def test_fetch_uses_peek_to_preserve_mail_flags():
    scanner = IMAPScanner({})
    conn = MagicMock()
    conn.uid.return_value = ("OK", [(b"1 (UID 42 BODY[] {4}", b"mail"), b")"])
    assert scanner._fetch_batch(conn, [b"42"]) == [(b"42", b"mail")]
    conn.uid.assert_called_once_with("fetch", b"42", "(UID BODY.PEEK[])")


@pytest.mark.parametrize("existing", [False, True])
def test_archive_commit_error_restores_old_cache_or_removes_new_cache(
    tmp_path, monkeypatch, existing
):
    from kiara.ingestion import mail_archive

    scanner, settings = _scanner(tmp_path)
    old_path = _save(scanner, tmp_path, MIMEText("Alter Inhalt", "plain")) if existing else None
    previous = old_path.read_bytes() if old_path else None
    original_replace = mail_archive.os.replace

    def fail_archive(source, target):
        if Path(target).suffix == ".json":
            raise OSError("Simulierter Archiv-Commitfehler")
        original_replace(source, target)

    monkeypatch.setattr(mail_archive.os, "replace", fail_archive)
    with pytest.raises(OSError, match="Archiv-Commitfehler"):
        _save(scanner, tmp_path, MIMEText("Neuer Inhalt", "plain"))
    if old_path:
        assert old_path.read_bytes() == previous
        assert mail_archive.read_mail_metadata(old_path, old_path.read_text(), settings)
    else:
        assert list((tmp_path / "cache").rglob("*.txt")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


def test_conversion_error_preserves_previous_cache_and_archive(tmp_path, monkeypatch):
    from kiara_document_loaders import html

    scanner, settings = _scanner(tmp_path)
    path = _save(scanner, tmp_path, MIMEText("Alter Inhalt", "plain"))
    previous = path.read_bytes()
    archive_path = next((tmp_path / "originals").rglob("*.json"))
    previous_archive = archive_path.read_bytes()

    def fail_conversion(_source):
        raise ValueError("Simulierter Konvertierungsfehler")

    monkeypatch.setattr(html, "convert_html", fail_conversion)
    with pytest.raises(ValueError, match="Konvertierungsfehler"):
        _save(scanner, tmp_path)
    assert path.read_bytes() == previous
    assert archive_path.read_bytes() == previous_archive


@pytest.mark.parametrize("failure", ["conversion", "archive"])
def test_failed_mail_is_not_registered_as_success(tmp_path, monkeypatch, failure):
    from kiara_document_loaders import html

    from kiara.ingestion import mail_archive

    scanner, _ = _scanner(tmp_path)
    output = tmp_path / "cache"
    output.mkdir()
    conn = MagicMock()
    conn.select.return_value = ("OK", [])
    monkeypatch.setattr(
        scanner, "_fetch_batch_with_reconnect", lambda *_: ([(b"42", _html().as_bytes())], conn)
    )

    def fail(*_args, **_kwargs):
        raise ValueError("Simulierter Verarbeitungsfehler")

    if failure == "conversion":
        monkeypatch.setattr(html, "convert_html", fail)
    else:
        original_replace = mail_archive.os.replace

        def fail_archive(source, target):
            if Path(target).suffix == ".json" and Path(target).name != ".uid_registry.json":
                raise ValueError("Simulierter Verarbeitungsfehler")
            original_replace(source, target)

        monkeypatch.setattr(mail_archive.os, "replace", fail_archive)
    folder_key = scanner._folder_key("INBOX")
    registry = {folder_key: {"uidvalidity": "123", "uids": {}}}
    files, _, _ = scanner._fetch_and_save_mails(
        conn, [("INBOX", [b"42"], folder_key)], registry, str(output), None, None
    )
    assert files == []
    assert "42" not in registry[folder_key]["uids"]
    assert list(output.rglob("*.txt")) == []


@pytest.mark.parametrize("field", ["body_hash", "mime_type", "charset", "metadata"])
def test_corrupt_original_provenance_is_rejected(tmp_path, field):
    scanner, settings = _scanner(tmp_path)
    path = _save(scanner, tmp_path)
    archive_path = next((tmp_path / "originals").rglob("*.json"))
    record = json.loads(archive_path.read_text())
    record[field] = None if field == "charset" else "korrupt"
    archive_path.write_text(json.dumps(record))
    from kiara.ingestion.mail_archive import read_mail_metadata

    with pytest.raises(ValueError, match="Provenienz"):
        read_mail_metadata(path, path.read_text(), settings)


def test_output_hash_must_match_converted_body(tmp_path):
    scanner, settings = _scanner(tmp_path)
    path = _save(scanner, tmp_path)
    archive_path = next((tmp_path / "originals").rglob("*.json"))
    record = json.loads(archive_path.read_text())
    record["metadata"]["html_output_hash"] = "0" * 64
    archive_path.write_text(json.dumps(record))
    from kiara.ingestion.mail_archive import read_mail_metadata

    with pytest.raises(ValueError, match="Provenienz"):
        read_mail_metadata(path, path.read_text(), settings)


def test_plain_mail_crlf_roundtrip_accepts_loader_newline_normalization(tmp_path):
    scanner, settings = _scanner(tmp_path)
    original = "V42\r\nnicht fuer Trinkwasser\r\n"
    path = _save(scanner, tmp_path, MIMEText(original, "plain", "utf-8"))
    assert b"V42\r\n" in path.read_bytes()
    from kiara.ingestion.mail_archive import read_mail_metadata

    metadata = read_mail_metadata(path, path.read_text(), settings)
    assert metadata["mail_source_hash"] == hashlib.sha256(original.encode()).hexdigest()


def test_raw_cache_changes_cannot_hide_behind_newline_normalization(tmp_path):
    scanner, settings = _scanner(tmp_path)
    path = _save(scanner, tmp_path, MIMEText("V42\r\nzweite Zeile", "plain", "utf-8"))
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    from kiara.ingestion.mail_archive import read_mail_metadata

    with pytest.raises(ValueError, match="Provenienz"):
        read_mail_metadata(path, path.read_text(), settings)
