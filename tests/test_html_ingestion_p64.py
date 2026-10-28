"""Gemeinsamer HTML-Pfad und unveränderte Provenienz bis zum Chunk."""

from email.message import EmailMessage
from pathlib import Path

import pytest

from kiara.ingestion.chunk_strategy import detect_chunk_params
from kiara.ingestion.chunker import chunk_document
from kiara.ingestion.imap_scanner import IMAPScanner
from kiara.ingestion.loader import load_file
from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner


def test_html_file_is_markdown_with_conversion_provenance(tmp_path):
    path = tmp_path / "auftrag.html"
    path.write_text("<h1>Auftrag</h1><p>Bauteil: Ventil</p><ul><li>24 V</li></ul>")
    doc = load_file(str(path))
    assert doc is not None
    assert "# Auftrag" in doc["text"]
    assert doc["metadata"]["content_format"] == "markdown"
    assert doc["metadata"]["source_mime_type"] == "text/html"
    assert doc["metadata"]["parser"] == "html-to-markdown"
    assert doc["metadata"]["html_parser_revision"]
    assert len(doc["metadata"]["html_source_hash"]) == 64
    size, overlap = detect_chunk_params(doc["text"], doc["metadata"])
    assert doc["metadata"]["chunk_strategy"] == "prose"
    chunks = chunk_document(doc["text"], doc["metadata"], size, overlap)
    assert chunks
    for chunk in chunks:
        assert chunk["text"] == doc["text"][chunk["char_start"] : chunk["char_end"]]
        assert chunk["metadata"]["html_source_hash"] == doc["metadata"]["html_source_hash"]


def test_html_extension_htm_uses_same_converter(tmp_path):
    path = tmp_path / "hinweis.htm"
    path.write_text("<h2>Hinweis</h2><p>Nicht einschalten.</p>")
    doc = load_file(str(path))
    assert doc is not None
    assert doc["metadata"]["content_format"] == "markdown"
    assert doc["text"].startswith("## Hinweis")


@pytest.mark.parametrize("scanner_cls", [IMAPScanner, OAuth2IMAPScanner])
def test_mail_original_to_cache_loader_and_chunk_metadata(tmp_path, scanner_cls):
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 10000}
    scanner = scanner_cls({"mail_archive": archive})
    mail = EmailMessage()
    mail["Subject"] = "Prüfanweisung"
    mail.set_content("<h1>Prüfung</h1><p>Nicht einschalten.</p>", subtype="html")
    paths = scanner._save_mail(mail.as_bytes(), "INBOX", str(tmp_path / "cache"), uid="81")
    doc = load_file(paths[0], mail_archive=archive)
    assert doc is not None
    assert "# Prüfung" in doc["text"]
    assert "Nicht einschalten." in doc["text"]
    assert doc["metadata"]["content_format"] == "markdown"
    assert doc["metadata"]["source_mime_type"] == "text/html"
    assert doc["metadata"]["html_source_hash"]
    assert doc["metadata"]["mail_cache_hash"]
    size, overlap = detect_chunk_params(doc["text"], doc["metadata"])
    assert doc["metadata"]["chunk_strategy"] == "email"
    chunks = chunk_document(doc["text"], doc["metadata"], size, overlap)
    assert chunks
    for chunk in chunks:
        assert chunk["text"] == doc["text"][chunk["char_start"] : chunk["char_end"]]
        assert chunk["metadata"]["html_output_hash"] == doc["metadata"]["html_output_hash"]


def test_loader_rejects_modified_mail_cache_with_stale_provenance(tmp_path):
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 10000}
    scanner = IMAPScanner({"mail_archive": archive})
    mail = EmailMessage()
    mail.set_content("<p>Freigabe fehlt.</p>", subtype="html")
    path = scanner._save_mail(mail.as_bytes(), "INBOX", str(tmp_path / "cache"), uid="82")[0]
    with open(path, "a") as stream:
        stream.write("Zusatz ohne passende Provenienz")
    assert load_file(path, mail_archive=archive) is None


@pytest.mark.parametrize("case", ["title_body", "title_only", "reddit"])
def test_page_title_reaches_actual_full_text_and_chunk_spans(tmp_path, case):
    from kiara_ingestion_core.models import ProcessingContext

    from kiara.ingestion.runtime import _process_single_file

    title = "Reviewkennung AB-314"
    body = "Lieferung am 29. September."
    source = f"<html><head><title>{title}</title></head><body>"
    source += f"<p>{body}</p>" if case == "title_body" else ""
    source += "</body></html>"
    if case == "reddit":
        source = (
            Path(__file__).parents[1]
            / "packages/kiara-document-loaders/tests/fixtures/reddit_title.html"
        ).read_text()
        title = (
            "Requiem for a Dream: Ellen Burstyn should have took home the Oscar for "
            "Best Actress in 2000 over Julia Roberts. : r/movies"
        )
        body = "facebookexternalhit/1.1"
    path = tmp_path / "title.html"
    path.write_text(source)
    doc = load_file(str(path))
    assert doc is not None
    assert not doc["metadata"].get("no_content")
    assert title in doc["text"]
    stats = {"errors": 0}
    chunks, metadata, ids, full_text, spans = _process_single_file(
        str(path),
        {"ingestion": {"chunk_size": 640, "chunk_overlap": 100}},
        {},
        1,
        ProcessingContext(),
        stats,
    )
    assert stats["errors"] == 0
    assert title in full_text
    assert any(title in chunk for chunk in chunks)
    if case != "title_only":
        assert body in full_text
        assert any(body in chunk for chunk in chunks)
    assert len(chunks) == len(metadata) == len(ids) == len(spans)
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )


@pytest.mark.parametrize("scanner_cls", [IMAPScanner, OAuth2IMAPScanner])
def test_title_only_html_mail_reaches_canonical_cache_text(tmp_path, scanner_cls):
    archive = {"path": str(tmp_path / "originals"), "max_body_bytes": 10000}
    scanner = scanner_cls({"mail_archive": archive})
    mail = EmailMessage()
    mail.set_content(
        "<html><head><title>Freigabe AB-314</title></head><body></body></html>", subtype="html"
    )
    path = scanner._save_mail(mail.as_bytes(), "INBOX", str(tmp_path / "cache"), uid="83")[0]
    doc = load_file(path, mail_archive=archive)
    assert doc is not None
    assert "# Freigabe AB-314" in doc["text"]
    assert not doc["metadata"].get("no_content")
    assert doc["metadata"]["html_profile_revision"] == "kiara-html-7"
