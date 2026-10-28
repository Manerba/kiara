"""Tests fuer IMAP-Anhang-Indexierung: _sanitize_text(), PDF-Null-Byte-Handling,
scan_files() mit Attachment-Verzeichnissen, ds_index_attachments-Override,
build_datasource_overrides(), DB-Migration.
"""

import os
from unittest.mock import patch

import kiara.ingestion.loader as loader_module
from kiara.ingestion._file_utils import scan_files
from kiara.ingestion.datasource_runtime import build_datasource_overrides
from kiara.ingestion.loader import _sanitize_text, load_file
from kiara.ingestion.pipeline import _apply_ds_overrides

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_file(tmp_path, name, content="Dummy content for testing"):
    """Erstelle eine temporaere Datei mit gegebenem Inhalt."""
    f = tmp_path / name
    f.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        f.write_bytes(content)
    else:
        f.write_text(content)
    return str(f)


# ── 1. TestSanitizeText ─────────────────────────────────────────────────────


class TestSanitizeText:
    """Tests fuer den _sanitize_text() Helper."""

    def test_clean_text_unchanged(self):
        """Text ohne Null-Bytes wird unveraendert zurueckgegeben."""
        result = _sanitize_text("Normaler Text", "test.pdf", "standard")
        assert result == "Normaler Text"

    def test_few_null_bytes_sanitized(self):
        """Wenige Null-Bytes (<5%) werden entfernt."""
        text = "Langer Text " * 20 + "\x00"  # 1 Null-Byte bei ~260 Zeichen
        result = _sanitize_text(text, "test.pdf", "standard")
        assert result is not None
        assert "\x00" not in result

    def test_many_null_bytes_rejected(self):
        """Hoher Null-Byte-Anteil (>5%) → return None."""
        text = "A\x00" * 50  # 50% Null-Bytes
        result = _sanitize_text(text, "binary.dat", "standard")
        assert result is None

    def test_boundary_5_percent(self):
        """Genau an der 5%-Grenze: 5 Null-Bytes bei 100 Zeichen → abgelehnt."""
        # 95 normale Zeichen + 5 Null-Bytes = 100 Zeichen, 5% → ueber Schwelle
        text = "A" * 94 + "\x00" * 6  # 6% → abgelehnt
        result = _sanitize_text(text, "test.pdf", "standard")
        assert result is None

    def test_just_below_boundary(self):
        """Knapp unter 5%: 4 Null-Bytes bei 100 Zeichen → bereinigt."""
        text = "A" * 96 + "\x00" * 4  # 4% → OK
        result = _sanitize_text(text, "test.pdf", "standard")
        assert result is not None
        assert "\x00" not in result
        assert len(result) == 96

    def test_empty_string(self):
        """Leerer String ohne Null-Bytes → unveraendert."""
        result = _sanitize_text("", "test.pdf", "standard")
        assert result == ""


# ── 2. TestPdfNullByteHandling ──────────────────────────────────────────────


class TestPdfNullByteHandling:
    """Tests fuer load_file() mit PDF-typischen Null-Byte-Artefakten."""

    def test_standard_path_pdf_with_few_nulls(self, tmp_path):
        """Standard-Loader: PDF mit wenigen Null-Bytes → bereinigter Text."""
        path = _make_file(tmp_path, "invoice.pdf")
        text_with_nulls = (
            "Rechnung Nr. 12345 - Firma GmbH\x00\nBetrag: 100,00 EUR netto zzgl. MwSt."
        )

        with patch.dict(loader_module._LOADERS, {".pdf": lambda p: text_with_nulls}):
            result = load_file(path)

        assert result is not None
        assert "\x00" not in result["text"]
        assert "Rechnung Nr. 12345" in result["text"]

    @patch("kiara.ingestion.loader._load_with_docling")
    def test_ddu_path_pdf_with_few_nulls(self, mock_docling, tmp_path):
        """DDU-Pfad: PDF mit wenigen Null-Bytes → bereinigter Text."""
        path = _make_file(tmp_path, "scan.pdf")
        mock_docling.return_value = (
            "Gescannter Inhalt mit OCR-Artefakten\x00 aus dem Dokumentenscanner",
            "",
            "docling-remote",
        )

        result = load_file(
            path,
            ddu_enabled=True,
            ddu_backends=[
                {
                    "url": "http://ddu:5001",
                    "is_local": False,
                    "model": "docling",
                    "table_mode": "accurate",
                    "do_ocr": True,
                    "ocr_lang": ["de", "en"],
                    "timeout": 120.0,
                    "concurrency_limit": None,
                }
            ],
        )

        assert result is not None
        assert "\x00" not in result["text"]
        assert "Gescannter Inhalt" in result["text"]

    def test_real_binary_pdf_rejected(self, tmp_path):
        """Echte Binaerdatei (viele Null-Bytes) → return None."""
        path = _make_file(tmp_path, "corrupt.pdf")
        binary_text = "\x00\x01\x02" * 100  # 33% Null-Bytes

        with patch.dict(loader_module._LOADERS, {".pdf": lambda p: binary_text}):
            result = load_file(path)

        assert result is None


# ── 3. TestScanFilesFindsAttachments ────────────────────────────────────────


class TestScanFilesFindsAttachments:
    """Tests fuer scan_files() mit IMAP-Attachment-Verzeichnissen."""

    def test_scan_finds_attachments(self, tmp_path):
        """scan_files() findet Dateien in attachments/-Unterverzeichnissen."""
        # IMAP-typische Struktur simulieren
        _make_file(tmp_path, "INBOX/mail_001.txt", "E-Mail-Text")
        _make_file(tmp_path, "INBOX/attachments/20251130_Invoice.pdf", "PDF")

        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt", ".pdf"],
                "exclude_patterns": [],
            },
        }
        files = scan_files(config)
        basenames = [os.path.basename(f) for f in files]
        assert "mail_001.txt" in basenames
        assert "20251130_Invoice.pdf" in basenames

    def test_scan_excludes_attachments_pattern(self, tmp_path):
        """scan_files() mit */attachments/* Pattern schliesst Anhaenge aus."""
        _make_file(tmp_path, "INBOX/mail_001.txt", "E-Mail-Text")
        _make_file(tmp_path, "INBOX/attachments/20251130_Invoice.pdf", "PDF")

        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt", ".pdf"],
                "exclude_patterns": ["*/attachments/*"],
            },
        }
        files = scan_files(config)
        basenames = [os.path.basename(f) for f in files]
        assert "mail_001.txt" in basenames
        assert "20251130_Invoice.pdf" not in basenames

    def test_scan_excludes_connector_metadata_from_datasource_root(self, tmp_path):
        """IMAP-Steuerdateien werden auch bei indexierbarer Endung nie erfasst."""
        _make_file(tmp_path, ".uid_registry.json", '{"INBOX": {"uids": {}}}')
        _make_file(tmp_path, ".uid_registry.json.tmp", '{"INBOX": {"uids": {}}}')
        _make_file(tmp_path, ".scan_done", "2026-08-29T16:36:19")
        _make_file(tmp_path, ".customer.json", '{"customer": "legitimate"}')
        _make_file(tmp_path, "INBOX/.uid_registry.json", '{"document": "legitimate"}')

        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": ["", ".json", ".tmp"],
                "exclude_patterns": [],
            },
        }

        relative_files = {os.path.relpath(path, tmp_path) for path in scan_files(config)}
        assert ".uid_registry.json" not in relative_files
        assert ".uid_registry.json.tmp" not in relative_files
        assert ".scan_done" not in relative_files
        assert ".customer.json" in relative_files
        assert os.path.join("INBOX", ".uid_registry.json") in relative_files


# ── 4. TestIndexAttachmentsOverride ─────────────────────────────────────────


class TestIndexAttachmentsOverride:
    """Tests fuer ds_index_attachments Override in der Pipeline."""

    def test_index_attachments_true_no_exclude(self):
        """index_attachments=True → keine zusaetzlichen Exclude-Patterns."""
        config = {
            "datasource": {
                "path": "/tmp/test",
                "file_types": [".pdf", ".txt"],
                "exclude_patterns": [],
            },
            "ingestion": {"chunk_size": 768, "chunk_overlap": 150},
        }
        _apply_ds_overrides(config, {"index_attachments": True})
        assert "*/attachments/*" not in config["datasource"]["exclude_patterns"]

    def test_index_attachments_false_adds_exclude(self):
        """index_attachments=False → */attachments/* wird zu Excludes hinzugefuegt."""
        config = {
            "datasource": {
                "path": "/tmp/test",
                "file_types": [".pdf", ".txt"],
                "exclude_patterns": ["*.bak"],
            },
            "ingestion": {"chunk_size": 768, "chunk_overlap": 150},
        }
        _apply_ds_overrides(config, {"index_attachments": False})
        assert "*/attachments/*" in config["datasource"]["exclude_patterns"]
        assert "*.bak" in config["datasource"]["exclude_patterns"]  # bestehende bleiben

    def test_index_attachments_default_no_exclude(self):
        """Ohne index_attachments-Key → Standard-Verhalten (indexieren)."""
        config = {
            "datasource": {
                "path": "/tmp/test",
                "file_types": [".pdf", ".txt"],
                "exclude_patterns": [],
            },
            "ingestion": {"chunk_size": 768, "chunk_overlap": 150},
        }
        _apply_ds_overrides(config, {"ddu_enabled": False})
        assert "*/attachments/*" not in config["datasource"]["exclude_patterns"]

    def test_no_overrides_no_exclude(self):
        """Keine Overrides → Standard-Verhalten."""
        config = {
            "datasource": {
                "path": "/tmp/test",
                "file_types": [".pdf", ".txt"],
                "exclude_patterns": [],
            },
            "ingestion": {"chunk_size": 768, "chunk_overlap": 150},
        }
        _apply_ds_overrides(config, None)
        assert "*/attachments/*" not in config["datasource"]["exclude_patterns"]


# ── 5. TestImapAttachmentEndToEnd ───────────────────────────────────────────


class TestImapAttachmentEndToEnd:
    """End-to-End: Attachment-Dateien werden korrekt gefunden und geladen."""

    def test_attachment_pdf_loaded_successfully(self, tmp_path):
        """PDF-Anhang mit typischen Null-Byte-Artefakten wird erfolgreich geladen."""
        path = _make_file(tmp_path, "INBOX/attachments/20251130_Invoice.pdf", "Dummy")
        text_with_nulls = "Invoice OUZSP0UW-0004\x00\nTotal: 150.00\x00 EUR"

        with patch.dict(loader_module._LOADERS, {".pdf": lambda p: text_with_nulls}):
            result = load_file(path)

        assert result is not None
        assert "Invoice OUZSP0UW-0004" in result["text"]
        assert "\x00" not in result["text"]
        assert result["metadata"]["file_type"] == ".pdf"

    def test_e2e_scan_and_load_with_attachments(self, tmp_path):
        """Gesamtflow: scan_files() → load_file() fuer IMAP-Struktur."""
        _make_file(tmp_path, "INBOX/msg_001.txt", "Betreff: Test-Mail")
        _make_file(tmp_path, "INBOX/attachments/report.pdf", "Dummy")

        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt", ".pdf"],
                "exclude_patterns": [],
            },
        }
        files = scan_files(config)
        assert len(files) == 2

        # Lade beide Dateien
        results = []
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext == ".pdf":
                with patch.dict(loader_module._LOADERS, {".pdf": lambda p: "PDF-Inhalt"}):
                    r = load_file(f)
            else:
                r = load_file(f)
            if r:
                results.append(r)

        assert len(results) == 2

    def test_e2e_scan_with_index_attachments_disabled(self, tmp_path):
        """index_attachments=False → Anhaenge werden nicht gescannt."""
        _make_file(tmp_path, "INBOX/msg_001.txt", "Betreff: Test-Mail")
        _make_file(tmp_path, "INBOX/attachments/report.pdf", "Dummy")

        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt", ".pdf"],
                "exclude_patterns": [],
            },
            "ingestion": {"chunk_size": 768, "chunk_overlap": 150},
        }
        _apply_ds_overrides(config, {"index_attachments": False})
        files = scan_files(config)

        basenames = [os.path.basename(f) for f in files]
        assert "msg_001.txt" in basenames
        assert "report.pdf" not in basenames


# ── 6. TestBuildDsOverrides (pipeline.py) ────────────────────────────────────


class TestBuildDsOverrides:
    """Tests fuer build_datasource_overrides() aus pipeline.py (DB-Row → Override-Dict)."""

    def _make_row(self, **kwargs):
        """Erstellt ein dict-aehnliches Row-Objekt mit Datasource-Feldern."""
        defaults = {
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": "auto",
            "ds_ddu_enabled": False,
            "ds_vlm_enabled": False,
            "ds_index_attachments": 1,
            "ds_accept_all_text": 0,
            "ds_contextual_retrieval": 0,
            "ds_enrichment_prompt_id": None,
            "ds_enrichment_backend_id": None,
            "restriction_level": "anonymize",
        }
        defaults.update(kwargs)
        return defaults

    def test_defaults_no_feature_overrides(self):
        """Standard-Werte → keine Feature-Overrides (nur restriction_level)."""
        row = self._make_row()
        overrides = build_datasource_overrides(row)
        assert "chunk_size" not in overrides
        assert "ddu_enabled" not in overrides
        assert "vlm_enabled" not in overrides
        assert "index_attachments" not in overrides
        assert "accept_all_text" not in overrides
        assert "contextual_retrieval" not in overrides

    def test_index_attachments_disabled(self):
        """ds_index_attachments=0 → index_attachments=False im Override."""
        row = self._make_row(ds_index_attachments=0)
        overrides = build_datasource_overrides(row)
        assert overrides["index_attachments"] is False

    def test_index_attachments_enabled(self):
        """ds_index_attachments=1 → kein index_attachments Key (Standard = True)."""
        row = self._make_row(ds_index_attachments=1)
        overrides = build_datasource_overrides(row)
        assert overrides is not None  # restriction_level wird immer gesetzt
        assert "index_attachments" not in overrides

    def test_index_attachments_combined_with_ddu(self):
        """index_attachments + ddu_enabled zusammen gesetzt."""
        row = self._make_row(ds_index_attachments=0, ds_ddu_enabled=True)
        overrides = build_datasource_overrides(row)
        assert overrides["index_attachments"] is False
        assert overrides["ddu_enabled"] is True

    def test_index_attachments_missing_column_fallback(self):
        """Alte DB ohne ds_index_attachments → Fallback auf aktiv (kein Override)."""
        # Row ohne ds_index_attachments Key — KeyError-Fallback in build_datasource_overrides
        row = {
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": "auto",
            "ds_ddu_enabled": False,
            "ds_vlm_enabled": False,
            "ds_accept_all_text": 0,
            "ds_contextual_retrieval": 0,
            "ds_enrichment_prompt_id": None,
            "ds_enrichment_backend_id": None,
            "restriction_level": "anonymize",
        }

        overrides = build_datasource_overrides(row)
        assert "index_attachments" not in overrides

    def test_all_overrides_set(self):
        """Alle Felder gleichzeitig gesetzt → alle Keys im Override-Dict."""
        import json as _json

        row = self._make_row(
            ds_chunk_size=500,
            ds_chunk_overlap=100,
            ds_chunk_strategy="email",
            ds_file_types=_json.dumps([".pdf", ".txt"]),
            ds_exclude_patterns=_json.dumps(["*.bak"]),
            ds_ddu_enabled=True,
            ds_vlm_enabled=True,
            ds_index_attachments=0,
        )
        overrides = build_datasource_overrides(row)
        assert overrides["chunk_size"] == 500
        assert overrides["chunk_overlap"] == 100
        assert overrides["chunk_strategy"] == "email"
        assert overrides["file_types"] == [".pdf", ".txt"]
        assert overrides["exclude_patterns"] == ["*.bak"]
        assert overrides["ddu_enabled"] is True
        assert overrides["vlm_enabled"] is True
        assert overrides["index_attachments"] is False


# Die urspruenglichen TestMigrateDbIndexAttachments-Tests wurden entfernt:
# Sie importierten aus kiara.web.db_migrations (Modul existiert nicht mehr);
# ds_index_attachments ist heute Teil der initialen Schema-Erstellung (schema.sql),
# ein separates Migrations-Helfer-Modul wird nicht mehr gepflegt.
