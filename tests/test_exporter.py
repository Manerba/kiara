"""Tests fuer den DocumentExporter (kiara.workflows.exporter)."""

import builtins
import os
import time

import pytest

from kiara.workflows.exporter import DocumentExporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def export_dir(tmp_path):
    """Temporaeres Export-Verzeichnis."""
    d = tmp_path / "exports"
    d.mkdir()
    return str(d)


@pytest.fixture
def config(export_dir):
    return {"workflows": {"output_dir": export_dir}}


@pytest.fixture
def exporter(config):
    return DocumentExporter(config)


@pytest.fixture
def sample_content():
    return (
        "# Testdokument\n\n"
        "## Einleitung\n\n"
        "Dies ist ein **wichtiger** Testtext.\n\n"
        "### Details\n\n"
        "- Punkt 1\n"
        "- Punkt 2\n"
        "- [ ] Aufgabe 1\n"
        "- [x] Aufgabe 2\n\n"
        "---\n\n"
        "Normaler Absatz am Ende."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportMarkdown:
    @pytest.mark.asyncio
    async def test_export_markdown(self, exporter, sample_content, export_dir):
        """Markdown-Export schreibt den Inhalt korrekt."""
        path = await exporter.export(sample_content, "test_doc", format="markdown")
        assert path.endswith(".md")
        assert os.path.exists(path)

        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "# Testdokument" in content
        assert "**wichtiger**" in content


class TestExportText:
    @pytest.mark.asyncio
    async def test_export_text(self, exporter, sample_content, export_dir):
        """Text-Export entfernt Markdown-Formatierung."""
        path = await exporter.export(sample_content, "test_doc", format="text")
        assert path.endswith(".txt")
        assert os.path.exists(path)

        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Markdown-Marker sollten entfernt sein
        assert "**" not in content
        assert "# " not in content
        # Text-Inhalt sollte erhalten bleiben
        assert "wichtiger" in content
        assert "Testdokument" in content


try:
    import reportlab  # noqa: F401

    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

try:
    import docx  # noqa: F401

    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


class TestExportPDF:
    @pytest.mark.asyncio
    @pytest.mark.skipif(not _REPORTLAB_AVAILABLE, reason="reportlab nicht installiert")
    async def test_export_pdf(self, exporter, sample_content, export_dir):
        """PDF-Export erstellt eine gueltige PDF-Datei."""
        path = await exporter.export(
            sample_content,
            "test_pdf",
            format="pdf",
            title="Mein Testdokument",
        )
        assert path.endswith(".pdf")
        assert os.path.exists(path)

        # Minimale PDF-Validierung: Datei beginnt mit %PDF
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

        # Datei hat vernuenftige Groesse
        size = os.path.getsize(path)
        assert size > 100

    @pytest.mark.asyncio
    async def test_export_pdf_missing_reportlab_fails_cleanly(
        self, exporter, sample_content, tmp_path, monkeypatch
    ):
        """Fehlendes reportlab wird als kontrollierter Exportfehler gemeldet."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("reportlab"):
                raise ImportError("reportlab unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ValueError, match="reportlab"):
            await exporter._export_pdf(sample_content, tmp_path / "missing.pdf")


class TestExportDocx:
    @pytest.mark.asyncio
    @pytest.mark.skipif(not _DOCX_AVAILABLE, reason="python-docx nicht installiert")
    async def test_export_docx(self, exporter, sample_content, export_dir):
        """DOCX-Export erstellt eine gueltige DOCX-Datei."""
        path = await exporter.export(
            sample_content,
            "test_docx",
            format="docx",
            title="DOCX Test",
        )
        assert path.endswith(".docx")
        assert os.path.exists(path)

        # DOCX ist ein ZIP — pruefen dass es ein gueltiges ZIP ist
        import zipfile

        assert zipfile.is_zipfile(path)

        # Grundlegende DOCX-Struktur pruefen
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "word/document.xml" in names

        # Datei hat vernuenftige Groesse
        size = os.path.getsize(path)
        assert size > 100


class TestListExports:
    @pytest.mark.asyncio
    async def test_list_exports(self, exporter, export_dir):
        """list_exports gibt die korrekten Dateien zurueck."""
        # Einige Testdateien erstellen
        await exporter.export("Test 1", "doc1", format="markdown")
        await exporter.export("Test 2", "doc2", format="text")

        exports = exporter.list_exports()
        assert len(exports) == 2

        filenames = {e["filename"] for e in exports}
        assert "doc1.md" in filenames
        assert "doc2.txt" in filenames

        for exp in exports:
            assert "size_bytes" in exp
            assert "created_at" in exp
            assert "format" in exp
            assert exp["size_bytes"] > 0

    def test_list_exports_empty(self, exporter):
        """Leeres Verzeichnis liefert leere Liste."""
        exports = exporter.list_exports()
        assert exports == []


class TestCleanupExports:
    @pytest.mark.asyncio
    async def test_cleanup_exports(self, exporter, export_dir):
        """Alte Dateien werden geloescht, neue bleiben."""
        # Neue Datei erstellen
        path_new = await exporter.export("Neu", "new_doc", format="markdown")

        # Alte Datei erstellen und mtime in die Vergangenheit setzen
        path_old = await exporter.export("Alt", "old_doc", format="text")
        old_time = time.time() - (31 * 86400)  # 31 Tage alt
        os.utime(path_old, (old_time, old_time))

        deleted = exporter.cleanup_exports(max_age_days=30)
        assert deleted == 1

        # Neue Datei sollte noch existieren
        assert os.path.exists(path_new)
        # Alte Datei sollte geloescht sein
        assert not os.path.exists(path_old)


class TestExportCreatesDir:
    @pytest.mark.asyncio
    async def test_export_creates_output_dir(self, tmp_path):
        """Export erstellt das output_dir wenn es nicht existiert."""
        non_existent = str(tmp_path / "nicht" / "vorhanden" / "exports")
        config = {"workflows": {"output_dir": non_existent}}
        exporter = DocumentExporter(config)

        path = await exporter.export("Test", "auto_dir", format="markdown")
        assert os.path.exists(path)
        assert os.path.isdir(non_existent)


class TestSanitizeFilename:
    def test_sanitize_removes_unsafe_chars(self):
        """Unsichere Zeichen werden aus dem Dateinamen entfernt."""
        result = DocumentExporter._sanitize_filename("../../etc/passwd")
        # Pfad-Traversal-Zeichen (/) werden entfernt/ersetzt
        assert "/" not in result
        assert "passwd" in result
        assert DocumentExporter._sanitize_filename("normal-name.txt") == "normal-name.txt"
        assert DocumentExporter._sanitize_filename("with spaces here") == "with_spaces_here"

    def test_sanitize_empty_returns_default(self):
        """Leerer Name ergibt 'export'."""
        assert DocumentExporter._sanitize_filename("") == "export"
