"""Tests fuer Sprint 09: Workflows Admin-Funktionen.

Testet workflows.templates.list, .templates.detail, .generate,
.summarize, .export, .exports.list, .exports.delete — 12 Unit-Tests.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# workflows.templates.list
# ---------------------------------------------------------------------------


class TestWorkflowsTemplatesList:
    @patch("kiara.workflows.template_engine.TemplateEngine.list_templates")
    @patch("kiara.config.load_config")
    def test_templates_list(self, mock_config, mock_list):
        """Test 1: Templates auflisten."""
        mock_config.return_value = {}
        mock_list.return_value = [
            {"name": "checklist", "description": "Checkliste", "variables": ["topic"]},
            {"name": "protocol", "description": "Protokoll", "variables": ["title"]},
        ]

        from kiara.admin.functions.workflows import workflows_templates_list

        result = _run(workflows_templates_list(MOCK_USER))

        assert result.success is True
        assert len(result.data["templates"]) == 2


# ---------------------------------------------------------------------------
# workflows.templates.detail
# ---------------------------------------------------------------------------


class TestWorkflowsTemplatesDetail:
    @patch("kiara.config.get_template_dir")
    @patch("kiara.workflows.template_engine.TemplateEngine.list_templates")
    @patch("kiara.config.load_config")
    def test_template_detail_found(self, mock_config, mock_list, mock_dir):
        """Test 2: Template-Details mit Inhalt."""
        mock_config.return_value = {}
        mock_list.return_value = [
            {"name": "checklist.md", "description": "Checkliste", "variables": ["topic"]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_dir.return_value = tmpdir
            (Path(tmpdir) / "checklist.md").write_text("# {{ topic }}", encoding="utf-8")

            from kiara.admin.functions.workflows import workflows_templates_detail

            result = _run(workflows_templates_detail(MOCK_USER, name="checklist.md"))

        assert result.success is True
        assert result.data["content"] == "# {{ topic }}"
        assert result.data["name"] == "checklist.md"

    @patch("kiara.workflows.template_engine.TemplateEngine.list_templates")
    @patch("kiara.config.load_config")
    def test_template_detail_not_found(self, mock_config, mock_list):
        """Test 3: Nicht existentes Template → fail."""
        mock_config.return_value = {}
        mock_list.return_value = []

        from kiara.admin.functions.workflows import workflows_templates_detail

        result = _run(workflows_templates_detail(MOCK_USER, name="nonexistent"))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# workflows.generate
# ---------------------------------------------------------------------------


class TestWorkflowsGenerate:
    @patch(
        "kiara.workflows.template_engine.TemplateEngine.generate_checklist", new_callable=AsyncMock
    )
    @patch("kiara.config.load_config")
    def test_generate_checklist(self, mock_config, mock_gen):
        """Test 4: Checkliste generieren."""
        mock_config.return_value = {}
        mock_gen.return_value = "- [ ] Punkt 1\n- [ ] Punkt 2"

        from kiara.admin.functions.workflows import workflows_generate

        result = _run(
            workflows_generate(
                MOCK_USER,
                template="checklist",
                variables={"topic": "Backup"},
            )
        )

        assert result.success is True
        assert "Punkt 1" in result.data["content"]
        assert result.data["template"] == "checklist"

    @patch("kiara.workflows.template_engine.TemplateEngine.render_template", new_callable=AsyncMock)
    @patch("kiara.config.load_config")
    def test_generate_custom(self, mock_config, mock_render):
        """Test 5: Benutzerdefiniertes Template rendern."""
        mock_config.return_value = {}
        mock_render.return_value = "Bericht: Sicherheit"

        from kiara.admin.functions.workflows import workflows_generate

        result = _run(
            workflows_generate(
                MOCK_USER,
                template="bericht",
                variables={"title": "Sicherheit"},
            )
        )

        assert result.success is True
        assert result.data["content"] == "Bericht: Sicherheit"

    @patch(
        "kiara.workflows.template_engine.TemplateEngine.generate_checklist", new_callable=AsyncMock
    )
    @patch("kiara.config.load_config")
    def test_generate_failure(self, mock_config, mock_gen):
        """Test 6: Generierung fehlgeschlagen → fail."""
        mock_config.return_value = {}
        mock_gen.side_effect = Exception("LLM timeout")

        from kiara.admin.functions.workflows import workflows_generate

        result = _run(workflows_generate(MOCK_USER, template="checklist"))

        assert result.success is False
        assert "fehlgeschlagen" in result.message


# ---------------------------------------------------------------------------
# workflows.summarize
# ---------------------------------------------------------------------------


class TestWorkflowsSummarize:
    @patch("kiara.workflows.summarizer.DocumentSummarizer.summarize", new_callable=AsyncMock)
    @patch("kiara.retrieval.retriever.Retriever")
    @patch("kiara.config.load_config")
    def test_summarize_success(self, mock_config, mock_retriever_cls, mock_summarize):
        """Test 7: Zusammenfassung erstellt."""
        mock_config.return_value = {}
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"text": "Absatz 1", "metadata": {"source": "doc1.pdf"}},
            {"text": "Absatz 2", "metadata": {"source": "doc2.pdf"}},
        ]
        mock_retriever_cls.return_value = mock_retriever
        mock_summarize.return_value = "Zusammenfassung der Dokumente."

        from kiara.admin.functions.workflows import workflows_summarize

        result = _run(workflows_summarize(MOCK_USER, source="Datenschutz"))

        assert result.success is True
        assert result.data["summary"] == "Zusammenfassung der Dokumente."
        assert len(result.data["sources"]) == 2
        assert result.data["chunk_count"] == 2

    @patch("kiara.retrieval.retriever.Retriever")
    @patch("kiara.config.load_config")
    def test_summarize_no_results(self, mock_config, mock_retriever_cls):
        """Test 8: Keine Chunks gefunden → fail."""
        mock_config.return_value = {}
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        from kiara.admin.functions.workflows import workflows_summarize

        result = _run(workflows_summarize(MOCK_USER, source="gibberish"))

        assert result.success is False
        assert "Keine Dokumente" in result.message


# ---------------------------------------------------------------------------
# workflows.export
# ---------------------------------------------------------------------------


class TestWorkflowsExport:
    @patch("kiara.workflows.exporter.DocumentExporter.export", new_callable=AsyncMock)
    @patch("kiara.config.load_config")
    def test_export_success(self, mock_config, mock_export):
        """Test 9: Dokument exportieren."""
        mock_config.return_value = {}
        mock_export.return_value = "/var/lib/kiara/exports/bericht.pdf"

        from kiara.admin.functions.workflows import workflows_export

        result = _run(
            workflows_export(
                MOCK_USER,
                content="# Bericht",
                filename="bericht",
                format="pdf",
            )
        )

        assert result.success is True
        assert result.data["filename"] == "bericht.pdf"
        assert result.data["format"] == "pdf"


# ---------------------------------------------------------------------------
# workflows.exports.list
# ---------------------------------------------------------------------------


class TestWorkflowsExportsList:
    @patch("kiara.workflows.exporter.DocumentExporter.list_exports")
    @patch("kiara.config.load_config")
    def test_exports_list(self, mock_config, mock_list):
        """Test 10: Exportierte Dateien auflisten."""
        mock_config.return_value = {}
        mock_list.return_value = [
            {"filename": "bericht.pdf", "format": "pdf", "size_bytes": 1024},
        ]

        from kiara.admin.functions.workflows import workflows_exports_list

        result = _run(workflows_exports_list(MOCK_USER))

        assert result.success is True
        assert len(result.data["exports"]) == 1


# ---------------------------------------------------------------------------
# workflows.exports.delete
# ---------------------------------------------------------------------------


class TestWorkflowsExportsDelete:
    @patch("kiara.workflows.exporter.DocumentExporter")
    @patch("kiara.config.load_config")
    def test_exports_delete_success(self, mock_config, mock_exporter_cls):
        """Test 11: Export loeschen."""
        mock_config.return_value = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bericht.pdf"
            test_file.write_bytes(b"PDF content")

            mock_exporter_cls.return_value.output_dir = tmpdir

            from kiara.admin.functions.workflows import workflows_exports_delete

            result = _run(workflows_exports_delete(MOCK_USER, filename="bericht.pdf"))

            assert result.success is True
            assert not test_file.exists()

    @patch("kiara.workflows.exporter.DocumentExporter")
    @patch("kiara.config.load_config")
    def test_exports_delete_traversal(self, mock_config, mock_exporter_cls):
        """Test 12: Pfad-Traversal → fail."""
        mock_config.return_value = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_exporter_cls.return_value.output_dir = tmpdir

            from kiara.admin.functions.workflows import workflows_exports_delete

            result = _run(workflows_exports_delete(MOCK_USER, filename="../../../etc/passwd"))

        assert result.success is False
