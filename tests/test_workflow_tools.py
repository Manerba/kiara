"""Tests fuer die Workflow-Tools (kiara.tools.workflow_tools)."""

import pytest

from kiara.tools.workflow_tools import (
    ExportDocumentTool,
    GenerateFromTemplateTool,
    SummarizeDocumentTool,
)

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockRetriever:
    """Simuliert den Retriever."""

    def __init__(self, results=None):
        self._results = results or []

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[dict]:
        return self._results


class MockSummarizer:
    """Simuliert den DocumentSummarizer."""

    def __init__(self, response: str = "Mock-Zusammenfassung"):
        self._response = response

    async def summarize(self, text, focus=None, output_format="paragraph", max_length=500):
        return self._response


class MockTemplateEngine:
    """Simuliert die TemplateEngine."""

    def __init__(self, response: str = "Mock-Template-Inhalt"):
        self._response = response
        self.last_method: str | None = None

    async def generate_checklist(self, topic, context="", num_items=10):
        self.last_method = "checklist"
        return f"# Checkliste: {topic}\n\n- [ ] Mock-Punkt"

    async def generate_protocol(self, title, context="", participants=None, date=None):
        self.last_method = "protocol"
        return f"# Protokoll: {title}\n\n**Datum:** {date}"

    async def generate_summary(self, documents, focus=None, max_length=500):
        self.last_method = "summary"
        return "Mock-Zusammenfassung"

    async def render_template(self, template_name, variables):
        self.last_method = "render"
        return self._response


class MockExporter:
    """Simuliert den DocumentExporter."""

    def __init__(self, output_path: str = "/tmp/mock_export.md"):
        self._output_path = output_path

    async def export(self, content, filename, format="markdown", title=None, metadata=None):
        return self._output_path


# ---------------------------------------------------------------------------
# Tests: Tool Definitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_summarize_tool_definition(self):
        """SummarizeDocumentTool hat korrekte Definition."""
        tool = SummarizeDocumentTool(
            retriever=MockRetriever(),
            summarizer=MockSummarizer(),
        )
        defn = tool.definition()
        assert defn.name == "summarize_document"
        assert defn.description != ""
        assert defn.parameters["type"] == "object"
        assert "source" in defn.parameters["properties"]
        assert "source" in defn.parameters["required"]
        assert "focus" in defn.parameters["properties"]
        assert "format" in defn.parameters["properties"]
        assert defn.requires_confirmation is False

    def test_generate_template_tool_definition(self):
        """GenerateFromTemplateTool hat korrekte Definition."""
        tool = GenerateFromTemplateTool(template_engine=MockTemplateEngine())
        defn = tool.definition()
        assert defn.name == "generate_from_template"
        assert defn.parameters["type"] == "object"
        assert "template" in defn.parameters["properties"]
        assert "topic" in defn.parameters["properties"]
        assert "variables" in defn.parameters["properties"]
        assert "template" in defn.parameters["required"]
        assert "topic" in defn.parameters["required"]
        assert defn.requires_confirmation is False

    def test_export_tool_definition(self):
        """ExportDocumentTool hat korrekte Definition."""
        tool = ExportDocumentTool(exporter=MockExporter())
        defn = tool.definition()
        assert defn.name == "export_document"
        assert defn.parameters["type"] == "object"
        assert "content" in defn.parameters["properties"]
        assert "filename" in defn.parameters["properties"]
        assert "format" in defn.parameters["properties"]
        assert "title" in defn.parameters["properties"]
        assert "content" in defn.parameters["required"]
        assert "filename" in defn.parameters["required"]

    def test_export_tool_requires_confirmation(self):
        """ExportDocumentTool erfordert Bestaetigung."""
        tool = ExportDocumentTool(exporter=MockExporter())
        defn = tool.definition()
        assert defn.requires_confirmation is True


# ---------------------------------------------------------------------------
# Tests: Tool Execution
# ---------------------------------------------------------------------------


class TestGenerateFromTemplateExecute:
    @pytest.mark.asyncio
    async def test_generate_from_template_checklist(self):
        """Checklist-Template wird korrekt generiert."""
        engine = MockTemplateEngine()
        tool = GenerateFromTemplateTool(template_engine=engine)

        result = await tool.execute(
            {
                "template": "checklist",
                "topic": "Serverumzug",
            }
        )

        assert result.success is True
        assert "Checkliste" in result.display_text
        assert "Serverumzug" in result.display_text
        assert engine.last_method == "checklist"

    @pytest.mark.asyncio
    async def test_generate_from_template_protocol(self):
        """Protocol-Template wird korrekt generiert."""
        engine = MockTemplateEngine()
        tool = GenerateFromTemplateTool(template_engine=engine)

        result = await tool.execute(
            {
                "template": "protocol",
                "topic": "Sprint Review",
                "variables": {"participants": ["Alice"], "date": "2026-02-24"},
            }
        )

        assert result.success is True
        assert "Protokoll" in result.display_text
        assert engine.last_method == "protocol"

    @pytest.mark.asyncio
    async def test_generate_from_template_custom(self):
        """Benutzerdefiniertes Template wird via render_template gerendert."""
        engine = MockTemplateEngine("Gerenderter Custom-Inhalt")
        tool = GenerateFromTemplateTool(template_engine=engine)

        result = await tool.execute(
            {
                "template": "status_mail.md.j2",
                "topic": "Auftrag A-1001",
                "variables": {"order_id": "A-1001"},
            }
        )

        assert result.success is True
        assert "Gerenderter Custom-Inhalt" in result.display_text
        assert engine.last_method == "render"

    @pytest.mark.asyncio
    async def test_generate_from_template_invalid_params(self):
        """Fehlende Pflichtparameter fuehren zu Fehler."""
        engine = MockTemplateEngine()
        tool = GenerateFromTemplateTool(template_engine=engine)

        result = await tool.execute({"template": "checklist"})
        assert result.success is False
        assert result.error is not None


class TestSummarizeDocumentToolExecute:
    @pytest.mark.asyncio
    async def test_summarize_success(self):
        """Erfolgreiche Zusammenfassung mit gefundenen Chunks."""
        retriever = MockRetriever(
            results=[
                {"text": "Inhalt von Chunk 1", "metadata": {"source": "doc1.txt"}},
                {"text": "Inhalt von Chunk 2", "metadata": {"source": "doc2.txt"}},
            ]
        )
        summarizer = MockSummarizer("Zusammenfassung der Dokumente.")
        tool = SummarizeDocumentTool(retriever=retriever, summarizer=summarizer)

        result = await tool.execute({"source": "Testdokument"})
        assert result.success is True
        assert "Zusammenfassung" in result.display_text
        assert result.data["chunk_count"] == 2
        assert "doc1.txt" in result.data["sources"]

    @pytest.mark.asyncio
    async def test_summarize_no_results(self):
        """Keine gefundenen Chunks fuehren zu Fehler."""
        retriever = MockRetriever(results=[])
        summarizer = MockSummarizer()
        tool = SummarizeDocumentTool(retriever=retriever, summarizer=summarizer)

        result = await tool.execute({"source": "Nichtexistent"})
        assert result.success is False
        assert "nicht gefunden" in result.error.lower() or "keine" in result.error.lower()


class TestExportDocumentToolExecute:
    @pytest.mark.asyncio
    async def test_export_success(self):
        """Erfolgreicher Export gibt Dateipfad zurueck."""
        exporter = MockExporter("/tmp/test_export.pdf")
        tool = ExportDocumentTool(exporter=exporter)

        result = await tool.execute(
            {
                "content": "# Test\n\nInhalt",
                "filename": "test_export",
                "format": "pdf",
                "title": "Testdokument",
            }
        )

        assert result.success is True
        assert "/tmp/test_export.pdf" in result.data["filepath"]
        assert "pdf" in result.data["format"]
