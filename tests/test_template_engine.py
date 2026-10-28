"""Tests fuer die TemplateEngine (kiara.workflows.template_engine)."""

import os

import pytest

from kiara.workflows.template_engine import TemplateEngine

# ---------------------------------------------------------------------------
# Mock-LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Simuliert ein LLM mit fester Antwort."""

    def __init__(self, response: str = "Mock-Antwort vom LLM") -> None:
        self._response = response
        self.last_prompt: str | None = None

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.last_prompt = user_message
        return self._response


class FailingLLM:
    """LLM das immer einen Fehler wirft."""

    def generate(self, system_prompt: str, user_message: str) -> str:
        raise RuntimeError("LLM nicht erreichbar")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tpl_dir(tmp_path):
    """Erstellt ein temporaeres Template-Verzeichnis."""
    d = tmp_path / "templates"
    d.mkdir()
    return str(d)


@pytest.fixture
def config(tpl_dir):
    return {"workflows": {"template_dir": tpl_dir}}


@pytest.fixture
def engine(config):
    return TemplateEngine(config)


@pytest.fixture
def engine_with_llm(config):
    llm = MockLLM("# Checkliste: Testthema\n\n- [ ] Punkt A\n- [ ] Punkt B")
    return TemplateEngine(config, llm=llm), llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    @pytest.mark.asyncio
    async def test_render_template_basic(self, engine, tpl_dir):
        """Einfaches Template wird korrekt gerendert."""
        tpl_file = os.path.join(tpl_dir, "test.md.j2")
        with open(tpl_file, "w") as f:
            f.write("Hallo {{ name }}, du bist {{ age }} Jahre alt.")

        result = await engine.render_template("test.md.j2", {"name": "Max", "age": 30})
        assert "Hallo Max" in result
        assert "30 Jahre alt" in result

    @pytest.mark.asyncio
    async def test_render_template_missing_variable(self, engine, tpl_dir):
        """Fehlende Variablen werden als leerer String gerendert."""
        tpl_file = os.path.join(tpl_dir, "partial.md.j2")
        with open(tpl_file, "w") as f:
            f.write("Name: {{ name | default('unbekannt') }}")

        result = await engine.render_template("partial.md.j2", {})
        assert "unbekannt" in result


class TestGenerateChecklist:
    @pytest.mark.asyncio
    async def test_generate_checklist_without_llm(self, engine):
        """Ohne LLM wird eine statische Checkliste generiert."""
        result = await engine.generate_checklist("Serverumzug", num_items=5)
        assert "# Checkliste: Serverumzug" in result
        assert result.count("- [ ]") == 5
        assert "Generiert von KIara" in result

    @pytest.mark.asyncio
    async def test_generate_checklist_with_llm(self, engine_with_llm):
        """Mit LLM wird die LLM-Antwort zurueckgegeben."""
        engine, llm = engine_with_llm
        result = await engine.generate_checklist("Testthema", num_items=2)
        assert "Punkt A" in result
        assert "Punkt B" in result
        assert llm.last_prompt is not None
        assert "Testthema" in llm.last_prompt

    @pytest.mark.asyncio
    async def test_generate_checklist_llm_failure_fallback(self, config):
        """Bei LLM-Fehler wird auf statische Checkliste zurueckgefallen."""
        engine = TemplateEngine(config, llm=FailingLLM())
        result = await engine.generate_checklist("Test", num_items=3)
        assert "# Checkliste: Test" in result
        assert result.count("- [ ]") == 3


class TestGenerateProtocol:
    @pytest.mark.asyncio
    async def test_generate_protocol_without_llm(self, engine):
        """Ohne LLM wird ein statisches Protokoll-Template generiert."""
        result = await engine.generate_protocol(
            title="Team-Meeting",
            participants=["Alice", "Bob"],
            date="2026-02-24",
        )
        assert "# Protokoll: Team-Meeting" in result
        assert "Alice, Bob" in result
        assert "2026-02-24" in result
        assert "Tagesordnung" in result
        assert "Beschluesse" in result

    @pytest.mark.asyncio
    async def test_generate_protocol_with_llm(self, config):
        """Mit LLM wird die LLM-Antwort fuer das Protokoll verwendet."""
        llm = MockLLM(
            "# Protokoll: Sprint Review\n\n"
            "**Datum:** 2026-02-24\n"
            "**Teilnehmer:** Team A\n\n"
            "## TOP 1: Ergebnisse\nAlles gut."
        )
        engine = TemplateEngine(config, llm=llm)
        result = await engine.generate_protocol(
            title="Sprint Review",
            participants=["Team A"],
            date="2026-02-24",
        )
        assert "Sprint Review" in result
        assert "Ergebnisse" in result
        assert llm.last_prompt is not None


class TestListTemplates:
    def test_list_templates(self, engine, tpl_dir):
        """Templates werden mit Metadaten aufgelistet."""
        tpl_file = os.path.join(tpl_dir, "example.md.j2")
        with open(tpl_file, "w") as f:
            f.write(
                "{# description: Ein Beispiel-Template #}\n"
                "{# variables: name, role, team #}\n"
                "Hallo {{ name }}!"
            )

        templates = engine.list_templates()
        assert len(templates) == 1
        assert templates[0]["name"] == "example.md.j2"
        assert templates[0]["description"] == "Ein Beispiel-Template"
        assert "name" in templates[0]["variables"]
        assert "role" in templates[0]["variables"]
        assert "team" in templates[0]["variables"]

    def test_list_templates_no_extra_templates(self, engine, tpl_dir):
        """Verzeichnis ohne eigene Templates liefert nur Default-Templates."""
        # Nach _get_env werden Default-Templates kopiert, daher
        # pruefen wir dass keine zusaetzlichen Templates da sind
        templates = engine.list_templates()
        names = {t["name"] for t in templates}
        # Nur die Standard-Templates (kein benutzerdefiniertes)
        assert names.issubset({"checklist.md.j2", "protocol.md.j2", "status_mail.md.j2"})


class TestDefaultTemplates:
    def test_default_templates_copied(self, tmp_path):
        """Default-Templates werden in ein leeres Verzeichnis kopiert."""
        empty_dir = tmp_path / "empty_templates"
        # Verzeichnis noch NICHT erstellen, _get_env soll es anlegen
        config = {"workflows": {"template_dir": str(empty_dir)}}
        engine = TemplateEngine(config)

        # _get_env triggern -> kopiert Defaults
        engine._get_env()

        assert empty_dir.exists()
        j2_files = list(empty_dir.glob("*.j2"))
        # Mindestens unsere 3 Default-Templates
        assert len(j2_files) >= 3

        names = {f.name for f in j2_files}
        assert "checklist.md.j2" in names
        assert "protocol.md.j2" in names
        assert "status_mail.md.j2" in names


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_generate_summary_without_llm(self, engine):
        """Ohne LLM wird der Text gekuerzt zurueckgegeben."""
        text = "Dies ist ein langer Text. " * 50
        result = await engine.generate_summary([text], max_length=100)
        assert len(result) <= 200  # truncated + "..."
        assert "..." in result

    @pytest.mark.asyncio
    async def test_generate_summary_with_llm(self, config):
        """Mit LLM wird die LLM-Zusammenfassung zurueckgegeben."""
        llm = MockLLM("Zusammenfassung: Alles in Ordnung.")
        engine = TemplateEngine(config, llm=llm)
        result = await engine.generate_summary(
            ["Text 1", "Text 2"],
            focus="Sicherheit",
        )
        assert "Zusammenfassung" in result
        assert llm.last_prompt is not None
        assert "Sicherheit" in llm.last_prompt
