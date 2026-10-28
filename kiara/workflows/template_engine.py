"""Template-Engine fuer KIara Dokumenten-Workflows.

Rendert Jinja2-Templates, generiert Checklisten, Protokolle und
Zusammenfassungen wahlweise statisch oder LLM-gestuetzt.
"""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from jinja2 import FileSystemLoader, select_autoescape
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)

# Pfad zu den mitgelieferten Default-Templates innerhalb des Pakets
_DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "default_templates"


class TemplateEngine:
    """Jinja2-basierte Template-Engine mit optionaler LLM-Unterstuetzung.

    Args:
        config: KIara-Konfiguration. Relevanter Schluessel:
                ``workflows.template_dir`` (default ``/etc/kiara/templates``).
        llm: Optionales Backend-Objekt mit ``generate("", prompt)``-Methode.
    """

    def __init__(self, config: dict, llm=None) -> None:
        from kiara.config import get_template_dir

        self._template_dir = get_template_dir(config)
        self._llm = llm
        self._env: SandboxedEnvironment | None = None

    # ------------------------------------------------------------------
    # Jinja2 Environment
    # ------------------------------------------------------------------

    def _get_env(self) -> SandboxedEnvironment:
        """Lazy-initialisiert das Jinja2-Environment.

        Erstellt das Template-Verzeichnis, falls es nicht existiert, und
        kopiert Default-Templates hinein, wenn das Verzeichnis leer ist.
        """
        if self._env is not None:
            return self._env

        tpl_path = Path(self._template_dir)
        tpl_path.mkdir(parents=True, exist_ok=True)

        # Default-Templates kopieren wenn Verzeichnis leer
        existing = list(tpl_path.glob("*.j2"))
        if not existing and _DEFAULT_TEMPLATES_DIR.is_dir():
            for src in _DEFAULT_TEMPLATES_DIR.glob("*.j2"):
                dst = tpl_path / src.name
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    logger.info("Default-Template kopiert: %s", dst)

        self._env = SandboxedEnvironment(
            loader=FileSystemLoader(str(tpl_path)),
            autoescape=select_autoescape(
                enabled_extensions=("html", "htm", "xml", "html.j2", "xml.j2"),
                default_for_string=False,
                default=False,
            ),
            keep_trailing_newline=True,
        )
        return self._env

    # ------------------------------------------------------------------
    # Template-Rendering
    # ------------------------------------------------------------------

    async def render_template(self, template_name: str, variables: dict) -> str:
        """Rendert ein Jinja2-Template mit den uebergebenen Variablen.

        Args:
            template_name: Dateiname des Templates (z.B. ``checklist.md.j2``).
            variables: Dictionary mit Template-Variablen.

        Returns:
            Gerenderter Markdown-String.

        Raises:
            TemplateNotFound: Wenn das Template nicht existiert.
        """
        env = self._get_env()
        template = env.get_template(template_name)
        return template.render(**variables)

    # ------------------------------------------------------------------
    # LLM-gestuetzte Generierung
    # ------------------------------------------------------------------

    async def generate_checklist(
        self,
        topic: str,
        context: str = "",
        num_items: int = 10,
    ) -> str:
        """Generiert eine Checkliste, wahlweise via LLM.

        Args:
            topic: Thema der Checkliste.
            context: Optionaler Kontext (z.B. aus Retrieval).
            num_items: Gewuenschte Anzahl der Punkte.

        Returns:
            Markdown-formatierte Checkliste.
        """
        if self._llm is not None:
            ctx_block = f"\nKontext:\n{context}\n" if context else ""
            prompt = (
                f"Erstelle eine praeziese Checkliste zum Thema '{topic}' "
                f"mit genau {num_items} Punkten.{ctx_block}\n"
                f"Gib die Checkliste als Markdown mit '- [ ]' Checkboxen zurueck. "
                f"Beginne mit einer Ueberschrift '# Checkliste: {topic}'.\n"
                f"Antworte NUR mit der Checkliste, kein zusaetzlicher Text."
            )
            try:
                result = self._llm.generate("", prompt)
                return result.strip()
            except Exception as exc:
                logger.warning("LLM-Generierung fehlgeschlagen, Fallback auf statisch: %s", exc)

        # Statische Fallback-Checkliste
        lines = [f"# Checkliste: {topic}", ""]
        for i in range(1, num_items + 1):
            lines.append(f"- [ ] Punkt {i}: {topic}")
        lines.append("")
        lines.append("---")
        lines.append("*Generiert von KIara*")
        return "\n".join(lines)

    async def generate_protocol(
        self,
        title: str,
        context: str = "",
        participants: list[str] | None = None,
        date: str | None = None,
    ) -> str:
        """Generiert ein Besprechungsprotokoll-Template.

        Args:
            title: Titel der Besprechung.
            context: Optionaler Kontext.
            participants: Liste der Teilnehmer.
            date: Datum der Besprechung (default: heute).

        Returns:
            Markdown-formatiertes Protokoll-Template.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        if participants is None:
            participants = []

        participants_str = ", ".join(participants) if participants else "(keine angegeben)"

        if self._llm is not None:
            ctx_block = f"\nKontext:\n{context}\n" if context else ""
            prompt = (
                f"Erstelle ein Besprechungsprotokoll-Template fuer:\n"
                f"Titel: {title}\n"
                f"Datum: {date}\n"
                f"Teilnehmer: {participants_str}\n"
                f"{ctx_block}\n"
                f"Das Protokoll soll enthalten: Ueberschrift, Datum, Teilnehmer, "
                f"Tagesordnungspunkte (mit Platz fuer Notizen), Beschluesse/Aufgaben.\n"
                f"Antworte NUR mit dem Protokoll im Markdown-Format."
            )
            try:
                result = self._llm.generate("", prompt)
                return result.strip()
            except Exception as exc:
                logger.warning("LLM-Generierung fehlgeschlagen, Fallback auf statisch: %s", exc)

        # Statische Fallback-Struktur
        lines = [
            f"# Protokoll: {title}",
            "",
            f"**Datum:** {date}",
            f"**Teilnehmer:** {participants_str}",
            "",
            "## Tagesordnung",
            "",
            "### 1. Begruessung und Agenda",
            "",
            "(Notizen hier einfuegen)",
            "",
            "### 2. Thema 1",
            "",
            "(Notizen hier einfuegen)",
            "",
            "### 3. Verschiedenes",
            "",
            "(Notizen hier einfuegen)",
            "",
            "## Beschluesse & Aufgaben",
            "",
            "- **Aufgabe 1** - Zustaendig: (Name) - Frist: (Datum)",
            "",
            "---",
            "*Generiert von KIara*",
        ]
        return "\n".join(lines)

    async def generate_summary(
        self,
        documents: list[str],
        focus: str | None = None,
        max_length: int = 500,
    ) -> str:
        """Generiert eine Kurzzusammenfassung mehrerer Texte via LLM.

        Args:
            documents: Liste von Textabschnitten.
            focus: Optionaler Fokus / Fragestellung.
            max_length: Maximale Laenge der Zusammenfassung (Zeichen).

        Returns:
            Zusammenfassung als Markdown-Text.
        """
        combined = "\n\n---\n\n".join(documents)

        if self._llm is not None:
            focus_hint = f"\nFokus: {focus}" if focus else ""
            prompt = (
                f"Fasse die folgenden Texte zusammen (maximal {max_length} Zeichen)."
                f"{focus_hint}\n\n"
                f"Texte:\n{combined}\n\n"
                f"Antworte NUR mit der Zusammenfassung."
            )
            try:
                result = self._llm.generate("", prompt)
                return result.strip()
            except Exception as exc:
                logger.warning("LLM-Generierung fehlgeschlagen: %s", exc)

        # Fallback: Ersten Text kuerzen
        if combined:
            truncated = combined[:max_length]
            if len(combined) > max_length:
                truncated += "..."
            return truncated
        return "(Keine Texte vorhanden)"

    # ------------------------------------------------------------------
    # Template-Verwaltung
    # ------------------------------------------------------------------

    def list_templates(self) -> list[dict]:
        """Listet alle verfuegbaren Templates mit Metadaten auf.

        Parst die Header-Kommentare ``{# description: ... #}`` und
        ``{# variables: ... #}`` aus jeder Template-Datei.

        Returns:
            Liste von Dicts mit ``name``, ``description``, ``variables``.
        """
        self._get_env()
        tpl_path = Path(self._template_dir)
        templates = []

        for tpl_file in sorted(tpl_path.glob("*.j2")):
            name = tpl_file.name
            description = ""
            variables = []

            try:
                content = tpl_file.read_text(encoding="utf-8")
                # Description parsen
                desc_match = re.search(r"\{#\s*description:\s*(.+?)\s*#\}", content)
                if desc_match:
                    description = desc_match.group(1).strip()

                # Variables parsen
                var_match = re.search(r"\{#\s*variables:\s*(.+?)\s*#\}", content)
                if var_match:
                    variables = [v.strip() for v in var_match.group(1).split(",")]
            except Exception as exc:
                logger.warning("Fehler beim Parsen von Template '%s': %s", name, exc)

            templates.append(
                {
                    "name": name,
                    "description": description,
                    "variables": variables,
                }
            )

        return templates
