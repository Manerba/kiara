"""Workflows: Template-Engine, Dokumentenzusammenfassung und Export.

Stellt hoeherwertige Dokumentenverarbeitungs-Workflows bereit, die
ueber einfache RAG-Queries hinausgehen: Template-basierte Textgenerierung,
automatische Zusammenfassungen und Export in verschiedene Formate.

Beziehungen:
    - generation.chain liefert die LLM-Anbindung fuer Zusammenfassungen
    - ingestion.loader stellt die Quelldokumente bereit
    - web.admin_app bindet Workflow-Endpoints ein

Module:
    template_engine — TemplateEngine (Jinja2-basierte Textgenerierung)
    summarizer      — DocumentSummarizer (LLM-gestuetzte Zusammenfassung)
    exporter        — DocumentExporter (PDF, DOCX, Markdown Export)
"""

from .exporter import DocumentExporter
from .summarizer import DocumentSummarizer
from .template_engine import TemplateEngine
