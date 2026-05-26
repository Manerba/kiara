from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WikiPage:
    slug: str
    label: str
    rel_path: str
    section: str  # "guide" | "workflow" | "concept"
    show_in_sidebar: bool = False
    parent: str = ""  # Slug des Eltern-Guides (fuer Sidebar-Nesting)


CONTENT_ROOT = Path(__file__).resolve().parent / "wiki_content"

WIKI_PAGES = [
    WikiPage("admin-chat", "Admin-Chat", "guides/admin-chat.md", "guide", True),
    WikiPage("users", "Benutzerverwaltung", "guides/users.md", "guide", True),
    WikiPage("datasources", "Datenquellen", "guides/datasources.md", "guide", True),
    WikiPage("jobs", "Jobs", "guides/jobs.md", "guide", True),
    WikiPage("vectordb", "VektorDB", "guides/vectordb.md", "guide", True),
    WikiPage("ki", "KI", "guides/ki.md", "guide", True),
    WikiPage("tools", "Tools & Integrationen", "guides/tools.md", "guide", True),
    WikiPage("compliance", "Compliance", "guides/compliance.md", "guide", True),
    WikiPage(
        "contextual-retrieval", "Contextual Retrieval", "guides/contextual-retrieval.md", "guide"
    ),
    WikiPage("late-chunking", "Late Chunking", "guides/late-chunking.md", "guide"),
    WikiPage("colbert", "ColBERT", "guides/colbert.md", "guide"),
    WikiPage("system", "System", "guides/system.md", "guide", True),
    WikiPage(
        "workflow-setup-llm-backend",
        "Workflow: KI-Backend",
        "workflows/setup-llm-backend.md",
        "workflow",
    ),
    WikiPage(
        "workflow-setup-datasource",
        "Workflow: Datenquelle",
        "workflows/setup-datasource.md",
        "workflow",
    ),
    WikiPage(
        "workflow-setup-ingestion",
        "Workflow: Indexierung",
        "workflows/setup-ingestion.md",
        "workflow",
    ),
    WikiPage(
        "workflow-first-setup", "Workflow: Ersteinrichtung", "workflows/first-setup.md", "workflow"
    ),
    WikiPage("architecture", "Architektur", "concepts/architecture.md", "concept"),
    WikiPage("admin-agent", "Admin-Agent", "concepts/admin-agent.md", "concept"),
]

WIKI_PAGE_MAP = {page.slug: page for page in WIKI_PAGES}
