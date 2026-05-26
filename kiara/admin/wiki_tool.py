import logging
import re
from pathlib import Path

from kiara.admin.wiki_manifest import CONTENT_ROOT, WIKI_PAGE_MAP, WIKI_PAGES
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Fallback-Pfad fuer Guide-Seiten die als Jinja2-HTML-Templates existieren
# (nicht als .md). Siehe api_wiki._LEGACY_GUIDE_TEMPLATES.
_TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "web" / "templates" / "wiki"

_HTML_TAG = re.compile(r"<[^>]+>")
_JINJA_BLOCK = re.compile(r"\{%[^%]*%\}|\{\{[^}]*\}\}")
_HTML_ENTITY = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&raquo;": "»",
    "&laquo;": "«",
    "&rsaquo;": "›",
    "&lsaquo;": "‹",
    "&nbsp;": " ",
    "&mdash;": "—",
    "&ndash;": "–",
}


def _load_page_text(page) -> str | None:
    """Lade Text-Inhalt einer Wiki-Seite.

    Priorisiert .md (workflows/concepts), faellt auf Jinja2-Template
    fuer Guides zurueck (HTML-Tags und Jinja-Bloecke werden entfernt).
    Gibt None zurueck wenn weder .md noch Template existieren.
    """
    md_path = CONTENT_ROOT / page.rel_path
    if md_path.exists():
        try:
            return md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("wiki.search ueberspringt %s wegen Lesefehler: %s", md_path, exc)
            return None

    # Fallback: Guide-Template als HTML
    template_name = f"guide_{page.slug.replace('-', '_')}.html"
    template_path = _TEMPLATE_ROOT / template_name
    if not template_path.exists():
        logger.warning(
            "wiki.search: weder %s noch %s gefunden — Seite '%s' uebersprungen",
            md_path,
            template_path,
            page.slug,
        )
        return None
    try:
        html = template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "wiki.search ueberspringt Template %s wegen Lesefehler: %s", template_path, exc
        )
        return None

    text = _JINJA_BLOCK.sub("", html)
    text = _HTML_TAG.sub("\n", text)
    for entity, char in _HTML_ENTITY.items():
        text = text.replace(entity, char)
    return text


class WikiSearchTool(BaseTool):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="wiki.search",
            description="Durchsucht die lokale KIara-Wiki-Dokumentation in Markdown-Dateien.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Suchfrage oder Suchbegriffe"},
                    "guide": {
                        "type": "string",
                        "description": "Optionaler Dokument-Slug aus wiki_manifest.py",
                    },
                },
                "required": ["query"],
            },
            category="wiki",
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        error = self._validate_or_error(params)
        if error:
            return error

        query = params["query"].strip().lower()
        if len(query) < 2:
            return ToolResult(
                success=True,
                data={"hits": []},
                error=None,
                display_text="Bitte mindestens 2 sinnvolle Zeichen fuer wiki.search angeben.",
            )
        guide = (params.get("guide") or "").strip()
        if guide and guide not in WIKI_PAGE_MAP:
            return ToolResult(
                success=True,
                data={"hits": []},
                error=None,
                display_text=f"Wiki-Dokument '{guide}' nicht gefunden",
            )

        pages = [WIKI_PAGE_MAP[guide]] if guide else list(WIKI_PAGES)
        tokens = list(dict.fromkeys(t for t in re.split(r"\W+", query) if len(t) >= 2))
        hits: list[dict] = []

        try:
            for page in pages:
                text = _load_page_text(page)
                if text is None:
                    continue
                current_heading = page.label
                for raw_block in re.split(r"\n\s*\n", text):
                    block = raw_block.strip()
                    if not block:
                        continue
                    if block.startswith("#"):
                        head_line, _, rest = block.partition("\n")
                        current_heading = head_line.lstrip("#").strip() or current_heading
                        if not rest.strip():
                            continue
                        block = rest.strip()

                    lower_block = block.lower()
                    score = 0
                    if query in lower_block:
                        score += 20
                    score += 5 * sum(1 for token in tokens if token in lower_block)
                    score += 3 * sum(1 for token in tokens if token in current_heading.lower())
                    if score <= 0:
                        continue

                    hits.append(
                        {
                            "slug": page.slug,
                            "label": page.label,
                            "section": current_heading,
                            "score": score,
                            "excerpt": block[:500],
                        }
                    )
        except Exception as exc:
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=f"Fehler bei wiki.search: {exc}",
            )

        hits.sort(key=lambda item: item["score"], reverse=True)
        hits = hits[:5]
        return ToolResult(
            success=True,
            data={"hits": hits},
            error=None,
            display_text=(
                "\n\n".join(f"[{hit['label']}] {hit['section']}: {hit['excerpt']}" for hit in hits)
                if hits
                else "Keine passenden Wiki-Abschnitte gefunden."
            ),
        )
