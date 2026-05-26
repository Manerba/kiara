"""Wiki/Docs-System fuer die Admin-UI — API-Katalog, Guides, Self-Check."""

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from kiara.admin.registry import admin_registry
from kiara.admin.wiki_manifest import CONTENT_ROOT, WIKI_PAGE_MAP, WIKI_PAGES
from kiara.web.auth import _extract_user_from_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wiki", tags=["wiki"])

_templates_dir = Path(__file__).parent / "templates" / "wiki"
_templates = Jinja2Templates(directory=str(_templates_dir))


def _require_admin(request: Request):
    """Auth-Check fuer Wiki-HTML-Seiten. Redirect statt Exception."""
    user = _extract_user_from_request(request)
    if not user or not user["is_admin"]:
        return None
    return user


def _base_context(request: Request, **extra) -> dict:
    csp_nonce = getattr(request.state, "csp_nonce", None)
    if not csp_nonce:
        raise RuntimeError("CSP-Nonce fehlt im Request-State")
    return {
        "request": request,
        "csp_nonce": csp_nonce,
        "guide_pages": [
            page for page in WIKI_PAGES if page.section == "guide" and page.show_in_sidebar
        ],
        **extra,
    }


@router.get("/", response_class=HTMLResponse)
async def wiki_index(request: Request):
    if not _require_admin(request):
        return RedirectResponse(url="/login", status_code=302)
    summary = admin_registry.get_self_check_summary()
    return _templates.TemplateResponse(
        request,
        "index.html",
        context=_base_context(
            request,
            summary=summary,
        ),
    )


@router.get("/api/", response_class=HTMLResponse)
async def wiki_api_index(request: Request):
    if not _require_admin(request):
        return RedirectResponse(url="/login", status_code=302)
    functions = admin_registry.serialize_all()
    summary = admin_registry.get_self_check_summary()
    categories = {}
    for func in functions:
        cat = func["category"]
        categories.setdefault(cat, []).append(func)
    return _templates.TemplateResponse(
        request,
        "api_index.html",
        context=_base_context(
            request,
            categories=categories,
            summary=summary,
        ),
    )


@router.get("/api/fn/{name}", response_class=HTMLResponse)
async def wiki_api_function(name: str, request: Request):
    if not _require_admin(request):
        return RedirectResponse(url="/login", status_code=302)
    data = admin_registry.get_serialized(name)
    if not data:
        return _templates.TemplateResponse(
            request,
            "404.html",
            context=_base_context(
                request,
                message=f"Funktion '{name}' nicht gefunden",
            ),
            status_code=404,
        )
    return _templates.TemplateResponse(
        request,
        "api_function.html",
        context=_base_context(
            request,
            func=data,
        ),
    )


@router.get("/api/{category}", response_class=HTMLResponse)
async def wiki_api_category(category: str, request: Request):
    if not _require_admin(request):
        return RedirectResponse(url="/login", status_code=302)
    functions = [f for f in admin_registry.serialize_all() if f["category"] == category]
    if not functions:
        return _templates.TemplateResponse(
            request,
            "404.html",
            context=_base_context(
                request,
                message=f"Kategorie '{category}' nicht gefunden",
            ),
            status_code=404,
        )
    return _templates.TemplateResponse(
        request,
        "api_category.html",
        context=_base_context(
            request,
            category=category,
            functions=functions,
        ),
    )


_LEGACY_GUIDE_TEMPLATES = {
    "users": "guide_users.html",
    "datasources": "guide_datasources.html",
    "jobs": "guide_jobs.html",
    "vectordb": "guide_vectordb.html",
    "ki": "guide_ki.html",
    "tools": "guide_tools.html",
    "compliance": "guide_compliance.html",
    "system": "guide_system.html",
    "admin-chat": "guide_admin_chat.html",
    "contextual-retrieval": "guide_contextual_retrieval.html",
    "late-chunking": "guide_late_chunking.html",
    "colbert": "guide_colbert.html",
}


@router.get("/guide/{tab}", response_class=HTMLResponse)
async def wiki_guide(tab: str, request: Request):
    if not _require_admin(request):
        return RedirectResponse(url="/login", status_code=302)

    page = WIKI_PAGE_MAP.get(tab)
    if not page:
        return _templates.TemplateResponse(
            request,
            "guide.html",
            context=_base_context(request, tab=tab),
            status_code=404,
        )

    md_path = (CONTENT_ROOT / page.rel_path).resolve()
    try:
        if not md_path.is_relative_to(CONTENT_ROOT.resolve()):
            raise FileNotFoundError(md_path)
        markdown_text = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass  # fall through to legacy template check
    except Exception:
        logger.exception("Wiki-Guide '%s' konnte nicht geladen werden", tab)
        return _templates.TemplateResponse(
            request,
            "404.html",
            context=_base_context(
                request,
                message=f"Guide '{tab}' konnte nicht geladen werden",
            ),
            status_code=500,
        )
    else:
        return _templates.TemplateResponse(
            request,
            "guide_dynamic.html",
            context=_base_context(
                request,
                tab=tab,
                title=page.label,
                markdown_text=markdown_text,
            ),
        )

    legacy_template = _LEGACY_GUIDE_TEMPLATES.get(tab)
    if legacy_template:
        return _templates.TemplateResponse(
            request,
            legacy_template,
            context=_base_context(request, tab=tab),
        )

    return _templates.TemplateResponse(
        request,
        "guide.html",
        context=_base_context(request, tab=tab),
        status_code=404,
    )
