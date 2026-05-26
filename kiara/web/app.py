"""
KIara - User Web-App
FastAPI Backend, Port 8501
"""

# Faulthandler so frueh wie moeglich — vor anderen Imports — damit auch
# Crashes waehrend des restlichen Imports einen Stack-Trace nach stderr
# dumpen (landet im journald, kiara-worker-death-watcher liest mit).
from kiara.web.faulthandler_setup import enable_faulthandler

enable_faulthandler()

import logging
import logging.handlers
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from kiara.config import DEFAULT_PATHS
from kiara.shared.constants import LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES

LOG_PATH = DEFAULT_PATHS["log"]
_log_formatter = logging.Formatter(
    fmt="%(asctime)s %(name)s[%(process)d]: %(levelname)s %(message)s",
    datefmt="%b %d %H:%M:%S",
)
try:
    _file_handler = logging.handlers.RotatingFileHandler(
        LOG_PATH,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    _file_handler.setFormatter(_log_formatter)
except (PermissionError, OSError):
    _file_handler = logging.StreamHandler(sys.stderr)
    _file_handler.setFormatter(_log_formatter)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
if not _root_logger.handlers:
    _root_logger.addHandler(_file_handler)

for _uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uv_logger = logging.getLogger(_uv_name)
    _uv_logger.handlers.clear()
    _uv_logger.addHandler(_file_handler)
    _uv_logger.propagate = False

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from kiara.web.api_auth import router as auth_router
from kiara.web.api_chat import router as chat_router
from kiara.web.api_csp import router as csp_router
from kiara.web.api_health import router as health_router
from kiara.web.api_onlyoffice import (
    callback_router as onlyoffice_callback_router,
)
from kiara.web.api_onlyoffice import (
    editor_router as onlyoffice_editor_router,
)
from kiara.web.api_onlyoffice import (
    wopi_router as onlyoffice_wopi_router,
)
from kiara.web.api_search import router as search_router
from kiara.web.api_workflows import router as workflows_router
from kiara.web.auth import _extract_user_from_request, get_chat_user
from kiara.web.database import init_db
from kiara.web.html_utils import render_html_file
from kiara.web.logging_config import setup_security_logger
from kiara.web.middleware import (
    add_csrf_middleware,
    add_no_cache_static_middleware,
    add_security_headers_middleware,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from kiara.config import load_config
    from kiara.shared.db import init_db_backend

    config = load_config()
    init_db_backend(config)
    try:
        init_db()
    except Exception as exc:
        logger.critical("DB-Initialisierung fehlgeschlagen — App kann nicht starten: %s", exc)
        raise
    # Security-Logger + DB-Handler erst nach erfolgreicher DB-Migration anhaengen,
    # damit die system_events-Tabelle beim ersten Emit garantiert existiert.
    setup_security_logger()
    from kiara.shared.tracing import init_tracing

    init_tracing(config)
    # BM25-Index beim Startup bauen — blockierend, bevor Requests angenommen werden.
    # Verhindert GIL-Deadlocks die entstehen wenn der BM25-Build (629k Dokumente
    # laden + tokenisieren) parallel zu laufenden Requests passiert.
    from kiara.retrieval.retriever import HybridRetriever

    HybridRetriever.warmup_bm25_index(config)
    from kiara.web.api_health import ensure_background_checks, shutdown_degradation_manager

    await ensure_background_checks()
    try:
        yield
    finally:
        await shutdown_degradation_manager()


app = FastAPI(title="KIara", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

app.include_router(auth_router)
app.include_router(chat_router, dependencies=[Depends(get_chat_user)])
app.include_router(onlyoffice_editor_router)
app.include_router(onlyoffice_wopi_router)
app.include_router(onlyoffice_callback_router)
app.include_router(health_router)  # Phase 3: Health-Endpoints (kein Auth fuer Monitoring)
app.include_router(workflows_router, dependencies=[Depends(get_chat_user)])  # Phase 3: Workflows
app.include_router(search_router, dependencies=[Depends(get_chat_user)])
app.include_router(
    csp_router
)  # V30 Welle 4: CSP-Violation-Reports (browser-initiiert, kein Auth/CSRF)

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=static_dir), name="static")


def _get_onlyoffice_script_origin(request: Request | None = None) -> str:
    """Liefert die OnlyOffice-Origin fuer CSP script-src (dynamisch aus DB)."""
    try:
        from kiara.repositories.config import ConfigRepository

        oo = ConfigRepository().get("onlyoffice_config")
    except Exception as exc:
        logger.error(
            "OnlyOffice-CSP-Origin: ConfigRepository-Zugriff fehlgeschlagen — "
            "Origin fehlt in CSP, OnlyOffice-Skript wird vom Browser blockiert: %s",
            exc,
            exc_info=True,
        )
        return ""

    if not oo:
        logger.debug(
            "OnlyOffice-CSP-Origin: kein onlyoffice_config-Eintrag in DB — "
            "Origin leer (OnlyOffice nicht konfiguriert)",
        )
        return ""

    url = oo.get("document_server_url")
    if not url:
        logger.debug(
            "OnlyOffice-CSP-Origin: document_server_url fehlt/leer in onlyoffice_config — Origin leer",
        )
        return ""

    try:
        from urllib.parse import urlparse

        p = urlparse(url)
    except Exception as exc:
        logger.warning(
            "OnlyOffice-CSP-Origin: urlparse fuer document_server_url=%r fehlgeschlagen: %s",
            url,
            exc,
        )
        return ""

    if not p.scheme or not p.netloc:
        logger.warning(
            "OnlyOffice-CSP-Origin: document_server_url=%r ist keine gueltige URL "
            "(scheme=%r, netloc=%r) — Origin leer, OnlyOffice-Skript wird blockiert",
            url,
            p.scheme,
            p.netloc,
        )
        return ""

    return f"{p.scheme}://{p.netloc}"


# Middleware (Reihenfolge: zuletzt registriert = zuerst ausgefuehrt)
# HSTS-Config lesen (0 = deaktiviert, sinnvoll fuer interne HTTP-Deployments)
_config: dict = {}
_hsts_max_age = 0
_csp_strict = False
_csp_report_only = False
_csp_report_uri = ""
try:
    from kiara.config import load_config

    _config = load_config() or {}
    _security = _config.get("security") or {}
    _hsts_max_age = int(_security.get("hsts_max_age") or 0)
    _csp_strict = bool(_security.get("csp_strict") or False)
    _csp_report_only = bool(_security.get("csp_report_only") or False)
    _csp_report_uri = str(_security.get("csp_report_uri") or "")
except Exception as exc:
    logger.critical(
        "Web-App Config-Load fehlgeschlagen — Sicherheits-Header (HSTS/CSP) "
        "koennen nicht initialisiert werden, fail-closed: %s",
        exc,
    )
    raise
_web_port = int((_config.get("web") or {}).get("port") or 8501)
from kiara.web.cors import compute_trusted_origins

_trusted_origins = compute_trusted_origins(_config, ports=[_web_port])

add_csrf_middleware(
    app,
    exempt_prefixes=[
        "/api/wopi/",
        "/api/onlyoffice/callback",
        "/api/csp-report",
        "/api/admin/csp-report",
    ],
    token_exempt_paths={"/api/auth/login"},
    trusted_origins=_trusted_origins,
)
add_security_headers_middleware(
    app,
    extra_img_src="blob:",
    frame_src="*",
    extra_script_src=_get_onlyoffice_script_origin,
    hsts_max_age=_hsts_max_age,
    csp_strict=_csp_strict,
    csp_report_only=_csp_report_only,
    csp_report_uri=_csp_report_uri,
)
add_no_cache_static_middleware(app)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return render_html_file(templates_dir / "login.html", request)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = _extract_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return render_html_file(templates_dir / "index.html", request)


@app.get("/editor", response_class=HTMLResponse)
async def editor_page(request: Request):
    user = _extract_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return render_html_file(templates_dir / "editor.html", request)


@app.get("/{path:path}", response_class=HTMLResponse)
async def catch_all(path: str, request: Request):
    if path.startswith("api/") or path.startswith("static/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    user = _extract_user_from_request(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return render_html_file(templates_dir / "index.html", request)
