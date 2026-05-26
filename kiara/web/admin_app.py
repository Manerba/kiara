"""
KIara Admin - Verwaltungsoberfläche
FastAPI Backend, Port 8502
"""

from kiara.web.faulthandler_setup import enable_faulthandler

enable_faulthandler()

from kiara.web.logging_init import init_root_logger

init_root_logger()

import logging
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from kiara.shared.constants import ADMIN_CHAT_PORT_DEFAULT
from kiara.web.api_admin import router as admin_router
from kiara.web.api_admin_system_events import router as system_events_csv_router
from kiara.web.api_admin_workflows import router as admin_workflows_router
from kiara.web.api_auth import router as auth_router
from kiara.web.api_backup import backup_router
from kiara.web.api_cache import cache_router
from kiara.web.api_config_changelog import router as config_changelog_router
from kiara.web.api_csp import router as csp_router
from kiara.web.api_dashboard import router as dashboard_router
from kiara.web.api_datasource import router as datasource_router
from kiara.web.api_elasticsearch import router as elasticsearch_router
from kiara.web.api_graph import router as graph_router
from kiara.web.api_groups import router as groups_router

# Sprint 09: cloud, compliance, tools, testsuite Router entfernt — migriert zu Admin-Registry
from kiara.web.api_health import router as health_router
from kiara.web.api_ingestion_jobs import router as ingestion_jobs_router
from kiara.web.api_ki import router as ki_router
from kiara.web.api_ki_backends import router as ki_backends_router
from kiara.web.api_ldap import router as ldap_router
from kiara.web.api_mail import router as mail_router
from kiara.web.api_mcp import router as mcp_router
from kiara.web.api_monitoring import router as monitoring_router
from kiara.web.api_onlyoffice import admin_router as onlyoffice_admin_router
from kiara.web.api_settings import router as settings_router
from kiara.web.api_tracing import router as tracing_router
from kiara.web.auth import _extract_user_from_request, get_admin_user, set_cookie_name
from kiara.web.cors import compute_trusted_origins
from kiara.web.database import init_db
from kiara.web.html_utils import render_html_file
from kiara.web.logging_config import setup_security_logger
from kiara.web.middleware import (
    add_csrf_middleware,
    add_no_cache_static_middleware,
    add_security_headers_middleware,
)

# Separates Session-Cookie fuer Admin-UI (Browser unterscheidet nicht nach Port)
set_cookie_name("kiara_admin_session")


def _start_downsampling_thread(stop_event):
    """Prueft stuendlich ob Downsampling faellig ist (Standard: 03:00)."""
    import threading

    from kiara.shared.constants import METRICS_DOWNSAMPLE_HOUR

    def _loop():
        import datetime
        import logging

        log = logging.getLogger(__name__ + ".downsampling")
        last_run_date = None
        target_hour = METRICS_DOWNSAMPLE_HOUR
        while True:
            now = datetime.datetime.now()
            if now.hour == target_hour and now.date() != last_run_date:
                try:
                    from kiara.monitoring.downsampling import run_downsampling

                    run_downsampling()
                    last_run_date = now.date()
                except Exception as exc:
                    log.warning("Downsampling fehlgeschlagen: %s", exc)
            if stop_event.wait(3600):
                break

    t = threading.Thread(target=_loop, daemon=True, name="metrics-downsampling")
    t.start()
    return t


def _start_wopi_cleanup_thread(stop_event):
    """Bereinigt abgelaufene WOPI-Tokens und -Locks stuendlich."""
    import threading

    def _loop():
        import logging

        log = logging.getLogger(__name__ + ".wopi_cleanup")
        while not stop_event.wait(3600):
            try:
                from kiara.repositories.wopi import WopiRepository

                repo = WopiRepository()
                deleted_tokens = repo.cleanup_tokens()
                deleted_locks = repo.cleanup_locks()
                if deleted_tokens:
                    log.info("WOPI-Cleanup: %d abgelaufene Tokens entfernt", deleted_tokens)
                if deleted_locks:
                    log.info("WOPI-Cleanup: %d abgelaufene Locks entfernt", deleted_locks)
            except Exception as exc:
                log.warning("WOPI-Cleanup fehlgeschlagen: %s", exc)

    t = threading.Thread(target=_loop, daemon=True, name="wopi-cleanup")
    t.start()
    return t


def _start_cache_refresh_thread(stop_event):
    """Background-Thread der die Admin-UI-Caches periodisch aktualisiert."""
    import threading

    from kiara.web.cache_refresh import refresh_all, refresh_vectordb_cache

    def _loop():
        import logging

        log = logging.getLogger(__name__ + ".cache_refresh")
        cycle = 0
        # Initialer Cold-Start-Refresh
        try:
            from kiara.web.cache_refresh import refresh_all_with_vectordb

            refresh_all_with_vectordb()
            log.info("Cache-Refresh: Initial-Befuellung abgeschlossen")
        except Exception as exc:
            log.warning("Cache-Refresh: Initial-Befuellung fehlgeschlagen: %s", exc)
        while not stop_event.wait(60):
            cycle += 1
            try:
                refresh_all()
            except Exception as exc:
                log.debug("Cache-Refresh (periodisch): %s", exc)
            if cycle % 2 == 0:  # alle 120s
                try:
                    refresh_vectordb_cache()
                except Exception as exc:
                    log.debug("VektorDB-Cache-Refresh: %s", exc)

    t = threading.Thread(target=_loop, daemon=True, name="cache-refresh")
    t.start()
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    import threading

    from kiara.config import load_config
    from kiara.shared.db import init_db_backend

    _log = logging.getLogger(__name__)
    config = load_config()
    init_db_backend(config)
    try:
        init_db()
    except Exception as exc:
        _log.critical("DB-Initialisierung fehlgeschlagen — Admin-App kann nicht starten: %s", exc)
        raise
    # Security-Logger (File + DB-Handler) nach erfolgreicher Migration anhaengen.
    # Admin-UI-Prozess erzeugt eigene Login-/Self-Audit-Events (z.B. SYSLOG_VIEW),
    # die sonst im separaten Prozess ohne dedizierten Handler bleiben wuerden.
    setup_security_logger()
    from kiara.shared.tracing import init_tracing

    init_tracing(config)
    # Worker wird NICHT mehr hier gestartet — laeuft als kiara-worker.service
    # Background-Cache-Refresh starten
    _cache_stop = threading.Event()
    _cache_thread = _start_cache_refresh_thread(_cache_stop)

    # NEU: System-Metrik-Collector starten (1s Takt + SSE-Broadcast)
    from kiara.monitoring.collector import start_collector

    _metrics_stop = threading.Event()
    _metrics_thread = start_collector(_metrics_stop)

    # NEU: Downsampling-Timer starten (taegliches 03xx-Fenster, nach erstem Hourly-Tick)
    _downsample_stop = threading.Event()
    _downsample_thread = _start_downsampling_thread(_downsample_stop)

    # WOPI-Token-Cleanup (stuendlich abgelaufene Tokens entfernen)
    _wopi_stop = threading.Event()
    _wopi_thread = _start_wopi_cleanup_thread(_wopi_stop)

    from kiara.web.api_health import ensure_background_checks, shutdown_degradation_manager

    await ensure_background_checks()
    try:
        yield
    finally:
        _cache_stop.set()
        _metrics_stop.set()
        _downsample_stop.set()
        _wopi_stop.set()
        _shutdown_threads = (
            ("cache_refresh", _cache_thread),
            ("metrics", _metrics_thread),
            ("downsample", _downsample_thread),
            ("wopi_cleanup", _wopi_thread),
        )
        for _name, _t in _shutdown_threads:
            if _t is None:
                continue
            _t.join(timeout=30)
            if _t.is_alive():
                _log.warning(
                    "Background-Thread '%s' nach 30s Shutdown-Timeout noch aktiv — "
                    "Daemon-Thread wird beim Prozessende abgewuergt, "
                    "moegliche halbfertige DB-Transaktion.",
                    _name,
                )
        await shutdown_degradation_manager()


app = FastAPI(
    title="KIara Admin", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
)

app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(settings_router)
app.include_router(admin_router)
app.include_router(ldap_router)
app.include_router(datasource_router)
app.include_router(groups_router)
app.include_router(mail_router)
app.include_router(onlyoffice_admin_router)
app.include_router(ki_router)
app.include_router(ki_backends_router)
# Sprint 09: cloud, compliance, tools, testsuite Router entfernt — migriert zu Admin-Registry
app.include_router(health_router)
app.include_router(backup_router)
app.include_router(cache_router)
app.include_router(graph_router)
app.include_router(mcp_router)
app.include_router(ingestion_jobs_router)
app.include_router(elasticsearch_router)
app.include_router(config_changelog_router)
app.include_router(monitoring_router)
app.include_router(tracing_router)
app.include_router(
    csp_router
)  # V30 Welle 4: CSP-Violation-Reports (browser-initiiert, kein Auth/CSRF)

# Wiki-Router (nach den API-Routern, vor generate_admin_routes)
from kiara.web.api_wiki import router as wiki_router

app.include_router(wiki_router)

# CSV-Router fuer system-events vor dem Auto-Router registrieren — sonst
# wuerde /{event_id} den Pfad /export.csv als dynamischen Event-Id matchen.
app.include_router(system_events_csv_router)
app.include_router(admin_workflows_router)

# Admin-Funktions-Registry laden und Auto-Routen generieren
# WICHTIG: Muss VOR app.mount() und dem catch-all /{path:path} stehen,
# da Starlette Routen in Registrierungs-Reihenfolge matcht.
from kiara.admin.registry import admin_registry, load_admin_functions
from kiara.admin.router import generate_admin_routes

load_admin_functions()
generate_admin_routes(app, admin_registry)

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=static_dir), name="static")

from kiara.config import load_config as _load_config

_config = _load_config() or {}
_admin_port = int((_config.get("web") or {}).get("admin_port") or 8502)
_admin_chat_port = int((_config.get("web") or {}).get("admin_chat_port") or ADMIN_CHAT_PORT_DEFAULT)
_security = _config.get("security") or {}
_hsts_max_age = int(_security.get("hsts_max_age") or 0)
_csp_strict = bool(_security.get("csp_strict") or False)
_csp_report_only = bool(_security.get("csp_report_only") or False)
_csp_report_uri = str(_security.get("csp_report_uri") or "")
_trusted_origins = compute_trusted_origins(_config, ports=[_admin_port])


def _admin_chat_connect_src(request: Request | None = None) -> str:
    sources = compute_trusted_origins(_config, ports=[_admin_chat_port])
    if request is not None and request.url.hostname:
        sources.insert(0, f"{request.url.scheme}://{request.url.hostname}:{_admin_chat_port}")
    return " ".join(dict.fromkeys(sources))


add_csrf_middleware(
    app,
    exempt_prefixes=["/api/csp-report", "/api/admin/csp-report"],
    token_exempt_paths={"/api/auth/login"},
    trusted_origins=_trusted_origins,
)
add_security_headers_middleware(
    app,
    extra_connect_src=_admin_chat_connect_src,
    hsts_max_age=_hsts_max_age,
    csp_strict=_csp_strict,
    csp_report_only=_csp_report_only,
    csp_report_uri=_csp_report_uri,
)
add_no_cache_static_middleware(app)


_RESTART_RATE_LIMIT_SECONDS = 60.0
_restart_last_call: dict[str, float] = {}
_restart_lock = threading.Lock()


@app.post("/api/dashboard/restart")
async def restart_dashboard(user=Depends(get_admin_user)):
    user_key = str(user.get("username") or user.get("id") or "unknown")
    now = time.monotonic()
    with _restart_lock:
        last = _restart_last_call.get(user_key)
        if last is not None and (now - last) < _RESTART_RATE_LIMIT_SECONDS:
            retry_after = max(1, int(_RESTART_RATE_LIMIT_SECONDS - (now - last)))
            raise HTTPException(
                status_code=429,
                detail="Restart bereits ausgeloest, bitte kurz warten.",
                headers={"Retry-After": str(retry_after)},
            )
        _restart_last_call[user_key] = now

    log = logging.getLogger(__name__)
    try:
        proc = subprocess.Popen(
            ["systemctl", "restart", "kiara.service", "kiara-admin.service"],
        )
        log.info("Service-Restart durch %s ausgeloest (pid=%s)", user_key, proc.pid)
    except Exception as exc:
        log.error("Restart fehlgeschlagen: %s", exc, exc_info=True)
        raise HTTPException(500, "Restart fehlgeschlagen")
    return {"status": "restarting", "message": "Services werden neu gestartet..."}


@app.get("/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return render_html_file(templates_dir / "admin_login.html", request)


def _serve_admin_html(request: Request) -> HTMLResponse:
    """admin.html mit Request-Nonce und konfiguriertem Admin-Chat-Port ausliefern."""
    return render_html_file(
        templates_dir / "admin.html",
        request,
        replacements={"admin_chat_port": _admin_chat_port},
    )


@app.get("/", response_class=HTMLResponse)
async def admin_index(request: Request):
    user = _extract_user_from_request(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/login", status_code=302)
    return _serve_admin_html(request)


@app.get("/audit/{audit_id}", response_class=HTMLResponse)
async def audit_detail_page(audit_id: int, request: Request):
    user = _extract_user_from_request(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url=f"/login?next=/audit/{audit_id}", status_code=302)
    return render_html_file(templates_dir / "audit_detail.html", request)


@app.get("/{path:path}", response_class=HTMLResponse)
async def admin_catch_all(path: str, request: Request):
    if path.startswith("api/") or path.startswith("static/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    user = _extract_user_from_request(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/login", status_code=302)
    return _serve_admin_html(request)
