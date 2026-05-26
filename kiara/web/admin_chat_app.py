from kiara.web.faulthandler_setup import enable_faulthandler

enable_faulthandler()

from kiara.web.logging_init import init_root_logger

init_root_logger()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kiara.admin.registry import load_admin_functions
from kiara.config import load_config
from kiara.shared.constants import ADMIN_PORT_DEFAULT
from kiara.shared.db import init_db_backend
from kiara.web.auth import set_cookie_name
from kiara.web.cors import compute_trusted_origins

logger = logging.getLogger(__name__)

# Session-Cookie-Name MUSS identisch mit admin_app.py sein.
# HINWEIS (H7): set_cookie_name() setzt eine globale Variable in auth.py.
# Das ist sicher, weil jeder Service als eigener Prozess laeuft
# (kiara.service, kiara-admin.service, kiara-admin-chat.service).
# Bei Multi-Worker-Setups (z.B. Gunicorn mit fork) waere dies problematisch,
# da Workers den Modul-Namespace teilen. Aktuell nicht relevant.
# bugscan:by-design[global-state] — Prozess-gebundener Cookie-Name-Override, siehe H7 oben; Multi-Worker aktuell nicht Scope.
set_cookie_name("kiara_admin_session")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    init_db_backend(config)

    from kiara.shared.tracing import init_tracing

    init_tracing(config)

    # Tabellen erstellen + Migrationen ausfuehren (idempotent)
    # Gemeinsamer Pfad mit init_db() aus database.py: Advisory Lock schuetzt
    # gegen parallele Migrationen von Haupt-/Admin-App.
    from kiara.web.database import init_db

    try:
        init_db()
    except Exception as exc:
        logger.critical(
            "DB-Initialisierung fehlgeschlagen — Admin-Chat-App kann nicht starten: %s", exc
        )
        raise

    # Security-Logger (File + DB-Handler) nach erfolgreicher Migration anhaengen.
    # Admin-Chat-Prozess erzeugt eigene SYSLOG_VIEW-Events via system_events.list;
    # ohne diesen Aufruf bliebe der DB-Handler nicht am kiara.security-Logger.
    from kiara.web.logging_config import setup_security_logger

    setup_security_logger()

    load_admin_functions()
    logger.info("Admin-Chat-Service gestartet")
    yield
    logger.info("Admin-Chat-Service gestoppt")


app = FastAPI(
    title="KIara Admin Chat", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
)

# CORS — Admin-UI auf Port 8502 muss SSE-Streams von Port 8503 empfangen
# Origins ueber kiara.web.cors zentralisiert (port-spezifisch + web.trusted_origins additiv)
config = load_config()
admin_port = config.get("web", {}).get("admin_port", ADMIN_PORT_DEFAULT)
_trusted_origins = compute_trusted_origins(config, ports=[admin_port])

# CSRF-Schutz VOR CORSMiddleware registrieren → im LIFO-Stack liegt CORS aussen
# und umhuellt CSRF-403-Responses mit CORS-Headern, damit Cross-Origin-Browser
# den JSON-Code `csrf_invalid` lesen und den Refresh-Pfad ausloesen koennen.
from kiara.web.middleware import add_csrf_middleware, add_security_headers_middleware

add_csrf_middleware(
    app,
    token_exempt_paths={"/api/auth/login"},
    trusted_origins=_trusted_origins,
)

# Security-Headers (HSTS, CSP, Referrer-Policy, X-Frame-Options).
# V30 Welle 5 (#1049): Admin-Chat-Prozess ist API/SSE-only, text/html wird hier
# nicht ausgeliefert — die CSP-Header-Middleware setzt CSP sowieso nur fuer
# text/html-Responses. Trotzdem reichen wir die strict-CSP-Config durch, damit
# falls kuenftig HTML-Endpoints hinzukommen, sie automatisch denselben
# strict-Enforce + Report-Only bekommen wie app.py / admin_app.py.
_security_cfg = config.get("security") or {}
_hsts_max_age = int(_security_cfg.get("hsts_max_age") or 0)
_csp_strict = bool(_security_cfg.get("csp_strict") or False)
_csp_report_only = bool(_security_cfg.get("csp_report_only") or False)
_csp_report_uri = str(_security_cfg.get("csp_report_uri") or "")
add_security_headers_middleware(
    app,
    hsts_max_age=_hsts_max_age,
    csp_strict=_csp_strict,
    csp_report_only=_csp_report_only,
    csp_report_uri=_csp_report_uri,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_trusted_origins,
    allow_credentials=True,  # Wichtig fuer Session-Cookie
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "X-Requested-With", "X-CSRF-Token"],
)

# Router einbinden
from kiara.web.api_admin_chat import router as admin_chat_router

app.include_router(admin_chat_router)
