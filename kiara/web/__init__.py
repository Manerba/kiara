"""KIara Web Application — FastAPI-Backend fuer Admin-UI und Chat-UI.

Zwei Apps:
    app.py       — FastAPI Chat-UI + API (Port 8501)
    admin_app.py — FastAPI Admin-API + SPA (Port 8502)

API-Module (api_*.py):
    auth, dashboard, settings, admin, ldap, datasource, groups,
    mail, onlyoffice, ki, cloud, compliance, tools, health,
    backup, cache, graph, mcp

Utilities:
    database     — PostgreSQL-Verbindungsmanagement, init_db()
    schema       — Schema-Loader (liest schema.sql, Source-of-Truth)
    db_seeding   — Default-Admin, ai_config, lokales DDU-Backend
    db_utils     — db_query/db_transaction Context-Manager
    auth         — Hashing, Sessions, Cookie-Signing, FastAPI-Dependencies
    crypto_utils — AES-Verschluesselung fuer Passwoerter/Secrets
    config_utils — JSON-Parsing, User-Permissions, Settings-Helper
    api_utils    — api_endpoint Decorator, lazy_singleton

Beziehungen:
    - kiara.config liefert die YAML-Konfiguration
    - kiara.ingestion wird fuer Ingestion-Jobs aufgerufen
    - kiara.generation wird fuer RAG-Queries aufgerufen
    - kiara.connectors liefert Konnektor-Typen und -Logik
"""
