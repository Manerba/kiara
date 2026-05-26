"""Gemeinsame API-Infrastruktur fuer alle FastAPI-Endpoints.

API-Response-Konventionen (fuer neue Endpoints):
    Erfolg (Liste):   {"data": [...], "total": N}
    Erfolg (Einzel):  {"data": {...}, "message": "Erstellt"}
    Erfolg (Aktion):  {"message": "Cache geleert"}
    Fehler:           HTTPException (nicht im Body)

Stellt wiederverwendbare Decorator, Helper und Patterns bereit,
die das DRY-Prinzip in den api_*.py-Modulen durchsetzen.

Beziehungen:
    - Alle kiara.web.api_* Module nutzen diese Utilities
    - kiara.web.admin_app registriert die Router
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Callable

from fastapi import HTTPException

from kiara.shared.constants import MAX_WALK_FILE_COUNT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard-Fehlermeldungen (fuer HTTPException.detail)
# Nur Meldungen, die in 3+ Stellen verwendet werden.
# ---------------------------------------------------------------------------
ERR_DATASOURCE_NOT_FOUND = "Datenquelle nicht gefunden"
ERR_BACKEND_NOT_FOUND = "Backend nicht gefunden"
ERR_USER_NOT_FOUND = "Benutzer nicht gefunden"


def api_endpoint(func: Callable) -> Callable:
    """Decorator fuer API-Endpoints mit einheitlicher Fehlerbehandlung.

    Faengt Exceptions und wandelt sie in passende HTTPExceptions um:
    - ``HTTPException`` → unveraendert weitergeworfen
    - ``ValueError`` → 400 Bad Request (generische Meldung)
    - ``PermissionError`` → 403 Forbidden (generische Meldung)
    - ``FileNotFoundError`` → 404 Not Found (generische Meldung)
    - Sonstige → 500 (generische Meldung)

    Alle Exception-Details werden serverseitig geloggt (mit Stacktrace),
    aber nie als ``detail`` zum Client durchgereicht — das verhindert
    Information-Disclosure (Dateipfade, SQL-Fehlertexte, interne Zustaende).
    Endpoints, die benutzerfreundliche Detail-Texte benoetigen, sollen
    stattdessen selbst ``HTTPException(detail="...")`` werfen.

    Usage::

        @router.post("/example")
        @api_endpoint
        async def create_example(data: ExampleRequest, user=Depends(get_admin_user)):
            ...
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as exc:
            logger.warning("%s: ValueError: %s", func.__name__, exc)
            raise HTTPException(status_code=400, detail="Ungueltige Anfrage") from exc
        except PermissionError as exc:
            logger.warning("%s: PermissionError: %s", func.__name__, exc)
            raise HTTPException(status_code=403, detail="Zugriff verweigert") from exc
        except FileNotFoundError as exc:
            logger.warning("%s: FileNotFoundError: %s", func.__name__, exc)
            raise HTTPException(status_code=404, detail="Ressource nicht gefunden") from exc
        except Exception as exc:
            logger.error("%s fehlgeschlagen: %s", func.__name__, exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Interner Fehler") from exc

    return wrapper


def lazy_singleton(factory: Callable) -> Callable:
    """Decorator: Erstellt Instanz beim ersten Aufruf, cached danach (thread-safe).

    Usage::

        @lazy_singleton
        def _get_cache():
            return QueryCache(load_config())

        cache = _get_cache()  # Erstellt beim 1. Aufruf, cached danach
    """
    unset = object()
    instance = unset
    lock = threading.Lock()

    @functools.wraps(factory)
    def getter() -> Any:
        nonlocal instance
        if instance is unset:
            with lock:
                if instance is unset:
                    instance = factory()
        return instance

    def reset() -> None:
        """Setzt den Cache zurueck (fuer Tests)."""
        nonlocal instance
        with lock:
            instance = unset

    getter.reset = reset  # type: ignore[attr-defined]
    return getter


# Re-Export aus shared.ollama_utils (ehemals hier definiert, seit v163 verschoben)
from kiara.shared.ollama_utils import (  # noqa: F401
    check_ollama_models,
    check_ollama_models_sync,
)


def require_datasource(ds_id: int, *, db_query_fn=None) -> dict:
    """Laedt eine Datasource per ID oder wirft HTTPException(404).

    Args:
        ds_id: Datasource-ID.
        db_query_fn: Optionaler Kontextmanager fuer DB-Zugriff (Default: db_query aus shared.db).

    Returns:
        Die Datasource-Row als dict.

    Raises:
        HTTPException(404): Wenn die Datasource nicht existiert.
    """
    if db_query_fn is None:
        from kiara.shared.db import db_query as db_query_fn
    with db_query_fn() as cur:
        row = cur.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
    return dict(row)


# ---------------------------------------------------------------------------
# Pattern A: Resource-Existenzpruefung
# ---------------------------------------------------------------------------

_ENSURE_EXISTS_TABLES: frozenset[str] = frozenset(
    {
        "datasources",
        "users",
        "groups",
        "ki_backends",
        "api_keys",
        "prompt_templates",
        "ingestion_jobs",
    }
)
"""Erlaubte Tabellennamen fuer :func:`ensure_exists` (SQL-Injection-Schutz)."""


def ensure_exists(
    db,
    table: str,
    id_value: int,
    label: str = "Ressource",
) -> None:
    """Prueft ob eine Ressource existiert, wirft 404 wenn nicht.

    Validiert den Tabellennamen gegen eine Allowlist, um SQL-Injection
    zu verhindern.

    Args:
        db:       DB-Connection (ConnectionWrapper).
        table:    Tabellenname (muss in ``_ENSURE_EXISTS_TABLES`` sein).
        id_value: Primaerschluessel-Wert.
        label:    Menschenlesbarer Name fuer die Fehlermeldung.

    Raises:
        HTTPException(404): Wenn die Ressource nicht existiert.
        ValueError:         Wenn der Tabellenname nicht erlaubt ist.
    """
    if table not in _ENSURE_EXISTS_TABLES:
        raise ValueError(f"ensure_exists: Tabelle '{table}' nicht erlaubt")
    row = db.execute(f"SELECT id FROM {table} WHERE id = %s", (id_value,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"{label} nicht gefunden")


# ---------------------------------------------------------------------------
# Pattern B: INSERT mit RETURNING (PostgreSQL)
# ---------------------------------------------------------------------------


def insert_returning(db, sql: str, params: tuple | list) -> int:
    """Fuehrt ein INSERT aus und gibt die neue Row-ID zurueck.

    Haengt ``RETURNING id`` an das INSERT-Statement an und liest
    die ID aus dem Ergebnis.

    Das SQL wird OHNE ``RETURNING id`` uebergeben — der Helper
    haengt es automatisch an.

    Args:
        db:     DB-Connection (ConnectionWrapper).
        sql:    INSERT-Statement OHNE ``RETURNING id``.
        params: Parameter fuer das Statement.

    Returns:
        Die ID der eingefuegten Zeile.
    """
    cursor = db.execute(f"{sql} RETURNING id", params)
    return cursor.fetchone()["id"]


# ---------------------------------------------------------------------------
# Pattern C: Safe Cache-Refresh
# ---------------------------------------------------------------------------


def safe_refresh(fn: Callable, *args: Any) -> None:
    """Ruft eine Cache-Refresh-Funktion auf, schluckt Fehler.

    Verhindert, dass ein fehlgeschlagener Cache-Refresh die
    eigentliche API-Operation zum Scheitern bringt.

    Args:
        fn:   Die aufzurufende Refresh-Funktion.
        args: Argumente fuer die Funktion.
    """
    try:
        fn(*args)
    except Exception as exc:
        logger.debug("Cache-Refresh fehlgeschlagen: %s", exc)


def count_files_limited(path: str, limit: int = MAX_WALK_FILE_COUNT) -> int:
    """Zaehlt Dateien in einem Verzeichnis (mit Obergrenze).

    Bricht bei ``limit`` ab, um bei grossen Verzeichnissen
    keine lange Wartezeit zu verursachen.
    """
    count = 0
    try:
        for _root, _dirs, files in os.walk(path):
            count += len(files)
            if count > limit:
                break
    except Exception as exc:
        logger.debug("count_files_limited(%s) fehlgeschlagen: %s", path, exc)
    return count
