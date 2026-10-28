"""KIara Admin - KI-Backend CRUD API (Phase 13).

Multi-Backend-Verwaltung mit CRUD-Operationen, Verbindungstest,
Aktiv/Inaktiv-Toggle und Modell-Konsistenzpruefung.

Beziehungen:
    - kiara.backend_selector  -> invalidate_cache() nach CUD
    - kiara.web.api_ki        -> _test_ollama_connection() wiederverwendet
    - kiara.shared.crypto     -> encrypt/decrypt fuer API-Keys
    - kiara.shared.constants  -> PASSWORD_MASK
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from kiara.repositories.backend import KiBackendRepository
from kiara.shared.constants import PASSWORD_MASK
from kiara.web.auth import get_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/ki/backends", tags=["ki-backends"])

# --- Repository-Instanz (Modul-Ebene) ---
backend_repo = KiBackendRepository()

# ---------------------------------------------------------------------------
# Ollama num_ctx Ermittlung
# ---------------------------------------------------------------------------


async def _fetch_model_info(url: str, model: str) -> dict:
    """Ermittelt num_ctx und embedding_dim eines Ollama-Modells via /api/show."""
    from kiara.retrieval.embedding import fetch_ollama_model_info

    return fetch_ollama_model_info(url, model, timeout=10.0)


# ---------------------------------------------------------------------------
# Validierung
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {"embedding", "llm", "cloud", "vlm", "ddu"}

_VALID_PROVIDERS = {
    "embedding": ["ollama", "openai", "voyage"],
    "llm": ["ollama"],
    "cloud": ["anthropic", "openai"],
    "vlm": ["ollama"],
    "ddu": ["docling"],
}


def _validate_category(category: str) -> None:
    if category not in _VALID_CATEGORIES:
        raise HTTPException(status_code=422, detail=f"Ungueltige Kategorie: {category}")


def _validate_provider(category: str, provider: str) -> None:
    valid = _VALID_PROVIDERS.get(category, [])
    if provider not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Ungueltiger Provider '{provider}' fuer Kategorie '{category}'. "
            f"Erlaubt: {', '.join(valid)}",
        )


## _check_model_consistency() entfernt (V13):
## Verschiedene Embedding-Modelle sind gewollt — die Konsistenz wird ueber
## separate Collections sichergestellt, nicht ueber Backend-Restriktionen.

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class KiBackendRequest(BaseModel):
    """Request zum Erstellen oder Aktualisieren eines KI-Backends."""

    category: str
    provider: str
    name: str = ""
    url: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    is_active: bool = True
    priority: int = 10
    extra_config: dict = {}
    trusted: bool = False


class KiBackendToggleRequest(BaseModel):
    """Request zum Aktivieren oder Deaktivieren eines KI-Backends."""

    is_active: bool


# ---------------------------------------------------------------------------
# CRUD Endpoints
# ---------------------------------------------------------------------------


# --- Route migriert zu Admin-Registry: ki.backends.categories (Sprint 04) ---
# @router.get("/categories")
async def get_categories(user: dict = Depends(get_admin_user)) -> dict:
    """Gibt verfuegbare Kategorien und ihre erlaubten Provider zurueck."""
    return {
        "categories": [
            {
                "id": "embedding",
                "label": "Embedding",
                "providers": _VALID_PROVIDERS["embedding"],
                "description": "Vektorisierung fuer Retrieval (Failover)",
            },
            {
                "id": "llm",
                "label": "LLM",
                "providers": _VALID_PROVIDERS["llm"],
                "description": "Sprachmodelle (Round-Robin)",
            },
            {
                "id": "cloud",
                "label": "Cloud-AI",
                "providers": _VALID_PROVIDERS["cloud"],
                "description": "Cloud-Sprachmodelle (Round-Robin)",
            },
            {
                "id": "vlm",
                "label": "VLM (Vision)",
                "providers": _VALID_PROVIDERS["vlm"],
                "description": "Bildbeschreibung / Vision (Failover)",
            },
            {
                "id": "ddu",
                "label": "DDU (Docling)",
                "providers": _VALID_PROVIDERS["ddu"],
                "description": "Dokumentenkonvertierung / OCR (Failover)",
            },
        ]
    }


# --- Route migriert zu Admin-Registry: ki.backends.list (Sprint 04) ---
# @router.get("/")
async def list_backends(user: dict = Depends(get_admin_user)) -> dict:
    """Alle Backends auflisten."""
    rows = backend_repo.list_ordered()

    backends = []
    for d in rows:
        # API-Key maskieren
        if d.get("api_key"):
            d["api_key"] = PASSWORD_MASK
        # extra_config als dict
        try:
            d["extra_config"] = json.loads(d.get("extra_config") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["extra_config"] = {}
        d["is_active"] = bool(d["is_active"])
        d["trusted"] = bool(d.get("trusted", 0))
        d["last_test_ok"] = bool(d["last_test_ok"]) if d["last_test_ok"] is not None else None
        backends.append(d)

    return {"backends": backends}


# --- Migriert zu Admin-Registry: ki.backends.create (Sprint 06) ---
# @router.post("/")
# async def create_backend(data: KiBackendRequest, user: dict = Depends(get_admin_user)) -> dict:
#     ...

# --- Migriert zu Admin-Registry: ki.backends.update (Sprint 06) ---
# @router.put("/{backend_id}")
# async def update_backend(backend_id: int, data: KiBackendRequest, user: dict = Depends(get_admin_user)) -> dict:
#     ...

# --- Migriert zu Admin-Registry: ki.backends.delete (Sprint 06) ---
# @router.delete("/{backend_id}")
# async def delete_backend(backend_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     ...

# --- Migriert zu Admin-Registry: ki.backends.toggle (Sprint 06) ---
# @router.patch("/{backend_id}/toggle")
# async def toggle_backend(backend_id: int, data: KiBackendToggleRequest, user: dict = Depends(get_admin_user)) -> dict:
#     ...

# --- Migriert zu Admin-Registry: ki.backends.test (Sprint 06) ---
# @router.post("/{backend_id}/test")
# async def test_backend(backend_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     ...
