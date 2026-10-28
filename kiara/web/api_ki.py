"""KIara Admin - KI API.

Verwaltung der KI-Backend-Konfiguration mit drei Bereichen:
1. **Ollama/LLM**: Embedding- und LLM-URLs, Modelle, Prompt-Template.
2. **Cloud-AI**: Anthropic/OpenAI-Integration mit Provider-Auswahl und Routing.
3. **Anonymisierung**: ERP-/LDAP-basierte Pseudonymisierung mit Aho-Corasick-Automat.

Alle Einstellungen werden in der PostgreSQL-Tabelle ``ai_config`` (Singleton, id=1)
gespeichert und zusaetzlich in ``kiara.yaml`` synchronisiert, damit
Laufzeit-Consumer (chain, retriever) die Aenderungen uebernehmen.

Beziehungen:
    - kiara.config             → load_config/save_config fuer YAML-Sync
    - kiara.anonymization.*    → Pseudonymizer, Stammdaten, Automaton
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from kiara.config import DEFAULT_PATHS, DEFAULT_TIMEOUTS, load_config, save_config
from kiara.repositories.config import ConfigRepository
from kiara.repositories.prompt import PromptTemplateRepository
from kiara.shared.constants import (
    ANON_LLM_CACHE_SIZE,
    ANON_LLM_NUM_CTX,
    ANON_LLM_TIMEOUT_SECS,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_CLOUD_PROVIDER,
    PASSWORD_MASK,
)
from kiara.shared.crypto import encrypt, is_encrypted
from kiara.shared.db import db_query, paginated_query
from kiara.shared.enums import RoutingMode
from kiara.web.api_utils import check_ollama_models
from kiara.web.auth import get_admin_user
from kiara.web.config_changelog import log_config_change
from kiara.web.config_utils import safe_json_loads

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/ki", tags=["ki"])

config_repo = ConfigRepository()
prompt_repo = PromptTemplateRepository()


@router.get("/source-profiles")
async def get_chat_source_profiles(user=Depends(get_admin_user)):
    from kiara.retrieval.source_policy import get_source_profiles

    return {"profiles": get_source_profiles(load_config())}


class SourceProfilesRequest(BaseModel):
    profiles: dict[str, dict]


@router.post("/source-profiles")
async def update_chat_source_profiles(data: SourceProfilesRequest, user=Depends(get_admin_user)):
    from kiara.retrieval.source_policy import (
        get_source_profiles,
        validate_source_backend_budget,
        validate_source_profiles,
    )

    config = load_config()
    before = get_source_profiles(config)
    if data.profiles.keys() != before.keys():
        raise HTTPException(status_code=400, detail="Alle drei Quellenprofile sind erforderlich")
    config.setdefault("chat", {})["source_profiles"] = data.profiles
    try:
        validate_source_profiles(config)
        validate_source_backend_budget(config)
        save_config(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Quellenprofile konnten nicht gespeichert werden")
        raise HTTPException(
            status_code=503, detail="Quellenprofile konnten nicht geprueft oder gespeichert werden"
        ) from exc
    log_config_change(
        user,
        "chat",
        "update",
        target_label="Quellenprofile",
        field_name="source_profiles",
        old_value=json.dumps(before, ensure_ascii=False),
        new_value=json.dumps(data.profiles, ensure_ascii=False),
    )
    return {"profiles": get_source_profiles(config), "message": "Quellenprofile gespeichert"}


# -- Modul-Level Cache fuer den Automaten --
_automaton_cache = {
    "automaton": None,
    "last_built": None,
    "erp_count": 0,
    "ldap_count": 0,
}


class KiConfigRequest(BaseModel):
    """Request zum Speichern der KI-Backend-Konfiguration (Ollama, Cloud, Anonymisierung)."""

    embedding_url: str = ""
    embedding_model: str = ""
    llm_url: str = ""
    llm_model: str = ""
    prompt_template: str = ""
    cloud_enabled: bool = False
    cloud_provider: str = DEFAULT_CLOUD_PROVIDER
    cloud_api_key: str = ""
    cloud_model: str = DEFAULT_CLOUD_MODEL
    cloud_base_url: str = ""
    cloud_mode: Literal["auto", "local", "cloud"] = "auto"
    anon_enabled: bool = True
    anon_erp_enabled: bool = False
    anon_erp_path: str = DEFAULT_PATHS["stammdaten"]
    anon_ldap_enabled: bool = True
    anon_word_boundaries: bool = True
    anon_min_term_length: int = 3
    anon_excluded_terms: str = "[]"
    anon_custom_patterns: str = "[]"
    anon_audit_log: bool = True
    anon_custom_entries: str = "[]"
    anon_llm_enabled: bool = False
    anon_llm_model: str = ""
    anon_llm_temperature: str = "0.1"
    anon_llm_confidence: str = "0.7"
    anon_llm_timeout: int = ANON_LLM_TIMEOUT_SECS
    anon_llm_cache_size: int = ANON_LLM_CACHE_SIZE
    anon_llm_num_ctx: int = ANON_LLM_NUM_CTX
    agent_system_prompt: str = ""


class PromptTemplateCreate(BaseModel):
    """Request zum Erstellen eines neuen Prompt-Templates."""

    name: str
    label: str = ""
    description: str = ""
    content: str
    final_prompt: str = ""
    chain_type: str = "standard"
    is_cloud: bool = False
    show_in_chat: bool = False
    badge_color: str = ""
    sort_order: int = 0


class PromptTemplateUpdate(BaseModel):
    """Request zum Aktualisieren eines bestehenden Prompt-Templates."""

    label: str | None = None
    description: str | None = None
    content: str | None = None
    final_prompt: str | None = None
    chain_type: str | None = None
    is_cloud: bool | None = None
    show_in_chat: bool | None = None
    badge_color: str | None = None
    sort_order: int | None = None
    is_active: bool | None = None


class TestBackendRequest(BaseModel):
    """Request zum Testen einer Ollama-Backend-Verbindung."""

    url: str


class TestCloudRequest(BaseModel):
    """Request zum Testen einer Cloud-AI-Verbindung (Anthropic/OpenAI)."""

    provider: str = DEFAULT_CLOUD_PROVIDER
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class TestEmbeddingCloudRequest(BaseModel):
    """Request zum Testen einer Cloud-Embedding-Verbindung (OpenAI/Voyage)."""

    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class TestAnonymizeRequest(BaseModel):
    """Request zum Testen der Anonymisierung mit einem Beispieltext."""

    text: str = Field(..., min_length=1, max_length=100_000)


def _sync_config_section(config: dict, section: str, mapping: dict[str, Any]) -> None:
    """Sync eines Config-Abschnitts in das Config-Dict.

    Unterstuetzt verschachtelte Abschnitte ueber Punkt-Notation in den Keys
    (z.B. ``"erp_source.enabled"`` → ``config[section]["erp_source"]["enabled"]``).

    Args:
        config:  Das gesamte Config-Dict (wird in-place mutiert).
        section: Top-Level-Schluessel (z.B. ``"ollama"``, ``"cloud"``).
        mapping: Dict von ``{yaml_key: value}``; Punkt-Keys erzeugen Sub-Dicts.
    """
    config.setdefault(section, {})
    for key, value in mapping.items():
        if "." in key:
            parts = key.split(".", 1)
            config[section].setdefault(parts[0], {})
            config[section][parts[0]][parts[1]] = value
        else:
            config[section][key] = value


def _sync_to_yaml(
    embedding_url: str, embedding_model: str, llm_url: str, llm_model: str, **cloud_anon_kwargs: Any
) -> None:
    """Sync AI config to kiara.yaml so runtime consumers pick up changes."""
    config = load_config()

    # Backend-Felder nur in YAML schreiben wenn nicht-leer (Phase 13 Kompatibilitaet)
    ollama_updates: dict[str, Any] = {}
    if embedding_url:
        ollama_updates["embedding_url"] = embedding_url
    if embedding_model:
        ollama_updates["embedding_model"] = embedding_model
    if llm_url:
        ollama_updates["llm_url"] = llm_url
        ollama_updates["url"] = llm_url
    if llm_model:
        ollama_updates["llm_model"] = llm_model
    if ollama_updates:
        _sync_config_section(config, "ollama", ollama_updates)

    if "cloud_enabled" in cloud_anon_kwargs:
        config.setdefault("cloud", {}).pop("api_key", None)
        _sync_config_section(
            config,
            "cloud",
            {
                "enabled": cloud_anon_kwargs.get("cloud_enabled", False),
                "provider": cloud_anon_kwargs.get("cloud_provider", DEFAULT_CLOUD_PROVIDER),
                "model": cloud_anon_kwargs.get("cloud_model", DEFAULT_CLOUD_MODEL),
                "base_url": cloud_anon_kwargs.get("cloud_base_url", ""),
                "mode": cloud_anon_kwargs.get("cloud_mode", RoutingMode.AUTO.value),
            },
        )

    if "anon_enabled" in cloud_anon_kwargs:
        _sync_config_section(
            config,
            "anonymization",
            {
                "enabled": cloud_anon_kwargs.get("anon_enabled", True),
                "word_boundaries": cloud_anon_kwargs.get("anon_word_boundaries", True),
                "min_term_length": cloud_anon_kwargs.get("anon_min_term_length", 3),
                "audit_log": cloud_anon_kwargs.get("anon_audit_log", True),
                "erp_source.enabled": cloud_anon_kwargs.get("anon_erp_enabled", False),
                "erp_source.path": cloud_anon_kwargs.get(
                    "anon_erp_path", DEFAULT_PATHS["stammdaten"]
                ),
                "ldap_source.enabled": cloud_anon_kwargs.get("anon_ldap_enabled", True),
                "excluded_terms": safe_json_loads(
                    cloud_anon_kwargs.get("anon_excluded_terms", "[]"), []
                ),
                "custom_patterns": safe_json_loads(
                    cloud_anon_kwargs.get("anon_custom_patterns", "[]"), []
                ),
                "custom_entries": safe_json_loads(
                    cloud_anon_kwargs.get("anon_custom_entries", "[]"), []
                ),
            },
        )
        # LLM-Detection Sub-Block
        if "anon_llm_enabled" in cloud_anon_kwargs:
            llm_temp = cloud_anon_kwargs.get("anon_llm_temperature", "0.1")
            llm_conf = cloud_anon_kwargs.get("anon_llm_confidence", "0.7")
            try:
                llm_temp = float(llm_temp)
            except (TypeError, ValueError):
                llm_temp = 0.1
            llm_temp = max(0.0, min(2.0, llm_temp))
            try:
                llm_conf = float(llm_conf)
            except (TypeError, ValueError):
                llm_conf = 0.7
            llm_conf = max(0.0, min(1.0, llm_conf))
            _sync_config_section(
                config,
                "anonymization",
                {
                    "llm_detection.enabled": cloud_anon_kwargs.get("anon_llm_enabled", False),
                    "llm_detection.model": cloud_anon_kwargs.get("anon_llm_model", ""),
                    "llm_detection.temperature": llm_temp,
                    "llm_detection.confidence_threshold": llm_conf,
                    "llm_detection.timeout_seconds": cloud_anon_kwargs.get("anon_llm_timeout", 10),
                    "llm_detection.cache_size": cloud_anon_kwargs.get("anon_llm_cache_size", 256),
                    "llm_detection.num_ctx": cloud_anon_kwargs.get("anon_llm_num_ctx", 4096),
                },
            )

    save_config(config)


# -- Modell-Familien fuer Klassifizierung --
_EMBEDDING_FAMILIES = {"bert", "nomic-bert", "xlm-roberta"}
_VLM_FAMILIES = {"mllama"}
_VLM_NAME_PATTERN = re.compile(
    r"(?:vision|(?<![a-z])vl(?![a-z])|llava|moondream|minicpm-v)", re.IGNORECASE
)
_EMBEDDING_NAME_PATTERN = re.compile(r"(?:embed|bge-|gte-|e5-|arctic-embed)", re.IGNORECASE)

_CATEGORY_LABELS = {"embedding": "Embedding", "vlm": "VLM", "llm": "LLM"}


def classify_ollama_model(model: dict) -> str:
    """Klassifiziere ein Ollama-Modell als 'embedding', 'vlm' oder 'llm'.

    Heuristik-Kaskade:
    1. ``details.families`` enthaelt bekannte Embedding-Familie → ``"embedding"``
    2. ``details.families`` enthaelt bekannte VLM-Familie → ``"vlm"``
    3. Name matcht VLM-Pattern → ``"vlm"``
    4. Name matcht Embedding-Pattern → ``"embedding"``
    5. Fallback → ``"llm"``
    """
    families = set(model.get("details", {}).get("families") or [])
    name = model.get("name", "")

    # 1. Embedding-Familien (bert, nomic-bert, xlm-roberta)
    if families & _EMBEDDING_FAMILIES:
        return "embedding"

    # 2. VLM-Familien (mllama)
    if families & _VLM_FAMILIES:
        return "vlm"

    # 3. VLM-Namensmuster (vision, vl, llava, moondream, minicpm-v)
    if _VLM_NAME_PATTERN.search(name):
        return "vlm"

    # 4. Embedding-Namensmuster (embed, bge-, gte-, e5-, arctic-embed)
    if _EMBEDDING_NAME_PATTERN.search(name):
        return "embedding"

    # 5. Fallback
    return "llm"


class TestBackendFilteredRequest(BaseModel):
    """Request zum Testen einer Ollama-Verbindung mit optionalem Kategorie-Filter."""

    url: str
    category: Literal["embedding", "vlm", "llm"] | None = None


async def _test_ollama_connection(url: str, category: str | None = None) -> dict:
    """Test Ollama connection and return available models.

    Args:
        url: Ollama-Server-URL.
        category: Optionaler Filter ('embedding', 'vlm', 'llm').
                  ``None`` liefert alle Modelle (Rueckwaertskompatibilitaet).
    """
    ok, names, raw_models, error = await check_ollama_models(
        url,
        timeout=DEFAULT_TIMEOUTS["http_api"],
    )
    if not ok:
        return {"success": False, "models": [], "error": error}

    if category:
        filtered = [m["name"] for m in raw_models if classify_ollama_model(m) == category]
        return {"success": True, "models": filtered}

    return {"success": True, "models": names}


# -- Config CRUD --


# --- Route migriert zu Admin-Registry: ki.config (Sprint 04) ---
# @router.get("/config")
async def get_ki_config(user=Depends(get_admin_user)):
    """Get AI backend configuration including cloud and anonymization settings."""
    row = config_repo.get("ai_config")

    if not row:
        return {"config": None}

    row_dict = dict(row)
    return {
        "config": {
            "embedding_url": row_dict.get("embedding_url", ""),
            "embedding_model": row_dict.get("embedding_model", ""),
            "llm_url": row_dict.get("llm_url", ""),
            "llm_model": row_dict.get("llm_model", ""),
            "prompt_template": row_dict.get("prompt_template") or "",
            "updated_at": row_dict.get("updated_at", ""),
            # Cloud
            "cloud_enabled": bool(row_dict.get("cloud_enabled", 0)),
            "cloud_provider": row_dict.get("cloud_provider", DEFAULT_CLOUD_PROVIDER),
            "cloud_api_key": PASSWORD_MASK if row_dict.get("cloud_api_key") else "",
            "cloud_model": row_dict.get("cloud_model", DEFAULT_CLOUD_MODEL),
            "cloud_base_url": row_dict.get("cloud_base_url", ""),
            "cloud_mode": row_dict.get("cloud_mode", "auto"),
            # Anonymisierung
            "anon_enabled": bool(row_dict.get("anon_enabled", 1)),
            "anon_erp_enabled": bool(row_dict.get("anon_erp_enabled", 0)),
            "anon_erp_path": row_dict.get("anon_erp_path", DEFAULT_PATHS["stammdaten"]),
            "anon_ldap_enabled": bool(row_dict.get("anon_ldap_enabled", 1)),
            "anon_word_boundaries": bool(row_dict.get("anon_word_boundaries", 1)),
            "anon_min_term_length": row_dict.get("anon_min_term_length", 3),
            "anon_excluded_terms": row_dict.get("anon_excluded_terms", "[]"),
            "anon_custom_patterns": row_dict.get("anon_custom_patterns", "[]"),
            "anon_audit_log": bool(row_dict.get("anon_audit_log", 1)),
            "anon_custom_entries": row_dict.get("anon_custom_entries", "[]"),
            # LLM-PII-Erkennung (Schicht 4)
            "anon_llm_enabled": bool(row_dict.get("anon_llm_enabled", 0)),
            "anon_llm_model": row_dict.get("anon_llm_model", ""),
            "anon_llm_temperature": row_dict.get("anon_llm_temperature", "0.1"),
            "anon_llm_confidence": row_dict.get("anon_llm_confidence", "0.7"),
            "anon_llm_timeout": row_dict.get("anon_llm_timeout", 10),
            "anon_llm_cache_size": row_dict.get("anon_llm_cache_size", 256),
            "anon_llm_num_ctx": row_dict.get("anon_llm_num_ctx", 4096),
            # Agent-Systemprompt
            "agent_system_prompt": row_dict.get("agent_system_prompt") or "",
        }
    }


@router.post("/config")
async def save_ki_config(data: KiConfigRequest, user=Depends(get_admin_user)):
    """Save AI backend configuration and sync to YAML."""
    if data.prompt_template and (
        "{context}" not in data.prompt_template or "{question}" not in data.prompt_template
    ):
        raise HTTPException(
            status_code=422,
            detail="Prompt-Template muss die Platzhalter {context} und {question} enthalten",
        )

    if data.agent_system_prompt and "{tools_prompt}" not in data.agent_system_prompt:
        raise HTTPException(
            status_code=422,
            detail="Agent-Systemprompt muss den Platzhalter {tools_prompt} enthalten",
        )

    # F-72: Cloud-API-Key: PASSWORD_MASK-Handling + Verschluesselung
    if data.cloud_api_key == PASSWORD_MASK:
        # Bestehenden Key aus DB behalten
        _existing = config_repo.get("ai_config")
        resolved_cloud_key = _existing.get("cloud_api_key", "") if _existing else ""
        if not resolved_cloud_key:
            raise HTTPException(
                status_code=422,
                detail="PASSWORD_MASK kann ohne bestehenden Cloud-API-Key nicht uebernommen werden",
            )
        # Alten Klartext-Key opportunistisch verschluesseln (analog zu save_cloud_config)
        if not is_encrypted(resolved_cloud_key):
            resolved_cloud_key = encrypt(resolved_cloud_key)
    else:
        resolved_cloud_key = encrypt(data.cloud_api_key) if data.cloud_api_key else ""

    fields = {
        "embedding_url": data.embedding_url,
        "embedding_model": data.embedding_model,
        "llm_url": data.llm_url,
        "llm_model": data.llm_model,
        "prompt_template": data.prompt_template,
        "cloud_enabled": data.cloud_enabled,
        "cloud_provider": data.cloud_provider,
        "cloud_api_key": resolved_cloud_key,
        "cloud_model": data.cloud_model,
        "cloud_base_url": data.cloud_base_url,
        "cloud_mode": data.cloud_mode,
        "anon_enabled": data.anon_enabled,
        "anon_erp_enabled": data.anon_erp_enabled,
        "anon_erp_path": data.anon_erp_path,
        "anon_ldap_enabled": data.anon_ldap_enabled,
        "anon_word_boundaries": int(data.anon_word_boundaries),
        "anon_min_term_length": data.anon_min_term_length,
        "anon_excluded_terms": data.anon_excluded_terms,
        "anon_custom_patterns": data.anon_custom_patterns,
        "anon_audit_log": int(data.anon_audit_log),
        "anon_custom_entries": data.anon_custom_entries,
        "anon_llm_enabled": data.anon_llm_enabled,
        "anon_llm_model": data.anon_llm_model,
        "anon_llm_temperature": data.anon_llm_temperature,
        "anon_llm_confidence": data.anon_llm_confidence,
        "anon_llm_timeout": data.anon_llm_timeout,
        "anon_llm_cache_size": data.anon_llm_cache_size,
        "anon_llm_num_ctx": data.anon_llm_num_ctx,
        "agent_system_prompt": data.agent_system_prompt,
    }

    config_repo.set("ai_config", fields)
    log_config_change(user, area="ai_config", action="update", source="gui")

    _sync_to_yaml(
        data.embedding_url,
        data.embedding_model,
        data.llm_url,
        data.llm_model,
        cloud_enabled=data.cloud_enabled,
        cloud_provider=data.cloud_provider,
        cloud_api_key=resolved_cloud_key,
        cloud_model=data.cloud_model,
        cloud_base_url=data.cloud_base_url,
        cloud_mode=data.cloud_mode,
        anon_enabled=data.anon_enabled,
        anon_erp_enabled=data.anon_erp_enabled,
        anon_erp_path=data.anon_erp_path,
        anon_ldap_enabled=data.anon_ldap_enabled,
        anon_word_boundaries=data.anon_word_boundaries,
        anon_min_term_length=data.anon_min_term_length,
        anon_excluded_terms=data.anon_excluded_terms,
        anon_custom_patterns=data.anon_custom_patterns,
        anon_audit_log=data.anon_audit_log,
        anon_custom_entries=data.anon_custom_entries,
        anon_llm_enabled=data.anon_llm_enabled,
        anon_llm_model=data.anon_llm_model,
        anon_llm_temperature=data.anon_llm_temperature,
        anon_llm_confidence=data.anon_llm_confidence,
        anon_llm_timeout=data.anon_llm_timeout,
        anon_llm_cache_size=data.anon_llm_cache_size,
        anon_llm_num_ctx=data.anon_llm_num_ctx,
    )

    logger.info("AI config saved by %s", user["username"])
    return {"message": "KI-Konfiguration gespeichert"}


@router.delete("/config")
async def delete_ki_config(user=Depends(get_admin_user)):
    """Reset AI configuration to defaults."""
    config_repo.delete("ai_config")
    log_config_change(user, area="ai_config", action="delete", source="gui")

    config = load_config()
    _sync_config_section(
        config,
        "ollama",
        {
            "embedding_url": "",
            "llm_url": "",
        },
    )

    _sync_config_section(
        config,
        "cloud",
        {
            "enabled": False,
            "provider": DEFAULT_CLOUD_PROVIDER,
            "model": DEFAULT_CLOUD_MODEL,
            "base_url": "",
            "mode": "auto",
        },
    )

    _sync_config_section(
        config,
        "anonymization",
        {
            "enabled": True,
            "word_boundaries": True,
            "min_term_length": 3,
            "audit_log": True,
            "erp_source.enabled": False,
            "erp_source.path": DEFAULT_PATHS["stammdaten"],
            "ldap_source.enabled": True,
            "excluded_terms": [],
            "custom_patterns": [],
            "custom_entries": [],
            "llm_detection.enabled": False,
            "llm_detection.model": "",
            "llm_detection.temperature": 0.1,
            "llm_detection.confidence_threshold": 0.7,
            "llm_detection.timeout_seconds": ANON_LLM_TIMEOUT_SECS,
            "llm_detection.cache_size": ANON_LLM_CACHE_SIZE,
            "llm_detection.num_ctx": ANON_LLM_NUM_CTX,
        },
    )

    save_config(config)

    logger.info("AI config reset by %s", user["username"])
    return {"message": "KI-Konfiguration zurueckgesetzt"}


# -- Backend-Tests --


@router.post("/test-embedding")
async def test_embedding_backend(data: TestBackendRequest, user=Depends(get_admin_user)):
    """Test embedding backend connection and return available models."""
    result = await _test_ollama_connection(data.url)
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"models": result["models"]}


@router.post("/test-llm")
async def test_llm_backend(data: TestBackendRequest, user=Depends(get_admin_user)):
    """Test LLM backend connection and return available models."""
    result = await _test_ollama_connection(data.url)
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"models": result["models"]}


@router.post("/test-connection")
async def test_backend_filtered(data: TestBackendFilteredRequest, user=Depends(get_admin_user)):
    """Test Ollama-Verbindung mit optionalem Kategorie-Filter.

    Liefert nur Modelle zurueck, die zur angegebenen Kategorie passen
    (embedding/vlm/llm). Ohne Kategorie werden alle Modelle geliefert.
    """
    result = await _test_ollama_connection(data.url, data.category)
    if not result["success"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"models": result["models"], "category": data.category}


# -- Cloud-Test --


@router.post("/test-cloud")
async def test_cloud_backend(data: TestCloudRequest, user=Depends(get_admin_user)):
    """Test Cloud-AI-Verbindung (Anthropic / OpenAI)."""
    provider = data.provider or DEFAULT_CLOUD_PROVIDER
    api_key = data.api_key

    if not api_key:
        raise HTTPException(status_code=422, detail="API-Key ist erforderlich")

    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            models = client.models.list()
            model_ids = [m.id for m in models.data] if hasattr(models, "data") else []
            return {"success": True, "models": model_ids}

        elif provider == "openai":
            import openai

            base_url = data.base_url or None
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
            model_ids = [m.id for m in models.data] if hasattr(models, "data") else []
            return {"success": True, "models": model_ids}

        else:
            raise HTTPException(status_code=422, detail=f"Unbekannter Provider: {provider}")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Cloud-Verbindungstest fehlgeschlagen (provider=%s)", provider)
        raise HTTPException(
            status_code=502,
            detail="Cloud-Verbindung fehlgeschlagen, Details siehe Server-Logs",
        )


# -- Cloud-Embedding-Test --


@router.post("/test-embedding-cloud")
async def test_embedding_cloud(data: TestEmbeddingCloudRequest, user=Depends(get_admin_user)):
    """Test Cloud-Embedding-Verbindung (OpenAI / Voyage AI) mit Probe-Embedding."""
    provider = data.provider
    api_key = data.api_key
    model = data.model

    if not api_key:
        raise HTTPException(status_code=422, detail="API-Key ist erforderlich")
    if not model:
        raise HTTPException(status_code=422, detail="Modell ist erforderlich")
    if provider not in ("openai", "voyage"):
        raise HTTPException(status_code=422, detail=f"Ungueltiger Embedding-Provider: {provider}")

    from kiara.admin.services.ki_backend_service import _test_embedding_cloud

    success, error, dimension = _test_embedding_cloud(provider, api_key, model, data.base_url)

    if success:
        result = {"success": True, "dimension": dimension}
        return result
    raise HTTPException(status_code=502, detail=error or "Probe-Embedding fehlgeschlagen")


# -- Anonymisierung --


@router.post("/test-anonymize")
async def test_anonymize(data: TestAnonymizeRequest, user=Depends(get_admin_user)):
    """Beispieltext anonymisieren (Vorschau)."""
    config = load_config()

    from kiara.anonymization.anonymizer import Anonymizer
    from kiara.anonymization.stammdaten import build_automaton_from_sources

    automaton = build_automaton_from_sources(config)
    pseudonymizer = Anonymizer(automaton, config)
    anonymized = pseudonymizer.anonymize(data.text)

    matches = []
    for pseudonym, original in pseudonymizer.reverse_map.items():
        matches.append(
            {
                "original": original,
                "pseudonym": pseudonym,
            }
        )

    return {
        "original": data.text,
        "anonymized": anonymized,
        "matches": matches,
        "stats": pseudonymizer.stats,
    }


@router.post("/reload-automaton")
async def reload_automaton(user=Depends(get_admin_user)):
    """Stammdaten neu laden und Automaten aktualisieren."""
    config = load_config()

    from kiara.anonymization.stammdaten import build_automaton_from_sources

    automaton = build_automaton_from_sources(config)

    _automaton_cache["automaton"] = automaton
    _automaton_cache["last_built"] = datetime.now(timezone.utc).isoformat()
    _automaton_cache["entry_count"] = getattr(automaton, "entry_count", 0)
    _automaton_cache["erp_count"] = getattr(automaton, "source_counts", {}).get("erp", 0)
    _automaton_cache["ldap_count"] = getattr(automaton, "source_counts", {}).get("ldap", 0)

    logger.info(
        "Automaton rebuilt by %s: %d entries", user["username"], _automaton_cache["entry_count"]
    )
    return {
        "entry_count": _automaton_cache["entry_count"],
        "sources": {
            "erp": _automaton_cache["erp_count"],
            "ldap": _automaton_cache["ldap_count"],
        },
    }


# --- Route migriert zu Admin-Registry: ki.automaton_stats (Sprint 04) ---
# @router.get("/automaton-stats")
async def get_automaton_stats(user=Depends(get_admin_user)):
    """Automaten-Statistiken."""
    return {
        "entry_count": _automaton_cache.get("entry_count", 0),
        "erp_entries": _automaton_cache.get("erp_count", 0),
        "ldap_entries": _automaton_cache.get("ldap_count", 0),
        "last_built": _automaton_cache.get("last_built"),
    }


# -- Audit-Log --


# --- Route migriert zu Admin-Registry: ki.audit (Sprint 04) ---
# @router.get("/audit")
async def get_audit_log(
    user=Depends(get_admin_user),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """Audit-Log paginiert abrufen."""
    with db_query() as db:
        result = paginated_query(
            db,
            table="audit_anonymization a",
            columns="a.*, u.username",
            joins="LEFT JOIN users u ON a.user_id = u.id",
            order_by="a.timestamp DESC",
            page=page,
            page_size=per_page,
        )

    data = []
    for d in result["data"]:
        data.append(
            {
                "id": d["id"],
                "timestamp": d["timestamp"],
                "username": d.get("username", ""),
                "session_id": d["session_id"],
                "question_hash": d["question_hash"],
                "entity_count": d["entity_count"],
                "entity_types": d["entity_types"],
                "route": d["route"],
                "content_class": d.get("content_class"),
                "cloud_provider": d.get("cloud_provider"),
                "cloud_model": d.get("cloud_model"),
                "response_time_ms": d.get("response_time_ms"),
            }
        )

    return {"data": data, "total": result["total"], "page": page}


# -- Prompt-Templates --


def _validate_template_content(content: str, chain_type: str) -> None:
    """Validiert Prompt-Template-Inhalt basierend auf dem chain_type.

    Raises:
        HTTPException(422): Bei fehlenden Pflicht-Platzhaltern.
    """
    if chain_type == "standard":
        if "{context}" not in content or "{question}" not in content:
            raise HTTPException(
                status_code=422,
                detail="Standard-Template muss die Platzhalter {context} und {question} enthalten",
            )
    elif chain_type in ("agent", "admin_agent"):
        if "{tools_prompt}" not in content:
            raise HTTPException(
                status_code=422,
                detail="Agent-Template muss den Platzhalter {tools_prompt} enthalten",
            )
    elif chain_type == "enrichment":
        if not content or not content.strip():
            raise HTTPException(
                status_code=422,
                detail="Enrichment-Template darf nicht leer sein",
            )
        if "{chunk_text}" not in content:
            raise HTTPException(
                status_code=422,
                detail="Enrichment-Template muss den Platzhalter {chunk_text} enthalten",
            )
    # unchained, rewriter: keine Pflicht-Platzhalter


def _template_row_to_dict(row) -> dict:
    """Konvertiert eine DB-Row in ein Template-Dict."""
    d = dict(row)
    return {
        "id": d["id"],
        "name": d["name"],
        "label": d.get("label", ""),
        "description": d.get("description", ""),
        "content": d["content"],
        "final_prompt": d.get("final_prompt", ""),
        "chain_type": d["chain_type"],
        "is_cloud": bool(d.get("is_cloud", 0)),
        "show_in_chat": bool(d.get("show_in_chat", 0)),
        "badge_color": d.get("badge_color", ""),
        "sort_order": d.get("sort_order", 0),
        "is_system": bool(d.get("is_system", 0)),
        "is_active": bool(d.get("is_active", 1)),
        "created_at": d.get("created_at", ""),
        "updated_at": d.get("updated_at", ""),
    }


# --- Route migriert zu Admin-Registry: ki.templates.list (Sprint 04) ---
# @router.get("/prompt-templates")
async def list_prompt_templates(user=Depends(get_admin_user)):
    """Alle Prompt-Templates auflisten."""
    rows = prompt_repo.list_all(sort="sort_order ASC, id ASC")
    return {"data": [_template_row_to_dict(r) for r in rows]}


# --- Migriert zu Admin-Registry: ki.templates.create (Sprint 06) ---
# @router.post("/prompt-templates")
# async def create_prompt_template(data: PromptTemplateCreate, user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: ki.templates.update (Sprint 06) ---
# @router.put("/prompt-templates/{template_id}")
# async def update_prompt_template(template_id: int, data: PromptTemplateUpdate, user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: ki.templates.delete (Sprint 06) ---
# @router.delete("/prompt-templates/{template_id}")
# async def delete_prompt_template(template_id: int, user=Depends(get_admin_user)):
#     ...
