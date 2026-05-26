"""Configuration management for KIara.

Laedt, validiert und speichert die YAML-Konfiguration. Stellt
Helper-Funktionen fuer haeufig benoetigte Config-Zugriffe bereit.

Beziehungen:
    - kiara.shared.constants liefert die Default-Werte fuer Magic Numbers
    - Alle Module importieren load_config() oder die get_*-Helper
    - /etc/kiara/kiara.yaml ist das primaere Config-File (conffile)
"""

import copy
import json
import logging
import os
from enum import Enum
from typing import Any

import yaml

from kiara.shared.constants import (
    ACCEPT_ALL_BLOCKLIST_DEFAULT,
    ADMIN_PORT_DEFAULT,
    ANON_LLM_CACHE_SIZE,
    ANON_LLM_CONFIDENCE_THRESHOLD_DEFAULT,
    ANON_LLM_NUM_CTX,
    ANON_LLM_TIMEOUT_SECS,
    ANON_MIN_TERM_LENGTH_DEFAULT,
    BACKUP_MAX_COUNT_DEFAULT,
    CACHE_MAX_SIZE_DEFAULT,
    CACHE_TTL_SECS_DEFAULT,
    CLOUD_BUDGET_ALERT_THRESHOLD_PCT,
    CLOUD_MONTHLY_BUDGET_EUR,
    COLBERT_DEFAULT_MODEL,
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_AGENT_NUM_CTX,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_ELASTICSEARCH_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_CONTEXT_CHUNKS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QDRANT_URL,
    DEFAULT_REWRITER_NUM_CTX,
    DEFAULT_SUMMARY_CHUNK_SIZE,
    DEFAULT_UNCHAINED_TEMPERATURE,
    DEFAULT_VECTOR_SIZE,
    ENRICHMENT_MAX_CONTEXT_CHARS,
    EXPORT_MAX_AGE_DAYS_DEFAULT,
    GRAPH_MAX_CHUNKS_DEFAULT,
    GRAPH_MAX_TRAVERSAL_DEPTH_DEFAULT,
    HEALTH_CHECK_INTERVAL_SECS_DEFAULT,
    HEALTH_LOG_INTERVAL_SECS_DEFAULT,
    INGESTION_DEFAULT_BATCH_SIZE,
    MCP_PORT_DEFAULT,
    RERANK_BATCH_SIZE_DEFAULT,
    RERANK_FETCH_MULTIPLIER_DEFAULT,
    RERANK_TOP_K_DEFAULT,
    TRACING_DEFAULT_RETENTION_DAYS,
    WEB_PORT_DEFAULT,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "/etc/kiara/kiara.yaml"

# ---------------------------------------------------------------------------
# Zentrale System-Pfade (Single Source of Truth fuer alle Default-Pfade)
# ---------------------------------------------------------------------------
DEFAULT_PATHS: dict[str, str] = {
    "vectordb_data": "/var/lib/kiara/vectordb",
    "secret_key": "/var/lib/kiara/secret.key",
    "backups": "/var/lib/kiara/backups",
    "exports": "/var/lib/kiara/exports",
    "mail_cache": "/var/lib/kiara/mail-cache",
    "sql_cache": "/var/lib/kiara/sql-cache",
    "log": "/var/log/kiara/kiara.log",
    "config": "/etc/kiara/kiara.yaml",
    "templates": "/etc/kiara/templates",
    "stammdaten": "/etc/kiara/stammdaten",
    "erp_data": "/etc/kiara/erp-data/",
}

# ---------------------------------------------------------------------------
# Config-Pfad-Helper (kapseln die .get()-Ketten)
# ---------------------------------------------------------------------------


def get_vectordb_data_path(config: dict) -> str:
    """Return the VectorDB data path from config."""
    return (config.get("vectordb") or {}).get("path", DEFAULT_PATHS["vectordb_data"])


def get_backup_dir(config: dict) -> str:
    """Return the backup directory from config."""
    return (config.get("backup") or {}).get("dir", DEFAULT_PATHS["backups"])


def get_config_file_path() -> str:
    """Return the YAML config file path (env override supported)."""
    return os.environ.get("KIARA_CONFIG", DEFAULT_PATHS["config"])


def get_template_dir(config: dict) -> str:
    """Return the workflow template directory from config."""
    return (config.get("workflows") or {}).get("template_dir", DEFAULT_PATHS["templates"])


def get_export_dir(config: dict) -> str:
    """Return the workflow export directory from config."""
    return (config.get("workflows") or {}).get("output_dir", DEFAULT_PATHS["exports"])


# ---------------------------------------------------------------------------
# Zentrale Timeout-Defaults (Sekunden)
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUTS: dict[str, float] = {
    "http_health": 5.0,
    "http_api": 10.0,
    "ldap_connect": 10,
    "ldap_receive": 30,
    "smtp": 15,
    "subprocess_quick": 5,
    "subprocess_service_logs": 15,
    "subprocess_restart": 30,
    "subprocess_mount": 30,
    "subprocess_umount": 10,
    "graph_extraction": 120.0,
    "erp_query": 30.0,
    "erp_bulk": 300.0,
    "db_connect": 5,
    "ddu_convert": 600.0,
    "subprocess_network_ping": 15,
    "subprocess_network_scan": 30,
    "subprocess_network_smb": 10,
    "subprocess_network_dns": 10,
    "subprocess_network_http": 10,
    "subprocess_network_traceroute": 30,
}


def get_timeout(config: dict, key: str) -> float:
    """Return a timeout value from config, falling back to DEFAULT_TIMEOUTS."""
    return (config.get("timeouts") or {}).get(key, DEFAULT_TIMEOUTS.get(key, 30.0))


_SYSTEM_EVENTS_LEVEL_ALLOWLIST = ("INFO", "WARNING", "ERROR", "CRITICAL")


def get_system_events_config(config: dict) -> dict:
    """Liefert den normalisierten `system_events`-Config-Block.

    Defaults: min_level="INFO", retention_days=90. Diese Funktion ist nur fuer
    direkte Handler-/CLI-Nutzung defensiv; regulaer geladene YAML-Config wird
    in _validate() fail-closed auf ungueltige Werte geprueft.
    """
    block = config.get("system_events") if isinstance(config, dict) else None
    if not isinstance(block, dict):
        block = {}

    raw_level = str(block.get("min_level", "INFO")).upper()
    min_level = raw_level if raw_level in _SYSTEM_EVENTS_LEVEL_ALLOWLIST else "INFO"

    retention_raw = block.get("retention_days", 90)
    try:
        retention_days = int(retention_raw)
        if retention_days <= 0:
            retention_days = 90
    except (TypeError, ValueError):
        retention_days = 90

    return {"min_level": min_level, "retention_days": retention_days}


DEFAULT_CONFIG = {
    "datasource": {
        "path": "/mnt/knowledgebase",
        "file_types": [
            ".docx",
            ".txt",
            ".pdf",
            ".ps1",
            ".cs",
            ".vb",
            ".xml",
            ".odt",
            ".xlsx",
            ".xlsm",
            ".md",
            ".py",
            ".json",
            ".html",
            ".bat",
            ".sh",
            ".config",
            ".ini",
            ".pl",
            ".js",
            ".css",
            ".sln",
            ".pptx",
            ".csv",
            ".dxf",
            ".step",
            ".stp",
            ".iges",
            ".igs",
        ],
        "exclude_patterns": ["*/bin/*", "*/obj/*", "*/.git/*"],
    },
    "ollama": {
        "url": DEFAULT_OLLAMA_URL,
        "llm_url": "",
        "embedding_url": "",
        "llm_model": DEFAULT_LLM_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "vlm_model": "",
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "num_ctx": DEFAULT_LLM_NUM_CTX,
    },
    "vectordb": {
        "backend": "qdrant",
        "path": "/var/lib/kiara/vectordb",
        "collection": "knowledgebase",
        "vector_size": DEFAULT_VECTOR_SIZE,
        "qdrant_url": DEFAULT_QDRANT_URL,
        "qdrant_api_key": "",
        "qdrant_grpc": False,
    },
    "ingestion": {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "batch_size": INGESTION_DEFAULT_BATCH_SIZE,
        "vlm_prompt": "",
        "accept_all_blocklist": sorted(ACCEPT_ALL_BLOCKLIST_DEFAULT),
    },
    "web": {
        "host": "0.0.0.0",
        "port": WEB_PORT_DEFAULT,
        "admin_port": ADMIN_PORT_DEFAULT,
        "mcp_port": MCP_PORT_DEFAULT,
        "mcp_enabled": False,
        "workers": 4,
        "trusted_origins": [],
    },
    "database": {
        "backend": "postgresql",
        "postgresql": {
            "dsn": "postgresql://kiara:kiara@localhost:5432/kiara",
            "statement_timeout_ms": 30000,
        },
    },
    "timeouts": dict(DEFAULT_TIMEOUTS),
    "cloud": {
        "enabled": False,
        "provider": "openai",
        "api_key": "",
        "model": DEFAULT_OPENAI_MODEL,
        "base_url": "",
        "mode": "auto",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "routing": {
            "mode": "auto",
            "confidence_threshold": 0.3,
            "min_relevant_chunks": 2,
            "cloud_keywords": [],
        },
        "budget": {
            "monthly_limit_eur": CLOUD_MONTHLY_BUDGET_EUR,
            "alert_threshold_pct": CLOUD_BUDGET_ALERT_THRESHOLD_PCT,
        },
    },
    "security": {
        "cookie_secure": False,
        "hsts_max_age": 0,
        # V30 Welle 5 (#1049): Default-Flip — Strict-CSP ist jetzt der
        # Enforce-Modus. Report-Only bleibt als Observability-Kanal aktiv
        # (Browser sendet Violations an csp_report_uri auch wenn sie
        # enforcement-seitig bereits geblockt werden).
        "csp_strict": True,
        "csp_report_only": True,
        "csp_report_uri": "/api/admin/csp-report",
    },
    "tools": {
        "enabled": False,
        "max_calls_per_query": 3,
        "erp": {
            "backend": "mock",
            "rest_url": "",
            "rest_api_key": "",
            "sql_dsn": "",
            "sql_driver": "auto",
            "mock_data": "",
            "mapping": {},
            "connections": [],
            "tool_assignments": {},
        },
        "permissions": {
            "ERP-Leser": ["erp_order_status", "erp_stock_level"],
            "ERP-Vertrieb": [
                "erp_order_status",
                "erp_stock_level",
                "erp_customer_info",
                "generate_status_mail",
            ],
            "ERP-Fertigung": ["erp_production_status"],
            "IT-Admin": ["*"],
        },
    },
    "chat": {
        "default_mode": "standard",
        "unchained": {
            "temperature": DEFAULT_UNCHAINED_TEMPERATURE,
        },
        "agent": {
            "max_iterations": DEFAULT_AGENT_MAX_ITERATIONS,
            "num_ctx": DEFAULT_AGENT_NUM_CTX,
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "raw_chunk_compaction": True,
            "final_reranking": True,
            "invoice_number_grouping": True,
        },
        "rewriter": {
            "num_ctx": DEFAULT_REWRITER_NUM_CTX,
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "rewrite_prompt": "",
        },
    },
    "retrieval": {
        "hybrid_search": "none",
        "elasticsearch_url": DEFAULT_ELASTICSEARCH_URL,
        "elasticsearch_index": "kiara",
        "elasticsearch_fuzziness": "none",
        "reranking": False,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_top_k": RERANK_TOP_K_DEFAULT,
        "rerank_fetch_multiplier": RERANK_FETCH_MULTIPLIER_DEFAULT,
        "rerank_batch_size": RERANK_BATCH_SIZE_DEFAULT,
        "query_expansion": False,
        "rrf_k": 60,
        "max_context_chunks": DEFAULT_MAX_CONTEXT_CHUNKS,
        "agent_dedup": True,
        "colbert": {
            "enabled": False,
            "model": COLBERT_DEFAULT_MODEL,
            "url": "",
        },
    },
    "workflows": {
        "enabled": True,
        "template_dir": "/etc/kiara/templates",
        "output_dir": "/var/lib/kiara/exports",
        "max_context_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
        "summary_chunk_size": DEFAULT_SUMMARY_CHUNK_SIZE,
        "export_max_age_days": EXPORT_MAX_AGE_DAYS_DEFAULT,
    },
    "health": {
        "check_interval_seconds": HEALTH_CHECK_INTERVAL_SECS_DEFAULT,
        "log_interval_seconds": HEALTH_LOG_INTERVAL_SECS_DEFAULT,
        "disk_warning_gb": 1.0,
        "disk_critical_gb": 0.1,
    },
    "monitoring": {
        "persist_interval_seconds": 60,
        "downsample_hour": 3,
        "raw_retention_hours": 24,
        "retention_5min_days": 30,
        "retention_1h_days": 365,
        "qdrant_data_path": "",  # Leer = Auto-Detect (Docker-Volume / /opt/qdrant)
    },
    "backup": {
        "dir": "/var/lib/kiara/backups",
        "max_backups": BACKUP_MAX_COUNT_DEFAULT,
        "include_vectordb_data": False,
    },
    "cache": {
        "enabled": True,
        "max_size": CACHE_MAX_SIZE_DEFAULT,
        "ttl_seconds": CACHE_TTL_SECS_DEFAULT,
    },
    "graph": {
        "enabled": False,
        "extract_on_ingestion": True,
        "erp_sync_enabled": True,
        "max_traversal_depth": GRAPH_MAX_TRAVERSAL_DEPTH_DEFAULT,
        "max_chunks": GRAPH_MAX_CHUNKS_DEFAULT,
        "extraction_model": "",
    },
    "enrichment": {
        "enabled": False,
        "max_context_chars": ENRICHMENT_MAX_CONTEXT_CHARS,
        "max_workers": 2,
        "late_chunking": True,
    },
    "tracing": {
        "enabled": False,
        "exporter": "postgresql",  # "postgresql", "otlp" oder "both"
        "retention_days": TRACING_DEFAULT_RETENTION_DAYS,
        "pipelines": {
            "retrieval": True,
            "agent": True,
            "ingestion": True,
        },
        "otlp": {
            "endpoint": "",  # z.B. "http://otel-collector.local:4317"
            "insecure": True,  # Kein TLS im internen Netz
        },
        "chat_debug": False,  # Trace-Debug-Bar in Chat-UI
    },
    "system_events": {
        "min_level": "INFO",
        "retention_days": 90,
    },
    "anonymization": {
        "enabled": True,
        "word_boundaries": True,
        "min_term_length": ANON_MIN_TERM_LENGTH_DEFAULT,
        "erp": {
            "backend": "csv",
            "csv_dir": "/etc/kiara/erp-data/",
        },
        "erp_source": {
            "enabled": False,
            "type": "csv",
            "path": "/etc/kiara/stammdaten",
        },
        "ldap": {
            "enrich_from_db": True,
            "enrich_from_ldap": False,
        },
        "ldap_source": {"enabled": True},
        "placeholder_format": "bracket",
        "custom_patterns": [],
        "excluded_terms": [],
        "audit_log": True,
        "custom_entries": [],
        "llm_detection": {
            "enabled": False,
            "model": "",
            "temperature": 0.1,
            "confidence_threshold": ANON_LLM_CONFIDENCE_THRESHOLD_DEFAULT,
            "timeout_seconds": ANON_LLM_TIMEOUT_SECS,
            "cache_size": ANON_LLM_CACHE_SIZE,
            "num_ctx": ANON_LLM_NUM_CTX,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    if not isinstance(override, dict):
        logger.warning(
            "Config-Override ist kein dict (Typ: %s), wird ignoriert", type(override).__name__
        )
        return base.copy()
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file, merged with defaults.

    Args:
        config_path: Path to YAML config file. Falls back to
                     KIARA_CONFIG env var, then DEFAULT_CONFIG_PATH.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If an explicitly specified config file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = config_path or os.environ.get("KIARA_CONFIG") or DEFAULT_CONFIG_PATH

    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                user_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"Ungueltiges YAML in {path}: {exc}") from exc
        if user_config is None:
            user_config = {}
        elif not isinstance(user_config, dict):
            raise ValueError(
                f"Config in {path} muss ein YAML-Mapping (dict) sein, "
                f"ist aber {type(user_config).__name__}"
            )
        config = _deep_merge(config, user_config)
    elif config_path:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _validate(config)
    return config


def _normalize_for_yaml(value: Any) -> Any:
    """Konvertiert Enum-Werte rekursiv zu primitiven .value-Strings.

    Hintergrund: yaml.safe_dump kann (str, Enum)-Subklassen nicht serialisieren,
    yaml.dump erzeugt !!python/object/apply-Tags die yaml.safe_load ablehnt.
    Vor dem Dump werden Enums daher zu ihrem zugrundeliegenden Wert reduziert.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _normalize_for_yaml(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_yaml(item) for item in value]
    return value


def save_config(config: dict, config_path: str | None = None) -> None:
    """Validate and write configuration back to YAML file.

    Schreibt via sudo tee damit die Datei root:root bleibt.
    Fallback auf direktes Schreiben (z.B. als root oder in Tests).

    Args:
        config: Configuration dictionary to save.
        config_path: Path to write to. Falls back to
                     KIARA_CONFIG env var, then DEFAULT_CONFIG_PATH.
    """
    import subprocess

    _validate(config)
    path = config_path or os.environ.get("KIARA_CONFIG") or DEFAULT_CONFIG_PATH
    content = yaml.safe_dump(
        _normalize_for_yaml(config),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    # Versuch 1: sudo tee (kiara-User hat sudoers-Eintrag)
    try:
        result = subprocess.run(
            ["sudo", "tee", path],
            input=content,
            text=True,
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: direkt schreiben (als root oder in Tests)
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Secret-Masking fuer /api/admin/settings (Bug #610)
# ---------------------------------------------------------------------------

_API_KEY_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("vectordb", "qdrant_api_key"),
    ("cloud", "api_key"),
    ("tools", "erp", "rest_api_key"),
)
"""Pfade zu Klartext-API-Keys, die beim Ausliefern ans Frontend maskiert werden.

DSN-Felder (`database.postgresql.dsn`, `tools.erp.sql_dsn`) sind bewusst NICHT
enthalten — komplettes Masking wuerde die URL-Anzeige zerstoeren, selektives
Passwort-Masking in URL erfordert eigene Parsing-Logik (separates TODO).
Verschluesselte Credentials (Fernet-Prefix `gAAAAA`) in ERP-Connections duerfen
ausgeliefert werden, da sie ohne Secret-Key ohnehin nicht verwertbar sind.
"""


def _walk_to_parent(d: dict, path: tuple[str, ...]) -> dict | None:
    """Navigiert bis zum Parent des letzten Pfadelements. None wenn Pfad fehlt."""
    node: Any = d
    for key in path[:-1]:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def mask_config_secrets(config: dict) -> dict:
    """Gibt eine Kopie der Config zurueck, in der gesetzte API-Keys durch
    ``PASSWORD_MASK`` ersetzt sind.

    Leere Secret-Felder bleiben leer (kein Masken-Leak, dass der Wert gesetzt ist).
    Aufrufer nutzt dies vor der Uebermittlung ans Admin-Frontend.
    """
    import copy

    from kiara.shared.constants import PASSWORD_MASK

    result = copy.deepcopy(config)
    for path in _API_KEY_CONFIG_PATHS:
        parent = _walk_to_parent(result, path)
        if parent is None:
            continue
        leaf = path[-1]
        if parent.get(leaf):
            parent[leaf] = PASSWORD_MASK
    return result


def restore_config_secrets(incoming: dict, existing: dict) -> dict:
    """Ersetzt ``PASSWORD_MASK`` in ``incoming`` durch den zugehoerigen Wert aus
    ``existing``.

    Wird beim Settings-Update aufgerufen, damit der Admin beim Speichern ohne
    expliziten Key-Eintrag den gespeicherten Wert nicht versehentlich loescht.
    Leerer String in ``incoming`` bleibt leerer String (= explizit geloescht).
    """
    import copy

    from kiara.shared.constants import PASSWORD_MASK

    result = copy.deepcopy(incoming)
    for path in _API_KEY_CONFIG_PATHS:
        new_parent = _walk_to_parent(result, path)
        if new_parent is None:
            continue
        leaf = path[-1]
        if new_parent.get(leaf) == PASSWORD_MASK:
            old_parent = _walk_to_parent(existing, path)
            new_parent[leaf] = (old_parent or {}).get(leaf, "")
    return result


# ---------------------------------------------------------------------------
# DB-only Backend-Wrapper (Phase 13)
# ---------------------------------------------------------------------------


def _get_backend(category: str, *, required: bool = True) -> dict | None:
    """Holt das naechste verfuegbare Backend fuer eine Kategorie.

    Args:
        category: Backend-Kategorie (embedding, llm, vlm, cloud).
        required: True = RuntimeError bei Fehler, False = None zurueckgeben.

    Returns:
        Backend-Dict oder None (nur bei required=False).

    Raises:
        RuntimeError: Wenn required=True und kein Backend verfuegbar.
    """
    from kiara.backend_selector import get_next_backend

    try:
        result = get_next_backend(category)
        if not result and required:
            raise RuntimeError(
                f"Kein {category}-Backend in ki_backends konfiguriert. "
                "Bitte in der Admin-UI unter KI-Backends ein Backend anlegen."
            )
        return result
    except Exception as exc:
        if required:
            raise
        logger.debug("Backend-Abfrage '%s' fehlgeschlagen: %s", category, exc)
        return None


def get_embedding_backend(config: dict) -> dict:
    """Return Embedding-Backend als ``{"url", "model"}`` aus ki_backends.

    Raises RuntimeError wenn kein aktives Embedding-Backend konfiguriert ist.
    """
    backend = _get_backend("embedding", required=True)
    return {"url": backend["url"], "model": backend["model"]}


def get_embedding_model(config: dict) -> str:
    """Return das Embedding-Modell aus ki_backends."""
    return get_embedding_backend(config)["model"]


def get_llm_backend(config: dict) -> dict:
    """Return LLM-Backend als ``{"url", "model"}`` aus ki_backends.

    Raises RuntimeError wenn kein aktives LLM-Backend konfiguriert ist.
    """
    backend = _get_backend("llm", required=True)
    return {"url": backend["url"], "model": backend["model"]}


def get_vlm_backend(config: dict) -> dict | None:
    """Return VLM-Backend als dict oder None wenn nicht konfiguriert."""
    backend = _get_backend("vlm", required=False)
    if backend:
        return {"url": backend["url"], "model": backend["model"]}
    return None


def get_cloud_backend(config: dict) -> dict | None:
    """Return Cloud-Backend als dict oder None wenn nicht konfiguriert.

    Keys: provider, api_key, model, base_url.
    """
    backend = _get_backend("cloud", required=False)
    if backend:
        return {
            "provider": backend["provider"],
            "api_key": backend["api_key"],
            "model": backend["model"],
            "base_url": backend.get("base_url", ""),
        }
    return None


def get_ddu_backends(config: dict) -> list[dict]:
    """Alle aktiven DDU-Backends als Liste, sortiert nach Prioritaet.

    Jedes Dict enthaelt die gleichen Keys wie get_ddu_backend() plus:
    - is_local: immer False; KIara nutzt DDU nur noch als Remote-Service.

    Hinweis: ``config`` wird aktuell nicht verwendet (kein YAML-Fallback),
    aber als Parameter beibehalten fuer API-Symmetrie.
    """
    try:
        from kiara.backend_selector import get_all_active

        backends = get_all_active("ddu")
    except Exception as exc:
        logger.warning("DDU-Backend-Abfrage fehlgeschlagen: %s", exc)
        return []

    result = []
    for backend in backends:
        url = (backend.get("url") or "").strip()
        if not url:
            logger.info(
                "DDU-Backend %s ohne URL ignoriert: lokale Docling-Ausfuehrung ist deaktiviert",
                backend.get("name", "?"),
            )
            continue
        try:
            extra = json.loads(backend.get("extra_config") or "{}")
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Ungueltige extra_config fuer DDU-Backend %s, verwende Defaults",
                backend.get("name", "?"),
            )
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        result.append(
            {
                "url": url,
                "is_local": False,
                "model": backend.get("model", "docling"),
                "table_mode": extra.get("table_mode", "accurate"),
                "do_ocr": extra.get("do_ocr", True),
                "ocr_lang": extra.get("ocr_lang", ["de", "en"]),
                "timeout": extra.get("timeout", DEFAULT_TIMEOUTS["ddu_convert"]),
                "concurrency_limit": extra.get("concurrency_limit"),
            }
        )
    return result


def get_ddu_backend(config: dict) -> dict | None:
    """Return hoechstpriorisiertes DDU-Backend oder None.

    Wrapper um get_ddu_backends() — gibt nur das erste Backend zurueck.
    Fuer Failover ueber mehrere Backends stattdessen get_ddu_backends() nutzen.
    """
    backends = get_ddu_backends(config)
    return backends[0] if backends else None


def get_ddu_internal_cidrs(config: dict) -> list[str]:
    """Liefert ``ddu.internal_cidrs`` aus der Config oder ``[]``.

    Fail-closed-Vertrag:
    - Fehlender ``ddu``-Abschnitt -> leerer Default ``[]``.
    - Vorhandener ``ddu``-Abschnitt mit falschem Typ -> ``PinningConfigError``.

    Der ``PinningConfigError``-Import bleibt lokal, damit
    ``kiara.shared.http_pinning`` nicht von ``kiara.config`` abhaengt
    (Zyklusvermeidung).
    """
    from kiara.shared.http_pinning import PinningConfigError

    ddu_config = config.get("ddu")
    if ddu_config is None:
        return []
    if not isinstance(ddu_config, dict):
        raise PinningConfigError("ungueltige ddu-Konfiguration")
    return ddu_config.get("internal_cidrs", [])


def _get_config_value(config: dict, keys: tuple[str, ...]) -> tuple[bool, Any]:
    """Return ``(exists, value)`` fuer verschachtelte Config-Pfade."""
    obj: Any = config
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return False, None
        obj = obj[key]
    return True, obj


def _require_int_range(
    config: dict,
    keys: tuple[str, ...],
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> None:
    exists, value = _get_config_value(config, keys)
    if not exists:
        return
    path = ".".join(keys)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config field must be integer: {path}")
    if min_value is not None and value < min_value:
        raise ValueError(f"Config field below minimum {min_value}: {path}")
    if max_value is not None and value > max_value:
        raise ValueError(f"Config field above maximum {max_value}: {path}")


def _require_number_range(
    config: dict,
    keys: tuple[str, ...],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_exclusive: bool = False,
) -> None:
    exists, value = _get_config_value(config, keys)
    if not exists:
        return
    path = ".".join(keys)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config field must be numeric: {path}")
    if min_value is not None:
        if min_exclusive and value <= min_value:
            raise ValueError(f"Config field must be greater than {min_value}: {path}")
        if not min_exclusive and value < min_value:
            raise ValueError(f"Config field below minimum {min_value}: {path}")
    if max_value is not None and value > max_value:
        raise ValueError(f"Config field above maximum {max_value}: {path}")


def _require_choice(config: dict, keys: tuple[str, ...], allowed: set[str]) -> None:
    exists, value = _get_config_value(config, keys)
    if not exists:
        return
    path = ".".join(keys)
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid config value for {path}: {value!r} (allowed: {allowed_values})")


def _validate(config: dict) -> None:
    """Validate required and semantic config fields.

    KIara ist in initialer Entwicklung: ungueltige Config-Werte werden
    fail-closed abgelehnt statt still normalisiert.
    """
    required = [
        ("datasource", "path"),
        ("ollama", "url"),
        ("ollama", "llm_model"),
        ("ollama", "embedding_model"),
        ("vectordb", "path"),
    ]
    for keys in required:
        obj = config
        for key in keys:
            if not isinstance(obj, dict) or key not in obj:
                raise ValueError(f"Missing required config field: {'.'.join(keys)}")
            obj = obj[key]
        if obj is None or (isinstance(obj, str) and not obj.strip()):
            raise ValueError(f"Required config field is empty: {'.'.join(keys)}")

    for keys in [
        ("web", "port"),
        ("web", "admin_port"),
        ("web", "mcp_port"),
    ]:
        _require_int_range(config, keys, min_value=1, max_value=65535)

    for keys in [
        ("web", "workers"),
        ("ingestion", "chunk_size"),
        ("ingestion", "batch_size"),
        ("ollama", "num_ctx"),
        ("cloud", "max_tokens"),
        ("cloud", "routing", "min_relevant_chunks"),
        ("chat", "agent", "max_iterations"),
        ("chat", "agent", "num_ctx"),
        ("chat", "rewriter", "num_ctx"),
        ("retrieval", "rerank_top_k"),
        ("retrieval", "rerank_fetch_multiplier"),
        ("retrieval", "rerank_batch_size"),
        ("retrieval", "rrf_k"),
        ("retrieval", "max_context_chunks"),
        ("workflows", "max_context_tokens"),
        ("workflows", "summary_chunk_size"),
        ("workflows", "export_max_age_days"),
        ("backup", "max_backups"),
        ("cache", "max_size"),
        ("cache", "ttl_seconds"),
        ("graph", "max_chunks"),
        ("enrichment", "max_context_chars"),
        ("enrichment", "max_workers"),
        ("tracing", "retention_days"),
        ("system_events", "retention_days"),
        ("anonymization", "min_term_length"),
        ("anonymization", "llm_detection", "timeout_seconds"),
        ("anonymization", "llm_detection", "cache_size"),
        ("anonymization", "llm_detection", "num_ctx"),
    ]:
        _require_int_range(config, keys, min_value=1)

    _require_int_range(config, ("ingestion", "chunk_overlap"), min_value=0)
    _require_int_range(config, ("database", "postgresql", "statement_timeout_ms"), min_value=0)
    _require_int_range(config, ("security", "hsts_max_age"), min_value=0)
    _require_int_range(config, ("graph", "max_traversal_depth"), min_value=0)

    for keys in [
        ("timeouts", "http_health"),
        ("timeouts", "http_api"),
        ("timeouts", "ldap_connect"),
        ("timeouts", "ldap_receive"),
        ("timeouts", "smtp"),
        ("timeouts", "subprocess_quick"),
        ("timeouts", "subprocess_service_logs"),
        ("timeouts", "subprocess_restart"),
        ("timeouts", "subprocess_mount"),
        ("timeouts", "subprocess_umount"),
        ("timeouts", "graph_extraction"),
        ("timeouts", "erp_query"),
        ("timeouts", "erp_bulk"),
        ("timeouts", "db_connect"),
        ("timeouts", "ddu_convert"),
        ("timeouts", "subprocess_network_ping"),
        ("timeouts", "subprocess_network_scan"),
        ("timeouts", "subprocess_network_smb"),
        ("timeouts", "subprocess_network_dns"),
        ("timeouts", "subprocess_network_http"),
        ("timeouts", "subprocess_network_traceroute"),
        ("health", "check_interval_seconds"),
        ("health", "log_interval_seconds"),
        ("health", "disk_warning_gb"),
        ("health", "disk_critical_gb"),
        ("monitoring", "persist_interval_seconds"),
        ("monitoring", "downsample_hour"),
        ("monitoring", "raw_retention_hours"),
        ("monitoring", "retention_5min_days"),
        ("monitoring", "retention_1h_days"),
    ]:
        _require_number_range(config, keys, min_value=0, min_exclusive=True)

    for keys in [
        ("ollama", "temperature"),
        ("cloud", "temperature"),
        ("chat", "unchained", "temperature"),
        ("chat", "agent", "temperature"),
        ("chat", "rewriter", "temperature"),
        ("anonymization", "llm_detection", "temperature"),
    ]:
        _require_number_range(config, keys, min_value=0, max_value=2)

    for keys in [
        ("cloud", "routing", "confidence_threshold"),
        ("anonymization", "llm_detection", "confidence_threshold"),
    ]:
        _require_number_range(config, keys, min_value=0, max_value=1)

    _require_number_range(config, ("cloud", "budget", "monthly_limit_eur"), min_value=0)
    _require_number_range(
        config, ("cloud", "budget", "alert_threshold_pct"), min_value=0, max_value=100
    )

    exists, chunk_size = _get_config_value(config, ("ingestion", "chunk_size"))
    overlap_exists, chunk_overlap = _get_config_value(config, ("ingestion", "chunk_overlap"))
    if exists and overlap_exists and chunk_overlap >= chunk_size:
        raise ValueError("ingestion.chunk_overlap must be smaller than ingestion.chunk_size")

    _require_choice(config, ("database", "backend"), {"postgresql"})
    _require_choice(config, ("vectordb", "backend"), {"qdrant"})
    _require_choice(config, ("cloud", "mode"), {"auto", "local", "cloud"})
    _require_choice(config, ("cloud", "routing", "mode"), {"auto", "local", "cloud"})
    _require_choice(config, ("retrieval", "hybrid_search"), {"none", "bm25", "elasticsearch"})
    _require_choice(config, ("retrieval", "elasticsearch_fuzziness"), {"none", "AUTO", "1", "2"})
    _require_choice(config, ("tracing", "exporter"), {"postgresql", "otlp", "both"})
    _require_choice(config, ("system_events", "min_level"), set(_SYSTEM_EVENTS_LEVEL_ALLOWLIST))
    _require_choice(config, ("anonymization", "placeholder_format"), {"bracket", "angle", "typed"})
