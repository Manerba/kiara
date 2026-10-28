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
from urllib.parse import urlparse

import yaml

from kiara.generation.research_config import DEFAULT_RESEARCH, research_settings
from kiara.retrieval.source_policy import DEFAULT_SOURCE_PROFILES, validate_source_profiles
from kiara.shared.constants import (
    ABSTAIN_INSTRUCTION_MAX_SCORE_DEFAULT,
    ABSTAIN_MIN_RERANK_SCORE_DEFAULT,
    ACCEPT_ALL_BLOCKLIST_DEFAULT,
    ADMIN_PORT_DEFAULT,
    ADMIN_TOOL_TEST_MAX_SESSIONS,
    ADMIN_TOOL_TEST_SESSION_TTL_SECS,
    AGENT_CONTEXT_BUDGET_CHARS,
    AGENT_CONTEXT_READ_RESERVE_CHARS,
    AGENT_CONTEXT_RESULT_CHARS,
    AGENT_DOCUMENT_READ_MAX_CHUNKS,
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
    GROUNDING_MIN_ENTAILMENT_DEFAULT,
    GROUNDING_MODEL_DEFAULT,
    GROUNDING_TIMEOUT_SECS_DEFAULT,
    HEALTH_CHECK_INTERVAL_SECS_DEFAULT,
    HEALTH_LOG_INTERVAL_SECS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_ATTEMPTS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_BACKOFF_INITIAL_SECS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_BACKOFF_MAX_SECS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_TIMEOUT_SECS_DEFAULT,
    INDEX_CONTRACT_RECOVERY_POLL_SECS_DEFAULT,
    INGESTION_D_STATE_TIMEOUT_SECS_DEFAULT,
    INGESTION_D_STATE_WARNING_SECS_DEFAULT,
    INGESTION_DEFAULT_BATCH_SIZE,
    INGESTION_STALL_POLL_INTERVAL_SECS_DEFAULT,
    MAIL_ARCHIVE_MAX_BODY_BYTES_DEFAULT,
    MCP_PORT_DEFAULT,
    RERANK_BACKEND_DEFAULT,
    RERANK_BATCH_SIZE_DEFAULT,
    RERANK_FAIL_MODE_DEFAULT,
    RERANK_FETCH_MULTIPLIER_DEFAULT,
    RERANK_TIMEOUT_SECS_DEFAULT,
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
    "mail_originals": "/var/lib/kiara/mail-originals",
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


def get_mail_archive_config(config: dict) -> dict:
    """Nur den aktuellen ausgewählten Body je Mail außerhalb des Indexes halten."""
    block = config.get("ingestion", {}).get("mail_archive", {})
    if not isinstance(block, dict):
        raise ValueError("ingestion.mail_archive muss ein Mapping sein")
    path = block.get("path", DEFAULT_PATHS["mail_originals"])
    limit = block.get("max_body_bytes", MAIL_ARCHIVE_MAX_BODY_BYTES_DEFAULT)
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("ingestion.mail_archive.path muss ein absoluter Pfad sein")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("ingestion.mail_archive.max_body_bytes muss positiv sein")
    return {"path": path, "max_body_bytes": limit}


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
    "qdrant_index": 300.0,
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


def get_ingestion_stall_detection_config(config: dict) -> dict:
    """Liefert die normalisierte Konfiguration der D-State-Erkennung.

    Regulaer geladene Konfiguration wird von :func:`_validate` fail-closed
    geprueft. Die defensiven Defaults halten direkte Test-/Library-Aufrufe
    ohne vollstaendigen YAML-Block funktionsfaehig.
    """
    ingestion = config.get("ingestion") if isinstance(config, dict) else None
    if not isinstance(ingestion, dict):
        ingestion = {}
    block = ingestion.get("stall_detection")
    if not isinstance(block, dict):
        block = {}
    return {
        "enabled": block.get("enabled", True),
        "poll_interval_seconds": block.get(
            "poll_interval_seconds",
            INGESTION_STALL_POLL_INTERVAL_SECS_DEFAULT,
        ),
        "warning_after_seconds": block.get(
            "warning_after_seconds",
            INGESTION_D_STATE_WARNING_SECS_DEFAULT,
        ),
        "timeout_seconds": block.get(
            "timeout_seconds",
            INGESTION_D_STATE_TIMEOUT_SECS_DEFAULT,
        ),
    }


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
        "mail_archive": {
            "path": DEFAULT_PATHS["mail_originals"],
            "max_body_bytes": MAIL_ARCHIVE_MAX_BODY_BYTES_DEFAULT,
        },
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "batch_size": INGESTION_DEFAULT_BATCH_SIZE,
        "admin_cache_refresh_seconds": 3,
        "ds_ids_sync_attempts": 3,
        "vlm_prompt": "",
        "accept_all_blocklist": sorted(ACCEPT_ALL_BLOCKLIST_DEFAULT),
        "stall_detection": {
            "enabled": True,
            "poll_interval_seconds": INGESTION_STALL_POLL_INTERVAL_SECS_DEFAULT,
            "warning_after_seconds": INGESTION_D_STATE_WARNING_SECS_DEFAULT,
            "timeout_seconds": INGESTION_D_STATE_TIMEOUT_SECS_DEFAULT,
        },
    },
    "index_contract": {
        "discovery_timeout_seconds": INDEX_CONTRACT_DISCOVERY_TIMEOUT_SECS_DEFAULT,
        "discovery_attempts": INDEX_CONTRACT_DISCOVERY_ATTEMPTS_DEFAULT,
        "discovery_backoff_initial_seconds": (
            INDEX_CONTRACT_DISCOVERY_BACKOFF_INITIAL_SECS_DEFAULT
        ),
        "discovery_backoff_max_seconds": INDEX_CONTRACT_DISCOVERY_BACKOFF_MAX_SECS_DEFAULT,
        "recovery_poll_seconds": INDEX_CONTRACT_RECOVERY_POLL_SECS_DEFAULT,
    },
    "web": {
        "host": "0.0.0.0",
        "port": WEB_PORT_DEFAULT,
        "admin_port": ADMIN_PORT_DEFAULT,
        "mcp_port": MCP_PORT_DEFAULT,
        "mcp_enabled": False,
        "mcp_retrieval_enabled": False,
        "mcp_retrieval_allow_raw_chunks": False,
        "mcp_retrieval_include_text_default": False,
        "mcp_retrieval_max_k": 50,
        "mcp_retrieval_max_query_length": 10000,
        "mcp_retrieval_max_chars_per_result": 4000,
        "mcp_retrieval_max_context_window": 5,
        "mcp_retrieval_max_context_chunks": 10,
        "mcp_retrieval_max_chars_total": 20000,
        "oidc_retrieval": {
            "enabled": False,
            "issuer": "",
            "discovery_url": "",
            "jwks_url": "",
            "audience": "kiara-retrieval",
            "client_binding_claim": "azp",
            "client_binding_value": "open-webui",
            "algorithms": ["RS256"],
            "identity_claim": "ad_sid",
            "identity_claim_format": "ad_object_sid_or_base64",
            "required_scopes": [],
            "required_roles": [],
            "clock_skew_seconds": 60,
            "ca_bundle_path": "",
            "allow_admin_full_access": False,
        },
        "workers": 4,
        "trusted_origins": [],
    },
    "database": {
        "backend": "postgresql",
        "postgresql": {
            "dsn": "postgresql://kiara:kiara@localhost:5432/kiara",
            "statement_timeout_ms": 30000,
            "pool_max_size": 16,
            "pool_wait_seconds": 5,
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
        "test_session_ttl_secs": ADMIN_TOOL_TEST_SESSION_TTL_SECS,
        "test_max_sessions": ADMIN_TOOL_TEST_MAX_SESSIONS,
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
            "ERP-Leser": ["erp_order_status", "erp_orders", "erp_stock_level"],
            "ERP-Vertrieb": [
                "erp_order_status",
                "erp_orders",
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
        "research": DEFAULT_RESEARCH,
        "source_profiles": DEFAULT_SOURCE_PROFILES,
        "unchained": {
            "temperature": DEFAULT_UNCHAINED_TEMPERATURE,
        },
        "agent": {
            "max_iterations": DEFAULT_AGENT_MAX_ITERATIONS,
            "num_ctx": DEFAULT_AGENT_NUM_CTX,
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "raw_chunk_compaction": True,
            "final_reranking": True,
            "context_budget_chars": AGENT_CONTEXT_BUDGET_CHARS,
            "context_result_chars": AGENT_CONTEXT_RESULT_CHARS,
            "context_read_reserve_chars": AGENT_CONTEXT_READ_RESERVE_CHARS,
            "document_read_max_chunks": AGENT_DOCUMENT_READ_MAX_CHUNKS,
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
        "elasticsearch_index": "kiara",  # Praefix fuer <prefix>_ds_<id>
        "elasticsearch_maintenance_file": "/var/lib/kiara/elasticsearch-maintenance.json",
        "elasticsearch_fuzziness": "none",
        "reranking": False,
        "rerank_backend": RERANK_BACKEND_DEFAULT,
        "rerank_url": "",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_top_k": RERANK_TOP_K_DEFAULT,
        "rerank_fetch_multiplier": RERANK_FETCH_MULTIPLIER_DEFAULT,
        "rerank_batch_size": RERANK_BATCH_SIZE_DEFAULT,
        "rerank_timeout_s": RERANK_TIMEOUT_SECS_DEFAULT,
        "rerank_fail_mode": RERANK_FAIL_MODE_DEFAULT,
        "abstain_enabled": False,
        "abstain_min_rerank_score": ABSTAIN_MIN_RERANK_SCORE_DEFAULT,
        "abstain_instruction": False,
        "abstain_instruction_max_score": ABSTAIN_INSTRUCTION_MAX_SCORE_DEFAULT,
        "abstain_message": "",
        "grounding_retry": False,
        "grounding_url": "",
        "grounding_model": GROUNDING_MODEL_DEFAULT,
        "grounding_min_entailment": GROUNDING_MIN_ENTAILMENT_DEFAULT,
        "grounding_timeout_s": GROUNDING_TIMEOUT_SECS_DEFAULT,
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
        "qdrant_data_path": "",  # Leer = lokale Auto-Erkennung; sonst gemountetes Qdrant-storage
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

    # Auto-Snapshot des ALTEN Stands vor jedem Schreiben (best-effort, dedupliziert).
    # Lazy-Import gegen Zyklus (der Service importiert kiara.config).
    if path == DEFAULT_CONFIG_PATH:
        try:
            from kiara.admin.services.config_snapshot_service import auto_snapshot

            auto_snapshot()
        except Exception:
            logger.warning("Auto-Config-Snapshot uebersprungen", exc_info=True)
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


def _require_bool(config: dict, keys: tuple[str, ...]) -> None:
    exists, value = _get_config_value(config, keys)
    if not exists:
        return
    if not isinstance(value, bool):
        raise ValueError(f"Config field must be boolean: {'.'.join(keys)}")


def _require_string_list(block: dict, field: str, *, path_prefix: str) -> list[str]:
    value = block.get(field, [])
    path = f"{path_prefix}.{field}"
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Config field must be list of strings: {path}")
    return value


def _validate_https_url(value: str, path: str, *, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field must be non-empty HTTPS URL: {path}")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Config field must be HTTPS URL: {path}")


def _validate_oidc_retrieval_config(config: dict) -> None:
    web = config.get("web")
    if not isinstance(web, dict):
        raise ValueError("Config section must be mapping: web")
    oidc = web.get("oidc_retrieval")
    if not isinstance(oidc, dict):
        raise ValueError("Config section must be mapping: web.oidc_retrieval")

    for field in ["enabled", "allow_admin_full_access"]:
        if not isinstance(oidc.get(field), bool):
            raise ValueError(f"Config field must be boolean: web.oidc_retrieval.{field}")

    required_string_fields = [
        "issuer",
        "discovery_url",
        "jwks_url",
        "audience",
        "client_binding_claim",
        "client_binding_value",
        "identity_claim",
        "identity_claim_format",
        "ca_bundle_path",
    ]
    for field in required_string_fields:
        if field in oidc and not isinstance(oidc[field], str):
            raise ValueError(f"Config field must be string: web.oidc_retrieval.{field}")

    _require_string_list(oidc, "required_scopes", path_prefix="web.oidc_retrieval")
    _require_string_list(oidc, "required_roles", path_prefix="web.oidc_retrieval")

    algorithms = oidc.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("Config field must be non-empty list: web.oidc_retrieval.algorithms")
    if any(not isinstance(item, str) or not item.strip() for item in algorithms):
        raise ValueError(
            "Config field must be list of non-empty strings: web.oidc_retrieval.algorithms"
        )
    if algorithms != ["RS256"]:
        raise ValueError("web.oidc_retrieval.algorithms must be exactly ['RS256'] for openDesk")

    clock_skew = oidc.get("clock_skew_seconds")
    if isinstance(clock_skew, bool) or not isinstance(clock_skew, int):
        raise ValueError("Config field must be integer: web.oidc_retrieval.clock_skew_seconds")
    if clock_skew < 0 or clock_skew > 300:
        raise ValueError("web.oidc_retrieval.clock_skew_seconds must be between 0 and 300")

    ca_bundle_path = oidc.get("ca_bundle_path", "")
    if ca_bundle_path and (
        not os.path.isfile(ca_bundle_path) or not os.access(ca_bundle_path, os.R_OK)
    ):
        raise ValueError("web.oidc_retrieval.ca_bundle_path must be readable")

    if oidc.get("identity_claim") not in ("", "ad_sid"):
        raise ValueError("web.oidc_retrieval.identity_claim must be ad_sid for openDesk")
    allowed_identity_formats = {"ad_object_sid", "ad_object_sid_base64", "ad_object_sid_or_base64"}
    if oidc.get("identity_claim_format") not in allowed_identity_formats:
        raise ValueError("Invalid web.oidc_retrieval.identity_claim_format")

    for url_field in ["issuer", "discovery_url", "jwks_url"]:
        value = oidc.get(url_field, "")
        _validate_https_url(value, f"web.oidc_retrieval.{url_field}", allow_empty=True)
        if url_field == "issuer" and value:
            parsed = urlparse(value)
            if parsed.query or parsed.fragment or value.endswith("/"):
                raise ValueError(
                    "web.oidc_retrieval.issuer must be exact and contain no query, fragment or trailing slash"
                )

    if oidc.get("enabled") is not True:
        return

    for field in [
        "issuer",
        "audience",
        "identity_claim",
        "client_binding_claim",
        "client_binding_value",
    ]:
        value = oidc.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Required OIDC config field is empty: web.oidc_retrieval.{field}")

    if oidc["identity_claim"] != "ad_sid":
        raise ValueError("web.oidc_retrieval.identity_claim must be ad_sid for openDesk")
    if oidc["client_binding_claim"] != "azp":
        raise ValueError("web.oidc_retrieval.client_binding_claim must be azp for openDesk")
    if oidc["client_binding_value"] != "open-webui":
        raise ValueError("web.oidc_retrieval.client_binding_value must be open-webui for openDesk")

    if not (oidc.get("discovery_url") or oidc.get("jwks_url")):
        raise ValueError("web.oidc_retrieval requires discovery_url or jwks_url when enabled")


def _validate(config: dict) -> None:
    """Validate required and semantic config fields.

    KIara ist in initialer Entwicklung: ungueltige Config-Werte werden
    fail-closed abgelehnt statt still normalisiert.
    """
    validate_source_profiles(config)
    research_settings(config)
    get_mail_archive_config(config)
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
        ("web", "mcp_retrieval_max_k"),
        ("web", "mcp_retrieval_max_query_length"),
        ("web", "mcp_retrieval_max_chars_per_result"),
        ("web", "mcp_retrieval_max_context_window"),
        ("web", "mcp_retrieval_max_context_chunks"),
        ("web", "mcp_retrieval_max_chars_total"),
        ("ingestion", "chunk_size"),
        ("ingestion", "batch_size"),
        ("index_contract", "discovery_attempts"),
        ("ollama", "num_ctx"),
        ("cloud", "max_tokens"),
        ("cloud", "routing", "min_relevant_chunks"),
        ("chat", "agent", "max_iterations"),
        ("chat", "agent", "num_ctx"),
        ("chat", "agent", "context_budget_chars"),
        ("chat", "agent", "context_result_chars"),
        ("chat", "agent", "document_read_max_chunks"),
        ("tools", "test_session_ttl_secs"),
        ("tools", "test_max_sessions"),
        ("chat", "rewriter", "num_ctx"),
        ("retrieval", "rerank_top_k"),
        ("retrieval", "rerank_fetch_multiplier"),
        ("retrieval", "rerank_batch_size"),
        ("retrieval", "rerank_timeout_s"),
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
    _require_int_range(config, ("chat", "agent", "context_read_reserve_chars"), min_value=0)
    agent = config.get("chat", {}).get("agent", {})
    if agent.get("context_read_reserve_chars", AGENT_CONTEXT_READ_RESERVE_CHARS) >= agent.get(
        "context_budget_chars", AGENT_CONTEXT_BUDGET_CHARS
    ):
        raise ValueError("chat.agent.context_read_reserve_chars must be below context_budget_chars")
    _require_bool(config, ("ingestion", "stall_detection", "enabled"))
    for keys in [
        ("ingestion", "stall_detection", "poll_interval_seconds"),
        ("ingestion", "stall_detection", "warning_after_seconds"),
        ("ingestion", "stall_detection", "timeout_seconds"),
        ("index_contract", "discovery_timeout_seconds"),
        ("index_contract", "discovery_backoff_initial_seconds"),
        ("index_contract", "discovery_backoff_max_seconds"),
        ("index_contract", "recovery_poll_seconds"),
    ]:
        _require_number_range(config, keys, min_value=0, min_exclusive=True)

    warning_exists, warning_after = _get_config_value(
        config,
        ("ingestion", "stall_detection", "warning_after_seconds"),
    )
    timeout_exists, timeout_seconds = _get_config_value(
        config,
        ("ingestion", "stall_detection", "timeout_seconds"),
    )
    if warning_exists and timeout_exists and warning_after > timeout_seconds:
        raise ValueError(
            "ingestion.stall_detection.warning_after_seconds must not exceed "
            "ingestion.stall_detection.timeout_seconds"
        )
    _require_int_range(config, ("database", "postgresql", "statement_timeout_ms"), min_value=0)
    _require_int_range(config, ("database", "postgresql", "pool_max_size"), min_value=1)
    _require_int_range(config, ("database", "postgresql", "pool_wait_seconds"), min_value=1)
    _require_int_range(config, ("ingestion", "admin_cache_refresh_seconds"), min_value=1)
    _require_int_range(config, ("ingestion", "ds_ids_sync_attempts"), min_value=1)
    _require_int_range(config, ("security", "hsts_max_age"), min_value=0)
    _require_int_range(config, ("graph", "max_traversal_depth"), min_value=0)

    _require_bool(config, ("web", "mcp_enabled"))
    _require_bool(config, ("web", "mcp_retrieval_enabled"))

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
    if config.get("retrieval", {}).get("hybrid_search") == "elasticsearch":
        from kiara.retrieval.elasticsearch_indices import datasource_index_name

        datasource_index_name(config, 0)
    maintenance_file = config.get("retrieval", {}).get("elasticsearch_maintenance_file")
    if maintenance_file is not None and (
        not isinstance(maintenance_file, str) or not os.path.isabs(maintenance_file)
    ):
        raise ValueError("retrieval.elasticsearch_maintenance_file must be an absolute path")

    _require_choice(config, ("retrieval", "rerank_backend"), {"local", "kiron"})
    _require_choice(config, ("retrieval", "rerank_fail_mode"), {"degrade", "error"})
    _require_choice(config, ("tracing", "exporter"), {"postgresql", "otlp", "both"})
    _require_choice(config, ("system_events", "min_level"), set(_SYSTEM_EVENTS_LEVEL_ALLOWLIST))
    _require_choice(config, ("anonymization", "placeholder_format"), {"bracket", "angle", "typed"})
    _validate_oidc_retrieval_config(config)
