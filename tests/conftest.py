"""Shared Fixtures fuer KIara-Tests.

Alle Tests laufen gegen PostgreSQL (separate Test-DB kiara_test).
DSN wird auf Modul-Level gesetzt damit Imports waehrend Test-Collection
bereits funktionieren.
"""

# ---------------------------------------------------------------------------
# DB-Backend SOFORT initialisieren (vor allen Imports die DB brauchen)
# ---------------------------------------------------------------------------
import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"

# CSRF-Middleware-Bypass fuer die bestehende Test-Suite. test_middleware_csrf.py
# importiert die Original-Middleware gezielt. Die hier gesetzte No-Op-Variante
# ersetzt ``add_csrf_middleware`` VOR dem Import der Web-Apps, damit in der
# Test-Suite keine CSRF-Pruefung registriert wird. Dieser Mechanismus lebt
# ausschliesslich im Test-Code — Produktions-Middleware hat keinen Runtime-
# Bypass mehr (ImplScan-Finding 1, Spec §1a).
import kiara.web.middleware as _kiara_middleware

# Original-Funktion als Modul-Attribut ablegen, damit test_middleware_csrf.py
# die echte Middleware weiter aufrufen kann.
_kiara_middleware._ORIGINAL_ADD_CSRF_MIDDLEWARE = _kiara_middleware.add_csrf_middleware


def _noop_add_csrf_middleware(app, *args, **kwargs):
    """Test-Pass-Through: registriert weder CSRF-Middleware noch Handler.

    Produktions-Apps rufen add_csrf_middleware einmal zum Startup auf. In der
    Test-Suite (ausser test_middleware_csrf.py) wird der Call zu einem No-Op,
    damit bestehende Tests ohne CSRF-Token/Origin-Header laufen.
    """
    return None


_kiara_middleware.add_csrf_middleware = _noop_add_csrf_middleware

import pytest

from kiara.anonymization.anonymizer import Anonymizer
from kiara.anonymization.automaton import AnonymizationAutomaton


@pytest.fixture
def s4_guard_passthrough(monkeypatch):
    """Isolate legacy component tests from the S4 persistence boundary.

    Tests opting in still exercise their Qdrant/cache/HTTP behavior. The real
    fail-closed state machine, locks and constructor ordering have dedicated
    S4 tests and are deliberately not replaced globally.
    """
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import kiara.config as config_module
    import kiara.shared.db as shared_db
    from kiara.retrieval import index_compatibility as module

    monkeypatch.setattr(
        config_module,
        "get_embedding_model",
        lambda _config: "nomic-embed-text",
    )

    @contextmanager
    def passthrough(
        _config,
        target,
        *,
        operation,
        write=False,
        require_configured_default=False,
        validation_scope=module.IndexValidationScope.LIVE_PROFILE,
        **_kwargs,
    ):
        active = next(
            (
                token
                for token in reversed(module._active_tokens.get())
                if token.target.logical_index_key == target.logical_index_key
            ),
            None,
        )
        if active is not None:
            yield active
            return

        snapshot = {
            "contract_logical_index_key": target.logical_index_key,
            "contract_index_compatibility_id": "sha256:" + "1" * 64,
            "contract_query_compatibility_id": "sha256:" + "2" * 64,
            "contract_canonical_model_id": target.requested_model,
            "contract_profile_id": "s4-test-profile",
            "contract_profile_kind": target.vector_kind.value,
            "contract_profile_endpoint": {
                "dense": "/api/embed",
                "late_chunking": "/api/embed_late",
                "multi_vector": "/api/embed_colbert",
            }[target.vector_kind.value],
            "contract_snapshot_sha256": "3" * 64,
            "contract_fence_version": 1,
        }
        resolved = SimpleNamespace(
            state=module.ContractState.VALID,
            backend=None,
            backend_url=None,
            index_compatibility_id=snapshot["contract_index_compatibility_id"],
            query_compatibility_id=snapshot["contract_query_compatibility_id"],
            canonical_model_id=snapshot["contract_canonical_model_id"],
            profile_id=snapshot["contract_profile_id"],
            profile_kind=snapshot["contract_profile_kind"],
            profile_endpoint=snapshot["contract_profile_endpoint"],
            profile_supported_roles=("search_document", "search_query"),
            profile_role_sensitive=True,
            profile_input_type_required=False,
            profile_formatting_owner="server",
            contract_snapshot_sha256=snapshot["contract_snapshot_sha256"],
        )
        # Eine echte Test-Transaktion erhaelt die atomaren Semantiken der
        # aufrufenden Komponente (z. B. enqueue_job), ohne den S4-Zustands-
        # automaten in fachfremden Legacy-Unit-Tests erneut aufzubauen.
        with shared_db.db_transaction() as connection:
            token = SimpleNamespace(
                target=target,
                operation=str(getattr(operation, "value", operation)),
                write=write,
                validation_scope=module.IndexValidationScope(validation_scope),
                require_configured_default=require_configured_default,
                resolved=resolved,
                stored={},
                connection=connection,
                fence_version=1,
                job_fields=lambda: dict(snapshot),
            )
            previous = module._active_tokens.get()
            reset_token = module._active_tokens.set((*previous, token))
            try:
                yield token
            finally:
                module._active_tokens.reset(reset_token)

    monkeypatch.setattr(module, "guard_index", passthrough)
    monkeypatch.setattr(module, "raw_qdrant_client", lambda _config: MagicMock())
    monkeypatch.setattr(module, "retire_index_contract", lambda _token: None)

    # Manche alte ColBERT-Helpertests erzeugen bewusst Instanzen via __new__.
    # Die Klassenattribute liefern nur dort den fuer den Decorator benoetigten
    # Kontext; reale Instanzen ueberschreiben sie stets in __init__.
    from kiara.retrieval.colbert import ColBERTStore

    colbert_target = module.colbert_index_target({}, "s4-test-colbert")
    monkeypatch.setattr(ColBERTStore, "_config", {}, raising=False)
    monkeypatch.setattr(ColBERTStore, "_index_target", colbert_target, raising=False)
    monkeypatch.setattr(ColBERTStore, "_intent", "write", raising=False)


@pytest.fixture
def s4_worker_guard_passthrough(s4_guard_passthrough, monkeypatch):
    """Unwrap only the old full-reingest behavior tests.

    The atomic job-row fence is covered with real PostgreSQL rows in the S4
    suite; these legacy tests use the deliberately invalid id ``job-1`` and
    exclusively verify cleanup ordering after that boundary.
    """
    from kiara.web.ingestion_worker import IngestionWorker

    wrapped = IngestionWorker._prepare_full_reingest
    monkeypatch.setattr(IngestionWorker, "_prepare_full_reingest", wrapped.__wrapped__)


# ---------------------------------------------------------------------------
# Temporaere Config mit Anonymisierung aktiviert
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path):
    """Minimale Config mit aktivierter Anonymisierung (ohne externe Dienste)."""
    csv_dir = tmp_path / "stammdaten"
    csv_dir.mkdir()
    return {
        "datasource": {
            "path": str(tmp_path / "kb"),
            "file_types": [".txt"],
            "exclude_patterns": [],
        },
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "test",
            "embedding_model": "test",
            "prompt_template": "",
        },
        "vectordb": {
            "path": str(tmp_path / "vectordb"),
            "collection": "test",
        },
        "anonymization": {
            "enabled": True,
            "word_boundaries": True,
            "min_term_length": 3,
            "erp": {
                "enabled": True,
                "backend": "csv",
                "csv_dir": str(csv_dir),
            },
            "erp_source": {
                "enabled": True,
                "type": "csv",
                "path": str(csv_dir),
            },
            "ldap": {
                "enrich_from_db": False,
                "enrich_from_ldap": False,
            },
            "ldap_source": {"enabled": False},
            "custom_patterns": [],
            "excluded_terms": [],
            "custom_entries": [],
        },
    }


# ---------------------------------------------------------------------------
# Test-Entitaeten
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_entities():
    """Repraesentative Test-Entitaeten aus verschiedenen Kategorien."""
    return [
        {"value": "Mueller GmbH", "category": "customer", "id": "K-1001"},
        {"value": "Schmidt AG", "category": "customer", "id": "K-1002"},
        {"value": "Bauer & Soehne KG", "category": "supplier", "id": "L-2001"},
        {"value": "TechParts International", "category": "supplier", "id": "L-2002"},
        {"value": "Hans Mueller", "category": "employee", "id": "M-3001"},
        {"value": "Anna Schmidt", "category": "employee", "id": "M-3002"},
        {"value": "Petra Bauer", "category": "employee", "id": "M-3003"},
        {"value": "ServerRack Pro X1", "category": "product", "id": "P-4001"},
        {"value": "KabelManager 3000", "category": "product", "id": "P-4002"},
        {"value": "Stuttgart", "category": "location", "id": "O-5001"},
        {"value": "Muenchen", "category": "location", "id": "O-5002"},
        {"value": "Berlin-Mitte", "category": "location", "id": "O-5003"},
    ]


# ---------------------------------------------------------------------------
# Test-DB (PostgreSQL, per Transaktion isoliert)
# ---------------------------------------------------------------------------

_DB_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT,
        vorname TEXT DEFAULT '',
        nachname TEXT DEFAULT '',
        email TEXT DEFAULT '',
        is_admin BOOLEAN DEFAULT false,
        auth_source TEXT DEFAULT 'local',
        ldap_dn TEXT DEFAULT '',
        is_active BOOLEAN DEFAULT true,
        must_change_password BOOLEAN DEFAULT false,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token TEXT NOT NULL UNIQUE,
        csrf_token_hash CHAR(64),
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        expires_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS api_keys (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL DEFAULT '',
        key_hash TEXT NOT NULL,
        key_prefix TEXT NOT NULL DEFAULT '',
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        is_active BOOLEAN DEFAULT true,
        expires_at TEXT,
        last_used_at TEXT,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS chats (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        status TEXT DEFAULT 'active',
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources TEXT,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS admin_chats (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT NOT NULL DEFAULT 'Neuer Chat',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT DEFAULT (NOW()),
        updated_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE IF NOT EXISTS admin_chat_messages (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER NOT NULL REFERENCES admin_chats(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources TEXT,
        tool_calls TEXT,
        metadata TEXT,
        created_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE IF NOT EXISTS admin_chat_confirmations (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER NOT NULL REFERENCES admin_chats(id) ON DELETE CASCADE,
        action_id TEXT NOT NULL UNIQUE,
        function_name TEXT NOT NULL,
        params TEXT NOT NULL DEFAULT '{}',
        confirm_string TEXT NOT NULL,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TIMESTAMPTZ DEFAULT (NOW()),
        expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 minutes')
    );
    CREATE INDEX IF NOT EXISTS idx_admin_confirm_chat
        ON admin_chat_confirmations(chat_id, status);
    CREATE INDEX IF NOT EXISTS idx_admin_confirm_expires
        ON admin_chat_confirmations(status, expires_at);

    CREATE TABLE IF NOT EXISTS admin_chat_wizards (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER NOT NULL REFERENCES admin_chats(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        function_name TEXT NOT NULL,
        collected_params TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'collecting',
        created_at TEXT DEFAULT (NOW()),
        updated_at TEXT DEFAULT (NOW())
    );
    CREATE INDEX IF NOT EXISTS idx_admin_wizard_chat
        ON admin_chat_wizards(chat_id, status);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_wizard_active
        ON admin_chat_wizards(chat_id) WHERE status = 'collecting';

    CREATE TABLE IF NOT EXISTS ldap_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        server_url TEXT NOT NULL,
        port INTEGER DEFAULT 389,
        use_ssl INTEGER DEFAULT 0,
        bind_dn TEXT NOT NULL,
        bind_password TEXT NOT NULL,
        search_base TEXT NOT NULL,
        group_dn TEXT,
        group_name TEXT,
        last_sync_at TEXT,
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS datasource_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        type TEXT NOT NULL,
        webdav_url TEXT DEFAULT '',
        webdav_username TEXT DEFAULT '',
        webdav_password TEXT DEFAULT '',
        webdav_base_path TEXT DEFAULT '/',
        smb_server TEXT DEFAULT '',
        smb_share TEXT DEFAULT '',
        smb_username TEXT DEFAULT '',
        smb_password TEXT DEFAULT '',
        smb_domain TEXT DEFAULT '',
        smb_mount_point TEXT DEFAULT '/mnt/kiara-smb',
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS datasources (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'local',
        path TEXT DEFAULT '',
        is_active BOOLEAN DEFAULT true,
        connector_config TEXT DEFAULT '{}',
        ingestion_status TEXT DEFAULT 'idle',
        restriction_level TEXT DEFAULT 'anonymize' NOT NULL,
        ds_chunk_strategy TEXT DEFAULT 'auto',
        ds_chunk_size INTEGER DEFAULT NULL,
        ds_chunk_overlap INTEGER DEFAULT NULL,
        ds_file_types TEXT DEFAULT NULL,
        ds_exclude_patterns TEXT DEFAULT NULL,
        ds_ddu_enabled BOOLEAN DEFAULT false,
        ds_ddu_ocr_images INTEGER DEFAULT 0,
        ds_vlm_enabled BOOLEAN DEFAULT false,
        ds_accept_all_text INTEGER DEFAULT 0,
        ds_index_attachments INTEGER DEFAULT 1,
        ds_contextual_retrieval INTEGER DEFAULT 0,
        ds_enrichment_prompt_id INTEGER DEFAULT NULL,
        ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false,
        is_public INTEGER DEFAULT NULL,
        sync_interval TEXT DEFAULT 'global',
        progress_total INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS groups (
        id SERIAL PRIMARY KEY,
        sid TEXT UNIQUE,
        name TEXT NOT NULL DEFAULT '',
        cn TEXT NOT NULL,
        dn TEXT NOT NULL,
        member_count INTEGER DEFAULT 0,
        imported_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS user_groups (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        UNIQUE(user_id, group_id)
    );

    CREATE TABLE IF NOT EXISTS ai_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        embedding_url TEXT NOT NULL DEFAULT 'http://localhost:11434',
        embedding_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
        llm_url TEXT NOT NULL DEFAULT 'http://gpu-server.local:11434',
        llm_model TEXT NOT NULL DEFAULT 'qwen3:8b',
        prompt_template TEXT DEFAULT '',
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );

    CREATE TABLE IF NOT EXISTS audit_anonymization (
        id SERIAL PRIMARY KEY,
        timestamp TEXT NOT NULL DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        user_id INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        question_hash TEXT NOT NULL,
        entity_count INTEGER NOT NULL DEFAULT 0,
        entity_types TEXT NOT NULL DEFAULT '{}',
        route TEXT NOT NULL,
        content_class TEXT,
        cloud_provider TEXT,
        cloud_model TEXT,
        response_time_ms INTEGER,
        correlation_id TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS datasource_embeddings (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
        embedding_model TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        last_ingested_at TEXT DEFAULT NULL,
        UNIQUE(datasource_id, embedding_model)
    );

    CREATE TABLE IF NOT EXISTS datasource_groups (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        UNIQUE(datasource_id, group_id)
    );

    CREATE TABLE IF NOT EXISTS datasource_users (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(datasource_id, user_id)
    );

    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
        status TEXT NOT NULL DEFAULT 'pending',
        job_type TEXT NOT NULL DEFAULT 'incremental',
        triggered_by TEXT NOT NULL DEFAULT 'manual',
        embedding_model TEXT DEFAULT '',
        requested_action TEXT DEFAULT NULL,
        progress_processed INTEGER DEFAULT 0,
        progress_total INTEGER DEFAULT 0,
        queued_at TIMESTAMPTZ DEFAULT NOW(),
        started_at TIMESTAMPTZ DEFAULT NULL,
        finished_at TIMESTAMPTZ DEFAULT NULL,
        stats_json TEXT DEFAULT '{}',
        error_message TEXT DEFAULT '',
        resumed_from_job INTEGER DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS ingestion_file_logs (
        id SERIAL PRIMARY KEY,
        job_id INTEGER NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
        file_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_size BIGINT DEFAULT 0,
        file_ext TEXT DEFAULT '',
        parser TEXT DEFAULT '',
        parser_fallback INTEGER DEFAULT 0,
        chunk_strategy TEXT DEFAULT '',
        chunk_count INTEGER DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'success',
        error_message TEXT DEFAULT '',
        error_phase TEXT DEFAULT '',
        duration_ms INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE IF NOT EXISTS query_log (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        question_hash TEXT NOT NULL,
        question_excerpt TEXT,
        answer_excerpt TEXT,
        routed_to TEXT NOT NULL DEFAULT 'local',
        model_used TEXT,
        sources_count INTEGER DEFAULT 0,
        tools_used TEXT,
        chunks_retrieved INTEGER DEFAULT 0,
        entities_anonymized INTEGER DEFAULT 0,
        duration_ms INTEGER DEFAULT 0,
        chat_id INTEGER,
        chain_type TEXT,
        full_prompt TEXT,
        full_answer TEXT,
        tool_calls TEXT,
        context_sources TEXT,
        trace_id TEXT,
        created_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE IF NOT EXISTS file_hashes (
        datasource_id INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        file_size BIGINT,
        PRIMARY KEY (datasource_id, embedding_model, file_path)
    );

    CREATE TABLE IF NOT EXISTS hash_mode_meta (
        datasource_id INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        hash_mode TEXT NOT NULL DEFAULT 'mtime',
        PRIMARY KEY (datasource_id, embedding_model)
    );

    CREATE TABLE IF NOT EXISTS chunk_datasources (
        chunk_id TEXT NOT NULL,
        datasource_id INTEGER NOT NULL,
        embedding_model TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (chunk_id, datasource_id, embedding_model)
    );

    CREATE TABLE IF NOT EXISTS cache_datasources (
        datasource_id INTEGER PRIMARY KEY,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cache_jobs (
        job_id INTEGER PRIMARY KEY,
        datasource_id INTEGER,
        status TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS onlyoffice_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        document_server_url TEXT NOT NULL,
        jwt_secret TEXT NOT NULL,
        wopi_external_base_url TEXT DEFAULT '',
        proof_validation_enabled BOOLEAN DEFAULT TRUE,
        updated_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE IF NOT EXISTS mail_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        smtp_server TEXT NOT NULL,
        smtp_port INTEGER DEFAULT 587,
        encryption TEXT DEFAULT 'starttls',
        username TEXT DEFAULT '',
        password TEXT DEFAULT '',
        sender_address TEXT NOT NULL,
        recipient_addresses TEXT NOT NULL,
        updated_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE IF NOT EXISTS wopi_tokens (
        id SERIAL PRIMARY KEY,
        token TEXT UNIQUE NOT NULL,
        file_id TEXT NOT NULL,
        file_path TEXT,
        user_id INTEGER NOT NULL,
        expires_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS wopi_locks (
        file_id TEXT PRIMARY KEY,
        lock_id TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        expires_at TEXT NOT NULL
    );
"""


def restore_test_schema():
    """Stellt das vollstaendige Test-DB-Schema wieder her.

    Wrapper um db_helpers.restore_test_schema() — importierbar aus
    conftest oder direkt aus db_helpers.
    """
    from db_helpers import restore_test_schema as _restore

    _restore()


@pytest.fixture
def test_db():
    """PostgreSQL-Verbindung mit isoliertem Schema und leeren Tabellen.

    Nach dem Schema-Reset-Sprint: Source-of-Truth ist kiara/web/schema.sql,
    geladen via create_tables(). Das vereinfachte _DB_SCHEMA ist als Fallback
    fuer Altlasten dokumentiert, wird aber nicht mehr aktiv benutzt.
    """
    from db_helpers import reset_test_data

    from kiara.shared.db import get_connection

    conn = get_connection()
    reset_test_data(conn)
    try:
        yield conn
    finally:
        conn.rollback()
        reset_test_data(conn)
        conn.close()


# ---------------------------------------------------------------------------
# PG-Nachweise fuer Tests mit gemockten Suchindizes
# ---------------------------------------------------------------------------


@pytest.fixture
def chunk_memberships(test_db):
    """Registriert die Kandidaten isolierter Rankingtests im echten PG-Vertrag."""

    def register(retriever, *chunk_ids):
        from kiara.ingestion._file_utils import ChunkDatasources

        model = getattr(retriever, "_embed_model_name", "test-model")
        retriever._store._embed_model = model
        if not isinstance(retriever._store.get_collection_name(), str):
            retriever._store.get_collection_name.return_value = "kb_test"
        ds = test_db.execute(
            "INSERT INTO datasources (name, type, path) "
            "VALUES ('Rankingtest', 'local', '/ranking') RETURNING id"
        ).fetchone()["id"]
        test_db.commit()
        ChunkDatasources().add_associations(list(chunk_ids), ds, model)
        return ds

    return register


# ---------------------------------------------------------------------------
# Anonymizer mit vorgeladenen Testdaten
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_detection_config(tmp_config):
    """Config mit aktivierter LLM-PII-Erkennung fuer Tests."""
    tmp_config["anonymization"]["llm_detection"] = {
        "enabled": True,
        "model": "",
        "temperature": 0.1,
        "confidence_threshold": 0.7,
        "timeout_seconds": 5,
        "cache_size": 16,
        "num_ctx": 2048,
    }
    return tmp_config


@pytest.fixture
def loaded_anonymizer(sample_entities, tmp_config):
    """Anonymizer mit vorgeladenen Test-Entitaeten (ohne Config/LDAP)."""
    automaton = AnonymizationAutomaton()
    automaton.add_entities(sample_entities)
    automaton.build()
    return Anonymizer(automaton, tmp_config)
