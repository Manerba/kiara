"""Idempotente S4-Migration fuer persistente Index-Vertragsmetadaten.

Die Migration ist rein additiv.  Sie fuehrt weder Qdrant-Aufrufe noch
Embedding-Requests oder Bestands-Backfills aus.  Alle DDL-Schritte laufen in
der vom Aufrufer bereitgestellten PostgreSQL-Transaktion und werden durch
einen eigenen Advisory-Lock gegen parallele Service-Starts serialisiert.
"""

from __future__ import annotations

_MIGRATION_LOCK_KEY = 681034

# Startup startet mehrere KIara-Dienste dicht hintereinander. Der normale
# Query-Timeout (aktuell 30 s) ist fuer DDL-/Advisory-Lock-Wartezeiten zu kurz;
# die Abweichung gilt nur fuer die umgebende Migrationstransaktion und wird
# beim Commit/Rollback automatisch verworfen.
_MIGRATION_STATEMENT_TIMEOUT = "5min"

# Derselbe Key wird vom Laufzeit-Guard als Shared-Lock gehalten.  Mutationen
# der Embedding-Backend-Konfiguration erhalten ihn per DB-Trigger exklusiv.
INDEX_CONTRACT_CONFIG_LOCK_KEY = 493727140001


def migrate_index_contract_schema(conn=None) -> None:
    """Legt S4-Tabellen, Job-Fence-Spalten und Config-Trigger idempotent an."""

    close_conn = False
    if conn is None:
        from kiara.shared.db import get_db

        conn = get_db()
        close_conn = True

    try:
        conn.execute(f"SET LOCAL statement_timeout = '{_MIGRATION_STATEMENT_TIMEOUT}'")
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
        # CREATE/ALTER ... IF NOT EXISTS fordert auf PostgreSQL teils trotzdem
        # starke Tabellenlocks an. Nach dem ersten erfolgreichen Startup darf
        # deshalb kein weiterer Dienst die DDL erneut anfassen.
        if _schema_is_current(conn):
            if close_conn:
                conn.commit()
            return
        _create_evidence_table(conn)
        _create_contract_table(conn)
        _migrate_ingestion_jobs(conn)
        _create_backend_fence_trigger(conn)
        if close_conn:
            conn.commit()
    except Exception:
        if close_conn:
            conn.rollback()
        raise
    finally:
        if close_conn:
            conn.close()


_REQUIRED_COLUMNS = {
    "index_contract_evidence": {
        "id",
        "logical_index_key",
        "evidence_type",
        "result_state",
        "reason_code",
        "sample_count",
        "max_abs_error",
        "max_cosine_distance",
        "abs_tolerance",
        "cosine_tolerance",
        "collection_points_count",
        "collection_config_sha256",
        "sample_set_sha256",
        "references_json_text",
        "references_sha256",
        "diagnostics_json_text",
        "diagnostics_sha256",
        "candidate_index_compatibility_id",
        "candidate_query_compatibility_id",
        "contract_snapshot_sha256",
        "backfill_run_id",
        "created_at",
    },
    "index_contracts": {
        "logical_index_key",
        "lock_key",
        "store_type",
        "store_locator",
        "collection_name",
        "vector_kind",
        "requested_model",
        "backend_id",
        "backend_provider",
        "backend_url",
        "canonical_model_id",
        "profile_id",
        "profile_kind",
        "profile_endpoint",
        "contract_schema_version",
        "contract_schema_sha256",
        "contract_snapshot_json_text",
        "contract_snapshot_canonical_text",
        "contract_snapshot_sha256",
        "index_compatibility_id",
        "query_compatibility_id",
        "backend_type",
        "backend_implementation",
        "backend_implementation_revision",
        "artifact_repository",
        "artifact_revision",
        "artifact_manifest_digest",
        "artifact_json_text",
        "verification_state",
        "reason_code",
        "evidence_id",
        "evidence_type",
        "evidence_sample_count",
        "evidence_at",
        "observed_points_count",
        "backfill_run_id",
        "fence_version",
        "created_at",
        "updated_at",
    },
    "ingestion_jobs": {
        "contract_logical_index_key",
        "contract_index_compatibility_id",
        "contract_query_compatibility_id",
        "contract_canonical_model_id",
        "contract_profile_id",
        "contract_profile_kind",
        "contract_profile_endpoint",
        "contract_snapshot_sha256",
        "contract_fence_version",
    },
}

_REQUIRED_CONSTRAINTS = {
    "index_contract_evidence_pkey",
    "index_contract_evidence_type_check",
    "index_contract_evidence_state_check",
    "index_contract_evidence_sample_count_check",
    "index_contract_evidence_witness_count_check",
    "index_contracts_pkey",
    "index_contracts_lock_key_key",
    "index_contracts_physical_unique",
    "index_contracts_evidence_fk",
    "index_contracts_store_type_check",
    "index_contracts_vector_kind_check",
    "index_contracts_profile_kind_check",
    "index_contracts_verification_state_check",
    "index_contracts_fence_check",
    "index_contracts_sample_count_check",
    "index_contracts_snapshot_bundle_check",
    "index_contracts_verified_fields_check",
}

_REQUIRED_INDEXES = {
    "idx_index_contract_evidence_key_created",
    "idx_index_contracts_backend",
    "idx_index_contracts_state",
    "idx_ingestion_jobs_contract_key",
}

_REQUIRED_TRIGGERS = {
    "trg_ki_backends_contract_fence_insert_delete",
    "trg_ki_backends_contract_fence_update",
}


def _schema_is_current(conn) -> bool:
    """Return true only for the complete S4 schema, without taking DDL locks."""

    rows = conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = ANY(%s)",
        (list(_REQUIRED_COLUMNS),),
    ).fetchall()
    available_columns: dict[str, set[str]] = {table: set() for table in _REQUIRED_COLUMNS}
    for row in rows:
        available_columns[str(row["table_name"])].add(str(row["column_name"]))
    if any(
        not required.issubset(available_columns[table])
        for table, required in _REQUIRED_COLUMNS.items()
    ):
        return False

    constraints = {
        str(row["conname"])
        for row in conn.execute(
            "SELECT conname FROM pg_constraint WHERE connamespace = 'public'::regnamespace "
            "AND conrelid IN ('public.index_contracts'::regclass, "
            "'public.index_contract_evidence'::regclass)"
        ).fetchall()
    }
    if not _REQUIRED_CONSTRAINTS.issubset(constraints):
        return False

    indexes = {
        str(row["indexname"])
        for row in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
            "AND tablename IN ('index_contracts', 'index_contract_evidence', "
            "'ingestion_jobs')"
        ).fetchall()
    }
    if not _REQUIRED_INDEXES.issubset(indexes):
        return False

    triggers = {
        str(row["tgname"])
        for row in conn.execute(
            "SELECT tgname FROM pg_trigger WHERE tgrelid = 'public.ki_backends'::regclass "
            "AND NOT tgisinternal"
        ).fetchall()
    }
    return _REQUIRED_TRIGGERS.issubset(triggers)


def _create_evidence_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public.index_contract_evidence (
            id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            logical_index_key text NOT NULL,
            evidence_type text NOT NULL,
            result_state text NOT NULL,
            reason_code text NOT NULL,
            sample_count integer NOT NULL DEFAULT 0,
            max_abs_error double precision,
            max_cosine_distance double precision,
            abs_tolerance double precision,
            cosine_tolerance double precision,
            collection_points_count bigint NOT NULL DEFAULT 0,
            collection_config_sha256 text,
            sample_set_sha256 text,
            references_json_text text NOT NULL DEFAULT '[]',
            references_sha256 text NOT NULL,
            diagnostics_json_text text NOT NULL DEFAULT '{}',
            diagnostics_sha256 text NOT NULL,
            candidate_index_compatibility_id text,
            candidate_query_compatibility_id text,
            contract_snapshot_sha256 text,
            backfill_run_id text,
            created_at timestamp with time zone NOT NULL DEFAULT now(),
            CONSTRAINT index_contract_evidence_type_check CHECK (
                evidence_type IN ('witness', 'provenance', 'empty_first_write',
                                  'unverified', 'evidence_failed')
            ),
            CONSTRAINT index_contract_evidence_state_check CHECK (
                result_state IN ('verified', 'unverified', 'evidence_failed')
            ),
            CONSTRAINT index_contract_evidence_sample_count_check CHECK (sample_count >= 0),
            CONSTRAINT index_contract_evidence_witness_count_check CHECK (
                evidence_type <> 'witness' OR sample_count >= 16
            )
        )
        """
    )
    conn.execute(
        "ALTER TABLE public.index_contract_evidence ADD COLUMN IF NOT EXISTS backfill_run_id text"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_contract_evidence_key_created "
        "ON public.index_contract_evidence (logical_index_key, created_at DESC, id DESC)"
    )


def _create_contract_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public.index_contracts (
            logical_index_key text PRIMARY KEY,
            lock_key bigint NOT NULL UNIQUE,
            store_type text NOT NULL,
            store_locator text NOT NULL,
            collection_name text NOT NULL,
            vector_kind text NOT NULL,
            requested_model text NOT NULL,
            backend_id integer,
            backend_provider text,
            backend_url text,
            canonical_model_id text,
            profile_id text,
            profile_kind text,
            profile_endpoint text,
            contract_schema_version integer,
            contract_schema_sha256 text,
            contract_snapshot_json_text text,
            contract_snapshot_canonical_text text,
            contract_snapshot_sha256 text,
            index_compatibility_id text,
            query_compatibility_id text,
            backend_type text,
            backend_implementation text,
            backend_implementation_revision text,
            artifact_repository text,
            artifact_revision text,
            artifact_manifest_digest text,
            artifact_json_text text,
            verification_state text NOT NULL,
            reason_code text NOT NULL,
            evidence_id bigint NOT NULL,
            evidence_type text NOT NULL,
            evidence_sample_count integer NOT NULL DEFAULT 0,
            evidence_at timestamp with time zone NOT NULL,
            observed_points_count bigint NOT NULL DEFAULT 0,
            backfill_run_id text,
            fence_version bigint NOT NULL DEFAULT 1,
            created_at timestamp with time zone NOT NULL DEFAULT now(),
            updated_at timestamp with time zone NOT NULL DEFAULT now(),
            CONSTRAINT index_contracts_physical_unique UNIQUE (
                store_type, store_locator, collection_name
            ),
            CONSTRAINT index_contracts_evidence_fk FOREIGN KEY (evidence_id)
                REFERENCES public.index_contract_evidence(id) ON DELETE RESTRICT,
            CONSTRAINT index_contracts_store_type_check CHECK (store_type = 'qdrant'),
            CONSTRAINT index_contracts_vector_kind_check CHECK (
                vector_kind IN ('dense', 'late_chunking', 'multi_vector')
            ),
            CONSTRAINT index_contracts_profile_kind_check CHECK (
                profile_kind IS NULL OR profile_kind IN ('dense', 'late_chunking', 'multi_vector')
            ),
            CONSTRAINT index_contracts_verification_state_check CHECK (
                verification_state IN ('verified', 'unverified', 'evidence_failed')
            ),
            CONSTRAINT index_contracts_fence_check CHECK (fence_version > 0),
            CONSTRAINT index_contracts_sample_count_check CHECK (evidence_sample_count >= 0),
            CONSTRAINT index_contracts_snapshot_bundle_check CHECK (
                (contract_snapshot_json_text IS NULL
                 AND contract_snapshot_canonical_text IS NULL
                 AND contract_snapshot_sha256 IS NULL)
                OR
                (contract_snapshot_json_text IS NOT NULL
                 AND contract_snapshot_canonical_text IS NOT NULL
                 AND contract_snapshot_sha256 IS NOT NULL)
            ),
            CONSTRAINT index_contracts_verified_fields_check CHECK (
                (verification_state = 'verified'
                 AND canonical_model_id IS NOT NULL
                 AND profile_id IS NOT NULL
                 AND profile_kind IS NOT NULL
                 AND profile_endpoint IS NOT NULL
                 AND contract_schema_version IS NOT NULL
                 AND contract_schema_sha256 IS NOT NULL
                 AND contract_snapshot_sha256 IS NOT NULL
                 AND index_compatibility_id IS NOT NULL
                 AND query_compatibility_id IS NOT NULL)
                OR
                (verification_state <> 'verified'
                 AND index_compatibility_id IS NULL
                 AND query_compatibility_id IS NULL)
            )
        )
        """
    )
    conn.execute("ALTER TABLE public.index_contracts ADD COLUMN IF NOT EXISTS backfill_run_id text")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_contracts_backend "
        "ON public.index_contracts (backend_id) WHERE backend_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_contracts_state "
        "ON public.index_contracts (verification_state, reason_code)"
    )


def _migrate_ingestion_jobs(conn) -> None:
    columns = (
        ("contract_logical_index_key", "text"),
        ("contract_index_compatibility_id", "text"),
        ("contract_query_compatibility_id", "text"),
        ("contract_canonical_model_id", "text"),
        ("contract_profile_id", "text"),
        ("contract_profile_kind", "text"),
        ("contract_profile_endpoint", "text"),
        ("contract_snapshot_sha256", "text"),
        ("contract_fence_version", "bigint"),
    )
    for column, sql_type in columns:
        conn.execute(
            f"ALTER TABLE public.ingestion_jobs ADD COLUMN IF NOT EXISTS {column} {sql_type}"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_contract_key "
        "ON public.ingestion_jobs (contract_logical_index_key) "
        "WHERE contract_logical_index_key IS NOT NULL"
    )


def _create_backend_fence_trigger(conn) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.kiara_lock_embedding_contract_config()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $function$
        DECLARE
            old_category text;
            new_category text;
        BEGIN
            IF TG_OP <> 'INSERT' THEN
                old_category := OLD.category;
            END IF;
            IF TG_OP <> 'DELETE' THEN
                new_category := NEW.category;
            END IF;
            IF old_category = 'embedding' OR new_category = 'embedding' THEN
                PERFORM pg_advisory_xact_lock({INDEX_CONTRACT_CONFIG_LOCK_KEY});
            END IF;
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END
        $function$
        """
    )
    conn.execute(
        "DROP TRIGGER IF EXISTS trg_ki_backends_contract_fence_insert_delete ON public.ki_backends"
    )
    conn.execute(
        "CREATE TRIGGER trg_ki_backends_contract_fence_insert_delete "
        "BEFORE INSERT OR DELETE ON public.ki_backends "
        "FOR EACH ROW EXECUTE FUNCTION public.kiara_lock_embedding_contract_config()"
    )
    conn.execute(
        "DROP TRIGGER IF EXISTS trg_ki_backends_contract_fence_update ON public.ki_backends"
    )
    conn.execute(
        "CREATE TRIGGER trg_ki_backends_contract_fence_update "
        "BEFORE UPDATE OF category, provider, url, base_url, model, is_active, priority, extra_config "
        "ON public.ki_backends FOR EACH ROW "
        "EXECUTE FUNCTION public.kiara_lock_embedding_contract_config()"
    )
