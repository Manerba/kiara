"""Idempotente Live-Migration fuer Graph-Schema-Erweiterungen.

``schema.sql`` bleibt Source of Truth fuer frische Datenbanken. Dieser Helper
repariert bestehende DBs, in denen Graph-Provenienz oder Chunk-Junctions aus
frueheren Versionen fehlen, ohne Tabellen zu droppen.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

from kiara.shared.db import db_transaction

logger = logging.getLogger(__name__)

_MIGRATION_LOCK_KEY = 681033


def migrate_graph_schema(conn=None) -> None:
    """Wendet die Graph-Schema-Migration idempotent an.

    Bei leerer Web-DB oder fehlenden Graph-Basistabellen laeuft der Helper no-op;
    der Greenfield-Pfad wendet anschliessend ``schema.sql`` an.
    """
    ctx = nullcontext(conn) if conn is not None else db_transaction()
    with ctx as db:
        db.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
        if not _table_exists(db, "graph_edges") or not _table_exists(db, "graph_nodes"):
            return

        _ensure_edge_source(db)
        _ensure_edge_node_cascade(db)
        _ensure_junction_tables(db)
        _backfill_edge_chunks(db)
        _ensure_edge_unique_index(db)
        _diagnose_unlinked_extraction_nodes(db)


def _table_exists(db, table_name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(db, table_name: str, column_name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
        (table_name, column_name),
    ).fetchone()
    return row is not None


def _constraint_exists(db, table_name: str, constraint_name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE table_schema = 'public' AND table_name = %s "
        "AND constraint_name = %s",
        (table_name, constraint_name),
    ).fetchone()
    return row is not None


def _primary_key_matches(
    db,
    table_name: str,
    constraint_name: str,
    columns: list[str],
) -> bool:
    rows = db.execute(
        """SELECT kcu.column_name
           FROM information_schema.table_constraints tc
           JOIN information_schema.key_column_usage kcu
             ON kcu.constraint_schema = tc.constraint_schema
            AND kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
            AND kcu.table_name = tc.table_name
           WHERE tc.table_schema = 'public'
             AND tc.table_name = %s
             AND tc.constraint_name = %s
             AND tc.constraint_type = 'PRIMARY KEY'
           ORDER BY kcu.ordinal_position""",
        (table_name, constraint_name),
    ).fetchall()
    return [row["column_name"] for row in rows] == columns


def _foreign_key_delete_rule(db, constraint_name: str) -> str | None:
    row = db.execute(
        """SELECT delete_rule
           FROM information_schema.referential_constraints
           WHERE constraint_schema = 'public'
             AND constraint_name = %s""",
        (constraint_name,),
    ).fetchone()
    return row["delete_rule"] if row else None


def _drop_constraint(db, table_name: str, constraint_name: str) -> None:
    db.execute(f"ALTER TABLE public.{table_name} DROP CONSTRAINT IF EXISTS {constraint_name}")


def _index_definition(db, index_name: str) -> str | None:
    row = db.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s",
        (index_name,),
    ).fetchone()
    return row["indexdef"] if row else None


def _ensure_plain_btree_index(
    db,
    index_name: str,
    table_name: str,
    column_name: str,
) -> None:
    expected_suffix = f"ON public.{table_name} USING btree ({column_name})"
    current = _index_definition(db, index_name)
    if current and expected_suffix not in current:
        db.execute(f"DROP INDEX IF EXISTS public.{index_name}")
        current = None
    if not current:
        db.execute(f"CREATE INDEX {index_name} ON public.{table_name} USING btree ({column_name})")


def _ensure_edge_source(db) -> None:
    if not _column_exists(db, "graph_edges", "source"):
        # Add without a default first so legacy rows stay NULL long enough for
        # the ERP-edge recovery heuristic below. Adding the default immediately
        # would make all old rows look like ordinary extraction edges.
        db.execute("ALTER TABLE public.graph_edges ADD COLUMN source text")

    # Heuristik fuer legacy ERP-Kanten ohne Chunk-Provenienz.
    # Greift in zwei Partial-State-Szenarien:
    #   a) source IS NULL — Spalte wurde hier gerade ohne DEFAULT angelegt
    #      (Fresh-Migration-Pfad).
    #   b) source = 'extraction' — Spalte war bereits vorhanden mit
    #      DEFAULT 'extraction' durch manuelle/halbfertige Migration, und alle
    #      Legacy-Rows haben diesen Default geerbt (ImplScan V28 Finding 1).
    # Die ERP-Relation-/Prefix-Paare sind eindeutig ERP-spezifisch, daher ist
    # die Heuristik auch fuer 'extraction'-Rows kollisionssicher.
    # Falls graph_edges.chunk_id (noch) nicht existiert, faellt die NULL-Bedingung
    # weg — semantisch aequivalent zu "alle Rows ohne chunk_id-Provenienz".
    chunk_id_clause = (
        "AND chunk_id IS NULL" if _column_exists(db, "graph_edges", "chunk_id") else ""
    )
    db.execute(
        f"""UPDATE public.graph_edges
           SET source = 'erp_import'
           WHERE (source IS NULL OR source = 'extraction')
             {chunk_id_clause}
             AND (
               (source_id LIKE 'order:%%' AND target_id LIKE 'customer:%%'
                    AND relation = 'bestellt_bei')
               OR (source_id LIKE 'order:%%' AND target_id LIKE 'article:%%'
                    AND relation = 'enthaelt')
               OR (source_id LIKE 'production_order:%%' AND target_id LIKE 'order:%%'
                    AND relation = 'produziert_fuer')
               OR (source_id LIKE 'supplier:%%' AND target_id LIKE 'article:%%'
                    AND relation = 'liefert')
             )"""
    )
    db.execute("UPDATE public.graph_edges SET source = 'extraction' WHERE source IS NULL")
    db.execute("ALTER TABLE public.graph_edges ALTER COLUMN source SET DEFAULT 'extraction'::text")
    db.execute("ALTER TABLE public.graph_edges ALTER COLUMN source SET NOT NULL")

    _ensure_plain_btree_index(db, "idx_graph_edges_source_prov", "graph_edges", "source")


def _ensure_edge_node_cascade(db) -> None:
    """Stellt ON DELETE CASCADE fuer graph_edges → graph_nodes FKs sicher.

    Anwendungscode (store.py, repositories/graph.py) geht von CASCADE-Semantik
    aus; ohne diese FK-Klausel schlaegt jeder direkte DELETE FROM graph_nodes
    mit FK-Constraint-Verletzung fehl, sobald noch Edges referenzieren.
    """
    for column in ("source_id", "target_id"):
        constraint = f"graph_edges_{column}_fkey"
        if _foreign_key_delete_rule(db, constraint) not in (None, "CASCADE"):
            _drop_constraint(db, "graph_edges", constraint)
        if not _constraint_exists(db, "graph_edges", constraint):
            db.execute(
                f"ALTER TABLE public.graph_edges "
                f"ADD CONSTRAINT {constraint} "
                f"FOREIGN KEY ({column}) REFERENCES public.graph_nodes(id) "
                f"ON DELETE CASCADE"
            )


def _ensure_edge_unique_index(db) -> None:
    """Erzwingt eindeutige Graph-Edges pro Provenienz."""
    _dedupe_edges_by_provenance_key(db)
    expected_suffix = "ON public.graph_edges USING btree (source_id, target_id, relation, source)"
    current = _index_definition(db, "idx_graph_edges_unique")
    if current and expected_suffix not in current:
        db.execute("DROP INDEX IF EXISTS public.idx_graph_edges_unique")
        current = None
    if not current:
        db.execute(
            "CREATE UNIQUE INDEX idx_graph_edges_unique "
            "ON public.graph_edges USING btree "
            "(source_id, target_id, relation, source)"
        )


def _dedupe_edges_by_provenance_key(db) -> None:
    """Fuehrt exakt gleiche physische Edges zusammen, bevor der Unique-Key greift."""
    if _table_exists(db, "graph_edge_chunks"):
        db.execute(
            """WITH duplicate_edges AS (
                   SELECT id,
                          MIN(id) OVER (
                              PARTITION BY source_id, target_id, relation, source
                          ) AS keep_id
                   FROM public.graph_edges
               )
               INSERT INTO public.graph_edge_chunks (edge_id, chunk_id)
               SELECT de.keep_id, gec.chunk_id
               FROM duplicate_edges de
               JOIN public.graph_edge_chunks gec ON gec.edge_id = de.id
               WHERE de.id <> de.keep_id
               ON CONFLICT (edge_id, chunk_id) DO NOTHING"""
        )
        db.execute(
            """WITH duplicate_edges AS (
                   SELECT id,
                          MIN(id) OVER (
                              PARTITION BY source_id, target_id, relation, source
                          ) AS keep_id
                   FROM public.graph_edges
               )
               DELETE FROM public.graph_edge_chunks gec
               USING duplicate_edges de
               WHERE gec.edge_id = de.id
                 AND de.id <> de.keep_id"""
        )
    db.execute(
        """WITH duplicate_edges AS (
               SELECT id,
                      MIN(id) OVER (
                          PARTITION BY source_id, target_id, relation, source
                      ) AS keep_id
               FROM public.graph_edges
           )
           DELETE FROM public.graph_edges ge
           USING duplicate_edges de
           WHERE ge.id = de.id
             AND de.id <> de.keep_id"""
    )


def _ensure_junction_tables(db) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS public.graph_node_chunks (
               node_id text NOT NULL,
               chunk_id text NOT NULL,
               PRIMARY KEY (node_id, chunk_id)
           )"""
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS public.graph_edge_chunks (
               edge_id integer NOT NULL,
               chunk_id text NOT NULL,
               PRIMARY KEY (edge_id, chunk_id)
           )"""
    )
    _dedupe_junction(db, "graph_node_chunks", ["node_id", "chunk_id"])
    _dedupe_junction(db, "graph_edge_chunks", ["edge_id", "chunk_id"])

    if _constraint_exists(
        db, "graph_node_chunks", "graph_node_chunks_pkey"
    ) and not _primary_key_matches(
        db,
        "graph_node_chunks",
        "graph_node_chunks_pkey",
        ["node_id", "chunk_id"],
    ):
        _drop_constraint(db, "graph_node_chunks", "graph_node_chunks_pkey")
    if _constraint_exists(
        db, "graph_edge_chunks", "graph_edge_chunks_pkey"
    ) and not _primary_key_matches(
        db,
        "graph_edge_chunks",
        "graph_edge_chunks_pkey",
        ["edge_id", "chunk_id"],
    ):
        _drop_constraint(db, "graph_edge_chunks", "graph_edge_chunks_pkey")

    if not _constraint_exists(db, "graph_node_chunks", "graph_node_chunks_pkey"):
        db.execute(
            "ALTER TABLE public.graph_node_chunks "
            "ADD CONSTRAINT graph_node_chunks_pkey PRIMARY KEY (node_id, chunk_id)"
        )
    if not _constraint_exists(db, "graph_edge_chunks", "graph_edge_chunks_pkey"):
        db.execute(
            "ALTER TABLE public.graph_edge_chunks "
            "ADD CONSTRAINT graph_edge_chunks_pkey PRIMARY KEY (edge_id, chunk_id)"
        )

    if _foreign_key_delete_rule(db, "graph_node_chunks_node_id_fkey") not in (
        None,
        "CASCADE",
    ):
        _drop_constraint(db, "graph_node_chunks", "graph_node_chunks_node_id_fkey")
    if _foreign_key_delete_rule(db, "graph_edge_chunks_edge_id_fkey") not in (
        None,
        "CASCADE",
    ):
        _drop_constraint(db, "graph_edge_chunks", "graph_edge_chunks_edge_id_fkey")

    if not _constraint_exists(db, "graph_node_chunks", "graph_node_chunks_node_id_fkey"):
        db.execute(
            "ALTER TABLE public.graph_node_chunks "
            "ADD CONSTRAINT graph_node_chunks_node_id_fkey "
            "FOREIGN KEY (node_id) REFERENCES public.graph_nodes(id) ON DELETE CASCADE"
        )
    if not _constraint_exists(db, "graph_edge_chunks", "graph_edge_chunks_edge_id_fkey"):
        db.execute(
            "ALTER TABLE public.graph_edge_chunks "
            "ADD CONSTRAINT graph_edge_chunks_edge_id_fkey "
            "FOREIGN KEY (edge_id) REFERENCES public.graph_edges(id) ON DELETE CASCADE"
        )

    _ensure_plain_btree_index(db, "idx_graph_node_chunks_chunk", "graph_node_chunks", "chunk_id")
    _ensure_plain_btree_index(db, "idx_graph_edge_chunks_chunk", "graph_edge_chunks", "chunk_id")


def _dedupe_junction(db, table_name: str, columns: list[str]) -> None:
    db.execute(
        f"""DELETE FROM public.{table_name}
            WHERE {" OR ".join(f"{c} IS NULL" for c in columns)}"""
    )
    db.execute(
        f"""DELETE FROM public.{table_name} a
            USING public.{table_name} b
            WHERE a.ctid < b.ctid
              AND ({" AND ".join(f"a.{c} = b.{c}" for c in columns)})"""
    )
    for col in columns:
        db.execute(f"ALTER TABLE public.{table_name} ALTER COLUMN {col} SET NOT NULL")


def _backfill_edge_chunks(db) -> None:
    if not _column_exists(db, "graph_edges", "chunk_id"):
        return
    db.execute(
        """INSERT INTO public.graph_edge_chunks (edge_id, chunk_id)
           SELECT id, chunk_id
           FROM public.graph_edges
           WHERE chunk_id IS NOT NULL
           ON CONFLICT (edge_id, chunk_id) DO NOTHING"""
    )


def _diagnose_unlinked_extraction_nodes(db) -> None:
    row = db.execute(
        """SELECT COUNT(*) AS cnt
           FROM public.graph_nodes gn
           WHERE gn.source = 'extraction'
             AND NOT EXISTS (
               SELECT 1 FROM public.graph_node_chunks gnc
               WHERE gnc.node_id = gn.id
             )"""
    ).fetchone()
    count = (row["cnt"] if row else 0) or 0
    if count:
        logger.warning(
            "Graph-Schema-Migration: %d Extraction-Nodes haben keinen "
            "graph_node_chunks-Link; Re-Ingest/Cleanup kann erforderlich sein.",
            count,
        )
