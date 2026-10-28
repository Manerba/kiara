"""Regressionen fuer die idempotente Graph-Schema-Migration."""

import pytest

from kiara.shared.db import get_db
from kiara.web.graph_schema_migration import migrate_graph_schema


@pytest.fixture
def old_graph_schema():
    """Erzeugt ein altes minimales Graph-Schema in einer Rollback-TX."""
    conn = get_db()
    try:
        conn.execute("DROP TABLE IF EXISTS graph_edge_chunks CASCADE")
        conn.execute("DROP TABLE IF EXISTS graph_node_chunks CASCADE")
        conn.execute("DROP TABLE IF EXISTS graph_edges CASCADE")
        conn.execute("DROP TABLE IF EXISTS graph_nodes CASCADE")
        conn.execute("DROP SEQUENCE IF EXISTS graph_edges_id_seq CASCADE")
        conn.execute(
            """CREATE TABLE public.graph_nodes (
                   id text PRIMARY KEY,
                   entity_type text NOT NULL,
                   label text NOT NULL,
                   properties text DEFAULT '{}'::text,
                   source text DEFAULT 'extraction'::text,
                   created_at timestamp with time zone DEFAULT now()
               )"""
        )
        conn.execute(
            """CREATE TABLE public.graph_edges (
                   id serial PRIMARY KEY,
                   source_id text NOT NULL REFERENCES public.graph_nodes(id),
                   target_id text NOT NULL REFERENCES public.graph_nodes(id),
                   relation text NOT NULL,
                   properties text DEFAULT '{}'::text,
                   chunk_id text,
                   created_at timestamp with time zone DEFAULT now()
               )"""
        )
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _insert_node(conn, node_id: str, entity_type: str) -> None:
    conn.execute(
        "INSERT INTO graph_nodes (id, entity_type, label) VALUES (%s, %s, %s)",
        (node_id, entity_type, node_id),
    )


def test_migration_classifies_legacy_erp_edges_before_default(old_graph_schema):
    conn = old_graph_schema
    for node_id, entity_type in (
        ("order:a-1", "order"),
        ("customer:k-1", "customer"),
        ("topic:manual", "topic"),
        ("article:art-1", "article"),
    ):
        _insert_node(conn, node_id, entity_type)
    conn.execute(
        """INSERT INTO graph_edges (source_id, target_id, relation, chunk_id)
           VALUES
             ('order:a-1', 'customer:k-1', 'bestellt_bei', NULL),
             ('topic:manual', 'article:art-1', 'betrifft', 'chunk-1')"""
    )

    migrate_graph_schema(conn)

    rows = conn.execute("SELECT relation, source FROM graph_edges ORDER BY id").fetchall()
    assert [dict(r) for r in rows] == [
        {"relation": "bestellt_bei", "source": "erp_import"},
        {"relation": "betrifft", "source": "extraction"},
    ]
    column = conn.execute(
        """SELECT is_nullable, column_default
           FROM information_schema.columns
           WHERE table_schema = 'public'
             AND table_name = 'graph_edges'
             AND column_name = 'source'"""
    ).fetchone()
    assert column["is_nullable"] == "NO"
    assert column["column_default"] == "'extraction'::text"
    edge_chunks = conn.execute(
        "SELECT chunk_id FROM graph_edge_chunks ORDER BY chunk_id"
    ).fetchall()
    assert [r["chunk_id"] for r in edge_chunks] == ["chunk-1"]


def test_migration_repairs_partial_junction_constraints_and_index(old_graph_schema):
    conn = old_graph_schema
    conn.execute("ALTER TABLE graph_edges ADD COLUMN source text")
    conn.execute("CREATE INDEX idx_graph_edges_source_prov ON graph_edges(source_id)")
    conn.execute(
        "CREATE UNIQUE INDEX idx_graph_edges_unique ON graph_edges(source_id, target_id, relation)"
    )
    conn.execute(
        """CREATE TABLE graph_node_chunks (
               node_id text,
               chunk_id text
           )"""
    )
    conn.execute(
        """CREATE TABLE graph_edge_chunks (
               edge_id integer,
               chunk_id text
           )"""
    )
    conn.execute(
        "ALTER TABLE graph_node_chunks "
        "ADD CONSTRAINT graph_node_chunks_pkey PRIMARY KEY (chunk_id, node_id)"
    )
    conn.execute(
        "ALTER TABLE graph_node_chunks "
        "ADD CONSTRAINT graph_node_chunks_node_id_fkey "
        "FOREIGN KEY (node_id) REFERENCES graph_nodes(id)"
    )
    conn.execute(
        "ALTER TABLE graph_edge_chunks "
        "ADD CONSTRAINT graph_edge_chunks_edge_id_fkey "
        "FOREIGN KEY (edge_id) REFERENCES graph_edges(id)"
    )

    migrate_graph_schema(conn)

    pk_rows = conn.execute(
        """SELECT kcu.column_name
           FROM information_schema.table_constraints tc
           JOIN information_schema.key_column_usage kcu
             ON kcu.constraint_schema = tc.constraint_schema
            AND kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
            AND kcu.table_name = tc.table_name
           WHERE tc.table_schema = 'public'
             AND tc.table_name = 'graph_node_chunks'
             AND tc.constraint_name = 'graph_node_chunks_pkey'
           ORDER BY kcu.ordinal_position"""
    ).fetchall()
    pk_cols = [row["column_name"] for row in pk_rows]
    assert pk_cols == ["node_id", "chunk_id"]

    rules = conn.execute(
        """SELECT constraint_name, delete_rule
           FROM information_schema.referential_constraints
           WHERE constraint_schema = 'public'
             AND constraint_name IN (
               'graph_node_chunks_node_id_fkey',
               'graph_edge_chunks_edge_id_fkey'
             )
           ORDER BY constraint_name"""
    ).fetchall()
    assert {r["constraint_name"]: r["delete_rule"] for r in rules} == {
        "graph_edge_chunks_edge_id_fkey": "CASCADE",
        "graph_node_chunks_node_id_fkey": "CASCADE",
    }

    indexdef = conn.execute(
        """SELECT indexdef FROM pg_indexes
           WHERE schemaname = 'public'
             AND indexname = 'idx_graph_edges_source_prov'"""
    ).fetchone()["indexdef"]
    assert "ON public.graph_edges USING btree (source)" in indexdef

    unique_indexdef = conn.execute(
        """SELECT indexdef FROM pg_indexes
           WHERE schemaname = 'public'
             AND indexname = 'idx_graph_edges_unique'"""
    ).fetchone()["indexdef"]
    assert (
        "ON public.graph_edges USING btree (source_id, target_id, relation, source)"
    ) in unique_indexdef
