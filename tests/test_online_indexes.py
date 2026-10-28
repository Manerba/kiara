"""Online-Index-Deploy gegen isolierte Schemas der PostgreSQL-Testdatenbank."""

import uuid

import psycopg2
import pytest
from psycopg2 import sql

from scripts.deploy_online_indexes import ensure_file_hash_path_index


@pytest.fixture
def index_db():
    from db_helpers import _TEST_DSN

    conn = psycopg2.connect(_TEST_DSN, options="-c statement_timeout=5000")
    conn.autocommit = True
    schema = "checkpoint_test_" + uuid.uuid4().hex
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        assert cur.fetchone()[0] == "kiara_test"
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                "CREATE TABLE {}.file_hashes ("
                "datasource_id integer, embedding_model text, file_path text, "
                "file_hash text, file_size bigint, "
                "PRIMARY KEY (datasource_id, embedding_model, file_path))"
            ).format(sql.Identifier(schema))
        )
    try:
        yield conn, schema
    finally:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        conn.close()


def test_online_index_preserves_cross_model_updates_and_is_idempotent(index_db):
    conn, schema = index_db
    table = sql.Identifier(schema, "file_hashes")
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "INSERT INTO {} VALUES "
                "(1,'a','/doc','hash-a',10), (1,'b','/doc','hash-b',20), "
                "(2,'a','/doc','other-ds',30), (1,'a','/other','other-file',40)"
            ).format(table)
        )
    assert ensure_file_hash_path_index(conn, schema=schema)["created"] is True
    assert ensure_file_hash_path_index(conn, schema=schema)["created"] is False
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("UPDATE {} SET file_size=99 WHERE datasource_id=1 AND file_path='/doc'").format(
                table
            )
        )
        assert cur.rowcount == 2
        cur.execute(
            sql.SQL(
                "SELECT embedding_model,file_hash,file_size FROM {} "
                "ORDER BY datasource_id,file_path,embedding_model"
            ).format(table)
        )
        assert cur.fetchall() == [
            ("a", "hash-a", 99),
            ("b", "hash-b", 99),
            ("a", "other-file", 40),
            ("a", "other-ds", 30),
        ]


@pytest.mark.parametrize("repair_invalid", [False, True])
def test_online_index_rejects_existing_wrong_definition(index_db, repair_invalid):
    conn, schema = index_db
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE INDEX idx_file_hashes_ds_path ON {}.file_hashes (datasource_id)"
            ).format(sql.Identifier(schema))
        )
    with pytest.raises(RuntimeError, match="weicht vom Soll ab"):
        ensure_file_hash_path_index(conn, schema=schema, repair_invalid=repair_invalid)


@pytest.mark.parametrize("repair_invalid", [False, True])
def test_online_index_rejects_unfinished_concurrent_build(index_db, repair_invalid):
    conn, schema = index_db
    # Ein fehlgeschlagener Unique-Aufbau hinterlaesst real indisvalid=false.
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "INSERT INTO {}.file_hashes VALUES (1,'a','/doc','a',10), (1,'b','/doc','b',20)"
            ).format(sql.Identifier(schema))
        )
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX CONCURRENTLY idx_file_hashes_ds_path "
                    "ON {}.file_hashes (datasource_id,file_path)"
                ).format(sql.Identifier(schema))
            )
    with pytest.raises(RuntimeError, match="ungueltig"):
        ensure_file_hash_path_index(conn, schema=schema, repair_invalid=repair_invalid)


def test_online_index_repairs_own_definition_after_lock_timeout(index_db):
    from db_helpers import _TEST_DSN

    conn, schema = index_db
    writer = psycopg2.connect(_TEST_DSN)
    try:
        with writer.cursor() as cur:
            cur.execute(
                sql.SQL("INSERT INTO {}.file_hashes VALUES (1,'a','/doc','a',10)").format(
                    sql.Identifier(schema)
                )
            )
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout=100")
        with pytest.raises(psycopg2.errors.LockNotAvailable):
            ensure_file_hash_path_index(conn, schema=schema)
        writer.commit()
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout=0")
        with pytest.raises(RuntimeError, match="ungueltig"):
            ensure_file_hash_path_index(conn, schema=schema)
        result = ensure_file_hash_path_index(conn, schema=schema, repair_invalid=True)
        assert result["valid"] is True
        assert result["repaired_invalid"] is True
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT file_hash,file_size FROM {}.file_hashes").format(
                    sql.Identifier(schema)
                )
            )
            assert cur.fetchall() == [("a", 10)]
    finally:
        writer.close()


def test_online_index_rejects_open_transaction(index_db):
    conn, schema = index_db
    conn.autocommit = False
    try:
        with pytest.raises(ValueError, match="autocommit"):
            ensure_file_hash_path_index(conn, schema=schema)
    finally:
        conn.autocommit = True
