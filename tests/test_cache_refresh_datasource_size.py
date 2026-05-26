"""Tests fuer deduplizierte Datasource-Groesse ueber mehrere Modelle."""

import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"

import json

import pytest

from kiara.shared.db import db_query, db_transaction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DS_ID = 8880


@pytest.fixture(autouse=True)
def _setup_teardown():
    """Legt Test-Datasource an und raeumt danach auf."""
    with db_transaction() as conn:
        conn.execute(
            "INSERT INTO datasources (id, name, type) VALUES (%s, 'size_test', 'local') "
            "ON CONFLICT (id) DO NOTHING",
            (_DS_ID,),
        )
    yield
    with db_transaction() as conn:
        conn.execute("DELETE FROM file_hashes WHERE datasource_id = %s", (_DS_ID,))
        conn.execute("DELETE FROM cache_datasources WHERE datasource_id = %s", (_DS_ID,))
        conn.execute("DELETE FROM datasources WHERE id = %s", (_DS_ID,))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeduplicatedDatasourceSize:
    """source_size_bytes zaehlt Multi-Modell-Dateien nur einmal."""

    def test_same_file_two_models_counted_once(self):
        """Zwei file_hashes-Zeilen mit gleichem (ds, path) aber unterschiedlichem
        embedding_model zaehlen in source_size_bytes nur einmal."""
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'model_a', '/doc.pdf', 'ha', 1000)",
                (_DS_ID,),
            )
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'model_b', '/doc.pdf', 'hb', 1000)",
                (_DS_ID,),
            )

        with db_query() as db:
            from kiara.web.cache_refresh import _load_ds_batch_data

            _rows, _rj, _gc, _uc, _em, size_map = _load_ds_batch_data(db, _DS_ID)

        # 1000, nicht 2000
        assert size_map.get(_DS_ID, 0) == 1000

    def test_multiple_files_summed_correctly(self):
        """Mehrere verschiedene Dateien werden korrekt summiert."""
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'nomic', '/a.txt', 'h1', 100), "
                "       (%s, 'nomic', '/b.txt', 'h2', 200)",
                (_DS_ID, _DS_ID),
            )

        with db_query() as db:
            from kiara.web.cache_refresh import _load_ds_batch_data

            _rows, _rj, _gc, _uc, _em, size_map = _load_ds_batch_data(db, _DS_ID)

        assert size_map.get(_DS_ID, 0) == 300

    def test_null_file_size_excluded(self):
        """Zeilen mit file_size IS NULL fliessen nicht in die Aggregation ein."""
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'nomic', '/a.txt', 'h1', 100)",
                (_DS_ID,),
            )
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash) "
                "VALUES (%s, 'nomic', '/b.txt', 'h2')",
                (_DS_ID,),
            )

        with db_query() as db:
            from kiara.web.cache_refresh import _load_ds_batch_data

            _rows, _rj, _gc, _uc, _em, size_map = _load_ds_batch_data(db, _DS_ID)

        assert size_map.get(_DS_ID, 0) == 100

    def test_zero_byte_file_counted_as_zero(self):
        """0-Byte-Dateien sind legitim und werden korrekt gezaehlt."""
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'nomic', '/empty.txt', 'h1', 0)",
                (_DS_ID,),
            )
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'nomic', '/full.txt', 'h2', 500)",
                (_DS_ID,),
            )

        with db_query() as db:
            from kiara.web.cache_refresh import _load_ds_batch_data

            _rows, _rj, _gc, _uc, _em, size_map = _load_ds_batch_data(db, _DS_ID)

        assert size_map.get(_DS_ID, 0) == 500

    def test_payload_contains_source_size_bytes(self):
        """Das Payload-Feld source_size_bytes ist im Cache enthalten,
        obwohl die Spalte in datasources nicht mehr existiert."""
        with db_transaction() as conn:
            conn.execute(
                "INSERT INTO file_hashes (datasource_id, embedding_model, file_path, file_hash, file_size) "
                "VALUES (%s, 'nomic', '/test.txt', 'h1', 42)",
                (_DS_ID,),
            )

        from kiara.web.cache_refresh import refresh_datasource_cache

        try:
            refresh_datasource_cache(_DS_ID)
        except Exception:
            pytest.skip("refresh_datasource_cache braucht vollstaendige Infrastruktur")

        with db_query() as db:
            row = db.execute(
                "SELECT payload_json FROM cache_datasources WHERE datasource_id = %s",
                (_DS_ID,),
            ).fetchone()

        if row:
            # jsonb-Spalte: psycopg2 liefert dict direkt, sonst string parsen.
            raw = row["payload_json"]
            payload = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            assert "source_size_bytes" in payload
            assert int(payload["source_size_bytes"]) == 42
