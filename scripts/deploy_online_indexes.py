"""Gezielter Online-Index-Deploy; Aufruf ueber ./deploy.sh --online-indexes.

Keine App-Initialisierung, Seeds, Datenaenderungen oder Service-Neustarts.
Neue Datenbanken erhalten denselben Index aus kiara/web/schema.sql.
"""

import argparse
import json
import time
from pathlib import Path

import psycopg2
import yaml
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

_INDEX_NAME = "idx_file_hashes_ds_path"


def _read_index(conn, schema):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT i.indisvalid, i.indisready, i.indislive, i.indisunique, "
            "i.indnatts, i.indnkeyatts, i.indpred IS NULL AS unfiltered, "
            "i.indexprs IS NULL AS plain, t.relname AS table_name, "
            "tn.nspname AS table_schema, am.amname AS method, "
            "ARRAY(SELECT pg_get_indexdef(idx.oid, n, false) "
            "      FROM generate_series(1, i.indnatts) n) AS columns "
            "FROM pg_class idx JOIN pg_namespace ns ON ns.oid=idx.relnamespace "
            "LEFT JOIN pg_index i ON i.indexrelid=idx.oid "
            "LEFT JOIN pg_class t ON t.oid=i.indrelid "
            "LEFT JOIN pg_namespace tn ON tn.oid=t.relnamespace "
            "LEFT JOIN pg_am am ON am.oid=idx.relam "
            "WHERE ns.nspname=%s AND idx.relname=%s",
            (schema, _INDEX_NAME),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _validate_index(state, schema):
    expected = {
        "indisvalid": True,
        "indisready": True,
        "indislive": True,
        "indisunique": False,
        "indnatts": 2,
        "indnkeyatts": 2,
        "unfiltered": True,
        "plain": True,
        "table_name": "file_hashes",
        "table_schema": schema,
        "method": "btree",
        "columns": ["datasource_id", "file_path"],
    }
    if state != expected:
        raise RuntimeError(
            f"{schema}.{_INDEX_NAME}: Index fehlt, ist ungueltig oder weicht vom Soll ab: {state}"
        )


def ensure_file_hash_path_index(conn, *, schema="public", repair_invalid=False):
    """Erstellt den vollstaendigen Pfadindex online und prueft seinen Zustand.

    Ein fehlgeschlagener CONCURRENTLY-Aufbau kann einen ungueltigen Index
    hinterlassen. Dieser wird bei Wiederholung explizit abgewiesen, sofern
    keine Reparatur verlangt wurde. Repariert wird nur die exakte, nicht
    eindeutige Soll-Definition; gueltige oder fremde Indizes bleiben erhalten.
    """
    if not conn.autocommit:
        raise ValueError("Online-Index-Deploy braucht autocommit ausserhalb einer Transaktion")
    lock_name = f"kiara:online-index:{schema}.{_INDEX_NAME}"
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (lock_name,))
        if not cur.fetchone()[0]:
            raise RuntimeError("Ein Online-Deploy dieses Indexes laeuft bereits")
    try:
        existing = _read_index(conn, schema)
        repaired = False
        if existing is not None and not existing["indisvalid"] and repair_invalid:
            _validate_index({**existing, "indisvalid": True, "indisready": True}, schema)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP INDEX CONCURRENTLY {}.{}").format(
                        sql.Identifier(schema), sql.Identifier(_INDEX_NAME)
                    )
                )
            existing = None
            repaired = True
        if existing is not None:
            _validate_index(existing, schema)
            return {"index": f"{schema}.{_INDEX_NAME}", "created": False, "valid": True}
        started = time.monotonic()
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE INDEX CONCURRENTLY {} ON {}.file_hashes "
                    "USING btree (datasource_id, file_path)"
                ).format(sql.Identifier(_INDEX_NAME), sql.Identifier(schema))
            )
        _validate_index(_read_index(conn, schema), schema)
        return {
            "index": f"{schema}.{_INDEX_NAME}",
            "created": True,
            "repaired_invalid": repaired,
            "valid": True,
            "seconds": round(time.monotonic() - started, 3),
        }
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (lock_name,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/etc/kiara/kiara.yaml"))
    parser.add_argument("--repair-invalid", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    conn = psycopg2.connect(
        config["database"]["postgresql"]["dsn"],
        application_name="kiara_online_indexes",
        connect_timeout=5,
        options="-c lock_timeout=30000 -c statement_timeout=120000 "
        "-c max_parallel_maintenance_workers=0",
    )
    conn.autocommit = True
    try:
        print(
            json.dumps(ensure_file_hash_path_index(conn, repair_invalid=args.repair_invalid)),
            flush=True,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
