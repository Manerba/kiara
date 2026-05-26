"""KIara - Datenbank.

Schema und Initialisierung. Verbindungsmanagement liegt in :mod:`kiara.shared.db`.

- Schema: ``schema.sql`` (pg_dump-basierte Source-of-Truth).
- Seeds: ``db_seeding.py`` (Default-Admin, ai_config, lokales DDU-Backend).

Keine Migrations-Kette — Alpha-Phase arbeitet auf frischem Schema, das
schema.sql definiert. Siehe ``feedback_no_backward_compat.md``.
"""

import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

# Canonical source fuer get_db ist kiara.shared.db
from kiara.shared.db import get_db

_SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


def init_db() -> None:
    """Erstellt Tabellen (bei leerer DB) und seed-Daten beim App-Startup.

    Schema-Apply ist idempotent per Tabellen-Count-Check — nur wenn die DB
    keine ``public``-Tabellen hat, wird ``schema.sql`` ausgefuehrt. Seeding
    laeuft immer (eigene Idempotenz-Checks in den Seed-Funktionen).
    """
    from kiara.web.db_seeding import run_seeds

    conn = get_db()
    try:
        # Advisory Lock: Nur ein Worker initialisiert (verhindert Race-Conditions).
        conn.execute("SELECT pg_advisory_lock(42)")

        # Schema nur bei leerer DB anwenden.
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchone()
        if row["cnt"] == 0:
            _logger.info("Leere DB erkannt — schema.sql anwenden")
            schema_sql = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
            conn.execute(schema_sql)

        from kiara.web.core_schema_migration import migrate_core_schema
        from kiara.web.graph_schema_migration import migrate_graph_schema
        from kiara.web.group_schema_migration import migrate_group_schema

        migrate_group_schema(conn)
        migrate_graph_schema(conn)
        migrate_core_schema(conn)

        # Seeds (idempotent).
        run_seeds(conn)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            conn.execute("SELECT pg_advisory_unlock(42)")
            conn.commit()
        except Exception as exc:
            _logger.warning("Advisory-Lock-Freigabe fehlgeschlagen: %s", exc)
        conn.close()
