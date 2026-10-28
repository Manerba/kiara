"""Tests fuer Build 72: Resume-Chain-Gruppierung.

Testet:
- Migration _migrate_job_resume_chain: Spalte, Idempotenz, Backfill
- NOT IN Filter: Intermediate-Jobs ausgeblendet, nur Blatt-Jobs sichtbar
- _enrich_resume_chains(): Kettenlaenge, kumulierte Dauer, Verlauf
- Einzelner Job ohne Resume: Kettenlaenge 1, kein Verlauf
"""

import json
import sqlite3
from contextlib import contextmanager

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_schema(conn, *, with_resumed_column=True):
    """Erstellt das Basis-Schema fuer ingestion_jobs + datasources.

    Args:
        with_resumed_column: Wenn True, wird resumed_from_job direkt angelegt.
            Wenn False, wird die Spalte weggelassen (fuer Migrations-Tests).
    """
    resumed_col = "resumed_from_job INTEGER DEFAULT NULL," if with_resumed_column else ""
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS datasources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'local',
            is_active BOOLEAN DEFAULT true,
            ingestion_status TEXT DEFAULT 'idle',
            connector_config TEXT DEFAULT '{{}}'
        );
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datasource_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            job_type TEXT NOT NULL DEFAULT 'incremental',
            triggered_by TEXT NOT NULL DEFAULT 'manual',
            embedding_model TEXT DEFAULT '',
            requested_action TEXT DEFAULT NULL,
            progress_processed INTEGER DEFAULT 0,
            progress_total INTEGER DEFAULT 0,
            queued_at TEXT DEFAULT (datetime('now')),
            started_at TEXT DEFAULT NULL,
            finished_at TEXT DEFAULT NULL,
            stats_json TEXT DEFAULT '{{}}',
            error_message TEXT DEFAULT '',
            {resumed_col}
            FOREIGN KEY (datasource_id) REFERENCES datasources(id) ON DELETE CASCADE
        );
        INSERT INTO datasources (name, type) VALUES ('Test DS', 'local');
    """)
    conn.commit()


@pytest.fixture
def db_with_column(tmp_path):
    """DB mit resumed_from_job Spalte (fuer API/Enrich-Tests)."""
    db_path = str(tmp_path / "kiara.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    _create_schema(conn, with_resumed_column=True)
    conn.close()
    return db_path


@pytest.fixture
def db_without_column(tmp_path):
    """DB ohne resumed_from_job Spalte (fuer Migrations-Tests)."""
    db_path = str(tmp_path / "kiara.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    _create_schema(conn, with_resumed_column=False)
    conn.close()
    return db_path


def _make_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _patch_db(monkeypatch, db_path):
    """Patcht db_query/db_transaction in allen relevanten Modulen."""
    import kiara.shared.db as shared_db
    import kiara.web.db_utils as db_utils

    @contextmanager
    def fake_db_query():
        conn = _make_conn(db_path)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def fake_db_transaction():
        conn = _make_conn(db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_utils, "db_query", fake_db_query)
    monkeypatch.setattr(db_utils, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(shared_db, "db_query", fake_db_query)
    monkeypatch.setattr(shared_db, "db_transaction", fake_db_transaction)

    import kiara.web.ingestion_worker as worker_mod

    monkeypatch.setattr(worker_mod, "db_query", fake_db_query)
    monkeypatch.setattr(worker_mod, "db_transaction", fake_db_transaction)

    # BaseRepository loest die Connection-Helper in kiara_db.base auf.
    import kiara_db.base as repo_base

    monkeypatch.setattr(repo_base, "db_query", fake_db_query)
    monkeypatch.setattr(repo_base, "db_transaction", fake_db_transaction)


# ---------------------------------------------------------------------------
# NOT IN Filter: Intermediate-Jobs ausblenden
# ---------------------------------------------------------------------------


class TestNotInFilter:
    """Testet, dass der NOT IN Filter in der Job-Liste Intermediate-Jobs ausblendet."""

    def test_intermediate_jobs_hidden(self, db_with_column, monkeypatch):
        """Jobs, auf die ein Nachfolger via resumed_from_job verweist, werden ausgeblendet."""
        _patch_db(monkeypatch, db_with_column)

        conn = _make_conn(db_with_column)
        # Job 1: Original → wird zum Intermediate
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (1, 'error', '2026-01-01 10:00:00', '2026-01-01 10:05:00', "
            "'{\"duration_seconds\": 300}')"
        )
        # Job 2: Resumed von 1 → Blatt-Job
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, resumed_from_job, "
            "started_at, finished_at, stats_json) "
            "VALUES (1, 'success', 1, '2026-01-01 10:10:00', '2026-01-01 10:20:00', "
            "'{\"duration_seconds\": 600}')"
        )
        conn.commit()
        conn.close()

        from kiara.shared.db import db_query

        # SQL-Query aus dem Endpoint nachbauen
        chain_filter = (
            "j.id NOT IN ("
            "SELECT resumed_from_job FROM ingestion_jobs "
            "WHERE resumed_from_job IS NOT NULL)"
        )
        query = (
            "SELECT j.*, d.name AS datasource_name "
            "FROM ingestion_jobs j "
            "LEFT JOIN datasources d ON j.datasource_id = d.id "
            f"WHERE {chain_filter} ORDER BY j.id DESC"
        )

        with db_query() as db:
            rows = db.execute(query).fetchall()
            jobs = [dict(r) for r in rows]

        # Nur Job 2 soll sichtbar sein (Blatt)
        assert len(jobs) == 1
        assert jobs[0]["id"] == 2

    def test_leaf_jobs_without_chain_visible(self, db_with_column, monkeypatch):
        """Jobs ohne Resume-Kette (resumed_from_job IS NULL) bleiben sichtbar."""
        _patch_db(monkeypatch, db_with_column)

        conn = _make_conn(db_with_column)
        conn.execute("INSERT INTO ingestion_jobs (datasource_id, status) VALUES (1, 'success')")
        conn.execute("INSERT INTO ingestion_jobs (datasource_id, status) VALUES (1, 'error')")
        conn.commit()
        conn.close()

        from kiara.shared.db import db_query

        chain_filter = (
            "j.id NOT IN ("
            "SELECT resumed_from_job FROM ingestion_jobs "
            "WHERE resumed_from_job IS NOT NULL)"
        )
        query = f"SELECT j.id FROM ingestion_jobs j WHERE {chain_filter} ORDER BY j.id DESC"

        with db_query() as db:
            rows = db.execute(query).fetchall()

        assert len(rows) == 2

    def test_three_level_chain_only_leaf_visible(self, db_with_column, monkeypatch):
        """Bei einer 3-stufigen Kette (A→B→C) ist nur C sichtbar."""
        _patch_db(monkeypatch, db_with_column)

        conn = _make_conn(db_with_column)
        # Job A (Original)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, stats_json) "
            "VALUES (1, 'error', '{\"duration_seconds\": 100}')"
        )
        # Job B (Resumed von A)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, resumed_from_job, stats_json) "
            "VALUES (1, 'error', 1, '{\"duration_seconds\": 200}')"
        )
        # Job C (Resumed von B) — Blatt
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, resumed_from_job, stats_json) "
            "VALUES (1, 'success', 2, '{\"duration_seconds\": 300}')"
        )
        conn.commit()
        conn.close()

        from kiara.shared.db import db_query

        chain_filter = (
            "j.id NOT IN ("
            "SELECT resumed_from_job FROM ingestion_jobs "
            "WHERE resumed_from_job IS NOT NULL)"
        )
        query = f"SELECT j.id FROM ingestion_jobs j WHERE {chain_filter} ORDER BY j.id DESC"

        with db_query() as db:
            rows = db.execute(query).fetchall()

        assert len(rows) == 1
        assert rows[0]["id"] == 3


# ---------------------------------------------------------------------------
# PostgreSQL-Fixture fuer _enrich_resume_chains und Worker-Tests
# (SQLite kann %s-Placeholder nicht verarbeiten)
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_enrich_db():
    """PostgreSQL-Test-DB mit dem vollstaendigen aktuellen Schema."""
    from db_helpers import restore_test_schema

    from kiara.shared.db import get_connection

    # Der Helper verweigert destructive Resets ausserhalb von ``kiara_test``.
    # Damit enthalten Job-Fixtures auch alle aktuellen Contract-/Evidence-Felder.
    restore_test_schema()
    conn = get_connection()
    ds_id = conn.execute(
        "INSERT INTO datasources (name, type) VALUES (%s, %s) RETURNING id",
        ("Test DS", "local"),
    ).fetchone()["id"]
    conn.commit()
    conn.close()

    yield ds_id

    # Cleanup — Testdaten loeschen
    cleanup = get_connection()
    cleanup.execute("DELETE FROM ingestion_jobs")
    cleanup.execute("DELETE FROM datasources")
    cleanup.commit()
    cleanup.close()


def _insert_running_job_with_contract(conn, datasource_id: int) -> int:
    """Erzeugt einen laufenden Job mit vollstaendigem aktuellem S4-Snapshot."""
    from kiara.retrieval.index_compatibility import dense_index_target

    model = "nomic-embed-text"
    target = dense_index_target({}, model)
    return conn.execute(
        "INSERT INTO ingestion_jobs ("
        "datasource_id, status, embedding_model, contract_logical_index_key, "
        "contract_index_compatibility_id, contract_query_compatibility_id, "
        "contract_canonical_model_id, contract_profile_id, contract_profile_kind, "
        "contract_profile_endpoint, contract_snapshot_sha256, contract_fence_version"
        ") VALUES (%s, 'running', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "RETURNING id",
        (
            datasource_id,
            model,
            target.logical_index_key,
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            model,
            "s4-test-profile",
            "dense",
            "/api/embed",
            "3" * 64,
            1,
        ),
    ).fetchone()["id"]


def _pg_conn():
    """Erstellt eine PostgreSQL-Verbindung fuer Tests."""
    from kiara.shared.db import get_connection

    return get_connection()


# ---------------------------------------------------------------------------
# _enrich_resume_chains
# ---------------------------------------------------------------------------


class TestEnrichResumeChains:
    """Testet die Anreicherung von Jobs mit Resume-Ketten-Metadaten."""

    def test_chain_length_two_jobs(self, pg_enrich_db):
        """Kette aus 2 Jobs: chain_length == 2."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        # Job 1: Original
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                ds_id,
                "error",
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
                '{"duration_seconds": 300}',
            ),
        )
        conn.commit()
        job1_id = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Job 2: Resumed von 1
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, resumed_from_job, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                ds_id,
                "success",
                job1_id,
                "2026-01-01 10:10:00",
                "2026-01-01 10:15:00",
                '{"duration_seconds": 300}',
            ),
        )
        conn.commit()
        job2_id = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]

        job = dict(
            conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job2_id,)).fetchone()
        )

        _enrich_resume_chains([job], conn)
        conn.close()

        assert job["resume_chain_length"] == 2
        assert job["resume_chain_original_job_id"] == job1_id

    def test_chain_length_three_jobs(self, pg_enrich_db):
        """Kette aus 3 Jobs (A→B→C): chain_length == 3."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        ids = []
        for status, stats in [
            ("error", '{"duration_seconds": 100}'),
            ("error", '{"duration_seconds": 200}'),
            ("success", '{"duration_seconds": 300}'),
        ]:
            resumed = ids[-1] if ids else None
            conn.execute(
                "INSERT INTO ingestion_jobs "
                "(datasource_id, status, resumed_from_job, started_at, finished_at, stats_json) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (ds_id, status, resumed, "2026-01-01 10:00:00", "2026-01-01 10:05:00", stats),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
                (ds_id,),
            ).fetchone()
            ids.append(row["id"])

        job = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (ids[2],)).fetchone())

        _enrich_resume_chains([job], conn)
        conn.close()

        assert job["resume_chain_length"] == 3
        assert job["resume_chain_original_job_id"] == ids[0]

    def test_cumulative_duration(self, pg_enrich_db):
        """Kumulierte Dauer summiert duration_seconds aller Vorfahren."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        ids = []
        for status, stats, _resumed in [
            ("error", '{"duration_seconds": 120}', None),
            ("error", '{"duration_seconds": 180}', 0),  # placeholder
            ("success", '{"duration_seconds": 300}', 0),  # placeholder
        ]:
            rfj = ids[-1] if ids else None
            conn.execute(
                "INSERT INTO ingestion_jobs "
                "(datasource_id, status, resumed_from_job, started_at, finished_at, stats_json) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (ds_id, status, rfj, "2026-01-01 10:00:00", "2026-01-01 10:05:00", stats),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
                (ds_id,),
            ).fetchone()
            ids.append(row["id"])

        job = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (ids[2],)).fetchone())

        _enrich_resume_chains([job], conn)
        conn.close()

        # ancestor_duration = Job1 (120) + Job2 (180) = 300
        assert job["resume_chain_ancestor_duration"] == 300

    def test_chain_history_contains_all_ancestors(self, pg_enrich_db):
        """resume_chain enthaelt alle Vorfahren-Jobs in Reihenfolge (neuester zuerst)."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        # Job 1
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                ds_id,
                "error",
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
                '{"duration_seconds": 100}',
            ),
        )
        conn.commit()
        id1 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Job 2 — resumed von 1
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, resumed_from_job, started_at, finished_at, "
            "stats_json, error_message) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                ds_id,
                "error",
                id1,
                "2026-01-01 10:10:00",
                "2026-01-01 10:15:00",
                '{"duration_seconds": 200}',
                "Connection lost",
            ),
        )
        conn.commit()
        id2 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Job 3 — resumed von 2 (Blatt)
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, resumed_from_job, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                ds_id,
                "success",
                id2,
                "2026-01-01 10:20:00",
                "2026-01-01 10:30:00",
                '{"duration_seconds": 300}',
            ),
        )
        conn.commit()
        id3 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]

        job = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (id3,)).fetchone())

        _enrich_resume_chains([job], conn)
        conn.close()

        chain = job["resume_chain"]
        assert len(chain) == 2

        # Erster Eintrag: Job 2 (direkter Vorgaenger)
        assert chain[0]["id"] == id2
        assert chain[0]["status"] == "error"
        assert chain[0]["duration_seconds"] == 200
        assert chain[0]["error_message"] == "Connection lost"

        # Zweiter Eintrag: Job 1 (Original)
        assert chain[1]["id"] == id1
        assert chain[1]["status"] == "error"
        assert chain[1]["duration_seconds"] == 100

    def test_single_job_without_resume_no_chain(self, pg_enrich_db):
        """Job ohne resumed_from_job: Kein resume_chain_length, kein Verlauf."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, stats_json) VALUES (%s, %s, %s)",
            (ds_id, "success", '{"duration_seconds": 60}'),
        )
        conn.commit()
        job_id = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]

        job = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone())

        _enrich_resume_chains([job], conn)
        conn.close()

        # Keine Chain-Felder vorhanden
        assert "resume_chain_length" not in job
        assert "resume_chain" not in job
        assert "resume_chain_original_job_id" not in job
        assert "resume_chain_ancestor_duration" not in job

    def test_enrich_multiple_jobs_batch(self, pg_enrich_db):
        """_enrich_resume_chains kann mehrere Jobs gleichzeitig anreichern."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        # Kette 1: Job1 → Job2
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                ds_id,
                "error",
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
                '{"duration_seconds": 100}',
            ),
        )
        conn.commit()
        id1 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, resumed_from_job, stats_json) "
            "VALUES (%s, %s, %s, %s)",
            (ds_id, "success", id1, '{"duration_seconds": 200}'),
        )
        conn.commit()
        id2 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s AND status = 'success' ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Standalone Job 3
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, stats_json) VALUES (%s, %s, %s)",
            (ds_id, "success", '{"duration_seconds": 50}'),
        )
        conn.commit()
        id3 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]

        job2 = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (id2,)).fetchone())
        job3 = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (id3,)).fetchone())

        _enrich_resume_chains([job2, job3], conn)
        conn.close()

        # Job 2 hat Chain-Metadaten
        assert job2["resume_chain_length"] == 2
        assert job2["resume_chain_original_job_id"] == id1

        # Job 3 hat keine Chain-Metadaten
        assert "resume_chain_length" not in job3

    def test_ancestor_with_missing_duration_defaults_to_zero(self, pg_enrich_db):
        """Vorfahren ohne duration_seconds in stats_json zaehlen als 0."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        # Job 1: Keine duration_seconds
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s)",
            (ds_id, "error", "2026-01-01 10:00:00", "2026-01-01 10:05:00", '{"total_files": 5}'),
        )
        conn.commit()
        id1 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Job 2: resumed von 1, hat duration
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, resumed_from_job, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                ds_id,
                "success",
                id1,
                "2026-01-01 10:10:00",
                "2026-01-01 10:15:00",
                '{"duration_seconds": 300}',
            ),
        )
        conn.commit()
        id2 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]

        job = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (id2,)).fetchone())

        _enrich_resume_chains([job], conn)
        conn.close()

        # ancestor_duration = Job1 (0, da kein duration_seconds) = 0
        assert job["resume_chain_ancestor_duration"] == 0

    def test_circular_reference_protection(self, pg_enrich_db):
        """Zirkulaere Referenz (Job A → B → A) fuehrt nicht zu Endlosschleife."""
        from kiara.web.api_ingestion_jobs import _enrich_resume_chains

        ds_id = pg_enrich_db
        conn = _pg_conn()
        # Job 1: erstmal ohne resumed_from_job
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                ds_id,
                "error",
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
                '{"duration_seconds": 100}',
            ),
        )
        conn.commit()
        id1 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Job 2: resumed_from_job = id1
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, resumed_from_job, started_at, finished_at, stats_json) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                ds_id,
                "error",
                id1,
                "2026-01-01 10:10:00",
                "2026-01-01 10:15:00",
                '{"duration_seconds": 200}',
            ),
        )
        conn.commit()
        id2 = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()["id"]
        # Job 1 zirkulaer auf Job 2 verweisen
        conn.execute(
            "UPDATE ingestion_jobs SET resumed_from_job = %s WHERE id = %s",
            (id2, id1),
        )
        conn.commit()

        job = dict(conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (id2,)).fetchone())

        # Darf nicht haengen bleiben
        _enrich_resume_chains([job], conn)
        conn.close()

        # Chain existiert, aber ist begrenzt (seen-Set verhindert Endlosschleife)
        assert "resume_chain" in job
        assert len(job["resume_chain"]) >= 1


# ---------------------------------------------------------------------------
# Worker: resumed_from_job beim Auto-Resume
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("s4_guard_passthrough")
class TestWorkerResume:
    """Testet, dass der Worker beim Auto-Resume die resumed_from_job Spalte setzt."""

    def test_recover_stale_sets_resumed_from_job(self, pg_enrich_db, monkeypatch):
        """_recover_stale_jobs erstellt Resume-Job mit resumed_from_job Spalte."""
        ds_id = pg_enrich_db

        # Sicherstellen dass db_query/db_transaction die echte PostgreSQL-Version ist
        # (vorherige Tests patchen diese auf SQLite, monkeypatch-undo reicht nicht immer)
        import kiara_db.base as rb

        import kiara.shared.db as sdb
        import kiara.web.ingestion_worker as wm

        monkeypatch.setattr(wm, "db_query", sdb.db_query)
        monkeypatch.setattr(wm, "db_transaction", sdb.db_transaction)
        monkeypatch.setattr(rb, "db_query", sdb.db_query)
        monkeypatch.setattr(rb, "db_transaction", sdb.db_transaction)

        conn = _pg_conn()
        # Laufenden Job simulieren
        orig_id = _insert_running_job_with_contract(conn, ds_id)
        conn.execute(
            "UPDATE datasources SET ingestion_status = %s WHERE id = %s",
            ("running", ds_id),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = _pg_conn()

        # Original-Job wurde als error markiert
        orig = conn.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (orig_id,)).fetchone()
        assert orig["status"] == "error"

        # Resume-Job wurde erstellt
        resume = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND resumed_from_job = %s",
            (ds_id, orig_id),
        ).fetchone()
        assert resume is not None
        assert resume["status"] == "pending"
        assert resume["datasource_id"] == ds_id

        # stats_json spiegelt den Ursprung fuer die Resume-Kettenanzeige.
        # jsonb-Spalte: psycopg2 liefert dict direkt, sonst string parsen.
        stats_raw = resume["stats_json"]
        stats = stats_raw if isinstance(stats_raw, dict) else json.loads(stats_raw or "{}")
        assert stats["resumed_from_job"] == orig_id

        conn.close()

    def test_recover_stale_no_duplicate_resume(self, pg_enrich_db, monkeypatch):
        """Wenn bereits ein pending Job existiert, wird kein Resume erstellt."""
        ds_id = pg_enrich_db

        # Sicherstellen dass db_transaction die echte PostgreSQL-Version ist
        import kiara_db.base as rb

        import kiara.shared.db as sdb
        import kiara.web.ingestion_worker as wm

        monkeypatch.setattr(wm, "db_query", sdb.db_query)
        monkeypatch.setattr(wm, "db_transaction", sdb.db_transaction)
        monkeypatch.setattr(rb, "db_query", sdb.db_query)
        monkeypatch.setattr(rb, "db_transaction", sdb.db_transaction)

        conn = _pg_conn()
        # Laufender Job
        _insert_running_job_with_contract(conn, ds_id)
        # Bereits existierender pending Job
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)",
            (ds_id, "pending"),
        )
        conn.execute(
            "UPDATE datasources SET ingestion_status = %s WHERE id = %s",
            ("running", ds_id),
        )
        conn.commit()

        # Zaehle Jobs vorher
        before_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM ingestion_jobs WHERE datasource_id = %s",
            (ds_id,),
        ).fetchone()["cnt"]
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = _pg_conn()
        after_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM ingestion_jobs WHERE datasource_id = %s",
            (ds_id,),
        ).fetchone()["cnt"]
        conn.close()

        # Kein zusaetzlicher Job — Resume wurde uebersprungen
        assert after_count == before_count
