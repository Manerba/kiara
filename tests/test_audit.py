"""Tests fuer den AuditLogger (kiara.cloud.audit).

Testet Log-Eintraege schreiben/lesen, Filterung, Statistik-Aggregation,
Budget-Check und DSGVO-Konformitaet (nur anonymisierter Text).
"""

from datetime import datetime, timedelta

import pytest

from kiara.cloud.audit import _COST_PER_1K, AuditLogger, _stats_since_date
from kiara.shared.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AUDIT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS cloud_audit_log (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        question_anonymized TEXT NOT NULL,
        answer_anonymized TEXT,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        entities_anonymized INTEGER DEFAULT 0,
        duration_ms INTEGER DEFAULT 0,
        context_sources TEXT,
        correlation_id TEXT,
        created_at TEXT DEFAULT (NOW()),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE INDEX IF NOT EXISTS idx_cloud_audit_created ON cloud_audit_log(created_at);
    CREATE INDEX IF NOT EXISTS idx_cloud_audit_user ON cloud_audit_log(user_id);
"""


@pytest.fixture
def audit_db():
    """PostgreSQL-DB mit Audit-Schema und Test-User."""
    conn = get_connection()
    # Schema sicherstellen — DROP+CREATE fuer Robustheit gegen Schema-Aenderungen durch andere Tests
    conn.execute("DROP TABLE IF EXISTS cloud_audit_log CASCADE")
    conn.executescript(_AUDIT_SCHEMA)
    conn.commit()

    # Testdaten bereinigen — hohe IDs um FK-Konflikte mit anderen Tests zu vermeiden
    conn.execute(
        "DELETE FROM sessions WHERE user_id IN "
        "(SELECT id FROM users WHERE username IN ('audit_testuser', 'audit_admin'))"
    )
    conn.execute("DELETE FROM users WHERE username IN ('audit_testuser', 'audit_admin')")
    conn.commit()

    # Test-User mit hohen IDs anlegen (keine Kollision mit anderen Tests)
    conn.execute(
        "INSERT INTO users (id, username, password_hash) VALUES (9901, 'audit_testuser', 'x') "
        "ON CONFLICT (id) DO NOTHING"
    )
    conn.execute(
        "INSERT INTO users (id, username, password_hash) VALUES (9902, 'audit_admin', 'x') "
        "ON CONFLICT (id) DO NOTHING"
    )
    conn.commit()

    yield conn, "pg"

    # Aufraeumen
    cleanup = get_connection()
    try:
        cleanup.execute("DELETE FROM cloud_audit_log WHERE user_id IN (9901, 9902)")
        cleanup.execute("DELETE FROM cloud_audit_log WHERE user_id IS NULL")
        cleanup.execute(
            "DELETE FROM sessions WHERE user_id IN "
            "(SELECT id FROM users WHERE username IN ('audit_testuser', 'audit_admin'))"
        )
        cleanup.execute("DELETE FROM users WHERE username IN ('audit_testuser', 'audit_admin')")
        cleanup.commit()
    except Exception:
        cleanup.rollback()
    cleanup.close()

    conn.close()


@pytest.fixture
def audit_config():
    """Standard-Config fuer AuditLogger."""
    return {
        "cloud": {
            "budget": {
                "monthly_limit_eur": 50.0,
                "alert_threshold_pct": 80,
            },
        },
    }


@pytest.fixture
def audit_config_low_budget():
    """Config mit sehr niedrigem Budget fuer Budget-Tests."""
    return {
        "cloud": {
            "budget": {
                "monthly_limit_eur": 0.01,
                "alert_threshold_pct": 80,
            },
        },
    }


def _patch_db(audit_db_fixture):
    """Liefert No-Op-Patches (Repos nutzen dieselbe PostgreSQL-DB direkt).

    Die alten Patches auf kiara.cloud.audit.db_query existieren nicht mehr,
    da cloud/audit.py jetzt ueber _cloud_audit_repo (CloudAuditLogRepository)
    arbeitet. Die Repos nutzen intern kiara.shared.db.db_query, was
    dieselbe PostgreSQL-DB anspricht die auch die Tests verwenden.
    """
    from contextlib import nullcontext

    return nullcontext(), nullcontext()


class TestStatsPeriodBoundaries:
    def test_month_period_uses_calendar_month_start(self):
        """Monatsbudget startet am ersten Kalendertag, nicht vor 30 Tagen."""
        now = datetime(2026, 4, 26, 15, 30)
        assert _stats_since_date("month", now) == "2026-04-01"

    def test_week_and_day_keep_rolling_windows(self):
        """Day/Week behalten die bestehende rollierende Semantik."""
        now = datetime(2026, 4, 26, 15, 30)
        assert _stats_since_date("day", now) == "2026-04-25"
        assert _stats_since_date("week", now) == "2026-04-19"


# ---------------------------------------------------------------------------
# Log-Eintrag schreiben und lesen
# ---------------------------------------------------------------------------


class TestLogRequest:
    def test_log_request_returns_positive_id(self, audit_db, audit_config):
        """log_request gibt eine positive Audit-ID zurueck."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            audit_id = logger.log_request(
                user_id=9901,
                question_anonymized="Was ist [KUNDE_001]?",
                answer_anonymized="[KUNDE_001] ist ein Unternehmen.",
                provider="openai",
                model="gpt-4o-mini",
                tokens_in=100,
                tokens_out=50,
                entities_anonymized=1,
                duration_ms=500,
            )

        assert isinstance(audit_id, int)
        assert audit_id > 0

    def test_log_request_persisted_in_db(self, audit_db, audit_config):
        """Geschriebener Log-Eintrag ist in der DB vorhanden."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            audit_id = logger.log_request(
                user_id=9901,
                question_anonymized="Frage [MITARBEITER_001]",
                answer_anonymized="Antwort zu [MITARBEITER_001]",
                provider="anthropic",
                model="claude-sonnet-4-6",
                tokens_in=200,
                tokens_out=80,
                entities_anonymized=2,
                duration_ms=750,
            )

        # Direkt in DB pruefen
        row = conn.execute("SELECT * FROM cloud_audit_log WHERE id = %s", (audit_id,)).fetchone()
        assert row is not None
        assert row["provider"] == "anthropic"
        assert row["model"] == "claude-sonnet-4-6"
        assert row["tokens_in"] == 200
        assert row["tokens_out"] == 80
        assert row["entities_anonymized"] == 2
        assert row["duration_ms"] == 750

    def test_log_request_stores_only_anonymized_text(self, audit_db, audit_config):
        """NUR anonymisierter Text wird gespeichert, nie Originaltext."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        original_name = "Mueller GmbH"
        anonymized_question = "Wie ist der Umsatz von [KUNDE_001]?"
        anonymized_answer = "[KUNDE_001] hat 5 Mio EUR Umsatz."

        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            audit_id = logger.log_request(
                user_id=9901,
                question_anonymized=anonymized_question,
                answer_anonymized=anonymized_answer,
                provider="openai",
                model="gpt-4o-mini",
                tokens_in=50,
                tokens_out=30,
                entities_anonymized=1,
                duration_ms=300,
            )

        row = conn.execute("SELECT * FROM cloud_audit_log WHERE id = %s", (audit_id,)).fetchone()

        # Anonymisierter Text vorhanden
        assert "[KUNDE_001]" in row["question_anonymized"]
        assert "[KUNDE_001]" in row["answer_anonymized"]
        # Originalname NICHT vorhanden
        assert original_name not in row["question_anonymized"]
        assert original_name not in row["answer_anonymized"]

    def test_log_request_with_correlation_id(self, audit_db, audit_config):
        """log_request mit correlation_id speichert den Wert korrekt."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            audit_id = logger.log_request(
                user_id=9901,
                question_anonymized="Frage mit Korrelation",
                answer_anonymized="Antwort",
                provider="openai",
                model="gpt-4o-mini",
                tokens_in=100,
                tokens_out=50,
                entities_anonymized=1,
                duration_ms=500,
                correlation_id="test-corr-id-123",
            )

        row = conn.execute(
            "SELECT correlation_id FROM cloud_audit_log WHERE id = %s", (audit_id,)
        ).fetchone()
        assert row is not None
        assert row["correlation_id"] == "test-corr-id-123"

    def test_log_request_without_correlation_id(self, audit_db, audit_config):
        """log_request ohne correlation_id laesst Spalte NULL."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            audit_id = logger.log_request(
                user_id=9901,
                question_anonymized="Frage ohne Korrelation",
                answer_anonymized="Antwort",
                provider="openai",
                model="gpt-4o-mini",
                tokens_in=100,
                tokens_out=50,
                entities_anonymized=0,
                duration_ms=300,
            )

        row = conn.execute(
            "SELECT correlation_id FROM cloud_audit_log WHERE id = %s", (audit_id,)
        ).fetchone()
        assert row is not None
        assert row["correlation_id"] is None

    def test_correlation_id_unique_constraint(self, audit_db, audit_config):
        """Doppelte correlation_id verletzt den UNIQUE-Index."""
        conn, db_path = audit_db
        # UNIQUE-Index anlegen (wie in Migration)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_cloud_audit_corr_id "
            "ON cloud_audit_log(correlation_id) WHERE correlation_id IS NOT NULL"
        )
        conn.commit()

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            logger.log_request(
                user_id=9901,
                question_anonymized="Erste Frage",
                answer_anonymized="Erste Antwort",
                provider="openai",
                model="gpt-4o-mini",
                tokens_in=10,
                tokens_out=5,
                entities_anonymized=0,
                duration_ms=100,
                correlation_id="unique-corr-id",
            )
            # Zweiter Eintrag mit gleicher ID → Audit-Persistenz muss fail-closed sein
            with pytest.raises(RuntimeError, match="Cloud-Audit-Log"):
                logger.log_request(
                    user_id=9901,
                    question_anonymized="Zweite Frage",
                    answer_anonymized="Zweite Antwort",
                    provider="openai",
                    model="gpt-4o-mini",
                    tokens_in=10,
                    tokens_out=5,
                    entities_anonymized=0,
                    duration_ms=100,
                    correlation_id="unique-corr-id",
                )

    def test_multiple_log_entries(self, audit_db, audit_config):
        """Mehrere Log-Eintraege werden korrekt gespeichert."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            ids = []
            for i in range(5):
                aid = logger.log_request(
                    user_id=9901,
                    question_anonymized=f"Frage {i}",
                    answer_anonymized=f"Antwort {i}",
                    provider="openai",
                    model="gpt-4o-mini",
                    tokens_in=10 * (i + 1),
                    tokens_out=5 * (i + 1),
                    entities_anonymized=i,
                    duration_ms=100 * (i + 1),
                )
                ids.append(aid)

        assert len(set(ids)) == 5  # Alle IDs einzigartig
        count = conn.execute("SELECT COUNT(*) AS cnt FROM cloud_audit_log").fetchone()["cnt"]
        assert count == 5


# ---------------------------------------------------------------------------
# Filterung (get_log)
# ---------------------------------------------------------------------------


class TestGetLog:
    def _seed_entries(self, conn, db_path, audit_config):
        """Fuellt die DB mit Test-Eintraegen."""
        patch_q, patch_t = _patch_db((conn, db_path))
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            # User 1: 3 Eintraege
            for i in range(3):
                logger.log_request(
                    user_id=9901,
                    question_anonymized=f"User1 Frage {i}",
                    answer_anonymized=f"User1 Antwort {i}",
                    provider="openai",
                    model="gpt-4o-mini",
                    tokens_in=100,
                    tokens_out=50,
                    entities_anonymized=1,
                    duration_ms=300,
                )
            # User 2: 2 Eintraege
            for i in range(2):
                logger.log_request(
                    user_id=9902,
                    question_anonymized=f"User2 Frage {i}",
                    answer_anonymized=f"User2 Antwort {i}",
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    tokens_in=200,
                    tokens_out=80,
                    entities_anonymized=2,
                    duration_ms=500,
                )

    def test_get_log_all_entries(self, audit_db, audit_config):
        """Alle Eintraege abrufen ohne Filter."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            entries = logger.get_log(limit=100)

        assert len(entries) == 5

    def test_get_log_filter_by_user(self, audit_db, audit_config):
        """Filterung nach user_id."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            entries = logger.get_log(user_id=9901)

        assert len(entries) == 3
        assert all(e["user_id"] == 9901 for e in entries)

    def test_get_log_filter_by_user_2(self, audit_db, audit_config):
        """Filterung nach user_id=9902."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            entries = logger.get_log(user_id=9902)

        assert len(entries) == 2
        assert all(e["provider"] == "anthropic" for e in entries)

    def test_get_log_with_limit(self, audit_db, audit_config):
        """Limit begrenzt die Anzahl."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            entries = logger.get_log(limit=2)

        assert len(entries) == 2

    def test_get_log_with_offset(self, audit_db, audit_config):
        """Offset fuer Paginierung."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            page1 = logger.get_log(limit=2, offset=0)
            page2 = logger.get_log(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        # Keine Ueberlappung
        ids_1 = {e["id"] for e in page1}
        ids_2 = {e["id"] for e in page2}
        assert ids_1.isdisjoint(ids_2)

    def test_get_log_filter_by_date(self, audit_db, audit_config):
        """Filterung nach Datum (date_from)."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        # Einen alten Eintrag manuell einfuegen
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """INSERT INTO cloud_audit_log
               (user_id, question_anonymized, answer_anonymized,
                provider, model, tokens_in, tokens_out, created_at)
               VALUES (9901, 'Alte Frage', 'Alte Antwort', 'openai', 'gpt-4o-mini',
                       10, 5, %s)""",
            (old_date,),
        )
        conn.commit()

        # Nur Eintraege von heute
        today = datetime.now().strftime("%Y-%m-%d")
        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            entries = logger.get_log(date_from=today)

        # Der alte Eintrag sollte nicht dabei sein
        assert all(e["created_at"] >= today for e in entries)

    def test_get_log_includes_username(self, audit_db, audit_config):
        """get_log enthaelt den Usernamen ueber JOIN."""
        conn, db_path = audit_db
        self._seed_entries(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            entries = logger.get_log(user_id=9901, limit=1)

        assert len(entries) == 1
        assert entries[0]["username"] == "audit_testuser"


# ---------------------------------------------------------------------------
# Statistik-Aggregation (get_stats)
# ---------------------------------------------------------------------------


class TestGetStats:
    def _seed_for_stats(self, conn, db_path, audit_config):
        """Fuellt die DB mit Eintraegen fuer Statistik-Tests."""
        patch_q, patch_t = _patch_db((conn, db_path))
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            # 3 OpenAI-Anfragen
            for i in range(3):
                logger.log_request(
                    user_id=9901,
                    question_anonymized=f"OpenAI Frage {i}",
                    answer_anonymized=f"OpenAI Antwort {i}",
                    provider="openai",
                    model="gpt-4o-mini",
                    tokens_in=1000,
                    tokens_out=500,
                    entities_anonymized=2,
                    duration_ms=400,
                )
            # 2 Anthropic-Anfragen
            for i in range(2):
                logger.log_request(
                    user_id=9902,
                    question_anonymized=f"Anthropic Frage {i}",
                    answer_anonymized=f"Anthropic Antwort {i}",
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    tokens_in=800,
                    tokens_out=400,
                    entities_anonymized=3,
                    duration_ms=600,
                )

    def test_get_stats_request_count(self, audit_db, audit_config):
        """Korrekte Anzahl Requests."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        assert stats["request_count"] == 5

    def test_get_stats_token_totals(self, audit_db, audit_config):
        """Korrekte Token-Summen."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        # 3 * 1000 + 2 * 800 = 4600
        assert stats["total_tokens_in"] == 4600
        # 3 * 500 + 2 * 400 = 2300
        assert stats["total_tokens_out"] == 2300

    def test_get_stats_by_model(self, audit_db, audit_config):
        """Aufschluesselung nach Provider/Model."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        by_model = {m["model"]: m for m in stats["by_model"]}
        assert "gpt-4o-mini" in by_model
        assert "claude-sonnet-4-6" in by_model
        assert by_model["gpt-4o-mini"]["count"] == 3
        assert by_model["claude-sonnet-4-6"]["count"] == 2

    def test_get_stats_cost_estimation(self, audit_db, audit_config):
        """Kosten-Schaetzung ist positiv und plausibel."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        assert stats["estimated_cost_eur"] > 0
        assert stats["budget_limit_eur"] == 50.0
        assert 0 <= stats["budget_used_pct"] <= 100

    def test_get_stats_period_day(self, audit_db, audit_config):
        """Statistik fuer period='day'."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("day")

        assert stats["period"] == "day"
        assert "since" in stats

    def test_get_stats_period_week(self, audit_db, audit_config):
        """Statistik fuer period='week'."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("week")

        assert stats["period"] == "week"

    def test_get_stats_empty_db(self, audit_db, audit_config):
        """Statistik bei leerer DB."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        assert stats["request_count"] == 0
        assert stats["total_tokens_in"] == 0
        assert stats["estimated_cost_eur"] == 0.0

    def test_get_stats_daily_breakdown(self, audit_db, audit_config):
        """Tages-Aufschluesselung ist vorhanden."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        assert "daily" in stats
        assert isinstance(stats["daily"], list)
        # Alle Eintraege von heute
        if stats["daily"]:
            today = datetime.now().strftime("%Y-%m-%d")
            # PG gibt date-Objekt zurueck, daher str() fuer Vergleich
            assert str(stats["daily"][0]["day"]) == today
            assert stats["daily"][0]["count"] == 5

    def test_get_stats_entities_total(self, audit_db, audit_config):
        """Gesamtzahl anonymisierter Entitaeten."""
        conn, db_path = audit_db
        self._seed_for_stats(conn, db_path, audit_config)

        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            stats = logger.get_stats("month")

        # 3 * 2 + 2 * 3 = 12
        assert stats["total_entities"] == 12


# ---------------------------------------------------------------------------
# Budget-Check
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_budget_ok_with_room(self, audit_db, audit_config):
        """Budget ist OK wenn Kosten unter Limit."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            allowed, remaining = logger.check_budget()

        assert allowed is True
        assert remaining == 50.0  # Keine Ausgaben

    def test_budget_exceeded(self, audit_db, audit_config_low_budget):
        """Budget erschoepft bei hohen Kosten."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)

        with patch_q, patch_t:
            logger = AuditLogger(audit_config_low_budget)
            # Viele Token-Ausgaben erzeugen
            for _ in range(10):
                logger.log_request(
                    user_id=9901,
                    question_anonymized="Frage",
                    answer_anonymized="Antwort",
                    provider="openai",
                    model="gpt-4o",  # Teures Modell
                    tokens_in=10000,
                    tokens_out=5000,
                    entities_anonymized=0,
                    duration_ms=100,
                )

            allowed, remaining = logger.check_budget()

        assert allowed is False
        assert remaining < 0

    def test_budget_returns_correct_remaining(self, audit_db, audit_config):
        """Verbleibendes Budget korrekt berechnet."""
        conn, db_path = audit_db
        patch_q, patch_t = _patch_db(audit_db)
        with patch_q, patch_t:
            logger = AuditLogger(audit_config)
            _, remaining = logger.check_budget()

        assert remaining == 50.0  # 50 EUR Limit, 0 EUR Kosten


# ---------------------------------------------------------------------------
# Kosten-Tabelle
# ---------------------------------------------------------------------------


class TestCostTable:
    def test_known_models_have_costs(self):
        """Alle bekannten Modelle haben Kosteneintraege."""
        known_models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-6"]
        for model in known_models:
            assert model in _COST_PER_1K, f"Kein Kosteneintrag fuer {model}"
            assert "input" in _COST_PER_1K[model]
            assert "output" in _COST_PER_1K[model]

    def test_costs_are_positive(self):
        """Alle Kosten sind positiv."""
        for model, costs in _COST_PER_1K.items():
            assert costs["input"] > 0, f"Input-Kosten fuer {model} nicht positiv"
            assert costs["output"] > 0, f"Output-Kosten fuer {model} nicht positiv"

    def test_output_more_expensive_than_input(self):
        """Output-Tokens sind teurer als Input-Tokens (bei allen Modellen)."""
        for model, costs in _COST_PER_1K.items():
            assert costs["output"] >= costs["input"], f"Output billiger als Input bei {model}"
