"""Prio-1 Tests fuer konkrete Repositories.

Testet domaenenspezifische Methoden mit komplexem SQL:
- IngestionJobRepository: claim_next, cleanup_old
- DatasourceRepository: delete_cascade
- PermissionRepository: get_user_datasource_ids, get_public_datasource_ids
- QueryLogRepository: list_with_joins
- GroupRepository: delete_cascade, upsert_ldap
"""

import uuid

import pytest

from kiara.repositories.audit import QueryLogRepository
from kiara.repositories.chat import ChatRepository, MessageRepository
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.repositories.group import GroupRepository
from kiara.repositories.job import FileLogRepository, IngestionJobRepository
from kiara.repositories.permission import PermissionRepository
from kiara.repositories.user import UserRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def job_repo():
    return IngestionJobRepository()


@pytest.fixture
def file_log_repo():
    return FileLogRepository()


@pytest.fixture
def ds_repo():
    return DatasourceRepository()


@pytest.fixture
def emb_repo():
    return DatasourceEmbeddingRepository()


@pytest.fixture
def perm_repo():
    return PermissionRepository()


@pytest.fixture
def audit_repo():
    return QueryLogRepository()


@pytest.fixture
def group_repo():
    return GroupRepository()


@pytest.fixture
def user_repo():
    return UserRepository()


@pytest.fixture
def chat_repo():
    return ChatRepository()


@pytest.fixture
def msg_repo():
    return MessageRepository()


def _uid(test_db):
    """Erstellt einen eindeutigen Test-User."""
    name = f"repo_test_{uuid.uuid4().hex[:8]}"
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (name, "hash"),
    )
    test_db.commit()
    return cursor.fetchone()["id"]


def _ds(test_db, name="TestDS"):
    """Erstellt eine Test-Datenquelle."""
    cursor = test_db.execute(
        "INSERT INTO datasources (name, type, is_active) VALUES (%s, 'local', true) RETURNING id",
        (f"{name}_{uuid.uuid4().hex[:6]}",),
    )
    test_db.commit()
    return cursor.fetchone()["id"]


def _group(test_db, cn="TestGroup"):
    """Erstellt eine Test-Gruppe mit stabiler SID."""
    dn = f"cn={cn}_{uuid.uuid4().hex[:6]},ou=groups,dc=test"
    sid = f"S-1-5-21-{uuid.uuid4().int % 1_000_000_000}"
    unique_name = f"{cn}_{uuid.uuid4().hex[:6]}"
    cursor = test_db.execute(
        "INSERT INTO groups (sid, name, cn, dn, member_count) "
        "VALUES (%s, %s, %s, %s, 0) RETURNING id",
        (sid, unique_name, cn, dn),
    )
    test_db.commit()
    return cursor.fetchone()["id"]


# ===========================================================================
# IngestionJobRepository
# ===========================================================================


class TestIngestionJobClaim:
    """Tests fuer claim_next() — atomares Job-Claiming mit FOR UPDATE SKIP LOCKED."""

    def _clear_pending(self, test_db):
        """Bereinigt alle pending Jobs aus vorherigen Tests."""
        test_db.execute(
            "UPDATE ingestion_jobs SET status = 'cancelled' "
            "WHERE status IN ('pending', 'initializing')"
        )
        test_db.commit()

    def test_claim_next_returns_job(self, test_db, job_repo):
        self._clear_pending(test_db)
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="pending",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        claimed = job_repo.claim_next()
        assert claimed is not None
        assert claimed["id"] == job_id
        assert claimed["status"] == "initializing"
        assert claimed["started_at"] is not None

    def test_claim_next_empty_queue(self, test_db, job_repo):
        self._clear_pending(test_db)
        claimed = job_repo.claim_next()
        assert claimed is None

    def test_claim_next_skips_running(self, test_db, job_repo):
        self._clear_pending(test_db)
        ds_id = _ds(test_db)
        job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="running",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        claimed = job_repo.claim_next()
        assert claimed is None

    def test_claim_next_fifo_order(self, test_db, job_repo):
        self._clear_pending(test_db)
        ds_id = _ds(test_db)
        id1 = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="pending",
            job_type="incremental",
            triggered_by="manual",
        )
        id2 = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="pending",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        first = job_repo.claim_next()
        assert first["id"] == id1
        second = job_repo.claim_next()
        assert second["id"] == id2


class TestIngestionJobCleanup:
    """Tests fuer cleanup_old() — loescht alte Jobs + FileLogss."""

    def test_cleanup_old_deletes_finished(self, test_db, job_repo, file_log_repo):
        ds_id = _ds(test_db)
        # Alten Job erstellen (finished vor 60 Tagen)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.execute(
            "UPDATE ingestion_jobs SET finished_at = NOW() - INTERVAL '60 days' WHERE id = %s",
            (job_id,),
        )
        # FileLogs zum Job
        file_log_repo.create(
            conn=test_db,
            job_id=job_id,
            file_path="/test.txt",
            file_name="test.txt",
            status="success",
        )
        test_db.commit()

        deleted = job_repo.cleanup_old(days=30)
        assert deleted >= 1
        # Job und FileLogs geloescht
        assert job_repo.get_by_id(job_id) is None

    def test_cleanup_keeps_recent(self, test_db, job_repo):
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.execute(
            "UPDATE ingestion_jobs SET finished_at = NOW() WHERE id = %s",
            (job_id,),
        )
        test_db.commit()

        deleted = job_repo.cleanup_old(days=30)
        assert deleted == 0
        assert job_repo.get_by_id(job_id) is not None

    def test_cleanup_keeps_running(self, test_db, job_repo):
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="running",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        deleted = job_repo.cleanup_old(days=0)
        # Running Job darf NICHT geloescht werden (nur success/error/cancelled)
        assert job_repo.get_by_id(job_id) is not None


class TestIngestionJobActions:
    """Tests fuer request_action und list_paginated."""

    def test_request_action(self, test_db, job_repo):
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="running",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        assert job_repo.request_action(job_id, "pause") is True
        row = job_repo.get_by_id(job_id)
        assert row["requested_action"] == "pause"

        assert job_repo.request_action(job_id, None) is True
        row = job_repo.get_by_id(job_id)
        assert row["requested_action"] is None

    def test_list_paginated_filter_status(self, test_db, job_repo):
        ds_id = _ds(test_db)
        job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="pending",
            job_type="incremental",
            triggered_by="manual",
        )
        job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        rows, total = job_repo.list_paginated(status="pending")
        assert all(r["status"] == "pending" for r in rows)

    def test_list_paginated_multi_status(self, test_db, job_repo):
        ds_id = _ds(test_db)
        job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="pending",
            job_type="incremental",
            triggered_by="manual",
        )
        job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="error",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        rows, total = job_repo.list_paginated(status="pending,success")
        statuses = {r["status"] for r in rows}
        assert "error" not in statuses


# ===========================================================================
# DatasourceRepository
# ===========================================================================


class TestDatasourceEmbeddingStats:
    def test_get_for_datasource_model(self, test_db, emb_repo):
        ds_id = _ds(test_db)
        emb_repo.upsert_stats(ds_id, "nomic", 123, 7)

        result = emb_repo.get_for_datasource_model(ds_id, "nomic")

        assert result["datasource_id"] == ds_id
        assert result["embedding_model"] == "nomic"
        assert result["chunk_count"] == 123
        assert result["file_count"] == 7
        assert emb_repo.get_for_datasource_model(ds_id, "other") is None


class TestDatasourceCascadeDelete:
    """Tests fuer delete_cascade() — loescht DS + alle Abhaengigkeiten."""

    def test_cascade_deletes_all_dependents(self, test_db, ds_repo, emb_repo, perm_repo, job_repo):
        ds_id = _ds(test_db)
        user_id = _uid(test_db)
        group_id = _group(test_db)

        # Abhaengige Daten erstellen
        emb_repo.upsert_stats(ds_id, "nomic", 100, 10, conn=test_db)
        perm_repo.set_datasource_groups(ds_id, [group_id], conn=test_db)
        perm_repo.set_datasource_users(ds_id, [user_id], conn=test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        test_db.commit()

        result = ds_repo.delete_cascade(ds_id)
        assert result["deleted"] is True
        assert result["name"] is not None

        # Alles geloescht
        assert ds_repo.get_by_id(ds_id) is None
        assert perm_repo.get_datasource_group_ids(ds_id) == []
        assert job_repo.get_by_id(job_id) is None

    def test_cascade_nonexistent_ds(self, test_db, ds_repo):
        result = ds_repo.delete_cascade(999999)
        assert result["deleted"] is False

    def test_cascade_deletes_file_logs(self, test_db, ds_repo, job_repo, file_log_repo):
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        file_log_repo.create(
            conn=test_db,
            job_id=job_id,
            file_path="/a.txt",
            file_name="a.txt",
            status="success",
        )
        test_db.commit()

        result = ds_repo.delete_cascade(ds_id)
        assert result["deleted"] is True
        assert result["dependents"]["ingestion_file_logs"] >= 1


# ===========================================================================
# PermissionRepository
# ===========================================================================


class TestPermissionUserDatasources:
    """Tests fuer get_user_datasource_ids() — Kern der Zugriffskontrolle."""

    def test_direct_user_assignment(self, test_db, perm_repo):
        user_id = _uid(test_db)
        ds_id = _ds(test_db)
        perm_repo.set_datasource_users(ds_id, [user_id], conn=test_db)
        test_db.commit()

        ds_ids = perm_repo.get_user_datasource_ids(user_id)
        assert ds_id in ds_ids

    def test_group_assignment(self, test_db, perm_repo):
        user_id = _uid(test_db)
        ds_id = _ds(test_db)
        group_id = _group(test_db)

        # User in Gruppe, Gruppe hat DS-Zugriff
        perm_repo.set_user_groups(user_id, [group_id], conn=test_db)
        perm_repo.set_datasource_groups(ds_id, [group_id], conn=test_db)
        test_db.commit()

        ds_ids = perm_repo.get_user_datasource_ids(user_id)
        assert ds_id in ds_ids

    def test_combined_direct_and_group(self, test_db, perm_repo):
        user_id = _uid(test_db)
        ds1 = _ds(test_db, "Direct")
        ds2 = _ds(test_db, "ViaGroup")
        group_id = _group(test_db)

        perm_repo.set_datasource_users(ds1, [user_id], conn=test_db)
        perm_repo.set_user_groups(user_id, [group_id], conn=test_db)
        perm_repo.set_datasource_groups(ds2, [group_id], conn=test_db)
        test_db.commit()

        ds_ids = perm_repo.get_user_datasource_ids(user_id)
        assert ds1 in ds_ids
        assert ds2 in ds_ids

    def test_no_duplicates(self, test_db, perm_repo):
        """DS sowohl direkt als auch via Gruppe zugewiesen → nur 1x in Liste."""
        user_id = _uid(test_db)
        ds_id = _ds(test_db)
        group_id = _group(test_db)

        perm_repo.set_datasource_users(ds_id, [user_id], conn=test_db)
        perm_repo.set_user_groups(user_id, [group_id], conn=test_db)
        perm_repo.set_datasource_groups(ds_id, [group_id], conn=test_db)
        test_db.commit()

        ds_ids = perm_repo.get_user_datasource_ids(user_id)
        assert ds_ids.count(ds_id) == 1

    def test_user_without_assignments(self, test_db, perm_repo):
        user_id = _uid(test_db)
        ds_ids = perm_repo.get_user_datasource_ids(user_id)
        assert ds_ids == []


class TestPermissionPublicDatasources:
    """Tests fuer get_public_datasource_ids() — DS mit is_public=True oder NULL."""

    def test_unassigned_ds_is_public(self, test_db, perm_repo):
        # DS ohne explizites is_public (NULL) gilt als oeffentlich (COALESCE default)
        ds_id = _ds(test_db)
        test_db.commit()
        public_ids = perm_repo.get_public_datasource_ids()
        assert ds_id in public_ids

    def test_assigned_ds_not_public(self, test_db, perm_repo):
        # is_public=False macht DS explizit nicht-oeffentlich (unabhaengig von Gruppen)
        ds_id = _ds(test_db)
        group_id = _group(test_db)
        perm_repo.set_datasource_groups(ds_id, [group_id], conn=test_db)
        test_db.execute("UPDATE datasources SET is_public = false WHERE id = %s", (ds_id,))
        test_db.commit()

        public_ids = perm_repo.get_public_datasource_ids()
        assert ds_id not in public_ids

    def test_user_assigned_ds_not_public(self, test_db, perm_repo):
        # is_public=False macht DS explizit nicht-oeffentlich (unabhaengig von User-Zuweisungen)
        ds_id = _ds(test_db)
        user_id = _uid(test_db)
        perm_repo.set_datasource_users(ds_id, [user_id], conn=test_db)
        test_db.execute("UPDATE datasources SET is_public = false WHERE id = %s", (ds_id,))
        test_db.commit()

        public_ids = perm_repo.get_public_datasource_ids()
        assert ds_id not in public_ids

    def test_inactive_ds_not_public(self, test_db, ds_repo, perm_repo):
        ds_id = _ds(test_db)
        ds_repo.update(ds_id, conn=test_db, is_active=False)
        test_db.commit()

        public_ids = perm_repo.get_public_datasource_ids()
        assert ds_id not in public_ids


class TestPermissionSetGroups:
    """Tests fuer set_datasource_groups() — M2M-Zuweisung."""

    def test_set_groups_replaces_all(self, test_db, perm_repo):
        ds_id = _ds(test_db)
        g1 = _group(test_db, "G1")
        g2 = _group(test_db, "G2")
        g3 = _group(test_db, "G3")

        perm_repo.set_datasource_groups(ds_id, [g1, g2], conn=test_db)
        test_db.commit()
        assert set(perm_repo.get_datasource_group_ids(ds_id)) == {g1, g2}

        perm_repo.set_datasource_groups(ds_id, [g2, g3], conn=test_db)
        test_db.commit()
        assert set(perm_repo.get_datasource_group_ids(ds_id)) == {g2, g3}

    def test_set_empty_clears_all(self, test_db, perm_repo):
        ds_id = _ds(test_db)
        g1 = _group(test_db, "ClearG")
        perm_repo.set_datasource_groups(ds_id, [g1], conn=test_db)
        test_db.commit()

        perm_repo.set_datasource_groups(ds_id, [], conn=test_db)
        test_db.commit()
        assert perm_repo.get_datasource_group_ids(ds_id) == []

    def test_invalid_group_keeps_existing_assignments(self, test_db, perm_repo):
        ds_id = _ds(test_db)
        g1 = _group(test_db, "KeepG")
        perm_repo.set_datasource_groups(ds_id, [g1], conn=test_db)
        test_db.commit()

        with pytest.raises(ValueError):
            perm_repo.set_datasource_groups(ds_id, [g1, 999999], conn=test_db)

        assert perm_repo.get_datasource_group_ids(ds_id) == [g1]

    def test_invalid_datasource_fails_before_write(self, test_db, perm_repo):
        g1 = _group(test_db, "NoDSG")

        with pytest.raises(ValueError):
            perm_repo.set_datasource_groups(999999, [g1], conn=test_db)

        rows = test_db.execute(
            "SELECT * FROM datasource_groups WHERE group_id = %s", (g1,)
        ).fetchall()
        assert rows == []


class TestPermissionSetUsers:
    """Tests fuer set_datasource_users() — fail-closed bei Referenzfehlern."""

    def test_invalid_user_keeps_existing_assignments(self, test_db, perm_repo):
        ds_id = _ds(test_db)
        user_id = _uid(test_db)
        perm_repo.set_datasource_users(ds_id, [user_id], conn=test_db)
        test_db.commit()

        with pytest.raises(ValueError):
            perm_repo.set_datasource_users(ds_id, [user_id, 999999], conn=test_db)

        rows = test_db.execute(
            "SELECT user_id FROM datasource_users WHERE datasource_id = %s",
            (ds_id,),
        ).fetchall()
        assert [r["user_id"] for r in rows] == [user_id]

    def test_invalid_datasource_fails_before_write(self, test_db, perm_repo):
        user_id = _uid(test_db)

        with pytest.raises(ValueError):
            perm_repo.set_datasource_users(999999, [user_id], conn=test_db)

        rows = test_db.execute(
            "SELECT * FROM datasource_users WHERE user_id = %s", (user_id,)
        ).fetchall()
        assert rows == []


class TestPermissionSetUserGroups:
    """Tests fuer set_user_groups() — fail-closed bei Referenzfehlern."""

    def test_invalid_group_keeps_existing_assignments(self, test_db, perm_repo):
        user_id = _uid(test_db)
        g1 = _group(test_db, "KeepUG")
        perm_repo.set_user_groups(user_id, [g1], conn=test_db)
        test_db.commit()

        with pytest.raises(ValueError):
            perm_repo.set_user_groups(user_id, [g1, 999999], conn=test_db)

        rows = test_db.execute(
            "SELECT group_id FROM user_groups WHERE user_id = %s",
            (user_id,),
        ).fetchall()
        assert [r["group_id"] for r in rows] == [g1]

    def test_invalid_user_fails_before_write(self, test_db, perm_repo):
        g1 = _group(test_db, "NoUserG")

        with pytest.raises(ValueError):
            perm_repo.set_user_groups(999999, [g1], conn=test_db)

        rows = test_db.execute("SELECT * FROM user_groups WHERE group_id = %s", (g1,)).fetchall()
        assert rows == []


# ===========================================================================
# QueryLogRepository
# ===========================================================================


class TestQueryLogWithJoins:
    """Tests fuer list_with_joins() — paginierte Audit-Liste."""

    def _create_audit_entry(
        self, test_db, audit_repo, user_id, routed_to="local", chain_type="standard"
    ):
        return audit_repo.create(
            conn=test_db,
            user_id=user_id,
            question_hash="hash123",
            question_excerpt="Testfrage",
            answer_excerpt="Testantwort",
            routed_to=routed_to,
            chain_type=chain_type,
            duration_ms=100,
        )

    def test_basic_list(self, test_db, audit_repo):
        user_id = _uid(test_db)
        self._create_audit_entry(test_db, audit_repo, user_id)
        test_db.commit()

        result = audit_repo.list_with_joins(page=1, page_size=10)
        assert result["pagination"]["total_items"] >= 1
        assert len(result["data"]) >= 1
        # JOIN liefert username
        assert "username" in result["data"][0]

    def test_filter_by_user(self, test_db, audit_repo):
        u1 = _uid(test_db)
        u2 = _uid(test_db)
        self._create_audit_entry(test_db, audit_repo, u1)
        self._create_audit_entry(test_db, audit_repo, u2)
        test_db.commit()

        result = audit_repo.list_with_joins(user_id=u1)
        assert all(r["user_id"] == u1 for r in result["data"])

    def test_filter_by_route(self, test_db, audit_repo):
        user_id = _uid(test_db)
        self._create_audit_entry(test_db, audit_repo, user_id, routed_to="local")
        self._create_audit_entry(test_db, audit_repo, user_id, routed_to="cloud")
        test_db.commit()

        result = audit_repo.list_with_joins(routed_to="cloud")
        assert all(r["routed_to"] == "cloud" for r in result["data"])

    def test_filter_by_chain_type(self, test_db, audit_repo):
        user_id = _uid(test_db)
        self._create_audit_entry(test_db, audit_repo, user_id, chain_type="agent")
        self._create_audit_entry(test_db, audit_repo, user_id, chain_type="standard")
        test_db.commit()

        result = audit_repo.list_with_joins(chain_type="agent")
        assert all(r["chain_type"] == "agent" for r in result["data"])

    def test_search_text(self, test_db, audit_repo):
        user_id = _uid(test_db)
        audit_repo.create(
            conn=test_db,
            user_id=user_id,
            question_hash="h1",
            question_excerpt="Was kostet Produkt XYZ123?",
            answer_excerpt="Der Preis betraegt...",
            routed_to="local",
            duration_ms=50,
        )
        test_db.commit()

        result = audit_repo.list_with_joins(search_text="XYZ123")
        assert result["pagination"]["total_items"] >= 1

    def test_pagination(self, test_db, audit_repo):
        user_id = _uid(test_db)
        for _i in range(5):
            self._create_audit_entry(test_db, audit_repo, user_id)
        test_db.commit()

        page1 = audit_repo.list_with_joins(user_id=user_id, page=1, page_size=2)
        page2 = audit_repo.list_with_joins(user_id=user_id, page=2, page_size=2)
        assert len(page1["data"]) == 2
        assert len(page2["data"]) == 2
        assert page1["pagination"]["total_pages"] >= 3

    def test_sort_direction(self, test_db, audit_repo):
        user_id = _uid(test_db)
        self._create_audit_entry(test_db, audit_repo, user_id)
        self._create_audit_entry(test_db, audit_repo, user_id)
        test_db.commit()

        asc = audit_repo.list_with_joins(
            user_id=user_id,
            sort_field="id",
            sort_dir="asc",
        )
        desc = audit_repo.list_with_joins(
            user_id=user_id,
            sort_field="id",
            sort_dir="desc",
        )
        if len(asc["data"]) >= 2:
            assert asc["data"][0]["id"] < asc["data"][1]["id"]
            assert desc["data"][0]["id"] > desc["data"][1]["id"]


# ===========================================================================
# UserRepository
# ===========================================================================


class TestUserDeleteCascade:
    """Tests fuer delete_with_sessions() — entfernt User-Daten konsistent."""

    def test_delete_removes_acl_sessions_and_api_keys(self, test_db, user_repo, perm_repo):
        user_id = _uid(test_db)
        ds_id = _ds(test_db)
        group_id = _group(test_db)
        perm_repo.set_datasource_users(ds_id, [user_id], conn=test_db)
        perm_repo.set_user_groups(user_id, [group_id], conn=test_db)
        test_db.execute(
            "INSERT INTO sessions (user_id, token, expires_at) "
            "VALUES (%s, %s, NOW() + interval '1 hour')",
            (user_id, f"token_{uuid.uuid4().hex}"),
        )
        test_db.execute(
            "INSERT INTO api_keys (user_id, name, key_hash, key_prefix) VALUES (%s, %s, %s, %s)",
            (user_id, "test", f"hash_{uuid.uuid4().hex}", "kiara_"),
        )
        test_db.commit()

        assert user_repo.delete_with_sessions(user_id) is True

        for table in ("sessions", "user_groups", "datasource_users", "api_keys"):
            row = test_db.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE user_id = %s",
                (user_id,),
            ).fetchone()
            assert row["count"] == 0
        assert user_repo.get_by_id(user_id) is None


# ===========================================================================
# GroupRepository
# ===========================================================================


class TestGroupCascadeDelete:
    """Tests fuer delete_cascade() — loescht Gruppe + Zuweisungen."""

    def test_cascade_deletes_assignments(self, test_db, group_repo, perm_repo):
        ds_id = _ds(test_db)
        user_id = _uid(test_db)
        group_id = _group(test_db)

        perm_repo.set_datasource_groups(ds_id, [group_id], conn=test_db)
        perm_repo.set_user_groups(user_id, [group_id], conn=test_db)
        test_db.commit()

        assert group_repo.delete_cascade(group_id) is True
        assert group_repo.get_by_id(group_id) is None
        assert perm_repo.get_datasource_group_ids(ds_id) == []

    def test_cascade_nonexistent(self, test_db, group_repo):
        assert group_repo.delete_cascade(999999) is False


class TestGroupUpsertLdap:
    """Tests fuer upsert_ldap() — LDAP-Gruppen importieren/aktualisieren."""

    def test_insert_new(self, test_db, group_repo):
        suffix = uuid.uuid4().hex[:6]
        sid = f"S-1-5-21-{uuid.uuid4().int % 1_000_000_000}"
        cn = f"NewGroup_{suffix}"
        dn = f"cn={cn},ou=groups,dc=test"
        gid = group_repo.upsert_ldap(sid=sid, cn=cn, dn=dn, member_count=5)
        assert gid > 0
        row = group_repo.get_by_id(gid)
        assert row["sid"] == sid
        assert row["cn"] == cn
        assert row["member_count"] == 5

    def test_update_existing(self, test_db, group_repo):
        suffix = uuid.uuid4().hex[:6]
        sid = f"S-1-5-21-{uuid.uuid4().int % 1_000_000_000}"
        cn1 = f"UpdateGroup_{suffix}"
        cn2 = f"UpdateGroup-Renamed_{suffix}"
        dn = f"cn={cn1},ou=groups,dc=test"
        dn2 = f"cn={cn2},ou=groups,dc=test"
        gid1 = group_repo.upsert_ldap(sid=sid, cn=cn1, dn=dn, member_count=3)
        gid2 = group_repo.upsert_ldap(sid=sid, cn=cn2, dn=dn2, member_count=10)
        assert gid1 == gid2
        row = group_repo.get_by_id(gid1)
        assert row["cn"] == cn2
        assert row["dn"] == dn2
        assert row["member_count"] == 10

    def test_same_cn_different_sid_creates_distinct_groups(self, test_db, group_repo):
        suffix = uuid.uuid4().hex[:6]
        cn = f"SharedCN_{suffix}"
        sid1 = f"S-1-5-21-{uuid.uuid4().int % 1_000_000_000}"
        sid2 = f"S-1-5-21-{uuid.uuid4().int % 1_000_000_000}"
        gid1 = group_repo.upsert_ldap(
            sid=sid1,
            cn=cn,
            dn=f"cn={cn},ou=A,dc=test",
            member_count=1,
        )
        gid2 = group_repo.upsert_ldap(
            sid=sid2,
            cn=cn,
            dn=f"cn={cn},ou=B,dc=test",
            member_count=2,
        )

        assert gid1 != gid2
        assert group_repo.get_by_sid(sid1)["id"] == gid1
        assert group_repo.get_by_sid(sid2)["id"] == gid2


# ===========================================================================
# FileLogRepository
# ===========================================================================


class TestFileLogAggregate:
    """Tests fuer aggregate_by_status()."""

    def test_aggregate(self, test_db, file_log_repo, job_repo):
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        file_log_repo.create(
            conn=test_db,
            job_id=job_id,
            file_path="/a.txt",
            file_name="a.txt",
            status="success",
            chunk_count=5,
            duration_ms=100,
        )
        file_log_repo.create(
            conn=test_db,
            job_id=job_id,
            file_path="/b.txt",
            file_name="b.txt",
            status="success",
            chunk_count=3,
            duration_ms=200,
        )
        file_log_repo.create(
            conn=test_db,
            job_id=job_id,
            file_path="/c.txt",
            file_name="c.txt",
            status="error",
            chunk_count=0,
            duration_ms=50,
        )
        test_db.commit()

        agg = file_log_repo.aggregate_by_status(job_id)
        assert "success" in agg
        assert agg["success"]["count"] == 2
        assert agg["error"]["count"] == 1


class TestFileLogSort:
    """Tests fuer list_by_job() Sort-Whitelist."""

    def test_invalid_sort_falls_back(self, test_db, file_log_repo, job_repo):
        ds_id = _ds(test_db)
        job_id = job_repo.create(
            conn=test_db,
            datasource_id=ds_id,
            status="success",
            job_type="incremental",
            triggered_by="manual",
        )
        file_log_repo.create(
            conn=test_db, job_id=job_id, file_path="/x.txt", file_name="x.txt", status="success"
        )
        test_db.commit()

        # SQL-Injection-Versuch via sort → wird auf "file_name" zurueckgesetzt
        rows, total = file_log_repo.list_by_job(
            job_id,
            sort="file_name; DROP TABLE users;--",
            direction="asc",
        )
        assert total >= 1  # Kein Crash, Fallback greift
