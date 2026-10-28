"""Prio-2 Tests fuer Repositories mit eigenem SQL (nicht BaseRepository).

Testet:
- ConfigRepository: Singleton-Pattern (id=1), GET/SET/DELETE, Table-Whitelist
- WopiRepository: Token-CRUD, Lock-UPSERT, Expiry-Cleanup
- CacheRepository: Bulk-Refresh (DELETE ALL + INSERT), Filtered Reads
- FileHashRepository: Composite-PK UPSERT, Batch-Delete, get_all_hashes
- ChunkDatasourceRepository: Upsert ON CONFLICT DO NOTHING, Batch-Read
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kiara.repositories.cache import CacheRepository
from kiara.repositories.config import ConfigRepository
from kiara.repositories.file_hash import ChunkDatasourceRepository, FileHashRepository
from kiara.repositories.wopi import WopiRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_repo():
    return ConfigRepository()


@pytest.fixture
def wopi_repo():
    return WopiRepository()


@pytest.fixture
def cache_repo():
    return CacheRepository()


@pytest.fixture
def hash_repo():
    return FileHashRepository()


@pytest.fixture
def chunk_ds_repo():
    return ChunkDatasourceRepository()


def _ds_id(test_db):
    """Erstellt eine Test-Datenquelle und gibt die ID zurueck."""
    cursor = test_db.execute(
        "INSERT INTO datasources (name, type) VALUES (%s, 'local') RETURNING id",
        (f"ds_{uuid.uuid4().hex[:6]}",),
    )
    test_db.commit()
    return cursor.fetchone()["id"]


def _user_id(test_db):
    """Erstellt einen Test-User und gibt die ID zurueck."""
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, 'h') RETURNING id",
        (f"user_{uuid.uuid4().hex[:8]}",),
    )
    test_db.commit()
    return cursor.fetchone()["id"]


# ===========================================================================
# ConfigRepository — Singleton-Pattern
# ===========================================================================


class TestConfigGet:
    def test_get_existing(self, test_db, config_repo):
        config_repo.set(
            "ldap_config",
            {
                "server_url": "ldap://test",
                "bind_dn": "cn=admin",
                "bind_password": "secret",
                "search_base": "dc=test",
            },
        )
        result = config_repo.get("ldap_config")
        assert result is not None
        assert result["server_url"] == "ldap://test"
        assert result["id"] == 1  # Singleton

    def test_get_nonexistent(self, test_db, config_repo):
        config_repo.delete("onlyoffice_config")  # Sicherstellen
        result = config_repo.get("onlyoffice_config")
        assert result is None

    def test_get_invalid_table_raises(self, config_repo):
        with pytest.raises(ValueError, match="Unbekannte Config-Tabelle"):
            config_repo.get("users")  # Nicht in Whitelist


class TestConfigSet:
    def test_set_creates_new(self, test_db, config_repo):
        config_repo.set(
            "onlyoffice_config",
            {
                "document_server_url": "http://office:8080",
                "jwt_secret": "abc123",
            },
        )
        result = config_repo.get("onlyoffice_config")
        assert result is not None
        assert result["document_server_url"] == "http://office:8080"

    def test_set_updates_existing(self, test_db, config_repo):
        config_repo.set(
            "onlyoffice_config",
            {
                "document_server_url": "http://old:8080",
                "jwt_secret": "old",
            },
        )
        config_repo.set(
            "onlyoffice_config",
            {
                "document_server_url": "http://new:9090",
                "jwt_secret": "new",
            },
        )
        result = config_repo.get("onlyoffice_config")
        assert result["document_server_url"] == "http://new:9090"
        assert result["jwt_secret"] == "new"

    def test_set_invalid_table_raises(self, config_repo):
        with pytest.raises(ValueError):
            config_repo.set("sessions", {"token": "hack"})


class TestConfigDelete:
    def test_delete_existing(self, test_db, config_repo):
        config_repo.set(
            "mail_config",
            {
                "smtp_server": "smtp.test",
                "sender_address": "a@b",
                "recipient_addresses": "c@d",
            },
        )
        assert config_repo.delete("mail_config") is True
        assert config_repo.get("mail_config") is None

    def test_delete_nonexistent(self, test_db, config_repo):
        # Sicherstellen dass keine Zeile existiert
        config_repo.delete("mail_config")
        assert config_repo.delete("mail_config") is False


# ===========================================================================
# WopiRepository — Tokens + Locks
# ===========================================================================


class TestWopiTokens:
    def test_create_and_get_token(self, test_db, wopi_repo):
        uid = _user_id(test_db)
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        wopi_repo.create_token("tok_123", "file_abc", uid, expires)
        result = wopi_repo.get_token("tok_123")
        assert result is not None
        assert result["file_id"] == "file_abc"
        assert result["user_id"] == uid

    def test_get_nonexistent_token(self, test_db, wopi_repo):
        assert wopi_repo.get_token("nonexistent") is None

    def test_cleanup_expired_tokens(self, test_db, wopi_repo):
        uid = _user_id(test_db)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        future = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

        tok_old = f"tok_old_{uuid.uuid4().hex[:6]}"
        tok_new = f"tok_new_{uuid.uuid4().hex[:6]}"
        wopi_repo.create_token(tok_old, "file1", uid, past)
        wopi_repo.create_token(tok_new, "file2", uid, future)

        # Verify tokens exist
        assert wopi_repo.get_token(tok_old) is not None
        assert wopi_repo.get_token(tok_new) is not None

        deleted = wopi_repo.cleanup_tokens()
        assert deleted >= 1
        assert wopi_repo.get_token(tok_old) is None
        assert wopi_repo.get_token(tok_new) is not None


class TestWopiLocks:
    def test_set_and_get_lock(self, test_db, wopi_repo):
        uid = _user_id(test_db)
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        wopi_repo.set_lock("doc_1", "lock_abc", uid, expires)

        lock = wopi_repo.get_lock("doc_1")
        assert lock is not None
        assert lock["lock_id"] == "lock_abc"

    def test_upsert_replaces_lock(self, test_db, wopi_repo):
        uid = _user_id(test_db)
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        wopi_repo.set_lock("doc_2", "lock_old", uid, expires)
        wopi_repo.set_lock("doc_2", "lock_new", uid, expires)

        lock = wopi_repo.get_lock("doc_2")
        assert lock["lock_id"] == "lock_new"

    def test_release_lock(self, test_db, wopi_repo):
        uid = _user_id(test_db)
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        wopi_repo.set_lock("doc_3", "lock_x", uid, expires)

        assert wopi_repo.release_lock("doc_3") is True
        assert wopi_repo.get_lock("doc_3") is None

    def test_release_nonexistent(self, test_db, wopi_repo):
        assert wopi_repo.release_lock("nonexistent") is False


# ===========================================================================
# CacheRepository — Bulk-Refresh
# ===========================================================================


class TestCacheDatasources:
    def test_refresh_and_read(self, test_db, cache_repo):
        entries = [
            {"id": 1, "name": "DS-A", "type": "local"},
            {"id": 2, "name": "DS-B", "type": "smb"},
        ]
        cache_repo.refresh_datasources(entries)
        result = cache_repo.get_cached_datasources()
        names = [r["name"] for r in result]
        assert "DS-A" in names
        assert "DS-B" in names

    def test_refresh_replaces_all(self, test_db, cache_repo):
        cache_repo.refresh_datasources([{"id": 1, "name": "Old"}])
        cache_repo.refresh_datasources([{"id": 2, "name": "New"}])
        result = cache_repo.get_cached_datasources()
        names = [r["name"] for r in result]
        assert "Old" not in names
        assert "New" in names

    def test_refresh_empty(self, test_db, cache_repo):
        cache_repo.refresh_datasources([{"id": 1, "name": "X"}])
        cache_repo.refresh_datasources([])
        assert cache_repo.get_cached_datasources() == []


class TestCacheJobs:
    def test_refresh_and_filter(self, test_db, cache_repo):
        entries = [
            {"id": 1, "datasource_id": 10, "status": "pending", "type": "incr"},
            {"id": 2, "datasource_id": 10, "status": "success", "type": "incr"},
            {"id": 3, "datasource_id": 20, "status": "pending", "type": "full"},
        ]
        cache_repo.refresh_jobs(entries)

        # Alle
        all_jobs, total = cache_repo.get_cached_jobs()
        assert total == 3

        # Nach DS filtern
        ds10, total_10 = cache_repo.get_cached_jobs(ds_id=10)
        assert total_10 == 2

        # Nach Status filtern
        pending, total_p = cache_repo.get_cached_jobs(status="pending")
        assert total_p == 2

        # Multi-Status
        multi, total_m = cache_repo.get_cached_jobs(status="pending,success")
        assert total_m == 3

    def test_pagination(self, test_db, cache_repo):
        entries = [{"id": i, "datasource_id": 1, "status": "success"} for i in range(5)]
        cache_repo.refresh_jobs(entries)

        page1, total = cache_repo.get_cached_jobs(limit=2, offset=0)
        assert len(page1) == 2
        assert total == 5

        page2, _ = cache_repo.get_cached_jobs(limit=2, offset=2)
        assert len(page2) == 2


# ===========================================================================
# FileHashRepository — Composite-PK, Batch-Ops
# ===========================================================================


class TestFileHashCRUD:
    def test_upsert_and_get(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/doc.pdf", "abc123")
        result = hash_repo.get_hash(ds_id, "nomic", "/doc.pdf")
        assert result == "abc123"

    def test_upsert_updates_existing(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/doc.pdf", "old")
        hash_repo.upsert_hash(ds_id, "nomic", "/doc.pdf", "new")
        assert hash_repo.get_hash(ds_id, "nomic", "/doc.pdf") == "new"

    def test_get_nonexistent(self, test_db, hash_repo):
        assert hash_repo.get_hash(999, "nomic", "/nope") is None

    def test_delete_hash(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/del.txt", "xxx")
        assert hash_repo.delete_hash(ds_id, "nomic", "/del.txt") is True
        assert hash_repo.get_hash(ds_id, "nomic", "/del.txt") is None

    def test_delete_nonexistent(self, test_db, hash_repo):
        assert hash_repo.delete_hash(999, "nomic", "/nope") is False

    def test_upsert_hash_with_file_size(self, test_db, hash_repo):
        """upsert_hash mit file_size speichert Groesse."""
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/doc.pdf", "abc123", file_size=1024)
        from kiara.shared.db import db_query

        with db_query() as db:
            row = db.execute(
                "SELECT file_size FROM file_hashes WHERE datasource_id = %s AND file_path = %s",
                (ds_id, "/doc.pdf"),
            ).fetchone()
        assert row["file_size"] == 1024

    def test_upsert_hash_negative_size_raises(self, test_db, hash_repo):
        """Negative file_size wirft ValueError."""
        ds_id = _ds_id(test_db)
        with pytest.raises(ValueError, match="negativ"):
            hash_repo.upsert_hash(ds_id, "nomic", "/doc.pdf", "abc123", file_size=-1)

    def test_upsert_hash_cross_model_sync(self, test_db, hash_repo):
        """file_size wird ueber alle Modell-Zeilen synchronisiert."""
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "model_a", "/f.txt", "h1", file_size=500)
        hash_repo.upsert_hash(ds_id, "model_b", "/f.txt", "h2", file_size=700)
        from kiara.shared.db import db_query

        with db_query() as db:
            rows = db.execute(
                "SELECT embedding_model, file_size FROM file_hashes "
                "WHERE datasource_id = %s AND file_path = %s",
                (ds_id, "/f.txt"),
            ).fetchall()
        assert all(r["file_size"] == 700 for r in rows)

    def test_get_paths_missing_size(self, test_db, hash_repo):
        """get_paths_missing_size findet Pfade mit file_size IS NULL."""
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/a.txt", "h1", file_size=100)
        hash_repo.upsert_hash(ds_id, "nomic", "/b.txt", "h2")  # kein file_size
        missing = hash_repo.get_paths_missing_size(ds_id)
        assert "/b.txt" in missing
        assert "/a.txt" not in missing


class TestFileHashBatch:
    def test_get_all_hashes(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/a.txt", "h1")
        hash_repo.upsert_hash(ds_id, "nomic", "/b.txt", "h2")
        hash_repo.upsert_hash(ds_id, "nomic", "/c.txt", "h3")

        all_h = hash_repo.get_all_hashes(ds_id, "nomic")
        assert all_h == {"/a.txt": "h1", "/b.txt": "h2", "/c.txt": "h3"}

    def test_get_all_hashes_isolates_models(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/x.txt", "h1")
        hash_repo.upsert_hash(ds_id, "other", "/x.txt", "h2")

        nomic = hash_repo.get_all_hashes(ds_id, "nomic")
        other = hash_repo.get_all_hashes(ds_id, "other")
        assert nomic["/x.txt"] == "h1"
        assert other["/x.txt"] == "h2"

    def test_list_paginated_uses_stable_pages(self, test_db, hash_repo):
        """Dateipfade werden sortiert, gezaehlt und echte DB-seitig paginiert."""
        ds_id = _ds_id(test_db)
        for index, path in enumerate(
            ["/c.txt", "/a.txt", "/e.txt", "/b.txt", "/d.txt"],
            start=1,
        ):
            hash_repo.upsert_hash(ds_id, "nomic", path, f"h{index}", file_size=index)

        result = hash_repo.list_paginated(ds_id, "nomic", page=2, page_size=2)

        assert [row["file_path"] for row in result["data"]] == ["/c.txt", "/d.txt"]
        assert result["pagination"] == {
            "page": 2,
            "page_size": 2,
            "total_items": 5,
            "total_pages": 3,
        }
        assert result["total_all_items"] == 5

    def test_list_paginated_search_is_literal_and_clamps_page(self, test_db, hash_repo):
        """Prozent/Unterstrich sind Suchtext; zu hohe Seiten landen auf der letzten."""
        ds_id = _ds_id(test_db)
        for index, path in enumerate(
            ["/Invoice_100%.PDF", "/invoice_1000.pdf", "/other.txt"],
            start=1,
        ):
            hash_repo.upsert_hash(ds_id, "nomic", path, f"h{index}")

        result = hash_repo.list_paginated(
            ds_id,
            "nomic",
            page=99,
            page_size=1,
            search="100%",
            direction="desc",
        )

        assert [row["file_path"] for row in result["data"]] == ["/Invoice_100%.PDF"]
        assert result["pagination"]["page"] == 1
        assert result["pagination"]["total_items"] == 1
        assert result["total_all_items"] == 3

    def test_delete_by_paths(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/a.txt", "h1")
        hash_repo.upsert_hash(ds_id, "nomic", "/b.txt", "h2")
        hash_repo.upsert_hash(ds_id, "nomic", "/c.txt", "h3")

        deleted = hash_repo.delete_by_paths(ds_id, "nomic", ["/a.txt", "/b.txt"])
        assert deleted == 2
        assert hash_repo.get_hash(ds_id, "nomic", "/c.txt") == "h3"
        assert hash_repo.get_hash(ds_id, "nomic", "/a.txt") is None

    def test_delete_by_paths_empty(self, test_db, hash_repo):
        assert hash_repo.delete_by_paths(1, "nomic", []) == 0

    def test_delete_by_datasource(self, test_db, hash_repo):
        ds_id = _ds_id(test_db)
        hash_repo.upsert_hash(ds_id, "nomic", "/x.txt", "h1")
        hash_repo.upsert_hash(ds_id, "nomic", "/y.txt", "h2")

        deleted = hash_repo.delete_by_datasource(ds_id)
        assert deleted == 2
        assert hash_repo.get_all_hashes(ds_id, "nomic") == {}


# ===========================================================================
# ChunkDatasourceRepository — M2M Chunk <-> DS
# ===========================================================================


class TestChunkDatasource:
    def test_upsert_and_read(self, test_db, chunk_ds_repo):
        ds_id = _ds_id(test_db)
        # Cleanup: vorherige Testdaten entfernen
        chunk_ds_repo.delete_by_datasource(ds_id)
        chunk_ds_repo.upsert("chunk_001", ds_id, "nomic")
        chunk_ds_repo.upsert("chunk_002", ds_id, "nomic")

        chunks = chunk_ds_repo.get_chunks_for_datasource(ds_id, "nomic")
        assert set(chunks) == {"chunk_001", "chunk_002"}

    def test_upsert_idempotent(self, test_db, chunk_ds_repo):
        ds_id = _ds_id(test_db)
        chunk_ds_repo.upsert("chunk_dup", ds_id, "nomic")
        chunk_ds_repo.upsert("chunk_dup", ds_id, "nomic")  # Duplikat

        chunks = chunk_ds_repo.get_chunks_for_datasource(ds_id, "nomic")
        assert chunks.count("chunk_dup") == 1

    def test_isolates_models(self, test_db, chunk_ds_repo):
        ds_id = _ds_id(test_db)
        chunk_ds_repo.upsert("chunk_x", ds_id, "nomic")
        chunk_ds_repo.upsert("chunk_y", ds_id, "other")

        nomic = chunk_ds_repo.get_chunks_for_datasource(ds_id, "nomic")
        other = chunk_ds_repo.get_chunks_for_datasource(ds_id, "other")
        assert "chunk_x" in nomic and "chunk_y" not in nomic
        assert "chunk_y" in other and "chunk_x" not in other

    def test_delete_by_datasource(self, test_db, chunk_ds_repo):
        ds_id = _ds_id(test_db)
        chunk_ds_repo.upsert("c1", ds_id, "nomic")
        chunk_ds_repo.upsert("c2", ds_id, "nomic")

        deleted = chunk_ds_repo.delete_by_datasource(ds_id)
        assert deleted == 2
        assert chunk_ds_repo.get_chunks_for_datasource(ds_id, "nomic") == []
