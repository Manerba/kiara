"""Tests fuer BaseRepository — Phase 1 des Repository-Patterns.

Testet alle CRUD-Operationen, Filter-Operatoren, RawSQL-Marker,
Option-B-Connection-Durchreichung und Edge Cases.

Laeuft gegen PostgreSQL (kiara_test DB via conftest.py).
"""

import pytest

from kiara.repositories.base import BaseRepository, RawSQL

# ---------------------------------------------------------------------------
# Test-Repository (Concrete Subclass)
# ---------------------------------------------------------------------------


class ChatTestRepo(BaseRepository):
    table = "chats"
    pk = "id"
    default_sort = "id DESC"


class MessageTestRepo(BaseRepository):
    table = "chat_messages"
    pk = "id"
    default_sort = "id ASC"


class DatasourceTestRepo(BaseRepository):
    table = "datasources"
    pk = "id"
    default_sort = "id ASC"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_repo():
    return ChatTestRepo()


@pytest.fixture
def msg_repo():
    return MessageTestRepo()


@pytest.fixture
def ds_repo():
    return DatasourceTestRepo()


@pytest.fixture
def user_id(test_db):
    """Erstellt einen Test-User und gibt dessen ID zurueck."""
    import uuid

    unique_name = f"testrepo_{uuid.uuid4().hex[:8]}"
    cursor = test_db.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
        (unique_name, "hash123"),
    )
    uid = cursor.fetchone()["id"]
    test_db.commit()
    return uid


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_returns_id(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(
            conn=test_db,
            user_id=user_id,
            title="Erster Chat",
        )
        assert isinstance(chat_id, int)
        assert chat_id > 0

    def test_create_with_raw_sql(self, test_db, ds_repo):
        ds_id = ds_repo.create(
            conn=test_db,
            name="TestDS",
            type="local",
            updated_at=RawSQL("NOW()"),
        )
        row = ds_repo.get_by_id(ds_id, conn=test_db)
        assert row is not None
        assert row["name"] == "TestDS"
        # updated_at sollte gesetzt sein (nicht None)
        assert row["updated_at"] is not None

    def test_create_multiple(self, test_db, chat_repo, user_id):
        id1 = chat_repo.create(conn=test_db, user_id=user_id, title="Chat A")
        id2 = chat_repo.create(conn=test_db, user_id=user_id, title="Chat B")
        assert id1 != id2

    def test_create_rejects_invalid_column(self, test_db, ds_repo):
        with pytest.raises(ValueError, match="Ungueltiger Spaltenname"):
            ds_repo.create(conn=test_db, **{"name) VALUES ('x');--": "bad"})


# ---------------------------------------------------------------------------
# get_by_id()
# ---------------------------------------------------------------------------


class TestGetById:
    def test_get_existing(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id, title="FindMe")
        result = chat_repo.get_by_id(chat_id, conn=test_db)
        assert result is not None
        assert result["title"] == "FindMe"
        assert result["user_id"] == user_id

    def test_get_nonexistent(self, test_db, chat_repo):
        result = chat_repo.get_by_id(999999, conn=test_db)
        assert result is None


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


class TestExists:
    def test_exists_true(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id, title="Exists")
        assert chat_repo.exists(chat_id, conn=test_db) is True

    def test_exists_false(self, test_db, chat_repo):
        assert chat_repo.exists(999999, conn=test_db) is False


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_fields(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id, title="Alt")
        success = chat_repo.update(chat_id, conn=test_db, title="Neu")
        assert success is True
        row = chat_repo.get_by_id(chat_id, conn=test_db)
        assert row["title"] == "Neu"

    def test_update_nonexistent(self, test_db, chat_repo):
        success = chat_repo.update(999999, conn=test_db, title="Nope")
        assert success is False

    def test_update_no_fields(self, test_db, chat_repo):
        success = chat_repo.update(1, conn=test_db)
        assert success is False

    def test_update_with_raw_sql(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id, title="Raw")
        success = chat_repo.update(
            chat_id,
            conn=test_db,
            status="archived",
            updated_at=RawSQL("NOW()"),
        )
        assert success is True
        row = chat_repo.get_by_id(chat_id, conn=test_db)
        assert row["status"] == "archived"

    def test_update_rejects_invalid_column(self, test_db, chat_repo):
        with pytest.raises(ValueError, match="Ungueltiger Spaltenname"):
            chat_repo.update(1, conn=test_db, **{"title = title; DROP TABLE chats;--": "bad"})

    def test_update_where_rejects_invalid_column(self, test_db, chat_repo):
        with pytest.raises(ValueError, match="Ungueltiger Spaltenname"):
            chat_repo.update_where(
                {"id": 1}, conn=test_db, **{"status; DROP TABLE chats;--": "bad"}
            )


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing(self, test_db, chat_repo, user_id):
        chat_id = chat_repo.create(conn=test_db, user_id=user_id, title="Bye")
        assert chat_repo.delete(chat_id, conn=test_db) is True
        assert chat_repo.get_by_id(chat_id, conn=test_db) is None

    def test_delete_nonexistent(self, test_db, chat_repo):
        assert chat_repo.delete(999999, conn=test_db) is False


# ---------------------------------------------------------------------------
# delete_where()
# ---------------------------------------------------------------------------


class TestDeleteWhere:
    def test_delete_where(self, test_db, chat_repo, user_id):
        chat_repo.create(conn=test_db, user_id=user_id, title="A", status="active")
        chat_repo.create(conn=test_db, user_id=user_id, title="B", status="archived")
        chat_repo.create(conn=test_db, user_id=user_id, title="C", status="archived")
        deleted = chat_repo.delete_where(
            {"status": "archived", "user_id": user_id},
            conn=test_db,
        )
        assert deleted == 2

    def test_delete_where_empty_filter_raises(self, test_db, chat_repo):
        with pytest.raises(ValueError, match="mindestens einen Filter"):
            chat_repo.delete_where({}, conn=test_db)


# ---------------------------------------------------------------------------
# list_all()
# ---------------------------------------------------------------------------


class TestListAll:
    def test_list_all(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="DS1", type="local")
        ds_repo.create(conn=test_db, name="DS2", type="smb")
        result = ds_repo.list_all(conn=test_db)
        names = [r["name"] for r in result]
        assert "DS1" in names
        assert "DS2" in names

    def test_list_all_with_limit(self, test_db, ds_repo):
        for i in range(5):
            ds_repo.create(conn=test_db, name=f"LimitDS{i}", type="local")
        result = ds_repo.list_all(limit=2, conn=test_db)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# list_filtered() + _build_where()
# ---------------------------------------------------------------------------


class TestListFiltered:
    def test_filter_eq(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="FilterEQ", type="local")
        ds_repo.create(conn=test_db, name="Other", type="smb")
        rows, total = ds_repo.list_filtered(
            {"type": "local", "name": "FilterEQ"},
            conn=test_db,
        )
        assert total >= 1
        assert all(r["type"] == "local" for r in rows)

    def test_filter_ne(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="NETest1", type="local")
        ds_repo.create(conn=test_db, name="NETest2", type="smb")
        rows, total = ds_repo.list_filtered(
            {"type__ne": "smb", "name__like": "NETest"},
            conn=test_db,
        )
        assert all(r["type"] != "smb" for r in rows)

    def test_filter_in(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="InA", type="local")
        ds_repo.create(conn=test_db, name="InB", type="smb")
        ds_repo.create(conn=test_db, name="InC", type="webdav")
        rows, total = ds_repo.list_filtered(
            {"type__in": ["local", "smb"], "name__like": "In"},
            conn=test_db,
        )
        assert all(r["type"] in ("local", "smb") for r in rows)

    def test_filter_in_empty(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="EmptyIn", type="local")
        rows, total = ds_repo.list_filtered(
            {"type__in": []},
            conn=test_db,
        )
        assert total == 0
        assert rows == []

    def test_filter_like(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="SearchMe123", type="local")
        ds_repo.create(conn=test_db, name="Nope", type="local")
        rows, total = ds_repo.list_filtered(
            {"name__like": "SearchMe"},
            conn=test_db,
        )
        assert total >= 1
        assert all("searchme" in r["name"].lower() for r in rows)

    def test_filter_like_case_insensitive(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="CaseTest", type="local")
        rows, total = ds_repo.list_filtered(
            {"name__like": "casetest"},
            conn=test_db,
        )
        assert total >= 1

    def test_filter_gte_lte(self, test_db, chat_repo, user_id):
        chat_repo.create(conn=test_db, user_id=user_id, title="GTE-LTE")
        # Alle Chats des Users mit created_at >= gestern
        rows, total = chat_repo.list_filtered(
            {"user_id": user_id, "created_at__gte": "2020-01-01"},
            conn=test_db,
        )
        assert total >= 1

    def test_filter_none_is_null(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="NullTest", type="local", path="")
        rows, _ = ds_repo.list_filtered(
            {"name": "NullTest"},
            conn=test_db,
        )
        assert len(rows) >= 1

    def test_filter_unknown_operator_raises(self, test_db, ds_repo):
        with pytest.raises(ValueError, match="Unknown filter operator"):
            ds_repo.list_filtered({"name__invalid": "x"}, conn=test_db)

    def test_pagination(self, test_db, ds_repo):
        for i in range(5):
            ds_repo.create(conn=test_db, name=f"Page{i}", type="local")
        rows, total = ds_repo.list_filtered(
            {"name__like": "Page"},
            limit=2,
            offset=0,
            conn=test_db,
        )
        assert len(rows) == 2
        assert total == 5
        # Zweite Seite
        rows2, total2 = ds_repo.list_filtered(
            {"name__like": "Page"},
            limit=2,
            offset=2,
            conn=test_db,
        )
        assert len(rows2) == 2
        assert total2 == 5


# ---------------------------------------------------------------------------
# count()
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_all(self, test_db, ds_repo):
        before = ds_repo.count(conn=test_db)
        ds_repo.create(conn=test_db, name="CountMe", type="local")
        after = ds_repo.count(conn=test_db)
        assert after == before + 1

    def test_count_filtered(self, test_db, ds_repo):
        ds_repo.create(conn=test_db, name="CountF1", type="local")
        ds_repo.create(conn=test_db, name="CountF2", type="smb")
        local_count = ds_repo.count(
            {"type": "local", "name__like": "CountF"},
            conn=test_db,
        )
        assert local_count == 1


# ---------------------------------------------------------------------------
# upsert()
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_insert(self, test_db, ds_repo):
        """Erster UPSERT fuegt ein."""
        before = ds_repo.count(conn=test_db)
        ds_repo.upsert(
            conflict_columns=["id"],
            conn=test_db,
            name="Upserted",
            type="local",
        )
        after = ds_repo.count(conn=test_db)
        assert after == before + 1

    def test_upsert_update_on_conflict(self, test_db):
        """UPSERT mit UNIQUE-Constraint auf (datasource_id, embedding_model)."""

        class EmbeddingRepo(BaseRepository):
            table = "datasource_embeddings"

        repo = EmbeddingRepo()

        # Erst Datasource erstellen
        ds_id = DatasourceTestRepo().create(
            conn=test_db,
            name="UpsertDS",
            type="local",
        )

        # Erster UPSERT — INSERT
        repo.upsert(
            conflict_columns=["datasource_id", "embedding_model"],
            conn=test_db,
            datasource_id=ds_id,
            embedding_model="nomic",
            chunk_count=10,
            file_count=3,
        )
        # Pruefen
        row = test_db.execute(
            "SELECT * FROM datasource_embeddings WHERE datasource_id = %s AND embedding_model = %s",
            (ds_id, "nomic"),
        ).fetchone()
        assert row["chunk_count"] == 10

        # Zweiter UPSERT — UPDATE
        repo.upsert(
            conflict_columns=["datasource_id", "embedding_model"],
            conn=test_db,
            datasource_id=ds_id,
            embedding_model="nomic",
            chunk_count=25,
            file_count=8,
        )
        row = test_db.execute(
            "SELECT * FROM datasource_embeddings WHERE datasource_id = %s AND embedding_model = %s",
            (ds_id, "nomic"),
        ).fetchone()
        assert row["chunk_count"] == 25
        assert row["file_count"] == 8

    def test_upsert_rejects_invalid_conflict_column(self, test_db, ds_repo):
        with pytest.raises(ValueError, match="Ungueltiger Spaltenname"):
            ds_repo.upsert(
                conflict_columns=["id) DO UPDATE SET name='x';--"],
                conn=test_db,
                name="BadUpsert",
                type="local",
            )


# ---------------------------------------------------------------------------
# create_many()
# ---------------------------------------------------------------------------


class TestCreateMany:
    def test_create_many(self, test_db, ds_repo):
        before = ds_repo.count(conn=test_db)
        count = ds_repo.create_many(
            [
                {"name": "Batch1", "type": "local"},
                {"name": "Batch2", "type": "smb"},
                {"name": "Batch3", "type": "webdav"},
            ],
            conn=test_db,
        )
        assert count == 3
        after = ds_repo.count(conn=test_db)
        assert after == before + 3

    def test_create_many_empty(self, test_db, ds_repo):
        count = ds_repo.create_many([], conn=test_db)
        assert count == 0

    def test_create_many_rejects_invalid_column(self, test_db, ds_repo):
        with pytest.raises(ValueError, match="Ungueltiger Spaltenname"):
            ds_repo.create_many(
                [
                    {"name; DROP TABLE datasources;--": "Batch1", "type": "local"},
                ],
                conn=test_db,
            )


# ---------------------------------------------------------------------------
# Option B: Externe Connection (Cross-Table-Transaction)
# ---------------------------------------------------------------------------


class TestOptionB:
    def test_cross_table_transaction(self, test_db, chat_repo, user_id):
        """Zwei Repositories teilen sich eine Transaction."""
        msg_repo = MessageTestRepo()

        # Chat + Nachricht in einer Transaction
        chat_id = chat_repo.create(
            conn=test_db,
            user_id=user_id,
            title="CrossTable",
        )
        msg_repo.create(
            conn=test_db,
            chat_id=chat_id,
            role="user",
            content="Hallo",
        )

        # Beide sichtbar in derselben Connection
        chat = chat_repo.get_by_id(chat_id, conn=test_db)
        assert chat is not None
        msgs, total = msg_repo.list_filtered(
            {"chat_id": chat_id},
            conn=test_db,
        )
        assert total == 1
        assert msgs[0]["content"] == "Hallo"

    def test_cross_table_delete(self, test_db, chat_repo, user_id):
        """Atomares Loeschen ueber zwei Tabellen."""
        msg_repo = MessageTestRepo()

        chat_id = chat_repo.create(
            conn=test_db,
            user_id=user_id,
            title="DeleteCross",
        )
        msg_repo.create(
            conn=test_db,
            chat_id=chat_id,
            role="user",
            content="Bye",
        )

        # Erst Messages, dann Chat loeschen (FK-Reihenfolge)
        msg_repo.delete_where({"chat_id": chat_id}, conn=test_db)
        chat_repo.delete(chat_id, conn=test_db)

        assert chat_repo.get_by_id(chat_id, conn=test_db) is None
        msgs, _ = msg_repo.list_filtered({"chat_id": chat_id}, conn=test_db)
        assert msgs == []


# ---------------------------------------------------------------------------
# RawSQL
# ---------------------------------------------------------------------------


class TestRawSQL:
    def test_raw_sql_repr(self):
        r = RawSQL("NOW()")
        assert repr(r) == "RawSQL('NOW()')"

    def test_raw_sql_rejects_statement_payload(self):
        with pytest.raises(ValueError, match="Ungueltiger RawSQL-Ausdruck"):
            RawSQL("NOW(); DROP TABLE users")

    def test_raw_sql_in_create(self, test_db, ds_repo):
        ds_id = ds_repo.create(
            conn=test_db,
            name="RawCreate",
            type="local",
            created_at=RawSQL("NOW()"),
            updated_at=RawSQL("NOW()"),
        )
        row = ds_repo.get_by_id(ds_id, conn=test_db)
        assert row["created_at"] is not None
        assert row["updated_at"] is not None


# ---------------------------------------------------------------------------
# _build_where() Unit-Tests (statische Methode)
# ---------------------------------------------------------------------------


class TestBuildWhere:
    def test_empty_filters(self):
        where, params = BaseRepository._build_where(None)
        assert where == ""
        assert params == []

    def test_eq(self):
        where, params = BaseRepository._build_where({"status": "active"})
        assert "status = %s" in where
        assert params == ["active"]

    def test_eq_none(self):
        where, params = BaseRepository._build_where({"status": None})
        assert "IS NULL" in where
        assert params == []

    def test_ne(self):
        where, params = BaseRepository._build_where({"status__ne": "deleted"})
        assert "status != %s" in where
        assert params == ["deleted"]

    def test_ne_none(self):
        where, params = BaseRepository._build_where({"status__ne": None})
        assert "IS NOT NULL" in where
        assert params == []

    def test_in(self):
        where, params = BaseRepository._build_where({"status__in": ["a", "b"]})
        assert "IN (%s, %s)" in where
        assert params == ["a", "b"]

    def test_in_empty(self):
        where, params = BaseRepository._build_where({"status__in": []})
        assert "FALSE" in where
        assert params == []

    def test_like(self):
        where, params = BaseRepository._build_where({"name__like": "test"})
        assert "ILIKE" in where
        assert params == ["%test%"]

    def test_gte(self):
        where, params = BaseRepository._build_where({"age__gte": 18})
        assert ">=" in where
        assert params == [18]

    def test_lte(self):
        where, params = BaseRepository._build_where({"age__lte": 65})
        assert "<=" in where
        assert params == [65]

    def test_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown filter operator"):
            BaseRepository._build_where({"name__regex": ".*"})

    def test_multiple_filters(self):
        where, params = BaseRepository._build_where(
            {
                "status": "active",
                "type__in": ["a", "b"],
                "name__like": "test",
            }
        )
        assert "AND" in where
        assert len(params) == 4  # active + a + b + %test%
