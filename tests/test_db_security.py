"""Tests fuer SQL-Injection-Schutz in kiara.shared.db (F-21).

Prueft validate_identifier(), paginated_query(), insert_row(), update_row()
und die _SQL_IDENT_RE-Regex gegen Injection-Versuche.
"""

import pytest

from kiara.shared.db import (
    ALLOWED_COLUMNS,
    ALLOWED_TABLES,
    get_connection,
    insert_row,
    paginated_query,
    update_row,
    validate_identifier,
)

# ---------------------------------------------------------------------------
# Hilfsfunktion: PostgreSQL-DB mit Minimaltabelle
# ---------------------------------------------------------------------------


def _make_db():
    """Erstellt eine PostgreSQL-Verbindung mit einer 'datasources'-Tabelle."""
    conn = get_connection()
    # Testdaten bereinigen statt DROP+CREATE (erhalt Produktions-Schema)
    conn.execute("DELETE FROM datasources WHERE name = 'Test DS'")
    conn.execute(
        "INSERT INTO datasources (name, type, path) VALUES (%s, %s, %s)",
        ("Test DS", "local", "/data"),
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# paginated_query: SQL-Injection in table
# ---------------------------------------------------------------------------


class TestPaginatedQuerySecurity:
    def test_paginated_query_rejects_sql_injection_in_table(self):
        """table='users; DROP TABLE users--' muss ValueError ausloesen."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(db, table="users; DROP TABLE users--")
        db.close()

    def test_paginated_query_rejects_injection_in_order_by(self):
        """order_by='id; DROP TABLE users' muss ValueError ausloesen."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(db, table="datasources", order_by="id; DROP TABLE users")
        db.close()

    def test_paginated_query_accepts_valid_identifiers(self):
        """Gueltige Bezeichner (table='datasources', order_by='id') funktionieren."""
        db = _make_db()
        result = paginated_query(db, table="datasources", order_by="id DESC")
        assert "data" in result
        assert "total" in result
        assert result["total"] >= 1
        assert len(result["data"]) >= 1
        db.close()

    def test_paginated_query_accepts_static_join_contract(self):
        """Der Audit-Log-Join bleibt als expliziter, validierter Vertrag erlaubt."""
        db = _make_db()
        result = paginated_query(
            db,
            table="audit_anonymization a",
            columns="a.*, u.username",
            joins="LEFT JOIN users u ON a.user_id = u.id",
            order_by="a.timestamp DESC",
        )
        assert "data" in result
        assert "total" in result
        db.close()

    def test_paginated_query_rejects_join_injection(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(
                db,
                table="audit_anonymization a",
                joins="LEFT JOIN users u ON a.user_id = u.id; DROP TABLE users",
            )
        db.close()

    def test_paginated_query_rejects_condition_injection(self):
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiges SQL-Condition-Fragment"):
            paginated_query(
                db,
                table="datasources",
                conditions=["name = 'x' OR '1'='1'"],
            )
        db.close()


# ---------------------------------------------------------------------------
# validate_identifier: Whitelist-Pruefung
# ---------------------------------------------------------------------------


class TestValidateIdentifier:
    def test_validate_identifier_rejects_unknown_table(self):
        """Unbekannter Tabellenname muss ValueError ausloesen."""
        with pytest.raises(ValueError, match="Invalid SQL table"):
            validate_identifier("nonexistent_table", ALLOWED_TABLES, "table")

    def test_validate_identifier_accepts_known_table(self):
        """Bekannter Tabellenname wird zurueckgegeben."""
        result = validate_identifier("datasources", ALLOWED_TABLES, "table")
        assert result == "datasources"

    def test_validate_identifier_rejects_unknown_column(self):
        """Unbekannter Spaltenname muss ValueError ausloesen."""
        with pytest.raises(ValueError, match="Invalid SQL column"):
            validate_identifier("evil_column", ALLOWED_COLUMNS, "column")

    def test_validate_identifier_accepts_known_column(self):
        """Bekannter Spaltenname wird zurueckgegeben."""
        result = validate_identifier("name", ALLOWED_COLUMNS, "column")
        assert result == "name"


# ---------------------------------------------------------------------------
# _SQL_IDENT_RE: Regex blockiert Sonderzeichen
# ---------------------------------------------------------------------------


class TestSqlIdentRegex:
    def test_sql_ident_regex_blocks_semicolons(self):
        """Semikolons in Tabellennamen werden abgelehnt."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(db, table="users; DROP")
        db.close()

    def test_sql_ident_regex_blocks_double_dash_comment(self):
        """SQL-Kommentare (--) werden abgelehnt."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(db, table="users -- comment")
        db.close()

    def test_sql_ident_regex_blocks_hash_comment(self):
        """Hash-Kommentare (#) in order_by werden abgelehnt."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(db, table="datasources", order_by="id # comment")
        db.close()

    def test_sql_ident_regex_blocks_backticks(self):
        """Backticks werden abgelehnt."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(db, table="`users`")
        db.close()

    def test_sql_ident_regex_blocks_union_injection(self):
        """UNION SELECT Injection in columns wird abgelehnt (Semikolon)."""
        db = _make_db()
        with pytest.raises(ValueError, match="Ungueltiger SQL-Bezeichner"):
            paginated_query(
                db,
                table="datasources",
                columns="*; SELECT * FROM pg_catalog.pg_tables",
            )
        db.close()


# ---------------------------------------------------------------------------
# insert_row / update_row: Whitelist-Validierung
# ---------------------------------------------------------------------------


class TestInsertUpdateRowSecurity:
    def test_insert_row_rejects_unknown_table(self):
        """insert_row mit unbekannter Tabelle muss ValueError ausloesen."""
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid SQL table"):
            insert_row(db, "evil_table", {"name": "test"})
        db.close()

    def test_insert_row_rejects_unknown_column(self):
        """insert_row mit unbekannter Spalte muss ValueError ausloesen."""
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid SQL column"):
            insert_row(db, "datasources", {"evil_col": "test"})
        db.close()

    def test_update_row_rejects_unknown_table(self):
        """update_row mit unbekannter Tabelle muss ValueError ausloesen."""
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid SQL table"):
            update_row(db, "evil_table", 1, {"name": "test"})
        db.close()

    def test_update_row_rejects_unknown_column(self):
        """update_row mit unbekannter Spalte muss ValueError ausloesen."""
        db = _make_db()
        with pytest.raises(ValueError, match="Invalid SQL column"):
            update_row(db, "datasources", 1, {"evil_col": "test"})
        db.close()
