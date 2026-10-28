"""Tests fuer die aktuelle SID-basierte Group-Schema-Migration."""

from kiara.web.group_schema_migration import migrate_group_schema


def test_migrate_group_schema_without_cn_dn_columns(test_db):
    """Frisches aktuelles Schema braucht keine cn/dn-Fallback-Spalten."""
    test_db.execute("DROP TABLE IF EXISTS groups CASCADE")
    test_db.execute(
        """CREATE TABLE groups (
               id SERIAL PRIMARY KEY,
               sid text,
               name text
           )"""
    )
    cursor = test_db.execute(
        "INSERT INTO groups (sid, name) VALUES (%s, %s) RETURNING id",
        ("S-1-5-21-test", None),
    )
    group_id = cursor.fetchone()["id"]

    migrate_group_schema(test_db)

    row = test_db.execute(
        "SELECT sid, name FROM groups WHERE id = %s",
        (group_id,),
    ).fetchone()
    assert row["sid"] == "S-1-5-21-test"
    assert row["name"] == f"Gruppe {group_id}"
