"""Group-Repository.

Ersetzt SQL in: web/api_groups.py (9), web/ldap_sync.py
"""

from __future__ import annotations

from kiara_db.base import BaseRepository


class GroupRepository(BaseRepository):
    table = "groups"
    default_sort = "cn ASC"

    def list_with_counts(self, *, conn=None) -> list[dict]:
        """Alle Gruppen mit Datasource- und User-Anzahl."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT g.*,
                          (SELECT COUNT(*) FROM datasource_groups dg
                           WHERE dg.group_id = g.id) AS datasource_count,
                          (SELECT COUNT(*) FROM user_groups ug
                           WHERE ug.group_id = g.id) AS user_count
                   FROM groups g ORDER BY g.cn""",
            ).fetchall()
            return [dict(r) for r in rows]

    def get_members(self, group_id: int, *, conn=None) -> list[dict]:
        """Alle Mitglieder einer Gruppe."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT u.id, u.username, u.vorname, u.nachname,
                          u.email, u.auth_source, u.is_active
                   FROM users u
                   JOIN user_groups ug ON u.id = ug.user_id
                   WHERE ug.group_id = %s
                   ORDER BY u.username""",
                (group_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_sid(self, sid: str, *, conn=None) -> dict | None:
        """Gruppe anhand der stabilen Active-Directory-SID."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM groups WHERE sid = %s",
                (sid,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_sids(self, *, conn=None) -> list[str]:
        """Alle importierten LDAP-SIDs."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT sid FROM groups WHERE sid IS NOT NULL AND sid != ''"
            ).fetchall()
            return [r["sid"] for r in rows]

    def get_all_cns(self, *, conn=None) -> list[str]:
        """Alle Gruppen-Common-Names."""
        with self._read(conn) as db:
            rows = db.execute("SELECT cn FROM groups").fetchall()
            return [r["cn"] for r in rows]

    def upsert_ldap(
        self,
        *,
        sid: str,
        cn: str,
        dn: str,
        member_count: int,
        conn=None,
    ) -> int:
        """LDAP-Gruppe importieren oder aktualisieren. Gibt ID zurueck."""
        if not sid:
            raise ValueError("LDAP-Gruppe ohne SID wird nicht importiert")
        with self._write(conn) as db:
            existing = db.execute(
                "SELECT id FROM groups WHERE sid = %s",
                (sid,),
            ).fetchone()
            if existing:
                db.execute(
                    """UPDATE groups SET cn = %s, name = %s, dn = %s, member_count = %s,
                              updated_at = NOW() WHERE id = %s""",
                    (cn, cn, dn, member_count, existing["id"]),
                )
                return existing["id"]
            else:
                cursor = db.execute(
                    "INSERT INTO groups (sid, name, cn, dn, member_count) "
                    "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (sid, cn, cn, dn, member_count),
                )
                return cursor.fetchone()["id"]

    def delete_cascade(self, group_id: int, *, conn=None) -> bool:
        """Loescht Gruppe mit allen Zuweisungen."""
        with self._write(conn) as db:
            db.execute("DELETE FROM user_groups WHERE group_id = %s", (group_id,))
            db.execute("DELETE FROM datasource_groups WHERE group_id = %s", (group_id,))
            cursor = db.execute("DELETE FROM groups WHERE id = %s", (group_id,))
            return cursor.rowcount > 0
