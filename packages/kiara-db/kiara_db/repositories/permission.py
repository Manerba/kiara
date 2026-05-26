"""Permission-Repository — M2M-Beziehungen.

Verwaltet: datasource_groups, datasource_users, user_groups.
Erbt NICHT von BaseRepository (3 Junction-Tabellen).

Ersetzt SQL in: web/api_datasource.py, web/api_groups.py, web/ldap_sync.py
"""

from __future__ import annotations

from contextlib import contextmanager

from kiara_db.connection import db_query, db_transaction


class PermissionRepository:
    """Verwaltet M2M-Beziehungen (DS <-> Groups, DS <-> Users, User <-> Groups)."""

    @contextmanager
    def _read(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_query() as db:
                yield db

    @contextmanager
    def _write(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_transaction() as db:
                yield db

    @staticmethod
    def _normalize_ids(ids: list[int]) -> list[int]:
        normalized: list[int] = []
        seen: set[int] = set()
        for raw_id in ids:
            value = int(raw_id)
            if value not in seen:
                normalized.append(value)
                seen.add(value)
        return normalized

    @staticmethod
    def _assert_id_exists(db, table: str, row_id: int, label: str) -> None:
        row = db.execute(
            f"SELECT id FROM {table} WHERE id = %s",
            (int(row_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"{label} nicht gefunden: {row_id}")

    def _assert_all_ids_exist(
        self,
        db,
        table: str,
        row_ids: list[int],
        label: str,
    ) -> list[int]:
        normalized = self._normalize_ids(row_ids)
        if not normalized:
            return []
        placeholders = ", ".join(["%s"] * len(normalized))
        rows = db.execute(
            f"SELECT id FROM {table} WHERE id IN ({placeholders})",
            tuple(normalized),
        ).fetchall()
        existing = {int(r["id"]) for r in rows}
        missing = [row_id for row_id in normalized if row_id not in existing]
        if missing:
            raise ValueError(f"{label} nicht gefunden: {missing}")
        return normalized

    # ----- Datasource <-> Groups -----

    def set_datasource_groups(
        self,
        ds_id: int,
        group_ids: list[int],
        *,
        conn=None,
    ) -> None:
        """Ersetzt alle Gruppen-Zuweisungen fuer eine Datenquelle."""
        with self._write(conn) as db:
            self._assert_id_exists(db, "datasources", ds_id, "Datenquelle")
            valid_group_ids = self._assert_all_ids_exist(db, "groups", group_ids, "Gruppen")
            db.execute("DELETE FROM datasource_groups WHERE datasource_id = %s", (ds_id,))
            if valid_group_ids:
                db.executemany(
                    "INSERT INTO datasource_groups (datasource_id, group_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(ds_id, gid) for gid in valid_group_ids],
                )

    def get_datasource_groups(self, ds_id: int, *, conn=None) -> list[dict]:
        """Gruppen einer Datenquelle mit Details."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT g.id, g.cn, g.dn FROM groups g
                   JOIN datasource_groups dg ON g.id = dg.group_id
                   WHERE dg.datasource_id = %s
                   ORDER BY g.cn""",
                (ds_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_group_datasources(self, group_id: int, *, conn=None) -> list[dict]:
        """Datenquellen einer Gruppe mit Details."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT d.id, d.name, d.type, d.is_active,
                          d.restriction_level
                   FROM datasources d
                   JOIN datasource_groups dg ON d.id = dg.datasource_id
                   WHERE dg.group_id = %s
                   ORDER BY d.name""",
                (group_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_unassigned_datasources(self, *, conn=None) -> int:
        """Anzahl aktiver Datenquellen ohne Gruppenzuordnung."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT COUNT(*) AS count FROM datasources d
                   WHERE d.is_active = true
                   AND NOT EXISTS (
                       SELECT 1 FROM datasource_groups dg WHERE dg.datasource_id = d.id
                   )"""
            ).fetchone()
            return row["count"] if row else 0

    def get_datasource_group_ids(self, ds_id: int, *, conn=None) -> list[int]:
        """Nur die Group-IDs einer Datenquelle."""
        with self._read(conn) as db:
            rows = db.execute(
                "SELECT group_id FROM datasource_groups WHERE datasource_id = %s",
                (ds_id,),
            ).fetchall()
            return [r["group_id"] for r in rows]

    # ----- Datasource <-> Users -----

    def set_datasource_users(
        self,
        ds_id: int,
        user_ids: list[int],
        *,
        conn=None,
    ) -> None:
        """Ersetzt alle User-Zuweisungen fuer eine Datenquelle."""
        with self._write(conn) as db:
            self._assert_id_exists(db, "datasources", ds_id, "Datenquelle")
            valid_user_ids = self._assert_all_ids_exist(db, "users", user_ids, "Benutzer")
            db.execute("DELETE FROM datasource_users WHERE datasource_id = %s", (ds_id,))
            if valid_user_ids:
                db.executemany(
                    "INSERT INTO datasource_users (datasource_id, user_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(ds_id, uid) for uid in valid_user_ids],
                )

    def get_datasource_users(self, ds_id: int, *, conn=None) -> list[dict]:
        """User einer Datenquelle mit Details."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT u.id, u.username, u.vorname, u.nachname, u.auth_source
                   FROM users u
                   JOIN datasource_users du ON u.id = du.user_id
                   WHERE du.datasource_id = %s
                   ORDER BY u.username""",
                (ds_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ----- User <-> Groups -----

    def set_user_groups(
        self,
        user_id: int,
        group_ids: list[int],
        *,
        conn=None,
    ) -> None:
        """Ersetzt alle Gruppen eines Users (LDAP-Sync)."""
        with self._write(conn) as db:
            self._assert_id_exists(db, "users", user_id, "Benutzer")
            valid_group_ids = self._assert_all_ids_exist(db, "groups", group_ids, "Gruppen")
            db.execute("DELETE FROM user_groups WHERE user_id = %s", (user_id,))
            if valid_group_ids:
                db.executemany(
                    "INSERT INTO user_groups (user_id, group_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(user_id, gid) for gid in valid_group_ids],
                )

    # ----- Berechtigungs-Queries -----

    def get_user_datasource_ids(self, user_id: int, *, conn=None) -> list[int]:
        """Alle DS-IDs auf die ein User Zugriff hat (direkt + via Gruppen)."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT DISTINCT datasource_id FROM (
                       SELECT du.datasource_id FROM datasource_users du
                       JOIN datasources d ON d.id = du.datasource_id AND d.is_active = true
                       WHERE du.user_id = %s
                       UNION
                       SELECT dg.datasource_id FROM datasource_groups dg
                       JOIN user_groups ug ON dg.group_id = ug.group_id
                       JOIN datasources d ON d.id = dg.datasource_id AND d.is_active = true
                       WHERE ug.user_id = %s
                   ) AS combined""",
                (user_id, user_id),
            ).fetchall()
            return [r["datasource_id"] for r in rows]

    def get_public_datasource_ids(self, *, conn=None) -> list[int]:
        """DS-IDs die explizit als oeffentlich markiert sind."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT id FROM datasources
                   WHERE is_active = true
                   AND is_public = true""",
            ).fetchall()
            return [r["id"] for r in rows]
