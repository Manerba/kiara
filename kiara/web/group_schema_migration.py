"""Aktualisiert das Gruppenschema auf SID-basierte LDAP-Identitaet."""

from __future__ import annotations

_MIGRATION_LOCK_KEY = 681032


def migrate_group_schema(conn=None) -> None:
    """Passt bestehende Instanzen an das aktuelle Groups-Schema an."""
    close_conn = False
    if conn is None:
        from kiara.shared.db import get_connection

        conn = get_connection()
        close_conn = True
    try:
        db = conn
        db.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
        db.execute("ALTER TABLE public.groups ADD COLUMN IF NOT EXISTS sid text")
        db.execute("ALTER TABLE public.groups ADD COLUMN IF NOT EXISTS name text")
        db.execute(
            """UPDATE public.groups
               SET name = COALESCE(NULLIF(name, ''), 'Gruppe ' || id::text)
               WHERE name IS NULL OR name = ''"""
        )
        db.execute("ALTER TABLE public.groups ALTER COLUMN name SET NOT NULL")
        db.execute("UPDATE public.groups SET sid = NULL WHERE sid = ''")
        db.execute("ALTER TABLE public.groups DROP CONSTRAINT IF EXISTS groups_name_key")
        db.execute("ALTER TABLE public.groups DROP CONSTRAINT IF EXISTS groups_dn_key")
        db.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_groups_sid_unique
               ON public.groups (sid)
               WHERE sid IS NOT NULL AND sid <> ''"""
        )
        if close_conn:
            db.commit()
    except Exception:
        if close_conn:
            conn.rollback()
        raise
    finally:
        if close_conn:
            conn.close()
