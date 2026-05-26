"""Idempotente Kernschema-Korrekturen fuer die laufende KIara-DB."""

from __future__ import annotations

_MIGRATION_LOCK_KEY = 681031


def migrate_core_schema(conn=None) -> None:
    """Erzwingt zentrale FK-/Expiry-Invarianten fuer bestehende Datenbanken."""
    close_conn = False
    if conn is None:
        from kiara.shared.db import get_db

        conn = get_db()
        close_conn = True

    try:
        if not _has_required_tables(conn):
            return
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
        _migrate_confirmation_expiry(conn)
        _migrate_user_cascade_constraints(conn)
        if close_conn:
            conn.commit()
    except Exception:
        if close_conn:
            conn.rollback()
        raise
    finally:
        if close_conn:
            conn.close()


def _has_required_tables(conn) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS count FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        "AND table_name IN ("
        "'users', 'admin_chat_confirmations', 'sessions', 'api_keys', "
        "'chats', 'admin_chats', 'admin_chat_wizards', 'user_groups', "
        "'groups', 'datasource_users', 'datasources'"
        ")"
    ).fetchone()
    return bool(row and row["count"] == 11)


def _migrate_confirmation_expiry(conn) -> None:
    conn.execute(
        "ALTER TABLE public.admin_chat_confirmations "
        "ADD COLUMN IF NOT EXISTS expires_at timestamp with time zone"
    )
    conn.execute(
        "UPDATE public.admin_chat_confirmations "
        "SET expires_at = COALESCE(expires_at, created_at::timestamptz + interval '30 minutes') "
        "WHERE expires_at IS NULL"
    )
    conn.execute(
        "ALTER TABLE public.admin_chat_confirmations "
        "ALTER COLUMN expires_at SET DEFAULT (now() + interval '30 minutes')"
    )
    conn.execute("ALTER TABLE public.admin_chat_confirmations ALTER COLUMN expires_at SET NOT NULL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_confirm_expires "
        "ON public.admin_chat_confirmations USING btree (status, expires_at)"
    )


def _migrate_user_cascade_constraints(conn) -> None:
    cleanup_statements = [
        (
            "DELETE FROM public.sessions s "
            "WHERE NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = s.user_id)"
        ),
        (
            "DELETE FROM public.api_keys k "
            "WHERE k.user_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = k.user_id)"
        ),
        (
            "DELETE FROM public.chats c "
            "WHERE c.user_id IS NULL "
            "OR NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = c.user_id)"
        ),
        (
            "DELETE FROM public.admin_chats c "
            "WHERE c.user_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = c.user_id)"
        ),
        (
            "DELETE FROM public.admin_chat_confirmations c "
            "WHERE c.user_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = c.user_id)"
        ),
        (
            "DELETE FROM public.admin_chat_wizards w "
            "WHERE w.user_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = w.user_id)"
        ),
        (
            "DELETE FROM public.user_groups ug "
            "WHERE NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = ug.user_id) "
            "OR NOT EXISTS (SELECT 1 FROM public.groups g WHERE g.id = ug.group_id)"
        ),
        (
            "DELETE FROM public.datasource_users du "
            "WHERE NOT EXISTS (SELECT 1 FROM public.users u WHERE u.id = du.user_id) "
            "OR NOT EXISTS (SELECT 1 FROM public.datasources d WHERE d.id = du.datasource_id)"
        ),
    ]
    for statement in cleanup_statements:
        conn.execute(statement)

    if not _column_is_not_null(conn, "chats", "user_id"):
        conn.execute("ALTER TABLE public.chats ALTER COLUMN user_id SET NOT NULL")

    constraints = [
        ("sessions", "sessions_user_id_fkey", "user_id", "users", "id"),
        ("api_keys", "api_keys_user_id_fkey", "user_id", "users", "id"),
        ("chats", "chats_user_id_fkey", "user_id", "users", "id"),
        ("admin_chats", "admin_chats_user_id_fkey", "user_id", "users", "id"),
        (
            "admin_chat_confirmations",
            "admin_chat_confirmations_user_id_fkey",
            "user_id",
            "users",
            "id",
        ),
        ("admin_chat_wizards", "admin_chat_wizards_user_id_fkey", "user_id", "users", "id"),
        ("user_groups", "user_groups_user_id_fkey", "user_id", "users", "id"),
        ("user_groups", "user_groups_group_id_fkey", "group_id", "groups", "id"),
        ("datasource_users", "datasource_users_user_id_fkey", "user_id", "users", "id"),
        (
            "datasource_users",
            "datasource_users_datasource_id_fkey",
            "datasource_id",
            "datasources",
            "id",
        ),
    ]
    for table, constraint, column, ref_table, ref_column in constraints:
        expected_def = (
            f"FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column}) ON DELETE CASCADE"
        )
        if _constraint_matches(conn, table, constraint, expected_def):
            continue
        conn.execute(f"ALTER TABLE public.{table} DROP CONSTRAINT IF EXISTS {constraint}")
        conn.execute(
            f"ALTER TABLE public.{table} ADD CONSTRAINT {constraint} "
            f"FOREIGN KEY ({column}) REFERENCES public.{ref_table}({ref_column}) "
            "ON DELETE CASCADE"
        )


def _column_is_not_null(conn, table: str, column: str) -> bool:
    row = conn.execute(
        "SELECT attnotnull FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' "
        "AND c.relname = %s "
        "AND a.attname = %s "
        "AND NOT a.attisdropped",
        (table, column),
    ).fetchone()
    return bool(row and row["attnotnull"])


def _constraint_matches(conn, table: str, constraint: str, expected_def: str) -> bool:
    row = conn.execute(
        "SELECT pg_get_constraintdef(con.oid) AS definition "
        "FROM pg_constraint con "
        "JOIN pg_class rel ON rel.oid = con.conrelid "
        "JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace "
        "WHERE nsp.nspname = 'public' "
        "AND rel.relname = %s "
        "AND con.conname = %s "
        "AND con.contype = 'f'",
        (table, constraint),
    ).fetchone()
    return bool(row and row["definition"] == expected_def)
