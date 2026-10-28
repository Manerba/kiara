"""Config-Repository — Singleton-Tabellen (id=1).

Verwaltet: ldap_config, mail_config, ai_config, onlyoffice_config, datasource_config.
Erbt NICHT von BaseRepository (5 Tabellen, Singleton-Pattern).

Ersetzt: shared/db.py get_singleton_config() + upsert_singleton_config()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from kiara_db.base import _validate_column
from kiara_db.connection import db_query, db_transaction

# Erlaubte Singleton-Tabellen (Whitelist gegen SQL-Injection)
_SINGLETON_TABLES = frozenset(
    {
        "ldap_config",
        "mail_config",
        "ai_config",
        "onlyoffice_config",
        "datasource_config",
    }
)


class ConfigRepository:
    """GET/SET fuer Singleton-Konfigurationstabellen (id=1)."""

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
    def _validate_table(table: str) -> str:
        if table not in _SINGLETON_TABLES:
            raise ValueError(f"Unbekannte Config-Tabelle: {table}")
        return table

    def get(self, table: str, *, conn=None) -> dict | None:
        """Liest die Singleton-Zeile (id=1)."""
        self._validate_table(table)
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT * FROM {table} WHERE id = 1",
            ).fetchone()
            return dict(row) if row else None

    def set(self, table: str, fields: dict[str, Any], *, conn=None) -> None:
        """Aktualisiert oder erstellt die Singleton-Zeile."""
        self._validate_table(table)
        fields = {k: v for k, v in fields.items() if k != "id"}
        for k in fields:
            _validate_column(k)
        with self._write(conn) as db:
            cols = ", ".join(["id", *list(fields.keys())])
            placeholders = ", ".join(["1"] + ["%s"] * len(fields))
            update_clause = ", ".join(f"{k} = EXCLUDED.{k}" for k in fields)
            db.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT (id) DO UPDATE SET {update_clause}",
                list(fields.values()),
            )

    def delete(self, table: str, *, conn=None) -> bool:
        """Loescht die Singleton-Zeile."""
        self._validate_table(table)
        with self._write(conn) as db:
            cursor = db.execute(f"DELETE FROM {table} WHERE id = 1")
            return cursor.rowcount > 0
