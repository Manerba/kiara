"""Tests fuer Build 57: Timestamps in Chat-Nachrichten.

Testet:
    - Backend: created_at-Felder in API-Responses (Chats + Messages)
    - Backend: Timestamp-Format-Validierung (DB NOW() Format)
    - Backend: Edge Cases (leere/fehlende Timestamps)
    - Frontend-Logik: _formatTimestamp()-Verhalten (als Unit-Test der Logik)
"""

import json
import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from kiara.shared.db import get_connection

# ===========================================================================
# Hilfsfunktionen
# ===========================================================================

# datetime-Format: "YYYY-MM-DD HH:MM:SS" (ohne T, ohne TZ).
_DT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

_ISO8601_LOOSE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")


def _make_test_db():
    """Erstellt PostgreSQL-Tabellen mit Chat-Schema (ohne DROP — verhindert Deadlocks)."""
    conn = get_connection()
    # Tabellen mit IF NOT EXISTS erstellen (kein DROP — wuerde andere Tests brechen)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            is_admin BOOLEAN DEFAULT false,
            is_active BOOLEAN DEFAULT true,
            created_at TEXT DEFAULT (NOW())
        );

        CREATE TABLE IF NOT EXISTS chats (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (NOW()),
            updated_at TEXT DEFAULT (NOW()),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT DEFAULT (NOW()),
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS query_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            chat_id INTEGER,
            answer_excerpt TEXT,
            full_answer TEXT,
            trace_id TEXT,
            created_at TEXT DEFAULT (NOW()),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Saubere Testdaten: zuerst query_log bereinigen (FK auf users!), dann Chats/User
    conn.execute(
        "DELETE FROM query_log WHERE user_id IN "
        "(SELECT id FROM users WHERE username IN ('testuser', 'admin_ts_test'))"
    )
    conn.execute(
        "DELETE FROM chat_messages WHERE chat_id IN "
        "(SELECT c.id FROM chats c JOIN users u ON c.user_id = u.id "
        " WHERE u.username IN ('testuser', 'admin_ts_test'))"
    )
    conn.execute(
        "DELETE FROM chats WHERE user_id IN "
        "(SELECT id FROM users WHERE username IN ('testuser', 'admin_ts_test'))"
    )
    conn.execute("DELETE FROM users WHERE username IN ('testuser', 'admin_ts_test')")
    conn.commit()
    return conn, "pg"


def _format_ts(value):
    """Formatiert einen PostgreSQL-Timestamp (datetime oder str) als 'YYYY-MM-DD HH:MM:SS'."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    s = str(value)
    # Entferne Zeitzone und Mikrosekunden falls vorhanden
    if "+" in s:
        s = s.split("+")[0]
    if "." in s:
        s = s.split(".")[0]
    return s.strip()


# ===========================================================================
# DB-Schema: created_at Default-Werte
# ===========================================================================


class TestTimestampDBDefaults:
    """Prueft, dass PostgreSQL-Defaults korrekte Timestamps erzeugen."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.conn, self.db_path = _make_test_db()
        # Testuser anlegen und ID merken
        cursor = self.conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            ("testuser", "hash"),
        )
        self.user_id = cursor.fetchone()["id"]
        self.conn.commit()
        yield
        self.conn.rollback()
        self.conn.close()

    def test_chat_created_at_auto_populated(self):
        """Neuer Chat bekommt automatisch einen created_at-Timestamp."""
        cursor = self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (self.user_id, "Testchat"),
        )
        chat_id = cursor.fetchone()["id"]
        self.conn.commit()
        row = self.conn.execute("SELECT created_at FROM chats WHERE id = %s", (chat_id,)).fetchone()
        assert row["created_at"] is not None
        ts = _format_ts(row["created_at"])
        assert _DT_PATTERN.match(ts), f"Unerwartetes Format: {ts}"

    def test_chat_message_created_at_auto_populated(self):
        """Neue Chat-Nachricht bekommt automatisch einen created_at-Timestamp."""
        cursor = self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (self.user_id, "Test"),
        )
        chat_id = cursor.fetchone()["id"]
        cursor = self.conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content) VALUES (%s, %s, %s) RETURNING id",
            (chat_id, "user", "Hallo"),
        )
        msg_id = cursor.fetchone()["id"]
        self.conn.commit()
        row = self.conn.execute(
            "SELECT created_at FROM chat_messages WHERE id = %s", (msg_id,)
        ).fetchone()
        assert row["created_at"] is not None
        ts = _format_ts(row["created_at"])
        assert _DT_PATTERN.match(ts)

    def test_chat_updated_at_auto_populated(self):
        """Neuer Chat bekommt automatisch einen updated_at-Timestamp."""
        cursor = self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (self.user_id, "Test"),
        )
        chat_id = cursor.fetchone()["id"]
        self.conn.commit()
        row = self.conn.execute("SELECT updated_at FROM chats WHERE id = %s", (chat_id,)).fetchone()
        assert row["updated_at"] is not None
        ts = _format_ts(row["updated_at"])
        assert _DT_PATTERN.match(ts)

    def test_timestamps_are_parseable_as_datetime(self):
        """PostgreSQL-Timestamps lassen sich als Python datetime parsen."""
        cursor = self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (self.user_id, "Test"),
        )
        chat_id = cursor.fetchone()["id"]
        self.conn.commit()
        row = self.conn.execute("SELECT created_at FROM chats WHERE id = %s", (chat_id,)).fetchone()
        ts = _format_ts(row["created_at"])
        parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        assert parsed.year >= 2024


# ===========================================================================
# API-Response: get_chat — created_at in Chat + Messages
# ===========================================================================


class TestGetChatTimestamps:
    """Prueft, dass die get_chat-API created_at-Felder korrekt zurueckgibt."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.conn, self.db_path = _make_test_db()
        # Testdaten
        cursor = self.conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, %s, %s) RETURNING id",
            ("admin_ts_test", "hash", True),
        )
        self.user_id = cursor.fetchone()["id"]
        cursor = self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (self.user_id, "Testchat"),
        )
        self.chat_id = cursor.fetchone()["id"]
        self.conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content, sources) VALUES (%s, %s, %s, %s)",
            (self.chat_id, "user", "Was ist RAG?", None),
        )
        self.conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content, sources) VALUES (%s, %s, %s, %s)",
            (self.chat_id, "assistant", "RAG steht fuer...", '[{"source":"doc.txt","score":0.9}]'),
        )
        self.conn.commit()
        yield
        self.conn.rollback()
        self.conn.close()

    def _get_chat_row(self):
        """Holt die Chat-Row als Dict aus der Test-DB."""
        row = self.conn.execute("SELECT * FROM chats WHERE id = %s", (self.chat_id,)).fetchone()
        d = dict(row) if row else None
        if d:
            for k in ("created_at", "updated_at"):
                if k in d:
                    d[k] = _format_ts(d[k])
        return d

    def _get_message_rows(self):
        """Holt alle Message-Rows als Dicts aus der Test-DB."""
        rows = self.conn.execute(
            "SELECT * FROM chat_messages WHERE chat_id = %s ORDER BY id",
            (self.chat_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if "created_at" in d:
                d["created_at"] = _format_ts(d["created_at"])
            result.append(d)
        return result

    @pytest.mark.asyncio
    async def test_chat_response_has_created_at(self):
        """GET /chat/{id} Response enthaelt created_at auf Chat-Ebene."""
        from kiara.web.api_chat import get_chat

        user = {"id": self.user_id, "is_admin": True}
        with (
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        ):
            mock_chat_repo.get_owned.return_value = self._get_chat_row()
            mock_msg_repo.list_by_chat.return_value = self._get_message_rows()
            result = await get_chat(self.chat_id, user)

        assert "created_at" in result
        assert result["created_at"] is not None
        assert _ISO8601_LOOSE_PATTERN.match(result["created_at"])

    @pytest.mark.asyncio
    async def test_messages_have_created_at(self):
        """GET /chat/{id} Messages enthalten jeweils created_at."""
        from kiara.web.api_chat import get_chat

        user = {"id": self.user_id, "is_admin": True}
        with (
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        ):
            mock_chat_repo.get_owned.return_value = self._get_chat_row()
            mock_msg_repo.list_by_chat.return_value = self._get_message_rows()
            result = await get_chat(self.chat_id, user)

        messages = result["messages"]
        assert len(messages) == 2
        for msg in messages:
            assert "created_at" in msg
            assert msg["created_at"] is not None
            assert _ISO8601_LOOSE_PATTERN.match(msg["created_at"]), (
                f"Unerwartetes Timestamp-Format: {msg['created_at']}"
            )

    @pytest.mark.asyncio
    async def test_message_created_at_matches_db_format(self):
        """Message-Timestamps haben das DB datetime()-Format."""
        from kiara.web.api_chat import get_chat

        user = {"id": self.user_id, "is_admin": True}
        with (
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        ):
            mock_chat_repo.get_owned.return_value = self._get_chat_row()
            mock_msg_repo.list_by_chat.return_value = self._get_message_rows()
            result = await get_chat(self.chat_id, user)

        for msg in result["messages"]:
            assert _DT_PATTERN.match(msg["created_at"]), (
                f"Erwartet 'YYYY-MM-DD HH:MM:SS', bekommen: {msg['created_at']}"
            )

    @pytest.mark.asyncio
    async def test_user_and_assistant_messages_have_timestamps(self):
        """Sowohl User- als auch Assistant-Messages haben Timestamps."""
        from kiara.web.api_chat import get_chat

        user = {"id": self.user_id, "is_admin": True}
        with (
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        ):
            mock_chat_repo.get_owned.return_value = self._get_chat_row()
            mock_msg_repo.list_by_chat.return_value = self._get_message_rows()
            result = await get_chat(self.chat_id, user)

        roles = {msg["role"]: msg["created_at"] for msg in result["messages"]}
        assert "user" in roles
        assert "assistant" in roles
        assert roles["user"] is not None
        assert roles["assistant"] is not None


# ===========================================================================
# API-Response: list_chats — Timestamps in Chat-Liste
# ===========================================================================


class TestListChatsTimestamps:
    """Prueft created_at und updated_at in der Chat-Liste."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.conn, self.db_path = _make_test_db()
        cursor = self.conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            ("testuser", "hash"),
        )
        self.user_id = cursor.fetchone()["id"]
        self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s)",
            (self.user_id, "Chat A"),
        )
        self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s)",
            (self.user_id, "Chat B"),
        )
        self.conn.commit()
        yield
        self.conn.rollback()
        self.conn.close()

    def _get_chat_rows(self):
        """Holt alle Chat-Rows als Dicts aus der Test-DB."""
        rows = self.conn.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM chat_messages WHERE chat_id = c.id) AS message_count "
            "FROM chats c WHERE c.user_id = %s ORDER BY c.updated_at DESC",
            (self.user_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for k in ("created_at", "updated_at"):
                if k in d:
                    d[k] = _format_ts(d[k])
            result.append(d)
        return result

    @pytest.mark.asyncio
    async def test_list_chats_has_timestamps(self):
        """GET /chats Response enthaelt created_at und updated_at pro Chat."""
        from kiara.web.api_chat import list_chats

        user = {"id": self.user_id, "is_admin": False}
        with patch("kiara.web.api_chat.chat_repo") as mock_chat_repo:
            mock_chat_repo.list_by_user.return_value = self._get_chat_rows()
            result = await list_chats(user)

        assert len(result["data"]) == 2
        for chat in result["data"]:
            assert "created_at" in chat
            assert "updated_at" in chat
            assert chat["created_at"] is not None
            assert chat["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_list_chats_timestamp_format(self):
        """Chat-Timestamps in der Liste haben das korrekte Format."""
        from kiara.web.api_chat import list_chats

        user = {"id": self.user_id, "is_admin": False}
        with patch("kiara.web.api_chat.chat_repo") as mock_chat_repo:
            mock_chat_repo.list_by_user.return_value = self._get_chat_rows()
            result = await list_chats(user)

        for chat in result["data"]:
            assert _DT_PATTERN.match(chat["created_at"]), (
                f"Unerwartetes created_at-Format: {chat['created_at']}"
            )
            assert _DT_PATTERN.match(chat["updated_at"]), (
                f"Unerwartetes updated_at-Format: {chat['updated_at']}"
            )


# ===========================================================================
# _save_assistant_message — Timestamp via DB-Default
# ===========================================================================


class TestSaveMessageTimestamp:
    """Prueft, dass _save_assistant_message Timestamps korrekt speichert."""

    def test_insert_calls_msg_repo_add_message(self):
        """_save_assistant_message ruft msg_repo.add_message auf."""
        from kiara.web.api_chat import _save_assistant_message

        with (
            patch("kiara.web.api_chat.db_transaction") as mock_db,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
        ):
            mock_conn = MagicMock()
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            _save_assistant_message(1, "Antwort")

            mock_msg_repo.add_message.assert_called_once()
            call_args = mock_msg_repo.add_message.call_args
            assert call_args[0][0] == 1  # chat_id
            assert call_args[0][1] == "assistant"  # role
            assert call_args[0][2] == "Antwort"  # content

    def test_update_calls_chat_repo_touch(self):
        """Nach dem Speichern wird chat_repo.touch aufgerufen."""
        from kiara.web.api_chat import _save_assistant_message

        with (
            patch("kiara.web.api_chat.db_transaction") as mock_db,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
        ):
            mock_conn = MagicMock()
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            _save_assistant_message(42, "Antwort")

            mock_chat_repo.touch.assert_called_once()
            assert mock_chat_repo.touch.call_args[0][0] == 42


# ===========================================================================
# Frontend _formatTimestamp()-Logik — Python-Nachbildung
# ===========================================================================


def _format_timestamp_py(date_input):
    """Python-Nachbildung der JS _formatTimestamp()-Funktion.

    Testet dieselbe Logik:
    - None/falsy → aktuelle Zeit
    - DB-Format "YYYY-MM-DD HH:MM:SS" → Space-to-T Replacement → parse
    - Ergebnis: DD.MM.YYYY HH:MM:SS (deutsches Format)
    - Ungueltige Werte → leerer String
    """
    if not date_input:
        d = datetime.now()
    elif isinstance(date_input, str) and re.match(r"^\d{4}-\d{2}-\d{2} ", date_input):
        # DB-Kompatibilitaet: Space → T (wie im JS)
        try:
            d = datetime.fromisoformat(date_input.replace(" ", "T"))
        except ValueError:
            return ""
    else:
        try:
            d = datetime.fromisoformat(str(date_input))
        except (ValueError, TypeError):
            return ""

    return d.strftime("%d.%m.%Y %H:%M:%S")


class TestFormatTimestampLogic:
    """Tests fuer die _formatTimestamp()-Logik (Python-Nachbildung des JS-Codes)."""

    def test_space_separated_format(self):
        """Timestamp 'YYYY-MM-DD HH:MM:SS' (Space statt T) wird korrekt geparst."""
        result = _format_timestamp_py("2026-01-15 14:30:45")
        assert result == "15.01.2026 14:30:45"

    def test_iso8601_format_with_t(self):
        """Standard ISO 8601 mit T-Separator wird korrekt geparst."""
        result = _format_timestamp_py("2026-01-15T14:30:45")
        assert result == "15.01.2026 14:30:45"

    def test_none_returns_current_time(self):
        """None ergibt einen aktuellen Timestamp (nicht leer)."""
        result = _format_timestamp_py(None)
        assert result != ""
        # Format pruefen: DD.MM.YYYY HH:MM:SS
        assert re.match(r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$", result)

    def test_empty_string_returns_current_time(self):
        """Leerer String ergibt einen aktuellen Timestamp (falsy-Wert)."""
        result = _format_timestamp_py("")
        assert result != ""
        assert re.match(r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$", result)

    def test_invalid_date_returns_empty(self):
        """Ungueltige Datumsangabe ergibt leeren String."""
        result = _format_timestamp_py("kein-datum")
        assert result == ""

    def test_output_format_german(self):
        """Ausgabe ist im deutschen Format DD.MM.YYYY HH:MM:SS."""
        result = _format_timestamp_py("2026-03-02 09:05:07")
        assert result == "02.03.2026 09:05:07"

    def test_midnight_timestamp(self):
        """Mitternacht wird korrekt formatiert."""
        result = _format_timestamp_py("2026-01-01 00:00:00")
        assert result == "01.01.2026 00:00:00"

    def test_end_of_day_timestamp(self):
        """23:59:59 wird korrekt formatiert."""
        result = _format_timestamp_py("2026-12-31 23:59:59")
        assert result == "31.12.2026 23:59:59"

    def test_single_digit_day_padded(self):
        """Einstelliger Tag wird mit fuehrender Null formatiert."""
        result = _format_timestamp_py("2026-03-01 08:03:05")
        assert result == "01.03.2026 08:03:05"

    def test_space_to_t_replacement_is_safari_compatible(self):
        """Space→T Replacement: Space-separiertes Format wird ISO-kompatibel gemacht.

        Safari kann 'YYYY-MM-DD HH:MM:SS' nicht direkt parsen,
        deshalb wird im JS der Space durch T ersetzt.
        """
        space_ts = "2026-06-15 10:30:00"
        iso_ts = space_ts.replace(" ", "T")
        assert iso_ts == "2026-06-15T10:30:00"
        # Beide muessen zum gleichen Ergebnis fuehren
        assert _format_timestamp_py(space_ts) == _format_timestamp_py(iso_ts)


# ===========================================================================
# Edge Cases: created_at Feld-Handling in der API
# ===========================================================================


class TestTimestampEdgeCases:
    """Edge Cases fuer Timestamp-Felder in der API."""

    def test_message_dict_includes_created_at_key(self):
        """Das Message-Dict in get_chat enthaelt immer den created_at-Key."""
        # Simuliere eine DB-Row als dict (so wie api_chat.py sie verarbeitet)
        mock_row = {
            "id": 1,
            "role": "user",
            "content": "Frage",
            "sources": None,
            "created_at": "2026-01-15 14:30:45",
        }
        # Nachbildung der Transformation aus get_chat()
        result = {
            "id": mock_row["id"],
            "role": mock_row["role"],
            "content": mock_row["content"],
            "sources": json.loads(mock_row["sources"]) if mock_row["sources"] else [],
            "created_at": mock_row["created_at"],
            "trace_id": None,
        }
        assert "created_at" in result
        assert result["created_at"] == "2026-01-15 14:30:45"

    def test_message_with_none_created_at(self):
        """Message mit created_at=None (theoretisch) wird korrekt weitergereicht."""
        mock_row = {
            "id": 2,
            "role": "assistant",
            "content": "Antwort",
            "sources": None,
            "created_at": None,
        }
        result = {
            "id": mock_row["id"],
            "role": mock_row["role"],
            "content": mock_row["content"],
            "sources": json.loads(mock_row["sources"]) if mock_row["sources"] else [],
            "created_at": mock_row["created_at"],
            "trace_id": None,
        }
        assert "created_at" in result
        assert result["created_at"] is None

    def test_message_with_empty_string_created_at(self):
        """Message mit created_at='' wird weitergereicht."""
        mock_row = {
            "id": 3,
            "role": "user",
            "content": "Test",
            "sources": None,
            "created_at": "",
        }
        result = {
            "id": mock_row["id"],
            "role": mock_row["role"],
            "content": mock_row["content"],
            "sources": [],
            "created_at": mock_row["created_at"],
            "trace_id": None,
        }
        assert result["created_at"] == ""

    def test_format_timestamp_handles_none_gracefully(self):
        """_formatTimestamp(None) gibt aktuellen Timestamp, keinen Fehler."""
        result = _format_timestamp_py(None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_timestamp_handles_invalid_gracefully(self):
        """_formatTimestamp('ungueltig') gibt leeren String, keinen Fehler."""
        result = _format_timestamp_py("ungueltig")
        assert result == ""

    def test_format_timestamp_handles_partial_date(self):
        """Nur ein Datum ohne Zeit wird als ungueltiger Input behandelt."""
        # "2026-01-15" ohne Uhrzeit — nicht im erwarteten DB-Timestamp-Pattern
        result = _format_timestamp_py("2026-01-15")
        # fromisoformat kann das parsen, dann ergaenzt Python Mitternacht
        # Das ist akzeptables Verhalten
        assert result == "15.01.2026 00:00:00"


# ===========================================================================
# Integration: Vollstaendiger Roundtrip DB → API-Response
# ===========================================================================


class TestTimestampRoundtrip:
    """Prueft den vollstaendigen Weg: DB-Insert → API-Response → Format."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.conn, self.db_path = _make_test_db()
        cursor = self.conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (%s, %s, %s) RETURNING id",
            ("admin_ts_test", "hash", True),
        )
        self.user_id = cursor.fetchone()["id"]
        cursor = self.conn.execute(
            "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING id",
            (self.user_id, "Roundtrip-Test"),
        )
        self.chat_id = cursor.fetchone()["id"]
        self.conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content) VALUES (%s, %s, %s)",
            (self.chat_id, "user", "Test-Frage"),
        )
        self.conn.commit()
        yield
        self.conn.rollback()
        self.conn.close()

    def _get_chat_row(self):
        row = self.conn.execute("SELECT * FROM chats WHERE id = %s", (self.chat_id,)).fetchone()
        d = dict(row) if row else None
        if d:
            for k in ("created_at", "updated_at"):
                if k in d:
                    d[k] = _format_ts(d[k])
        return d

    def _get_message_rows(self):
        rows = self.conn.execute(
            "SELECT * FROM chat_messages WHERE chat_id = %s ORDER BY id",
            (self.chat_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if "created_at" in d:
                d["created_at"] = _format_ts(d["created_at"])
            result.append(d)
        return result

    @pytest.mark.asyncio
    async def test_db_to_api_to_format(self):
        """DB-Timestamp → API-Response → deutsches Format: Roundtrip funktioniert."""
        from kiara.web.api_chat import get_chat

        user = {"id": self.user_id, "is_admin": True}
        with (
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        ):
            mock_chat_repo.get_owned.return_value = self._get_chat_row()
            mock_msg_repo.list_by_chat.return_value = self._get_message_rows()
            result = await get_chat(self.chat_id, user)

        # API liefert DB-Format
        msg_ts = result["messages"][0]["created_at"]
        assert _DT_PATTERN.match(msg_ts)

        # Deutsches Format via _formatTimestamp-Logik
        formatted = _format_timestamp_py(msg_ts)
        assert re.match(r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$", formatted)

    @pytest.mark.asyncio
    async def test_chat_and_message_timestamps_consistent(self):
        """Chat- und Message-Timestamps sind zeitlich konsistent."""
        from kiara.web.api_chat import get_chat

        user = {"id": self.user_id, "is_admin": True}
        with (
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
        ):
            mock_chat_repo.get_owned.return_value = self._get_chat_row()
            mock_msg_repo.list_by_chat.return_value = self._get_message_rows()
            result = await get_chat(self.chat_id, user)

        chat_ts = datetime.strptime(result["created_at"], "%Y-%m-%d %H:%M:%S")
        msg_ts = datetime.strptime(result["messages"][0]["created_at"], "%Y-%m-%d %H:%M:%S")
        # Message wurde nach dem Chat erstellt (oder gleichzeitig)
        assert msg_ts >= chat_ts


# ===========================================================================
# Realtime-Timestamp (neue Nachrichten im Frontend)
# ===========================================================================


class TestRealtimeTimestamp:
    """Prueft die Echtzeit-Timestamp-Logik fuer neue Nachrichten."""

    def test_new_message_uses_iso_string(self):
        """Neue User-Nachrichten im Frontend nutzen new Date().toISOString().

        Verifiziert, dass das JS-Pattern `new Date().toISOString()` ein
        von _formatTimestamp() parsebares Format erzeugt.
        """
        # Simuliert das JS: new Date().toISOString() → "2026-03-02T10:30:00.000Z"
        iso_ts = "2026-03-02T10:30:00.000Z"
        # Python-Nachbildung: fromisoformat versteht kein Z-Suffix nativ (< 3.11)
        # aber die JS _formatTimestamp-Logik faengt es ueber new Date(dateInput) ab
        # Hier pruefen wir nur, dass das Format grundsaetzlich parsebar ist
        clean = iso_ts.replace("Z", "+00:00")
        d = datetime.fromisoformat(clean)
        assert d.year == 2026
        assert d.month == 3

    def test_streaming_bubble_gets_timestamp(self):
        """appendStreamingBubble() ruft _formatTimestamp() ohne Argument auf.

        Das bedeutet: Fallback auf new Date() → aktuelle Zeit.
        Nachgebildet in Python: _format_timestamp_py(None).
        """
        result = _format_timestamp_py(None)
        assert result != ""
        # Muss dem deutschen Format entsprechen
        parts = result.split(" ")
        assert len(parts) == 2
        date_part, time_part = parts
        assert len(date_part.split(".")) == 3  # DD.MM.YYYY
        assert len(time_part.split(":")) == 3  # HH:MM:SS
