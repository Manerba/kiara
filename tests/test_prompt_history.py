"""Tests fuer PromptTemplateRepository: save_history() + get_history().

Testet History-Methoden des Prompt-Repos (Sprint 16.1).
"""

import asyncio
import hashlib
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# save_history
# ---------------------------------------------------------------------------


class TestSaveHistory:
    @patch("kiara_db.repositories.prompt.PromptTemplateRepository._write")
    def test_save_history_returns_id(self, mock_write):
        """Test 5: save_history speichert alte Version und gibt ID zurueck."""
        mock_db = MagicMock()
        mock_row = {"id": 42}
        mock_db.execute.return_value.fetchone.return_value = mock_row
        mock_write.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_write.return_value.__exit__ = MagicMock(return_value=False)

        from kiara_db.repositories.prompt import PromptTemplateRepository

        repo = PromptTemplateRepository()
        result = repo.save_history(
            1, "alter content", cloud_system_prompt="sys", changed_by="admin"
        )

        assert result == 42
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        assert "INSERT INTO prompt_template_history" in call_args[0][0]

    @patch("kiara_db.repositories.prompt.PromptTemplateRepository._write")
    def test_save_history_content_hash_is_sha256(self, mock_write):
        """Test 6: content_hash ist SHA256 des Contents."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchone.return_value = {"id": 1}
        mock_write.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_write.return_value.__exit__ = MagicMock(return_value=False)

        from kiara_db.repositories.prompt import PromptTemplateRepository

        repo = PromptTemplateRepository()
        content = "Test-Content fuer Hash"
        repo.save_history(1, content)

        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        call_args = mock_db.execute.call_args[0][1]
        # Tuple: (template_id, content, content_hash, cloud_system_prompt, changed_by, change_reason)
        assert call_args[2] == expected_hash

    @patch("kiara_db.repositories.prompt.PromptTemplateRepository._write")
    def test_save_history_with_external_conn(self, mock_write):
        """Test 21: save_history mit externer conn."""
        mock_tx = MagicMock()
        mock_tx.execute.return_value.fetchone.return_value = {"id": 7}

        # _write mit externem conn soll den externen conn durchreichen
        mock_write.return_value.__enter__ = MagicMock(return_value=mock_tx)
        mock_write.return_value.__exit__ = MagicMock(return_value=False)

        from kiara_db.repositories.prompt import PromptTemplateRepository

        repo = PromptTemplateRepository()
        result = repo.save_history(1, "content", conn=mock_tx)

        # _write wurde mit conn aufgerufen
        mock_write.assert_called_once_with(mock_tx)


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    @patch("kiara_db.repositories.prompt.PromptTemplateRepository._read")
    def test_get_history_ordered(self, mock_read):
        """Test 7: sortiert nach created_at DESC, id DESC."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [
            {
                "id": 3,
                "template_id": 1,
                "content": "v3",
                "content_hash": "h3",
                "cloud_system_prompt": "",
                "changed_by": "admin",
                "change_reason": "",
                "created_at": "2026-04-03",
            },
            {
                "id": 2,
                "template_id": 1,
                "content": "v2",
                "content_hash": "h2",
                "cloud_system_prompt": "",
                "changed_by": "admin",
                "change_reason": "",
                "created_at": "2026-04-02",
            },
        ]
        mock_read.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_read.return_value.__exit__ = MagicMock(return_value=False)

        from kiara_db.repositories.prompt import PromptTemplateRepository

        repo = PromptTemplateRepository()
        entries = repo.get_history(1)

        assert len(entries) == 2
        assert entries[0]["id"] == 3  # Neuester zuerst
        # SQL enthaelt ORDER BY created_at DESC, id DESC
        sql = mock_db.execute.call_args[0][0]
        assert "ORDER BY created_at DESC, id DESC" in sql

    @patch("kiara_db.repositories.prompt.PromptTemplateRepository._read")
    def test_get_history_limit(self, mock_read):
        """Test 8: limit begrenzt Ergebnismenge."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [
            {
                "id": 1,
                "template_id": 1,
                "content": "v1",
                "content_hash": "h1",
                "cloud_system_prompt": "",
                "changed_by": "",
                "change_reason": "",
                "created_at": "2026-04-01",
            },
        ]
        mock_read.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_read.return_value.__exit__ = MagicMock(return_value=False)

        from kiara_db.repositories.prompt import PromptTemplateRepository

        repo = PromptTemplateRepository()
        entries = repo.get_history(1, limit=1)

        assert len(entries) == 1
        call_args = mock_db.execute.call_args[0][1]
        assert call_args[1] == 1  # limit-Wert
