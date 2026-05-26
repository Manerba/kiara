"""Tests fuer Sprint 06: KI-Config Admin-Funktionen (Chat-only).

Testet ki.config.update, .test_cloud, .test_anonymize,
.test_connection, ki.automaton.rebuild — 10 Unit-Tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# ki.config.update
# ---------------------------------------------------------------------------


class TestKiConfigUpdate:
    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_partial(self, mock_get, mock_set, mock_sync):
        """Test 36: Merge mit aktueller Config."""
        mock_get.return_value = {
            "embedding_url": "http://old:11434",
            "embedding_model": "nomic",
            "llm_url": "http://old:11434",
            "llm_model": "old-model",
            "cloud_api_key": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                llm_model="qwen3:8b",
            )
        )

        assert result.success is True
        assert "1 Felder" in result.message
        # Verify set was called with merged config
        call_args = mock_set.call_args[0]
        assert call_args[0] == "ai_config"

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_validate_prompt(self, mock_get, mock_set, mock_sync):
        """Test 37: prompt_template ohne {context} — Validierung wurde entfernt,
        Feld wird akzeptiert (success=True)."""
        mock_get.return_value = {"cloud_api_key": ""}

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                prompt_template="Kein Kontext hier",
            )
        )

        assert result.success is True

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_validate_agent(self, mock_get, mock_set, mock_sync):
        """Test 38: agent_system_prompt ohne {tools_prompt} — Validierung wurde entfernt,
        Feld wird akzeptiert (success=True)."""
        mock_get.return_value = {"cloud_api_key": ""}

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                agent_system_prompt="Kein Tools-Platzhalter",
            )
        )

        assert result.success is True

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_password_mask(self, mock_get, mock_set, mock_sync):
        """Test 39: PASSWORD_MASK → bestehender Key bleibt erhalten."""
        from kiara.shared.constants import PASSWORD_MASK
        from kiara.shared.crypto import encrypt as real_encrypt

        # Bestehender Key ist bereits verschluesselt (Fernet-Format)
        encrypted_key = real_encrypt("test-api-key")
        mock_get.return_value = {
            "cloud_api_key": encrypted_key,
            "llm_url": "",
            "llm_model": "",
            "embedding_url": "",
            "embedding_model": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                cloud_api_key=PASSWORD_MASK,
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        saved = call_args[1]
        # Bei PASSWORD_MASK muss der bestehende verschluesselte Key beibehalten werden
        assert saved["cloud_api_key"] == encrypted_key

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_password_mask_without_existing_key_fails(
        self, mock_get, mock_set, mock_sync
    ):
        """PASSWORD_MASK darf keine initiale Cloud-Konfig ohne echten Key erzeugen."""
        from kiara.shared.constants import PASSWORD_MASK

        mock_get.return_value = {
            "cloud_api_key": "",
            "llm_url": "",
            "llm_model": "",
            "embedding_url": "",
            "embedding_model": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                cloud_api_key=PASSWORD_MASK,
            )
        )

        assert result.success is False
        assert "PASSWORD_MASK" in result.message
        mock_sync.assert_not_called()
        mock_set.assert_not_called()

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_rejects_old_cloud_mode(self, mock_get, mock_set, mock_sync):
        """ki.config.update akzeptiert nur auto/local/cloud."""
        mock_get.return_value = {"cloud_api_key": ""}

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                cloud_mode="fallback",
            )
        )

        assert result.success is False
        mock_sync.assert_not_called()
        mock_set.assert_not_called()

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_yaml_sync(self, mock_get, mock_set, mock_sync):
        """Test 40: _sync_to_yaml aufgerufen."""
        mock_get.return_value = {
            "cloud_api_key": "",
            "llm_url": "",
            "llm_model": "",
            "embedding_url": "",
            "embedding_model": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                llm_model="qwen3:8b",
            )
        )

        assert result.success is True
        mock_sync.assert_called_once()


# ---------------------------------------------------------------------------
# ki.config.test_cloud
# ---------------------------------------------------------------------------


class TestKiConfigTestCloud:
    def test_config_test_cloud_anthropic(self):
        """Test 41: Anthropic models.list."""
        mock_module = MagicMock()
        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="claude-3")]
        mock_module.Anthropic.return_value.models.list.return_value = mock_models

        with patch.dict("sys.modules", {"anthropic": mock_module}):
            from kiara.admin.functions.ki import ki_config_test_cloud

            result = _run(
                ki_config_test_cloud(
                    user=MOCK_USER,
                    provider="anthropic",
                    api_key="sk-test",
                )
            )

        assert result.success is True
        assert "claude-3" in result.data["models"]

    def test_config_test_cloud_openai(self):
        """Test 42: OpenAI models.list."""
        mock_module = MagicMock()
        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="gpt-4")]
        mock_module.OpenAI.return_value.models.list.return_value = mock_models

        with patch.dict("sys.modules", {"openai": mock_module}):
            from kiara.admin.functions.ki import ki_config_test_cloud

            result = _run(
                ki_config_test_cloud(
                    user=MOCK_USER,
                    provider="openai",
                    api_key="sk-test",
                )
            )

        assert result.success is True
        assert "gpt-4" in result.data["models"]


# ---------------------------------------------------------------------------
# ki.config.test_anonymize
# ---------------------------------------------------------------------------


class TestKiConfigTestAnonymize:
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources")
    @patch("kiara.config.load_config")
    def test_config_test_anonymize(self, mock_config, mock_build, mock_pseudo_cls):
        """Test 43: Pseudonymizer aufgerufen."""
        mock_config.return_value = {}
        mock_build.return_value = MagicMock()

        mock_pseudo = MagicMock()
        mock_pseudo.anonymize.return_value = "***"
        mock_pseudo.reverse_map = {"***": "Max"}
        mock_pseudo.stats = {"total": 1}
        mock_pseudo_cls.return_value = mock_pseudo

        from kiara.admin.functions.ki import ki_config_test_anonymize

        result = _run(
            ki_config_test_anonymize(
                user=MOCK_USER,
                text="Max Mustermann",
            )
        )

        assert result.success is True
        assert result.data["anonymized"] == "***"


# ---------------------------------------------------------------------------
# ki.config.test_connection
# ---------------------------------------------------------------------------


class TestKiConfigTestConnection:
    @patch("kiara.web.api_ki._test_ollama_connection", new_callable=AsyncMock)
    def test_config_test_connection(self, mock_test):
        """Test 44: _test_ollama_connection."""
        mock_test.return_value = {"success": True, "models": ["qwen3:8b"]}

        from kiara.admin.functions.ki import ki_config_test_connection

        result = _run(
            ki_config_test_connection(
                user=MOCK_USER,
                url="http://localhost:11434",
            )
        )

        assert result.success is True
        assert "qwen3:8b" in result.data["models"]


# ---------------------------------------------------------------------------
# ki.automaton.rebuild
# ---------------------------------------------------------------------------


class TestKiAutomatonRebuild:
    @patch(
        "kiara.web.api_ki._automaton_cache",
        {"automaton": None, "last_built": None, "erp_count": 0, "ldap_count": 0},
    )
    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources")
    @patch("kiara.config.load_config")
    def test_automaton_rebuild(self, mock_config, mock_build):
        """Test 45: build_automaton_from_sources + Cache."""
        mock_config.return_value = {}
        mock_automaton = MagicMock()
        mock_automaton.entry_count = 42
        mock_build.return_value = mock_automaton

        from kiara.admin.functions.ki import ki_automaton_rebuild

        result = _run(ki_automaton_rebuild(user=MOCK_USER))

        assert result.success is True
        assert result.data["entry_count"] == 42
        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# Ergaenzende Edge-Case Tests (Coverage-Erweiterung)
# ---------------------------------------------------------------------------


class TestKiConfigUpdateEdgeCases:
    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_bool_to_int_conversion(self, mock_get, mock_set, mock_sync):
        """Bool-Felder muessen vor DB-Speicherung zu int konvertiert werden."""
        mock_get.return_value = {
            "cloud_api_key": "",
            "llm_url": "",
            "llm_model": "",
            "embedding_url": "",
            "embedding_model": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                cloud_enabled=True,
                anon_enabled=False,
                anon_llm_enabled=True,
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        saved = call_args[1]
        # Bool-Felder muessen als int gespeichert werden (DB-Konvention)
        assert saved["cloud_enabled"]
        assert not saved["anon_enabled"]
        assert saved["anon_llm_enabled"]
        assert not isinstance(saved["cloud_enabled"], bool)

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_new_api_key_encrypted(self, mock_get, mock_set, mock_sync):
        """Neuer API-Key (nicht PASSWORD_MASK) wird verschluesselt gespeichert."""
        mock_get.return_value = {
            "cloud_api_key": "",
            "llm_url": "",
            "llm_model": "",
            "embedding_url": "",
            "embedding_model": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                cloud_api_key="sk-ant-new-key-12345",
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        saved = call_args[1]
        # Neuer Key darf NICHT im Klartext gespeichert werden
        assert saved["cloud_api_key"] != "sk-ant-new-key-12345"
        assert saved["cloud_api_key"] != ""
        # Muss ein Fernet-Token sein (beginnt mit gAAAAA)
        assert saved["cloud_api_key"].startswith("gAAAAA")

    @patch("kiara.web.api_ki._sync_to_yaml")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_config_update_empty_api_key_clears(self, mock_get, mock_set, mock_sync):
        """Leerer API-Key loescht den bestehenden Key."""
        from kiara.shared.crypto import encrypt as real_encrypt

        mock_get.return_value = {
            "cloud_api_key": real_encrypt("old-key"),
            "llm_url": "",
            "llm_model": "",
            "embedding_url": "",
            "embedding_model": "",
        }

        from kiara.admin.functions.ki import ki_config_update

        result = _run(
            ki_config_update(
                user=MOCK_USER,
                cloud_api_key="",
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        saved = call_args[1]
        assert saved["cloud_api_key"] == ""


class TestKiAutomatonRebuildEdgeCases:
    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources")
    @patch("kiara.config.load_config")
    def test_automaton_rebuild_build_exception(self, mock_config, mock_build):
        """build_automaton_from_sources Exception → fail (nicht crash)."""
        mock_config.return_value = {}
        mock_build.side_effect = RuntimeError("Stammdaten nicht lesbar")

        from kiara.admin.functions.ki import ki_automaton_rebuild

        result = _run(ki_automaton_rebuild(user=MOCK_USER))

        assert result.success is False
        assert "fehlgeschlagen" in result.message
