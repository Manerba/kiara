"""Tests fuer Sprint 09: Cloud Admin-Funktionen.

Testet cloud.config.read, .config.update, .audit, .stats, .test — 9 Unit-Tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kiara.shared.constants import PASSWORD_MASK


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# cloud.config.read
# ---------------------------------------------------------------------------


class TestCloudConfigRead:
    @patch("kiara.config.load_config")
    def test_config_read_with_key(self, mock_config):
        """Test 1: API-Key wird durch PASSWORD_MASK ersetzt."""
        mock_config.return_value = {
            "cloud": {
                "enabled": True,
                "provider": "anthropic",
                "api_key": "sk-ant-secret123",
                "model": "claude-3-opus",
                "base_url": "",
                "max_tokens": 4096,
                "temperature": 0.3,
                "routing": {
                    "mode": "auto",
                    "confidence_threshold": 0.3,
                    "min_relevant_chunks": 2,
                    "cloud_keywords": ["komplex"],
                },
                "budget": {
                    "monthly_limit_eur": 100,
                    "alert_threshold_pct": 80,
                },
            }
        }

        from kiara.admin.functions.cloud import cloud_config_read

        result = _run(cloud_config_read(MOCK_USER))

        assert result.success is True
        cfg = result.data["config"]
        assert cfg["api_key"] == PASSWORD_MASK
        assert cfg["provider"] == "anthropic"
        assert cfg["enabled"] is True
        assert cfg["monthly_limit_eur"] == 100

    @patch("kiara.config.load_config")
    def test_config_read_no_key(self, mock_config):
        """Test 2: Kein API-Key → leerer String (nicht PASSWORD_MASK)."""
        mock_config.return_value = {"cloud": {"api_key": ""}}

        from kiara.admin.functions.cloud import cloud_config_read

        result = _run(cloud_config_read(MOCK_USER))

        assert result.success is True
        assert result.data["config"]["api_key"] == ""


# ---------------------------------------------------------------------------
# cloud.config.update
# ---------------------------------------------------------------------------


class TestCloudConfigUpdate:
    @patch("kiara.config.save_config")
    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.config.load_config")
    def test_config_update_new_key(self, mock_config, mock_encrypt, mock_save):
        """Test 3: Neuer API-Key wird verschluesselt."""
        mock_config.return_value = {}
        mock_encrypt.return_value = "gAAAAABencrypted..."

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(
            cloud_config_update(
                MOCK_USER,
                enabled=True,
                provider="openai",
                api_key="sk-new-key",
                model="gpt-4o",
            )
        )

        assert result.success is True
        mock_encrypt.assert_called_once_with("sk-new-key")
        saved_config = mock_save.call_args[0][0]
        assert saved_config["cloud"]["api_key"] == "gAAAAABencrypted..."

    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_config_update_writes_runtime_routing_mode(self, mock_config, mock_save):
        """Routing-Modus wird top-level fuer generation.cloud_routing gespeichert."""
        mock_config.return_value = {}

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(
            cloud_config_update(
                MOCK_USER,
                enabled=True,
                routing_mode="cloud",
            )
        )

        assert result.success is True
        saved_config = mock_save.call_args[0][0]
        assert saved_config["cloud"]["mode"] == "cloud"
        assert saved_config["cloud"]["routing"]["mode"] == "cloud"

    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_config_update_rejects_old_routing_mode(self, mock_config, mock_save):
        """Alte Routing-Werte werden nicht mehr akzeptiert."""
        mock_config.return_value = {}

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(
            cloud_config_update(
                MOCK_USER,
                enabled=True,
                routing_mode="always",
            )
        )

        assert result.success is False
        mock_save.assert_not_called()

    @patch("kiara.config.save_config")
    @patch("kiara.shared.crypto.is_encrypted")
    @patch("kiara.config.load_config")
    def test_config_update_mask_keeps_existing(self, mock_config, mock_is_enc, mock_save):
        """Test 4: PASSWORD_MASK → bestehendes verschluesseltes PW beibehalten."""
        mock_config.return_value = {
            "cloud": {"api_key": "gAAAAABexisting..."},
        }
        mock_is_enc.return_value = True

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(
            cloud_config_update(
                MOCK_USER,
                api_key=PASSWORD_MASK,
            )
        )

        assert result.success is True
        saved_config = mock_save.call_args[0][0]
        assert saved_config["cloud"]["api_key"] == "gAAAAABexisting..."

    @patch("kiara.config.save_config")
    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.shared.crypto.is_encrypted")
    @patch("kiara.config.load_config")
    def test_config_update_mask_encrypts_plaintext(
        self, mock_config, mock_is_enc, mock_encrypt, mock_save
    ):
        """Test 5: PASSWORD_MASK + Klartext-Key → opportunistische Verschluesselung."""
        mock_config.return_value = {
            "cloud": {"api_key": "sk-plaintext-old"},
        }
        mock_is_enc.return_value = False
        mock_encrypt.return_value = "gAAAAABnewly_encrypted..."

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(
            cloud_config_update(
                MOCK_USER,
                api_key=PASSWORD_MASK,
            )
        )

        assert result.success is True
        mock_encrypt.assert_called_once_with("sk-plaintext-old")

    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_config_update_partial_budget_preserves_existing_cloud_config(
        self,
        mock_config,
        mock_save,
    ):
        """Partieller Update darf Provider, Key und Runtime-Felder nicht loeschen."""
        mock_config.return_value = {
            "cloud": {
                "enabled": True,
                "provider": "anthropic",
                "api_key": "gAAAAABexisting...",
                "model": "claude-sonnet-4-6",
                "base_url": "https://api.anthropic.com",
                "max_tokens": 8192,
                "temperature": 0.2,
                "mode": "cloud",
                "routing": {
                    "mode": "cloud",
                    "confidence_threshold": 0.4,
                    "min_relevant_chunks": 3,
                    "cloud_keywords": ["extern"],
                },
                "budget": {
                    "monthly_limit_eur": 50,
                    "alert_threshold_pct": 70,
                },
            }
        }

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(
            cloud_config_update(
                MOCK_USER,
                monthly_limit_eur=75,
            )
        )

        assert result.success is True
        saved_cloud = mock_save.call_args[0][0]["cloud"]
        assert saved_cloud["enabled"] is True
        assert saved_cloud["provider"] == "anthropic"
        assert saved_cloud["api_key"] == "gAAAAABexisting..."
        assert saved_cloud["model"] == "claude-sonnet-4-6"
        assert saved_cloud["base_url"] == "https://api.anthropic.com"
        assert saved_cloud["max_tokens"] == 8192
        assert saved_cloud["temperature"] == 0.2
        assert saved_cloud["mode"] == "cloud"
        assert saved_cloud["routing"]["mode"] == "cloud"
        assert saved_cloud["routing"]["cloud_keywords"] == ["extern"]
        assert saved_cloud["budget"]["monthly_limit_eur"] == 75
        assert saved_cloud["budget"]["alert_threshold_pct"] == 70

    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_config_update_empty_key_explicitly_clears_existing_key(
        self,
        mock_config,
        mock_save,
    ):
        """Explizit leerer API-Key loescht den Key; omitted wuerde erhalten."""
        mock_config.return_value = {
            "cloud": {
                "enabled": True,
                "provider": "openai",
                "api_key": "gAAAAABexisting...",
            }
        }

        from kiara.admin.functions.cloud import cloud_config_update

        result = _run(cloud_config_update(MOCK_USER, api_key=""))

        assert result.success is True
        saved_cloud = mock_save.call_args[0][0]["cloud"]
        assert saved_cloud["api_key"] == ""
        assert saved_cloud["enabled"] is True
        assert saved_cloud["provider"] == "openai"


# ---------------------------------------------------------------------------
# cloud.audit
# ---------------------------------------------------------------------------


class TestCloudAudit:
    @patch("kiara.shared.db.db_query")
    @patch("kiara.cloud.audit.AuditLogger.get_log")
    @patch("kiara.config.load_config")
    def test_cloud_audit(self, mock_config, mock_log, mock_db):
        """Test 6: Cloud-Audit-Log paginiert."""
        mock_config.return_value = {}
        mock_log.return_value = [
            {"id": 1, "provider": "openai", "tokens_in": 100},
        ]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.execute.return_value.fetchone.return_value = {"count": 42}
        mock_db.return_value = mock_ctx

        from kiara.admin.functions.cloud import cloud_audit

        result = _run(cloud_audit(MOCK_USER, page=2, per_page=10))

        assert result.success is True
        assert result.data["page"] == 2
        assert len(result.data["data"]) == 1
        mock_log.assert_called_once_with(limit=10, offset=10, user_id=None, date_from=None)


# ---------------------------------------------------------------------------
# cloud.stats
# ---------------------------------------------------------------------------


class TestCloudStats:
    @patch("kiara.cloud.audit.AuditLogger.get_stats")
    @patch("kiara.config.load_config")
    def test_cloud_stats(self, mock_config, mock_stats):
        """Test 7: Cloud-Nutzungsstatistiken."""
        mock_config.return_value = {}
        mock_stats.return_value = {
            "period": "month",
            "request_count": 100,
            "total_tokens_in": 50000,
            "total_tokens_out": 30000,
            "estimated_cost_eur": 12.50,
            "budget_limit_eur": 50,
            "budget_used_pct": 25.0,
        }

        from kiara.admin.functions.cloud import cloud_stats

        result = _run(cloud_stats(MOCK_USER, period="month"))

        assert result.success is True
        assert result.data["request_count"] == 100
        assert result.data["estimated_cost_eur"] == 12.50


# ---------------------------------------------------------------------------
# cloud.test
# ---------------------------------------------------------------------------


class TestCloudTest:
    def test_cloud_test_no_key(self):
        """Test 8: Leerer API-Key → Fehler."""
        from kiara.admin.functions.cloud import cloud_test

        result = _run(cloud_test(MOCK_USER, api_key=""))

        assert result.success is False
        assert "erforderlich" in result.message

    @patch("kiara.cloud.client.CloudKIClient.generate", new_callable=AsyncMock)
    def test_cloud_test_success(self, mock_generate):
        """Test 9: Erfolgreicher Cloud-Verbindungstest."""
        mock_generate.return_value = "OK"

        from kiara.admin.functions.cloud import cloud_test

        result = _run(
            cloud_test(
                MOCK_USER,
                api_key="sk-test",
                provider="openai",
            )
        )

        assert result.success is True
        assert result.data["success"] is True
        assert result.data["response"] == "OK"
