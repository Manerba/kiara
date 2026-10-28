"""Tests fuer Sprint 06: KI-Backend-CRUD Admin-Funktionen.

Testet ki.backends.create/update/delete/toggle/test — 13 Unit-Tests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# ki.backends.create
# ---------------------------------------------------------------------------


class TestKiBackendsCreate:
    @patch("kiara.admin.services.ki_backend_service._invalidate_ddu")
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.create")
    @patch("kiara.shared.crypto.encrypt")
    def test_create_backend(self, mock_encrypt, mock_create, mock_inv, mock_ddu):
        """Test 23: validate + encrypt + create."""
        mock_encrypt.return_value = "enc_key"
        mock_create.return_value = 10

        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="cloud",
                provider="anthropic",
                model="claude-3",
                api_key="sk-test",
            )
        )

        assert result.success is True
        assert result.data["id"] == 10
        mock_encrypt.assert_called_once_with("sk-test")
        # trusted defaults to 0 (int)
        call_kwargs = mock_create.call_args[1]
        assert not call_kwargs["trusted"]

    @patch("kiara.admin.services.ki_backend_service._invalidate_ddu")
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.create")
    @patch("kiara.shared.crypto.encrypt")
    def test_create_backend_trusted(self, mock_encrypt, mock_create, mock_inv, mock_ddu):
        """trusted=True wird an Repository.create durchgereicht."""
        mock_encrypt.return_value = "enc_key"
        mock_create.return_value = 15

        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="cloud",
                provider="anthropic",
                model="claude-3",
                api_key="sk-test",
                trusted=True,
            )
        )

        assert result.success is True
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["trusted"]  # int(bool(True))

    def test_create_backend_invalid_category(self):
        """Test 24: ungueltige Kategorie → fail."""
        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="invalid",
                provider="ollama",
                model="test",
            )
        )
        assert result.success is False
        assert "Ungueltige Kategorie" in result.message

    def test_create_backend_invalid_provider(self):
        """Test 25: falscher Provider fuer Kategorie → fail."""
        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="embedding",
                provider="anthropic",
                model="test",
            )
        )
        assert result.success is False
        assert "Ungueltiger Provider" in result.message

    def test_create_backend_ollama_requires_url(self):
        """Ollama-Backends brauchen beim Anlegen eine URL."""
        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="llm",
                provider="ollama",
                model="llama3",
            )
        )
        assert result.success is False
        assert "url" in result.message

    def test_create_backend_cloud_requires_api_key(self):
        """Cloud-Provider brauchen beim Anlegen einen echten API-Key."""
        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="cloud",
                provider="openai",
                model="gpt-4o",
            )
        )
        assert result.success is False
        assert "api_key" in result.message

    def test_create_backend_cloud_rejects_password_mask_as_new_key(self):
        """PASSWORD_MASK ist beim Anlegen kein gueltiger API-Key."""
        from kiara.admin.functions.ki import ki_backends_create
        from kiara.shared.constants import PASSWORD_MASK

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="cloud",
                provider="anthropic",
                model="claude-3",
                api_key=PASSWORD_MASK,
            )
        )

        assert result.success is False
        assert "api_key" in result.message

    def test_create_backend_ddu_requires_url(self):
        """DDU-Backends brauchen beim Anlegen eine Remote-URL."""
        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="ddu",
                provider="docling",
            )
        )
        assert result.success is False
        assert "url" in result.message

    @patch("kiara.admin.services.ki_backend_service._invalidate_ddu")
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.create")
    def test_create_backend_ddu_auto_model(self, mock_create, mock_inv, mock_ddu):
        """Test 26: category=ddu → model='docling'."""
        mock_create.return_value = 11

        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="ddu",
                provider="docling",
                url="http://ddu:5001",
            )
        )

        assert result.success is True
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "docling"

    @patch("kiara.admin.services.ki_backend_service._fetch_model_info", new_callable=AsyncMock)
    @patch("kiara.admin.services.ki_backend_service._invalidate_ddu")
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.create")
    def test_create_backend_ollama_fetch_info(
        self, mock_create, mock_update, mock_inv, mock_ddu, mock_fetch
    ):
        """Test 27: _fetch_model_info aufgerufen."""
        mock_create.return_value = 12
        mock_fetch.return_value = {"num_ctx": 4096, "embedding_dim": 768}

        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="embedding",
                provider="ollama",
                url="http://localhost:11434",
                model="nomic-embed-text",
            )
        )

        assert result.success is True
        mock_fetch.assert_called_once()
        # extra_config mit num_ctx/embedding_dim sollte aktualisiert worden sein
        mock_update.assert_called_once()


# ---------------------------------------------------------------------------
# ki.backends.update
# ---------------------------------------------------------------------------


class TestKiBackendsUpdate:
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_update_backend_password_mask(self, mock_get, mock_update, mock_inv):
        """Test 28: PASSWORD_MASK → bestehender Key bleibt erhalten."""
        from kiara.shared.constants import PASSWORD_MASK

        mock_get.return_value = {
            "id": 3,
            "category": "cloud",
            "provider": "anthropic",
            "name": "Cloud",
            "url": "",
            "base_url": "",
            "model": "claude-3",
            "api_key": "existing_enc_key",
            "is_active": True,
            "priority": 10,
            "trusted": False,
            "extra_config": "{}",
        }

        from kiara.admin.functions.ki import ki_backends_update

        result = _run(
            ki_backends_update(
                user=MOCK_USER,
                backend_id=3,
                api_key=PASSWORD_MASK,
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        # Bei PASSWORD_MASK muss der bestehende Key beibehalten werden
        assert call_kwargs["api_key"] == "existing_enc_key"

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_update_backend_partial(self, mock_get, mock_update, mock_inv):
        """Test 29: nur uebergebene Felder geaendert."""
        mock_get.return_value = {
            "id": 3,
            "category": "llm",
            "provider": "ollama",
            "name": "LLM",
            "url": "http://old:11434",
            "base_url": "",
            "model": "old-model",
            "api_key": "",
            "is_active": True,
            "priority": 10,
            "trusted": False,
            "extra_config": "{}",
        }

        from kiara.admin.functions.ki import ki_backends_update

        result = _run(
            ki_backends_update(
                user=MOCK_USER,
                backend_id=3,
                url="http://new:11434",
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["url"] == "http://new:11434"
        assert call_kwargs["model"] == "old-model"  # Unveraendert

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_update_backend_provider_change_requires_new_key(self, mock_get, mock_update, mock_inv):
        """Providerwechsel zu Key-Provider darf altes Secret nicht still uebernehmen."""
        mock_get.return_value = {
            "id": 3,
            "category": "cloud",
            "provider": "openai",
            "name": "Cloud",
            "url": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key": "existing_enc_key",
            "is_active": True,
            "priority": 10,
            "trusted": False,
            "extra_config": "{}",
        }

        from kiara.admin.functions.ki import ki_backends_update

        result = _run(
            ki_backends_update(
                user=MOCK_USER,
                backend_id=3,
                provider="anthropic",
                model="claude-3",
            )
        )

        assert result.success is False
        assert "Providerwechsel" in result.message
        mock_update.assert_not_called()

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    @patch("kiara.shared.crypto.encrypt")
    def test_update_backend_provider_change_encrypts_new_key(
        self, mock_encrypt, mock_get, mock_update, mock_inv
    ):
        """Providerwechsel mit neuem Key speichert nur den neuen verschluesselten Key."""
        mock_encrypt.return_value = "new_enc_key"
        mock_get.return_value = {
            "id": 3,
            "category": "cloud",
            "provider": "openai",
            "name": "Cloud",
            "url": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key": "existing_enc_key",
            "is_active": True,
            "priority": 10,
            "trusted": False,
            "extra_config": "{}",
        }

        from kiara.admin.functions.ki import ki_backends_update

        result = _run(
            ki_backends_update(
                user=MOCK_USER,
                backend_id=3,
                provider="anthropic",
                base_url="https://api.anthropic.com",
                model="claude-3",
                api_key="sk-ant-new",
            )
        )

        assert result.success is True
        mock_encrypt.assert_called_once_with("sk-ant-new")
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["api_key"] == "new_enc_key"

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_update_backend_provider_change_to_no_key_clears_secret(
        self, mock_get, mock_update, mock_inv
    ):
        """Providerwechsel zu Provider ohne Key loescht das alte Secret."""
        mock_get.return_value = {
            "id": 3,
            "category": "embedding",
            "provider": "openai",
            "name": "Embedding",
            "url": "",
            "base_url": "https://api.openai.com/v1",
            "model": "text-embedding-3-small",
            "api_key": "existing_enc_key",
            "is_active": True,
            "priority": 10,
            "trusted": False,
            "extra_config": "{}",
        }

        from kiara.admin.functions.ki import ki_backends_update

        result = _run(
            ki_backends_update(
                user=MOCK_USER,
                backend_id=3,
                provider="ollama",
                url="http://localhost:11434",
                base_url="",
                model="nomic-embed-text",
            )
        )

        assert result.success is True
        call_kwargs = mock_update.call_args[1]
        assert call_kwargs["provider"] == "ollama"
        assert call_kwargs["api_key"] == ""


# ---------------------------------------------------------------------------
# ki.backends.delete
# ---------------------------------------------------------------------------


class TestKiBackendsDelete:
    @patch("kiara.admin.services.ki_backend_service._invalidate_ddu")
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.delete")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_delete_backend_ddu_hard(self, mock_get, mock_delete, mock_inv, mock_ddu):
        """Test 30: DDU → delete."""
        mock_get.return_value = {
            "id": 7,
            "category": "ddu",
            "url": "",
            "name": "DDU Lokal",
        }

        from kiara.admin.functions.ki import ki_backends_delete

        result = _run(ki_backends_delete(user=MOCK_USER, backend_id=7))

        assert result.success is True
        mock_delete.assert_called_once_with(7)
        assert "geloescht" in result.data["message"]

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.delete")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_delete_backend_hard(self, mock_get, mock_delete, mock_inv):
        """Test 31: non-DDU → delete."""
        mock_get.return_value = {
            "id": 8,
            "category": "llm",
            "url": "http://test:11434",
            "name": "LLM",
        }

        from kiara.admin.functions.ki import ki_backends_delete

        result = _run(ki_backends_delete(user=MOCK_USER, backend_id=8))

        assert result.success is True
        mock_delete.assert_called_once_with(8)


# ---------------------------------------------------------------------------
# ki.backends.toggle
# ---------------------------------------------------------------------------


class TestKiBackendsToggle:
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.set_active")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_toggle_backend(self, mock_get, mock_set, mock_inv):
        """Test 32: set_active + invalidate_cache."""
        mock_get.return_value = {
            "id": 3,
            "category": "llm",
            "name": "LLM",
        }

        from kiara.admin.functions.ki import ki_backends_toggle

        result = _run(
            ki_backends_toggle(
                user=MOCK_USER,
                backend_id=3,
                is_active=False,
            )
        )

        assert result.success is True
        mock_set.assert_called_once_with(3, False)
        mock_inv.assert_called_once_with("llm")
        assert "deaktiviert" in result.data["message"]


# ---------------------------------------------------------------------------
# ki.backends.test
# ---------------------------------------------------------------------------


class TestKiBackendsTest:
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update_test_result")
    @patch("kiara.web.api_ki._test_ollama_connection", new_callable=AsyncMock)
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_test_backend_ollama(self, mock_get, mock_test_conn, mock_update_result, mock_inv):
        """Test 33: _test_ollama_connection aufgerufen."""
        mock_get.return_value = {
            "id": 1,
            "provider": "ollama",
            "category": "llm",
            "url": "http://localhost:11434",
            "model": "qwen3:8b",
            "api_key": "",
            "extra_config": "{}",
        }
        mock_test_conn.return_value = {"success": True, "models": ["qwen3:8b"]}

        from kiara.admin.functions.ki import ki_backends_test

        result = _run(ki_backends_test(user=MOCK_USER, backend_id=1))

        assert result.success is True
        assert "qwen3:8b" in result.data["models"]

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update_test_result")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_test_backend_anthropic(self, mock_get, mock_update_result, mock_inv):
        """Test 34: models.list aufgerufen."""
        mock_get.return_value = {
            "id": 2,
            "provider": "anthropic",
            "category": "cloud",
            "url": "",
            "model": "claude-3",
            "api_key": "plain_key",
            "extra_config": "{}",
        }

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="claude-3-opus")]

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.models.list.return_value = mock_models
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch("kiara.shared.crypto.is_encrypted", return_value=False):
                from kiara.admin.functions.ki import ki_backends_test

                result = _run(ki_backends_test(user=MOCK_USER, backend_id=2))

        assert result.success is True

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update_test_result")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_test_backend_docling_remote(self, mock_get, mock_update_result, mock_inv):
        """Test 35: Health-Check GET /health."""
        mock_get.return_value = {
            "id": 5,
            "provider": "docling",
            "category": "ddu",
            "url": "http://localhost:5001",
            "model": "docling",
            "api_key": "",
            "extra_config": "{}",
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_resp
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_instance

            from kiara.admin.functions.ki import ki_backends_test

            result = _run(ki_backends_test(user=MOCK_USER, backend_id=5))

        assert result.success is True


# ---------------------------------------------------------------------------
# Ergaenzende Error-Path Tests (Coverage-Erweiterung)
# ---------------------------------------------------------------------------


class TestKiBackendsCreateErrorPaths:
    @patch("kiara.admin.services.ki_backend_service._fetch_model_info", new_callable=AsyncMock)
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update")
    @patch("kiara.repositories.backend.KiBackendRepository.create")
    def test_create_ollama_fetch_info_exception_non_fatal(
        self, mock_create, mock_update, mock_inv, mock_fetch
    ):
        """_fetch_model_info Exception ist non-fatal — Backend wird trotzdem erstellt."""
        mock_create.return_value = 20
        mock_fetch.side_effect = ConnectionError("Ollama nicht erreichbar")

        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="llm",
                provider="ollama",
                url="http://unreachable:11434",
                model="qwen3:8b",
            )
        )

        assert result.success is True
        assert result.data["id"] == 20
        mock_create.assert_called_once()
        # update sollte NICHT aufgerufen werden (model_info = None wegen Exception)
        mock_update.assert_not_called()

    def test_create_backend_missing_model(self):
        """Fehlendes Modell bei nicht-DDU-Kategorie → fail."""
        from kiara.admin.functions.ki import ki_backends_create

        result = _run(
            ki_backends_create(
                user=MOCK_USER,
                category="llm",
                provider="ollama",
                url="http://test:11434",
                model="",
            )
        )
        assert result.success is False
        assert "Modell" in result.message


class TestKiBackendsTestErrorPaths:
    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update_test_result")
    @patch("kiara.web.api_ki._test_ollama_connection", new_callable=AsyncMock)
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_test_backend_ollama_connection_failed(
        self, mock_get, mock_test_conn, mock_update_result, mock_inv
    ):
        """Ollama Verbindungsfehler → fail mit Fehlermeldung."""
        mock_get.return_value = {
            "id": 1,
            "provider": "ollama",
            "category": "llm",
            "url": "http://down:11434",
            "model": "qwen3:8b",
            "api_key": "",
            "extra_config": "{}",
        }
        mock_test_conn.return_value = {
            "success": False,
            "models": [],
            "error": "Connection refused",
        }

        from kiara.admin.functions.ki import ki_backends_test

        result = _run(ki_backends_test(user=MOCK_USER, backend_id=1))

        assert result.success is False
        assert "Connection refused" in result.message
        # Testergebnis muss trotzdem gespeichert werden
        mock_update_result.assert_called_once()

    @patch("kiara.backend_selector.invalidate_cache")
    @patch("kiara.repositories.backend.KiBackendRepository.update_test_result")
    @patch("kiara.repositories.backend.KiBackendRepository.get_by_id")
    def test_test_backend_unknown_provider_fallback(self, mock_get, mock_update_result, mock_inv):
        """Unbekannter Provider → Fallback-Fehlermeldung."""
        mock_get.return_value = {
            "id": 99,
            "provider": "unknown_provider",
            "category": "llm",
            "url": "",
            "model": "test",
            "api_key": "",
            "extra_config": "{}",
        }

        from kiara.admin.functions.ki import ki_backends_test

        result = _run(ki_backends_test(user=MOCK_USER, backend_id=99))

        assert result.success is False
        assert "Unbekannter Provider" in result.message

    def test_test_backend_not_found(self):
        """Backend-ID existiert nicht → fail."""
        with patch("kiara.repositories.backend.KiBackendRepository.get_by_id", return_value=None):
            from kiara.admin.functions.ki import ki_backends_test

            result = _run(ki_backends_test(user=MOCK_USER, backend_id=9999))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# _normalize_embedding_dim Regression-Tests
# ---------------------------------------------------------------------------


class TestNormalizeEmbeddingDim:
    def test_normalize_embedding_dim_bge_m3(self):
        """bge-m3 resolves to 1024 from KNOWN_EMBEDDING_DIMENSIONS."""
        from kiara.admin.services.ki_backend_service import _normalize_embedding_dim

        extra = _normalize_embedding_dim({}, "bge-m3")
        assert extra["embedding_dim"] == 1024

    def test_normalize_embedding_dim_bge_m3_with_latest_tag(self):
        """bge-m3:latest also resolves to 1024 (tag stripped by normalize_model_name)."""
        from kiara.admin.services.ki_backend_service import _normalize_embedding_dim

        extra = _normalize_embedding_dim({}, "bge-m3:latest")
        assert extra["embedding_dim"] == 1024

    def test_normalize_embedding_dim_overwrites_stale_value(self):
        """Known model overwrites a stale/wrong embedding_dim in extra."""
        from kiara.admin.services.ki_backend_service import _normalize_embedding_dim

        extra = _normalize_embedding_dim({"embedding_dim": 999}, "bge-m3")
        assert extra["embedding_dim"] == 1024

    def test_normalize_embedding_dim_unknown_model_requires_dim(self):
        """Unknown model without embedding_dim raises ValueError."""
        from kiara.admin.services.ki_backend_service import _normalize_embedding_dim

        with pytest.raises(ValueError, match="Embedding-Dimension unbekannt"):
            _normalize_embedding_dim({}, "totally-unknown-model")
