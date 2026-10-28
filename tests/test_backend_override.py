"""Tests fuer Backend-Override und Backend-Injection in Chains.

Testet:
    - get_backend_by_id() und create_backend() arbeiten korrekt mit Backend-IDs
    - RAGChain/UnchainedChain arbeiten korrekt mit injizierten Backends
    - create_backend() und _create_chain_backend() behandeln Backend-IDs korrekt
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.backend_selector import get_backend_by_id, invalidate_cache
from kiara.generation.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    id_=1,
    category="llm",
    provider="ollama",
    model="qwen3:8b",
    url="http://gpu-server.local:11434",
    api_key="",
    is_active=True,
    priority=10,
    **overrides,
):
    """Erzeugt ein einzelnes Mock-Backend-Dict."""
    base = {
        "id": id_,
        "category": category,
        "provider": provider,
        "name": f"Backend {id_}",
        "url": url,
        "base_url": "",
        "model": model,
        "api_key": api_key,
        "is_active": is_active,
        "priority": priority,
        "extra_config": "{}",
    }
    base.update(overrides)
    return base


def _base_config():
    """Minimale Config fuer Tests."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "default-model",
            "embedding_model": "nomic-embed-text",
            "prompt_template": "",
        },
        "chat": {
            "unchained": {
                "temperature": 0.8,
                "system_prompt": "",
            },
        },
        "vectordb": {
            "path": "/tmp/test-vectordb",
            "collection": "test",
        },
    }


@pytest.fixture(autouse=True)
def _clear_selector_cache():
    """Cache des Backend-Selectors vor und nach jedem Test leeren."""
    invalidate_cache()
    yield
    invalidate_cache()


# ===================================================================
# A) get_backend_by_id()
# ===================================================================


class TestGetBackendById:
    """Tests fuer kiara.backend_selector.get_backend_by_id()."""

    def test_finds_correct_backend(self):
        """get_backend_by_id() gibt das richtige Backend zurueck."""
        backend = _make_backend(id_=42, model="special-model")

        mock_db = MagicMock()
        mock_row = MagicMock()
        mock_row.__iter__ = lambda self: iter(backend.items())
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.execute.return_value.fetchone.return_value = backend

        with patch("kiara_db.base.db_query", return_value=mock_db):
            result = get_backend_by_id(42)

        assert result is not None
        assert result["id"] == 42
        assert result["model"] == "special-model"

    def test_returns_none_for_unknown_id(self):
        """get_backend_by_id() gibt None bei nicht existierender ID zurueck."""
        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.execute.return_value.fetchone.return_value = None

        with patch("kiara_db.base.db_query", return_value=mock_db):
            result = get_backend_by_id(9999)

        assert result is None

    def test_returns_none_on_db_error(self):
        """get_backend_by_id() gibt None bei DB-Fehler zurueck (kein Crash)."""
        with patch("kiara_db.base.db_query", side_effect=Exception("DB kaputt")):
            result = get_backend_by_id(1)

        assert result is None

    def test_decrypts_api_key(self):
        """get_backend_by_id() entschluesselt den API-Key des Backends."""
        backend = _make_backend(id_=5, api_key="gAAAAA_encrypted")

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.execute.return_value.fetchone.return_value = backend

        with (
            patch("kiara_db.base.db_query", return_value=mock_db),
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-decrypted"),
        ):
            result = get_backend_by_id(5)

        assert result["api_key"] == "sk-decrypted"

    def test_plaintext_api_key_unchanged(self):
        """Klartext-Key wird nicht veraendert."""
        from kiara.backend_selector import _decrypt_backend

        backend = _make_backend(api_key="sk-plain-key")
        with patch("kiara.shared.crypto.is_encrypted", return_value=False):
            result = _decrypt_backend(backend)
        assert result["api_key"] == "sk-plain-key"

    def test_corrupt_encrypted_key_becomes_empty(self):
        """Korrupt verschluesselter Key wird fail-closed auf '' gesetzt."""
        from kiara.backend_selector import _decrypt_backend

        backend = _make_backend(api_key="gAAAAA_bad_token")
        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", side_effect=Exception("InvalidToken")),
        ):
            result = _decrypt_backend(backend)
        assert result["api_key"] == ""

    def test_empty_api_key_stays_empty(self):
        """Leerer api_key wird nicht angefasst."""
        from kiara.backend_selector import _decrypt_backend

        backend = _make_backend(api_key="")
        result = _decrypt_backend(backend)
        assert result["api_key"] == ""

    def test_missing_api_key_defaults_empty(self):
        """Fehlendes api_key-Feld verwendet Default ''."""
        from kiara.backend_selector import _decrypt_backend

        backend = _make_backend()
        del backend["api_key"]
        result = _decrypt_backend(backend)
        assert result.get("api_key", "") == ""


# ===================================================================
# C) RAGChain mit injiziertem Backend
# ===================================================================


class TestRAGChainBackendInjection:
    """Tests fuer RAGChain mit Backend-Injection (Sprint 2 Architektur)."""

    def _patch_chain_deps(self):
        """Gibt Context-Manager zurueck der Retriever/Reranker mockt."""
        return (
            patch("kiara.generation.chain.HybridRetriever"),
            patch("kiara.retrieval.reranker.CrossEncoderReranker"),
            patch("kiara.generation.chain.QueryExpander"),
        )

    def _mock_backend(self, provider="ollama"):
        """Erstellt ein Mock-LLMBackend mit gegebenem Provider."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = provider
        backend.last_usage = {}
        return backend

    def test_ollama_backend_injection(self):
        """RAGChain akzeptiert ein injiziertes Ollama-Backend."""
        backend = self._mock_backend(provider="ollama")

        p_retr, p_rerank, p_expander = self._patch_chain_deps()
        with p_retr, p_rerank, p_expander:
            from kiara.generation.chain import RAGChain

            chain = RAGChain(_base_config(), backend=backend)

        assert chain._backend is backend
        assert chain._backend.provider == "ollama"

    def test_cloud_backend_injection(self):
        """RAGChain akzeptiert ein injiziertes Cloud-Backend (anthropic)."""
        backend = self._mock_backend(provider="anthropic")

        p_retr, p_rerank, p_expander = self._patch_chain_deps()
        with p_retr, p_rerank, p_expander:
            from kiara.generation.chain import RAGChain

            chain = RAGChain(_base_config(), backend=backend)

        assert chain._backend is backend
        assert chain._backend.provider == "anthropic"

    def test_backend_accessible_via_property(self):
        """Das injizierte Backend ist ueber chain.backend erreichbar."""
        backend = self._mock_backend(provider="openai")

        p_retr, p_rerank, p_expander = self._patch_chain_deps()
        with p_retr, p_rerank, p_expander:
            from kiara.generation.chain import RAGChain

            chain = RAGChain(_base_config(), backend=backend)

        assert chain.backend is backend

    def test_system_prompt_stored(self):
        """Der system_prompt Parameter wird korrekt gespeichert."""
        backend = self._mock_backend()

        p_retr, p_rerank, p_expander = self._patch_chain_deps()
        with p_retr, p_rerank, p_expander:
            from kiara.generation.chain import RAGChain

            chain = RAGChain(
                _base_config(), backend=backend, system_prompt="Du bist ein Assistent."
            )

        assert chain._system_prompt == "Du bist ein Assistent."


# ===================================================================
# D) UnchainedChain mit injiziertem Backend
# ===================================================================


class TestUnchainedChainBackendInjection:
    """Tests fuer UnchainedChain mit Backend-Injection (Sprint 2 Architektur)."""

    def _mock_backend(self, provider="ollama"):
        """Erstellt ein Mock-LLMBackend mit gegebenem Provider."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = provider
        backend.last_usage = {}
        return backend

    def test_ollama_backend_injection(self):
        """UnchainedChain akzeptiert ein injiziertes Ollama-Backend."""
        backend = self._mock_backend(provider="ollama")

        from kiara.generation.unchained_chain import UnchainedChain

        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="Test-Prompt")

        assert chain._backend is backend
        assert chain._backend.provider == "ollama"

    def test_cloud_backend_injection(self):
        """UnchainedChain akzeptiert ein injiziertes Cloud-Backend (openai)."""
        backend = self._mock_backend(provider="openai")

        from kiara.generation.unchained_chain import UnchainedChain

        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="Test-Prompt")

        assert chain._backend is backend
        assert chain._backend.provider == "openai"

    def test_route_used_local_for_ollama(self):
        """route_used gibt 'local' fuer Ollama-Backend zurueck."""
        backend = self._mock_backend(provider="ollama")

        from kiara.generation.unchained_chain import UnchainedChain

        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="Test")

        assert chain.route_used == "local"

    def test_route_used_cloud_for_anthropic(self):
        """route_used gibt 'cloud' fuer Anthropic-Backend zurueck."""
        backend = self._mock_backend(provider="anthropic")

        from kiara.generation.unchained_chain import UnchainedChain

        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="Test")

        assert chain.route_used == "cloud"

    def test_system_prompt_stored(self):
        """Der system_prompt Parameter wird korrekt gespeichert."""
        backend = self._mock_backend()

        from kiara.generation.unchained_chain import UnchainedChain

        chain = UnchainedChain(
            _base_config(), backend=backend, system_prompt="Du bist ein Assistent."
        )

        assert chain._system_prompt == "Du bist ein Assistent."

    def test_backend_accessible_via_property(self):
        """Das injizierte Backend ist ueber chain.backend erreichbar."""
        backend = self._mock_backend(provider="openai")

        from kiara.generation.unchained_chain import UnchainedChain

        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="Test")

        assert chain.backend is backend


# ===================================================================
# F) create_backend() und _create_chain_backend() Factory-Tests
# ===================================================================


class TestCreateBackendFactory:
    """Tests fuer create_backend() aus kiara.generation.backends.factory."""

    def test_valid_backend_id_creates_backend(self):
        """create_backend() mit gueltiger ID erstellt das passende Backend."""
        backend_dict = _make_backend(
            id_=7, provider="ollama", model="test-model", url="http://test:11434"
        )

        with (
            patch("kiara.backend_selector.get_backend_by_id", return_value=backend_dict),
            patch("kiara.generation.backends.ollama.OllamaBackend") as mock_ollama,
        ):
            from kiara.generation.backends.factory import create_backend

            create_backend(_base_config(), backend_id=7)

        mock_ollama.assert_called_once()

    def test_invalid_backend_id_raises_value_error(self):
        """create_backend() mit unbekannter ID wirft ValueError."""
        with patch("kiara.backend_selector.get_backend_by_id", return_value=None):
            from kiara.generation.backends.factory import create_backend

            with pytest.raises(ValueError, match="nicht gefunden"):
                create_backend(_base_config(), backend_id=9999)

    def test_anthropic_provider_creates_anthropic_backend(self):
        """create_backend() erstellt AnthropicBackend fuer provider=anthropic."""
        backend_dict = _make_backend(
            id_=10, provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test"
        )

        with (
            patch("kiara.backend_selector.get_backend_by_id", return_value=backend_dict),
            patch("kiara.generation.backends.anthropic.AnthropicBackend") as mock_anth,
        ):
            from kiara.generation.backends.factory import create_backend

            create_backend(_base_config(), backend_id=10)

        mock_anth.assert_called_once()
        call_kwargs = mock_anth.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["api_key"] == "sk-test"

    def test_openai_provider_creates_openai_backend(self):
        """create_backend() erstellt OpenAIBackend fuer provider=openai."""
        backend_dict = _make_backend(id_=11, provider="openai", model="gpt-4o", api_key="sk-openai")

        with (
            patch("kiara.backend_selector.get_backend_by_id", return_value=backend_dict),
            patch("kiara.generation.backends.openai.OpenAIBackend") as mock_oai,
        ):
            from kiara.generation.backends.factory import create_backend

            create_backend(_base_config(), backend_id=11)

        mock_oai.assert_called_once()
        call_kwargs = mock_oai.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_no_backend_id_uses_default_ollama(self):
        """create_backend() ohne ID verwendet get_llm_backend() und erstellt OllamaBackend."""
        default_backend = {"url": "http://default:11434", "model": "default-model"}

        with (
            patch("kiara.config.get_llm_backend", return_value=default_backend),
            patch("kiara.generation.backends.ollama.OllamaBackend") as mock_ollama,
        ):
            from kiara.generation.backends.factory import create_backend

            create_backend(_base_config())

        mock_ollama.assert_called_once()
        call_kwargs = mock_ollama.call_args.kwargs
        assert call_kwargs["model"] == "default-model"
        assert call_kwargs["url"] == "http://default:11434"


class TestCreateChainBackend:
    """Tests fuer _create_chain_backend() aus kiara.generation.cloud_routing."""

    def test_invalid_backend_id_raises_value_error(self):
        """_create_chain_backend() mit unbekannter ID wirft ValueError."""
        with patch("kiara.generation.cloud_routing.get_backend_by_id", return_value=None):
            from kiara.generation.cloud_routing import _create_chain_backend

            with pytest.raises(ValueError, match="nicht gefunden"):
                _create_chain_backend(
                    _base_config(), "test frage", llm_backend_id=9999, filter_datasource_ids=None
                )

    def test_explicit_cloud_backend_wraps_with_anonymization(self):
        """Explizites untrusted Cloud-Backend wird mit Anonymisierung gewrappt bei anonymize-Level."""
        cloud_dict = _make_backend(
            id_=10, provider="anthropic", model="claude-sonnet-4-20250514", api_key="sk-test"
        )
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.provider = "anthropic"

        with (
            patch("kiara.generation.cloud_routing.get_backend_by_id", return_value=cloud_dict),
            patch(
                "kiara.generation.cloud_routing.create_backend", return_value=mock_backend
            ) as mock_create,
            patch(
                "kiara.generation.cloud_routing.wrap_with_anonymization", return_value=mock_backend
            ) as mock_wrap,
            patch(
                "kiara.generation.cloud_routing._load_restriction_levels",
                return_value=["anonymize"],
            ),
        ):
            from kiara.generation.cloud_routing import _create_chain_backend

            result = _create_chain_backend(
                _base_config(), "test", llm_backend_id=10, filter_datasource_ids=[1]
            )

        mock_create.assert_called_once_with(_base_config(), 10)
        mock_wrap.assert_called_once()
        assert result is mock_backend

    def test_explicit_trusted_cloud_backend_no_restriction_check(self):
        """Explizites trusted Cloud-Backend ueberspringt Restriction-Pruefung."""
        cloud_dict = _make_backend(
            id_=10,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-test",
            trusted=True,
        )
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.provider = "anthropic"

        with (
            patch("kiara.generation.cloud_routing.get_backend_by_id", return_value=cloud_dict),
            patch(
                "kiara.generation.cloud_routing.create_backend", return_value=mock_backend
            ) as mock_create,
        ):
            from kiara.generation.cloud_routing import _create_chain_backend

            result = _create_chain_backend(
                _base_config(), "test", llm_backend_id=10, filter_datasource_ids=[1]
            )

        mock_create.assert_called_once_with(_base_config(), 10)
        assert result is mock_backend

    def test_no_backend_id_uses_default(self):
        """Ohne backend_id wird Default-Backend via create_backend() erstellt."""
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.provider = "ollama"

        with (
            patch("kiara.generation.cloud_routing._route_backend", return_value="local"),
            patch("kiara.generation.cloud_routing.create_backend", return_value=mock_backend),
        ):
            from kiara.generation.cloud_routing import _create_chain_backend

            result = _create_chain_backend(
                _base_config(), "test", llm_backend_id=None, filter_datasource_ids=None
            )

        assert result is mock_backend
