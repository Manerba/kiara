"""Tests fuer kiara.config — DB-only Backend-Wrapper (Phase 13).

Testet get_embedding_backend(), get_llm_backend(), get_vlm_backend(),
get_cloud_backend() und get_embedding_model() mit DB-only-Logik
(kein YAML-Fallback).
"""

from unittest.mock import patch

import pytest

from kiara.config import (
    get_cloud_backend,
    get_embedding_backend,
    get_embedding_model,
    get_llm_backend,
    get_vlm_backend,
)


def _mock_db_backend(category="llm", **overrides):
    """Simuliert ein Backend-Dict wie es get_next_backend() zurueckgibt."""
    base = {
        "id": 1,
        "category": category,
        "provider": "ollama",
        "name": f"DB {category}",
        "url": f"http://db-{category}:11434",
        "base_url": "",
        "model": f"db-model-{category}",
        "api_key": "",
        "is_active": True,
        "priority": 10,
    }
    base.update(overrides)
    return base


# ===================================================================
# A) get_embedding_backend()
# ===================================================================


class TestGetEmbeddingBackend:
    def test_returns_db_values_when_available(self):
        """DB-Backend wird zurueckgegeben."""
        db_backend = _mock_db_backend("embedding", url="http://db-emb:11434", model="db-nomic")
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            result = get_embedding_backend({})
        assert result == {"url": "http://db-emb:11434", "model": "db-nomic"}

    def test_raises_when_db_returns_none(self):
        """RuntimeError wenn DB kein Backend liefert."""
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            with pytest.raises(RuntimeError, match="Kein embedding-Backend"):
                get_embedding_backend({})

    def test_raises_on_import_error(self):
        """ImportError propagiert (kein stiller Fallback)."""
        with patch("kiara.backend_selector.get_next_backend", side_effect=ImportError("No module")):
            with pytest.raises(ImportError):
                get_embedding_backend({})

    def test_raises_on_db_exception(self):
        """DB-Exceptions propagieren (kein stiller Fallback)."""
        with patch(
            "kiara.backend_selector.get_next_backend", side_effect=RuntimeError("DB crashed")
        ):
            with pytest.raises(RuntimeError, match="DB crashed"):
                get_embedding_backend({})


# ===================================================================
# B) get_embedding_model()
# ===================================================================


class TestGetEmbeddingModel:
    def test_returns_model_from_db(self):
        """get_embedding_model() gibt Modell aus DB zurueck."""
        db_backend = _mock_db_backend("embedding", model="mxbai-embed-large")
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            assert get_embedding_model({}) == "mxbai-embed-large"

    def test_raises_when_no_backend(self):
        """RuntimeError wenn kein Embedding-Backend konfiguriert."""
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            with pytest.raises(RuntimeError, match="Kein embedding-Backend"):
                get_embedding_model({})


# ===================================================================
# C) get_llm_backend()
# ===================================================================


class TestGetLlmBackend:
    def test_returns_db_values(self):
        """LLM-Backend aus DB."""
        db_backend = _mock_db_backend("llm", url="http://db-llm:11434", model="llama3:70b")
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            result = get_llm_backend({})
        assert result == {"url": "http://db-llm:11434", "model": "llama3:70b"}

    def test_raises_when_no_backend(self):
        """RuntimeError wenn kein LLM-Backend konfiguriert."""
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            with pytest.raises(RuntimeError, match="Kein llm-Backend"):
                get_llm_backend({})

    def test_raises_on_exception(self):
        """Exceptions propagieren."""
        with patch("kiara.backend_selector.get_next_backend", side_effect=Exception("timeout")):
            with pytest.raises(Exception, match="timeout"):
                get_llm_backend({})

    def test_returns_dict_with_url_and_model_keys(self):
        """Rueckgabe-Dict hat 'url' und 'model' Keys."""
        db_backend = _mock_db_backend("llm")
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            result = get_llm_backend({})
        assert "url" in result
        assert "model" in result


# ===================================================================
# D) get_vlm_backend()
# ===================================================================


class TestGetVlmBackend:
    def test_returns_db_values(self):
        """VLM-Backend aus DB."""
        db_backend = _mock_db_backend("vlm", url="http://db-vlm:11434", model="llava:13b")
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            result = get_vlm_backend({})
        assert result == {"url": "http://db-vlm:11434", "model": "llava:13b"}

    def test_returns_none_when_db_empty(self):
        """None wenn DB kein VLM-Backend hat."""
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            result = get_vlm_backend({})
        assert result is None

    def test_returns_none_on_exception(self):
        """None bei DB-Exception (VLM ist optional)."""
        with patch("kiara.backend_selector.get_next_backend", side_effect=Exception("err")):
            result = get_vlm_backend({})
        assert result is None


# ===================================================================
# E) get_cloud_backend()
# ===================================================================


class TestGetCloudBackend:
    def test_returns_db_values(self):
        """Cloud-Backend aus DB mit allen Feldern."""
        db_backend = _mock_db_backend(
            "cloud",
            provider="anthropic",
            api_key="sk-ant-key",
            model="claude-sonnet-4-6",
            base_url="https://api.anthropic.com",
        )
        with patch("kiara.backend_selector.get_next_backend", return_value=db_backend):
            result = get_cloud_backend({})
        assert result["provider"] == "anthropic"
        assert result["api_key"] == "sk-ant-key"
        assert result["model"] == "claude-sonnet-4-6"
        assert result["base_url"] == "https://api.anthropic.com"

    def test_returns_none_when_db_empty(self):
        """None wenn DB kein Cloud-Backend hat."""
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            result = get_cloud_backend({})
        assert result is None

    def test_returns_none_on_exception(self):
        """None bei DB-Exception (Cloud ist optional)."""
        with patch("kiara.backend_selector.get_next_backend", side_effect=Exception("err")):
            result = get_cloud_backend({})
        assert result is None


# ===================================================================
# F) Konsistenz-Tests
# ===================================================================


class TestConsistency:
    def test_all_backends_return_dict_with_required_keys(self):
        """Alle Backend-Funktionen geben dicts mit erwarteten Keys zurueck."""
        emb_be = _mock_db_backend("embedding")
        llm_be = _mock_db_backend("llm")
        vlm_be = _mock_db_backend("vlm")
        cloud_be = _mock_db_backend(
            "cloud",
            provider="anthropic",
            api_key="sk-test",
            model="claude-sonnet-4-6",
            base_url="",
        )

        def _side_effect(category):
            return {"embedding": emb_be, "llm": llm_be, "vlm": vlm_be, "cloud": cloud_be}[category]

        with patch("kiara.backend_selector.get_next_backend", side_effect=_side_effect):
            emb = get_embedding_backend({})
            llm = get_llm_backend({})
            vlm = get_vlm_backend({})
            cloud = get_cloud_backend({})

        # Embedding + LLM: immer url + model
        assert "url" in emb and "model" in emb
        assert "url" in llm and "model" in llm
        # VLM: url + model
        assert vlm is not None
        assert "url" in vlm and "model" in vlm
        # Cloud: provider + api_key + model + base_url
        assert cloud is not None
        assert all(k in cloud for k in ("provider", "api_key", "model", "base_url"))

    def test_mandatory_backends_raise_optional_return_none(self):
        """Embedding/LLM raisen bei fehlendem Backend, VLM/Cloud geben None."""
        with patch("kiara.backend_selector.get_next_backend", return_value=None):
            with pytest.raises(RuntimeError):
                get_embedding_backend({})
            with pytest.raises(RuntimeError):
                get_llm_backend({})
            assert get_vlm_backend({}) is None
            assert get_cloud_backend({}) is None
