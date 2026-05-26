"""Tests fuer VectorDB-Config-Integration.

Prueft, dass DEFAULT_CONFIG korrekte vectordb-Defaults hat und die
Config-Merging-Logik Backend-spezifische Werte korrekt handhabt.
"""

import os
import tempfile

import pytest
import yaml

from kiara.config import DEFAULT_CONFIG, _validate, load_config

# ---------------------------------------------------------------------------
# 1. test_default_config_has_vectordb_backend
# ---------------------------------------------------------------------------


def test_default_config_has_vectordb_backend():
    """DEFAULT_CONFIG['vectordb']['backend'] existiert und ist 'qdrant'."""
    assert "vectordb" in DEFAULT_CONFIG
    assert "backend" in DEFAULT_CONFIG["vectordb"]
    assert DEFAULT_CONFIG["vectordb"]["backend"] == "qdrant"
    assert DEFAULT_CONFIG["vectordb"]["path"] != "/var/lib/kiara/chromadb"
    assert DEFAULT_CONFIG["backup"]["include_vectordb_data"] is False


# ---------------------------------------------------------------------------
# 2. test_config_merge_preserves_backend
# ---------------------------------------------------------------------------


def test_config_merge_preserves_backend():
    """Benutzerdefinierte Config mit vectordb.backend='qdrant' wird korrekt gemergt."""
    user_config = {
        "vectordb": {
            "backend": "qdrant",
            "qdrant_url": "http://my-qdrant:6333",
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(user_config, f)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config["vectordb"]["backend"] == "qdrant"
        assert config["vectordb"]["qdrant_url"] == "http://my-qdrant:6333"
        # Andere Defaults bleiben erhalten
        assert config["vectordb"]["path"] == DEFAULT_CONFIG["vectordb"]["path"]
        assert config["vectordb"]["vector_size"] == 768
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 3. test_config_validation_requires_vectordb_path
# ---------------------------------------------------------------------------


def test_config_validation_requires_vectordb_path():
    """Config ohne vectordb.path -> ValueError bei Validierung."""
    config = {
        "datasource": {"path": "/tmp"},
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {
            "backend": "qdrant",
            # "path" fehlt absichtlich
        },
    }
    with pytest.raises(ValueError, match="vectordb.path"):
        _validate(config)


# ---------------------------------------------------------------------------
# 4. test_config_qdrant_url_defaults
# ---------------------------------------------------------------------------


def test_config_qdrant_url_defaults():
    """Wenn backend=qdrant aber kein qdrant_url, greift Default."""
    assert "qdrant_url" in DEFAULT_CONFIG["vectordb"]
    assert DEFAULT_CONFIG["vectordb"]["qdrant_url"] == "http://localhost:6333"

    # Auch nach Merge mit leerer User-Config bleibt der Default
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"vectordb": {"backend": "qdrant"}}, f)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config["vectordb"]["qdrant_url"] == "http://localhost:6333"
    finally:
        os.unlink(tmp_path)
