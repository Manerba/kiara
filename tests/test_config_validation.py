"""Semantische Config-Validierung."""

import copy

import pytest

from kiara.config import DEFAULT_CONFIG, _validate


def _config():
    return copy.deepcopy(DEFAULT_CONFIG)


def test_default_config_semantic_validation_passes():
    _validate(_config())


def test_config_validation_rejects_invalid_port():
    config = _config()
    config["web"]["admin_port"] = 70000

    with pytest.raises(ValueError, match="web.admin_port"):
        _validate(config)


def test_config_validation_rejects_negative_timeout():
    config = _config()
    config["timeouts"]["http_api"] = -1

    with pytest.raises(ValueError, match="timeouts.http_api"):
        _validate(config)


def test_config_validation_rejects_chunk_overlap_at_or_above_chunk_size():
    config = _config()
    config["ingestion"]["chunk_size"] = 1000
    config["ingestion"]["chunk_overlap"] = 1000

    with pytest.raises(ValueError, match="chunk_overlap"):
        _validate(config)


def test_config_validation_rejects_unknown_cloud_mode():
    config = _config()
    config["cloud"]["mode"] = "fallback"

    with pytest.raises(ValueError, match="cloud.mode"):
        _validate(config)


def test_config_validation_rejects_boolean_hybrid_search():
    config = _config()
    config["retrieval"]["hybrid_search"] = True

    with pytest.raises(ValueError, match="retrieval.hybrid_search"):
        _validate(config)
