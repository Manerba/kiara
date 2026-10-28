"""Semantische Config-Validierung."""

import copy

import pytest
import yaml

from kiara.config import DEFAULT_CONFIG, _validate, load_config, save_config


def _config():
    return copy.deepcopy(DEFAULT_CONFIG)


def test_default_config_semantic_validation_passes():
    _validate(_config())


@pytest.mark.parametrize(
    "key,value",
    [
        ("context_budget_chars", 0),
        ("context_result_chars", -1),
        ("context_read_reserve_chars", -1),
        ("context_result_chars", True),
    ],
)
def test_agent_context_budget_validation(key, value):
    config = _config()
    config["chat"]["agent"][key] = value
    with pytest.raises(ValueError, match=key):
        _validate(config)


def test_agent_read_reserve_leaves_space_for_search():
    config = _config()
    config["chat"]["agent"]["context_read_reserve_chars"] = config["chat"]["agent"][
        "context_budget_chars"
    ]
    with pytest.raises(ValueError, match="context_read_reserve_chars"):
        _validate(config)


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


def _valid_oidc_config():
    config = _config()
    config["web"]["mcp_retrieval_enabled"] = True
    config["web"]["oidc_retrieval"].update(
        {
            "enabled": True,
            "issuer": "https://id.opendesk.hirschgasse5.de/realms/opendesk",
            "discovery_url": (
                "https://id.opendesk.hirschgasse5.de/realms/opendesk/"
                ".well-known/openid-configuration"
            ),
            "jwks_url": "",
            "audience": "kiara-retrieval",
            "client_binding_claim": "azp",
            "client_binding_value": "open-webui",
            "identity_claim": "ad_sid",
            "identity_claim_format": "ad_object_sid_or_base64",
            "algorithms": ["RS256"],
            "required_scopes": [],
            "required_roles": [],
            "clock_skew_seconds": 60,
            "ca_bundle_path": "",
        }
    )
    return config


def test_oidc_retrieval_valid_config_passes():
    _validate(_valid_oidc_config())


def test_oidc_retrieval_enabled_requires_static_fields():
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["issuer"] = ""

    with pytest.raises(ValueError, match="issuer"):
        _validate(config)


def test_oidc_retrieval_rejects_non_https_urls():
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["discovery_url"] = "http://issuer/.well-known/openid"

    with pytest.raises(ValueError, match="discovery_url"):
        _validate(config)


def test_oidc_retrieval_rejects_unsafe_algorithms():
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["algorithms"] = ["HS256"]

    with pytest.raises(ValueError, match="algorithm|RS256"):
        _validate(config)


def test_oidc_retrieval_rejects_additional_asymmetric_algorithms():
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["algorithms"] = ["RS256", "RS512"]

    with pytest.raises(ValueError, match="exactly"):
        _validate(config)


def test_oidc_retrieval_rejects_invalid_scope_role_types():
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["required_scopes"] = "kiara-retrieval"

    with pytest.raises(ValueError, match="required_scopes"):
        _validate(config)


def test_oidc_retrieval_rejects_clock_skew_out_of_range():
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["clock_skew_seconds"] = 301

    with pytest.raises(ValueError, match="clock_skew"):
        _validate(config)


def test_load_and_save_config_reject_static_invalid_oidc_config(tmp_path):
    config = _valid_oidc_config()
    config["web"]["oidc_retrieval"]["jwks_url"] = "http://issuer/certs"
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="jwks_url"):
        load_config(str(config_path))

    with pytest.raises(ValueError, match="jwks_url"):
        save_config(config, str(tmp_path / "out.yaml"))


@pytest.mark.parametrize("prefix", ["*", "Other", "a,b", "", "../index"])
def test_es_source_index_prefix_validation(prefix):
    config = _config()
    config["retrieval"].update(hybrid_search="elasticsearch", elasticsearch_index=prefix)
    with pytest.raises(ValueError, match="Elasticsearch"):
        _validate(config)


@pytest.mark.parametrize("path", ["relative.json", "", 42])
def test_es_maintenance_path_is_absolute(path):
    config = _config()
    config["retrieval"]["elasticsearch_maintenance_file"] = path
    with pytest.raises(ValueError, match="elasticsearch_maintenance_file"):
        _validate(config)
