"""Quellensteuerung: UI-Vorrang, Agent-Automatik und Backend-Budget."""

import asyncio
import json
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from kiara.generation.adapters.ollama import OllamaToolAdapter
from kiara.retrieval.source_policy import (
    get_source_profiles,
    resolve_chat_sources,
    validate_source_profiles,
)
from kiara.tools.builtin_tools import SearchDocumentsTool


@pytest.fixture
def config():
    profiles = get_source_profiles({})
    profiles["standard"]["default_k"] = 12
    profiles["rewriter"]["default_k"] = 32
    profiles["agent"].update(auto_min_k=3, auto_default_k=18, auto_max_k=60)
    return {"chat": {"source_profiles": profiles}, "retrieval": {"reranking": True}}


@pytest.mark.parametrize("mode,expected", [("standard", 12), ("rewriter", 32), ("agent", 20)])
def test_mode_defaults_and_request_local_config(config, mode, expected):
    before = deepcopy(config)
    resolved, k = resolve_chat_sources(config, mode, None)
    assert k == expected
    assert resolved["_chat_source_policy"]["ui_k"] is None
    assert resolved["_chat_source_policy"]["effective_k"] == expected
    assert resolved["retrieval"]["rerank_fetch_multiplier"] == 4
    assert resolved["retrieval"]["max_sibling_chunks"] == (30 if mode == "agent" else 5)
    assert config == before


@pytest.mark.parametrize(
    "mode,k", [("standard", 0), ("rewriter", 0), ("agent", 101), ("agent", True)]
)
def test_rejects_disallowed_ui_value(config, mode, k):
    with pytest.raises(ValueError):
        resolve_chat_sources(config, mode, k)


def test_agent_auto_can_be_disabled(config):
    config["chat"]["source_profiles"]["agent"]["auto_enabled"] = False
    with pytest.raises(ValueError, match="Automatische Quellenwahl"):
        resolve_chat_sources(config, "agent", 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("default_k", 101),
        ("auto_default_k", 2),
        ("auto_max_k", 101),
        ("max_context_chunks", 0),
        ("keyword_fetch_multiplier", 1.5),
        ("auto_enabled", "true"),
        ("auto_instructions", " "),
        ("unknown_setting", 10),
    ],
)
def test_invalid_profiles_fail_closed(config, field, value):
    config["chat"]["source_profiles"]["agent"][field] = value
    with pytest.raises(ValueError):
        validate_source_profiles(config)


@pytest.mark.parametrize("model_k", [2025, "2025", "summe", 1, None])
def test_fixed_ui_k_wins_over_every_model_proposal(config, model_k):
    config, k = resolve_chat_sources(config, "agent", 50)
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    tool = SearchDocumentsTool(retriever, None, config, default_k=k)
    params = {"query": "Rechnungen 2025", "k": model_k}
    assert "k" not in tool.definition().parameters["properties"]
    assert tool.validate_params(params)[0]
    result = asyncio.run(tool.execute(params))
    assert result.success
    assert retriever.retrieve.call_args.kwargs["k"] == 50
    assert result.diagnostics["source_selection"] == {
        "ui_k": 50,
        "requested_k": 50,
        "selection": "fixed",
        "model_k": model_k,
        "effective_k": 50,
        "model_k_ignored": True,
    }


@pytest.mark.parametrize("params,expected", [({}, 18), ({"k": 3}, 3), ({"k": 60}, 60)])
def test_auto_uses_valid_model_choice_or_admin_default(config, params, expected):
    config, k = resolve_chat_sources(config, "agent", 0)
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    tool = SearchDocumentsTool(retriever, None, config, default_k=k)
    result = asyncio.run(tool.execute({"query": "Rechnungen 2025", **params}))
    assert result.success
    assert retriever.retrieve.call_args.kwargs["k"] == expected
    assert result.diagnostics["source_selection"]["effective_k"] == expected


@pytest.mark.parametrize("model_k", [2025, 2, 61, 0, -1, "2025", "summe", True, 3.5, None])
def test_invalid_auto_k_never_reaches_retriever(config, model_k):
    config, k = resolve_chat_sources(config, "agent", 0)
    retriever = MagicMock()
    tool = SearchDocumentsTool(retriever, None, config, default_k=k)
    result = asyncio.run(tool.execute({"query": "Rechnungen 2025", "k": model_k}))
    assert not result.success
    assert result.diagnostics["source_selection"]["effective_k"] is None
    retriever.retrieve.assert_not_called()


def test_ollama_auto_prompt_explains_quantity_and_actual_bounds(config):
    config["chat"]["source_profiles"]["agent"]["auto_instructions"] = "Test-Anweisung aus Admin-UI."
    tool = SearchDocumentsTool(MagicMock(), None, config, default_k=0)
    prompt = OllamaToolAdapter._build_tools_prompt([tool.definition()])
    assert "Textabschnitte pro Suche: 3 bis 60, Standard 18" in prompt
    assert "Test-Anweisung aus Admin-UI." in prompt


def test_profile_api_checks_real_es_budget_before_saving(config):
    from kiara.retrieval.elasticsearch_indices import DatasourceIndexTarget
    from kiara.web.api_ki import SourceProfilesRequest, update_chat_source_profiles

    config["retrieval"].update(hybrid_search="elasticsearch", elasticsearch_index="kiara_test")
    es = MagicMock()
    es.indices.get_settings.return_value = {
        "kiara_test_ds_1": {"settings": {"index.max_result_window": "1000"}}
    }
    with (
        patch("kiara.web.api_ki.load_config", return_value=deepcopy(config)),
        patch("kiara.web.api_ki.save_config") as save,
        patch("kiara.web.api_ki.log_config_change") as audit,
        patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es),
        patch(
            "kiara.retrieval.elasticsearch_service.search_scope",
            return_value=nullcontext((DatasourceIndexTarget(1, "kiara_test_ds_1", 1),)),
        ),
    ):
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                update_chat_source_profiles(
                    SourceProfilesRequest(profiles=config["chat"]["source_profiles"]),
                    user={"id": 1},
                )
            )
        assert error.value.status_code == 400
        assert "1600" in error.value.detail
        save.assert_not_called()
        audit.assert_not_called()


def test_profile_api_saves_all_modes_and_audits_changes(config):
    from kiara.web.api_ki import SourceProfilesRequest, update_chat_source_profiles

    changed = deepcopy(config["chat"]["source_profiles"])
    changed["standard"]["default_k"] = 15
    changed["agent"]["default_k"] = 0
    with (
        patch("kiara.web.api_ki.load_config", return_value=deepcopy(config)),
        patch("kiara.web.api_ki.save_config") as save,
        patch("kiara.web.api_ki.log_config_change") as audit,
    ):
        result = asyncio.run(
            update_chat_source_profiles(
                SourceProfilesRequest(profiles=changed),
                user={"id": 1},
            )
        )
    assert result["profiles"] == changed
    assert save.call_args.args[0]["chat"]["source_profiles"] == changed
    assert json.loads(audit.call_args.kwargs["new_value"]) == changed


def test_query_audit_persists_source_policy_snapshot(config):
    from kiara.web.api_chat import _log_query_audit

    config, _ = resolve_chat_sources(config, "agent", 50)
    with patch("kiara.web.api_chat.audit_repo") as repo:
        _log_query_audit(
            user_id=1,
            question="Rechnungen 2025",
            answer="Test",
            chain_type="agent",
            request_settings=config["_chat_source_policy"],
        )
    assert json.loads(repo.create.call_args.kwargs["request_settings"])["ui_k"] == 50
