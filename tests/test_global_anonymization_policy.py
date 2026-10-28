"""Der globale Aus-Schalter gilt fuer alle Quellen; Cloud-Sperren bleiben wirksam."""

from unittest.mock import MagicMock, patch

import pytest

from kiara.generation._validators import needs_anonymization
from kiara.generation.backends.base import LLMBackend
from kiara.generation.backends.factory import wrap_with_anonymization
from kiara.generation.cloud_routing import _create_chain_backend, _resolve_backend_dict
from kiara.shared.enums import RoutingMode
from kiara.web.api_chat import _cloud_requires_agent_anonymizer, _create_agent_anonymizer


@pytest.mark.parametrize("enabled", [False, True, None])
@pytest.mark.parametrize(
    "levels,source_requires",
    [
        (None, True),
        ([], True),
        (["public"], False),
        (["anonymize"], True),
        (["public", "anonymize"], True),
    ],
)
def test_global_switch_precedes_source_policy(enabled, levels, source_requires):
    config = {} if enabled is None else {"anonymization": {"enabled": enabled}}
    expected = enabled is not False and source_requires
    assert needs_anonymization(config, levels) is expected
    for provider in ("anthropic", "openai"):
        assert _cloud_requires_agent_anonymizer(config, {"provider": provider}, levels) is expected


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("levels", [None, [], ["public"], ["anonymize"], ["public", "anonymize"]])
def test_disabled_skips_initialization_and_passes_original_text(provider, levels):
    config = {"anonymization": {"enabled": False}}
    backend = MagicMock(spec=LLMBackend)
    backend.provider = provider
    backend.generate.return_value = "Antwort fuer Eva Beispiel"
    with patch(
        "kiara.anonymization.stammdaten.build_automaton_from_sources",
        side_effect=AssertionError("Bei globalem Aus keine Stammdaten laden"),
    ) as build:
        wrapped = wrap_with_anonymization(backend, config, levels)
        assert wrapped is backend
        assert _create_agent_anonymizer(config, {"provider": provider}, levels) is None
        assert wrapped.generate("System", "Frage zu Eva Beispiel") == "Antwort fuer Eva Beispiel"
        backend.generate.assert_called_once_with("System", "Frage zu Eva Beispiel")
        build.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("level", ["blocked", "unknown", None])
def test_blocked_and_invalid_sources_rejected_even_when_disabled(enabled, level):
    config = {"anonymization": {"enabled": enabled}}
    backend = MagicMock(spec=LLMBackend)
    backend.provider = "openai"
    levels = ["public", level]
    with pytest.raises(ValueError, match="blocked/unbekannt"):
        wrap_with_anonymization(backend, config, levels)
    with pytest.raises(ValueError, match="blocked/unbekannt"):
        _create_agent_anonymizer(config, {"provider": "openai"}, levels)
    with pytest.raises(ValueError, match="blocked/unbekannt"):
        _cloud_requires_agent_anonymizer(config, {"provider": "openai"}, levels)


@pytest.mark.parametrize("explicit", [False, True])
def test_standard_routing_respects_global_off(explicit):
    config = {"anonymization": {"enabled": False}}
    backend = MagicMock(spec=LLMBackend)
    backend.provider = "openai"
    backend_row = {"id": 5, "provider": "openai", "category": "cloud", "trusted": False}
    with (
        patch("kiara.generation.cloud_routing.get_backend_by_id", return_value=backend_row),
        patch("kiara.generation.cloud_routing.get_next_backend", return_value=backend_row),
        patch("kiara.generation.cloud_routing._route_backend", return_value=RoutingMode.CLOUD),
        patch(
            "kiara.generation.cloud_routing._load_restriction_levels", return_value=["anonymize"]
        ),
        patch("kiara.generation.cloud_routing.create_backend", return_value=backend),
        patch("kiara.anonymization.stammdaten.build_automaton_from_sources") as build,
    ):
        result = _create_chain_backend(config, "Frage", 5 if explicit else None, [12])
    assert result is backend
    build.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["standard", "agent"])
@pytest.mark.parametrize("level", ["blocked", "unknown", None])
def test_global_off_does_not_bypass_cloud_routing_block(entrypoint, level):
    config = {"anonymization": {"enabled": False}}
    backend_row = {"id": 5, "provider": "openai", "category": "cloud", "trusted": False}
    with (
        patch("kiara.generation.cloud_routing.get_backend_by_id", return_value=backend_row),
        patch("kiara.generation.cloud_routing._load_restriction_levels", return_value=[level]),
        patch("kiara.generation.cloud_routing.create_backend") as create,
        pytest.raises(ValueError, match="Restriction-Level"),
    ):
        if entrypoint == "standard":
            _create_chain_backend(config, "Frage", 5, [12])
        else:
            _resolve_backend_dict(config, llm_backend_id=5, filter_datasource_ids=[12])
    create.assert_not_called()


def test_enrichment_uses_original_text_when_globally_disabled():
    from kiara.ingestion.runtime import _get_enrichment_llm

    config = {"anonymization": {"enabled": False}}
    backend = MagicMock(spec=LLMBackend)
    backend.provider = "openai"
    backend.generate.return_value = "Eva Beispiel"
    backend_row = {"id": 5, "provider": "openai", "category": "cloud", "trusted": False}
    with (
        patch("kiara.backend_selector.get_backend_by_id", return_value=backend_row),
        patch("kiara.generation.backends.factory.create_backend", return_value=backend),
        patch("kiara.anonymization.stammdaten.build_automaton_from_sources") as build,
        patch("kiara.cloud.audit.AuditLogger"),
        patch("kiara.anonymization.audit.log_anonymization_event") as audit,
    ):
        generate, _, _ = _get_enrichment_llm(
            config, {"enrichment_backend_id": 5, "restriction_level": "anonymize"}
        )
        assert generate("System", "Eva Beispiel") == "Eva Beispiel"
    backend.generate.assert_called_once_with("System", "Eva Beispiel")
    build.assert_not_called()
    assert audit.call_args.kwargs["entity_count"] == 0
