"""Tests fuer das Cloud-Routing und die Backend-Resolver in
kiara.generation.cloud_routing.

Routing- und Resolver-Logik als Shared-Helper fuer api_chat.py,
api_admin_chat.py und mcp_server.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.shared.enums import RoutingMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cloud_config(enabled=True, mode=RoutingMode.AUTO):
    return {
        "cloud": {"enabled": enabled, "mode": mode},
        "anonymization": {"enabled": True},
    }


# ---------------------------------------------------------------------------
# Tests: _route_backend()
# ---------------------------------------------------------------------------


class TestRouteBackend:
    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    def test_route_backend_auto_keyword_match(self, mock_active):
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(),
            "bitte eine strategische Analyse erstellen",
            None,
        )
        assert result == RoutingMode.CLOUD

    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    @patch("kiara.generation.cloud_routing._degradation_feature_available", return_value=True)
    def test_route_backend_auto_no_keyword(self, mock_health, mock_active):
        """Keyword routing assumes healthy backends, independently of earlier health tests."""
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(),
            "wie heisst die Datei?",
            None,
        )
        assert result == RoutingMode.LOCAL

    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    @patch("kiara.generation.cloud_routing._is_cloud_blocked", return_value=False)
    def test_route_backend_no_llm_uses_cloud_without_keyword(self, mock_blocked, mock_active):
        """NO_LLM nutzt eine gesunde Cloud auch ohne strategische Keywords."""
        from kiara.generation.cloud_routing import _route_backend

        mock_dm = MagicMock()
        mock_dm.is_feature_available.side_effect = lambda feature: feature == "cloud"

        with patch("kiara.web.api_health._get_degradation_manager", return_value=mock_dm):
            result = _route_backend(
                _cloud_config(),
                "wie heisst die Datei?",
                None,
            )

        assert result == RoutingMode.CLOUD

    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    def test_route_backend_invalid_mode_fails_closed_to_local(self, mock_active):
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(mode="always"),
            "strategische analyse",
            None,
        )
        assert result == RoutingMode.LOCAL

    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    def test_route_backend_forced_cloud(self, mock_active):
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(mode=RoutingMode.CLOUD),
            "x",
            None,
        )
        assert result == RoutingMode.CLOUD

    def test_route_backend_forced_local(self):
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(mode=RoutingMode.LOCAL),
            "erklaere alles",
            None,
        )
        assert result == RoutingMode.LOCAL

    def test_route_backend_cloud_disabled(self):
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(enabled=False),
            "erklaere mir das",
            None,
        )
        assert result == RoutingMode.LOCAL

    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    @patch("kiara.generation.cloud_routing._is_cloud_blocked", return_value=True)
    def test_route_backend_restriction_level_blocked(self, mock_blocked, mock_active):
        from kiara.generation.cloud_routing import _route_backend

        result = _route_backend(
            _cloud_config(mode=RoutingMode.CLOUD),
            "x",
            [1, 2],
        )
        assert result == RoutingMode.LOCAL

    @patch("kiara.generation.cloud_routing.get_all_active", return_value=[{"id": 1}])
    def test_route_backend_checks_availability_without_consuming_round_robin(self, mock_active):
        """Verfuegbarkeitscheck nutzt get_all_active, nicht get_next_backend."""
        from kiara.generation.cloud_routing import _route_backend

        with patch("kiara.generation.cloud_routing.get_next_backend") as mock_next:
            _route_backend(_cloud_config(), "erklaere", None)
            mock_next.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _is_cloud_blocked()
# ---------------------------------------------------------------------------


class TestIsCloudBlocked:
    def test_no_ds_ids_not_blocked(self):
        from kiara.generation.cloud_routing import _is_cloud_blocked

        assert _is_cloud_blocked({}, None) is False
        assert _is_cloud_blocked({}, []) is False

    @patch("kiara.repositories.datasource.DatasourceRepository")
    def test_is_cloud_blocked_fail_closed(self, MockRepo):
        """DB-Fehler -> True (fail-closed)."""
        from kiara.generation.cloud_routing import _is_cloud_blocked

        MockRepo.return_value.get_restriction_levels.side_effect = RuntimeError("DB down")
        assert _is_cloud_blocked({}, [1, 2]) is True

    @patch("kiara.repositories.datasource.DatasourceRepository")
    def test_public_restriction_allows_cloud(self, MockRepo):
        """public-Daten erlauben Cloud-Nutzung."""
        from kiara.generation.cloud_routing import _is_cloud_blocked

        MockRepo.return_value.get_restriction_levels.return_value = [
            {"restriction_level": "public"},
            {"restriction_level": "public"},
        ]
        assert _is_cloud_blocked({}, [1, 2]) is False

    @patch("kiara.repositories.datasource.DatasourceRepository")
    def test_anonymize_restriction_allows_cloud(self, MockRepo):
        """anonymize-Daten erlauben Cloud-Nutzung (mit Wrapping)."""
        from kiara.generation.cloud_routing import _is_cloud_blocked

        MockRepo.return_value.get_restriction_levels.return_value = [
            {"restriction_level": "anonymize"},
            {"restriction_level": "public"},
        ]
        assert _is_cloud_blocked({}, [1, 2]) is False

    @patch("kiara.repositories.datasource.DatasourceRepository")
    def test_any_blocked_blocks_all(self, MockRepo):
        """Eine blocked-Quelle reicht aus um Cloud zu blockieren."""
        from kiara.generation.cloud_routing import _is_cloud_blocked

        MockRepo.return_value.get_restriction_levels.return_value = [
            {"restriction_level": "public"},
            {"restriction_level": "blocked"},
        ]
        assert _is_cloud_blocked({}, [1, 2]) is True

    @patch("kiara.repositories.datasource.DatasourceRepository")
    def test_missing_row_guard_fail_closed(self, MockRepo):
        """Fehlende Rows (weniger als IDs) -> True (fail-closed)."""
        from kiara.generation.cloud_routing import _is_cloud_blocked

        # Nur 1 Row fuer 2 IDs -> _load_restriction_levels wirft RuntimeError
        MockRepo.return_value.get_restriction_levels.return_value = [
            {"restriction_level": "public"},
        ]
        assert _is_cloud_blocked({}, [1, 2]) is True


# ---------------------------------------------------------------------------
# Tests: _create_chain_backend()
# ---------------------------------------------------------------------------


class TestCreateChainBackend:
    @patch("kiara.generation.cloud_routing.get_backend_by_id")
    @patch("kiara.generation.cloud_routing._load_restriction_levels", return_value=["anonymize"])
    @patch("kiara.generation.cloud_routing.create_backend")
    @patch("kiara.generation.cloud_routing.wrap_with_anonymization")
    def test_create_chain_backend_explicit_cloud(
        self, mock_wrap, mock_create, mock_levels, mock_get_by_id
    ):
        """Explizites untrusted Cloud-Backend wird gewrappt bei anonymize-Level."""
        from kiara.generation.cloud_routing import _create_chain_backend

        mock_get_by_id.return_value = {"id": 5, "provider": "anthropic", "model": "claude"}
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        mock_wrap.return_value = mock_backend

        result = _create_chain_backend({}, "frage", llm_backend_id=5, filter_datasource_ids=[1])
        mock_wrap.assert_called_once()

    @patch("kiara.generation.cloud_routing.get_backend_by_id")
    @patch("kiara.generation.cloud_routing._load_restriction_levels", return_value=["blocked"])
    def test_create_chain_backend_blocked_raises(self, mock_levels, mock_get_by_id):
        """Explizites Cloud-Backend mit blocked-Level wirft ValueError."""
        from kiara.generation.cloud_routing import _create_chain_backend

        mock_get_by_id.return_value = {"id": 5, "provider": "anthropic", "model": "claude"}

        with pytest.raises(ValueError, match="Restriction-Level"):
            _create_chain_backend({}, "frage", llm_backend_id=5, filter_datasource_ids=[1])

    @patch("kiara.generation.cloud_routing.get_backend_by_id")
    @patch("kiara.generation.cloud_routing.create_backend")
    def test_create_chain_backend_trusted_skips_restriction(self, mock_create, mock_get_by_id):
        """Trusted Cloud-Backend ueberspringt Restriction-Pruefung."""
        from kiara.generation.cloud_routing import _create_chain_backend

        mock_get_by_id.return_value = {
            "id": 5,
            "provider": "anthropic",
            "model": "claude",
            "trusted": True,
        }
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend

        result = _create_chain_backend({}, "frage", llm_backend_id=5, filter_datasource_ids=[1])
        mock_create.assert_called_once_with({}, 5)

    @patch("kiara.generation.cloud_routing._route_backend", return_value=RoutingMode.CLOUD)
    @patch("kiara.generation.cloud_routing.get_next_backend")
    @patch("kiara.generation.cloud_routing._load_restriction_levels", return_value=["anonymize"])
    @patch("kiara.generation.cloud_routing.create_backend")
    @patch("kiara.generation.cloud_routing.wrap_with_anonymization")
    def test_create_chain_backend_uses_get_next_backend_for_cloud(
        self, mock_wrap, mock_create, mock_levels, mock_next, mock_route
    ):
        """Auto-Cloud nutzt den Selector zur Backend-Auswahl."""
        from kiara.generation.cloud_routing import _create_chain_backend

        mock_next.return_value = {"id": 7, "provider": "openai", "model": "gpt-4"}
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        mock_wrap.return_value = mock_backend

        _create_chain_backend({}, "erklaere alles", llm_backend_id=None, filter_datasource_ids=[1])
        mock_next.assert_called_once_with("cloud")
        mock_wrap.assert_called_once()

    @patch("kiara.generation.cloud_routing._route_backend", return_value=RoutingMode.CLOUD)
    @patch("kiara.generation.cloud_routing.get_next_backend", return_value=None)
    @patch("kiara.generation.cloud_routing.create_backend")
    def test_create_chain_backend_cloud_no_backend_falls_back(
        self, mock_create, mock_next, mock_route
    ):
        """Cloud-Route, aber kein Cloud-Backend verfuegbar -> Fallback auf Default."""
        from kiara.generation.cloud_routing import _create_chain_backend

        mock_backend = MagicMock()
        mock_create.return_value = mock_backend

        result = _create_chain_backend({}, "frage", llm_backend_id=None, filter_datasource_ids=None)
        # Fallback: create_backend ohne ID
        mock_create.assert_called_once_with({})
        assert result is mock_backend

    @patch("kiara.generation.cloud_routing.get_backend_by_id")
    @patch("kiara.generation.cloud_routing._route_backend", return_value=RoutingMode.CLOUD)
    @patch("kiara.generation.cloud_routing.get_next_backend")
    @patch("kiara.generation.cloud_routing.create_backend")
    def test_create_chain_backend_explicit_local_not_overridden_by_auto_cloud(
        self,
        mock_create,
        mock_next,
        mock_route,
        mock_get_by_id,
    ):
        """Explizit gewaehltes lokales Backend darf nicht durch Auto-Cloud-Routing ersetzt werden."""
        from kiara.generation.cloud_routing import _create_chain_backend

        mock_get_by_id.return_value = {
            "id": 5,
            "provider": "ollama",
            "model": "qwen3:8b",
            "category": "llm",
        }
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend

        result = _create_chain_backend(
            {},
            "bitte eine strategische Analyse",
            llm_backend_id=5,
            filter_datasource_ids=[1],
        )

        mock_create.assert_called_once_with({}, 5)
        # Auto-Routing darf bei expliziter Wahl nicht ausgeloest werden
        mock_route.assert_not_called()
        mock_next.assert_not_called()
        assert result is mock_backend


# ---------------------------------------------------------------------------
# Tests: _resolve_backend_dict()
# ---------------------------------------------------------------------------


class TestResolveBackendDict:
    @patch("kiara.generation.cloud_routing.get_backend_by_id", return_value=None)
    def test_explicit_invalid_backend_id_raises_value_error(self, mock_get_by_id):
        from kiara.generation.cloud_routing import _resolve_backend_dict

        with pytest.raises(ValueError, match="nicht gefunden"):
            _resolve_backend_dict({}, llm_backend_id=9999, filter_datasource_ids=None)

        mock_get_by_id.assert_called_once_with(9999)

    @patch("kiara.generation.cloud_routing.get_next_backend")
    def test_resolve_backend_dict_prefers_local_llm(self, mock_next):
        from kiara.generation.cloud_routing import _resolve_backend_dict

        mock_next.side_effect = [
            {"id": 1, "provider": "ollama", "url": "http://ollama", "model": "qwen3:8b"},
        ]

        backend = _resolve_backend_dict({}, llm_backend_id=None, filter_datasource_ids=None)

        assert backend["provider"] == "ollama"
        assert mock_next.call_args_list[0].args == ("llm",)

    @patch("kiara.generation.cloud_routing.get_next_backend")
    def test_resolve_backend_dict_falls_back_to_cloud(self, mock_next):
        from kiara.generation.cloud_routing import _resolve_backend_dict

        mock_next.side_effect = [
            None,
            {"id": 7, "provider": "openai", "model": "gpt-4o", "api_key": "sk-test"},
        ]

        backend = _resolve_backend_dict({}, llm_backend_id=None, filter_datasource_ids=None)

        assert backend["id"] == 7
        assert backend["provider"] == "openai"

    @patch("kiara.generation.cloud_routing._is_cloud_blocked", return_value=True)
    @patch("kiara.generation.cloud_routing.get_backend_by_id")
    def test_resolve_backend_dict_explicit_blocked_cloud_raises_value_error(
        self,
        mock_get_by_id,
        mock_blocked,
    ):
        """Explizit gewaehltes Cloud-Backend + blocked -> ValueError."""
        from kiara.generation.cloud_routing import _resolve_backend_dict

        mock_get_by_id.return_value = {
            "id": 5,
            "provider": "anthropic",
            "model": "claude",
            "api_key": "sk-ant",
            "category": "cloud",
        }

        with pytest.raises(ValueError, match="Restriction-Level"):
            _resolve_backend_dict({}, llm_backend_id=5, filter_datasource_ids=[1, 2])

    @patch("kiara.generation.cloud_routing.get_next_backend")
    @patch("kiara.generation.cloud_routing._is_cloud_blocked", return_value=True)
    def test_resolve_backend_dict_auto_blocked_cloud_falls_back_to_local(
        self,
        mock_blocked,
        mock_next,
    ):
        """Auto-gewaehltes Cloud-Backend + blocked -> Fallback auf lokales LLM."""
        from kiara.generation.cloud_routing import _resolve_backend_dict

        mock_next.side_effect = [
            None,  # kein LLM
            {"id": 5, "provider": "anthropic", "model": "claude", "api_key": "sk-ant"},  # Cloud
            {
                "id": 1,
                "provider": "ollama",
                "url": "http://ollama",
                "model": "qwen3:8b",
            },  # Fallback LLM
        ]

        backend = _resolve_backend_dict({}, llm_backend_id=None, filter_datasource_ids=[1, 2])

        assert backend["provider"] == "ollama"

    @patch("kiara.generation.cloud_routing.get_next_backend")
    @patch("kiara.generation.cloud_routing._is_cloud_blocked", return_value=True)
    def test_resolve_backend_dict_auto_blocked_cloud_without_local_raises(
        self,
        mock_blocked,
        mock_next,
    ):
        """Auto-gewaehltes Cloud-Backend + blocked + kein lokales LLM -> RuntimeError."""
        from kiara.generation.cloud_routing import _resolve_backend_dict

        mock_next.side_effect = [
            None,  # kein LLM
            {"id": 5, "provider": "openai", "model": "gpt-4o", "api_key": "sk-openai"},  # Cloud
            None,  # kein Fallback-LLM
        ]

        with pytest.raises(
            RuntimeError,
            match="Cloud-Backend durch Restriction-Level gesperrt und kein lokales LLM-Backend verfuegbar",
        ):
            _resolve_backend_dict({}, llm_backend_id=None, filter_datasource_ids=[1, 2])

    @patch("kiara.generation.cloud_routing.get_backend_by_id")
    def test_resolve_backend_dict_trusted_skips_restriction_check(
        self,
        mock_get_by_id,
    ):
        """trusted=True Cloud-Backend ueberspringt Restriction-Pruefung."""
        from kiara.generation.cloud_routing import _resolve_backend_dict

        mock_get_by_id.return_value = {
            "id": 5,
            "provider": "anthropic",
            "model": "claude",
            "api_key": "sk-ant",
            "trusted": True,
            "category": "cloud",
        }

        backend = _resolve_backend_dict({}, llm_backend_id=5, filter_datasource_ids=[1, 2])

        assert backend["provider"] == "anthropic"
        assert backend["trusted"]
