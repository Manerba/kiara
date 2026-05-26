"""Tests fuer kiara.shared.enums."""

import json

import pytest

from kiara.shared.enums import ChatMode, HealthStatus, IngestionStatus, RoutingMode


class TestChatMode:
    """Tests fuer das ChatMode-Enum."""

    def test_values(self):
        assert ChatMode.STANDARD == "standard"
        assert ChatMode.UNCHAINED == "unchained"
        assert ChatMode.AGENT == "agent"

    def test_string_comparison(self):
        """ChatMode erbt von str — Vergleich mit Strings muss funktionieren."""
        assert ChatMode.STANDARD == "standard"
        assert ChatMode.UNCHAINED == "unchained"
        assert ChatMode.AGENT == "agent"

    def test_json_serializable(self):
        """str-Enum ist direkt JSON-serialisierbar."""
        data = {"mode": ChatMode.AGENT}
        result = json.dumps(data)
        assert '"agent"' in result

    def test_from_string(self):
        """ChatMode kann aus String erzeugt werden."""
        assert ChatMode("standard") is ChatMode.STANDARD
        assert ChatMode("unchained") is ChatMode.UNCHAINED
        assert ChatMode("agent") is ChatMode.AGENT

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ChatMode("invalid")

    def test_all_members(self):
        members = list(ChatMode)
        assert len(members) == 4


class TestExistingEnums:
    """Sicherstellen, dass bestehende Enums nicht kaputt sind."""

    def test_ingestion_status_values(self):
        assert IngestionStatus.PENDING == "pending"
        assert IngestionStatus.RUNNING == "running"

    def test_health_status_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.UNHEALTHY == "unhealthy"

    def test_routing_mode_values(self):
        assert RoutingMode.AUTO == "auto"
        assert RoutingMode.LOCAL == "local"
        assert RoutingMode.CLOUD == "cloud"
