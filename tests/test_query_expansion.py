"""Tests fuer den QueryExpander (kiara.retrieval.query_expansion)."""

from unittest.mock import MagicMock

import pytest

from kiara.retrieval.query_expansion import QueryExpander
from kiara.shared.constants import QUERY_EXPANSION_TIMEOUT_SECS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def disabled_config():
    """Config mit deaktivierter Query-Expansion."""
    return {
        "retrieval": {
            "query_expansion": False,
        },
    }


@pytest.fixture
def enabled_config():
    """Config mit aktivierter Query-Expansion."""
    return {
        "retrieval": {
            "query_expansion": True,
        },
    }


@pytest.fixture
def mock_backend():
    """Mock-Backend das alternative Suchbegriffe liefert."""
    backend = MagicMock()
    backend.provider = "ollama"
    backend.generate = MagicMock(
        return_value=("firewall rules\nPaketfilter\niptables Konfiguration")
    )
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQueryExpander:
    def test_expansion_disabled(self, disabled_config, mock_backend):
        """Wenn deaktiviert: nur Original-Query zurueck."""
        expander = QueryExpander(disabled_config, backend=mock_backend)
        assert expander.enabled is False

        result = expander.expand("Firewall-Regeln")
        assert result == ["Firewall-Regeln"]
        mock_backend.generate.assert_not_called()

    def test_expansion_no_backend(self, enabled_config):
        """Ohne Backend: nur Original-Query zurueck."""
        expander = QueryExpander(enabled_config)
        assert expander.enabled is True

        result = expander.expand("Firewall-Regeln")
        assert result == ["Firewall-Regeln"]

    def test_disabled_and_no_backend_paths_return_trimmed_original(
        self, disabled_config, enabled_config, mock_backend
    ):
        """Alle No-Expansion-Pfade liefern dieselbe normalisierte Original-Query."""
        disabled = QueryExpander(disabled_config, backend=mock_backend)
        no_backend = QueryExpander(enabled_config)

        assert disabled.expand("  Firewall-Regeln  ") == ["Firewall-Regeln"]
        assert no_backend.expand("  Firewall-Regeln  ") == ["Firewall-Regeln"]
        mock_backend.generate.assert_not_called()

    def test_expansion_with_backend(self, enabled_config, mock_backend):
        """Mit Backend werden Expansionen erzeugt."""
        expander = QueryExpander(enabled_config, backend=mock_backend)
        result = expander.expand("Firewall-Regeln")

        assert result[0] == "Firewall-Regeln"
        assert len(result) >= 2
        assert len(result) <= 4
        mock_backend.generate.assert_called_once()

    def test_parse_expansions(self, enabled_config):
        """Parsing der LLM-Antwort in einzelne Begriffe."""
        expander = QueryExpander(enabled_config)
        response = "firewall rules\nPaketfilter\niptables Konfiguration"
        result = expander._parse_expansions(response)
        assert result == ["firewall rules", "Paketfilter", "iptables Konfiguration"]

    def test_parse_expansions_with_numbers(self, enabled_config):
        """Nummerierte Antwort wird bereinigt."""
        expander = QueryExpander(enabled_config)
        response = "1. firewall rules\n2. Paketfilter\n3. iptables Konfiguration"
        result = expander._parse_expansions(response)
        assert result == ["firewall rules", "Paketfilter", "iptables Konfiguration"]

    def test_parse_expansions_with_bullets(self, enabled_config):
        """Aufzaehlungszeichen werden entfernt."""
        expander = QueryExpander(enabled_config)
        response = "- firewall rules\n- Paketfilter\n- iptables"
        result = expander._parse_expansions(response)
        assert result == ["firewall rules", "Paketfilter", "iptables"]

    def test_parse_expansions_with_typographic_bullets(self, enabled_config):
        """Typografische Bullets/Dashes (–, —, •) werden ebenfalls entfernt."""
        expander = QueryExpander(enabled_config)
        response = "• firewall rules\n– Paketfilter\n— iptables"
        result = expander._parse_expansions(response)
        assert result == ["firewall rules", "Paketfilter", "iptables"]

    def test_parse_expansions_max_three(self, enabled_config):
        """Maximal 3 Expansionen werden zurueckgegeben."""
        expander = QueryExpander(enabled_config)
        response = "eins\nzwei\ndrei\nvier\nfuenf"
        result = expander._parse_expansions(response)
        assert len(result) == 3

    def test_expansion_backend_error_returns_original(self, enabled_config):
        """Bei Backend-Fehler nur Original-Query zurueck."""
        backend = MagicMock()
        backend.provider = "ollama"
        backend.generate = MagicMock(side_effect=RuntimeError("LLM offline"))
        expander = QueryExpander(enabled_config, backend=backend)

        result = expander.expand("Test-Query")
        assert result == ["Test-Query"]

    def test_query_expander_uses_backend_generate(self, enabled_config):
        """Lokaler Backend-Pfad nutzt LLMBackend.generate()."""
        mock_backend = MagicMock()
        mock_backend.provider = "ollama"
        mock_backend.generate.return_value = "alt1\nalt2\nalt3"

        expander = QueryExpander(enabled_config, backend=mock_backend)
        result = expander.expand("Firewall-Regeln")

        assert result[0] == "Firewall-Regeln"
        assert len(result) >= 2
        mock_backend.generate.assert_called_once()
        assert mock_backend.generate.call_args.kwargs["timeout"] == QUERY_EXPANSION_TIMEOUT_SECS

    def test_query_expander_cloud_backend_skips_expansion(self, enabled_config):
        """Cloud-Backend gibt nur [query] zurueck (keine Cloud-Expansion)."""
        mock_backend = MagicMock()
        mock_backend.provider = "anthropic"

        expander = QueryExpander(enabled_config, backend=mock_backend)
        result = expander.expand("  Test-Query  ")

        assert result == ["Test-Query"]
        mock_backend.generate.assert_not_called()

    def test_query_expander_has_no_llm_parameter(self):
        """QueryExpander.__init__() akzeptiert kein llm= mehr."""
        import inspect

        sig = inspect.signature(QueryExpander.__init__)
        assert "llm" not in sig.parameters

    @pytest.mark.parametrize("empty_query", ["", "   ", "\t\n", None])
    def test_expansion_empty_query_returns_empty_list(
        self, enabled_config, mock_backend, empty_query
    ):
        """Leere/Whitespace-Query liefert [] (kein Leerstring-Retrieval)."""
        expander = QueryExpander(enabled_config, backend=mock_backend)
        assert expander.expand(empty_query) == []
        mock_backend.generate.assert_not_called()

    def test_expansion_empty_query_returns_empty_list_when_disabled(
        self, disabled_config, mock_backend
    ):
        """Leere Query liefert auch bei deaktivierter Expansion []."""
        expander = QueryExpander(disabled_config, backend=mock_backend)
        assert expander.expand("") == []
        assert expander.expand("   ") == []
