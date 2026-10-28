"""Tests fuer zentrale Graph-ID-Normalisierung."""

import pytest

from kiara.graph.normalization import normalize_node_id


def test_normalize_node_id_unicode_ascii_slug():
    assert normalize_node_id("supplier", "Müller GmbH") == "supplier:muller-gmbh"
    assert normalize_node_id("topic", "Qualitäts Prüfung!") == "topic:qualitats-prufung"


def test_normalize_node_id_invalid_inputs():
    with pytest.raises(ValueError):
        normalize_node_id("bad-type", "x")
    with pytest.raises(ValueError):
        normalize_node_id("topic", None)
    with pytest.raises(ValueError):
        normalize_node_id("topic", "ß")
