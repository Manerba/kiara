"""Tests fuer elasticsearch_service: check_missing_ds_ids() und update_ds_ids().

Alle Tests mocken den ES-Client, um keine laufende ES-Instanz zu benoetigen.
"""

import sys
from unittest.mock import MagicMock, patch

# elasticsearch ist im Test-Env ein Namespace-Stub — fehlende Attribute ergaenzen
import elasticsearch as _es_pkg
import pytest

if not hasattr(_es_pkg, "helpers"):
    _mock_helpers = MagicMock()
    _es_pkg.helpers = _mock_helpers
    sys.modules["elasticsearch.helpers"] = _mock_helpers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _es_config(enabled=True):
    """Liefert eine Config mit aktiviertem/deaktiviertem ES."""
    return {
        "retrieval": {
            "hybrid_search": "elasticsearch" if enabled else "none",
            "elasticsearch_url": "http://localhost:9200",
            "elasticsearch_index": "kiara_test",
        }
    }


def _mock_es_client(count_total=100, count_missing=0, index_exists=True, raise_on_count=None):
    """Erzeugt einen gemockten ES-Client."""
    es = MagicMock()
    es.indices.exists.return_value = index_exists
    if raise_on_count:
        es.count.side_effect = raise_on_count
    else:

        def _count_side_effect(index, body=None):
            if body and "must_not" in str(body):
                return {"count": count_missing}
            return {"count": count_total}

        es.count.side_effect = _count_side_effect
    return es


# ---------------------------------------------------------------------------
# check_missing_ds_ids
# ---------------------------------------------------------------------------


class TestCheckMissingDsIds:
    """Tests fuer check_missing_ds_ids()."""

    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_all_docs_have_ds_ids(self, mock_get_client):
        """Alle Dokumente haben ds_ids -> safe_to_proceed=True."""
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        mock_get_client.return_value = _mock_es_client(count_total=500, count_missing=0)
        result = check_missing_ds_ids(_es_config())

        assert result["total_docs"] == 500
        assert result["missing_ds_ids"] == 0
        assert result["safe_to_proceed"] is True
        assert result["error"] is None

    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_some_docs_missing_ds_ids(self, mock_get_client):
        """Einige Dokumente ohne ds_ids -> safe_to_proceed=False."""
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        mock_get_client.return_value = _mock_es_client(count_total=500, count_missing=42)
        result = check_missing_ds_ids(_es_config())

        assert result["total_docs"] == 500
        assert result["missing_ds_ids"] == 42
        assert result["safe_to_proceed"] is False
        assert result["error"] is None

    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_es_error_returns_unsafe(self, mock_get_client):
        """ES-Fehler -> safe_to_proceed=False + error gesetzt."""
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        mock_get_client.return_value = _mock_es_client(raise_on_count=ConnectionError("ES down"))
        result = check_missing_ds_ids(_es_config())

        assert result["safe_to_proceed"] is False
        assert result["error"] is not None
        assert "ES down" in result["error"]

    def test_es_not_configured(self):
        """Nicht konfiguriertes ES -> safe_to_proceed=True (kein ES = kein Problem)."""
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        result = check_missing_ds_ids(_es_config(enabled=False))
        assert result["safe_to_proceed"] is True
        assert result["total_docs"] == 0

    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_index_not_exists(self, mock_get_client):
        """Index existiert nicht -> safe_to_proceed=True."""
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        mock_get_client.return_value = _mock_es_client(index_exists=False)
        result = check_missing_ds_ids(_es_config())

        assert result["safe_to_proceed"] is True
        assert result["total_docs"] == 0


# ---------------------------------------------------------------------------
# update_ds_ids
# ---------------------------------------------------------------------------


class TestUpdateDsIds:
    """Tests fuer update_ds_ids()."""

    @patch("elasticsearch.helpers.bulk", return_value=(3, []))
    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_bulk_update(self, mock_get_client, mock_bulk):
        """Normaler Bulk-Update aktualisiert ds_ids."""
        from kiara.retrieval.elasticsearch_service import update_ds_ids

        es = MagicMock()
        es.indices.exists.return_value = True
        mock_get_client.return_value = es

        result = update_ds_ids(
            _es_config(),
            {
                "chunk_1": [1, 2],
                "chunk_2": [2],
                "chunk_3": [1, 2, 3],
            },
        )

        assert result == 3
        mock_bulk.assert_called_once()
        actions = mock_bulk.call_args[0][1]
        assert len(actions) == 3
        assert actions[0]["_op_type"] == "update"
        assert actions[0]["doc"] == {"metadata": {"ds_ids": [1, 2]}}

    def test_empty_map_returns_zero(self):
        """Leere ds_ids_map -> 0 ohne Backend-Call."""
        from kiara.retrieval.elasticsearch_service import update_ds_ids

        assert update_ds_ids(_es_config(), {}) == 0

    def test_not_configured_returns_zero(self):
        """ES nicht konfiguriert -> 0."""
        from kiara.retrieval.elasticsearch_service import update_ds_ids

        assert update_ds_ids(_es_config(enabled=False), {"c1": [1]}) == 0

    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_index_not_exists_returns_zero(self, mock_get_client):
        """Index nicht vorhanden -> 0."""
        from kiara.retrieval.elasticsearch_service import update_ds_ids

        es = MagicMock()
        es.indices.exists.return_value = False
        mock_get_client.return_value = es

        assert update_ds_ids(_es_config(), {"c1": [1]}) == 0

    @patch("elasticsearch.helpers.bulk", return_value=(1, [{"update": {"error": "not found"}}]))
    @patch("kiara.retrieval.elasticsearch_service._get_client")
    def test_bulk_errors_logged(self, mock_get_client, mock_bulk):
        """Bulk-Fehler werfen ElasticsearchPartialFailureError fail-loud (seit 0.0.1+395)."""
        from kiara.retrieval.elasticsearch_service import (
            ElasticsearchPartialFailureError,
            update_ds_ids,
        )

        es = MagicMock()
        es.indices.exists.return_value = True
        mock_get_client.return_value = es

        with pytest.raises(ElasticsearchPartialFailureError, match="unvollstaendig"):
            update_ds_ids(_es_config(), {"c1": [1], "c2": [2]})
