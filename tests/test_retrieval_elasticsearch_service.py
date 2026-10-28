"""Tests fuer elasticsearch_service: check_missing_ds_ids() und update_ds_ids().

Alle Tests mocken den ES-Client, um keine laufende ES-Instanz zu benoetigen.
"""

from unittest.mock import MagicMock, patch

import elasticsearch.helpers  # noqa: F401 -- real installed client, no global module stub
import pytest

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


@pytest.mark.usefixtures("s4_guard_passthrough")
class TestCheckMissingDsIds:
    @pytest.mark.parametrize("missing", [0, 10])
    def test_membership_totals(self, missing):
        from kiara.retrieval import elasticsearch_service as service
        from kiara.retrieval.elasticsearch_indices import (
            DatasourceIndexAvailability,
            DatasourceIndexTarget,
        )

        target = DatasourceIndexTarget(1, "kiara_test_ds_1", 500)
        counts = {"doc_count": 500, "counts_match": True, "datasources": []}
        with (
            patch(
                "kiara.retrieval.elasticsearch_inventory.inventory",
                return_value=(counts, DatasourceIndexAvailability((target,), (), {1: 500})),
            ),
            patch.object(service, "_get_client"),
            patch.object(
                service,
                "search_explicit_targets",
                return_value={"hits": {"total": {"value": missing, "relation": "eq"}}},
            ) as search,
        ):
            result = service.check_missing_ds_ids(_es_config())
        assert result["total_docs"] == 500
        assert result["missing_ds_ids"] == missing
        assert result["safe_to_proceed"] == (missing == 0)
        assert search.call_args.args[1] == (target,)

    def test_es_error_returns_unsafe(self):
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        with patch(
            "kiara.retrieval.elasticsearch_inventory.inventory",
            side_effect=RuntimeError("missing expected index"),
        ):
            result = check_missing_ds_ids(_es_config())
        assert not result["safe_to_proceed"] and result["total_docs"] is None
        assert "missing expected" in result["error"]

    def test_es_not_configured(self):
        from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

        assert check_missing_ds_ids(_es_config(False))["safe_to_proceed"]

    def test_known_empty(self):
        from kiara.retrieval import elasticsearch_service as service
        from kiara.retrieval.elasticsearch_indices import DatasourceIndexAvailability

        with (
            patch(
                "kiara.retrieval.elasticsearch_inventory.inventory",
                return_value=(
                    {"doc_count": 0, "counts_match": True, "datasources": []},
                    DatasourceIndexAvailability((), (), {}),
                ),
            ),
            patch.object(service, "_get_client") as client,
        ):
            result = service.check_missing_ds_ids(_es_config())
        assert result["safe_to_proceed"] and result["total_docs"] == 0
        client.assert_not_called()


from kiara.retrieval.elasticsearch_indices import document_address


def _address(cid, ds=1):
    return document_address(_es_config(), ds, cid)


def _mget_response(*, docs, **kwargs):
    return {
        "docs": [
            {
                **a,
                "found": True,
                "_seq_no": 1,
                "_primary_term": 1,
                "_source": {"metadata": {"ds_ids": [1]}},
            }
            for a in docs
        ]
    }


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
        es.mget.side_effect = _mget_response

        result = update_ds_ids(
            _es_config(),
            {
                _address("chunk_1"): [1, 2],
                _address("chunk_2", 2): [2],
                _address("chunk_3"): [1, 2, 3],
            },
            refresh=False,
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
    def test_index_not_exists_is_fatal(self, mock_get_client):
        """Index nicht vorhanden -> 0."""
        from kiara.retrieval.elasticsearch_service import update_ds_ids

        es = MagicMock()
        es.indices.exists.return_value = False
        mock_get_client.return_value = es

        from kiara.retrieval.elasticsearch_service import ElasticsearchPartialFailureError

        es.mget.return_value = {
            "docs": [
                {
                    "_index": _address("c1").index_name,
                    "_id": "c1",
                    "error": {"type": "index_not_found_exception"},
                }
            ]
        }
        with pytest.raises(ElasticsearchPartialFailureError):
            update_ds_ids(_es_config(), {_address("c1"): [1]})

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

        es.mget.side_effect = _mget_response
        with pytest.raises(ElasticsearchPartialFailureError, match="unvollstaendig"):
            update_ds_ids(_es_config(), {_address("c1"): [1], _address("c2", 2): [2]})
