"""Fail-closed Tests fuer Admin-Dashboard-KPIs."""

from unittest.mock import patch


def test_backends_stats_db_error_not_productive_zero():
    from kiara.web.api_dashboard import _get_backends_stats

    with patch("kiara.web.api_dashboard.db_query", side_effect=RuntimeError("db down")):
        result = _get_backends_stats()

    assert result["status"] == "error"
    assert result["total"] is None
    assert result["active"] is None
    assert result["by_category"] == []
    assert "db down" in result["error"]


def test_datasources_stats_db_error_not_productive_zero():
    from kiara.web.api_dashboard import _get_datasources_stats

    with patch("kiara.web.api_dashboard.ds_repo.count", side_effect=RuntimeError("db down")):
        result = _get_datasources_stats()

    assert result["status"] == "error"
    assert result["total"] is None
    assert result["active"] is None
    assert result["by_type"] == []
    assert "db down" in result["error"]


def test_jobs_stats_db_error_not_productive_zero():
    from kiara.web.api_dashboard import _get_jobs_stats

    with patch("kiara.web.api_dashboard.db_query", side_effect=RuntimeError("db down")):
        result = _get_jobs_stats()

    assert result["status"] == "error"
    assert result["running"] is None
    assert result["pending"] is None
    assert "db down" in result["error"]
