"""Tests fuer atomaren Full-Refresh des Datasource-Caches."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from kiara.web.cache_refresh import refresh_datasource_cache


@contextmanager
def _db_cm(db):
    yield db


def test_full_refresh_replaces_cache_before_upsert(monkeypatch):
    """Full-Refresh entfernt verwaiste cache_datasources-Zeilen."""
    db = MagicMock()

    monkeypatch.setattr(
        "kiara.web.cache_refresh._load_ds_batch_data",
        lambda _db, ds_id: (
            [{"id": 1, "name": "DS", "type": "local"}],
            {},
            {},
            {},
            {},
            {},
        ),
    )
    monkeypatch.setattr("kiara.web.cache_refresh._load_live_progress", lambda *_: {})
    monkeypatch.setattr(
        "kiara.web.cache_refresh._build_ds_payload",
        lambda *args, **kwargs: (1, "{}"),
    )
    monkeypatch.setattr(
        "kiara.web.cache_refresh.db_transaction",
        lambda: _db_cm(db),
    )

    with (
        patch("kiara.connectors.registry.ensure_loaded"),
        patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None),
    ):
        refresh_datasource_cache()

    sql_calls = [call.args[0] for call in db.execute.call_args_list]
    assert sql_calls[0] == "DELETE FROM cache_datasources"
    assert "INSERT INTO cache_datasources" in sql_calls[1]


def test_full_refresh_with_no_rows_clears_cache(monkeypatch):
    """Wenn keine Datasources existieren, wird der Cache komplett geleert."""
    db = MagicMock()

    monkeypatch.setattr(
        "kiara.web.cache_refresh._load_ds_batch_data",
        lambda _db, ds_id: ([], {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        "kiara.web.cache_refresh.db_transaction",
        lambda: _db_cm(db),
    )

    with patch("kiara.connectors.registry.ensure_loaded"):
        refresh_datasource_cache()

    db.execute.assert_called_once_with("DELETE FROM cache_datasources")
