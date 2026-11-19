"""Fehlende Konnektoren duerfen gespeicherte Datenquellen nicht verstecken."""

import asyncio
import json

from kiara.admin.functions.datasources import datasource_list
from kiara.admin.services.datasource_service import create_datasource
from kiara.connectors.registry import ensure_loaded
from kiara.repositories.datasource import DatasourceRepository
from kiara.retrieval.colbert import ColBERTStore
from kiara.web.cache_refresh import refresh_datasource_cache


def test_create_and_list_with_missing_connector(test_db, monkeypatch):
    """Eine vorhandene, unbekannte Quelle blockiert Create + Full-Refresh nicht."""
    ensure_loaded()
    monkeypatch.setattr(ColBERTStore, "list_colbert_collections", lambda _config: [])
    repo = DatasourceRepository()
    missing_id = repo.create(
        name="Archiv mit fehlendem Plugin",
        type="missing_test_connector",
        connector_config=json.dumps({"path": "/archive", "custom_secret": "private-value"}),
    )
    refresh_datasource_cache()

    created = asyncio.run(
        create_datasource(
            name="Neue SMB-Quelle",
            ds_type="smb",
            connector_config={"server": "fileserver", "share": "documents"},
            sync_interval="manual",
        )
    )
    result = asyncio.run(datasource_list({"username": "admin"}))
    assert result.success
    rows = {row["id"]: row for row in result.data["data"]}
    assert rows[created["id"]]["name"] == "Neue SMB-Quelle"
    assert rows[created["id"]]["connector_available"] is True
    missing = rows[missing_id]
    assert missing["connector_available"] is False
    assert missing["connector_config"] == {}
    assert "nicht verfuegbar" in missing["source_display"]
    assert "private-value" not in json.dumps(missing)
    assert "private-value" in repo.get_by_id(missing_id)["connector_config"]

    # Auch ein spaeterer Voll-Refresh darf keine der Quellen wieder verlieren.
    refresh_datasource_cache()
    result = asyncio.run(datasource_list({"username": "admin"}))
    assert {missing_id, created["id"]} <= {row["id"] for row in result.data["data"]}
