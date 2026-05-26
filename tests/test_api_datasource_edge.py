"""Edge-Case-Tests fuer API-Endpoint-Hilfsfunktionen (Phase 11.1).

Testet _mask_row, _source_display, _effective_path, _prepare_ds_config
und den connector-types Endpoint.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors.registry import ensure_loaded
from kiara.web.api_datasource import (
    _effective_path,
    _mask_row,
    _prepare_ds_config,
    _source_display,
)


@pytest.fixture(autouse=True)
def _load_connectors():
    from kiara.connectors import registry as _reg

    if _reg._loaded and not _reg._registry:
        _reg._loaded = False
    ensure_loaded()


def _make_row(data: dict) -> dict:
    """Erzeugt ein Dict (kompatibel mit RealDictCursor-Rows)."""
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = json.dumps(v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# 5.1 _mask_row() — Sonderfaelle
# ---------------------------------------------------------------------------


class TestMaskRow:
    def test_mask_row_unknown_type(self):
        """Unbekannter Typ: kein Crash, connector_config bleibt."""
        row = _make_row({"type": "ftp", "connector_config": "{}", "path": ""})
        result = _mask_row(row)
        assert isinstance(result, dict)
        assert result["connector_config"] == {}

    def test_mask_row_invalid_json_in_connector_config(self):
        """Ungültiges JSON: kein Crash, connector_config = {}."""
        row = _make_row({"type": "imap", "connector_config": "INVALID JSON", "path": ""})
        result = _mask_row(row)
        assert result["connector_config"] == {}

    def test_mask_row_null_connector_config(self):
        """None connector_config: kein Crash."""
        row = _make_row({"type": "imap", "connector_config": None, "path": ""})
        result = _mask_row(row)
        assert result["connector_config"] == {} or isinstance(result["connector_config"], dict)

    def test_mask_row_local_mirrors_path(self):
        """Local-Typ: path wird aus connector_config gespiegelt."""
        row = _make_row(
            {
                "type": "local",
                "connector_config": '{"path": "/mnt/data"}',
                "path": "/old/path",
            }
        )
        result = _mask_row(row)
        assert result["path"] == "/mnt/data"

    def test_mask_row_non_local_no_path_mirror(self):
        """Nicht-Local-Typ: path bleibt der Originalwert."""
        row = _make_row(
            {
                "type": "imap",
                "connector_config": '{"server": "mail.de"}',
                "path": "/original",
            }
        )
        result = _mask_row(row)
        assert result["path"] == "/original"


# ---------------------------------------------------------------------------
# 5.2 _source_display() / _effective_path() — Fehlertoleranz
# ---------------------------------------------------------------------------


class TestSourceDisplayFallback:
    def test_source_display_unknown_type_fallback(self):
        """Unbekannter Typ: Fallback auf row['path']."""
        row = _make_row(
            {
                "type": "ftp",
                "path": "/fallback",
                "connector_config": "{}",
            }
        )
        result = _source_display(row)
        assert result == "/fallback"

    def test_source_display_no_connector_config_key(self):
        """Kein connector_config Key: kein KeyError."""
        row = {"type": "local", "path": "/some/path"}

        result = _source_display(row)
        assert isinstance(result, str)

    def test_effective_path_unknown_type_returns_empty(self):
        """Unbekannter Typ: leerer String."""
        row = _make_row(
            {
                "type": "ftp",
                "connector_config": "{}",
                "path": "",
            }
        )
        result = _effective_path(row)
        assert result == ""


# ---------------------------------------------------------------------------
# 5.3 _prepare_ds_config() — Deepcopy
# ---------------------------------------------------------------------------


class TestPrepareDsConfig:
    @patch("kiara.web.api_datasource_ingest.load_config")
    @patch("kiara.web.api_datasource_ingest.get_connector")
    @patch("kiara.web.api_datasource_ingest._get_connector_config", return_value={"path": "/data"})
    @patch("kiara.web.api_datasource_ingest._decrypt_config", return_value={"path": "/data"})
    def test_prepare_ds_config_does_not_mutate_global_config(
        self, mock_decrypt, mock_get_cc, mock_get_conn, mock_load
    ):
        """Das Original-Dict von load_config() darf nicht mutiert werden."""
        original_config = {
            "datasource": {"path": "/original/path"},
            "ingestion": {"chunk_size": 1000},
        }
        mock_load.return_value = original_config

        mock_cls = MagicMock()
        mock_cls.prepare_and_scan.return_value = "/new/path"
        mock_cls.secret_fields.return_value = []
        mock_get_conn.return_value = mock_cls

        row = _make_row(
            {
                "type": "local",
                "connector_config": '{"path": "/data"}',
            }
        )

        config, error = _prepare_ds_config(row)
        assert error is None
        assert config["datasource"]["path"] == "/new/path"
        # Original darf NICHT veraendert sein
        assert original_config["datasource"]["path"] == "/original/path"


# ---------------------------------------------------------------------------
# 5.4 Connector-Types Endpoint
# ---------------------------------------------------------------------------


class TestConnectorTypesEndpoint:
    @pytest.mark.asyncio
    async def test_connector_types_endpoint(self):
        """datasource.connector_types liefert alle 5 Typen."""
        from kiara.admin.functions.datasources import datasource_connector_types
        from kiara.connectors.registry import ensure_loaded

        ensure_loaded()

        mock_user = {"username": "admin", "is_admin": True}
        result = await datasource_connector_types(user=mock_user)

        assert result.success is True
        by_name = {t["name"]: t for t in result.data["data"]}
        names = list(by_name)
        for expected in ("local", "smb", "webdav", "imap", "exchange_online", "sql"):
            assert expected in names, f"Konnektor '{expected}' fehlt im Endpoint"

        for entry in result.data["data"]:
            assert "name" in entry
            assert "label" in entry
            assert "badge_class" in entry
            assert "schema" in entry
            assert "supports_browse" in entry

        assert by_name["local"]["supports_browse"] is True
        assert by_name["smb"]["supports_browse"] is True
        assert by_name["webdav"]["supports_browse"] is True
        assert by_name["imap"]["supports_browse"] is False
        assert by_name["exchange_online"]["supports_browse"] is False
        assert by_name["sql"]["supports_browse"] is False
