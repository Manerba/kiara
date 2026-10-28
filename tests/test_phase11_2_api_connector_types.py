"""Tests fuer Phase 11.2 — /api/admin/connector-types Endpoint.

Testet die Response-Struktur, Sortierung und Konsistenz
des connector-types API-Endpoints.

HINWEIS: list_connector_types wurde aus kiara.web.api_datasource entfernt und als
datasource_connector_types nach kiara.admin.functions.datasources migriert (Sprint 04).
Die neue Funktion wird in test_admin_readonly.py::TestDatasourceConnectorTypes getestet.
"""

import pytest

pytest.skip(
    "list_connector_types migriert zu kiara.admin.functions.datasources.datasource_connector_types "
    "(getestet in test_admin_readonly.py)",
    allow_module_level=True,
)


from kiara.connectors.registry import ensure_loaded

_CONNECTOR_MODULE_PREFIXES = (
    "kiara.connectors.local_connector",
    "kiara.connectors.smb_connector",
    "kiara.connectors.webdav_connector",
    "kiara.connectors.imap_connector",
    "kiara.connectors.sql_ds_connector",
)


@pytest.fixture(autouse=True)
def _load_connectors():
    from kiara.connectors import registry as _reg

    if _reg._loaded and not _reg._registry:
        _reg._loaded = False
    ensure_loaded()


# ---------------------------------------------------------------------------
# 1. Endpoint gibt alle Konnektoren zurueck
# ---------------------------------------------------------------------------


class TestEndpointReturnsAll:
    @pytest.mark.asyncio
    async def test_endpoint_returns_at_least_5(self):
        """Response enthaelt mindestens 5 Eintraege (Built-ins)."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        assert "data" in result
        assert len(result["data"]) >= 5

    @pytest.mark.asyncio
    async def test_each_entry_has_required_keys(self):
        """Jeder Eintrag hat name, label, badge_class, schema, default_config."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        for entry in result["data"]:
            assert "name" in entry, f"Eintrag hat kein 'name': {entry}"
            assert "label" in entry, f"Eintrag hat kein 'label': {entry}"
            assert "badge_class" in entry, f"Eintrag hat kein 'badge_class': {entry}"
            assert "schema" in entry, f"Eintrag hat kein 'schema': {entry}"
            assert "default_config" in entry, f"Eintrag hat kein 'default_config': {entry}"

    @pytest.mark.asyncio
    async def test_all_builtin_types_present(self):
        """Alle 5 Built-in-Typen sind in der Response."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        names = [t["name"] for t in result["data"]]
        for expected in ("local", "smb", "webdav", "imap", "sql"):
            assert expected in names, f"Konnektor '{expected}' fehlt im Endpoint"


# ---------------------------------------------------------------------------
# 2. Sortierung nach Label (H7)
# ---------------------------------------------------------------------------


class TestEndpointSorting:
    @pytest.mark.asyncio
    async def test_sorted_by_label(self):
        """Die Response-Liste ist alphabetisch nach label sortiert."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        labels = [t["label"] for t in result["data"]]
        assert labels == sorted(labels), f"Labels nicht sortiert: {labels}"


# ---------------------------------------------------------------------------
# 3. Schema-Struktur in der Response
# ---------------------------------------------------------------------------


class TestSchemaInResponse:
    @pytest.mark.asyncio
    async def test_schema_is_dict_with_string_keys(self):
        """Jedes schema ist ein Dict mit String-Keys."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        for entry in result["data"]:
            schema = entry["schema"]
            assert isinstance(schema, dict), f"{entry['name']}: schema ist kein Dict"
            for key in schema:
                assert isinstance(key, str), f"{entry['name']}: Schema-Key {key} ist kein String"

    @pytest.mark.asyncio
    async def test_schema_fields_have_required_keys(self):
        """Jedes Schema-Feld hat mindestens type, default, label."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        for entry in result["data"]:
            name = entry["name"]
            for field, spec in entry["schema"].items():
                assert "type" in spec, f"{name}.{field}: kein 'type'"
                assert "default" in spec, f"{name}.{field}: kein 'default'"
                assert "label" in spec, f"{name}.{field}: kein 'label'"


# ---------------------------------------------------------------------------
# 4. default_config in der Response
# ---------------------------------------------------------------------------


class TestDefaultConfigInResponse:
    @pytest.mark.asyncio
    async def test_default_config_is_dict(self):
        """Jedes default_config ist ein Dict."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        for entry in result["data"]:
            assert isinstance(entry["default_config"], dict), (
                f"{entry['name']}: default_config ist kein Dict"
            )

    @pytest.mark.asyncio
    async def test_default_config_keys_match_schema(self):
        """default_config-Keys stimmen mit Schema-Fields ueberein."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        for entry in result["data"]:
            name = entry["name"]
            schema_keys = set(entry["schema"].keys())
            default_keys = set(entry["default_config"].keys())
            assert schema_keys == default_keys, (
                f"{name}: Schema-Keys {schema_keys} != default_config-Keys {default_keys}"
            )

    @pytest.mark.asyncio
    async def test_default_config_values_match_schema_defaults(self):
        """default_config-Werte stimmen mit Schema-Defaults ueberein."""
        from kiara.web.api_datasource import list_connector_types

        mock_user = {"username": "admin", "is_admin": True}
        result = await list_connector_types(user=mock_user)

        for entry in result["data"]:
            name = entry["name"]
            for field, spec in entry["schema"].items():
                expected = spec.get("default", "" if spec.get("type") == "str" else None)
                actual = entry["default_config"].get(field)
                assert actual == expected, (
                    f"{name}.{field}: default_config={actual} != schema default={expected}"
                )
