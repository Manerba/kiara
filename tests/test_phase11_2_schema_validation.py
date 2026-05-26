"""Tests fuer Phase 11.2 — Schema-Validierung aller Konnektoren.

Stellt sicher, dass config_schema(), default_config() und secret_fields()
konsistent sind — Grundlage fuer die datengetriebene UI.
"""

import sys

import pytest

from kiara.connectors import registry
from kiara.connectors.registry import all_connectors, ensure_loaded
from kiara.connectors.registry import get as get_connector

_CONNECTOR_MODULE_PREFIXES = (
    "kiara.connectors.local_connector",
    "kiara.connectors.smb_connector",
    "kiara.connectors.webdav_connector",
    "kiara.connectors.imap_connector",
    "kiara.connectors.exchange_connector",
    "kiara.connectors.sql_ds_connector",
)

_VALID_TYPES = {"str", "int", "bool"}


@pytest.fixture(autouse=True)
def _load():
    if not registry._registry:
        for key in list(sys.modules):
            if key.startswith(_CONNECTOR_MODULE_PREFIXES):
                del sys.modules[key]
        registry._loaded = False
        ensure_loaded()
    elif not registry._loaded:
        ensure_loaded()


# ---------------------------------------------------------------------------
# 1. Alle Konnektoren haben gueltige Schema-Typen
# ---------------------------------------------------------------------------


class TestSchemaTypes:
    def test_all_fields_have_valid_type(self):
        """Jedes Feld in config_schema() hat einen type aus str/int/bool."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                assert "type" in spec, f"{name}.{field} hat keinen 'type'"
                assert spec["type"] in _VALID_TYPES, (
                    f"{name}.{field} hat ungueltigen type: {spec['type']}"
                )

    def test_all_fields_have_default(self):
        """Jedes Feld hat einen default-Wert."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                assert "default" in spec, f"{name}.{field} hat keinen 'default'"

    def test_all_fields_have_label(self):
        """Jedes Feld hat ein label."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                assert "label" in spec, f"{name}.{field} hat kein 'label'"
                assert isinstance(spec["label"], str)
                assert len(spec["label"]) > 0, f"{name}.{field} hat leeres label"


# ---------------------------------------------------------------------------
# 2. Secret-Fields sind korrekt markiert
# ---------------------------------------------------------------------------


class TestSecretFields:
    def test_secret_fields_match_schema(self):
        """Felder mit secret=True im Schema erscheinen in secret_fields()."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            schema_secrets = {f for f, spec in schema.items() if spec.get("secret", False)}
            reported_secrets = set(cls.secret_fields())
            assert schema_secrets == reported_secrets, (
                f"{name}: Schema-Secrets {schema_secrets} != secret_fields() {reported_secrets}"
            )

    def test_non_secret_fields_not_in_secret_fields(self):
        """Felder ohne secret=True erscheinen NICHT in secret_fields()."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            secret_list = cls.secret_fields()
            for field, spec in schema.items():
                if not spec.get("secret", False):
                    assert field not in secret_list, (
                        f"{name}.{field} ist in secret_fields() aber hat secret != True im Schema"
                    )


# ---------------------------------------------------------------------------
# 3. default_config() ist konsistent mit config_schema()
# ---------------------------------------------------------------------------


class TestDefaultConfigConsistency:
    def test_all_schema_fields_in_default_config(self):
        """Alle Schema-Felder sind in default_config() enthalten."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            defaults = cls.default_config()
            for field in schema:
                assert field in defaults, f"{name}.default_config() fehlt Feld '{field}'"

    def test_default_values_match_schema(self):
        """Die Default-Werte stimmen ueberein."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            defaults = cls.default_config()
            for field, spec in schema.items():
                expected = spec.get("default", "" if spec.get("type") == "str" else None)
                assert defaults[field] == expected, (
                    f"{name}.{field}: default_config={defaults[field]} != schema default={expected}"
                )

    def test_no_extra_keys_in_default_config(self):
        """Keine Extra-Keys in default_config() die nicht im Schema sind."""
        for name, cls in all_connectors().items():
            schema_keys = set(cls.config_schema().keys())
            default_keys = set(cls.default_config().keys())
            extra = default_keys - schema_keys
            assert not extra, f"{name}.default_config() hat Extra-Keys: {extra}"


# ---------------------------------------------------------------------------
# 4. Required-Felder haben leeren Default
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_required_fields_have_empty_default(self):
        """Felder mit required=True haben default='' (leerer String)."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if spec.get("required", False):
                    assert spec["default"] == "", (
                        f"{name}.{field} ist required, "
                        f"aber default ist '{spec['default']}' statt ''"
                    )

    def test_required_fields_are_string_type(self):
        """Required-Felder sollten String-Typ haben (fuer leeren Default)."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if spec.get("required", False):
                    assert spec["type"] == "str", (
                        f"{name}.{field} ist required, aber type ist '{spec['type']}' statt 'str'"
                    )


# ---------------------------------------------------------------------------
# 5. WebCrawler-Schema im Kontext aller Konnektoren
# ---------------------------------------------------------------------------


class TestWebCrawlerSchema:
    def _get_webcrawler(self):
        cls = get_connector("webcrawler")
        if cls is None:
            pytest.skip("WebCrawler-Plugin nicht installiert")
        return cls

    def test_webcrawler_has_14_fields(self):
        """WebCrawler hat genau 14 Schema-Felder."""
        cls = self._get_webcrawler()
        assert len(cls.config_schema()) == 14

    def test_start_url_is_only_required_field(self):
        """start_url ist das einzige required-Feld."""
        cls = self._get_webcrawler()
        schema = cls.config_schema()
        required = [f for f, spec in schema.items() if spec.get("required", False)]
        assert required == ["start_url"]

    def test_auth_password_is_only_secret(self):
        """auth_password ist das einzige Secret-Feld."""
        cls = self._get_webcrawler()
        assert cls.secret_fields() == ["auth_password"]

    def test_request_delay_is_string_type(self):
        """request_delay ist ein String-Typ (kein Float, da HTML-Input)."""
        cls = self._get_webcrawler()
        schema = cls.config_schema()
        assert schema["request_delay"]["type"] == "str"

    def test_webcrawler_appears_in_all_connectors(self):
        """WebCrawler erscheint in all_connectors()."""
        cls = self._get_webcrawler()
        assert "webcrawler" in all_connectors()


# ---------------------------------------------------------------------------
# 6. Options-Felder haben korrekte Struktur
# ---------------------------------------------------------------------------


class TestOptionsFields:
    def test_options_are_list_of_dicts(self):
        """options ist eine Liste von Dicts mit value und label."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if "options" in spec:
                    assert isinstance(spec["options"], list), (
                        f"{name}.{field}.options ist keine Liste"
                    )
                    for i, opt in enumerate(spec["options"]):
                        assert isinstance(opt, dict), f"{name}.{field}.options[{i}] ist kein Dict"
                        assert "value" in opt, f"{name}.{field}.options[{i}] hat kein 'value'"
                        assert "label" in opt, f"{name}.{field}.options[{i}] hat kein 'label'"

    def test_options_values_are_unique(self):
        """Option-Values sind eindeutig."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if "options" in spec:
                    values = [o["value"] for o in spec["options"]]
                    assert len(values) == len(set(values)), (
                        f"{name}.{field}.options hat doppelte Values: {values}"
                    )

    def test_default_is_valid_option_or_empty(self):
        """Default-Wert ist einer der option-Values oder leer."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if "options" in spec:
                    values = [o["value"] for o in spec["options"]]
                    default = spec.get("default", "")
                    if default:
                        assert default in values, (
                            f"{name}.{field}: Default '{default}' "
                            f"ist keiner der Options-Values {values}"
                        )


# ---------------------------------------------------------------------------
# 7. Browse-Felder haben korrekte Abhaengigkeiten
# ---------------------------------------------------------------------------


class TestBrowseFields:
    def test_browse_depends_on_references_existing_fields(self):
        """browse.depends_on referenziert nur existierende Schema-Felder."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if "browse" in spec:
                    browse = spec["browse"]
                    assert "depends_on" in browse, f"{name}.{field}.browse hat kein 'depends_on'"
                    assert "endpoint" in browse, f"{name}.{field}.browse hat kein 'endpoint'"
                    for dep in browse["depends_on"]:
                        assert dep in schema, (
                            f"{name}.{field}.browse.depends_on "
                            f"referenziert '{dep}', existiert nicht im Schema"
                        )

    def test_browse_endpoint_is_valid_path(self):
        """browse.endpoint ist ein gueltiger URL-Pfad."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if "browse" in spec:
                    endpoint = spec["browse"]["endpoint"]
                    assert isinstance(endpoint, str)
                    assert endpoint.startswith("/"), (
                        f"{name}.{field}.browse.endpoint '{endpoint}' beginnt nicht mit '/'"
                    )


# ---------------------------------------------------------------------------
# 8. Hint-Texte sind Strings und nicht leer
# ---------------------------------------------------------------------------


class TestHintTexts:
    def test_hints_are_non_empty_strings(self):
        """Wenn ein Feld hint hat, ist es ein nicht-leerer String."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field, spec in schema.items():
                if "hint" in spec:
                    assert isinstance(spec["hint"], str), f"{name}.{field}.hint ist kein String"
                    assert len(spec["hint"]) > 0, f"{name}.{field}.hint ist leer"

    def test_hint_count(self):
        """Mindestens einige Konnektoren haben Hints."""
        hint_count = 0
        for _name, cls in all_connectors().items():
            schema = cls.config_schema()
            for _field, spec in schema.items():
                if "hint" in spec:
                    hint_count += 1
        assert hint_count >= 5, f"Nur {hint_count} Hints gefunden, erwartet mindestens 5"
