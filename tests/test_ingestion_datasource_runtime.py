"""Tests fuer kiara.ingestion.datasource_runtime.

Kategorie A: Feld-Paritaet (build_datasource_overrides)
Kategorie B: Connector-Config-Loading (load_connector_config)
Kategorie D: Rueckwaertskompatibilitaet
"""

import json
import logging
from unittest.mock import MagicMock, patch

from kiara.ingestion.datasource_runtime import (
    build_datasource_overrides,
    load_connector_config,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _full_row(**overrides):
    """Vollstaendige DB-Row mit allen 15 Override-Feldern."""
    defaults = {
        "ds_chunk_size": None,
        "ds_chunk_overlap": None,
        "ds_file_types": None,
        "ds_exclude_patterns": None,
        "ds_chunk_strategy": "auto",
        "ds_ddu_enabled": False,
        "ds_ddu_ocr_images": 1,
        "ds_vlm_enabled": False,
        "ds_index_attachments": 1,
        "ds_accept_all_text": 0,
        "ds_contextual_retrieval": 0,
        "ds_enrichment_prompt_id": None,
        "ds_enrichment_backend_id": None,
        "restriction_level": "anonymize",
        "ds_colbert_enabled": False,
    }
    defaults.update(overrides)
    return defaults


def _make_connector_cls(secret_fields=None):
    """Erzeugt einen Mock-Connector mit secret_fields()."""
    cls = MagicMock()
    cls.secret_fields.return_value = secret_fields or []
    return cls


# ═══════════════════════════════════════════════════════════════════════════
# Kategorie A: Feld-Paritaet
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildOverridesFieldParity:
    def test_all_fields_set(self):
        """Alle 15 Felder in Row gesetzt → alle im Dict."""
        row = _full_row(
            ds_chunk_size=500,
            ds_chunk_overlap=100,
            ds_file_types=json.dumps([".pdf", ".txt"]),
            ds_exclude_patterns=json.dumps(["*.bak"]),
            ds_chunk_strategy="prose",
            ds_ddu_enabled=True,
            ds_ddu_ocr_images=0,
            ds_vlm_enabled=True,
            ds_index_attachments=0,
            ds_accept_all_text=1,
            ds_contextual_retrieval=1,
            ds_enrichment_prompt_id=42,
            ds_enrichment_backend_id=7,
            restriction_level="blocked",
            ds_colbert_enabled=True,
        )
        result = build_datasource_overrides(row)
        assert result["chunk_size"] == 500
        assert result["chunk_overlap"] == 100
        assert result["file_types"] == [".pdf", ".txt"]
        assert result["exclude_patterns"] == ["*.bak"]
        assert result["chunk_strategy"] == "prose"
        assert result["ddu_enabled"] is True
        assert result["ddu_ocr_images"] is False
        assert result["vlm_enabled"] is True
        assert result["index_attachments"] is False
        assert result["accept_all_text"] is True
        assert result["contextual_retrieval"] is True
        assert result["enrichment_prompt_id"] == 42
        assert result["enrichment_backend_id"] == 7
        assert result["restriction_level"] == "blocked"
        assert result["colbert_enabled"] is True

    def test_default_row_only_restriction_level(self):
        """Default-Row → nur restriction_level im Dict."""
        row = _full_row()
        result = build_datasource_overrides(row)
        assert result == {"restriction_level": "anonymize"}

    def test_partial_fields(self):
        """Nur Subset gesetzt → nur diese im Dict."""
        row = _full_row(ds_ddu_enabled=True, ds_vlm_enabled=True)
        result = build_datasource_overrides(row)
        assert result["ddu_enabled"] is True
        assert result["vlm_enabled"] is True
        assert "chunk_size" not in result
        assert "colbert_enabled" not in result

    def test_ddu_ocr_images_false(self):
        """ds_ddu_ocr_images=0 → ddu_ocr_images=False."""
        row = _full_row(ds_ddu_ocr_images=0)
        result = build_datasource_overrides(row)
        assert result["ddu_ocr_images"] is False

    def test_ddu_ocr_images_default(self):
        """Spalte fehlt → nicht im Dict (Default True wird in _apply_ds_overrides gesetzt)."""
        row = _full_row()
        del row["ds_ddu_ocr_images"]
        result = build_datasource_overrides(row)
        assert "ddu_ocr_images" not in result

    def test_chunk_strategy_auto_ignored(self):
        """chunk_strategy='auto' → nicht im Dict."""
        row = _full_row(ds_chunk_strategy="auto")
        result = build_datasource_overrides(row)
        assert "chunk_strategy" not in result

    def test_chunk_strategy_set(self):
        """chunk_strategy='prose' → im Dict."""
        row = _full_row(ds_chunk_strategy="prose")
        result = build_datasource_overrides(row)
        assert result["chunk_strategy"] == "prose"

    def test_colbert_enabled_true(self):
        """ds_colbert_enabled=True → colbert_enabled=True."""
        row = _full_row(ds_colbert_enabled=True)
        result = build_datasource_overrides(row)
        assert result["colbert_enabled"] is True

    def test_json_fields_valid(self):
        """Gueltige JSON-Arrays fuer file_types/exclude_patterns."""
        row = _full_row(
            ds_file_types=json.dumps([".doc", ".xlsx"]),
            ds_exclude_patterns=json.dumps(["tmp/*"]),
        )
        result = build_datasource_overrides(row)
        assert result["file_types"] == [".doc", ".xlsx"]
        assert result["exclude_patterns"] == ["tmp/*"]

    def test_json_fields_invalid(self):
        """Ungueltiges JSON → Feld wird ignoriert (kein Crash)."""
        row = _full_row(
            ds_file_types="{broken json",
            ds_exclude_patterns="not-json",
        )
        result = build_datasource_overrides(row)
        assert "file_types" not in result
        assert "exclude_patterns" not in result

    def test_missing_columns_robust(self):
        """Row ohne optionale Spalten → keine Exception."""
        row = {"restriction_level": "anonymize"}
        result = build_datasource_overrides(row)
        assert result == {"restriction_level": "anonymize"}

    def test_restriction_level_string(self):
        """restriction_level ist str, nicht int."""
        row = _full_row(restriction_level="blocked")
        result = build_datasource_overrides(row)
        assert result["restriction_level"] == "blocked"
        assert isinstance(result["restriction_level"], str)


# ═══════════════════════════════════════════════════════════════════════════
# Kategorie B: Connector-Config-Loading
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadConnectorConfig:
    @patch("kiara.shared.crypto.is_encrypted", return_value=True)
    @patch("kiara.shared.crypto.decrypt", return_value="geheim")
    def test_encrypted(self, mock_decrypt, mock_is_enc):
        """Verschluesseltes Feld wird entschluesselt."""
        row = {"connector_config": json.dumps({"password": "ENC:abc"})}
        cls = _make_connector_cls(secret_fields=["password"])
        result = load_connector_config(row, cls)
        assert result["password"] == "geheim"
        mock_decrypt.assert_called_with("ENC:abc")

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_plaintext(self, mock_is_enc):
        """Klartext-Feld bleibt unveraendert."""
        row = {"connector_config": json.dumps({"server": "mail.firma.de"})}
        cls = _make_connector_cls(secret_fields=["password"])
        result = load_connector_config(row, cls)
        assert result["server"] == "mail.firma.de"

    @patch("kiara.shared.crypto.is_encrypted", return_value=True)
    @patch("kiara.shared.crypto.decrypt", side_effect=Exception("decrypt fail"))
    def test_decrypt_error_graceful(self, mock_decrypt, mock_is_enc, caplog):
        """Decrypt-Fehler → leerer String, WARNING-Log, kein Crash."""
        row = {"connector_config": json.dumps({"password": "ENC:broken"})}
        cls = _make_connector_cls(secret_fields=["password"])
        with caplog.at_level(logging.WARNING):
            result = load_connector_config(row, cls)
        assert result["password"] == ""
        assert "Decrypt-Fehler" in caplog.text

    def test_empty_config(self):
        """connector_config='' → {}."""
        row = {"connector_config": ""}
        cls = _make_connector_cls()
        result = load_connector_config(row, cls)
        assert result == {}

    def test_missing_connector_config_key(self):
        """Row ohne connector_config → {}."""
        row = {"id": 1, "name": "test"}
        cls = _make_connector_cls()
        result = load_connector_config(row, cls)
        assert result == {}

    def test_invalid_json_returns_empty_dict(self):
        """Ungueltiges JSON → {}."""
        row = {"connector_config": "{broken"}
        cls = _make_connector_cls()
        result = load_connector_config(row, cls)
        assert result == {}

    def test_none_value_returns_empty_dict(self):
        """connector_config=None → {}."""
        row = {"connector_config": None}
        cls = _make_connector_cls()
        result = load_connector_config(row, cls)
        assert result == {}

    def test_non_dict_json_returns_empty_dict(self):
        """JSON wie [] oder 'foo' wird nicht als Connector-Config akzeptiert."""
        for val in [json.dumps([1, 2, 3]), json.dumps("foo")]:
            row = {"connector_config": val}
            cls = _make_connector_cls()
            result = load_connector_config(row, cls)
            assert result == {}

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    def test_no_secret_fields(self, mock_is_enc):
        """Connector ohne Secrets → Config unveraendert."""
        row = {"connector_config": json.dumps({"path": "/data"})}
        cls = _make_connector_cls(secret_fields=[])
        result = load_connector_config(row, cls)
        assert result == {"path": "/data"}


# ═══════════════════════════════════════════════════════════════════════════
# Kategorie D: Rueckwaertskompatibilitaet
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardsCompatibility:
    def test_matches_old_pipeline_behavior(self):
        """Vergleich mit Erwartungswerten des alten Pipeline-Builders auf vollstaendiger Row.

        Bewusste Abweichungen:
        - ddu_ocr_images wird jetzt gesetzt (Drift-Fix)
        - per-field Error-Handling statt catch-all
        """
        row = _full_row(
            ds_chunk_size=500,
            ds_chunk_overlap=100,
            ds_file_types=json.dumps([".pdf"]),
            ds_exclude_patterns=json.dumps(["*.tmp"]),
            ds_chunk_strategy="email",
            ds_ddu_enabled=True,
            ds_ddu_ocr_images=0,
            ds_vlm_enabled=True,
            ds_index_attachments=0,
            ds_accept_all_text=1,
            ds_contextual_retrieval=1,
            ds_enrichment_prompt_id=5,
            ds_enrichment_backend_id=3,
            restriction_level="blocked",
            ds_colbert_enabled=True,
        )
        result = build_datasource_overrides(row)
        assert result == {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "file_types": [".pdf"],
            "exclude_patterns": ["*.tmp"],
            "chunk_strategy": "email",
            "ddu_enabled": True,
            "ddu_ocr_images": False,
            "vlm_enabled": True,
            "index_attachments": False,
            "accept_all_text": True,
            "contextual_retrieval": True,
            "enrichment_prompt_id": 5,
            "enrichment_backend_id": 3,
            "restriction_level": "blocked",
            "colbert_enabled": True,
        }

    def test_matches_old_worker_behavior(self):
        """Vergleich mit dem alten Worker-Builder.

        Bewusste Abweichung: Return-Type ist dict statt None bei leerem Ergebnis.
        """
        row = _full_row()
        result = build_datasource_overrides(row)
        assert isinstance(result, dict)
        assert result == {"restriction_level": "anonymize"}
