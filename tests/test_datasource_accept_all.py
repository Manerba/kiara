"""Tests fuer ds_accept_all_text Flag-Durchreichung (Phase 15).

Testet Pydantic-Model, Overrides-Builder (Pipeline + Worker),
und _apply_ds_overrides().
"""

from kiara.web.api_datasource import DatasourceRequest


class TestDatasourceRequestModel:
    """Pydantic-Model akzeptiert ds_accept_all_text."""

    def test_default_false(self):
        """Default-Wert ist False."""
        req = DatasourceRequest(name="Test")
        assert req.ds_accept_all_text is False

    def test_explicit_true(self):
        """Explizit True wird akzeptiert."""
        req = DatasourceRequest(name="Test", ds_accept_all_text=True)
        assert req.ds_accept_all_text is True

    def test_int_coercion(self):
        """Integer 1 wird zu True konvertiert."""
        req = DatasourceRequest(name="Test", ds_accept_all_text=1)
        assert req.ds_accept_all_text is True


class TestBuildDsOverridesPipeline:
    """build_datasource_overrides() liest ds_accept_all_text."""

    def _make_row(self, accept_all=0):
        """Simuliert eine DB-Row als Dict mit allen noetigen Spalten."""

        class FakeRow(dict):
            def keys(self):
                return list(super().keys())

        return FakeRow(
            {
                "ds_chunk_size": None,
                "ds_chunk_overlap": None,
                "ds_file_types": None,
                "ds_exclude_patterns": None,
                "ds_chunk_strategy": "auto",
                "ds_ddu_enabled": False,
                "ds_vlm_enabled": False,
                "ds_index_attachments": 1,
                "ds_accept_all_text": accept_all,
            }
        )

    def test_accept_all_text_true(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(accept_all=1)
        overrides = build_datasource_overrides(row)
        assert overrides.get("accept_all_text") is True

    def test_accept_all_text_false(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(accept_all=0)
        overrides = build_datasource_overrides(row)
        assert "accept_all_text" not in overrides

    def test_accept_all_text_missing_column(self):
        """Fehlende Spalte (alte DB) → kein Fehler, Default False."""
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        class FakeRow(dict):
            def keys(self):
                return list(super().keys())

        row = FakeRow(
            {
                "ds_chunk_size": None,
                "ds_chunk_overlap": None,
                "ds_file_types": None,
                "ds_exclude_patterns": None,
                "ds_chunk_strategy": "auto",
                "ds_ddu_enabled": False,
                "ds_vlm_enabled": False,
                "ds_index_attachments": 1,
                # ds_accept_all_text fehlt!
            }
        )
        overrides = build_datasource_overrides(row)
        assert "accept_all_text" not in overrides


class TestBuildOverridesWorker:
    """build_datasource_overrides() liest ds_accept_all_text."""

    def _make_row(self, accept_all=0):
        class FakeRow(dict):
            def keys(self):
                return list(super().keys())

        return FakeRow(
            {
                "ds_chunk_size": None,
                "ds_chunk_overlap": None,
                "ds_file_types": None,
                "ds_exclude_patterns": None,
                "ds_chunk_strategy": "auto",
                "ds_ddu_enabled": False,
                "ds_ddu_ocr_images": 1,
                "ds_vlm_enabled": False,
                "ds_index_attachments": 1,
                "ds_accept_all_text": accept_all,
                "ds_contextual_retrieval": 0,
                "ds_enrichment_prompt_id": None,
                "ds_enrichment_backend_id": None,
                "restriction_level": "anonymize",
            }
        )

    def test_accept_all_text_true(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(accept_all=1)
        overrides = build_datasource_overrides(row)
        assert overrides is not None
        assert overrides.get("accept_all_text") is True

    def test_accept_all_text_false(self):
        from kiara.ingestion.datasource_runtime import build_datasource_overrides

        row = self._make_row(accept_all=0)
        overrides = build_datasource_overrides(row)
        # restriction_level wird immer gesetzt, aber keine Feature-Overrides
        assert overrides is not None
        assert "accept_all_text" not in overrides


class TestApplyDsOverrides:
    """_apply_ds_overrides() schreibt accept_all_text in Config."""

    def _base_config(self):
        return {
            "datasource": {
                "path": "/tmp/test",
                "file_types": [".txt"],
                "exclude_patterns": [],
            },
            "ingestion": {
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
        }

    def test_accept_all_text_applied(self):
        from kiara.ingestion.pipeline import _apply_ds_overrides

        config = self._base_config()
        _apply_ds_overrides(config, {"accept_all_text": True})
        assert config["datasource"].get("accept_all_text") is True

    def test_accept_all_text_not_applied(self):
        from kiara.ingestion.pipeline import _apply_ds_overrides

        config = self._base_config()
        _apply_ds_overrides(config, {})
        assert "accept_all_text" not in config["datasource"]

    def test_no_overrides(self):
        from kiara.ingestion.pipeline import _apply_ds_overrides

        config = self._base_config()
        _apply_ds_overrides(config, None)
        assert "accept_all_text" not in config["datasource"]
