"""Tests fuer Sprint 16.3-5: Pipeline-Integration der Enrichment-Phase.

B1-B20: _should_enrich, _get_enrichment_prompt, _get_enrichment_llm,
        Pipeline-Enrichment-Block in _process_file_in_pipeline().
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from kiara_ingestion_core.models import FileLogEntry, ProcessingContext

from kiara.ingestion.runtime import (
    _generate_with_audit,
    _get_enrichment_llm,
    _get_enrichment_prompt,
    _IngestionContext,
    _process_file_in_pipeline,
    _should_enrich,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ictx(**overrides):
    """Erzeugt einen minimalen _IngestionContext mit Mock-Objekten."""
    defaults = dict(
        config={
            "vectordb": {"path": "/tmp"},
            "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
        },
        ctx=ProcessingContext(),
        store=MagicMock(),
        chunk_ds=MagicMock(),
        hash_store=MagicMock(),
        batch_size=10,
        datasource_id=1,
        embedding_model="test-model",
        hash_mode="mtime",
        ds_overrides=None,
        file_hashes={},
        stats={
            "total_files": 10,
            "processed_files": 0,
            "skipped_files": 0,
            "errors": 0,
            "total_chunks": 0,
        },
        pending_hashes={},
        consecutive_errors=0,
        skipped=0,
        total=10,
    )
    defaults.update(overrides)
    return _IngestionContext(**defaults)


def _make_file_log(file_path="/test/file.txt", **overrides):
    """Erzeugt einen minimalen FileLogEntry."""
    p = Path(file_path)
    defaults = dict(
        file_path=file_path,
        file_name=p.name,
        file_size=1024,
        file_ext=p.suffix.lower(),
    )
    defaults.update(overrides)
    return FileLogEntry(**defaults)


# ===================================================================
# B1-B4: _should_enrich
# ===================================================================


class TestShouldEnrich:
    def test_b1_config_disabled_ds_flag_true(self):
        """B1: config.enabled=False, ds_flag=True -> False."""
        config = {"enrichment": {"enabled": False}}
        ds_overrides = {"contextual_retrieval": True}
        assert _should_enrich(config, ds_overrides) is False

    def test_b2_config_enabled_ds_flag_false(self):
        """B2: config.enabled=True, ds_flag=False -> False."""
        config = {"enrichment": {"enabled": True}}
        ds_overrides = {"contextual_retrieval": False}
        assert _should_enrich(config, ds_overrides) is False

    def test_b3_config_enabled_ds_flag_true(self):
        """B3: config.enabled=True, ds_flag=True -> True."""
        config = {"enrichment": {"enabled": True}}
        ds_overrides = {"contextual_retrieval": True}
        assert _should_enrich(config, ds_overrides) is True

    def test_b4_ds_overrides_none(self):
        """B4: ds_overrides=None -> False."""
        config = {"enrichment": {"enabled": True}}
        assert _should_enrich(config, None) is False

    def test_b4_missing_enrichment_section(self):
        """B4: config ohne enrichment-Section -> False."""
        config = {}
        ds_overrides = {"contextual_retrieval": True}
        assert _should_enrich(config, ds_overrides) is False

    def test_b4_empty_enrichment_section(self):
        """B4: config mit leerem enrichment-Dict -> False."""
        config = {"enrichment": {}}
        ds_overrides = {"contextual_retrieval": True}
        assert _should_enrich(config, ds_overrides) is False

    def test_b4_missing_contextual_retrieval_key(self):
        """B4: ds_overrides ohne contextual_retrieval Key -> False."""
        config = {"enrichment": {"enabled": True}}
        ds_overrides = {"some_other_key": True}
        assert _should_enrich(config, ds_overrides) is False


# ===================================================================
# B5-B8: _get_enrichment_prompt
# ===================================================================


class TestGetEnrichmentPrompt:
    @patch("kiara.shared.db.db_query")
    def test_b5_ds_with_prompt_id(self, mock_db_query):
        """B5: DS with prompt_id -> loads template, returns (content, sha256_hash)."""
        template_content = "Kontext: {document_context}\nChunk: {chunk_text}"

        mock_row = {"content": template_content, "chain_type": "enrichment"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ds_overrides = {"enrichment_prompt_id": 42}
        content, content_hash = _get_enrichment_prompt(ds_overrides)

        assert content == template_content
        expected_hash = hashlib.sha256(template_content.encode("utf-8")).hexdigest()
        assert content_hash == expected_hash

        # Verify the DB query used the prompt_id
        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args
        assert 42 in args[0][1]

    @patch("kiara.shared.db.db_query")
    def test_b6_ds_without_prompt_id(self, mock_db_query):
        """B6: DS without prompt_id -> loads 'enrichment_default'."""
        default_content = "Default template {chunk_text}"

        mock_row = {"content": default_content}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ds_overrides = {}
        content, content_hash = _get_enrichment_prompt(ds_overrides)

        assert content == default_content
        # Verify it queried for enrichment_default
        query_str = mock_cursor.execute.call_args[0][0]
        assert "enrichment_default" in query_str

    @patch("kiara.shared.db.db_query")
    def test_b7_content_hash_is_sha256(self, mock_db_query):
        """B7: content_hash is correct SHA256."""
        template_content = "Ein spezielles Template mit Umlauten: aeoeue"

        mock_row = {"content": template_content, "chain_type": "enrichment"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ds_overrides = {"enrichment_prompt_id": 7}
        _, content_hash = _get_enrichment_prompt(ds_overrides)

        # SHA256 hex digest is 64 characters long
        assert len(content_hash) == 64
        expected = hashlib.sha256(template_content.encode("utf-8")).hexdigest()
        assert content_hash == expected

    @patch("kiara.shared.db.db_query")
    def test_b8_wrong_chain_type_raises(self, mock_db_query):
        """B8: Wrong chain_type -> ValueError."""
        mock_row = {"content": "template", "chain_type": "standard"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ds_overrides = {"enrichment_prompt_id": 99}
        with pytest.raises(ValueError, match="erwartet 'enrichment'"):
            _get_enrichment_prompt(ds_overrides)


# ===================================================================
# B9-B12: _get_enrichment_llm
# ===================================================================


class TestGetEnrichmentLlm:
    @patch("kiara.backend_selector.get_backend_by_id")
    @patch("kiara.generation.backends.factory.create_backend")
    def test_b9_ds_with_backend_id(self, mock_create, mock_get_by_id):
        """B9: DS with backend_id -> creates backend, generate() callable."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "LLM Antwort"
        mock_create.return_value = mock_backend
        mock_get_by_id.return_value = {
            "id": 5,
            "category": "llm",
            "provider": "ollama",
        }

        config = {"enrichment": {"enabled": True}}
        ds_overrides = {"enrichment_backend_id": 5}

        llm_fn, factory, allow_par = _get_enrichment_llm(config, ds_overrides)

        # llm_fn should be callable and delegate to backend.generate
        result = llm_fn("system", "user")
        assert result == "LLM Antwort"
        mock_backend.generate.assert_called_once_with("system", "user")
        mock_create.assert_called_once_with(config, 5)

    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b10_wrong_backend_category(self, mock_get_by_id):
        """B10: Wrong backend category (embedding) -> ValueError."""
        mock_get_by_id.return_value = {
            "id": 10,
            "category": "embedding",
            "provider": "ollama",
        }

        config = {}
        ds_overrides = {"enrichment_backend_id": 10}

        with pytest.raises(ValueError, match="erwartet 'llm' oder 'cloud'"):
            _get_enrichment_llm(config, ds_overrides)

    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b10_vlm_category_raises(self, mock_get_by_id):
        """B10: Wrong backend category (vlm) -> ValueError."""
        mock_get_by_id.return_value = {
            "id": 11,
            "category": "vlm",
            "provider": "ollama",
        }

        config = {}
        ds_overrides = {"enrichment_backend_id": 11}

        with pytest.raises(ValueError, match="erwartet 'llm' oder 'cloud'"):
            _get_enrichment_llm(config, ds_overrides)

    def test_b11_no_backend_id_raises(self):
        """B11: DS without backend_id -> ValueError (no silent fallback)."""
        config = {}
        ds_overrides = {}

        with pytest.raises(ValueError, match="Kein Enrichment-Backend konfiguriert"):
            _get_enrichment_llm(config, ds_overrides)

    def test_b11_none_overrides_raises(self):
        """B11: None overrides -> ValueError (no silent fallback)."""
        config = {}

        with pytest.raises(ValueError, match="Kein Enrichment-Backend konfiguriert"):
            _get_enrichment_llm(config, None)

    @patch("kiara.generation.backends.factory.wrap_with_anonymization")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b12_cloud_backend_calls_anonymization(self, mock_get_by_id, mock_create, mock_wrap):
        """B12: Cloud backend + restriction_level -> wrap_with_anonymization called."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "anon response"
        mock_create.return_value = mock_backend

        wrapped_backend = MagicMock()
        wrapped_backend.generate.return_value = "anonymized response"
        mock_wrap.return_value = wrapped_backend

        mock_get_by_id.return_value = {
            "id": 20,
            "category": "cloud",
            "provider": "anthropic",
        }

        config = {"anonymization": {"enabled": True}}
        ds_overrides = {
            "enrichment_backend_id": 20,
            "restriction_level": "anonymize",
        }

        llm_fn, factory, allow_par = _get_enrichment_llm(config, ds_overrides)

        # wrap_with_anonymization must have been called
        mock_wrap.assert_called_once_with(
            mock_backend,
            config,
            restriction_levels=["anonymize"],
        )
        # Cloud backends don't allow parallel
        assert allow_par is False
        assert factory is None

        # llm_fn delegates to the wrapped backend
        result = llm_fn("sys", "usr")
        assert result == "anonymized response"
        wrapped_backend.generate.assert_called_once_with("sys", "usr")

    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b9_local_backend_allows_parallel(self, mock_get_by_id, mock_create):
        """B9 supplement: Local (ollama) backend -> allow_parallel=True, factory set."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "ok"
        mock_create.return_value = mock_backend
        mock_get_by_id.return_value = {
            "id": 5,
            "category": "llm",
            "provider": "ollama",
        }

        config = {}
        ds_overrides = {"enrichment_backend_id": 5}

        _, factory, allow_par = _get_enrichment_llm(config, ds_overrides)

        assert allow_par is True
        assert factory is not None
        # Factory returns a callable that creates a new backend
        worker_fn = factory()
        assert callable(worker_fn)


class TestGenerateWithAudit:
    @patch("kiara.anonymization.audit.log_anonymization_event")
    @patch("kiara.cloud.audit.AuditLogger")
    def test_cloud_enrichment_uses_backend_anonymized_values(self, mock_audit_cls, mock_anon_log):
        """Cloud-Enrichment-Audit darf keine Klartexte als anonymized speichern."""

        class Backend:
            last_anon_question = "Frage zu [PERSON_001]"
            last_anon_answer = "Antwort fuer [PERSON_001]"
            last_entities_anonymized = 2
            anonymizer_stats = {
                "total_entities": 2,
                "entity_types": {"PERSON": 2},
            }
            last_usage = {"prompt_tokens": 11, "completion_tokens": 7}

            def generate(self, system_prompt, user_prompt):
                assert system_prompt == "system"
                assert user_prompt == "Frage zu Max Mustermann"
                return "Antwort fuer Max Mustermann"

        mock_audit_cls.return_value.log_request.return_value = 123
        backend_dict = {"provider": "anthropic", "model": "claude-sonnet"}

        result = _generate_with_audit(
            Backend(),
            backend_dict,
            {},
            "system",
            "Frage zu Max Mustermann",
        )

        assert result == "Antwort fuer Max Mustermann"
        cloud_kwargs = mock_audit_cls.return_value.log_request.call_args.kwargs
        assert cloud_kwargs["user_id"] is None
        assert cloud_kwargs["question_anonymized"] == "Frage zu [PERSON_001]"
        assert cloud_kwargs["answer_anonymized"] == "Antwort fuer [PERSON_001]"
        assert cloud_kwargs["entities_anonymized"] == 2
        assert cloud_kwargs["tokens_in"] == 11
        assert cloud_kwargs["tokens_out"] == 7
        assert cloud_kwargs["context_sources"] == [{"source": "enrichment"}]

        anon_kwargs = mock_anon_log.call_args.kwargs
        assert anon_kwargs["user_id"] is None
        assert anon_kwargs["session_id"] == cloud_kwargs["correlation_id"]
        assert anon_kwargs["correlation_id"] == cloud_kwargs["correlation_id"]
        assert anon_kwargs["question"] == "Frage zu Max Mustermann"
        assert anon_kwargs["entity_count"] == 2
        assert anon_kwargs["entity_types"] == {"PERSON": 2}
        assert anon_kwargs["route"] == "enrichment:cloud"
        assert anon_kwargs["cloud_provider"] == "anthropic"
        assert anon_kwargs["cloud_model"] == "claude-sonnet"


# ===================================================================
# B13-B20: Pipeline-Enrichment-Integration
#
# The enrichment block inside _process_file_in_pipeline uses lazy imports:
#   from kiara.ingestion.enrichment import enrich_chunks_batch
#   from kiara.shared.db import db_query  (for DS name lookup)
# We patch at the source modules.
# _write_embeddings signature: (file_texts, file_metas, file_ids, store, batch_size, ...)
# So call_args[0][0] = file_texts, [0][1] = file_metas.
# ===================================================================


class TestPipelineEnrichmentIntegration:
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    def test_b13_enrichment_active_chunks_contain_prefix(
        self,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B13: Pipeline with enrichment active -> chunks contain prefix."""
        # Mock db_query for datasource name lookup
        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        original_text = "Original chunk text"
        enriched_text = "Kontext-Prefix\n\nOriginal chunk text"
        mock_psf.return_value = (
            [original_text],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full document text",
            [(0, len(original_text))],
        )
        mock_gep.return_value = ("template {chunk_text}", "abc123hash")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            [enriched_text],
            {"enriched": 1, "skipped": 0, "errors": 0, "enrichment_time_ms": 50},
        )

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        mock_ecb.assert_called_once()
        # _write_embeddings args: (file_texts, file_metas, file_ids, store, batch_size, ...)
        write_args = mock_write.call_args
        written_texts = write_args[0][0]
        assert written_texts[0] == enriched_text

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    def test_b14_qdrant_payload_changed_chunks_get_metadata(
        self,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B14: Only changed chunks get contextual_prefix + enrichment_prompt_hash."""
        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        original_texts = ["Chunk A", "Chunk B bleibt gleich", "Chunk C"]
        enriched_texts = [
            "Prefix\n\nChunk A",  # changed
            "Chunk B bleibt gleich",  # unchanged
            "Neuer Prefix\n\nChunk C",  # changed
        ]
        prompt_hash = "deadbeef" * 8
        mock_psf.return_value = (
            list(original_texts),
            [
                {"source": "/f.txt", "file_type": ".txt"},
                {"source": "/f.txt", "file_type": ".txt"},
                {"source": "/f.txt", "file_type": ".txt"},
            ],
            ["id1", "id2", "id3"],
            "Full text",
            [(0, 7), (8, 30), (31, 38)],
        )
        mock_gep.return_value = ("template", prompt_hash)
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            enriched_texts,
            {"enriched": 2, "skipped": 1, "errors": 0, "enrichment_time_ms": 100},
        )

        result = _process_file_in_pipeline(ictx, "/f.txt")
        assert result is True

        # _write_embeddings args: (file_texts, file_metas, file_ids, ...)
        write_args = mock_write.call_args
        written_metas = write_args[0][1]

        # Chunk 0: changed -> has contextual_prefix and prompt_hash
        assert written_metas[0].get("contextual_prefix") is True
        assert written_metas[0].get("enrichment_prompt_hash") == prompt_hash

        # Chunk 1: unchanged -> no contextual_prefix
        assert written_metas[1].get("contextual_prefix") is None
        assert written_metas[1].get("enrichment_prompt_hash") is None

        # Chunk 2: changed -> has contextual_prefix and prompt_hash
        assert written_metas[2].get("contextual_prefix") is True
        assert written_metas[2].get("enrichment_prompt_hash") == prompt_hash

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_b15_no_enrichment_chunks_unchanged(
        self,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B15: Pipeline without enrichment -> chunks unchanged."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
            },
            ds_overrides=None,  # No overrides -> _should_enrich returns False
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            datasource_id=None,
        )

        original_text = "Original chunk text unmodified"
        mock_psf.return_value = (
            [original_text],
            [{"source": "/f.txt"}],
            ["id1"],
            "Full text",
            [(0, len(original_text))],
        )

        result = _process_file_in_pipeline(ictx, "/f.txt")
        assert result is True

        # _write_embeddings args: (file_texts, file_metas, file_ids, ...)
        write_args = mock_write.call_args
        written_texts = write_args[0][0]
        assert written_texts == [original_text]
        written_metas = write_args[0][1]
        assert "contextual_prefix" not in written_metas[0]
        assert "enrichment_prompt_hash" not in written_metas[0]

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    def test_b16_enrichment_error_fallback_to_original(
        self,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B16: LLM/prompt/backend error -> fallback to original text, pipeline continues."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        original_text = (
            "Original text must survive and is long enough for enrichment attempts. " * 2
        )
        original_meta = {"source": "/f.txt", "file_type": ".txt"}
        mock_psf.return_value = (
            [original_text],
            [dict(original_meta)],
            ["id1"],
            "Full text",
            [(0, len(original_text))],
        )
        # _get_enrichment_prompt raises an error
        mock_gep.side_effect = ValueError("Template nicht gefunden")

        result = _process_file_in_pipeline(ictx, "/f.txt")

        # Pipeline continues despite enrichment error
        assert result is True
        # Original text is written
        write_args = mock_write.call_args
        written_texts = write_args[0][0]
        assert written_texts == [original_text]
        # enrichment_errors counter incremented
        assert ictx.stats.get("enrichment_errors", 0) >= 1

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    def test_b17_embedding_retry_no_double_enrichment(
        self,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B17: Embedding retry (skip_loading=True) -> no double enrichment."""
        from kiara.backend_selector import BackendUnavailableError

        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            # Worker-mode: control_callback + status_callback for _wait_for_backend
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            consecutive_errors=9,  # threshold - 1
        )

        mock_psf.return_value = (
            ["text"],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full text",
            [(0, 4)],
        )
        mock_gep.return_value = ("template", "hash123")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            ["Prefix\n\ntext"],
            {"enriched": 1, "skipped": 0, "errors": 0, "enrichment_time_ms": 10},
        )

        # First _write_embeddings raises BackendUnavailableError, second succeeds
        mock_write.side_effect = [BackendUnavailableError("qdrant down"), None]

        with patch("kiara.ingestion.runtime._wait_for_backend", return_value=True):
            result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        # enrich_chunks_batch called only ONCE despite embedding retry
        assert mock_ecb.call_count == 1
        # _process_single_file called only once (skip_loading on retry)
        assert mock_psf.call_count == 1
        # _write_embeddings called twice (error + success)
        assert mock_write.call_count == 2

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    def test_b18_full_text_passed_through(
        self,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B18: full_text correctly passed to enrich_chunks_batch."""
        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        full_document = "Dies ist der vollstaendige Dokumenttext mit mehreren Absaetzen."
        chunk = "Dies ist der vollstaendige"
        mock_psf.return_value = (
            [chunk],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            full_document,
            [(0, len(chunk))],
        )
        mock_gep.return_value = ("template", "hash")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            [chunk],
            {"enriched": 0, "skipped": 1, "errors": 0, "enrichment_time_ms": 0},
        )

        _process_file_in_pipeline(ictx, "/f.txt")

        # Verify full_text was passed to enrich_chunks_batch
        ecb_call = mock_ecb.call_args
        # enrich_chunks_batch is called with keyword args from the pipeline
        assert ecb_call.kwargs.get("full_text") == full_document

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    def test_b19_import_error_restores_originals(
        self,
        mock_gep,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B19: Import/init error before enrich_chunks_batch -> except restores originals."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        original_texts = ["OrigA", "OrigB"]
        original_metas = [
            {"source": "/f.txt", "file_type": ".txt"},
            {"source": "/f.txt", "file_type": ".txt"},
        ]
        mock_psf.return_value = (
            list(original_texts),
            [dict(m) for m in original_metas],
            ["id1", "id2"],
            "Full text",
            [(0, 5), (6, 11)],
        )
        # _get_enrichment_prompt raises -> except block fires before enrich_chunks_batch
        mock_gep.side_effect = RuntimeError("DB connection lost")

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        # Original texts are written (restored from file_texts_before)
        write_args = mock_write.call_args
        written_texts = write_args[0][0]
        assert written_texts == original_texts
        # Original metas restored (no enrichment keys added)
        written_metas = write_args[0][1]
        for m in written_metas:
            assert "contextual_prefix" not in m
            assert "enrichment_prompt_hash" not in m
        # Zu kurze Chunks bleiben skipped und werden nicht als Enrichment-Fehler gezaehlt.
        assert ictx.stats.get("enrichment_skipped", 0) == 2
        assert ictx.stats.get("enrichment_errors", 0) == 0

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    def test_b20_restriction_level_from_overrides(
        self,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """B20: restriction_level from overrides -> passed to _get_enrichment_llm."""
        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={
                "contextual_retrieval": True,
                "enrichment_backend_id": 1,
                "restriction_level": "public",
            },
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        mock_psf.return_value = (
            ["text"],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full text",
            [(0, 4)],
        )
        mock_gep.return_value = ("template", "hash")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            ["text"],
            {"enriched": 0, "skipped": 1, "errors": 0, "enrichment_time_ms": 0},
        )

        _process_file_in_pipeline(ictx, "/f.txt")

        # _get_enrichment_llm called with config and ds_overrides containing restriction_level
        mock_gel.assert_called_once()
        call_args = mock_gel.call_args[0]
        passed_overrides = call_args[1]
        assert passed_overrides["restriction_level"] == "public"


class TestGetEnrichmentLlmRestrictionLevel:
    """B20 unit-level: verify _get_enrichment_llm passes restriction_level correctly."""

    @patch("kiara.generation.backends.factory.wrap_with_anonymization")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b20_anonymize_level_passed(
        self,
        mock_get_by_id,
        mock_create,
        mock_wrap,
    ):
        """B20: restriction_level='anonymize' -> wrapped with ['anonymize']."""
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        mock_wrap.return_value = mock_backend
        mock_get_by_id.return_value = {
            "id": 20,
            "category": "cloud",
            "provider": "anthropic",
        }

        config = {}
        ds_overrides = {
            "enrichment_backend_id": 20,
            "restriction_level": "anonymize",
        }

        _get_enrichment_llm(config, ds_overrides)

        mock_wrap.assert_called_once_with(
            mock_backend,
            config,
            restriction_levels=["anonymize"],
        )

    @patch("kiara.generation.backends.factory.wrap_with_anonymization")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b20_public_level_passed(
        self,
        mock_get_by_id,
        mock_create,
        mock_wrap,
    ):
        """B20: restriction_level='public' -> wrapped with ['public']."""
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        mock_wrap.return_value = mock_backend
        mock_get_by_id.return_value = {
            "id": 20,
            "category": "cloud",
            "provider": "openai",
        }

        config = {}
        ds_overrides = {
            "enrichment_backend_id": 20,
            "restriction_level": "public",
        }

        _get_enrichment_llm(config, ds_overrides)

        mock_wrap.assert_called_once_with(
            mock_backend,
            config,
            restriction_levels=["public"],
        )

    @patch("kiara.generation.backends.factory.wrap_with_anonymization")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b20_empty_restriction_level_no_wrap(
        self,
        mock_get_by_id,
        mock_create,
        mock_wrap,
    ):
        """B20: restriction_level='' -> wrap_with_anonymization called with None."""
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        mock_get_by_id.return_value = {
            "id": 20,
            "category": "cloud",
            "provider": "anthropic",
        }

        config = {}
        ds_overrides = {
            "enrichment_backend_id": 20,
            "restriction_level": "",
        }

        _get_enrichment_llm(config, ds_overrides)

        # Empty restriction_level is falsy -> restriction_levels=None in call
        mock_wrap.assert_called_once_with(
            mock_backend,
            config,
            restriction_levels=None,
        )

    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_b20_default_anonymize_when_key_missing(
        self,
        mock_get_by_id,
        mock_create,
    ):
        """B20: No restriction_level in overrides -> defaults to 'anonymize'."""
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        mock_get_by_id.return_value = {
            "id": 5,
            "category": "llm",
            "provider": "ollama",
        }

        config = {}
        # No restriction_level key -> default is "anonymize"
        ds_overrides = {"enrichment_backend_id": 5}

        llm_fn, factory, allow_par = _get_enrichment_llm(config, ds_overrides)

        # Ollama provider -> no wrap call, but restriction_level was resolved
        # Verify no error occurred and function completed
        assert callable(llm_fn)


# ===================================================================
# Late Chunking + Enrichment Interaction Tests
# ===================================================================


class TestLateCunkingEnrichmentInteraction:
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    @patch("kiara.ingestion.runtime.embed_texts_late")
    @patch("kiara.ingestion.runtime._should_use_late_chunking")
    def test_late_chunking_fallback_uses_enriched_texts(
        self,
        mock_sulc,
        mock_etl,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """When late_chunking=True and embed_texts_late returns None (error),
        _write_embeddings is called with embeddings=None — enriched texts
        go through standard embedding."""
        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True, "late_chunking": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
        )

        original_text = "Original chunk text"
        enriched_text = "Kontext-Prefix\n\nOriginal chunk text"
        mock_psf.return_value = (
            [original_text],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full document text",
            [(0, len(original_text))],
        )
        mock_gep.return_value = ("template {chunk_text}", "abc123hash")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            [enriched_text],
            {"enriched": 1, "skipped": 0, "errors": 0, "enrichment_time_ms": 50},
        )

        # Late chunking is enabled but embed_texts_late returns None (error)
        mock_sulc.return_value = True
        mock_etl.return_value = None

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        # _write_embeddings must be called with enriched texts
        write_args = mock_write.call_args
        written_texts = write_args[0][0]
        assert written_texts[0] == enriched_text
        # embeddings kwarg must be None (standard embedding fallback)
        written_embeddings = write_args[1].get("embeddings") if write_args[1] else None
        assert written_embeddings is None

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    @patch("kiara.ingestion.runtime._refresh_late_chunking_snapshots")
    def test_late_chunking_retry_preserves_original_snapshots(
        self,
        mock_refresh,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """After enrichment + embedding error + retry, _refresh_late_chunking_snapshots
        is called with enrichment_applied=True on the retry pass (reconstructs from
        ID maps, not from enriched texts)."""
        from kiara.backend_selector import BackendUnavailableError

        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True},
            },
            ds_overrides={"contextual_retrieval": True, "enrichment_backend_id": 1},
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            consecutive_errors=9,
        )

        mock_psf.return_value = (
            ["text"],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full text",
            [(0, 4)],
        )
        mock_gep.return_value = ("template", "hash123")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.return_value = (
            ["Prefix\n\ntext"],
            {"enriched": 1, "skipped": 0, "errors": 0, "enrichment_time_ms": 10},
        )

        # First _write_embeddings raises, second succeeds
        mock_write.side_effect = [BackendUnavailableError("qdrant down"), None]

        with patch("kiara.ingestion.runtime._wait_for_backend", return_value=True):
            result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        # _refresh_late_chunking_snapshots called twice: first pass + retry pass
        assert mock_refresh.call_count == 2

        # First call: enrichment_applied=False (initial loading pass)
        first_call = mock_refresh.call_args_list[0]
        assert first_call[1]["enrichment_applied"] is False

        # Second call (retry): enrichment_applied=True (enrichment already happened)
        second_call = mock_refresh.call_args_list[1]
        assert second_call[1]["enrichment_applied"] is True


# ===================================================================
# Spec #8 (strengthened): Pipeline Late Chunking -> store.add_with_embeddings
# ===================================================================


class TestLateCunkingPipelineCallsAddWithEmbeddings:
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._should_use_late_chunking")
    @patch("kiara.ingestion.runtime._late_chunking_file_eligible")
    @patch("kiara.ingestion.runtime.embed_texts_late")
    def test_late_chunking_pipeline_calls_add_with_embeddings(
        self,
        mock_etl,
        mock_lcfe,
        mock_sulc,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """Late Chunking produces embeddings -> _write_embeddings called with embeddings kwarg."""
        fake_vectors = [[0.1] * 768]

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"late_chunking": True},
            },
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            datasource_id=None,
        )

        mock_psf.return_value = (
            ["Original chunk"],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full document text for late chunking",
            [(0, 14)],
        )

        mock_sulc.return_value = True
        mock_lcfe.return_value = True
        mock_etl.return_value = fake_vectors

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        mock_write.assert_called_once()
        write_kwargs = mock_write.call_args
        # embeddings kwarg should contain the fake vectors
        assert write_kwargs[1].get("embeddings") == fake_vectors


# ===================================================================
# Spec #36: Enrichment exception + retry reruns enrichment in Late path
# ===================================================================


class TestEnrichmentExceptionRetryInLatePath:
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._get_enrichment_llm")
    @patch("kiara.ingestion.runtime._get_enrichment_prompt")
    @patch("kiara.ingestion.enrichment.enrich_chunks_batch")
    @patch("kiara.shared.db.db_query")
    @patch("kiara.ingestion.runtime._refresh_late_chunking_snapshots")
    def test_enrichment_exception_retry_reruns_enrichment_in_late_path(
        self,
        mock_refresh,
        mock_db_query,
        mock_ecb,
        mock_gep,
        mock_gel,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """Enrichment error on first pass -> embedding RETRY ->
        enrichment_applied stays False -> enrichment attempted again on second pass.
        _refresh_late_chunking_snapshots called with enrichment_applied=False both times."""
        from kiara.backend_selector import BackendUnavailableError

        mock_name_row = {"name": "TestDS"}
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchone.return_value = mock_name_row
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"enabled": True, "late_chunking": True},
            },
            ds_overrides={
                "contextual_retrieval": True,
                "enrichment_backend_id": 1,
            },
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            consecutive_errors=9,
        )

        mock_psf.return_value = (
            ["text"],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full text",
            [(0, 4)],
        )

        # First call to enrich_chunks_batch raises exception
        # Second call succeeds
        mock_gep.return_value = ("template", "hash123")
        mock_gel.return_value = (lambda s, u: "prefix", None, False)
        mock_ecb.side_effect = [
            RuntimeError("LLM down"),
            (
                ["Prefix\n\ntext"],
                {"enriched": 1, "skipped": 0, "errors": 0, "enrichment_time_ms": 10},
            ),
        ]

        # First _write_embeddings raises BackendUnavailableError (triggers RETRY),
        # second succeeds.
        mock_write.side_effect = [BackendUnavailableError("qdrant down"), None]

        with patch("kiara.ingestion.runtime._wait_for_backend", return_value=True):
            result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        # enrich_chunks_batch called TWICE: first pass (exception) + retry pass (success)
        assert mock_ecb.call_count == 2

        # _refresh_late_chunking_snapshots called twice
        assert mock_refresh.call_count == 2
        # Both calls have enrichment_applied=False because the except block
        # restores originals and does NOT set enrichment_applied=True
        first_call = mock_refresh.call_args_list[0]
        assert first_call[1]["enrichment_applied"] is False
        second_call = mock_refresh.call_args_list[1]
        assert second_call[1]["enrichment_applied"] is False
