"""Tests D1-D19: Enrichment Integrity + Admin-Funktionen (Sprint 16.3-5).

Testet check_enrichment_prompt_consistency(), run_full_check(),
ki_enrichment_test, ki_templates_rollback, Update-History-Trigger
und Admin-Registry-Eintraege.
"""

import asyncio
import hashlib
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}

# Konstante fuer Tests: sha256 eines bekannten Prompt-Contents
_PROMPT_CONTENT_A = "Fasse den folgenden Text zusammen: {chunk_text}"
_PROMPT_HASH_A = hashlib.sha256(_PROMPT_CONTENT_A.encode("utf-8")).hexdigest()

_PROMPT_CONTENT_B = "Beschreibe den Kontext: {chunk_text}"
_PROMPT_HASH_B = hashlib.sha256(_PROMPT_CONTENT_B.encode("utf-8")).hexdigest()


def _mock_db_query_cm(call_responses, idx):
    """Erstellt einen db_query()-Context-Manager mit sequenziellen Antworten."""
    mock_conn = MagicMock()

    def _execute(*args, **kwargs):
        result = MagicMock()
        if idx[0] < len(call_responses):
            resp = call_responses[idx[0]]
            result.fetchone.return_value = resp.get("fetchone")
            result.fetchall.return_value = resp.get("fetchall", [])
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        idx[0] += 1
        return result

    mock_conn.execute = MagicMock(side_effect=_execute)
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_conn)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


def _mock_tx_context(row_data=None, hist_row=None):
    """Erstellt einen Mock fuer db_transaction() mit SELECT FOR UPDATE + History-Lookup."""
    mock_tx = MagicMock()

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        result = MagicMock()
        if call_count[0] == 1:
            result.fetchone.return_value = row_data
        elif call_count[0] == 2 and hist_row is not None:
            result.fetchone.return_value = hist_row
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    mock_tx.execute = MagicMock(side_effect=side_effect)
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_tx)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm, mock_tx


# ===========================================================================
# Integrity Check: check_enrichment_prompt_consistency (D1-D6, D17)
# ===========================================================================


class TestEnrichmentPromptConsistency:
    """Tests D1-D6: check_enrichment_prompt_consistency mit Mock DB + Qdrant."""

    def _build_db_responses(self, ds_rows, prompt_rows, history_hashes=None):
        """Baut eine Liste von db_query-Antworten fuer den Check.

        Reihenfolge der Queries im Code (nach N+1-Fix, Batch-Load):
        1. SELECT ... FROM datasources WHERE is_active=True AND ds_contextual_retrieval=1
        2. SELECT id, content FROM prompt_templates WHERE id IN (...)  (Batch)
        3. SELECT content FROM prompt_templates WHERE name = 'enrichment_default'
        4. SELECT DISTINCT content_hash FROM prompt_template_history
        """
        responses = []
        # Query 1: DS-Liste
        responses.append({"fetchall": ds_rows})

        # Query 2: Batch-Prompt-Load — alle Prompt-IDs auf einmal
        batch_prompts = []
        for i, pr in enumerate(prompt_rows):
            prompt_id = ds_rows[i]["ds_enrichment_prompt_id"] if i < len(ds_rows) else None
            if prompt_id is not None and pr is not None:
                batch_prompts.append({"id": prompt_id, "content": pr["content"]})
        responses.append({"fetchall": batch_prompts})

        # Query 3: Default-Prompt (fetchone, None wenn nicht benoetigt)
        responses.append({"fetchone": None})

        # Query 4: History-Hashes
        history_rows = [{"content_hash": h} for h in (history_hashes or [])]
        responses.append({"fetchall": history_rows})

        return responses

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.shared.db.db_query")
    def test_d1_all_consistent(self, mock_db_query, mock_vs_cls):
        """D1: Alle Chunks konsistent → status='ok', inconsistent=0."""
        ds_rows = [{"id": 1, "name": "TestDS", "ds_enrichment_prompt_id": 10}]
        prompt_row = {"content": _PROMPT_CONTENT_A}
        db_responses = self._build_db_responses(ds_rows, [prompt_row])
        idx = [0]
        mock_db_query.side_effect = lambda: _mock_db_query_cm(db_responses, idx)

        # Qdrant: alle Chunks mit korrektem Hash
        chunks = [
            (
                "c1",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_A,
                },
            ),
            (
                "c2",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_A,
                },
            ),
        ]
        mock_vs_cls.list_collections.return_value = [
            {"name": "kb_test", "model": "test", "chunks": 2},
        ]
        mock_store = MagicMock()
        mock_store.scroll_all.return_value = iter([chunks])
        mock_vs_cls.return_value = mock_store

        from kiara.integrity.checks import check_enrichment_prompt_consistency

        result = check_enrichment_prompt_consistency({})

        assert result["status"] == "ok"
        assert result["inconsistent"] == 0
        assert result["consistent"] == 2
        assert result["total_datasources_checked"] == 1

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.shared.db.db_query")
    def test_d2_mixed_hashes(self, mock_db_query, mock_vs_cls):
        """D2: Mixed hashes → status='warning', found_hashes shows distribution."""
        ds_rows = [{"id": 1, "name": "MixedDS", "ds_enrichment_prompt_id": 10}]
        prompt_row = {"content": _PROMPT_CONTENT_A}
        db_responses = self._build_db_responses(
            ds_rows,
            [prompt_row],
            history_hashes=[_PROMPT_HASH_B],
        )
        idx = [0]
        mock_db_query.side_effect = lambda: _mock_db_query_cm(db_responses, idx)

        # Chunks: 1 mit aktuellem Hash, 1 mit altem Hash
        chunks = [
            (
                "c1",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_A,
                },
            ),
            (
                "c2",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_B,
                },
            ),
        ]
        mock_vs_cls.list_collections.return_value = [
            {"name": "kb_test", "model": "test", "chunks": 2},
        ]
        mock_store = MagicMock()
        mock_store.scroll_all.return_value = iter([chunks])
        mock_vs_cls.return_value = mock_store

        from kiara.integrity.checks import check_enrichment_prompt_consistency

        result = check_enrichment_prompt_consistency({})

        assert result["status"] == "warning"
        assert result["inconsistent"] == 1
        assert result["consistent"] == 1
        # found_hashes enthaelt beide Hashes
        detail = result["details"][0]
        found_hash_values = {h["hash"] for h in detail["found_hashes"]}
        assert _PROMPT_HASH_A[:12] in found_hash_values
        assert _PROMPT_HASH_B[:12] in found_hash_values

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.shared.db.db_query")
    def test_d3_ds_without_enrichment_skipped(self, mock_db_query, mock_vs_cls):
        """D3: DS ohne Enrichment (ds_contextual_retrieval=0) → nicht geprueft."""
        # Wenn keine DS mit ds_contextual_retrieval=1 existiert,
        # gibt die Funktion sofort ein leeres Ergebnis zurueck
        db_responses = [{"fetchall": []}]  # Leere DS-Liste
        idx = [0]
        mock_db_query.side_effect = lambda: _mock_db_query_cm(db_responses, idx)

        from kiara.integrity.checks import check_enrichment_prompt_consistency

        result = check_enrichment_prompt_consistency({})

        assert result["status"] == "ok"
        assert result["total_datasources_checked"] == 0
        assert result["total_chunks_checked"] == 0
        # VectorStore sollte nicht aufgerufen worden sein
        mock_vs_cls.list_collections.assert_not_called()

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.shared.db.db_query")
    def test_d4_chunks_without_prefix_counted(self, mock_db_query, mock_vs_cls):
        """D4: Chunks ohne contextual_prefix → not_enriched gezaehlt."""
        ds_rows = [{"id": 1, "name": "TestDS", "ds_enrichment_prompt_id": 10}]
        prompt_row = {"content": _PROMPT_CONTENT_A}
        db_responses = self._build_db_responses(ds_rows, [prompt_row])
        idx = [0]
        mock_db_query.side_effect = lambda: _mock_db_query_cm(db_responses, idx)

        # Chunks: 1 mit Prefix, 2 ohne Prefix
        chunks = [
            (
                "c1",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_A,
                },
            ),
            ("c2", {"ds_ids": [1], "contextual_prefix": False, "enrichment_prompt_hash": None}),
            ("c3", {"ds_ids": [1]}),  # kein contextual_prefix key
        ]
        mock_vs_cls.list_collections.return_value = [
            {"name": "kb_test", "model": "test", "chunks": 3},
        ]
        mock_store = MagicMock()
        mock_store.scroll_all.return_value = iter([chunks])
        mock_vs_cls.return_value = mock_store

        from kiara.integrity.checks import check_enrichment_prompt_consistency

        result = check_enrichment_prompt_consistency({})

        assert result["not_enriched"] == 2
        assert result["consistent"] == 1

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.shared.db.db_query")
    def test_d5_hash_resolution_via_history(self, mock_db_query, mock_vs_cls):
        """D5: Hash-Aufloesung via History → found_hashes enthalten Status current/historical."""
        ds_rows = [{"id": 1, "name": "HistDS", "ds_enrichment_prompt_id": 10}]
        prompt_row = {"content": _PROMPT_CONTENT_A}
        db_responses = self._build_db_responses(
            ds_rows,
            [prompt_row],
            history_hashes=[_PROMPT_HASH_B],  # Hash B ist historisch bekannt
        )
        idx = [0]
        mock_db_query.side_effect = lambda: _mock_db_query_cm(db_responses, idx)

        # Chunks mit aktuellem und historischem Hash
        chunks = [
            (
                "c1",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_A,
                },
            ),
            (
                "c2",
                {
                    "ds_ids": [1],
                    "contextual_prefix": True,
                    "enrichment_prompt_hash": _PROMPT_HASH_B,
                },
            ),
        ]
        mock_vs_cls.list_collections.return_value = [
            {"name": "kb_test", "model": "test", "chunks": 2},
        ]
        mock_store = MagicMock()
        mock_store.scroll_all.return_value = iter([chunks])
        mock_vs_cls.return_value = mock_store

        from kiara.integrity.checks import check_enrichment_prompt_consistency

        result = check_enrichment_prompt_consistency({})

        detail = result["details"][0]
        found_map = {h["hash"]: h["status"] for h in detail["found_hashes"]}
        # Aktueller Hash hat Status "current"
        assert found_map[_PROMPT_HASH_A[:12]] == "current"
        # Historischer Hash hat Status "historical"
        assert found_map[_PROMPT_HASH_B[:12]] == "historical"

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.shared.db.db_query")
    def test_d6_no_ds_with_enrichment(self, mock_db_query, mock_vs_cls):
        """D6: Keine DS mit Enrichment → status='ok', total_datasources_checked=0."""
        db_responses = [{"fetchall": []}]
        idx = [0]
        mock_db_query.side_effect = lambda: _mock_db_query_cm(db_responses, idx)

        from kiara.integrity.checks import check_enrichment_prompt_consistency

        result = check_enrichment_prompt_consistency({})

        assert result["status"] == "ok"
        assert result["total_datasources_checked"] == 0
        assert result["consistent"] == 0
        assert result["inconsistent"] == 0
        assert result["not_enriched"] == 0
        assert result["details"] == []


class TestRunFullCheck:
    """Test D17: run_full_check enthaelt enrichment_prompt_consistency."""

    def test_d17_enrichment_prompt_consistency_is_7th_check(self):
        """D17: run_full_check() enthaelt enrichment_prompt_consistency + stale_colbert (8 Checks)."""
        # Alle 8 Checks mocken damit run_full_check durchlaeuft
        with (
            patch("kiara.integrity.checks.check_layer_counts", return_value={"status": "ok"}),
            patch(
                "kiara.integrity.checks.check_ds_coverage",
                return_value={
                    "status": "ok",
                    "es_without_ds_ids": 0,
                    "es_total": 0,
                    "datasources": [],
                },
            ),
            patch(
                "kiara.integrity.checks.check_es_duplicates",
                return_value={"status": "ok", "suspected_duplicates": 0},
            ),
            patch(
                "kiara.integrity.checks.check_orphan_chunks",
                return_value={"status": "ok", "orphan_pct": 0},
            ),
            patch(
                "kiara.integrity.checks.check_overlapping_datasources",
                return_value={"status": "ok", "overlaps": []},
            ),
            patch(
                "kiara.integrity.checks.check_qdrant_fragmentation",
                return_value={"status": "ok", "fragmented_count": 0, "collections": []},
            ),
            patch(
                "kiara.integrity.checks.check_enrichment_prompt_consistency",
                return_value={"status": "ok"},
            ) as mock_epc,
            patch(
                "kiara.integrity.checks.check_stale_colbert_collections",
                return_value={"status": "ok"},
            ) as mock_stale,
        ):
            from kiara.integrity.checks import run_full_check

            result = run_full_check({})

            # Muss aufgerufen worden sein
            mock_epc.assert_called_once_with({})
            mock_stale.assert_called_once_with({})
            # 8 Checks im Ergebnis
            assert len(result["checks"]) == 8
            assert "enrichment_prompt_consistency" in result["checks"]
            assert "stale_colbert" in result["checks"]
            assert result["checks"]["enrichment_prompt_consistency"]["status"] == "ok"
            assert result["checks"]["stale_colbert"]["status"] == "ok"
            assert result["overall_status"] == "ok"

    def test_run_full_check_summary_includes_colbert_warnings(self):
        """run_full_check() Summary erwaehnt colbert_warnings explizit."""
        ds_coverage_result = {
            "status": "warning",
            "es_without_ds_ids": 0,
            "es_total": 0,
            "datasources": [],
            "ds_warnings": 0,
            "colbert_warnings": [{"id": 1, "name": "TestDS"}],
            "colbert_warning_count": 1,
        }

        with (
            patch("kiara.integrity.checks.check_layer_counts", return_value={"status": "ok"}),
            patch("kiara.integrity.checks.check_ds_coverage", return_value=ds_coverage_result),
            patch(
                "kiara.integrity.checks.check_es_duplicates",
                return_value={"status": "ok", "suspected_duplicates": 0},
            ),
            patch(
                "kiara.integrity.checks.check_orphan_chunks",
                return_value={"status": "ok", "orphan_pct": 0},
            ),
            patch(
                "kiara.integrity.checks.check_overlapping_datasources",
                return_value={"status": "ok", "overlaps": []},
            ),
            patch(
                "kiara.integrity.checks.check_qdrant_fragmentation",
                return_value={"status": "ok", "fragmented_count": 0, "collections": []},
            ),
            patch(
                "kiara.integrity.checks.check_enrichment_prompt_consistency",
                return_value={"status": "ok"},
            ),
            patch(
                "kiara.integrity.checks.check_stale_colbert_collections",
                return_value={"status": "ok"},
            ),
        ):
            from kiara.integrity.checks import run_full_check

            result = run_full_check({})

            assert result["overall_status"] == "warning"
            assert "ColBERT aktiviert aber ohne Chunks" in result["summary"]
            assert "TestDS" in result["summary"]


# ===========================================================================
# Prompt Test Admin Function: ki_enrichment_test (D7-D11)
# ===========================================================================


class TestKiEnrichmentTest:
    """Tests D7-D11: ki.enrichment.test Admin-Funktion."""

    @patch("kiara.ingestion.enrichment.enrich_chunk")
    @patch("kiara.ingestion.enrichment._render_prompt")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.config.load_config", return_value={})
    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_d7_valid_template_generates_prefix(
        self,
        mock_repo_cls,
        mock_config,
        mock_create_backend,
        mock_render,
        mock_enrich,
    ):
        """D7: Gueltige template + text + backend → prefix erzeugt, duration_ms > 0."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {
            "id": 5,
            "name": "test-enrich",
            "chain_type": "enrichment",
            "content": "Fasse zusammen: {chunk_text}",
        }
        mock_repo_cls.return_value = mock_repo

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"
        mock_backend.model = "llama3"
        mock_backend.generate = MagicMock(return_value="Generated prefix")
        mock_create_backend.return_value = mock_backend

        mock_render.return_value = "Fasse zusammen: Beispieltext"

        sample = "Beispieltext"
        enriched = "Kontext des Dokuments\n\n" + sample
        mock_enrich.return_value = enriched

        from kiara.admin.functions.ki import ki_enrichment_test

        result = _run(
            ki_enrichment_test(
                user=MOCK_USER,
                template_id=5,
                sample_text=sample,
            )
        )

        assert result.success is True
        assert result.data["prefix"] == "Kontext des Dokuments"
        assert result.data["duration_ms"] >= 0
        assert result.data["backend_used"] == "ollama/llama3"

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_d8_invalid_template_id(self, mock_repo_cls):
        """D8: Ungueltiges template_id → AdminResult.fail mit 'nicht gefunden'."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = None
        mock_repo_cls.return_value = mock_repo

        from kiara.admin.functions.ki import ki_enrichment_test

        result = _run(
            ki_enrichment_test(
                user=MOCK_USER,
                template_id=999,
                sample_text="test",
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.backend_selector.get_backend_by_id")
    @patch("kiara.config.load_config", return_value={})
    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_d9_invalid_backend_wrong_category(
        self,
        mock_repo_cls,
        mock_config,
        mock_get_backend,
    ):
        """D9: Backend mit falscher Kategorie (embedding) → error 400."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {
            "id": 5,
            "name": "test-enrich",
            "chain_type": "enrichment",
            "content": "{chunk_text}",
        }
        mock_repo_cls.return_value = mock_repo

        # Backend existiert aber hat falsche Kategorie
        mock_get_backend.return_value = {
            "id": 10,
            "category": "embedding",
            "provider": "ollama",
        }

        from kiara.admin.functions.ki import ki_enrichment_test

        result = _run(
            ki_enrichment_test(
                user=MOCK_USER,
                template_id=5,
                sample_text="test",
                backend_id=10,
            )
        )

        assert result.success is False
        assert "embedding" in result.message or "llm/cloud" in result.message

    @patch("kiara.backend_selector.get_backend_by_id")
    @patch("kiara.config.load_config", return_value={})
    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_d9b_invalid_backend_not_found(
        self,
        mock_repo_cls,
        mock_config,
        mock_get_backend,
    ):
        """D9b: Backend-ID nicht gefunden → error."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {
            "id": 5,
            "name": "test-enrich",
            "chain_type": "enrichment",
            "content": "{chunk_text}",
        }
        mock_repo_cls.return_value = mock_repo
        mock_get_backend.return_value = None

        from kiara.admin.functions.ki import ki_enrichment_test

        result = _run(
            ki_enrichment_test(
                user=MOCK_USER,
                template_id=5,
                sample_text="test",
                backend_id=999,
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message or "inaktiv" in result.message

    @patch("kiara.ingestion.enrichment.enrich_chunk")
    @patch("kiara.ingestion.enrichment._render_prompt")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.config.load_config", return_value={})
    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_d10_llm_unreachable(
        self,
        mock_repo_cls,
        mock_config,
        mock_create_backend,
        mock_render,
        mock_enrich,
    ):
        """D10: LLM nicht erreichbar → Fehlermeldung, kein Crash."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {
            "id": 5,
            "name": "test-enrich",
            "chain_type": "enrichment",
            "content": "{chunk_text}",
        }
        mock_repo_cls.return_value = mock_repo

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"
        mock_backend.model = "llama3"
        mock_create_backend.return_value = mock_backend

        mock_render.return_value = "Rendered prompt"
        mock_enrich.side_effect = ConnectionError("Connection refused")

        from kiara.admin.functions.ki import ki_enrichment_test

        result = _run(
            ki_enrichment_test(
                user=MOCK_USER,
                template_id=5,
                sample_text="test",
            )
        )

        assert result.success is False
        assert "fehlgeschlagen" in result.message

    @patch("kiara.ingestion.enrichment.enrich_chunk")
    @patch("kiara.ingestion.enrichment._render_prompt")
    @patch("kiara.generation.backends.factory.create_backend")
    @patch("kiara.config.load_config", return_value={})
    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_d11_prompt_used_contains_rendered(
        self,
        mock_repo_cls,
        mock_config,
        mock_create_backend,
        mock_render,
        mock_enrich,
    ):
        """D11: prompt_used enthaelt gerenderten Prompt."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {
            "id": 5,
            "name": "test-enrich",
            "chain_type": "enrichment",
            "content": "Fasse zusammen: {chunk_text}",
        }
        mock_repo_cls.return_value = mock_repo

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"
        mock_backend.model = "llama3"
        mock_create_backend.return_value = mock_backend

        rendered = "Fasse zusammen: Beispieltext zum Testen"
        mock_render.return_value = rendered

        sample = "Beispieltext zum Testen"
        mock_enrich.return_value = "Prefix\n\n" + sample

        from kiara.admin.functions.ki import ki_enrichment_test

        result = _run(
            ki_enrichment_test(
                user=MOCK_USER,
                template_id=5,
                sample_text=sample,
            )
        )

        assert result.success is True
        assert result.data["prompt_used"] == rendered


# ===========================================================================
# Rollback Admin Function: ki_templates_rollback (D12-D16)
# ===========================================================================


class TestKiTemplatesRollback:
    """Tests D12-D16: ki.templates.rollback Admin-Funktion."""

    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.shared.db.db_transaction")
    def test_d12_rollback_restores_content(
        self,
        mock_tx_factory,
        mock_save_history,
        mock_update,
    ):
        """D12: Rollback stellt content + cloud_system_prompt wieder her, History erweitert."""
        template_row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "enrichment",
            "is_system": False,
            "content": "aktueller Content mit {chunk_text}",
            "cloud_system_prompt": "aktueller CSP",
        }
        history_row = {
            "id": 42,
            "template_id": 3,
            "content": "alter Content mit {chunk_text}",
            "cloud_system_prompt": "alter CSP",
        }
        mock_cm, mock_tx = _mock_tx_context(template_row, history_row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_rollback

        result = _run(
            ki_templates_rollback(
                user=MOCK_USER,
                template_id=3,
                history_id=42,
            )
        )

        assert result.success is True
        assert "zurueckgesetzt" in result.message

        # save_history muss alten Stand archiviert haben
        mock_save_history.assert_called_once()
        save_args = mock_save_history.call_args
        assert save_args[0][0] == 3  # template_id
        assert save_args[0][1] == "aktueller Content mit {chunk_text}"
        assert save_args[1]["cloud_system_prompt"] == "aktueller CSP"
        assert save_args[1]["conn"] is mock_tx

        # update muss mit History-Daten aufgerufen worden sein
        mock_update.assert_called_once()
        update_kwargs = mock_update.call_args[1]
        assert update_kwargs["content"] == "alter Content mit {chunk_text}"
        assert update_kwargs["cloud_system_prompt"] == "alter CSP"

    @patch("kiara.shared.db.db_transaction")
    def test_d13_invalid_history_id(self, mock_tx_factory):
        """D13: Ungueltiges history_id → AdminResult.fail mit 'nicht gefunden'."""
        template_row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "enrichment",
            "is_system": False,
            "content": "content",
            "cloud_system_prompt": "",
        }
        mock_cm, mock_tx = _mock_tx_context(template_row, hist_row=None)
        # Override: make the second execute return None explicitly
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.fetchone.return_value = template_row
            else:
                result.fetchone.return_value = None
            return result

        mock_tx.execute = MagicMock(side_effect=side_effect)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_rollback

        result = _run(
            ki_templates_rollback(
                user=MOCK_USER,
                template_id=3,
                history_id=999,
            )
        )

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.shared.db.db_transaction")
    def test_d14_history_belongs_to_different_template(self, mock_tx_factory):
        """D14: history_id gehoert zu anderem Template → error 400."""
        template_row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "enrichment",
            "is_system": False,
            "content": "content",
            "cloud_system_prompt": "",
        }
        # History gehoert zu Template 99, nicht 3
        history_row = {
            "id": 42,
            "template_id": 99,
            "content": "some content",
            "cloud_system_prompt": "",
        }
        mock_cm, mock_tx = _mock_tx_context(template_row, history_row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_rollback

        result = _run(
            ki_templates_rollback(
                user=MOCK_USER,
                template_id=3,
                history_id=42,
            )
        )

        assert result.success is False
        assert "gehoert nicht" in result.message

    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.shared.db.db_transaction")
    def test_d15_rollback_to_current_content(
        self,
        mock_tx_factory,
        mock_save_history,
        mock_update,
    ):
        """D15: Rollback auf identischen Inhalt → kein No-Op, History wird trotzdem geschrieben."""
        current_content = "gleicher Content mit {chunk_text}"
        template_row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "enrichment",
            "is_system": False,
            "content": current_content,
            "cloud_system_prompt": "sys",
        }
        # History mit gleichem Content
        history_row = {
            "id": 42,
            "template_id": 3,
            "content": current_content,
            "cloud_system_prompt": "sys",
        }
        mock_cm, mock_tx = _mock_tx_context(template_row, history_row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_rollback

        result = _run(
            ki_templates_rollback(
                user=MOCK_USER,
                template_id=3,
                history_id=42,
            )
        )

        assert result.success is True
        # History wird dennoch geschrieben (rollback wird protokolliert)
        mock_save_history.assert_called_once()
        # Update wird auch aufgerufen
        mock_update.assert_called_once()

    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.shared.db.db_transaction")
    def test_d16_system_template_rollback(
        self,
        mock_tx_factory,
        mock_save_history,
        mock_update,
    ):
        """D16: System-Template → Rollback funktioniert (nur versionierte Felder)."""
        template_row = {
            "id": 1,
            "name": "system_default",
            "label": "Standard",
            "chain_type": "standard",
            "is_system": True,
            "content": "neuer System-Content {context} {question}",
            "cloud_system_prompt": "",
        }
        history_row = {
            "id": 10,
            "template_id": 1,
            "content": "alter System-Content {context} {question}",
            "cloud_system_prompt": "",
        }
        mock_cm, mock_tx = _mock_tx_context(template_row, history_row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_rollback

        result = _run(
            ki_templates_rollback(
                user=MOCK_USER,
                template_id=1,
                history_id=10,
            )
        )

        assert result.success is True
        mock_update.assert_called_once()
        update_kwargs = mock_update.call_args[1]
        assert update_kwargs["content"] == "alter System-Content {context} {question}"


# ===========================================================================
# Template Update History Trigger (D18)
# ===========================================================================


class TestUpdateHistoryTrigger:
    """Test D18: cloud_system_prompt-Aenderung erzeugt History-Eintrag."""

    @patch("kiara.repositories.prompt.PromptTemplateRepository.save_history")
    @patch("kiara.repositories.prompt.PromptTemplateRepository.update")
    @patch("kiara.shared.db.db_transaction")
    def test_d18_only_csp_changed_creates_history(
        self,
        mock_tx_factory,
        mock_update,
        mock_save_history,
    ):
        """D18: Nur cloud_system_prompt geaendert → History-Eintrag wird erstellt."""
        row = {
            "id": 3,
            "name": "test",
            "label": "Test",
            "chain_type": "unchained",
            "is_system": False,
            "content": "unveraenderter Content",
            "cloud_system_prompt": "alter System-Prompt",
            "is_cloud": False,
            "show_in_chat": 0,
        }
        mock_cm, mock_tx = _mock_tx_context(row)
        mock_tx_factory.return_value = mock_cm

        from kiara.admin.functions.ki import ki_templates_update

        result = _run(
            ki_templates_update(
                user=MOCK_USER,
                template_id=3,
                cloud_system_prompt="neuer System-Prompt",
            )
        )

        assert result.success is True
        # save_history MUSS aufgerufen worden sein, obwohl content gleich blieb
        mock_save_history.assert_called_once()
        save_args = mock_save_history.call_args
        assert save_args[0][1] == "unveraenderter Content"  # alter Content gesichert
        assert save_args[1]["cloud_system_prompt"] == "alter System-Prompt"


# ===========================================================================
# Admin Registry (D19)
# ===========================================================================


class TestAdminRegistry:
    """Test D19: ki.enrichment.test + ki.templates.rollback in Registry."""

    def test_d19_enrichment_test_in_registry(self):
        """D19a: ki.enrichment.test in Registry mit POST + korrektem Pfad."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.enrichment.test")
        assert func is not None, "ki.enrichment.test nicht in Registry"
        assert func.http_method == "POST"
        assert func.http_path == "/api/admin/ki/enrichment/test"

    def test_d19_templates_rollback_in_registry(self):
        """D19b: ki.templates.rollback in Registry mit POST + korrektem Pfad."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.templates.rollback")
        assert func is not None, "ki.templates.rollback nicht in Registry"
        assert func.http_method == "POST"
        assert (
            func.http_path == "/api/admin/ki/prompt-templates/{template_id}/rollback/{history_id}"
        )
        assert func.requires_confirmation is True
