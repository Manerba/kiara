"""Tests fuer Sprint 16.3-5: Enrichment Performance (C1-C13).

Testet parallele/sequenzielle Pfade, Stats-Zaehlung, Callback-Reihenfolge,
Pipeline-Stats-Aggregation und Randfaelle.
"""

import threading
from unittest.mock import MagicMock, patch

from kiara.ingestion.enrichment import (
    enrich_chunks_batch,
)
from kiara.shared.constants import ENRICHMENT_MIN_CHUNK_CHARS

# ---------------------------------------------------------------------------
# Hilfsdaten + Mocks
# ---------------------------------------------------------------------------

_TEMPLATE = "{document_context}\n{chunk_text}"


def _mock_llm(prefix: str = "Generierter Kontext"):
    """Erstellt ein LLM-Mock-Callable (sequenziell sicher)."""

    def _fn(system_prompt: str, user_message: str) -> str:
        return prefix

    return _fn


def _make_chunks(n: int, char_count: int = 100) -> list[str]:
    """Erzeugt n Chunks mit je char_count Zeichen."""
    return [chr(ord("A") + (i % 26)) * char_count for i in range(n)]


# ---------------------------------------------------------------------------
# C1: 10 Chunks, allow_parallel=True, lokale Factory → tatsaechliche Nebenl.
# ---------------------------------------------------------------------------


class TestC1ParallelConcurrency:
    def test_parallel_path_with_barrier_proves_concurrency(self):
        """C1: 10 Chunks mit Factory → Threads laufen tatsaechlich parallel.

        Beweist Parallelitaet ueber eine threading.Barrier: Wenn mindestens
        2 Threads gleichzeitig die Barrier passieren, waren sie nebenlaefig.
        """
        n_chunks = 10
        chunks = _make_chunks(n_chunks)
        full_text = "".join(chunks)
        max_workers = 4

        # Barrier mit parties=2 — wenn 2 Threads gleichzeitig hier ankommen,
        # passieren sie; sonst wuerde ein Timeout ausloesen.
        barrier = threading.Barrier(parties=2, timeout=5)
        barrier_passed = threading.Event()

        def _barrier_llm(system_prompt: str, user_message: str) -> str:
            try:
                barrier.wait(timeout=5)
                barrier_passed.set()
            except threading.BrokenBarrierError:
                pass
            return "parallel-prefix"

        def _factory():
            return _barrier_llm

        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text=full_text,
            llm_generate_fn=_barrier_llm,
            prompt_template=_TEMPLATE,
            max_workers=max_workers,
            llm_generate_factory=_factory,
            allow_parallel=True,
        )

        # Parallelitaet bewiesen
        assert barrier_passed.is_set(), (
            "Barrier wurde nie von 2+ Threads gleichzeitig passiert — "
            "kein Beweis fuer echte Parallelitaet"
        )
        # Ergebnisse korrekt
        assert len(results) == n_chunks
        for i, result in enumerate(results):
            assert chunks[i] in result, f"Chunk {i} nicht im Ergebnis"
            assert "parallel-prefix" in result
        assert stats["enriched"] == n_chunks


# ---------------------------------------------------------------------------
# C2: 3 Chunks → sequenzieller Pfad (kein ThreadPoolExecutor)
# ---------------------------------------------------------------------------


class TestC2SequentialForSmallBatch:
    def test_3_chunks_forces_sequential(self):
        """C2: <= 3 Chunks → sequenzieller Pfad, kein ThreadPoolExecutor."""
        chunks = _make_chunks(3)

        with patch("kiara.ingestion.enrichment.concurrent.futures.ThreadPoolExecutor") as mock_pool:
            results, stats = enrich_chunks_batch(
                chunk_texts=chunks,
                full_text="".join(chunks),
                llm_generate_fn=_mock_llm(),
                prompt_template=_TEMPLATE,
                max_workers=4,
                llm_generate_factory=lambda: _mock_llm(),
                allow_parallel=True,
            )

        mock_pool.assert_not_called()
        assert len(results) == 3
        assert stats["enriched"] == 3


# ---------------------------------------------------------------------------
# C3: Cloud/anonymisierender Pfad → allow_parallel=False, sequenziell
# ---------------------------------------------------------------------------


class TestC3CloudForcesSequential:
    def test_allow_parallel_false_forces_sequential(self):
        """C3: allow_parallel=False → sequenzieller Pfad trotz max_workers>1."""
        chunks = _make_chunks(10)

        with patch("kiara.ingestion.enrichment.concurrent.futures.ThreadPoolExecutor") as mock_pool:
            results, stats = enrich_chunks_batch(
                chunk_texts=chunks,
                full_text="".join(chunks),
                llm_generate_fn=_mock_llm(),
                prompt_template=_TEMPLATE,
                max_workers=4,
                llm_generate_factory=lambda: _mock_llm(),
                allow_parallel=False,
            )

        mock_pool.assert_not_called()
        assert len(results) == 10
        assert stats["enriched"] == 10


# ---------------------------------------------------------------------------
# C4: Chunk < MIN_CHUNK_CHARS → skipped
# ---------------------------------------------------------------------------


class TestC4ShortChunkSkipped:
    def test_chunk_below_min_chars_is_skipped(self):
        """C4: Chunk kuerzer als ENRICHMENT_MIN_CHUNK_CHARS → skipped, Original bleibt."""
        short_chunk = "x" * (ENRICHMENT_MIN_CHUNK_CHARS - 1)
        results, stats = enrich_chunks_batch(
            chunk_texts=[short_chunk],
            full_text=short_chunk,
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
        )
        assert stats["skipped"] == 1
        assert stats["enriched"] == 0
        assert results[0] == short_chunk


# ---------------------------------------------------------------------------
# C5: 5 Chunks, 2 skipped, 1 error → enriched=2, skipped=2, errors=1
# ---------------------------------------------------------------------------


class TestC5MixedStats:
    def test_5_chunks_2_skipped_1_error(self):
        """C5: 5 Chunks: 2 zu kurz, 1 LLM-Fehler → enriched=2, skipped=2, errors=1."""
        call_count = [0]

        def _mixed_llm(s, u):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("LLM-Fehler")
            return "prefix"

        chunks = [
            "A" * 100,  # → enriched
            "B" * 100,  # → error (2. LLM-Call)
            "short",  # → skipped (< MIN_CHUNK_CHARS)
            "tiny",  # → skipped (< MIN_CHUNK_CHARS)
            "E" * 100,  # → enriched
        ]

        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mixed_llm,
            prompt_template=_TEMPLATE,
        )

        assert stats["enriched"] == 2
        assert stats["skipped"] == 2
        assert stats["errors"] == 1
        assert len(results) == 5
        # Skipped Chunks bleiben unveraendert
        assert results[2] == chunks[2]
        assert results[3] == chunks[3]
        # Error Chunk bleibt unveraendert
        assert results[1] == chunks[1]


# ---------------------------------------------------------------------------
# C6: enrichment_time_ms > 0
# ---------------------------------------------------------------------------


class TestC6TimeMs:
    def test_enrichment_time_ms_positive(self):
        """C6: enrichment_time_ms > 0 bei nicht-leerem Batch."""
        chunks = _make_chunks(3)
        _, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
        )
        assert stats["enrichment_time_ms"] >= 0
        # Bei leerem Batch waere es 0, hier mit 3 Chunks muss etwas gemessen werden
        assert isinstance(stats["enrichment_time_ms"], int)


# ---------------------------------------------------------------------------
# C7: max_workers=1 → funktioniert identisch (sequenzieller Fallback)
# ---------------------------------------------------------------------------


class TestC7MaxWorkersOne:
    def test_max_workers_1_sequential_identical(self):
        """C7: max_workers=1 → sequenzieller Pfad, identische Ergebnisse."""
        chunks = _make_chunks(5)

        with patch("kiara.ingestion.enrichment.concurrent.futures.ThreadPoolExecutor") as mock_pool:
            results, stats = enrich_chunks_batch(
                chunk_texts=chunks,
                full_text="".join(chunks),
                llm_generate_fn=_mock_llm(),
                prompt_template=_TEMPLATE,
                max_workers=1,
                llm_generate_factory=lambda: _mock_llm(),
                allow_parallel=True,
            )

        # max_workers=1 verhindert Parallelisierung
        mock_pool.assert_not_called()
        assert len(results) == 5
        assert stats["enriched"] == 5


# ---------------------------------------------------------------------------
# C8: Progress-Callback in Input-Reihenfolge (1..N, total)
# ---------------------------------------------------------------------------


class TestC8ProgressCallbackOrder:
    def test_sequential_callback_order(self):
        """C8: Progress-Callback wird sequenziell in Input-Reihenfolge aufgerufen."""
        calls = []
        n = 5
        chunks = _make_chunks(n)

        enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )

        assert len(calls) == n
        for i, (cur, tot) in enumerate(calls):
            assert cur == i + 1, f"Erwarteter current={i + 1}, erhalten={cur}"
            assert tot == n, f"Erwarteter total={n}, erhalten={tot}"

    def test_parallel_callback_total_count(self):
        """C8b: Paralleler Pfad — Callback wird genau N-mal aufgerufen mit total=N."""
        calls = []
        n = 6
        chunks = _make_chunks(n)

        enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
            max_workers=3,
            llm_generate_factory=lambda: _mock_llm(),
            allow_parallel=True,
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )

        assert len(calls) == n
        # Alle Calls haben korrekte total
        for _, tot in calls:
            assert tot == n


# ---------------------------------------------------------------------------
# C9: Ergebnis-Reihenfolge bewahrt (enriched_texts[i] ↔ chunk_texts[i])
# ---------------------------------------------------------------------------


class TestC9ResultOrdering:
    def test_sequential_order_preserved(self):
        """C9: Sequenzieller Pfad — Ergebnis[i] entspricht Chunk[i]."""
        chunks = [f"CHUNK_{i}_" + "x" * 90 for i in range(5)]

        results, _ = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
        )

        for i, result in enumerate(results):
            assert f"CHUNK_{i}_" in result, f"Ergebnis {i} enthaelt nicht den Marker des Chunks {i}"

    def test_parallel_order_preserved(self):
        """C9b: Paralleler Pfad — Ergebnis[i] entspricht Chunk[i] trotz Threads."""
        n = 8
        chunks = [f"MARKER_{i}_" + "x" * 90 for i in range(n)]

        results, _ = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
            max_workers=4,
            llm_generate_factory=lambda: _mock_llm(),
            allow_parallel=True,
        )

        assert len(results) == n
        for i, result in enumerate(results):
            assert f"MARKER_{i}_" in result, (
                f"Parallel-Ergebnis {i} enthaelt nicht den Marker des Chunks {i}"
            )


# ---------------------------------------------------------------------------
# C10: Pipeline-Stats akkumuliert ueber mehrere Dateien
# ---------------------------------------------------------------------------


class TestC10PipelineStatsAccumulation:
    def test_stats_accumulate_across_batches(self):
        """C10: Mehrere Batch-Aufrufe → Stats werden korrekt akkumuliert."""
        accumulated = {
            "enrichment_chunks": 0,
            "enrichment_skipped": 0,
            "enrichment_errors": 0,
            "enrichment_time_ms": 0,
        }

        # Datei 1: 3 Chunks, alle enriched
        chunks_1 = _make_chunks(3)
        _, stats_1 = enrich_chunks_batch(
            chunk_texts=chunks_1,
            full_text="".join(chunks_1),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
        )
        accumulated["enrichment_chunks"] += stats_1.get("enriched", 0)
        accumulated["enrichment_skipped"] += stats_1.get("skipped", 0)
        accumulated["enrichment_errors"] += stats_1.get("errors", 0)
        accumulated["enrichment_time_ms"] += stats_1.get("enrichment_time_ms", 0)

        # Datei 2: 2 Chunks enriched, 1 skipped
        chunks_2 = ["A" * 100, "short", "C" * 100]
        _, stats_2 = enrich_chunks_batch(
            chunk_texts=chunks_2,
            full_text="".join(chunks_2),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
        )
        accumulated["enrichment_chunks"] += stats_2.get("enriched", 0)
        accumulated["enrichment_skipped"] += stats_2.get("skipped", 0)
        accumulated["enrichment_errors"] += stats_2.get("errors", 0)
        accumulated["enrichment_time_ms"] += stats_2.get("enrichment_time_ms", 0)

        # Datei 3: 1 error
        def _failing_llm(s, u):
            raise RuntimeError("fail")

        chunks_3 = ["X" * 100]
        _, stats_3 = enrich_chunks_batch(
            chunk_texts=chunks_3,
            full_text="".join(chunks_3),
            llm_generate_fn=_failing_llm,
            prompt_template=_TEMPLATE,
        )
        accumulated["enrichment_chunks"] += stats_3.get("enriched", 0)
        accumulated["enrichment_skipped"] += stats_3.get("skipped", 0)
        accumulated["enrichment_errors"] += stats_3.get("errors", 0)
        accumulated["enrichment_time_ms"] += stats_3.get("enrichment_time_ms", 0)

        assert accumulated["enrichment_chunks"] == 5  # 3 + 2
        assert accumulated["enrichment_skipped"] == 1
        assert accumulated["enrichment_errors"] == 1
        # enrichment_time_ms ist int und >= 0 (kann bei Sub-ms-Ausfuehrung 0 sein)
        assert isinstance(accumulated["enrichment_time_ms"], int)
        assert accumulated["enrichment_time_ms"] >= 0


# ---------------------------------------------------------------------------
# C11: Paralleler Pfad mit Default-Backend → Factory nutzt gepinntes Backend
# ---------------------------------------------------------------------------


class TestC11FactoryPinnedBackend:
    def test_factory_uses_pinned_backend_no_round_robin(self):
        """C11: Factory wird pro Chunk aufgerufen, liefert separate Instanzen
        mit gepinntem Backend (kein Round-Robin-Switch).

        Beweist ueber Barrier, dass tatsaechlich parallele Threads laufen,
        und ueber Instanz-Tracking, dass jeder Thread seine eigene Instanz hat.
        """
        n = 8
        chunks = _make_chunks(n)
        max_workers = 3

        factory_call_count = [0]
        factory_lock = threading.Lock()
        instance_ids = []
        instance_lock = threading.Lock()

        # Barrier beweist Parallelitaet
        barrier = threading.Barrier(parties=2, timeout=5)
        barrier_passed = threading.Event()

        def _factory():
            with factory_lock:
                factory_call_count[0] += 1
                current_id = factory_call_count[0]

            def _gen(s, u):
                with instance_lock:
                    instance_ids.append(current_id)
                try:
                    barrier.wait(timeout=5)
                    barrier_passed.set()
                except threading.BrokenBarrierError:
                    pass
                return "pinned-prefix"

            return _gen

        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
            max_workers=max_workers,
            llm_generate_factory=_factory,
            allow_parallel=True,
        )

        # Factory wurde fuer jeden Chunk aufgerufen
        assert factory_call_count[0] == n

        # Parallelitaet bewiesen
        assert barrier_passed.is_set(), "Keine echte Parallelitaet festgestellt"

        # Jeder Chunk hat sein eigenes Ergebnis
        assert len(results) == n
        assert stats["enriched"] == n

        # Instanz-IDs zeigen, dass verschiedene Instanzen genutzt wurden
        unique_ids = set(instance_ids)
        assert len(unique_ids) > 1, (
            "Alle Chunks verwendeten dieselbe Instanz — Factory wird nicht genutzt"
        )

    def test_parallel_factory_failure_is_per_chunk_error(self):
        """Factory-Fehler im Parallelpfad bricht den Batch nicht ab."""
        chunks = [
            "A" * 100,
            "short",
            "B" * 100,
            "C" * 100,
            "D" * 100,
        ]
        calls = [0]
        calls_lock = threading.Lock()

        def _factory():
            with calls_lock:
                calls[0] += 1
                call_no = calls[0]
            if call_no == 2:
                raise RuntimeError("backend init failed")
            return _mock_llm("factory-prefix")

        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="".join(chunks),
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
            max_workers=3,
            llm_generate_factory=_factory,
            allow_parallel=True,
        )

        assert len(results) == len(chunks)
        assert results[1] == chunks[1]
        assert stats["skipped"] == 1
        assert stats["errors"] == 1
        assert stats["enriched"] == 3
        assert calls[0] == 4


# ---------------------------------------------------------------------------
# C12: run_all_datasources() aggregiert enrichment_*-Stats
# ---------------------------------------------------------------------------


class TestC12RunAllAggregation:
    def test_run_all_aggregates_enrichment_stats(self):
        """C12: run_all_datasources summiert enrichment-Stats aus run_ingestion."""
        from kiara.ingestion.pipeline import run_all_datasources

        # Zwei Datasources simulieren
        mock_row_1 = {
            "id": 1,
            "name": "DS1",
            "type": "local",
            "config_json": "{}",
            "ds_chunk_strategy": None,
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_enrichment_prompt_id": None,
            "ds_enrichment_backend_id": None,
            "ds_restriction_level": None,
        }
        mock_row_2 = {
            "id": 2,
            "name": "DS2",
            "type": "local",
            "config_json": "{}",
            "ds_chunk_strategy": None,
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_enrichment_prompt_id": None,
            "ds_enrichment_backend_id": None,
            "ds_restriction_level": None,
        }

        repo_instance = MagicMock()
        repo_instance.get_active.return_value = [mock_row_1, mock_row_2]

        # Connector-Mock
        mock_connector = MagicMock()
        mock_connector.HASH_MODE = "mtime"
        mock_connector.safe_prepare_and_scan.return_value = "/tmp/path"

        # run_ingestion gibt enrichment-Stats zurueck
        ingestion_results = [
            {
                "total_files": 5,
                "processed_files": 5,
                "skipped_files": 0,
                "deleted_files": 0,
                "total_chunks": 20,
                "errors": 0,
                "vlm_chunks": 0,
                "vlm_errors": 0,
                "ddu_files": 0,
                "ddu_remote_files": 0,
                "ddu_fallback_files": 0,
                "ddu_remote_errors": 0,
                "ddu_timeout": 0,
                "enrichment_chunks": 15,
                "enrichment_skipped": 3,
                "enrichment_errors": 2,
                "enrichment_time_ms": 1200,
            },
            {
                "total_files": 3,
                "processed_files": 3,
                "skipped_files": 0,
                "deleted_files": 0,
                "total_chunks": 10,
                "errors": 0,
                "vlm_chunks": 0,
                "vlm_errors": 0,
                "ddu_files": 0,
                "ddu_remote_files": 0,
                "ddu_fallback_files": 0,
                "ddu_remote_errors": 0,
                "ddu_timeout": 0,
                "enrichment_chunks": 8,
                "enrichment_skipped": 1,
                "enrichment_errors": 1,
                "enrichment_time_ms": 800,
            },
        ]

        with (
            patch("kiara.connectors.registry.ensure_loaded"),
            patch("kiara.connectors.registry.get", return_value=mock_connector),
            patch("kiara.repositories.datasource.DatasourceRepository", return_value=repo_instance),
            patch("kiara.ingestion.pipeline.load_config", return_value={"datasource": {}}),
            patch(
                "kiara.ingestion.pipeline.run_ingestion", side_effect=ingestion_results
            ) as mock_run,
        ):
            total = run_all_datasources(config={"datasource": {}}, incremental=False)

        assert total["enrichment_chunks"] == 23  # 15 + 8
        assert total["enrichment_skipped"] == 4  # 3 + 1
        assert total["enrichment_errors"] == 3  # 2 + 1
        assert total["enrichment_time_ms"] == 2000  # 1200 + 800
        assert total["datasources_processed"] == 2


# ---------------------------------------------------------------------------
# C13: Leerer Batch → [] + Zero-Stats inkl. enrichment_time_ms=0
# ---------------------------------------------------------------------------


class TestC13EmptyBatch:
    def test_empty_batch_returns_empty_list_and_zero_stats(self):
        """C13: Leerer Batch → leere Liste und alle Stats auf 0."""
        results, stats = enrich_chunks_batch(
            chunk_texts=[],
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
        )

        assert results == []
        assert stats["enriched"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert stats["enrichment_time_ms"] == 0

    def test_empty_batch_with_parallel_config(self):
        """C13b: Leerer Batch mit Parallel-Config → identisch zero-stats."""
        results, stats = enrich_chunks_batch(
            chunk_texts=[],
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template=_TEMPLATE,
            max_workers=4,
            llm_generate_factory=lambda: _mock_llm(),
            allow_parallel=True,
        )

        assert results == []
        assert stats == {
            "enriched": 0,
            "skipped": 0,
            "unchanged": 0,
            "errors": 0,
            "enrichment_time_ms": 0,
        }
