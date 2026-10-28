"""Pipeline-Integrationstests fuer Chunk-Strategie (Phase 7).

Testet die Strategie-Auswahl in run_ingestion():
  - Auto-Erkennung via detect_chunk_params()
  - Feste Strategie aus STRATEGIES-Dict
  - Custom chunk_size/overlap Override
  - Prioritaet: feste Strategie > custom > auto
  - Fallback bei Exception in detect_chunk_params()
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from kiara.ingestion.chunk_strategy import STRATEGIES

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _base_config(tmp_path):
    """Minimale Config fuer Pipeline-Tests."""
    kb = tmp_path / "kb"
    kb.mkdir(exist_ok=True)
    vdb = tmp_path / "vectordb"
    vdb.mkdir(exist_ok=True)
    return {
        "datasource": {"path": str(kb), "file_types": [".txt"], "exclude_patterns": []},
        "vectordb": {"path": str(vdb), "collection": "test"},
        "ingestion": {"chunk_size": 768, "chunk_overlap": 200, "batch_size": 50},
        "ollama": {"embedding_url": "", "embedding_model": "test"},
        "graph": {"enabled": False},
    }


def _make_doc(source="/data/test.txt"):
    """Erzeugt ein Mock-Dokument wie load_file() es zurueckgibt."""
    return {
        "text": "Normaler Testtext mit genuegend Inhalt fuer Chunk-Strategie-Tests.",
        "metadata": {"file_type": ".txt", "source": source},
    }


def _make_chunks(source="/data/test.txt"):
    """Erzeugt Mock-Chunks wie chunk_document() sie liefert."""
    return [{"text": "chunk1", "metadata": {"source": source, "chunk_index": 0}}]


# ---------------------------------------------------------------------------
# Fixture: Alle Pipeline-Abhaengigkeiten gemockt
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_env(tmp_path):
    """Richtet alle Pipeline-Mocks ein und liefert Config + Mocks."""
    path = tmp_path / "test.txt"
    path.write_text(_make_doc()["text"])
    test_file = str(path)
    config = _base_config(tmp_path)

    mock_hash_store = MagicMock()
    mock_hash_store.load.return_value = {}
    mock_hash_store.upsert_many = MagicMock()
    mock_hash_store.remove_files = MagicMock()
    mock_hash_store.set_hash_mode = MagicMock()

    def fake_load_and_diff(cfg, store, incremental, ds_id, emb_model, **kwargs):
        return (
            [test_file],  # all_files
            [test_file],  # files_to_process
            {test_file: ("abc123", 100)},  # file_hashes
            [],  # deleted_files
            mock_hash_store,  # hash_store
            {},  # resumed_hashes
        )

    with (
        patch("kiara.ingestion.pipeline._load_and_diff_files", side_effect=fake_load_and_diff),
        patch("kiara.ingestion.runtime.load_file", return_value=_make_doc(test_file)) as m_load,
        patch(
            "kiara.ingestion.runtime.chunk_document", return_value=_make_chunks(test_file)
        ) as m_chunk,
        patch("kiara.ingestion.pipeline.VectorStore") as m_store,
        patch("kiara.ingestion.pipeline.ChunkDatasources") as m_chunk_ds,
        patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
    ):
        m_store.return_value.finalize_document_versions.side_effect = (
            lambda ids, *, publish_search, on_complete: on_complete(
                {test_file: {"standard": "test-version"}}
            )
        )
        yield {
            "config": config,
            "chunk": m_chunk,
            "load": m_load,
            "store": m_store,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_auto_detect_called_when_no_overrides(pipeline_env):
    """run_ingestion() ruft detect_chunk_params() auf wenn ds_overrides leer/None."""
    from kiara.ingestion.pipeline import run_ingestion

    with patch(
        "kiara.ingestion.chunk_strategy.detect_chunk_params", return_value=(768, 150)
    ) as mock_detect:
        run_ingestion(pipeline_env["config"], ds_overrides=None)
        mock_detect.assert_called_once()


def test_auto_detect_called_when_strategy_is_auto(pipeline_env):
    """run_ingestion() ruft detect_chunk_params() auf wenn chunk_strategy='auto'."""
    from kiara.ingestion.pipeline import run_ingestion

    with patch(
        "kiara.ingestion.chunk_strategy.detect_chunk_params", return_value=(768, 150)
    ) as mock_detect:
        run_ingestion(pipeline_env["config"], ds_overrides={"chunk_strategy": "auto"})
        mock_detect.assert_called_once()


def test_fixed_strategy_uses_strategies_dict(pipeline_env):
    """run_ingestion() nutzt STRATEGIES['tabular'] bei chunk_strategy='tabular'."""
    from kiara.ingestion.pipeline import run_ingestion

    run_ingestion(pipeline_env["config"], ds_overrides={"chunk_strategy": "tabular"})

    call_kwargs = pipeline_env["chunk"].call_args.kwargs
    assert call_kwargs["chunk_size"] == STRATEGIES["tabular"][0]  # 256
    assert call_kwargs["chunk_overlap"] == STRATEGIES["tabular"][1]  # 0


def test_custom_chunk_size_override(pipeline_env):
    """run_ingestion() nutzt custom chunk_size/overlap aus ds_overrides."""
    from kiara.ingestion.pipeline import run_ingestion

    run_ingestion(pipeline_env["config"], ds_overrides={"chunk_size": 500, "chunk_overlap": 50})

    call_kwargs = pipeline_env["chunk"].call_args.kwargs
    assert call_kwargs["chunk_size"] == 500
    assert call_kwargs["chunk_overlap"] == 50


def test_override_priority_strategy_beats_custom(pipeline_env):
    """Feste Strategie hat Vorrang vor custom chunk_size/overlap."""
    from kiara.ingestion.pipeline import run_ingestion

    run_ingestion(
        pipeline_env["config"],
        ds_overrides={
            "chunk_strategy": "tabular",
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
    )

    call_kwargs = pipeline_env["chunk"].call_args.kwargs
    # Strategie-Werte statt custom
    assert call_kwargs["chunk_size"] == STRATEGIES["tabular"][0]  # 256
    assert call_kwargs["chunk_overlap"] == STRATEGIES["tabular"][1]  # 0


def test_detect_exception_fallback_to_default(pipeline_env, caplog):
    """Exception in detect_chunk_params() → Fallback auf Config-Default + Warning."""
    from kiara.ingestion.pipeline import run_ingestion

    with (
        patch(
            "kiara.ingestion.chunk_strategy.detect_chunk_params",
            side_effect=RuntimeError("Test-Fehler"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        run_ingestion(pipeline_env["config"], ds_overrides=None)

    call_kwargs = pipeline_env["chunk"].call_args.kwargs
    # Fallback auf globale Config-Defaults
    assert call_kwargs["chunk_size"] == 768
    assert call_kwargs["chunk_overlap"] == 200
    # Warning-Log vorhanden
    assert any("fehlgeschlagen" in r.message for r in caplog.records)
