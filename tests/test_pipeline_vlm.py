"""Tests fuer VLM-Integration in der Ingestion-Pipeline (Phase 10).

Testet VLM-Toggle Durchreichung, Bild-Extensions Injektion, Multi-Doc,
Chunk-ID mit Parser-Typ, VLM-Einzelchunk-Strategie, VLM Stats-Tracking,
Config-Mutations-Fix und Stats-Aggregation.
Alle Tests laufen OHNE Ollama-Server und OHNE VLM-Modell (komplett gemockt).
"""

import copy
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from kiara.ingestion.pipeline import run_all_datasources, run_ingestion


@pytest.fixture(autouse=True)
def _mock_backend_selectors():
    """Mockt Backend-Selektoren und DDU-Preflight fuer alle Tests.

    get_vlm_backend gibt ein gueltiges VLM-Backend zurueck,
    get_ddu_backends gibt ein gueltiges DDU-Backend zurueck,
    _check_ddu_preflight wird uebersprungen (kein echtes DDU-Backend noetig).
    Tests die explizit kein VLM-Backend wollen, koennen mit eigenem Patch ueberschreiben.
    """
    ddu_backend = [
        {
            "url": "http://ddu:5001",
            "is_local": False,
            "model": "docling",
            "table_mode": "accurate",
            "do_ocr": True,
            "ocr_lang": ["de", "en"],
            "timeout": 120.0,
            "concurrency_limit": None,
        }
    ]
    with (
        patch(
            "kiara.config.get_vlm_backend",
            return_value={"url": "http://localhost:11434", "model": "qwen2.5-vl:7b"},
        ),
        patch("kiara.config.get_ddu_backends", return_value=ddu_backend),
        patch("kiara.ingestion.pipeline._check_ddu_preflight"),
    ):
        yield


def _mock_db_query_for_rows(rows):
    """Erstellt einen Patch fuer DatasourceRepository.get_active mit Mock-Rows."""
    return patch("kiara.repositories.datasource.DatasourceRepository.get_active", return_value=rows)


# -- Helpers ------------------------------------------------------------------


def _base_config(tmp_path):
    """Minimale Config fuer Pipeline-Tests (mit VLM-Model)."""
    ds_path = tmp_path / "docs"
    ds_path.mkdir(exist_ok=True)
    vdb_path = tmp_path / "vdb"
    vdb_path.mkdir(exist_ok=True)
    return {
        "datasource": {
            "path": str(ds_path),
            "file_types": [".pdf", ".txt", ".docx"],
            "exclude_patterns": [],
        },
        "ingestion": {
            "chunk_size": 768,
            "chunk_overlap": 200,
            "batch_size": 50,
        },
        "vectordb": {
            "path": str(vdb_path),
        },
        "ollama": {
            "url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "vlm_model": "qwen2.5-vl:7b",
        },
        "graph": {"enabled": False},
    }


def _make_file(directory, name, content="Test content"):
    """Erstelle eine Datei im angegebenen Verzeichnis."""
    f = directory / name
    if isinstance(content, bytes):
        f.write_bytes(content)
    else:
        f.write_text(content)
    return str(f)


# -- 2.1 run_ingestion() — VLM-Toggle Durchreichung --------------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_enabled_calls_describe_image(mock_load, mock_describe, mock_vs, tmp_path):
    """ds_overrides={'vlm_enabled': True} + config mit vlm_model -> describe_image() aufgerufen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "photo.png", b"\x89PNG\x00\x00")

    mock_load.return_value = None  # kein Standard-Loader fuer Bilder
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": str(ds_path / "photo.png"), "parser": "vlm", "file_type": ".png"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    mock_describe.assert_called_once()


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_disabled_no_describe_image_call(mock_load, mock_describe, mock_vs, tmp_path):
    """ds_overrides ohne vlm_enabled -> describe_image() wird NICHT aufgerufen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={})

    mock_describe.assert_not_called()


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
@patch("kiara.config.get_vlm_backend", return_value=None)
def test_vlm_enabled_no_model_no_describe_image_call(
    mock_vlm_backend, mock_load, mock_describe, mock_vs, tmp_path
):
    """ds_overrides={'vlm_enabled': True} aber kein VLM-Backend -> describe_image() NICHT aufgerufen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    mock_describe.assert_not_called()


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_overrides_none_no_describe_image_call(mock_load, mock_describe, mock_vs, tmp_path):
    """ds_overrides=None -> describe_image() wird NICHT aufgerufen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)

    mock_describe.assert_not_called()


# -- 2.1b run_ingestion() — VLM-Prompt Durchreichung --------------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_prompt_from_config_passed_to_describe_image(
    mock_load, mock_describe, mock_vs, tmp_path
):
    """ingestion.vlm_prompt aus Config wird als prompt= an describe_image() durchgereicht."""
    config = _base_config(tmp_path)
    config["ingestion"]["vlm_prompt"] = "Describe in English please."
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "photo.png", b"\x89PNG\x00\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Image description",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    mock_describe.assert_called_once()
    call_kwargs = mock_describe.call_args
    assert call_kwargs.kwargs.get("prompt") == "Describe in English please."


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_prompt_empty_passes_empty_string(mock_load, mock_describe, mock_vs, tmp_path):
    """Leerer vlm_prompt in Config -> prompt='' an describe_image() (Fallback in vision.py)."""
    config = _base_config(tmp_path)
    # vlm_prompt nicht gesetzt -> Default ""
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "photo.png", b"\x89PNG\x00\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    mock_describe.assert_called_once()
    call_kwargs = mock_describe.call_args
    assert call_kwargs.kwargs.get("prompt") == ""


# -- 2.2 run_ingestion() — Bild-Extensions -----------------------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_enabled_adds_image_types(mock_load, mock_describe, mock_vs, tmp_path):
    """vlm_enabled=True -> .png/.jpg/.tiff werden in file_types aufgenommen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "scan.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "VLM text",
        "metadata": {"source": str(ds_path / "scan.png"), "parser": "vlm", "file_type": ".png"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    # scan.png muss verarbeitet worden sein
    assert mock_describe.call_count == 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_disabled_no_image_types(mock_load, mock_describe, mock_vs, tmp_path):
    """vlm_enabled=False -> Bilddateien werden NICHT gescannt."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "scan.png", b"\x89PNG\x00")

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)

    # scan.png darf nicht verarbeitet werden
    mock_load.assert_not_called()
    mock_describe.assert_not_called()


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_or_vlm_both_add_image_types(mock_load, mock_describe, mock_vs, tmp_path):
    """Entweder ddu_enabled=True ODER vlm_enabled=True -> Bild-Extensions vorhanden."""
    for overrides_dict in [{"ddu_enabled": True}, {"vlm_enabled": True}]:
        config = _base_config(tmp_path)
        ds_path = tmp_path / "docs"
        _make_file(ds_path, "img.png", b"\x89PNG\x00")

        mock_load.return_value = None
        mock_describe.return_value = {
            "text": "VLM text",
            "metadata": {"source": str(ds_path / "img.png"), "parser": "vlm", "file_type": ".png"},
        }

        run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=overrides_dict)

        # Bild-Extensions sollten in file_types sein
        ft = config["datasource"]["file_types"]
        for ext in [".png", ".tiff", ".tif", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic"]:
            assert ext in ft, f"{ext} missing in file_types for {overrides_dict}"

        mock_load.reset_mock()
        mock_describe.reset_mock()


# -- 2.3 run_ingestion() — Multi-Doc (docs_to_chunk) -------------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_both_ddu_and_vlm_produce_two_chunks(mock_load, mock_describe, mock_vs, tmp_path):
    """load_file -> doc1 (docling-remote), describe_image -> doc2 (vlm)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = {
        "text": "OCR text from docling-remote",
        "metadata": {
            "source": img_path,
            "parser": "docling-remote",
            "file_type": ".png",
            "chunk_index": 0,
        },
    }
    mock_describe.return_value = {
        "text": "Bildbeschreibung vom VLM",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(
        config,
        embedding_model="nomic-embed-text",
        ds_overrides={"ddu_enabled": True, "vlm_enabled": True},
    )

    # store.add() muss mit 2 Chunks aufgerufen worden sein (DDU + VLM)
    total_texts = []
    for c in mock_store.add.call_args_list:
        total_texts.extend(c[0][0])
    assert len(total_texts) == 2


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_only_vlm_doc_when_load_file_returns_none(mock_load, mock_describe, mock_vs, tmp_path):
    """load_file -> None, describe_image -> doc. errors wird NICHT erhoeht."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    assert stats["errors"] == 0
    # VLM-Chunk wurde geschrieben
    assert mock_store.add.call_count >= 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_only_load_file_doc_when_vlm_fails(mock_load, mock_describe, mock_vs, tmp_path):
    """load_file -> doc, describe_image -> None. 1 Chunk, vlm_errors erhoeht."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = {
        "text": "Standard text",
        "metadata": {
            "source": img_path,
            "parser": "standard",
            "file_type": ".png",
            "chunk_index": 0,
        },
    }
    mock_describe.return_value = None  # VLM fehlgeschlagen

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    assert stats.get("vlm_errors", 0) == 1
    # 1 Chunk von load_file
    total_texts = []
    for c in mock_store.add.call_args_list:
        total_texts.extend(c[0][0])
    assert len(total_texts) == 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_both_none_increments_errors(mock_load, mock_describe, mock_vs, tmp_path):
    """load_file -> None, describe_image -> None. errors erhoeht."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = None

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    assert stats["errors"] >= 1


# -- 2.4 run_ingestion() — Chunk-ID mit Parser-Typ (K1-Fix) ------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_chunk_id_contains_parser_type(mock_load, mock_describe, mock_vs, tmp_path):
    """Chunk-ID muss Format '{source}::{parser}::{chunk_index}' haben (MD5)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    # Verifiziere: store.add() erhaelt IDs im neuen Format
    add_call = mock_store.add.call_args
    ids = add_call[0][2]
    expected_id = hashlib.md5(f"{img_path}::vlm::0".encode()).hexdigest()
    assert expected_id in ids


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_chunk_id_ddu_and_vlm_different(mock_load, mock_describe, mock_vs, tmp_path):
    """Fuer dieselbe Bilddatei: DDU-Chunk-ID != VLM-Chunk-ID."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = {
        "text": "OCR text",
        "metadata": {
            "source": img_path,
            "parser": "docling-remote",
            "file_type": ".png",
            "chunk_index": 0,
        },
    }
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(
        config,
        embedding_model="nomic-embed-text",
        ds_overrides={"ddu_enabled": True, "vlm_enabled": True},
    )

    # Sammle alle IDs
    all_ids = []
    for c in mock_store.add.call_args_list:
        all_ids.extend(c[0][2])

    # Muss mindestens 2 verschiedene IDs geben (DDU vs VLM)
    assert len(all_ids) >= 2
    assert len(set(all_ids)) == len(all_ids), "Chunk-IDs muessen eindeutig sein"


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_chunk_id_standard_parser_for_text_files(mock_load, mock_describe, mock_vs, tmp_path):
    """Text-Datei -> Chunk-ID enthaelt '::standard::'."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    txt_path = _make_file(ds_path, "readme.txt", "Some documentation text")

    mock_load.return_value = {
        "text": "Some documentation text",
        "metadata": {
            "source": txt_path,
            "parser": "standard",
            "file_type": ".txt",
            "chunk_index": 0,
        },
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text")

    add_call = mock_store.add.call_args
    ids = add_call[0][2]
    expected_id = hashlib.md5(f"{txt_path}::standard::0".encode()).hexdigest()
    assert expected_id in ids


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_chunk_id_deterministic(mock_load, mock_describe, mock_vs, tmp_path):
    """Gleiche Datei + gleicher Parser -> gleiche Chunk-ID (MD5 ist deterministisch)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    # Erster Lauf
    run_ingestion(
        copy.deepcopy(config),
        embedding_model="nomic-embed-text",
        ds_overrides={"vlm_enabled": True},
    )
    ids_run1 = []
    for c in mock_store.add.call_args_list:
        ids_run1.extend(c[0][2])

    mock_store.reset_mock()

    # Zweiter Lauf
    run_ingestion(
        copy.deepcopy(config),
        embedding_model="nomic-embed-text",
        ds_overrides={"vlm_enabled": True},
    )
    ids_run2 = []
    for c in mock_store.add.call_args_list:
        ids_run2.extend(c[0][2])

    assert ids_run1 == ids_run2


# -- 2.5 run_ingestion() — VLM-Einzelchunk-Strategie -------------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_doc_is_single_chunk(mock_load, mock_describe, mock_vs, tmp_path):
    """VLM-Beschreibung -> genau 1 Chunk (kein Splitting durch chunk_document)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    # Genau 1 Chunk geschrieben
    total_texts = []
    for c in mock_store.add.call_args_list:
        total_texts.extend(c[0][0])
    assert len(total_texts) == 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_doc_chunk_strategy_is_vlm_single(mock_load, mock_describe, mock_vs, tmp_path):
    """VLM-Chunk-Metadata: chunk_strategy == 'vlm_single'."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    add_call = mock_store.add.call_args
    metas = add_call[0][1]
    assert len(metas) == 1
    assert metas[0]["chunk_strategy"] == "vlm_single"


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_doc_chunk_index_is_zero(mock_load, mock_describe, mock_vs, tmp_path):
    """VLM-Chunk-Metadata: chunk_index == 0."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    add_call = mock_store.add.call_args
    metas = add_call[0][1]
    assert metas[0]["chunk_index"] == 0


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_doc_long_text_still_single_chunk(mock_load, mock_describe, mock_vs, tmp_path):
    """VLM-Beschreibung mit 3000 Zeichen -> trotzdem nur 1 Chunk (kein Split bei 768)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    long_text = "A" * 3000  # Deutlich ueber chunk_size (768)
    mock_load.return_value = None
    mock_describe.return_value = {
        "text": long_text,
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    total_texts = []
    for c in mock_store.add.call_args_list:
        total_texts.extend(c[0][0])
    assert len(total_texts) == 1
    assert total_texts[0] == long_text


# -- 2.6 run_ingestion() — VLM Stats-Tracking --------------------------------


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_chunks_counter_incremented(mock_load, mock_describe, mock_vs, tmp_path):
    """Erfolgreiche VLM-Beschreibung -> stats['vlm_chunks'] == 1."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = {
        "text": "Bildbeschreibung",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    assert stats.get("vlm_chunks", 0) == 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_errors_counter_incremented(mock_load, mock_describe, mock_vs, tmp_path):
    """Fehlgeschlagene VLM-Beschreibung fuer Bilddatei -> stats['vlm_errors'] == 1."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "img.png", b"\x89PNG\x00")

    mock_load.return_value = None
    mock_describe.return_value = None  # VLM fehlgeschlagen

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    assert stats.get("vlm_errors", 0) == 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_errors_not_incremented_for_non_images(mock_load, mock_describe, mock_vs, tmp_path):
    """describe_image() gibt None fuer .pdf -> vlm_errors wird NICHT erhoeht."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    pdf_path = _make_file(ds_path, "doc.pdf")

    mock_load.return_value = {
        "text": "PDF content",
        "metadata": {
            "source": pdf_path,
            "parser": "standard",
            "file_type": ".pdf",
            "chunk_index": 0,
        },
    }
    mock_describe.return_value = None  # .pdf ist kein Bild → describe_image gibt None

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    # vlm_errors darf NICHT erhoeht werden (weil .pdf nicht in IMAGE_FORMATS)
    assert stats.get("vlm_errors", 0) == 0


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
def test_vlm_stats_multiple_files(mock_load, mock_describe, mock_vs, tmp_path):
    """3 Bilder: 2 erfolgreich, 1 fehlgeschlagen -> vlm_chunks==2, vlm_errors==1."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    img1 = _make_file(ds_path, "img1.png", b"\x89PNG\x00")
    img2 = _make_file(ds_path, "img2.jpg", b"\xff\xd8\xff\x00")
    img3 = _make_file(ds_path, "img3.tiff", b"\x49\x49\x2a\x00")

    mock_load.return_value = None  # Kein Standard-Loader fuer Bilder

    def describe_side_effect(file_path, **kwargs):
        if "img3" in file_path:
            return None  # Fehlgeschlagen
        return {
            "text": f"Beschreibung von {file_path}",
            "metadata": {"source": file_path, "parser": "vlm", "file_type": ".png"},
        }

    mock_describe.side_effect = describe_side_effect

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True}
    )

    assert stats.get("vlm_chunks", 0) == 2
    assert stats.get("vlm_errors", 0) == 1


# -- 2.7 run_all_datasources() — Config-Mutations-Fix ------------------------


def _create_ds_rows_vlm(tmp_path, datasources):
    """Erstelle Mock-Rows fuer mehrere Datasources fuer VLM-Tests."""
    rows = []
    for ds in datasources:
        doc_path = tmp_path / f"docs_{ds['id']}"
        doc_path.mkdir(exist_ok=True)
        rows.append(
            {
                "id": ds["id"],
                "name": ds["name"],
                "type": "local",
                "path": str(doc_path),
                "is_active": True,
                "connector_config": json.dumps({"path": str(doc_path)}),
                "ds_chunk_size": None,
                "ds_chunk_overlap": None,
                "ds_file_types": None,
                "ds_exclude_patterns": None,
                "ds_chunk_strategy": "auto",
                "ds_ddu_enabled": ds.get("ddu_enabled", 0),
                "ds_vlm_enabled": ds.get("vlm_enabled", 0),
            }
        )
    return rows


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_config_file_types_reset_between_datasources(mock_mounted, mock_mount, mock_run, tmp_path):
    """DS1 vlm_enabled=True, DS2 vlm_enabled=False. Nach DS1 duerfen .png/.jpg NICHT in DS2 file_types sein."""
    rows = _create_ds_rows_vlm(
        tmp_path,
        [
            {"id": 1, "name": "VLM DS", "vlm_enabled": 1},
            {"id": 2, "name": "Normal DS", "vlm_enabled": 0},
        ],
    )
    config = _base_config(tmp_path)

    captured_configs = []

    def capture_config(cfg, **kwargs):
        captured_configs.append(copy.deepcopy(cfg))
        return {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

    mock_run.side_effect = capture_config

    with _mock_db_query_for_rows(rows):
        run_all_datasources(config, incremental=False)

    assert mock_run.call_count == 2
    # DS2 (Normal DS) sollte KEINE Bild-Extensions haben
    ds2_call_kwargs = mock_run.call_args_list[1][1]
    ds2_overrides = ds2_call_kwargs.get("ds_overrides")
    # VLM sollte nicht in den Overrides sein
    assert ds2_overrides is None or "vlm_enabled" not in ds2_overrides


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_config_deepcopy_isolates_datasources(mock_mounted, mock_mount, mock_run, tmp_path):
    """Aenderungen an config in DS1 duerfen DS2 nicht beeinflussen."""
    rows = _create_ds_rows_vlm(
        tmp_path,
        [
            {"id": 1, "name": "DS1", "vlm_enabled": 1},
            {"id": 2, "name": "DS2", "vlm_enabled": 0},
        ],
    )
    config = _base_config(tmp_path)

    original_file_types = list(config["datasource"]["file_types"])

    captured_configs = []

    def capture_config(cfg, **kwargs):
        captured_configs.append(copy.deepcopy(cfg))
        return {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }

    mock_run.side_effect = capture_config

    with _mock_db_query_for_rows(rows):
        run_all_datasources(config, incremental=False)

    assert len(captured_configs) == 2
    # DS2 config sollte die Original-file_types haben (deepcopy wirkt)
    ds2_ft = set(captured_configs[1]["datasource"]["file_types"])
    # DS2 hat kein VLM, also sollten keine Bildformate hinzugefuegt sein
    # (es sei denn, sie waren schon in der Original-Config)
    orig_set = set(original_file_types)
    for ext in [".png", ".tiff", ".tif", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic"]:
        if ext not in orig_set:
            assert ext not in ds2_ft, f"{ext} leaked from DS1 to DS2"


# -- 2.8 run_all_datasources() — VLM Stats-Aggregation (G2-Fix) -------------


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_run_all_ds_aggregates_vlm_chunks(mock_mounted, mock_mount, mock_run, tmp_path):
    """2 Datasources mit VLM -> total_stats['vlm_chunks'] ist Summe beider."""
    rows = _create_ds_rows_vlm(
        tmp_path,
        [
            {"id": 1, "name": "DS1", "vlm_enabled": 1},
            {"id": 2, "name": "DS2", "vlm_enabled": 1},
        ],
    )
    config = _base_config(tmp_path)

    call_count = [0]

    def side_effect(cfg, **kwargs):
        call_count[0] += 1
        return {
            "total_files": 1,
            "processed_files": 1,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 1,
            "errors": 0,
            "vlm_chunks": 3 if call_count[0] == 1 else 2,
            "vlm_errors": 0,
        }

    mock_run.side_effect = side_effect

    with _mock_db_query_for_rows(rows):
        total = run_all_datasources(config, incremental=False)

    assert total["vlm_chunks"] == 5  # 3 + 2


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_run_all_ds_aggregates_vlm_errors(mock_mounted, mock_mount, mock_run, tmp_path):
    """2 Datasources mit VLM-Fehlern -> total_stats['vlm_errors'] ist Summe beider."""
    rows = _create_ds_rows_vlm(
        tmp_path,
        [
            {"id": 1, "name": "DS1", "vlm_enabled": 1},
            {"id": 2, "name": "DS2", "vlm_enabled": 1},
        ],
    )
    config = _base_config(tmp_path)

    call_count = [0]

    def side_effect(cfg, **kwargs):
        call_count[0] += 1
        return {
            "total_files": 1,
            "processed_files": 1,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 1,
            "errors": 0,
            "vlm_chunks": 0,
            "vlm_errors": 1 if call_count[0] == 1 else 2,
        }

    mock_run.side_effect = side_effect

    with _mock_db_query_for_rows(rows):
        total = run_all_datasources(config, incremental=False)

    assert total["vlm_errors"] == 3  # 1 + 2


# -- 2.9 Late Chunking — VLM-Dateien werden ausgeschlossen ---------------------


@patch("kiara.ingestion.runtime.embed_texts_late")
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.describe_image")
@patch("kiara.ingestion.runtime.load_file")
@patch("kiara.backend_selector.normalize_model_name", side_effect=lambda m: m or "")
@patch(
    "kiara.backend_selector.get_embedding_backend_for_model",
    return_value={
        "provider": "ollama",
        "url": "http://localhost:11434",
        "model": "nomic-embed-text",
    },
)
def test_vlm_file_skips_late_chunking(
    mock_backend, mock_normalize, mock_load, mock_describe, mock_vs, mock_embed_late, tmp_path
):
    """VLM-Datei mit parser=vlm -> embed_texts_late wird NICHT aufgerufen,
    store.add_with_embeddings wird NICHT aufgerufen (Standard store.add Pfad)."""
    config = _base_config(tmp_path)
    config["enrichment"] = {"late_chunking": True}
    ds_path = tmp_path / "docs"
    img_path = _make_file(ds_path, "photo.png", b"\x89PNG\x00\x00")

    # load_file gibt None zurueck (reines Bild, kein Textinhalt)
    mock_load.return_value = None
    # VLM liefert Beschreibung mit parser=vlm
    mock_describe.return_value = {
        "text": "Bildbeschreibung vom VLM-Modell",
        "metadata": {"source": img_path, "parser": "vlm", "file_type": ".png"},
    }

    mock_store = MagicMock()
    mock_vs.return_value = mock_store

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"vlm_enabled": True})

    # embed_texts_late darf NICHT aufgerufen werden (VLM-Chunks sind nicht eligible)
    mock_embed_late.assert_not_called()
    # store.add_with_embeddings darf NICHT aufgerufen werden (kein Late-Chunking-Pfad)
    mock_store.add_with_embeddings.assert_not_called()
    # Aber store.add MUSS aufgerufen werden (Standard-Embedding-Pfad)
    assert mock_store.add.call_count >= 1
