"""Tests fuer DDU-Integration in der Ingestion-Pipeline (Phase 9).

Testet ddu_enabled Durchreichung, Bild-Extensions Injektion,
Pre-flight DDU Health-Check und run_all_datasources DDU-Auswertung aus der DB.
Alle Tests laufen OHNE installiertes Docling (komplett gemockt).
"""

import json
from unittest.mock import call, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

from kiara.ingestion.pipeline import run_all_datasources, run_ingestion


def _mock_db_query_for_rows(rows):
    """Erstellt einen Patch fuer DatasourceRepository.get_active mit Mock-Rows."""
    return patch("kiara.repositories.datasource.DatasourceRepository.get_active", return_value=rows)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_ddu_health_check():
    """Pre-flight DDU Health-Check im Testkontext immer bestehen lassen.

    Tests die das Pre-flight-Verhalten selbst testen, ueberschreiben
    diesen Mock mit eigenem @patch.
    """
    with patch("kiara.ingestion.ddu_client.check_ddu_health", return_value=(True, "")):
        yield


# ── Helpers ──────────────────────────────────────────────────────────────────

_REMOTE_BACKEND = {
    "url": "http://ddu:5001",
    "is_local": False,
    "model": "docling",
    "table_mode": "accurate",
    "do_ocr": True,
    "ocr_lang": ["de", "en"],
    "timeout": 120.0,
    "concurrency_limit": 2,
}


def _base_config(tmp_path):
    """Minimale Config fuer Pipeline-Tests."""
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
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "batch_size": 50,
        },
        "vectordb": {
            "path": str(vdb_path),
        },
        "ollama": {
            "url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
        },
        "graph": {"enabled": False},
    }


def _make_file(directory, name, content="Test content"):
    """Erstelle eine Datei im angegebenen Verzeichnis."""
    f = directory / name
    f.write_text(content)
    return str(f)


# ── 2.1 run_ingestion() — ddu_enabled Durchreichung ─────────────────────────


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ds_overrides_ddu_true_passes_to_load_file(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ds_overrides={'ddu_enabled': True} → load_file bekommt ddu_enabled=True."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.pdf"), "file_type": ".pdf"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True})

    # load_file wurde mit ddu_enabled=True aufgerufen
    assert mock_load_file.call_count == 1
    _, kwargs = mock_load_file.call_args
    assert kwargs.get("ddu_enabled") is True
    assert kwargs.get("ddu_backends") == [_REMOTE_BACKEND]


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ds_overrides_none_passes_ddu_false(mock_load_file, mock_vs, tmp_path):
    """ds_overrides=None → load_file bekommt ddu_enabled=False."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load_file.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)

    call_args = mock_load_file.call_args
    assert call_args == call(str(ds_path / "test.txt"), ddu_enabled=False, ddu_backends=None)


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ds_overrides_empty_passes_ddu_false(mock_load_file, mock_vs, tmp_path):
    """ds_overrides={} (ohne ddu_enabled) → load_file bekommt ddu_enabled=False."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load_file.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={})

    call_args = mock_load_file.call_args
    assert call_args == call(str(ds_path / "test.txt"), ddu_enabled=False, ddu_backends=None)


@patch("kiara.config.get_ddu_backends", return_value=[])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ds_overrides_ddu_true_no_backends_disables_ddu(
    mock_load_file, mock_vs, mock_ddu_be, tmp_path
):
    """ddu_enabled=True + get_ddu_backends returns [] → Warning + ddu_enabled=False."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load_file.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True})

    _, kwargs = mock_load_file.call_args
    assert kwargs.get("ddu_enabled") is False


# ── 2.2 run_ingestion() — Bild-Extensions Injektion ─────────────────────────


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_enabled_adds_image_types(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_enabled=True → .png/.jpg/.tiff werden in file_types aufgenommen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    # Erstelle eine Bilddatei um zu pruefen ob sie gescannt wird
    _make_file(ds_path, "scan.png")

    mock_load_file.return_value = {
        "text": "OCR text",
        "metadata": {"source": str(ds_path / "scan.png"), "file_type": ".png"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True})

    # scan.png muss verarbeitet worden sein (weil .png jetzt in file_types ist)
    assert mock_load_file.call_count == 1
    assert "scan.png" in mock_load_file.call_args[0][0]


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_disabled_no_image_types(mock_load_file, mock_vs, tmp_path):
    """ddu_enabled=False → .png/.jpg/.tiff werden NICHT aufgenommen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "scan.png")  # Bilddatei ohne DDU

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)

    # scan.png darf nicht verarbeitet werden (.png nicht in default file_types)
    mock_load_file.assert_not_called()


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_enabled_no_duplicate_types(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_enabled=True + bestehende file_types → Union (keine Duplikate)."""
    config = _base_config(tmp_path)
    # .png ist bereits in file_types
    config["datasource"]["file_types"] = [".pdf", ".txt", ".png"]
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load_file.return_value = {
        "text": "Hello",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True})

    # file_types sollte .png nur einmal enthalten
    ft = config["datasource"]["file_types"]
    assert ft.count(".png") == 1
    # Alle Bildtypen muessen enthalten sein
    for ext in [".png", ".tiff", ".tif", ".jpg", ".jpeg"]:
        assert ext in ft


# ── 2.3 run_all_datasources() — ddu_enabled aus DB ──────────────────────────


def _create_ds_rows(tmp_path, ddu_enabled=0, include_ddu_column=True):
    """Erstelle Mock-Rows fuer eine Datasource."""
    doc_path = tmp_path / "docs"
    doc_path.mkdir(exist_ok=True)

    row = {
        "id": 1,
        "name": "Test DS",
        "type": "local",
        "path": str(doc_path),
        "is_active": True,
        "connector_config": json.dumps({"path": str(doc_path)}),
        "ds_chunk_size": None,
        "ds_chunk_overlap": None,
        "ds_file_types": None,
        "ds_exclude_patterns": None,
        "ds_chunk_strategy": "auto",
    }
    if include_ddu_column:
        row["ds_ddu_enabled"] = ddu_enabled
        row["ds_vlm_enabled"] = 0
    return [row]


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_run_all_ddu_enabled_in_db(mock_mounted, mock_mount, mock_run, tmp_path):
    """DB-Row mit ds_ddu_enabled=True → ds_overrides enthaelt ddu_enabled=True."""
    rows = _create_ds_rows(tmp_path, ddu_enabled=1)
    config = _base_config(tmp_path)

    mock_run.return_value = {
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "deleted_files": 0,
        "total_chunks": 0,
        "errors": 0,
    }

    with _mock_db_query_for_rows(rows):
        run_all_datasources(config, incremental=False)

    assert mock_run.call_count == 1
    _, kwargs = mock_run.call_args
    overrides = kwargs.get("ds_overrides")
    assert overrides is not None
    assert overrides.get("ddu_enabled") is True


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_run_all_ddu_disabled_in_db(mock_mounted, mock_mount, mock_run, tmp_path):
    """DB-Row mit ds_ddu_enabled=False → ds_overrides enthaelt KEIN ddu_enabled."""
    rows = _create_ds_rows(tmp_path, ddu_enabled=0)
    config = _base_config(tmp_path)

    mock_run.return_value = {
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "deleted_files": 0,
        "total_chunks": 0,
        "errors": 0,
    }

    with _mock_db_query_for_rows(rows):
        run_all_datasources(config, incremental=False)

    assert mock_run.call_count == 1
    _, kwargs = mock_run.call_args
    overrides = kwargs.get("ds_overrides")
    # Wenn kein Override gesetzt ist, ist overrides None
    assert overrides is None or "ddu_enabled" not in overrides


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_run_all_no_ddu_column_no_crash(mock_mounted, mock_mount, mock_run, tmp_path):
    """DB-Row OHNE ds_ddu_enabled Spalte → kein Crash (try/except greift)."""
    rows = _create_ds_rows(tmp_path, include_ddu_column=False)
    config = _base_config(tmp_path)

    mock_run.return_value = {
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "deleted_files": 0,
        "total_chunks": 0,
        "errors": 0,
    }

    # Darf nicht crashen
    with _mock_db_query_for_rows(rows):
        run_all_datasources(config, incremental=False)

    assert mock_run.call_count == 1
    _, kwargs = mock_run.call_args
    overrides = kwargs.get("ds_overrides")
    assert overrides is None or "ddu_enabled" not in overrides


# ── 2.4 _process_single_file() — DDU-Stats ──────────────────────────────────


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_timeout_increments_stats(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_error=timeout → stats['ddu_timeout']==1, errors>=1, total_chunks==0."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "parser": "docling-remote",
            "ddu_error": "timeout",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_timeout", 0) >= 1
    assert stats.get("errors", 0) >= 1
    assert stats.get("total_chunks", 0) == 0


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_remote_success_tracked(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """parser=docling-remote → ddu_files=1, ddu_remote_files=1."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "Remote content",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "parser": "docling-remote",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_files", 0) == 1
    assert stats.get("ddu_remote_files", 0) == 1


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_legacy_parser_tracked(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """Legacy parser=docling wird weiter als DDU-Datei gezaehlt."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "Legacy docling content",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "parser": "docling",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_files", 0) == 1
    assert stats.get("ddu_remote_files", 0) == 0


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_connection_error_skips_file(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_error=connection → Datei wird uebersprungen (errors>=1, total_chunks==0)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "ddu_error": "connection",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_remote_errors", 0) >= 1
    assert stats.get("errors", 0) >= 1
    assert stats.get("total_chunks", 0) == 0


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_remote_connection_error_tracked(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_error=connection → ddu_remote_errors=1, Datei uebersprungen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "ddu_error": "connection",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_remote_errors", 0) >= 1
    assert stats.get("errors", 0) >= 1


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_remote_http_error_tracked(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_error=http_error → ddu_remote_errors=1, Datei uebersprungen."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "ddu_error": "http_error",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_remote_errors", 0) >= 1
    assert stats.get("errors", 0) >= 1


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_http_error_skips_file(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """ddu_error=http_error → Datei wird uebersprungen (errors>=1, total_chunks==0)."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "ddu_error": "http_error",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("ddu_remote_errors", 0) >= 1
    assert stats.get("errors", 0) >= 1
    assert stats.get("total_chunks", 0) == 0


@patch("kiara.ingestion.pipeline.run_ingestion")
@patch("kiara.shared.mount.mount_smb")
@patch("kiara.shared.mount.is_mounted", return_value=False)
def test_run_all_aggregates_ddu_stats(mock_mounted, mock_mount, mock_run, tmp_path):
    """2 Datasources: DDU-Stats korrekt summiert in total_stats."""
    doc_path = tmp_path / "docs"
    doc_path.mkdir(exist_ok=True)
    doc_path2 = tmp_path / "docs2"
    doc_path2.mkdir(exist_ok=True)

    rows = [
        {
            "id": 1,
            "name": "Test DS",
            "type": "local",
            "path": str(doc_path),
            "is_active": True,
            "connector_config": json.dumps({"path": str(doc_path)}),
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": "auto",
            "ds_ddu_enabled": True,
            "ds_vlm_enabled": False,
        },
        {
            "id": 2,
            "name": "Test DS 2",
            "type": "local",
            "path": str(doc_path2),
            "is_active": True,
            "connector_config": json.dumps({"path": str(doc_path2)}),
            "ds_chunk_size": None,
            "ds_chunk_overlap": None,
            "ds_file_types": None,
            "ds_exclude_patterns": None,
            "ds_chunk_strategy": "auto",
            "ds_ddu_enabled": True,
            "ds_vlm_enabled": False,
        },
    ]

    config = _base_config(tmp_path)

    # Zwei Aufrufe: erster mit Remote-Files, zweiter mit Fallback
    mock_run.side_effect = [
        {
            "total_files": 3,
            "processed_files": 3,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 10,
            "errors": 0,
            "ddu_files": 2,
            "ddu_remote_files": 2,
            "ddu_fallback_files": 0,
            "ddu_remote_errors": 0,
            "ddu_timeout": 0,
        },
        {
            "total_files": 2,
            "processed_files": 2,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 5,
            "errors": 1,
            "ddu_files": 1,
            "ddu_remote_files": 0,
            "ddu_fallback_files": 1,
            "ddu_remote_errors": 1,
            "ddu_timeout": 1,
        },
    ]

    with _mock_db_query_for_rows(rows):
        total = run_all_datasources(config, incremental=False)

    assert total["ddu_files"] == 3
    assert total["ddu_remote_files"] == 2
    assert total["ddu_fallback_files"] == 1
    assert total["ddu_remote_errors"] == 1
    assert total["ddu_timeout"] == 1
    assert total["datasources_processed"] == 2


# ── 2.5 run_ingestion() — DDU Pre-flight Check ───────────────────────────────


@patch("kiara.ingestion.ddu_client.check_ddu_health", return_value=(False, "Verbindung verweigert"))
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
def test_ddu_preflight_fails_raises_runtime_error(mock_ddu_be, mock_health, tmp_path):
    """DDU aktiviert + Backend nicht erreichbar → RuntimeError vor Dateiverarbeitung."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    with pytest.raises(RuntimeError, match="DDU.*erreichbar"):
        run_ingestion(
            config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
        )


@patch("kiara.ingestion.ddu_client.check_ddu_health", return_value=(True, ""))
@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_ddu_preflight_passes_continues(
    mock_load_file, mock_vs, mock_ddu_be, mock_health, tmp_path
):
    """DDU aktiviert + Backend erreichbar → Ingestion laeuft normal."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "DDU content",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "parser": "docling-remote",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    assert stats.get("errors", 0) == 0
    assert stats.get("total_chunks", 0) >= 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_no_ddu_skips_preflight(mock_load_file, mock_vs, tmp_path):
    """DDU nicht aktiviert → kein Pre-flight Check, Ingestion laeuft direkt."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.txt")

    mock_load_file.return_value = {
        "text": "Standard content",
        "metadata": {"source": str(ds_path / "test.txt"), "file_type": ".txt"},
    }

    # Kein check_ddu_health mock noetig — wird nie aufgerufen
    stats = run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)

    assert stats.get("errors", 0) == 0


# ── 2.6 Circuit-Breaker — aufeinanderfolgende Fehler ─────────────────────────


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_circuit_breaker_consecutive_ddu_errors(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """10+ aufeinanderfolgende DDU-Fehler → alle als ddu_remote_errors gezaehlt."""
    from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"

    # Mehr Dateien als der Schwellwert erstellen
    for i in range(INGESTION_MAX_CONSECUTIVE_ERRORS + 2):
        _make_file(ds_path, f"doc{i:03d}.pdf")

    # Alle Dateien liefern DDU-Fehler
    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": "irrelevant",
            "file_type": ".pdf",
            "ddu_error": "connection",
        },
    }

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )
    # DDU-Fehler werden als errors + ddu_remote_errors gezaehlt.
    # Seit dem DDU-Fail-Safe-Refactoring (kein lautloser Fallback)
    # durchlaufen DDU-Fehler nicht den Circuit-Breaker sondern werden
    # direkt uebersprungen — daher kein abort_reason.
    assert stats["errors"] == INGESTION_MAX_CONSECUTIVE_ERRORS + 2
    assert stats.get("ddu_remote_errors", 0) == INGESTION_MAX_CONSECUTIVE_ERRORS + 2


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_circuit_breaker_consecutive_embedding_errors(mock_load_file, mock_vs, tmp_path):
    """10+ aufeinanderfolgende kategoriale Embedding-Fehler → abort_reason im Stats-Dict."""
    from kiara.backend_selector import BackendUnavailableError
    from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"

    for i in range(INGESTION_MAX_CONSECUTIVE_ERRORS + 2):
        _make_file(ds_path, f"doc{i:03d}.txt")

    mock_load_file.return_value = {
        "text": "Inhalt",
        "metadata": {"source": "irrelevant", "file_type": ".txt"},
    }

    # VectorStore.add wirft einen kategorialen Fehler (Backend nicht erreichbar)
    mock_vs.return_value.add.side_effect = BackendUnavailableError(
        "Embedding-Backend nicht erreichbar"
    )

    stats = run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)
    assert "abort_reason" in stats
    assert "aufeinanderfolgende" in stats["abort_reason"]


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_circuit_breaker_resets_on_success(mock_load_file, mock_vs, tmp_path):
    """Erfolg zwischen Fehlern setzt den Zaehler zurueck — kein Abbruch."""
    from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"

    n = INGESTION_MAX_CONSECUTIVE_ERRORS - 1
    # n Fehler-Dateien + 1 Erfolg + n Fehler-Dateien = 2n+1 Dateien, kein Abbruch
    total = 2 * n + 1
    for i in range(total):
        _make_file(ds_path, f"doc{i:03d}.txt")

    call_count = 0

    def _load_side_effect(path, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "text": "Inhalt",
            "metadata": {"source": path, "file_type": ".txt"},
        }

    mock_load_file.side_effect = _load_side_effect

    add_count = 0

    def _add_side_effect(texts, metas, ids):
        nonlocal add_count
        add_count += 1
        # Jeder n-te Aufruf gelingt (der mittlere), alle anderen scheitern
        if add_count == n + 1:
            return  # Erfolg
        raise Exception("Embedding-Fehler")

    mock_vs.return_value.add.side_effect = _add_side_effect

    # Darf NICHT mit RuntimeError abbrechen, weil der Zaehler nach dem
    # einen Erfolg zurueckgesetzt wird
    stats = run_ingestion(config, embedding_model="nomic-embed-text", ds_overrides=None)

    assert stats.get("errors", 0) >= 2 * n - 2  # Mindestens einige Fehler, aber kein Abbruch


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_circuit_breaker_single_error_no_abort(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """Ein einzelner DDU-Fehler bricht die Pipeline NICHT ab."""
    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"
    _make_file(ds_path, "test.pdf")

    mock_load_file.return_value = {
        "text": "",
        "metadata": {
            "source": str(ds_path / "test.pdf"),
            "file_type": ".pdf",
            "ddu_error": "connection",
        },
    }

    # Darf nicht crashen — nur 1 Fehler, unter dem Schwellwert
    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )
    assert stats.get("errors", 0) >= 1


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_circuit_breaker_ignores_content_errors(mock_load_file, mock_vs, tmp_path):
    """Content-/Format-Fehler (unbekanntes Format) loesen den Circuit-Breaker NICHT aus.

    Regression-Test: 10+ .xlsm-Dateien die nicht geladen werden koennen
    duerfen die Pipeline nicht abbrechen, da es kein Backend-Problem ist.
    """
    from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"

    n = INGESTION_MAX_CONSECUTIVE_ERRORS + 5
    for i in range(n):
        _make_file(ds_path, f"tutorial{i:03d}.txt")

    # load_file gibt None zurueck (Datei kann nicht geladen werden)
    mock_load_file.return_value = None

    file_logs = []

    def _log_callback(entry):
        file_logs.append(entry)

    stats = run_ingestion(
        config,
        embedding_model="nomic-embed-text",
        ds_overrides=None,
        file_log_callback=_log_callback,
    )

    # Alle Dateien scheitern, aber kein Abbruch — es sind Content-Fehler
    assert "abort_reason" not in stats, (
        f"Circuit-Breaker haette bei Content-Fehlern NICHT ausloesen duerfen: {stats.get('abort_reason')}"
    )
    assert stats.get("errors", 0) == n
    assert all(not log.is_infra_error for log in file_logs)


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_no_content_files_skipped_not_errored(mock_load_file, mock_vs, tmp_path):
    """Dateien ohne extrahierbaren Inhalt (no_content) werden als 'skipped' geloggt.

    Pipeline-Integration-Test fuer den no_content-Pfad:
    - status = 'skipped' (nicht 'error')
    - error_message = 'Kein extrahierbarer Inhalt'
    - Kein Error-Increment in stats
    - Hash wird NICHT gespeichert — Datei soll beim naechsten Lauf erneut
      geprueft werden, falls sie inzwischen Content bekommen hat
      (siehe runtime._handle_loading_result no-error/no-content-Pfad)
    - consecutive_errors wird zurueckgesetzt (kein Circuit-Breaker-Einfluss)
    """
    from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"

    n = INGESTION_MAX_CONSECUTIVE_ERRORS + 5
    for i in range(n):
        _make_file(ds_path, f"empty{i:03d}.txt")

    # load_file gibt no_content-Dict zurueck (Loader lief, aber leer)
    def _load_side_effect(path, **kwargs):
        return {
            "text": "",
            "metadata": {
                "source": path,
                "file_type": ".txt",
                "parser": "standard",
                "no_content": True,
            },
        }

    mock_load_file.side_effect = _load_side_effect

    file_logs = []

    def _log_callback(entry):
        file_logs.append(entry)

    stats = run_ingestion(
        config,
        embedding_model="nomic-embed-text",
        ds_overrides=None,
        file_log_callback=_log_callback,
    )

    # Kein Abbruch
    assert "abort_reason" not in stats

    # Keine Fehler — no_content ist kein Error
    assert stats.get("errors", 0) == 0

    # no_content-Dateien werden weder als processed noch als error gezaehlt
    # (siehe runtime._handle_loading_result no-error/no-content-Pfad: nur
    # ictx.skipped++ — wird nicht in stats persistiert)
    assert stats.get("processed_files", 0) == 0
    assert stats.get("errors", 0) == 0

    # File-Logs: alle 'skipped' mit korrekter Meldung
    assert len(file_logs) == n
    assert all(log.status == "skipped" for log in file_logs)
    assert all(log.error_message == "Kein extrahierbarer Inhalt" for log in file_logs)
    assert all(not log.is_infra_error for log in file_logs)


@patch("kiara.config.get_ddu_backends", return_value=[_REMOTE_BACKEND])
@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
def test_no_content_resets_consecutive_errors(mock_load_file, mock_vs, mock_ddu_be, tmp_path):
    """no_content-Datei zwischen DDU-Fehlern setzt consecutive_errors zurueck.

    Sequenz: (N-1) DDU-Fehler → 1 no_content → (N-1) DDU-Fehler = kein Abort,
    weil die no_content-Datei den Zaehler zuruecksetzt.
    """
    from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

    config = _base_config(tmp_path)
    ds_path = tmp_path / "docs"

    n = INGESTION_MAX_CONSECUTIVE_ERRORS - 1  # z.B. 9
    # n DDU-Fehler + 1 no_content + n DDU-Fehler = 2n+1 Dateien
    total = 2 * n + 1
    for i in range(total):
        _make_file(ds_path, f"doc{i:03d}.pdf")

    call_count = 0

    def _load_side_effect(path, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == n + 1:
            # Mittlere Datei: no_content (leere Datei)
            return {
                "text": "",
                "metadata": {
                    "source": path,
                    "file_type": ".pdf",
                    "parser": "standard",
                    "no_content": True,
                },
            }
        # Alle anderen: DDU-Fehler
        return {
            "text": "",
            "metadata": {
                "source": path,
                "file_type": ".pdf",
                "ddu_error": "connection",
            },
        }

    mock_load_file.side_effect = _load_side_effect

    stats = run_ingestion(
        config, embedding_model="nomic-embed-text", ds_overrides={"ddu_enabled": True}
    )

    # Kein Abbruch — die no_content-Datei hat den Zaehler zurueckgesetzt
    assert "abort_reason" not in stats
    # 2*n DDU-Fehler
    assert stats.get("errors", 0) == 2 * n
