"""Tests fuer DDU-Integration im Loader (Phase 9).

Testet _load_with_docling() Remote-Failover, load_file() DDU-Pfad,
Fallback, Binary-Guard und Nicht-DDU-Formate.
"""

from unittest.mock import patch

import kiara.ingestion.loader as loader_module
from kiara.ingestion.loader import (
    _load_with_docling,
    load_file,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


_LOCAL_BACKEND = {
    "url": "",
    "is_local": True,
    "model": "docling",
    "table_mode": "accurate",
    "do_ocr": True,
    "ocr_lang": ["de", "en"],
    "timeout": 120.0,
    "concurrency_limit": None,
}
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


def _make_file(tmp_path, name, content="Dummy content for testing"):
    """Erstelle eine temporaere Datei mit gegebenem Inhalt."""
    f = tmp_path / name
    f.write_text(content)
    return str(f)


# ── 1.1 _load_with_docling() — Remote-only Basisverhalten ────────────────────


def test_load_with_docling_local_backend_ignored():
    """Backend ohne URL wird nicht lokal ausgefuehrt."""
    text, error, source = _load_with_docling("/tmp/test.pdf", ddu_backends=[_LOCAL_BACKEND])

    assert text is None
    assert error == "missing_url"
    assert source == ""


def test_load_with_docling_empty_list():
    """Leere Backend-Liste → (None, '', '')."""
    text, error, source = _load_with_docling("/tmp/test.pdf", ddu_backends=[])
    assert text is None
    assert error == ""
    assert source == ""


def test_load_with_docling_none_backends():
    """ddu_backends=None → (None, '', '')."""
    text, error, source = _load_with_docling("/tmp/test.pdf", ddu_backends=None)
    assert text is None
    assert error == ""
    assert source == ""


# ── 1.2 _load_with_docling() — Remote-Pfad ──────────────────────────────────


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_success(mock_convert):
    """Remote-Erfolg → ('# MD', '', 'docling-remote')."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content="# Remote MD", error="")

    text, remote_error, source = _load_with_docling("/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND])

    assert text == "# Remote MD"
    assert remote_error == ""
    assert source == "docling-remote"


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_timeout(mock_convert):
    """Remote-Timeout → (None, 'timeout', '') — kein Failover."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content=None, error="timeout")

    text, remote_error, source = _load_with_docling("/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND])

    assert text is None
    assert remote_error == "timeout"
    assert source == ""


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_timeout_no_failover_to_local(mock_convert):
    """Remote-Timeout → kein Failover auf lokales Backend."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content=None, error="timeout")

    text, remote_error, source = _load_with_docling(
        "/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND, _LOCAL_BACKEND]
    )

    assert text is None
    assert remote_error == "timeout"
    assert source == ""


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_error_no_local(mock_convert):
    """Remote-Connection-Fehler ohne lokales Backend → (None, 'connection', '')."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content=None, error="connection")

    text, remote_error, source = _load_with_docling("/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND])

    assert text is None
    assert remote_error == "connection"
    assert source == ""


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_error_ignores_local_backend(mock_convert):
    """Remote-Connection-Fehler → lokales Backend wird ignoriert."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content=None, error="connection")

    text, remote_error, source = _load_with_docling(
        "/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND, _LOCAL_BACKEND]
    )

    assert text is None
    assert remote_error == "connection"
    assert source == ""


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_http_error_ignores_local_backend(mock_convert):
    """Remote-HTTP-Error → lokales Backend wird ignoriert."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content=None, error="http_error")

    text, remote_error, source = _load_with_docling(
        "/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND, _LOCAL_BACKEND]
    )

    assert text is None
    assert remote_error == "http_error"
    assert source == ""


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_params_forwarded(mock_convert):
    """Backend-Dict Felder werden an convert_document uebergeben."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content="# OK", error="")

    backend = {
        "url": "http://ddu:5001",
        "is_local": False,
        "table_mode": "fast",
        "do_ocr": False,
        "ocr_lang": ["fr"],
        "timeout": 60,
        "concurrency_limit": 4,
    }
    _load_with_docling("/tmp/t.pdf", ddu_backends=[backend])

    _, kwargs = mock_convert.call_args
    assert kwargs["ddu_url"] == "http://ddu:5001"
    assert kwargs["table_mode"] == "fast"
    assert kwargs["do_ocr"] is False
    assert kwargs["ocr_lang"] == ["fr"]
    assert kwargs["timeout"] == 60
    assert kwargs["concurrency_limit"] == 4


# ── 1.3 load_file() — DDU-Pfad ─────────────────────────────────────────────


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_pdf_calls_docling(mock_docling, tmp_path):
    """ddu_enabled=True + ddu_backends + .pdf → ruft _load_with_docling auf."""
    path = _make_file(tmp_path, "test.pdf")
    mock_docling.return_value = ("# PDF Content", "", "docling-remote")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    mock_docling.assert_called_once_with(path, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == "# PDF Content"


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_png_calls_docling(mock_docling, tmp_path):
    """ddu_enabled=True + ddu_backends + .png → ruft _load_with_docling auf."""
    path = _make_file(tmp_path, "scan.png")
    mock_docling.return_value = ("OCR text from image", "", "docling-remote")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    mock_docling.assert_called_once_with(path, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == "OCR text from image"


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_docx_calls_docling(mock_docling, tmp_path):
    """ddu_enabled=True + ddu_backends + .docx → ruft _load_with_docling auf."""
    path = _make_file(tmp_path, "doc.docx")
    mock_docling.return_value = ("# Word Content\n\nParagraph text", "", "docling-remote")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    mock_docling.assert_called_once_with(path, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None


def test_ddu_no_backends_falls_back(tmp_path):
    """ddu_enabled=True + ddu_backends=None → Standard-Loader (Fallback)."""
    path = _make_file(tmp_path, "test.pdf", "Standard PDF content")
    with patch.dict(loader_module._LOADERS, {".pdf": lambda p: "Standard PDF content"}):
        result = load_file(path, ddu_enabled=True, ddu_backends=None)
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


def test_ddu_empty_backends_falls_back(tmp_path):
    """ddu_enabled=True + ddu_backends=[] → Standard-Loader (Fallback)."""
    path = _make_file(tmp_path, "test.pdf", "Standard PDF content")
    with patch.dict(loader_module._LOADERS, {".pdf": lambda p: "Standard PDF content"}):
        result = load_file(path, ddu_enabled=True, ddu_backends=[])
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_disabled_uses_standard(mock_docling, tmp_path):
    """ddu_enabled=False + .pdf → Standard-Loader."""
    path = _make_file(tmp_path, "test.pdf", "Plain text")
    with patch.dict(loader_module._LOADERS, {".pdf": lambda p: "Standard content"}):
        result = load_file(path, ddu_enabled=False)
    mock_docling.assert_not_called()
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_success_sets_parser_docling(mock_docling, tmp_path):
    """metadata['parser'] == 'docling-remote' wenn DDU erfolgreich."""
    path = _make_file(tmp_path, "test.pdf")
    mock_docling.return_value = ("# Docling Output", "", "docling-remote")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result["metadata"]["parser"] == "docling-remote"


def test_standard_loader_sets_parser_standard(tmp_path):
    """metadata['parser'] == 'standard' wenn Standard-Loader genutzt."""
    path = _make_file(tmp_path, "test.txt", "Hello World")
    result = load_file(path)
    assert result["metadata"]["parser"] == "standard"


# ── 1.4 load_file() — Fallback-Verhalten ────────────────────────────────────


@patch("kiara.ingestion.loader._load_with_docling")
def test_docling_fails_pdf_fallback_standard(mock_docling, tmp_path):
    """Docling schlaegt fehl bei .pdf → Fallback auf Standard-Loader."""
    path = _make_file(tmp_path, "test.pdf")
    mock_docling.side_effect = RuntimeError("Docling crash")

    with patch.dict(loader_module._LOADERS, {".pdf": lambda p: "Standard PDF text"}):
        result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


@patch("kiara.ingestion.loader._load_with_docling")
def test_docling_fails_png_returns_none(mock_docling, tmp_path):
    """Docling schlaegt fehl bei .png → return dict with no_content (kein Standard-Loader)."""
    path = _make_file(tmp_path, "scan.png")
    mock_docling.side_effect = RuntimeError("OCR failed")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == ""
    assert result["metadata"]["no_content"] is True


@patch("kiara.ingestion.loader._load_with_docling")
def test_docling_fails_jpg_returns_none(mock_docling, tmp_path):
    """Docling schlaegt fehl bei .jpg → return dict with no_content (kein Standard-Loader)."""
    path = _make_file(tmp_path, "photo.jpg")
    mock_docling.side_effect = RuntimeError("OCR failed")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == ""
    assert result["metadata"]["no_content"] is True


@patch("kiara.ingestion.loader._load_with_docling")
def test_docling_fails_tiff_returns_none(mock_docling, tmp_path):
    """Docling schlaegt fehl bei .tiff → return dict with no_content (kein Standard-Loader)."""
    path = _make_file(tmp_path, "scan.tiff")
    mock_docling.side_effect = RuntimeError("OCR failed")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == ""
    assert result["metadata"]["no_content"] is True


@patch("kiara.ingestion.loader._load_with_docling")
def test_docling_empty_pdf_fallback_standard(mock_docling, tmp_path):
    """Docling liefert leeren String bei .pdf → Fallback auf Standard-Loader."""
    path = _make_file(tmp_path, "empty.pdf")
    mock_docling.return_value = ("", "", "docling-remote")

    with patch.dict(loader_module._LOADERS, {".pdf": lambda p: "Standard text"}):
        result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


# ── 1.5 load_file() — Binaer-Guard ──────────────────────────────────────────


@patch("kiara.ingestion.loader._load_with_docling")
def test_binary_guard_ddu_path_high_null_ratio(mock_docling, tmp_path):
    """Docling liefert Text mit >5% Null-Bytes → return None (echte Binaerdatei)."""
    path = _make_file(tmp_path, "test.pdf")
    # >5% Null-Bytes = echte Binaerdatei
    mock_docling.return_value = ("AB\x00" * 10, "", "docling-remote")  # 33% Null-Bytes

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is None


@patch("kiara.ingestion.loader._load_with_docling")
def test_binary_guard_ddu_path_few_nulls_sanitized(mock_docling, tmp_path):
    """Docling liefert Text mit wenigen Null-Bytes → bereinigt, nicht abgelehnt."""
    path = _make_file(tmp_path, "test.pdf")
    mock_docling.return_value = ("Normaler Text mit\x00 einem Artefakt", "", "docling-remote")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert "\x00" not in result["text"]
    assert "Normaler Text mit einem Artefakt" in result["text"]


def test_binary_guard_standard_path_high_null_ratio(tmp_path):
    """Standard-Loader liefert Text mit >5% Null-Bytes → return None."""
    path = _make_file(tmp_path, "test.txt", "AB\x00" * 10)
    result = load_file(path)
    assert result is None


def test_binary_guard_standard_path_few_nulls_sanitized(tmp_path):
    """Standard-Loader liefert Text mit wenigen Null-Bytes → bereinigt."""
    path = _make_file(
        tmp_path,
        "test.txt",
        "Guter Text hier mit etwas mehr Inhalt damit der Nullbyte-Anteil\x00 niedrig bleibt",
    )
    result = load_file(path)
    assert result is not None
    assert "\x00" not in result["text"]
    assert "Guter Text hier" in result["text"]


def test_clean_text_returned_normally(tmp_path):
    """Sauberer Text ohne \\x00 → wird normal zurueckgegeben."""
    path = _make_file(tmp_path, "clean.txt", "Sauberer Text ohne Null-Bytes")
    result = load_file(path)
    assert result is not None
    assert "Sauberer Text" in result["text"]


# ── 1.6 load_file() — Nicht-DDU-Formate ─────────────────────────────────────


@patch("kiara.ingestion.loader._load_with_docling")
def test_py_not_sent_to_docling(mock_docling, tmp_path):
    """ddu_enabled=True + .py → Standard-Loader (_load_code), nicht Docling."""
    path = _make_file(tmp_path, "script.py", "def hello(): pass")
    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    mock_docling.assert_not_called()
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


@patch("kiara.ingestion.loader._load_with_docling")
def test_csv_not_sent_to_docling(mock_docling, tmp_path):
    """ddu_enabled=True + .csv → Standard-Loader (_load_csv), nicht Docling."""
    path = _make_file(tmp_path, "data.csv", "col1,col2\nval1,val2")
    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    mock_docling.assert_not_called()
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


@patch("kiara.ingestion.loader._load_with_docling")
def test_dxf_not_sent_to_docling(mock_docling, tmp_path):
    """ddu_enabled=True + .dxf → Standard-Loader, nicht Docling."""
    path = _make_file(tmp_path, "drawing.dxf", "DXF dummy content")
    # DXF-Loader braucht ezdxf — wir mocken ihn
    with patch.dict(loader_module._LOADERS, {".dxf": lambda p: "CAD content"}):
        result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    mock_docling.assert_not_called()
    assert result is not None
    assert result["metadata"]["parser"] == "standard"


# ── 1.7 load_file() — DDU-Timeout + Edge-Cases ──────────────────────────────


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_timeout_returns_dict_with_empty_text(mock_docling, tmp_path):
    """DDU-Timeout → Dict mit leerem Text + metadata['ddu_error']='timeout'.

    Nicht None zurueckgeben, damit _process_single_file() den Fehler tracken kann.
    """
    path = _make_file(tmp_path, "big.pdf")
    mock_docling.return_value = (None, "timeout", "")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == ""
    assert result["metadata"]["ddu_error"] == "timeout"


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_connection_error_no_fallback(mock_docling, tmp_path):
    """DDU-Connection-Error → Dict mit leerem Text, KEIN Fallback auf Standard-Loader."""
    path = _make_file(tmp_path, "test.pdf")
    mock_docling.return_value = (None, "connection", "")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == ""
    assert result["metadata"]["ddu_error"] == "connection"


@patch("kiara.ingestion.loader._load_with_docling")
def test_ddu_http_error_no_fallback(mock_docling, tmp_path):
    """DDU-HTTP-Error → Dict mit leerem Text, KEIN Fallback auf Standard-Loader."""
    path = _make_file(tmp_path, "test.pdf")
    mock_docling.return_value = (None, "http_error", "")

    result = load_file(path, ddu_enabled=True, ddu_backends=[_REMOTE_BACKEND])
    assert result is not None
    assert result["text"] == ""
    assert result["metadata"]["ddu_error"] == "http_error"


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_remote_empty_content_is_success(mock_convert):
    """Remote liefert leeren String → gilt als Erfolg (content is not None).

    load_file() wird dann den leeren Text als Fallback-Trigger behandeln,
    aber _load_with_docling() selbst gibt ('', '', 'docling-remote') zurueck.
    """
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.return_value = DDUResult(content="", error="")

    text, remote_error, source = _load_with_docling("/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND])

    assert text == ""
    assert remote_error == ""
    assert source == "docling-remote"


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_multi_remote_failover(mock_convert):
    """Zwei Remote-Backends: erstes Connection-Error → zweites erfolgreich."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.side_effect = [
        DDUResult(content=None, error="connection"),
        DDUResult(content="# From Backend 2", error=""),
    ]
    remote_2 = {
        **_REMOTE_BACKEND,
        "url": "http://ddu2:5001",
    }

    text, remote_error, source = _load_with_docling(
        "/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND, remote_2]
    )

    assert text == "# From Backend 2"
    assert remote_error == ""
    assert source == "docling-remote"
    assert mock_convert.call_count == 2


@patch("kiara.ingestion.ddu_client.convert_document")
def test_load_with_docling_first_remote_error_preserved(mock_convert):
    """Erster Remote-Fehler bleibt erhalten, auch wenn weitere Backends anders fehlen."""
    from kiara.ingestion.ddu_client import DDUResult

    mock_convert.side_effect = [
        DDUResult(content=None, error="connection"),
        DDUResult(content=None, error="http_error"),
    ]

    remote_2 = {**_REMOTE_BACKEND, "url": "http://ddu2:5001"}
    text, remote_error, source = _load_with_docling(
        "/tmp/t.pdf", ddu_backends=[_REMOTE_BACKEND, remote_2, _LOCAL_BACKEND]
    )

    assert text is None
    assert remote_error == "connection"  # erster Fehler bleibt
    assert source == ""
