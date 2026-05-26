"""Tests fuer kiara/ingestion/ddu_client.py.

Testet _get_semaphore() (Singleton, Defaults, Custom-Limit, Thread-Safety)
und convert_document() (Success, Params, Timeout, Fehlerarten, URL-Normalisierung, MIME).
Alle HTTP-Aufrufe werden via Modul-Namespace-Patch gemockt.
"""

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

import kiara.ingestion.ddu_client as ddu_module
from kiara.config import DEFAULT_TIMEOUTS
from kiara.ingestion.ddu_client import (
    DDUResult,
    _get_semaphore,
    check_ddu_health,
    convert_document,
)
from kiara.shared.constants import DDU_DEFAULT_CONCURRENCY as _DEFAULT_CONCURRENCY

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_semaphore():
    """Semaphore-Singleton vor und nach jedem Test zuruecksetzen."""
    ddu_module._DDU_SEMAPHORE = None
    yield
    ddu_module._DDU_SEMAPHORE = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_response(json_data=None, status_code=200, raise_for_status=None):
    """Erzeugt ein Mock-httpx-Response-Objekt."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if raise_for_status:
        resp.raise_for_status.side_effect = raise_for_status
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── 1.1 _get_semaphore() ────────────────────────────────────────────────────


def test_get_semaphore_default_limit():
    """Ohne Argument → _DEFAULT_CONCURRENCY (2)."""
    sem = _get_semaphore()
    assert sem._value == _DEFAULT_CONCURRENCY


def test_get_semaphore_custom_limit():
    """_get_semaphore(5) → sem._value == 5."""
    sem = _get_semaphore(5)
    assert sem._value == 5


def test_get_semaphore_singleton():
    """Zweiter Aufruf gibt selbes Objekt zurueck."""
    s1 = _get_semaphore(3)
    s2 = _get_semaphore(99)  # Limit wird ignoriert (Singleton)
    assert s1 is s2
    assert s2._value == 3  # Wert des ersten Aufrufs


def test_get_semaphore_thread_safety():
    """10 Threads, nur 1 Instanz."""
    results = []

    def get_sem():
        s = _get_semaphore(2)
        results.append(id(s))

    threads = [threading.Thread(target=get_sem) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1


# ── 1.2 convert_document() ──────────────────────────────────────────────────


def test_convert_success(tmp_path):
    """HTTP 200 + md_content → DDUResult('# Hello', '')."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {"md_content": "# Hello"}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult("# Hello", "")


def test_convert_params_forwarded(tmp_path):
    """table_mode, do_ocr, ocr_lang werden in data uebergeben."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {"md_content": "# OK"}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        convert_document(
            str(f), "http://ddu:5001", table_mode="fast", do_ocr=False, ocr_lang=["fr"]
        )

    _, call_kwargs = mock_httpx.post.call_args
    assert call_kwargs["data"]["table_mode"] == "fast"
    assert call_kwargs["data"]["do_ocr"] == "false"
    assert call_kwargs["data"]["ocr_lang"] == ["fr"]


def test_convert_default_timeout(tmp_path):
    """Ohne timeout → DEFAULT_TIMEOUTS['ddu_convert'] (120.0)."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {"md_content": "# OK"}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        convert_document(str(f), "http://ddu:5001")

    _, call_kwargs = mock_httpx.post.call_args
    # The timeout passed to httpx.post is 'remaining' (deadline - elapsed), which is
    # slightly less than DEFAULT_TIMEOUTS["ddu_convert"] due to execution time.
    expected = DEFAULT_TIMEOUTS["ddu_convert"]
    assert abs(call_kwargs["timeout"] - expected) < 1.0, (
        f"timeout {call_kwargs['timeout']} not within 1s of expected {expected}"
    )


def test_convert_empty_md_returns_http_error(tmp_path):
    """{'document': {}} → DDUResult(None, 'http_error')."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult(None, "http_error")


def test_convert_empty_md_content_is_success(tmp_path):
    """Leerer md_content-String ist eine erfolgreiche leere Konvertierung."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {"md_content": ""}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult("", "")


def test_convert_timeout_exception(tmp_path):
    """httpx.TimeoutException → DDUResult(None, 'timeout')."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.side_effect = httpx.TimeoutException("timed out")
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult(None, "timeout")


def test_convert_connect_error(tmp_path):
    """httpx.ConnectError → DDUResult(None, 'connection')."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.side_effect = httpx.ConnectError("refused")
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult(None, "connection")


def test_convert_http_status_error(tmp_path):
    """resp.raise_for_status() raises → DDUResult(None, 'http_error')."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    error = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_resp = _mock_response(raise_for_status=error)
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult(None, "http_error")


def test_convert_generic_exception(tmp_path):
    """RuntimeError → DDUResult(None, 'connection')."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.side_effect = RuntimeError("unexpected")
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        result = convert_document(str(f), "http://ddu:5001")

    assert result == DDUResult(None, "connection")


def test_convert_url_trailing_slash(tmp_path):
    """URL 'http://ddu:5001/' → Request an 'http://ddu:5001/v1/convert/file'."""
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {"md_content": "# OK"}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        convert_document(str(f), "http://ddu:5001/")

    call_args = mock_httpx.post.call_args
    assert call_args[0][0] == "http://ddu:5001/v1/convert/file"


def test_convert_mime_type_pdf(tmp_path):
    """.pdf Datei → MIME 'application/pdf' im files-Tuple."""
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")

    mock_resp = _mock_response({"document": {"md_content": "# PDF"}})

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError

        convert_document(str(f), "http://ddu:5001")

    _, call_kwargs = mock_httpx.post.call_args
    # files ist eine Liste von Tuples: [("files", (filename, file_obj, mime_type))]
    files_param = call_kwargs["files"]
    assert len(files_param) == 1
    field_name, (filename, _fobj, mime_type) = files_param[0]
    assert field_name == "files"
    assert filename == "report.pdf"
    assert mime_type == "application/pdf"


# ── 1.3 check_ddu_health() ──────────────────────────────────────────────────


def test_health_empty_backends():
    """Leere Backend-Liste → (False, 'Keine DDU-Backends...')."""
    ok, err = check_ddu_health([])
    assert ok is False
    assert "Keine DDU-Backends" in err


def test_health_remote_available():
    """Remote-Backend antwortet → (True, '')."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        ok, err = check_ddu_health(
            [
                {"url": "http://ddu:5001", "is_local": False},
            ]
        )

    assert ok is True
    assert err == ""


def test_health_remote_404_not_available():
    """Remote-Backend antwortet mit 404 → gilt als nicht verfuegbar (status >= 400)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.get.return_value = mock_resp
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        ok, err = check_ddu_health(
            [
                {"url": "http://ddu:5001", "is_local": False},
            ]
        )

    assert ok is False
    assert "404" in err


def test_health_remote_connect_error():
    """Remote-Backend nicht erreichbar → (False, '...Verbindung verweigert')."""
    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.get.side_effect = httpx.ConnectError("refused")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        ok, err = check_ddu_health(
            [
                {"url": "http://ddu:5001", "is_local": False},
            ]
        )

    assert ok is False
    assert "Verbindung verweigert" in err


def test_health_remote_timeout():
    """Remote-Backend Timeout → (False, '...Timeout')."""
    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.get.side_effect = httpx.TimeoutException("timed out")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        ok, err = check_ddu_health(
            [
                {"url": "http://ddu:5001", "is_local": False},
            ],
            timeout=5.0,
        )

    assert ok is False
    assert "Timeout" in err


def test_health_local_backend_ignored():
    """Lokales Backend ohne URL wird remote-only ignoriert."""
    ok, err = check_ddu_health(
        [
            {"url": "", "is_local": True},
        ]
    )

    assert ok is False
    assert "remote-only" in err


def test_health_missing_url_ignored():
    """Backend mit leerer URL wird ignoriert, ohne docling zu importieren."""
    ok, err = check_ddu_health(
        [
            {"url": "", "is_local": False},
        ]
    )

    assert ok is False
    assert "ohne URL" in err


def test_health_first_fails_second_ok():
    """Erstes Backend unerreichbar, zweites antwortet → (True, '')."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.get.side_effect = [
            httpx.ConnectError("refused"),
            mock_resp,
        ]
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        ok, err = check_ddu_health(
            [
                {"url": "http://ddu1:5001", "is_local": False},
                {"url": "http://ddu2:5001", "is_local": False},
            ]
        )

    assert ok is True
    assert err == ""


def test_health_all_fail_combined_errors():
    """Alle Backends fehlgeschlagen → Fehler aller Backends in error_message."""
    with patch("kiara.ingestion.ddu_client.httpx") as mock_httpx:
        mock_httpx.get.side_effect = httpx.ConnectError("refused")
        mock_httpx.ConnectError = httpx.ConnectError
        mock_httpx.TimeoutException = httpx.TimeoutException

        ok, err = check_ddu_health(
            [
                {"url": "http://ddu1:5001", "is_local": False},
                {"url": "http://ddu2:5001", "is_local": False},
            ]
        )

    assert ok is False
    assert "ddu1" in err
    assert "ddu2" in err
