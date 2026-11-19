"""Regressionen fuer echte Bilddateien, Docling-404 und Backend-Recovery."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from kiara_ingestion_core.models import FileLogEntry, ProcessingContext
from PIL import Image

from kiara.config import get_ddu_backends
from kiara.exceptions import ConfigurationError
from kiara.ingestion import ddu_client, runtime
from kiara.ingestion.ddu_images import check_image
from kiara.ingestion.loader import _load_with_docling, load_file

BACKEND = {"url": "http://ddu.test:5001", "timeout": 2}
TASK_MISSING = "Task result not found. Please wait for a completion status."


def _image(tmp_path, size=(32, 16), name="bild.png"):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return str(path)


def _http(monkeypatch, status=200, body=None):
    response = httpx.Response(
        status,
        json=body if body is not None else {"document": {"md_content": "Text"}},
        request=httpx.Request("POST", BACKEND["url"] + "/v1/convert/file"),
    )
    post = Mock(return_value=response)
    monkeypatch.setattr(ddu_client.httpx, "post", post)
    return post


def _process(path, **backend_options):
    context = ProcessingContext(ddu_enabled=True, ddu_backends=[BACKEND | backend_options])
    log = FileLogEntry(path, "bild", 0, ".png")
    stats = {"errors": 0}
    result = runtime._process_single_file(
        path,
        {"ingestion": {"chunk_size": 100, "chunk_overlap": 0}},
        {},
        1,
        context,
        stats,
        file_log=log,
    )
    return result, log, stats


@pytest.mark.parametrize("size", [(1, 1), (1, 64), (3, 3)])
def test_tiny_images_skip_before_http_and_do_not_confirm_empty(tmp_path, monkeypatch, size):
    path = _image(tmp_path, size)
    post = _http(monkeypatch)
    result, log, stats = _process(path)
    post.assert_not_called()
    assert result[0] == []
    assert log.status == "skipped" and not log.is_infra_error
    assert not log.empty_content_confirmed
    assert "Mindestseitenlaenge 4" in log.error_message
    assert stats == {"errors": 0, "ddu_images_skipped": 1}


@pytest.mark.parametrize("size", [(4, 4), (32, 16)])
def test_valid_images_reach_ddu_unchanged(tmp_path, monkeypatch, size):
    path = _image(tmp_path, size)
    original = open(path, "rb").read()
    post = _http(monkeypatch)
    result = ddu_client.convert_document(path, BACKEND["url"])
    assert result.content == "Text" and not result.error
    post.assert_called_once()
    assert open(path, "rb").read() == original


def test_oversize_is_visible_file_error_without_upload_or_empty_completion(tmp_path, monkeypatch):
    path = _image(tmp_path)
    post = _http(monkeypatch)
    _, log, stats = _process(path, image_max_pixels=511)
    post.assert_not_called()
    assert log.status == "error" and not log.is_infra_error
    assert not log.empty_content_confirmed
    assert "32x16" in log.error_message and "511" in log.error_message
    assert stats["errors"] == stats["ddu_document_errors"] == 1


def test_pillow_safety_limit_is_not_disabled(tmp_path, monkeypatch):
    path = _image(tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    error, detail = check_image(path, max_pixels=10000)
    assert error == "image_too_large" and "Pillow" in detail
    assert Image.MAX_IMAGE_PIXELS == 100


def test_broken_image_is_not_an_infrastructure_failure(tmp_path, monkeypatch):
    path = tmp_path / "defekt.jpg"
    path.write_bytes(b"\xff\xd8\xffkaputter JPEG-Header")
    post = _http(monkeypatch)
    _, log, stats = _process(str(path))
    post.assert_not_called()
    assert log.status == "error" and not log.is_infra_error
    assert "nicht lesbar" in log.error_message
    assert stats["ddu_document_errors"] == 1


def test_broken_mpo_frame_is_detected_without_upload(tmp_path, monkeypatch):
    path = tmp_path / "stereo.jpg"
    frame = Image.new("RGB", (16, 16), "white")
    frame.save(path, format="MPO", save_all=True, append_images=[frame.copy()])
    data = path.read_bytes()
    second_frame = data.index(b"\xff\xd8", 2)
    path.write_bytes(data[:second_frame])
    post = _http(monkeypatch)
    result = ddu_client.convert_document(str(path), BACKEND["url"])
    post.assert_not_called()
    assert result.error == "invalid_image" and "frame" in result.detail.lower()


@pytest.mark.parametrize(
    "second_size,error", [((16, 16), ""), ((1, 1), "invalid_image"), ((33, 33), "image_too_large")]
)
def test_all_tiff_frames_are_checked(tmp_path, second_size, error):
    path = tmp_path / "seiten.tiff"
    Image.new("RGB", (16, 16)).save(
        path, save_all=True, append_images=[Image.new("RGB", second_size)]
    )
    assert check_image(str(path), max_pixels=1024)[0] == error


@pytest.mark.parametrize(
    "status,body,error,infra",
    [
        (404, {"detail": TASK_MISSING}, "document_error", False),
        (404, {"detail": "Not Found"}, "request_error", False),
        (422, {"detail": "invalid document"}, "request_error", False),
        (401, {"detail": "unauthorized"}, "request_error", False),
        (429, {"detail": "busy"}, "http_error", True),
        (503, {"error": "Docling derzeit nicht verfuegbar"}, "http_error", True),
        (500, {"detail": "Internal Server Error"}, "http_error", True),
    ],
)
def test_http_classification_and_cause_reach_file_log(
    tmp_path, monkeypatch, status, body, error, infra
):
    path = _image(tmp_path)
    post = _http(monkeypatch, status, body)
    result = ddu_client.convert_document(path, BACKEND["url"])
    assert result.error == error
    _, log, stats = _process(path)
    assert log.status == "error" and log.is_infra_error is infra
    assert f"HTTP {status}" in log.error_message
    assert next(iter(body.values())) in log.error_message
    assert stats["errors"] == 1
    assert post.call_count == 2


def test_document_errors_never_fail_over_or_start_recovery(tmp_path, monkeypatch):
    path = _image(tmp_path)
    post = _http(monkeypatch, 404, {"detail": TASK_MISSING})
    result = _load_with_docling(path, [BACKEND, {"url": "http://second.test"}])
    assert result[1] == "document_error"
    post.assert_called_once()
    _, log, _ = _process(path)
    recovery = Mock()
    monkeypatch.setattr(runtime, "_wait_for_backend", recovery)
    context = SimpleNamespace(consecutive_errors=100)
    action, retries = runtime._handle_circuit_breaker(
        context,
        path,
        log,
        "loading",
        log.error_message,
        log.is_infra_error,
        0,
        0,
    )
    assert action == runtime._CircuitBreakerResult.CONTINUE and retries == 0
    recovery.assert_not_called()


def test_503_can_fail_over_and_success_clears_previous_error(tmp_path, monkeypatch):
    path = _image(tmp_path)
    post = _http(monkeypatch)
    post.side_effect = [
        httpx.Response(503, json={"error": "busy"}, request=httpx.Request("POST", BACKEND["url"])),
        post.return_value,
    ]
    doc = load_file(path, True, [BACKEND, {"url": "http://second.test"}])
    assert doc["text"] == "Text"
    assert "ddu_error" not in doc["metadata"] and "ddu_error_detail" not in doc["metadata"]
    assert post.call_count == 2


def test_partial_success_is_not_indexed_as_complete(tmp_path, monkeypatch):
    path = _image(tmp_path)
    _http(monkeypatch, body={"status": "partial_success", "document": {"md_content": "Teiltext"}})
    result, log, stats = _process(path)
    assert result[0] == [] and log.status == "error" and not log.is_infra_error
    assert not log.empty_content_confirmed and stats["errors"] == 1


def test_image_limits_are_read_from_yaml_configuration(monkeypatch):
    monkeypatch.setattr("kiara.backend_selector.get_all_active", lambda _: [BACKEND])
    backend = get_ddu_backends({"ddu": {"image_min_side": 8, "image_max_pixels": 1024}})[0]
    assert backend["image_min_side"] == 8 and backend["image_max_pixels"] == 1024


@pytest.mark.parametrize(
    "limits",
    [
        {"image_min_side": 1},
        {"image_min_side": True},
        {"image_max_pixels": 0},
        {"image_max_pixels": "100"},
    ],
)
def test_invalid_limits_fail_closed(limits):
    with pytest.raises(ConfigurationError):
        get_ddu_backends({"ddu": limits})
