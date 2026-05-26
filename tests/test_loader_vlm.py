"""Tests fuer VLM-Integration im Loader (Phase 10).

Testet describe_image() Grundfunktion, Dateigroessen-Limit,
Fehlerbehandlung, Ollama-Aufruf-Korrektheit und IMAGE_FORMATS Konstante.
Alle Tests laufen OHNE Ollama-Server und OHNE VLM-Modell (komplett gemockt).
"""

import base64
from unittest.mock import MagicMock, patch

from kiara.ingestion.loader import (
    _IMAGE_ONLY_FORMATS,
    _OLLAMA_NATIVE,
    IMAGE_FORMATS,
    MAX_VLM_IMAGE_SIZE,
    _vlm_prompt_de,
    describe_image,
)

# -- Helpers ------------------------------------------------------------------


def _make_image(tmp_path, name, size_bytes=1024):
    """Erstelle eine temporaere gueltige Bilddatei (1x1 Pixel PNG)."""
    from PIL import Image

    f = tmp_path / name
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    img.save(str(f), format="PNG")
    # Auf gewuenschte Groesse auffuellen falls noetig (PIL liest nur den Header)
    current = f.stat().st_size
    if current < size_bytes:
        data = f.read_bytes()
        f.write_bytes(data + b"\x00" * (size_bytes - current))
    return str(f)


def _mock_ollama():
    """Erzeuge ein gemocktes ollama-Modul mit Client."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "Beschreibung des Bildes"}}
    mock_module.Client.return_value = mock_client
    return mock_module, mock_client


# -- 1.1 describe_image() — Grundfunktion ------------------------------------


def test_describe_image_returns_text_and_metadata(tmp_path):
    """Bild (.png) + gemockter Ollama-Client -> dict mit 'text' und 'metadata'."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is not None
    assert "text" in result
    assert "metadata" in result
    meta = result["metadata"]
    assert meta["parser"] == "vlm"
    assert meta["vlm_model"] == "qwen2.5-vl:7b"
    assert meta["source"] == path
    assert meta["filename"] == "test.png"
    assert meta["file_type"] == ".png"
    assert "modified" in meta
    assert "size_bytes" in meta
    assert "directory" in meta


def test_describe_image_non_image_returns_none(tmp_path):
    """Nicht-Bilddatei (.pdf, .txt) -> return None (fruehes Return)."""
    for name in ("doc.pdf", "readme.txt"):
        f = tmp_path / name
        f.write_text("dummy content")
        result = describe_image(
            str(f), vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )
        assert result is None


def test_describe_image_all_native_formats(tmp_path):
    """Fuer jedes Ollama-native Format (.png, .jpg, .jpeg) -> return ist nicht None."""
    mock_module, mock_client = _mock_ollama()

    for ext in _OLLAMA_NATIVE:
        path = _make_image(tmp_path, f"img{ext}")
        with patch.dict("sys.modules", {"ollama": mock_module}):
            result = describe_image(
                path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
            )
        assert result is not None, f"Expected result for {ext}, got None"


def test_describe_image_non_native_formats_with_pillow(tmp_path):
    """Nicht-native Formate (.webp, .bmp, .gif, ...) -> Pillow-Konvertierung -> result nicht None."""
    mock_module, mock_client = _mock_ollama()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    non_native = IMAGE_FORMATS - _OLLAMA_NATIVE
    for ext in non_native:
        path = tmp_path / f"img{ext}"
        path.write_bytes(b"\x00" * 100)
        with (
            patch.dict("sys.modules", {"ollama": mock_module}),
            patch("kiara.ingestion.loaders.vision._to_png_bytes", return_value=png_bytes),
        ):
            result = describe_image(
                str(path), vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
            )
        assert result is not None, f"Expected result for {ext}, got None"


def test_describe_image_unknown_extension_returns_none(tmp_path):
    """.svg, .psd, .raw -> return None (nicht in IMAGE_FORMATS)."""
    for ext in (".svg", ".psd", ".raw"):
        f = tmp_path / f"img{ext}"
        f.write_bytes(b"\x00" * 100)
        result = describe_image(
            str(f), vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )
        assert result is None, f"Expected None for {ext}"


# -- 1.2 describe_image() — Dateigroessen-Limit ------------------------------


def test_describe_image_file_too_large_returns_none(tmp_path):
    """Datei > MAX_VLM_IMAGE_SIZE (20 MB) -> return None. Ollama darf NICHT aufgerufen werden."""
    path = _make_image(tmp_path, "huge.png", size_bytes=MAX_VLM_IMAGE_SIZE + 1)
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is None
    mock_client.chat.assert_not_called()


def test_describe_image_file_at_limit_returns_result(tmp_path):
    """Datei == MAX_VLM_IMAGE_SIZE -> wird verarbeitet (Grenzfall)."""
    path = _make_image(tmp_path, "exact.png", size_bytes=MAX_VLM_IMAGE_SIZE)
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is not None


def test_describe_image_file_below_limit_returns_result(tmp_path):
    """Datei < MAX_VLM_IMAGE_SIZE -> wird verarbeitet."""
    path = _make_image(tmp_path, "small.png", size_bytes=1024)
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is not None


# -- 1.3 describe_image() — Fehlerbehandlung ---------------------------------


def test_describe_image_ollama_exception_returns_none(tmp_path):
    """Ollama client.chat() wirft Exception -> return None, kein Crash."""
    path = _make_image(tmp_path, "test.png")
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionError("Ollama down")
    mock_module.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is None


def test_describe_image_empty_response_returns_none(tmp_path):
    """VLM liefert leeren String -> return None."""
    path = _make_image(tmp_path, "test.png")
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": ""}}
    mock_module.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is None


def test_describe_image_whitespace_response_returns_none(tmp_path):
    """VLM liefert nur Whitespace -> return None."""
    path = _make_image(tmp_path, "test.png")
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "   \n\t  "}}
    mock_module.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is None


def test_describe_image_binary_guard_null_bytes(tmp_path):
    """VLM-Antwort enthaelt \\x00 -> return None (Binaer-Guard)."""
    path = _make_image(tmp_path, "test.png")
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "Text mit \x00 null bytes"}}
    mock_module.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is None


def test_describe_image_strips_whitespace(tmp_path):
    """VLM-Antwort mit fuehrenden/nachfolgenden Leerzeichen -> text ist gestrippt."""
    path = _make_image(tmp_path, "test.png")
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "  Bildbeschreibung  \n"}}
    mock_module.Client.return_value = mock_client

    with patch.dict("sys.modules", {"ollama": mock_module}):
        result = describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is not None
    assert result["text"] == "Bildbeschreibung"


# -- 1.4 describe_image() — Ollama-Aufruf korrekt ----------------------------


def test_describe_image_uses_chat_not_generate(tmp_path):
    """Verifiziere: client.chat() wird aufgerufen (nicht client.generate())."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        describe_image(path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434")

    mock_client.chat.assert_called_once()
    mock_client.generate.assert_not_called()


def test_describe_image_passes_base64_image(tmp_path):
    """Verifiziere: images=[base64_encoded_data] wird an chat() uebergeben."""
    path = _make_image(tmp_path, "test.png", size_bytes=100)
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        describe_image(path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434")

    call_kwargs = mock_client.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert len(messages) == 1
    msg = messages[0]
    assert "images" in msg
    assert len(msg["images"]) == 1
    # Verifiziere: ist gueltige Base64-Zeichenkette
    decoded = base64.b64decode(msg["images"][0])
    assert len(decoded) > 0


def test_describe_image_uses_correct_prompt(tmp_path):
    """Verifiziere: content == _vlm_prompt_de (wenn kein custom prompt)."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        describe_image(path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434")

    call_kwargs = mock_client.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert messages[0]["content"] == _vlm_prompt_de


def test_describe_image_uses_custom_prompt(tmp_path):
    """Custom prompt wird anstelle des Default-Prompts an Ollama uebergeben."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()
    custom = "Describe this image in English."

    with patch.dict("sys.modules", {"ollama": mock_module}):
        describe_image(
            path, vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434", prompt=custom
        )

    call_kwargs = mock_client.chat.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert messages[0]["content"] == custom


def test_describe_image_empty_prompt_uses_default(tmp_path):
    """Leerer prompt-Parameter -> Fallback auf _vlm_prompt_de."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()

    for empty in ("", "   ", None):
        mock_client.reset_mock()
        with patch.dict("sys.modules", {"ollama": mock_module}):
            describe_image(
                path,
                vlm_model="qwen2.5-vl:7b",
                ollama_url="http://localhost:11434",
                prompt=empty if empty is not None else "",
            )

        call_kwargs = mock_client.chat.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages[0]["content"] == _vlm_prompt_de, f"Failed for prompt={empty!r}"


def test_describe_image_uses_correct_model(tmp_path):
    """Verifiziere: model==vlm_model wird an chat() uebergeben."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        describe_image(path, vlm_model="my-special-vlm:13b", ollama_url="http://localhost:11434")

    call_kwargs = mock_client.chat.call_args
    model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
    assert model == "my-special-vlm:13b"


def test_describe_image_uses_correct_url(tmp_path):
    """Verifiziere: Client(host=ollama_url) wird mit der uebergebenen URL erstellt."""
    path = _make_image(tmp_path, "test.png")
    mock_module, mock_client = _mock_ollama()

    with patch.dict("sys.modules", {"ollama": mock_module}):
        describe_image(path, vlm_model="qwen2.5-vl:7b", ollama_url="http://gpu-server.local:11434")

    mock_module.Client.assert_called_once_with(host="http://gpu-server.local:11434", timeout=300.0)


# -- 1.5 IMAGE_FORMATS Konstante ---------------------------------------------


def test_image_formats_is_frozenset():
    """IMAGE_FORMATS ist ein frozenset."""
    assert isinstance(IMAGE_FORMATS, frozenset)


def test_image_formats_contains_all_expected():
    """Enthaelt alle unterstuetzten Bildformate (17 Eintraege)."""
    expected = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".ico",
        ".ppm",
        ".pcx",
        ".tga",
        ".dds",
        ".avif",
        ".jp2",
        ".heif",
        ".heic",
    }
    assert expected == IMAGE_FORMATS
    assert len(IMAGE_FORMATS) == 17


def test_image_only_formats_backward_compat():
    """_IMAGE_ONLY_FORMATS ist identisch mit IMAGE_FORMATS (backward compat)."""
    assert _IMAGE_ONLY_FORMATS is IMAGE_FORMATS


def test_ollama_native_is_subset_of_image_formats():
    """_OLLAMA_NATIVE muss Teilmenge von IMAGE_FORMATS sein."""
    assert _OLLAMA_NATIVE.issubset(IMAGE_FORMATS)
    assert {".png", ".jpg", ".jpeg"} == _OLLAMA_NATIVE


# -- 1.6 _to_png_bytes + Konvertierungspfad -----------------------------------


def test_native_format_skips_pillow(tmp_path):
    """PNG-Datei wird direkt gelesen, ohne Pillow-Konvertierung."""
    path = _make_image(tmp_path, "test.png", size_bytes=200)
    mock_module, mock_client = _mock_ollama()

    with (
        patch.dict("sys.modules", {"ollama": mock_module}),
        patch("kiara.ingestion.loaders.vision._to_png_bytes") as mock_convert,
    ):
        result = describe_image(
            str(path), vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    mock_convert.assert_not_called()
    assert result is not None


def test_non_native_format_calls_to_png_bytes(tmp_path):
    """WebP-Datei loest Pillow-Konvertierung aus."""
    path = tmp_path / "test.webp"
    path.write_bytes(b"\x00" * 200)
    mock_module, mock_client = _mock_ollama()
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    with (
        patch.dict("sys.modules", {"ollama": mock_module}),
        patch(
            "kiara.ingestion.loaders.vision._to_png_bytes", return_value=png_bytes
        ) as mock_convert,
    ):
        result = describe_image(
            str(path), vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    mock_convert.assert_called_once_with(str(path))
    assert result is not None


def test_non_native_format_pillow_failure_returns_none(tmp_path):
    """Wenn _to_png_bytes None liefert (Pillow-Fehler) -> describe_image gibt None zurueck."""
    path = tmp_path / "broken.heic"
    path.write_bytes(b"\x00" * 200)
    mock_module, mock_client = _mock_ollama()

    with (
        patch.dict("sys.modules", {"ollama": mock_module}),
        patch("kiara.ingestion.loaders.vision._to_png_bytes", return_value=None),
    ):
        result = describe_image(
            str(path), vlm_model="qwen2.5-vl:7b", ollama_url="http://localhost:11434"
        )

    assert result is None
    mock_client.chat.assert_not_called()
