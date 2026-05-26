"""Tests fuer VLM-Downscaling in vision.py (Phase 15).

Testet die Bildskalierung: grosse Bilder werden herunterskaliert,
kleine bleiben unveraendert, Native-Formate werden korrekt behandelt.
"""

import io

import pytest

PIL = pytest.importorskip("PIL", reason="Pillow nicht installiert")
from PIL import Image

from kiara.ingestion.loaders.vision import (
    _MAX_VLM_RESOLUTION,
    _to_png_bytes,
)


def _create_image(tmp_path, name, width, height, fmt="PNG"):
    """Erstellt ein Testbild mit gegebenen Dimensionen."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    p = tmp_path / name
    img.save(str(p), format=fmt)
    return str(p)


class TestToPngBytesDownscale:
    """_to_png_bytes() skaliert grosse Bilder herunter."""

    def test_large_image_downscaled(self, tmp_path):
        """8000x6000 → laengere Seite wird auf 4096 skaliert."""
        path = _create_image(tmp_path, "big.bmp", 8000, 6000, "BMP")
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert max(img.size) == _MAX_VLM_RESOLUTION
        # Seitenverhaeltnis beibehalten
        w, h = img.size
        assert abs(w / h - 8000 / 6000) < 0.01

    def test_small_image_not_scaled(self, tmp_path):
        """1024x768 → bleibt unveraendert."""
        path = _create_image(tmp_path, "small.bmp", 1024, 768, "BMP")
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert img.size == (1024, 768)

    def test_exact_threshold_not_scaled(self, tmp_path):
        """4096x4096 → Grenzfall, wird NICHT skaliert."""
        path = _create_image(tmp_path, "exact.bmp", 4096, 4096, "BMP")
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert img.size == (4096, 4096)

    def test_tall_image_downscaled(self, tmp_path):
        """100x8000 → Hoehe wird auf 4096 skaliert."""
        path = _create_image(tmp_path, "tall.bmp", 100, 8000, "BMP")
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert max(img.size) == _MAX_VLM_RESOLUTION
        assert img.size[1] == _MAX_VLM_RESOLUTION  # Hoehe ist die laengere Seite


class TestNativeFormatDownscale:
    """Grosse Native-Bilder (JPG, PNG) werden ebenfalls herunterskaliert."""

    def test_large_jpeg_gets_downscaled(self, tmp_path):
        """Grosses JPEG (Native-Format) → wird herunterskaliert (Issue #1 Fix)."""
        path = _create_image(tmp_path, "big.jpg", 6000, 5000, "JPEG")

        # Verifiziere dass die Datei als JPEG existiert
        assert Image.open(path).size == (6000, 5000)

        # _to_png_bytes sollte herunterskalieren
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert max(img.size) == _MAX_VLM_RESOLUTION

    def test_small_jpeg_not_scaled(self, tmp_path):
        """Kleines JPEG → bleibt unveraendert."""
        path = _create_image(tmp_path, "small.jpg", 800, 600, "JPEG")
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert img.size == (800, 600)

    def test_large_png_gets_downscaled(self, tmp_path):
        """Grosses PNG (Native-Format) → wird herunterskaliert."""
        path = _create_image(tmp_path, "big.png", 5000, 5000, "PNG")
        result = _to_png_bytes(path)
        assert result is not None

        img = Image.open(io.BytesIO(result))
        assert max(img.size) == _MAX_VLM_RESOLUTION
