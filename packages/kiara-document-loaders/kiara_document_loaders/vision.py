"""Vision Language Model (VLM) Bildbeschreibung."""

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Alle unterstuetzten Bildformate (Konvertierung via Pillow)
IMAGE_FORMATS = frozenset(
    {
        # Ollama-nativ (keine Konvertierung noetig)
        ".png",
        ".jpg",
        ".jpeg",
        # Pillow-nativ
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
        # Pillow >= 10.1
        ".avif",
        # Pillow + libopenjp2-7
        ".jp2",
        # pillow-heif Plugin
        ".heif",
        ".heic",
    }
)

# Formate die Ollama direkt akzeptiert — kein Umweg ueber Pillow
_OLLAMA_NATIVE = frozenset({".png", ".jpg", ".jpeg"})

MAX_VLM_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB — schuetzt vor OOM bei grossen TIFFs
_MAX_VLM_RESOLUTION = 4096  # Laengste Seite — groessere Bilder werden herunterskaliert

# HEIF/HEIC-Unterstuetzung registrieren (optional)
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

_DEFAULT_VLM_PROMPT: str = (
    "Beschreibe den Inhalt dieses Bildes detailliert auf Deutsch. "
    "Wenn es sich um eine technische Zeichnung handelt, nenne Masse, Werkstoffe, "
    "Artikelnummern und Fertigungshinweise. "
    "Wenn es sich um ein Typenschild handelt, nenne alle ablesbaren Daten. "
    "Wenn es sich um ein Foto handelt, beschreibe was zu sehen ist."
)
_vlm_prompt_de = _DEFAULT_VLM_PROMPT  # Alias fuer Rueckwaertskompatibilitaet


def _to_png_bytes(file_path: str) -> bytes | None:
    """Konvertiert ein beliebiges Bildformat via Pillow nach PNG.

    Skaliert Bilder ueber _MAX_VLM_RESOLUTION herunter und konvertiert
    nicht-RGB-Modi nach RGB. Gibt None bei Fehler zurueck.

    Args:
        file_path: Pfad zur Bilddatei.

    Returns:
        PNG-Bytes oder None bei Konvertierungsfehler.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow nicht installiert — Bildkonvertierung nicht moeglich")
        return None
    try:
        with Image.open(file_path) as img:
            if max(img.size) > _MAX_VLM_RESOLUTION:
                img.thumbnail((_MAX_VLM_RESOLUTION, _MAX_VLM_RESOLUTION), Image.LANCZOS)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as exc:
        logger.warning("Bildkonvertierung fehlgeschlagen fuer %s: %s", file_path, exc)
        return None


def describe_image(
    file_path: str,
    vlm_model: str,
    ollama_url: str,
    prompt: str = "",
    timeout: float = 300.0,
) -> dict[str, Any] | None:
    """Describe an image file using a Vision Language Model.

    Returns:
        Dict with 'text' and 'metadata' keys, or None if description fails.
        The text is a natural-language description of the image content.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in IMAGE_FORMATS:
        return None

    try:
        stat = path.stat()
    except OSError as exc:
        logger.warning("Bilddatei nicht lesbar (%s): %s", exc, file_path)
        return None

    # Dateigroessen-Limit: schuetzt vor RAM-Explosion und GPU-OOM
    if stat.st_size > MAX_VLM_IMAGE_SIZE:
        logger.warning(
            "Image too large for VLM (%d bytes, limit %d): %s",
            stat.st_size,
            MAX_VLM_IMAGE_SIZE,
            file_path,
        )
        return None

    metadata = {
        "source": str(path),
        "directory": str(path.parent),
        "filename": path.name,
        "file_type": ext,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
        "parser": "vlm",
        "vlm_model": vlm_model,
    }

    # Bilddaten vorbereiten: Ollama-native Formate direkt (wenn klein genug),
    # grosse Bilder und Nicht-Native via Pillow nach PNG (mit Downscaling)
    if ext in _OLLAMA_NATIVE:
        try:
            from PIL import Image

            pil_available = True
        except ImportError:
            pil_available = False

        if pil_available:
            try:
                with Image.open(file_path) as img:
                    if max(img.size) > _MAX_VLM_RESOLUTION:
                        image_bytes = _to_png_bytes(file_path)
                        if image_bytes is None:
                            return None
                    else:
                        with open(file_path, "rb") as f:
                            image_bytes = f.read()
            except Exception as exc:
                logger.warning("Native Bildvorbereitung fehlgeschlagen (%s): %s", exc, file_path)
                return None
        else:
            try:
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
            except OSError as exc:
                logger.warning("Native Bildlesen fehlgeschlagen (%s): %s", exc, file_path)
                return None
    else:
        image_bytes = _to_png_bytes(file_path)
        if image_bytes is None:
            return None

    image_data = base64.b64encode(image_bytes).decode()

    effective_prompt = prompt.strip() if prompt and prompt.strip() else _vlm_prompt_de

    try:
        import ollama as ollama_client

        client = ollama_client.Client(host=ollama_url, timeout=timeout)
        response = client.chat(
            model=vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": effective_prompt,
                    "images": [image_data],
                }
            ],
        )
        text = response.get("message", {}).get("content", "")
    except Exception as exc:
        logger.warning("VLM description failed for %s: %s", file_path, exc)
        return None

    if not text or not text.strip():
        return None

    # Binaer-Guard (sollte bei VLM-Output nicht vorkommen, aber defensiv)
    if "\x00" in text:
        logger.warning("Binary content in VLM response for %s, skipping", file_path)
        return None

    return {"text": text.strip(), "metadata": metadata}
