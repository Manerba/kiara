"""Bildpruefung vor DDU, ohne Skalierung oder Dekodierung grosser Pixelpuffer."""

from pathlib import Path

from kiara.shared.constants import DDU_IMAGE_MAX_PIXELS, DDU_IMAGE_MIN_SIDE


def check_image(
    path: str,
    *,
    min_side: int = DDU_IMAGE_MIN_SIDE,
    max_pixels: int = DDU_IMAGE_MAX_PIXELS,
    detected_extension: str | None = None,
) -> tuple[str, str]:
    """Prueft Header und Frames; nur vollstaendig winzige Bilder sind Skips.

    Groesse allein beweist keine Inhaltsleere. Uebergrosse, gemischte oder
    defekte Bilder bleiben Dateifehler und erhalten keinen Leer-Abschluss.
    Pillow-Schutzgrenzen bleiben auch bei groesserem konfiguriertem Limit aktiv.
    """
    if (detected_extension or Path(path).suffix.lower()) not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:
        return "", ""

    from PIL import Image

    try:
        with Image.open(path) as image:
            frames = getattr(image, "n_frames", 1)
            small_frames = 0
            first_size = image.size
            for frame in range(frames):
                image.seek(frame)  # Auch MPO-Frames pruefen, nicht nur das Vorschaubild.
                width, height = image.size
                if width * height > max_pixels:
                    return "image_too_large", (
                        f"Bild zu gross fuer DDU: {width}x{height} Pixel "
                        f"(Frame {frame + 1}, Limit {max_pixels} Pixel)"
                    )
                if min(width, height) < min_side:
                    small_frames += 1
            # PNG prueft hier auch die Checksummen, ohne den Pixelpuffer zu laden.
            image.verify()
        if small_frames == frames:
            return "image_too_small", (
                f"Bild fuer DDU uebersprungen: {first_size[0]}x{first_size[1]} Pixel, "
                f"alle {frames} Frames unter der Mindestseitenlaenge {min_side}"
            )
        if small_frames:
            return "invalid_image", (
                f"Bild fuer DDU nicht vollstaendig verarbeitbar: {small_frames} von "
                f"{frames} Frames unter der Mindestseitenlaenge {min_side}"
            )
    except Image.DecompressionBombError as exc:
        return "image_too_large", f"Bild ueberschreitet die Pillow-Schutzgrenze: {exc}"
    except (OSError, ValueError, EOFError, SyntaxError) as exc:
        return "invalid_image", f"Bild oder Frame nicht lesbar: {exc}"
    return "", ""
