"""Content-basierte Dateierkennung.

Erkennt ob eine Datei lesbaren Text enthaelt, unabhaengig von der Dateiendung.
Nutzt charset_normalizer fuer Encoding-Erkennung und einen Non-Printable-Ratio
Check als Binary-Guard.
"""

import os
import stat

_DEFAULT_BINARY_THRESHOLD = 0.05
_DEFAULT_CHUNK_SIZE = 8192

# BOM-Signaturen (laengere zuerst pruefen)
_BOM_UTF8 = b"\xef\xbb\xbf"
_BOM_UTF16_LE = b"\xff\xfe"
_BOM_UTF16_BE = b"\xfe\xff"


def detect_content(
    path: str, *, binary_threshold=_DEFAULT_BINARY_THRESHOLD, chunk_size=_DEFAULT_CHUNK_SIZE
) -> tuple[bool, str | None, float]:
    """Erkennt ob eine Datei lesbaren Text enthaelt.

    Args:
        path: Dateipfad.
        binary_threshold: Anteil Non-Printable-Zeichen ab dem binaer (default 0.05).
        chunk_size: Bytes zum Lesen fuer die Erkennung (default 8192).

    Returns:
        Tuple (is_text, encoding, confidence).
        - is_text: True wenn die Datei als Text erkannt wurde.
        - encoding: Erkanntes Encoding (z.B. 'utf-8', 'latin-1') oder None.
        - confidence: Konfidenz der Erkennung (0.0 - 1.0).
    """
    from charset_normalizer import from_bytes  # lazy — war vorher Modul-Ebene

    # Leere Datei
    try:
        st = os.stat(path)
    except (OSError, PermissionError):
        return (False, None, 0.0)

    if not stat.S_ISREG(st.st_mode):
        return (False, None, 0.0)

    size = st.st_size
    if size == 0:
        return (True, "utf-8", 1.0)

    # Erste Bytes lesen
    try:
        with open(path, "rb") as f:
            chunk = f.read(chunk_size)
    except (OSError, PermissionError):
        return (False, None, 0.0)

    # BOM-Check
    encoding = None
    text = None

    if chunk.startswith(_BOM_UTF8):
        encoding = "utf-8-sig"
        try:
            text = chunk.decode("utf-8-sig")
        except UnicodeDecodeError:
            return (False, None, 0.5)
    elif chunk.startswith(_BOM_UTF16_LE):
        encoding = "utf-16-le"
        try:
            raw = chunk[2:]
            raw = raw[: len(raw) - len(raw) % 2]  # gerade Byte-Anzahl
            if not raw:
                return (False, None, 0.5)
            text = raw.decode("utf-16-le")
        except UnicodeDecodeError:
            return (False, None, 0.5)
    elif chunk.startswith(_BOM_UTF16_BE):
        encoding = "utf-16-be"
        try:
            raw = chunk[2:]
            raw = raw[: len(raw) - len(raw) % 2]  # gerade Byte-Anzahl
            if not raw:
                return (False, None, 0.5)
            text = raw.decode("utf-16-be")
        except UnicodeDecodeError:
            return (False, None, 0.5)

    # Kein BOM → charset_normalizer
    confidence = 1.0
    if text is None:
        try:
            result = from_bytes(chunk)
        except Exception:
            return (False, None, 0.5)
        best = result.best()
        if best is not None:
            encoding = best.encoding
            # UTF-16/UTF-32 ohne BOM ist ein bekannter False-Positive:
            # Binaerdaten (PNG-/EXE-Header) decoden in CJK-Codepoints,
            # die isprintable() den Binary-Guard ueberlisten lassen.
            # Echte UTF-16/32-Dateien tragen praktisch immer ein BOM
            # (Behandlung weiter oben).
            enc_norm = encoding.replace("_", "-").lower() if encoding else ""
            if enc_norm.startswith("utf-16") or enc_norm.startswith("utf-32"):
                return (False, encoding, 0.0)
            confidence = max(0.0, min(1.0, 1.0 - best.chaos))
            if confidence < 0.5:
                return (False, encoding, confidence)
            text = str(best)
        else:
            # charset_normalizer findet kein Encoding → binaer
            return (False, None, 0.0)

    # Binary-Check auf dekodiertem Text
    if not text:
        return (False, None, confidence)

    non_printable = sum(
        1 for c in text if not c.isprintable() and c not in ("\n", "\r", "\t", "\f")
    )
    ratio = non_printable / len(text) if text else 1.0

    if ratio > binary_threshold:
        return (False, encoding, confidence)

    return (True, encoding, confidence)


def is_text_file(path: str) -> bool:
    """Prueft ob eine Datei lesbaren Text enthaelt.

    Wrapper um detect_content() — gibt nur den is_text-Wert zurueck.
    """
    is_text, _, _ = detect_content(path)
    return is_text
