"""Text-Hilfsfunktionen fuer Encoding-Sicherheit."""

import logging

logger = logging.getLogger(__name__)


def sanitize_utf8(text: str) -> str:
    """Stellt sicher, dass ein String sicher serialisierbar ist.

    Entfernt Lone-Surrogate (U+D800-U+DFFF) und andere nicht-serialisierbare
    Unicode-Codepoints durch einen encode/decode-Roundtrip.

    In Python 3 sind alle str-Objekte intern Unicode. Jedoch koennen durch
    fehlerhafte Dekodierung (z.B. errors='surrogateescape') Lone-Surrogate
    entstehen, die bei JSON-Serialisierung oder UTF-8-Encoding crashen.
    Dieser Roundtrip bereinigt solche Faelle.

    Regulaere Umlaute (ae=U+00E4, oe=U+00F6, ue=U+00FC, ss=U+00DF) sind
    valide Unicode-Codepoints und werden NICHT veraendert.

    Args:
        text: Eingabe-String (kann ungueltige Codepoints enthalten).
              Bei falsy Input (None, "") wird "" zurueckgegeben.

    Returns:
        Bereinigter String — garantiert JSON- und UTF-8-sicher.
    """
    if not text:
        return text or ""
    try:
        # Schneller Pfad: Wenn encode() ohne Fehler durchlaeuft,
        # ist der String bereits sauber → kein Replacement noetig.
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        cleaned = text.encode("utf-8", errors="replace").decode("utf-8")
        logger.debug(
            "sanitize_utf8: String mit ungueltigen Codepoints bereinigt (Laenge: %d)", len(text)
        )
        return cleaned
