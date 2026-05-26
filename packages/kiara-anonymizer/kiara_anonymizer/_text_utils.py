"""Gemeinsame Text-Hilfsfunktionen fuer die Anonymisierung.

Stellt Wortgrenzen-Pruefung und Overlap-Resolution bereit,
die vom Aho-Corasick-Automaten genutzt werden.

SICHERHEITSHINWEIS: Aenderungen an diesen Funktionen wirken sich
auf die Anonymisierungs-Pipeline aus.
"""

import logging

logger = logging.getLogger(__name__)


_WORD_CONNECTORS = frozenset("-_'–—’ʼ")


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Prueft ob der Treffer an Wortgrenzen liegt.

    Verhindert Teil-Matches wie "Mueller" in "Benzmueller" sowie
    in Bindestrich-/Apostroph-Komposita wie "Mueller-Schmidt" oder "O'Brien".

    Args:
        text: Der durchsuchte Text (lowercase).
        start: Start-Index des Treffers (inklusive).
        end: End-Index des Treffers (exklusive).
    """
    if start > 0:
        ch = text[start - 1]
        if ch.isalnum() or ch in _WORD_CONNECTORS:
            return False
    if end < len(text):
        ch = text[end]
        if ch.isalnum() or ch in _WORD_CONNECTORS:
            return False
    return True


def _resolve_overlaps(matches: list[dict]) -> list[dict]:
    """Loest ueberlappende Treffer auf: laengster Treffer gewinnt.

    Bei "Hans Mueller GmbH" darf nicht sowohl "Hans Mueller" als auch
    "Mueller GmbH" matchen — der laengste Match gewinnt.

    Matches muessen mindestens "start" und "end" Keys enthalten.
    Sortierung: laengste Treffer zuerst, bei Gleichstand frueheste Position.

    Args:
        matches: Liste von Dicts mit "start" und "end" Keys.

    Returns:
        Gefilterte Liste ohne Ueberlappungen, sortiert nach Position.
    """
    if not matches:
        return []

    matches.sort(key=lambda m: (-(m["end"] - m["start"]), m["start"]))

    resolved = []
    occupied: list[tuple[int, int]] = []
    for match in matches:
        start, end = match["start"], match["end"]
        if start >= end:
            continue
        if any(s < end and start < e for s, e in occupied):
            continue
        resolved.append(match)
        occupied.append((start, end))

    resolved.sort(key=lambda m: m["start"])
    return resolved


def open_with_fallback(path, **kwargs):
    """Oeffnet eine Textdatei mit Encoding-Fallback (UTF-8 -> CP1252 -> Latin-1).

    Deutsche ERP-Systeme exportieren CSV/JSON oft in CP1252 (Windows-1252).
    CP1252 ist ein Superset von Latin-1 fuer den Bereich 0x80-0x9F, wo Latin-1
    Steuerzeichen hat, CP1252 aber druckbare Zeichen (Euro, Anfuehrungszeichen).
    Latin-1 als letzter Fallback decodiert alles (jedes Byte ist gueltig).
    """
    # Binaer-Modus ist mit Encoding-Detection inkompatibel — fail-closed
    # statt stillschweigend an open() weiterzureichen und dort TypeError zu
    # produzieren (encoding ist bei mode='rb' nicht erlaubt).
    mode = kwargs.get("mode", "r")
    if "b" in mode:
        raise ValueError(f"open_with_fallback unterstuetzt nur Text-Modus, nicht {mode!r}")

    # utf-8-sig strippt automatisch BOM (Excel/Windows-CSV-Exports), faellt
    # ansonsten auf normales UTF-8-Verhalten zurueck — also abwaertskompatibel.
    for enc in ("utf-8-sig", "cp1252"):
        # Encoding chunkweise pruefen — verhindert RAM-Spike bei mehreren GB
        # grossen ERP-Stammdaten-Dumps. Datei wird fuer den Konsumenten danach
        # neu geoeffnet, das ist robuster als seek(0) (das bei nicht-seekbaren
        # Streams scheitert) und billiger als die ganze Datei in den Decoder
        # zu schieben.
        try:
            with open(path, encoding=enc, **kwargs) as probe:
                while probe.read(65536):
                    pass
        except UnicodeDecodeError:
            continue
        return open(path, encoding=enc, **kwargs)
    logger.warning(
        "open_with_fallback: %s konnte weder als UTF-8 noch CP1252 dekodiert werden — "
        "Fallback auf Latin-1 (Datenkorruption moeglich, jede Byte-Sequenz wird akzeptiert).",
        path,
    )
    return open(path, encoding="latin-1", **kwargs)
