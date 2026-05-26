"""Datums- und Zeit-Hilfsfunktionen fuer KIara.

Stellt bereit:
- Deutsche Datumsformatierung ohne locale-Abhaengigkeit (Prompts)
- elapsed_ms() — Convenience fuer Dauer-Messungen mit time.monotonic()
"""

import time
from datetime import datetime

_WOCHENTAGE = [
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Freitag",
    "Samstag",
    "Sonntag",
]

_MONATE = [
    "Januar",
    "Februar",
    "Maerz",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]


def get_german_date_string() -> str:
    """Gibt einen deutschen Datums-String fuer Prompt-Injection zurueck.

    Returns:
        z.B. ``"Heute ist Montag, der 2. Maerz 2026."``
    """
    now = datetime.now()
    wochentag = _WOCHENTAGE[now.weekday()]
    monat = _MONATE[now.month - 1]
    return f"Heute ist {wochentag}, der {now.day}. {monat} {now.year}."


def elapsed_ms(t0: float) -> int:
    """Berechnet vergangene Millisekunden seit einem monotonic-Startzeitpunkt."""
    return round((time.monotonic() - t0) * 1000)
