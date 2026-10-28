"""Cloud-Sperren und Anonymisierungsentscheidung fuer alle LLM-Pfade.

Zentralisiert die Konstante und Validierungslogik, die von
cloud_routing.py und backends/factory.py gemeinsam benoetigt wird.
Neutral platziert um Import-Zirkularitaeten zu vermeiden.
"""

import logging

logger = logging.getLogger(__name__)

VALID_RESTRICTION_LEVELS = frozenset({"public", "anonymize", "blocked"})


def is_cloud_blocked_by_restrictions(restriction_levels: list[str]) -> bool:
    """Prueft ob Restriction-Levels Cloud-Nutzung blockieren.

    Ein Level blockiert wenn es nicht 'public' oder 'anonymize' ist
    (d.h. 'blocked', NULL oder unbekannt -> fail-closed).

    Args:
        restriction_levels: Liste der Restriction-Levels.

    Returns:
        True wenn mindestens ein Level Cloud blockiert.
    """
    return any(level not in ("public", "anonymize") for level in restriction_levels)


def needs_anonymization(config: dict, restriction_levels: list[str] | None = None) -> bool:
    """Globaler Schalter vor Quellenregeln; Cloud-Sperren gelten unabhaengig.

    Ohne Quellenkontext (None oder []) gilt der globale Wert, standardmaessig
    True. Bei globalem Aus kann keine Quelle Anonymisierung einschalten.
    Bei globalem An erfordert mindestens eine anonymize-Quelle Maskierung;
    ausschliesslich public-Quellen benoetigen sie nicht.

    blocked/unbekannte Werte sind kein Anonymisierungsmodus. Sie muessen
    bereits beim Routing abgefangen werden und werden auch bei globalem Aus
    hier abgewiesen. Lokale/trusted Backends entscheidet der Aufrufer.
    """
    for level in restriction_levels or []:
        if level not in ("public", "anonymize"):
            raise ValueError(
                f"blocked/unbekanntes Restriction-Level '{level}' darf nicht bis zur "
                "Anonymisierungsentscheidung durchkommen — Programmfehler im Aufrufer"
            )

    if not config.get("anonymization", {}).get("enabled", True):
        return False
    return not restriction_levels or "anonymize" in restriction_levels
