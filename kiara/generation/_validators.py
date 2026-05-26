"""Restriction-Level-Validierung fuer Cloud-Routing.

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


def needs_anonymization(restriction_levels: list[str]) -> bool:
    """Prueft ob mindestens ein Level Anonymisierung erfordert."""
    return bool(restriction_levels) and any(l == "anonymize" for l in restriction_levels)
