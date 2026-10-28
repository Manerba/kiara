"""Schema-agnostische Boolean-Helper fuer DB-Zeilen.

Hintergrund: KIara hat historisch 0/1 INTEGER-Spalten fuer boolesche Flags
(is_active, is_admin, is_public, ..., ~22 Spalten). Diese werden schrittweise
auf native PostgreSQL BOOLEAN umgestellt. Damit Python-Code unabhaengig vom
Spaltentyp arbeitet, nutzt er diese Helper:

    from kiara.shared.flags import is_true, is_false

    if is_true(row["is_active"]):
        ...
    if is_false(row.get("is_admin")):
        ...

Die Helper funktionieren mit:
- ``1`` / ``0`` (INTEGER-Spalte)
- ``True`` / ``False`` (BOOLEAN-Spalte)
- ``None``    -> False
- Beliebigen wahrheitsfaehigen Python-Werten (fallback via ``bool()``)

Zweck: Die Spalten-Migration INTEGER -> BOOLEAN kann spaeter ohne einen
zweiten Code-Sweep durchgefuehrt werden.
"""


def is_true(value) -> bool:
    """True wenn der Wert wahrheitsfaehig ist (1, True, nicht-leerer String, ...).

    ``None`` liefert False. Akzeptiert INT (0/1), BOOL, sonstige Python-Werte.
    """
    return bool(value)


def is_false(value) -> bool:
    """Invertierter ``is_true``. Nuetzlich fuer sprechende Bedingungen."""
    return not bool(value)
