"""Rueckwaertskompatibilitaets-Modul fuer DB-Verbindungsverwaltung.

Canonical source: :mod:`kiara.shared.db`

Dieses Modul re-exportiert die DB-Funktionen aus ``kiara.shared.db``,
damit bestehende Imports (``from kiara.db import ...``) weiterhin
funktionieren.
"""

# Re-Exports aus kiara.shared.db (Single Source of Truth)
from kiara.shared.db import (  # noqa: F401
    connect_query,
    connect_transaction,
    db_query,
    db_transaction,
    get_connection,
    init_db_backend,
)
