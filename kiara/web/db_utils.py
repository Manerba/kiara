"""Rueckwaertskompatibilitaets-Modul fuer DB-Hilfsfunktionen.

.. deprecated::
    Dieses Modul wird nur noch fuer Rueckwaertskompatibilitaet vorgehalten.
    Neue Imports bitte direkt aus ``kiara.shared.db`` beziehen::

        from kiara.shared.db import db_query, db_transaction

Canonical source: :mod:`kiara.shared.db`
"""

# DEPRECATED: Bitte direkt aus kiara.shared.db importieren.
# Re-Exports nur fuer Rueckwaertskompatibilitaet (externe Nutzer, Plugins).
from kiara.shared.db import (  # noqa: F401
    ALLOWED_COLUMNS,
    ALLOWED_TABLES,
    db_query,
    db_transaction,
    insert_row,
    paginated_query,
    update_row,
    upsert_singleton_config,
    validate_identifier,
)
