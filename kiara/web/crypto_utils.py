"""Rueckwaertskompatibilitaets-Modul fuer Crypto-Utilities.

.. deprecated::
    Dieses Modul wird nur noch fuer Rueckwaertskompatibilitaet vorgehalten.
    Neue Imports bitte direkt aus ``kiara.shared.crypto`` beziehen::

        from kiara.shared.crypto import encrypt, decrypt, is_encrypted

Canonical source: :mod:`kiara.shared.crypto`
"""

# DEPRECATED: Bitte direkt aus kiara.shared.crypto importieren.
# Re-Exports nur fuer Rueckwaertskompatibilitaet (externe Nutzer, Plugins).
from kiara.shared.crypto import (  # noqa: F401
    decrypt,
    encrypt,
    get_secret_key,
    is_encrypted,
)
