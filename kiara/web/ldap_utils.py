"""Rueckwaertskompatibilitaets-Modul fuer LDAP-Utilities.

Canonical source: :mod:`kiara.shared.ldap`
"""

# Re-Exports aus kiara.shared.ldap (Single Source of Truth)
from kiara.shared.ldap import (  # noqa: F401
    authenticate_user,
    get_group_members,
    get_groups,
    get_user_groups,
    test_connection,
)
