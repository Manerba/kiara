"""WebDAV-Mount-Utilities — Rueckwaertskompatible Re-Exports.

Die kanonische Implementierung liegt in :mod:`kiara.shared.mount`.
Dieses Modul re-exportiert die Symbole, damit bestehender Code
(``from kiara.web.webdav_utils import mount_webdav, is_mounted``) weiter
funktioniert.
"""

from kiara.shared.mount import (  # noqa: F401
    exec_mount,
    is_mounted,
    mount_webdav,
    unmount,
    unmount_webdav,
)
