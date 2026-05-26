"""Generische Filesystem-Mount-Operationen.

Stellt gemeinsame Mount-/Unmount-Logik bereit, die von den
protokollspezifischen Modulen (smb_utils, webdav_utils) und
den Konnektoren direkt verwendet wird.

Beziehungen:
    - kiara.web.smb_utils      → re-exportiert mount_smb, unmount_smb, is_mounted
    - kiara.web.webdav_utils   → re-exportiert mount_webdav, unmount_webdav, is_mounted
    - kiara.connectors.smb_connector    → nutzt mount_smb, is_mounted (direkt)
    - kiara.connectors.webdav_connector → nutzt mount_webdav, is_mounted (direkt)
"""

from __future__ import annotations

import fcntl
import logging
import os
import shlex
import subprocess
import tempfile
import threading
from contextlib import contextmanager

from kiara.config import DEFAULT_TIMEOUTS
from kiara.shared.constants import DAVFS_SECRETS_LOCK_PATH

logger = logging.getLogger(__name__)

_mount_locks_guard = threading.Lock()
_mount_locks: dict[str, threading.Lock] = {}

_davfs_secrets_lock = threading.Lock()


@contextmanager
def _davfs_secrets_cross_process_lock():
    """Serialisiert Read-Modify-Write auf /etc/davfs2/secrets.

    Kombiniert thread-internen ``_davfs_secrets_lock`` mit fcntl.flock auf
    DAVFS_SECRETS_LOCK_PATH, damit auch parallele KIara-Prozesse
    (kiara/kiara-admin/kiara-worker) sich gegenseitig serialisieren.
    Ohne dieses Lock kann ein paralleler Mount/Unmount die Credentials
    eines anderen Mount-Points ueberschreiben (Lost-Update-Race).
    """
    with _davfs_secrets_lock:
        os.makedirs(os.path.dirname(DAVFS_SECRETS_LOCK_PATH), exist_ok=True)
        with open(DAVFS_SECRETS_LOCK_PATH, "a") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _mount_lock_for(mount_point: str) -> threading.Lock:
    """Return per-Mount-Point lock fuer Check+Action-Sequenzen."""
    normalized = os.path.abspath(mount_point)
    with _mount_locks_guard:
        lock = _mount_locks.get(normalized)
        if lock is None:
            lock = threading.Lock()
            _mount_locks[normalized] = lock
        return lock


def is_mounted(mount_point: str) -> bool:
    """Prueft ob ein Pfad ein aktiver Mount-Point ist."""
    try:
        return os.path.ismount(mount_point)
    except OSError:
        return False


def _decode_proc_mount_field(value: str) -> str:
    """Dekodiert die Escape-Sequenzen in /proc/*/mounts-Feldern."""
    return (
        value.replace("\\134", "\\")
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
    )


def mount_source(mount_point: str) -> str | None:
    """Liest die aktuell gemountete Source fuer einen Mount-Point."""
    target = os.path.abspath(mount_point)
    try:
        with open("/proc/self/mounts", encoding="utf-8") as mounts:
            for line in mounts:
                parts = line.split()
                if len(parts) < 2:
                    continue
                source = _decode_proc_mount_field(parts[0])
                mounted_at = os.path.abspath(_decode_proc_mount_field(parts[1]))
                if mounted_at == target:
                    return source
    except OSError:
        return None
    return None


def mount_source_matches(actual: str | None, expected: str) -> bool:
    """Vergleicht Mount-Sources normalisiert, aber ohne Fuzzy-Fallback."""
    if not actual:
        return False
    return actual.rstrip("/") == expected.rstrip("/")


def unmount(mount_point: str) -> tuple[bool, str]:
    """Haengt ein Dateisystem aus (generisch, unabhaengig vom Protokoll).

    Returns:
        (success, message) Tuple.
    """
    with _mount_lock_for(mount_point):
        if not is_mounted(mount_point):
            return True, "Nicht gemountet"

        try:
            result = subprocess.run(
                ["sudo", "umount", mount_point],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUTS["subprocess_umount"],
            )
            if result.returncode == 0:
                logger.info("Filesystem unmounted: %s", mount_point)
                return True, "Erfolgreich getrennt"
            return False, result.stderr.strip() or "Unmount fehlgeschlagen"
        except Exception as exc:
            return False, str(exc)


def exec_mount(
    cmd: list[str],
    mount_point: str,
    *,
    label: str = "Filesystem",
) -> tuple[bool, str]:
    """Fuehrt einen Mount-Befehl aus mit einheitlicher Fehlerbehandlung.

    Erstellt den Mount-Point falls noetig, prueft ob bereits gemountet,
    fuehrt den Befehl aus und liefert ein (success, message) Tuple.

    Args:
        cmd:         Der komplette Mount-Befehl als Liste (z.B. ["sudo", "mount", ...]).
        mount_point: Ziel-Verzeichnis.
        label:       Beschreibung fuer Log-Meldungen (z.B. "SMB share", "WebDAV").
    """
    try:
        lock = _mount_lock_for(mount_point)
        with lock:
            if os.path.islink(mount_point):
                return False, f"Mount-Point '{mount_point}' ist ein Symlink — abgelehnt"

            subprocess.run(
                ["sudo", "install", "-d", "-m", "0750", mount_point],
                capture_output=True,
                timeout=DEFAULT_TIMEOUTS["subprocess_quick"],
            )

            if os.path.islink(mount_point):
                return False, f"Mount-Point '{mount_point}' ist ein Symlink — abgelehnt"

            if is_mounted(mount_point):
                return True, "Bereits gemountet"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUTS["subprocess_mount"],
            )
            if result.returncode == 0:
                logger.info("%s mounted: %s", label, mount_point)
                return True, "Erfolgreich gemountet"
            return False, result.stderr.strip() or "Mount fehlgeschlagen"
    except subprocess.TimeoutExpired:
        return False, "Timeout beim Mount"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# SMB/CIFS-spezifische Hilfsfunktionen
# ---------------------------------------------------------------------------


def mount_smb(
    server: str,
    share: str,
    username: str,
    password: str,
    domain: str = "",
    mount_point: str = "/mnt/kiara-smb",
) -> tuple[bool, str]:
    """Mountet eine SMB/CIFS-Freigabe.

    Verwendet ein temporaeres Credentials-File statt Kommandozeile,
    um Komma-Injection in Passwoertern zu verhindern.
    """
    import tempfile

    # CR/LF in Credentials ablehnen (Zeileninjection verhindern)
    for _label, _val in [("username", username), ("password", password), ("domain", domain)]:
        if "\n" in _val or "\r" in _val:
            return False, f"SMB-Credential '{_label}' enthaelt unerlaubte Zeilenumbrueche"

    # Credentials-File: sicher gegen Komma/Sonderzeichen im Passwort
    cred_lines = f"username={username}\npassword={password}\n"
    if domain:
        cred_lines += f"domain={domain}\n"

    fd, cred_path = tempfile.mkstemp(prefix="kiara-smb-", suffix=".cred")
    try:
        os.write(fd, cred_lines.encode())
        os.close(fd)
        os.chmod(cred_path, 0o600)

        cmd = [
            "sudo",
            "mount",
            "-t",
            "cifs",
            f"//{server}/{share}",
            mount_point,
            "-o",
            f"credentials={cred_path},iocharset=utf8,file_mode=0644,dir_mode=0755,vers=3.0,seal",
        ]
        return exec_mount(cmd, mount_point, label=f"SMB //{server}/{share}")
    finally:
        try:
            os.unlink(cred_path)
        except OSError:
            pass


def unmount_smb(mount_point: str = "/mnt/kiara-smb") -> tuple[bool, str]:
    """Haengt eine SMB-Freigabe aus."""
    return unmount(mount_point)


# ---------------------------------------------------------------------------
# WebDAV/davfs2-spezifische Hilfsfunktionen
# ---------------------------------------------------------------------------


def mount_webdav(url: str, username: str, password: str, mount_point: str) -> tuple[bool, str]:
    """Mountet eine WebDAV-Freigabe via davfs2."""
    secrets_file = "/etc/davfs2/secrets"
    _ensure_davfs_secret(secrets_file, mount_point, username, password)

    success = False
    try:
        cmd = [
            "sudo",
            "mount",
            "-t",
            "davfs",
            url,
            mount_point,
            "-o",
            "uid=root,gid=root",
        ]
        success, msg = exec_mount(cmd, mount_point, label=f"WebDAV {url}")
        return success, msg
    finally:
        if not success:
            _remove_davfs_secret(secrets_file, mount_point)


def unmount_webdav(mount_point: str) -> tuple[bool, str]:
    """Haengt eine WebDAV-Freigabe aus und entfernt Credentials."""
    success, msg = unmount(mount_point)
    if success:
        _remove_davfs_secret("/etc/davfs2/secrets", mount_point)
    return success, msg


def _remove_davfs_secret(secrets_file: str, mount_point: str) -> None:
    """Entfernt Credentials fuer einen Mount-Point aus /etc/davfs2/secrets."""
    if not os.path.exists(secrets_file):
        return
    with _davfs_secrets_cross_process_lock():
        lines = []
        try:
            result = subprocess.run(
                ["sudo", "cat", secrets_file],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.splitlines(keepends=True)
            else:
                logger.warning(
                    "davfs2-Secrets Cleanup: sudo cat fehlgeschlagen: %s", result.stderr.strip()
                )
        except Exception as exc:
            logger.warning("davfs2-Secrets Cleanup: sudo cat fehlgeschlagen: %s", exc)
        if not lines:
            try:
                with open(secrets_file, encoding="utf-8") as fh:
                    lines = fh.readlines()
            except Exception as exc:
                logger.warning("davfs2-Secrets Cleanup: Datei nicht lesbar: %s", exc)
                return

        mp = _clean_davfs_field(mount_point).strip()
        filtered = [line for line in lines if _davfs_secret_mount(line) != mp]
        if len(filtered) == len(lines):
            return

        content = "".join(filtered)
        try:
            result = subprocess.run(
                ["sudo", "tee", secrets_file],
                input=content,
                text=True,
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
            logger.warning(
                "davfs2-Secrets Cleanup: sudo tee fehlgeschlagen: %s", result.stderr.strip()
            )
        except Exception as exc:
            logger.warning("davfs2-Secrets Cleanup fehlgeschlagen: %s", exc)
        # Atomares Schreiben via tempfile + os.replace: Bei Fehler bleibt die
        # Originaldatei intakt statt truncated/leer (O_TRUNC-Fallback verlor
        # bei ENOSPC/EIO alle Credentials, nicht nur die zu entfernenden).
        dir_path = os.path.dirname(secrets_file) or "/"
        tmp_path: str | None = None
        fd = -1
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".secrets.", suffix=".tmp")
            os.write(fd, content.encode())
            os.fchmod(fd, 0o600)
            os.close(fd)
            fd = -1
            os.replace(tmp_path, secrets_file)
            tmp_path = None
        except Exception as exc:
            logger.warning(
                "davfs2-Secrets Cleanup: direkter Schreib-Fallback fehlgeschlagen: %s", exc
            )
        finally:
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _clean_davfs_field(value: str) -> str:
    """Entfernt Zeilensteuerzeichen aus einem davfs2-Secrets-Feld."""
    return str(value).replace("\n", "").replace("\r", "").replace("\t", "")


def _quote_davfs_field(value: str) -> str:
    """Quotet ein Feld fuer /etc/davfs2/secrets."""
    clean = _clean_davfs_field(value)
    return '"' + clean.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _davfs_secret_mount(line: str) -> str | None:
    """Extrahiert den Mount-Point aus einer davfs2-Secrets-Zeile."""
    try:
        parts = shlex.split(line, comments=False, posix=True)
    except ValueError:
        return None
    return parts[0] if parts else None


def _ensure_davfs_secret(secrets_file: str, mount_point: str, username: str, password: str) -> None:
    """Schreibt Credentials in /etc/davfs2/secrets via sudo tee (idempotent).

    Nutzt sudo tee damit die Datei root:root bleibt (davfs2-Pflicht).
    Entfernt vorhandene Eintraege fuer den Mount-Point, fuegt den neuen an.
    """
    with _davfs_secrets_cross_process_lock():
        lines = []
        if os.path.exists(secrets_file):
            try:
                result = subprocess.run(
                    ["sudo", "cat", secrets_file],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.splitlines(keepends=True)
            except Exception:
                pass

        # Vorhandene Eintraege fuer diesen Mount-Point entfernen
        mp = _clean_davfs_field(mount_point).strip()
        filtered = [line for line in lines if _davfs_secret_mount(line) != mp]

        # Neuen Eintrag anfuegen
        filtered.append(
            f"{_quote_davfs_field(mount_point)} "
            f"{_quote_davfs_field(username)} "
            f"{_quote_davfs_field(password)}\n"
        )

        content = "".join(filtered)

        # Via sudo tee schreiben (root-Ownership bleibt erhalten)
        try:
            result = subprocess.run(
                ["sudo", "tee", secrets_file],
                input=content,
                text=True,
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sudo tee fehlgeschlagen: {result.stderr.strip()}")
            result = subprocess.run(
                ["sudo", "chmod", "0600", secrets_file],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sudo chmod fehlgeschlagen: {result.stderr.strip()}")
        except Exception as exc:
            # Fallback: direkt schreiben (z.B. als root)
            logger.warning(
                "davfs2-Secrets: sudo tee fehlgeschlagen (%s), Fallback auf direktes Schreiben", exc
            )
            os.makedirs(os.path.dirname(secrets_file), exist_ok=True)
            fd = os.open(secrets_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, content.encode())
            finally:
                os.close(fd)
