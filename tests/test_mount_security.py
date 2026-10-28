"""Security-Tests fuer F-34: davfs2 Newline-Injection Prevention.

Testet dass _ensure_davfs_secret() Steuerzeichen (\\n, \\r, \\t) aus
Username, Password und Mount-Point entfernt, bevor sie in
/etc/davfs2/secrets geschrieben werden.
"""

import os
import subprocess

import pytest

from kiara.shared import mount as mount_module

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
from kiara.shared.mount import _ensure_davfs_secret, _mount_lock_for, _remove_davfs_secret


@pytest.fixture(autouse=True)
def _isolated_mount_io(monkeypatch, tmp_path):
    """sudo-Aufrufe umleiten damit Tests auch ohne passwordless-sudo laufen.

    sudo tee/cat/chmod → FileNotFoundError, damit der Fallback in
    _ensure_davfs_secret() auf direktes Schreiben greift.
    """
    _real_run = subprocess.run

    def _no_sudo_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "sudo":
            raise FileNotFoundError("sudo gepatcht fuer Tests")
        return _real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _no_sudo_run)
    monkeypatch.setattr("kiara.shared.mount.subprocess.run", _no_sudo_run)
    monkeypatch.setattr(
        mount_module,
        "DAVFS_SECRETS_LOCK_PATH",
        str(tmp_path / "davfs-secrets.lock"),
    )


class TestDavfsNewlineInjection:
    """F-34: Steuerzeichen muessen aus davfs-Credentials gestrippt werden."""

    def test_ensure_davfs_secret_strips_newlines_from_username(self, tmp_path):
        """Username mit \\n wird bereinigt — kein Newline in Secrets-Datei."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav", "user\nadmin", "pass123")

        content = open(secrets_file).read()
        # Die Zeile muss "useradmin" enthalten (Newline entfernt)
        assert "useradmin" in content
        # Es darf keine Zeile mit nur "admin" stehen (Injection)
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        assert len(lines) == 1, f"Erwarte genau eine Zeile, habe: {lines}"

    def test_ensure_davfs_secret_strips_carriage_return(self, tmp_path):
        """Username mit \\r wird bereinigt."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav", "user\radmin", "pass123")

        content = open(secrets_file).read()
        assert "\r" not in content.replace("\n", "")
        assert "useradmin" in content

    def test_ensure_davfs_secret_strips_tabs(self, tmp_path):
        """Username mit \\t wird bereinigt."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav", "user\tadmin", "pass123")

        content = open(secrets_file).read()
        assert "\t" not in content
        assert "useradmin" in content

    def test_ensure_davfs_secret_strips_newlines_from_password(self, tmp_path):
        """Password mit \\n wird bereinigt."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav", "user", "pass\nword")

        content = open(secrets_file).read()
        assert "password" in content
        # Nur eine Datenzeile (plus evtl. leere Zeilen)
        data_lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        assert len(data_lines) == 1

    def test_ensure_davfs_secret_strips_newlines_from_mount_point(self, tmp_path):
        """Mount-Point mit \\n wird bereinigt."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav\n/etc/shadow", "user", "pass")

        content = open(secrets_file).read()
        assert "/etc/shadow" not in content.split("\n")[0] or "/mnt/dav/etc/shadow" in content
        # Nur eine Datenzeile
        data_lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        assert len(data_lines) == 1

    def test_ensure_davfs_secret_combined_injection(self, tmp_path):
        """Kombination aus \\n, \\r, \\t in allen Feldern."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(
            secrets_file,
            "/mnt/dav\r\n",
            "user\t\nname",
            "pa\r\nss\tword",
        )

        content = open(secrets_file).read()
        data_lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        assert len(data_lines) == 1
        line = data_lines[0]
        assert "\r" not in line
        assert "\t" not in line
        assert "username" in line
        assert "password" in line

    def test_ensure_davfs_secret_sets_permissions(self, tmp_path):
        """Secrets-Datei muss Modus 0600 haben."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav", "user", "pass")

        mode = os.stat(secrets_file).st_mode & 0o777
        assert mode == 0o600

    def test_ensure_davfs_secret_escapes_backslashes(self, tmp_path):
        """Backslashes in Credentials muessen fuer davfs2 geescaped werden."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/dav", r"DOMAIN\\user", r"pa\\ss")

        content = open(secrets_file).read()
        assert r'"DOMAIN\\\\user"' in content
        assert r'"pa\\\\ss"' in content

    def test_ensure_davfs_secret_quotes_mount_point(self, tmp_path):
        """Mount-Points mit Leerzeichen duerfen die davfs2-Zeile nicht zerlegen."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/kiara ds", "user", "pass")

        content = open(secrets_file).read()
        assert content.startswith('"/mnt/kiara ds" ')

    def test_remove_davfs_secret_handles_quoted_mount_point(self, tmp_path):
        """Cleanup entfernt auch Eintraege mit gequotetem Mount-Point."""
        secrets_file = str(tmp_path / "secrets")
        _ensure_davfs_secret(secrets_file, "/mnt/kiara ds", "user", "pass")
        _ensure_davfs_secret(secrets_file, "/mnt/other", "user", "pass")

        _remove_davfs_secret(secrets_file, "/mnt/kiara ds")

        content = open(secrets_file).read()
        assert "/mnt/kiara ds" not in content
        assert "/mnt/other" in content


class TestMountCriticalSection:
    """TOCTOU-Schutz fuer Mount-/Unmount-Sequenzen."""

    def test_mount_lock_normalizes_mount_point(self, tmp_path):
        base = tmp_path / "mount"
        lock1 = _mount_lock_for(str(base))
        lock2 = _mount_lock_for(str(base.parent / "." / base.name))

        assert lock1 is lock2

    def test_exec_mount_keeps_check_and_action_inside_same_lock(self, monkeypatch):
        events: list[str] = []

        class RecordingLock:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")

        def fake_run(cmd, *args, **kwargs):
            events.append("mkdir" if cmd[:3] == ["sudo", "install", "-d"] else "mount")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(mount_module, "_mount_lock_for", lambda _mount_point: RecordingLock())
        monkeypatch.setattr(
            mount_module.os.path, "islink", lambda _path: events.append("islink") or False
        )
        monkeypatch.setattr(
            mount_module, "is_mounted", lambda _path: events.append("is_mounted") or False
        )
        monkeypatch.setattr(mount_module.subprocess, "run", fake_run)

        success, _message = mount_module.exec_mount(
            ["sudo", "mount", "/dev/test", "/mnt/test"],
            "/mnt/test",
        )

        assert success is True
        assert events == ["enter", "islink", "mkdir", "islink", "is_mounted", "mount", "exit"]
