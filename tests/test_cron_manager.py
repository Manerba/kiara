"""Tests fuer den Cron-Manager.

Testet:
- validate_cron_expr: Gueltige/ungueltige Ausdruecke
- read_schedules: Parsen der Cron-Datei
- write_schedule / remove_schedule: Datei-Manipulation
"""

import os
import subprocess

import pytest

from kiara.shared import cron_utils

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cron_file(tmp_path, monkeypatch):
    """Temporaere Cron-Datei.

    Patcht subprocess.run so dass sudo-Aufrufe eine FileNotFoundError
    ausloesen — damit greift der Fallback auf direktes Schreiben, auch
    wenn der Test-User kein passwordless-sudo hat.

    Patcht zusaetzlich CRON_LOCK_PATH auf tmp_path, damit der flock-
    basierte Inter-Process-Lock nicht in /var/lib/kiara/ schreibt.
    """
    path = str(tmp_path / "kiara")
    lock_path = str(tmp_path / "cron.lock")
    monkeypatch.setattr(cron_utils, "CRON_FILE", path)
    monkeypatch.setattr(cron_utils, "CRON_LOCK_PATH", lock_path)

    _real_run = subprocess.run

    def _no_sudo_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "sudo":
            # sudo chmod → ohne sudo ausfuehren (fuer Permissions-Test)
            if len(cmd) > 1 and cmd[1] == "chmod":
                return _real_run(cmd[1:], *args, **kwargs)
            raise FileNotFoundError("sudo gepatcht fuer Tests")
        return _real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _no_sudo_run)
    return path


# ---------------------------------------------------------------------------
# validate_cron_expr
# ---------------------------------------------------------------------------


class TestValidateCronExpr:
    def test_valid_every_2h(self):
        assert cron_utils.validate_cron_expr("0 */2 * * *") is True

    def test_valid_daily(self):
        assert cron_utils.validate_cron_expr("0 3 * * *") is True

    def test_valid_complex(self):
        assert cron_utils.validate_cron_expr("*/15 9-17 * * 1-5") is True

    def test_invalid_empty(self):
        assert cron_utils.validate_cron_expr("") is False

    def test_invalid_too_few_fields(self):
        assert cron_utils.validate_cron_expr("0 3 *") is False

    def test_invalid_too_many_fields(self):
        assert cron_utils.validate_cron_expr("0 3 * * * *") is False

    def test_invalid_letters(self):
        assert cron_utils.validate_cron_expr("0 abc * * *") is False

    def test_invalid_none(self):
        assert cron_utils.validate_cron_expr(None) is False

    def test_whitespace_only(self):
        assert cron_utils.validate_cron_expr("   ") is False

    def test_invalid_single_value_step_syntax(self):
        """M/N ist nach Cron-Spec ungueltig — nur */N oder M-N/S erlaubt."""
        # Minute-Feld: 5/10 ist ungueltig
        assert cron_utils.validate_cron_expr("5/10 * * * *") is False
        # Hour-Feld: 3/2 ist ungueltig
        assert cron_utils.validate_cron_expr("0 3/2 * * *") is False
        # In Liste: ein ungueltiges M/N kippt das ganze Feld
        assert cron_utils.validate_cron_expr("0,5/10 * * * *") is False

    def test_valid_step_syntax_variants(self):
        """*/N und M-N/S bleiben gueltig (Regressionsschutz fuer M/N-Fix)."""
        assert cron_utils.validate_cron_expr("*/15 * * * *") is True
        assert cron_utils.validate_cron_expr("0 0-23/2 * * *") is True
        assert cron_utils.validate_cron_expr("*/5 9-17/2 * * 1-5") is True


# ---------------------------------------------------------------------------
# read_schedules / write_schedule / remove_schedule
# ---------------------------------------------------------------------------


class TestCronFileOperations:
    @pytest.fixture(autouse=True)
    def _mock_embedding_model(self, monkeypatch):
        """Mock _get_ds_embedding_model to return a fixed model name."""
        monkeypatch.setattr(cron_utils, "_get_ds_embedding_model", lambda ds_id: "nomic-embed-text")

    def test_read_empty_file(self, cron_file):
        """Keine Datei → leeres Dict."""
        assert cron_utils.read_schedules() == {}

    def test_write_and_read_schedule(self, cron_file):
        cron_utils.write_schedule(3, "IT-Wiki", "0 */2 * * *")

        schedules = cron_utils.read_schedules()
        assert 3 in schedules
        assert schedules[3]["name"] == "IT-Wiki"
        assert schedules[3]["cron_expr"] == "0 */2 * * *"

    def test_write_preset(self, cron_file):
        cron_utils.write_schedule(5, "E-Mails", "2h")

        schedules = cron_utils.read_schedules()
        assert 5 in schedules
        assert schedules[5]["cron_expr"] == "0 */2 * * *"

    def test_update_existing_schedule(self, cron_file):
        cron_utils.write_schedule(3, "IT-Wiki", "0 */2 * * *")
        cron_utils.write_schedule(3, "IT-Wiki v2", "0 3 * * *")

        schedules = cron_utils.read_schedules()
        assert len(schedules) == 1
        assert schedules[3]["cron_expr"] == "0 3 * * *"

    def test_multiple_schedules(self, cron_file):
        cron_utils.write_schedule(1, "DS-A", "0 * * * *")
        cron_utils.write_schedule(2, "DS-B", "0 3 * * *")
        cron_utils.write_schedule(3, "DS-C", "*/30 * * * *")

        schedules = cron_utils.read_schedules()
        assert len(schedules) == 3

    def test_remove_schedule(self, cron_file):
        cron_utils.write_schedule(3, "IT-Wiki", "0 */2 * * *")
        cron_utils.write_schedule(5, "E-Mails", "0 3 * * *")
        cron_utils.remove_schedule(3)

        schedules = cron_utils.read_schedules()
        assert 3 not in schedules
        assert 5 in schedules

    def test_remove_nonexistent(self, cron_file):
        """Entfernen eines nicht-existierenden Schedules ist ein No-Op."""
        cron_utils.write_schedule(1, "Test", "0 * * * *")
        cron_utils.remove_schedule(999)
        schedules = cron_utils.read_schedules()
        assert 1 in schedules

    def test_invalid_cron_raises(self, cron_file):
        with pytest.raises(ValueError, match="Ungueltig"):
            cron_utils.write_schedule(1, "Test", "invalid")

    def test_file_permissions(self, cron_file):
        cron_utils.write_schedule(1, "Test", "0 * * * *")
        assert os.path.isfile(cron_file)
        mode = os.stat(cron_file).st_mode & 0o777
        assert mode == 0o644

    def test_cron_line_format(self, cron_file):
        cron_utils.write_schedule(7, "Mein DS", "0 3 * * *")

        with open(cron_file) as f:
            content = f.read()

        assert "# kiara-ds-7: Mein DS" in content
        assert "0 3 * * * kiara" in content
        assert "queue-ingest --ds-id 7 --triggered-by cron" in content

    def test_cron_line_contains_embedding_model(self, cron_file):
        """Cron-Zeile muss --embedding-model enthalten (Drift-Check-Kompatibilitaet)."""
        cron_utils.write_schedule(7, "Mein DS", "0 3 * * *")

        with open(cron_file) as f:
            content = f.read()

        assert "--embedding-model nomic-embed-text" in content

    def test_write_no_embedding_model_raises(self, cron_file, monkeypatch):
        """Ohne bekanntes DS- oder Default-Modell darf kein Schedule erstellt werden."""
        monkeypatch.setattr(cron_utils, "_get_ds_embedding_model", lambda ds_id: "")
        monkeypatch.setattr(cron_utils, "_get_default_embedding_model", lambda: "")

        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            cron_utils.write_schedule(1, "Test", "0 * * * *")

    def test_ds_name_special_chars(self, cron_file):
        """DS-Namen mit Sonderzeichen muessen korrekt durchgereicht werden."""
        cron_utils.write_schedule(9, "IT Doku (Alt) & Neu", "0 3 * * *")

        schedules = cron_utils.read_schedules()
        assert 9 in schedules
        assert schedules[9]["name"] == "IT Doku (Alt) & Neu"


# ---------------------------------------------------------------------------
# INTERVAL_PRESETS
# ---------------------------------------------------------------------------


class TestPresets:
    def test_all_presets_valid(self):
        for name, expr in cron_utils.INTERVAL_PRESETS.items():
            assert cron_utils.validate_cron_expr(expr), f"Preset {name} ungueltig: {expr}"


# ---------------------------------------------------------------------------
# read_schedules - Robustheit bei korrupten/unvollstaendigen Dateien
# ---------------------------------------------------------------------------


class TestReadSchedulesRobustness:
    """Testet read_schedules mit handgeschriebenen, fehlerhaften Cron-Dateien."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self._path = str(tmp_path / "kiara")
        monkeypatch.setattr(cron_utils, "CRON_FILE", self._path)

    def _write_raw(self, content):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            f.write(content)

    def test_non_numeric_ds_id(self):
        """Marker mit nicht-numerischer ID wird uebersprungen."""
        self._write_raw(
            "# kiara-ds-abc: Test\n0 * * * * kiara /opt/kiara/venv/bin/kiara queue-ingest\n"
        )
        assert cron_utils.read_schedules() == {}

    def test_marker_at_end_of_file(self):
        """Marker ohne nachfolgende Cron-Zeile wird uebersprungen."""
        self._write_raw("# kiara-ds-1: Test\n")
        assert cron_utils.read_schedules() == {}

    def test_marker_followed_by_comment(self):
        """Marker gefolgt von Kommentar statt Cron-Zeile wird uebersprungen."""
        self._write_raw("# kiara-ds-1: Test\n# noch ein Kommentar\n")
        assert cron_utils.read_schedules() == {}

    def test_cron_line_too_few_fields(self):
        """Cron-Zeile mit weniger als 6 Feldern wird uebersprungen."""
        self._write_raw("# kiara-ds-1: Test\n0 * *\n")
        assert cron_utils.read_schedules() == {}

    def test_valid_among_corrupt_entries(self):
        """Gueltige Eintraege werden trotz korrupter Nachbarn korrekt gelesen."""
        self._write_raw(
            "# Managed by KIara\n"
            "SHELL=/bin/bash\n"
            "# kiara-ds-abc: Kaputt\n"
            "0 * * * * kiara cmd\n"
            "# kiara-ds-5: Gueltig\n"
            "0 3 * * * kiara /opt/kiara/venv/bin/kiara queue-ingest --ds-id 5\n"
        )
        schedules = cron_utils.read_schedules()
        assert 5 in schedules
        assert schedules[5]["name"] == "Gueltig"
        assert len(schedules) == 1

    def test_oserror_returns_empty(self, monkeypatch):
        """Bei OSError beim Lesen wird leeres Dict zurueckgegeben."""
        orig_isfile = os.path.isfile
        monkeypatch.setattr(
            os.path,
            "isfile",
            lambda p: True if p == self._path else orig_isfile(p),
        )
        # Datei existiert nicht wirklich → FileNotFoundError (OSError-Subclass)
        assert cron_utils.read_schedules() == {}

    def test_marker_without_colon(self):
        """Marker ohne Doppelpunkt loest ValueError aus und wird uebersprungen."""
        self._write_raw("# kiara-ds-1 Test ohne Doppelpunkt\n0 * * * * kiara cmd\n")
        assert cron_utils.read_schedules() == {}


# ---------------------------------------------------------------------------
# Cross-Process-Lock (fcntl.flock)
# ---------------------------------------------------------------------------


class TestCrossProcessLock:
    """Verifiziert, dass _cross_process_lock Inter-Process-Sync liefert.

    Zweck: Bei Gunicorn/uvicorn-Workern>1 reicht threading.Lock nicht —
    ``_cross_process_lock`` muss einen fcntl.flock-basierten Lock halten,
    damit ein zweiter Prozess beim Schreiben wartet.
    """

    def test_flock_blocks_second_acquirer(self, tmp_path, monkeypatch):
        """Waehrend der Context-Manager aktiv ist, scheitert ein
        zweiter LOCK_EX|LOCK_NB-Versuch auf derselben Datei."""
        import fcntl

        lock_path = str(tmp_path / "cron.lock")
        monkeypatch.setattr(cron_utils, "CRON_LOCK_PATH", lock_path)

        with cron_utils._cross_process_lock(), open(lock_path, "a") as other_fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    other_fh.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )

        # Nach Verlassen des Context-Managers muss der Lock wieder frei sein
        with open(lock_path, "a") as other_fh:
            fcntl.flock(other_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(other_fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Re-Export-Wrapper (kiara.web.cron_manager)
# ---------------------------------------------------------------------------


class TestReExportWrapper:
    """Regressions-Schutz: Web-Layer-Imports muessen stabil bleiben."""

    def test_all_public_symbols_importable(self):
        from kiara.web import cron_manager

        for name in (
            "CRON_FILE",
            "MARKER_PREFIX",
            "KIARA_BIN",
            "CRON_USER",
            "INTERVAL_PRESETS",
            "validate_cron_expr",
            "read_schedules",
            "write_schedule",
            "remove_schedule",
        ):
            assert hasattr(cron_manager, name), f"{name} fehlt im Re-Export"

    def test_reexported_functions_are_identical(self):
        """Re-exportierte Funktionen muessen identische Objekte sein (kein Kopie)."""
        from kiara.web import cron_manager

        assert cron_manager.read_schedules is cron_utils.read_schedules
        assert cron_manager.write_schedule is cron_utils.write_schedule
        assert cron_manager.remove_schedule is cron_utils.remove_schedule
        assert cron_manager.validate_cron_expr is cron_utils.validate_cron_expr
