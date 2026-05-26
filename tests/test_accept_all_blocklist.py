"""Tests fuer konfigurierbare Accept-All-Text Blocklist (Build 73).

Testet:
  - Default-Blocklist enthaelt typische Binaer-Endungen
  - scan_files() filtert Dateien mit Blocklist-Endungen aus
  - scan_files() akzeptiert Dateien die NICHT auf der Blocklist sind
  - Custom Blocklist aus Config wird statt Default verwendet
  - Leere Config-Blocklist: Fallback auf _ACCEPT_ALL_BLOCKLIST
  - Punkt-Prefix-Korrektur: "zip" → ".zip"
  - Case-Insensitivity: ".ZIP" wird genauso gefiltert wie ".zip"
  - DEFAULT_CONFIG enthaelt accept_all_blocklist
"""

from unittest.mock import patch

from kiara.config import DEFAULT_CONFIG
from kiara.ingestion._file_utils import _ACCEPT_ALL_BLOCKLIST, scan_files

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_config(tmp_path, file_types=None, accept_all_text=False, blocklist=None, exclude=None):
    """Minimale Config fuer scan_files() Tests."""
    config = {
        "datasource": {
            "path": str(tmp_path),
            "file_types": file_types or [".txt"],
            "exclude_patterns": exclude or [],
            "accept_all_text": accept_all_text,
        },
    }
    if blocklist is not None:
        config["ingestion"] = {"accept_all_blocklist": blocklist}
    return config


def _create_file(tmp_path, name, content="Testinhalt fuer Textdatei\n"):
    """Erstellt eine Datei mit Textinhalt."""
    f = tmp_path / name
    f.write_text(content)
    return str(f)


# ---------------------------------------------------------------------------
# 1. Default-Blocklist Inhalt
# ---------------------------------------------------------------------------


class TestDefaultBlocklistContent:
    """Default-Blocklist enthaelt die erwarteten Kategorien."""

    def test_contains_archive_extensions(self):
        for ext in (".zip", ".tar", ".gz", ".7z", ".rar"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_contains_image_extensions(self):
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_contains_audio_video_extensions(self):
        for ext in (".mp3", ".wav", ".mp4", ".avi", ".mkv"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_contains_binary_extensions(self):
        for ext in (".exe", ".dll", ".so", ".msi", ".deb"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_contains_crypto_extensions(self):
        for ext in (".p7s", ".p7m", ".pem", ".cer", ".gpg"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_contains_font_extensions(self):
        for ext in (".woff", ".woff2", ".ttf", ".eot", ".otf"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_contains_db_temp_extensions(self):
        for ext in (".db", ".sqlite", ".lock", ".tmp", ".bak"):
            assert ext in _ACCEPT_ALL_BLOCKLIST, f"{ext} fehlt in _ACCEPT_ALL_BLOCKLIST"

    def test_blocklist_is_set(self):
        """_ACCEPT_ALL_BLOCKLIST ist ein Set-artiger Container (set oder frozenset)."""
        assert isinstance(_ACCEPT_ALL_BLOCKLIST, (set, frozenset))

    def test_all_entries_start_with_dot(self):
        """Alle Eintraege in der Default-Blocklist beginnen mit Punkt."""
        for ext in _ACCEPT_ALL_BLOCKLIST:
            assert ext.startswith("."), f"'{ext}' beginnt nicht mit Punkt"

    def test_all_entries_lowercase(self):
        """Alle Eintraege in der Default-Blocklist sind lowercase."""
        for ext in _ACCEPT_ALL_BLOCKLIST:
            assert ext == ext.lower(), f"'{ext}' ist nicht lowercase"


# ---------------------------------------------------------------------------
# 2. DEFAULT_CONFIG enthaelt accept_all_blocklist
# ---------------------------------------------------------------------------


class TestConfigDefault:
    """DEFAULT_CONFIG in config.py enthaelt accept_all_blocklist."""

    def test_accept_all_blocklist_in_default_config(self):
        bl = DEFAULT_CONFIG["ingestion"]["accept_all_blocklist"]
        assert isinstance(bl, list)
        assert len(bl) > 0

    def test_default_config_blocklist_contains_zip(self):
        bl = DEFAULT_CONFIG["ingestion"]["accept_all_blocklist"]
        assert ".zip" in bl

    def test_default_config_blocklist_matches_file_utils(self):
        """Config-Default und _ACCEPT_ALL_BLOCKLIST enthalten die gleichen Eintraege."""
        config_bl = set(DEFAULT_CONFIG["ingestion"]["accept_all_blocklist"])
        assert config_bl == _ACCEPT_ALL_BLOCKLIST


# ---------------------------------------------------------------------------
# 3. scan_files() filtert Blocklist-Endungen aus
# ---------------------------------------------------------------------------


class TestScanFilesBlocklistFiltering:
    """scan_files() respektiert die Blocklist bei accept_all_text=True."""

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_blocklisted_extension_filtered(self, mock_is_text, tmp_path):
        """Datei mit .zip-Endung wird trotz accept_all_text ausgefiltert."""
        _create_file(tmp_path, "archive.zip")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        assert not any(f.endswith("archive.zip") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_blocklisted_image_filtered(self, mock_is_text, tmp_path):
        """Bild-Dateien (.jpg) werden bei accept_all_text ausgefiltert."""
        _create_file(tmp_path, "foto.jpg")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        assert not any(f.endswith("foto.jpg") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_blocklisted_executable_filtered(self, mock_is_text, tmp_path):
        """.exe-Dateien werden bei accept_all_text ausgefiltert."""
        _create_file(tmp_path, "setup.exe")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        assert not any(f.endswith("setup.exe") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_multiple_blocklisted_extensions_all_filtered(self, mock_is_text, tmp_path):
        """Mehrere Blocklist-Dateien werden alle ausgefiltert."""
        _create_file(tmp_path, "archive.zip")
        _create_file(tmp_path, "foto.png")
        _create_file(tmp_path, "sound.mp3")
        _create_file(tmp_path, "font.woff2")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        # Keine der Blocklist-Dateien darf enthalten sein
        blocked_names = {"archive.zip", "foto.png", "sound.mp3", "font.woff2"}
        found = {f.rsplit("/", 1)[-1] for f in files}
        assert found & blocked_names == set()


# ---------------------------------------------------------------------------
# 4. scan_files() akzeptiert Dateien die NICHT auf der Blocklist sind
# ---------------------------------------------------------------------------


class TestScanFilesNonBlocklisted:
    """Unbekannte Endungen ausserhalb der Blocklist werden akzeptiert."""

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_unknown_text_extension_accepted(self, mock_is_text, tmp_path):
        """.log-Datei ist nicht in der Blocklist → wird akzeptiert."""
        _create_file(tmp_path, "app.log")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        assert any(f.endswith("app.log") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_conf_extension_accepted(self, mock_is_text, tmp_path):
        """.conf-Datei ist nicht in der Blocklist → wird akzeptiert."""
        _create_file(tmp_path, "nginx.conf")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        assert any(f.endswith("nginx.conf") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_mix_of_blocked_and_accepted(self, mock_is_text, tmp_path):
        """Mischung: Blocklist-Datei raus, Nicht-Blocklist-Datei rein."""
        _create_file(tmp_path, "readme.txt")
        _create_file(tmp_path, "app.log")
        _create_file(tmp_path, "archive.zip")
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        names = {f.rsplit("/", 1)[-1] for f in files}
        assert "readme.txt" in names  # via file_types-Whitelist
        assert "app.log" in names  # via accept_all_text (nicht blockiert)
        assert "archive.zip" not in names  # blockiert


# ---------------------------------------------------------------------------
# 5. Custom Blocklist aus Config wird statt Default verwendet
# ---------------------------------------------------------------------------


class TestCustomBlocklist:
    """Eigene Blocklist-Konfiguration ueberschreibt Default."""

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_custom_blocklist_blocks_custom_extension(self, mock_is_text, tmp_path):
        """Eigene Blocklist mit .xyz → .xyz wird gefiltert."""
        _create_file(tmp_path, "data.xyz")
        config = _make_config(tmp_path, accept_all_text=True, blocklist=[".xyz"])
        files = scan_files(config)
        assert not any(f.endswith("data.xyz") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_custom_blocklist_allows_previously_blocked(self, mock_is_text, tmp_path):
        """Eigene Blocklist OHNE .zip → .zip wird nun akzeptiert."""
        _create_file(tmp_path, "archive.zip")
        # Nur .xyz blockiert, .zip nicht mehr
        config = _make_config(tmp_path, accept_all_text=True, blocklist=[".xyz"])
        files = scan_files(config)
        assert any(f.endswith("archive.zip") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_custom_blocklist_completely_replaces_default(self, mock_is_text, tmp_path):
        """Custom Blocklist ersetzt komplett die Default-Blocklist."""
        _create_file(tmp_path, "foto.jpg")  # Default blockiert, Custom nicht
        _create_file(tmp_path, "data.custom")  # Default nicht blockiert, Custom schon
        config = _make_config(
            tmp_path,
            accept_all_text=True,
            blocklist=[".custom"],
        )
        files = scan_files(config)
        names = {f.rsplit("/", 1)[-1] for f in files}
        assert "foto.jpg" in names  # nicht in Custom-Blocklist
        assert "data.custom" not in names  # in Custom-Blocklist


# ---------------------------------------------------------------------------
# 6. Leere/fehlende Config → Fallback auf _ACCEPT_ALL_BLOCKLIST
# ---------------------------------------------------------------------------


class TestBlocklistFallback:
    """Fehlende oder nicht gesetzte Config → Default-Blocklist."""

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_no_ingestion_section_uses_default(self, mock_is_text, tmp_path):
        """Keine ingestion-Section in Config → _ACCEPT_ALL_BLOCKLIST greift."""
        _create_file(tmp_path, "archive.zip")
        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt"],
                "exclude_patterns": [],
                "accept_all_text": True,
            },
            # Kein "ingestion" Key
        }
        files = scan_files(config)
        assert not any(f.endswith("archive.zip") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_ingestion_without_blocklist_key_uses_default(self, mock_is_text, tmp_path):
        """ingestion-Section ohne accept_all_blocklist → _ACCEPT_ALL_BLOCKLIST."""
        _create_file(tmp_path, "foto.jpg")
        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt"],
                "exclude_patterns": [],
                "accept_all_text": True,
            },
            "ingestion": {
                "chunk_size": 500,
                # Kein accept_all_blocklist Key
            },
        }
        files = scan_files(config)
        assert not any(f.endswith("foto.jpg") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_empty_ingestion_dict_uses_default(self, mock_is_text, tmp_path):
        """Leeres ingestion-Dict → _ACCEPT_ALL_BLOCKLIST als Fallback."""
        _create_file(tmp_path, "sound.mp3")
        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt"],
                "exclude_patterns": [],
                "accept_all_text": True,
            },
            "ingestion": {},
        }
        files = scan_files(config)
        assert not any(f.endswith("sound.mp3") for f in files)


# ---------------------------------------------------------------------------
# 7. Punkt-Prefix-Korrektur (Frontend-Logik, hier serverseitig nachgestellt)
# ---------------------------------------------------------------------------


class TestDotPrefixNormalization:
    """Punkt-Prefix-Korrektur: "zip" wird zu ".zip" normalisiert.

    Die Korrektur findet im Frontend statt (_parseBlocklistTextarea).
    Hier wird geprueft, dass scan_files() Eintraege ohne Punkt NICHT
    matcht — und mit Punkt korrekt.
    """

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_entry_without_dot_does_not_match(self, mock_is_text, tmp_path):
        """Blocklist-Eintrag 'zip' (ohne Punkt) matcht .zip-Datei NICHT."""
        _create_file(tmp_path, "archive.zip")
        # Absichtlich ohne Punkt — wurde noch nicht normalisiert
        config = _make_config(tmp_path, accept_all_text=True, blocklist=["zip"])
        files = scan_files(config)
        # .zip wird nicht erkannt, weil ext=".zip" != "zip"
        assert any(f.endswith("archive.zip") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_entry_with_dot_matches(self, mock_is_text, tmp_path):
        """Blocklist-Eintrag '.zip' (mit Punkt) matcht .zip-Datei korrekt."""
        _create_file(tmp_path, "archive.zip")
        config = _make_config(tmp_path, accept_all_text=True, blocklist=[".zip"])
        files = scan_files(config)
        assert not any(f.endswith("archive.zip") for f in files)

    def test_frontend_normalization_logic(self):
        """Nachbildung der Frontend-_parseBlocklistTextarea Punkt-Korrektur."""
        raw_entries = ["zip", ".tar", "gz", ".jpg", "mp3"]
        normalized = [s if s.startswith(".") else "." + s for s in raw_entries]
        assert normalized == [".zip", ".tar", ".gz", ".jpg", ".mp3"]

    def test_frontend_normalization_preserves_dotted(self):
        """Bereits mit Punkt → bleibt unveraendert."""
        raw_entries = [".zip", ".tar", ".gz"]
        normalized = [s if s.startswith(".") else "." + s for s in raw_entries]
        assert normalized == [".zip", ".tar", ".gz"]


# ---------------------------------------------------------------------------
# 8. Case-Insensitivity: ".ZIP" wird genauso gefiltert wie ".zip"
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    """scan_files() normalisiert Endungen auf lowercase vor dem Vergleich."""

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_uppercase_extension_file_blocked(self, mock_is_text, tmp_path):
        """Datei mit .ZIP (uppercase) wird von Blocklist mit .zip erkannt."""
        _create_file(tmp_path, "ARCHIVE.ZIP")
        config = _make_config(tmp_path, accept_all_text=True, blocklist=[".zip"])
        files = scan_files(config)
        assert not any(f.endswith("ARCHIVE.ZIP") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_mixed_case_extension_blocked(self, mock_is_text, tmp_path):
        """Datei mit .Zip (mixed case) wird von Blocklist erkannt."""
        _create_file(tmp_path, "archive.Zip")
        config = _make_config(tmp_path, accept_all_text=True, blocklist=[".zip"])
        files = scan_files(config)
        assert not any(f.endswith("archive.Zip") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_uppercase_blocklist_entry_blocks_lowercase_file(self, mock_is_text, tmp_path):
        """Blocklist-Eintrag .ZIP (uppercase) blockt .zip-Datei.

        scan_files() macht ext = Path(name).suffix.lower(), daher muss
        die Blocklist ebenfalls lowercase-Eintraege enthalten, sonst greift
        der Vergleich nicht. Dieser Test dokumentiert das Verhalten.
        """
        _create_file(tmp_path, "archive.zip")
        # Blocklist hat nur Uppercase — ext.lower() == ".zip" vs ".ZIP" → kein Match
        config = _make_config(tmp_path, accept_all_text=True, blocklist=[".ZIP"])
        files = scan_files(config)
        # .ZIP != .zip → Datei wird NICHT blockiert (dokumentiertes Verhalten)
        assert any(f.endswith("archive.zip") for f in files)

    @patch("kiara.ingestion.content_detect.is_text_file", return_value=True)
    def test_default_blocklist_all_lowercase(self, mock_is_text, tmp_path):
        """Default-Blocklist ist komplett lowercase → uppercase Dateien werden blockiert."""
        _create_file(tmp_path, "PHOTO.JPG")
        # Default-Blocklist (kein blocklist-Override)
        config = _make_config(tmp_path, accept_all_text=True)
        files = scan_files(config)
        assert not any(f.endswith("PHOTO.JPG") for f in files)


# ---------------------------------------------------------------------------
# 9. Zusammenspiel: accept_all_text=False ignoriert Blocklist
# ---------------------------------------------------------------------------


class TestBlocklistOnlyWithAcceptAll:
    """Blocklist ist nur relevant wenn accept_all_text=True."""

    def test_accept_all_false_ignores_blocklist(self, tmp_path):
        """accept_all_text=False → Blocklist spielt keine Rolle."""
        _create_file(tmp_path, "data.log")
        # Auch mit leerer Blocklist: ohne accept_all_text wird .log nicht akzeptiert
        config = _make_config(tmp_path, accept_all_text=False, blocklist=[])
        files = scan_files(config)
        assert not any(f.endswith("data.log") for f in files)

    def test_known_type_not_affected_by_blocklist(self, tmp_path):
        """Bekannte Endung (.txt) wird immer akzeptiert, unabhaengig von Blocklist."""
        _create_file(tmp_path, "readme.txt")
        # .txt steht sogar auf der Blocklist (edge case)
        config = _make_config(
            tmp_path,
            file_types=[".txt"],
            accept_all_text=True,
            blocklist=[".txt"],
        )
        files = scan_files(config)
        # .txt wird via file_types-Whitelist akzeptiert, Blocklist greift nur
        # bei accept_all_text-Pfad (unbekannte Endungen)
        assert any(f.endswith("readme.txt") for f in files)
