"""Tests fuer accept_all_text Content-Detection in scan_files() (Phase 15).

Testet das Zusammenspiel von accept_all_text Flag und Content-Detection
in der File-Scan-Pipeline.
"""

from kiara.ingestion._file_utils import scan_files


class TestScanFilesAcceptAllText:
    """scan_files() mit accept_all_text Flag."""

    def _make_config(self, path, file_types, accept_all_text=False, exclude=None):
        return {
            "datasource": {
                "path": str(path),
                "file_types": file_types,
                "exclude_patterns": exclude or [],
                "accept_all_text": accept_all_text,
            }
        }

    def test_accept_all_text_includes_log(self, tmp_path):
        """accept_all_text=True → .log-Datei mit Textinhalt wird akzeptiert."""
        (tmp_path / "app.log").write_text("2024-01-01 Server started\n")
        (tmp_path / "readme.txt").write_text("Hello\n")

        config = self._make_config(tmp_path, [".txt"], accept_all_text=True)
        files = scan_files(config)

        assert any(f.endswith("app.log") for f in files)
        assert any(f.endswith("readme.txt") for f in files)

    def test_accept_all_text_false_skips_log(self, tmp_path):
        """accept_all_text=False → .log-Datei wird uebersprungen."""
        (tmp_path / "app.log").write_text("2024-01-01 Server started\n")
        (tmp_path / "readme.txt").write_text("Hello\n")

        config = self._make_config(tmp_path, [".txt"], accept_all_text=False)
        files = scan_files(config)

        assert not any(f.endswith("app.log") for f in files)
        assert any(f.endswith("readme.txt") for f in files)

    def test_known_extension_still_uses_whitelist(self, tmp_path):
        """Bekannte Endung (.txt) wird weiterhin per Whitelist akzeptiert."""
        (tmp_path / "readme.txt").write_text("Hello\n")

        config = self._make_config(tmp_path, [".txt"], accept_all_text=True)
        files = scan_files(config)

        assert len(files) == 1
        assert files[0].endswith("readme.txt")

    def test_binary_with_unknown_extension_skipped(self, tmp_path):
        """Binaer-Datei mit unbekannter Endung → uebersprungen auch mit accept_all."""
        (tmp_path / "data.xyz").write_bytes(bytes(range(256)) * 4)

        config = self._make_config(tmp_path, [".txt"], accept_all_text=True)
        files = scan_files(config)

        assert not any(f.endswith("data.xyz") for f in files)

    def test_accept_all_text_with_conf_file(self, tmp_path):
        """accept_all_text → .conf-Datei mit Text wird akzeptiert."""
        (tmp_path / "nginx.conf").write_text("server {\n    listen 80;\n}\n")

        config = self._make_config(tmp_path, [".txt"], accept_all_text=True)
        files = scan_files(config)

        assert any(f.endswith("nginx.conf") for f in files)

    def test_accept_all_respects_exclude(self, tmp_path):
        """accept_all_text beachtet exclude_patterns."""
        (tmp_path / "app.log").write_text("Log-Eintrag\n")

        config = self._make_config(
            tmp_path,
            [".txt"],
            accept_all_text=True,
            exclude=["*.log"],
        )
        files = scan_files(config)

        assert not any(f.endswith("app.log") for f in files)

    def test_default_accept_all_text_false(self, tmp_path):
        """Ohne accept_all_text in Config → Default False."""
        (tmp_path / "app.log").write_text("Log-Eintrag\n")

        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt"],
                "exclude_patterns": [],
            }
        }
        files = scan_files(config)

        assert not any(f.endswith("app.log") for f in files)
