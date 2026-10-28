"""Tests fuer den Config-Snapshot-Service (kiara.admin.services.config_snapshot_service)."""

import json
from unittest.mock import patch

import pytest

from kiara.admin.services import config_snapshot_service as svc


@pytest.fixture()
def snapshot_env(tmp_path, monkeypatch):
    """Isolierte Snapshot-Umgebung: eigene Config-Datei + Snapshot-Verzeichnis."""
    config_file = tmp_path / "kiara.yaml"
    config_file.write_text("retrieval:\n  reranking: true\n  rerank_top_k: 5\n", encoding="utf-8")
    snap_dir = tmp_path / "snapshots"
    monkeypatch.setattr(svc, "SNAPSHOT_DIR", snap_dir)
    monkeypatch.setattr(svc, "INDEX_FILE", snap_dir / "index.jsonl")
    monkeypatch.setattr(svc, "BENCHMARK_ROOT", tmp_path / "benchmarks")
    monkeypatch.setattr(svc, "_config_path", lambda: config_file)
    monkeypatch.setattr(svc, "_max_auto_snapshots", lambda: 3)
    return config_file, snap_dir


class TestCreateAndDedup:
    def test_create_writes_file_and_index(self, snapshot_env):
        config_file, snap_dir = snapshot_env
        entry = svc.create_snapshot(label="test-stand", comment="Kommentar", user="tester")
        assert (snap_dir / entry["file"]).read_bytes() == config_file.read_bytes()
        assert entry["label"] == "test-stand"
        assert len(entry["config_sha256"]) == 64
        index = svc._read_index()
        assert index[-1]["file"] == entry["file"]

    def test_auto_snapshot_dedups_identical_content(self, snapshot_env):
        first = svc.create_snapshot(trigger="auto")
        second = svc.create_snapshot(trigger="auto")
        assert second.get("deduplicated") is True
        assert second["file"] == first["file"]
        assert len(svc._read_index()) == 1

    def test_manual_snapshot_not_deduplicated(self, snapshot_env):
        svc.create_snapshot(trigger="auto")
        entry = svc.create_snapshot(label="marker", trigger="manual")
        assert entry.get("deduplicated") is None
        assert len(svc._read_index()) == 2

    def test_invalid_label_rejected(self, snapshot_env):
        with pytest.raises(svc.ConfigSnapshotError, match="Label"):
            svc.create_snapshot(label="kein label mit spaces!")

    def test_retention_removes_oldest_unnamed(self, snapshot_env):
        config_file, snap_dir = snapshot_env
        files = []
        for i in range(5):
            config_file.write_text(f"retrieval:\n  rerank_top_k: {i}\n", encoding="utf-8")
            files.append(svc.create_snapshot(trigger="auto")["file"])
        index = svc._read_index()
        assert len(index) == 3
        assert not (snap_dir / files[0]).exists()
        assert (snap_dir / files[-1]).exists()

    def test_named_snapshots_survive_retention(self, snapshot_env):
        config_file, _ = snapshot_env
        named = svc.create_snapshot(label="behalten")["file"]
        for i in range(5):
            config_file.write_text(f"x: {i}\n", encoding="utf-8")
            svc.create_snapshot(trigger="auto")
        assert any(e["file"] == named for e in svc._read_index())


class TestListAndJoin:
    def test_list_marks_current_and_joins_benchmarks(self, snapshot_env, tmp_path):
        entry = svc.create_snapshot(label="live")
        results_dir = tmp_path / "benchmarks" / "kiara_golden" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "lauf1_summary.json").write_text(
            json.dumps(
                {
                    "run": {"run_label": "lauf1"},
                    "reproducibility": {"config_file_sha256": entry["config_sha256"]},
                }
            ),
            encoding="utf-8",
        )
        snapshots = svc.list_snapshots()
        assert snapshots[0]["is_current"] is True
        assert snapshots[0]["benchmark_runs"] == ["lauf1"]

    def test_resolve_by_label_sha_and_file(self, snapshot_env):
        entry = svc.create_snapshot(label="findbar")
        assert svc._resolve("findbar")["file"] == entry["file"]
        assert svc._resolve(entry["config_sha256"][:8])["file"] == entry["file"]
        assert svc._resolve(entry["file"])["file"] == entry["file"]
        with pytest.raises(svc.ConfigSnapshotError):
            svc._resolve("gibtsnicht")


class TestDiff:
    def test_diff_masks_secrets_and_reports_changes(self, snapshot_env):
        config_file, _ = snapshot_env
        svc.create_snapshot(label="alt")
        config_file.write_text(
            "retrieval:\n  reranking: false\n  rerank_top_k: 5\ndatabase:\n  password: geheim\n",
            encoding="utf-8",
        )
        changes = svc.diff_snapshots("alt", "current")
        keys = {c["key"]: c for c in changes[1:]}
        assert keys["retrieval.reranking"]["against"] == "false"
        assert keys["database.password"]["against"] == "***"


class TestRestore:
    def test_restore_roundtrip_with_pre_snapshot(self, snapshot_env):
        config_file, _ = snapshot_env
        svc.create_snapshot(label="stand-a")
        config_file.write_text("retrieval:\n  reranking: false\n", encoding="utf-8")

        saved = {}
        with (
            patch.object(svc, "_validated_load", create=True),
            patch("kiara.config.load_config", return_value={"retrieval": {"reranking": True}}),
            patch("kiara.config.save_config", side_effect=lambda cfg: saved.update(cfg)),
        ):
            result = svc.restore_snapshot("stand-a", user="tester")
        assert result["restored"]["label"] == "stand-a"
        assert saved == {"retrieval": {"reranking": True}}
        pre = [e for e in svc._read_index() if e["trigger"] == "pre_restore"]
        assert len(pre) == 1

    def test_restore_rejects_invalid_snapshot_fail_closed(self, snapshot_env):
        svc.create_snapshot(label="kaputt")
        with (
            patch("kiara.config.load_config", side_effect=ValueError("Invalid config")),
            patch("kiara.config.save_config") as mock_save,
            pytest.raises(svc.ConfigSnapshotError, match="Validierung"),
        ):
            svc.restore_snapshot("kaputt")
        mock_save.assert_not_called()
        assert not any(e["trigger"] == "pre_restore" for e in svc._read_index())

    def test_restore_missing_file_rejected(self, snapshot_env):
        entry = svc.create_snapshot(label="wegdamit")
        (svc.SNAPSHOT_DIR / entry["file"]).unlink()
        with pytest.raises(svc.ConfigSnapshotError, match="fehlt"):
            svc.restore_snapshot("wegdamit")


class TestSaveConfigHook:
    def test_save_config_triggers_auto_snapshot(self, snapshot_env, monkeypatch, tmp_path):
        from kiara import config as config_module

        called = {}
        monkeypatch.setattr(svc, "create_snapshot", lambda **kw: called.update(kw) or {})
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(snapshot_env[0]))
        cfg = config_module.load_config(str(snapshot_env[0]))
        config_module.save_config(cfg, config_path=str(snapshot_env[0]))
        assert called.get("trigger") == "auto"

    def test_save_config_snapshot_failure_does_not_block(self, snapshot_env, monkeypatch):
        from kiara import config as config_module

        def boom(**kw):
            raise RuntimeError("Platte voll")

        monkeypatch.setattr(svc, "create_snapshot", boom)
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(snapshot_env[0]))
        cfg = config_module.load_config(str(snapshot_env[0]))
        config_module.save_config(cfg, config_path=str(snapshot_env[0]))
        assert snapshot_env[0].exists()


class TestPermissions:
    def test_snapshot_file_mode_640(self, snapshot_env):
        _, snap_dir = snapshot_env
        entry = svc.create_snapshot(label="rechte")
        mode = (snap_dir / entry["file"]).stat().st_mode & 0o777
        assert mode == 0o640
