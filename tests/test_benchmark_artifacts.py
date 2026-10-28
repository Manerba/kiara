import stat

from kiara.benchmarks.artifacts import (
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    write_private_json,
    write_private_jsonl,
)


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_private_artifact_writers_harden_file_and_directory_permissions(tmp_path):
    json_path = tmp_path / "run" / "summary.json"
    jsonl_path = tmp_path / "run" / "details.jsonl"

    write_private_json(json_path, {"ok": True})
    write_private_jsonl(jsonl_path, [{"case": "KGS-009"}])

    assert _mode(json_path.parent) == PRIVATE_DIR_MODE
    assert _mode(json_path) == PRIVATE_FILE_MODE
    assert _mode(jsonl_path) == PRIVATE_FILE_MODE
