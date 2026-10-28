"""Freshness and metadata-call contracts across scan, diff and hashing."""

import hashlib
import os
from unittest.mock import patch

import pytest
from kiara_ingestion_core import file_diff

from kiara.ingestion._file_utils import scan_files


def config(root):
    return {"datasource": {"path": str(root), "file_types": [".md"], "exclude_patterns": []}}


@pytest.mark.parametrize("mode", ["mtime", "content"])
def test_scan_snapshot_does_not_hide_a_later_change(tmp_path, mode):
    path = tmp_path / "document.md"
    path.write_text("initial content")
    old_hash = file_diff.compute_file_hash(str(path), mode)
    metadata = {}
    paths = scan_files(config(tmp_path), file_stats=metadata)
    path.write_text("a changed, longer document")
    changed, hashes, deleted = file_diff._compute_file_diff(
        paths, True, {str(path): old_hash}, hash_mode=mode, file_stats=metadata
    )
    assert changed == [str(path)]
    assert hashes[str(path)] == (file_diff.compute_file_hash(str(path), mode), path.stat().st_size)
    assert not deleted


@pytest.mark.parametrize("replacement", ["missing", "directory", "symlink", "fifo"])
@pytest.mark.parametrize("mode", ["mtime", "content"])
def test_disappeared_and_nonregular_candidates_are_not_hashed(tmp_path, replacement, mode):
    path = tmp_path / "document.md"
    path.write_text("initial")
    metadata = {}
    paths = scan_files(config(tmp_path), file_stats=metadata)
    path.unlink()
    if replacement == "directory":
        path.mkdir()
    elif replacement == "symlink":
        other = tmp_path / "other.txt"
        other.write_text("different")
        path.symlink_to(other)
    elif replacement == "fifo":
        os.mkfifo(path)
    changed, hashes, deleted = file_diff._compute_file_diff(
        paths, True, {str(path): "old"}, hash_mode=mode, file_stats=metadata
    )
    assert (changed, hashes, deleted) == ([], {}, [str(path)])


def test_unreadable_file_cannot_be_mistaken_for_deleted(tmp_path):
    path = str(tmp_path / "document.md")
    with patch.object(file_diff.os, "stat", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError):
            file_diff._compute_file_diff([path], True, {path: "old"})


def test_mtime_scan_and_diff_use_two_metadata_queries_not_three(tmp_path):
    path = tmp_path / "document.md"
    path.write_text("stable")
    metadata = {}
    with (
        patch("os.stat", wraps=os.stat) as stat_call,
        patch("os.lstat", wraps=os.lstat) as lstat_call,
    ):
        files = scan_files(config(tmp_path), file_stats=metadata)
        _, hashes, _ = file_diff._compute_file_diff(files, False, {}, file_stats=metadata)
    assert lstat_call.call_count == 1
    assert stat_call.call_count == 1
    info = path.stat()
    expected = hashlib.md5(f"{path}:{info.st_size}:{info.st_mtime}".encode()).hexdigest()
    assert hashes[str(path)] == (expected, info.st_size)


@pytest.mark.parametrize("change", ["content", "replacement"])
def test_content_change_during_hash_fails_closed(tmp_path, monkeypatch, change):
    path = tmp_path / "document.md"
    path.write_text("initial content")
    original_fstat = os.fstat
    calls = 0

    def changed_fstat(fd):
        nonlocal calls
        calls += 1
        info = original_fstat(fd)
        if calls == 1:
            if change == "content":
                # A size change is observable even on filesystems whose
                # timestamp granularity coalesces two immediate writes.
                path.write_text("changed longer content")
            else:
                replacement = tmp_path / "replacement"
                replacement.write_text("initial content")
                replacement.replace(path)
        return info

    monkeypatch.setattr(file_diff.os, "fstat", changed_fstat)
    with pytest.raises(file_diff.FileChangedDuringHashError):
        file_diff._compute_file_diff([str(path)], True, {}, hash_mode="content")
