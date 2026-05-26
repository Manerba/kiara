"""File-Hashing und Incremental-Diff-Logik.

Extrahiert aus kiara.ingestion._file_utils (Hash-Funktionen) und
kiara.ingestion.pipeline (_compute_file_diff, _filter_resumed_files).

Return-Contract:
    file_hashes / resumed_hashes sind dict[str, tuple[str, int]]
    mit (file_hash, file_size). Dieser Tuple-Contract ist Downstream
    verpflichtend fuer HashStore.upsert_many() und _load_and_diff_files().
"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


_CONTENT_HASH_CHUNK_SIZE = 8192  # Buffer-Groesse fuer streamende Hash-Berechnung

_VALID_HASH_MODES = frozenset({"mtime", "content"})


def _file_hash(path: str) -> str:
    """Compute a fast hash based on file path, size, and mtime."""
    stat = os.stat(path)
    key = f"{path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


def _content_file_hash(path: str) -> str:
    """Content-basierter Hash: MD5(path + size + full_content_hash).

    Hasht den kompletten Dateiinhalt streamend (8-KB-Chunks).
    Immun gegen mtime-Aenderungen, erkennt alle Inhaltsaenderungen.
    """
    stat = os.stat(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CONTENT_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    content_digest = h.hexdigest()
    key = f"{path}:{stat.st_size}:{content_digest}"
    return hashlib.md5(key.encode()).hexdigest()


def compute_file_hash(path: str, hash_mode: str = "mtime") -> str:
    """Berechnet den File-Hash mit der angegebenen Strategie.

    Args:
        path: Absoluter Dateipfad.
        hash_mode: "mtime" (Default, schnell) oder "content" (robust).

    Raises:
        ValueError: Bei unbekanntem hash_mode.
    """
    if hash_mode not in _VALID_HASH_MODES:
        raise ValueError(
            f"Unbekannter hash_mode: {hash_mode!r} "
            f"(erlaubt: {', '.join(sorted(_VALID_HASH_MODES))})"
        )
    if hash_mode == "content":
        return _content_file_hash(path)
    return _file_hash(path)


def _compute_file_diff(all_files, incremental, stored_hashes, hash_mode="mtime"):
    """Berechnet den Incremental-Diff: neue/geaenderte und geloeschte Dateien.

    Returns:
        Tuple (files_to_process, file_hashes, deleted_files).
        file_hashes ist dict[str, tuple[str, int]] mit (file_hash, file_size)
        fuer alle gescannten Dateien.
    """
    file_hashes = {}
    files_to_process = []

    for fp in all_files:
        stat = os.stat(fp)
        h = compute_file_hash(fp, hash_mode)
        file_hashes[fp] = (h, stat.st_size)
        if incremental and stored_hashes.get(fp) == h:
            continue
        files_to_process.append(fp)

    deleted_files = []
    if incremental:
        current_paths = set(all_files)
        for old_path in stored_hashes:
            if old_path not in current_paths:
                deleted_files.append(old_path)

    return files_to_process, file_hashes, deleted_files


def _filter_resumed_files(files_to_process, file_hashes, resume_skip_files):
    """Filtert bereits verarbeitete Dateien aus einem Resume-Job heraus.

    Args:
        files_to_process: Liste der zu verarbeitenden Dateipfade.
        file_hashes: Dict mit Dateipfad -> (hash, file_size) fuer alle
            gescannten Dateien.
        resume_skip_files: Set von Dateipfaden, die im vorherigen Job bereits
            erfolgreich verarbeitet wurden.

    Returns:
        Tuple (filtered_files, resumed_hashes).
        filtered_files: files_to_process ohne bereits verarbeitete Dateien.
        resumed_hashes: dict[str, tuple[str, int]] der uebersprungenen Dateien
            mit ihren (hash, file_size)-Tupeln.
    """
    resumed_hashes = {}
    before = len(files_to_process)
    filtered = []
    for fp in files_to_process:
        if fp in resume_skip_files:
            resumed_hashes[fp] = file_hashes[fp]
        else:
            filtered.append(fp)
    if before != len(filtered):
        logger.info(
            "Resume: %d Dateien uebersprungen (bereits im vorherigen Job verarbeitet)",
            before - len(filtered),
        )
    return filtered, resumed_hashes
