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
import stat

logger = logging.getLogger(__name__)


_CONTENT_HASH_CHUNK_SIZE = 8192  # Buffer-Groesse fuer streamende Hash-Berechnung

_VALID_HASH_MODES = frozenset({"mtime", "content"})


def _file_hash(path: str, *, file_stat=None) -> str:
    """Compute a fast hash based on file path, size, and mtime."""
    info = file_stat if file_stat is not None else os.stat(path)
    key = f"{path}:{info.st_size}:{info.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


def _metadata_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class FileChangedDuringHashError(OSError):
    """Kein stabiler Dateiinhalt fuer einen erfolgreichen Hash-Checkpoint."""


def _content_file_hash(path: str, *, file_stat=None) -> str:
    """Content-basierter Hash: MD5(path + size + full_content_hash).

    Hasht den kompletten Dateiinhalt streamend (8-KB-Chunks).
    Immun gegen mtime-Aenderungen, erkennt alle Inhaltsaenderungen.
    """
    info = file_stat if file_stat is not None else os.stat(path, follow_symlinks=False)
    h = hashlib.md5()
    # A file replaced by a FIFO/link after scanning must not block or silently
    # change identity. O_NONBLOCK has no effect on ordinary file reads.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or _metadata_identity(before) != _metadata_identity(
            info
        ):
            raise FileChangedDuringHashError(f"Datei vor Content-Hash geaendert: {path}")
        while True:
            chunk = f.read(_CONTENT_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
        after = os.fstat(f.fileno())
        current = os.stat(path, follow_symlinks=False)
        if _metadata_identity(after) != _metadata_identity(before) or _metadata_identity(
            current
        ) != _metadata_identity(before):
            raise FileChangedDuringHashError(f"Datei waehrend Content-Hash geaendert: {path}")
    content_digest = h.hexdigest()
    key = f"{path}:{info.st_size}:{content_digest}"
    return hashlib.md5(key.encode()).hexdigest()


def compute_file_hash(path: str, hash_mode: str = "mtime", *, file_stat=None) -> str:
    """Berechnet den File-Hash mit der angegebenen Strategie.

    Args:
        path: Absoluter Dateipfad.
        hash_mode: "mtime" (Default, schnell) oder "content" (robust).
        file_stat: Unmittelbar zuvor validierte Metadaten des aufrufenden Diff.

    Raises:
        ValueError: Bei unbekanntem hash_mode.
    """
    if hash_mode not in _VALID_HASH_MODES:
        raise ValueError(
            f"Unbekannter hash_mode: {hash_mode!r} "
            f"(erlaubt: {', '.join(sorted(_VALID_HASH_MODES))})"
        )
    if hash_mode == "content":
        return _content_file_hash(path, file_stat=file_stat)
    return _file_hash(path, file_stat=file_stat)


def _compute_file_diff(
    all_files, incremental, stored_hashes, hash_mode="mtime", *, file_stats=None
):
    """Berechnet den Incremental-Diff: neue/geaenderte und geloeschte Dateien.

    Returns:
        Tuple (files_to_process, file_hashes, deleted_files).
        file_hashes ist dict[str, tuple[str, int]] mit (file_hash, file_size)
        fuer alle gescannten Dateien.
    """
    file_hashes = {}
    files_to_process = []

    for fp in all_files:
        try:
            current = os.stat(fp, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode):
                continue
            scanned = file_stats.get(fp) if file_stats is not None else None
            # Scan and diff can be minutes apart. Reuse only after comparing a
            # fresh regular-file identity; pass that validated snapshot to the
            # hash function instead of issuing another redundant path stat.
            info = (
                scanned
                if scanned is not None
                and _metadata_identity(scanned) == _metadata_identity(current)
                else current
            )
            h = compute_file_hash(fp, hash_mode, file_stat=info)
        except (FileNotFoundError, NotADirectoryError):
            continue
        file_hashes[fp] = (h, info.st_size)
        if incremental and stored_hashes.get(fp) == h:
            continue
        files_to_process.append(fp)

    deleted_files = []
    if incremental:
        current_paths = set(file_hashes)
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
