"""File-Utilities fuer die Ingestion-Pipeline.

Scanning, Pattern-Matching, Cancellation.
Hash-Persistenz und Index-State sind nach kiara_ingestion_core extrahiert.
"""

import fnmatch
import os
import stat
from pathlib import Path

from kiara_ingestion_core.file_diff import (  # noqa: F401
    _CONTENT_HASH_CHUNK_SIZE,
    _VALID_HASH_MODES,
    _content_file_hash,
    _file_hash,
    compute_file_hash,
)

# Re-Exports fuer Rueckwaertskompatibilitaet
from kiara_ingestion_core.index_state import ChunkDatasources, HashStore  # noqa: F401

from kiara.shared.constants import ACCEPT_ALL_BLOCKLIST_DEFAULT

# Backward-compat Alias — intern und in Tests referenziert
_ACCEPT_ALL_BLOCKLIST = ACCEPT_ALL_BLOCKLIST_DEFAULT


class _CancelledError(Exception):
    """Internal signal for ingestion cancellation."""


def _matches_any(path: str, patterns: list[str]) -> bool:
    """Check if path matches any exclude pattern."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def scan_files(config: dict) -> list[str]:
    """Scan the datasource directory for indexable files.

    Wenn accept_all_text aktiv ist, werden Dateien mit unbekannter Endung
    per Content-Detection geprueft und als Plaintext akzeptiert.

    Returns:
        Sorted list of absolute file paths.
    """
    ds = config["datasource"]
    base_path = ds["path"]
    file_types = set(ds["file_types"])
    exclude = ds.get("exclude_patterns", [])
    accept_all_text = ds.get("accept_all_text", False)
    blocklist = set(config.get("ingestion", {}).get("accept_all_blocklist", _ACCEPT_ALL_BLOCKLIST))

    files = []
    for root, dirs, filenames in os.walk(base_path):
        # Skip excluded directories early
        dirs[:] = [d for d in dirs if not _matches_any(os.path.join(root, d, ""), exclude)]
        for name in filenames:
            # Office-Lock-Dateien (~$doc.docx) und Backup-Dateien (doc.txt~)
            if name.startswith("~$") or name.endswith("~"):
                continue
            full_path = os.path.join(root, name)
            if _matches_any(full_path, exclude):
                continue
            try:
                lst = os.lstat(full_path)
            except OSError:
                continue
            if not stat.S_ISREG(lst.st_mode):
                continue
            ext = Path(name).suffix.lower()
            if ext in file_types:
                files.append(full_path)
            elif accept_all_text and ext not in blocklist:
                from kiara.ingestion.content_detect import is_text_file

                if is_text_file(full_path):
                    files.append(full_path)

    files.sort()
    return files
