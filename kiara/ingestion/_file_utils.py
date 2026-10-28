"""File-Utilities fuer die Ingestion-Pipeline.

Scanning, Pattern-Matching, Cancellation.
Hash-Persistenz und Index-State sind nach kiara_ingestion_core extrahiert.
"""

import fnmatch
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from kiara_ingestion_core.file_diff import (  # noqa: F401
    _CONTENT_HASH_CHUNK_SIZE,
    _VALID_HASH_MODES,
    FileChangedDuringHashError,
    _content_file_hash,
    _file_hash,
    _metadata_identity,
    compute_file_hash,
)

# Re-Exports fuer Rueckwaertskompatibilitaet
from kiara_ingestion_core.index_state import ChunkDatasources, HashStore  # noqa: F401

from kiara.shared.constants import ACCEPT_ALL_BLOCKLIST_DEFAULT

# Backward-compat Alias — intern und in Tests referenziert
_ACCEPT_ALL_BLOCKLIST = ACCEPT_ALL_BLOCKLIST_DEFAULT

# Connector-eigene Steuerdateien im Wurzelverzeichnis einer Datenquelle sind
# keine Nutzdokumente.  Insbesondere kann die IMAP-UID-Registry mehrere MB
# gross werden und darf niemals als JSON-Dokument vektorisiert werden.
_INTERNAL_DATASOURCE_FILES = frozenset(
    {
        ".scan_done",
        ".uid_registry.json",
        ".uid_registry.json.tmp",
    }
)


class _CancelledError(Exception):
    """Internal signal for ingestion cancellation."""


@dataclass(frozen=True)
class _LoadedFileVersion:
    """Dateifassung unmittelbar vor dem Laden, bis zum Abschluss geprueft.

    ctime und Inode erkennen auch Rueckaenderungen mit gleicher Groesse/mtime.
    Der Planungs-Hash entscheidet nur ueber Kandidaten, nie ueber ihren Abschluss.
    """

    path: str
    identity: tuple
    completion: tuple[str, int]

    @classmethod
    def capture(cls, path: str, hash_mode: str, config: dict):
        from kiara.ingestion.processing_revision import processing_file_hash

        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise FileChangedDuringHashError(f"Keine regulaere Quelldatei: {path}")
        digest = compute_file_hash(path, hash_mode, file_stat=info)
        version = cls(
            path,
            _metadata_identity(info),
            (processing_file_hash(config, path, digest), info.st_size),
        )
        version.validate()
        return version

    def validate(self):
        if _metadata_identity(os.stat(self.path, follow_symlinks=False)) != self.identity:
            raise FileChangedDuringHashError(
                f"Quelldatei waehrend der Verarbeitung geaendert: {self.path}"
            )


def _matches_any(path: str, patterns: list[str]) -> bool:
    """Check if path matches any exclude pattern."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def scan_files(config: dict, *, file_stats: dict[str, os.stat_result] | None = None) -> list[str]:
    """Scan the datasource directory for indexable files.

    Wenn accept_all_text aktiv ist, werden Dateien mit unbekannter Endung
    per Content-Detection geprueft und als Plaintext akzeptiert.

    Returns:
        Sorted list of absolute file paths.
        Wenn file_stats uebergeben ist, werden die Scan-Metadaten mitgeliefert.
        Sie ersetzen keine spaetere Aktualitaetspruefung im Diff.
    """
    from kiara.config import get_mail_archive_config

    archive_root = Path(get_mail_archive_config(config)["path"]).resolve()
    ds = config["datasource"]
    base_path = ds["path"]
    file_types = set(ds["file_types"])
    exclude = ds.get("exclude_patterns", [])
    accept_all_text = ds.get("accept_all_text", False)
    blocklist = set(config.get("ingestion", {}).get("accept_all_blocklist", _ACCEPT_ALL_BLOCKLIST))
    datasource_root = os.path.abspath(base_path)

    resolved_base = Path(base_path).resolve()
    if resolved_base.is_relative_to(archive_root):
        raise ValueError("Mailoriginalarchiv darf nicht als Datenquelle indexiert werden")

    files = []
    for root, dirs, filenames in os.walk(base_path):
        # os.walk folgt Verzeichnissymlinks nicht; keine zusätzlichen stat()-
        # Zugriffe auf jedes Quellverzeichnis für den Archivausschluss.
        relative_root = Path(root).relative_to(base_path)
        dirs[:] = [
            d for d in dirs if not (resolved_base / relative_root / d).is_relative_to(archive_root)
        ]
        # Skip excluded directories early
        dirs[:] = [d for d in dirs if not _matches_any(os.path.join(root, d, ""), exclude)]
        is_datasource_root = os.path.abspath(root) == datasource_root
        for name in filenames:
            if is_datasource_root and name in _INTERNAL_DATASOURCE_FILES:
                continue
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
                if file_stats is not None:
                    file_stats[full_path] = lst
            elif accept_all_text and ext not in blocklist:
                from kiara.ingestion.content_detect import is_text_file

                if is_text_file(full_path):
                    files.append(full_path)
                    if file_stats is not None:
                        file_stats[full_path] = lst

    files.sort()
    return files
