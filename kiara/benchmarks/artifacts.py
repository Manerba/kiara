"""Private artifact writers for benchmark outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRIVATE_DIR_MODE = 0o750
PRIVATE_FILE_MODE = 0o640


def ensure_private_dir(path: str | Path) -> Path:
    """Create an artifact directory and best-effort harden its permissions."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        target.chmod(PRIVATE_DIR_MODE)
    except OSError as exc:
        logger.warning(
            "Artefakt-Verzeichnisrechte konnten nicht gesetzt werden (%s): %s", target, exc
        )
    return target


def _chmod_file(path: Path) -> None:
    try:
        path.chmod(PRIVATE_FILE_MODE)
    except OSError as exc:
        logger.warning("Artefakt-Dateirechte konnten nicht gesetzt werden (%s): %s", path, exc)


def write_private_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write a text artifact with owner/group-only read permissions."""

    target = Path(path)
    ensure_private_dir(target.parent)
    target.write_text(content, encoding=encoding)
    _chmod_file(target)
    return target


def write_private_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Write a JSON artifact using KIara's benchmark permission policy."""

    content = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=sort_keys) + "\n"
    return write_private_text(path, content)


def write_private_jsonl(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    sort_keys: bool = True,
) -> Path:
    """Write a JSONL artifact using KIara's benchmark permission policy."""

    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n" for row in rows
    )
    return write_private_text(path, content)
