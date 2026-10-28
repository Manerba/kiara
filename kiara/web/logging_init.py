"""Root-Logger-Initialisierung fuer Web-Apps.

Muss vor dem Import von FastAPI/uvicorn am Modul-Anfang aufgerufen werden,
damit uvicorn-Logger den File-Handler bekommen, bevor sie ihre eigenen
Handler registrieren. Ohne diesen Aufruf landen uvicorn-Access-Logs und
Background-Thread-Logs nicht in /var/log/kiara/kiara.log.
"""

import logging
import logging.handlers
import sys

from kiara.config import DEFAULT_PATHS
from kiara.shared.constants import LOG_FILE_BACKUP_COUNT, LOG_FILE_MAX_BYTES


def init_root_logger() -> None:
    """Haengt RotatingFileHandler an Root-Logger und uvicorn-Logger.

    Idempotent: Mehrfach-Aufrufe haengen keine zusaetzlichen Handler an.
    Fail-soft: Bei PermissionError/OSError fallback auf stderr.
    """
    log_path = DEFAULT_PATHS["log"]
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s[%(process)d]: %(levelname)s %(message)s",
        datefmt="%b %d %H:%M:%S",
    )
    try:
        file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
    except (PermissionError, OSError):
        file_handler = logging.StreamHandler(sys.stderr)
        file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)

    for uv_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uv_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(file_handler)
        uv_logger.propagate = False
