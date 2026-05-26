"""Security-Logging Konfiguration."""

import logging
import os
from logging.handlers import RotatingFileHandler

from kiara.config import DEFAULT_PATHS
from kiara.shared.constants import LOG_MAX_BYTES

LOG_FILE = DEFAULT_PATHS["log"]
LOG_DIR = os.path.dirname(LOG_FILE)


def attach_db_handler(logger: logging.Logger, *, strict_db: bool = False) -> None:
    """Haengt einen `SystemEventDBHandler` an den Logger an (idempotent).

    Lazy-Import: DB-Abhaengigkeiten werden erst beim ersten Handler-Anhaengen
    geladen, damit der Import dieses Moduls ohne DB-Stack funktioniert.
    Im Normalbetrieb werden Fehler beim Import/Anhaengen geschluckt — der
    File-Handler bleibt primaerer Log-Pfad. Mit ``strict_db=True`` propagieren
    Attach- und spaetere DB-Fehler; genutzt von forensischen CLI-Pfaden.
    """
    try:
        from kiara.web.logging_db_handler import SystemEventDBHandler
    except Exception as exc:
        if strict_db:
            raise RuntimeError("SystemEventDBHandler konnte nicht geladen werden") from exc
        logger.warning("SystemEventDBHandler konnte nicht geladen werden: %s", exc)
        return

    for handler in logger.handlers:
        if isinstance(handler, SystemEventDBHandler):
            if strict_db:
                handler.strict = True
                handler.setLevel(logging.INFO)
            return
    try:
        handler = SystemEventDBHandler(strict=strict_db)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    except Exception as exc:
        if strict_db:
            raise RuntimeError("SystemEventDBHandler konnte nicht angehaengt werden") from exc
        if not any(isinstance(h, SystemEventDBHandler) for h in logger.handlers):
            logger.warning("SystemEventDBHandler konnte nicht angehaengt werden: %s", exc)


def setup_security_logger(*, strict_db: bool = False) -> logging.Logger:
    """Richtet den Security-Logger mit RotatingFileHandler + DB-Handler ein.

    Idempotent: Mehrfach-Aufrufe haengen keine zusaetzlichen Handler an.
    ``strict_db=True`` schaerft einen vorhandenen DB-Handler fuer diesen
    Prozess nach.
    `propagate = False` verhindert Doppel-Logging ueber den Root-Logger.
    """
    logger = logging.getLogger("kiara.security")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        attach_db_handler(logger, strict_db=strict_db)
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        if LOG_DIR:
            os.makedirs(LOG_DIR, exist_ok=True)
        handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=5)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except (PermissionError, OSError):
        fallback = logging.StreamHandler()
        fallback.setFormatter(formatter)
        logger.addHandler(fallback)
        logger.warning(
            "Security-Log konnte nicht nach %s geschrieben werden, verwende stderr",
            LOG_FILE,
        )

    attach_db_handler(logger, strict_db=strict_db)
    return logger


def get_security_logger():
    """Gibt den Security-Logger zurueck."""
    return logging.getLogger("kiara.security")
