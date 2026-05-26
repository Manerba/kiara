"""Ingestion-Datenmodelle: FileLogEntry und ProcessingContext.

Extrahiert aus kiara.ingestion.pipeline (Sprint 2, Package-Extraktion).
Shape-Contract: Feldnamen, Defaults und Mutierbarkeit muessen stabil bleiben,
da kiara.web.ingestion_worker und zahlreiche Tests diese direkt nutzen.
"""

from dataclasses import dataclass


@dataclass
class FileLogEntry:
    """Per-File-Protokolleintrag fuer Ingestion-Monitoring."""

    file_path: str
    file_name: str
    file_size: int
    file_ext: str
    parser: str = ""
    parser_fallback: bool = False
    chunk_strategy: str = ""
    chunk_count: int = 0
    status: str = "success"  # success | error | skipped
    error_message: str = ""
    error_phase: str = ""
    is_infra_error: bool = False  # True bei DDU/Backend-Fehlern (Circuit-Breaker-relevant)
    duration_ms: int = 0


@dataclass
class ProcessingContext:
    """Buendelt Feature-Flags und Backend-Config fuer die Dateiverarbeitung.

    Wird von _apply_ds_overrides() erstellt und an _process_single_file()
    durchgereicht. Ersetzt 5 einzelne Positional-Parameter.
    """

    ddu_enabled: bool = False
    ddu_backends: list | None = None
    ddu_ocr_images: bool = True
    vlm_enabled: bool = False
    vlm_model: str = ""
    vlm_url: str = ""
    vlm_prompt: str = ""
