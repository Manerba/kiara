"""Datasource-Runtime fuer das Kiara-Monorepo.

Monorepo-internes Helper-Modul fuer per-Datasource Overrides und Connector-
Config-Loading im Ingestion-Layer (`pipeline.py` + `ingestion_worker.py`).

Kein stabiler Standalone-Contract: Das Modul darf weiterhin `kiara.shared.*`
und andere interne Kiara-Module direkt voraussetzen.

Erstellt in Sprint 1 (Drift-Beseitigung) — ersetzt:
  - pipeline.py:_build_ds_overrides()
  - pipeline.py:_load_connector_config()
  - ingestion_worker.py:_build_overrides()
  - ingestion_worker.py:_get_connector_config/_decrypt_config Nutzung
"""

import json
import logging

logger = logging.getLogger(__name__)


def _safe_json_loads(value, default=None):
    """JSON-Parsing mit Null-Guard und Fehlertoleranz."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def build_datasource_overrides(row) -> dict:
    """Baut per-Datasource Overrides aus DB-Spalten auf.

    Konsolidiert die frueher getrennten Builder aus pipeline.py und
    ingestion_worker.py. Alle Spalten-Zugriffe sind einzeln abgesichert,
    damit ein fehlendes Feld (Schema-Evolution) nicht den gesamten
    Override-Build abbricht.

    Returns:
        Dict mit gesetzten Override-Werten (nie None).
    """
    overrides = {}

    # --- Chunk-Einstellungen ---
    try:
        if row["ds_chunk_size"] is not None:
            overrides["chunk_size"] = row["ds_chunk_size"]
    except (KeyError, IndexError):
        pass

    try:
        if row["ds_chunk_overlap"] is not None:
            overrides["chunk_overlap"] = row["ds_chunk_overlap"]
    except (KeyError, IndexError):
        pass

    try:
        if row["ds_file_types"]:
            parsed = _safe_json_loads(row["ds_file_types"])
            if parsed is not None:
                overrides["file_types"] = parsed
    except (KeyError, IndexError):
        pass

    try:
        if row["ds_exclude_patterns"]:
            parsed = _safe_json_loads(row["ds_exclude_patterns"])
            if parsed is not None:
                overrides["exclude_patterns"] = parsed
    except (KeyError, IndexError):
        pass

    try:
        chunk_strategy = row["ds_chunk_strategy"]
    except (KeyError, IndexError):
        chunk_strategy = None
    if chunk_strategy and chunk_strategy != "auto":
        overrides["chunk_strategy"] = chunk_strategy

    # --- Feature-Flags ---
    try:
        ddu = row["ds_ddu_enabled"]
    except (KeyError, IndexError):
        ddu = 0
    if ddu:
        overrides["ddu_enabled"] = True

    try:
        ddu_ocr_img = row["ds_ddu_ocr_images"]
    except (KeyError, IndexError):
        ddu_ocr_img = 1
    if not ddu_ocr_img:
        overrides["ddu_ocr_images"] = False

    try:
        vlm = row["ds_vlm_enabled"]
    except (KeyError, IndexError):
        vlm = 0
    if vlm:
        overrides["vlm_enabled"] = True

    try:
        idx_att = row["ds_index_attachments"]
    except (KeyError, IndexError):
        idx_att = 1
    if not idx_att:
        overrides["index_attachments"] = False

    try:
        accept_all = row["ds_accept_all_text"]
    except (KeyError, IndexError):
        accept_all = 0
    if accept_all:
        overrides["accept_all_text"] = True

    try:
        cr = row["ds_contextual_retrieval"]
    except (KeyError, IndexError):
        cr = 0
    if cr:
        overrides["contextual_retrieval"] = True

    # --- Enrichment ---
    try:
        enr_prompt = row["ds_enrichment_prompt_id"]
    except (KeyError, IndexError):
        enr_prompt = None
    if enr_prompt is not None:
        overrides["enrichment_prompt_id"] = enr_prompt

    try:
        enr_backend = row["ds_enrichment_backend_id"]
    except (KeyError, IndexError):
        enr_backend = None
    if enr_backend is not None:
        overrides["enrichment_backend_id"] = enr_backend

    # --- Restriction Level ---
    try:
        restriction_level = row["restriction_level"]
    except (KeyError, IndexError):
        restriction_level = "anonymize"
    if restriction_level:
        overrides["restriction_level"] = restriction_level

    # --- ColBERT ---
    try:
        colbert = row["ds_colbert_enabled"]
    except (KeyError, IndexError):
        colbert = 0
    if colbert:
        overrides["colbert_enabled"] = True

    return overrides


def load_connector_config(row, connector_cls) -> dict:
    """Laedt und entschluesselt Connector-Config aus DB-Row.

    Crypto-Imports sind bewusst funktionslokal, damit bestehende
    Patches auf kiara.shared.crypto.* in Tests weiter greifen.

    Returns:
        Dict mit Connector-Config (nie None). Bei Fehlern fail-closed zu {}.
    """
    from kiara.shared.crypto import decrypt, is_encrypted

    try:
        raw = row["connector_config"]
    except (KeyError, IndexError):
        raw = "{}"

    parsed = _safe_json_loads(raw, {})
    config = parsed if isinstance(parsed, dict) else {}

    for field in connector_cls.secret_fields():
        value = config.get(field, "")
        if value and is_encrypted(value):
            try:
                config[field] = decrypt(value)
            except Exception:
                logger.warning("Decrypt-Fehler fuer Feld '%s' — setze leeren String", field)
                config[field] = ""

    return config
