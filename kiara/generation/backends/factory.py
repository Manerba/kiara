"""Factory-Funktion fuer LLM-Backends und Anonymisierungs-Wrapper.

Erstellt anhand einer Backend-ID oder der Default-Config das passende
LLMBackend fuer EIN konkretes Backend. Cloud-Routing/Composite gehoert
ausdruecklich nicht zu Sprint 1.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from kiara.generation.backends.base import LLMBackend

logger = logging.getLogger(__name__)


def create_backend(config: dict, backend_id: int | None = None) -> LLMBackend:
    """Erstellt das passende LLM-Backend anhand der Backend-ID oder Default-Config.

    Ablauf:
    1. backend_id gegeben -> get_backend_by_id() -> Backend-Dict aus DB
    2. backend_id None -> get_llm_backend(config) -> naechstes aktives llm-Backend
    3. Provider-Dispatch: anthropic -> AnthropicBackend, openai -> OpenAIBackend,
       sonst -> OllamaBackend
    4. API-Key-Entschluesselung passiert in backend_selector._decrypt_backend()
       (get_backend_by_id ruft das bereits auf). Fuer Fallback ueber get_llm_backend()
       ist keine Entschluesselung noetig; dieser Pfad liefert nur url/model
       und wird in Sprint 1 als Ollama-kompatibles llm-Backend behandelt.

    Args:
        config: Globale KIara-Config (dict aus YAML).
        backend_id: Optionale ID aus ki_backends-Tabelle.
            None = naechstes aktives llm-Backend via get_llm_backend(config).

    Returns:
        Konfiguriertes LLMBackend.

    Raises:
        RuntimeError: Wenn kein aktives Backend verfuegbar ist.
        ValueError: Wenn backend_id angegeben aber nicht in DB gefunden.
    """
    backend_dict = None
    provider = "ollama"

    if backend_id is not None:
        from kiara.backend_selector import get_backend_by_id

        backend_dict = get_backend_by_id(backend_id)
        if backend_dict is None:
            raise ValueError(f"Backend mit ID {backend_id} nicht gefunden oder inaktiv")
        if backend_dict.get("category") not in ("llm", "cloud"):
            raise ValueError(
                f"Backend mit ID {backend_id} ist kein LLM-/Cloud-Backend "
                f"(Kategorie: {backend_dict.get('category')})"
            )
        provider = backend_dict.get("provider", "ollama")
    else:
        # Fallback: get_llm_backend() liefert nur {"url", "model"} ohne provider.
        # Sprint 1 behandelt diesen Pfad deshalb explizit als
        # Ollama-kompatibles llm-Backend.
        from kiara.config import get_llm_backend

        backend_dict = get_llm_backend(config)
        # get_llm_backend gibt {"url": ..., "model": ...} zurueck

    if provider == "anthropic":
        from kiara.generation.backends.anthropic import AnthropicBackend

        cloud_cfg = config.get("cloud", {})
        from kiara.shared.constants import DEFAULT_LLM_TEMPERATURE, DEFAULT_MAX_TOKENS

        return AnthropicBackend(
            api_key=backend_dict.get("api_key", ""),
            model=backend_dict["model"],
            max_tokens=cloud_cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
            temperature=cloud_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE),
            base_url=backend_dict.get("base_url", ""),
        )

    if provider == "openai":
        from kiara.generation.backends.openai import OpenAIBackend

        cloud_cfg = config.get("cloud", {})
        from kiara.shared.constants import DEFAULT_LLM_TEMPERATURE, DEFAULT_MAX_TOKENS

        return OpenAIBackend(
            api_key=backend_dict.get("api_key", ""),
            model=backend_dict["model"],
            max_tokens=cloud_cfg.get("max_tokens", DEFAULT_MAX_TOKENS),
            temperature=cloud_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE),
            base_url=backend_dict.get("base_url", ""),
        )

    # Default: Ollama
    from kiara.generation.backends.ollama import OllamaBackend
    from kiara.shared.constants import DEFAULT_LLM_NUM_CTX, DEFAULT_LLM_TEMPERATURE

    ollama_cfg = config.get("ollama", {})
    ollama_url = backend_dict.get("url", "")
    if not ollama_url:
        raise ValueError(
            f"Ollama-Backend benoetigt eine URL, aber backend_dict['url'] ist leer "
            f"(backend_id={backend_id}). Bitte URL in der Backend-Konfiguration setzen."
        )
    return OllamaBackend(
        url=ollama_url,
        model=backend_dict["model"],
        temperature=ollama_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE),
        num_ctx=ollama_cfg.get("num_ctx", DEFAULT_LLM_NUM_CTX),
    )


def _normalize_config_for_anonymizer(config: dict) -> dict:
    """Normalisiert bestehende Config-Varianten fuer Anonymizer.load_entities().

    Ziel: Anonymizer.load_entities() erwartet fuer ERP effektiv eine erp-Struktur
    mit enabled, waehrend die aktuelle Produktiv-Konfiguration haeufig ueber
    erp_source.enabled/type/path gesteuert wird.

    Prioritaet:
    1. Wenn erp_source vorhanden ist, gilt diese Legacy-Struktur als Source of Truth
       (auch fuer enabled=False, damit deaktivierte Quellen nicht versehentlich
       durch eine parallel vorhandene erp-Sektion reaktiviert werden; eine
       bestehende erp.enabled=True-Markierung wird in diesem Fall explizit
       auf False heruntergezogen).
    2. Nur wenn erp_source komplett fehlt, wird eine reine erp-Konfiguration
       in die von Anonymizer erwartete enabled-Form ueberfuehrt.
    """
    normalized = deepcopy(config)
    anon_cfg = normalized.setdefault("anonymization", {})

    erp_source = anon_cfg.get("erp_source", {})
    erp_cfg = dict(anon_cfg.get("erp", {}))
    if erp_source:
        if erp_source.get("enabled", False):
            anon_cfg["erp"] = {
                "enabled": True,
                "backend": erp_source.get("type", erp_cfg.get("backend", "csv")),
                "csv_dir": erp_source.get("path", erp_cfg.get("csv_dir", "")),
            }
        elif erp_cfg:
            anon_cfg["erp"] = {
                **erp_cfg,
                "enabled": False,
            }
    elif erp_cfg and "enabled" not in erp_cfg:
        # Reine erp-Configs ohne Legacy-erp_source als aktiv behandeln.
        anon_cfg["erp"] = {
            "enabled": True,
            **erp_cfg,
        }

    if "ldap" not in anon_cfg and "ldap_source" in anon_cfg:
        anon_cfg["ldap"] = dict(anon_cfg["ldap_source"])

    return normalized


def wrap_with_anonymization(
    backend: LLMBackend,
    config: dict,
    restriction_levels: list[str] | None = None,
) -> LLMBackend:
    """Wrapt ein Backend mit Anonymisierung basierend auf Restriction-Levels.

    Entscheidungslogik:
    - provider == "ollama": KEIN Wrapping (lokal, keine PII-Gefahr)
    - restriction_levels is None: bisheriger Fallback auf anonymization.enabled
    - leere Liste oder nur "public": KEIN Wrapping
    - mindestens ein "anonymize": Wrapping
    - "blocked" oder unbekannte Werte: ValueError (Programmfehler im Aufrufer)

    Args:
        backend: Das zu wrappende Backend.
        config: Globale KIara-Config.
        restriction_levels: Restriction-Levels der beteiligten Datasources.
            None = kein DS-Kontext (bisheriger Fallback).

    Returns:
        Das Original-Backend (wenn kein Wrapping noetig) oder
        ein AnonymizingBackend-Wrapper.

    Raises:
        ValueError: Wenn blocked/unknown in restriction_levels.
        RuntimeError: Wenn Anonymisierung bei explizitem anonymize fehlschlaegt.
    """
    if backend.provider == "ollama":
        return backend

    if restriction_levels is None:
        # Kein DS-Kontext: bisheriger Fallback auf globales anonymization.enabled
        anon_cfg = config.get("anonymization", {})
        # Default True — Init-Fehler werden fail-closed hart geworfen (Issue #1056).
        anon_enabled = anon_cfg.get("enabled", True)
        if not anon_enabled:
            logger.info("Cloud-Backend ohne Anonymisierung — PII werden ungeschuetzt gesendet")
            return backend
        try:
            from kiara.anonymization.anonymizer import Anonymizer
            from kiara.anonymization.llm_detector import build_llm_detector
            from kiara.anonymization.stammdaten import build_automaton_from_sources

            normalized_config = _normalize_config_for_anonymizer(config)
            automaton = build_automaton_from_sources(normalized_config)
            llm_detector = build_llm_detector(normalized_config)
            anonymizer = Anonymizer(automaton, normalized_config, llm_detector=llm_detector)
            from kiara.generation.backends.anonymizing import AnonymizingBackend

            return AnonymizingBackend(backend, anonymizer)
        except Exception as exc:
            raise RuntimeError(
                f"Anonymisierung ist aktiviert (anonymization.enabled=True, ggf. Default) "
                f"aber konnte nicht initialisiert werden: {exc}"
            ) from exc

    # Explizite restriction_levels: DS-Kontext vorhanden
    from kiara.generation._validators import (
        VALID_RESTRICTION_LEVELS,
    )

    for lvl in restriction_levels:
        if lvl not in VALID_RESTRICTION_LEVELS or lvl == "blocked":
            raise ValueError(
                f"blocked/unbekanntes Restriction-Level '{lvl}' darf nicht bis "
                f"wrap_with_anonymization() durchkommen — Programmfehler im Aufrufer"
            )

    # Leere Liste oder nur public -> kein Wrapping
    if not restriction_levels or all(l == "public" for l in restriction_levels):
        return backend

    # Mindestens ein anonymize -> Wrapping (fail-closed bei Init-Fehler)
    try:
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.llm_detector import build_llm_detector
        from kiara.anonymization.stammdaten import build_automaton_from_sources

        normalized_config = _normalize_config_for_anonymizer(config)
        automaton = build_automaton_from_sources(normalized_config)
        llm_detector = build_llm_detector(normalized_config)
        anonymizer = Anonymizer(automaton, normalized_config, llm_detector=llm_detector)
        from kiara.generation.backends.anonymizing import AnonymizingBackend

        return AnonymizingBackend(backend, anonymizer)
    except Exception as exc:
        raise RuntimeError(
            f"Anonymisierung fuer explizites anonymize-Level konnte nicht "
            f"initialisiert werden: {exc}"
        ) from exc
