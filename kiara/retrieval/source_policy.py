"""Gemeinsame, administrierbare Quellensteuerung fuer Chat und Agent-Tools."""

from copy import deepcopy

DEFAULT_SOURCE_PROFILES = {
    mode: {
        "default_k": 20,
        "max_k": 100,
        "rerank_fetch_multiplier": 4,
        "keyword_fetch_multiplier": 4,
        "vector_fetch_multiplier": 2,
        "max_sibling_chunks": 30 if mode == "agent" else 5,
        "max_context_chunks": 80,
        **(
            {
                "auto_enabled": True,
                "auto_default_k": 20,
                "auto_min_k": 1,
                "auto_max_k": 100,
                "auto_instructions": (
                    "k bestimmt die maximale Anzahl relevanter Textabschnitte pro Suche, "
                    "nicht die Anzahl aller Dokumente und keine Jahreszahl. "
                    "Waehle fuer konkrete Einzelfragen eher wenige Treffer, fuer "
                    "breite Uebersichten mehr Treffer innerhalb der angegebenen Grenzen. "
                    "Lasse k weg, wenn du keinen begruendeten anderen Bedarf hast. "
                    "Eine hohe Trefferzahl beweist keine vollstaendige Belegauswertung."
                ),
            }
            if mode == "agent"
            else {}
        ),
    }
    for mode in ("standard", "rewriter", "agent", "research")
}


def get_source_profiles(config: dict) -> dict:
    """Liefert vollstaendige Profile, auch fuer kleine Komponenten-Konfigurationen."""
    profiles = deepcopy(DEFAULT_SOURCE_PROFILES)
    configured = config.get("chat", {}).get("source_profiles", {})
    if not isinstance(configured, dict):
        raise ValueError("chat.source_profiles muss ein Objekt sein")
    for mode, values in configured.items():
        if mode not in profiles or not isinstance(values, dict):
            raise ValueError(f"Ungueltiges Quellenprofil: {mode}")
        profiles[mode].update(values)
    return profiles


def validate_source_profiles(config: dict) -> None:
    for mode, profile in get_source_profiles(config).items():
        unknown = profile.keys() - DEFAULT_SOURCE_PROFILES[mode].keys()
        if unknown:
            raise ValueError(f"Unbekannte Quellen-Einstellungen fuer {mode}: {sorted(unknown)}")
        for key, value in profile.items():
            if key in ("auto_enabled", "auto_instructions"):
                continue
            minimum = 0 if key == "default_k" and mode == "agent" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"chat.source_profiles.{mode}.{key} muss >= {minimum} sein")
        if profile["default_k"] > profile["max_k"]:
            raise ValueError(f"{mode}: Quellen-Standard darf das Maximum nicht ueberschreiten")
        if mode == "agent":
            if type(profile["auto_enabled"]) is not bool:
                raise ValueError("agent.auto_enabled muss boolesch sein")
            if (
                not isinstance(profile["auto_instructions"], str)
                or not profile["auto_instructions"].strip()
            ):
                raise ValueError(
                    "Die Erklaerung fuer die automatische Quellenwahl darf nicht leer sein"
                )
            if not (
                profile["auto_min_k"]
                <= profile["auto_default_k"]
                <= profile["auto_max_k"]
                <= profile["max_k"]
            ):
                raise ValueError(
                    "Agent: Auto-Minimum <= Auto-Standard <= Auto-Maximum <= Quellen-Maximum erforderlich"
                )
            if profile["default_k"] == 0 and not profile["auto_enabled"]:
                raise ValueError("Agent: Auto als Standard erfordert aktivierte Automatik")


def resolve_chat_sources(config: dict, mode: str, requested_k: int | None) -> tuple[dict, int]:
    """Wendet das Modusprofil an, bevor Retriever/Adapter erzeugt werden."""
    if mode not in DEFAULT_SOURCE_PROFILES:
        return config, requested_k if requested_k is not None else 0
    profile = get_source_profiles(config)[mode]
    k = profile["default_k"] if requested_k is None else requested_k
    if type(k) is not int or k < 0 or k > profile["max_k"]:
        raise ValueError(f"Quellen muss zwischen 1 und {profile['max_k']} liegen")
    if k == 0 and (mode != "agent" or not profile["auto_enabled"]):
        raise ValueError(
            "Automatische Quellenwahl ist nur fuer einen aktivierten Agent-Modus erlaubt"
        )
    resolved = dict(config)
    resolved["retrieval"] = dict(config.get("retrieval", {}))
    for key in (
        "rerank_fetch_multiplier",
        "keyword_fetch_multiplier",
        "vector_fetch_multiplier",
        "max_sibling_chunks",
        "max_context_chunks",
    ):
        resolved["retrieval"][key] = profile[key]
    resolved["_chat_source_policy"] = {
        "mode": mode,
        "ui_k": requested_k,
        "configured_default_k": profile["default_k"],
        "requested_k": k,
        "selection": "auto" if k == 0 else "fixed",
        "effective_k": None if k == 0 else k,
        "profile": profile,
    }
    return resolved, k


def elasticsearch_result_window(client, targets) -> int:
    """Strictest effective window of exactly the inspected physical targets."""
    names = [target.index_name for target in targets]
    if not names:
        raise ValueError("Elasticsearch-Ergebnisfenster braucht konkrete Ziele")
    settings = client.indices.get_settings(
        index=names,
        name="index.max_result_window",
        include_defaults=True,
        flat_settings=True,
        expand_wildcards="none",
        ignore_unavailable=False,
        allow_no_indices=False,
    )
    settings = getattr(settings, "body", settings)
    if not isinstance(settings, dict) or set(settings) != set(names):
        raise ValueError("Elasticsearch-Ergebnisfenster hat unvollstaendige oder fremde Ziele")
    windows = []
    for values in settings.values():
        value = values.get("settings", {}).get("index.max_result_window")
        if value is None:
            value = values.get("defaults", {}).get("index.max_result_window")
        try:
            if isinstance(value, (bool, float)):
                raise ValueError("Ungueltiger Fenstertyp")
            window = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Elasticsearch liefert kein gueltiges Ergebnisfenster") from exc
        if window < 1:
            raise ValueError("Elasticsearch liefert kein gueltiges Ergebnisfenster")
        windows.append(window)
    if not windows:
        raise ValueError("Elasticsearch-Ergebnisfenster konnte nicht ermittelt werden")
    return min(windows)


def validate_source_backend_budget(config: dict) -> None:
    """Verhindert beim Speichern Profile, die Elasticsearch ueberfordern."""
    retrieval = config.get("retrieval", {})
    if retrieval.get("hybrid_search") != "elasticsearch":
        return
    from kiara.retrieval.elasticsearch_service import _get_client, search_scope

    # No existing searchable target: saving is allowed. Every later concrete
    # search still enforces its actual index window, including newly created DS.
    with search_scope(config, None) as targets:
        if not targets:
            return
        limit = elasticsearch_result_window(_get_client(config), targets)
    for mode, profile in get_source_profiles(config).items():
        rerank = profile["rerank_fetch_multiplier"] if retrieval.get("reranking") else 1
        size = profile["max_k"] * rerank * profile["keyword_fetch_multiplier"]
        if size > limit:
            raise ValueError(
                f"{mode}: Quellen-Maximum und Vorabruf ergeben {size} Treffer; Elasticsearch erlaubt {limit}"
            )
