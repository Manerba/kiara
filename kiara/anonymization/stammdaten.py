"""Stammdaten-Loader fuer den Anonymisierungs-Automaten.

Laedt ERP-Stammdaten (CSV/JSON) und LDAP-Mitarbeiterdaten
und baut den Aho-Corasick-Automaten daraus auf.
"""

import csv
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from kiara.anonymization._text_utils import open_with_fallback
from kiara.anonymization.automaton import AnonymizationAutomaton, normalize_entity_type
from kiara.config import DEFAULT_PATHS

logger = logging.getLogger(__name__)


def build_automaton_from_sources(config: dict) -> AnonymizationAutomaton:
    """Baut den Anonymisierungs-Automaten aus allen konfigurierten Quellen auf."""
    anon_cfg = config.get("anonymization", {})
    word_boundaries = anon_cfg.get("word_boundaries", True)
    automaton = AnonymizationAutomaton(word_boundaries=word_boundaries)

    erp_added = 0
    ldap_added = 0

    # -- Stufe 1: ERP-Stammdaten (erp_source oder erp als Alias) --
    erp_config = anon_cfg.get("erp_source", anon_cfg.get("erp", {}))
    if erp_config.get("enabled", False):
        try:
            entries = _load_erp_stammdaten(erp_config)
            for entry in entries:
                etype = normalize_entity_type(entry["type"])
                if automaton.add_entry(
                    text=entry["value"],
                    entity_type=etype,
                    erp_key=entry["key"],
                ):
                    erp_added += 1
                for variant in entry.get("variants", []):
                    if automaton.add_entry(
                        text=variant,
                        entity_type=etype,
                        erp_key=entry["key"],
                    ):
                        erp_added += 1
            logger.info("ERP-Stammdaten geladen: %d/%d Eintraege", erp_added, len(entries))
        except Exception as exc:
            raise RuntimeError(
                f"ERP-Stammdaten fuer Anonymisierung aktiviert, aber nicht verfuegbar: {exc}"
            ) from exc

    # -- Stufe 2: LDAP/Active Directory (ldap_source oder ldap als Alias) --
    ldap_source = anon_cfg.get("ldap_source", anon_cfg.get("ldap", {}))
    if ldap_source.get("enabled", False):
        try:
            entries = _load_ldap_entries(config)
            for entry in entries:
                if automaton.add_entry(
                    text=entry["value"],
                    entity_type=entry["type"],
                    erp_key=entry["key"],
                ):
                    ldap_added += 1
            logger.info("LDAP-Eintraege geladen: %d", ldap_added)
        except Exception as exc:
            raise RuntimeError(
                f"LDAP fuer Anonymisierung aktiviert, aber nicht verfuegbar: {exc}"
            ) from exc

    # -- Benutzerdefinierte Eintraege (type/category, key/id als Aliases) --
    custom = anon_cfg.get("custom_entries", [])
    for entry in custom:
        if isinstance(entry, dict) and "text" in entry:
            etype = normalize_entity_type(entry.get("type", entry.get("category", "CUSTOM")))
            # Fallback-Key MUSS den entity_type enthalten, sonst recyceln
            # gleichnamige Entries unterschiedlichen Typs dasselbe Pseudonym.
            automaton.add_entry(
                text=entry["text"],
                entity_type=etype,
                erp_key=entry.get("key", entry.get("id", f"custom-{etype}-{entry['text']}")),
            )

    automaton.source_counts = {"erp": erp_added, "ldap": ldap_added}
    automaton.build()
    return automaton


ERP_DEFAULT_MAX_AGE_HOURS = 48
LDAP_CACHE_TTL_SECONDS = 600

_automaton_cache: dict = {"key": None, "automaton": None}
_automaton_lock = threading.Lock()


def _erp_path(erp_config: dict) -> Path:
    return Path(erp_config.get("csv_dir", erp_config.get("path", DEFAULT_PATHS["stammdaten"])))


def erp_source_signature(erp_config: dict) -> tuple:
    """Prueft die ERP-Exportdateien und liefert ihre Signatur (Name, Groesse, mtime).

    Fail-closed (Hausaenderung): Ist die ERP-Quelle aktiv, muss mindestens eine
    CSV/JSON-Datei vorliegen, und die neueste darf nicht aelter sein als
    ``max_age_hours`` (Vorgabe 48, 0 = keine Altersgrenze). Sonst RuntimeError,
    damit keine Ausgabe mit lueckenhaften Stammdaten entsteht.
    """
    path = _erp_path(erp_config)
    if not path.is_dir():
        raise RuntimeError(f"Stammdaten-Pfad fehlt: {path}")
    files = sorted([*path.glob("*.csv"), *path.glob("*.json")])
    if not files:
        raise RuntimeError(f"Keine Stammdaten-Datei (*.csv, *.json) in {path}")
    signature = tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files)
    max_age_hours = erp_config.get("max_age_hours", ERP_DEFAULT_MAX_AGE_HOURS)
    if max_age_hours:
        age_hours = (time.time() - max(m for _n, _s, m in signature) / 1e9) / 3600
        if age_hours > float(max_age_hours):
            raise RuntimeError(
                f"Stammdaten in {path} sind veraltet ({age_hours:.0f} h, erlaubt {max_age_hours} h)"
            )
    return signature


def _load_erp_stammdaten(erp_config: dict) -> list[dict]:
    """Laedt Stammdaten aus CSV/JSON-Exportdateien (fail-closed, siehe erp_source_signature)."""
    source_type = erp_config.get("type", "csv")

    if source_type in ("csv", "json", "mixed"):
        erp_source_signature(erp_config)
        entries = _load_from_csv(erp_config)
        if not entries:
            raise RuntimeError(
                f"Stammdaten-Dateien in {_erp_path(erp_config)} ohne gueltige Zeilen"
            )
        return entries

    return []


def cached_automaton(config: dict) -> AnonymizationAutomaton:
    """Automat aus build_automaton_from_sources, je Prozess zwischengespeichert.

    Bei 100.000+ Stammdaten kostet der Aufbau mehr als eine Sekunde und einige
    hundert MB je Anfrage. Neu gebaut wird, wenn sich die Anonymisierungs-
    Konfiguration oder die ERP-Dateien aendern und spaetestens nach
    LDAP_CACHE_TTL_SECONDS (LDAP-Aenderungen). Die Pruefungen der ERP-Quelle
    (Dateien vorhanden, nicht veraltet) laufen bei jedem Aufruf.
    """
    anon_cfg = config.get("anonymization", {})
    erp_config = anon_cfg.get("erp_source", anon_cfg.get("erp", {}))
    ldap_source = anon_cfg.get("ldap_source", anon_cfg.get("ldap", {}))
    erp_signature = erp_source_signature(erp_config) if erp_config.get("enabled") else None
    ldap_bucket = (
        int(time.monotonic() // LDAP_CACHE_TTL_SECONDS) if ldap_source.get("enabled") else None
    )
    config_hash = hashlib.sha256(
        json.dumps(anon_cfg, sort_keys=True, default=str).encode()
    ).hexdigest()
    key = (config_hash, erp_signature, ldap_bucket)
    with _automaton_lock:
        if _automaton_cache["key"] == key:
            return _automaton_cache["automaton"]
        automaton = build_automaton_from_sources(config)
        _automaton_cache.update(key=key, automaton=automaton)
        return automaton


def _load_from_csv(erp_config: dict) -> list[dict]:
    """Laedt Stammdaten aus CSV-Dateien (Semikolon-getrennt).

    Erwartetes Format:
        key;type;value;variants
        K-12345;ORG;Schmidt GmbH;Schmidt|Fa. Schmidt
    """
    entries = []
    path = _erp_path(erp_config)

    if not path.exists():
        logger.warning("Stammdaten-Pfad existiert nicht: %s", path)
        return []

    for file in sorted(path.glob("*.csv")):
        try:
            with open_with_fallback(file, newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    if not row.get("key", "").strip() or not row.get("value", "").strip():
                        continue
                    variants = []
                    if row.get("variants"):
                        variants = [v.strip() for v in row["variants"].split("|") if v.strip()]
                    entries.append(
                        {
                            "key": row["key"].strip(),
                            "type": row.get("type", "CUSTOM").strip() or "CUSTOM",
                            "value": row["value"].strip(),
                            "variants": variants,
                        }
                    )
        except Exception as exc:
            logger.error("CSV-Datei konnte nicht geladen werden: %s — %s", file, exc)
            raise RuntimeError(f"Stammdaten-Datei {file.name} nicht lesbar: {exc}") from exc

    # JSON-Dateien ebenfalls unterstuetzen
    for file in sorted(path.glob("*.json")):
        try:
            with open_with_fallback(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and "entries" in data:
                    items = data["entries"]
                else:
                    logger.warning(
                        "Unerwartete JSON-Struktur in %s (weder Liste noch entries-Key)", file
                    )
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    value = (item.get("value") or item.get("name") or "").strip()
                    key = (item.get("key") or item.get("id") or "").strip()
                    if not value or not key:
                        logger.debug("JSON-Eintrag uebersprungen (value/key fehlt): %s", item)
                        continue
                    entry_type = (
                        item.get("type") or item.get("category", "CUSTOM") or ""
                    ).strip() or "CUSTOM"
                    variants = []
                    raw_variants = item.get("variants")
                    if isinstance(raw_variants, list):
                        variants = [
                            v.strip() for v in raw_variants if isinstance(v, str) and v.strip()
                        ]
                    elif isinstance(raw_variants, str):
                        variants = [v.strip() for v in raw_variants.split("|") if v.strip()]
                    entries.append(
                        {
                            "key": key,
                            "type": entry_type,
                            "value": value,
                            "variants": variants,
                        }
                    )
        except Exception as exc:
            logger.error("JSON-Datei konnte nicht geladen werden: %s — %s", file, exc)
            raise RuntimeError(f"Stammdaten-Datei {file.name} nicht lesbar: {exc}") from exc

    return entries


def _load_ldap_entries(config: dict) -> list[dict]:
    """Laedt alle Personen aus LDAP/Active Directory (Benutzer und Kontakte).

    Nutzt die Zugangsdaten aus der ldap_config-Tabelle, beschraenkt sich aber
    nicht auf die Gruppe der Benutzerverwaltung: Anonymisiert werden muessen
    alle Personen, die in Dokumenten vorkommen koennen, auch ehemalige
    Mitarbeiter (deaktivierte Konten) und Kontakte (Hausaenderung 30.10.2026).

    Fail-closed: Bei aktiver LDAP-Source und fehlender Config / nicht
    erreichbarem Server wird eine Exception geworfen. Der Aufrufer
    (build_automaton) faengt diese und bricht die Automaton-Erstellung
    ab — keine stille Anonymisierungs-Luecke.
    """
    from kiara.repositories.config import ConfigRepository
    from kiara.shared.ldap import get_directory_people
    from kiara.web.config_utils import build_ldap_config

    config_repo = ConfigRepository()
    ldap_row = config_repo.get("ldap_config")

    if not ldap_row:
        raise RuntimeError(
            "LDAP-Anonymisierung aktiviert, aber ldap_config ist nicht "
            "gesetzt. Bitte LDAP im Admin-Tab konfigurieren oder "
            "Anonymisierungs-Source deaktivieren."
        )

    members = get_directory_people(build_ldap_config(ldap_row))

    entries = []
    for member in members:
        dn = member.get("dn", "")
        sam = member.get("sAMAccountName", "")
        surname = (member.get("sn") or "").strip()
        given_name = (member.get("givenName") or "").strip()
        mail = (member.get("mail") or "").strip()
        if not dn and not sam:
            continue
        key = f"ldap-{dn}" if dn else f"ldap-{sam}"

        # Vollstaendiger Name
        if given_name and surname:
            full_name = f"{given_name} {surname}"
            entries.append({"value": full_name, "type": "PERSON", "key": key})

        # Nachname einzeln
        if surname and len(surname) >= 3:
            entries.append({"value": surname, "type": "PERSON", "key": key})

        # Vorname einzeln (DSGVO: isolierte Vornamen in E-Mail-Betreffs etc.)
        if given_name and len(given_name) >= 3:
            entries.append({"value": given_name, "type": "PERSON", "key": key})

        # Nachname, Vorname
        if given_name and surname:
            entries.append({"value": f"{surname}, {given_name}", "type": "PERSON", "key": key})

        # E-Mail
        if mail:
            entries.append({"value": mail, "type": "EMAIL", "key": key})

    return entries
