"""Adapter fuer ERP-Stammdaten-Import.

Unterstuetzt CSV-Import (Standardweg), REST-API und SQL-View als Backends.
CSV-Format: Semikolon-getrennt (deutsches ERP-Standard), UTF-8.
"""

import csv
import json
import logging
from pathlib import Path

from kiara_anonymizer._text_utils import open_with_fallback

logger = logging.getLogger(__name__)

# Mapping von ERP-Typ-Bezeichnungen auf Anonymisierungs-Kategorien
_TYPE_TO_CATEGORY = {
    "KUNDE": "customer",
    "KUNDEN": "customer",
    "CUSTOMER": "customer",
    "ORG": "organization",
    "ORGANIZATION": "organization",
    "ORGANISATION": "organization",
    "UNTERNEHMEN": "organization",
    "FIRMA": "organization",
    "LIEFERANT": "supplier",
    "SUPPLIER": "supplier",
    "MITARBEITER": "employee",
    "EMPLOYEE": "employee",
    "PERSON": "employee",
    "PRODUKT": "product",
    "PRODUCT": "product",
    "ARTIKEL": "product",
    "STANDORT": "location",
    "LOCATION": "location",
    "ORT": "location",
    "ADDRESS": "location",
}


class ERPConnector:
    """Importiert Stammdaten aus dem ERP-System.

    Unterstuetzt mehrere Backends:
    - CSV-Import (Standardweg, immer verfuegbar)
    - REST-API (konfigurierbar)
    - SQL-View (konfigurierbar, read-only)
    """

    @staticmethod
    def from_csv(
        customers_csv: str = None,
        suppliers_csv: str = None,
        employees_csv: str = None,
        products_csv: str = None,
    ) -> list[dict]:
        """Importiert Entitaeten aus CSV-Dateien.

        CSV-Format: id;name;category;[additional_fields...]
        Semikolon-getrennt (deutsches ERP-Standard), UTF-8.

        Args:
            customers_csv: Pfad zur Kunden-CSV.
            suppliers_csv: Pfad zur Lieferanten-CSV.
            employees_csv: Pfad zur Mitarbeiter-CSV.
            products_csv: Pfad zur Produkt-CSV.

        Returns:
            Liste von Entitaeten im Format:
            [{"value": "Mueller GmbH", "category": "customer", "id": "K-1234"}, ...]
        """
        entities = []
        file_category_pairs = [
            (customers_csv, "customer"),
            (suppliers_csv, "supplier"),
            (employees_csv, "employee"),
            (products_csv, "product"),
        ]

        for filepath, default_category in file_category_pairs:
            if not filepath:
                continue
            path = Path(filepath)
            if not path.exists():
                logger.warning("CSV-Datei nicht gefunden: %s", path)
                continue
            entities.extend(_load_csv_file(path, default_category))

        return _deduplicate(entities)

    @staticmethod
    def from_directory(csv_dir: str) -> list[dict]:
        """Importiert alle CSV- und JSON-Dateien aus einem Verzeichnis.

        Erkennt den Typ anhand des Dateinamens:
        - customers.csv / kunden.csv -> customer
        - suppliers.csv / lieferanten.csv -> supplier
        - employees.csv / mitarbeiter.csv -> employee
        - products.csv / produkte.csv / artikel.csv -> product
        - locations.csv / standorte.csv -> location
        - Sonstige -> custom

        Args:
            csv_dir: Pfad zum Verzeichnis mit Stammdaten-Dateien.

        Returns:
            Liste von Entitaeten.
        """
        path = Path(csv_dir)
        if not path.exists():
            logger.warning("Stammdaten-Verzeichnis existiert nicht: %s", path)
            return []

        entities = []

        # CSV-Dateien
        for file in sorted(path.glob("*.csv")):
            category = _guess_category_from_filename(file.stem)
            entities.extend(_load_csv_file(file, category))

        # JSON-Dateien
        for file in sorted(path.glob("*.json")):
            category = _guess_category_from_filename(file.stem)
            entities.extend(_load_json_file(file, category))

        return _deduplicate(entities)

    @staticmethod
    def from_config(config: dict) -> list[dict]:
        """Liest ERP-Connector-Konfiguration und importiert.

        Args:
            config: Anonymization-Config mit erp-Sektion.

        Returns:
            Liste von Entitaeten.
        """
        erp_cfg = config.get("anonymization", {}).get("erp", {})
        if not erp_cfg:
            # Fallback: altes Config-Format
            erp_cfg = config.get("anonymization", {}).get("erp_source", {})

        if not erp_cfg.get("enabled", True):
            return []

        backend = erp_cfg.get("backend", erp_cfg.get("type", "csv"))

        if backend == "csv":
            csv_dir = erp_cfg.get("csv_dir", erp_cfg.get("path", ""))
            if csv_dir:
                return ERPConnector.from_directory(csv_dir)
            return []
        elif backend == "rest":
            logger.info("REST-Backend noch nicht implementiert")
            return []
        elif backend == "sql":
            logger.info("SQL-Backend noch nicht implementiert")
            return []
        else:
            logger.warning("Unbekanntes ERP-Backend: %s", backend)
            return []


def _load_csv_file(file_path: Path, default_category: str) -> list[dict]:
    """Laedt Entitaeten aus einer CSV-Datei.

    Unterstuetzt zwei Formate:
    1. Neues Format: id;name;category;variants
    2. Legacy-Format: key;type;value;variants (kompatibel mit stammdaten.py)
    """
    entities = []
    try:
        with open_with_fallback(file_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            headers = reader.fieldnames or []

            for row in reader:
                # Neues Format (id;name;category)
                if "name" in headers:
                    name = row.get("name", "").strip()
                    if not name:
                        continue
                    category = _normalize_category(row.get("category", default_category))
                    entity_id = row.get("id", "").strip()
                    entities.append(
                        {
                            "value": name,
                            "category": category,
                            "id": entity_id,
                        }
                    )
                    # Varianten
                    variants = row.get("variants", "")
                    if variants:
                        for v in variants.split("|"):
                            v = v.strip()
                            if v:
                                entities.append(
                                    {
                                        "value": v,
                                        "category": category,
                                        "id": entity_id,
                                    }
                                )
                # Legacy-Format (key;type;value;variants)
                elif "value" in headers:
                    value = row.get("value", "").strip()
                    if not value:
                        continue
                    category = _normalize_category(row.get("type", default_category))
                    entity_id = row.get("key", "").strip()
                    entities.append(
                        {
                            "value": value,
                            "category": category,
                            "id": entity_id,
                        }
                    )
                    variants = row.get("variants", "")
                    if variants:
                        for v in variants.split("|"):
                            v = v.strip()
                            if v:
                                entities.append(
                                    {
                                        "value": v,
                                        "category": category,
                                        "id": entity_id,
                                    }
                                )

    except Exception as exc:
        logger.warning("CSV-Datei konnte nicht geladen werden: %s — %s", file_path, exc)

    return entities


def _load_json_file(file_path: Path, default_category: str) -> list[dict]:
    """Laedt Entitaeten aus einer JSON-Datei."""
    try:
        with open_with_fallback(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "entries" in data:
            items = data["entries"]
        else:
            logger.warning(
                "Unerwartete JSON-Struktur in %s (weder Liste noch dict mit 'entries'-Key)",
                file_path,
            )
            return []
        entities = []

        for item in items:
            value = (item.get("value") or item.get("name") or "").strip()
            if not value:
                continue
            category = _normalize_category(
                item.get("category") or item.get("type") or default_category
            )
            entity_id = item.get("id") or item.get("key") or ""
            entities.append(
                {
                    "value": value,
                    "category": category,
                    "id": entity_id,
                }
            )

        return entities

    except Exception as exc:
        logger.warning("JSON-Datei konnte nicht geladen werden: %s — %s", file_path, exc)
        return []


def _normalize_category(raw: str) -> str:
    """Normalisiert ERP-Typ-Bezeichnungen auf Anonymisierungs-Kategorien."""
    raw_str = str(raw or "").strip()
    return _TYPE_TO_CATEGORY.get(raw_str.upper(), raw_str.lower())


def _guess_category_from_filename(stem: str) -> str:
    """Leitet die Kategorie aus dem Dateinamen ab."""
    stem_lower = stem.lower()
    mapping = {
        "customer": "customer",
        "kunden": "customer",
        "kunde": "customer",
        "supplier": "supplier",
        "lieferant": "supplier",
        "lieferanten": "supplier",
        "employee": "employee",
        "mitarbeiter": "employee",
        "personal": "employee",
        "product": "product",
        "produkt": "product",
        "produkte": "product",
        "artikel": "product",
        "location": "location",
        "standort": "location",
        "standorte": "location",
    }
    for key, cat in mapping.items():
        if key in stem_lower:
            return cat
    return "custom"


def _deduplicate(entities: list[dict]) -> list[dict]:
    """Entfernt Duplikate basierend auf (value_lower, category)."""
    seen = set()
    unique = []
    for e in entities:
        key = (e["value"].lower(), e.get("category", "custom"))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique
