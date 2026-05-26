"""Tests fuer den ERPConnector (Stammdaten-Import).

Testet: CSV-Import (Semikolon, UTF-8, Umlaute), Fehlerbehandlung,
Deduplizierung, Kategorie-Erkennung, JSON-Import und from_config.
"""

import json

from kiara.anonymization.erp_connector import ERPConnector

# ---------------------------------------------------------------------------
# CSV-Import: Standardfall
# ---------------------------------------------------------------------------


class TestCSVImport:
    def test_basic_csv_import(self, tmp_path):
        """Einfacher CSV-Import mit Semikolon-Trennung."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category\nK-1001;Mueller GmbH;customer\nK-1002;Schmidt AG;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert len(entities) == 2
        assert entities[0]["value"] == "Mueller GmbH"
        assert entities[0]["category"] == "customer"
        assert entities[0]["id"] == "K-1001"

    def test_csv_utf8_umlaute(self, tmp_path):
        """UTF-8 mit deutschen Umlauten."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category\n"
            "K-1;Muellers Baeckerei;customer\n"
            "K-2;Schroeder & Soehne;customer\n"
            "K-3;Buero fuer Uebergaenge;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert len(entities) == 3
        values = {e["value"] for e in entities}
        assert "Muellers Baeckerei" in values
        assert "Schroeder & Soehne" in values

    def test_csv_with_variants(self, tmp_path):
        """CSV mit Varianten-Spalte (Pipe-getrennt)."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category;variants\nK-1;Mueller GmbH;customer;Mueller|Mueller GmbH & Co\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        values = {e["value"] for e in entities}
        assert "Mueller GmbH" in values
        assert "Mueller" in values
        assert "Mueller GmbH & Co" in values

    def test_csv_multiple_files(self, tmp_path):
        """Import aus mehreren CSV-Dateien (Kunden + Lieferanten)."""
        customers = tmp_path / "kunden.csv"
        customers.write_text(
            "id;name;category\nK-1;Kunde A;customer\n",
            encoding="utf-8",
        )
        suppliers = tmp_path / "lieferanten.csv"
        suppliers.write_text(
            "id;name;category\nL-1;Lieferant B;supplier\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(
            customers_csv=str(customers),
            suppliers_csv=str(suppliers),
        )
        assert len(entities) == 2
        categories = {e["category"] for e in entities}
        assert "customer" in categories
        assert "supplier" in categories

    def test_csv_legacy_format(self, tmp_path):
        """Legacy-CSV-Format: key;type;value;variants."""
        csv_file = tmp_path / "stammdaten.csv"
        csv_file.write_text(
            "key;type;value;variants\nK-1;KUNDE;Mueller GmbH;Mueller\nL-1;LIEFERANT;Schmidt AG;\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        values = {e["value"] for e in entities}
        assert "Mueller GmbH" in values
        assert "Mueller" in values
        assert "Schmidt AG" in values

    def test_csv_empty_name_skipped(self, tmp_path):
        """Zeilen mit leerem Namen werden uebersprungen."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category\nK-1;;customer\nK-2;Gueltig;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert len(entities) == 1
        assert entities[0]["value"] == "Gueltig"


# ---------------------------------------------------------------------------
# Fehlerbehandlung
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_empty_csv(self, tmp_path):
        """Leere CSV-Datei (nur Header) ergibt leere Liste."""
        csv_file = tmp_path / "leer.csv"
        csv_file.write_text("id;name;category\n", encoding="utf-8")

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert entities == []

    def test_missing_file(self, tmp_path):
        """Nicht existierende Datei ergibt leere Liste (kein Fehler)."""
        entities = ERPConnector.from_csv(customers_csv=str(tmp_path / "gibts_nicht.csv"))
        assert entities == []

    def test_missing_columns_graceful(self, tmp_path):
        """CSV ohne erwartete Spalten wird toleriert."""
        csv_file = tmp_path / "kaputt.csv"
        csv_file.write_text(
            "spalte_a;spalte_b\nwert1;wert2\n",
            encoding="utf-8",
        )

        # Soll keinen Fehler werfen, einfach leere Liste
        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert isinstance(entities, list)

    def test_none_arguments(self):
        """Alle None-Argumente ergeben leere Liste."""
        entities = ERPConnector.from_csv()
        assert entities == []

    def test_directory_not_exists(self, tmp_path):
        """Nicht existierendes Verzeichnis ergibt leere Liste."""
        entities = ERPConnector.from_directory(str(tmp_path / "nicht_vorhanden"))
        assert entities == []

    def test_corrupt_csv_graceful(self, tmp_path):
        """Fehlerhafte CSV wird toleriert."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_bytes(b"\xff\xfe corrupted binary content")

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert isinstance(entities, list)


# ---------------------------------------------------------------------------
# Deduplizierung
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_exact_duplicates_removed(self, tmp_path):
        """Exakte Duplikate (gleicher Name, gleiche Kategorie) werden entfernt."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category\nK-1;Mueller GmbH;customer\nK-2;Mueller GmbH;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert len(entities) == 1

    def test_case_sensitive_dedup(self, tmp_path):
        """Deduplizierung ist case-insensitive (basiert auf value.lower())."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category\nK-1;Mueller GmbH;customer\nK-2;mueller gmbh;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert len(entities) == 1

    def test_different_category_not_deduped(self, tmp_path):
        """Gleicher Name in verschiedenen Kategorien wird NICHT dedupliziert."""
        csv_file = tmp_path / "kunden.csv"
        csv_file.write_text(
            "id;name;category\nK-1;Mueller;customer\nM-1;Mueller;employee\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        assert len(entities) == 2

    def test_dedup_across_files(self, tmp_path):
        """Duplikate ueber mehrere Dateien hinweg werden entfernt."""
        dir_path = tmp_path / "stammdaten"
        dir_path.mkdir()

        (dir_path / "kunden.csv").write_text(
            "id;name;category\nK-1;Mueller GmbH;customer\n",
            encoding="utf-8",
        )
        (dir_path / "kunden2.csv").write_text(
            "id;name;category\nK-2;Mueller GmbH;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        mueller = [e for e in entities if "mueller" in e["value"].lower()]
        assert len(mueller) == 1


# ---------------------------------------------------------------------------
# Kategorie-Erkennung aus Dateinamen
# ---------------------------------------------------------------------------


class TestCategoryFromFilename:
    def test_kunden_csv(self, tmp_path):
        """kunden.csv → Kategorie customer."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "kunden.csv").write_text(
            "id;name;category\nK-1;Test;customer\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "customer"

    def test_lieferanten_csv(self, tmp_path):
        """lieferanten.csv → Kategorie supplier."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "lieferanten.csv").write_text(
            "id;name\nL-1;Lieferant X\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "supplier"

    def test_mitarbeiter_csv(self, tmp_path):
        """mitarbeiter.csv → Kategorie employee."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "mitarbeiter.csv").write_text(
            "id;name\nM-1;Max Muster\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "employee"

    def test_produkte_csv(self, tmp_path):
        """produkte.csv → Kategorie product."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "produkte.csv").write_text(
            "id;name\nP-1;Widget Pro\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "product"

    def test_standorte_csv(self, tmp_path):
        """standorte.csv → Kategorie location."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "standorte.csv").write_text(
            "id;name\nO-1;Stuttgart\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "location"

    def test_artikel_csv(self, tmp_path):
        """artikel.csv → Kategorie product."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "artikel.csv").write_text(
            "id;name\nA-1;Schraube M8\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "product"

    def test_unknown_filename_custom(self, tmp_path):
        """Unbekannter Dateiname → Kategorie custom."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "sonstiges.csv").write_text(
            "id;name\nX-1;Unbekannt\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["category"] == "custom"

    def test_mixed_files_in_directory(self, tmp_path):
        """Mehrere Dateien mit verschiedenen Kategorien."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()

        (dir_path / "kunden.csv").write_text("id;name\nK-1;Firma A\n", encoding="utf-8")
        (dir_path / "lieferanten.csv").write_text("id;name\nL-1;Lieferant B\n", encoding="utf-8")
        (dir_path / "mitarbeiter.csv").write_text("id;name\nM-1;Hans Mueller\n", encoding="utf-8")

        entities = ERPConnector.from_directory(str(dir_path))
        categories = {e["category"] for e in entities}
        assert "customer" in categories
        assert "supplier" in categories
        assert "employee" in categories


# ---------------------------------------------------------------------------
# JSON-Import
# ---------------------------------------------------------------------------


class TestJSONImport:
    def test_json_list_format(self, tmp_path):
        """JSON als flache Liste von Entitaeten."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        json_file = dir_path / "kunden.json"
        data = [
            {"id": "K-1", "name": "Alpha GmbH", "category": "customer"},
            {"id": "K-2", "name": "Beta AG", "category": "customer"},
        ]
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 2
        values = {e["value"] for e in entities}
        assert "Alpha GmbH" in values
        assert "Beta AG" in values

    def test_json_entries_format(self, tmp_path):
        """JSON mit 'entries'-Schluessel."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        json_file = dir_path / "stammdaten.json"
        data = {
            "entries": [
                {"id": "K-1", "value": "Gamma OHG", "category": "customer"},
            ]
        }
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        entities = ERPConnector.from_directory(str(dir_path))
        assert len(entities) == 1
        assert entities[0]["value"] == "Gamma OHG"

    def test_json_empty_array(self, tmp_path):
        """Leeres JSON-Array ergibt leere Liste."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        json_file = dir_path / "leer.json"
        json_file.write_text("[]", encoding="utf-8")

        entities = ERPConnector.from_directory(str(dir_path))
        assert entities == []

    def test_json_corrupt_file(self, tmp_path):
        """Fehlerhafte JSON-Datei wird toleriert."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        json_file = dir_path / "kaputt.json"
        json_file.write_text("{invalid json", encoding="utf-8")

        entities = ERPConnector.from_directory(str(dir_path))
        assert isinstance(entities, list)

    def test_json_and_csv_combined(self, tmp_path):
        """CSV und JSON im selben Verzeichnis werden beide geladen."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()

        (dir_path / "kunden.csv").write_text("id;name\nK-1;CSV Kunde\n", encoding="utf-8")
        json_file = dir_path / "lieferanten.json"
        json_file.write_text(
            json.dumps([{"name": "JSON Lieferant", "category": "supplier"}]),
            encoding="utf-8",
        )

        entities = ERPConnector.from_directory(str(dir_path))
        values = {e["value"] for e in entities}
        assert "CSV Kunde" in values
        assert "JSON Lieferant" in values


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_csv_backend(self, tmp_path):
        """from_config mit backend=csv laedt aus csv_dir."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "kunden.csv").write_text("id;name\nK-1;Config Kunde\n", encoding="utf-8")

        config = {
            "anonymization": {
                "erp": {
                    "enabled": True,
                    "backend": "csv",
                    "csv_dir": str(dir_path),
                },
            },
        }

        entities = ERPConnector.from_config(config)
        assert len(entities) == 1
        assert entities[0]["value"] == "Config Kunde"

    def test_from_config_no_erp_section(self):
        """Config ohne erp-Sektion ergibt leere Liste."""
        config = {"anonymization": {}}
        entities = ERPConnector.from_config(config)
        assert entities == []

    def test_from_config_empty_csv_dir(self):
        """Config mit leerem csv_dir ergibt leere Liste."""
        config = {
            "anonymization": {
                "erp": {
                    "backend": "csv",
                    "csv_dir": "",
                },
            },
        }
        entities = ERPConnector.from_config(config)
        assert entities == []

    def test_from_config_unknown_backend(self):
        """Unbekanntes Backend ergibt leere Liste (kein Fehler)."""
        config = {
            "anonymization": {
                "erp": {
                    "backend": "unknown",
                },
            },
        }
        entities = ERPConnector.from_config(config)
        assert entities == []

    def test_from_config_legacy_format(self, tmp_path):
        """Legacy-Config-Format (erp_source statt erp) funktioniert."""
        dir_path = tmp_path / "erp"
        dir_path.mkdir()
        (dir_path / "kunden.csv").write_text("id;name\nK-1;Legacy Kunde\n", encoding="utf-8")

        config = {
            "anonymization": {
                "erp_source": {
                    "enabled": True,
                    "type": "csv",
                    "path": str(dir_path),
                },
            },
        }

        entities = ERPConnector.from_config(config)
        assert len(entities) == 1
        assert entities[0]["value"] == "Legacy Kunde"

    def test_from_config_rest_backend(self):
        """REST-Backend gibt leere Liste zurueck (nicht implementiert)."""
        config = {
            "anonymization": {
                "erp": {
                    "backend": "rest",
                    "api_url": "http://localhost/api",
                },
            },
        }
        entities = ERPConnector.from_config(config)
        assert entities == []

    def test_from_config_sql_backend(self):
        """SQL-Backend gibt leere Liste zurueck (nicht implementiert)."""
        config = {
            "anonymization": {
                "erp": {
                    "backend": "sql",
                    "connection_string": "sqlite:///test.db",
                },
            },
        }
        entities = ERPConnector.from_config(config)
        assert entities == []


# ---------------------------------------------------------------------------
# Kategorie-Normalisierung
# ---------------------------------------------------------------------------


class TestCategoryNormalization:
    def test_german_type_names(self, tmp_path):
        """Deutsche Typbezeichnungen werden auf englische Kategorien gemappt."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(
            "id;name;category\n"
            "1;Firma A;KUNDE\n"
            "2;Firma B;LIEFERANT\n"
            "3;Person C;MITARBEITER\n"
            "4;Ding D;PRODUKT\n"
            "5;Ort E;STANDORT\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        cats = {e["value"]: e["category"] for e in entities}
        assert cats["Firma A"] == "customer"
        assert cats["Firma B"] == "supplier"
        assert cats["Person C"] == "employee"
        assert cats["Ding D"] == "product"
        assert cats["Ort E"] == "location"

    def test_english_type_names(self, tmp_path):
        """Englische Typbezeichnungen werden ebenfalls erkannt."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(
            "id;name;category\n1;Firma A;CUSTOMER\n2;Firma B;SUPPLIER\n3;Person C;EMPLOYEE\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))
        cats = {e["value"]: e["category"] for e in entities}
        assert cats["Firma A"] == "customer"
        assert cats["Firma B"] == "supplier"
        assert cats["Person C"] == "employee"

    def test_generic_org_type_is_not_customer(self, tmp_path):
        """ORG aus Legacy-Stammdaten bleibt eine neutrale Organisation."""
        csv_file = tmp_path / "stammdaten.csv"
        csv_file.write_text(
            "key;type;value;variants\nO-1;ORG;Eigene Firma GmbH;\n",
            encoding="utf-8",
        )

        entities = ERPConnector.from_csv(customers_csv=str(csv_file))

        assert entities == [
            {
                "value": "Eigene Firma GmbH",
                "category": "organization",
                "id": "O-1",
            }
        ]

    def test_file_defaults_stay_customer_and_supplier(self, tmp_path):
        """CSV-Dateirollen setzen weiter spezifische Defaults ohne Typ-Spalte."""
        customers = tmp_path / "kunden.csv"
        customers.write_text("id;name\nK-1;Kunde GmbH\n", encoding="utf-8")
        suppliers = tmp_path / "lieferanten.csv"
        suppliers.write_text("id;name\nL-1;Lieferant AG\n", encoding="utf-8")

        entities = ERPConnector.from_csv(
            customers_csv=str(customers),
            suppliers_csv=str(suppliers),
        )

        cats = {e["value"]: e["category"] for e in entities}
        assert cats["Kunde GmbH"] == "customer"
        assert cats["Lieferant AG"] == "supplier"
