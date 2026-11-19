"""ERP-Stammdaten fuer die Anonymisierung: fail-closed bei fehlender/veralteter Datei, Cache."""

import os
import time

import pytest

from kiara.anonymization import stammdaten

HEADER = "key;type;value;variants\n"


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(stammdaten, "_automaton_cache", {"key": None, "automaton": None})


def _config(path, **erp):
    return {
        "anonymization": {
            "erp_source": {"enabled": True, "type": "csv", "path": str(path), **erp},
            "ldap_source": {"enabled": False},
        }
    }


def _write(path, rows, *, age_hours=0):
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))
    return path


def test_missing_directory_and_empty_directory_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="Pfad fehlt"):
        stammdaten.erp_source_signature({"path": str(tmp_path / "fehlt")})
    with pytest.raises(RuntimeError, match="Keine Stammdaten-Datei"):
        stammdaten.erp_source_signature({"path": str(tmp_path)})


def test_stale_file_fails_closed_unless_age_check_disabled(tmp_path):
    _write(tmp_path / "parity-partner.csv", ["T-1;ORG;Testfirma Beispiel GmbH;\n"], age_hours=50)

    with pytest.raises(RuntimeError, match="veraltet"):
        stammdaten.erp_source_signature({"path": str(tmp_path)})
    assert stammdaten.erp_source_signature({"path": str(tmp_path), "max_age_hours": 72})
    assert stammdaten.erp_source_signature({"path": str(tmp_path), "max_age_hours": 0})


def test_tmp_upload_is_ignored(tmp_path):
    (tmp_path / "parity-partner.csv.tmp").write_text(HEADER + "T-1;ORG;Halbfertig GmbH;\n")
    with pytest.raises(RuntimeError, match="Keine Stammdaten-Datei"):
        stammdaten.erp_source_signature({"path": str(tmp_path)})


def test_file_without_valid_rows_fails_closed(tmp_path):
    _write(tmp_path / "parity-partner.csv", [";ORG;;\n"])
    with pytest.raises(RuntimeError, match="ohne gueltige Zeilen"):
        stammdaten._load_erp_stammdaten({"path": str(tmp_path)})


def test_unreadable_json_fails_closed(tmp_path):
    _write(tmp_path / "a.csv", ["T-1;ORG;Testfirma Beispiel GmbH;\n"])
    (tmp_path / "b.json").write_text("{kaputt", encoding="utf-8")
    with pytest.raises(RuntimeError, match="nicht lesbar"):
        stammdaten._load_erp_stammdaten({"path": str(tmp_path)})


def test_build_uses_csv_entries_and_variants(tmp_path):
    _write(
        tmp_path / "parity-partner.csv",
        [
            "ans-1;ORG;Testfirma Beispiel GmbH;Testfirma\n",
            "anp-2;PERSON;Erika Musterfrau;Musterfrau, Erika|Musterfrau\n",
            "anp-2-mail;EMAIL;erika.musterfrau@example.org;\n",
        ],
    )

    automaton = stammdaten.build_automaton_from_sources(_config(tmp_path))

    assert automaton.source_counts["erp"] == 6
    hits = {m["entry"].entity_type for m in automaton.scan("Frau Musterfrau von Testfirma")}
    assert hits == {"PERSON", "ORG"}


def test_build_fails_closed_with_stale_export(tmp_path):
    _write(tmp_path / "parity-partner.csv", ["T-1;ORG;Testfirma Beispiel GmbH;\n"], age_hours=49)
    with pytest.raises(RuntimeError, match="ERP-Stammdaten"):
        stammdaten.build_automaton_from_sources(_config(tmp_path))


def test_cached_automaton_reuses_and_rebuilds_on_new_export(tmp_path):
    export = _write(tmp_path / "parity-partner.csv", ["T-1;ORG;Testfirma Beispiel GmbH;\n"])
    config = _config(tmp_path)

    first = stammdaten.cached_automaton(config)
    assert stammdaten.cached_automaton(config) is first

    _write(export, ["T-1;ORG;Testfirma Beispiel GmbH;\n", "T-2;ORG;Zweite Testfirma AG;\n"])
    second = stammdaten.cached_automaton(config)
    assert second is not first
    assert second.source_counts["erp"] == 2


def test_cached_automaton_checks_age_on_cache_hit(tmp_path):
    export = _write(tmp_path / "parity-partner.csv", ["T-1;ORG;Testfirma Beispiel GmbH;\n"])
    config = _config(tmp_path)
    stammdaten.cached_automaton(config)

    old = time.time() - 49 * 3600
    os.utime(export, (old, old))
    with pytest.raises(RuntimeError, match="veraltet"):
        stammdaten.cached_automaton(config)
