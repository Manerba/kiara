#!/usr/bin/env python3
"""DB_FLAGS, getrennte Schutz-/Bibliotheksdaten und Materialinventar lesen.

  python -I /absolut/analyse/metadaten.py MODELL.pkg [--details]

Die Standardausgabe enthaelt nur Klassen, Feldformen und Zaehler. --details
ergaenzt lokale IDs, Bibliothekskennungen und Materialtexte und darf bei
Kundenpaketen nicht veroeffentlicht werden. Kein Schreibzugriff.
"""
import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altformat import oeffnen  # noqa: E402
from pkgleser import benutzerattribute, ist_zahl, segment_lesen, segmentkoepfe, zahl  # noqa: E402
from analyse.protokollfelder import Tabellenfehler, tabelle_lesen  # noqa: E402
from analyse.materialschluessel import lesen as materialschluessel_lesen  # noqa: E402

KLASSEN = {"DB_FLAGS", "mbu_material", "GA_LW_MATERIAL", "PD_ATTR0", "PD_ATTR_SP_STRUCT0"}
BELEGE = {
    "db_flags": "Creo 20.3, 1.11.2026: Teil-Protected-/Library-Aktionen und isolierte Einbyte-Gegenproben; 12.11.2026: neutrale Baugruppe, unabhaengige 0/2/4/6-Gegenproben vor/nach Save",
    "schutz": "Integration Kit reference/db_attributes.html; separate UDEF-Ruecklesung in derselben Referenz",
    "bibliothek": "Creo 20.3, 3.11.2026: sd-inq-obj-symb-name/-version; sd-set-library-attr mit optionaler symbolischer Version; getrennte ID-/Name-/Versions-Gegenproben; INGEST_FELDER.md",
    "bibliothek_alt": "Creo 20.3, 12.11.2026: SD11/SD14 Einzeltext-Gegenproben und native symbolische Getter; positive CoCreate-2008-Originale mit Resave; INGEST_FELDER.md",
    "material": "Creo 20.3, 1.11.2026: 56 Material-Inhaltsvorkommen, 112 API-Feldvergleiche; 7 Inhalte unter MM/INCH/MM unveraendert; reference/sheetmetal_inq.html und util_misc.html; Schluesselgrammatik: sheetmetal_inq.html und SheetAdvisor/sha_demoshop.lsp",
    "render_material": "Hier nur Inventar; belegte Vorkommens-/Inhalts-Darstellungswahl separat in analyse/rendering.py; reference/inq_part.html",
}


def _felder(obj):
    return [(i, t) for i, t in enumerate(obj.felder) if t.art not in ("s", "8002")]


def _ort(pair):
    i, t = pair
    return {"feld": i, "offset": t.offset}


def _ziel(seg, nr):
    obj = seg.objekte.get(nr)
    return {"nr": nr, "klasse": obj.klasse if obj else None, "id": obj.id if obj else None}


def db_flags(seg, obj):
    """Exakte Form B1 v:part_struct/asbm_struct D; unabhaengige Bits 0x2/0x4.

    0x4 wurde durch 4->0 und 6->2 gegen contents-read-only-p isoliert.
    0x2 wurde durch 6->4 und Nichtbibliothek 0->2 gegen
    contents-modified-p(:check-database T) isoliert. Die normale Modified-
    Inquiry bleibt nach Laden NIL. Fuer asbm_struct bestaetigen isolierte
    0/2/4/6-Proben dieselben unabhaengigen Rollen; Protected bleibt dabei T
    und Library NIL. Protected und Bibliotheksstatus sind
    keine Bits dieser Dekodierung. Weitere Bits bleiben sichtbar.
    """
    fs = _felder(obj)
    valid = (obj.klasse == "DB_FLAGS" and len(fs) == 3
             and fs[0][1].art == "B" and fs[0][1].wert == 1
             and fs[1][1].art == "v" and isinstance(fs[1][1].wert, int)
             and fs[2][1].art == "D" and isinstance(fs[2][1].wert, int)
             and fs[2][1].wert >= 0)
    if not valid:
        return {"schema_erkannt": False}
    owner = _ziel(seg, fs[1][1].wert)
    value = fs[2][1].wert
    result = {"schema_erkannt": True, "besitzer": owner, "roh": value,
              "B1_roh": fs[0][1].wert,
              "felder": {"B1_roh": _ort(fs[0]), "besitzer": _ort(fs[1]), "flags": _ort(fs[2])},
              "unbekannte_bits": value & ~0x6, "beleg": BELEGE["db_flags"]}
    # Andere Inhaltsarten, insbesondere wp_struct, sind nicht mitfreigegeben.
    if owner["klasse"] in ("part_struct", "asbm_struct"):
        result.update(inhalt_schreibgeschuetzt=bool(value & 0x4), datenbank_geaendert=bool(value & 0x2))
    else:
        result["besitzersemantik"] = "unbekannt"
    return result


def bibliotheksmarker(obj):
    """Eigene Kennzeichnung direkt nach dem Zeitstempel, keine Fremddeklaration.

    8031 c Kennung 8032 c Zusatz steht im part_struct zwischen Zeitstempel
    und Vorkommensdaten. Dieselben Marken koennen in eingebetteten
    Inhaltsdeklarationen eines npart3 stehen; sie sind dort kein eigener
    Bibliotheksstatus des Vorkommens. 8032 ist die symbolische Version;
    ihre interne Kodierung wird nicht in eine numerische Revision zerlegt.
    """
    unknown = {"vorhanden": None, "eigener_inhalt_geprueft": False,
               "grund": "part_struct-Headerform nicht unterstuetzt"}
    if obj.klasse != "part_struct":
        return unknown
    fs = _felder(obj)
    try:
        table = tabelle_lesen(obj)
    except Tabellenfehler:
        return unknown
    if table is None:
        return unknown
    end = table["ende"]
    header = [p[1].art for p in fs[end:end + 3]]
    if header == ["v", "v", "F"]:
        tail = fs[end + 3:]
        if tail and tail[0][1].art == "B":
            return {"vorhanden": False, "eigener_inhalt_geprueft": True}
        if (len(tail) >= 5 and [p[1].art for p in tail[:5]] == ["8031", "c", "8032", "c", "B"]
                and type(tail[1][1].wert) is str and type(tail[3][1].wert) is str):
            return {"vorhanden": True, "eigener_inhalt_geprueft": True,
                    "kennung": tail[1][1].wert, "symbolische_version": tail[3][1].wert,
                    "zusatz_8032_roh": tail[3][1].wert,
                    "felder": {"kennung": _ort(tail[1]), "zusatz": _ort(tail[3]),
                               "symbolische_version": _ort(tail[3])},
                    "beleg": BELEGE["bibliothek"]}
        if (len(tail) >= 3 and [p[1].art for p in tail[:3]] == ["8032", "c", "B"]
                and type(tail[1][1].wert) is str and tail[1][1].wert):
            # Native H1 originals have NIL symbolic name, but a real version.
            # Preserve the version without classifying the content as a library part.
            return {"vorhanden": False, "eigener_inhalt_geprueft": True,
                    "version_ohne_kennung": True, "form": "version_ohne_8031",
                    "symbolische_version": tail[1][1].wert, "zusatz_8032_roh": tail[1][1].wert,
                    "felder": {"zusatz": _ort(tail[1]), "symbolische_version": _ort(tail[1])},
                    "beleg": "Creo 20.3, 12.11.2026: 15 Inhalte aus drei H1-Originalen, symbolischer Name NIL, "
                             "symbolische Version gleich 8032-Text; Getter vor/nach Resave; INGEST_FELDER.md"}
        form, missing = "soliddesigner_vvf", "o"
    elif header[:2] == ["v", "F"]:
        tail = fs[end + 2:]
        form, missing = "soliddesigner_vf", "I"
    else:
        return unknown
    # Older layouts store the two values directly, without 8031/8032.
    # I is altformat's local analysis alias of physical o, never a global rule.
    # Native single-name/single-version probes distinguish these slots. The
    # following B and all further occurrence data retain their existing meaning.
    if (len(tail) < 3 or tail[2][1].art != "B" or type(tail[2][1].wert) is not int
            or tail[2][1].wert < 0):
        return unknown
    fields = {"kennung": _ort(tail[0]), "zusatz": _ort(tail[1]), "symbolische_version": _ort(tail[1])}
    if all(p[1].art == missing and p[1].wert is None for p in tail[:2]):
        return {"vorhanden": False, "eigener_inhalt_geprueft": True, "form": form,
                "felder": fields, "beleg": BELEGE["bibliothek_alt"]}
    if all(p[1].art == "c" and type(p[1].wert) is str and p[1].wert for p in tail[:2]):
        return {"vorhanden": True, "eigener_inhalt_geprueft": True, "form": form,
                "kennung": tail[0][1].wert, "symbolische_version": tail[1][1].wert,
                "zusatz_8032_roh": tail[1][1].wert, "felder": fields, "beleg": BELEGE["bibliothek_alt"]}
    # Mixed missing/text pairs and empty library texts lack old-format native probes.
    return unknown


def schutz_und_bibliothek(seg, owner):
    """Protected-UDEF und Bibliotheksdaten getrennt vom DB_FLAGS-Wort."""
    attrs = benutzerattribute(seg, owner)
    protected = attrs.get("DB-PROTECTED")
    values = [v for k, v in protected.get("werte", []) if k == "PROTECTED"] if protected else []
    value = False if protected is None else None
    if len(values) == 1:
        value = {"T": True, "NIL": False}.get(values[0])
    return {"besitzer": _ziel(seg, owner.nr),
            "protected_attribut_vorhanden": protected is not None, "protected_wert": value,
            "protected_attribut_roh": protected, "bibliotheksmarker": bibliotheksmarker(owner),
            "sollib_info_roh": attrs.get("SOLLIB-INFO"), "beleg": BELEGE["schutz"]}


def material(seg, obj):
    """Blechmaterial: dokumentierter Werkzeugschluessel und Dicke in mm.

    c/N stimmen in 56 Inhaltsvorkommen mit der nativen Material-Inquiry
    ueberein. Sieben davon bleiben bei MM -> INCH -> MM unveraendert,
    waehrend die Kontrollumrechnung 1 -> 1/25.4 -> 1 wechselt.
    Die dokumentierte SHOPNAME/TABNAME/ROWKEY-Form wird als Daten gelesen;
    B1 und weitere Schluesselformen bleiben offen. Keine Werkstoffableitung.
    Render-Material wird weiter nur strukturell inventarisiert.
    """
    fs = _felder(obj)
    ts = [t for _, t in fs]
    if obj.klasse == "mbu_material":
        valid = (len(ts) == 5 and [t.art for t in ts[:4]] == ["ID", "B", "v", "c"]
                 and ts[1].wert == 1 and isinstance(ts[2].wert, int) and ist_zahl(ts[4])
                 and math.isfinite(zahl(ts[4])))
        if not valid:
            return {"schema_erkannt": False}
        owner = _ziel(seg, ts[2].wert)
        result = {"schema_erkannt": True, "besitzer": owner, "B1_roh": ts[1].wert,
                  "text_roh": ts[3].wert, "zahl_roh": zahl(ts[4]),
                  "felder": {"B1_roh": _ort(fs[1]), "besitzer": _ort(fs[2]),
                             "text": _ort(fs[3]), "zahl": _ort(fs[4])},
                  "offen": ["B1", "Weitere Materialschluesselformen und externe Katalogeigenschaften",
                            "Gueltigkeitsgrenzen der Blechdicke"],
                  "beleg": BELEGE["material"]}
        if owner["klasse"] == "body":
            result.update(materialwerkzeugschluessel=ts[3].wert, blechdicke_mm=zahl(ts[4]))
            result["materialschluessel"] = materialschluessel_lesen(ts[3].wert)
        else:
            result["besitzersemantik"] = "unbekannt"
        return result
    if obj.klasse == "GA_LW_MATERIAL":
        if len(ts) != 2 or [t.art for t in ts] != ["c", "B"]:
            return {"schema_erkannt": False}
        return {"schema_erkannt": True, "text_roh": ts[0].wert, "flag_roh": ts[1].wert,
                "felder": {"text": _ort(fs[0]), "flag": _ort(fs[1])},
                "hinweis": "Render-Materialbezug roh; B nur mit Halterkontext aus analyse/rendering.py deuten", "beleg": BELEGE["render_material"]}
    return {"schema_erkannt": False}


def lesen(pfad, details=False):
    counts, shapes, dbvalues, statecounts = (collections.Counter() for _ in range(4))
    records, warnings = [], []
    unknown_shapes = unknown_bits = unknown_owners = segments = 0
    for entry, data in oeffnen(pfad):
        if not data.startswith(b"Rz"):
            continue
        for start, _, _, _ in segmentkoepfe(data):
            seg = segment_lesen(data, start, entry, warnings)
            if seg is None:
                continue
            segments += 1
            local = {"eintrag": entry, "segment": start, "db_flags": [], "inhalte": [], "material": []}
            for obj in seg.objekte.values():
                if obj.klasse in KLASSEN:
                    counts[obj.klasse] += 1
                    shape = tuple("N" if ist_zahl(t) else t.art for _, t in _felder(obj))
                    shapes[(obj.klasse, shape)] += 1
                if obj.klasse == "DB_FLAGS":
                    rec = db_flags(seg, obj)
                    unknown_shapes += not rec["schema_erkannt"]
                    if rec["schema_erkannt"]:
                        dbvalues[rec["roh"]] += 1
                        unknown_bits += bool(rec["unbekannte_bits"])
                        unknown_owners += "besitzersemantik" in rec
                    local["db_flags"].append({"nr": obj.nr, **rec})
                elif obj.klasse == "part_struct":
                    rec = schutz_und_bibliothek(seg, obj)
                    statecounts["protected_attribute"] += rec["protected_attribut_vorhanden"]
                    statecounts["protected_werte_unbekannt"] += rec["protected_wert"] is None
                    statecounts["bibliotheksmarker"] += rec["bibliotheksmarker"]["vorhanden"] is True
                    if rec["bibliotheksmarker"].get("version_ohne_kennung"):
                        statecounts["symbolische_version_ohne_kennung"] += 1
                    statecounts["bibliotheksformen_unbekannt"] += not rec["bibliotheksmarker"]["eigener_inhalt_geprueft"]
                    statecounts["sollib_attribute"] += rec["sollib_info_roh"] is not None
                    if (rec["protected_attribut_vorhanden"] or rec["bibliotheksmarker"]["vorhanden"]
                            or rec["bibliotheksmarker"].get("version_ohne_kennung")
                            or rec["sollib_info_roh"] or not rec["bibliotheksmarker"]["eigener_inhalt_geprueft"]):
                        local["inhalte"].append(rec)
                elif obj.klasse in ("mbu_material", "GA_LW_MATERIAL"):
                    rec = material(seg, obj)
                    unknown_shapes += not rec["schema_erkannt"]
                    local["material"].append({"nr": obj.nr, "klasse": obj.klasse, **rec})
            if details and any(local[k] for k in ("db_flags", "inhalte", "material")):
                records.append(local)
    report = {"segmente": segments, "klassen": dict(sorted(counts.items())),
              "feldformen": [{"klasse": c, "form": list(s), "anzahl": n} for (c, s), n in sorted(shapes.items())],
              "db_rohwerte": dict(sorted(dbvalues.items())), "getrennte_inhaltsmerkmale": dict(statecounts),
              "unbekannte_feldformen": unknown_shapes, "db_objekte_mit_unbekannten_bits": unknown_bits,
              "db_objekte_mit_unbelegtem_besitzer": unknown_owners,
              "belege": BELEGE, "offen": ["DB_FLAGS ausser 0x2/0x4", "Interne Teilkodierung der symbolischen Bibliotheksversion",
                "B1 der Blechmaterialklasse", "Weitere Materialschluesselformen und externe Katalogeigenschaften",
                "Rendering-Kontextauswertung separat in analyse/rendering.py",
                "Weitere STEP-Objektformen und konstante Restfelder"], "warnungen": len(warnings)}
    if details:
        report["details_lokal"] = {"segmente": records, "warnungen": warnings}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pakete", nargs="+")
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args(argv)
    reports = [lesen(p, args.details) for p in args.pakete]
    print(json.dumps(reports, indent=2, ensure_ascii=True, allow_nan=False))
    return int(any(r["warnungen"] or r["unbekannte_feldformen"] for r in reports))


if __name__ == "__main__":
    sys.exit(main())
