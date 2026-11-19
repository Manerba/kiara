#!/usr/bin/env python3
"""Gemeinsame D-Tabellenköpfe lesen und ihre Semantik gegen PKG prüfen (371).

Grammatik direkt nach der Objekt-ID:
    D Fachzahl, D Tabellenparameter, D Eintragszahl,
    Eintragszahl * (D Fach, v Ziel | NIL).

Auch leere Tabellen gehören dazu: D0 D0 D0 und D1 D0 D0. Einträge zählen
ausdrücklich NIL-Werte mit. Der zweite D-Wert ist NICHT eindeutig durch die
Fachzahl bestimmt: z.B. 0/1/0, 0/2/0, 3/0/2 und 7/1/3 sind beobachtet.
Die ältere Bezeichnung Größenstufe ist deshalb keine bewiesene Semantik.
Fachzahl ist keine Eintragskapazität; gleiche Fächer dürfen mehrfach vorkommen.
Hashfunktion, Tabellenparameter und Bedeutung einzelner NIL-Plätze bleiben offen.

Standardbericht: nur Klassen, Protokollwerte und Zähler. --details enthält
zusätzlich lokale Fundstellen und Schlüsselsymbole; nicht veröffentlichen.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from altformat import oeffnen  # noqa: E402
from pkgleser import segment_lesen, segmentkoepfe  # noqa: E402


class Tabellenfehler(ValueError):
    pass


def felder(obj):
    """Symbol-/UUID-Deklarationen sind keine Objektfelder."""
    return [t for t in obj.felder if t.art not in ("s", "8002")]


def tabelle_lesen(obj):
    """None ohne D-Präfix; sonst vollständig begrenzter Kopf plus Einträge.

    ende ist der Index des ersten Nutzfeldes hinter der Tabelle in der von
    Symbol-/UUID-Deklarationen bereinigten Feldliste. Keine heuristische Suche
    nach dem nächsten Marker; genau Eintragszahl Paare werden konsumiert.
    """
    f = felder(obj)
    if len(f) < 2 or f[0].art != "ID" or f[1].art != "D":
        return None
    if len(f) < 4 or any(t.art != "D" for t in f[1:4]):
        raise Tabellenfehler("D-Präfix ohne vollständigen Tabellenkopf")
    groesse, stufe, anzahl = (t.wert for t in f[1:4])
    if any(type(x) is not int or x < 0 for x in (groesse, stufe, anzahl)):
        raise Tabellenfehler("Negative oder nicht-ganzzahlige Kopfwerte")
    if anzahl > (len(f)-4)//2:
        raise Tabellenfehler("Eintragszahl überschreitet die verfügbaren Feldpaare")
    out, vorher = [], -1
    for i in range(anzahl):
        fach, wert = f[4+2*i:6+2*i]
        if fach.art != "D" or type(fach.wert) is not int or not 0 <= fach.wert < groesse:
            raise Tabellenfehler("Fach liegt nicht innerhalb der Fachzahl")
        if fach.wert < vorher:
            raise Tabellenfehler("Einträge stehen nicht in aufsteigender Fachreihenfolge")
        # Zeigerform (CoCreate 2008/17.0): ein v ohne Bit 31, dessen Wert kein Objekt des Segments trifft, bleibt
        # ('roh', x). Es entspricht dem in 20.3 ebenfalls ziellosen Indexverweis (Pilotmodell: alle 34 A3D_TAG-
        # Tabellenwerte ohne lokales Ziel); daher als unaufgeloester Verweis zulaessig, Ziel None, Rohwert erhalten.
        roh = (wert.art == "v" and type(wert.wert) is tuple and len(wert.wert) == 2 and wert.wert[0] == "roh"
               and type(wert.wert[1]) is int and wert.wert[1] > 0)
        if not (wert.art == "NIL" and wert.wert == 1) and not roh and not (
                wert.art == "v" and type(wert.wert) is int and wert.wert >= 0):
            raise Tabellenfehler("Tabellenwert ist weder lokaler Verweis noch NIL")
        vorher = fach.wert
        eintrag = {"fach": fach.wert, "art": wert.art,
                   "ziel": wert.wert if wert.art == "v" and not roh else None,
                   "feld_fach": 4+2*i, "feld_wert": 5+2*i}
        if roh:
            eintrag["roh_verweis"] = wert.wert[1]
        out.append(eintrag)
    # stufe is retained as a historical spelling for local analysis consumers;
    # it has precisely the same unproven meaning as parameter, not a size index.
    return {"faecher": groesse, "parameter": stufe, "stufe": stufe, "anzahl": anzahl,
            "ende": 4+2*anzahl, "eintraege": out}


def feldrollen(obj):
    """Semantik nur der validierten Präfixfelder; keine Aussage zum Restobjekt.

    NIL-Platz und Tabellenparameter bleiben ausdrücklich unbekannt, obwohl der
    Parser sie begrenzen kann. Diese Trennung ist für das Wissensregister nötig.
    Die Feldindizes beziehen sich wie ende auf felder(obj), nicht rohe Token.
    """
    tab = tabelle_lesen(obj)
    if tab is None:
        return []
    out = [{"index": 0, "status": "belegt", "rolle": "Objektidentitaet"},
           {"index": 1, "status": "belegt", "rolle": "Fachzahl, obere exklusive Grenze der Fachnummern"},
           {"index": 2, "status": "unbekannt", "rolle": "Tabellenparameter; keine eindeutige Groessenstufe"},
           {"index": 3, "status": "belegt", "rolle": "Anzahl serialisierter Paare einschliesslich NIL"}]
    for e in tab["eintraege"]:
        out.append({"index": e["feld_fach"], "status": "belegt", "rolle": "Fachnummer; Kollisionen erlaubt"})
        out.append({"index": e["feld_wert"], "status": "belegt" if e["art"] == "v" else "unbekannt",
                    "rolle": "Objektverweis im Fach" if e["art"] == "v" else "NIL-Platz; Ursprung/Funktion offen"})
    return out


def schluessel(seg, nr):
    """Candidate key: ATTRIBUTE_LINK symbol, otherwise target class.

    This identifies a candidate key only. It does not claim to know the hash
    function. Missing objects/symbols are left unresolved, never guessed.
    """
    obj = seg.objekte.get(nr)
    if obj is None:
        return None
    if obj.klasse == "ATTRIBUTE_LINK":
        symbole = [t.wert for t in felder(obj) if t.art == "z"]
        return symbole[0] if len(symbole) == 1 else None
    return obj.klasse


class Untersuchung:
    def __init__(self, details=False):
        self.details = details
        self.zaehler = collections.Counter()
        self.klassen = collections.defaultdict(collections.Counter)
        self.kopfpaare = collections.Counter()
        self.kopf_formate = collections.Counter()
        self.eintragspaare = collections.Counter()
        self.schluesselfaecher = collections.defaultdict(collections.Counter)
        self.wertklassen, self.fehler = collections.Counter(), collections.Counter()
        self.fundstellen = []
        self.rollenzaehler = collections.Counter()
        self.paket_schluessel = collections.defaultdict(collections.Counter)
        self.format = "unbestimmt"

    def segment(self, seg, ort=None):
        for obj in seg.objekte.values():
            self.zaehler["objekte"] += 1
            try:
                tab = tabelle_lesen(obj)
            except Tabellenfehler as exc:
                self.zaehler["nicht_erklaerte_D_praefixe"] += 1
                self.klassen[obj.klasse]["nicht_erklaert"] += 1
                self.fehler[str(exc)] += 1
                if self.details:
                    self.fundstellen.append({"ort": ort, "objekt": obj.nr, "klasse": obj.klasse, "fehler": str(exc),
                                             "praefix": [(t.art, t.wert) for t in felder(obj)[:16]]})
                continue
            if tab is None:
                self.zaehler["ohne_D_praefix"] += 1
                continue
            k = self.klassen[obj.klasse]
            k["tabellen"] += 1
            k["leere_tabellen"] += tab["anzahl"] == 0
            self.zaehler["tabellen"] += 1
            self.zaehler["tabelleneintraege"] += tab["anzahl"]
            self.zaehler["tabellen_mit_mehr_eintraegen_als_faechern"] += tab["anzahl"] > tab["faecher"]
            self.rollenzaehler["belegt"] += 3 + tab["anzahl"] + sum(e["art"] == "v" for e in tab["eintraege"])
            self.rollenzaehler["unbekannt"] += 1 + sum(e["art"] == "NIL" for e in tab["eintraege"])
            self.kopfpaare[tab["faecher"], tab["stufe"]] += 1
            self.kopf_formate[self.format, tab["faecher"], tab["stufe"]] += 1
            self.eintragspaare[tab["faecher"], tab["stufe"], tab["anzahl"]] += 1
            for e in tab["eintraege"]:
                if e["art"] == "NIL":
                    self.zaehler["NIL_eintraege"] += 1
                    k["NIL_eintraege"] += 1
                    continue
                ziel = seg.objekte.get(e["ziel"])
                self.zaehler["verweise"] += 1
                if ziel is None:
                    self.zaehler["verweise_nicht_lokal_aufgeloest"] += 1
                    k["verweise_nicht_lokal_aufgeloest"] += 1
                    continue
                self.wertklassen[ziel.klasse] += 1
                if obj.nr in ziel.verweise():
                    self.zaehler["verweise_mit_besitzerrueckbezug"] += 1
                else:
                    k["verweise_ohne_besitzerrueckbezug"] += 1
                key = schluessel(seg, e["ziel"])
                if key is not None:
                    self.schluesselfaecher[key, tab["faecher"]][e["fach"]] += 1
                    self.paket_schluessel[self.zaehler["pakete"], key, tab["faecher"]][e["fach"]] += 1
            # One example per class/shape gives reproducible local evidence
            # without retaining an entire customer model in the report.
            if self.details and k["tabellen"] == 1:
                self.fundstellen.append({"ort": ort, "objekt": obj.nr, "klasse": obj.klasse, "tabelle": tab,
                                         "danach": [t.art for t in felder(obj)[tab["ende"]:tab["ende"]+6]]})

    def paket(self, pfad):
        self.zaehler["pakete"] += 1
        self.format = "ZIP" if zipfile.is_zipfile(pfad) else "SD11_normalisiert"
        for eintrag, daten in oeffnen(pfad):
            if not daten.startswith(b"Rz"):
                continue
            for start, _, _, _ in segmentkoepfe(daten):
                warn = []
                seg = segment_lesen(daten, start, eintrag, warn)
                if warn or seg is None:
                    self.fehler["Segment nicht vollständig lesbar"] += 1
                    continue
                self.zaehler["segmente"] += 1
                self.segment(seg, {"paket": str(Path(pfad).resolve()), "eintrag": eintrag, "segment": start}
                             if self.details else None)

    def bericht(self):
        konflikte = sum(len(faecher) != 1 for faecher in self.schluesselfaecher.values())
        return {"grammatik": "ID D Fachzahl D Parameter D Eintragszahl (D Fach (v Ziel | NIL))*Eintragszahl",
                "geltung": "Nur gemeinsame ID-D-Praefixe; eingebettete Sekundaertabellen und Restfelder nicht erfasst.",
                "praefix_feldsemantik": dict(sorted(self.rollenzaehler.items())),
                "zaehler": dict(sorted(self.zaehler.items())),
                "klassen": {k: dict(sorted(v.items())) for k, v in sorted(self.klassen.items())},
                "tabellenparameter": [{"faecher": a, "parameter": b, "tabellen": n} for (a, b), n in sorted(self.kopfpaare.items())],
                "parameter_nach_format": [{"format": a, "faecher": b, "parameter": c, "tabellen": n}
                                          for (a, b, c), n in sorted(self.kopf_formate.items())],
                "eintragszahlen": [{"faecher": a, "parameter": b, "eintraege": c, "tabellen": n}
                                   for (a, b, c), n in sorted(self.eintragspaare.items())],
                "wertklassen": dict(sorted(self.wertklassen.items())),
                "schluessel_groessen_paare": len(self.schluesselfaecher),
                "schluessel_groessen_mit_mehreren_faechern": konflikte,
                "schluessel_groessen_innerhalb_eines_pakets_mit_mehreren_faechern": sum(
                    len(faecher) != 1 for faecher in self.paket_schluessel.values()),
                "fehler": dict(sorted(self.fehler.items())),
                "offen": ["Hashfunktion und Ursprung der Schluesselwerte", "Wachstums-/Rehash-Schwellen",
                          "NIL: reservierter Eintrag, geloeschter Link oder andere Bedeutung",
                          "Zweiter D-Wert: keine eindeutige Funktion der Fachzahl; Allokations-/Historienrolle offen",
                          "Nicht lokal aufgeloeste v-Ziele: keine stillschweigende Umdeutung zu NIL"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pakete", nargs="*")
    ap.add_argument("--proben", action="store_true")
    ap.add_argument("--echt", action="store_true")
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--ausgabe", type=Path, required=True, help="neues lokales Ergebnisverzeichnis")
    a = ap.parse_args(argv)
    pfade = [Path(p) for p in a.pakete]
    for an, var, rekursiv in ((a.proben, "PKG_PROBEN", False), (a.echt, "PKG_ECHT", True), (a.alt, "PKG_ALT", True)):
        if an:
            if not os.environ.get(var):
                ap.error(var+" ist nicht gesetzt")
            for p in os.environ[var].split(os.pathsep):
                path = Path(p)
                pfade.extend(path.rglob("*.pkg") if rekursiv and path.is_dir() else path.glob("*.pkg") if path.is_dir() else [path])
    if not pfade:
        ap.error("Keine Pakete ausgewählt")
    a.ausgabe.mkdir(parents=True, exist_ok=False)
    u = Untersuchung(a.details)
    for pfad in sorted(set(p.resolve() for p in pfade)):
        u.paket(pfad)
    report = u.bericht()
    (a.ausgabe/"bericht.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if a.details:
        keys = [{"schluessel": k, "faecher": n, "beobachtet": dict(v)} for (k, n), v in sorted(u.schluesselfaecher.items())]
        (a.ausgabe/"details-lokal.json").write_text(json.dumps({"fundstellen": u.fundstellen, "schluesselfaecher": keys},
            ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"zaehler": report["zaehler"], "tabellenparameter": report["tabellenparameter"],
                      "fehler": report["fehler"], "schluesselkonflikte": report["schluessel_groessen_mit_mehreren_faechern"]}))
    return 1 if u.fehler else 0


if __name__ == "__main__":
    raise SystemExit(main())
