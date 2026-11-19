#!/usr/bin/env python3
"""Leser für Creo-Elements/Direct-.pkg: Baugruppenbaum, Namen, Lage, Farbe, Dichte, Eckpunkte.

  pkgleser.py datei.pkg [--kompakt]   JSON auf stdout; Rückgabewert 1, wenn es Warnungen gab
  pkgleser.py --baum datei.pkg         Textansicht des Baums

Als Modul: lesen(pfad) -> dict. Längen in mm, Dichte in g/cm³.
Lage: p_global = p · M + t; die Zeilen von M sind die Bilder der Basisvektoren (FORMAT.md).
Unbekannte Kennungen: das betroffene Segment wird übersprungen und als Warnung gemeldet.
"""
import argparse
import datetime
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pkgcontainer import oeffnen_mit_diagnose  # noqa: E402
from objstrom import Token, Unbekannt, tokens  # noqa: E402

# Kurzkonstanten 'O' an Zahlenstellen; 3/4 sind Hypothese (leerer Begrenzungsquader)
O_WERTE = {1: 0.0, 2: 1.0, 3: math.inf, 4: -math.inf}
# new_layout, 2d_view: Layouts und 2D-Ansichten als Kinder von Baugruppen (echte Pakete, 30.10.2026)
TYPEN = {"npart3": "teil", "nasmb3": "baugruppe", "nwp": "arbeitsebene", "new_layout": "layout", "2d_view": "ansicht"}
TOPOLOGIE = ("body", "shell", "face", "loop", "coedge", "edge", "vertex")
EINHEIT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def ist_zahl(t):
    return t.art == "F" or (t.art == "O" and t.wert in O_WERTE)


def zahl(t):
    return t.wert if t.art == "F" else O_WERTE[t.wert]


@dataclass
class Objekt:
    nr: int
    klasse: str
    offset: int
    felder: list = field(default_factory=list)

    @property
    def id(self):
        return next((t.wert for t in self.felder if t.art == "ID"), None)

    def verweise(self):
        return [t.wert for t in self.felder if t.art == "v" and isinstance(t.wert, int)]

    def zahlen(self):
        return [zahl(t) for t in self.felder if ist_zahl(t)]


@dataclass
class Segment:
    eintrag: str
    start: int
    ende: int = -1
    klasse: str = ""
    id: str = ""
    objekte: dict = field(default_factory=dict)

    def wurzel(self):
        return next((o for o in self.objekte.values() if o.klasse == self.klasse), None)

    def von_klasse(self, klasse):
        return [o for o in self.objekte.values() if o.klasse == klasse]


class Fehler(Exception):
    pass


def kopf_lesen(b):
    kopf = {}
    for t in tokens(b):
        if t.art in ("d", "1"):          # 1: 8-Bit-Versionstext älterer Ströme (SolidDesigner 11/14)
            kopf["version"] = t.wert
        elif t.art == "S":
            kopf["zeichensatz"] = t.wert[1]
        elif t.art == "q":
            break
    return kopf


def zeiger_aufloesen(seg, daten=None):
    """Segmentlokale Zeiger gegen vollständige gespeicherte t-Wörter auflösen.

    Die Segmentleser übergeben mit daten noch die unveränderten 32-Bit-t-Wörter.
    Eine Definition ohne Bit 31 kennzeichnet die Zeigerform. Darin gehört auch
    Bit 31 zur Adresse; nur ein exakt gleicher vollständiger v/t-Wert wird
    aufgelöst. Fehlende Ziele bleiben ('roh', Wert), auch bei gesetztem Bit 31.
    Ausschließlich markierte Definitionen behalten die bisherige indizierte
    Darstellung mit maskiertem Bit 31. Das bewahrt auch die Referenzgleichheit
    vollständig hoch liegender Zeiger, ohne deren Speicherart zu behaupten.

    Ohne daten gilt für direkt erzeugte Segmente der bisherige Vertrag: nur
    durchgehend rohe v gegen vorhandene Nummern auflösen. Native Original- und
    Resave-Hierarchien des gemischten 32-Bit-Falls sind separat belegt; siehe
    FORMAT.md. Gibt die Zahl der aufgelösten v zurück.
    """
    vs = [t for o in seg.objekte.values() for t in o.felder if t.art == "v"]
    if daten is not None:
        # The segment builders retain the full stored t word until this point.
        # An unmarked definition identifies the pointer representation. Its
        # addresses may also have bit 31 set; those bits belong to the address.
        # Match the complete stored v word against a complete t word, never a
        # guessed range or a masked address. Indexed streams retain their API.
        pointer = any(not (nr & 0x80000000) for nr in seg.objekte)
        raw_objects = seg.objekte
        numbers = {nr: nr if pointer else nr & 0x7fffffff for nr in raw_objects}
        for nr, obj in raw_objects.items():
            obj.nr = numbers[nr]
        seg.objekte = {numbers[nr]: obj for nr, obj in raw_objects.items()}
        if not pointer:
            return 0
        n = 0
        for t in vs:
            raw = struct.unpack_from(">I", daten, t.offset + 1)[0]
            if raw in raw_objects:
                t.wert = numbers[raw]
                n += 1
            else:
                t.wert = ("roh", raw)
        return n
    if not vs or any(not isinstance(t.wert, tuple) for t in vs):
        return 0
    n = 0
    for t in vs:
        if t.wert[1] in seg.objekte:
            t.wert = t.wert[1]
            n += 1
    return n


def segmente_lesen(b, eintrag, warnungen):
    """Alle Segmente eines Objektstroms; bei unbekannter Kennung weiter am nächsten 'q 80 21'."""
    segmente = []
    start = None
    while True:
        seg = obj = None
        im_kopf = False
        try:
            for t in tokens(b, start):
                if t.art == "q":
                    seg = Segment(eintrag, t.offset)
                    im_kopf = True
                elif seg is None:
                    continue
                elif t.art == "r":
                    seg.ende = t.offset
                    zeiger_aufloesen(seg, b)
                    segmente.append(seg)
                    seg = None
                elif im_kopf:
                    if t.art == "z":
                        seg.klasse = t.wert
                    elif t.art == "ID":
                        seg.id = t.wert
                    elif t.art == "8022":
                        im_kopf = False
                elif t.art == "t":
                    obj = Objekt(struct.unpack_from(">I", b, t.offset + 1)[0], t.wert[1], t.offset)
                    seg.objekte[obj.nr] = obj
                elif t.art == "u":
                    obj = None
                elif obj is not None:
                    obj.felder.append(t)
            return segmente
        except Unbekannt as e:
            wo = f"Segment ab {seg.start:#x}" if seg else "Kopf"
            warnungen.append(f"Eintrag {eintrag}: {e}; {wo} übersprungen")
            weiter = b.find(b"q\x80\x21", e.offset + 1)
            if weiter < 0:
                return segmente
            start = weiter


def segment_lesen(b, start, eintrag, warnungen):
    """Ein einzelnes Segment ab Offset `start` (Kennung q) bis zu seinem r. Für große Pakete, die nicht als Ganzes in
    den Speicher passen (sparsamer Leser); Gliederung wie segmente_lesen."""
    seg = obj = None
    im_kopf = False
    try:
        for t in tokens(b, start):
            if t.art == "q":
                if seg is not None:
                    break
                seg = Segment(eintrag, t.offset)
                im_kopf = True
            elif seg is None:
                continue
            elif t.art == "r":
                seg.ende = t.offset
                zeiger_aufloesen(seg, b)
                return seg
            elif im_kopf:
                if t.art == "z":
                    seg.klasse = t.wert
                elif t.art == "ID":
                    seg.id = t.wert
                elif t.art == "8022":
                    im_kopf = False
            elif t.art == "t":
                obj = Objekt(struct.unpack_from(">I", b, t.offset + 1)[0], t.wert[1], t.offset)
                seg.objekte[obj.nr] = obj
            elif t.art == "u":
                obj = None
            elif obj is not None:
                obj.felder.append(t)
    except Unbekannt as e:
        warnungen.append(f"Eintrag {eintrag}: {e}; Segment ab {start:#x} übersprungen")
        return None
    warnungen.append(f"Eintrag {eintrag}: Segment ab {start:#x} ohne Ende")
    return None


def segmentkoepfe(b):
    """Durchlauf ohne Objekte: je Segment (start, ende, klasse, id). Schnell (31 MB in 2 s) und ohne Speicherbedarf."""
    out, akt, im_kopf = [], None, False
    for t in tokens(b):
        if t.art == "q":
            akt, im_kopf = [t.offset, -1, "", ""], True
        elif akt is None:
            continue
        elif t.art == "r":
            akt[1] = t.offset
            out.append(tuple(akt))
            akt = None
        elif im_kopf:
            if t.art == "z" and not akt[2]:
                akt[2] = t.wert
            elif t.art == "ID" and not akt[3]:
                akt[3] = t.wert
            elif t.art == "8022":
                im_kopf = False
    return out


class Segmentverzeichnis:
    """ID → Segmente wie Leser.index, aber geladen erst beim Zugriff. Baugruppensegmente bleiben im Speicher, Teile
    (part_struct) werden bei jedem Zugriff neu gelesen und danach freigegeben – so passt auch ein Paket mit 31 MB
    Objektstrom (4,9 Mio. Tokens) neben anderen Prozessen in einen Container mit 2 GB (Praxistest 30.10.2026)."""

    BEHALTEN = ("asbm_struct", "nasmb3", "npart3")

    def __init__(self, b, eintrag, warnungen):
        self.b, self.eintrag, self.warnungen = b, eintrag, warnungen
        self.koepfe = segmentkoepfe(b)
        self.nach_id = {}
        for start, ende, klasse, sid in self.koepfe:
            self.nach_id.setdefault(sid, []).append((start, klasse))
        self.cache = {}

    def get(self, sid, standard=None):
        if sid not in self.nach_id:
            return [] if standard is None else standard
        if sid in self.cache:
            return self.cache[sid]
        segs = [s for s in (segment_lesen(self.b, start, self.eintrag, self.warnungen) for start, _ in self.nach_id[sid])
                if s is not None]
        if all(k in self.BEHALTEN for _, k in self.nach_id[sid]):
            self.cache[sid] = segs
        return segs

    def oberstes(self):
        """Das letzte Segment des Eintrags (oberstes Objekt der Sitzung)."""
        start, _, _, _ = self.koepfe[-1]
        return segment_lesen(self.b, start, self.eintrag, self.warnungen)

    def part_ids(self):
        return [sid for sid, liste in self.nach_id.items() if any(k == "part_struct" for _, k in liste)]


class Verzeichnisse:
    """Die Segmentverzeichnisse aller Objektströme eines Pakets als ein Index (sparsamer Leser). get() liefert
    wie das Index-Wörterbuch des vollständigen Lesens die Segmente einer ID aus allen Strömen in
    Eintragsreihenfolge. Bis 4.11.2026 ersetzte jeder Strom das Verzeichnis des vorigen: in Paketen mit
    mehreren Objektströmen fehlten dann Teile und Baugruppen früherer Ströme (Ingestproben 5.11.2026)."""

    def __init__(self):
        self.liste = []

    def get(self, sid, standard=None):
        segs = [s for v in self.liste for s in v.get(sid)]
        return segs if segs or standard is None else standard


def kurz_id(nutzlast):
    """Nutzlast von 80 11/80 12/80 14 (Marke, ID-hi, ID-lo, …) als ID im Format von 0x81 ("1:62f9"); Längen je Marke
    in objstrom.KURZ_ID_TEILE (z. B. 0x19: hi u8 lo u16, 0x1d: hi u8 lo u32, 0x1e: hi u16 lo u32, 0x15: hi u8 lo u8)."""
    from objstrom import KURZ_ID_TEILE
    b = bytes.fromhex(nutzlast)
    hl, ll = KURZ_ID_TEILE[b[0]]
    hi = int.from_bytes(b[1:1 + hl], "big")
    lo = int.from_bytes(b[1 + hl:1 + hl + ll], "big")
    return f"{hi:x}:{lo:x}"


def kurz_id_kodieren(kennung, rest="02", k_laenge=1):
    """Umkehrung von kurz_id: ID "hi:lo" als Nutzlast von 80 11/80 14 mit der kürzesten passenden Marke (0x19, 0x1a,
    0x1d, 0x1e); `rest` sind die Bytes dahinter als Hex (bei 80 11 ein Byte: der Deklarationsindex des Ziels im Segment, FORMAT.md).
    k_laenge: Länge des Indexfelds laut Marke; 2 nur, wenn eine vorhandene Kurz-ID mit 2-Byte-Index umgeschrieben wird."""
    from objstrom import KURZ_ID_FORMEN
    hi, lo = (int(x, 16) for x in kennung.split(":"))
    laengen = (1 if hi < 0x100 else 2, 2 if lo < 0x10000 else 4, k_laenge)
    marke = next(m for m, l in KURZ_ID_FORMEN.items() if l == laengen)
    return (bytes([marke]) + hi.to_bytes(laengen[0], "big") + lo.to_bytes(laengen[1], "big")).hex() + rest


def platzierung(obj):
    """Verweis auf die Struktur und Lage aus einem npart3/nasmb3. Der Inhalt steht meist in einem anderen Segment
    (Kurz-ID 80 11, nach einem zweiten 8002 auch 80 12). Eine Unterbaugruppe kann aber auch im selben Segment liegen:
    dann steht statt der Kurz-ID ein direkter Objektverweis vor der Matrix (Pilotmodell, 30.10.2026); geliefert wird
    dafür ("lokal", Objektnummer)."""
    i = next((n for n, t in enumerate(obj.felder) if t.art in ("8011", "8012")), None)
    if i is None:
        # direkter Verweis: ein v, dem neun Zahlen (Matrix) folgen
        i = next((n for n, t in enumerate(obj.felder) if t.art == "v" and len(obj.felder) > n + 9
                  and all(ist_zahl(x) for x in obj.felder[n + 1:n + 10])), None)
        if i is None:
            raise Fehler(f"{obj.klasse} #{obj.nr}: kein Verweis 80 11")
    rest = iter(obj.felder[i + 1:])

    def nimm(n, art):
        out = []
        for _ in range(n):
            t = next(rest, None)
            if t is None or (art == "zahl" and not ist_zahl(t)) or (art == "B" and t.art != "B"):
                raise Fehler(f"{obj.klasse} #{obj.nr}: Platzierung unerwartet bei {t}")
            out.append(zahl(t) if art == "zahl" else t.wert)
        return out

    m = nimm(9, "zahl")
    f1 = nimm(3, "B")
    t = nimm(3, "zahl")
    s = nimm(1, "zahl")[0]
    f2 = nimm(5, "B")
    lage = {"matrix": [m[0:3], m[3:6], m[6:9]], "verschiebung": t, "skalierung": s, "flags": [f1, f2]}
    if obj.felder[i].art == "v":
        return ("lokal", obj.felder[i].wert), lage
    return kurz_id(obj.felder[i].wert), lage


def verketten(kind, eltern):
    """Globale Lage: p·Mk + tk, dann ·Me + te."""
    mk, tk = kind["matrix"], kind["verschiebung"]
    me, te = eltern["matrix"], eltern["verschiebung"]
    m = [[sum(mk[r][k] * me[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    t = [sum(tk[k] * me[k][c] for k in range(3)) + te[c] for c in range(3)]
    return {"matrix": m, "verschiebung": t}


def anwenden(lage, p):
    m, t = lage["matrix"], lage["verschiebung"]
    return [sum(p[k] * m[k][c] for k in range(3)) + t[c] for c in range(3)]


def name_von(seg, obj):
    for n in seg.von_klasse("NAME"):
        if obj.nr in n.verweise():
            return next((t.wert for t in n.felder if t.art == "c"), None)
    return None


def zeitstempel(obj):
    for t in obj.felder:
        if t.art == "F" and 1e9 < t.wert < 1e10:
            return datetime.datetime.fromtimestamp(t.wert, datetime.timezone.utc).isoformat(timespec="seconds")
    return None


KOERPERARTEN = ("volumen", "flaeche", "draht")


def koerperart(seg, body=None):
    """Körperart aus den Verweisen des body (Anlagenplaner #464/#466, Beleg 7.11.2026: 6.057 nativ gemessene
    Teilinhalte aus zwei Kundenmodellen, keine Abweichung):

    - "volumen": body → shell, kein FACE_PART_ATT
    - "flaeche": body → shell und FACE_PART_ATT (`8002, ID, B 1, v body, c ''`)
    - "draht":   body → wire, keine shell

    Jede andere Verweisform (beides, keines) ist nicht belegt und ergibt None."""
    body = body if body is not None else next(iter(seg.von_klasse("body")), None)
    if body is None:
        return None
    klassen = [seg.objekte[v].klasse for v in body.verweise() if v in seg.objekte]
    shell, wire, fpa = klassen.count("shell"), klassen.count("wire"), klassen.count("FACE_PART_ATT")
    if shell and not wire:
        return "flaeche" if fpa else "volumen"
    if wire and not shell and not fpa:
        return "draht"
    return None


def body_lage(seg):
    """Matrix und Translation des body (die ersten 12 Zahlen nach dem Verweis auf die shell) oder None."""
    body = next(iter(seg.von_klasse("body")), None)
    z, nach_shell = [], False
    for t in (body.felder if body else []):
        if t.art == "v" and isinstance(t.wert, int) and t.wert in seg.objekte and seg.objekte[t.wert].klasse == "shell":
            nach_shell = True
        elif nach_shell and ist_zahl(t):
            z.append(zahl(t))
            if len(z) == 12:
                return {"matrix": [z[0:3], z[3:6], z[6:9]], "verschiebung": z[9:12]}
    return None


def koerper_quader(seg, teil, warnungen):
    """Altes Format (SolidDesigner 11): part_struct ohne Quader und Dichte. Der body trägt nach dem Verweis auf
    die shell seine Lage (9 + 3 Zahlen), eine Skalierung und den Quader in Körperkoordinaten; der Quader wird
    in Teilkoordinaten umgerechnet, eine Körperlage ungleich Einheit mitgeführt (pkg2step bäckt sie ein)."""
    body = next(iter(seg.von_klasse("body")), None)
    z, nach_shell = [], False
    for t in (body.felder if body else []):
        if t.art == "v" and isinstance(t.wert, int) and t.wert in seg.objekte and seg.objekte[t.wert].klasse == "shell":
            nach_shell = True
        elif nach_shell and ist_zahl(t):
            z.append(zahl(t))
            if len(z) == 19:
                break
    if len(z) < 19:
        warnungen.append(f"part_struct {seg.id}: Quader/Dichte nicht in erwarteter Form")
        return
    lage = {"matrix": [z[0:3], z[3:6], z[6:9]], "verschiebung": z[9:12]}
    q = z[13:19]
    if any(q[k] > q[k + 1] for k in (0, 2, 4)):
        teil["quader"] = None
    else:
        ecken = [anwenden(lage, [q[i], q[2 + j], q[4 + k]]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        teil["quader"] = {"min": [min(e[c] for e in ecken) for c in range(3)],
                          "max": [max(e[c] for e in ecken) for c in range(3)]}
    if lage["matrix"] != EINHEIT or lage["verschiebung"] != [0.0, 0.0, 0.0]:
        teil["koerperlage"] = lage


def farbe_aus(clr):
    """Grafikfarbe als #rrggbb: gepackte B-Werte oder SD11 B0 B0 und R G B als Zahlen 0…1.
    Gilt für GA_PART_CLR, GA_FACE_CLR und den RGB-Anteil alter Reflexionsattribute."""
    rgb = [zahl(t) for t in clr.felder if ist_zahl(t)]
    if len(rgb) >= 3:
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in rgb[:3]):
            return None
        return "#" + "".join("%02x" % round(x * 255) for x in rgb[:3])
    werte = [t.wert & 0xFFFFFF for t in clr.felder if t.art == "B"]
    return "#%06x" % werte[0] if werte else None


def vorkommensfarbe(seg, obj):
    """Farbe eines Vorkommens im Kontext seiner Baugruppe: npart3/nasmb3 → GEM_NPART/GEM_NASSEMBLY → G_ATTR_LIST
    → GA_PART_CLR (SolidDesigner 11: alle 86 Vorkommen des Fahrwerks; pkgbau schreibt hier NIL wie Creos Import)."""
    for v in obj.verweise():
        gem = seg.objekte.get(v)
        if gem is None or gem.klasse not in ("GEM_NPART", "GEM_NASSEMBLY"):
            continue
        for w in gem.verweise():
            liste = seg.objekte.get(w)
            if liste is None or liste.klasse != "G_ATTR_LIST":
                continue
            for x in liste.verweise():
                clr = seg.objekte.get(x)
                if clr is not None and clr.klasse == "GA_PART_CLR":
                    return farbe_aus(clr)
    return None


def attributliste(seg, halter):
    """Attribute (Klasse → Objekt) der G_ATTR_LIST, auf die ein Halter zeigt (GEM_NPART, GEM_NASSEMBLY, GEM_SP_STRUCT,
    GEM_FACE_ATTR; Teilaufgabe 334)."""
    for w in halter.verweise():
        liste = seg.objekte.get(w)
        if liste is not None and liste.klasse == "G_ATTR_LIST":
            return {seg.objekte[x].klasse: seg.objekte[x] for x in liste.verweise() if x in seg.objekte}
    return {}


# Kurzkonstanten an Stellen kurzer Gleitzahlen (K): O 7 = 0.0, O 8 = 1.0 – wie O 1/O 2 bei F. Belegt im Kundenmodell
# (Lisp-Lauf 7b): 144 Teile mit Reflexionsstärke O 8 meldet Creo als 1.0, 2 mit Glanz O 7 als 0.0. Auch GEM_FACE_ATTR.
K_KURZ = {7: 0.0, 8: 1.0}


def k_wert(t):
    """Zahl an einer K-Stelle: K, Kurzkonstante O 7/O 8 oder (selten) F/O 1/O 2; sonst die Marke als Text."""
    if t.art == "K":
        return t.wert
    if t.art == "O" and t.wert in K_KURZ:
        return K_KURZ[t.wert]
    if ist_zahl(t):
        return zahl(t)
    return f"{t.art}{t.wert}"


def attributwerte(attr):
    """Transparenz und Reflexion aus einer Attributliste, einschließlich belegter SD11-Feldformen.
    Moderne Reflexion: B Farbe, K Stärke, K Glanz. SD11: B0 B0 RGB Skalar, auch negativ."""
    out = {}
    tr = attr.get("GA_PART_TRANS")
    if tr is not None:
        f = tr.felder
        if len(f) == 3 and all(t.art == "B" and t.wert == 0 for t in f[:2]):
            f = f[2:]  # SD11: B0 B0 Transparenz; keine implizite Vorgabe.
        if len(f) == 1 and ist_zahl(f[0]):
            out["transparenz"] = zahl(f[0])
    rf = attr.get("GA_PART_REFLECT")
    if rf is not None:
        f = rf.felder
        if (len(f) == 6 and all(t.art == "B" and t.wert == 0 for t in f[:2])
                and all(ist_zahl(t) and math.isfinite(zahl(t)) for t in f[2:])
                and farbe_aus(rf) is not None):
            # SD11 B0 B0 RGB Skalar. Native Gegenwerte -128/-64/-32/-8/-1,
            # 0/5/8/16/32/64: Stärke = min(1, Skalar/32), Glanz = -1.
            # Keine untere Begrenzung: -64 liefert beispielsweise Stärke -2.
            value = zahl(f[5])
            out["reflexion"] = {"farbe": farbe_aus(rf),
                                "werte": [min(1., value/32.), -1.]}
        elif len(f) == 3 and f[0].art == "B":
            out["reflexion"] = {"farbe": "#%06x" % (f[0].wert & 0xFFFFFF), "werte": [k_wert(t) for t in f[1:3]]}
    return out


def vorkommenswerte(seg, obj):
    """Transparenz und Reflexion eines Vorkommens (wie vorkommensfarbe über GEM_NPART/GEM_NASSEMBLY)."""
    for v in obj.verweise():
        gem = seg.objekte.get(v)
        if gem is not None and gem.klasse in ("GEM_NPART", "GEM_NASSEMBLY"):
            return attributwerte(attributliste(seg, gem))
    return {}


def benutzerattribute(seg, besitzer):
    """Benutzerattribute (sd-attach-item-attribute) eines Objekts: ATTRIBUTE_LINK mit Symbol UDEF-ATTRIB → UDEF-ATTRIB
    (Verhaltensbits D, dann Schlüssel und Werte als c, Werte als gedruckte Lisp-Form, Ende I) → attr_stick → NAME mit
    dem Attributnamen. {name: {"bits": "0x222224", "werte": [[schlüssel, wert], …]}} (echte Pakete, 31.10.2026)."""
    out = {}
    for v in besitzer.verweise():
        link = seg.objekte.get(v)
        if link is None or link.klasse != "ATTRIBUTE_LINK" or not any(t.art == "z" and t.wert == "UDEF-ATTRIB" for t in link.felder):
            continue
        for a in link.verweise():
            ua = seg.objekte.get(a)
            if ua is None or ua.klasse != "UDEF-ATTRIB":
                continue
            texte, bits = [], None
            for t in ua.felder:
                if t.art == "I":
                    break
                if t.art == "D" and bits is None:
                    bits = t.wert
                elif t.art == "c":
                    texte.append(t.wert)
            stick = next((seg.objekte[x] for x in ua.verweise() if x in seg.objekte and seg.objekte[x].klasse == "attr_stick"), None)
            name_obj = next((seg.objekte[x] for x in (stick.verweise() if stick else []) if x in seg.objekte
                             and seg.objekte[x].klasse == "NAME"), None)
            name = next((t.wert for t in name_obj.felder if t.art == "c"), None) if name_obj else None
            out[name if name is not None else f"#{ua.nr}"] = {"bits": None if bits is None else hex(bits),
                                                             "werte": [list(p) for p in zip(texte[0::2], texte[1::2])]}
    return out


def ansichtsbezuege(seg):
    """Vorkommen (Objektnummer → Anzahl), auf die Ansichtspfade eines Baugruppensegments zeigen: view_attr (Ansicht einer
    Arbeitsebene) → subpath → subpath_struct (B1 v Vorkommen B1 v subpath) → npart3/nasmb3 (Teilaufgabe 336)."""
    out = {}
    for sp in seg.von_klasse("subpath_struct"):
        for v in sp.verweise():
            o = seg.objekte.get(v)
            if o is not None and o.klasse in ("npart3", "nasmb3"):
                out[v] = out.get(v, 0) + 1
    return out


def flaechenfolge(seg):
    """(ID, Klasse der Trägerfläche) der Flächen in der Reihenfolge von geometrie.py und pkg2step (body → shell → face)."""
    body = next(iter(seg.von_klasse("body")), None)
    out = []
    for sh in (seg.objekte.get(v) for v in (body.verweise() if body else [])):
        if sh is None or sh.klasse != "shell":
            continue
        for fo in schalenflaechen(seg, sh):
            art = next((seg.objekte[v].klasse for v in fo.verweise() if v in seg.objekte
                        and seg.objekte[v].klasse.startswith("k2_")), None)
            out.append((fo.id, art))
    return out


def schalenflaechen(seg, sh):
    """Flächen einer Schale wie geometrie.BRep.schalenflaechen: direkt verwiesen, dann über subshell (altes Format,
    auch Normteile in 20.3-Paketen, Lauf 7b vom 31.10.2026)."""
    out, gesehen = [], set()
    for f in sh.verweise():
        o = seg.objekte.get(f)
        if o is not None and o.klasse == "face" and f not in gesehen:
            gesehen.add(f)
            out.append(o)
    for o in seg.von_klasse("face"):
        subs = [seg.objekte[v] for v in o.verweise() if v in seg.objekte and seg.objekte[v].klasse == "subshell"]
        if subs and o.nr not in gesehen and any(sh.nr in s.verweise() for s in subs[:1]):
            gesehen.add(o.nr)
            out.append(o)
    return out


# Historische Schluessel bleiben erhalten. Bei erkannter Struktur sind ihre
# Positionen fest; die kontextdaten-Sicht trennt leere Texte von o.
STEP_PRODUKT_VORN = ("form_name", "form_beschreibung", "definition_beschreibung")
STEP_PRODUKT_HINTEN = ("lebenszyklus", "version_id", "version_beschreibung", "id", "name", "beschreibung", "kontext",
                       "feld_11", "disziplin")


def _step_kontextfelder(obj):
    """Validate the complete variant, then retain its 21 text slots and source anchors.

    Legacy headers differ in width; none of their extra flags/references is
    assigned a meaning here. I is the altformat analysis alias of physical o
    in the proven compressed forms, not a global missing-value rule.
    """
    from analyse.protokollfelder import Tabellenfehler, tabelle_lesen
    if obj.klasse != "PD_ATTR_SP_STRUCT0":
        return None
    try:
        table = tabelle_lesen(obj)
    except Tabellenfehler:
        return None
    if table is None:
        return None
    fs = [(i, t) for i, t in enumerate(obj.felder) if t.art not in ("s", "8002")][table["ende"]:]
    turns = [("O", v) for v in (3, 4, 3, 4, 3, 4)]
    variants = (
        ("creo_20_3", [("v",), ("v",), ("F",), ("B", 0), ("B", 1), ("v",),
                       ("B", 0), ("B", 0), ("NIL", 1)] + turns, False),
        ("cocreate_2008_ohne_b0", [("v",), ("v",), ("F",), ("B", 1), ("v",),
                                  ("B", 0), ("B", 0), ("NIL", 1)] + turns, False),
        ("soliddesigner_11_30", [("v",), ("F",), ("I", None), ("I", None), ("B", 1),
                                 ("v",), ("B", 0), ("B", 0), ("B", 0)], True),
        ("soliddesigner_11_28", [("v",), ("F",), ("I", None), ("I", None), ("B", 1),
                                 ("v",), ("B", 0)], True),
        ("soliddesigner_13_36", [("v",), ("F",), ("I", None), ("I", None), ("B", 1),
                                 ("v",), ("B", 0), ("B", 0), ("NIL", 1)] + turns, True),
        ("soliddesigner_14_37", [("v",), ("v",), ("F",), ("o", None), ("o", None), ("B", 1),
                                 ("v",), ("B", 0), ("B", 0), ("NIL", 1)] + turns, False),
    )
    for form, header, old_missing in variants:
        if len(fs) != len(header) + 21:
            continue
        if any(t.art != spec[0] or (len(spec) == 2 and t.wert != spec[1])
               for (_, t), spec in zip(fs, header)):
            continue
        text = fs[len(header):]
        ts = [t for _, t in text]
        if (ts[2].art, ts[2].wert) != ("B", 0):
            continue
        if any(ts[i].art != "c" or type(ts[i].wert) is not str
               for i in (0, 1, 3, 9, 10, 11, 12, 13, 14, 15, 20)):
            continue
        missing = ("I",) if old_missing else ("o",)
        if any(not ((ts[i].art in missing and ts[i].wert is None)
                    or (ts[i].art == "c" and type(ts[i].wert) is str))
               for i in (4, 5, 6, 7, 8, 16, 17, 18, 19)):
            continue
        return form, text
    return None


def step_kontextdaten(obj, mit_fundstellen=False):
    """Belegte AP214-Kontextfelder mit Leerstatus und optionalem Originalort.

    Beide Jahre sind Anwendungsprotokolljahre, keine Produkt-/Baujahre.
    Alte Kopfformen: Original-/Resave-Getter und isolierte Feldgegenproben
    vom 12.11.2026; die verbleibenden Kopfwerte erhalten keine neue Deutung.
    """
    layout = _step_kontextfelder(obj)
    if layout is None:
        return {"schema_erkannt": False, "status": "unbekannte_form"}
    form, fs = layout

    def textfeld(index, bedeutung):
        raw, t = fs[index]
        value = {"status": "nicht_gespeichert" if t.art != "c" else ("leer" if t.wert == "" else "vorhanden"),
                 "wert": t.wert if t.art == "c" else None, "tokenart": t.art, "bedeutung": bedeutung}
        if mit_fundstellen:
            value.update(feld=raw, offset=t.offset)
        return value

    erg = {"schema_erkannt": True, "status": "belegte_kontextfelder", "form": form,
           "definitionskontext_name": textfeld(4, "belegt"),
           "definitionskontext_anwendungsprotokoll_jahr": textfeld(7, "belegt"),
           "definitionskontext_zusatz_roh": textfeld(7, "belegt"),
           "anwendungsprotokoll_jahr": textfeld(18, "belegt"),
           "beleg": "Creo 20.3 AP214-Gegenproben und native Kontext-Setter/Getters 3.11.2026; INGEST_FELDER.md"}
    if form == "cocreate_2008_ohne_b0":
        erg["beleg_form"] = ("Feldfolge ab dem Zeitstempel wie 20.3 ohne B 0; Feldrollen aus den 20.3-Belegen "
                             "übertragen, für CoCreate Modeling 2008 nicht nativ gegengeprueft (Ingest 5.11.2026)")
    elif form != "creo_20_3":
        erg["beleg_form"] = ("Native Original-/Resave-Getter und isolierte Einfeldproben 12.11.2026; "
                             "INGEST_FELDER.md; Kopfkonstanten und Referenzrollen darueber hinaus offen")
    return erg


def teil_lesen(segmente, warnungen, mit_punkten=True):
    teil, folge, farben, transparenzen = {}, [], {}, {}
    for s in segmente:
        for p in s.von_klasse("part_struct"):
            f = p.felder
            keine = f and f[-1].art == "O" and f[-1].wert == 5
            # Die Dichte folgt auch im SD11-part_struct ohne Teilquader auf v.
            # O 5 = keine Teildichte; Creo nimmt dann die Vorgabe 1 g/mm³ (Lisp-Lauf 7b).
            if len(f) >= 2 and f[-2].art == "v" and (ist_zahl(f[-1]) or keine):
                teil["dichte_g_cm3"] = None if keine else zahl(f[-1]) * 1000
            if len(f) >= 8 and all(ist_zahl(t) for t in f[-8:-2]) and f[-2].art == "v" and (ist_zahl(f[-1]) or keine):
                q = [zahl(t) for t in f[-8:-2]]
                leer = any(q[k] > q[k + 1] for k in (0, 2, 4))
                teil["quader"] = None if leer else {"min": q[0::2], "max": q[1::2]}
            else:
                koerper_quader(s, teil, warnungen)
            teil["zeitstempel"] = zeitstempel(p)
            attr = benutzerattribute(s, p)
            if attr:
                teil["attribute"] = attr
            topo = {k: len(s.von_klasse(k)) for k in TOPOLOGIE}
            if topo["body"]:
                teil["topologie"] = topo
                arten = {koerperart(s, b) for b in s.von_klasse("body")}
                # Mehrere Körper verschiedener Art in einem Teil sind nicht belegt: dann None wie bei unbekannter Form
                teil["koerperart"] = arten.pop() if len(arten) == 1 else None
                if mit_punkten:
                    teil["eckpunkte"] = [o.zahlen()[-3:] for o in s.von_klasse("k2_point")]
                # Lage des Körpers im Teil (Matrix und Translation nach dem shell-Verweis, wie geometrie.koerperlage).
                # Auch Creo 20.3 nutzt sie: Teile, die in einem anderen Zusammenhang konstruiert und dann bewegt wurden,
                # tragen die Bewegung im body (Praxistest 30.10.2026). Die Eckpunkte sind Körperkoordinaten.
                if "koerperlage" not in teil:
                    lage = body_lage(s)
                    if lage and (lage["matrix"] != EINHEIT or lage["verschiebung"] != [0.0, 0.0, 0.0]):
                        teil["koerperlage"] = lage
        for sp in s.von_klasse("GEM_SP_STRUCT"):
            teil.update(attributwerte(attributliste(s, sp)))
        for pd in s.von_klasse("PD_ATTR_SP_STRUCT0"):
            texte = [t.wert for t in pd.felder if t.art == "c"]
            n_v, n_h = len(STEP_PRODUKT_VORN), len(STEP_PRODUKT_HINTEN)
            kontext = step_kontextdaten(pd)
            if kontext['schema_erkannt']:
                # Every supported header maps to the same validated 21 slots.
                # Keep actual source positions; optional texts never shift identity.
                _, mapped = _step_kontextfelder(pd)
                fs = [t for _, t in mapped]
                vorn = [fs[i].wert for i in (0, 1, 3)]
                hinten = [fs[i].wert for i in (9, 10, 11, 12, 13, 14, 15, 18, 20)]
                produkt = dict(zip(STEP_PRODUKT_VORN, vorn)) | dict(zip(STEP_PRODUKT_HINTEN, hinten))
                weitere = [fs[i].wert for i in range(4, 9) if fs[i].art == 'c']
                if weitere:
                    produkt['weitere'] = weitere
                produkt['kontextdaten'] = kontext
                teil['step_produkt'] = produkt
            elif len(texte) >= n_v + n_h:
                teil["step_produkt"] = dict(zip(STEP_PRODUKT_VORN, texte[:n_v])) | dict(zip(STEP_PRODUKT_HINTEN, texte[-n_h:]))
                if texte[n_v:-n_h]:
                    teil["step_produkt"]["weitere"] = texte[n_v:-n_h]
                teil["step_produkt"]["kontextdaten"] = kontext
            else:
                warnungen.append(f"PD_ATTR_SP_STRUCT0 #{pd.nr}: nur {len(texte)} Texte")
        for db in s.von_klasse("DB_FLAGS"):
            # B1 v Besitzer D n am part_struct (10 bzw. 6 Teile der Kundenmodelle, n = 6 bzw. 4), Bedeutung offen
            teil["db_flags"] = next((t.wert for t in db.felder if t.art == "D"), None)
        for ir in s.von_klasse("inheritance_relation"):
            # Vererbung: Kante oder Fläche → ID eines Elements aus einer früheren Sitzung (Teilaufgabe 336)
            ziel = next((s.objekte[v].klasse for v in ir.verweise() if v in s.objekte), "?")
            vererbung = teil.setdefault("vererbung", {})
            vererbung[ziel] = vererbung.get(ziel, 0) + 1
        if s.von_klasse("body"):
            folge = flaechenfolge(s)
        for fa in s.von_klasse("GEM_FACE_ATTR") + s.von_klasse("GEM_FACE"):
            attr = attributliste(s, fa)
            clr, tr = attr.get("GA_FACE_CLR"), attr.get("GA_FACE_TRANS")
            ref = next((t.wert for t in fa.felder if t.art == "8014"), None)
            fid = kurz_id(ref) if ref is not None else None
            if fa.klasse == "GEM_FACE":
                targets = [s.objekte[v] for v in fa.verweise() if v in s.objekte and s.objekte[v].klasse == "face"]
                fid = targets[0].id if len(targets) == 1 else None
            if clr is not None and fid is not None:
                color = farbe_aus(clr)
                if color is not None:
                    farben[fid] = color
            if tr is not None and fid is not None and tr.felder and ist_zahl(tr.felder[0]):
                transparenzen[fid] = zahl(tr.felder[0])
        for c in s.von_klasse("GA_PART_CLR"):
            if any(ist_zahl(t) for t in c.felder):
                teil["farbe"] = farbe_aus(c)
                continue
            werte = [t.wert & 0xFFFFFF for t in c.felder if t.art == "B"]
            if werte:
                teil["farbe"] = "#%06x" % werte[0]
                if len(set(werte)) > 1:
                    teil["farben_alle"] = ["#%06x" % w for w in werte]
    if transparenzen:
        nummer = {fid: n for n, (fid, _) in enumerate(folge, 1)}
        teil["flaechentransparenz"] = {nummer[fid]: w for fid, w in transparenzen.items() if fid in nummer}
    if farben:
        # Flächenfarben (GA_FACE_CLR am GEM_FACE_ATTR, über die Kurz-ID der Fläche): Flächennummer ab 1 → Farbe
        nummer = {fid: (n, art) for n, (fid, art) in enumerate(folge, 1)}
        teil["flaechenfarben"] = {nummer[fid][0]: f for fid, f in farben.items() if fid in nummer}
        teil["flaechenarten"] = {nummer[fid][0]: nummer[fid][1] for fid in farben if fid in nummer}
    return teil


class Leser:
    def __init__(self, mit_punkten=True):
        """mit_punkten=False: ohne Eckpunkte der Teile (pkgaendern; spart bei echten Paketen den größten Posten)."""
        self.mit_punkten = mit_punkten
        self._bezuege = {}
        self.warnungen = []
        self.index = {}
        self.teile = {}

    def struktur(self, ziel_id, klasse):
        return next((s for s in self.index.get(ziel_id, []) if s.klasse == klasse), None)

    def knoten(self, seg, obj, eltern=None, kontextfarbe=None):
        """Knoten des Baums. `farbe`: Farbe des Vorkommens in seiner Baugruppe (falls gesetzt); `farbe_wirksam` (nur
        Teile): äußerste Vorkommensfarbe auf dem Weg von der Wurzel, sonst die Teilfarbe – Annahme: der äußere
        Baugruppenkontext gewinnt (im Fahrwerk tragen nur Teilvorkommen Farben, die Frage stellt sich dort nicht)."""
        k = {"typ": TYPEN.get(obj.klasse, "unbekannt"), "klasse": obj.klasse,
             "name": name_von(seg, obj), "id": obj.id,
             "ort": [seg.eintrag, seg.start, obj.nr]}            # Eintrag, Segmentanfang, Objektnummer (pkgaendern)
        if obj.klasse not in ("npart3", "nasmb3", "new_layout"):
            return k
        eigene = vorkommensfarbe(seg, obj)
        if eigene:
            k["farbe"] = eigene
        k.update(vorkommenswerte(seg, obj))
        attr = benutzerattribute(seg, obj)
        if attr:
            k["exemplarattribute"] = attr
        schl = (seg.eintrag, seg.start)
        if schl not in self._bezuege:
            self._bezuege[schl] = ansichtsbezuege(seg)
        if self._bezuege[schl].get(obj.nr):
            k["ansichtsbezuege"] = self._bezuege[schl][obj.nr]
        # Exemplardichte (Creo 20.3, Lisp-Lauf 6: set_part_inst_density): letztes Feld des npart3 in g/mm³,
        # sonst O 5 („keine“)
        if obj.klasse == "npart3" and obj.felder and ist_zahl(obj.felder[-1]):
            # auch als Kurzkonstante: O 2 = 1.0 g/mm³ (Kundenmodell, Lisp-Lauf 7b vom 31.10.2026)
            k["dichte_g_cm3"] = zahl(obj.felder[-1]) * 1000
        kontextfarbe = kontextfarbe or eigene
        eltern = eltern or {"matrix": EINHEIT, "verschiebung": [0.0, 0.0, 0.0]}
        try:
            ziel, lage = platzierung(obj)
        except Fehler as e:
            self.warnungen.append(str(e))
            return k
        if lage["skalierung"] != 1.0:
            self.warnungen.append(f"{k['name']}: Skalierung {lage['skalierung']} wird nicht berücksichtigt")
        k["lage"] = lage
        k["lage_global"] = verketten(lage, eltern)
        if obj.klasse == "npart3":
            k["teil_id"] = ziel
            if ziel not in self.teile:
                segs = self.index.get(ziel, [])
                if not segs:
                    self.warnungen.append(f"{k['name']}: Teil {ziel} nicht im Paket")
                self.teile[ziel] = teil_lesen(segs, self.warnungen, self.mit_punkten)
            wirksam = kontextfarbe or self.teile[ziel].get("farbe")
            if wirksam:
                k["farbe_wirksam"] = wirksam
            return k
        # Struktur des Inhalts: asbm_struct bei Baugruppen, wp_struct bei Layouts (new_layout: Arbeitsebenen,
        # 2D-Ansichten, auch eingebettete Baugruppen; echte Pakete 30.10.2026)
        strukturklassen = ("asbm_struct", "wp_struct")
        if isinstance(ziel, tuple):                                 # Inhalt im selben Segment
            asbm, wurzel = seg, seg.objekte.get(ziel[1])
            if wurzel is None or wurzel.klasse not in strukturklassen:
                self.warnungen.append(f"{k['name']}: Struktur #{ziel[1]} nicht im Segment")
                return k
        else:
            asbm = self.struktur(ziel, "asbm_struct") or self.struktur(ziel, "wp_struct")
            if asbm is None:
                self.warnungen.append(f"{k['name']}: Baugruppenstruktur {ziel} nicht im Paket")
                return k
            wurzel = next((o for o in asbm.objekte.values() if o.klasse in strukturklassen and o.id == asbm.id), None) \
                or next((o for o in asbm.objekte.values() if o.klasse in strukturklassen), None)
        k["struktur"] = [asbm.eintrag, asbm.start, wurzel.nr]   # Ort der asbm_struct/wp_struct (Quader, Kinderliste)
        attr = benutzerattribute(asbm, wurzel)
        if attr:
            k["attribute"] = attr
        k["kinder"] = []
        # Kinder in der Reihenfolge der Verweisliste der Baugruppenstruktur, nicht alle child_link des Segments: ein
        # Segment kann weitere, verschachtelte Baugruppenstrukturen enthalten
        links = [asbm.objekte[v] for v in wurzel.verweise() if v in asbm.objekte and asbm.objekte[v].klasse == "child_link"]
        for link in links:
            try:
                cnode = asbm.objekte[link.verweise()[0]]
                kind = asbm.objekte[cnode.verweise()[0]]
            except (IndexError, KeyError):
                self.warnungen.append(f"{k['name']}: child_link #{link.nr} nicht auflösbar")
                continue
            k["kinder"].append(self.knoten(asbm, kind, k["lage_global"], kontextfarbe))
        return k

    def lesen(self, pfad, sparsam=False):
        """sparsam: Segmente erst beim Zugriff laden (Segmentverzeichnis), für Pakete, die nicht in den Speicher passen;
        nur ein Objektstrom je Paket."""
        erg = {"datei": os.path.basename(pfad), "einheiten": {"laenge": "mm", "dichte": "g/cm³"},
               "lage_konvention": "p_global = p · M + t", "eintraege": [], "objekte": []}
        oben = []
        eintraege, containerbefunde = oeffnen_mit_diagnose(pfad)
        self.warnungen.extend(containerbefunde)
        verzeichnisse = Verzeichnisse()
        stroeme = 0
        for name, d in eintraege:                          # ZIP (20.3) oder altes compress-Paket (altformat)
            if d[:2] == b"Rz":
                stroeme += 1
                erg.update({k: v for k, v in kopf_lesen(d).items() if k not in erg})
                if sparsam:
                    verzeichnis = Segmentverzeichnis(d, name, self.warnungen)
                    verzeichnisse.liste.append(verzeichnis)
                    seg = verzeichnis.oberstes() if verzeichnis.koepfe else None
                    if seg is not None:
                        oben.append(seg)
                else:
                    segs = segmente_lesen(d, name, self.warnungen)
                    for s in segs:
                        self.index.setdefault(s.id, []).append(s)
                    if segs:
                        oben.append(segs[-1])
                art = "objektstrom"
            else:
                art = "png" if d[:8] == b"\x89PNG\r\n\x1a\n" else "unbekannt"
            erg["eintraege"].append({"name": name, "art": art, "bytes": len(d)})
        del eintraege  # Entpackte Stroeme vor dem Aufbau des Modellbaums freigeben.
        erg["objektstroeme"] = stroeme                     # 0: kein Creo-Paket (z. B. gewöhnliches ZIP-Archiv)
        if len(verzeichnisse.liste) == 1:                  # ein Strom: wie bisher das Verzeichnis selbst (pkgaendern)
            self.index = verzeichnisse.liste[0]
        elif verzeichnisse.liste:                          # mehrere Ströme: gemeinsamer Index
            self.index = verzeichnisse
        for seg in oben:
            w = seg.wurzel()
            if w is None:
                self.warnungen.append(f"Eintrag {seg.eintrag}: kein Wurzelobjekt {seg.klasse}")
                continue
            k = self.knoten(seg, w)
            k["eintrag"] = seg.eintrag
            erg["objekte"].append(k)
        erg["teile"] = self.teile
        erg["warnungen"] = self.warnungen
        return erg


def lesen(pfad, sparsam=False):
    return Leser().lesen(pfad, sparsam)


def json_sauber(x):
    """inf/nan sind kein gültiges JSON -> None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: json_sauber(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_sauber(v) for v in x]
    return x


def fmt(v):
    return "(" + ", ".join(f"{x:g}" for x in v) + ")"


def baum_text(erg):
    zeilen = [f"{erg['datei']}: {erg.get('version', '?')}"]

    def knoten(k, tiefe):
        z = f"{'  ' * tiefe}{k['typ']} {k['name']!r}"
        if "lage_global" in k:
            lg = k["lage_global"]
            z += f"  t={fmt(lg['verschiebung'])}"
            if lg["matrix"] != EINHEIT:
                z += f" M={' '.join(fmt(r) for r in lg['matrix'])}"
        if k.get("farbe"):
            z += f"  Vorkommensfarbe {k['farbe']}"
        if k.get("dichte_g_cm3") is not None:
            z += f"  Vorkommensdichte {k['dichte_g_cm3']:g} g/cm³"
        if k.get("transparenz") is not None:
            z += f"  Vorkommenstransparenz {k['transparenz']:g}"
        if k.get("ansichtsbezuege"):
            z += f"  in {k['ansichtsbezuege']} Ansichtspfaden"
        for schl, text in (("attribute", "Attribute"), ("exemplarattribute", "Exemplarattribute")):
            if k.get(schl):
                z += f"  {text} {', '.join(k[schl])}"
        if "teil_id" in k:
            t = erg["teile"].get(k["teil_id"], {})
            z += f"  [Teil {k['teil_id']}"
            for schl, text in (("farbe", " {}"), ("dichte_g_cm3", " {:g} g/cm³")):
                if t.get(schl) is not None:
                    z += text.format(t[schl])
            if t.get("transparenz") is not None:
                z += f" Transparenz {t['transparenz']:g}"
            if t.get("topologie"):
                z += f" {t['topologie']['face']} Flächen, {len(t.get('eckpunkte', []))} Punkte"
            if t.get("attribute"):
                z += " Attribute " + ", ".join(t["attribute"])
            if t.get("step_produkt"):
                z += f" STEP-Produkt {t['step_produkt']['id']!r}/{t['step_produkt']['name']!r}"
            if t.get("flaechenfarben"):
                z += " Flächenfarben " + " ".join(f"{n}:{f}" for n, f in sorted(t["flaechenfarben"].items()))
            z += "]"
        zeilen.append(z)
        for c in k.get("kinder", []):
            knoten(c, tiefe + 1)

    for o in erg["objekte"]:
        knoten(o, 1)
    zeilen += [f"WARNUNG {w}" for w in erg["warnungen"]]
    return "\n".join(zeilen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datei")
    ap.add_argument("--kompakt", action="store_true", help="JSON ohne Einrückung")
    ap.add_argument("--baum", action="store_true", help="Textansicht statt JSON")
    ap.add_argument("--sparsam", action="store_true", help="Segmente erst bei Bedarf laden (große Pakete, wenig Speicher)")
    a = ap.parse_args()
    erg = lesen(a.datei, a.sparsam)
    if a.baum:
        print(baum_text(erg))
    else:
        # Escape isolated UTF-16 units as well as ordinary non-ASCII names.
        print(json.dumps(json_sauber(erg), ensure_ascii=True, indent=None if a.kompakt else 1, allow_nan=False))
    return 1 if erg["warnungen"] else 0


if __name__ == "__main__":
    sys.exit(main())
