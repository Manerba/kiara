#!/usr/bin/env python3
"""Belegte Inhalts-/Feature-Textbeschriftungen lesen, andere A3D-Formen inventarisieren.

  python -I /absolut/analyse/beschriftungen.py MODELL.pkg [--details]

Kein Schreiber und keine allgemeine A3D-Transformationsfreigabe. Die CLI
zeigt standardmaessig nur Zaehler; Texte, IDs und Rohfelder nur mit --details.
Erkannte Labelformen und vollstaendige Feldsemantik sind getrennte Aussagen.
"""
import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altformat import oeffnen  # noqa: E402
from pkgleser import ist_zahl, segment_lesen, segmentkoepfe, schalenflaechen, zahl  # noqa: E402
from analyse.protokollfelder import Tabellenfehler, tabelle_lesen  # noqa: E402

PARAMETER = {"PLAIN_TEXT", "FRAME_PARAM", "text_param", "line_param", "label_param", "arrow_param", "CLUSTER_PATH"}
BELEG = ("Creo 20.3, 1.11.2026: neutrale Inhalts-Textformen, kontrollierte Feld- und Textlistenmutanten, "
         "41 native Zustaende, lokale/gedrehte Richtungs-Gegenproben, gemeinsame Pfeilrolle, Textlisten mit 1/2/3 Eintraegen und Leerstring, Pfeilfuellung und Rahmen OFF/BOX, UTF-16-Paarerhalt, MM/INCH/MM-Sitzungswechsel; "
         "reference/3ddocu/anno_inq.html; DOC3D_ORIENTATE_ANNO, "
         "reference/dg_manual.html (:measure-direction/:two_pt); "
         "7.11.2026: label_param Feld 5 = 3 an 22 Labels (Creo 19.0 und 20.3) in allen gelesenen Darstellungswerten "
         "nativ gleich, nur mit Schrift hp_i3098_v; Einfeldmutanten 1/2/3 ohne native Wirkung; "
         "13.11.2026: Zwischenform mit A3D_TAG-B0 statt Feature-NIL und "
         "A3D_REF-Schluss-NIL ohne nachfolgendes B0 durch Original/Creo-Resave "
         "und native Beschriftungswerte belegt")
OFFEN = ["Andere Annotationstypen, Halter- und Referenzformen",
         "Weitere Pfeil-/Rahmentypen und getrennte ungefuellte Pfeilrollen", "Allgemeine Koordinatenrahmen und Vererbung",
         "Restliche A3D-Flags, N9..N11/N15 und Referenzrollen",
         "Genaue Bedeutung der abgeleiteten PLAIN_TEXT-Zahlen",
         "Textlisten mit anderer Laenge, Darstellung und Formatierungsrollen der Texte", "Groessengrenzen und weitere Fontparameter",
         "Allgemeine Transformation und Geometrieersetzung"]


RICHTUNGSTOLERANZ = 1e-12  # dimensionslos; keine Normierung der gespeicherten Werte
LABELPARAM_FELD5 = (0, 3)   # in Originalen beobachtet und nativ geprueft (7.11.2026)


class Unbelegt(ValueError):
    """Nur feste allgemeine Gruende, keine Kundenwerte in Fehlermeldungen."""


def _refs_of_kind(seg, obj, kinds):
    return [seg.objekte[nr] for nr in obj.verweise()
            if nr in seg.objekte and seg.objekte[nr].klasse in kinds]


def _koerperflaechen(seg, body, owner):
    """Membership with actual reverse links, also for older subshell chains.

    This checks topology membership, not the unrelated face-tail grammar.
    A forward shell list alone must not overrule a conflicting face parent.
    """
    _require(_refs_of_kind(seg, body, {"part_struct"}) == [owner],
             "Annotationsflaeche_Koerperbesitzer_widerspruechlich")
    faces = set()
    for shell in _refs_of_kind(seg, body, {"shell"}):
        _require(_refs_of_kind(seg, shell, {"body"}) == [body],
                 "Annotationsflaeche_Schalenbesitzer_widerspruechlich")
        for face in schalenflaechen(seg, shell):
            parents = _refs_of_kind(seg, face, {"shell", "subshell"})
            _require(len(parents) == 1, "Annotationsflaeche_Besitzer_mehrdeutig")
            parent = parents[0]
            if parent.klasse == "subshell":
                # An outer subshell can name the same shell twice (immediate
                # parent and owning shell). Every occurrence must agree.
                shell_refs = _refs_of_kind(seg, parent, {"shell"})
                _require(shell_refs and all(s is shell for s in shell_refs)
                         and face in _refs_of_kind(seg, parent, {"face"}),
                         "Annotationsflaeche_Unterschalenbesitzer_widerspruechlich")
            else:
                _require(parent is shell, "Annotationsflaeche_Besitzer_widerspruechlich")
            faces.add(face.nr)
    return faces


def _require(value, reason):
    if not value:
        raise Unbelegt(reason)


def _fields(obj):
    return [(i, t) for i, t in enumerate(obj.felder) if t.art not in ("s", "8002")]


def _finite(t):
    if not ist_zahl(t):
        return False
    value = zahl(t)
    return type(value) in (int, float) and math.isfinite(value)


def _shape(obj, expected, table=False):
    pairs = _fields(obj)
    if table:
        try:
            tab = tabelle_lesen(obj)
        except Tabellenfehler:
            raise Unbelegt("Tabellenpraefix_unbekannt") from None
        _require(tab is not None, "Tabellenpraefix_fehlt")
        pairs = pairs[tab["ende"]:]
    fs = [t for _, t in pairs]
    _require(len(fs) == len(expected), "Feldform_unbekannt")
    for t, kind in zip(fs, expected):
        if kind == "N":
            _require(_finite(t), "Numerisches_Feld_unbelegt")
        else:
            _require(t.art == kind, "Feldform_unbekannt")
            if kind in ("B", "D", "v"):
                _require(type(t.wert) is int, "Ganzzahlfeld_unbelegt")
            elif kind in ("c", "z", "ID"):
                _require(isinstance(t.wert, str), "Textfeld_unbelegt")
            elif kind == "NIL":
                _require(t.wert == 1, "NIL_Form_unbelegt")
    return pairs


def _plain_shape(obj):
    # Belegt sind 1/2/3 geordnete API-Strings, auch ein leerer mittlerer
    # Eintrag. Keine Zeilen-, Prefix-/Suffix- oder Fontmetriken ableiten.
    try:
        tab = tabelle_lesen(obj)
    except Tabellenfehler:
        raise Unbelegt("Tabellenpraefix_unbekannt") from None
    _require(tab is not None, "Tabellenpraefix_fehlt")
    raw = _fields(obj)[tab["ende"]:]
    if len(raw) > 2 and raw[2][1].art != "v":
        # SD14 bettet den Rahmen in PLAIN_TEXT ein. Native Altfassung und
        # Creo-20.3-Neuspeicherung stimmen in Texten und Rahmenwert ueberein.
        count = len(raw) - 13
        _require(count in (1, 2, 3), "Textlistenlaenge_unbelegt")
        pairs = _shape(obj, ["N"]*3 + ["B"] + ["N"]*4 + ["B", "v", "NIL", "B"]
                       + ["c"]*count + ["v"], table=True)
        _constants(pairs, {2: 0, 3: 0, 11: count})
        _require(pairs[8][1].wert in (0, 1), "Konstante_der_Referenzform_abweichend")
        return pairs[:2] + [None] + pairs[9:]
    count = len(raw) - 7
    _require(count in (1, 2, 3), "Textlistenlaenge_unbelegt")
    pairs = _shape(obj, ["N", "N", "v", "v", "NIL", "B"] + ["c"]*count + ["v"], table=True)
    _require(pairs[5][1].wert == count, "Textanzahl_widerspricht_Feldern")
    return pairs


def _constants(pairs, expected):
    for index, value in expected.items():
        t = pairs[index][1]
        actual = zahl(t) if _finite(t) else t.wert
        _require(actual == value, "Konstante_der_Referenzform_abweichend")


def _directions(pairs):
    """Die eng belegte Textbasis ist lokal und orthonormal; Rohwerte erhalten."""
    directions = {
        "textrichtung_lokal": [zahl(pairs[i][1]) for i in (11, 12, 13)],
        "obenrichtung_lokal": [zahl(pairs[i][1]) for i in (14, 15, 16)],
        "darstellungsnormale_lokal": [zahl(pairs[i][1]) for i in (24, 25, 26)],
    }
    u, v, n = directions.values()
    close = lambda a, b: math.isclose(a, b, rel_tol=0., abs_tol=RICHTUNGSTOLERANZ)
    _require(all(close(sum(x*x for x in vec), 1.) for vec in (u, v, n)),
             "Richtungslaenge_der_Textform_unbelegt")
    _require(close(sum(a*b for a, b in zip(u, v)), 0.), "Textbasis_nicht_orthogonal")
    cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    _require(all(close(a, b) for a, b in zip(cross, n)), "Darstellungsnormale_widerspricht_Textbasis")
    return directions


def _target(seg, pair, cls):
    t = pair[1]
    _require(t.art == "v" and type(t.wert) is int, "Verweisform_unbekannt")
    target = seg.objekte.get(t.wert)
    _require(target is not None and target.klasse == cls, "Verweisziel_unbelegt")
    return target


def rueckverweise(seg):
    """Ziel-Objektnummer -> Counter der verweisenden Objektnummern (Mehrfachverweise zaehlen) in einem Durchlauf.
    Gleiches Ergebnis wie die Einzelzaehlung in _incoming; fuer Pruefungen vieler Objekte desselben Segments."""
    rev = collections.defaultdict(collections.Counter)
    for o in seg.objekte.values():
        for nr in o.verweise():
            rev[nr][o.nr] += 1
    return rev


def _incoming(seg, obj, owners, rev=None):
    if rev is None:
        actual = collections.Counter(o.nr for o in seg.objekte.values() for nr in o.verweise() if nr == obj.nr)
    else:
        actual = rev[obj.nr] if obj.nr in rev else collections.Counter()
    _require(actual == collections.Counter(o.nr for o in owners), "Anheftung_mehrdeutig_oder_unvollstaendig")


def _attribute(seg, owner, target, symbol):
    """Genau ein ATTRIBUTE_LINK der Besitzertabelle `ID B1 v Besitzer z Symbol B n n×v` nennt das Ziel.

    n = 1 ist die nativ gepruefte Form. Zaehllisten mit n > 1 sind der allgemeine Mechanismus (FORMAT.md,
    Benutzerattribute); bei A3D_REF teilen sich so zwei Referenzen auf dieselbe Kante einen Link (6.11.2026).
    Jedes Ziel darf in der Liste nur einmal stehen."""
    matches = []
    try:
        table = tabelle_lesen(owner)
    except Tabellenfehler:
        raise Unbelegt("Besitzertabelle_unbekannt") from None
    _require(table is not None, "Besitzertabelle_fehlt")
    for entry in table["eintraege"]:
        link = seg.objekte.get(entry["ziel"]) if entry["art"] == "v" else None
        if link is None or link.klasse != "ATTRIBUTE_LINK":
            continue
        fs = [t for _, t in _fields(link)]
        if len(fs) < 6 or [t.art for t in fs[:5]] != ["ID", "B", "v", "z", "B"]:
            continue
        count = fs[4].wert
        if type(count) is not int or count < 1 or len(fs) != 5 + count or any(t.art != "v" for t in fs[5:]):
            continue
        ziele = [t.wert for t in fs[5:]]
        if (all(type(fs[i].wert) is int for i in (1, 2)) and all(type(z) is int for z in ziele)
                and len(set(ziele)) == count
                and fs[1].wert == 1 and fs[2].wert == owner.nr and fs[3].wert == symbol and target.nr in ziele):
            matches.append(link)
    _require(len(matches) == 1, "Attributanheftung_unbelegt")
    return matches[0]


def _attributziele(link):
    """Zielliste eines von _attribute gelieferten ATTRIBUTE_LINK."""
    return [t.wert for _, t in _fields(link)[5:]]


def _geteilte_ziele(seg, link, target, cls, owner):
    """Weitere Ziele eines geteilten Links: gleiche Klasse, Feld 2 verweist auf denselben Besitzer."""
    for nr in _attributziele(link):
        if nr == target.nr:
            continue
        other = seg.objekte.get(nr)
        fs = _fields(other) if other is not None else []
        _require(other is not None and other.klasse == cls and len(fs) > 2 and fs[2][1].art == "v"
                 and fs[2][1].wert == owner.nr, "Attributlink_fremdes_Ziel")


def _raw(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nicht_endliche_zahl": repr(value)}
    if isinstance(value, (list, tuple)):
        return [_raw(v) for v in value]
    return value


def _source(obj, pair):
    index, t = pair
    return {"klasse": obj.klasse, "objekt": obj.nr, "feld": index, "offset": t.offset}


def _darstellung(seg, label, *, feature_visibility=False):
    _require(label.klasse == "A3D_LBL_TAG", "Keine_Labelklasse")
    # Darstellungsfelder sind beiden belegten Textketten gemeinsam.
    # Der Aufrufer muss anschliessend die gesamte Halterkette pruefen.
    # SolidDesigner 14 speichert denselben Darstellungsblock ohne das letzte
    # B0. Original/Creo-20.3-Neuspeicherung und native Werte sind abgeglichen;
    # das fehlende Wort bekommt weder einen Ersatzwert noch eine Feldquelle.
    table = tabelle_lesen(label)
    _require(table is not None, "Tabellenpraefix_fehlt")
    legacy = len(_fields(label)) - table["ende"] == 35
    lp = _shape(label, ["v"]*5 + ["B"]*3 + ["N"]*9 + ["B", "B", "v", "B"]
                + ["N"]*10 + ["B", "NIL", "B", "B"] + ([] if legacy else ["B"]), table=True)
    _require(lp[6][1].wert in ((0, 2) if feature_visibility else (0,)),
             "Konstante_der_Referenzform_abweichend")
    _constants(lp, {7: 0, 17: 0, 18: 1, 20: 0, 21: 1, 22: 0, 23: 0,
                    27: 0, 31: 1, 33: 3, 34: 255})
    if not legacy:
        _constants(lp, {35: 0})
    directions = _directions(lp)
    plain = _target(seg, lp[0], "PLAIN_TEXT")
    line = _target(seg, lp[1], "line_param")
    label_param = _target(seg, lp[2], "label_param")
    arrow = _target(seg, lp[3], "arrow_param")
    tag = _target(seg, lp[4], "A3D_TAG")
    arrow2 = _target(seg, lp[19], "arrow_param")
    shared_arrow = arrow.nr == arrow2.nr
    pp = _plain_shape(plain)
    text_values = [pair[1].wert for pair in pp[6:-1]]
    inline_frame = pp[2] is None
    frame = plain if inline_frame else _target(seg, pp[2], "FRAME_PARAM")
    _require(_target(seg, pp[3], "A3D_LBL_TAG").nr == label.nr, "Text_Rueckverweis_unbelegt")
    text = _target(seg, pp[-1], "text_param")
    old_text = len(_fields(text)) == 10
    tp = _shape(text, ["D", "D", "D", "c", "B"] + ["N"]*5 + ([] if old_text else ["B"]))
    _constants(tp, {0: 0, 1: 0, 2: 0, 4: 1, 6: 1, 7: 0, 8: 0, 9: 2})
    if not old_text:
        _constants(tp, {10: -1})
    ap = _shape(arrow, ["D", "D", "D", "N", "B", "B"])
    _constants(ap, {0: 0, 1: 0, 2: 0, 4: 1})
    _require(ap[5][1].wert in (0, 1), "Pfeilfuellung_unbelegt")
    if not shared_arrow:
        # Die ungefuellte Gegenprobe betrifft die gemeinsame Rolle. Bei
        # getrennten Rollen bleiben die bisherigen Flagkombinationen streng.
        _constants(ap, {5: 1})
        ap2 = _shape(arrow2, ["D", "D", "D", "N", "B", "B"])
        _constants(ap2, {0: 0, 1: 0, 2: 0, 4: 1, 5: 0})
    if inline_frame:
        raw_plain = _fields(plain)[tabelle_lesen(plain)["ende"]:]
        fp = [None]*3 + raw_plain[3:8]
    else:
        fp = _shape(frame, ["D", "D", "D", "B"] + ["N"]*4)
        _constants(fp, {0: 0, 1: 0, 2: 0})
    # Neutraler Einfeldkontrast, Creo 20.3 (12.11.2026): B2 -> :FRAME_TYPE :BALLOON;
    # alle weiteren Einstellungen, Texte und Referenzpositionen unveraendert.
    frame_types = {0: "off", 1: "box", 2: "balloon"}
    _require(fp[3][1].wert in frame_types, "Rahmentyp_unbelegt")
    _constants(_shape(line, ["D", "D", "D", "N", "N", "B", "B", "B"]),
               {0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1, 6: 0, 7: 0})
    lpp = _shape(label_param, ["D", "D", "D", "N", "N", "B", "B", "B"])
    _constants(lpp, {0: 0, 1: 0, 2: 0, 3: 2, 4: 2, 6: 0, 7: 0})
    # Feld 5 = 0 oder 3 (FORMAT.md, 7.11.2026): 3 tritt nur mit der text_param-Schrift hp_i3098_v auf, 0 mit
    # osd_default. Creo wertet das Feld beim Laden nicht aus (Einfeldmutanten 1/2/3 ohne native Aenderung).
    # Der Wert bleibt roh; andere Werte bleiben unbelegt.
    _require(lpp[5][1].wert in LABELPARAM_FELD5, "Konstante_der_Referenzform_abweichend")
    return dict(label=label, lp=lp, plain=plain, pp=pp, text=text, tp=tp,
                arrow=arrow, ap=ap, arrow2=arrow2, frame=frame, fp=fp,
                line=line, label_param=label_param, tag=tag, directions=directions,
                text_values=text_values, shared_arrow=shared_arrow, frame_types=frame_types,
                feature_visibility=feature_visibility)


def _setfelder(aset):
    """SD14 lacks the second NIL before the counted tag list; keep real sources."""
    pairs = _fields(aset)
    old = len(pairs) > 4 and pairs[4][1].art == "B"
    at = 4 if old else 5
    _require(len(pairs) > at and pairs[at][1].art == "B" and type(pairs[at][1].wert) is int,
             "Feldform_unbekannt")
    count = pairs[at][1].wert
    _require(0 < count <= len(pairs), "Feldform_unbekannt")
    shape = ["ID", "B", "v", "NIL"] + ([] if old else ["NIL"]) + ["B"] + ["v"]*count + ["B"]*4
    pairs = _shape(aset, shape)
    return pairs[:4] + [None] + pairs[4:] if old else pairs


def _referenzfelder(ref):
    """Old forms omit B0 or NIL/B0; retain every real slot, invent no source."""
    count = len(_fields(ref))
    _require(count in (20, 21, 22), "Feldform_unbekannt")
    tail = {20: [], 21: ["NIL"], 22: ["NIL", "B"]}[count]
    pairs = _shape(ref, ["ID", "B", "v", "v", "N", "N", "B", "NIL", "v", "B"]
                   + ["N"]*6 + ["B"] + ["N"]*3 + tail)
    if count == 22:
        _constants(pairs, {21: 0})
    return pairs


def _tagfelder(tag):
    """Belegte alte/neue Tagablagen auf gemeinsame Leseindizes abbilden.

    Jede vorhandene Position bleibt ihr originales (Feldindex, Token)-Paar.
    None bezeichnet ausschliesslich das im alten Tag fehlende NIL-Wort;
    es wird weder als Rohfeld noch als Quelle ausgegeben. Das alte B0 an der
    Featureposition bleibt roh; B1+v wird auf den vorhandenen Verweis abgebildet.
    Die gesamte Rohform wird vor dieser Abbildung streng geprueft.
    """
    table = tabelle_lesen(tag)
    _require(table is not None, "Tabellenpraefix_fehlt")
    raw = _fields(tag)[table["ende"]:]
    _require(len(raw) > 2 and raw[1][1].art == "B" and type(raw[1][1].wert) is int,
             "Feldform_unbekannt")
    n = raw[1][1].wert
    _require(n > 0 and n <= len(raw) - 2, "Feldform_unbekannt")
    head = ["v", "B"] + ["v"] * n + ["B"]
    tail = [pair[1].art for pair in raw[len(head):]]
    modern = ["NIL", "v", "B", "NIL"]
    end = ["o", "B", "B", "B", "o"]
    if tail in (modern + ["NIL"] + end, modern + ["v"] + end):
        return _shape(tag, head + tail, table=True)
    # SD14: kein NIL vor dem Labelverweis, Feature als B0 oder B1+v.
    if tail in (["v", "B", "NIL", "B"] + end,
                ["v", "B", "NIL", "B", "v"] + end):
        pairs = _shape(tag, head + tail, table=True)
        flag = len(head) + 3
        feature = tail[4] == "v"
        _constants(pairs, {flag: int(feature)})
        mapped = pairs[:len(head)] + [None] + pairs[len(head):]
        if feature:
            del mapped[flag + 1]
        return mapped
    # Zwischenform: NIL vor dem Label, noch B0 ohne Featureverweis.
    # Original/Creo-Resave: B0 wird NIL; der echte alte Slot bleibt roh erhalten.
    if tail == modern + ["B"] + end:
        pairs = _shape(tag, head + tail, table=True)
        _constants(pairs, {len(head) + 4: 0})
        return pairs
    # Zwischenform mit Feature: noch B1 vor dem Featureverweis.
    if tail == modern + ["B", "v"] + end:
        pairs = _shape(tag, head + tail, table=True)
        flag = len(head) + 4
        _constants(pairs, {flag: 1})
        return pairs[:flag] + pairs[flag + 1:]
    raise Unbelegt("Feldform_unbekannt")


def _inhaltskette(seg, d):
    label, plain, text = (d[k] for k in ("label", "plain", "text"))
    line, label_param, tag = (d[k] for k in ("line", "label_param", "tag"))
    arrow, arrow2, frame = (d[k] for k in ("arrow", "arrow2", "frame"))
    shared_arrow = d["shared_arrow"]
    gp = _tagfelder(tag)
    _require(len(gp) == 14 and gp[8][1].art in ("NIL", "B"), "Feldform_unbekannt")
    _constants(gp, {1: 1, 3: 0, 6: 0, 10: 0, 11: 1, 12: 4})
    aset = _target(seg, gp[0], "A3D_SET")
    ref = _target(seg, gp[2], "A3D_REF")
    _require(_target(seg, gp[5], "A3D_LBL_TAG").nr == label.nr, "Tag_Rueckverweis_unbelegt")
    sp = _setfelder(aset)
    _require(len(sp) == 11, "Feldform_unbekannt")
    _constants(sp, {1: 1, 5: 1, 7: 1, 8: 2147483647, 9: 1, 10: 0})
    owner = _target(seg, sp[2], "part_struct")
    _require(_target(seg, sp[6], "A3D_TAG").nr == tag.nr, "Set_Rueckverweis_unbelegt")
    rp = _referenzfelder(ref)
    _constants(rp, {1: 1, 4: 0, 5: 0, 6: 0, 9: 1, 13: 1, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0})
    vertex = _target(seg, rp[2], "vertex")
    path = _target(seg, rp[3], "CLUSTER_PATH")
    _require(_target(seg, rp[8], "A3D_TAG").nr == tag.nr, "Referenz_Rueckverweis_unbelegt")
    cp = _shape(path, ["ID", "B", "B", "v"])
    _constants(cp, {1: 0, 2: 0})
    _require(_target(seg, cp[3], "A3D_SET").nr == aset.nr, "Pfad_Rueckverweis_unbelegt")
    owner_link = _attribute(seg, owner, aset, "A3D_SET")
    vertex_link = _attribute(seg, vertex, ref, "A3D_REF")
    _geteilte_ziele(seg, owner_link, aset, "A3D_SET", owner)
    _geteilte_ziele(seg, vertex_link, ref, "A3D_REF", vertex)
    rev = rueckverweise(seg)
    for obj, incoming in ((aset, [owner_link, tag, path]), (tag, [aset, label, ref]),
                          (label, [tag, plain]), (plain, [label]), (ref, [tag, vertex_link]),
                          (path, [ref]), (owner_link, [owner]), (vertex_link, [vertex]),
                          (text, [plain]), (frame, [plain]), (line, [label]),
                          (label_param, [label])):
        if obj is plain and incoming == [plain]:
            continue  # eingebetteter Rahmen: kein zusaetzlicher Graphverweis
        _incoming(seg, obj, incoming, rev)
    # Gemeinsame Rolle ist nur innerhalb dieses einen Labels belegt.
    _incoming(seg, arrow, [label, label] if shared_arrow else [label], rev)
    if not shared_arrow:
        _incoming(seg, arrow2, [label], rev)
    return {"inhalt": owner, "inhaltsattribut": owner_link, "set": aset, "tag": tag,
            "referenz": ref, "referenzpfad": path, "referenzvertex": vertex,
            "referenzattribut": vertex_link}


def _ergebnis(d, roles, *, feature=False):
    label, plain, text = (d[k] for k in ("label", "plain", "text"))
    arrow, arrow2, frame = (d[k] for k in ("arrow", "arrow2", "frame"))
    lp, pp, tp, ap, fp = (d[k] for k in ("lp", "pp", "tp", "ap", "fp"))
    text_values, shared_arrow = d["text_values"], d["shared_arrow"]
    known = {"text_anzahl": (plain, pp[5]), "farbe_code": (label, lp[5]),
             "textgroesse_mm": (text, tp[5]), "pfeilgroesse_mm": (arrow, ap[3]),
             "pfeil_gefuellt": (arrow, ap[5]), "rahmentyp": (frame, fp[3])}
    visibility = feature and d.get("feature_visibility", False)
    if visibility:
        known["individuell_ausgeblendet"] = (label, lp[6])
    if len(text_values) == 1:
        known["text"] = (plain, pp[6])  # kompatibler Einzelwert; nie eine Liste verkuerzen
    sources = {name: _source(obj, pair) for name, (obj, pair) in known.items()}
    sources["text_liste"] = [_source(plain, pair) for pair in pp[6:-1]]
    for name, indices in (("textrichtung_lokal", (11, 12, 13)),
                          ("obenrichtung_lokal", (14, 15, 16)),
                          ("darstellungsnormale_lokal", (24, 25, 26))):
        sources[name] = [_source(label, lp[i]) for i in indices]
    sources["versatz_lokal_mm"] = [_source(label, lp[i]) for i in (28, 29, 30)]
    sources["lokale_positionsablage_mm"] = [_source(label, lp[i]) for i in (8, 9, 10)]
    sources["plain_text_zahlen"] = [_source(plain, pp[i]) for i in (0, 1)]
    roles = {**roles, "label": label, "text": plain, "textparameter": text, "pfeilparameter_erstes": arrow,
             "pfeilparameter_zweites": arrow2, "rahmenparameter": frame, "linienparameter": d["line"],
             "labelparameter": d["label_param"]}
    used = collections.defaultdict(set)
    for value in sources.values():
        for source in value if isinstance(value, list) else [value]:
            used[source["objekt"]].add(source["feld"])
    raw = [{"rolle": role, "klasse": obj.klasse, "objekt": obj.nr,
            "felder": [{"feld": i, "offset": t.offset, "art": t.art, "wert": _raw(t.wert)}
                       for i, t in _fields(obj) if i not in used[obj.nr]]} for role, obj in roles.items()]
    return {"form_erkannt": True, "anheftung": "inhalt", "annotationstyp": "feature_text" if feature else "text",
            "semantik_vollstaendig": False,
            "pfeilparameter_gemeinsam": shared_arrow,
            "belegt": {"text_liste": text_values, "text_anzahl": pp[5][1].wert,
                       **({"text": text_values[0]} if len(text_values) == 1 else {}),
                       **({"individuell_ausgeblendet": lp[6][1].wert == 2} if visibility else {}),
                       "farbe_code": lp[5][1].wert,
                       "textgroesse_mm": zahl(tp[5][1]), "pfeilgroesse_mm": zahl(ap[3][1]),
                       "pfeil_gefuellt": ap[5][1].wert == 1, "rahmentyp": d["frame_types"][fp[3][1].wert],
                       "versatz_lokal_mm": [zahl(lp[i][1]) for i in (28, 29, 30)], **d["directions"]},
            "abgeleitet": {"lokale_positionsablage_mm": [zahl(lp[i][1]) for i in (8, 9, 10)],
                           "plain_text_zahlen": [zahl(pp[i][1]) for i in (0, 1)]},
            "kette": {role: {"objekt": obj.nr, "klasse": obj.klasse, "id": obj.id} for role, obj in roles.items()},
            "felder": sources, "restfelder_roh": raw, "beleg": BELEG, "offen": OFFEN}


def _decode(seg, label, *, index=None):
    # The initial parse permits the two proven feature visibility states only
    # for dispatch. Every non-feature path retains its previous strict value.
    d = _darstellung(seg, label, feature_visibility=True)
    gp = _fields(d["tag"])
    tab = tabelle_lesen(d["tag"])
    if tab is not None:
        gp = _tagfelder(d["tag"])
        if len(gp) == 14 and gp[8][1].art == "v":
            from analyse.featurebeschriftungen import decode as feature_decode
            return feature_decode(seg, label, index=index)
        _constants(d["lp"], {6: 0})
        # Inhalts-Textsets: v Set, B n, n × v A3D_REF, 11 feste Felder; Kanten-/Punktziele, mehrere
        # Referenzen oder Mitglieder (FORMAT.md). Die einfache Punktform bleibt bei _inhaltskette.
        from analyse import kantenbeschriftungen
        if kantenbeschriftungen.zustaendig(seg, gp):
            return kantenbeschriftungen.decode(seg, label)
    _constants(d["lp"], {6: 0})
    roles = _inhaltskette(seg, d)
    result = _ergebnis(d, roles)
    if any(len(_attributziele(roles[k])) > 1 for k in ("inhaltsattribut", "referenzattribut")):
        result["nicht_nativ_gegengeprueft"] = ["geteilter_attributlink"]
    return result


def decode(seg, label, *, index=None):
    """Detaildatensatz; externe Baugruppenreferenzen brauchen den Index derselben Datei.

    Ohne Index bleiben solche Referenzen unbekannt. Der Index muss das genau
    uebergebene Segment enthalten, nicht bloss eine gleichlautende Kurz-ID.
    """
    try:
        return _decode(seg, label, index=index)
    except (Unbelegt, Tabellenfehler) as exc:
        return {"form_erkannt": False, "grund": str(exc) if isinstance(exc, Unbelegt) else "Tabellenpraefix_unbekannt",
                "label_roh": [{"feld": i, "offset": t.offset, "art": t.art, "wert": _raw(t.wert)}
                              for i, t in _fields(label)]}


class _DateiIndex:
    """Dateigebundener, pro ausgewertetem Segment kurzlebiger Referenzindex.

    Zielsegmente werden erst fuer externe Referenzen gelesen. Der aktuelle
    Segmentgegenstand bleibt identisch; keine dauerhafte Mutations-Cacheannahme.
    Eintraege bleiben nach Ordinal unterschieden, auch bei gleichem ZIP-Namen.
    """
    def __init__(self, entries, warnings):
        self.entries, self.warnings = entries, warnings
        self.catalog = [(ordinal, entry, start, sid)
                        for ordinal, (entry, data) in enumerate(entries) if data.startswith(b"Rz")
                        for start, _, _, sid in segmentkoepfe(data)]
        self.cache = {}

    def active(self, ordinal, start, seg):
        self.cache = {(ordinal, start): seg}

    def get(self, sid, default=None):
        result = []
        for ordinal, entry, start, candidate in self.catalog:
            if candidate != sid:
                continue
            key = (ordinal, start)
            if key not in self.cache:
                self.cache[key] = segment_lesen(self.entries[ordinal][1], start, entry, self.warnings)
            if self.cache[key] is not None:
                result.append(self.cache[key])
        return result if result or default is None else default


def lesen(path, details=False):
    counts, shapes, reasons = (collections.Counter() for _ in range(3))
    warnings, records = [], []
    segments = supported = feature_supported = unknown_labels = uncovered_tags = uncovered_sets = 0
    entries = list(oeffnen(path))
    index = _DateiIndex(entries, warnings)
    for entry_ordinal, (entry, data) in enumerate(entries):
        if not data.startswith(b"Rz"):
            continue
        for ordinal, _, start, _ in index.catalog:
            if ordinal != entry_ordinal:
                continue
            seg = segment_lesen(data, start, entry, warnings)
            if seg is None:
                continue
            index.active(entry_ordinal, start, seg)
            segments += 1
            covered, covered_sets = set(), set()
            for obj in seg.objekte.values():
                if obj.klasse.startswith("A3D_") or obj.klasse in PARAMETER:
                    counts[obj.klasse] += 1
                    shapes[(obj.klasse, tuple("N" if ist_zahl(t) else t.art for _, t in _fields(obj)))] += 1
                if obj.klasse != "A3D_LBL_TAG":
                    continue
                rec = decode(seg, obj, index=index)
                if rec["form_erkannt"]:
                    if rec["annotationstyp"] == "feature_text":
                        feature_supported += 1
                    else:
                        supported += 1
                unknown_labels += not rec["form_erkannt"]
                if rec["form_erkannt"]:
                    covered.add(rec["kette"]["tag"]["objekt"])
                    covered_sets.add(rec["kette"]["set"]["objekt"])
                else:
                    reasons[rec["grund"]] += 1
                if details:
                    records.append({"eintrag": entry, "entry_ordinal": entry_ordinal,
                                    "segment_start": start, "segment_id": seg.id, "label": obj.nr, **rec})
            uncovered_tags += sum(o.nr not in covered for o in seg.von_klasse("A3D_TAG"))
            uncovered_sets += sum(o.nr not in covered_sets for o in seg.von_klasse("A3D_SET"))
    report = {"segmente": segments, "klassen": dict(sorted(counts.items())),
              "feldformen": [{"klasse": c, "form": list(s), "anzahl": n} for (c, s), n in sorted(shapes.items())],
              "belegte_inhalts_textformen": supported, "belegte_feature_textformen": feature_supported,
              "unbekannte_labelformen": unknown_labels,
              "nicht_abgedeckte_tags": uncovered_tags, "unbelegte_gruende": dict(sorted(reasons.items())),
              "nicht_abgedeckte_sets": uncovered_sets,
              "andere_a3d_klassen": {c: n for c, n in sorted(counts.items()) if c.startswith("A3D_")
                                      and c not in ("A3D_LBL_TAG", "A3D_TAG", "A3D_REF", "A3D_SET")},
              "warnungen": len(warnings), "semantik_vollstaendig": False,
              "beleg": BELEG, "offen": OFFEN}
    if feature_supported:
        from analyse.featurebeschriftungen import BELEG as FEATURE_BELEG, OFFEN as FEATURE_OFFEN
        report["feature_beleg"] = FEATURE_BELEG
        report["feature_offen"] = FEATURE_OFFEN
    if details:
        report["details_lokal"] = {"beschriftungen": records, "warnungen": warnings}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pakete", nargs="+")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    reports = [lesen(path, args.details) for path in args.pakete]
    print(json.dumps(reports, ensure_ascii=True, indent=2, allow_nan=False))
    return int(any(r["warnungen"] or r["unbekannte_labelformen"] or r["nicht_abgedeckte_tags"]
                   or r["nicht_abgedeckte_sets"] or r["andere_a3d_klassen"] for r in reports))


if __name__ == "__main__":
    # Extension decoders import analyse.beschriftungen. Run the canonical
    # package entry point so they share this decoder's Unbelegt class.
    from analyse.beschriftungen import main as package_main
    sys.exit(package_main())
