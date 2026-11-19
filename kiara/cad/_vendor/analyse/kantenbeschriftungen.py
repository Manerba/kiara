"""Contents text sets with counted edge/vertex references and shared parameters.

This reader validates every member/consumer of the set. A3D_REF fields 4/5 are
accepted only where the stored reference position (fields 10-12) is reproduced
from the referenced geometry (edge: curve at the normalised parameter; vertex:
its point). It does not authorize annotation rewriting.
"""
import math
from collections import defaultdict

from analyse import beschriftungen as a
from analyse.protokollfelder import tabelle_lesen
from geometrie import BRep, kurve_punkt, flaeche_punkt, kreuz, skal, plus
from analyse.topologieroh import objekt_lesen as topologie_lesen, TopologieFormfehler
from pkgleser import schalenflaechen, zahl

BELEG = ("Creo 20.3, 4.11.2026: 19 byteidentische Teilsegmente einer Creo-19.0-Ingestquelle; "
         "57 native Text-/Positionszuordnungen, 285 unabhaengige Darstellungsparameter, "
         "76 geordnete Kantenreferenzen mit Positions-/B-Spline-Geometrieabgleich; "
         "FORMAT.md: Inhalts-Textsets mit Kantenreferenzen; "
         "6.11.2026: A3D_REF-Felder 4/5 als normierter Parameter der Referenzposition, am Original an "
         "1.072 Kanten- und 29 Punktreferenzen geometrisch reproduziert (FORMAT.md); "
         "7.11.2026: native Gegenprobe an drei vollstaendig geladenen Soellingen-Originalen, 1.417 Datensaetze "
         "je Datei gezaehlt, darunter 1.008 mit Parameter ungleich 0.5, 121 mit 3 bis 14 Referenzen, 46 mit "
         "geteiltem Attributlink und 11 mit Punktreferenz im Set: Referenzliste, Elementart, Reihenfolge, "
         "gemeinsame Ziele, Kantenbereich, Position und Parameter stimmen mit Creo 20.3 ueberein; "
         "12.11.2026: neutrale u/v-Einfeldkontraste an einer Ebene und 14 native Flaechenreferenzen "
         "(Zylinder, Kegel, Torus), einschliesslich erweiterter Traegerflaeche")
OFFEN = ["Restflags und uebrige A3D_REF-Felder",
         "Weitere Traegerflaechen und widerspruechliche gespeicherte Referenzpositionen",
         "Genaue PLAIN_TEXT-Fontmetriken", "Allgemeine Transformation und Schreiben"]

# A3D_REF-Felder 4/5: normierter Parameter der Referenzposition (Felder 10-12) auf dem Ziel. Kante: Feld 4 im
# Parameterbereich [t0, t1] der Kante, Feld 5 = 0; Vertex: (0, 0), Position = Vertexpunkt. Am Original geprüft für
# k2_straight, k2_circle und k2_bspline_cv (FORMAT.md, „Referenzposition und normierter Parameter“). Andere
# Kurvenklassen nur mit dem bisher belegten Wert 0.5 und ohne Geometrieprüfung.
PARAMETER_KURVEN = ("k2_straight", "k2_circle", "k2_bspline_cv", "k2_ellipse")
PARAMETER_TOLERANZ = 1e-10      # relativ zur größten Koordinate der Referenzposition, mindestens 1e-10 mm


def _position(rp):
    position = [zahl(rp[i][1]) for i in (10, 11, 12)]
    a._require(all(math.isfinite(x) for x in position), "Referenzposition_unbelegt")
    return position


def _deckt(punkt, position):
    toleranz = PARAMETER_TOLERANZ * max(1., *(abs(x) for x in position))
    return (punkt is not None and all(math.isfinite(x) for x in punkt)
            and max(abs(p - q) for p, q in zip(punkt, position)) <= toleranz)


def _kantenparameter(seg, rp, edge):
    """Felder 4/5 einer Kantenreferenz. Rückgabe (Parameter [t, 0], Position, geometrisch geprüft)."""
    f4, f5 = rp[4][1], rp[5][1]
    a._require(a._finite(f4) and a._finite(f5) and zahl(f5) == 0, "Kantenreferenz_Parameter_unbelegt")
    t = zahl(f4)
    a._require(0. <= t <= 1., "Kantenreferenz_Parameter_unbelegt")
    position = _position(rp)
    brep = BRep(seg)
    kurven = brep.refs(edge, "k2_*")
    if not kurven or kurven[0].klasse not in PARAMETER_KURVEN:
        a._require(t == .5, "Kantenreferenz_Kurvenklasse_unbelegt")
        return [t, 0.], position, False
    kurve = kurven[0]
    try:
        geo = brep.geometrie(kurve)
        bereich = brep.zahlen_nach(edge, kurve, 2)
        if kurve.klasse == "k2_ellipse":
            a._require(all(len(geo[k]) == 3 and all(math.isfinite(x) for x in geo[k])
                           for k in ("n", "m", "x"))
                       and all(math.isfinite(geo[k]) for k in ("a", "b")),
                       "Kantenreferenz_Geometrie_unbelegt")
    except (ValueError, IndexError, KeyError, ZeroDivisionError):
        raise a.Unbelegt("Kantenreferenz_Geometrie_unbelegt") from None
    a._require(len(bereich) == 2 and all(math.isfinite(x) for x in bereich), "Kantenreferenz_Bereich_unbelegt")
    t0, t1 = bereich
    try:
        kurvenparameter = t0 + t * (t1 - t0)
        a._require(math.isfinite(kurvenparameter), "Kantenreferenz_Geometrie_unbelegt")
        if geo["typ"] == "bspline":
            p, U, P = geo["grad"], geo["knoten"], geo["pole"]
            periode = U[len(P)] - U[p]
            a._require(all(math.isfinite(x) for x in U)
                       and all(math.isfinite(x) for q in P for x in q)
                       and math.isfinite(periode) and periode > 0,
                       "Kantenreferenz_Geometrie_unbelegt")
            lo, hi = U[p], U[len(P)]
            if not lo - 1e-9 <= kurvenparameter <= hi + 1e-9:
                # Dieselbe Bereichs-/Schliessungspruefung wie geometrie.abrollen.
                # Nur den Punktparameter falten: keine Kurvenkopien und keine
                # Subtraktion einer gerundeten, eventuell riesigen Periodenzahl.
                start, ende = kurve_punkt(geo, lo), kurve_punkt(geo, hi)
                a._require(all(math.isfinite(x) for q in (start, ende) for x in q),
                           "Kantenreferenz_Geometrie_unbelegt")
                if math.dist(start, ende) < 1e-6:
                    abstand = kurvenparameter - lo
                    a._require(math.isfinite(abstand), "Kantenreferenz_Geometrie_unbelegt")
                    kurvenparameter = lo + abstand % periode
        punkt = kurve_punkt(geo, kurvenparameter)
    except (ValueError, IndexError, KeyError, ZeroDivisionError, OverflowError):
        raise a.Unbelegt("Kantenreferenz_Geometrie_unbelegt") from None
    a._require(_deckt(punkt, position),
               "Kantenreferenz_Position_widerspricht_Parameter")
    return [t, 0.], position, True


def _punktparameter(seg, rp, vertex):
    """Felder 4/5 einer Punktreferenz: (0, 0), Position = k2_point des Vertex."""
    a._constants(rp, {4: 0, 5: 0})
    position = _position(rp)
    brep = BRep(seg)
    punkte = brep.refs(vertex, "k2_point")
    a._require(len(punkte) == 1, "Punktreferenz_Geometrie_unbelegt")
    try:
        punkt = brep.geometrie(punkte[0])["p"]
    except (ValueError, IndexError, KeyError):
        raise a.Unbelegt("Punktreferenz_Geometrie_unbelegt") from None
    a._require(len(punkt) == 3 and _deckt(punkt, position), "Punktreferenz_Position_widerspricht_Vertex")
    return [0., 0.], position, True


def _flaechenparameter(seg, rp, face):
    """A3D_REF 4/5: normalisierte u/v der Position auf der Traegerflaeche.

    Eine Referenz darf auf der erweiterten Flaeche liegen. Die Bindung an die
    konkrete Koerperflaeche und die gespeicherte Position werden unabhaengig
    geprueft; es gibt keine Projektion oder Korrektur der PKG-Daten.
    """
    a._require(all(a._finite(rp[i][1]) for i in (4, 5)), "Flaechenreferenz_Parameter_unbelegt")
    params = [zahl(rp[i][1]) for i in (4, 5)]
    a._require(all(0. <= x <= 1. for x in params), "Flaechenreferenz_Parameter_unbelegt")
    position = _position(rp)
    try:
        raw = topologie_lesen(face, seg.objekte)
        box = [zahl(t) for t in raw.struktur['parameter_box_raw']]
        a._require(len(box) == 6 and all(math.isfinite(x) for x in box)
                   and box[0] < box[1] and box[3] < box[4], "Flaechenreferenz_Bereich_unbelegt")
        u, v = box[0] + params[0]*(box[1]-box[0]), box[3] + params[1]*(box[4]-box[3])
        brep = BRep(seg)
        surface = seg.objekte[raw.struktur['surface']]
        a._require(surface.klasse in ('k2_plane', 'k2_cylinder', 'k2_cone', 'k2_torus'),
                   "Flaechenreferenz_Flaechenklasse_unbelegt")
        # Plane/cylinder already use BRep's exact finite analytic grammar.
        # Cone/torus use the complete, distinct original layouts witnessed by
        # the native face inquiries. Keep their flags opaque; do not reuse the
        # plane/cylinder B0-v prefix or accept arbitrary numeric suffixes.
        if surface.klasse == 'k2_cone':
            sf = a._shape(surface, ['ID', 'D', 'D', 'D', 'B', 'N', 'B', 'v'] + ['N']*3)
            a._constants(sf, {1: 0, 2: 0, 3: 0, 4: 1, 6: 2})
            circle_obj = a._target(seg, sf[7], 'k2_circle')
        elif surface.klasse == 'k2_torus':
            sf = a._shape(surface, ['ID', 'D', 'D', 'D', 'B', 'N', 'B', 'N', 'B'] + ['N']*6 + ['v'])
            a._constants(sf, {1: 0, 2: 0, 3: 0, 4: 2, 6: 1, 8: 1})
            circle_obj = a._target(seg, sf[15], 'k2_circle')
        if surface.klasse in ('k2_cone', 'k2_torus'):
            circle_fields = a._shape(circle_obj, ['ID', 'D', 'D', 'D'] + ['N']*10)
            a._constants(circle_fields, {1: 0, 2: 0, 3: 0})
        geo = brep.geometrie(surface)
        if surface.klasse in ('k2_plane', 'k2_cylinder'):
            point = flaeche_punkt(geo, u, v)
        else:
            circles = brep.refs(surface, 'k2_circle')
            a._require(len(circles) == 1, "Flaechenreferenz_Grundkreis_unbelegt")
            circle = brep.geometrie(circles[0])
            a._require(circle['r'] > 0 and all(abs(skal(w,w)-1.) <= 1e-12 for w in (circle['n'], circle['x']))
                       and abs(skal(circle['n'],circle['x'])) <= 1e-12, "Flaechenreferenz_Grundkreis_unbelegt")
            c = kurve_punkt(circle, u)
            if surface.klasse == 'k2_cone':
                point = [(1.+v)*x-v*y for x,y in zip(c,geo['spitze'])]
            else:
                axis, origin = geo['a'], geo['m']
                a._require(abs(skal(axis,axis)-1.) <= 1e-12, "Flaechenreferenz_Achse_unbelegt")
                q = plus(c,origin,-1.)
                cross, dot = kreuz(axis,q), skal(axis,q)
                point = [m+x*math.cos(v)+y*math.sin(v)+z*dot*(1.-math.cos(v))
                         for m,x,y,z in zip(origin,q,cross,axis)]
    except (TopologieFormfehler, ValueError, IndexError, KeyError, ZeroDivisionError, OverflowError) as exc:
        if isinstance(exc, a.Unbelegt):
            raise
        raise a.Unbelegt("Flaechenreferenz_Geometrie_unbelegt") from None
    a._require(point is not None and len(point) == 3 and _deckt(point,position),
               "Flaechenreferenz_Position_widerspricht_Parameter")
    return params, position, True


def _tag(seg, tag):
    table = tabelle_lesen(tag)
    a._require(table is not None, "Kantentext_Tagtabelle_fehlt")
    pairs = a._tagfelder(tag)
    count = len(pairs) - 13
    # n = 1/2 nativ geprueft (4.11.2026), n = 3 bis 14 nativ geprueft (7.11.2026, Soellingen)
    a._require(count >= 1, "Kantentext_Referenzanzahl_unbelegt")
    a._require(pairs[7+count][1].art in ('NIL', 'B'), "Feldform_unbekannt")
    a._constants(pairs, {1: count, 2+count: 0, 5+count: 0,
                         9+count: 0, 10+count: 1, 11+count: 4})
    return pairs, count


def _edges(seg, owner):
    """Restrict edge and vertex targets to the owner's body/face/loop graph."""
    bodies = [o for o in seg.von_klasse('body') if owner.nr in o.verweise()]
    a._require(len(bodies) == 1, "Kantentext_Koerperbesitzer_mehrdeutig")
    body = bodies[0]
    faces = [face for nr in body.verweise() if nr in seg.objekte and seg.objekte[nr].klasse == 'shell'
             for face in schalenflaechen(seg, seg.objekte[nr])]
    pending = [seg.objekte[nr] for face in faces for nr in face.verweise()
               if nr in seg.objekte and seg.objekte[nr].klasse == 'loop']
    seen, edges = set(), set()
    while pending:
        obj = pending.pop()
        if obj.nr in seen:
            continue
        seen.add(obj.nr)
        for nr in obj.verweise():
            target = seg.objekte.get(nr)
            if target is None:
                continue
            if target.klasse == 'edge' and obj.klasse == 'coedge':
                edges.add(nr)
            elif target.klasse in ('loop', 'coedge'):
                pending.append(target)
    # Drahtkoerper haben keine shell/face-Kette. Nur explizit an diesen
    # Koerper und denselben wire rueckgebundene Kanten zulassen; fremde oder
    # lediglich im Segment vorhandene Vertices bleiben ausgeschlossen.
    wires = [seg.objekte[nr] for nr in body.verweise()
             if nr in seg.objekte and seg.objekte[nr].klasse == 'wire']
    for wire in wires:
        a._require(body.nr in wire.verweise(), "Kantentext_Drahtbesitzer_falsch")
        for nr in wire.verweise():
            edge = seg.objekte.get(nr)
            if edge is not None and edge.klasse == 'edge':
                a._require(wire.nr in edge.verweise(), "Kantentext_Drahtkante_falsch")
                edges.add(nr)
    vertices = {nr for e in edges for nr in seg.objekte[e].verweise()
                if nr in seg.objekte and seg.objekte[nr].klasse == 'vertex'}
    return body, edges, vertices


def zustaendig(seg, tag_pairs):
    """Soll ein Label mit diesen Tagfeldern (hinter der Tabelle) hierher? Kantenziel an erster Stelle, mehrere
    Referenzen oder ein Set mit mehreren Mitgliedern. Die einfache Punktform (ein Set-Mitglied, eine
    Punktreferenz) bleibt bei beschriftungen._inhaltskette."""
    gp = tag_pairs
    if not (len(gp) >= 14 and gp[1][1].art == "B" and gp[1][1].wert == len(gp) - 13 and gp[2][1].art == "v"):
        return False
    ref = seg.objekte.get(gp[2][1].wert)
    rp = a._fields(ref) if ref is not None and ref.klasse == "A3D_REF" else []
    ziel = seg.objekte.get(rp[2][1].wert) if len(rp) > 2 and rp[2][1].art == "v" else None
    if ziel is None or ziel.klasse not in ("edge", "vertex", "face"):
        return False
    if ziel.klasse in ("edge", "face") or gp[1][1].wert > 1:
        return True
    aset = seg.objekte.get(gp[0][1].wert) if gp[0][1].art == "v" else None
    sp = a._setfelder(aset) if aset is not None and aset.klasse == "A3D_SET" else []
    return len(sp) > 5 and sp[5][1].art == "B" and type(sp[5][1].wert) is int and sp[5][1].wert > 1


def decode(seg, label):
    """No partial success if another member or a shared parameter is unknown."""
    a._require(seg.objekte.get(label.nr) is label, "Kantentext_Label_nicht_im_Graph")
    d0 = a._darstellung(seg, label)
    gp0, _ = _tag(seg, d0['tag'])
    aset = a._target(seg, gp0[0], 'A3D_SET')
    sp = a._setfelder(aset)
    count = len(sp) - 10
    a._require(count > 0, "Kantentext_Setanzahl_unbelegt")
    a._constants(sp, {1: 1, 5: count, 6+count: 1, 7+count: 2147483647, 8+count: 1, 9+count: 0})
    owner = a._target(seg, sp[2], 'part_struct')
    a._require(len(seg.von_klasse('part_struct')) == 1, "Kantentext_Inhalt_mehrdeutig")
    owner_link = a._attribute(seg, owner, aset, 'A3D_SET')
    a._geteilte_ziele(seg, owner_link, aset, 'A3D_SET', owner)
    # Alle Verbraucher desselben Sets mit einem frischen Index prüfen. Der
    # Index lebt nur für diesen Decode; spätere Graphänderungen bleiben sichtbar.
    rev = a.rueckverweise(seg)
    a._incoming(seg, owner_link, [owner], rev)
    body, edges, vertices = _edges(seg, owner)
    faces = None
    tags = [a._target(seg, p, 'A3D_TAG') for p in sp[6:6+count]]
    a._require(len({o.nr for o in tags}) == count, "Kantentext_Setmitglied_doppelt")
    uses, objects, records, paths, all_refs = defaultdict(list), {}, {}, {}, []
    for tag in tags:
        gp, refs_count = _tag(seg, tag)
        a._require(a._target(seg, gp[0], 'A3D_SET') is aset, "Kantentext_Set_Rueckverweis_falsch")
        member = a._target(seg, gp[4+refs_count], 'A3D_LBL_TAG')
        d = a._darstellung(seg, member)
        a._require(d['tag'] is tag, "Kantentext_Label_Rueckverweis_falsch")
        refs = [a._target(seg, p, 'A3D_REF') for p in gp[2:2+refs_count]]
        a._require(len({o.nr for o in refs}) == refs_count, "Kantentext_Referenz_doppelt")
        roles = {'inhalt': owner, 'inhaltsattribut': owner_link, 'set': aset,
                 'tag': tag, 'koerper': body}
        targets, sources_ref = [], []
        for index, ref in enumerate(refs, 1):
            rp = a._referenzfelder(ref)
            # Felder 4/5 (Parameter der Referenzposition) prüfen _kantenparameter/_punktparameter an der Geometrie.
            a._constants(rp, {1: 1, 6: 0, 9: 1, 13: 1, 14: 0, 15: 0,
                              16: 0, 17: 0, 18: 0, 19: 0})
            ziel = seg.objekte.get(rp[2][1].wert) if rp[2][1].art == 'v' else None
            a._require(ziel is not None and ziel.klasse in ('edge', 'vertex', 'face'), "Verweisziel_unbelegt")
            if ziel.klasse == 'edge':
                a._require(ziel.nr in edges, "Kantentext_Referenz_ausserhalb_Koerper")
                parameter, position, geprueft = _kantenparameter(seg, rp, ziel)
            elif ziel.klasse == 'vertex':
                a._require(ziel.nr in vertices, "Kantentext_Referenz_ausserhalb_Koerper")
                parameter, position, geprueft = _punktparameter(seg, rp, ziel)
            else:
                if faces is None:
                    faces = a._koerperflaechen(seg, body, owner)
                a._require(ziel.nr in faces, "Kantentext_Referenz_ausserhalb_Koerper")
                parameter, position, geprueft = _flaechenparameter(seg, rp, ziel)
            a._require(a._target(seg, rp[8], 'A3D_TAG') is tag, "Kantentext_Referenz_Rueckverweis_falsch")
            path = a._target(seg, rp[3], 'CLUSTER_PATH')
            cp = a._shape(path, ['ID', 'B', 'B', 'v'])
            a._constants(cp, {1: 0, 2: 0})
            a._require(a._target(seg, cp[3], 'A3D_SET') is aset, "Kantentext_Pfad_Rueckverweis_falsch")
            link = a._attribute(seg, ziel, ref, 'A3D_REF')
            a._incoming(seg, link, [ziel], rev)
            # Ein geteilter Link nennt nur weitere Referenzen auf genau dieses Ziel.
            a._geteilte_ziele(seg, link, ref, 'A3D_REF', ziel)
            a._incoming(seg, ref, [tag, link], rev)
            paths[path.nr] = path
            all_refs.append(ref)
            rolle = {'edge': 'referenzkante', 'vertex': 'referenzvertex', 'face': 'referenzflaeche'}[ziel.klasse]
            roles.update({f'referenz_{index}': ref, f'{rolle}_{index}': ziel,
                          f'referenzattribut_{index}': link})
            targets.append({'referenz': ref.nr, 'element': ziel.nr, 'klasse': ziel.klasse,
                            'position_lokal_mm': position, 'parameter_normiert': parameter,
                            'parameter_geometrisch_geprueft': geprueft})
            sources_ref.append({'position_lokal_mm': [a._source(ref, rp[i]) for i in (10, 11, 12)],
                                'parameter_normiert': [a._source(ref, rp[i]) for i in (4, 5)]})
        a._incoming(seg, tag, [aset, member, *refs], rev)
        a._incoming(seg, member, [tag, d['plain']], rev)
        a._incoming(seg, d['plain'], [member], rev)
        roles['referenzpfad'] = path
        for key, consumer in (('text', d['plain']), ('frame', d['plain']), ('line', member),
                              ('label_param', member), ('arrow', member), ('arrow2', member)):
            obj = d[key]
            if key == 'frame' and obj is d['plain']:
                continue
            objects[obj.nr] = obj
            uses[obj.nr].append(consumer)
        result = a._ergebnis(d, roles)
        result.update(beleg=BELEG, offen=OFFEN,
                      referenzgraph={'set_beschriftungen': count, 'referenzanzahl': refs_count,
                                     'referenzen': targets})
        result['felder']['referenzanzahl'] = a._source(tag, gp[1])
        result['felder']['referenzen'] = [a._source(tag, pair) for pair in gp[2:2+refs_count]]
        result['felder']['referenzwerte'] = sources_ref
        a._require(member.nr not in records, "Kantentext_Label_mehrfach")
        records[member.nr] = result
    a._require(len(paths) == 1, "Kantentext_Setpfad_mehrdeutig")
    path = next(iter(paths.values()))
    a._incoming(seg, path, all_refs, rev)
    a._incoming(seg, aset, [owner_link, path, *tags], rev)
    for nr, obj in objects.items():
        a._incoming(seg, obj, uses[nr], rev)
    a._require(label.nr in records, "Kantentext_Label_nicht_im_Set")
    return records[label.nr]
