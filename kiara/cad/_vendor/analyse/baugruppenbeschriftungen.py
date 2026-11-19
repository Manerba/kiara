"""Contents-attached assembly feature texts with explicitly bound external faces.

The file index is required. Short IDs are resolved only inside the contents of
the exact npart3 selected by CLUSTER_PATH, never by a global object-ID search.
NIL placeholders on external targets are not presented as reciprocal pointers.
The opaque feature/legacy flags and selectors remain raw.
"""
import math
from collections import defaultdict
from copy import deepcopy

from analyse import beschriftungen as a
from analyse import featurebeschriftungen as f
from analyse import featurewissen
from analyse.protokollfelder import tabelle_lesen
from objstrom import KURZ_ID_FORMEN
from pkgleser import Fehler, kurz_id, platzierung, schalenflaechen

BELEG = ("Creo 20.3, 12.11.2026: elf Baugruppen-Featuretexte in drei Originalen; "
         "Besitzerart, Inhaltsnummer, Texte und Einstellungen, externe Flaechen mit "
         "Parameterbox, lokalen Referenzpositionen und u/v sowie globale Punkte "
         "ueber Kind- und Baugruppenlage unabhaengig verglichen; FORMAT.md")
OFFEN = ["Restflags, Featuredefinitionen, CF_REF-Selektoren und CFL_REF-Feldrollen",
         "Mehrstufige externe Pfade, weitere Zielklassen und allgemeine Transformation",
         "Allgemeine Feldsemantik, Schreiben und Geometrieersetzung"]


def _anchor(seg, obj):
    return dict(eintrag=seg.eintrag, segment_start=seg.start, segment_id=seg.id,
                objekt=obj.nr, klasse=obj.klasse, id=obj.id)


def _child(seg, owner, child):
    """Require the very occurrence in the owner's explicit child-link list."""
    matches = []
    for nr in owner.verweise():
        link = seg.objekte.get(nr)
        if link is None or link.klasse != "child_link":
            continue
        lp = a._shape(link, ["B", "v", "v"])
        a._constants(lp, {0: 1})
        a._require(a._target(seg, lp[2], "asbm_struct") is owner,
                   "Baugruppentext_Kindlink_Besitzer_falsch")
        node = a._target(seg, lp[1], "child_node")
        np = a._shape(node, ["ID", "B", "v", "v", "NIL"])
        a._constants(np, {1: 1})
        a._require(a._target(seg, np[3], "child_link") is link,
                   "Baugruppentext_Kindknoten_Rueckverweis_falsch")
        if np[2][1].wert == child.nr:
            matches.append((link, node))
    a._require(len(matches) == 1, "Baugruppentext_Kind_nicht_eindeutig_im_Besitzer")
    return matches[0]


def _path(seg, path, destination, owner, rev):
    cp = a._shape(path, ["ID", "B", "v", "B", "v", "v"])
    a._constants(cp, {1: 1, 3: 1})
    child = a._target(seg, cp[2], "npart3")
    a._require(a._target(seg, cp[4], "npart3") is child
               and a._target(seg, cp[5], destination.klasse) is destination,
               "Baugruppentext_Pfad_Rueckverweis_falsch")
    _child(seg, owner, child)
    link = f._attribute(seg, child, path, "CLUSTER_PATH", rev)
    return child, link


def _contents(index, child, cache):
    if child.nr not in cache:
        try:
            content_id, placement = platzierung(child)
        except (Fehler, ValueError, KeyError, IndexError):
            raise a.Unbelegt("Baugruppentext_Kindplatzierung_unbelegt") from None
        a._require(type(content_id) is str and placement["skalierung"] == 1.,
                   "Baugruppentext_Kindplatzierung_unbelegt")
        nums = [v for row in placement["matrix"] for v in row] + placement["verschiebung"]
        a._require(len(nums) == 12 and all(math.isfinite(v) for v in nums),
                   "Baugruppentext_Kindplatzierung_unbelegt")
        fragments = index.get(content_id, [])
        candidates = [s for s in fragments if s.von_klasse("part_struct")]
        a._require(len(candidates) == 1, "Baugruppentext_Kindinhalt_mehrdeutig")
        target = candidates[0]
        owners = [o for o in target.von_klasse("part_struct") if o.id == content_id]
        a._require(target.id == content_id and len(owners) == 1,
                   "Baugruppentext_Kindinhalt_falsch")
        bodies = [o for o in target.von_klasse("body") if owners[0].nr in o.verweise()]
        a._require(len(bodies) == 1, "Baugruppentext_Kindkoerper_mehrdeutig")
        body = bodies[0]
        faces = a._koerperflaechen(target, body, owners[0])
        cache[child.nr] = (target, body, faces, placement, fragments)
    return cache[child.nr]


def _external_id(pair):
    a._require(pair[1].art == "8012", "Baugruppentext_Externe_ID_unbelegt")
    try:
        raw = bytes.fromhex(pair[1].wert)
        a._require(len(raw) == 1 + sum(KURZ_ID_FORMEN[raw[0]]),
                   "Baugruppentext_Externe_ID_unbelegt")
        return kurz_id(pair[1].wert)
    except (ValueError, KeyError, IndexError, TypeError):
        raise a.Unbelegt("Baugruppentext_Externe_ID_unbelegt") from None


def _external(index, child, pair, kind, cache):
    target_id = _external_id(pair)
    target, body, faces, placement, fragments = _contents(index, child, cache)
    matches = [(s, o) for s in fragments for o in s.objekte.values() if o.id == target_id]
    a._require(len(matches) == 1 and matches[0][0] is target and matches[0][1].klasse == kind,
               "Baugruppentext_Externes_Ziel_mehrdeutig")
    obj = matches[0][1]
    a._require((kind == "face" and obj.nr in faces) or (kind == "body" and obj is body),
               "Baugruppentext_Externes_Ziel_ausserhalb_Kindkoerper")
    return target, obj, placement


def _placeholder(seg, obj, symbol):
    """Observed empty external backlink slot, not an invented reciprocal ID."""
    tab = tabelle_lesen(obj)
    a._require(tab is not None, "Baugruppentext_Externe_Tabelle_fehlt")
    found = []
    rev = a.rueckverweise(seg)
    for row in tab["eintraege"]:
        link = seg.objekte.get(row["ziel"]) if row["art"] == "v" else None
        if link is None or link.klasse != "ATTRIBUTE_LINK":
            continue
        lp = a._fields(link)
        if len(lp) < 4 or lp[3][1].art != "z" or lp[3][1].wert != symbol:
            continue
        lp = a._shape(link, ["ID", "B", "v", "z", "B", "NIL"])
        a._constants(lp, {1: 1, 4: 1})
        a._require(a._target(seg, lp[2], obj.klasse) is obj,
                   "Baugruppentext_Externer_Attributbesitzer_falsch")
        a._incoming(seg, link, [obj], rev)
        found.append(link)
    a._require(len(found) == 1, "Baugruppentext_Externer_Attributplatz_mehrdeutig")
    return found[0]


def _group(seg, aset, index):
    a._require(index is not None and callable(getattr(index, "get", None)),
               "Baugruppentext_Dateikontext_fehlt")
    same = index.get(seg.id, [])
    a._require(len(same) == 1 and same[0] is seg, "Baugruppentext_Fremder_Dateikontext")
    sp = a._setfelder(aset)
    n = len(sp)-10
    a._constants(sp, {1: 1, 5: n, 6+n: 1, 7+n: 2147483647, 8+n: 1, 9+n: 0})
    owner = a._target(seg, sp[2], "asbm_struct")
    rev = a.rueckverweise(seg)
    owner_link = f._attribute(seg, owner, aset, "A3D_SET", rev)
    tags = [a._target(seg, p, "A3D_TAG") for p in sp[6:6+n]]
    a._require(n > 0 and len({o.nr for o in tags}) == n, "Baugruppentext_Setmitglieder_unbelegt")
    paths, uses, objects, records, content_cache = {}, defaultdict(list), {}, {}, {}
    for tag in tags:
        gp = a._tagfelder(tag)
        a._require(len(gp) == 14 and gp[8][1].art == "v", "Feldform_unbekannt")
        # The assembly form has its own observed tail, distinct from part features.
        a._constants(gp, {1: 1, 3: 0, 6: 0, 10: 0, 11: 0, 12: 0})
        a._require(a._target(seg, gp[0], "A3D_SET") is aset, "Baugruppentext_Set_Rueckverweis_falsch")
        cust = a._target(seg, gp[8], "CUST_FEAT")
        cp = featurewissen.shape(cust, featurewissen.CUST_SHAPE)
        a._require(a._target(seg, cp[2], "asbm_struct") is owner
                   and a._target(seg, cp[50], "A3D_TAG") is tag,
                   "Baugruppentext_Feature_Rueckverweis_falsch")
        cust_link = f._attribute(seg, owner, cust, "CUST_FEAT", rev)
        stick = a._target(seg, cp[3], "attr_stick")
        feat = a._target(seg, cp[51], "FEAT_ATTR")
        ff = a._fields(feat)
        a._require(len(ff) > 3 and a._target(seg, ff[2], "asbm_struct") is owner
                   and a._target(seg, ff[3], "attr_stick") is stick,
                   "Baugruppentext_Featureattribut_falsch")
        dialog = a._target(seg, cp[52], "CF_DIALOG")
        a._shape(dialog, ["ID", "B", "B", "c", "B"])
        legacy = a._target(seg, cp[49], "CF_LABEL")
        lp = a._shape(legacy, ["ID", "B", "v", "NIL", "NIL", "D", "B", "v"])
        a._constants(lp, {1: 1, 6: 1})
        a._require(lp[5][1].wert in (135720, 2232872)
                   and a._target(seg, lp[2], "asbm_struct") is owner
                   and a._target(seg, lp[7], "CUST_FEAT") is cust,
                   "Baugruppentext_Legacyform_unbelegt")
        legacy_link = f._attribute(seg, owner, legacy, "CF_LABEL", rev)
        label = a._target(seg, gp[5], "A3D_LBL_TAG")
        d = a._darstellung(seg, label)
        a._require(d["tag"] is tag, "Baugruppentext_Label_Rueckverweis_falsch")
        ref = a._target(seg, gp[2], "A3D_REF")
        rp = a._shape(ref, ["ID", "B", "8012", "v", "N", "N", "B", "NIL", "v", "B"]
                      + ["N"]*6 + ["B"] + ["N"]*3 + ["NIL", "B"])
        a._constants(rp, {1: 1, 6: 0, 9: 0, 13: 1, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 21: 0})
        a._require(a._target(seg, rp[8], "A3D_TAG") is tag, "Baugruppentext_Referenz_Tag_falsch")
        path = a._target(seg, rp[3], "CLUSTER_PATH")
        child, path_link = _path(seg, path, aset, owner, rev)
        target, face, placement = _external(index, child, rp[2], "face", content_cache)
        ref_slot = _placeholder(target, face, "A3D_REF")
        feature_paths = [o for o in seg.von_klasse("CLUSTER_PATH") if cust.nr in o.verweise()]
        legacy_paths = [o for o in seg.von_klasse("CLUSTER_PATH") if legacy.nr in o.verweise()]
        a._require(len(feature_paths) == len(legacy_paths) == 1, "Baugruppentext_Featurepfad_mehrdeutig")
        feature_path, legacy_path = feature_paths[0], legacy_paths[0]
        for p, dest in ((feature_path, cust), (legacy_path, legacy)):
            other_child, other_link = _path(seg, p, dest, owner, rev)
            a._require(other_child is child and other_link is path_link, "Baugruppentext_Gekreuzte_Kindpfade")
        members = [o for o in seg.von_klasse("CF_REF") if feature_path.nr in o.verweise()]
        a._require(len(members) == 1, "Baugruppentext_Featureflaechenanzahl_unbelegt")
        mp = a._shape(members[0], ["ID", "B", "8012", "v", "B", "B"])
        a._constants(mp, {1: 1, 4: 1, 5: -1})
        _, selected, _ = _external(index, child, mp[2], "face", content_cache)
        a._require(selected is face, "Baugruppentext_Gekreuzte_Featurereferenz")
        _placeholder(target, face, "CF_REF")
        a._incoming(seg, members[0], [], rev)
        cfl = [o for o in seg.von_klasse("CFL_REF") if legacy_path.nr in o.verweise()]
        a._require(len(cfl) == 2, "Baugruppentext_Legacyreferenzen_unbelegt")
        resolved = []
        for other in cfl:
            op = a._shape(other, ["ID", "B", "8012", "v", "B"])
            a._constants(op, {1: 1, 4: 2})
            oid = _external_id(op[2])
            kind = "face" if oid == face.id else "body"
            _, obj, _ = _external(index, child, op[2], kind, content_cache)
            _placeholder(target, obj, "CFL_REF")
            a._incoming(seg, other, [], rev)
            resolved.append(obj)
        a._require({o.klasse for o in resolved} == {"body", "face"}, "Baugruppentext_Legacyziele_falsch")
        a._incoming(seg, feature_path, [path_link, *members], rev)
        a._incoming(seg, legacy_path, [path_link, *cfl], rev)
        paths.setdefault(path.nr, (path, path_link, []))[2].append(ref)
        a._incoming(seg, cust, [cust_link, feature_path, legacy, tag], rev)
        a._incoming(seg, legacy, [cust, legacy_link, legacy_path], rev)
        a._incoming(seg, tag, [aset, cust, label, ref], rev)
        a._incoming(seg, label, [tag, d["plain"]], rev)
        a._incoming(seg, d["plain"], [label], rev)
        a._incoming(seg, ref, [tag], rev)
        for key, consumer in (("text", d["plain"]), ("frame", d["plain"]),
                              ("arrow", label), ("arrow2", label), ("line", label), ("label_param", label)):
            obj = d[key]
            if key == "frame" and obj is d["plain"]:
                continue
            objects[obj.nr] = obj
            uses[obj.nr].append(consumer)
        roles = dict(inhalt=owner, inhaltsattribut=owner_link, set=aset, tag=tag, referenz=ref,
                     referenzpfad=path, referenzvorkommen=child, feature=cust, featureattribut=feat,
                     featuredialog=dialog, legacy_label=legacy, featurepfad=feature_path, legacypfad=legacy_path)
        result = a._ergebnis(d, roles, feature=True)
        result.update(beleg=BELEG, offen=OFFEN, referenzgraph=dict(set_beschriftungen=n,
                      externe_referenz=dict(ziel=_anchor(target, face), inhalt_id=target.id,
                      vorkommen=_anchor(seg, child), platzierung=placement,
                      externer_attributplatz=_anchor(target, ref_slot)),
                      cf_ref=[o.nr for o in members], cfl_ref=[o.nr for o in cfl]))
        result["kette"]["referenzflaeche"] = _anchor(target, face)
        result["felder"]["externe_referenz_id"] = a._source(ref, rp[2])
        result["restfelder_roh"].extend(dict(rolle="externe_featurereferenz", klasse=o.klasse, objekt=o.nr,
            felder=[dict(feld=i, offset=t.offset, art=t.art, wert=a._raw(t.wert)) for i,t in a._fields(o)])
            for o in members+cfl)
        a._require(label.nr not in records, "Baugruppentext_Label_mehrfach")
        records[label.nr] = result
    for path, link, refs in paths.values():
        a._incoming(seg, path, [link, *refs], rev)
    a._incoming(seg, aset, [owner_link, *tags, *(p[0] for p in paths.values())], rev)
    for nr, obj in objects.items():
        a._incoming(seg, obj, uses[nr], rev)
    return records


def decode(seg, label, *, index=None):
    a._require(seg.objekte.get(label.nr) is label, "Baugruppentext_Fremdes_Labelobjekt")
    d = a._darstellung(seg, label)
    gp = a._tagfelder(d["tag"])
    aset = a._target(seg, gp[0], "A3D_SET")
    records = _group(seg, aset, index)
    a._require(label.nr in records, "Baugruppentext_Label_nicht_im_Set")
    return deepcopy(records[label.nr])
