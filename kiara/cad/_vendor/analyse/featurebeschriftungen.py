"""Read contents-attached feature text labels with shared parameters.

Evidence: FORMAT.md, Feature-Textbeschriftungen. Complete reference checks are
structural guards, not a semantic release of opaque feature/CF_REF fields.
No setters, geometry replacement or arbitrary feature definition evaluation.
"""
from collections import defaultdict
from copy import deepcopy

from analyse import beschriftungen as a
from analyse import featurewissen as f
from analyse.protokollfelder import tabelle_lesen, Tabellenfehler
from pkgleser import schalenflaechen, zahl

BELEG = ("Creo 20.3: getrennte neutrale Text-/Parameter-/Richtungsproben; "
         "eindeutige Feature-/Annotationszuordnung in production_a (34 Features, "
         "45 Vorkommen) und production_b (15 Features, 16 Vorkommen); "
         "305 unabhaengig zurueckgehaltene Parametervergleiche, "
         "sechs isolierte Parameterkontraste; FORMAT.md: Feature-Textbeschriftungen; "
         "7.11.2026: label_param Feld 5 = 3 (Gewinde-/Bohrungstexte, Schrift hp_i3098_v) an 22 Labels nativ in "
         "Farbe, Text-/Pfeilgroesse, Pfeilfuellung, Rahmen und Typ gleich; "
         "13.11.2026: zehn CF_LABEL_TRACER_REL-Verknuepfungen in drei Originalen, "
         "22 Featuretexte nativ ueber Inhalt/Text und geordnete Referenzpunkte zugeordnet; "
         "Koerperlage, Flaechenparameter und Darstellungswerte getrennt abgeglichen; "
         "13.11.2026: CUST_FEAT_SHOW SHOW/HIDE/SHOW aendert an einem neutralen "
         "teilgebundenen Featurelabel ausschliesslich A3D_LBL_TAG-Nutzfeld 6 "
         "zwischen B0 und B2; nativer Hidden-Reload bestaetigt die Persistenz. "
         "Dies belegt den individuellen Ausblendzustand, keine globale Sichtbarkeit")
OFFEN = ["Restflags und vollstaendige Feature-/CF_REF-Semantik",
         "Native Identitaet jeder einzelnen CF_REF-Flaeche",
         "A3D_REF-Parameter und genaue PLAIN_TEXT-Fontmetriken",
         "Weitere Annotationstypen, Anheftungen und Darstellungsformen",
         "Allgemeine Transformation, Schreiben und Geometrieersetzung"]


def _attribute(seg, owner, target, symbol, rev=None):
    """A counted, unique list; require all entries, not just one matching link."""
    tab = tabelle_lesen(owner)
    a._require(tab is not None, "Feature_Besitzertabelle_fehlt")
    found = []
    for entry in tab["eintraege"]:
        link = seg.objekte.get(entry["ziel"]) if entry["art"] == "v" else None
        if link is None or link.klasse != "ATTRIBUTE_LINK":
            continue
        pairs = a._fields(link)
        if len(pairs) < 4 or pairs[3][1].art != "z" or pairs[3][1].wert != symbol:
            continue
        n = len(pairs) - 5
        a._require(n > 0, "Feature_Attributliste_leer")
        pairs = a._shape(link, ["ID", "B", "v", "z", "B"] + ["v"] * n)
        a._constants(pairs, {1: 1, 4: n})
        a._require(pairs[2][1].wert == owner.nr, "Feature_Attributbesitzer_falsch")
        members = [a._target(seg, pair, symbol) for pair in pairs[5:]]
        a._require(len({o.nr for o in members}) == n, "Feature_Attributliste_doppelt")
        a._incoming(seg, link, [owner], rev)
        found.append((link, members))
    a._require(len(found) == 1 and target in found[0][1], "Feature_Attributanheftung_falsch")
    return found[0][0]


def _path(seg, path, owner):
    pairs = a._shape(path, ["ID", "B", "B", "v"])
    a._constants(pairs, {1: 0, 2: 0})
    a._require(a._target(seg, pairs[3], owner.klasse) is owner, "Feature_Pfadbesitzer_falsch")


def _tracer(seg, cust, face, rev):
    """The observed extra graph relation; its flags/function stay opaque."""
    relations = [seg.objekte[nr] for nr in rev.get(cust.nr, {})
                 if seg.objekte[nr].klasse == "CF_LABEL_TRACER_REL"]
    a._require(len(relations) <= 1, "Feature_Traceranzahl_unbelegt")
    for relation in relations:
        rp = a._shape(relation, ["ID", "B", "v", "B", "v", "NIL", "NIL", "v"])
        a._constants(rp, {1: 1, 3: 2})
        a._require(a._target(seg, rp[2], "face") is face
                   and a._target(seg, rp[7], "face") is face
                   and a._target(seg, rp[4], "CUST_FEAT") is cust,
                   "Feature_Tracerverknuepfung_falsch")
        link = _attribute(seg, face, relation, "CF_LABEL_TRACER_REL", rev)
        a._require(len(a._fields(link)) == 6,
                   "Feature_Tracerattribut_geteilt_unbelegt")
        a._incoming(seg, relation, [link], rev)
    return relations


def _koerperlagenkontext(seg, body):
    """Existing body-matrix grammar, retaining its actual field anchors."""
    fields = a._fields(body)
    shells = [i for i, pair in enumerate(fields) if pair[1].art == "v"
              and pair[1].wert in seg.objekte and seg.objekte[pair[1].wert].klasse == "shell"]
    a._require(len(shells) == 1, "Feature_Tracerkoerperlage_unbelegt")
    tail = fields[shells[0]+1:shells[0]+18]
    kinds = ["B"] + ["N"]*9 + ["B"]*3 + ["N"]*4
    a._require(len(tail) == len(kinds) and all(a._finite(p[1]) if k == "N"
               else p[1].art == k and type(p[1].wert) is int for p,k in zip(tail,kinds)),
               "Feature_Tracerkoerperlage_unbelegt")
    a._require(zahl(tail[16][1]) == 1., "Feature_Tracerkoerperskalierung_unbelegt")
    nums = [zahl(p[1]) for p in tail[1:10]+tail[13:16]]
    value = dict(matrix=[nums[0:3],nums[3:6],nums[6:9]],verschiebung=nums[9:12],skalierung=1.)
    sources = dict(matrix=[[a._source(body,p) for p in tail[i:i+3]] for i in (1,4,7)],
                   verschiebung=[a._source(body,p) for p in tail[13:16]],skalierung=a._source(body,tail[16]))
    return value, sources


def _group(seg, aset):
    """Validate every label and every consumer of shared parameter objects."""
    sp = a._setfelder(aset)
    n = len(sp) - 10
    a._require(n > 0, "Feature_Setanzahl_unbelegt")
    a._constants(sp, {1: 1, 5: n, 6+n: 1, 7+n: 2147483647, 8+n: 1, 9+n: 0})
    owner = a._target(seg, sp[2], "part_struct")
    # Rueckverweise, Klassenlisten und Koerpertopologie einmal je Set statt je Pruefung (Ingest 5.11.2026:
    # 399 Feature-Labels in einem Set mit 17.730 Objekten); die Pruefungen selbst sind unveraendert.
    rev = a.rueckverweise(seg)
    cluster_paths = seg.von_klasse("CLUSTER_PATH")
    cf_refs = seg.von_klasse("CF_REF")
    topologien = {}
    owner_link = _attribute(seg, owner, aset, "A3D_SET", rev)
    tags = [a._target(seg, p, "A3D_TAG") for p in sp[6:6+n]]
    a._require(len({o.nr for o in tags}) == n, "Feature_Setmitglied_doppelt")
    a._require(len(seg.von_klasse("part_struct")) == 1, "Feature_Inhalt_mehrdeutig")
    uses = defaultdict(list)
    objects = {}
    records = {}
    refs = []
    paths = {}
    custs = set()
    for tag in tags:
        gp = a._tagfelder(tag)
        a._require(len(gp) == 14 and gp[8][1].art == "v", "Feldform_unbekannt")
        a._constants(gp, {1: 1, 3: 0, 6: 0, 10: 1, 11: 1, 12: 0})
        a._require(a._target(seg, gp[0], "A3D_SET") is aset, "Feature_Set_Rueckverweis_falsch")
        cust = a._target(seg, gp[8], "CUST_FEAT")
        a._require(cust.nr not in custs, "Feature_mehrfach_im_Set")
        custs.add(cust.nr)
        cp = f.shape(cust, f.CUST_SHAPE)
        body, cust_link = f._owner(seg, cust, cp)
        a._require(owner.nr in body.verweise(), "Feature_Koerperinhalt_falsch")
        a._require(a._target(seg, cp[50], "A3D_TAG") is tag, "Feature_Tag_Rueckverweis_falsch")
        stick = a._target(seg, cp[3], "attr_stick")
        feat = a._target(seg, cp[51], "FEAT_ATTR")
        feat_fields = a._fields(feat)
        a._require(len(feat_fields) > 3 and a._target(seg, feat_fields[2], "body") is body
                   and a._target(seg, feat_fields[3], "attr_stick") is stick,
                   "Feature_Attributbesitzer_falsch")
        dialog = a._target(seg, cp[52], "CF_DIALOG")
        a._shape(dialog, ["ID", "B", "B", "c", "B"])
        legacy = a._target(seg, cp[49], "CF_LABEL")
        lp = a._shape(legacy, ["ID", "B", "v", "v", "NIL", "NIL", "D", "B", "v"])
        a._constants(lp, {1: 2, 6: 2232872, 7: 1})
        face = a._target(seg, lp[3], "face")
        a._require(a._target(seg, lp[2], "body") is body
                   and a._target(seg, lp[8], "CUST_FEAT") is cust, "Feature_CF_LABEL_Rueckverweis_falsch")
        legacy_body = _attribute(seg, body, legacy, "CF_LABEL", rev)
        legacy_face = _attribute(seg, face, legacy, "CF_LABEL", rev)
        a._incoming(seg, legacy, [cust, legacy_body, legacy_face], rev)
        if body.nr not in topologien:
            topologien[body.nr] = {face.nr for nr in body.verweise()
                                   if nr in seg.objekte and seg.objekte[nr].klasse == "shell"
                                   for face in schalenflaechen(seg, seg.objekte[nr])}
        topology = topologien[body.nr]
        a._require(face.nr in topology, "Feature_Labelreferenz_ausserhalb_Koerper")
        clusters = [o for o in cluster_paths if cust.nr in o.verweise()]
        a._require(len(clusters) == 1, "Feature_Pfad_mehrdeutig")
        cluster = clusters[0]
        _path(seg, cluster, cust)
        members = [o for o in cf_refs if cluster.nr in o.verweise()]
        selected = []
        for member in members:
            mp = a._shape(member, ["ID", "B", "v", "v", "B", "B"])
            a._constants(mp, {1: 1, 4: 1})
            mf = a._target(seg, mp[2], "face")
            a._require(a._target(seg, mp[3], "CLUSTER_PATH") is cluster and mf.nr in topology,
                       "Feature_Mitgliedschaft_ausserhalb_Koerper")
            link = _attribute(seg, mf, member, "CF_REF", rev)
            a._incoming(seg, member, [link], rev)
            selected.append(mf.nr)
        a._require(face.nr in selected and len(set(selected)) == len(selected),
                   "Feature_Flaechenmitgliedschaft_falsch")
        a._incoming(seg, cluster, members, rev)
        tracers = _tracer(seg, cust, face, rev)
        if tracers:
            a._require(face.nr in a._koerperflaechen(seg, body, owner), "Feature_Tracerflaeche_ausserhalb_Koerper")
            body_placement, body_sources = _koerperlagenkontext(seg, body)
        a._incoming(seg, cust, [cust_link, cluster, legacy, tag, *tracers], rev)
        label = a._target(seg, gp[5], "A3D_LBL_TAG")
        d = a._darstellung(seg, label, feature_visibility=True)
        a._require(d["tag"] is tag, "Feature_Label_Rueckverweis_falsch")
        a._incoming(seg, label, [tag, d["plain"]], rev)
        a._incoming(seg, d["plain"], [label], rev)
        ref = a._target(seg, gp[2], "A3D_REF")
        rp = a._referenzfelder(ref)
        a._constants(rp, {1: 1, 6: 0, 9: 0, 13: 1, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0})
        a._require(a._target(seg, rp[2], "face") is face and a._target(seg, rp[8], "A3D_TAG") is tag,
                   "Feature_A3D_Referenz_falsch")
        ref_link = _attribute(seg, face, ref, "A3D_REF", rev)
        a._incoming(seg, ref, [tag, ref_link], rev)
        path = a._target(seg, rp[3], "CLUSTER_PATH")
        _path(seg, path, aset)
        refs.append(ref)
        paths[path.nr] = path
        a._incoming(seg, tag, [aset, cust, label, ref], rev)
        for key, consumer in (("text", d["plain"]), ("frame", d["plain"]),
                              ("arrow", label), ("arrow2", label), ("line", label), ("label_param", label)):
            obj = d[key]
            if key == "frame" and obj is d["plain"]:
                continue
            objects[obj.nr] = obj
            uses[obj.nr].append(consumer)
        roles = {"inhalt": owner, "inhaltsattribut": owner_link, "set": aset, "tag": tag,
                 "referenz": ref, "referenzpfad": path, "referenzflaeche": face,
                 "referenzattribut": ref_link, "feature": cust, "koerper": body,
                 "featureattribut": feat, "featuredialog": dialog, "legacy_label": legacy,
                 "featurepfad": cluster}
        result = a._ergebnis(d, roles, feature=True)
        result.update(beleg=BELEG, offen=OFFEN,
                      referenzgraph={"set_beschriftungen": n, "feature_flaechen": selected,
                                     "cf_ref": [o.nr for o in members],
                                     "einzelne_native_flaechenidentitaeten_belegt": False})
        # Retain the opaque CF_REF selector, never present it as a face ordinal.
        result["restfelder_roh"].extend({"rolle": "feature_flaechenreferenz", "klasse": o.klasse,
            "objekt": o.nr, "felder": [{"feld": i, "offset": t.offset, "art": t.art, "wert": a._raw(t.wert)}
                                       for i, t in a._fields(o)]} for o in members)
        if tracers:
            result["referenzgraph"].update(tracer_relationen=[o.nr for o in tracers],
                referenzposition_roh_raum="koerper",label_positionsablage_raum="teil",koerperlage=body_placement,
                koerper_ort=[seg.eintrag,seg.start,body.nr])
            result["felder"]["koerperlage"] = body_sources
            result["restfelder_roh"].extend({"rolle": "feature_tracer_relation", "klasse": o.klasse,
                "objekt": o.nr, "felder": [{"feld": i, "offset": t.offset, "art": t.art, "wert": a._raw(t.wert)}
                                           for i,t in a._fields(o)]} for o in tracers)
            result["offen"] = [*result["offen"], "Funktion und Flags der CF_LABEL_TRACER_REL"]
        a._require(label.nr not in records, "Feature_Label_mehrfach")
        records[label.nr] = result
    a._require(len(paths) == 1, "Feature_Setpfad_mehrdeutig")
    path = next(iter(paths.values()))
    a._incoming(seg, path, refs, rev)
    a._incoming(seg, aset, [owner_link, path, *tags], rev)
    for nr, obj in objects.items():
        a._incoming(seg, obj, uses[nr], rev)
    return records


def _gruppe(seg, aset):
    """_group je Set einmal je gelesenem Segment: decode() prueft fuer jedes Label das ganze Set; bei 399 Labels
    in einem Set lief dieselbe Pruefung 399-mal (Kiara-Zeitlimit 300 s ueberschritten, Ingest 5.11.2026).
    Voraussetzung wie bei allen Lesern: das Segment wird nach dem Lesen nicht veraendert. Schluessel sind die
    Identitaeten des Objektwoerterbuchs und des Set-Objekts. Nur lesen; decode() gibt Kopien einzelner Labels aus."""
    memo = seg.__dict__.setdefault("_feature_gruppen", {})
    key = (id(seg.objekte), len(seg.objekte), aset.nr, id(aset))
    if key not in memo:
        try:
            memo[key] = (None, _group(seg, aset))
        except (a.Unbelegt, f.Unbelegt, Tabellenfehler) as error:
            memo[key] = (type(error), error.args)
    fehler, wert = memo[key]
    if fehler is not None:
        raise fehler(*wert)
    return wert


def decode(seg, label, *, index=None):
    """A whole-set check prevents unexamined consumers hiding behind sharing."""
    try:
        a._require(seg.objekte.get(label.nr) is label, "Feature_Label_nicht_identisches_Graphobjekt")
        d = a._darstellung(seg, label, feature_visibility=True)
        gp = a._tagfelder(d["tag"])
        a._require(len(gp) == 14 and gp[8][1].art == "v", "Feldform_unbekannt")
        aset = a._target(seg, gp[0], "A3D_SET")
        owner = seg.objekte.get(a._setfelder(aset)[2][1].wert)
        if owner is not None and owner.klasse == "asbm_struct":
            from analyse.baugruppenbeschriftungen import decode as assembly_decode
            return assembly_decode(seg, label, index=index)
        result = _gruppe(seg, aset)
        a._require(label.nr in result, "Feature_Label_nicht_im_Set")
        return deepcopy(result[label.nr])
    except f.Unbelegt as error:
        raise a.Unbelegt(str(error)) from None
    except Tabellenfehler:
        raise a.Unbelegt("Feature_Tabellenform_unbelegt") from None


PARAMETERKLASSEN = frozenset({"A3D_LBL_TAG", "text_param", "arrow_param", "FRAME_PARAM"})
PARAMETERQUELLE = {"datei": "analyse/featurebeschriftungen.py", "abschnitt": "def parameterbelege(seg, native_features):"}
PARAMETERROLLEN = {
    "farbe_code": "Gespeicherte Annotationsfarbe aus :COLOR",
    "textgroesse_mm": "Gespeicherte Textgroesse in internen mm aus :TEXT_SIZE",
    "pfeilgroesse_mm": "Gespeicherte Pfeilgroesse in internen mm aus :ARROW_SIZE",
    "pfeil_gefuellt": "Gespeicherte Pfeilfuellung: 0 Umriss, 1 gefuellt",
    "rahmentyp": "Gespeicherter Textrahmentyp: 0 OFF, 1 BOX, 2 BALLOON",
}


def parameterbelege(seg, native_features):
    """Five held-out roles only, gated by all native-bound feature consumers.

    Text/position helped establish the native match, so they are not counted
    again as independent held-out evidence. Shared physical fields count once.
    This rule does not change the independent WARNING/FILING feature rule.
    """
    result = {}
    for aset in seg.von_klasse("A3D_SET"):
        try:
            records = _gruppe(seg, aset)
        except (a.Unbelegt, f.Unbelegt, Tabellenfehler):
            continue
        if any(r["kette"]["feature"]["objekt"] not in native_features for r in records.values()):
            continue
        by_object = defaultdict(dict)
        for record in records.values():
            for name, meaning in PARAMETERROLLEN.items():
                source = record["felder"][name]
                obj = seg.objekte[source["objekt"]]
                index = sum(t.art != "s" for t in obj.felder[:source["feld"]])
                by_object[obj.nr][index] = meaning
        for nr, roles in by_object.items():
            obj = seg.objekte[nr]
            key = (obj.offset, nr)
            a._require(key not in result, "Feature_Parameterbeleg_mehrdeutig")
            result[key] = f.Feldbeleg(nr, obj.klasse, obj.offset, f._fields_fingerprint(obj), tuple(sorted(roles.items())))
    return result
