"""Separate inventory of explicit part-owned CONTAINER/CONT_STRUCT branches.

The ordinary reader model is not extended or mutated. These branches are not
normal assembly occurrences, and this module assigns no quantity or mass role.
All names/paths in detailed results are private source data.
"""
import copy
import hashlib
import math

import pkgleser


def _anchor(seg, obj):
    return {"eintrag": seg.eintrag, "segment": seg.start, "objekt": obj.nr,
            "klasse": obj.klasse, "id": obj.id}


def _fields(obj):
    return [(i, t) for i, t in enumerate(obj.felder) if t.art not in ("s", "8002")]


def _place(pair):
    i, token = pair
    return {"feld": i, "offset": token.offset}


def _children(seg, owner):
    """Only the witnessed reciprocal owner/link/node relationship is accepted."""
    children, problems = [], []
    seen = set()
    for field, token in enumerate(owner.felder):
        if token.art != "v" or not isinstance(token.wert, int):
            continue
        link = seg.objekte.get(token.wert)
        if link is None or link.klasse != "child_link":
            continue
        source = {"besitzer_verweis": {"feld": field, "offset": token.offset},
                  "besitzer": _anchor(seg, owner), "link": _anchor(seg, link)}
        fs = _fields(link)
        source.update(link_feldform=[t.art for _, t in fs],
                      link_roh_verweise=[t.wert for _, t in fs if t.art == "v"])
        valid = (len(fs) == 3 and [t.art for _, t in fs] == ["B", "v", "v"] and fs[0][1].wert == 1)
        if not valid or fs[2][1].wert != owner.nr:
            problems.append({"grund": "Kindlink_Form_unbelegt" if not valid else "Kindlink_fremder_Besitzer", **source})
            continue
        node = seg.objekte.get(fs[1][1].wert) if isinstance(fs[1][1].wert, int) else None
        ns = _fields(node) if node is not None and node.klasse == "child_node" else []
        if node is None or node.klasse != "child_node":
            problems.append({"grund": "Kindknoten_nicht_aufloesbar" if node is None else "Kindknoten_Klasse_unbelegt", **source})
            continue
        source.update(knoten=_anchor(seg, node), knoten_feldform=[t.art for _, t in ns],
                      knoten_roh_verweise=[t.wert for _, t in ns if t.art == "v"])
        valid = (len(ns) == 5 and [t.art for _, t in ns] == ["ID", "B", "v", "v", "NIL"]
                 and ns[1][1].wert == 1 and ns[4][1].wert == 1)
        if not valid or ns[3][1].wert != link.nr:
            problems.append({"grund": "Kindknoten_Form_unbelegt" if not valid else "Kindknoten_fremder_Link", **source})
            continue
        child = seg.objekte.get(ns[2][1].wert) if isinstance(ns[2][1].wert, int) else None
        if child is None or child.nr in seen:
            problems.append({"grund": "Kindziel_nicht_aufloesbar" if child is None else "Kindziel_mehrfach", **source})
            continue
        seen.add(child.nr)
        source.update(link_ziel=_place(fs[1]),
                      link_besitzer=_place(fs[2]), knoten_ziel=_place(ns[2]),
                      knoten_link=_place(ns[3]), kind=_anchor(seg, child))
        children.append((child, source))
    return children, problems


def _unique_root(index, identity, klass):
    roots = [(s, o) for s in index.get(identity, []) if s.klasse == klass
             for o in s.von_klasse(klass) if o.id == identity]
    return roots[0] if len(roots) == 1 else None


def _raw_definition(index, seg, owner):
    """Bind a definition to complete original segment bytes and stream header.

    A parsed-only dictionary cannot prove byte identity. No duplicate is resolved
    in that case. The q..r span includes declarations, symbol tables and the final
    r; the complete stream header also binds the parser's text mode.
    """
    directories = index.liste if isinstance(index, pkgleser.Verzeichnisse) else [index]
    candidates = [(ordinal, v) for ordinal, v in enumerate(directories)
                  if isinstance(v, pkgleser.Segmentverzeichnis) and v.eintrag == seg.eintrag
                  and (seg.start, seg.ende, seg.klasse, seg.id) in v.koepfe]
    if len(candidates) != 1:
        return None
    ordinal, directory = candidates[0]
    raw = directory.b[seg.start:seg.ende + 1]
    if not raw.startswith(b"q") or not raw.endswith(b"r"):
        return None
    first = min(start for start, _, _, _ in directory.koepfe)
    header = directory.b[:first]
    # A mutated cache must not claim the identity of unchanged source bytes.
    warnings = []
    parsed = pkgleser.segment_lesen(directory.b, seg.start, seg.eintrag, warnings)
    if warnings or parsed != seg:
        return None
    return {"anker": _anchor(seg, owner), "index_ordinal": ordinal,
            "segment_ende_exklusiv": seg.ende + 1,
            "segment_sha256": hashlib.sha256(raw).hexdigest(),
            "stromkopf_ende_exklusiv": first,
            "stromkopf_sha256": hashlib.sha256(header).hexdigest()}


def _parent_definition(index, identity):
    roots = [(s, o) for s in index.get(identity, []) if s.klasse == "part_struct"
             for o in s.von_klasse("part_struct") if o.id == identity]
    if len(roots) == 1:
        return roots[0], None
    diagnostic = {"teil_id": identity, "definitionen": [_anchor(s, o) for s, o in roots]}
    if not roots:
        return None, diagnostic
    bindings = [_raw_definition(index, s, o) for s, o in roots]
    diagnostic["rohbindungen"] = bindings
    if any(binding is None for binding in bindings):
        diagnostic["detail"] = "Vollstaendige_Rohbindung_fehlt"
        return None, diagnostic
    locations = {(b["index_ordinal"], b["anker"]["segment"]) for b in bindings}
    if len(locations) != len(roots):
        diagnostic["detail"] = "Definitionsquelle_mehrfach"
        return None, diagnostic
    if len({(b["segment_sha256"], b["stromkopf_sha256"]) for b in bindings}) != 1:
        diagnostic["detail"] = "Segmentbytes_oder_Stromkontext_abweichend"
        return None, diagnostic
    diagnostic.update(zuordnung="identische_rohdefinitionen",
                      externe_ziele="weiterhin_separat_streng_aufloesen")
    return roots[0], diagnostic


KNOWN_ATTACHMENTS = {"PD_ATTR0": ("teil_produktdaten", "PD_ATTR_SP_STRUCT0"),
                     "new_layout": ("teil_layout", "wp_struct")}


def _known_attachment(seg, child, chain):
    """Classify the witnessed directed local attachment, not its remaining data."""
    relation, expected = KNOWN_ATTACHMENTS[child.klasse]
    record = {"beziehung": relation, "anhang_anker": _anchor(seg, child), "kette": chain,
              "mengenrolle": "nicht_aus_dieser_beziehung_abgeleitet",
              "restsemantik": "nicht_durch_klassifikation_freigegeben"}
    # One instance may not silently be attached to multiple child nodes.
    consumers = [obj for obj in seg.von_klasse("child_node") if child.nr in obj.verweise()]
    if len(consumers) != 1 or consumers[0].nr != chain["knoten"]["objekt"]:
        return record, "Kindziel_Besitzer_mehrdeutig"
    if sum(obj.id == child.id for obj in seg.objekte.values()) != 1 or not isinstance(child.id, str):
        return record, "Anhang_ID_nicht_eindeutig"
    # Only the observed local pointer + complete placement block is classified.
    # External references and multiple eligible blocks remain explicit unknowns.
    candidates = [(i, token) for i, token in enumerate(child.felder)
                  if token.art == "v" and isinstance(token.wert, int)
                  and len(child.felder) > i + 9
                  and all(pkgleser.ist_zahl(t) for t in child.felder[i + 1:i + 10])]
    if any(t.art in ("8011", "8012") for t in child.felder) or len(candidates) != 1:
        return record, "Anhang_Platzierungsform_unbelegt"
    field, token = candidates[0]
    try:
        identity, pose = pkgleser.platzierung(child)
    except (pkgleser.Fehler, ValueError, TypeError, KeyError, IndexError):
        return record, "Anhang_Platzierung_unlesbar"
    values = [v for row in pose["matrix"] for v in row] + pose["verschiebung"] + [pose["skalierung"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return record, "Anhang_Platzierung_nicht_endlich"
    if identity != ("lokal", token.wert):
        return record, "Anhang_Platzierungsziel_widerspruechlich"
    target = seg.objekte.get(token.wert)
    if target is None or target.klasse != expected:
        return record, "Anhang_Strukturklasse_unbelegt"
    if not isinstance(target.id, str) or sum(obj.id == target.id for obj in seg.objekte.values()) != 1:
        return record, "Anhang_Struktur_ID_nicht_eindeutig"
    record.update(zuordnung="expliziter_lokaler_verweis", struktur_anker=_anchor(seg, target),
                  lage_lokal=copy.deepcopy(pose), platzierungsverweis=_place(candidates[0]),
                  platzierung_ort=[{"feld": i, "offset": t.offset}
                                    for i, t in enumerate(child.felder)
                                    if field <= i <= field + 21],
                  beleg="FORMAT.md: STEP-Produktdaten und Teil-Layouts; native Teilbesitzer-Gegenprobe 13.11.2026")
    return record, None


def _placement(obj):
    identity, pose = pkgleser.platzierung(obj)
    if not isinstance(identity, str):
        raise ValueError("Lokale_Containerdeklaration_unbelegt")
    fields = [(i, t) for i, t in enumerate(obj.felder) if t.art in ("8011", "8012")]
    if len(fields) != 1:
        raise ValueError("Platzierungsverweis_mehrdeutig")
    # platzierung() prefers the first external reference. A competing local
    # placement prefix must not be silently ignored beside that reference.
    def complete(start):
        tail = obj.felder[start + 1:start + 22]
        return (len(tail) == 21
                and all(pkgleser.ist_zahl(t) for t in tail[:9])
                and all(t.art == "B" for t in tail[9:12])
                and all(pkgleser.ist_zahl(t) for t in tail[12:16])
                and all(t.art == "B" for t in tail[16:]))

    candidates = [i for i, t in enumerate(obj.felder)
                  if ((t.art in ("8011", "8012") and complete(i))
                      or (t.art == "v" and len(obj.felder) > i + 9
                          and all(pkgleser.ist_zahl(v) for v in obj.felder[i + 1:i + 10])))]
    if candidates != [fields[0][0]]:
        raise ValueError("Platzierungsverweis_mehrdeutig")
    values = [v for row in pose["matrix"] for v in row] + pose["verschiebung"] + [pose["skalierung"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Containerplatzierung_nicht_endlich")
    return identity, copy.deepcopy(pose), _place(fields[0])


def _diagnostic_index(index, warnings):
    """Share immutable bytes/header tables, isolate lazy-read warnings/cache."""
    if isinstance(index, pkgleser.Verzeichnisse):
        local = copy.copy(index)
        local.liste = [_diagnostic_index(item, warnings) for item in index.liste]
        return local
    if isinstance(index, pkgleser.Segmentverzeichnis):
        local = copy.copy(index)
        local.warnungen = warnings
        local.cache = dict(index.cache)
        return local
    if isinstance(index, dict):
        return index
    raise TypeError("Ein vorhandener pkgleser-Index wird benoetigt")


def inventar(modell, index):
    """Inventory additional contents using the existing reader index and model.

    Placement numbers use the existing placement decoder. They are reported
    separately, with their source anchor and no derived ordinary occurrence.
    The model's objekte/teile and all source bytes remain unchanged.
    """
    source_warnings = []
    index = _diagnostic_index(index, source_warnings)
    occurrences = {}
    stack = [(node, "") for node in modell["objekte"]]
    while stack:
        node, parent = stack.pop()
        name = node.get("name")
        path = parent + "/" + (name if isinstance(name, str) else "?")
        if node.get("klasse") == "npart3" and isinstance(node.get("teil_id"), str):
            occurrences.setdefault(node["teil_id"], []).append({"pfad": path,
                "vorkommen_anker": copy.deepcopy(node.get("ort")),
                "lage_global": copy.deepcopy(node.get("lage_global"))})
        stack.extend((child, path) for child in node.get("kinder", []))
    relations, issues, additional = [], [], set()
    attachments, duplicates = [], []
    for parent_id in sorted(occurrences):
        selected, definitions = _parent_definition(index, parent_id)
        if selected is None:
            issues.append({"grund": "Teilinhalt_nicht_eindeutig", **definitions})
            continue
        if definitions is not None:
            duplicates.append(definitions)
        seg, owner = selected
        children, problems = _children(seg, owner)
        issues.extend(problems)
        for container, chain in children:
            if container.klasse in KNOWN_ATTACHMENTS:
                attachment, problem = _known_attachment(seg, container, chain)
                attachment.update(eltern_teil_id=parent_id, eltern_inhalt_anker=_anchor(seg, owner),
                                  eltern_vorkommen=copy.deepcopy(occurrences[parent_id]))
                if definitions is not None:
                    attachment["eltern_rohdefinitionen"] = copy.deepcopy(definitions["rohbindungen"])
                if problem:
                    attachment.update(zuordnung="ungeklaert", grund=problem)
                    issues.append({"grund": problem, **chain})
                attachments.append(attachment)
                continue
            if container.klasse != "CONTAINER":
                issues.append({"grund": "Teilkindklasse_unbelegt", **chain})
                continue
            record = {"beziehung": "teil_inhaltscontainer", "mengenrolle": "ungeklaert",
                      "eltern_teil_id": parent_id, "eltern_inhalt_anker": _anchor(seg, owner),
                      "eltern_vorkommen": copy.deepcopy(occurrences[parent_id]),
                      "container_anker": _anchor(seg, container), "kette": chain,
                      "name": pkgleser.name_von(seg, container), "inhalte": []}
            relations.append(record)
            try:
                identity, pose, field = _placement(container)
            except (pkgleser.Fehler, ValueError, TypeError, KeyError, IndexError) as exc:
                record.update(zuordnung="ungeklaert", grund=str(exc))
                continue
            record.update(container_inhalt_id=identity, lage_lokal=pose,
                          platzierungsverweis=field, lage_beleg="INGEST_FELDER.md, F25")
            selected = _unique_root(index, identity, "CONT_STRUCT")
            if selected is None:
                record.update(zuordnung="ungeklaert", grund="CONT_STRUCT_nicht_eindeutig")
                continue
            cseg, content = selected
            record["container_struktur_anker"] = _anchor(cseg, content)
            cchildren, problems = _children(cseg, content)
            issues.extend(problems)
            for child, cchain in cchildren:
                item = {"anker": _anchor(cseg, child), "kette": cchain,
                        "name": pkgleser.name_von(cseg, child), "mengenrolle": "ungeklaert"}
                record["inhalte"].append(item)
                if child.klasse != "npart3":
                    item.update(zuordnung="ungeklaert", grund="Containerkindklasse_unbelegt")
                    continue
                try:
                    target, placement, place = _placement(child)
                except (pkgleser.Fehler, ValueError, TypeError, KeyError, IndexError) as exc:
                    item.update(zuordnung="ungeklaert", grund=str(exc))
                    continue
                item.update(teil_id=target, lage_lokal=placement, platzierungsverweis=place)
                target_root = _unique_root(index, target, "part_struct")
                if target_root is None:
                    item.update(zuordnung="ungeklaert", grund="Containerteil_nicht_eindeutig")
                    continue
                tseg, part = target_root
                item.update(zuordnung="expliziter_verweis", inhalt_anker=_anchor(tseg, part))
                if target not in modell["teile"]:
                    additional.add(target)
            record["zuordnung"] = "expliziter_verweis" if not problems and all(
                item.get("zuordnung") == "expliziter_verweis" for item in record["inhalte"]) else "ungeklaert"
    from analyse.baugruppencontainer import inventar as baugruppen_inventar
    return {"schema": 3, "beziehungen": relations, "zusaetzliche_teilinhalte": sorted(additional),
            "weitere_teilkinder": attachments, "identische_mehrfachdefinitionen": duplicates,
            "baugruppencontainer": baugruppen_inventar(modell, index),
            "unbekannte_verknuepfungen": issues, "mengenrolle": "ungeklaert",
            "quell_warnungen": source_warnings,
            "beleg": "Creo 20.3, 12.11.2026: 13 Containerinhalte, 94 Hierarchieknoten und 26 lokale Container-/Teilplatzierungen; INGEST_FELDER.md, F25",
            "native_pruefung": "separater_Referenzbeleg_erforderlich"}


def lesen(pfad):
    """Read-only convenience API; detailed return value contains private names."""
    reader = pkgleser.Leser(mit_punkten=False)
    model = reader.lesen(pfad, sparsam=True)
    result = inventar(model, reader.index)
    result["leser_warnungen"] = list(reader.warnungen)
    return result
