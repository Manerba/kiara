"""Separate inventory of witnessed assembly-owned, nested containers.

No ordinary occurrence, quantity, mass or source model is changed. Only explicit
CONTAINER/CONT_STRUCT/npart3 branches are followed; all other kinds stay unknown.
"""
import copy
import math

import pkgleser
from analyse import inhaltscontainer as base


def _key(seg, obj):
    return (seg.eintrag, seg.start, obj.nr)


def _at_anchor(index, anchor):
    if (not isinstance(anchor, (list, tuple)) or len(anchor) != 3
            or not isinstance(anchor[0], str) or not all(type(n) is int for n in anchor[1:])):
        return None
    entry, start, number = anchor
    if isinstance(index, dict):
        segments = {id(s): s for values in index.values() for s in values
                    if s.eintrag == entry and s.start == start}
        segments = list(segments.values())
    else:
        directories = index.liste if isinstance(index, pkgleser.Verzeichnisse) else [index]
        segments = [pkgleser.segment_lesen(d.b, start, entry, d.warnungen) for d in directories
                    if d.eintrag == entry and any(h[0] == start for h in d.koepfe)]
    if len(segments) != 1 or segments[0] is None:
        return None
    seg = segments[0]
    return (seg, seg.objekte[number]) if number in seg.objekte else None


def _assembly_owner(index, node):
    located = _at_anchor(index, node.get("ort"))
    if located is None:
        return None, "Baugruppenvorkommen_nicht_eindeutig"
    seg, occurrence = located
    if occurrence.klasse != "nasmb3" or occurrence.id != node.get("id"):
        return None, "Baugruppenvorkommen_fremd"
    references = [t for t in occurrence.felder if t.art in ("8011", "8012")]
    references += [t for i, t in enumerate(occurrence.felder) if t.art == "v"
                   and len(occurrence.felder) > i + 9
                   and all(pkgleser.ist_zahl(v) for v in occurrence.felder[i + 1:i + 10])]
    if len(references) != 1:
        return None, "Baugruppenstrukturverweis_mehrdeutig"
    try:
        target, _ = pkgleser.platzierung(occurrence)
    except (pkgleser.Fehler, ValueError, TypeError, KeyError, IndexError):
        return None, "Baugruppenstrukturverweis_unlesbar"
    if isinstance(target, tuple) and len(target) == 2 and target[0] == "lokal":
        root = seg.objekte.get(target[1])
        selected = (seg, root) if root is not None and root.klasse == "asbm_struct" else None
    else:
        selected = base._unique_root(index, target, "asbm_struct") if isinstance(target, str) else None
    if selected is None or list(_key(*selected)) != node.get("struktur"):
        return None, "Baugruppenstrukturanker_widerspruechlich"
    return selected, None


def _finite_placement(obj):
    target, pose, field = base._placement(obj)
    values = [v for row in pose["matrix"] for v in row] + pose["verschiebung"] + [pose["skalierung"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Containerplatzierung_nicht_endlich")
    return target, pose, field


def _walk(index, seg, instance, chain, active, visited, additional, model, depth=0):
    record = {"anker": base._anchor(seg, instance), "kette": chain,
              "name": pkgleser.name_von(seg, instance), "mengenrolle": "ungeklaert"}

    def unknown(reason):
        record.update(zuordnung="ungeklaert", grund=reason)
        return record

    if instance.klasse not in ("CONTAINER", "npart3"):
        return unknown("Containerkindklasse_unbelegt")
    consumers = [o for o in seg.von_klasse("child_node") if instance.nr in o.verweise()]
    if len(consumers) != 1 or consumers[0].nr != chain["knoten"]["objekt"]:
        return unknown("Containerkind_Besitzer_mehrdeutig")
    if not isinstance(instance.id, str) or sum(o.id == instance.id for o in seg.objekte.values()) != 1:
        return unknown("Containerkind_ID_nicht_eindeutig")
    try:
        target, pose, field = _finite_placement(instance)
    except (pkgleser.Fehler, ValueError, TypeError, KeyError, IndexError) as exc:
        return unknown(str(exc))
    klass = "CONT_STRUCT" if instance.klasse == "CONTAINER" else "part_struct"
    record.update(inhalt_id=target, lage_lokal=pose, platzierungsverweis=field)
    selected = base._unique_root(index, target, klass)
    if selected is None:
        return unknown(klass + "_nicht_eindeutig")
    content_seg, content = selected
    record["inhalt_anker"] = base._anchor(content_seg, content)
    if instance.klasse == "npart3":
        record.update(zuordnung="expliziter_verweis", beziehung="container_teilinhalt")
        if target not in model["teile"]:
            additional.add(target)
        return record
    record.update(beziehung="container_inhaltscontainer", inhalte=[])
    key = _key(content_seg, content)
    if key in active:
        return unknown("Containerzyklus")
    if key in visited:
        record["erste_verwendung"] = copy.deepcopy(visited[key])
        return unknown("Containerinhalt_mehrfach_verwendet")
    if depth >= 64:
        return unknown("Containerinventar_Tiefengrenze")
    visited[key] = base._anchor(seg, instance)
    children, problems = base._children(content_seg, content)
    record["unbekannte_verknuepfungen"] = problems
    for child, child_chain in children:
        record["inhalte"].append(_walk(index, content_seg, child, child_chain,
                                       active | {key}, visited, additional, model, depth + 1))
    record["zuordnung"] = "expliziter_verweis" if not problems and all(
        c["zuordnung"] == "expliziter_verweis" for c in record["inhalte"]) else "ungeklaert"
    return record


def inventar(modell, index):
    """Inspect anchored physical assembly branches already containing CONTAINER.

    Layout/proxy ancestry is excluded explicitly; selection is not inferred from
    names. Existing direct part-container relations are outside this API.
    """
    warnings = []
    index = base._diagnostic_index(index, warnings)
    groups, issues = {}, []
    stack = [(node, "", True) for node in modell["objekte"]]
    while stack:
        node, parent, physical_ancestors = stack.pop()
        name = node.get("name")
        path = parent + "/" + (name if isinstance(name, str) else "?")
        physical = physical_ancestors and node.get("klasse") == "nasmb3"
        if node.get("klasse") == "nasmb3" and any(c.get("klasse") == "CONTAINER" for c in node.get("kinder", [])):
            source = {"vorkommen_anker": copy.deepcopy(node.get("ort")), "pfad": path}
            if not physical:
                issues.append({"grund": "Baugruppencontainer_unbelegter_Vorfahrenkontext", **source})
            else:
                selected, problem = _assembly_owner(index, node)
                if problem:
                    issues.append({"grund": problem, **source})
                else:
                    key = _key(*selected)
                    if key not in groups:
                        groups[key] = {"selected": selected, "occurrences": []}
                    groups[key]["occurrences"].append(source)
        stack.extend((c, path, physical) for c in node.get("kinder", []))
    relations, additional = [], set()
    for key, group in sorted(groups.items()):
        seg, owner = group["selected"]
        children, problems = base._children(seg, owner)
        issues.extend(problems)
        selected = [(child, chain) for child, chain in children if child.klasse == "CONTAINER"]
        record = {"beziehung": "baugruppen_inhaltscontainer", "besitzer_anker": base._anchor(seg, owner),
                  "besitzer_vorkommen": group["occurrences"], "mengenrolle": "ungeklaert", "inhalte": []}
        visited = {}
        for child, chain in selected:
            record["inhalte"].append(_walk(index, seg, child, chain, set(), visited, additional, modell))
        record["zuordnung"] = "expliziter_verweis" if selected and not problems and all(
            item["zuordnung"] == "expliziter_verweis" for item in record["inhalte"]) else "ungeklaert"
        if not selected:
            record["grund"] = "Kein_streng_gebundener_Container"
        relations.append(record)
    return {"schema": 1, "beziehungen": relations, "zusaetzliche_teilinhalte": sorted(additional),
            "unbekannte_verknuepfungen": issues, "quell_warnungen": warnings, "mengenrolle": "ungeklaert",
            "beleg": "Creo 20.3, 13.11.2026: 5 Baugruppen mit je zwei Containerebenen, 20 native Objekte und 15 exakt gleiche lokale Lagen",
            "grenzen": "Keine Mengen-, Massen-, BOM- oder Stockrolle; weitere Kindklassen bleiben ungeklaert"}
