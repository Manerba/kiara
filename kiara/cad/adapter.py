"""PKG contents and occurrences with explicit provenance and object boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from kiara.cad import PARSER, READER_REVISION, SCHEMA_VERSION, revision


class CadReadError(OSError):
    pass


def _id(kind, value):
    encoded = json.dumps([kind, value], ensure_ascii=True, sort_keys=True).encode()
    return kind + "_" + hashlib.sha256(encoded).hexdigest()[:32]


def _source(value):
    return dict(zip(("entry", "segment", "object"), value, strict=True)) if value else None


def _property(name, value, *, unit=None, source=None, status=None, namespace="cad"):
    return {
        "namespace": namespace,
        "name": name,
        "value": value,
        "unit": unit,
        "status": status or ("not_stored" if value is None else "stored"),
        "source": source,
    }


def _attributes(attributes, source):
    # Attribute names themselves may contain private text: expose them as values.
    return [
        _property(str(key), value, source=source, namespace=str(namespace))
        for namespace, data in attributes.items()
        for key, value in data.get("werte", [])
    ]


def _labels(record):
    """Beschriftungen eines Inhalts (Teil, Baugruppe oder Containerinhalt)."""
    labels = []
    for item in record.get("labels", []):
        label = item["data"]
        if not label.get("form_erkannt", True):
            # Nur im Modus open_fields erreichbar: Rohfelder statt Deutung.
            labels.append(
                {
                    "text": [],
                    "source": item["source"],
                    "type": "unread",
                    "status": "form_unbekannt",
                    "reason": label.get("grund"),
                    "raw_fields": label.get("label_roh", []),
                }
            )
            continue
        labels.append(
            {
                "text": label["belegt"]["text_liste"],
                "source": item["source"],
                "type": label["annotationstyp"],
                "owner": label["kette"]["inhalt"],
                "fields": label["felder"]["text_liste"],
            }
        )
        if "individuell_ausgeblendet" in label["belegt"]:
            # Feature-Text einzeln ausgeblendet (cad-pkg-reverse a9f78e6); globale Sichtbarkeit offen.
            labels[-1]["hidden"] = label["belegt"]["individuell_ausgeblendet"]
            labels[-1]["hidden_field"] = label["felder"].get("individuell_ausgeblendet")
        if "referenzgraph" in label:
            labels[-1]["reference_graph"] = label["referenzgraph"]
            labels[-1]["reference_fields"] = {
                key: label["felder"][key]
                for key in ("referenzanzahl", "referenzen")
                if key in label["felder"]
            }
    return labels


# Befunde zu einzelnen Feldformen; das Modell ist ohne sie strukturell lesbar.
# Nur diese dürfen im ausdrücklich angeforderten Modus open_fields veröffentlicht
# werden (Nutzerentscheid 07.11.2026, nur für gezielte Läufe).
OPEN_FIELD_CODES = {
    "annotation_form_unknown",
    "material_form_unknown",
    "library_form_unknown",
    "db_flags_unknown",
    "product_context_unknown",
    "unmapped_content_metadata",
}


def normalize(raw, filename, *, open_fields=False):
    open_findings = []
    if raw.get("issues") and open_fields and all(
        issue["code"] in OPEN_FIELD_CODES for issue in raw["issues"]
    ):
        open_findings = [
            {k: issue[k] for k in ("code", "source", "native_id") if k in issue}
            for issue in raw["issues"]
        ]
    elif raw.get("issues"):
        # Preserve a compact, non-speculative diagnosis in the job log.
        findings = [
            {k: issue[k] for k in ("code", "source") if k in issue} for issue in raw["issues"][:32]
        ]
        raise CadReadError(
            "PKG enthält ungeklärte Ingest-Felder: " + json.dumps(findings, ensure_ascii=True)
        )
    model, records = raw["model"], raw["records"]
    if not model.get("objekte"):
        raise CadReadError("PKG enthält keine lesbare Modellwurzel")
    contents, occurrences, parts = {}, [], {}
    for native_id, part in model["teile"].items():
        record = records.get(native_id, {})
        source = next(iter(record.get("sources", [])), None)
        cid = _id("part", [native_id, source])
        parts[native_id] = cid
        props = _attributes(part.get("attribute", {}), source)
        for key, unit in (
            ("dichte_g_cm3", "g/cm³"),
            ("quader", "mm"),
            ("farbe", None),
            ("koerperart", None),
        ):
            props.append(
                _property(
                    key,
                    part.get(key),
                    unit=unit,
                    source=source,
                    status="not_read"
                    if key not in part
                    # Körperart: None heißt Verweisform unbelegt, nicht „leer gespeichert“.
                    else "unknown" if key == "koerperart" and part[key] is None
                    # Quader ±inf (H2): Rohmarker bleibt erhalten, kein Messwert.
                    else "not_finite" if _non_finite(part[key]) else None,
                )
            )
        for material in record.get("materials", []):
            data, loc = material["data"], material["source"]
            props.append(
                _property(
                    "blechdicke_mm",
                    data["blechdicke_mm"],
                    unit="mm",
                    source={**loc, **data["felder"]["zahl"]},
                )
            )
            key = data["materialschluessel"]
            for name in (
                "materialbezeichnung",
                "shopname",
                "tabellenname",
                "dickenwert_im_schluessel",
                "dickeneinheit_im_schluessel",
                "normierte_werkstoffguete",
                "katalogabgleich",
            ):
                props.append(
                    _property(
                        name,
                        key.get(name),
                        source=loc,
                        namespace="material_catalog",
                        status="unknown" if key.get(name) is None else "stored",
                    )
                )
            if key.get("status") == "legacy_oder_opak":
                # Werkzeugkennung ohne Tabellenform: nur als Rohschlüssel, ohne Deutung.
                props.append(
                    _property(
                        "werkzeugschluessel_roh",
                        data.get("materialwerkzeugschluessel"),
                        source={**loc, **data["felder"]["text"]},
                        namespace="material_catalog",
                    )
                )
        for lib in record.get("libraries", []):
            for key in ("kennung", "symbolische_version"):
                props.append(
                    _property(
                        key,
                        lib["data"][key],
                        namespace="library",
                        source={**lib["source"], **lib["data"]["felder"][key]},
                    )
                )
        product = part.get("step_produkt", {})
        for key, value in product.items():
            if key not in {"weitere", "feld_11", "kontextdaten"}:
                props.append(_property(key, value, namespace="step_product", source=source))
        for key in (
            "definitionskontext_name",
            "definitionskontext_anwendungsprotokoll_jahr",
            "anwendungsprotokoll_jahr",
        ):
            field = product.get("kontextdaten", {}).get(key)
            if field:
                props.append(
                    _property(
                        key,
                        field["wert"],
                        namespace="step_context",
                        source=source,
                        status={
                            "nicht_gespeichert": "not_stored",
                            "leer": "empty",
                            "vorhanden": "stored",
                        }[field["status"]],
                    )
                )
        labels = _labels(record)
        contents[cid] = {
            "object_id": cid,
            "kind": "part",
            "native_id": native_id,
            "names": [],
            "source": source,
            "has_body": bool(part.get("topologie", {}).get("body")),
            "properties": props,
            "labels": labels,
            "data": {
                "body_placement": part.get("koerperlage"),
                "topology": part.get("topologie"),
                "materials": record.get("materials", []),
                "libraries": record.get("libraries", []),
                "db_flags": record.get("db_flags", []),
            },
        }

    stack = [(n, None, [], [], True, i) for i, n in reversed(list(enumerate(model["objekte"])))]
    while stack:
        node, parent, path, ancestry, physical_parent, position = stack.pop()
        source = _source(node.get("ort"))
        name = node.get("name") or "(ohne Namen)"
        path = [*path, name]
        ancestry = [*ancestry, [position, node.get("ort")]]
        oid = _id("occ", ancestry)
        kind = {"teil": "part", "baugruppe": "assembly"}.get(node["typ"], "auxiliary")
        physical = physical_parent and kind in {"part", "assembly"}
        if kind == "part":
            cid = parts.get(node.get("teil_id"))
            if cid is None:
                raise CadReadError("PKG enthält eine nicht aufgelöste Teilreferenz")
            physical = physical and contents[cid]["has_body"]
        else:
            ref = node.get("struktur") or node.get("ort")
            cid = _id(kind, ref)
            if cid not in contents:
                # Baugruppentexte und -flags hängen an der Struktur-ID (cad-pkg-reverse 9ebaaa3).
                record = records.get(node.get("struktur_id"), {}) if kind == "assembly" else {}
                contents[cid] = {
                    "object_id": cid,
                    "kind": kind,
                    "native_id": node.get("id"),
                    "names": [],
                    "source": _source(ref),
                    "has_body": False,
                    "properties": _attributes(node.get("attribute", {}), _source(ref)),
                    "labels": _labels(record),
                    "data": {"node_type": node["typ"], "structure_id": node.get("struktur_id"),
                             "db_flags": record.get("db_flags", [])},
                }
        if name not in contents[cid]["names"]:
            contents[cid]["names"].append(name)
        occurrence_props = _attributes(node.get("exemplarattribute", {}), source)
        for key in ("dichte_g_cm3", "farbe"):
            if key in node:
                occurrence_props.append(
                    _property(
                        key,
                        node[key],
                        source=source,
                        unit="g/cm³" if key == "dichte_g_cm3" else None,
                    )
                )
        occurrences.append(
            {
                "occurrence_id": oid,
                "object_id": cid,
                "parent_id": parent,
                "position": position,
                "name": name,
                "path": path,
                "kind": kind,
                "physical": physical,
                "source": source,
                "properties": occurrence_props,
                "placement": node.get("lage"),
                "world_placement": node.get("lage_global"),
            }
        )
        stack.extend(
            (child, oid, path, ancestry, physical, i)
            for i, child in reversed(list(enumerate(node.get("kinder", []))))
        )
    inventory = raw.get("container_inventory") or {}
    _container_contents(inventory, model, records, contents)
    complete = not model["warnungen"]
    return {
        "schema_version": SCHEMA_VERSION,
        "reader_revision": READER_REVISION,
        "adapter_revision": revision(),
        "source_hash": raw["source_hash"],
        "filename": filename,
        "structure_complete": complete,
        "units": model["einheiten"],
        "placement_convention": model["lage_konvention"],
        "warnings": model["warnungen"],
        "verification_complete": not open_findings,
        "open_findings": open_findings,
        "content_containers": inventory,
        "contents": list(contents.values()),
        "occurrences": occurrences,
    }


def _non_finite(value):
    if isinstance(value, dict):
        return "nicht_endliche_zahl" in value or any(_non_finite(v) for v in value.values())
    if isinstance(value, list):
        return any(_non_finite(v) for v in value)
    return False


def _container_contents(inventory, model, records, contents):
    """Explizit verknüpfte Containerinhalte als eigene Inhaltsart ohne Vorkommen.

    Mengenrolle ungeklärt (cad-pkg-reverse 9ebaaa3, F25): keine Vorkommen, daher
    keine Stückzahl, keine Stückliste und keine Masse. Metadaten bleiben abfragbar.
    """
    items = [(relation.get("eltern_teil_id"), relation.get("container_anker"), item)
             for relation in inventory.get("beziehungen", []) for item in relation.get("inhalte", [])]
    # Verschachtelte Baugruppencontainer (cad-pkg-reverse a9f78e6): rekursiv, Teile tragen inhalt_id.
    stack = [(relation.get("besitzer_anker"), item)
             for relation in (inventory.get("baugruppencontainer") or {}).get("beziehungen", [])
             for item in relation.get("inhalte", [])]
    while stack:
        owner, item = stack.pop()
        if item.get("beziehung") == "container_teilinhalt":
            items.append((None, owner, {**item, "teil_id": item.get("inhalt_id")}))
        else:
            stack.extend((item.get("inhalt_anker") or owner, child) for child in item.get("inhalte", []))
    for parent_part, container_anchor, item in items:
        native_id = item.get("teil_id")
        if item.get("zuordnung") != "expliziter_verweis" or native_id in model["teile"]:
            continue
        record = records.get(native_id, {})
        cid = _id("container_part", [native_id, item.get("inhalt_anker")])
        entry = contents.setdefault(cid, {
            "object_id": cid,
            "kind": "container_part",
            "native_id": native_id,
            "names": [],
            "source": item.get("inhalt_anker"),
            "has_body": False,
            "properties": [],
            "labels": _labels(record),
            "data": {"materials": record.get("materials", []),
                     "libraries": record.get("libraries", []),
                     "db_flags": record.get("db_flags", []),
                     "containers": [], "quantity_role": "ungeklaert"},
        })
        name = item.get("name")
        if isinstance(name, str) and name not in entry["names"]:
            entry["names"].append(name)
        entry["data"]["containers"].append({
            "parent_part_id": parent_part,
            "container_anchor": container_anchor,
            "anchor": item.get("anker"),
            "local_placement": item.get("lage_lokal"),
        })


def load_pkg(path, *, timeout=600, open_fields=False):
    """Bound reader memory to a child process and preserve the original input.

    600 s: Containerinventar und Baugruppentexte (cad-pkg-reverse 9ebaaa3/a9f78e6) lesen
    Teilsegmente erneut, gemessen rund +55 % Lesezeit gegenüber 2b3e99e.
    """
    path = Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix="kiara-cad-") as directory:
        output = Path(directory) / "reader.json"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("reader_worker.py")),
                    str(path),
                    str(output),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CadReadError("PKG-Leser hat das Zeitlimit überschritten") from exc
        if result.returncode:
            # Parser stderr can contain model texts. Keep the public error generic.
            raise CadReadError("PKG-Leser konnte die Datei nicht vollständig auswerten")
        return normalize(
            json.loads(output.read_text(encoding="utf-8")), path.name, open_fields=open_fields
        )


def _property_text(properties):
    return [
        f"{p['namespace']}.{p['name']}: {p['value']}" + (f" {p['unit']}" if p.get("unit") else "")
        for p in properties
        if p["status"] == "stored" and isinstance(p["value"], (str, int, float, bool))
    ]


def document(snapshot, metadata):
    """One continuous context text with separately chunkable object sections."""
    sections = []
    physical = [o for o in snapshot["occurrences"] if o["physical"]]
    physical_ids = {o["object_id"] for o in physical}
    summary = [
        f"CAD-Datei: {snapshot['filename']}",
        f"Format: PKG; SHA-256: {snapshot['source_hash']}",
        f"Teilvorkommen: {sum(o['kind'] == 'part' for o in physical)}",
        f"Lesestatus: {'Struktur vollständig gelesen' if snapshot['structure_complete'] else 'Struktur unvollständig'}",
        *(
            [f"Prüfstatus: nicht vollständig geprüft, {len(snapshot['open_findings'])} ungeklärte Feldformen"]
            if not snapshot.get("verification_complete", True)
            else []
        ),
        "Stückzahlen und Verwendungen über die strukturierte CAD-Abfrage ermitteln.",
        "Gespeicherte Dichte ist kein geprüfter Werkstoffwert. Gespeicherte Quader können Zuschläge enthalten.",
    ]
    sections.append({"object_id": "file", "kind": "file", "text": "\n".join(summary)})
    for content in snapshot["contents"]:
        if content["object_id"] not in physical_ids:
            continue
        lines = [
            f"{'Bauteil' if content['kind'] == 'part' else 'Baugruppe'}: "
            + "; ".join(content["names"])
        ]
        lines.extend(_property_text(content["properties"]))
        for label in content["labels"]:
            lines.extend("Beschriftung: " + text for text in label["text"] if text)
        sections.append(
            {"object_id": content["object_id"], "kind": content["kind"], "text": "\n".join(lines)}
        )
    # Only occurrences with extra information receive their own search section.
    # Bare placements and repeated component names do not create duplicate chunks.
    for occurrence in physical:
        text_properties = [p for p in occurrence["properties"] if p["namespace"] != "cad"]
        if text_properties:
            sections.append(
                {
                    "object_id": occurrence["object_id"],
                    "occurrence_id": occurrence["occurrence_id"],
                    "kind": "occurrence",
                    "text": "Vorkommen: "
                    + " / ".join(occurrence["path"])
                    + "\n"
                    + "\n".join(_property_text(text_properties)),
                }
            )
    position, texts = 0, []
    for section in sections:
        section["start"] = position
        texts.append(section["text"])
        position += len(section["text"])
        section["end"] = position
        position += 2
    meta = {
        **metadata,
        "parser": PARSER,
        "cad_schema_version": SCHEMA_VERSION,
        "cad_revision": revision(),
        "cad_source_hash": snapshot["source_hash"],
        "cad_structure_complete": snapshot["structure_complete"],
    }
    return {
        "text": "\n\n".join(texts),
        "metadata": meta,
        "cad_snapshot": snapshot,
        "cad_sections": sections,
    }


def chunks(doc, chunk_size):
    from kiara.ingestion.chunker import chunk_document

    result = []
    for section in doc["cad_sections"]:
        meta = {
            **doc["metadata"],
            "cad_object_id": section["object_id"],
            "cad_kind": section["kind"],
        }
        if section.get("occurrence_id"):
            meta["cad_occurrence_id"] = section["occurrence_id"]
        for chunk in chunk_document(section["text"], meta, chunk_size=chunk_size, chunk_overlap=0):
            # The normal splitter supplies spans relative to the object text.
            for key in ("char_start", "char_end"):
                if isinstance(chunk.get(key), int):
                    chunk[key] += section["start"]
            result.append(chunk)
    for index, chunk in enumerate(result):
        chunk["metadata"].update(
            chunk_index=index,
            chunk_total=len(result),
            chunk_strategy="cad_objects",
            chunk_size_used=chunk_size,
            chunk_overlap_used=0,
        )
    return result
