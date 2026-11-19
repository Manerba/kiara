"""Isolated, dependency-free reader process; never imports Kiara or starts Creo."""

import gc
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))
import pkgleser
from analyse import beschriftungen, inhaltscontainer, metadaten


class _SegmentIndex:
    """Lazy segments across *all* streams, with complete metadata inspection.

    The vendor's sparsam=True reader retains only the last object stream.
    Combine its segment directories here so cross-stream references keep the
    same identity/order as the eager reader. Geometry-bearing part segments are
    released after use; only the vendor's small structural segments are cached.
    """

    def __init__(self):
        self.directories = []
        self.by_id = {}
        self.records = {}
        self.copies = {}
        self.ends = {}
        # Reiner Segmentindex ohne Datensatzberechnung für dateiweite Decoder
        # (Baugruppenbeschriftungen, Containerinventar). Über self.get würden
        # sich gegenseitig referenzierende Inhalte rekursiv ausgewertet.
        self.raw = pkgleser.Verzeichnisse()

    def add(self, data, name, warnings):
        directory = pkgleser.Segmentverzeichnis(data, name, warnings)
        self.directories.append(directory)
        self.raw.liste.append(directory)
        for start, end, _cls, _sid in directory.koepfe:
            self.ends[(id(directory), start)] = end
        for identity in directory.nach_id:
            self.by_id.setdefault(identity, []).append(directory)
        return directory

    def get(self, identity, default=None):
        if identity not in self.by_id:
            return [] if default is None else default
        pairs = [(d, seg) for d in self.by_id[identity] for seg in d.get(identity)]
        segments = [seg for _, seg in pairs]
        if identity not in self.records:
            # Mehrstrompakete speichern Segmente teils als byte-gleiche Kopie in
            # mehreren Strömen; diese nur einmal auswerten. Abweichende Segmente
            # derselben Teil-ID bleiben ein Befund (ambiguous_part_identity).
            distinct = self._distinct(pairs)
            if len(distinct) < len(segments):
                self.copies[identity] = len(segments) - len(distinct)
            self.records[identity] = _record(distinct, self.raw)
        return segments

    def _distinct(self, pairs):
        seen, distinct = set(), []
        for directory, seg in pairs:
            end = self.ends[(id(directory), seg.start)]
            digest = hashlib.sha256(memoryview(directory.b)[seg.start : end]).digest()
            if digest not in seen:
                seen.add(digest)
                distinct.append(seg)
        return distinct

    def finish(self, model, mapped=()):
        """mapped: weitere belegte Besitzer außer Teilen (Baugruppenstrukturen,
        explizit verknüpfte Containerinhalte), deren Metadaten zugeordnet sind."""
        records, issues = {}, []
        owners = set(model["teile"]) | set(mapped)
        # Inspect also unused segments: metadata gaps must never be hidden by
        # lazy traversal, and diagnostics retain the original file order.
        for identity in self.by_id:
            if identity not in self.records:
                self.get(identity)
            record, findings = self.records[identity]
            records[identity] = record
            issues.extend(findings)
            if len(record["sources"]) > 1:
                issues.append({"code": "ambiguous_part_identity", "native_id": identity})
            if (record["materials"] or record["labels"]) and identity not in owners:
                issues.append({"code": "unmapped_content_metadata", "native_id": identity})
        return records, issues


def _read_model(reader, path):
    """Use the vendor's tree/field decoders with a combined lazy segment index."""
    model = {
        "datei": Path(path).name,
        "einheiten": {"laenge": "mm", "dichte": "g/cm³"},
        "lage_konvention": "p_global = p · M + t",
        "eintraege": [],
        "objekte": [],
    }
    entries, findings = pkgleser.oeffnen_mit_diagnose(path)
    reader.warnungen.extend(findings)
    index = reader.index = _SegmentIndex()
    for name, data in entries:
        if data[:2] == b"Rz":
            model.update({k: v for k, v in pkgleser.kopf_lesen(data).items() if k not in model})
            index.add(data, name, reader.warnungen)
            kind = "objektstrom"
        else:
            kind = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "unbekannt"
        model["eintraege"].append({"name": name, "art": kind, "bytes": len(data)})
    del entries
    # 0: kein Creo-Paket (z. B. gewöhnliches ZIP-Archiv), wie der Vendor-Leser.
    model["objektstroeme"] = len(index.directories)
    for directory in index.directories:
        if not directory.koepfe:
            continue
        segment = directory.oberstes()
        if segment is None:
            continue
        root = segment.wurzel()
        if root is None:
            reader.warnungen.append(
                f"Eintrag {segment.eintrag}: kein Wurzelobjekt {segment.klasse}"
            )
            continue
        node = reader.knoten(segment, root)
        node["eintrag"] = segment.eintrag
        model["objekte"].append(node)
    model["teile"] = reader.teile
    model["warnungen"] = reader.warnungen
    return model


# Werkzeugkennung ohne Tabellenform (z. B. "Standard"): bekannter Zustand, der
# Rohschlüssel bleibt erhalten (cad-pkg-reverse INGEST_FELDER.md, 5./6.11.2026).
_KNOWN_MATERIAL_KEY_STATES = {"legacy_oder_opak"}


def _record(segments, index=None):
    """Decode metadata while a segment is resident; retain no geometry tokens."""
    issues = []
    record = {"sources": [], "materials": [], "libraries": [], "labels": [], "db_flags": []}
    for seg in segments:
        for obj in seg.objekte.values():
            source = {"entry": seg.eintrag, "segment": seg.start, "object": obj.nr}
            if obj.klasse == "part_struct":
                record["sources"].append(source)
                lib = metadaten.bibliotheksmarker(obj)
                if lib["vorhanden"] is True:
                    record["libraries"].append({"source": source, "data": lib})
                elif not lib["eigener_inhalt_geprueft"]:
                    issues.append({"code": "library_form_unknown", "source": source})
            elif obj.klasse == "mbu_material":
                material = metadaten.material(seg, obj)
                key = material.get("materialschluessel", {})
                if not key.get("schema_erkannt") and key.get("status") not in _KNOWN_MATERIAL_KEY_STATES:
                    issues.append({"code": "material_form_unknown", "source": source})
                record["materials"].append({"source": source, "data": material})
            elif obj.klasse == "DB_FLAGS":
                flags = metadaten.db_flags(seg, obj)
                if (
                    not flags.get("schema_erkannt")
                    or flags.get("unbekannte_bits")
                    or "besitzersemantik" in flags
                ):
                    issues.append({"code": "db_flags_unknown", "source": source})
                record["db_flags"].append({"source": source, "data": flags})
            elif obj.klasse == "A3D_LBL_TAG":
                # Baugruppentexte brauchen den dateiweiten Index (cad-pkg-reverse 9ebaaa3).
                label = beschriftungen.decode(seg, obj, index=index)
                if not label.get("form_erkannt"):
                    issues.append({"code": "annotation_form_unknown", "source": source})
                record["labels"].append({"source": source, "data": label})
            elif obj.klasse == "PD_ATTR_SP_STRUCT0":
                if not pkgleser.step_kontextdaten(obj)["schema_erkannt"]:
                    issues.append({"code": "product_context_unknown", "source": source})
    return record, issues


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(path):
    before = _digest(path)
    reader = pkgleser.Leser(mit_punkten=False)
    model = _read_model(reader, path)
    # Der fertige Baum bleibt für die Ausgabe referenziert; ohne Einfrieren
    # durchläuft die Speicherbereinigung ihn bei jeder Metadaten-/Beschriftungs-
    # analyse erneut (große Baugruppen sonst > 300 s, cad-pkg-reverse 6.11.2026).
    gc.freeze()
    structures = _assembly_structures(reader.index, model)
    inventory = inhaltscontainer.inventar(model, reader.index.raw)
    # Zugeordnet sind Metadaten auch an Baugruppenstrukturen und an explizit über
    # Teil/Container/Knoten verknüpften Containerinhalten (cad-pkg-reverse 9ebaaa3, F25).
    nested = inventory.get("baugruppencontainer") or {}
    mapped = (structures | set(inventory["zusaetzliche_teilinhalte"])
              | set(nested.get("zusaetzliche_teilinhalte", [])))
    records, issues = reader.index.finish(model, mapped)
    if _digest(path) != before:
        raise ValueError("CAD source changed during extraction")
    return {
        "source_hash": before,
        "model": model,
        "records": records,
        "issues": issues,
        "identical_copies": len(reader.index.copies),
        "container_inventory": inventory,
    }


def _assembly_structures(index, model):
    """Segment-ID der Struktur je Baugruppenknoten als struktur_id eintragen.

    Der Vendor-Knoten nennt nur den Ort (Eintrag, Segmentanfang, Objekt). Mehrdeutige
    Orte (gleicher Eintragsname mit verschiedenen IDs) bleiben ohne Zuordnung.
    """
    by_place = {}
    for directory in index.directories:
        for start, _end, _cls, sid in directory.koepfe:
            by_place.setdefault((directory.eintrag, start), set()).add(sid)
    found = set()
    stack = list(model["objekte"])
    while stack:
        node = stack.pop()
        place = node.get("struktur")
        if node.get("typ") == "baugruppe" and place:
            sids = by_place.get((place[0], place[1]), set())
            if len(sids) == 1:
                node["struktur_id"] = next(iter(sids))
                found.add(node["struktur_id"])
        stack.extend(node.get("kinder", []))
    return found


def _finite(value):
    """Nichtendliche Zahlen als Rohmarker erhalten (strenges JSON, kein Ersatzwert)."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nicht_endliche_zahl": repr(value)}
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


if __name__ == "__main__":
    result = extract(sys.argv[1])
    try:
        text = json.dumps(result, ensure_ascii=True, allow_nan=False)
    except ValueError:
        # Nur bei nichtendlichen Zahlen (z. B. Quader ±inf, H2) den teuren Umbau.
        text = json.dumps(_finite(result), ensure_ascii=True, allow_nan=False)
    Path(sys.argv[2]).write_text(text, encoding="utf-8")
