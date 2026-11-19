"""Independent, atomic CAD snapshots with per-datasource ownership."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal

from kiara.shared.db import db_query


def migrate(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cad_models (
            id text PRIMARY KEY, collection_name text NOT NULL, source text NOT NULL,
            embedding_model text NOT NULL, source_hash text NOT NULL, processing_version text NOT NULL,
            reader_revision text NOT NULL, structure_complete boolean NOT NULL, metadata jsonb NOT NULL,
            published_at timestamptz NOT NULL DEFAULT now(), UNIQUE(collection_name, source)
        );
        ALTER TABLE cad_models DROP CONSTRAINT IF EXISTS cad_models_collection_name_source_fkey;
        ALTER TABLE datasources ADD COLUMN IF NOT EXISTS ds_cad_vector_enabled boolean NOT NULL DEFAULT true;
        ALTER TABLE datasources ADD COLUMN IF NOT EXISTS ds_cad_relational_enabled boolean NOT NULL DEFAULT true;
        CREATE TABLE IF NOT EXISTS cad_model_owners (
            datasource_id integer NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
            source text NOT NULL, model_id text NOT NULL REFERENCES cad_models(id) ON DELETE CASCADE,
            PRIMARY KEY(datasource_id, source)
        );
        CREATE INDEX IF NOT EXISTS cad_model_owner_idx ON cad_model_owners(model_id);
        INSERT INTO cad_model_owners(datasource_id,source,model_id)
        SELECT DISTINCT ON (f.datasource_id,m.source) f.datasource_id,m.source,m.id
        FROM cad_models m JOIN file_hashes f ON f.file_path=m.source AND f.embedding_model=m.embedding_model
        JOIN document_publications p ON p.collection_name=m.collection_name AND p.source=m.source
        WHERE p.completed_at IS NOT NULL AND p.versions->>'cad-pkg'=m.processing_version
        ORDER BY f.datasource_id,m.source,m.published_at DESC,m.id
        ON CONFLICT(datasource_id,source) DO NOTHING;
        CREATE TABLE IF NOT EXISTS cad_contents (
            model_id text NOT NULL REFERENCES cad_models(id) ON DELETE CASCADE,
            object_id text NOT NULL, kind text NOT NULL, names text[] NOT NULL,
            properties jsonb NOT NULL, record jsonb NOT NULL, PRIMARY KEY(model_id, object_id)
        );
        CREATE TABLE IF NOT EXISTS cad_occurrences (
            model_id text NOT NULL, occurrence_id text NOT NULL, object_id text NOT NULL,
            parent_id text, physical boolean NOT NULL, path text[] NOT NULL, position integer NOT NULL,
            record jsonb NOT NULL, PRIMARY KEY(model_id, occurrence_id),
            FOREIGN KEY(model_id, object_id) REFERENCES cad_contents(model_id, object_id) ON DELETE CASCADE,
            FOREIGN KEY(model_id, parent_id) REFERENCES cad_occurrences(model_id, occurrence_id) DEFERRABLE INITIALLY DEFERRED
        );
        CREATE INDEX IF NOT EXISTS cad_content_properties_idx ON cad_contents USING gin(properties jsonb_path_ops);
        CREATE INDEX IF NOT EXISTS cad_occurrence_object_idx ON cad_occurrences(model_id, object_id);
        CREATE INDEX IF NOT EXISTS cad_occurrence_parent_idx ON cad_occurrences(model_id, parent_id);
    """)


def publish(conn, collection, source, model, processing_version, snapshot, *, datasource_id=None):
    """Publish one SQL target atomically; vector publication is independent.

    Legacy callers derive owners from completed file hashes. New imports pass an
    explicit owner and use a per-datasource namespace, preserving other owners.
    """
    if datasource_id is not None:
        collection, model = f"cad:{datasource_id}", ""
    owners = (
        [datasource_id]
        if datasource_id is not None
        else [
            r["datasource_id"]
            for r in conn.execute(
                "SELECT DISTINCT datasource_id FROM file_hashes WHERE file_path=%s AND embedding_model=%s",
                (source, model),
            ).fetchall()
        ]
    )
    model_id = hashlib.sha256(json.dumps([collection, source]).encode()).hexdigest()
    conn.execute("DELETE FROM cad_occurrences WHERE model_id = %s", (model_id,))
    conn.execute("DELETE FROM cad_contents WHERE model_id = %s", (model_id,))
    metadata = {k: v for k, v in snapshot.items() if k not in {"contents", "occurrences"}}
    conn.execute(
        "INSERT INTO cad_models(id,collection_name,source,embedding_model,source_hash,processing_version,reader_revision,structure_complete,metadata) "
        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) "
        "ON CONFLICT(id) DO UPDATE SET source_hash=excluded.source_hash, "
        "processing_version=excluded.processing_version,reader_revision=excluded.reader_revision, "
        "structure_complete=excluded.structure_complete,metadata=excluded.metadata,published_at=now()",
        (
            model_id,
            collection,
            source,
            model,
            snapshot["source_hash"],
            processing_version,
            snapshot["reader_revision"],
            snapshot["structure_complete"],
            json.dumps(metadata),
        ),
    )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO cad_contents(model_id,object_id,kind,names,properties,record) VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb)",
            [
                (
                    model_id,
                    c["object_id"],
                    c["kind"],
                    c["names"],
                    json.dumps(c["properties"]),
                    json.dumps(c),
                )
                for c in snapshot["contents"]
            ],
        )
        cur.executemany(
            "INSERT INTO cad_occurrences(model_id,occurrence_id,object_id,parent_id,physical,path,position,record) VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
            [
                (
                    model_id,
                    o["occurrence_id"],
                    o["object_id"],
                    o["parent_id"],
                    o["physical"],
                    o["path"],
                    o["position"],
                    json.dumps(o),
                )
                for o in snapshot["occurrences"]
            ],
        )

    for owner in owners:
        conn.execute(
            "INSERT INTO cad_model_owners(datasource_id,source,model_id) VALUES(%s,%s,%s) "
            "ON CONFLICT(datasource_id,source) DO UPDATE SET model_id=excluded.model_id",
            (owner, source, model_id),
        )
    # Only old snapshots of this source without any remaining owner can disappear.
    conn.execute(
        "DELETE FROM cad_models m WHERE source=%s AND NOT EXISTS "
        "(SELECT 1 FROM cad_model_owners o WHERE o.model_id=m.id)",
        (source,),
    )


def cleanup(conn, collection, sources):
    conn.execute(
        "DELETE FROM cad_models WHERE collection_name = %s AND source = ANY(%s)",
        (collection, sources),
    )


_VISIBLE = """
    SELECT m.*, owners.datasource_ids FROM cad_models m
    JOIN LATERAL (
        SELECT array_agg(DISTINCT o.datasource_id ORDER BY o.datasource_id) AS datasource_ids
        FROM cad_model_owners o JOIN datasources ds ON ds.id=o.datasource_id
        WHERE o.model_id=m.id AND ds.is_active=true AND o.datasource_id=ANY(%s)
    ) owners ON owners.datasource_ids IS NOT NULL
    WHERE true
"""


def validate(params):
    allowed = {
        "operation",
        "query",
        "model_id",
        "object_id",
        "occurrence_id",
        "source_hash",
        "recursive",
        "property_name",
        "property_namespace",
        "property_value",
        "numeric_field",
        "numeric_unit",
        "minimum",
        "maximum",
        "minimum_exclusive",
        "maximum_exclusive",
        "limit",
        "offset",
        "datasource_id",
    }
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError("Unbekannte CAD-Abfrageparameter")
    result = dict(params)
    result.setdefault("operation", "search")
    if result["operation"] not in {"search", "object", "bom", "where_used"}:
        raise ValueError("Unbekannte CAD-Abfrage")
    for name in ("query", "property_name", "property_namespace", "property_value"):
        if name in result and (not isinstance(result[name], str) or len(result[name]) > 2000):
            raise ValueError("Ungültiger Textfilter")
    for name in ("model_id", "source_hash", "object_id", "occurrence_id"):
        if name in result and (
            not isinstance(result[name], str) or not result[name] or len(result[name]) > 100
        ):
            raise ValueError("Ungültige CAD-Kennung")
    for name, default, maximum in (("limit", 50, 200), ("offset", 0, 1000000)):
        value = result.get(name, default)
        if type(value) is not int or value < (1 if name == "limit" else 0) or value > maximum:
            raise ValueError("Ungültige Ergebnisbegrenzung")
        result[name] = value
    if "recursive" in result and type(result["recursive"]) is not bool:
        raise ValueError("recursive muss boolesch sein")
    if result.get("numeric_field") not in (None, "dichte_g_cm3", "blechdicke_mm"):
        raise ValueError("Nicht unterstütztes Zahlenfeld")
    for key in ("minimum", "maximum"):
        if key in result and (
            type(result[key]) not in (int, float)
            or not math.isfinite(result[key])
            or not result.get("numeric_field")
        ):
            raise ValueError("Ungültiger Zahlenfilter")
        exclusive = key + "_exclusive"
        if exclusive in result and (
            type(result[exclusive]) is not bool or (result[exclusive] and key not in result)
        ):
            raise ValueError("Exklusive Grenze benötigt einen Zahlenwert")
    if "numeric_unit" in result and not result.get("numeric_field"):
        raise ValueError("Einheit benötigt ein Zahlenfeld")
    if result.get("numeric_field"):
        density = result["numeric_field"] == "dichte_g_cm3"
        factors = (
            {"g/cm³": "1", "g/cm3": "1", "kg/m³": ".001", "kg/m3": ".001"}
            if density
            else {"mm": "1", "cm": "10", "m": "1000", "in": "25.4"}
        )
        canonical = "g/cm³" if density else "mm"
        unit = result.get("numeric_unit", canonical)
        if not isinstance(unit, str) or unit not in factors:
            raise ValueError("Nicht unterstützte CAD-Einheit")
        for key in ("minimum", "maximum"):
            if key in result:
                result[key] = float(Decimal(str(result[key])) * Decimal(factors[unit]))
                if not math.isfinite(result[key]):
                    raise ValueError("Zahlenfilter liegt außerhalb des Wertebereichs")
        result["numeric_unit"] = canonical
    if result.get("minimum", -math.inf) > result.get("maximum", math.inf):
        raise ValueError("Minimum liegt über Maximum")
    if ("property_name" in result) != ("property_value" in result):
        raise ValueError("Eigenschaftsname und exakter Wert müssen gemeinsam angegeben werden")
    if "property_namespace" in result and "property_name" not in result:
        raise ValueError("Attributgruppe benötigt Eigenschaftsname und Wert")
    if result["operation"] != "search" and any(
        k in result for k in ("query", "property_name", "numeric_field")
    ):
        raise ValueError("Text- und Eigenschaftsfilter sind nur bei search zulässig")
    if result["operation"] == "bom" and "object_id" in result:
        raise ValueError("Für eine Baugruppenstückliste occurrence_id aus where_used verwenden")
    if result["operation"] != "bom" and "occurrence_id" in result:
        raise ValueError(
            "occurrence_id ist nur bei bom zulässig; Vorkommenseigenschaften liefert where_used"
        )
    if result["operation"] != "search" and not result.get("model_id"):
        raise ValueError("Diese CAD-Abfrage benötigt model_id")
    if result["operation"] in {"object", "where_used"} and not result.get("object_id"):
        raise ValueError("Diese CAD-Abfrage benötigt object_id")
    return result


def _model_info(row):
    from kiara.services.mcp_anonymization import source_ref

    return {
        "model_id": row["id"],
        "source_ref": source_ref(row["source"]),
        "filename": row["metadata"]["filename"],
        "file_path": row["source"],
        "source_hash": row["source_hash"],
        "processing_version": row["processing_version"],
        "datasource_ids": row["datasource_ids"],
        "structure_complete": row["structure_complete"],
        # Nur mit open_fields veröffentlichte Modelle: Feldformen ungeklärt.
        "verification_complete": row["metadata"].get("verification_complete", True),
        "open_findings": len(row["metadata"].get("open_findings", [])),
    }


def query(params, *, allowed_datasource_ids, embedding_model=None, collection_name=None):
    p = validate(params)
    ids = list(allowed_datasource_ids)
    if not ids:
        return {"operation": p["operation"], "items": [], "has_more": False}
    with db_query() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        if p["operation"] == "search":
            return _search(conn, p, ids, embedding_model, collection_name)
        row = conn.execute(_VISIBLE + " AND m.id=%s", (ids, p["model_id"])).fetchone()
        if row is None:
            raise LookupError("CAD-Modell ist nicht verfügbar oder nicht erlaubt")
        if p.get("source_hash") and p["source_hash"] != row["source_hash"]:
            raise ValueError(
                "Angefragte Dateiversion stimmt nicht mit dem relationalen CAD-Stand überein"
            )
        info = _model_info(row)
        if p["operation"] == "object":
            obj = conn.execute(
                "SELECT record FROM cad_contents WHERE model_id=%s AND object_id=%s",
                (row["id"], p["object_id"]),
            ).fetchone()
            if obj is None:
                raise LookupError("CAD-Objekt nicht vorhanden")
            # Raw parser fields are internal; only typed properties and labels leave SQL.
            content = {k: v for k, v in obj["record"].items() if k != "data"}
            return {"operation": "object", "items": [{**info, **content}], "has_more": False}
        if p["operation"] == "where_used":
            rows = conn.execute(
                "SELECT record FROM cad_occurrences WHERE model_id=%s AND object_id=%s AND physical "
                "ORDER BY path,occurrence_id LIMIT %s OFFSET %s",
                (row["id"], p["object_id"], p["limit"] + 1, p["offset"]),
            ).fetchall()
            return {
                "operation": "where_used",
                "items": [{**info, **x["record"]} for x in rows[: p["limit"]]],
                "has_more": len(rows) > p["limit"],
            }
        return _bom(conn, p, row, info)


def _search(conn, p, ids, model, collection):
    conditions, args = [], [ids]
    if p.get("object_id"):
        conditions.append("c.object_id=%s")
        args.append(p["object_id"])
    if p.get("model_id"):
        conditions.append("m.id=%s")
        args.append(p["model_id"])
    if p.get("source_hash"):
        conditions.append("m.source_hash=%s")
        args.append(p["source_hash"])
    if p.get("query"):
        conditions.append(
            "strpos(lower(array_to_string(c.names, ' ')), lower(%s))>0 OR "
            "EXISTS(SELECT 1 FROM jsonb_array_elements(c.properties) p WHERE p->>'status'='stored' AND strpos(lower(p->>'value'),lower(%s))>0)"
        )
        args.extend([p["query"], p["query"]])
    if "property_name" in p:
        condition = "p->>'name'=%s AND p->>'value'=%s AND p->>'status'='stored'"
        args.extend([p["property_name"], p["property_value"]])
        if "property_namespace" in p:
            condition += " AND p->>'namespace'=%s"
            args.append(p["property_namespace"])
        conditions.append(
            "EXISTS(SELECT 1 FROM jsonb_array_elements(c.properties) p WHERE " + condition + ")"
        )
    if p.get("numeric_field"):
        cond = "p->>'namespace'='cad' AND p->>'name'=%s AND p->>'status'='stored'"
        args.append(p["numeric_field"])
        for key, operator in (("minimum", ">="), ("maximum", "<=")):
            if key in p:
                if p.get(key + "_exclusive"):
                    operator = operator[0]
                cond += (
                    " AND (CASE WHEN jsonb_typeof(p->'value')='number' THEN (p->>'value')::numeric END) "
                    + operator
                    + " %s"
                )
                args.append(p[key])
        conditions.append(
            "EXISTS(SELECT 1 FROM jsonb_array_elements(c.properties) p WHERE " + cond + ")"
        )
    sql = (
        "WITH visible AS ("
        + _VISIBLE
        + ") SELECT m.*,c.record FROM visible m JOIN cad_contents c ON c.model_id=m.id "
    )
    sql += "WHERE c.kind IN ('part','assembly') AND EXISTS(SELECT 1 FROM cad_occurrences o WHERE o.model_id=m.id AND o.object_id=c.object_id AND o.physical)"
    sql += "".join(" AND (" + condition + ")" for condition in conditions)
    sql += " ORDER BY m.id,c.object_id LIMIT %s OFFSET %s"
    args.extend([p["limit"] + 1, p["offset"]])
    rows = conn.execute(sql, args).fetchall()
    items = [
        {
            **_model_info(r),
            **{
                k: v
                for k, v in r["record"].items()
                if k in {"object_id", "kind", "names", "properties", "labels", "source"}
            },
        }
        for r in rows[: p["limit"]]
    ]
    return {"operation": "search", "items": items, "has_more": len(rows) > p["limit"]}


def _bom(conn, p, row, info):
    if not row["structure_complete"]:
        raise ValueError("Unvollständig gelesene Struktur: keine verlässliche Stückliste verfügbar")
    parent = p.get("occurrence_id")
    if (
        parent
        and not conn.execute(
            "SELECT 1 FROM cad_occurrences o JOIN cad_contents c USING(model_id,object_id) "
            "WHERE o.model_id=%s AND occurrence_id=%s AND physical AND c.kind='assembly'",
            (row["id"], parent),
        ).fetchone()
    ):
        raise LookupError("Baugruppenvorkommen nicht vorhanden")
    sql = """WITH RECURSIVE selected AS (
        SELECT o.* FROM cad_occurrences o WHERE o.model_id=%s AND o.physical AND o.parent_id IS NOT DISTINCT FROM %s
        UNION ALL SELECT child.* FROM cad_occurrences child JOIN selected s
        ON child.model_id=s.model_id AND child.parent_id=s.occurrence_id WHERE child.physical AND %s
    ) SELECT c.object_id,c.kind,c.names,count(*)::integer AS quantity
      FROM selected o JOIN cad_contents c ON c.model_id=o.model_id AND c.object_id=o.object_id
      WHERE (%s=false OR c.kind='part') GROUP BY c.object_id,c.kind,c.names
      ORDER BY c.object_id LIMIT %s OFFSET %s"""
    recursive = p.get("recursive", True)
    rows = conn.execute(
        sql, (row["id"], parent, recursive, recursive, p["limit"] + 1, p["offset"])
    ).fetchall()
    return {
        "operation": "bom",
        "recursive": recursive,
        "scope_occurrence_id": parent,
        "items": [{**info, **dict(r)} for r in rows[: p["limit"]]],
        "has_more": len(rows) > p["limit"],
    }
