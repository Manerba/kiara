"""Opt-in pilot: user-supplied PKG, real local embeddings, isolated test DB/index.

CAD_PILOT_PKG and CAD_PILOT_EMBEDDING_URL must both be set explicitly.
The customer model and its text never enter the repository or test report.
"""

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
import pytest
from qdrant_client.models import Distance, VectorParams
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.cad import READER_REVISION, repository


@pytest.mark.skipif(
    not (os.getenv("CAD_PILOT_PKG") and os.getenv("CAD_PILOT_EMBEDDING_URL")),
    reason="Explicit local CAD pilot configuration required",
)
def test_real_pilot(whole_pipeline, test_db, tmp_path):
    started = time.monotonic()
    case = whole_pipeline
    pkg = tmp_path / "pilot.pkg"
    pkg.write_bytes(Path(os.environ["CAD_PILOT_PKG"]).read_bytes())
    source_hash = hashlib.sha256(pkg.read_bytes()).hexdigest()
    url = os.environ["CAD_PILOT_EMBEDDING_URL"].rstrip("/")
    model = os.getenv("CAD_PILOT_EMBEDDING_MODEL", "bge-m3")
    embedded = 0

    def embed(texts):
        nonlocal embedded
        vectors = []
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            response = httpx.post(
                url + "/api/embed", json={"model": model, "input": batch}, timeout=180
            )
            response.raise_for_status()
            result = response.json()["embeddings"]
            assert len(result) == len(batch)
            vectors.extend(result)
        embedded += len(texts)
        return vectors

    query_vector = embed(["Bauteile aus Blech mit zwei Millimeter Blechdicke"])[0]
    case.client.delete_collection("kb_test")
    case.client.create_collection(
        "kb_test", vectors_config=VectorParams(size=len(query_vector), distance=Distance.COSINE)
    )
    case.store.add = lambda texts, metas, ids: case.store.add_with_embeddings(
        texts, metas, ids, embed(texts)
    )
    case.store._config["datasource"]["file_types"] = [".pkg"]
    result = case.run_whole(case.ds, chunk_size=1000)
    assert result["errors"] == 0 and result["processed_files"] == 1
    case.enrich.assert_not_called()
    vector_data = case.store.get_documents_by_source(str(pkg))
    row = test_db.execute(
        "SELECT id,processing_version,structure_complete FROM cad_models WHERE source=%s",
        (str(pkg),),
    ).fetchone()
    assert row["structure_complete"]
    assert {m["processing_version"] for m in vector_data["metadatas"]} == {
        row["processing_version"]
    }
    assert {m["cad_source_hash"] for m in vector_data["metadatas"]} == {source_hash}

    def query(**params):
        return repository.query(
            params,
            allowed_datasource_ids=[case.ds],
            embedding_model="test-model",
            collection_name="kb_test",
        )

    def all_items(**params):
        items = []
        while True:
            page = query(**params, limit=200, offset=len(items))
            items.extend(page["items"])
            if not page["has_more"]:
                return items

    bom = all_items(operation="bom", model_id=row["id"], source_hash=source_hash)
    placements = test_db.execute(
        "SELECT count(*) AS n FROM cad_occurrences o JOIN cad_contents c USING(model_id,object_id) WHERE o.model_id=%s AND physical AND c.kind='part'",
        (row["id"],),
    ).fetchone()["n"]
    assert sum(item["quantity"] for item in bom) == placements
    most = max(bom, key=lambda item: item["quantity"])
    used = all_items(
        operation="where_used",
        model_id=row["id"],
        source_hash=source_hash,
        object_id=most["object_id"],
    )
    assert len(used) == most["quantity"] == len({i["occurrence_id"] for i in used})
    thicknesses = all_items(operation="search", model_id=row["id"], numeric_field="blechdicke_mm")
    counts = Counter(
        p["value"]
        for item in thicknesses
        for p in item["properties"]
        if p["namespace"] == "cad" and p["name"] == "blechdicke_mm" and p["status"] == "stored"
    )
    semantic = case.client.query_points(
        "kb_test", query=query_vector, limit=5, with_payload=True
    ).points
    assert len(semantic) == 5
    assert all(hit.payload["cad_source_hash"] == source_hash for hit in semantic)
    before = embedded
    repeat = case.run_whole(case.ds, chunk_size=1000)
    assert repeat["errors"] == 0 and repeat["processed_files"] == 0 and embedded == before
    report = {
        "source_sha256": source_hash,
        "reader_revision": READER_REVISION,
        "embedding_model": model,
        "dimensions": len(query_vector),
        "chunks": len(vector_data["ids"]),
        "part_contents": len(bom),
        "physical_part_occurrences": placements,
        "largest_where_used_count": len(used),
        "thickness_mm_counts": dict(counts),
        "semantic_top5_scores": [round(hit.score, 4) for hit in semantic],
        "semantic_top5_with_thickness_text": sum(
            "blechdicke_mm" in json.dumps(hit.payload) for hit in semantic
        ),
        "repeat_embeds": embedded - before,
        "publication_versions_match": True,
        "seconds": round(time.monotonic() - started, 2),
        "scope": "Isolated PostgreSQL test database and in-memory Qdrant; real embedding backend; test index contract",
    }
    if os.getenv("CAD_PILOT_REPORT"):
        Path(os.environ["CAD_PILOT_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
