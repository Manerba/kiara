#!/usr/bin/env python3
"""Ergaenzt den read_document-Positionsindex in allen bestehenden kb-Collections."""

import argparse
import json

from kiara.config import get_timeout, load_config
from kiara.retrieval.index_compatibility import IndexCompatibilityError
from kiara.retrieval.vectorstore import VectorStore


def ensure_indexes(config: dict) -> list[dict]:
    results = []
    # Indexaufbau darf laenger dauern als eine normale Retrieval-Anfrage.
    # Auch der gRPC-Client nutzt seinen Instanz-Timeout fuer CreateFieldIndex.
    maintenance_config = {
        **config,
        "vectordb": {**config.get("vectordb", {}), "timeout": get_timeout(config, "qdrant_index")},
    }
    for collection in VectorStore.list_collections(config):
        if collection["chunks"] == 0:
            # Leere, eventuell noch ungepruefte Collections bleiben unangetastet.
            # Ihr erster regulaerer Schreibzugriff legt den Index mit an.
            result = {"collection": collection["name"], "created": False, "empty": True}
            results.append(result)
            print(json.dumps(result), flush=True)
            continue
        print(f"Pruefe Dokument-Positionsindex: {collection['name']}", flush=True)
        try:
            store = VectorStore(
                maintenance_config, embedding_model=collection["model"], intent="maintenance"
            )
        except IndexCompatibilityError as exc:
            if exc.reason_code != "index_contract_unverified":
                raise
            # Diese Collection ist bereits fuer Lesen und Schreiben gesperrt.
            # Der Performance-Index darf ihren Vertrag nicht automatisch freigeben.
            result = {
                "collection": collection["name"],
                "created": False,
                "blocked": exc.reason_code,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            continue
        try:
            created = store.ensure_document_read_index()
            result = {"collection": collection["name"], "created": created}
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            store._client.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    ensure_indexes(load_config(args.config))


if __name__ == "__main__":
    main()
