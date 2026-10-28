"""VectorStore-Factory — Qdrant-Backend.

Zentrale Factory-/Enumerator-Stelle fuer alle Retrieval-Collections:
- Dense (``kb_*``): via ``create_vectorstore()`` / ``list_collections_for_backend()``
- ColBERT (``colbert_*``): via ``create_retrieval_store()`` / ``list_all_retrieval_collections()``

Bestehende Dense-only Pfade (Admin-Embedding-Models, VectorStore.list_collections())
importieren weiterhin ``list_collections_for_backend()`` und sehen nur ``kb_*``.
Neue Cache-/Integrity-/Admin-Pfade nutzen ``list_all_retrieval_collections()``.
"""

from kiara.backend_selector import normalize_model_name
from kiara.retrieval.vectorstore_base import VectorStoreBase


def create_vectorstore(
    config: dict,
    embedding_model: str | None = None,
    *,
    intent: str = "read",
    require_configured_default: bool | None = None,
) -> VectorStoreBase:
    """Erstellt einen QdrantVectorStore (einziges Dense-Backend)."""
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    return QdrantVectorStore(
        config,
        embedding_model,
        intent=intent,
        require_configured_default=require_configured_default,
    )


def list_collections_for_backend(config: dict) -> list[dict]:
    """Dense-Collections-Liste (``kb_*``) des Qdrant-Backends."""
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    return QdrantVectorStore.list_collections(config)


def list_all_retrieval_collections(config: dict) -> list[dict]:
    """Union aus ``kb_*``- und ``colbert_*``-Collections.

    Rueckgabe-Format:
        [{"kind": "kb", "name": "kb_nomic-embed-text", "model": "nomic-embed-text", "chunks": 42},
         {"kind": "colbert", "name": "colbert_colbert-xm", "model": "colbert-xm", "chunks": 10},
         ...]

    Wird in Cache-/Integrity-/Admin-Pfaden verwendet statt direkter
    ``VectorStore.list_collections()``-Aufrufe.
    """
    from kiara.retrieval.colbert import ColBERTStore
    from kiara.retrieval.embedding import model_name_from_collection

    dense = list_collections_for_backend(config)
    for entry in dense:
        entry["kind"] = "kb"
        if "model" not in entry:
            try:
                entry["model"] = model_name_from_collection(entry["name"])
            except ValueError:
                entry["model"] = entry["name"].removeprefix("kb_")

    colbert = ColBERTStore.list_colbert_collections(config)

    return dense + colbert


def create_retrieval_store(config: dict, kind: str, model: str, *, intent: str = "read"):
    """Factory fuer Cache-/Integrity-Pfade.

    Args:
        config: Globale Konfiguration.
        kind: ``"kb"`` fuer Dense, ``"colbert"`` fuer ColBERT.
        model: Normalisierter Modellname.

    Returns:
        QdrantVectorStore (kind="kb") oder ColBERTStore (kind="colbert").

    Raises:
        ValueError: Unbekannter ``kind``.
    """
    model = normalize_model_name(model)
    if not model:
        raise ValueError("create_retrieval_store: model darf nicht leer sein (fail-closed).")
    if kind == "kb":
        return create_vectorstore(config, embedding_model=model, intent=intent)
    if kind == "colbert":
        from kiara.retrieval.colbert import ColBERTStore

        return ColBERTStore(config, model, intent=intent)
    raise ValueError(f"Unbekannter Retrieval-Store-Typ: {kind!r}")
