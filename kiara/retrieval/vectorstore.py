"""Backward-compatibility wrapper.

Bestehender Code importiert VectorStore aus diesem Modul.
Intern wird direkt QdrantVectorStore instantiiert (ohne Factory-Indirektion).
"""


class VectorStore:
    """Factory-Klasse fuer VectorStore-Instanzen.

    Bleibt Klasse (statt Funktion) wegen list_collections()-classmethod
    die von 10+ Prod-Stellen genutzt wird.
    """

    def __new__(
        cls,
        config: dict,
        embedding_model: str | None = None,
        *,
        intent: str = "read",
        require_configured_default: bool | None = None,
    ):
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        return QdrantVectorStore(
            config,
            embedding_model,
            intent=intent,
            require_configured_default=require_configured_default,
        )

    @classmethod
    def list_collections(cls, config: dict) -> list[dict]:
        """Gibt alle VectorDB-Collections mit Modell-Info zurueck."""
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        return QdrantVectorStore.list_collections(config)
