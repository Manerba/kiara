"""Abstrakte Basisklasse fuer VectorStore-Backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from kiara.backend_selector import normalize_model_name
from kiara.config import get_embedding_model


class VectorStoreBase(ABC):
    """Interface fuer alle VectorStore-Implementierungen.

    Jedes Backend muss diese Methoden implementieren.
    Der Rest der Codebase (Retriever, Ingestion, Health-Check)
    arbeitet nur gegen dieses Interface.
    """

    def __init__(self, config: dict, embedding_model: str | None = None):
        self._config = config
        raw_model = embedding_model or get_embedding_model(config)
        self._embed_model = normalize_model_name(raw_model)
        if not self._embed_model:
            raise ValueError(
                "VectorStoreBase: Kein Embedding-Modell verfügbar. "
                "Bitte ein aktives Embedding-Backend in der Admin-UI konfigurieren "
                "oder embedding_model explizit übergeben."
            )

    @abstractmethod
    def add(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """Dokumente mit Embeddings zur Collection hinzufuegen (Upsert)."""

    @abstractmethod
    def query(self, query_text: str, k: int = 5, where: dict | None = None) -> list[dict]:
        """Vektorsuche. Gibt Liste von {text, metadata, score, _doc_id} zurueck."""

    @abstractmethod
    def delete_by_sources(self, sources: list[str]) -> None:
        """Alle Chunks der angegebenen Quelldateien loeschen."""

    @abstractmethod
    def get_stats(self) -> dict:
        """Collection-Statistiken zurueckgeben."""

    @abstractmethod
    def get_directories(self) -> list[str]:
        """Eindeutige Verzeichnisse in der Collection."""

    @abstractmethod
    def get_file_types(self) -> list[str]:
        """Eindeutige Dateitypen in der Collection."""

    @abstractmethod
    def add_with_embeddings(
        self, texts: list[str], metadatas: list[dict], ids: list[str], embeddings: list[list[float]]
    ) -> None:
        """Chunks MIT vorberechneten Embeddings hinzufuegen (fuer Migration)."""

    @abstractmethod
    def get_collection_name(self) -> str:
        """Name der aktuellen Collection/Tabelle zurueckgeben."""

    @abstractmethod
    def get_all_documents(
        self,
        include: list[str] | None = None,
    ) -> dict:
        """Alle Dokumente aus der Collection laden.

        Args:
            include: Welche Felder laden, z.B. ["documents", "metadatas"].

        Returns:
            Dict mit 'ids', 'documents', 'metadatas' (je nach include).
        """

    def get_all_documents_batched(
        self,
        include: list[str] | None = None,
        limit: int = 10_000,
        offset: int = 0,
    ) -> dict:
        """Dokumente seitenweise laden (fuer grosse Collections).

        Default-Implementierung delegiert an get_all_documents() und
        schneidet manuell. Backends koennen dies effizienter ueberschreiben.
        """
        result = self.get_all_documents(include=include)
        end = offset + limit
        return {
            "ids": result["ids"][offset:end],
            "documents": result.get("documents", [])[offset:end] if result.get("documents") else [],
            "metadatas": result.get("metadatas", [])[offset:end] if result.get("metadatas") else [],
        }

    @abstractmethod
    def get_documents_by_ids(
        self,
        ids: list[str],
        include: list[str] | None = None,
    ) -> dict:
        """Dokumente anhand ihrer IDs laden.

        Args:
            ids: Liste der Chunk-IDs.
            include: Welche Felder laden, z.B. ["documents", "metadatas"].

        Returns:
            Dict mit 'ids', 'documents', 'metadatas' (je nach include).
        """

    def get_documents_by_source(self, source: str) -> dict:
        """Alle Chunks einer Quelldatei laden (fuer Parent-Doc-Retrieval).

        Default-Implementierung: laedt alle Dokumente und filtert.
        Backends koennen dies effizienter ueberschreiben.
        """
        result = self.get_all_documents(include=["documents", "metadatas"])
        ids, docs, metas = [], [], []
        for i, meta in enumerate(result.get("metadatas", [])):
            if meta.get("source") == source:
                ids.append(result["ids"][i])
                docs.append(result.get("documents", [""])[i])
                metas.append(meta)
        return {"ids": ids, "documents": docs, "metadatas": metas}

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """Chunks anhand ihrer IDs loeschen."""

    @abstractmethod
    def get_ids_by_sources(self, sources: list[str], *, ds_id: int | None = None) -> list[str]:
        """Chunk-IDs fuer gegebene Quelldateien zurueckgeben (lightweight)."""

    @abstractmethod
    def list_sources_for_datasource(self, ds_id: int, batch_size: int = 1000) -> list[dict]:
        """Quelldateien mit Chunk-Counts fuer eine Datasource-ID zurueckgeben."""

    @abstractmethod
    def count_total(self) -> int:
        """Gesamtanzahl aller Chunks in der Collection."""

    @abstractmethod
    def query_by_ids(
        self, query_text: str, ids: list[str], k: int = 5, where: dict | None = None
    ) -> list[dict]:
        """ID-basierte ANN-Suche (Fallback fuer Post-Filter)."""

    @abstractmethod
    def search(
        self, query_text: str, k: int = 10, offset: int = 0, where: dict | None = None
    ) -> list[dict]:
        """Vektorsuche mit offset-Support fuer Pagination."""
        ...

    @abstractmethod
    def count_filtered(self, where: dict | None = None) -> int:
        """Anzahl Chunks mit optionalem Filter."""
        ...

    @abstractmethod
    def update_ds_ids(self, ds_ids_map: dict[str, list[int]], *, wait: bool = False) -> None:
        """Aktualisiert das ds_ids-Array fuer die gegebenen Chunks.

        Args:
            ds_ids_map: Dict chunk_id -> [ds_id_1, ds_id_2, ...].
            wait: Wenn True, wartet Qdrant auf Commit (fuer Loesch-Konsistenz).
        """

    @abstractmethod
    def scroll_all(self, batch_size: int = 1000) -> Iterator[list[tuple[str, dict]]]:
        """Iteriert ueber alle Chunks. Gibt (chunk_id, metadata) Tupel in Batches zurueck."""

    @abstractmethod
    def remove_payload_field(self, field_name: str) -> None:
        """Entfernt ein Feld aus allen Chunk-Metadaten + zugehoerigen Index."""

    @classmethod
    @abstractmethod
    def list_collections(cls, config: dict) -> list[dict]:
        """Alle kb_*-Collections mit Chunk-Anzahl auflisten."""
