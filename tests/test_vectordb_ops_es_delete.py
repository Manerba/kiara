"""Regressionstests fuer Elasticsearch-Loeschfehler im Delete-Pfad."""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest


@contextmanager
def _db_transaction_mock():
    yield MagicMock()


def test_delete_files_es_error_preserves_dense_retry_anchor():
    """ES-Fehler darf den Delete nicht als Erfolg abschliessen."""
    from kiara.ingestion.vectordb_ops import delete_files

    config = {"retrieval": {"hybrid_search": "elasticsearch"}}
    file_paths = ["/data/a.txt"]
    datasource_id = 7
    embedding_model = "dense-model"

    store = MagicMock()
    store.get_ids_by_sources.return_value = ["c1", "c2"]
    store.get_collection_name.return_value = "kb_dense"

    chunk_ds = MagicMock()
    chunk_ds.preview_chunk_removal_for_datasource.return_value = (
        ["c1"],
        {"c2": [99]},
    )
    hash_store = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.VectorStore", return_value=store),
        )
        stack.enter_context(
            patch(
                "kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds",
                return_value=[],
            ),
        )
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops._recount_datasource_embeddings"),
        )
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.db_transaction", _db_transaction_mock),
        )
        mock_bm25 = stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
        )
        stack.enter_context(
            patch(
                "kiara.retrieval.elasticsearch_service.is_configured",
                return_value=True,
            ),
        )
        mock_es_delete = stack.enter_context(
            patch(
                "kiara.retrieval.elasticsearch_service.delete_by_sources",
                side_effect=RuntimeError("es down"),
            ),
        )

        with pytest.raises(RuntimeError, match="es down"):
            delete_files(
                config,
                datasource_id,
                embedding_model,
                file_paths,
                chunk_ds=chunk_ds,
                hash_store=hash_store,
            )

    mock_es_delete.assert_called_once_with(
        config,
        file_paths,
        datasource_id=datasource_id,
    )
    store.delete_by_ids.assert_not_called()
    store.update_ds_ids.assert_not_called()
    mock_bm25.assert_not_called()


def test_delete_files_recounts_colbert_datasource_embeddings():
    """Datei-Delete muss Dense- und ColBERT-Counter aus der SoT neu setzen."""
    from kiara.ingestion.vectordb_ops import delete_files
    from kiara.retrieval.colbert import colbert_ds_model_key

    config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-model"}}}
    file_paths = ["/data/a.txt"]
    datasource_id = 7
    embedding_model = "dense-model"

    store = MagicMock()
    store.get_ids_by_sources.return_value = ["c1"]
    store.get_collection_name.return_value = "kb_dense"

    chunk_ds = MagicMock()
    chunk_ds.preview_chunk_removal_for_datasource.return_value = (["c1"], {})
    hash_store = MagicMock()

    conn = MagicMock()

    @contextmanager
    def fake_db_transaction():
        yield conn

    with ExitStack() as stack:
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.VectorStore", return_value=store),
        )
        stack.enter_context(
            patch(
                "kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds",
                return_value=["colbert-model"],
            ),
        )
        mock_recount = stack.enter_context(
            patch("kiara.ingestion.vectordb_ops._recount_datasource_embeddings"),
        )
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.db_transaction", fake_db_transaction),
        )
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch"),
        )
        stack.enter_context(
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
        )
        mock_colbert_store_cls = stack.enter_context(
            patch("kiara.retrieval.colbert.ColBERTStore"),
        )
        mock_colbert_store_cls.return_value = MagicMock()

        delete_files(
            config,
            datasource_id,
            embedding_model,
            file_paths,
            chunk_ds=chunk_ds,
            hash_store=hash_store,
        )

    mock_recount.assert_any_call(conn, datasource_id, embedding_model)
    mock_recount.assert_any_call(conn, datasource_id, colbert_ds_model_key("colbert-model"))
