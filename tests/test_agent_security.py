"""Tests fuer Agent-Security: Datasource-Berechtigungen auf Tool-Ebene."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from kiara.tools.builtin_tools import ListDatasourcesTool, SearchDocumentsTool

# ===========================================================================
# SearchDocumentsTool — Datasource-Berechtigungen
# ===========================================================================


class TestSearchDocumentsPermissions:
    @pytest.fixture
    def config(self):
        return {"retrieval": {"rerank_fetch_multiplier": 3}}

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Ergebnis", "metadata": {"source": "doc.txt"}, "score": 0.9},
        ]
        return retriever

    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()
        reranker.enabled = False
        return reranker

    def test_allowed_datasource_ids_passed_to_retriever(
        self, config, mock_retriever, mock_reranker
    ):
        """allowed_datasource_ids wird als filter_datasource_ids an Retriever uebergeben."""
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=[1, 2, 3]
        )
        asyncio.run(tool.execute({"query": "test"}))
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_datasource_ids"] == [1, 2, 3]

    def test_specific_datasource_id_overrides(self, config, mock_retriever, mock_reranker):
        """Spezifische datasource_id ueberschreibt allowed_datasource_ids."""
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=[1, 2, 3]
        )
        asyncio.run(tool.execute({"query": "test", "datasource_id": 2}))
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_datasource_ids"] == [2]

    def test_unauthorized_datasource_id_rejected(self, config, mock_retriever, mock_reranker):
        """datasource_id ausserhalb der erlaubten IDs → Fehler."""
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=[1, 2, 3]
        )
        result = asyncio.run(tool.execute({"query": "test", "datasource_id": 99}))
        assert result.success is False
        assert "nicht im Filter" in result.display_text

    def test_no_restrictions_when_none(self, config, mock_retriever, mock_reranker):
        """allowed_datasource_ids=None → kein Filter (Admin-Zugang)."""
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=None
        )
        asyncio.run(tool.execute({"query": "test"}))
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_datasource_ids"] is None

    def test_datasource_id_as_string(self, config, mock_retriever, mock_reranker):
        """datasource_id als String (kommt vom LLM-Parsing) wird in int konvertiert."""
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=[5, 10]
        )
        asyncio.run(tool.execute({"query": "test", "datasource_id": "5"}))
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_datasource_ids"] == [5]

    def test_datasource_id_string_unauthorized(self, config, mock_retriever, mock_reranker):
        """datasource_id als String, nicht erlaubt → Fehler."""
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=[5, 10]
        )
        result = asyncio.run(tool.execute({"query": "test", "datasource_id": "99"}))
        assert result.success is False


# ===========================================================================
# SearchDocumentsTool — Filter-Parameter
# ===========================================================================


class TestSearchDocumentsFilters:
    @pytest.fixture
    def config(self):
        return {"retrieval": {"rerank_fetch_multiplier": 3}}

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        return retriever

    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()
        reranker.enabled = False
        return reranker

    def test_file_type_filter(self, config, mock_retriever, mock_reranker):
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config)
        asyncio.run(tool.execute({"query": "test", "file_type": ".pdf"}))
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_type"] == ".pdf"

    def test_directory_filter(self, config, mock_retriever, mock_reranker):
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config)
        asyncio.run(tool.execute({"query": "test", "directory": "/docs/rechnungen"}))
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_dir"] == "/docs/rechnungen"

    def test_combined_filters(self, config, mock_retriever, mock_reranker):
        tool = SearchDocumentsTool(
            mock_retriever, mock_reranker, config, allowed_datasource_ids=[1]
        )
        asyncio.run(
            tool.execute(
                {
                    "query": "Rechnung",
                    "datasource_id": 1,
                    "file_type": ".eml",
                    "directory": "/emails",
                }
            )
        )
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["filter_datasource_ids"] == [1]
        assert call_kwargs["filter_type"] == ".eml"
        assert call_kwargs["filter_dir"] == "/emails"

    def test_definition_includes_new_params(self):
        retriever = MagicMock()
        reranker = MagicMock()
        tool = SearchDocumentsTool(retriever, reranker, {})
        defn = tool.definition()
        props = defn.parameters["properties"]
        assert "datasource_id" in props
        assert "file_type" in props
        assert "directory" in props


# ===========================================================================
# ListDatasourcesTool — Berechtigungen
# ===========================================================================


class TestListDatasourcesPermissions:
    def _mock_db_rows(self, rows):
        """Erstellt einen Mock fuer db_query der die gegebenen Zeilen zurueckgibt."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_db = MagicMock()
        mock_db.execute.return_value = mock_cursor
        return mock_db

    def _make_row(self, id, name, type_, chunk_count, file_count):
        """Erstellt ein dict-aehnliches Mock-Row-Objekt."""
        return {
            "id": id,
            "name": name,
            "type": type_,
            "chunk_count": chunk_count,
            "file_count": file_count,
        }

    def test_filters_by_allowed_ids(self):
        rows = [
            self._make_row(1, "Emails", "imap", 100, 50),
            self._make_row(2, "Docs", "local", 200, 80),
            self._make_row(3, "Geheim", "local", 50, 20),
        ]
        tool = ListDatasourcesTool(allowed_datasource_ids=[1, 2])

        with patch("kiara.repositories.datasource.DatasourceRepository") as mock_repo_cls:
            mock_repo_cls.return_value.list_active_with_chunks.return_value = rows
            result = asyncio.run(tool.execute({}))

        assert result.success is True
        ds_ids = [d["id"] for d in result.data["datasources"]]
        assert 3 not in ds_ids
        assert 1 in ds_ids
        assert 2 in ds_ids

    def test_no_restriction_when_none(self):
        rows = [
            self._make_row(1, "A", "local", 10, 5),
            self._make_row(2, "B", "local", 20, 10),
        ]
        tool = ListDatasourcesTool(allowed_datasource_ids=None)

        with patch("kiara.repositories.datasource.DatasourceRepository") as mock_repo_cls:
            mock_repo_cls.return_value.list_active_with_chunks.return_value = rows
            result = asyncio.run(tool.execute({}))

        assert len(result.data["datasources"]) == 2

    def test_definition(self):
        tool = ListDatasourcesTool()
        defn = tool.definition()
        assert defn.name == "list_datasources"
        assert defn.required_permissions == []
        assert defn.parameters["properties"] == {}

    def test_display_text_format(self):
        rows = [self._make_row(5, "Mails", "imap", 1000, 500)]
        tool = ListDatasourcesTool(allowed_datasource_ids=None)

        with patch("kiara.repositories.datasource.DatasourceRepository") as mock_repo_cls:
            mock_repo_cls.return_value.list_active_with_chunks.return_value = rows
            result = asyncio.run(tool.execute({}))

        assert "ID 5" in result.display_text
        assert "Mails" in result.display_text
        assert "1000 Chunks" in result.display_text

    def test_empty_db(self):
        tool = ListDatasourcesTool(allowed_datasource_ids=None)

        with patch("kiara.repositories.datasource.DatasourceRepository") as mock_repo_cls:
            mock_repo_cls.return_value.list_active_with_chunks.return_value = []
            result = asyncio.run(tool.execute({}))

        assert result.success is True
        assert "Keine" in result.display_text
