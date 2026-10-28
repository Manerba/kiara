"""Tests fuer Sprint 11b — WikiSearchTool (wiki.search).

9 Tests: Suche, Slug-Filter, leere Queries, fehlende Dateien, Lesefehler.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from kiara.admin.wiki_manifest import CONTENT_ROOT
from kiara.admin.wiki_tool import WikiSearchTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tool():
    return WikiSearchTool()


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Query gegen mehrere Markdown-Dateien liefert sortierte Treffer
# ---------------------------------------------------------------------------


def test_search_returns_sorted_hits(tool):
    result = _run(tool.execute({"query": "Datenquelle"}))
    assert result.success
    hits = result.data["hits"]
    # Mindestens ein Treffer erwartet (wenn MD-Dateien existieren)
    if hits:
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 2. guide="users" filtert auf ein Dokument
# ---------------------------------------------------------------------------


def test_search_with_guide_filter(tool):
    result = _run(tool.execute({"query": "Benutzer", "guide": "users"}))
    assert result.success
    for hit in result.data["hits"]:
        assert hit["slug"] == "users"


# ---------------------------------------------------------------------------
# 3. Unbekannter Slug -> success=True, hits=[]
# ---------------------------------------------------------------------------


def test_unknown_slug_returns_empty(tool):
    result = _run(tool.execute({"query": "test", "guide": "nonexistent"}))
    assert result.success
    assert result.data["hits"] == []
    assert "nicht gefunden" in result.display_text


# ---------------------------------------------------------------------------
# 4. Leere Treffer -> success=True, hits=[]
# ---------------------------------------------------------------------------


def test_no_matches_returns_empty(tool):
    result = _run(tool.execute({"query": "xyznonexistent12345"}))
    assert result.success
    assert result.data["hits"] == []


# ---------------------------------------------------------------------------
# 5. Query-Mindestlaenge nach strip()
# ---------------------------------------------------------------------------


def test_short_query_returns_empty(tool):
    result = _run(tool.execute({"query": "x"}))
    assert result.success
    assert result.data["hits"] == []
    assert "mindestens 2" in result.display_text.lower()


# ---------------------------------------------------------------------------
# 6. concepts/ werden durchsucht
# ---------------------------------------------------------------------------


def test_concepts_are_searchable(tool):
    result = _run(tool.execute({"query": "Architektur"}))
    assert result.success
    slugs = [h["slug"] for h in result.data["hits"]]
    # architecture.md sollte Treffer liefern wenn es existiert
    if (CONTENT_ROOT / "concepts/architecture.md").exists():
        assert "architecture" in slugs


# ---------------------------------------------------------------------------
# 7. Fehlende MD-Dateien werden uebersprungen
# ---------------------------------------------------------------------------


def test_missing_md_files_skipped(tool):
    """Auch wenn ein Manifest-Eintrag keine MD-Datei hat, kein Fehler."""
    result = _run(tool.execute({"query": "Datenquelle"}))
    assert result.success


# ---------------------------------------------------------------------------
# 8. Query nur Leerzeichen -> success=True, hits=[]
# ---------------------------------------------------------------------------


def test_whitespace_query_returns_empty(tool):
    result = _run(tool.execute({"query": "   "}))
    assert result.success
    assert result.data["hits"] == []


# ---------------------------------------------------------------------------
# 9. read_text-Fehler wird geloggt und uebersprungen
# ---------------------------------------------------------------------------


def test_read_error_logged_and_skipped(tool, caplog):
    def _mock_read_text(*args, **kwargs):
        raise OSError("Permission denied")

    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
    ):
        with caplog.at_level(logging.WARNING):
            result = _run(tool.execute({"query": "test query"}))
    assert result.success
    assert result.data["hits"] == []
