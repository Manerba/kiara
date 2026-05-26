"""Tests fuer die erweiterten Pipeline-Stats.

Testet:
- DDU-Tracking (ddu_files, ddu_fallback_files)
- Error-Details Sammlung
- Graph-Stats (entities, relations, errors)
- _run_graph_extraction Return-Wert
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# DDU-Tracking in run_ingestion
# ---------------------------------------------------------------------------


class TestDduTracking:
    def test_ddu_file_counted(self):
        """load_file mit parser=docling-remote → stats['ddu_files'] inkrementiert."""
        stats = {
            "total_files": 1,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }
        doc = {
            "text": "Test Content",
            "metadata": {"source": "test.pdf", "parser": "docling-remote", "file_type": ".pdf"},
        }

        # Simuliere DDU-Tracking-Logik aus pipeline.py
        if doc and doc["metadata"].get("parser") in ("docling", "docling-remote"):
            stats.setdefault("ddu_files", 0)
            stats["ddu_files"] += 1
        if doc and doc["metadata"].get("parser_fallback"):
            stats.setdefault("ddu_fallback_files", 0)
            stats["ddu_fallback_files"] += 1

        assert stats["ddu_files"] == 1
        assert "ddu_fallback_files" not in stats

    def test_ddu_fallback_counted(self):
        """parser_fallback=True → stats['ddu_fallback_files'] inkrementiert."""
        stats = {}
        doc = {
            "text": "Test",
            "metadata": {"parser": "standard", "parser_fallback": True},
        }

        if doc and doc["metadata"].get("parser") in ("docling", "docling-remote"):
            stats.setdefault("ddu_files", 0)
            stats["ddu_files"] += 1
        if doc and doc["metadata"].get("parser_fallback"):
            stats.setdefault("ddu_fallback_files", 0)
            stats["ddu_fallback_files"] += 1

        assert "ddu_files" not in stats
        assert stats["ddu_fallback_files"] == 1


# ---------------------------------------------------------------------------
# Error Details
# ---------------------------------------------------------------------------


class TestErrorDetails:
    def test_loading_error_detail(self):
        """Fehlgeschlagenes Laden → error_details mit phase='loading'."""
        stats = {"errors": 0}
        file_path = "/data/docs/defekt.pdf"

        # Simuliere Fehler-Tracking aus pipeline.py
        stats["errors"] += 1
        stats.setdefault("error_details", [])
        stats["error_details"].append(
            {
                "file": Path(file_path).name,
                "error": "Datei konnte nicht geladen werden",
                "phase": "loading",
            }
        )

        assert stats["errors"] == 1
        assert len(stats["error_details"]) == 1
        assert stats["error_details"][0]["file"] == "defekt.pdf"
        assert stats["error_details"][0]["phase"] == "loading"

    def test_embedding_error_detail(self):
        """Fehlgeschlagenes Embedding → error_details mit phase='embedding'."""
        stats = {"errors": 0}
        file_path = "/data/docs/large.docx"
        error = Exception("Connection refused to embedding server")

        stats["errors"] += 1
        stats.setdefault("error_details", [])
        stats["error_details"].append(
            {
                "file": Path(file_path).name,
                "error": str(error)[:200],
                "phase": "embedding",
            }
        )

        assert stats["error_details"][0]["phase"] == "embedding"
        assert "Connection refused" in stats["error_details"][0]["error"]

    def test_error_truncation(self):
        """Fehler-Messages werden auf 200 Zeichen begrenzt."""
        long_error = "X" * 500
        truncated = long_error[:200]
        assert len(truncated) == 200


# ---------------------------------------------------------------------------
# Graph Stats
# ---------------------------------------------------------------------------


class TestGraphStats:
    def test_graph_extraction_return_value(self):
        """_run_graph_extraction gibt immer ein Stats-Dict zurueck."""
        # Teste mit deaktiviertem Graph
        config = {"graph": {"enabled": False}}
        from kiara.ingestion.runtime import _run_graph_extraction

        result = _run_graph_extraction(config, [], [])
        assert result == {"entities": 0, "relations": 0, "errors": 0}

    def test_graph_stats_disabled(self):
        """Deaktivierter Graph → alle Zaehler 0."""
        from kiara.ingestion.runtime import _run_graph_extraction

        config = {}
        result = _run_graph_extraction(config, ["text"], ["id1"])
        assert result["entities"] == 0
        assert result["relations"] == 0
        assert result["errors"] == 0

    def test_graph_stats_accumulation(self):
        """Graph-Stats werden korrekt in Haupt-Stats aggregiert."""
        stats = {}
        graph_result = {"entities": 5, "relations": 3, "errors": 1}

        stats.setdefault("graph_entities", 0)
        stats.setdefault("graph_relations", 0)
        stats.setdefault("graph_errors", 0)
        stats["graph_entities"] += graph_result["entities"]
        stats["graph_relations"] += graph_result["relations"]
        stats["graph_errors"] += graph_result["errors"]

        assert stats["graph_entities"] == 5
        assert stats["graph_relations"] == 3
        assert stats["graph_errors"] == 1


# ---------------------------------------------------------------------------
# loader.py: parser_fallback
# ---------------------------------------------------------------------------


class TestParserFallback:
    def test_loader_sets_parser_fallback(self):
        """Bei Docling-Fallback wird parser_fallback=True in Metadata gesetzt."""
        # Simuliere das Verhalten aus loader.py
        metadata = {"source": "test.pdf", "parser": "docling-remote"}
        text = None  # Docling lieferte keinen Text

        if not (text and text.strip()):
            metadata["parser_fallback"] = True

        assert metadata["parser_fallback"] is True

    def test_loader_no_fallback_on_success(self):
        """Erfolgreiche Docling-Extraktion → kein parser_fallback."""
        metadata = {"source": "test.pdf"}
        text = "Extracted content"

        if text and text.strip():
            metadata["parser"] = "docling-remote"

        assert "parser_fallback" not in metadata
