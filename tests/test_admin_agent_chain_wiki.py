"""Tests fuer Sprint 11b — AdminAgentChain Wiki-/Tool-Integration.

6 Tests: Tool-Registry (wiki.search, ListDatasources), Fallback-Prompt.
"""

# DB-Backend initialisieren
import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"

from kiara.generation.admin_agent_chain import _ADMIN_SYSTEM_PROMPT_FALLBACK

# ---------------------------------------------------------------------------
# 1. Fallback-Prompt enthaelt wiki.search
# ---------------------------------------------------------------------------


def test_fallback_prompt_has_wiki_search():
    assert "wiki.search" in _ADMIN_SYSTEM_PROMPT_FALLBACK


# ---------------------------------------------------------------------------
# 2. Fallback-Prompt enthaelt NICHT search_documents
# ---------------------------------------------------------------------------


def test_fallback_prompt_no_search_documents():
    assert "SearchDocuments" not in _ADMIN_SYSTEM_PROMPT_FALLBACK
    assert "search_documents" not in _ADMIN_SYSTEM_PROMPT_FALLBACK


# ---------------------------------------------------------------------------
# 3. Fallback-Prompt enthaelt KEINEN User-Chat-Verweis
# ---------------------------------------------------------------------------


def test_fallback_prompt_no_user_chat_reference():
    assert "User-Chat verweisen" not in _ADMIN_SYSTEM_PROMPT_FALLBACK
    assert "Port 8501" not in _ADMIN_SYSTEM_PROMPT_FALLBACK


# ---------------------------------------------------------------------------
# 4. Fallback-Prompt enthaelt workflow.list und step-Details
# ---------------------------------------------------------------------------


def test_fallback_prompt_has_workflow_hints():
    assert "workflow.list" in _ADMIN_SYSTEM_PROMPT_FALLBACK
    assert "workflow.start" in _ADMIN_SYSTEM_PROMPT_FALLBACK
    assert "step.fixed_params" in _ADMIN_SYSTEM_PROMPT_FALLBACK
    assert "step.param_mapping" in _ADMIN_SYSTEM_PROMPT_FALLBACK


# ---------------------------------------------------------------------------
# 5. Fallback-Prompt enthaelt Gefuehrte Workflows und Dokumentation
# ---------------------------------------------------------------------------


def test_fallback_prompt_has_sections():
    assert "## Gefuehrte Workflows" in _ADMIN_SYSTEM_PROMPT_FALLBACK
    assert "## Dokumentation" in _ADMIN_SYSTEM_PROMPT_FALLBACK


# ---------------------------------------------------------------------------
# 6. Tool-Registry check (WikiSearchTool + ListDatasourcesTool vorhanden)
# ---------------------------------------------------------------------------


def test_tool_registry_contents():
    """Prueft dass WikiSearchTool und ListDatasourcesTool importierbar sind."""
    from kiara.admin.wiki_tool import WikiSearchTool
    from kiara.tools.builtin_tools import ListDatasourcesTool
    from kiara.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(WikiSearchTool())
    reg.register(ListDatasourcesTool())

    names = [t.name for t in reg.list_tools()]
    assert "wiki.search" in names
    assert "list_datasources" in names
    assert "search_documents" not in names
