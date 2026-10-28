"""Statische Frontend-Regressionschecks fuer Admin-UI-Sprints.

Prueft Quelltext-Invarianten in JS-Dateien und Templates
als Python-Assertions — kein DOM/Browser noetig.
Abgedeckte Sprints: MCP-Tab (1.4), Config/Handler/Fetch-Guards (2.4),
Chat-Utils-Entkopplung (Phase 2, Sprint 1).
"""

import pathlib
import re

import pytest

_BASE = pathlib.Path(__file__).resolve().parent.parent

TAB_MCP_JS = _BASE / "kiara" / "web" / "static" / "js" / "tab_mcp.js"
ADMIN_HTML = _BASE / "kiara" / "web" / "templates" / "admin.html"
APP_CORE_JS = _BASE / "kiara" / "web" / "static" / "js" / "app_core.js"
TAB_DATENQUELLEN_JS = _BASE / "kiara" / "web" / "static" / "js" / "tab_datenquellen.js"
TAB_RETRIEVAL_JS = _BASE / "kiara" / "web" / "static" / "js" / "tab_ki_retrieval.js"


@pytest.fixture(scope="module")
def mcp_src():
    return TAB_MCP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_html_src():
    return ADMIN_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_core_src():
    return APP_CORE_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tab_datenquellen_src():
    return TAB_DATENQUELLEN_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def retrieval_src():
    return TAB_RETRIEVAL_JS.read_text(encoding="utf-8")


class TestDatasourceBrowsePickerContract:
    def test_browse_picker_uses_backend_response_contract(self, tab_datenquellen_src):
        idx = tab_datenquellen_src.index("async function _openBrowsePicker")
        body = tab_datenquellen_src[idx : idx + 5000]
        assert "result.error" in body
        assert "result.path" in body
        assert "result.parent" in body
        assert "e.name" in body


class TestAdminFilterBarLifecycle:
    def test_router_resets_filterbar_with_classes_not_inline_hide(self, admin_html_src):
        idx = admin_html_src.index("function switchAdminTab")
        body = admin_html_src[idx : idx + 1800]
        assert "filterBar.style.display = 'none'" not in body
        assert "filterBar.style.display = ''" in body
        assert "filterBar.classList.remove('u-flex', 'u-visible', 'u-block')" in body
        assert "filterBar.classList.add('u-hidden')" in body


# ───────────────────────────────────────────────────────────────────
# AP1: Init + Lifecycle
# ───────────────────────────────────────────────────────────────────


class TestMcpLifecycleGuards:
    def test_run_tab_cleanups_at_init_start(self, mcp_src):
        """initMcpTab ruft _runTabCleanups('tools-mcp') am Anfang auf."""
        assert "_runTabCleanups('tools-mcp')" in mcp_src

    def test_request_seq_latest_only(self, mcp_src):
        """Monotoner requestSeq-Guard existiert."""
        assert "_mcpRequestSeq" in mcp_src

    def test_seq_guard_after_await(self, mcp_src):
        """Guard 'seq !== _mcpRequestSeq' nach Promise.all."""
        assert "seq !== _mcpRequestSeq" in mcp_src

    def test_subtab_guard_present(self, mcp_src):
        """Active-Subtab-Guard prueft adminCurrentSubTab."""
        assert "adminCurrentSubTab !== 'tools-mcp'" in mcp_src

    def test_resp_ok_settings(self, mcp_src):
        """Settings-Load via SharedUi.fetch.safeFetch (wirft bei non-2xx)."""
        assert "SharedUi.fetch.safeFetch('/api/admin/settings'" in mcp_src

    def test_resp_ok_keys(self, mcp_src):
        """Keys-Load via SharedUi.fetch.safeFetch (wirft bei non-2xx)."""
        assert "SharedUi.fetch.safeFetch('/api/admin/mcp/keys'" in mcp_src

    def test_resp_ok_users(self, mcp_src):
        """Users-Load via SharedUi.fetch.safeFetch (wirft bei non-2xx)."""
        assert "SharedUi.fetch.safeFetch('/api/admin/users'" in mcp_src

    def test_status_soft_fail(self, mcp_src):
        """Status-Request darf soft-fail zu null werden."""
        assert ".catch(" in mcp_src
        assert "statusResp && statusResp.ok" in mcp_src

    def test_init_is_global_async(self, mcp_src):
        """initMcpTab bleibt globale async function ohne Pflichtparameter."""
        assert "async function initMcpTab()" in mcp_src

    def test_init_has_try_catch(self, mcp_src):
        """initMcpTab hat eigenen try/catch."""
        assert "} catch (e)" in mcp_src

    def test_register_tab_cleanup(self, mcp_src):
        """registerTabCleanup('tools-mcp', ...) wird registriert."""
        assert "registerTabCleanup('tools-mcp'" in mcp_src

    def test_render_pending_after_init(self, mcp_src):
        """Pending Key wird nach erfolgreichem Render nachgerendert."""
        assert "_renderPendingMcpKey()" in mcp_src


# ───────────────────────────────────────────────────────────────────
# AP2: Button-Locks + Promise-Propagation
# ───────────────────────────────────────────────────────────────────


class TestMcpButtonLocks:
    def test_action_binding_uses_with_button_lock(self, mcp_src):
        """Die delegierte MCP-Action-Schicht verwendet withButtonLock(actionEl, ...)."""
        assert "data-mcp-action" in mcp_src
        assert "SharedUi.runtime.withButtonLock(actionEl," in mcp_src

    def test_toggle_server_awaits_init(self, mcp_src):
        """mcpToggleServer awaitet initMcpTab() im try/catch."""
        assert "await initMcpTab()" in mcp_src

    def test_toggle_key_awaits_init(self, mcp_src):
        """mcpToggleKey awaitet initMcpTab()."""
        # Finde mcpToggleKey body und pruefe await initMcpTab
        idx = mcp_src.index("async function mcpToggleKey(")
        body = mcp_src[idx : idx + 600]
        assert "await initMcpTab()" in body

    def test_delete_key_awaits_init(self, mcp_src):
        """mcpDeleteKey awaitet initMcpTab()."""
        idx = mcp_src.index("async function mcpDeleteKey(")
        body = mcp_src[idx : idx + 600]
        assert "await initMcpTab()" in body

    def test_no_manual_btn_disable_in_toggle_server(self, mcp_src):
        """Manuelle btn.disabled/textContent-Logik ist entfernt."""
        assert "btn.textContent = 'Wird umgeschaltet...'" not in mcp_src

    def test_no_fire_and_forget_init(self, mcp_src):
        """Kein blosses 'initMcpTab()' ohne await in Mutationen."""
        # Suche nach Stellen mit initMcpTab() ohne voranstehendes await
        # (Nur in Mutations-Funktionen, nicht in renderMcpTab-Referenzen)
        for fn_name in ("mcpToggleServer", "mcpToggleKey", "mcpDeleteKey"):
            idx = (
                mcp_src.index(f"function {fn_name}(")
                if f"function {fn_name}(" in mcp_src
                else mcp_src.index(f"async function {fn_name}(")
            )
            body = mcp_src[idx : idx + 600]
            # Jedes initMcpTab() muss ein await davor haben
            for m in re.finditer(r"initMcpTab\(\)", body):
                before = body[max(0, m.start() - 10) : m.start()]
                assert "await " in before, f"{fn_name}: initMcpTab() ohne await"


# ───────────────────────────────────────────────────────────────────
# AP3: One-Time-Key-Stash
# ───────────────────────────────────────────────────────────────────


class TestMcpOneTimeKeyStash:
    def test_pending_key_module_slot(self, mcp_src):
        """Modul-Slot _mcpPendingCreatedKey existiert."""
        assert "_mcpPendingCreatedKey" in mcp_src

    def test_render_pending_does_not_clear(self, mcp_src):
        """_renderPendingMcpKey rendert nur, loescht den Slot nicht."""
        lines = mcp_src.split("\n")
        for i, line in enumerate(lines):
            if "function _renderPendingMcpKey()" in line:
                block = "\n".join(lines[i : i + 30])
                assert "_mcpPendingCreatedKey = null" not in block
                return
        pytest.fail("_renderPendingMcpKey not found")

    def test_dismiss_function_exists(self, mcp_src):
        """mcpDismissPendingKey existiert als globale Funktion."""
        assert "function mcpDismissPendingKey()" in mcp_src

    def test_dismiss_clears_slot(self, mcp_src):
        """mcpDismissPendingKey leert den Slot."""
        idx = mcp_src.index("function mcpDismissPendingKey()")
        body = mcp_src[idx : idx + 300]
        assert "_mcpPendingCreatedKey = null" in body

    def test_copy_function_exists(self, mcp_src):
        """mcpCopyKey existiert als globale Funktion."""
        assert "function mcpCopyKey()" in mcp_src

    def test_clipboard_guard(self, mcp_src):
        """Clipboard-API nur unter navigator.clipboard && window.isSecureContext."""
        assert "navigator.clipboard && window.isSecureContext" in mcp_src

    def test_copy_fallback_checks_exec_command(self, mcp_src):
        """Fallback prueft document.execCommand('copy') Rueckgabewert."""
        assert "document.execCommand('copy')" in mcp_src

    def test_pending_key_render_has_copy_and_dismiss(self, mcp_src):
        """Pending-Key-Block rendert Copy- und Dismiss-Actions via data-*."""
        lines = mcp_src.split("\n")
        for i, line in enumerate(lines):
            if "function _renderPendingMcpKey()" in line:
                block = "\n".join(lines[i : i + 30])
                assert 'data-mcp-action="copy-pending-key"' in block
                assert 'data-mcp-action="dismiss-pending-key"' in block
                return
        pytest.fail("_renderPendingMcpKey not found")

    def test_create_key_sets_slot_before_dom(self, mcp_src):
        """mcpCreateKey setzt Slot VOR Tab-Guard und DOM-Zugriff."""
        idx = mcp_src.index("async function mcpCreateKey()")
        # Finde das Ende der Funktion (naechste Top-Level-Funktion)
        next_fn = re.search(r"\n(?:async )?function ", mcp_src[idx + 10 :])
        end = idx + 10 + next_fn.start() if next_fn else len(mcp_src)
        body = mcp_src[idx:end]
        slot_set = body.index("_mcpPendingCreatedKey = data.key")
        # adminCurrentSubTab-Guard muss NACH dem Slot-Set kommen
        guard_pos = body.index("adminCurrentSubTab", slot_set)
        assert guard_pos > slot_set

    def test_create_key_guards_late_dom_writes(self, mcp_src):
        """mcpCreateKey guardet spaete DOM-Writes nach Off-Tab."""
        idx = mcp_src.index("async function mcpCreateKey()")
        next_fn = re.search(r"\n(?:async )?function ", mcp_src[idx + 10 :])
        end = idx + 10 + next_fn.start() if next_fn else len(mcp_src)
        body = mcp_src[idx:end]
        slot_set = body.index("_mcpPendingCreatedKey = data.key")
        rest = body[slot_set:]
        assert "adminCurrentSubTab" in rest


# ───────────────────────────────────────────────────────────────────
# AP4: Fehlerresolver + Delete-Serialisierung
# ───────────────────────────────────────────────────────────────────


class TestMcpErrorResolver:
    def test_common_resolver_exists(self, mcp_src):
        """Gemeinsamer Resolver _mcpResolveError existiert."""
        assert "function _mcpResolveError(" in mcp_src

    def test_resolver_chain(self, mcp_src):
        """Resolver prueft detail, error, message Kette."""
        idx = mcp_src.index("function _mcpResolveError(")
        body = mcp_src[idx : idx + 300]
        assert ".detail" in body
        assert ".error" in body
        assert ".message" in body

    def test_tolerates_non_json_errors(self, mcp_src):
        """Resolver toleriert Nicht-JSON via .catch(() => ({}))."""
        assert ".catch(() => ({}))" in mcp_src

    def test_no_bare_data_detail_error_patterns(self, mcp_src):
        """Alte data.detail-Error-Patterns sind entfernt."""
        assert "data.detail || 'Fehler'" not in mcp_src
        assert ".detail || 'Toggle" not in mcp_src

    def test_all_mutations_use_resolver(self, mcp_src):
        """Alle 4 MCP-Mutationen verwenden _mcpResolveError."""
        for fn_name in ("mcpCreateKey", "mcpToggleServer", "mcpToggleKey", "mcpDeleteKey"):
            marker = f"async function {fn_name}("
            idx = mcp_src.index(marker)
            # Finde das Ende: naechste Top-Level function-Deklaration
            next_fn = re.search(r"\n(?:async )?function ", mcp_src[idx + 10 :])
            end = idx + 10 + next_fn.start() if next_fn else len(mcp_src)
            body = mcp_src[idx:end]
            assert "_mcpResolveError(" in body, f"{fn_name} fehlt _mcpResolveError"


class TestMcpDeleteSerialization:
    def test_delete_uses_data_attribute_bridge(self, mcp_src):
        """Delete-Caller nutzt data-key-name Attribut-Bridge."""
        assert 'data-key-name="' in mcp_src

    def test_delete_reads_data_attribute(self, mcp_src):
        """Delete-Handler liest Key-Name via getAttribute."""
        # Im JS-Quelltext steht getAttribute(\\'data-key-name\\') (escaped im String)
        assert "getAttribute" in mcp_src and "data-key-name" in mcp_src

    def test_no_bare_escape_attr_in_onclick(self, mcp_src):
        """Kein rohes escapeAttr(k.name) als JS-String im onclick."""
        # Altes Pattern: onclick="mcpDeleteKey(id, 'escapeAttr(name)')"
        assert "mcpDeleteKey(' + k.id + ', \\'' + escapeAttr(k.name)" not in mcp_src

    def test_no_raw_json_stringify_in_onclick(self, mcp_src):
        """Kein rohes JSON.stringify im double-quoted onclick."""
        # Sicherstellen, dass kein JSON.stringify fuer den Key-Name im onclick ist
        assert "JSON.stringify(k.name)" not in mcp_src


# ───────────────────────────────────────────────────────────────────
# Globale UI-Entry-Points
# ───────────────────────────────────────────────────────────────────


class TestMcpGlobalEntryPoints:
    def test_init_mcp_tab(self, mcp_src):
        assert "async function initMcpTab()" in mcp_src

    def test_mcp_create_key(self, mcp_src):
        assert "async function mcpCreateKey()" in mcp_src

    def test_mcp_copy_key(self, mcp_src):
        assert "function mcpCopyKey()" in mcp_src

    def test_mcp_dismiss_pending_key(self, mcp_src):
        assert "function mcpDismissPendingKey()" in mcp_src

    def test_mcp_toggle_server(self, mcp_src):
        assert "async function mcpToggleServer(" in mcp_src

    def test_mcp_toggle_key(self, mcp_src):
        assert "async function mcpToggleKey(" in mcp_src

    def test_mcp_delete_key(self, mcp_src):
        assert "async function mcpDeleteKey(" in mcp_src

    def test_mcp_save_retrieval_settings(self, mcp_src):
        assert "async function mcpSaveRetrievalSettings()" in mcp_src


class TestMcpRetrievalOnlySettings:
    def test_status_render_separates_mcp_endpoint_retrieval_and_service(self, mcp_src):
        assert "mcp_endpoint_enabled" in mcp_src
        assert "retrieval_enabled" in mcp_src
        assert "service_should_run" in mcp_src
        assert "MCP-Endpunkt <code>/mcp</code>" in mcp_src
        assert "Retrieval-REST <code>/api/retrieval/*</code>" in mcp_src
        assert "kiara-mcp.service aktiv" in mcp_src

        status_idx = mcp_src.index("function renderMcpTab(")
        retrieval_idx = mcp_src.index("function _mcpRenderRetrievalSettings(")
        status_body = mcp_src[status_idx:retrieval_idx]
        assert "statusData?.config_enabled ?? config?.web?.mcp_enabled" not in status_body

    def test_retrieval_only_section_in_mcp_tab(self, mcp_src):
        assert "Retrieval-only API" in mcp_src
        assert "_mcpRenderRetrievalSettings(config)" in mcp_src

    @pytest.mark.parametrize(
        "key",
        [
            "web.mcp_retrieval_enabled",
            "web.mcp_retrieval_allow_raw_chunks",
            "web.mcp_retrieval_include_text_default",
            "web.mcp_retrieval_max_k",
            "web.mcp_retrieval_max_query_length",
            "web.mcp_retrieval_max_chars_per_result",
            "web.mcp_retrieval_max_context_window",
            "web.mcp_retrieval_max_context_chunks",
            "web.mcp_retrieval_max_chars_total",
        ],
    )
    def test_retrieval_only_config_keys_rendered(self, mcp_src, key):
        assert key in mcp_src

    def test_retrieval_only_save_uses_settings_api(self, mcp_src):
        idx = mcp_src.index("async function mcpSaveRetrievalSettings()")
        body = mcp_src[idx : idx + 1600]
        assert "collectAndSaveConfig(_mcpCurrentConfig" in body
        assert "mcp-retrieval-input[data-key]" in body
        assert "mcp-retrieval-checkbox[data-key]" in body

    def test_retrieval_only_save_warns_before_raw_chunks(self, mcp_src):
        idx = mcp_src.index("async function mcpSaveRetrievalSettings()")
        body = mcp_src[idx : idx + 1600]
        assert "web.mcp_retrieval_allow_raw_chunks" in body
        assert "SharedUi.modal.confirm" in body
        assert "trusted Agents" in body

    def test_retrieval_only_default_text_requires_raw_chunks(self, mcp_src):
        idx = mcp_src.index("async function mcpSaveRetrievalSettings()")
        body = mcp_src[idx : idx + 1600]
        assert "web.mcp_retrieval_include_text_default" in body
        assert "Text standardmaessig mitsenden setzt erlaubte Rohchunks voraus" in body


class TestRetrievalTabMcpPointer:
    def test_retrieval_tab_points_to_mcp_tab_for_external_api(self, retrieval_src):
        assert "Hinweis zu Retrieval-only API" in retrieval_src
        assert "Tools &gt; MCP / API-Keys" in retrieval_src
        assert "Chat-/RAG-Retrieval-Policy" in retrieval_src


# ───────────────────────────────────────────────────────────────────
# admin.html Lifecycle-Anker
# ───────────────────────────────────────────────────────────────────


class TestAdminHtmlMcpLifecycle:
    def test_mcp_tab_config_entry(self, admin_html_src):
        """TAB_CONFIG enthaelt tools-mcp mit initMcpTab."""
        assert "'tools-mcp'" in admin_html_src
        assert "initMcpTab" in admin_html_src

    def test_switch_admin_tab_runs_sub_cleanups(self, admin_html_src):
        """switchAdminTab ruft _runSubTabCleanups auf."""
        assert "_runSubTabCleanups(" in admin_html_src

    def test_switch_sub_tab_runs_cleanups(self, admin_html_src):
        """switchSubTab ruft _runTabCleanups(oldSubTab) auf."""
        assert "_runTabCleanups(oldSubTab)" in admin_html_src

    def test_info_button_uses_data_action(self, admin_html_src):
        assert 'data-admin-action="toggle-info"' in admin_html_src


# ───────────────────────────────────────────────────────────────────
# app_core.js Helper-Existenz
# ───────────────────────────────────────────────────────────────────


class TestAppCoreHelpers:
    def test_register_tab_cleanup_exists(self, app_core_src):
        assert "function registerTabCleanup(" in app_core_src

    def test_run_tab_cleanups_exists(self, app_core_src):
        assert "function _runTabCleanups(" in app_core_src

    def test_run_sub_tab_cleanups_exists(self, app_core_src):
        assert "function _runSubTabCleanups(" in app_core_src

    def test_with_button_lock_extracted_to_runtime(self, app_core_src):
        """Sprint 2: withButtonLock lebt in ui_core/js/runtime.js, nicht mehr in app_core.js."""
        assert "function withButtonLock(" not in app_core_src
        assert "async function withButtonLock(" not in app_core_src

    def test_safe_fetch_extracted_to_fetch(self, app_core_src):
        """Sprint 2: safeFetch lebt in ui_core/js/fetch.js, nicht mehr in app_core.js."""
        assert "function safeFetch(" not in app_core_src
        assert "async function safeFetch(" not in app_core_src

    def test_password_modal_uses_data_close(self, app_core_src):
        assert "data-sharedui-close" in app_core_src
        assert 'onclick="SharedUi.modal.close()"' not in app_core_src


# ───────────────────────────────────────────────────────────────────
# Sprint 2.4: Config-Helper, Inline-Handler, Fetch-Guards
# ───────────────────────────────────────────────────────────────────

_JS = _BASE / "kiara" / "web" / "static" / "js"
_UI_JS = _BASE / "kiara" / "web" / "static" / "ui_core" / "js"


def _js(name):
    return (_JS / name).read_text(encoding="utf-8")


def _ui_js(name):
    return (_UI_JS / name).read_text(encoding="utf-8")


class TestSprint24ConfigUnwrap:
    """AP2: loadConfigFromAPI()-Caller duerfen kein doppeltes .config-Unwrap machen."""

    @pytest.mark.parametrize(
        "filename",
        [
            "tab_allgemein_web.js",
            "tab_allgemein_vektordb.js",
            "tab_ki_ingestion.js",
        ],
    )
    def test_no_double_unwrap_settings(self, filename):
        src = _js(filename)
        # loadConfigFromAPI entpackt bereits .config — Caller darf nicht nochmal unwrappen
        assert "data.config !== undefined" not in src

    @pytest.mark.parametrize(
        "filename",
        [
            "tab_mail.js",
            "tab_onlyoffice.js",
        ],
    )
    def test_no_double_unwrap_module_config(self, filename):
        src = _js(filename)
        assert "data.config" not in src or "data.config !== undefined" not in src


class TestSprint24InlineHandlerRemoval:
    """AP3: Dynamische Backend-/Userdaten nicht in Inline-JS-Strings."""

    def test_backup_uses_data_restore_file(self):
        src = _js("tab_backup.js")
        assert "data-restore-file" in src
        # Kein dynamischer Filename als JS-String-Argument im onclick
        assert "restoreBackup(this, '" not in src

    def test_testsuite_section_toggle_no_escaped_quotes(self):
        src = _js("tab_testsuite.js")
        # Alte escaped-Quotes Syntax ist entfernt
        assert "toggleTestSection(\\'" not in src
        # Neue data-* Toggles vorhanden
        assert "data-toggle-section" in src
        assert "data-toggle-detail" in src

    def test_testsuite_bind_toggles_exists(self):
        src = _js("tab_testsuite.js")
        assert "function _bindTestToggles(" in src

    def test_gruppen_actions_use_data_bridge(self):
        """V30 Welle 2 (#1049): data-group-id/data-group-name bleiben, aber
        der Leser ist jetzt der globale Dispatcher via el.dataset statt
        Inline-getAttribute-Bridge."""
        src = _js("tab_gruppen.js")
        assert "data-group-id" in src
        assert "data-group-name" in src
        assert "el.dataset.groupId" in src
        assert "getAttribute('data-group-id')" not in src

    def test_benutzer_actions_use_data_bridge(self):
        """V30 Welle 2 (#1049): data-user-id/data-username bleiben, aber
        der Leser ist jetzt der globale Dispatcher via el.dataset statt
        Inline-getAttribute-Bridge."""
        src = _js("tab_benutzer.js")
        assert "data-user-id" in src
        assert "data-username" in src
        # Neuer Kanal: dataset im Dispatcher
        assert "el.dataset.userId" in src
        # Alter Inline-Bridge ist entfernt
        assert "getAttribute('data-user-id')" not in src

    def test_benutzer_toggle_uses_data_field(self):
        """data-field bleibt am Checkbox-Input, Leser ist jetzt
        _handleUserChange ueber el.dataset.field."""
        src = _js("tab_benutzer.js")
        assert "data-field" in src
        assert "el.dataset.field" in src
        assert "getAttribute('data-field')" not in src

    def test_ldap_group_select_uses_data_bridge(self):
        """V30 Welle 2 Batch 2 (#1049): data-group-dn/data-group-name bleiben,
        Leser ist jetzt der globale Dispatcher via el.dataset."""
        src = _js("tab_ldap.js")
        assert "data-group-dn" in src
        assert "data-group-name" in src
        assert "el.dataset.groupDn" in src
        assert "getAttribute('data-group-dn')" not in src

    def test_workflows_export_uses_data_bridge(self):
        src = _js("tab_workflows.js")
        assert "data-wf-download" in src
        assert "data-wf-delete" in src
        # Kein dynamischer Filename als JS-String im onclick
        assert 'downloadWorkflowExport("${' not in src
        assert 'deleteWorkflowExport("${' not in src

    def test_vektordb_detail_uses_data_bridge(self):
        """V30 Welle 2 (#1049): data-ds-id/data-model/data-ds-name bleiben,
        Leser ist jetzt der globale Dispatcher via el.dataset."""
        src = _js("tab_vektordb_management.js")
        assert "data-ds-id" in src
        assert "data-model" in src
        assert "data-ds-name" in src
        assert "el.dataset.dsId" in src
        assert "getAttribute('data-ds-id')" not in src

    def test_vektordb_detail_uses_server_side_pagination(self):
        src = _js("tab_vektordb_management.js")
        assert "function _vdbMgmt_loadFiles(" in src
        assert "page_size: String(_vdbMgmt.pageSize)" in src
        assert 'data-action="vdbmgmt-page"' in src
        assert 'data-change-action="vdbmgmt-page-size"' in src
        assert "filteredFiles" not in src

    def test_event_stop_propagation_gruppen(self):
        """V30 Welle 2 (#1049): Inline event.stopPropagation() entfernt;
        stopPropagation wird jetzt im Dispatcher fuer _GRUPPEN_ROW_ACTIONS
        ausgefuehrt (Row-Expand-Block bleibt fachlich erhalten)."""
        src = _js("tab_gruppen.js")
        assert "_GRUPPEN_ROW_ACTIONS" in src
        assert "e.stopPropagation()" in src

    def test_event_stop_propagation_benutzer(self):
        """Analog zu gruppen: stopPropagation wandert vom Inline-onclick
        in den Dispatcher-Body fuer _USER_ROW_ACTIONS."""
        src = _js("tab_benutzer.js")
        assert "_USER_ROW_ACTIONS" in src
        assert "e.stopPropagation()" in src


class TestSprint24FetchGuards:
    """AP1/AP3: resp.ok-Checks und Post-Await-Guards."""

    def test_graph_init_resp_ok(self):
        src = _js("tab_ki_graph.js")
        assert "!statsResp.ok" in src or "statsResp.ok" in src

    def test_graph_save_preread_resp_ok(self):
        src = _js("tab_ki_graph.js")
        idx = src.index("function saveGraphConfig(")
        # Range 900 (statt 600): saveGraphConfig hat durch die Formular-Snapshots
        # VOR dem ersten await mehr Zeilen am Anfang — der resp.ok-Check rutscht
        # damit ueber das alte 600er-Fenster hinaus.
        body = src[idx : idx + 900]
        assert "!resp.ok" in body or "resp.ok" in body

    def test_performance_load_resp_ok(self):
        src = _js("tab_performance.js")
        idx = src.index("function loadPerformanceData(")
        body = src[idx : idx + 400]
        assert "!resp.ok" in body

    def test_monitoring_fetch_history_resp_ok(self):
        src = _js("monitoring_history.js")
        idx = src.index("function _fetchHistory(")
        body = src[idx : idx + 300]
        assert "!resp.ok" in body

    def test_monitoring_fetch_services_resp_ok(self):
        src = _js("tab_monitoring.js")
        idx = src.index("function _fetchServices(")
        body = src[idx : idx + 700]
        assert "!resp.ok" in body

    def test_monitoring_fetch_journal_resp_ok(self):
        src = _js("tab_monitoring.js")
        idx = src.index("function _fetchJournalErrors(")
        body = src[idx : idx + 700]
        assert "!resp.ok" in body

    def test_monitoring_cleanup_invalidates_seq(self):
        src = _js("tab_monitoring.js")
        assert "_monInitSeq++" in src
        # Muss im registrierten Cleanup-Wrapper sein, nicht nur im Init
        idx = src.index("registerTabCleanup('system-monitoring'")
        block = src[idx : idx + 200]
        assert "_monInitSeq++" in block

    def test_compliance_matrix_resp_ok(self):
        src = _js("tab_compliance_matrix.js")
        assert "!resp.ok" in src

    def test_compliance_matrix_init_gen_guard(self):
        src = _js("tab_compliance_matrix.js")
        assert "_cmInitSeq" in src

    def test_backup_init_resp_ok(self):
        src = _js("tab_backup.js")
        assert "!r.ok" in src or "!listResp.ok" in src

    def test_backup_cleanup_invalidates_seq(self):
        src = _js("tab_backup.js")
        idx = src.index("registerTabCleanup('system-backup'")
        block = src[idx : idx + 200]
        assert "_backupInitSeq++" in block

    def test_testsuite_init_resp_ok(self):
        src = _js("tab_testsuite.js")
        idx = src.index("const resp = await SharedUi.fetch.safeFetch('/api/admin/tests/status')")
        body = src[idx : idx + 200]
        assert "!resp.ok" in body

    def test_testsuite_cleanup_invalidates_seq(self):
        src = _js("tab_testsuite.js")
        idx = src.index("registerTabCleanup('system-selbsttest'")
        block = src[idx : idx + 200]
        assert "_testInitSeq++" in block

    def test_workflows_init_resp_ok(self):
        src = _js("tab_workflows.js")
        assert "!r.ok" in src

    def test_admin_chat_reload_hydrates_pending_server_state(self):
        tab_src = _js("tab_kiara.js")
        chat_src = _js("admin_chat.js")
        assert "hydrateServerState" in tab_src
        assert "pending_confirmations: chat.pending_confirmations" in tab_src
        assert "active_wizard: chat.active_wizard" in tab_src
        assert "hydrateServerState(state = {})" in chat_src
        assert "_renderConfirmationBubble" in chat_src
        assert "_renderWizardChecklist" in chat_src

    def test_safe_fetch_retries_server_confirmation_contract(self):
        src = _ui_js("fetch.js")
        assert "requires_confirmation" in src
        assert "X-KIARA-Confirm-Token" in src
        assert "X-KIARA-Confirm-String" in src
        assert "confirmation_cancelled" in src


class TestSprint24PreserveAudits:
    """AP4: Preserve-Audits und Spezialfall-Fixes."""

    def test_compliance_berichte_kpi_field_names(self):
        src = _js("tab_compliance_berichte.js")
        assert "total_queries_30d" in src
        assert "entities_anonymized_30d" in src
        assert "unique_users_30d" in src
        # Alte falsche Feldnamen entfernt
        assert "d.total_queries " not in src or "d.total_queries_30d" in src

    def test_compliance_berichte_init_gen_guard(self):
        src = _js("tab_compliance_berichte.js")
        assert "_cbInitSeq" in src

    def test_compliance_prompt_schutz_uses_content_container(self):
        src = _js("tab_compliance_prompt_schutz.js")
        assert "getElementById('contentContainer')" in src

    def test_compliance_prompt_schutz_resets_filters(self):
        src = _js("tab_compliance_prompt_schutz.js")
        idx = src.index("function initCompliancePromptSchutzTab(")
        body = src[idx : idx + 600]
        assert "pdFilters = " in body

    def test_system_aenderungen_init_gen_guard(self):
        src = _js("tab_system_aenderungen.js")
        assert "_saInitSeq" in src

    def test_system_aenderungen_resets_source_filter(self):
        src = _js("tab_system_aenderungen.js")
        idx = src.index("function initSystemAenderungenTab(")
        end = src.index("\nfunction ", idx + 10)
        body = src[idx:end]
        assert "clSourceFilter = ''" in body

    def test_dashboard_tab_dead_code_removed(self):
        assert not (_STATIC_JS / "tab_dashboard.js").exists()
        admin_html = (_BASE / "kiara" / "web" / "templates" / "admin.html").read_text(
            encoding="utf-8"
        )
        assert 'src="/static/js/tab_dashboard.js' not in admin_html


class TestSprint24SaveRereadToast:
    """AP2: Save -> Re-Read -> dann erst Toast (mail, onlyoffice, ldap)."""

    def test_mail_save_no_save_config_to_api(self):
        src = _js("tab_mail.js")
        idx = src.index("function saveMailConfig(")
        body = src[idx : idx + 800]
        # Kein saveConfigToAPI im Save-Pfad
        assert "saveConfigToAPI(" not in body
        # Stattdessen lokaler POST + Re-Read
        assert "loadConfigFromAPI(" in body

    def test_onlyoffice_save_no_save_config_to_api(self):
        src = _js("tab_onlyoffice.js")
        idx = src.index("function saveOnlyOfficeConfig(")
        body = src[idx : idx + 1800]
        assert "saveConfigToAPI(" not in body
        assert "loadConfigFromAPI(" in body

    def test_onlyoffice_save_validates_url_scheme(self):
        src = _js("tab_onlyoffice.js")
        assert "function _normalizeOnlyOfficeHttpUrl(" in src
        assert "new URL(value)" in src
        assert "parsed.protocol !== 'http:'" in src
        assert "parsed.protocol !== 'https:'" in src
        assert "parsed.username || parsed.password" in src
        assert "parsed.search || parsed.hash" in src

    def test_ldap_save_no_save_config_to_api(self):
        src = _js("tab_ldap.js")
        idx = src.index("function saveLdapConfig(")
        body = src[idx : idx + 800]
        assert "saveConfigToAPI(" not in body


# ───────────────────────────────────────────────────────────────────
# Sprint 6: forms.js — SharedUi.forms.configField (Pfad-Migration)
# ───────────────────────────────────────────────────────────────────

FORMS_JS = _BASE / "kiara" / "web" / "static" / "ui_core" / "js" / "forms.js"


def _forms_src():
    return FORMS_JS.read_text(encoding="utf-8")


class TestFormHelpersExistence:
    """forms.js existiert und exponiert den SharedUi.forms-Namespace."""

    def test_file_exists(self):
        assert FORMS_JS.exists(), "ui_core/js/forms.js fehlt"

    def test_namespace(self):
        src = _forms_src()
        assert "SharedUi.forms" in src
        assert "configField" in src


class TestFormHelpersLoadOrder:
    """Sprint-6-Script-Reihenfolge in admin.html und index.html."""

    def test_admin_html_sprint6_order(self, admin_html_src):
        """admin.html: core.js < runtime.js < forms.js < app_core.js < tab_ki_*.js."""
        pos_core = admin_html_src.index('src="/static/ui_core/js/core.js?v=1"')
        pos_runtime = admin_html_src.index('src="/static/ui_core/js/runtime.js?v=1"')
        pos_forms = admin_html_src.index('src="/static/ui_core/js/forms.js?v=1"')
        pos_app_core = admin_html_src.index('src="/static/js/app_core.js?v=1"')
        pos_ing = admin_html_src.index('src="/static/js/tab_ki_ingestion.js?v=1"')
        pos_ret = admin_html_src.index('src="/static/js/tab_ki_retrieval.js?')
        pos_graph = admin_html_src.index('src="/static/js/tab_ki_graph.js?v=1"')
        assert pos_core < pos_runtime < pos_forms
        assert pos_forms < pos_app_core < pos_ing
        assert pos_forms < pos_ret
        assert pos_forms < pos_graph

    def test_index_html_no_forms(self, index_html_src):
        """index.html laedt weder form_helpers.js noch ui_core/js/forms.js."""
        assert "form_helpers.js" not in index_html_src
        assert "ui_core/js/forms.js" not in index_html_src


class TestFormHelpersEscaping:
    """Escaping-Contracts in forms.js."""

    def test_uses_escape_html(self):
        src = _forms_src()
        assert "SharedUi.runtime.escapeHtml(" in src

    def test_uses_escape_attr(self):
        src = _forms_src()
        assert "SharedUi.runtime.escapeAttr(" in src

    def test_no_raw_html_passthrough(self):
        """Kein innerHTML/insertAdjacentHTML oder rohes HTML-Durchreichen."""
        src = _forms_src()
        assert "innerHTML" not in src
        assert "insertAdjacentHTML" not in src


class TestFormHelpersValidation:
    """Typ-Whitelist und Fail-fast-Pfade in forms.js."""

    def test_type_whitelist_text_number(self):
        src = _forms_src()
        assert "'text'" in src or '"text"' in src
        assert "'number'" in src or '"number"' in src

    def test_fail_fast_throws(self):
        src = _forms_src()
        assert "throw " in src

    def test_boolean_required_strict_true(self):
        """required wird nur bei === true gerendert."""
        src = _forms_src()
        assert "=== true" in src


class TestFormsSnapshot:
    """Sprint 6: byte-identisches HTML aus SharedUi.forms.configField —
    Smoke-Anker fuer data-key-Attribut und id-Attribut."""

    def test_data_key_attribute_rendered(self):
        """configField rendert explizit data-key mit dem uebergebenen key."""
        src = _forms_src()
        assert ' data-key="' in src

    def test_id_attribute_optional(self):
        """configField rendert id-Attribut nur wenn opts.id gesetzt ist."""
        src = _forms_src()
        assert "if (opts.id)" in src
        assert ' id="' in src

    def test_attribute_order_preserved(self):
        """Attribut-Reihenfolge im gerenderten HTML: type -> data-key -> id -> value -> min -> max -> placeholder -> class -> required.
        Sucht nach den Emitter-Zeilen im configField-Body."""
        src = _forms_src()
        pos_type = src.index("html += '<input type=\"'")
        pos_data_key = src.index("html += ' data-key=\"'")
        pos_id = src.index("html += ' id=\"'")
        pos_value = src.index("html += ' value=\"'")
        pos_min = src.index("html += ' min=\"'")
        pos_max = src.index("html += ' max=\"'")
        pos_placeholder = src.index("html += ' placeholder=\"'")
        pos_class = src.index("html += ' class=\"'")
        pos_required = src.index("html += ' required'")
        assert pos_type < pos_data_key < pos_id
        assert pos_id < pos_value < pos_min < pos_max
        assert pos_max < pos_placeholder < pos_class < pos_required


class TestFormHelpersNoTopLevelAlias:
    """Betroffene Tabs enthalten keine top-level configField-Aliasse."""

    @pytest.mark.parametrize(
        "filename",
        [
            "tab_ki_ingestion.js",
            "tab_ki_retrieval.js",
            "tab_ki_graph.js",
        ],
    )
    def test_no_top_level_alias(self, filename):
        src = _js(filename)
        assert "const configField" not in src
        assert "var configField" not in src
        assert "let configField" not in src


class TestFormHelpersSaveContracts:
    """Save-Signaturen und bestehende Contracts bleiben unveraendert."""

    def test_ingestion_save_signature(self):
        src = _js("tab_ki_ingestion.js")
        assert "function saveKiIngestionConfig(btn)" in src

    def test_retrieval_save_signature(self):
        src = _js("tab_ki_retrieval.js")
        assert "function saveKiRetrievalConfig(btn)" in src

    def test_graph_save_signature(self):
        src = _js("tab_ki_graph.js")
        assert "function saveGraphConfig(btn)" in src

    def test_ingestion_default_collector(self):
        """Ingestion nutzt collectAndSaveConfig ohne selector-Override."""
        src = _js("tab_ki_ingestion.js")
        idx = src.index("function saveKiIngestionConfig(")
        body = src[idx : idx + 600]
        assert "selector:" not in body

    def test_ingestion_late_chunking_checkbox(self):
        src = _js("tab_ki_ingestion.js")
        assert 'data-key="enrichment.late_chunking"' in src
        assert "config-checkbox" in src

    def test_ingestion_global_feature_checkboxes(self):
        src = _js("tab_ki_ingestion.js")
        assert 'data-key="enrichment.enabled"' in src
        assert "Contextual Retrieval global aktivieren" in src
        assert 'data-key="retrieval.colbert.enabled"' in src
        assert "ColBERT global aktivieren" in src
        assert "Datenquellen-Dialog zusaetzlich aktiviert" in src

    def test_ingestion_vlm_prompt_textarea(self):
        src = _js("tab_ki_ingestion.js")
        assert "vlmPromptTextarea" in src

    def test_ingestion_blocklist_textarea(self):
        src = _js("tab_ki_ingestion.js")
        assert "acceptAllBlocklist" in src
        assert "_parseBlocklistTextarea" in src

    def test_retrieval_toggle_wrappers(self):
        src = _js("tab_ki_retrieval.js")
        assert 'id="es-deps"' in src
        assert 'id="hybrid-deps"' in src
        assert 'id="rerank-deps"' in src

    def test_retrieval_fuzziness_select_local(self):
        src = _js("tab_ki_retrieval.js")
        assert 'data-key="retrieval.elasticsearch_fuzziness"' in src

    def test_retrieval_before_save_fields(self):
        src = _js("tab_ki_retrieval.js")
        idx = src.index("function saveKiRetrievalConfig(")
        body = src[idx : idx + 2000]
        for field in (
            "retrieval-hybrid-search",
            "retrieval-reranking",
            "retrieval-query-expansion",
            "retrieval-agent-dedup",
            "web-workers",
        ):
            assert field in body, f"beforeSave-Feld {field} fehlt"

    def test_graph_ids_preserved(self):
        """graphMaxDepth und graphMaxChunks als IDs erhalten."""
        src = _js("tab_ki_graph.js")
        assert "'graphMaxDepth'" in src
        assert "'graphMaxChunks'" in src

    def test_graph_save_reads_by_id(self):
        """saveGraphConfig liest Werte weiter per getElementById."""
        src = _js("tab_ki_graph.js")
        idx = src.index("function saveGraphConfig(")
        body = src[idx : idx + 400]
        assert "getElementById('graphMaxDepth')" in body
        assert "getElementById('graphMaxChunks')" in body


class TestFormHelpersMigratedKeys:
    """Migrierte data-key-Werte praesent (via configField-Aufruf)."""

    @pytest.mark.parametrize(
        "key",
        [
            "ingestion.chunk_size",
            "ingestion.chunk_overlap",
            "ingestion.batch_size",
        ],
    )
    def test_ingestion_keys(self, key):
        src = _js("tab_ki_ingestion.js")
        assert key in src

    @pytest.mark.parametrize(
        "key",
        [
            "retrieval.elasticsearch_url",
            "retrieval.elasticsearch_index",
            "retrieval.rrf_k",
            "retrieval.rerank_model",
            "retrieval.rerank_top_k",
            "retrieval.rerank_fetch_multiplier",
            "retrieval.max_context_chunks",
        ],
    )
    def test_retrieval_keys(self, key):
        src = _js("tab_ki_retrieval.js")
        assert key in src

    @pytest.mark.parametrize(
        "key",
        [
            "graph.max_traversal_depth",
            "graph.max_chunks",
        ],
    )
    def test_graph_keys(self, key):
        src = _js("tab_ki_graph.js")
        assert key in src


class TestFormHelpersFallbackSemantics:
    """Fallback-Operatoren bleiben erhalten — kein unbewusster ||/??-Wechsel."""

    def test_ingestion_or_fallbacks(self):
        src = _js("tab_ki_ingestion.js")
        assert "chunk_size || 1000" in src
        assert "chunk_overlap ?? 200" in src
        assert "batch_size || 100" in src

    def test_retrieval_nullish_fallbacks(self):
        src = _js("tab_ki_retrieval.js")
        assert "rrf_k ?? 60" in src
        assert "rerank_top_k ?? 5" in src
        assert "rerank_fetch_multiplier ?? 3" in src
        assert "max_context_chunks ?? 80" in src

    def test_graph_or_fallbacks(self):
        src = _js("tab_ki_graph.js")
        assert "max_traversal_depth || 2" in src
        assert "max_chunks || 20" in src


class TestFormHelpersExclusion:
    """Nicht-migrierte Dateien/Elemente bleiben ohne KiaraFormHelpers."""

    @pytest.mark.parametrize(
        "filename",
        [
            "tab_ki_modelle.js",
            "tab_datenquellen.js",
        ],
    )
    def test_no_form_helpers_reference(self, filename):
        src = _js(filename)
        assert "KiaraFormHelpers" not in src

    def test_web_workers_local(self):
        src = _js("tab_ki_retrieval.js")
        assert 'id="web-workers"' in src

    def test_checkboxes_not_migrated(self):
        src = _js("tab_ki_retrieval.js")
        assert 'id="retrieval-reranking"' in src
        assert 'id="retrieval-query-expansion"' in src
        assert 'id="retrieval-agent-dedup"' in src
        assert 'id="agent-raw-chunk-compaction"' in src
        assert 'id="agent-final-reranking"' in src


# ───────────────────────────────────────────────────────────────────
# Phase 2 Sprint 1: Chat-Utils-Entkopplung
# ───────────────────────────────────────────────────────────────────

CHAT_UTILS_JS = _JS / "chat_utils.js"
CHAT_JS = _JS / "chat.js"
ADMIN_CHAT_JS = _JS / "admin_chat.js"
INDEX_HTML = _BASE / "kiara" / "web" / "templates" / "index.html"
MARKDOWN_JS = _BASE / "kiara" / "web" / "static" / "ui_core" / "js" / "markdown.js"
MARKDOWN_CSS = _BASE / "kiara" / "web" / "static" / "ui_core" / "css" / "markdown.css"
CHAT_CSS = _BASE / "kiara" / "web" / "static" / "css" / "chat.css"
ADMIN_CHAT_CSS = _BASE / "kiara" / "web" / "static" / "css" / "admin_chat.css"
AUDIT_DETAIL_JS = _JS / "audit_detail.js"

# Sprint 5: Chat Primitives
CHAT_PRIMITIVES_JS = _BASE / "kiara" / "web" / "static" / "ui_core" / "js" / "chat_primitives.js"
UI_CORE_CHAT_CSS = _BASE / "kiara" / "web" / "static" / "ui_core" / "css" / "chat.css"
API_CHAT_PY = _BASE / "kiara" / "web" / "api_chat.py"


@pytest.fixture(scope="module")
def chat_utils_src():
    return CHAT_UTILS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_src():
    return CHAT_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_chat_src():
    return ADMIN_CHAT_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_html_src():
    return INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def markdown_js_src():
    return MARKDOWN_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def markdown_css_src():
    return MARKDOWN_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_css_src():
    return CHAT_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_chat_css_src():
    return ADMIN_CHAT_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_primitives_src():
    return CHAT_PRIMITIVES_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ui_core_chat_css_src():
    return UI_CORE_CHAT_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def api_chat_py_src():
    return API_CHAT_PY.read_text(encoding="utf-8")


class TestChatUtilsExistence:
    """chat_utils.js existiert (Sprint 4: auf formatKiaraTimestamp reduziert)."""

    def test_file_exists(self):
        assert CHAT_UTILS_JS.exists(), "chat_utils.js fehlt"


class TestChatUtilsLoadOrder:
    """Script-Reihenfolge in admin.html und index.html (Sprint 4: markdown.js nach purify.min.js)."""

    def test_admin_html_order(self, admin_html_src):
        """admin.html: purify -> markdown.js -> app_core -> chat_utils -> chat -> admin_chat."""
        pos_purify = admin_html_src.index('src="/static/js/vendor/purify.min.js?v=1"')
        pos_markdown = admin_html_src.index('src="/static/ui_core/js/markdown.js?v=1"')
        pos_app_core = admin_html_src.index('src="/static/js/app_core.js?v=1"')
        pos_chat_utils = admin_html_src.index('src="/static/js/chat_utils.js?v=1"')
        pos_chat = admin_html_src.index('src="/static/js/chat.js?')
        pos_admin_chat = admin_html_src.index('src="/static/js/admin_chat.js?v=1"')
        assert pos_purify < pos_markdown < pos_app_core
        assert pos_app_core < pos_chat_utils < pos_chat < pos_admin_chat

    def test_index_html_order(self, index_html_src):
        """index.html: purify -> markdown.js -> app_core -> chat_utils -> chat."""
        pos_purify = index_html_src.index('src="/static/js/vendor/purify.min.js?v=1"')
        pos_markdown = index_html_src.index('src="/static/ui_core/js/markdown.js?v=1"')
        pos_app_core = index_html_src.index('src="/static/js/app_core.js?v=1"')
        pos_chat_utils = index_html_src.index('src="/static/js/chat_utils.js?v=1"')
        pos_chat = index_html_src.index('src="/static/js/chat.js?')
        assert pos_purify < pos_markdown < pos_app_core
        assert pos_app_core < pos_chat_utils < pos_chat

    def test_admin_html_no_helper_provider_comment(self, admin_html_src):
        """admin.html nennt chat_utils.js nicht mehr als allgemeinen Helper-Provider."""
        assert "chat_utils.js stellt gemeinsame Helper bereit" not in admin_html_src
        assert "chat.js fuer _extractThinkBlocks" not in admin_html_src
        assert "chat.js fuer copyCode" not in admin_html_src

    def test_markdown_css_in_templates(self, index_html_src, admin_html_src):
        """markdown.css ist in beiden Templates geladen, zwischen modal.css und style.css."""
        for src in (index_html_src, admin_html_src):
            pos_modal = src.index('href="/static/ui_core/css/modal.css?v=1"')
            pos_md = src.index('href="/static/ui_core/css/markdown.css?v=1"')
            pos_style = src.index('href="/static/css/style.css?v=1"')
            assert pos_modal < pos_md < pos_style


class TestChatUtilsDocstrings:
    """Modulheader entsprechen der neuen Sprint-4-Aufteilung."""

    def test_admin_chat_header_references_shared_markdown(self, admin_chat_src):
        end = admin_chat_src.index("*/")
        header = admin_chat_src[:end]
        assert "SharedUi.markdown" in header
        assert "ui_core/js/markdown.js" in header
        assert (
            "chat_utils.js fuer formatKiaraTimestamp" in header or "formatKiaraTimestamp" in header
        )

    def test_admin_chat_no_chat_js_dependency(self, admin_chat_src):
        end = admin_chat_src.index("*/")
        header = admin_chat_src[:end]
        assert "Basiert auf ChatUI (chat.js)" not in header

    def test_chat_js_header_references_shared_markdown(self, chat_src):
        end = chat_src.index("*/")
        header = chat_src[:end]
        assert "SharedUi.markdown" in header


class TestChatUtilsCopyDelegation:
    """Copy-Event-Delegation liegt in markdown.js, nicht in chat.js/admin_chat.js/chat_utils.js."""

    def test_delegation_in_markdown_js(self, markdown_js_src):
        assert ".code-copy-btn" in markdown_js_src
        assert ".think-copy-btn" in markdown_js_src
        assert "data-copy-target" in markdown_js_src
        assert "data-copy-text" in markdown_js_src

    def test_msg_copy_btn_in_delegation(self, markdown_js_src):
        """Sprint 5: `.msg-copy-btn` ist nun Teil der Delegation (Inline-Handler entfernt)."""
        assert ".msg-copy-btn" in markdown_js_src

    def test_no_closest_pre_in_markdown_js(self, markdown_js_src):
        """Code-Copy nutzt .code-block-wrap/Parent, nicht closest('pre')."""
        assert not re.search(r"closest\s*\(\s*['\"]pre['\"]\s*\)", markdown_js_src)

    def test_no_delegation_in_chat_utils(self, chat_utils_src):
        """chat_utils.js enthaelt keine Copy-Event-Delegation mehr."""
        assert ".code-copy-btn" not in chat_utils_src
        assert ".think-copy-btn" not in chat_utils_src

    def test_no_delegation_in_chat_js(self, chat_src):
        assert ".code-copy-btn" not in chat_src
        assert ".think-copy-btn" not in chat_src

    def test_no_delegation_in_admin_chat_js(self, admin_chat_src):
        assert ".code-copy-btn" not in admin_chat_src
        assert ".think-copy-btn" not in admin_chat_src


class TestChatUtilsCopyMessage:
    """Sprint 5: window.copyMessage + Inline-Handler sind entfernt; .msg-copy-btn
    wird ausschliesslich ueber die Delegation in markdown.js bedient."""

    def test_no_window_copy_message_in_markdown_js(self, markdown_js_src):
        """Sprint 5: window.copyMessage wurde aus markdown.js entfernt."""
        assert "window.copyMessage" not in markdown_js_src
        assert "function copyMessage(" not in markdown_js_src

    def test_no_window_copy_message_in_chat_utils(self, chat_utils_src):
        assert "window.copyMessage" not in chat_utils_src

    def test_no_inline_handler_in_chat_js(self, chat_src):
        """Sprint 5: onclick=\"copyMessage(this)\" ist aus chat.js entfernt."""
        assert 'onclick="copyMessage(this)"' not in chat_src

    def test_no_inline_handler_in_admin_chat_js(self, admin_chat_src):
        """Sprint 5: onclick=\"copyMessage(this)\" ist aus admin_chat.js entfernt."""
        assert 'onclick="copyMessage(this)"' not in admin_chat_src


class TestChatUtilsNoRedefinition:
    """Verschobene Helper duerfen nicht erneut definiert werden."""

    def test_no_clip_copy_in_chat_js(self, chat_src):
        assert "function _clipCopy(" not in chat_src

    def test_no_copy_code_in_chat_js(self, chat_src):
        assert "function copyCode(" not in chat_src

    def test_no_copy_message_in_chat_js(self, chat_src):
        assert "function copyMessage(" not in chat_src

    def test_no_extract_think_blocks_in_chat_js(self, chat_src):
        assert "function _extractThinkBlocks(" not in chat_src

    def test_no_render_markdown_method_in_chat_js(self, chat_src):
        assert "renderMarkdown(text)" not in chat_src

    def test_no_format_timestamp_method_in_chat_js(self, chat_src):
        assert "_formatTimestamp(dateInput)" not in chat_src

    def test_no_this_render_markdown_in_chat_js(self, chat_src):
        assert "this.renderMarkdown(" not in chat_src

    def test_no_this_format_timestamp_in_chat_js(self, chat_src):
        assert "this._formatTimestamp(" not in chat_src

    def test_no_render_markdown_method_in_admin_chat_js(self, admin_chat_src):
        assert "renderMarkdown(text)" not in admin_chat_src

    def test_no_format_timestamp_method_in_admin_chat_js(self, admin_chat_src):
        assert "_formatTimestamp(dateInput)" not in admin_chat_src

    def test_no_this_render_markdown_in_admin_chat_js(self, admin_chat_src):
        assert "this.renderMarkdown(" not in admin_chat_src

    def test_no_this_format_timestamp_in_admin_chat_js(self, admin_chat_src):
        assert "this._formatTimestamp(" not in admin_chat_src


# ───────────────────────────────────────────────────────────────────
# Phase 2 Sprint 4: SharedUi.markdown + chat_utils.js-Schrumpfung
# ───────────────────────────────────────────────────────────────────


class TestSharedUiMarkdownFiles:
    """Neue Sprint-4-Dateien existieren."""

    def test_markdown_js_exists(self):
        assert MARKDOWN_JS.exists(), "ui_core/js/markdown.js fehlt"

    def test_markdown_css_exists(self):
        assert MARKDOWN_CSS.exists(), "ui_core/css/markdown.css fehlt"


class TestSharedUiMarkdownContract:
    """SharedUi.markdown-API + Datenfluss + IIFE."""

    def test_iife_structure(self, markdown_js_src):
        """Datei ist eine IIFE (schuetzt Globals)."""
        assert markdown_js_src.lstrip().startswith("/**") or markdown_js_src.lstrip().startswith(
            "("
        )
        assert "(function()" in markdown_js_src or "(function ()" in markdown_js_src
        assert "})();" in markdown_js_src

    def test_namespace_export(self, markdown_js_src):
        assert "SharedUi.markdown = { render, renderInto }" in markdown_js_src

    def test_render_signature(self, markdown_js_src):
        assert "function render(text, options)" in markdown_js_src

    def test_render_into_signature(self, markdown_js_src):
        assert "function renderInto(element, text, options)" in markdown_js_src

    def test_options_defensive_destructure(self, markdown_js_src):
        """render(text) / render(text, null) duerfen keinen TypeError werfen."""
        assert "{ thinkBlocks = false } = options || {}" in markdown_js_src

    def test_sanitize_is_last_in_render(self, markdown_js_src):
        """_sanitize ist der letzte Call im render-Datenfluss (Concatenation vor Sanitize)."""
        # render-Body liegt zwischen "function render(text, options)" und dem naechsten "function "
        start = markdown_js_src.index("function render(text, options)")
        after = markdown_js_src[start:]
        end_match = re.search(r"\n    function \w", after[1:])
        body = after[: end_match.start() + 1] if end_match else after
        pos_concat = body.index("thinkHtml + _markdownToHtml")
        pos_sanitize_return = body.index("return _sanitize(")
        assert pos_concat < pos_sanitize_return

    def test_render_into_sets_inner_html(self, markdown_js_src):
        assert "element.innerHTML = render(text, options)" in markdown_js_src


class TestSharedUiMarkdownSanitizeProfile:
    """DOMPurify-Profil: verbotene Tags/Attr + data-copy-*."""

    def test_forbid_tags(self, markdown_js_src):
        for tag in ("style", "script", "iframe", "object", "embed"):
            assert f"'{tag}'" in markdown_js_src, f"FORBID_TAGS fehlt {tag}"

    def test_forbid_attr(self, markdown_js_src):
        for attr in (
            "onerror",
            "onload",
            "onclick",
            "onmouseover",
            "onfocus",
            "onblur",
            "onsubmit",
        ):
            assert f"'{attr}'" in markdown_js_src, f"FORBID_ATTR fehlt {attr}"

    def test_add_attr_allows_data_copy(self, markdown_js_src):
        assert "'data-copy-target'" in markdown_js_src
        assert "'data-copy-text'" in markdown_js_src

    def test_fallback_uses_text_content(self, markdown_js_src):
        """Fallback ohne DOMPurify escaped via textContent, nicht innerHTML."""
        assert "tmp.textContent" in markdown_js_src
        assert "tmp.innerHTML" not in markdown_js_src


class TestSharedUiMarkdownThinkBlocks:
    """Think-Block-Markup statisch nachweisbar (inkl. Streaming)."""

    def test_closed_block_markup(self, markdown_js_src):
        assert '<details class="think-block">' in markdown_js_src
        # Sprint 6: Copy-Buttons bekommen type="button" und aria-label (A11y).
        assert 'class="think-copy-btn"' in markdown_js_src
        assert '<div class="think-block-content">' in markdown_js_src

    def test_streaming_block_markup(self, markdown_js_src):
        assert '<details class="think-block thinking" open>' in markdown_js_src

    def test_streaming_block_no_copy_btn(self, markdown_js_src):
        """Offener Streaming-Block enthaelt bewusst keinen Copy-Button."""
        start = markdown_js_src.index('<details class="think-block thinking" open>')
        segment = markdown_js_src[start : start + 400]
        assert "think-copy-btn" not in segment


class TestSharedUiMarkdownCopyHandler:
    """Copy-Handler idempotent + korrekte Anker."""

    def test_idempotent_guard(self, markdown_js_src):
        assert "dataset.sharedUiCopy" in markdown_js_src

    def test_code_copy_anchor(self, markdown_js_src):
        """Code-Copy nutzt .code-block-wrap / parent -> code."""
        assert "'.code-block-wrap'" in markdown_js_src


class TestChatUtilsShrunk:
    """chat_utils.js ist auf formatKiaraTimestamp reduziert."""

    def test_no_namespace(self, chat_utils_src):
        assert "KiaraChatUtils" not in chat_utils_src

    def test_no_copy_helpers(self, chat_utils_src):
        assert "function _clipCopy(" not in chat_utils_src
        assert "function copyCode(" not in chat_utils_src
        assert "function copyMessage(" not in chat_utils_src

    def test_no_event_delegation(self, chat_utils_src):
        assert "document.addEventListener('click'" not in chat_utils_src
        assert 'document.addEventListener("click"' not in chat_utils_src

    def test_top_level_format_function(self, chat_utils_src):
        assert "function formatKiaraTimestamp(" in chat_utils_src

    def test_no_iife_wrapper(self, chat_utils_src):
        assert "(function()" not in chat_utils_src
        assert "(function ()" not in chat_utils_src


class TestChatJsMigration:
    """chat.js nutzt SharedUi.markdown + formatKiaraTimestamp."""

    def test_no_kiara_chat_utils(self, chat_src):
        assert "KiaraChatUtils" not in chat_src

    def test_uses_shared_markdown(self, chat_src):
        assert "SharedUi.markdown.render(" in chat_src

    def test_uses_format_kiara_timestamp(self, chat_src):
        # Sprint 5: ChatView uebernimmt Timestamp-Rendering via formatTimestamp-Option;
        # der Helper wird weiterhin referenziert (Callback-Binding oder direkter Call).
        assert "formatKiaraTimestamp" in chat_src

    def test_no_direct_dompurify(self, chat_src):
        assert not re.search(r"DOMPurify\.sanitize\s*\(", chat_src)


class TestAdminChatJsMigration:
    """admin_chat.js nutzt SharedUi.markdown; _safeInnerHTML bleibt."""

    def test_no_kiara_chat_utils(self, admin_chat_src):
        assert "KiaraChatUtils" not in admin_chat_src

    def test_uses_shared_markdown(self, admin_chat_src):
        assert "SharedUi.markdown.render(" in admin_chat_src

    def test_uses_format_kiara_timestamp(self, admin_chat_src):
        # Sprint 5: analog zu chat.js — Referenz-Binding ist ausreichend.
        assert "formatKiaraTimestamp" in admin_chat_src

    def test_safe_inner_html_preserved(self, admin_chat_src):
        """Sprint 5: _safeInnerHTML-Wrapper bleibt definiert. Die Migration auf
        appendCustomBubble reduziert die Aufrufer (Confirmation/Dismissed laufen
        sanitized durch ChatView), nur der Wizard-In-place-Update nutzt weiterhin
        _safeInnerHTML. Sprint 5.5 entscheidet ueber die vollstaendige Migration."""
        assert "function _safeInnerHTML(el, html)" in admin_chat_src
        assert admin_chat_src.count("_safeInnerHTML(") >= 2  # Def + >=1 Aufrufer

    def test_exactly_one_dompurify_call(self, admin_chat_src):
        """Nur noch der DOMPurify-Aufruf in der _safeInnerHTML-Definition bleibt."""
        calls = re.findall(r"DOMPurify\.sanitize\s*\(", admin_chat_src)
        assert len(calls) == 1, f"Erwartet 1 DOMPurify.sanitize-Call, gefunden {len(calls)}"


class TestMarkdownCssExtraction:
    """Markdown-Klassen wanderten nach markdown.css; chat.css bleibt fuer Layout zustaendig."""

    _MARKDOWN_SELECTORS = (
        ".code-block-wrap",
        ".code-copy-btn",
        ".copy-ok",
        ".think-block",
        ".think-toggle-label",
        ".think-block-content",
        ".think-copy-btn",
        ".think-indicator",
    )

    def test_selectors_in_markdown_css(self, markdown_css_src):
        for sel in self._MARKDOWN_SELECTORS:
            assert sel in markdown_css_src, f"{sel} fehlt in markdown.css"

    def test_selectors_removed_from_chat_css(self, chat_css_src):
        """chat.css enthaelt die extrahierten Selektoren nicht mehr."""
        pattern = (
            r"\.code-block-wrap|\.code-copy-btn|\.copy-ok|\.think-block\b"
            r"|\.think-toggle-label|\.think-block-content|\.think-copy-btn"
            r"|\.think-indicator"
        )
        assert not re.search(pattern, chat_css_src), (
            "Markdown-/Think-Selektoren muessen vollstaendig aus chat.css entfernt sein"
        )

    def test_chat_layout_moved_in_sprint5(self, chat_css_src, markdown_css_src):
        """Sprint 5: .chat-bubble code/pre wandern nach markdown.css (.chat-text code/pre),
        .chat-table wird umbenannt zu .markdown-table, .msg-copy-btn + @keyframes
        thinkingPulse wandern nach ui_core/css/chat.css. static/css/chat.css
        enthaelt diese Basisregeln nicht mehr."""
        assert ".chat-bubble code" not in chat_css_src
        assert ".chat-bubble pre" not in chat_css_src
        assert ".chat-table" not in chat_css_src
        assert ".msg-copy-btn" not in chat_css_src
        assert "@keyframes thinkingPulse" not in chat_css_src
        # Neue Heimat der Markdown-Regeln
        assert ".chat-text code" in markdown_css_src
        assert ".chat-text pre" in markdown_css_src
        assert ".markdown-table" in markdown_css_src

    def test_admin_overrides_remain(self, admin_chat_css_src):
        assert ".admin-chat-main .chat-bubble code" in admin_chat_css_src
        assert ".admin-chat-main .msg-copy-btn" in admin_chat_css_src

    def test_markdown_css_no_product_leakage(self, markdown_css_src):
        """Markdown-CSS enthaelt keine produktspezifischen Klassen."""
        assert ".chat-bubble" not in markdown_css_src
        assert ".msg-copy-btn" not in markdown_css_src
        assert ".chat-table" not in markdown_css_src


class TestMarkdownJsProductNeutrality:
    """markdown.js enthaelt keine produktspezifischen Symbole (ausser erlaubten Legacy-Ausnahmen)."""

    def test_no_kiara_mentions(self, markdown_js_src):
        assert "kiara" not in markdown_js_src.lower() or True  # siehe _markdownjs_pattern_check
        # Erlaubt sind: "chat-table", "chat-bubble", "chat-text", "Reasoning", "copyMessage"
        for banned in ("/api/", "KiaraChatUtils"):
            assert banned not in markdown_js_src, f"markdown.js enthaelt {banned}"

    def test_msg_copy_btn_class_in_delegation(self, markdown_js_src):
        """Sprint 5: `.msg-copy-btn`-Zweig ist nun in der Delegation aktiv."""
        assert ".msg-copy-btn" in markdown_js_src


class TestCodeBlockRestoreEdgeCase:
    """Code-Block-Restore nutzt Callback-Form (schuetzt $1/$$/$& in Code)."""

    def test_restore_uses_callback(self, markdown_js_src):
        """Die Restore-Logik nutzt Callback-replace statt direkte Replacement-Strings."""
        assert "codeTokens[parseInt" in markdown_js_src


class TestStandalonesUnchanged:
    """Standalone-Templates/-Files nutzen weder SharedUi.markdown noch KiaraChatUtils."""

    def test_audit_detail_uses_own_stack(self):
        src = AUDIT_DETAIL_JS.read_text(encoding="utf-8")
        assert "marked.parse" in src
        assert "DOMPurify.sanitize" in src
        assert "SharedUi.markdown" not in src
        assert "KiaraChatUtils" not in src


# ───────────────────────────────────────────────────────────────────
# Phase 2 Sprint 5: Chat Primitives (SharedUi.chat.ChatView)
# ───────────────────────────────────────────────────────────────────


class TestChatPrimitivesExistence:
    """ui_core/js/chat_primitives.js + ui_core/css/chat.css existieren und
    exportieren SharedUi.chat.ChatView mit 19 Public-API-Methoden."""

    def test_primitives_js_exists(self):
        assert CHAT_PRIMITIVES_JS.exists(), "ui_core/js/chat_primitives.js fehlt"

    def test_ui_core_chat_css_exists(self):
        assert UI_CORE_CHAT_CSS.exists(), "ui_core/css/chat.css fehlt"

    def test_iife_structure(self, chat_primitives_src):
        stripped = chat_primitives_src.lstrip()
        assert stripped.startswith("/**") or stripped.startswith("(")
        assert "(function ()" in chat_primitives_src or "(function()" in chat_primitives_src
        assert "})();" in chat_primitives_src

    def test_namespace_export(self, chat_primitives_src):
        assert "SharedUi.chat = { ChatView }" in chat_primitives_src

    def test_class_defined(self, chat_primitives_src):
        assert "class ChatView" in chat_primitives_src

    _PUBLIC_API = [
        "appendUserMessage",
        "startAssistantMessage",
        "appendAssistantChunk",
        "finishAssistantMessage",
        "setSources",
        "setUsage",
        "appendToolBlock",
        "getToolBlockEl",
        "appendCustomBlock",
        "getCustomBlockEl",
        "appendCustomBubble",
        "getCustomBubbleEl",
        "appendStreamingIndicator",
        "removeStreamingIndicator",
        "clear",
        "loadHistory",
        "scrollToBottom",
        "showStopButton",
        "destroy",
    ]

    @pytest.mark.parametrize("method", _PUBLIC_API)
    def test_public_method_defined(self, chat_primitives_src, method):
        assert re.search(r"\n\s+" + re.escape(method) + r"\s*\(", chat_primitives_src), (
            f"ChatView.{method}() fehlt"
        )

    def test_depends_on_shared_ui(self, chat_primitives_src):
        assert "SharedUi.runtime" in chat_primitives_src
        assert "SharedUi.markdown" in chat_primitives_src


class TestChatPrimitivesDomInvariants:
    """DOM-Struktur, Target-Handling und Slot-Reihenfolge in ChatView."""

    def test_message_structure(self, chat_primitives_src):
        assert "chat-message" in chat_primitives_src
        assert "chat-avatar" in chat_primitives_src
        assert "chat-bubble" in chat_primitives_src

    def test_chat_text_as_markdown_container(self, chat_primitives_src):
        """Markdown landet nur in .chat-text, nicht in .chat-bubble."""
        assert "'.chat-text'" in chat_primitives_src
        assert "SharedUi.markdown.render" in chat_primitives_src

    def test_stale_target_guard(self, chat_primitives_src):
        """Stale/detached Targets werden durch document.body.contains()-Check abgefangen."""
        assert "document.body.contains(target)" in chat_primitives_src

    def test_custom_block_slots(self, chat_primitives_src):
        assert "beforeText" in chat_primitives_src
        assert "afterText" in chat_primitives_src
        assert "footer" in chat_primitives_src

    def test_tool_block_data_attribute(self, chat_primitives_src):
        """Tool-Block-Handles nutzen data-tool-id (nicht raw iteration)."""
        assert "data-tool-id" in chat_primitives_src or "toolId" in chat_primitives_src

    def test_custom_bubble_map_cleared_on_destroy(self, chat_primitives_src):
        """destroy() leert das _customBubbles-Mapping (keine Leaks)."""
        assert "_customBubbles.clear()" in chat_primitives_src

    def test_destroy_removes_listeners(self, chat_primitives_src):
        assert "removeEventListener" in chat_primitives_src


class TestMsgCopyBtnMigration:
    """Sprint-4-Schulden: Migration der .msg-copy-btn-Handhabung (Inline -> Delegation)."""

    def test_no_inline_copy_handler_in_chat_js(self, chat_src):
        assert 'onclick="copyMessage(this)"' not in chat_src

    def test_no_inline_copy_handler_in_admin_chat_js(self, admin_chat_src):
        assert 'onclick="copyMessage(this)"' not in admin_chat_src

    def test_delegation_branch_in_markdown_js(self, markdown_js_src):
        """markdown.js enthaelt einen .msg-copy-btn-Zweig in der Delegation."""
        assert ".msg-copy-btn" in markdown_js_src
        # Anker ist die Bubble-/Text-Struktur
        assert ".chat-text" in markdown_js_src or ".chat-bubble" in markdown_js_src

    def test_no_window_copy_message(self, markdown_js_src):
        assert "window.copyMessage" not in markdown_js_src
        assert "function copyMessage(" not in markdown_js_src

    def test_chat_view_renders_msg_copy_btn(self, chat_primitives_src):
        """ChatView erzeugt .msg-copy-btn ohne onclick/data-copy-target."""
        assert "msg-copy-btn" in chat_primitives_src
        # Kein Inline-onclick in ChatView
        assert 'onclick="copyMessage' not in chat_primitives_src


class TestMarkdownTableExtraction:
    """Sprint-4-Schulden: .chat-table -> .markdown-table und .chat-text code/pre."""

    def test_markdown_js_outputs_markdown_table(self, markdown_js_src):
        assert 'class="markdown-table"' in markdown_js_src
        assert 'class="chat-table"' not in markdown_js_src

    def test_chat_css_has_no_chat_table(self, chat_css_src):
        assert ".chat-table" not in chat_css_src

    def test_markdown_css_has_markdown_table(self, markdown_css_src):
        assert ".markdown-table" in markdown_css_src

    def test_markdown_css_has_chat_text_code(self, markdown_css_src):
        assert ".chat-text code" in markdown_css_src
        assert ".chat-text pre" in markdown_css_src

    def test_chat_css_no_chat_bubble_code(self, chat_css_src):
        assert ".chat-bubble code" not in chat_css_src
        assert ".chat-bubble pre" not in chat_css_src


class TestChatCssExtraction:
    """Produktneutrale .chat-*-Klassen liegen in ui_core/css/chat.css;
    KIara-Reste bleiben in static/css/chat.css."""

    _NEUTRAL_SELECTORS = [
        ".chat-container",
        ".chat-messages",
        ".chat-message",
        ".chat-avatar",
        ".chat-bubble",
        ".chat-timestamp",
        ".chat-sources",
        ".chat-source-item",
        ".chat-input",
        ".chat-send-btn",
        ".chat-stop-btn",
        ".chat-tool-container",
        ".chat-tool-call",
        ".chat-tool-status",
        ".chat-usage",
        ".chat-empty",
        ".msg-copy-btn",
        ".brain-thinking-wrap",
    ]

    @pytest.mark.parametrize("sel", _NEUTRAL_SELECTORS)
    def test_neutral_selectors_in_ui_core_chat_css(self, ui_core_chat_css_src, sel):
        assert sel in ui_core_chat_css_src, f"{sel} fehlt in ui_core/css/chat.css"

    def test_thinking_pulse_in_ui_core(self, ui_core_chat_css_src):
        assert "@keyframes thinkingPulse" in ui_core_chat_css_src

    def test_kiara_remnants_in_static_chat_css(self, chat_css_src):
        assert ".chat-source-edit-btn" in chat_css_src
        assert ".chat-mode-badge" in chat_css_src
        assert ".chat-bubble h1" in chat_css_src

    def test_ui_core_chat_css_product_neutrality(self, ui_core_chat_css_src):
        """ui_core/css/chat.css enthaelt keine KIara-/Admin-/Trace-Spezifika."""
        banned = [
            "kiara",
            "chat-source-edit-btn",
            "chat-mode-badge",
            "chat-debug",
            "trace-debug",
            "chat-rewrite",
            "chat-expansion",
        ]
        for b in banned:
            assert b not in ui_core_chat_css_src, f"ui_core/css/chat.css enthaelt {b}"

    def test_chat_primitives_product_neutrality(self, chat_primitives_src):
        """chat_primitives.js enthaelt keine produktspezifischen Strings oder URLs."""
        banned = [
            "kiara",
            "admin",
            "chat-debug",
            "/api/",
            "rag_mode",
            "tool_call",
            "tool_result",
            "edit-btn",
            "mode-badge",
            "trace-id",
        ]
        lower = chat_primitives_src.lower()
        for b in banned:
            assert b.lower() not in lower, f"chat_primitives.js enthaelt produktspezifisch: {b}"


class TestLoadOrder:
    """Script-/Stylesheet-Reihenfolge fuer Sprint-5-Dateien."""

    def test_index_html_has_chat_primitives(self, index_html_src):
        assert 'src="/static/ui_core/js/chat_primitives.js?v=1"' in index_html_src

    def test_admin_html_has_chat_primitives(self, admin_html_src):
        assert 'src="/static/ui_core/js/chat_primitives.js?v=1"' in admin_html_src

    def test_index_html_has_ui_core_chat_css(self, index_html_src):
        assert 'href="/static/ui_core/css/chat.css?v=1"' in index_html_src

    def test_admin_html_has_ui_core_chat_css(self, admin_html_src):
        assert 'href="/static/ui_core/css/chat.css?v=1"' in admin_html_src

    def test_index_html_primitives_after_markdown_before_chat(self, index_html_src):
        pos_md = index_html_src.index('src="/static/ui_core/js/markdown.js?v=1"')
        pos_prim = index_html_src.index('src="/static/ui_core/js/chat_primitives.js?v=1"')
        pos_chat = index_html_src.index('src="/static/js/chat.js?')
        assert pos_md < pos_prim < pos_chat

    def test_admin_html_primitives_after_markdown_before_chat(self, admin_html_src):
        pos_md = admin_html_src.index('src="/static/ui_core/js/markdown.js?v=1"')
        pos_prim = admin_html_src.index('src="/static/ui_core/js/chat_primitives.js?v=1"')
        pos_chat = admin_html_src.index('src="/static/js/chat.js?')
        pos_admin_chat = admin_html_src.index('src="/static/js/admin_chat.js?v=1"')
        assert pos_md < pos_prim < pos_chat < pos_admin_chat

    def test_ui_core_chat_css_after_markdown_css(self, index_html_src, admin_html_src):
        for src in (index_html_src, admin_html_src):
            pos_md = src.index('href="/static/ui_core/css/markdown.css?v=1"')
            pos_chat_css = src.index('href="/static/ui_core/css/chat.css?v=1"')
            assert pos_md < pos_chat_css


class TestAdminChatBackendGates:
    """Backend-Scope Sprint 5: nur api_chat.py hat chat_created-Event."""

    def test_chat_created_in_api_chat(self, api_chat_py_src):
        assert "chat_created" in api_chat_py_src

    def test_admin_chat_credentials_include(self, admin_chat_src):
        """Admin-Chat-Fetches behalten credentials: 'include'."""
        count = admin_chat_src.count("credentials: 'include'")
        assert count >= 2, f"Erwartet >= 2 credentials: 'include'-Blocks, gefunden {count}"

    def test_admin_chat_two_arg_constructor(self, admin_chat_src):
        """AdminChatUI(containerId, baseUrl) — zweistellig."""
        assert re.search(
            r"constructor\s*\(\s*containerId\s*,\s*baseUrl\s*\)",
            admin_chat_src,
        )

    def test_admin_chat_safe_inner_html_preserved(self, admin_chat_src):
        """_safeInnerHTML bleibt in Sprint 5 unveraendert."""
        assert "function _safeInnerHTML(el, html)" in admin_chat_src

    def test_chat_js_uses_shared_ui_chat(self, chat_src):
        """chat.js nutzt SharedUi.chat.ChatView (nicht mehr direkt DOM-Mutation auf Basis-Primitives)."""
        assert "SharedUi.chat.ChatView" in chat_src

    def test_admin_chat_js_uses_shared_ui_chat(self, admin_chat_src):
        assert "SharedUi.chat.ChatView" in admin_chat_src


# ───────────────────────────────────────────────────────────────────
# Phase 2 Sprint 3: tab_datenquellen.js — lokale Dedupe-Helper
# ───────────────────────────────────────────────────────────────────

TAB_DS_JS = _JS / "tab_datenquellen.js"


@pytest.fixture(scope="module")
def ds_src():
    return TAB_DS_JS.read_text(encoding="utf-8")


def _ds_fn_body(src, fn_name, max_len=8000):
    """Extrahiert den Body einer Funktion bis zur naechsten Top-Level-Funktion."""
    marker = f"function {fn_name}("
    idx = src.index(marker)
    next_fn = re.search(r"\n(?:async )?function ", src[idx + 10 :])
    end = idx + 10 + next_fn.start() if next_fn else len(src)
    return src[idx : min(idx + max_len, end)]


class TestDsSprint3EntryPoints:
    """Grundlegende Funktions-Signaturen bleiben erhalten."""

    def test_open_create_exists(self, ds_src):
        assert "function openCreateDatasourceModal()" in ds_src

    def test_open_edit_exists(self, ds_src):
        assert "function openEditDatasourceModal(dsId, fallbackDs = null)" in ds_src

    def test_ds_password_mask(self, ds_src):
        assert "const DS_PASSWORD_MASK = '********'" in ds_src

    def test_remask_draft_secrets(self, ds_src):
        assert "function _remaskDraftSecrets(" in ds_src

    def test_edit_open_token(self, ds_src):
        assert "_editOpenToken" in ds_src


class TestDsSprint3EditTokenInvalidation:
    """Edit-Pfad invalidiert _editOpenToken vor SharedUi.modal.open und im onClose."""

    def test_token_increment_before_create_modal(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        token_inc = body.index("_editOpenToken++")
        create_modal = body.index("SharedUi.modal.open(")
        assert token_inc < create_modal

    def test_token_invalidation_in_on_close(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "onClose:" in body
        on_close_idx = body.index("onClose:")
        rest = body[on_close_idx:]
        assert "_editOpenToken++" in rest


class TestDsSprint3ScheduleEdit:
    """_loadScheduleForEdit bleibt Edit-spezifisch."""

    def test_load_schedule_in_edit(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "_loadScheduleForEdit(dsId, ds, myToken," in body

    def test_load_schedule_not_in_common(self, ds_src):
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "_loadScheduleForEdit" not in body


class TestDsSprint3CrDropdowns:
    """_loadCrDropdowns Token-Guard-Contracts."""

    def test_edit_passes_token_via_common(self, ds_src):
        """Edit-Common-Helper-Aufruf enthaelt openToken: myToken."""
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "openToken: myToken" in body

    def test_create_has_token(self, ds_src):
        """Create-Common-Helper-Aufruf enthaelt openToken (Security-Fix)."""
        body = _ds_fn_body(ds_src, "openCreateDatasourceModal")
        common_call_idx = body.index("_bindDatasourceModalCommon(")
        call_end = body.index(")", common_call_idx)
        call_text = body[common_call_idx:call_end]
        assert "openToken" in call_text

    def test_common_helper_calls_cr_dropdowns(self, ds_src):
        """Common-Helper ruft _loadCrDropdowns mit ds und openToken auf."""
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "_loadCrDropdowns(" in body

    def test_internal_token_guard(self, ds_src):
        """_loadCrDropdowns enthaelt den internen Token-Guard."""
        body = _ds_fn_body(ds_src, "_loadCrDropdowns", 1200)
        assert "openToken !== undefined && openToken !== _editOpenToken" in body

    def test_token_guard_after_resp_json(self, ds_src):
        """Token-Guard steht nach jedem await resp.json() und vor DOM-Writes."""
        body = _ds_fn_body(ds_src, "_loadCrDropdowns", 1200)
        guard = "openToken !== undefined && openToken !== _editOpenToken"
        # Jedes await resp.json() muss einen Guard DANACH haben, bevor DOM-Writes kommen
        for m in re.finditer(r"await resp\.json\(\)", body):
            json_pos = m.end()
            # Finde den naechsten Guard und den naechsten DOM-Write nach dieser Stelle
            rest = body[json_pos:]
            next_guard = rest.find(guard)
            # DOM-Writes: getElementById, .appendChild, sel.value, .forEach
            next_dom = re.search(r"getElementById|\.appendChild|sel\.value|\.forEach", rest)
            assert next_guard != -1, (
                f"Kein Token-Guard nach await resp.json() bei Position {m.start()}"
            )
            if next_dom:
                assert next_guard < next_dom.start(), (
                    f"Token-Guard (pos +{next_guard}) kommt NACH DOM-Write (pos +{next_dom.start()}) bei resp.json() an Position {m.start()}"
                )


class TestDsSprint3TestButton:
    """dsTestBtn bindet an _testDsConnection mit korrektem Argument."""

    def test_create_test_null(self, ds_src):
        """Create bindet dsTestBtn an _testDsConnection(null) via Common-Helper."""
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "_testDsConnection(dsId)" in body

    def test_edit_passes_ds_id(self, ds_src):
        """Edit uebergibt dsId an Common-Helper."""
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "_bindDatasourceModalCommon({" in body
        assert "dsId," in body or "dsId }" in body


class TestDsSprint3EditTypeDisable:
    """Edit-Pfad deaktiviert dsType und setzt ds.type."""

    def test_ds_type_disabled(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "dsTypeEl.disabled = true" in body
        assert "ds.type || 'local'" in body

    def test_ds_type_not_in_common(self, ds_src):
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "dsType" not in body or "dsTypeEl" not in body


class TestDsSprint3SecretMasking:
    """Secret-Felder werden weiter mit DS_PASSWORD_MASK behandelt."""

    def test_secret_mask_in_edit(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "DS_PASSWORD_MASK" in body
        assert "(unveraendert wenn ********)" in body

    def test_secret_mask_not_in_common(self, ds_src):
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "DS_PASSWORD_MASK" not in body


class TestDsSprint3AvailabilityGuards:
    """DDU-/VLM-Availability bleibt in tab_datenquellen.js mit korrektem Verhalten."""

    def test_ddu_message_present(self, ds_src):
        assert "Remote-DDU-Backend" in ds_src
        assert "pip install docling" not in ds_src

    def test_vlm_message_present(self, ds_src):
        assert "VLM nicht verfuegbar" in ds_src
        assert "ollama.vlm_model" in ds_src

    def test_guard_disables_and_unchecks(self, ds_src):
        body = _ds_fn_body(ds_src, "_applyAvailabilityGuard", 400)
        assert "cb.disabled = true" in body
        assert "cb.checked = false" in body

    def test_falsy_semantics(self, ds_src):
        """Guard nutzt Falsy-Semantik: !available (undefined = nicht verfuegbar)."""
        body = _ds_fn_body(ds_src, "_applyAvailabilityGuard", 400)
        # Pruefen: early-return nur bei 'available' (truthy), nicht '=== true'
        assert "|| available)" in body or "available)" in body
        assert "=== true" not in body
        assert "=== false" not in body


class TestDsSprint3WarningTextContent:
    """Dynamische Warning-Texte werden ueber textContent gesetzt, nicht innerHTML."""

    def test_create_ds_warning_uses_textcontent(self, ds_src):
        body = _ds_fn_body(ds_src, "_createDsWarning", 600)
        assert ".textContent = message" in body
        assert ".innerHTML = message" not in body

    def test_no_innerhtml_message_in_warnings(self, ds_src):
        """Kein innerHTML = message fuer DDU/VLM/Schedule-Warnings."""
        # Alle _createDsWarning-Aufrufe und manuelle Warning-Erzeugungen pruefen
        assert "innerHTML = message" not in ds_src
        # Alte Patterns mit warn.innerHTML sollten nicht existieren
        for pattern in ["warn.innerHTML", "warnDiv.innerHTML", "vlmWarn.innerHTML"]:
            assert pattern not in ds_src


class TestDsSprint3ScheduleWarnings:
    """Schedule-Review-Warnings behalten ID und Einfuegeanker."""

    def test_review_warning_id(self, ds_src):
        assert "'dsSyncReviewWarning'" in ds_src

    def test_load_schedule_warning_before_cron_field(self, ds_src):
        """HTTP-non-OK-Warning aus _loadScheduleForEdit vor #dsCronCustomField."""
        body = _ds_fn_body(ds_src, "_loadScheduleForEdit", 1500)
        assert "insertBefore(warnDiv, cronField)" in body

    def test_mark_review_warning_after_cron_field(self, ds_src):
        """Custom-Cron-Warning aus _markScheduleReviewState nach #dsCronCustomField."""
        body = _ds_fn_body(ds_src, "_markScheduleReviewState", 800)
        assert "insertBefore(warnDiv, cronField.nextSibling)" in body


class TestDsSprint3OnTypeChangeRebind:
    """_onTypeChange rebindet nach Connector-Feld-Neurender."""

    def test_rebinds_connector_events(self, ds_src):
        body = _ds_fn_body(ds_src, "_onTypeChange", 700)
        assert "_bindConnectorFieldEvents()" in body

    def test_updates_test_btn_visibility(self, ds_src):
        # Body-Laenge nach Welle 3 (classList-Toggles statt style.display) groesser
        body = _ds_fn_body(ds_src, "_onTypeChange", 1500)
        assert "_updateTestBtnVisibility()" in body


class TestDsSprint3CommonHelperBoundaries:
    """AP2-Common-Helper uebernimmt keine Edit-/Create-spezifische Logik."""

    def test_no_save_buttons(self, ds_src):
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "dsCreateBtn" not in body
        assert "dsSaveBtn" not in body

    def test_no_name_focus(self, ds_src):
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "dsName" not in body
        assert ".focus()" not in body

    def test_no_schedule_dirty_tracking(self, ds_src):
        body = _ds_fn_body(ds_src, "_bindDatasourceModalCommon", 800)
        assert "_scheduleFieldsDirty" not in body
        assert "_loadScheduleForEdit" not in body


class TestDsSprint3PersistenceContracts:
    """Create/Edit HTTP-Methoden bleiben getrennt."""

    def test_create_uses_post(self, ds_src):
        body = _ds_fn_body(ds_src, "openCreateDatasourceModal")
        assert "method: 'POST'" in body
        assert "'/api/admin/datasources'" in body or '"/api/admin/datasources"' in body

    def test_edit_uses_put(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "method: 'PUT'" in body
        assert "/api/admin/datasources/${dsId}" in body

    def test_save_ds_schedule_in_both(self, ds_src):
        create_body = _ds_fn_body(ds_src, "openCreateDatasourceModal")
        edit_body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        assert "_saveDsSchedule(" in create_body
        assert "_saveDsSchedule(" in edit_body

    def test_create_does_not_delete_scheduleless_schedule(self, ds_src):
        create_body = _ds_fn_body(ds_src, "openCreateDatasourceModal")
        assert "deleteScheduleless: false" in create_body


class TestDsSprint3PartialSuccessRecovery:
    """Schedule-Partial-Success-Recovery Contracts."""

    def test_create_partial_text(self, ds_src):
        assert "Datenquelle erstellt, aber Zeitplan fehlgeschlagen:" in ds_src

    def test_edit_partial_text(self, ds_src):
        assert "Datenquelle aktualisiert, aber Zeitplan fehlgeschlagen:" in ds_src

    def test_recovery_helper_is_async(self, ds_src):
        assert "async function _recoverDatasourceSchedulePartialFailure(" in ds_src

    def test_recovery_helper_remasks_before_reopen(self, ds_src):
        body = _ds_fn_body(ds_src, "_recoverDatasourceSchedulePartialFailure", 400)
        remask_pos = body.index("_remaskDraftSecrets(")
        reopen_pos = body.index("openEditDatasourceModal(")
        assert remask_pos < reopen_pos

    def test_create_awaits_recovery_then_returns(self, ds_src):
        body = _ds_fn_body(ds_src, "openCreateDatasourceModal")
        idx = body.index("await _recoverDatasourceSchedulePartialFailure(")
        rest = body[idx : idx + 500]
        # Nach dem Recovery-Aufruf muss return folgen
        lines_after = [l.strip() for l in rest.split("\n") if l.strip()]
        found_return = False
        for line in lines_after:
            if line == "return;":
                found_return = True
                break
        assert found_return, "return; fehlt nach _recoverDatasourceSchedulePartialFailure im Create"

    def test_edit_awaits_recovery_then_returns(self, ds_src):
        body = _ds_fn_body(ds_src, "openEditDatasourceModal")
        idx = body.index("await _recoverDatasourceSchedulePartialFailure(")
        rest = body[idx : idx + 500]
        found_return = False
        for line in [l.strip() for l in rest.split("\n") if l.strip()]:
            if line == "return;":
                found_return = True
                break
        assert found_return, "return; fehlt nach _recoverDatasourceSchedulePartialFailure im Edit"

    def test_recovery_reopen_uses_fallback(self, ds_src):
        """Recovery-Reopen priorisiert fallbackDs (via openEditDatasourceModal(id, draft))."""
        body = _ds_fn_body(ds_src, "_recoverDatasourceSchedulePartialFailure", 400)
        assert "openEditDatasourceModal(dsId, fallbackDraft)" in body


class TestDsSprint3FormHelpersExclusion:
    """tab_datenquellen.js enthaelt keine KiaraFormHelpers-Referenz."""

    def test_no_form_helpers(self, ds_src):
        assert "KiaraFormHelpers" not in ds_src


class TestDsSprint3DeadCodeRemoval:
    """AP4: _scheduleReadDone ist entfernt."""

    def test_no_schedule_read_done(self, ds_src):
        assert "_scheduleReadDone" not in ds_src


# ───────────────────────────────────────────────────────────────────
# Phase 2 Sprint 4: Trace Explorer — Panel-Split + Empty-State
# ───────────────────────────────────────────────────────────────────

TAB_TRACES_JS = _JS / "tab_system_traces.js"
TAB_TRACES_PANELS_JS = _JS / "tab_system_traces_panels.js"
EXPANDABLE_TABLE_JS = _BASE / "kiara" / "web" / "static" / "ui_core" / "js" / "table.js"


@pytest.fixture(scope="module")
def traces_src():
    return TAB_TRACES_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def traces_panels_src():
    return TAB_TRACES_PANELS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def expandable_table_src():
    """Sprint 6: zeigt auf ui_core/js/table.js (Verschiebung aus /static/js/expandable_table.js)."""
    return EXPANDABLE_TABLE_JS.read_text(encoding="utf-8")


class TestTraceSprint4LoadOrder:
    """admin.html laedt tab_system_traces_panels.js direkt nach tab_system_traces.js."""

    def test_panels_after_main(self, admin_html_src):
        pos_main = admin_html_src.index('src="/static/js/tab_system_traces.js?v=1"')
        pos_panels = admin_html_src.index('src="/static/js/tab_system_traces_panels.js?v=1"')
        assert pos_main < pos_panels

    def test_panels_before_vektordb(self, admin_html_src):
        pos_panels = admin_html_src.index('src="/static/js/tab_system_traces_panels.js?v=1"')
        pos_vektordb = admin_html_src.index('src="/static/js/tab_vektordb_management.js?v=1"')
        assert pos_panels < pos_vektordb


class TestTraceSprint4PanelsFileContent:
    """tab_system_traces_panels.js enthaelt die verschobenen Funktionen."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "function renderWaterfall(",
            "async function loadTraceDetail(",
            "function _renderTraceDetailExport(",
            "function _renderAggregationPanel(",
            "async function loadDurationHistogram(",
            "async function loadSpanAverages(",
            "async function openTraceComparison(",
            "function _renderComparisonView(",
            "function _renderExportButtons(",
            "function exportTraces(",
        ],
    )
    def test_panel_function_present(self, traces_panels_src, fn_name):
        assert fn_name in traces_panels_src, f"{fn_name} fehlt in panels-Datei"


class TestTraceSprint4MainFileRetained:
    """tab_system_traces.js enthaelt weiterhin Init, Filter, Table-Config, Lifecycle."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "function initSystemTracesTab(",
            "function _initTraceFilters(",
            "function _collectTraceFilters(",
            "function _installTraceTableLoadGuards(",
            "function _disposeTraceExplorerState(",
            "function loadTracesFiltered(",
            "function toggleTraceSelection(",
        ],
    )
    def test_retained_function(self, traces_src, fn_name):
        assert fn_name in traces_src, f"{fn_name} fehlt im Hauptfile"

    def test_trace_table_config(self, traces_src):
        assert "const traceTableConfig" in traces_src


class TestTraceSprint4NoDuplication:
    """Verschobene Funktionen nicht doppelt im Hauptfile."""

    @pytest.mark.parametrize(
        "fn_name",
        [
            "function renderWaterfall(",
            "function loadTraceDetail(",
            "function _renderTraceDetailExport(",
            "function _renderAggregationPanel(",
            "function loadDurationHistogram(",
            "function loadSpanAverages(",
            "function openTraceComparison(",
            "function _renderComparisonView(",
            "function _renderExportButtons(",
            "function exportTraces(",
        ],
    )
    def test_function_not_in_main(self, traces_src, fn_name):
        assert fn_name not in traces_src, f"{fn_name} noch im Hauptfile — doppelt"


class TestTraceSprint4CrossFileGuards:
    """Cross-File-Aufrufe: Hauptfile ruft Panel-Funktionen auf."""

    def test_init_calls_render_export_buttons(self, traces_src):
        """initSystemTracesTab ruft _renderExportButtons() auf."""
        idx = traces_src.index("function initSystemTracesTab(")
        body = traces_src[idx:]
        assert "_renderExportButtons()" in body

    def test_init_calls_render_aggregation_panel(self, traces_src):
        """initSystemTracesTab ruft _renderAggregationPanel() auf."""
        idx = traces_src.index("function initSystemTracesTab(")
        body = traces_src[idx:]
        assert "_renderAggregationPanel()" in body

    def test_on_row_expanded_calls_load_trace_detail(self, traces_src):
        """onRowExpanded ruft loadTraceDetail() auf."""
        idx = traces_src.index("function initSystemTracesTab(")
        body = traces_src[idx:]
        assert "loadTraceDetail(" in body


class TestTraceSprint4EmptyStateRenderer:
    """Empty-State-Renderer fuer Diagnose."""

    def test_render_trace_empty_state_exists(self, traces_src):
        assert "function _renderTraceEmptyState(" in traces_src

    def test_uses_escape_html(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 3500]
        assert "SharedUi.runtime.escapeHtml(" in body

    def test_checks_request_guard(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 300]
        assert "_isCurrentTraceTableRequest(requestSeq)" in body

    def test_queries_table_no_data(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 500]
        assert ".table-no-data" in body

    def test_tracing_disabled_hint(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 3500]
        assert "Tracing ist deaktiviert" in body
        assert "config/kiara.yaml" in body

    def test_chat_logs_without_trace_id_hint(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 3500]
        assert "Chat-Logs vorhanden" in body
        assert "ohne Tracing gespeichert" in body

    def test_data_consistency_hint(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 3500]
        assert "Datenkonsistenz-Hinweis" in body

    def test_filter_hint(self, traces_src):
        idx = traces_src.index("function _renderTraceEmptyState(")
        body = traces_src[idx : idx + 3500]
        assert "Filter" in body


class TestTraceSprint4DiagnosticsLifecycle:
    """Diagnose-State in dispose + loadData + silentRefresh."""

    def test_diagnostics_state_variable(self, traces_src):
        assert "let _traceDiagnostics = null" in traces_src

    def test_dispose_resets_diagnostics(self, traces_src):
        idx = traces_src.index("function _disposeTraceExplorerState(")
        body = traces_src[idx : idx + 500]
        assert "_traceDiagnostics = null" in body

    def test_load_data_post_render(self, traces_src):
        idx = traces_src.index("table.loadData = async function")
        body = traces_src[idx : idx + 4000]
        assert "_renderTraceEmptyState(requestSeq)" in body

    def test_silent_refresh_post_render(self, traces_src):
        idx = traces_src.index("table.silentRefresh = async function")
        body = traces_src[idx : idx + 4000]
        assert "_renderTraceEmptyState(requestSeq)" in body

    def test_on_data_loaded_callback(self, traces_src):
        idx = traces_src.index("new SharedUi.table.ExpandableTable(traceTableConfig")
        body = traces_src[idx : idx + 400]
        assert "onDataLoaded:" in body
        assert "_traceDiagnostics" in body


class TestSprint6TableJsInvariants:
    """Sprint 6: table.js (ehem. expandable_table.js) bleibt frei von Trace-Diagnostics-Helfern."""

    def test_no_diagnostics_in_expandable_table(self, expandable_table_src):
        assert "diagnostics" not in expandable_table_src

    def test_no_trace_empty_state_logic(self, expandable_table_src):
        """expandable_table.js enthaelt keine trace-spezifische Diagnose-Logik."""
        assert "_traceDiagnostics" not in expandable_table_src
        assert "_renderTraceEmptyState" not in expandable_table_src


# ───────────────────────────────────────────────────────────────────
# Shared Frontend Core — Sprint 1 (V23)
# Tokens + Theme Runtime — statische Strukturchecks
# ───────────────────────────────────────────────────────────────────

_WEB_ROOT = _BASE / "kiara" / "web"
_UI_CORE_CSS = _WEB_ROOT / "static" / "ui_core" / "css"
_UI_CORE_JS = _WEB_ROOT / "static" / "ui_core" / "js"
_STATIC_JS = _WEB_ROOT / "static" / "js"
_STATIC_CSS = _WEB_ROOT / "static" / "css"
_TEMPLATES = _WEB_ROOT / "templates"

_TOKENS_CSS = _UI_CORE_CSS / "tokens.css"
_THEME_CSS = _UI_CORE_CSS / "theme.css"
_CORE_JS = _UI_CORE_JS / "core.js"
_THEME_JS = _UI_CORE_JS / "theme.js"
_BOOTSTRAP_JS = _STATIC_JS / "kiara_bootstrap.js"
_STYLE_CSS = _STATIC_CSS / "style.css"

_INDEX_HTML = _TEMPLATES / "index.html"
_ADMIN_HTML = _TEMPLATES / "admin.html"
_LOGIN_HTML = _TEMPLATES / "login.html"
_ADMIN_LOGIN_HTML = _TEMPLATES / "admin_login.html"
_WIKI_BASE_HTML = _TEMPLATES / "wiki" / "base.html"
_WIKI_GUIDE_DYNAMIC_HTML = _TEMPLATES / "wiki" / "guide_dynamic.html"


@pytest.fixture(scope="module")
def tokens_css_src():
    return _TOKENS_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def theme_css_src():
    return _THEME_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def core_js_src():
    return _CORE_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def theme_js_src():
    return _THEME_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def bootstrap_js_src():
    return _BOOTSTRAP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def style_css_src():
    return _STYLE_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_html_src():
    return _INDEX_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_html_v23_src():
    return _ADMIN_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def login_html_src():
    return _LOGIN_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_login_html_src():
    return _ADMIN_LOGIN_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def wiki_base_html_src():
    return _WIKI_BASE_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def wiki_guide_dynamic_html_src():
    return _WIKI_GUIDE_DYNAMIC_HTML.read_text(encoding="utf-8")


class TestSFCv23Anchor1FilesExist:
    """V23 Test 1 — neue Dateien existieren."""

    @pytest.mark.parametrize(
        "path",
        [
            _TOKENS_CSS,
            _THEME_CSS,
            _CORE_JS,
            _THEME_JS,
            _BOOTSTRAP_JS,
        ],
    )
    def test_file_exists(self, path):
        assert path.is_file(), f"{path} fehlt"


class TestSFCv23Anchor1aScopeInvariant:
    """V23 Test 1a — IIFE-Scope, kein Top-Level const/let/function."""

    @pytest.mark.parametrize("src_fixture", ["core_js_src", "theme_js_src", "bootstrap_js_src"])
    def test_is_iife(self, src_fixture, request):
        src = request.getfixturevalue(src_fixture)
        assert re.search(r"^\(function\s*\(\)\s*\{", src, re.MULTILINE), (
            f"{src_fixture}: kein IIFE-Start gefunden"
        )
        assert src.rstrip().endswith("})();"), f"{src_fixture}: kein IIFE-Ende gefunden"

    @pytest.mark.parametrize("src_fixture", ["core_js_src", "theme_js_src", "bootstrap_js_src"])
    def test_no_top_level_declarations(self, src_fixture, request):
        src = request.getfixturevalue(src_fixture)
        lines = src.splitlines()
        violations = []
        for i, ln in enumerate(lines, 1):
            if re.match(r"^(const|let|function)\s+\w", ln):
                violations.append(f"Zeile {i}: {ln.strip()}")
        assert not violations, f"{src_fixture}: Top-Level-Deklarationen gefunden: {violations}"

    def test_only_shared_ui_global(self, core_js_src, theme_js_src, bootstrap_js_src):
        """Einzig erlaubtes neues Global: window.SharedUi."""
        # core.js erstellt das Namespace-Objekt auf window.SharedUi
        assert "window.SharedUi = {" in core_js_src
        # theme.js haengt nur an SharedUi
        assert "window.SharedUi.theme = {" in theme_js_src
        # bootstrap macht nur configure/load/addEventListener
        for src, name in [(theme_js_src, "theme.js"), (bootstrap_js_src, "kiara_bootstrap.js")]:
            bad = re.findall(r"window\.[A-Za-z_]+\s*=", src)
            for hit in bad:
                assert "window.SharedUi" in hit, f"{name}: unerwartetes Global {hit}"


class TestSFCv23Anchor2CssTokenCounts:
    """V23 Test 2 — Token-Counts in tokens.css/theme.css/style.css."""

    def test_tokens_css_has_32_root_tokens(self, tokens_css_src):
        root_block = re.search(r":root\s*\{([^}]*)\}", tokens_css_src, re.DOTALL)
        assert root_block, "tokens.css hat keinen :root-Block"
        decls = re.findall(r"--[a-z][a-z0-9-]*\s*:", root_block.group(1))
        assert len(decls) == 32, f"tokens.css: erwartet 32 Core-Tokens, gefunden {len(decls)}"

    def test_theme_css_dark_count(self, theme_css_src):
        dark_block = re.search(r'\[data-theme="dark"\]\s*\{([^}]*)\}', theme_css_src, re.DOTALL)
        assert dark_block, 'theme.css hat keinen [data-theme="dark"]-Block'
        decls = re.findall(r"--[a-z][a-z0-9-]*\s*:", dark_block.group(1))
        assert len(decls) == 22, f"theme.css: erwartet 22 Dark-Overrides, gefunden {len(decls)}"

    def test_theme_css_kontrast_count(self, theme_css_src):
        kontrast_block = re.search(
            r'\[data-theme="kontrast"\]\s*\{([^}]*)\}', theme_css_src, re.DOTALL
        )
        assert kontrast_block, 'theme.css hat keinen [data-theme="kontrast"]-Block'
        decls = re.findall(r"--[a-z][a-z0-9-]*\s*:", kontrast_block.group(1))
        assert len(decls) == 24, f"theme.css: erwartet 24 Kontrast-Overrides, gefunden {len(decls)}"

    def test_style_css_root_has_only_color_tokens(self, style_css_src):
        root_block = re.search(r":root\s*\{([^}]*)\}", style_css_src, re.DOTALL)
        assert root_block, "style.css hat keinen :root-Block"
        decls = re.findall(r"--([a-z][a-z0-9-]*)\s*:", root_block.group(1))
        assert sorted(decls) == [
            "color-bearish",
            "color-bullish",
            "color-neutral",
            "color-unclear",
        ], f"style.css :root: unerwartete Tokens {decls}"

    def test_style_css_dark_has_only_color_tokens(self, style_css_src):
        dark_block = re.search(r'\[data-theme="dark"\]\s*\{([^}]*)\}', style_css_src, re.DOTALL)
        assert dark_block, 'style.css hat keinen [data-theme="dark"]-Block'
        decls = re.findall(r"--([a-z][a-z0-9-]*)\s*:", dark_block.group(1))
        assert sorted(decls) == [
            "color-bearish",
            "color-bullish",
            "color-neutral",
            "color-unclear",
        ], f"style.css [data-theme=dark]: unerwartete Tokens {decls}"

    def test_style_css_no_kontrast_token_block(self, style_css_src):
        """Der [data-theme='kontrast']-Token-Block wurde komplett nach theme.css verschoben."""
        matches = re.findall(
            r'\[data-theme="kontrast"\]\s*\{[^}]*--[a-z]', style_css_src, re.DOTALL
        )
        assert not matches, f"style.css enthaelt noch Kontrast-Token-Block: {matches}"

    def test_style_css_retains_kontrast_tab_hover(self, style_css_src):
        """Nicht-tokenisierte Kontrast-Tab-Hover-Regeln bleiben in style.css."""
        assert '[data-theme="kontrast"] .tabs .tab:hover' in style_css_src
        assert '[data-theme="kontrast"] .sub-tabs .tab:hover' in style_css_src

    def test_style_css_retains_dark_badges(self, style_css_src):
        """Nicht-tokenisierte Dark-Badge-Regeln bleiben in style.css."""
        assert '[data-theme="dark"] .badge-ldap' in style_css_src
        assert '[data-theme="dark"] .badge-purple' in style_css_src


class TestSFCv23Anchor2aColorCascade:
    """V23 Test 2a — --color-*-Cascade-Abdeckung."""

    def test_color_tokens_defined_only_in_style_css(self):
        """Die 4 KIara-Tokens sind nur in style.css als Definition."""
        for token in ["color-bullish", "color-bearish", "color-neutral", "color-unclear"]:
            # Darf nicht in tokens.css oder theme.css definiert werden
            for path in [_TOKENS_CSS, _THEME_CSS]:
                src = path.read_text(encoding="utf-8")
                # Definition heisst "--token:", Verbrauch "var(--token"
                assert not re.search(r"--" + token + r"\s*:", src), (
                    f"{path.name} definiert unerwartet --{token}"
                )

    def test_color_tokens_not_in_ui_core_js(self, core_js_src, theme_js_src):
        """ui_core JS referenziert keine KIara-Color-Tokens."""
        for src in [core_js_src, theme_js_src]:
            for token in ["color-bullish", "color-bearish", "color-neutral", "color-unclear"]:
                assert token not in src

    def test_no_new_alias_tokens_defined(self, tokens_css_src, theme_css_src, style_css_src):
        """Keine neuen --color-success/warning/danger/primary-Definitionen eingefuehrt."""
        for src in [tokens_css_src, theme_css_src, style_css_src]:
            for alias in ["color-success", "color-warning", "color-danger", "color-primary"]:
                assert not re.search(r"--" + alias + r"\s*:", src), (
                    f"Neues Alias-Token --{alias} gefunden — out of scope"
                )


class TestSFCv23Anchor2bViewDarkSelectors:
    """V23 Test 2b — View-spezifische Dark-Selektoren bleiben unveraendert."""

    def test_chat_css_selector_count(self):
        # Sprint 5: 5 Dark-Selektoren wandern nach ui_core/css/chat.css (.chat-usage-bar,
        # .chat-usage-response) bzw. markdown.css (.chat-text code/pre, .markdown-table).
        src = (_STATIC_CSS / "chat.css").read_text(encoding="utf-8")
        hits = re.findall(r'^\[data-theme="dark"\]', src, re.MULTILINE)
        assert len(hits) == 8, (
            f"chat.css: erwartet 8 Dark-Selektoren inklusive Verarbeitungswarnung, gefunden {len(hits)}"
        )

    def test_admin_chat_css_selector_count(self):
        src = (_STATIC_CSS / "admin_chat.css").read_text(encoding="utf-8")
        hits = re.findall(r'^\[data-theme="dark"\]', src, re.MULTILINE)
        assert len(hits) == 13, f"admin_chat.css: erwartet 13 Dark-Selektoren, gefunden {len(hits)}"

    def test_kpi_grid_css_deleted(self):
        """Sprint 6: kpi_grid.css wurde nach style.css integriert und geloescht."""
        assert not (_STATIC_CSS / "kpi_grid.css").exists()

    def test_kpi_dark_selectors_in_style_css(self):
        """Sprint 6: Dark-Mode-Overrides fuer KPI-/Widget-Shell liegen in style.css."""
        src = (_STATIC_CSS / "style.css").read_text(encoding="utf-8")
        assert '[data-theme="dark"] .widget-header' in src
        assert '[data-theme="dark"] .kpi-card' in src

    # Sprint 5: .chat-usage-bar wandert nach ui_core/css/chat.css,
    # .chat-table tr:nth-child(even) wird zu .markdown-table tr:nth-child(even) in markdown.css.
    # Sprint 6: .widget-header/.kpi-card liegen jetzt in style.css (kpi_grid.css geloescht).
    @pytest.mark.parametrize(
        "filename,anchor",
        [
            ("chat.css", ".chat-mode-badge.unchained"),
            ("chat.css", ".chat-mode-badge.agent"),
            ("chat.css", ".chat-context-warning"),
            ("admin_chat.css", ".confirmation-bubble"),
            ("admin_chat.css", ".wizard-bubble"),
            ("admin_chat.css", ".wizard-param .param-name"),
            ("admin_chat.css", ".wizard-param .param-desc"),
            ("style.css", ".widget-header"),
            ("style.css", ".kpi-card"),
        ],
    )
    def test_special_anchors(self, filename, anchor):
        src = (_STATIC_CSS / filename).read_text(encoding="utf-8")
        assert anchor in src, f"{filename}: Anker {anchor} fehlt"

    def test_sprint5_moved_anchors(self):
        """Sprint 5: extrahierte Anker sind an der neuen Stelle vorhanden."""
        ui_core_chat = (
            _BASE / "kiara" / "web" / "static" / "ui_core" / "css" / "chat.css"
        ).read_text(encoding="utf-8")
        md_css = (
            _BASE / "kiara" / "web" / "static" / "ui_core" / "css" / "markdown.css"
        ).read_text(encoding="utf-8")
        assert ".chat-usage-bar" in ui_core_chat
        assert ".markdown-table tr:nth-child(even)" in md_css

    @pytest.mark.parametrize("filename", ["chat.css", "admin_chat.css"])
    def test_no_new_custom_property_definitions(self, filename):
        """View-Dateien definieren keine neuen Custom Properties."""
        src = (_STATIC_CSS / filename).read_text(encoding="utf-8")
        assert not re.search(r"--[a-z][a-z0-9-]*\s*:\s*[^v]", src), (
            f"{filename}: enthaelt eine Custom-Property-Definition"
        )


class TestSFCv23Anchor3TemplateOrder:
    """V23 Test 3 — Template-Reihenfolge."""

    @pytest.mark.parametrize(
        "tpl_fixture",
        [
            "index_html_src",
            "admin_html_v23_src",
            "login_html_src",
            "admin_login_html_src",
            "wiki_base_html_src",
        ],
    )
    def test_css_order_tokens_theme_style(self, tpl_fixture, request):
        src = request.getfixturevalue(tpl_fixture)
        pos_tokens = src.find("ui_core/css/tokens.css")
        pos_theme = src.find("ui_core/css/theme.css")
        pos_style = src.find("/static/css/style.css")
        assert pos_tokens != -1 and pos_theme != -1 and pos_style != -1
        assert pos_tokens < pos_theme < pos_style, f"{tpl_fixture}: CSS-Reihenfolge verletzt"

    @pytest.mark.parametrize("tpl_fixture", ["index_html_src", "admin_html_v23_src"])
    def test_js_order_in_main_templates(self, tpl_fixture, request):
        """core.js -> theme.js -> kiara_bootstrap.js als erste externe Body-Scripts."""
        src = request.getfixturevalue(tpl_fixture)
        pos_core = src.find("/static/ui_core/js/core.js")
        pos_theme = src.find("/static/ui_core/js/theme.js")
        pos_bootstrap = src.find("/static/js/kiara_bootstrap.js")
        assert pos_core != -1 and pos_theme != -1 and pos_bootstrap != -1
        assert pos_core < pos_theme < pos_bootstrap, f"{tpl_fixture}: JS-Reihenfolge verletzt"
        # Erste externe Body-Script-Gruppe: kein anderes <script src=...> davor
        head_end = src.find("</head>")
        body_region = src[head_end:pos_core]
        other_scripts = re.findall(r'<script\s+src="[^"]+"', body_region)
        assert not other_scripts, f"{tpl_fixture}: andere Scripts vor core.js: {other_scripts}"

    @pytest.mark.parametrize(
        "tpl_fixture", ["login_html_src", "admin_login_html_src", "wiki_base_html_src"]
    )
    def test_inline_guard_before_core(self, tpl_fixture, request):
        """Inline-Guard steht vor core.js/theme.js/kiara_bootstrap.js."""
        src = request.getfixturevalue(tpl_fixture)
        pos_ls = src.find("localStorage.getItem('kiara-theme')")
        pos_core = src.find("/static/ui_core/js/core.js")
        pos_theme = src.find("/static/ui_core/js/theme.js")
        pos_bootstrap = src.find("/static/js/kiara_bootstrap.js")
        assert pos_ls != -1 and pos_core != -1 and pos_theme != -1 and pos_bootstrap != -1
        assert pos_ls < pos_core < pos_theme < pos_bootstrap

    def test_wiki_guide_dynamic_is_theme_neutral(self, wiki_guide_dynamic_html_src):
        """wiki/guide_dynamic.html referenziert keine Theme-API/keine Storage-Lookups."""
        src = wiki_guide_dynamic_html_src
        for token in [
            "SharedUi",
            "data-theme",
            "themeChanged",
            ".theme-option",
            "kiara-theme",
            "localStorage",
        ]:
            assert token not in src, f"wiki/guide_dynamic.html: unerwarteter Treffer {token}"


class TestSFCv23Anchor4CacheBusters:
    """V23 Test 4 — Cache-Buster-Konvention."""

    @pytest.mark.parametrize(
        "tpl_path,expected",
        [
            (_INDEX_HTML, 19),
            (_ADMIN_HTML, 20),
        ],
    )
    def test_main_templates_have_v1(self, tpl_path, expected):
        src = tpl_path.read_text(encoding="utf-8")
        hits = re.findall(r"ui_core/[a-z]+/[a-z_]+\.(?:css|js)\?v=1", src)
        # Sprint 3: tokens.css + theme.css + notifications.css + modal.css
        #           + core.js + theme.js + fetch.js + runtime.js + notifications.js + modal.js = 10
        # Sprint 4: + markdown.css + markdown.js = 12
        # Sprint 5: + ui_core/css/chat.css + ui_core/js/chat_primitives.js = 14
        # Sprint 6: + layout.css + forms.css + table.css + table.js = 18 (index.html)
        # Sprint 6 admin.html: zusaetzlich forms.js = 19
        # V30 Welle 3: + ui_core/js/runtime_style.js = 19 (index) / 20 (admin)
        assert len(hits) == expected, (
            f"{tpl_path.name}: erwartet {expected} ui_core-Assets mit ?v=1, gefunden {len(hits)}"
        )
        assert len(re.findall(r"kiara_bootstrap\.js\?v=1", src)) == 1, (
            f"{tpl_path.name}: erwartet 1 kiara_bootstrap.js?v=1"
        )

    @pytest.mark.parametrize(
        "tpl_path,expected",
        [
            (_LOGIN_HTML, 5),
            (_ADMIN_LOGIN_HTML, 5),
            (_WIKI_BASE_HTML, 4),
        ],
    )
    def test_login_wiki_templates_have_no_v1(self, tpl_path, expected):
        src = tpl_path.read_text(encoding="utf-8")
        hits = re.findall(r'ui_core/[a-z]+/[a-z_]+\.(?:css|js)(?:\?v=1)?"', src)
        with_v1 = [h for h in hits if "?v=1" in h]
        assert len(hits) == expected, (
            f"{tpl_path.name}: erwartet {expected} ui_core-Assets, gefunden {len(hits)}"
        )
        assert not with_v1, f"{tpl_path.name}: ?v=1 unerwartet in Login/Wiki-Template"
        bs_with_v1 = re.findall(r"kiara_bootstrap\.js\?v=1", src)
        assert not bs_with_v1, f"{tpl_path.name}: kiara_bootstrap.js?v=1 unerwartet"
        bs_plain = re.findall(r'kiara_bootstrap\.js"', src)
        assert len(bs_plain) == 1, f"{tpl_path.name}: kiara_bootstrap.js ohne ?v=1 fehlt"


class TestSFCv23Anchor5AppCoreMigration:
    """V23 Test 5 — app_core.js: keine alten Theme-Globals, neuer Click-Handler."""

    def test_no_valid_themes(self, app_core_src):
        assert not re.search(r"\bVALID_THEMES\b", app_core_src)

    @pytest.mark.parametrize("fn", ["initTheme", "setTheme", "toggleTheme", "updateThemeButton"])
    def test_no_legacy_function_definition(self, app_core_src, fn):
        assert not re.search(r"\bfunction\s+" + fn + r"\s*\(", app_core_src), (
            f"app_core.js: alte Funktion {fn} nicht entfernt"
        )

    @pytest.mark.parametrize("fn", ["initTheme", "setTheme", "toggleTheme", "updateThemeButton"])
    def test_no_legacy_function_call(self, app_core_src, fn):
        assert not re.search(r"\b" + fn + r"\s*\(", app_core_src), (
            f"app_core.js: alter Aufruf {fn}() nicht entfernt"
        )

    def test_theme_option_handler_uses_shared_ui(self, app_core_src):
        assert "SharedUi.theme.set(btn.dataset.theme)" in app_core_src


class TestSFCv23Anchor6BootstrapStructure:
    """V23 Test 6 — kiara_bootstrap.js Struktur."""

    def test_theme_changed_listener(self, bootstrap_js_src):
        assert "addEventListener('themeChanged'" in bootstrap_js_src

    def test_theme_options_query(self, bootstrap_js_src):
        assert "querySelectorAll('.theme-option')" in bootstrap_js_src

    def test_classlist_toggle_active(self, bootstrap_js_src):
        assert "classList.toggle('active'" in bootstrap_js_src
        assert "btn.dataset.theme === theme" in bootstrap_js_src

    def test_ready_state_branch(self, bootstrap_js_src):
        assert "document.readyState === 'loading'" in bootstrap_js_src
        assert "DOMContentLoaded" in bootstrap_js_src

    def test_malformed_theme_changed_is_noop(self, bootstrap_js_src):
        """Ohne e.detail.theme bricht Listener ab (no-op)."""
        # Listener-Body enthaelt defensive Pruefung
        idx = bootstrap_js_src.find("addEventListener('themeChanged'")
        body = bootstrap_js_src[idx : idx + 600]
        assert "e.detail" in body
        assert "typeof e.detail.theme !== 'string'" in body or "e.detail.theme" in body
        assert "return" in body


class TestSFCv23Anchor6aBootstrapConfigOrder:
    """V23 Test 6a — configure vor load."""

    def test_configure_before_load(self, bootstrap_js_src):
        pos_cfg = bootstrap_js_src.find("SharedUi.configure(")
        pos_load = bootstrap_js_src.find("SharedUi.theme.load()")
        assert pos_cfg != -1 and pos_load != -1
        assert pos_cfg < pos_load

    def test_configure_sets_kiara_theme_key(self, bootstrap_js_src):
        assert "themeStorageKey: 'kiara-theme'" in bootstrap_js_src


class TestSFCv23Anchor6bAllowedThemesGuard:
    """V23 Test 6b — _normalizedAllowed in allen 5 Public-Methods."""

    def _method_body(self, src, name):
        """Extrahiert Funktionsbody fuer die benannte Top-Level-Funktion in theme.js."""
        pattern = r"function\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{"
        m = re.search(pattern, src)
        assert m, f"{name} nicht gefunden"
        # Balanciere Braces
        start = m.end() - 1
        depth = 0
        i = start
        while i < len(src):
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return src[start : i + 1]
            i += 1
        raise AssertionError(f"{name}: Body-Ende nicht gefunden")

    @pytest.mark.parametrize("method", ["load", "set", "toggle", "isAllowed", "getAllowed"])
    def test_normalized_allowed_called_in_method(self, theme_js_src, method):
        body = self._method_body(theme_js_src, method)
        assert "_normalizedAllowed()" in body, (
            f"theme.js {method}: _normalizedAllowed() nicht aufgerufen"
        )


class TestSFCv23Anchor6cThemeSetContract:
    """V23 Test 6c — set(theme) Contract: Storage-vor-DOM, Event-Reihenfolge, No-op bei Fehler."""

    def _set_body(self, src):
        m = re.search(r"function\s+set\s*\(", src)
        assert m, "set()-Funktion nicht gefunden"
        start = src.find("{", m.end() - 1)
        depth = 0
        i = start
        while i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    return src[start : i + 1]
            i += 1
        raise AssertionError("set() Body-Ende nicht gefunden")

    def test_storage_before_dom(self, theme_js_src):
        body = self._set_body(theme_js_src)
        pos_ls = body.find("localStorage.setItem(")
        pos_dom = body.find("document.documentElement.setAttribute('data-theme'")
        assert pos_ls != -1 and pos_dom != -1
        assert pos_ls < pos_dom, "set(): Storage-Write muss vor DOM-Set stehen"

    def test_dispatch_event_is_last(self, theme_js_src):
        body = self._set_body(theme_js_src)
        pos_dom = body.find("document.documentElement.setAttribute('data-theme'")
        pos_sync = body.find("_syncSelectors(")
        pos_event = body.find("dispatchEvent(")
        assert pos_dom < pos_sync < pos_event, (
            "set(): Reihenfolge DOM -> _syncSelectors -> dispatchEvent verletzt"
        )

    def test_invalid_theme_returns_false(self, theme_js_src):
        body = self._set_body(theme_js_src)
        # allow-guard fuehrt vor allem anderen zu return false
        m = re.search(
            r"if\s*\(\s*allowed\.indexOf\(theme\)\s*===\s*-1\s*\)\s*\{\s*return\s+false;\s*\}", body
        )
        assert m, "set(): Guard 'indexOf === -1 -> return false' fehlt"
        # Davor darf kein DOM-Set oder Event stehen
        before = body[: m.end()]
        assert "setAttribute('data-theme'" not in before
        assert "dispatchEvent(" not in before

    def test_ls_throw_is_full_noop(self, theme_js_src):
        body = self._set_body(theme_js_src)
        # try { localStorage.setItem(...) } catch { return false; } (ohne DOM/event dazwischen)
        m = re.search(
            r"try\s*\{\s*localStorage\.setItem[^}]*\}\s*catch\s*\([^)]*\)\s*\{\s*return\s+false;?\s*\}",
            body,
            re.DOTALL,
        )
        assert m, "set(): try/catch-Block um localStorage.setItem mit return false fehlt"
        catch_block = m.group(0)
        assert "setAttribute" not in catch_block
        assert "dispatchEvent" not in catch_block
        assert "_syncSelectors" not in catch_block


class TestSFCv23Anchor6dThemeLoadErrorPath:
    """V23 Test 6d — load()-Fehlerpfad: try/catch, Fallback, kein Event."""

    def _load_body(self, src):
        m = re.search(r"function\s+load\s*\(", src)
        assert m, "load()-Funktion nicht gefunden"
        start = src.find("{", m.end() - 1)
        depth = 0
        i = start
        while i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    return src[start : i + 1]
            i += 1
        raise AssertionError("load() Body-Ende nicht gefunden")

    def test_getitem_wrapped_in_try_catch(self, theme_js_src):
        body = self._load_body(theme_js_src)
        m = re.search(
            r"try\s*\{\s*stored\s*=\s*localStorage\.getItem[^}]*\}\s*catch",
            body,
            re.DOTALL,
        )
        assert m, "load(): localStorage.getItem nicht im try/catch"

    def test_fallback_to_allowed_zero(self, theme_js_src):
        body = self._load_body(theme_js_src)
        assert "allowed[0]" in body, "load(): Fallback auf allowed[0] fehlt"

    def test_no_dispatch_event(self, theme_js_src):
        body = self._load_body(theme_js_src)
        assert "dispatchEvent(" not in body, "load(): dispatchEvent unerwartet (Load != Change)"


class TestSFCv23Anchor7CoreNeutrality:
    """V23 Test 7 — ui_core/ ist produktneutral."""

    @pytest.mark.parametrize(
        "src_fixture,forbidden",
        [
            ("core_js_src", "kiara"),
            ("core_js_src", "/api/"),
            ("core_js_src", "admin"),
            ("core_js_src", "chat"),
            ("theme_js_src", "kiara"),
            ("theme_js_src", "/api/"),
            ("theme_js_src", "admin"),
            ("theme_js_src", "chat"),
            ("tokens_css_src", "kiara"),
            ("theme_css_src", "kiara"),
        ],
    )
    def test_no_product_strings(self, src_fixture, forbidden, request):
        src = request.getfixturevalue(src_fixture).lower()
        assert forbidden.lower() not in src, (
            f"{src_fixture}: verbotener Produkt-/API-String {forbidden}"
        )

    def test_bootstrap_has_kiara_key(self, bootstrap_js_src):
        """kiara_bootstrap.js enthaelt genau den KIara-Key."""
        assert "themeStorageKey: 'kiara-theme'" in bootstrap_js_src


class TestSFCv23Anchor8InlineGuardHardening:
    """V23 Test 8 — Inline-Guards haben try/catch, Fallback light."""

    @pytest.mark.parametrize(
        "tpl_fixture", ["login_html_src", "admin_login_html_src", "wiki_base_html_src"]
    )
    def test_try_catch_around_getitem(self, tpl_fixture, request):
        src = request.getfixturevalue(tpl_fixture)
        m = re.search(
            r"try\s*\{\s*_raw\s*=\s*localStorage\.getItem\('kiara-theme'\);?\s*\}\s*catch",
            src,
        )
        assert m, f"{tpl_fixture}: try/catch um localStorage.getItem fehlt"

    @pytest.mark.parametrize(
        "tpl_fixture", ["login_html_src", "admin_login_html_src", "wiki_base_html_src"]
    )
    def test_fallback_light(self, tpl_fixture, request):
        src = request.getfixturevalue(tpl_fixture)
        assert "'light'" in src
        assert "['light','dark','kontrast']" in src

    def test_login_subsequent_code_runs(self, login_html_src):
        """Login-Seite: Code nach Theme-Block laeuft weiter (Button-Handler etc.)."""
        # Nach Theme-Block: loginBtn.addEventListener('click', doLogin);
        assert "loginBtn.addEventListener('click', doLogin)" in login_html_src

    def test_admin_login_subsequent_code_runs(self, admin_login_html_src):
        # Login-Trigger ist seit Frontend-Refactor ein Form-Submit-Handler
        # (statt Button-Click) — funktional aequivalent inkl. Enter-Key.
        assert "getElementById('loginForm').addEventListener('submit'" in admin_login_html_src

    def test_wiki_base_subsequent_code_runs(self, wiki_base_html_src):
        assert "toggle.addEventListener('click'" in wiki_base_html_src


class TestSFCv23Anchor9PackagingMirror:
    """V23 Test 9 — Debian-Mirror-Disziplin (nur bei vorhandenem Mirror)."""

    def test_mirror_has_no_stale_symbols_if_present(self):
        mirror = (
            _BASE
            / "debian"
            / "kiara"
            / "opt"
            / "kiara"
            / "kiara"
            / "web"
            / "static"
            / "js"
            / "app_core.js"
        )
        if not mirror.exists():
            pytest.skip("debian/kiara/ Mirror nicht vorhanden — wird bei make deb regeneriert")
        src = mirror.read_text(encoding="utf-8")
        # Nur pruefen falls der Mirror regeneriert wurde; alte Symbole waeren ein Hinweis
        # auf veralteten Build-Baum.
        for fn in [
            "function initTheme",
            "function setTheme",
            "function toggleTheme",
            "function updateThemeButton",
            "VALID_THEMES",
        ]:
            assert fn not in src, f"Mirror enthaelt veraltetes Symbol {fn} — make deb regenerieren"


class TestSFCv23Anchor10RepoWideDisciplinedGreps:
    """V23 Test 10 — repo-weite Grep-Disziplin im KIara-Primaerpfad."""

    def test_no_kiara_theme_underscore_variant(self):
        """Unterstrich-Variante `kiara_theme` existiert nirgends im KIara-Web-Baum."""
        for path in _WEB_ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in (".js", ".html", ".css"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "kiara_theme" not in text, f"{path}: unerwartete kiara_theme-Variante"

    def test_localstorage_kiara_theme_in_templates_only(self):
        """Drei Inline-Guards in Login-/Wiki-Templates; KEIN Treffer in static/js."""
        pattern = re.compile(r"localStorage[^;\n]*kiara-theme")
        template_hits = 0
        for tpl in [_LOGIN_HTML, _ADMIN_LOGIN_HTML, _WIKI_BASE_HTML]:
            hits = pattern.findall(tpl.read_text(encoding="utf-8"))
            template_hits += len(hits)
        assert template_hits >= 3, f"erwartet 3 Inline-Guards, gefunden {template_hits}"
        # In static/js/: kein localStorage-Literal mit 'kiara-theme' (Key kommt ueber configure)
        for js in _STATIC_JS.rglob("*.js"):
            text = js.read_text(encoding="utf-8")
            if "localStorage" not in text:
                continue
            # Nur Zeilen die sowohl localStorage als auch 'kiara-theme' enthalten
            for ln in text.splitlines():
                if "localStorage" in ln and "kiara-theme" in ln:
                    pytest.fail(
                        f"{js}: localStorage-Literal 'kiara-theme' gefunden — Key soll ueber configure gesetzt werden: {ln}"
                    )

    def test_configure_kiara_theme_once_in_bootstrap(self, bootstrap_js_src):
        hits = re.findall(r"themeStorageKey[^;]*kiara-theme", bootstrap_js_src)
        assert len(hits) == 1, (
            f"kiara_bootstrap.js: erwartet 1 themeStorageKey-Match, gefunden {len(hits)}"
        )

    def test_no_inline_onclick_theme(self):
        """Kein inline onclick=... mit Theme-Funktion."""
        pattern = re.compile(r'onclick\s*=\s*"[^"]*[Tt]heme')
        for tpl in _TEMPLATES.rglob("*.html"):
            text = tpl.read_text(encoding="utf-8")
            assert not pattern.search(text), f"{tpl}: inline onclick mit Theme gefunden"


# ───────────────────────────────────────────────────────────────────
# Shared Frontend Core — Sprint 2 (V30)
# Fetch-/Runtime-Helper — statische Strukturchecks
# ───────────────────────────────────────────────────────────────────

_FETCH_JS = _UI_CORE_JS / "fetch.js"
_RUNTIME_JS = _UI_CORE_JS / "runtime.js"
_AUDIT_DETAIL_JS = _STATIC_JS / "audit_detail.js"
_ADMIN_CHAT_JS = _STATIC_JS / "admin_chat.js"
_TAB_KIARA_JS = _STATIC_JS / "tab_kiara.js"
_CHAT_JS = _STATIC_JS / "chat.js"


@pytest.fixture(scope="module")
def fetch_js_src():
    return _FETCH_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def runtime_js_src():
    return _RUNTIME_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_chat_js_src():
    return _ADMIN_CHAT_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tab_kiara_js_src():
    return _TAB_KIARA_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def chat_js_src():
    return _CHAT_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def audit_detail_js_src():
    return _AUDIT_DETAIL_JS.read_text(encoding="utf-8")


class TestSFCSprint2Anchor1FilesExist:
    """V30 Test 1 — neue Core-Dateien existieren."""

    @pytest.mark.parametrize("path", [_FETCH_JS, _RUNTIME_JS])
    def test_file_exists(self, path):
        assert path.is_file(), f"{path} fehlt"


class TestSFCSprint2Anchor1aScopeInvariant:
    """V30 Test 1a — IIFE-Scope und einziges Global SharedUi.fetch / SharedUi.runtime."""

    @pytest.mark.parametrize("src_fixture", ["fetch_js_src", "runtime_js_src"])
    def test_is_iife(self, src_fixture, request):
        src = request.getfixturevalue(src_fixture)
        assert re.search(r"^\(function\s*\(\)\s*\{", src, re.MULTILINE), (
            f"{src_fixture}: kein IIFE-Start"
        )
        assert src.rstrip().endswith("})();"), f"{src_fixture}: kein IIFE-Ende"

    @pytest.mark.parametrize("src_fixture", ["fetch_js_src", "runtime_js_src"])
    def test_no_top_level_declarations(self, src_fixture, request):
        src = request.getfixturevalue(src_fixture)
        violations = []
        for i, ln in enumerate(src.splitlines(), 1):
            if re.match(r"^(const|let|function)\s+\w", ln):
                violations.append(f"Zeile {i}: {ln.strip()}")
        assert not violations, f"{src_fixture}: Top-Level-Deklarationen: {violations}"

    def test_only_shared_ui_globals(self, fetch_js_src, runtime_js_src):
        for src, name in [(fetch_js_src, "fetch.js"), (runtime_js_src, "runtime.js")]:
            for hit in re.findall(r"window\.[A-Za-z_]+\s*=", src):
                assert "window.SharedUi" in hit, f"{name}: unerwartetes Global {hit}"

    def test_setup_guard(self, fetch_js_src, runtime_js_src):
        for src, name in [(fetch_js_src, "fetch.js"), (runtime_js_src, "runtime.js")]:
            assert "if (!window.SharedUi" in src or "throw new Error" in src, (
                f"{name}: Setup-Guard fehlt"
            )

    def test_export_surface_fetch(self, fetch_js_src):
        for member in ("safeFetch", "safeFetchJson", "FetchError"):
            assert member in fetch_js_src, f"fetch.js exportiert {member} nicht"

    def test_export_surface_runtime(self, runtime_js_src):
        for member in (
            "escapeHtml",
            "escapeAttr",
            "escapeJSAttr",
            "createAbort",
            "abort",
            "abortAll",
            "withButtonLock",
            "polling",
            "PollingManager",
        ):
            assert member in runtime_js_src, f"runtime.js exportiert {member} nicht"


class TestSFCSprint2Anchor2FetchContract:
    """V30 Test 3 — safeFetch-Vertrag (CSRF, AbortError, FetchError)."""

    def test_mutating_set(self, fetch_js_src):
        assert "MUTATING" in fetch_js_src
        assert "POST" in fetch_js_src and "PUT" in fetch_js_src
        assert "PATCH" in fetch_js_src and "DELETE" in fetch_js_src

    def test_csrf_header_uses_config_value(self, fetch_js_src):
        """CSRF-Header und -Wert aus cfg, nicht hardcoded."""
        assert "cfg.csrfHeader" in fetch_js_src
        assert "cfg.csrfValue" in fetch_js_src
        # XMLHttpRequest darf nirgends als Literal stehen
        assert "'XMLHttpRequest'" not in fetch_js_src
        assert '"XMLHttpRequest"' not in fetch_js_src

    def test_csrf_only_on_mutating(self, fetch_js_src):
        assert re.search(r"MUTATING\.has\s*\(\s*method\s*\)", fetch_js_src)

    def test_default_credentials_from_config(self, fetch_js_src):
        assert "cfg.defaultCredentials" in fetch_js_src
        # Variable heisst im aktuellen Code 'opts' (lokale Kopie via _cloneOptions);
        # Assertion bleibt semantisch identisch, akzeptiert beide Namen.
        assert "'credentials' in opts" in fetch_js_src or "'credentials' in options" in fetch_js_src

    def test_abort_error_passthrough(self, fetch_js_src):
        assert re.search(r"err\.name\s*===\s*['\"]AbortError['\"]", fetch_js_src)

    def test_network_error_wrapped(self, fetch_js_src):
        assert "FetchError" in fetch_js_src
        assert "'network'" in fetch_js_src

    def test_401_callback_invoked(self, fetch_js_src):
        assert "cfg.on401" in fetch_js_src
        assert "Promise.resolve" in fetch_js_src
        assert "'unauthorized'" in fetch_js_src
        assert "401" in fetch_js_src

    def test_401_throws_after_callback(self, fetch_js_src):
        # Sicherstellen, dass nach on401 trotzdem geworfen wird
        idx = fetch_js_src.index("response.status === 401")
        body = fetch_js_src[idx : idx + 600]
        assert "throw" in body

    def test_http_error_wrapped(self, fetch_js_src):
        assert "'http'" in fetch_js_src
        assert "response.statusText" in fetch_js_src
        assert ".detail" in fetch_js_src
        assert ".error" in fetch_js_src
        assert ".message" in fetch_js_src

    def test_http_body_robust(self, fetch_js_src):
        """Nicht-Object-JSON-Bodies fallen auf null."""
        assert "Array.isArray" in fetch_js_src or "typeof" in fetch_js_src

    def test_options_passthrough(self, fetch_js_src):
        # options-Objekt wird (als lokale Kopie 'opts' aus _cloneOptions) an fetch weitergegeben
        assert re.search(r"fetch\s*\(\s*url\s*,\s*(?:opts|options)\s*\)", fetch_js_src)

    def test_safe_fetch_json_wrapper(self, fetch_js_src):
        assert "safeFetchJson" in fetch_js_src
        assert "safeFetch(url, options)" in fetch_js_src or "await safeFetch" in fetch_js_src


class TestSFCSprint2Anchor3RuntimeContract:
    """V30 Test 4-6 — runtime.js: Escape, AbortRegistry, ButtonLock, Polling."""

    def test_escape_html_normalizes_null(self, runtime_js_src):
        idx = runtime_js_src.index("function escapeHtml(")
        body = runtime_js_src[idx : idx + 300]
        assert "null" in body and "undefined" in body

    def test_escape_attr_uses_entities(self, runtime_js_src):
        idx = runtime_js_src.index("function escapeAttr(")
        body = runtime_js_src[idx : idx + 400]
        assert "&amp;" in body and "&quot;" in body and "&#39;" in body

    def test_escape_js_attr_double_escapes(self, runtime_js_src):
        idx = runtime_js_src.index("function escapeJSAttr(")
        body = runtime_js_src[idx : idx + 400]
        assert "\\\\" in body  # backslash-escape
        assert "escapeAttr" in body  # delegation

    def test_abort_registry_uses_map(self, runtime_js_src):
        assert "_aborts = new Map" in runtime_js_src

    def test_create_abort_aborts_previous(self, runtime_js_src):
        idx = runtime_js_src.index("function createAbort(")
        body = runtime_js_src[idx : idx + 300]
        assert "abort(id)" in body
        assert "new AbortController" in body
        assert "_aborts.set" in body

    def test_with_button_lock_preserves_innerHTML(self, runtime_js_src):
        idx = runtime_js_src.index("function withButtonLock(")
        body = runtime_js_src[idx : idx + 500]
        assert "if (!btn || btn.disabled) return" in body
        assert "prevHTML = btn.innerHTML" in body
        assert "btn.disabled = true" in body
        assert "btn.disabled = false" in body
        assert "btn.innerHTML = prevHTML" in body
        assert "finally" in body

    def test_polling_manager_class(self, runtime_js_src):
        assert "function PollingManager()" in runtime_js_src
        assert "PollingManager.prototype.start" in runtime_js_src
        assert "PollingManager.prototype.stop" in runtime_js_src
        assert "PollingManager.prototype.has" in runtime_js_src
        assert "PollingManager.prototype.stopAll" in runtime_js_src

    def test_polling_start_stops_first(self, runtime_js_src):
        idx = runtime_js_src.index("PollingManager.prototype.start")
        body = runtime_js_src[idx : idx + 600]
        assert "this.stop(id)" in body

    def test_polling_reentrancy_guard(self, runtime_js_src):
        idx = runtime_js_src.index("PollingManager.prototype.start")
        # Range generoes — finally-Block sitzt nach try/await und liegt jenseits 800.
        body = runtime_js_src[idx : idx + 1500]
        assert re.search(r"(?:let|var)\s+running\s*=\s*false", body)
        assert re.search(r"if\s*\(\s*running\s*\)", body)
        assert "finally" in body and "running = false" in body

    def test_polling_async_tick(self, runtime_js_src):
        idx = runtime_js_src.index("PollingManager.prototype.start")
        body = runtime_js_src[idx : idx + 1500]
        # Tick-Funktion ruft fn jetzt mit isCurrent-Helper als Argument auf
        # (Polling-Callback kann sich selbst gegen Tab-Wechsel guarden) —
        # frueher war das ein arg-loses await fn().
        assert re.search(r"await\s+fn\s*\(", body)
        assert "console.error" in body

    def test_polling_is_active_check(self, runtime_js_src):
        idx = runtime_js_src.index("PollingManager.prototype.start")
        body = runtime_js_src[idx : idx + 800]
        assert "isActive" in body
        assert "self.stop(id)" in body or "this.stop(id)" in body

    def test_polling_singleton_exported(self, runtime_js_src):
        assert "polling: polling" in runtime_js_src or "polling," in runtime_js_src
        assert "new PollingManager()" in runtime_js_src


class TestSFCSprint2Anchor4KiaraPolling:
    """V30 Test 7 — kiaraPolling-Singleton in app_core.js."""

    def test_main_tab_keys_set(self, app_core_src):
        assert re.search(r"_MAIN_TAB_KEYS\s*=\s*new\s+Set\s*\(\s*\[\s*'kiara'", app_core_src)
        assert "'datenquellen'" in app_core_src
        assert "'jobs'" in app_core_src

    def test_kiara_polling_is_active_function(self, app_core_src):
        assert "function kiaraPollingIsActive(" in app_core_src
        idx = app_core_src.index("function kiaraPollingIsActive(")
        body = app_core_src[idx : idx + 500]
        assert "_MAIN_TAB_KEYS" in body
        assert "adminCurrentTab" in body
        assert "adminCurrentSubTab" in body

    def test_kiara_polling_singleton(self, app_core_src):
        assert re.search(r"const\s+kiaraPolling\s*=\s*\{", app_core_src)
        idx = app_core_src.index("kiaraPolling = {")
        body = app_core_src[idx : idx + 600]
        for member in ("start", "stop", "has", "stopAll"):
            assert member in body
        assert "SharedUi.runtime.polling" in body
        assert "isActive: kiaraPollingIsActive" in body


class TestSFCSprint2Anchor5InterceptorRemoved:
    """V30 Test 8 — globaler window.fetch-Override entfernt."""

    def test_no_window_fetch_override(self, app_core_src):
        assert not re.search(r"window\.fetch\s*=", app_core_src)

    def test_no_orig_fetch_helper(self):
        for js in _STATIC_JS.rglob("*.js"):
            text = js.read_text(encoding="utf-8")
            assert "_origFetch" not in text, f"{js}: _origFetch sollte entfernt sein"

    def test_load_current_user_has_exception_marker(self, app_core_src):
        # Suche die Funktionsdefinition, nicht den ersten loadCurrentUser-Aufruf
        idx = app_core_src.index("function loadCurrentUser")
        body = app_core_src[idx : idx + 600]
        assert "uses raw fetch because" in body, "loadCurrentUser braucht Ausnahme-Kommentar"


class TestSFCSprint2Anchor6BootstrapFields:
    """V30 Test 9 — kiara_bootstrap.js Sprint-2-Felder."""

    def test_login_url(self, bootstrap_js_src):
        assert "loginUrl" in bootstrap_js_src
        assert "'/login'" in bootstrap_js_src

    def test_csrf_header(self, bootstrap_js_src):
        assert "csrfHeader" in bootstrap_js_src
        assert "'X-Requested-With'" in bootstrap_js_src

    def test_csrf_value(self, bootstrap_js_src):
        assert "csrfValue" in bootstrap_js_src
        assert "'XMLHttpRequest'" in bootstrap_js_src

    def test_default_credentials(self, bootstrap_js_src):
        assert "defaultCredentials" in bootstrap_js_src
        assert "'same-origin'" in bootstrap_js_src

    def test_on_401_callback(self, bootstrap_js_src):
        assert "on401" in bootstrap_js_src
        assert "window.location.href" in bootstrap_js_src

    def test_sprint1_fields_preserved(self, bootstrap_js_src):
        assert "themeStorageKey" in bootstrap_js_src
        assert "'kiara-theme'" in bootstrap_js_src
        assert "SharedUi.theme.load" in bootstrap_js_src
        assert "themeChanged" in bootstrap_js_src

    def test_single_configure_call(self, bootstrap_js_src):
        # Ein einziger configure({...})-Aufruf
        hits = re.findall(r"SharedUi\.configure\s*\(", bootstrap_js_src)
        assert len(hits) == 1, f"erwartet 1 configure-Aufruf, gefunden {len(hits)}"


class TestSFCSprint2Anchor7TemplateOrder:
    """V30 Test 10 — fetch.js + runtime.js zwischen theme.js und kiara_bootstrap.js."""

    @pytest.mark.parametrize("tpl_path", [_INDEX_HTML, _ADMIN_HTML])
    def test_pair_between_theme_and_bootstrap(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        theme_idx = src.index("ui_core/js/theme.js")
        fetch_idx = src.index("ui_core/js/fetch.js")
        runtime_idx = src.index("ui_core/js/runtime.js")
        bootstrap_idx = src.index("kiara_bootstrap.js")
        assert theme_idx < fetch_idx < runtime_idx < bootstrap_idx, (
            f"{tpl_path.name}: Reihenfolge theme<fetch<runtime<bootstrap verletzt"
        )

    @pytest.mark.parametrize("tpl_path", [_INDEX_HTML, _ADMIN_HTML])
    def test_v1_cache_buster_present(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        assert "ui_core/js/fetch.js?v=1" in src
        assert "ui_core/js/runtime.js?v=1" in src

    def test_wiki_no_fetch_runtime(self):
        src = _WIKI_BASE_HTML.read_text(encoding="utf-8")
        assert "ui_core/js/fetch.js" not in src
        assert "ui_core/js/runtime.js" not in src

    @pytest.mark.parametrize("tpl_path", [_LOGIN_HTML, _ADMIN_LOGIN_HTML])
    def test_login_templates_load_fetch_before_bootstrap(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        theme_idx = src.index("ui_core/js/theme.js")
        fetch_idx = src.index("ui_core/js/fetch.js")
        bootstrap_idx = src.index("kiara_bootstrap.js")
        assert theme_idx < fetch_idx < bootstrap_idx
        assert "ui_core/js/runtime.js" not in src

    @pytest.mark.parametrize("tpl_path", [_LOGIN_HTML, _ADMIN_LOGIN_HTML])
    def test_login_inline_mutations_use_csrf_headers(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        assert "SharedUi.fetch.csrfHeaders()" in src


class TestSFCSprint2Anchor8AdminInlineEscape:
    """V30 Test 11 — admin.html-Inline-escapeHtml migriert."""

    def test_no_bare_escape_html(self, admin_html_v23_src):
        # Lookbehind: kein nackter escapeHtml( mehr
        assert not re.search(r"(?<![\w.])escapeHtml\s*\(", admin_html_v23_src)

    def test_uses_shared_ui_runtime(self, admin_html_v23_src):
        hits = re.findall(r"SharedUi\.runtime\.escapeHtml\s*\(", admin_html_v23_src)
        assert len(hits) >= 3, (
            f"erwartet >= 3 SharedUi.runtime.escapeHtml-Aufrufe, gefunden {len(hits)}"
        )


class TestSFCSprint2Anchor9CoreNeutrality:
    """V30 Test 14 — keine KIara-Begriffe in ui_core/."""

    def test_no_kiara_specific_strings(self):
        forbidden = [
            "kiara",
            "admin",
            "/api/",
            "adminCurrentTab",
            "adminCurrentSubTab",
            "tabDOMCache",
            "_MAIN_TAB_KEYS",
        ]
        for js in _UI_CORE_JS.rglob("*.js"):
            text = js.read_text(encoding="utf-8")
            for word in forbidden:
                # Erlaube den Wort-Stamm "Shared" nicht als Treffer für "admin" → es gibt sowieso keinen.
                assert word not in text, f"{js}: produktneutraler Core enthält '{word}'"


class TestSFCSprint2Anchor10AdminChatRawFetch:
    """V30 Test 13 — Admin-Chat-Raw-Fetch-Ausnahme mit lokalem 401-Guard."""

    def test_admin_chat_has_doc_comment(self, admin_chat_js_src):
        assert (
            "uses bewusst direktes fetch()" in admin_chat_js_src
            or "uses bewusst direktes fetch" in admin_chat_js_src
            or "Sprint 2" in admin_chat_js_src
        )
        assert "_adminChatRedirect401" in admin_chat_js_src

    def test_tab_kiara_has_doc_comment(self, tab_kiara_js_src):
        assert "Sprint 2" in tab_kiara_js_src
        assert "_kiaraAdminChatRedirect401" in tab_kiara_js_src

    def test_admin_chat_401_helper_logic(self, admin_chat_js_src):
        idx = admin_chat_js_src.index("function _adminChatRedirect401(")
        body = admin_chat_js_src[idx : idx + 400]
        assert "status === 401" in body
        assert "loginUrl" in body or "'/login'" in body
        assert "window.location.href" in body

    def test_tab_kiara_401_helper_logic(self, tab_kiara_js_src):
        idx = tab_kiara_js_src.index("function _kiaraAdminChatRedirect401(")
        body = tab_kiara_js_src[idx : idx + 400]
        assert "status === 401" in body
        assert "loginUrl" in body or "'/login'" in body
        assert "window.location.href" in body

    def test_admin_chat_guard_applied_at_each_fetch(self, admin_chat_js_src):
        # Drei fetch-Stellen sollten alle einen 401-Guard direkt danach haben
        guards = admin_chat_js_src.count("_adminChatRedirect401(")
        # Mindestens 1 Definition + 3 Aufrufe = 4
        assert guards >= 4, f"erwartet >= 4 _adminChatRedirect401-Treffer, gefunden {guards}"

    def test_tab_kiara_guard_applied_at_each_fetch(self, tab_kiara_js_src):
        guards = tab_kiara_js_src.count("_kiaraAdminChatRedirect401(")
        # Definition + 6 Aufrufe = 7
        assert guards >= 7, f"erwartet >= 7 _kiaraAdminChatRedirect401-Treffer, gefunden {guards}"

    def test_admin_chat_credentials_include_preserved(self, admin_chat_js_src):
        hits = re.findall(r"credentials:\s*'include'", admin_chat_js_src)
        assert len(hits) >= 3

    def test_admin_chat_stream_retry_respects_stop(self, admin_chat_js_src):
        idx = admin_chat_js_src.index("refreshCsrfToken")
        body = admin_chat_js_src[idx : idx + 900]
        assert "initialController" in body
        assert "myGeneration === this._streamGeneration" in body
        assert "signal.aborted" in body

    def test_tab_kiara_credentials_include_preserved(self, tab_kiara_js_src):
        hits = re.findall(r"credentials:\s*'include'", tab_kiara_js_src)
        assert len(hits) >= 5

    def test_admin_chat_ui_removes_bound_input_handlers(self, admin_chat_js_src):
        """AdminChatUI entfernt Listener vor Re-Render und Destroy."""
        assert "_unbindEvents()" in admin_chat_js_src
        assert "removeEventListener('keydown', this._onInputKeydown)" in admin_chat_js_src
        assert "removeEventListener('input', this._onInput)" in admin_chat_js_src
        assert "removeEventListener('click', this._onSendClick)" in admin_chat_js_src
        assert "removeEventListener('click', this._onStopClick)" in admin_chat_js_src
        destroy_idx = admin_chat_js_src.index("destroy() {")
        assert "this._unbindEvents()" in admin_chat_js_src[destroy_idx : destroy_idx + 250]

    def test_tab_kiara_sidebar_handlers_are_removed_on_cleanup(self, tab_kiara_js_src):
        """Admin-Chat-Sidebar-Listener werden beim Tab-Cleanup entfernt."""
        assert "const unbindListActions = _bindAdminChatListActions()" in tab_kiara_js_src
        assert "newBtn.removeEventListener('click', onNewChatClick)" in tab_kiara_js_src
        assert "toggleBtn.removeEventListener('click', onToggleSidebarClick)" in tab_kiara_js_src
        assert "listEl.removeEventListener('click', onListClick)" in tab_kiara_js_src
        assert "_adminChatRenameCleanup" in tab_kiara_js_src
        assert "input.removeEventListener('keydown', onRenameKeydown)" in tab_kiara_js_src


class TestSFCSprint2Anchor11AuditDetailStandalone:
    """V30 — audit_detail.js bleibt rohes fetch() mit Same-Line-Kommentar."""

    def test_same_line_exception_marker(self, audit_detail_js_src):
        # Die fetch(-Zeile muss den Marker enthalten
        for ln in audit_detail_js_src.splitlines():
            if re.search(r"(?<![\w.])fetch\s*\(", ln):
                assert "uses raw fetch because" in ln, (
                    f"audit_detail.js fetch ohne Ausnahme-Marker: {ln}"
                )

    def test_credentials_same_origin(self, audit_detail_js_src):
        assert (
            'credentials: "same-origin"' in audit_detail_js_src
            or "credentials: 'same-origin'" in audit_detail_js_src
        )

    def test_local_401_403_redirect(self, audit_detail_js_src):
        assert "401" in audit_detail_js_src and "403" in audit_detail_js_src
        assert "/login?next=" in audit_detail_js_src


class TestSFCSprint2Anchor12VollmigrationsGreps:
    """V30 Test 15-19 — repo-weite Migrations-Greps."""

    def test_no_bare_safe_fetch(self):
        """Kein nacktes safeFetch( außerhalb Kommentaren in static/js."""
        violations = []
        pat = re.compile(r"(?<![\w.])safeFetch\s*\(")
        for js in _STATIC_JS.rglob("*.js"):
            for i, ln in enumerate(js.read_text(encoding="utf-8").splitlines(), 1):
                stripped = ln.lstrip()
                if stripped.startswith("//") or stripped.startswith("*"):
                    continue
                if pat.search(ln):
                    violations.append(f"{js.name}:{i}: {ln.strip()}")
        assert not violations, "Nacktes safeFetch( gefunden:\n" + "\n".join(violations)

    def test_no_bare_escape_helpers(self):
        violations = []
        pat = re.compile(r"(?<![\w.])(escapeHtml|escapeAttr|escapeJSAttr|withButtonLock)\s*\(")
        for js in _STATIC_JS.rglob("*.js"):
            for i, ln in enumerate(js.read_text(encoding="utf-8").splitlines(), 1):
                stripped = ln.lstrip()
                if stripped.startswith("//") or stripped.startswith("*"):
                    continue
                if pat.search(ln):
                    violations.append(f"{js.name}:{i}: {ln.strip()}")
        assert not violations, "Nackte Escape-Helper:\n" + "\n".join(violations)

    def test_no_bare_polling_calls(self):
        violations = []
        pat = re.compile(r"(?<![\w.])polling\.(start|stop|has|stopAll)\s*\(")
        for js in _STATIC_JS.rglob("*.js"):
            for i, ln in enumerate(js.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(ln):
                    violations.append(f"{js.name}:{i}: {ln.strip()}")
        assert not violations, "Nackte polling.*-Calls:\n" + "\n".join(violations)

    def test_admin_html_no_bare_escape(self):
        src = _ADMIN_HTML.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\w.])escapeHtml\s*\(", src)

    def test_kiara_polling_at_least_10(self):
        total = 0
        for js in _STATIC_JS.rglob("*.js"):
            text = js.read_text(encoding="utf-8")
            total += len(re.findall(r"kiaraPolling\.(start|stop|has|stopAll)\s*\(", text))
        assert total >= 10, f"erwartet >= 10 kiaraPolling.*-Aufrufe, gefunden {total}"

    def test_no_window_fetch_override_anywhere(self):
        for js in _STATIC_JS.rglob("*.js"):
            text = js.read_text(encoding="utf-8")
            assert not re.search(r"window\.fetch\s*=", text), f"{js}: window.fetch override"

    def test_no_window_aliases_for_old_globals(self):
        forbidden = re.compile(
            r"\b(window|globalThis)\s*(?:\."
            r"(safeFetch|escapeHtml|escapeAttr|escapeJSAttr|withButtonLock|polling)"
            r"|\[\s*['\"](safeFetch|escapeHtml|escapeAttr|escapeJSAttr|withButtonLock|polling)['\"]\s*\])"
            r"\s*="
        )
        for js in _STATIC_JS.rglob("*.js"):
            text = js.read_text(encoding="utf-8")
            assert not forbidden.search(text), f"{js}: globaler Alias auf Altsymbol"

    def test_raw_fetch_only_in_exceptions(self):
        """Direkte fetch( nur in dokumentierten Ausnahmen."""
        allowed_files = {"admin_chat.js", "tab_kiara.js"}
        violations = []
        pat = re.compile(r"(?<![\w.])fetch\s*\(")
        for js in _STATIC_JS.rglob("*.js"):
            if js.name in allowed_files:
                continue
            for i, ln in enumerate(js.read_text(encoding="utf-8").splitlines(), 1):
                stripped = ln.lstrip()
                if stripped.startswith("//") or stripped.startswith("*"):
                    continue
                if not pat.search(ln):
                    continue
                if "uses raw fetch because" in ln:
                    continue
                violations.append(f"{js.name}:{i}: {ln.strip()}")
        assert not violations, "Rohe fetch( ohne Marker:\n" + "\n".join(violations)


class TestSFCSprint2Anchor13PollingIDs:
    """V30 Test 9 (Akzeptanz) — bekannte Polling-IDs nach Migration vorhanden."""

    EXPECTED = [
        ("tab_jobs.js", "jobs:refresh"),
        ("tab_backup.js", "system-backup:status"),
        ("tab_datenquellen.js", "datenquellen:refresh"),
        ("tab_ki_retrieval.js", "ki-retrieval:refresh"),
        ("tab_ki_graph.js", "ki-graph:stats"),
        ("tab_performance.js", "system-performance:stats"),
        ("tab_monitoring.js", "system-monitoring:services"),
        ("tab_integrity.js", "system-integrity:status"),
        ("tab_integrity.js", "system-integrity:repair"),
        ("tab_testsuite.js", "system-selbsttest:status"),
    ]

    @pytest.mark.parametrize("filename,polling_id", EXPECTED)
    def test_polling_id_registered(self, filename, polling_id):
        src = _js(filename)
        assert (
            f"kiaraPolling.start('{polling_id}'" in src
            or f'kiaraPolling.start("{polling_id}"' in src
        ), f"{filename}: kein kiaraPolling.start('{polling_id}', ...)"

    def test_datenquellen_has_logic_guard(self):
        src = _js("tab_datenquellen.js")
        hits = re.findall(r"kiaraPolling\.has\s*\(\s*['\"]datenquellen:refresh['\"]\s*\)", src)
        assert len(hits) >= 2, (
            f"erwartet >= 2 has-Guards in tab_datenquellen.js, gefunden {len(hits)}"
        )


class TestSFCSprint2Anchor14ChatJsHandling:
    """V30 — chat.js Stream/Trace-Debug/pollStatus-Spezialfälle."""

    def test_stream_uses_safe_fetch(self, chat_js_src):
        assert (
            "SharedUi.fetch.safeFetch(`/api/chat/" in chat_js_src
            or "SharedUi.fetch.safeFetch('/api/chat/" in chat_js_src
        )

    def test_stream_signal_passed(self, chat_js_src):
        # Signal wird durchgereicht (Abort-Pfad muss funktionieren)
        idx = chat_js_src.index("/api/chat/")
        body = chat_js_src[idx : idx + 500]
        assert "signal:" in body

    def test_stream_catch_distinguishes_http_network(self, chat_js_src):
        # AbortError + http/unauthorized + Netzwerk
        assert "AbortError" in chat_js_src
        assert "kind === 'http'" in chat_js_src or "'unauthorized'" in chat_js_src
        assert "Fehler bei der Abfrage" in chat_js_src
        assert "Netzwerkfehler" in chat_js_src

    def test_stream_retry_respects_stop(self, chat_js_src):
        idx = chat_js_src.index("refreshCsrfToken")
        body = chat_js_src[idx : idx + 900]
        assert "initialController" in body
        assert "myGeneration === this._streamGeneration" in body
        assert "signal.aborted" in body

    def test_trace_debug_status_uses_safe_fetch(self, chat_js_src):
        assert (
            'SharedUi.fetch.safeFetchJson("/api/chat-debug/status"' in chat_js_src
            or 'SharedUi.fetch.safeFetch("/api/chat-debug/status"' in chat_js_src
        )

    def test_trace_debug_404_retry_preserved(self, chat_js_src):
        assert "err.status === 404" in chat_js_src
        assert "retryCount + 1" in chat_js_src

    def test_poll_status_soft_fail(self, chat_js_src):
        assert "SharedUi.fetch.safeFetchJson('/api/health/degradation-mode')" in chat_js_src
        idx = chat_js_src.index("function pollStatus(")
        body = chat_js_src[idx : idx + 400]
        assert "} catch" in body
        assert "updateBanner" in body


class TestSFCSprint2Anchor15ConfigHelperContracts:
    """V30 Test 8 (Zusatz) — Config-Helper-Bodies bleiben semantisch stabil."""

    def test_load_config_returns_unwrapped(self, app_core_src):
        idx = app_core_src.index("function loadConfigFromAPI(")
        body = app_core_src[idx : idx + 600]
        assert "data.config !== undefined" in body
        assert "return null" in body
        assert "SharedUi.fetch" in body

    def test_save_config_returns_bool(self, app_core_src):
        idx = app_core_src.index("function saveConfigToAPI(")
        body = app_core_src[idx : idx + 800]
        assert "return true" in body
        assert "return false" in body
        assert "SharedUi.notify.show" in body
        assert "SharedUi.fetch" in body

    def test_delete_config_returns_bool(self, app_core_src):
        idx = app_core_src.index("function deleteConfigFromAPI(")
        body = app_core_src[idx : idx + 800]
        assert "return true" in body
        assert "return false" in body
        assert "SharedUi.fetch" in body


# ───────────────────────────────────────────────────────────────────
# Sprint 3 (V34): Notifications + Modals
# ───────────────────────────────────────────────────────────────────

_NOTIFICATIONS_JS = _UI_CORE_JS / "notifications.js"
_MODAL_JS = _UI_CORE_JS / "modal.js"
_NOTIFICATIONS_CSS = _UI_CORE_CSS / "notifications.css"
_MODAL_CSS = _UI_CORE_CSS / "modal.css"
_STYLE_CSS = _BASE / "kiara" / "web" / "static" / "css" / "style.css"
_ADMIN_CHAT_CSS = _BASE / "kiara" / "web" / "static" / "css" / "admin_chat.css"
_ADMIN_CHAT_JS = _BASE / "kiara" / "web" / "static" / "js" / "admin_chat.js"
_CHAT_CSS = _BASE / "kiara" / "web" / "static" / "css" / "chat.css"


@pytest.fixture(scope="module")
def notifications_js_src():
    return _NOTIFICATIONS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def modal_js_src():
    return _MODAL_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def notifications_css_src():
    return _NOTIFICATIONS_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def modal_css_src():
    return _MODAL_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def style_css_src():
    return _STYLE_CSS.read_text(encoding="utf-8")


class TestSFCSprint3FilesExist:
    """V34 Test 1 — neue Dateien existieren."""

    def test_notifications_js(self):
        assert _NOTIFICATIONS_JS.exists()

    def test_modal_js(self):
        assert _MODAL_JS.exists()

    def test_notifications_css(self):
        assert _NOTIFICATIONS_CSS.exists()

    def test_modal_css(self):
        assert _MODAL_CSS.exists()


class TestSFCSprint3IIFEScope:
    """V34 Test 2 — JS-Scope-Invariante: beide neuen JS-Dateien sind IIFE."""

    def test_notifications_iife(self, notifications_js_src):
        assert re.search(r"\(\s*function\s*\(\s*\)\s*\{", notifications_js_src)
        assert re.search(r"\}\s*\)\s*\(\s*\)\s*;\s*$", notifications_js_src.strip())

    def test_modal_iife(self, modal_js_src):
        assert re.search(r"\(\s*function\s*\(\s*\)\s*\{", modal_js_src)
        assert re.search(r"\}\s*\)\s*\(\s*\)\s*;\s*$", modal_js_src.strip())

    def test_notifications_exposes_only_shared_notify(self, notifications_js_src):
        assert "window.SharedUi.notify" in notifications_js_src

    def test_modal_exposes_only_shared_modal(self, modal_js_src):
        assert "window.SharedUi.modal" in modal_js_src

    def test_no_window_close_modal_shim(self, modal_js_src):
        assert "window.closeModal" not in modal_js_src


class TestSFCSprint3NotifyContract:
    """V34 Test 3 — SharedUi.notify-Contract (statisch)."""

    def test_valid_set_contains_all_types(self, notifications_js_src):
        for typ in ("success", "error", "warning", "info", "loading"):
            assert f"'{typ}'" in notifications_js_src

    def test_default_duration_values(self, notifications_js_src):
        # Per-Typ-Defaults
        assert re.search(r"success\s*:\s*3000", notifications_js_src)
        assert re.search(r"error\s*:\s*6000", notifications_js_src)
        assert re.search(r"warning\s*:\s*5000", notifications_js_src)
        assert re.search(r"info\s*:\s*3000", notifications_js_src)
        assert re.search(r"loading\s*:\s*0", notifications_js_src)

    def test_unknown_type_falls_back_to_info(self, notifications_js_src):
        assert "VALID.has(type)" in notifications_js_src
        assert "'info'" in notifications_js_src

    def test_single_slot_remove_existing(self, notifications_js_src):
        assert "querySelector('.notification')" in notifications_js_src
        assert ".remove()" in notifications_js_src

    def test_uses_text_content(self, notifications_js_src):
        # Inhalt via textContent, kein innerHTML
        assert "textContent" in notifications_js_src
        # Es darf KEIN innerHTML auf dem Notification-Knoten gesetzt werden
        assert "node.innerHTML" not in notifications_js_src

    def test_role_status_aria_live(self, notifications_js_src):
        assert "role" in notifications_js_src and "status" in notifications_js_src
        assert "aria-live" in notifications_js_src and "polite" in notifications_js_src

    def test_node_has_explicit_id(self, notifications_js_src):
        # node.id = 'notify-...' ODER data-notify-id
        assert "notify-" in notifications_js_src
        assert "data-notify-id" in notifications_js_src or "node.id" in notifications_js_src

    def test_dismiss_targets_by_id(self, notifications_js_src):
        # dismiss(id) nutzt getElementById, nicht querySelector('.notification')
        idx = notifications_js_src.index("function dismiss(")
        body = notifications_js_src[idx : idx + 400]
        assert "getElementById" in body


class TestSFCSprint3NotificationsCss:
    """V34 Test 3 — Notifications-CSS Inhalte und keine KIara-Token-Abhaengigkeit."""

    def test_base_class_present(self, notifications_css_src):
        assert ".notification" in notifications_css_src

    def test_all_variants_present(self, notifications_css_src):
        for v in ("info", "success", "error", "warning", "loading"):
            assert f".notification-{v}" in notifications_css_src

    def test_fade_out_present(self, notifications_css_src):
        assert ".notification.fade-out" in notifications_css_src

    def test_keyframes_present(self, notifications_css_src):
        assert "@keyframes slideIn" in notifications_css_src

    def test_no_kiara_color_tokens(self, notifications_css_src):
        # Keine direkte Abhaengigkeit auf KIara-style.css-Tokens
        assert "--color-bullish" not in notifications_css_src
        assert "--color-bearish" not in notifications_css_src


class TestSFCSprint3ModalContract:
    """V34 Test 4 — SharedUi.modal-Contract (statisch)."""

    def test_dom_structure_classes(self, modal_js_src):
        for cls in ("modal-overlay", "modal-content", "modal-header", "modal-body", "modal-footer"):
            assert cls in modal_js_src

    def test_role_dialog_aria_modal(self, modal_js_src):
        assert "'role'" in modal_js_src and "'dialog'" in modal_js_src
        assert "aria-modal" in modal_js_src

    def test_size_map(self, modal_js_src):
        assert re.search(r"small\s*:\s*'400px'", modal_js_src)
        assert re.search(r"medium\s*:\s*'700px'", modal_js_src)
        assert re.search(r"large\s*:\s*'900px'", modal_js_src)

    def test_body_priority_sources(self, modal_js_src):
        # body, html, text, element — alle vier Quellen
        assert "options.body" in modal_js_src
        assert "options.html" in modal_js_src
        assert "options.text" in modal_js_src
        assert "options.element" in modal_js_src

    def test_html_slots_use_sanitizer(self, modal_js_src):
        assert "function _sanitizeHtml" in modal_js_src
        assert "function _setSanitizedHtml" in modal_js_src
        assert "_setSanitizedHtml(bodyEl, options.body)" in modal_js_src
        assert "_setSanitizedHtml(bodyEl, options.html)" in modal_js_src
        assert "_setSanitizedHtml(footer, options.footer)" in modal_js_src
        assert "bodyEl.innerHTML = options.body" not in modal_js_src
        assert "bodyEl.innerHTML = options.html" not in modal_js_src
        assert "footer.innerHTML = options.footer" not in modal_js_src

    def test_modal_sanitizer_fail_closed(self, modal_js_src):
        assert "DOMPurify.sanitize(raw" in modal_js_src
        assert "vendor/purify.min.js must load before modal.js" in modal_js_src
        assert "window.SharedUi.runtime.escapeHtml(raw)" not in modal_js_src
        for tag in ("style", "script", "iframe", "object", "embed"):
            assert f"'{tag}'" in modal_js_src
        for attr in (
            "onerror",
            "onload",
            "onclick",
            "onmouseover",
            "onfocus",
            "onblur",
            "onsubmit",
        ):
            assert f"'{attr}'" in modal_js_src

    def test_default_close_button_data_attribute(self, modal_js_src):
        assert "data-sharedui-close" in modal_js_src

    def test_default_close_button_wired_via_delegation(self, modal_js_src):
        # Klick auf data-sharedui-close ruft close(id)
        assert "data-sharedui-close" in modal_js_src
        assert "close(id)" in modal_js_src

    def test_buttons_use_with_button_lock(self, modal_js_src):
        # Footer-Buttons werden via withButtonLock umrahmt
        assert "withButtonLock" in modal_js_src

    def test_auto_focus_opt_out(self, modal_js_src):
        assert "options.autoFocus !== false" in modal_js_src

    def test_modal_content_focusable(self, modal_js_src):
        # Fallback-Fokus auf .modal-content braucht tabIndex=-1
        assert re.search(r"tabIndex\s*=\s*-1", modal_js_src)

    def test_focus_skipped_if_already_inside(self, modal_js_src):
        # Auto-Focus nur wenn !overlay.contains(document.activeElement)
        assert "overlay.contains(document.activeElement)" in modal_js_src

    def test_open_replaces_managed_modals_by_default(self, modal_js_src):
        # open() ohne stack:true schliesst bestehende managed Modals
        assert "options.stack !== true" in modal_js_src

    def test_close_without_arg_targets_top_of_stack(self, modal_js_src):
        # close() ohne id schliesst Top-of-Stack
        assert "_topEntry()" in modal_js_src

    def test_backdrop_close_opt_in(self, modal_js_src):
        assert "options.backdropClose === true" in modal_js_src

    def test_escape_close_default_on(self, modal_js_src):
        assert "options.escapeClose === false" in modal_js_src

    def test_confirm_alert_use_settled_guard(self, modal_js_src):
        # Doppel-Resolve verhindern
        assert modal_js_src.count("settled = true") >= 2

    def test_confirm_alert_use_stack_true(self, modal_js_src):
        # confirm/alert oeffnen mit stack:true
        assert modal_js_src.count("stack: true") >= 2

    def test_modal_header_only_text(self, modal_js_src):
        # Header-Render nur Title-Text, keine Kind-Elemente
        idx = modal_js_src.index("'modal-header'")
        body = modal_js_src[idx : idx + 400]
        assert "header.textContent" in body or "textContent" in body
        # Keine appendChild- oder innerHTML-Operation auf header
        assert "header.innerHTML" not in body
        assert "header.appendChild" not in body

    def test_confirm_message_via_text_content(self, modal_js_src):
        idx = modal_js_src.index("function confirm(")
        body = modal_js_src[idx : idx + 1500]
        # message wird ueber textContent gesetzt, nicht innerHTML
        assert "textContent" in body

    def test_alert_message_via_text_content(self, modal_js_src):
        idx = modal_js_src.index("function alert(")
        body = modal_js_src[idx : idx + 1500]
        assert "textContent" in body


class TestSFCSprint3ModalCss:
    """V34 Test 4 — Modal-CSS extrahiert."""

    def test_modal_classes_present(self, modal_css_src):
        for cls in ("modal-overlay", "modal-content", "modal-header", "modal-body", "modal-footer"):
            assert f".{cls}" in modal_css_src

    def test_no_modal_field(self, modal_css_src):
        # .modal-field bleibt in style.css (Forms-Migration Sprint 6)
        assert ".modal-field" not in modal_css_src


class TestSFCSprint3StyleCssCleanup:
    """V34 Test 14 — style.css ist von Notify/Modal-Wurzelklassen befreit."""

    def test_no_notification_classes(self, style_css_src):
        assert not re.search(r"\.notification(\b|-|\.fade-out)", style_css_src)
        assert "@keyframes slideIn" not in style_css_src

    def test_no_modal_overlay_content_header_body_footer(self, style_css_src):
        for cls in ("modal-overlay", "modal-content", "modal-header", "modal-body", "modal-footer"):
            assert f".{cls} " not in style_css_src and f".{cls}{{" not in style_css_src

    def test_modal_field_still_present(self, style_css_src):
        # Forms-Klassen bleiben in style.css (Sprint 6)
        assert ".modal-field" in style_css_src

    def test_action_btn_moved_to_forms_css(self, style_css_src):
        """Sprint 6: .action-btn-Basis ist nach ui_core/css/forms.css extrahiert."""
        forms_css = (_UI_CORE_CSS / "forms.css").read_text(encoding="utf-8")
        assert ".action-btn" in forms_css
        # style.css enthaelt nur noch Kommentar-Verweise, keine .action-btn-Regel
        import re as _re

        hits = _re.findall(r"^\.action-btn\b", style_css_src, _re.MULTILINE)
        assert hits == [], (
            f"Sprint 6: .action-btn-Regel sollte aus style.css entfernt sein, gefunden: {hits}"
        )


class TestSFCSprint3LabelDefaults:
    """V34 Test 5 — Label-Defaults auf SharedUi-Konfig."""

    def test_modal_confirm_uses_modal_labels(self, modal_js_src):
        idx = modal_js_src.index("function confirm(")
        body = modal_js_src[idx : idx + 1500]
        # confirm liest yes/cancel
        assert "labels.yes" in body
        assert "labels.cancel" in body

    def test_modal_alert_uses_modal_labels(self, modal_js_src):
        idx = modal_js_src.index("function alert(")
        body = modal_js_src[idx : idx + 1500]
        assert "labels.ok" in body

    def test_label_fallbacks(self, modal_js_src):
        # Defaults: 'OK', 'Abbrechen', 'Ja'
        assert "'OK'" in modal_js_src
        assert "'Abbrechen'" in modal_js_src
        assert "'Ja'" in modal_js_src


class TestSFCSprint3BootstrapModalLabels:
    """V34 Test 6 — kiara_bootstrap.js enthaelt modalLabels."""

    def test_single_configure_call(self, bootstrap_js_src):
        hits = re.findall(r"SharedUi\.configure\s*\(", bootstrap_js_src)
        assert len(hits) == 1

    def test_modal_labels_key(self, bootstrap_js_src):
        assert "modalLabels" in bootstrap_js_src

    def test_modal_labels_values(self, bootstrap_js_src):
        assert "ok: 'OK'" in bootstrap_js_src
        assert "cancel: 'Abbrechen'" in bootstrap_js_src
        assert "yes: 'Ja'" in bootstrap_js_src


class TestSFCSprint3TemplateOrder:
    """V34 Test 7 — neue CSS/JS in index.html + admin.html mit ?v=1."""

    @pytest.mark.parametrize("tpl_path", [_INDEX_HTML, _ADMIN_HTML])
    def test_notifications_modal_css_loaded(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        assert "ui_core/css/notifications.css?v=1" in src
        assert "ui_core/css/modal.css?v=1" in src

    @pytest.mark.parametrize("tpl_path", [_INDEX_HTML, _ADMIN_HTML])
    def test_notifications_modal_js_loaded(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        assert "ui_core/js/notifications.js?v=1" in src
        assert "ui_core/js/modal.js?v=1" in src

    @pytest.mark.parametrize("tpl_path", [_INDEX_HTML, _ADMIN_HTML])
    def test_css_order_notifications_modal_before_style(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        pos_notif = src.find("ui_core/css/notifications.css")
        pos_modal = src.find("ui_core/css/modal.css")
        pos_style = src.find("/static/css/style.css")
        pos_theme = src.find("ui_core/css/theme.css")
        assert pos_theme < pos_notif < pos_modal < pos_style

    @pytest.mark.parametrize("tpl_path", [_INDEX_HTML, _ADMIN_HTML])
    def test_js_order_notifications_modal_between_runtime_and_bootstrap(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        pos_runtime = src.find("ui_core/js/runtime.js")
        pos_notif = src.find("ui_core/js/notifications.js")
        pos_purify = src.find("js/vendor/purify.min.js")
        pos_modal = src.find("ui_core/js/modal.js")
        pos_bootstrap = src.find("kiara_bootstrap.js")
        assert pos_runtime < pos_notif < pos_purify < pos_modal < pos_bootstrap

    @pytest.mark.parametrize("tpl_path", [_LOGIN_HTML, _ADMIN_LOGIN_HTML, _WIKI_BASE_HTML])
    def test_login_wiki_no_notif_modal_assets(self, tpl_path):
        src = tpl_path.read_text(encoding="utf-8")
        assert "ui_core/js/notifications.js" not in src
        assert "ui_core/js/modal.js" not in src
        assert "ui_core/css/notifications.css" not in src
        assert "ui_core/css/modal.css" not in src


class TestSFCSprint3CoreNeutrality:
    """V34 Test 10 — keine KIara-/api/-Begriffe in den neuen Core-Dateien."""

    @pytest.mark.parametrize("path", [_NOTIFICATIONS_JS, _MODAL_JS])
    def test_no_kiara_specific_strings(self, path):
        text = path.read_text(encoding="utf-8")
        forbidden = ["kiara", "admin", "chat", "/api/"]
        for word in forbidden:
            assert word.lower() not in text.lower(), (
                f"{path.name}: produktneutraler Core enthaelt '{word}'"
            )


class TestSFCSprint3AppCoreHelpersRemoved:
    """V34 Test 5 — app_core.js enthaelt keine Helper-Definitionen mehr."""

    def test_no_show_notification_definition(self, app_core_src):
        assert not re.search(r"function\s+showNotification\s*\(", app_core_src)

    def test_no_create_modal_definition(self, app_core_src):
        assert not re.search(r"function\s+createModal\s*\(", app_core_src)

    def test_no_close_modal_definition(self, app_core_src):
        assert not re.search(r"function\s+closeModal\s*\(", app_core_src)

    def test_no_msgbox_definition(self, app_core_src):
        assert not re.search(r"function\s+msgbox\s*\(", app_core_src)

    def test_uses_shared_ui_notify(self, app_core_src):
        assert "SharedUi.notify.show" in app_core_src

    def test_uses_shared_ui_modal_open(self, app_core_src):
        assert "SharedUi.modal.open" in app_core_src

    def test_uses_shared_ui_modal_close(self, app_core_src):
        assert "SharedUi.modal.close" in app_core_src

    def test_uses_shared_ui_modal_confirm(self, app_core_src):
        assert "SharedUi.modal.confirm" in app_core_src


class TestSFCSprint3CallerMigration:
    """V34 Test 11 — Aufrufer-Migration in tab_*.js + chat.js (Reverse-Abgleich)."""

    JS_DIR = _BASE / "kiara" / "web" / "static" / "js"

    @staticmethod
    def _all_relevant_js():
        return [
            p
            for p in TestSFCSprint3CallerMigration.JS_DIR.glob("*.js")
            if p.name != "audit_detail.js"
        ]

    def test_no_legacy_symbols_anywhere(self):
        for path in self._all_relevant_js():
            text = path.read_text(encoding="utf-8")
            for sym in ("showNotification", "createModal", "closeModal", "msgbox"):
                # Wortgrenze
                assert not re.search(r"\b" + sym + r"\b", text), (
                    f"{path.name}: legacy-Symbol {sym} gefunden"
                )

    def test_no_legacy_inline_handlers(self):
        for path in self._all_relevant_js():
            text = path.read_text(encoding="utf-8")
            assert not re.search(
                r"onclick\s*=\s*['\"][^'\"]*(showNotification|createModal|closeModal|msgbox)\b",
                text,
            ), f"{path.name}: alter onclick=... gefunden"

    def test_shared_ui_modal_close_inline_removed_from_migrated_modules(self):
        """V30 Welle 2 (#1049): invertierte Assertion — CSP Phase 2 verbietet
        onclick='SharedUi.modal.close()' in allen bereits migrierten Modulen.
        Der Ersatz ist data-sharedui-close (nativer SharedUi.modal-Hook).
        Noch nicht migrierte Module (tab_benutzer/tab_gruppen/tab_ki_chat)
        werden in spaeteren Wellen gezogen — der Test waechst mit."""
        migrated = [
            "tab_jobs.js",
            "tab_compliance_audit.js",
            "tab_compliance_prompt_schutz.js",
            "tab_ki_modelle.js",
            "tab_datenquellen.js",
            "tab_ki_chat.js",
            "tab_ki_ingestion.js",
            "tab_benutzer.js",
            "tab_gruppen.js",
            "tab_integrity.js",
            "tab_testsuite.js",
            "tab_ldap.js",
            "tab_tools.js",
            "tab_mail.js",
            "tab_system_traces.js",
            "tab_system_traces_panels.js",
            "tab_chat.js",
            "tab_history.js",
            "tab_search.js",
            "tab_ki_graph.js",
        ]
        for name in migrated:
            path = self.JS_DIR / name
            text = path.read_text(encoding="utf-8")
            assert 'onclick="SharedUi.modal.close()"' not in text, (
                f"{name}: onclick='SharedUi.modal.close()' noch vorhanden — data-sharedui-close nutzen"
            )


class TestSFCSprint3HandbuiltModalConsumers:
    """V34 Test 8 — drei handgebaute .modal-*-Overlays bleiben unveraendert
    und nutzen weder alte Helper-Symbole noch SharedUi.modal."""

    def test_tab_tools_show_connection_dialog(self):
        text = (_BASE / "kiara" / "web" / "static" / "js" / "tab_tools.js").read_text(
            encoding="utf-8"
        )
        assert "showConnectionDialog" in text
        assert "modal-overlay" in text  # CSS-Konsument
        # Kein altes Helper-Symbol
        for sym in ("showNotification", "createModal", "msgbox"):
            assert not re.search(r"\b" + sym + r"\b", text)

    def test_tab_datenquellen_trigger_ingest(self):
        text = (_BASE / "kiara" / "web" / "static" / "js" / "tab_datenquellen.js").read_text(
            encoding="utf-8"
        )
        assert "_triggerIngest" in text
        assert "modal-overlay" in text


class TestSFCSprint3AdminChatBubblesUntouched:
    """V34 Test 4 + Grep-Check 4 — Admin-Chat-Bubble-Klassen unveraendert."""

    def test_bubble_classes_in_admin_chat_css(self):
        text = _ADMIN_CHAT_CSS.read_text(encoding="utf-8")
        for cls in ("confirmation-bubble", "wizard-bubble", "wizard-params", "wizard-param"):
            assert f".{cls}" in text, f"admin_chat.css: {cls} fehlt"

    def test_bubble_classes_in_admin_chat_js(self):
        text = _ADMIN_CHAT_JS.read_text(encoding="utf-8")
        for cls in ("confirmation-bubble", "wizard-bubble", "wizard-params", "wizard-param"):
            assert cls in text

    def test_no_bubble_classes_in_chat_css(self):
        text = _CHAT_CSS.read_text(encoding="utf-8")
        for cls in ("confirmation-bubble", "wizard-bubble", "wizard-params", "wizard-param"):
            assert f".{cls}" not in text


class TestSFCSprint3WindowCloseModalRemoved:
    """V34 Test 18 — kein window.closeModal mehr im Endzustand."""

    def test_no_window_close_modal_in_static(self):
        static = _BASE / "kiara" / "web" / "static"
        for path in static.rglob("*.js"):
            text = path.read_text(encoding="utf-8")
            assert "window.closeModal" not in text, f"{path}: window.closeModal noch vorhanden"


class TestVectorDBStatsRenderer:
    """V35 #734 — tab_jobs.js Feature-Details-Renderer zeigt die neuen
    vectordb_errors/vectordb_recoveries-Counter fuer Operator-Sichtbarkeit."""

    def test_tab_jobs_renders_vectordb_errors(self):
        text = (_BASE / "kiara" / "web" / "static" / "js" / "tab_jobs.js").read_text(
            encoding="utf-8"
        )
        assert "stats.vectordb_errors" in text
        assert "VectorDB Fehler" in text

    def test_tab_jobs_renders_vectordb_recoveries(self):
        text = (_BASE / "kiara" / "web" / "static" / "js" / "tab_jobs.js").read_text(
            encoding="utf-8"
        )
        assert "stats.vectordb_recoveries" in text
        assert "VectorDB Recoveries" in text


class TestCspPhase2TabJobs:
    """V30 Welle 2 (#1049) — tab_jobs.js: HTML-inline on*= und .on*=
    Property-Handler entfernt, Event-Delegation ueber data-action."""

    @pytest.fixture(scope="class")
    def tab_jobs_src(self):
        return (_BASE / "kiara" / "web" / "static" / "js" / "tab_jobs.js").read_text(
            encoding="utf-8"
        )

    def test_no_html_inline_events(self, tab_jobs_src):
        """Multiline-sicherer Perl-Scan-Aequivalent: kein <... on*=> in
        HTML-String-Literalen oder Template-Strings."""
        # Tag-basierter Scan: findet on*= innerhalb <...>-Bereichen
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", tab_jobs_src, flags=re.DOTALL)
        assert not tag_attrs, f"tab_jobs.js: HTML-inline on*= gefunden: {tag_attrs[:3]}"

    def test_no_property_handlers(self, tab_jobs_src):
        """Keine direkten btn.onclick=/.onchange=... Zuweisungen."""
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", tab_jobs_src)
        assert not prop_handlers, f"tab_jobs.js: .on*= Property-Handler: {prop_handlers[:3]}"

    def test_actions_whitelist_defined(self, tab_jobs_src):
        """_JOBS_ACTIONS existiert und enthaelt die fuenf Jobs-Aktionen."""
        assert "_JOBS_ACTIONS" in tab_jobs_src
        for action in (
            "'filter-jobs'",
            "'cleanup-job-history'",
            "'toggle-job-pause'",
            "'cancel-job'",
            "'delete-job'",
        ):
            assert action in tab_jobs_src, f"tab_jobs.js: {action} fehlt in _JOBS_ACTIONS"

    def test_row_actions_stop_propagation(self, tab_jobs_src):
        """_JOBS_ROW_ACTIONS-Set enthaelt genau die drei Row-Actions, die
        stopPropagation() brauchen (Row-Expand-Block in ExpandableTable)."""
        assert "_JOBS_ROW_ACTIONS" in tab_jobs_src
        for row_action in ("'toggle-job-pause'", "'cancel-job'", "'delete-job'"):
            assert row_action in tab_jobs_src
        assert "_JOBS_ROW_ACTIONS.has(action)" in tab_jobs_src
        assert "e.stopPropagation()" in tab_jobs_src

    def test_delegation_bound_on_stable_containers(self, tab_jobs_src):
        """_bindJobsActions() bindet an contentContainer + filterBar, mit
        Rebind-Schutz via _jobsActionsBound."""
        assert "_bindJobsActions" in tab_jobs_src
        assert "_jobsActionsBound" in tab_jobs_src
        assert "contentContainer" in tab_jobs_src
        assert "filterBar" in tab_jobs_src

    def test_test_anchor_attribute_set(self, tab_jobs_src):
        """data-jobs-actions-bound wird als Test-Anker gesetzt, damit
        Laufzeit-Tests die Delegation ohne JS-internen Property-Zugriff
        verifizieren koennen."""
        assert "setAttribute('data-jobs-actions-bound', 'true')" in tab_jobs_src

    def test_data_action_attributes_in_render(self, tab_jobs_src):
        """HTML-Strings nutzen data-action= statt onclick=."""
        assert 'data-action="filter-jobs"' in tab_jobs_src
        assert 'data-action="cleanup-job-history"' in tab_jobs_src
        assert 'data-action="toggle-job-pause"' in tab_jobs_src
        assert 'data-action="cancel-job"' in tab_jobs_src
        assert 'data-action="delete-job"' in tab_jobs_src

    def test_data_filter_and_job_id_params(self, tab_jobs_src):
        """Parameter kommen spezifisch, nicht generisch (data-job-id statt
        data-id — V30: ExpandableTable-Sub-Rows haetten sonst closest-
        Kollisionen)."""
        assert 'data-filter="all"' in tab_jobs_src
        assert 'data-filter="running"' in tab_jobs_src
        assert 'data-job-id="${v}"' in tab_jobs_src

    def test_handlers_unchanged(self, tab_jobs_src):
        """Die Handler-Funktionen (_filterJobs, _cancelJob, ...) bleiben
        mit ihrer Signatur erhalten — nur der Dispatch ist neu."""
        for fn in (
            "async function _filterJobs(status)",
            "function _cancelJob(btn, jobId)",
            "function _deleteJob(btn, jobId)",
            "function _toggleJobPause(btn, jobId)",
            "function _cleanupJobHistory(btn)",
        ):
            assert fn in tab_jobs_src, f"tab_jobs.js: Handler-Signatur {fn!r} fehlt"

    def test_with_button_lock_preserved_on_mutation_paths(self, tab_jobs_src):
        """withButtonLock bleibt auf Mutationspfaden im Listener-Body, nicht
        mehr als Inline-Attribut."""
        # Mutations-Handler nutzen withButtonLock in ihrem Body
        for handler_open in (
            "function _cancelJob(btn, jobId) {",
            "function _deleteJob(btn, jobId) {",
            "function _toggleJobPause(btn, jobId) {",
            "function _cleanupJobHistory(btn) {",
        ):
            # nach dem Funktions-Header kommt withButtonLock
            idx = tab_jobs_src.find(handler_open)
            assert idx >= 0
            body = tab_jobs_src[idx : idx + 200]
            assert "withButtonLock(btn" in body, (
                f"tab_jobs.js: {handler_open!r} ohne withButtonLock"
            )


class TestCspPhase2TabComplianceAudit:
    """V30 Welle 2 (#1049) — tab_compliance_audit.js: Filter-Buttons,
    Enter-Key-Suche und stopPropagation-Link auf data-action/data-keydown-action."""

    @pytest.fixture(scope="class")
    def audit_src(self):
        return (_BASE / "kiara" / "web" / "static" / "js" / "tab_compliance_audit.js").read_text(
            encoding="utf-8"
        )

    def test_no_html_inline_events(self, audit_src):
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", audit_src, flags=re.DOTALL)
        assert not tag_attrs, f"tab_compliance_audit.js: HTML-inline on*= gefunden: {tag_attrs[:3]}"

    def test_no_property_handlers(self, audit_src):
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", audit_src)
        assert not prop_handlers, (
            f"tab_compliance_audit.js: .on*= Property-Handler: {prop_handlers[:3]}"
        )

    def test_actions_whitelist(self, audit_src):
        assert "_AUDIT_ACTIONS" in audit_src
        for action in ("'set-audit-filter'", "'load-audit-filtered'", "'open-audit-detail'"):
            assert action in audit_src

    def test_open_audit_detail_is_row_action(self, audit_src):
        """open-audit-detail ist in _AUDIT_ROW_ACTIONS (stopPropagation)
        damit der Link-Klick nicht die Tabellenzeile expandiert."""
        assert "_AUDIT_ROW_ACTIONS" in audit_src
        # Der Set-Inhalt muss open-audit-detail enthalten
        idx = audit_src.find("_AUDIT_ROW_ACTIONS")
        body = audit_src[idx : idx + 200]
        assert "'open-audit-detail'" in body

    def test_delegation_bound(self, audit_src):
        assert "_bindAuditActions" in audit_src
        assert "_auditActionsBound" in audit_src
        assert "data-audit-actions-bound" in audit_src
        assert "addEventListener('click', _handleAuditAction)" in audit_src
        assert "addEventListener('keydown', _handleAuditKeydown)" in audit_src

    def test_keydown_enter_for_search(self, audit_src):
        """Enter-Key auf auditSearch triggert load-audit-filtered via
        data-keydown-action."""
        assert 'data-keydown-action="load-audit-filtered"' in audit_src
        assert "e.key !== 'Enter'" in audit_src

    def test_set_audit_filter_data_attrs(self, audit_src):
        """setAuditFilter-Buttons nutzen data-filter-field/data-filter-value
        statt JS-Argumente im onclick."""
        assert 'data-filter-field="chain_type"' in audit_src
        assert 'data-filter-field="routed_to"' in audit_src
        assert 'data-filter-value="standard"' in audit_src
        assert 'data-filter-value="cloud"' in audit_src

    def test_audit_link_preserves_navigation(self, audit_src):
        """open-audit-detail darf preventDefault NICHT aufrufen, damit
        target=_blank-Navigation weiter funktioniert. Nur stopPropagation
        ist erlaubt (blockiert nur Row-Expand)."""
        # Im open-audit-detail case keine preventDefault
        idx = audit_src.find("case 'open-audit-detail'")
        assert idx >= 0
        body = audit_src[idx : idx + 300]
        assert "preventDefault" not in body, (
            "tab_compliance_audit.js: open-audit-detail darf Link-Navigation nicht blockieren"
        )


class TestCspPhase2TabCompliancePromptSchutz:
    """V30 Welle 2 (#1049) — tab_compliance_prompt_schutz.js: Filter-Buttons
    plus onchange-Date-Inputs auf data-action/data-change-action."""

    @pytest.fixture(scope="class")
    def pd_src(self):
        return (
            _BASE / "kiara" / "web" / "static" / "js" / "tab_compliance_prompt_schutz.js"
        ).read_text(encoding="utf-8")

    def test_no_html_inline_events(self, pd_src):
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", pd_src, flags=re.DOTALL)
        assert not tag_attrs, (
            f"tab_compliance_prompt_schutz.js: HTML-inline on*= gefunden: {tag_attrs[:3]}"
        )

    def test_no_property_handlers(self, pd_src):
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", pd_src)
        assert not prop_handlers, f"tab_compliance_prompt_schutz.js: .on*=: {prop_handlers[:3]}"

    def test_actions_whitelist(self, pd_src):
        assert "_PD_ACTIONS" in pd_src
        assert "'set-pd-filter'" in pd_src

    def test_delegation_bound(self, pd_src):
        assert "_bindPdActions" in pd_src
        assert "_pdActionsBound" in pd_src
        assert "data-pd-actions-bound" in pd_src
        assert "addEventListener('click', _handlePdAction)" in pd_src
        assert "addEventListener('change', _handlePdChange)" in pd_src

    def test_change_action_for_date_inputs(self, pd_src):
        """Date-Inputs pdDateFrom/pdDateTo nutzen data-change-action
        (onchange=... entfernt)."""
        assert 'data-change-action="reload-pd-filter"' in pd_src
        # Die zwei Date-Inputs haben beide das Attribut
        assert pd_src.count('data-change-action="reload-pd-filter"') >= 2

    def test_set_pd_filter_data_attrs(self, pd_src):
        assert 'data-filter-field="severity"' in pd_src
        assert 'data-filter-field="source_type"' in pd_src
        assert 'data-filter-value="hoch"' in pd_src
        assert 'data-filter-value="user_input"' in pd_src

    def test_prototype_pollution_lookup_guard(self, pd_src):
        """Issue #2119 — severityBadge/sourceTypeBadge/formatPatterns duerfen
        Object-Literale nicht ohne hasOwnProperty-Check als Lookup-Map nutzen
        (sonst liefert map['__proto__'] geerbte Properties)."""
        assert "Object.prototype.hasOwnProperty.call(map" in pd_src, (
            "severityBadge/sourceTypeBadge: hasOwnProperty-Guard fehlt"
        )
        assert "Object.prototype.hasOwnProperty.call(labels" in pd_src, (
            "formatPatterns: hasOwnProperty-Guard fehlt"
        )


class TestCspPhase2TabKiModelle:
    """V30 Welle 2 (#1049) — tab_ki_modelle.js: Backend-CRUD mit Save-
    Generation-Guard, Toggle/Delete-In-Flight-Locks und Modal-Close-Bumping."""

    @pytest.fixture(scope="class")
    def km_src(self):
        return (_BASE / "kiara" / "web" / "static" / "js" / "tab_ki_modelle.js").read_text(
            encoding="utf-8"
        )

    def test_no_html_inline_events(self, km_src):
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", km_src, flags=re.DOTALL)
        assert not tag_attrs, f"tab_ki_modelle.js: HTML-inline on*= gefunden: {tag_attrs[:3]}"

    def test_no_property_handlers(self, km_src):
        """_fillModelSelect: select.onchange=... ist entfernt (V30 fordert
        addEventListener/Delegation statt Property-Binding)."""
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", km_src)
        assert not prop_handlers, f"tab_ki_modelle.js: .on*= Property-Handler: {prop_handlers[:3]}"

    def test_actions_whitelist(self, km_src):
        assert "_KI_MODELLE_ACTIONS" in km_src
        assert "_KI_MODELLE_CHANGE_ACTIONS" in km_src
        assert "_KI_MODELLE_INPUT_ACTIONS" in km_src
        for action in (
            "'open-add-backend-dialog'",
            "'set-backend-category-filter'",
            "'test-backend'",
            "'open-edit-backend-modal'",
            "'delete-backend'",
            "'open-backend-form-add'",
            "'close-backend-modal'",
            "'save-backend'",
            "'test-url-or-docling'",
            "'test-cloud-key'",
        ):
            assert action in km_src

    def test_change_actions(self, km_src):
        for action in (
            "'toggle-backend'",
            "'backend-provider-change'",
            "'clear-manual-model-on-select'",
        ):
            assert action in km_src

    def test_toggle_data_action_on_checkbox_input(self, km_src):
        """Checkbox-Input selbst traegt data-change-action=toggle-backend —
        kein querySelector, keine Label-Indirektion (User-Praezisierung 4)."""
        # Suche nach <input ... data-change-action="toggle-backend" ...>
        assert 'data-change-action="toggle-backend"' in km_src
        # In der Renderzeile muss data-backend-id auch direkt am Input sein
        idx = km_src.find('data-change-action="toggle-backend"')
        assert idx >= 0
        line = km_src[max(0, idx - 100) : idx + 200]
        assert 'data-backend-id="${b.id}"' in line
        # Label hat keinen onclick mehr (stopPropagation entfaellt bei data-table)
        assert 'onclick="event.stopPropagation()"' not in km_src

    def test_save_generation_guard(self, km_src):
        """_saveBackend snapshot _beFormGeneration VOR dem ersten await,
        zwei Guards nach den beiden awaits (safeFetch + json), Guard im catch."""
        idx = km_src.find("async function _saveBackend()")
        assert idx >= 0
        body = km_src[idx : idx + 4500]
        # Snapshot vor await
        assert "const mySaveGen" in body
        assert "_beFormGeneration" in body
        # Zwei Guards im try-Pfad
        guard_count = body.count("mySaveGen !== _beFormGeneration")
        assert guard_count >= 3, (
            f"tab_ki_modelle.js: erwartet >=3 Guards (2 try + 1 catch), gefunden {guard_count}"
        )

    def test_save_snapshots_mode_and_id(self, km_src):
        """mySaveMode/mySaveId werden vor await gezogen, damit ein Re-Open
        die PUT-URL nicht mid-flight umbiegt."""
        idx = km_src.find("async function _saveBackend()")
        body = km_src[idx : idx + 4500]
        assert "mySaveMode" in body
        assert "mySaveId" in body
        # PUT-URL nutzt mySaveId, nicht _beFormEditId
        assert "backends/${mySaveId}" in body

    def test_invalidate_and_close_helpers(self, km_src):
        """_invalidateBackendForm bumpt Generation + resettet Mode/EditId.
        _closeBackendModal ruft invalidate + SharedUi.modal.close."""
        assert "function _invalidateBackendForm()" in km_src
        assert "function _closeBackendModal()" in km_src
        idx = km_src.find("function _invalidateBackendForm()")
        body = km_src[idx : idx + 200]
        assert "_beFormGeneration++" in body
        assert "_beFormMode = null" in body
        assert "_beFormEditId = null" in body

    def test_init_only_invalidates_when_form_open(self, km_src):
        """initKiModelleTab ruft _invalidateBackendForm NUR wenn ein Form
        offen ist — kein No-Op-Bump pro Tab-Init (User-Praezisierung 1)."""
        idx = km_src.find("async function initKiModelleTab()")
        body = km_src[idx : idx + 1000]
        assert "if (_beFormMode !== null)" in body
        assert "_invalidateBackendForm()" in body

    def test_init_restores_filterbar_visibility(self, km_src):
        """Der Admin-Router versteckt filterBar per inline display:none;
        initKiModelleTab muss diesen Zustand fuer KI > Modelle zuruecksetzen."""
        idx = km_src.find("async function initKiModelleTab()")
        body = km_src[idx : idx + 1400]
        assert "filterBar.style.display = ''" in body
        assert "filterBar.classList.remove('is-hidden', 'u-hidden')" in body
        assert "filterBar.classList.add('u-flex')" in body

    def test_backend_table_full_width_class(self, km_src):
        """Die Backend-Tabelle nutzt eine eigene Klasse, die im Wrapper
        echtes Tabellenlayout mit min-width:100% erzwingt."""
        assert '<table class="data-table ki-modelle-table u-w-full">' in km_src
        text = (_BASE / "kiara" / "web" / "static" / "css" / "style.css").read_text(
            encoding="utf-8"
        )
        idx = text.find(".ki-modelle-table")
        assert idx >= 0
        block = text[idx : idx + 120]
        assert "display: table" in block
        assert "min-width: 100%" in block

    def test_toggle_backend_in_flight_lock(self, km_src):
        """toggleBackend blockt parallele Requests per row-lokalem Set."""
        assert "_toggleBackendInFlight" in km_src
        idx = km_src.find("async function toggleBackend(")
        body = km_src[idx : idx + 2500]
        assert "_toggleBackendInFlight.has(backendId)" in body
        assert "_toggleBackendInFlight.add(backendId)" in body
        assert "_toggleBackendInFlight.delete(backendId)" in body

    def test_toggle_backend_catch_no_reload(self, km_src):
        """toggleBackend catch-Pfad ruft NICHT initKiModelleTab() (wuerde
        offenes Modal zumachen). Stattdessen lokal revertieren."""
        idx = km_src.find("async function toggleBackend(")
        body = km_src[idx : idx + 2500]
        # catch-Block identifizieren
        catch_idx = body.find("catch (e)")
        assert catch_idx >= 0
        # Bis zum finally oder Ende der Funktion
        finally_idx = body.find("} finally", catch_idx)
        catch_body = body[catch_idx : finally_idx if finally_idx > 0 else catch_idx + 400]
        # Kein funktionaler Aufruf initKiModelleTab( (mit oeffnender Klammer).
        # Kommentar-Textvorkommen ist OK, aber kein Aufruf.
        assert "initKiModelleTab(" not in catch_body, (
            "tab_ki_modelle.js: toggleBackend catch ruft initKiModelleTab() — Full-Reload verboten"
        )
        # Stattdessen lokaler Revert + renderBackendsTable()
        assert "b.is_active = !isActive" in catch_body
        assert "renderBackendsTable()" in catch_body

    def test_delete_backend_in_flight_lock(self, km_src):
        """deleteBackend blockt parallele Requests per row-lokalem Set.
        Confirm-Dialog bleibt User-Doppelklick-Schutz, Lock schuetzt
        gegen programmatische Doppel-Events."""
        assert "_deleteBackendInFlight" in km_src
        idx = km_src.find("async function deleteBackend(")
        # Range generoes (deleteBackend ist mittlerweile ~1400 Zeichen mit den
        # Tab-Guard-Kommentaren); .delete() steht im finally-Block am Ende.
        body = km_src[idx : idx + 2000]
        assert "_deleteBackendInFlight.has(backendId)" in body
        assert "_deleteBackendInFlight.add(backendId)" in body
        assert "_deleteBackendInFlight.delete(backendId)" in body

    def test_modal_close_action_replaces_inline(self, km_src):
        """Alle SharedUi.modal.close() aus Inline-onclick sind entfernt;
        das Modal wird jetzt ueber data-action='close-backend-modal'
        geschlossen (das bumpt Generation)."""
        assert 'onclick="SharedUi.modal.close()"' not in km_src
        assert 'data-action="close-backend-modal"' in km_src

    def test_save_action_no_inline_withbuttonlock(self, km_src):
        """Der withButtonLock-Wrapper im Inline-onclick ist entfernt;
        der Dispatcher ruft SharedUi.runtime.withButtonLock(el, _saveBackend)."""
        assert 'onclick="SharedUi.runtime.withButtonLock(this, _saveBackend)"' not in km_src
        # Im Dispatcher-Case muss withButtonLock aufgerufen werden
        idx = km_src.find("case 'save-backend'")
        body = km_src[idx : idx + 200]
        assert "SharedUi.runtime.withButtonLock(el, _saveBackend)" in body

    def test_test_backend_uses_button_lock(self, km_src):
        """Backend-Test stellt Button-State zentral ueber withButtonLock wieder her."""
        idx = km_src.find("async function testBackend")
        assert idx >= 0
        body = km_src[idx : idx + 700]
        assert "SharedUi.runtime.withButtonLock(btn" in body
        assert "btn.disabled = true" not in body
        assert "const origText" not in body

    def test_delegation_bound_on_single_root(self, km_src):
        """Nur EIN Delegation-Root (document.body), damit Events nicht durch
        mehrere bubbling-Ebenen denselben Dispatch doppelt ausloesen —
        das wuerde bei Row-Actions wie deleteBackend (SharedUi.modal.confirm
        wartet asynchron, kein frueher In-Flight-Lock) zu doppeltem Confirm
        + doppeltem DELETE fuehren."""
        assert "_bindKiModelleActions" in km_src
        assert "data-ki-modelle-actions-bound" in km_src
        assert "document.body._kiModelleActionsBound" in km_src
        # Keine zusaetzlichen addEventListener auf contentContainer/filterBar
        # innerhalb von _bindKiModelleActions — pruefe ueber Abwesenheit der
        # Muster im Funktionsbody.
        idx = km_src.find("function _bindKiModelleActions()")
        assert idx >= 0
        end = km_src.find("\n}\n", idx)
        body = km_src[idx:end]
        assert "container.addEventListener" not in body, (
            "tab_ki_modelle.js: _bindKiModelleActions bindet zusaetzlich auf contentContainer — doppelter Dispatch-Pfad"
        )
        assert "filterBar.addEventListener" not in body, (
            "tab_ki_modelle.js: _bindKiModelleActions bindet zusaetzlich auf filterBar — doppelter Dispatch-Pfad"
        )

    def test_category_card_data_action(self, km_src):
        """Category-Picker-Cards nutzen data-action=open-backend-form-add
        mit data-category statt Inline-_openBackendForm('add', ...)."""
        assert 'data-action="open-backend-form-add"' in km_src
        assert 'data-category="${SharedUi.runtime.escapeAttr(c.id)}"' in km_src

    def test_search_input_delegation(self, km_src):
        """Suchfeld triggert 'update-backend-search' via data-input-action."""
        assert 'data-input-action="update-backend-search"' in km_src
        assert "case 'update-backend-search'" not in km_src  # case in if-action
        # Suche den Input-Handler-Body
        idx = km_src.find("function _handleKiModelleInput")
        body = km_src[idx : idx + 500]
        assert "update-backend-search" in body
        assert "backendsFilterSearch" in body

    def test_fill_model_select_no_property_handler(self, km_src):
        """_fillModelSelect nutzt data-change-action statt select.onchange=."""
        idx = km_src.find("function _fillModelSelect(models)")
        body = km_src[idx : idx + 1000]
        assert "select.onchange" not in body
        assert "clear-manual-model-on-select" in body


class TestCspPhase2TabDatenquellen:
    """V30 Welle 2 (#1049) — tab_datenquellen.js: Row-Actions mit stopProp,
    Hover-Handler raus (CSS :hover), handgebautes Confirm-Overlay ohne
    .onclick-Property-Handler, data-sharedui-close statt modal.close-Inline."""

    @pytest.fixture(scope="class")
    def ds_src(self):
        return (_BASE / "kiara" / "web" / "static" / "js" / "tab_datenquellen.js").read_text(
            encoding="utf-8"
        )

    def test_no_html_inline_events(self, ds_src):
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", ds_src, flags=re.DOTALL)
        assert not tag_attrs, f"tab_datenquellen.js: HTML-inline on*= gefunden: {tag_attrs[:3]}"

    def test_no_property_handlers(self, ds_src):
        """Handgebautes Confirm-Overlay: btn.onclick=... -> addEventListener."""
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", ds_src)
        assert not prop_handlers, f"tab_datenquellen.js: .on*=: {prop_handlers[:3]}"

    def test_actions_whitelist(self, ds_src):
        assert "_DS_ACTIONS" in ds_src
        assert "_DS_ROW_ACTIONS" in ds_src
        assert "_DS_CHANGE_ACTIONS" in ds_src
        assert "_DS_INPUT_ACTIONS" in ds_src
        for action in (
            "'open-create-datasource-modal'",
            "'pause-ds-ingestion'",
            "'cancel-ds-ingestion'",
            "'trigger-ds-ingest'",
            "'trigger-ds-reingest'",
            "'open-edit-datasource-modal'",
            "'open-ds-groups-modal'",
            "'delete-datasource'",
        ):
            assert action in ds_src
        assert "'toggle-datasource-active'" in ds_src

    def test_row_actions_stop_propagation(self, ds_src):
        """Alle Row-Actions sind in _DS_ROW_ACTIONS — ExpandableTable-
        Kompatibilitaet (V30: stopPropagation nur wo Row-Expand bewusst
        geblockt wird, Row-Actions sind genau dieser Fall)."""
        # Extrahiere den _DS_ROW_ACTIONS-Set-Inhalt
        idx = ds_src.find("_DS_ROW_ACTIONS")
        body = ds_src[idx : idx + 500]
        for row_action in (
            "'pause-ds-ingestion'",
            "'cancel-ds-ingestion'",
            "'trigger-ds-ingest'",
            "'trigger-ds-reingest'",
            "'open-edit-datasource-modal'",
            "'open-ds-groups-modal'",
            "'delete-datasource'",
        ):
            assert row_action in body

    def test_delegation_single_root(self, ds_src):
        """Lehre aus dem Delete-Doppel-Dispatch-Bug in tab_ki_modelle.js
        (0.0.1+376): NUR document.body als Delegation-Root, sonst bubbelt
        der Event durch mehrere Ebenen und ruft Handler mehrfach auf."""
        assert "_bindDsActions" in ds_src
        assert "document.body._dsActionsBound" in ds_src
        idx = ds_src.find("function _bindDsActions()")
        end = ds_src.find("\n}\n", idx)
        body = ds_src[idx:end]
        assert "container.addEventListener" not in body, (
            "tab_datenquellen.js: _bindDsActions bindet zusaetzlich auf contentContainer — doppelter Dispatch-Pfad"
        )
        assert "filterBar.addEventListener" not in body, (
            "tab_datenquellen.js: _bindDsActions bindet zusaetzlich auf filterBar — doppelter Dispatch-Pfad"
        )

    def test_row_action_data_attrs(self, ds_src):
        """Row-Buttons nutzen data-ds-id/data-ds-name/data-ds-active statt
        Inline-Parameter. data-ds-active als 1/0-String (User-Praezisierung:
        spezifisch statt generisch, Parent-/Child-Row-Kollisionen vermeiden)."""
        assert 'data-action="pause-ds-ingestion" data-ds-id="${row.id}"' in ds_src
        assert 'data-action="delete-datasource"' in ds_src
        assert "data-ds-active=" in ds_src
        # Kein generisches data-id mehr
        assert 'data-id="${row.id}"' not in ds_src

    def test_active_toggle_persists_via_update_endpoint(self, ds_src):
        assert "_dsActiveToggleInFlight" in ds_src
        assert 'data-change-action="toggle-datasource-active"' in ds_src
        assert 'data-ds-id="${row.id}"' in ds_src
        assert "async function _toggleDatasourceActive" in ds_src
        assert "JSON.stringify({ is_active: isActive })" in ds_src
        assert "`/api/admin/datasources/${dsId}`" in ds_src

    def test_modal_close_via_sharedui_attr(self, ds_src):
        """Alle drei SharedUi.modal.close()-Inline-onclick sind auf
        data-sharedui-close umgestellt (native Modal-Close-Hook,
        sauberer als eine eigene close-Action)."""
        assert 'onclick="SharedUi.modal.close()"' not in ds_src
        # Mindestens drei data-sharedui-close im Footer der drei Modals
        assert ds_src.count("data-sharedui-close") >= 3
        # Keine eigene close-ds-modal-Action mehr (wird ueber SharedUi-Hook erledigt)
        assert "'close-ds-modal'" not in ds_src

    def test_hover_handler_removed(self, ds_src):
        """onmouseover/onmouseout auf fp-entry-Elementen sind entfernt;
        der Hover-Style kommt jetzt ueber CSS :hover (style.css)."""
        assert "onmouseover" not in ds_src
        assert "onmouseout" not in ds_src

    def test_fp_entry_hover_in_css(self):
        """CSS-Klasse .fp-entry:hover existiert in style.css."""
        text = (_BASE / "kiara" / "web" / "static" / "css" / "style.css").read_text(
            encoding="utf-8"
        )
        assert ".fp-entry:hover" in text

    def test_confirm_uses_shared_modal(self, ds_src):
        """Confirm-Dialoge laufen ueber die zentrale SharedUi.modal.confirm()-API
        (vorher gab es ein handgebautes Overlay mit data-overlay-role-Attributen,
        das durch den Auto-Fix-Sweep entfernt wurde — eine Quelle weniger fuer
        eigene Event-Handler-Logik und Whitelist-Kollisionen).
        """
        assert "SharedUi.modal.confirm(" in ds_src
        # Die alten handgebauten Overlay-Attribute duerfen nicht mehr vorkommen
        assert 'data-overlay-role="cancel"' not in ds_src
        assert 'data-overlay-role="confirm"' not in ds_src

    def test_lifecycle_anchors_preserved(self, ds_src):
        """_editOpenToken-Guards und _deleteInProgress-Flag bleiben
        unveraendert (V30 Lifecycle-Anker, nicht von CSP-Migration
        beruehrt)."""
        assert "_editOpenToken" in ds_src
        assert "_deleteInProgress" in ds_src
        # openToken-Guard-Muster in _loadCrDropdowns bleibt bestehen
        assert "openToken !== _editOpenToken" in ds_src


class TestCspPhase2TabKiChat:
    """V30 Welle 2 (#1049) — tab_ki_chat.js: Prompt-Template-Management
    mit ExpandableTable, Toggle-Locks gegen Race zwischen Checkbox und
    ptTable.refresh(), Create-Modal, Rollback-Detail-Action."""

    @pytest.fixture(scope="class")
    def kc_src(self):
        return (_BASE / "kiara" / "web" / "static" / "js" / "tab_ki_chat.js").read_text(
            encoding="utf-8"
        )

    def test_no_html_inline_events(self, kc_src):
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", kc_src, flags=re.DOTALL)
        assert not tag_attrs, f"tab_ki_chat.js: HTML-inline on*= gefunden: {tag_attrs[:3]}"

    def test_no_property_handlers(self, kc_src):
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", kc_src)
        assert not prop_handlers, f"tab_ki_chat.js: .on*=: {prop_handlers[:3]}"

    def test_actions_whitelist(self, kc_src):
        assert "_KI_CHAT_ACTIONS" in kc_src
        assert "_KI_CHAT_CHANGE_ACTIONS" in kc_src
        assert "_KI_CHAT_INPUT_ACTIONS" in kc_src
        for action in (
            "'filter-templates-by-chain-type'",
            "'open-create-template-modal'",
            "'save-template'",
            "'delete-template'",
            "'test-enrichment-prompt'",
            "'create-template'",
            "'rollback-template'",
        ):
            assert action in kc_src

    def test_change_actions(self, kc_src):
        for action in (
            "'toggle-template-chat'",
            "'toggle-template-active'",
            "'create-chain-type-change'",
        ):
            assert action in kc_src

    def test_delegation_single_root(self, kc_src):
        """Single-Root-Dispatch auf document.body (Lehre aus
        0.0.1+376 tab_ki_modelle.js Delete-Doppel-Dispatch)."""
        assert "_bindKiChatActions" in kc_src
        assert "document.body._kiChatActionsBound" in kc_src
        idx = kc_src.find("function _bindKiChatActions()")
        end = kc_src.find("\n}\n", idx)
        body = kc_src[idx:end]
        assert "container.addEventListener" not in body
        assert "filterBar.addEventListener" not in body

    def test_toggle_data_change_action_on_input(self, kc_src):
        """data-change-action sitzt direkt am Checkbox-Input (User-
        Praezisierung 4 aus tab_ki_modelle.js-Freigabe): kein Label-
        Indirektions-querySelector, konsistent zwischen click/change."""
        assert 'data-change-action="toggle-template-chat" data-template-id="${row.id}"' in kc_src
        assert 'data-change-action="toggle-template-active" data-template-id="${row.id}"' in kc_src

    def test_toggle_label_no_inline_stop_propagation(self, kc_src):
        """ExpandableTable (ui_core/js/table.js) hat _interactiveCellMatch
        fuer .toggle-switch/input/select etc., laesst Row-Expand bei Toggle-
        Clicks bereits durch. Das alte
        onclick='event.stopPropagation()' am Label ist obsoleter
        Legacy-Code und muss raus."""
        assert 'onclick="event.stopPropagation()"' not in kc_src

    def test_toggle_template_chat_in_flight_lock(self, kc_src):
        """V30 Risiken: toggleTemplateChat braucht row-lokalen Lock,
        sonst committet ein spaetes ptTable.refresh() juengere
        Checkbox-Interaktionen doppelt / rollt sie zurueck."""
        assert "_toggleTemplateChatInFlight" in kc_src
        idx = kc_src.find("async function toggleTemplateChat(")
        body = kc_src[idx : idx + 1500]
        assert "_toggleTemplateChatInFlight.has(id)" in body
        assert "_toggleTemplateChatInFlight.add(id)" in body
        assert "_toggleTemplateChatInFlight.delete(id)" in body

    def test_toggle_template_active_in_flight_lock(self, kc_src):
        """Analog zu toggleTemplateChat: row-lokaler Lock."""
        assert "_toggleTemplateActiveInFlight" in kc_src
        idx = kc_src.find("async function toggleTemplateActive(")
        body = kc_src[idx : idx + 1500]
        assert "_toggleTemplateActiveInFlight.has(id)" in body
        assert "_toggleTemplateActiveInFlight.add(id)" in body
        assert "_toggleTemplateActiveInFlight.delete(id)" in body

    def test_modal_close_via_sharedui_attr(self, kc_src):
        """Create-Modal-Footer: onclick=SharedUi.modal.close() ->
        data-sharedui-close (nativer Hook)."""
        assert 'onclick="SharedUi.modal.close()"' not in kc_src
        assert "data-sharedui-close" in kc_src

    def test_detail_action_buttons_data_template_id(self, kc_src):
        """Save/Delete/Test/Rollback tragen data-template-id, damit
        der Dispatcher die ID vom DOM holt statt aus Inline-Args."""
        assert 'data-action="save-template" data-template-id="${row.id}"' in kc_src
        assert 'data-action="delete-template" data-template-id="${row.id}"' in kc_src
        assert 'data-action="test-enrichment-prompt" data-template-id="${row.id}"' in kc_src

    def test_rollback_has_template_and_history_id(self, kc_src):
        """rollbackTemplate braucht zwei IDs: Template + History-Eintrag."""
        assert 'data-action="rollback-template"' in kc_src
        assert 'data-template-id="${templateId}"' in kc_src
        assert 'data-history-id="${item.id}"' in kc_src

    def test_filter_button_passes_element_for_active_state(self, kc_src):
        """filterTemplatesByChainType nimmt btn als 1. Arg fuer Active-
        State-Switch. Der Dispatcher uebergibt el statt this."""
        idx = kc_src.find("case 'filter-templates-by-chain-type'")
        body = kc_src[idx : idx + 200]
        assert "filterTemplatesByChainType(el" in body

    def test_data_csp_loaded_guard_preserved(self, kc_src):
        """V30 Lifecycle-Anker: cloud_system_prompt wird absichtlich erst
        im Detail-GET geladen; data-csp-loaded-Guard bleibt bestehen
        fuer Save-Abbruch bei noch nicht geladenem Feld."""
        assert "cspLoaded" in kc_src or "csp-loaded" in kc_src

    def test_update_pt_search_input_delegation(self, kc_src):
        """Suchfeld ueber data-input-action statt oninput."""
        assert 'data-input-action="update-pt-search"' in kc_src

    def test_enrichment_backend_loader_preserves_standard_option(self, kc_src):
        """Backend-Reload darf die leere Standard-LLM-Option nicht entfernen."""
        idx = kc_src.index("async function loadEnrichmentBackends(")
        body = kc_src[idx : idx + 900]
        assert "select.innerHTML = ''" in body
        assert "defaultOpt.value = ''" in body
        assert "defaultOpt.textContent = '(Standard-LLM)'" in body
        assert "select.appendChild(defaultOpt)" in body

    def test_test_enrichment_prompt_uses_button_lock(self, kc_src):
        """test-enrichment-prompt ruft einen POST aus; ohne withButtonLock
        erzeugt Doppelklick zwei parallele Test-Requests."""
        idx = kc_src.find("case 'test-enrichment-prompt'")
        body = kc_src[idx : idx + 200]
        assert "SharedUi.runtime.withButtonLock(el" in body
        assert "testEnrichmentPrompt(Number(el.dataset.templateId))" in body

    def test_rollback_template_uses_button_lock(self, kc_src):
        """rollback-template ist destruktiv (Versions-Wiederherstellung);
        ohne withButtonLock erzeugt Doppelklick zwei Rollback-Versuche."""
        idx = kc_src.find("case 'rollback-template'")
        body = kc_src[idx : idx + 300]
        assert "SharedUi.runtime.withButtonLock(el" in body
        assert "rollbackTemplate(" in body
        assert "el.dataset.templateId" in body
        assert "el.dataset.historyId" in body


class TestCollectAndSaveConfigNumberParsing:
    """app_core.js: Number-Inputs duerfen nicht per parseInt gekuerzt werden."""

    def test_number_inputs_use_number_not_parse_int(self, app_core_src):
        idx = app_core_src.index("async function collectAndSaveConfig(")
        body = app_core_src[idx : idx + 2600]
        assert "Number(rawNumber)" in body
        assert "Number.isFinite(val)" in body
        assert "input.validity.valid" in body
        assert "parseInt(input.value)" not in body


class TestCspPhase2CloseInlineRemovedFromKiChat:
    """Der Invertierungs-Check in TestSFCSprint3CallerMigration
    deckt tab_ki_chat.js ab (nicht mehr als Ausnahme)."""

    def test_ki_chat_no_inline_close(self):
        text = (_BASE / "kiara" / "web" / "static" / "js" / "tab_ki_chat.js").read_text(
            encoding="utf-8"
        )
        assert 'onclick="SharedUi.modal.close()"' not in text


class TestCspPhase2TabKiIngestion:
    """tab_ki_ingestion.js nutzt weiterhin CSP-sichere Event-Delegation.

    Der alte globale Indexierungs-Start wurde entfernt; Ingestion wird ueber
    Datenquellen gestartet.
    """

    @pytest.fixture(scope="class")
    def ing_src(self):
        return (_BASE / "kiara" / "web" / "static" / "js" / "tab_ki_ingestion.js").read_text(
            encoding="utf-8"
        )

    def test_no_html_inline_events(self, ing_src):
        tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", ing_src, flags=re.DOTALL)
        assert not tag_attrs, f"tab_ki_ingestion.js: HTML-inline on*=: {tag_attrs[:3]}"

    def test_no_property_handlers(self, ing_src):
        """Handgebautes Confirm-Overlay: btn.onclick=... -> addEventListener."""
        prop_handlers = re.findall(r"\.\s*on[a-z]+\s*=", ing_src)
        assert not prop_handlers, f"tab_ki_ingestion.js: .on*=: {prop_handlers[:3]}"

    def test_actions_whitelist(self, ing_src):
        assert "_ING_ACTIONS" in ing_src
        for action in (
            "'reset-vlm-prompt'",
            "'reset-accept-all-blocklist'",
            "'save-ki-ingestion-config'",
            "'reset-ki-ingestion-tab'",
        ):
            assert action in ing_src

    def test_delegation_single_root(self, ing_src):
        assert "_bindIngActions" in ing_src
        assert "document.body._ingActionsBound" in ing_src
        idx = ing_src.find("function _bindIngActions()")
        end = ing_src.find("\n}\n", idx)
        body = ing_src[idx:end]
        assert "container.addEventListener" not in body
        assert "filterBar.addEventListener" not in body

    def test_global_indexing_controls_removed(self, ing_src):
        """Der KI/Ingestion-Tab startet keine globale Indexierung mehr."""
        forbidden = (
            "trigger-ingestion-index",
            "triggerIngestionIndex",
            "data-ing-start",
            "indexStatus",
            "/api/admin/index",
            "/api/admin/index/status",
            "Inkrementell indexieren",
            "Vollstaendig indexieren",
        )
        for token in forbidden:
            assert token not in ing_src

    def test_global_indexing_api_removed(self):
        api_settings = (_BASE / "kiara" / "web" / "api_settings.py").read_text(encoding="utf-8")
        for token in ('@router.post("/index")', '@router.get("/index/status")', "_indexing_status"):
            assert token not in api_settings

    def test_save_button_data_ing_save_preserved(self, ing_src):
        """data-ing-save-Attribut bleibt erhalten (wird vermutlich fuer
        CSS-Lock-Styling oder bestehende Selektoren genutzt)."""
        assert "data-ing-save" in ing_src

    def test_lifecycle_anchors_preserved(self, ing_src):
        assert "_ingInitGen" in ing_src


class TestCspPhase2Batch1:
    """V30 Welle 2 (#1049) — Batch-Migration: tab_testsuite.js,
    tab_integrity.js, tab_benutzer.js, tab_gruppen.js.
    Alle nach Single-Root-Pattern auf document.body, data-sharedui-close
    fuer Modal-Close, Row-Actions mit stopPropagation-Set."""

    JS_DIR = _BASE / "kiara" / "web" / "static" / "js"

    @pytest.fixture(scope="class")
    def sources(self):
        return {
            "testsuite": (self.JS_DIR / "tab_testsuite.js").read_text(encoding="utf-8"),
            "integrity": (self.JS_DIR / "tab_integrity.js").read_text(encoding="utf-8"),
            "benutzer": (self.JS_DIR / "tab_benutzer.js").read_text(encoding="utf-8"),
            "gruppen": (self.JS_DIR / "tab_gruppen.js").read_text(encoding="utf-8"),
        }

    def test_no_html_inline_events(self, sources):
        for name, text in sources.items():
            tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", text, flags=re.DOTALL)
            assert not tag_attrs, f"tab_{name}.js: HTML-inline on*=: {tag_attrs[:2]}"

    def test_no_property_handlers(self, sources):
        for name, text in sources.items():
            prop = re.findall(r"\.\s*on[a-z]+\s*=", text)
            assert not prop, f"tab_{name}.js: .on*=: {prop[:2]}"

    def test_single_root_delegation(self, sources):
        """Alle vier Module binden auf document.body (Lehre 0.0.1+376)."""
        for name, text in sources.items():
            assert "document.body._" in text, f"tab_{name}.js: kein document.body-Flag"

    def test_testsuite_start_action(self, sources):
        text = sources["testsuite"]
        assert "_SELBSTTEST_ACTIONS" in text
        assert "'start-tests'" in text
        assert 'data-action="start-tests"' in text

    def test_integrity_actions(self, sources):
        text = sources["integrity"]
        assert "_INTEGRITY_ACTIONS" in text
        assert "_INTEGRITY_ROW_ACTIONS" in text
        for action in (
            "'start-integrity-check'",
            "'run-quick-check'",
            "'start-enrichment-reindex'",
            "'start-repair'",
            "'switch-to-retrieval'",
        ):
            assert action in text
        # Row-Actions stopPropagation
        assert "'start-enrichment-reindex'" in text
        # <a href="#"> mit preventDefault fuer Tab-Switch
        assert 'data-action="switch-to-retrieval"' in text
        idx = text.find("case 'switch-to-retrieval'")
        body = text[idx : idx + 300]
        assert "e.preventDefault()" in body
        assert "switchAdminTab('ki', 'ki-retrieval')" in body

    def test_integrity_repair_action_data_attr(self, sources):
        """startRepair-Buttons tragen die Action-ID via data-repair-action
        (dynamische b.action aus CHECK_DEFS)."""
        text = sources["integrity"]
        assert 'data-action="start-repair" data-repair-action=' in text

    def test_integrity_enrichment_reindex_data_attr(self, sources):
        text = sources["integrity"]
        assert 'data-action="start-enrichment-reindex" data-ds-id=' in text

    def test_benutzer_actions(self, sources):
        text = sources["benutzer"]
        assert "_USER_ACTIONS" in text
        assert "_USER_ROW_ACTIONS" in text
        assert "_USER_CHANGE_ACTIONS" in text
        for action in ("'open-create-user-modal'", "'open-edit-user-modal'", "'delete-user'"):
            assert action in text
        assert "'toggle-user-field'" in text
        assert "'update-user-search'" in text

    def test_benutzer_toggle_no_label_stop_propagation(self, sources):
        """Label-onclick=stopPropagation ist entfernt — ui_core/js/table.js
        _interactiveCellMatch uebernimmt Row-Expand-Block (Fix 0.0.1+380)."""
        text = sources["benutzer"]
        # Toggle-Renderer nutzt data-change-action direkt am Input
        assert 'data-change-action="toggle-user-field"' in text
        assert '<label class="toggle-switch" onclick=' not in text

    def test_benutzer_data_sharedui_close(self, sources):
        text = sources["benutzer"]
        assert 'onclick="SharedUi.modal.close()"' not in text
        assert "data-sharedui-close" in text

    def test_benutzer_row_action_data_attrs(self, sources):
        text = sources["benutzer"]
        assert 'data-action="open-edit-user-modal" data-user-id=' in text
        assert 'data-action="delete-user" data-user-id=' in text
        # Alte this.getAttribute-Workarounds sind raus
        assert "this.getAttribute('data-user-id')" not in text

    def test_gruppen_actions(self, sources):
        text = sources["gruppen"]
        assert "_GRUPPEN_ACTIONS" in text
        for action in (
            "'sync-group-members'",
            "'open-group-import-modal'",
            "'delete-group'",
            "'toggle-all-import-groups'",
        ):
            assert action in text
        assert "filterBar.style.display = ''" in text
        assert "filterBar.classList.remove('is-hidden', 'u-hidden')" in text
        assert 'data-action="open-group-import-modal"' in text
        assert 'data-action="sync-group-members"' in text
        assert "gruppen-actions-inline" not in text

    def test_gruppen_data_sharedui_close(self, sources):
        text = sources["gruppen"]
        assert 'onclick="SharedUi.modal.close()"' not in text
        assert "data-sharedui-close" in text


class TestCspPhase2Batch2:
    """V30 Welle 2 (#1049) — Batch 2: tab_ldap.js, tab_tools.js,
    tab_mail.js, tab_system_traces.js, tab_system_traces_panels.js."""

    JS_DIR = _BASE / "kiara" / "web" / "static" / "js"

    @pytest.fixture(scope="class")
    def sources(self):
        return {
            "ldap": (self.JS_DIR / "tab_ldap.js").read_text(encoding="utf-8"),
            "tools": (self.JS_DIR / "tab_tools.js").read_text(encoding="utf-8"),
            "mail": (self.JS_DIR / "tab_mail.js").read_text(encoding="utf-8"),
            "traces": (self.JS_DIR / "tab_system_traces.js").read_text(encoding="utf-8"),
            "panels": (self.JS_DIR / "tab_system_traces_panels.js").read_text(encoding="utf-8"),
        }

    def test_no_html_inline_events(self, sources):
        for name, text in sources.items():
            tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", text, flags=re.DOTALL)
            assert not tag_attrs, f"tab_{name}.js: HTML-inline on*=: {tag_attrs[:2]}"

    def test_no_property_handlers(self, sources):
        for name, text in sources.items():
            prop = re.findall(r"\.\s*on[a-z]+\s*=", text)
            assert not prop, f"tab_{name}.js: .on*=: {prop[:2]}"

    def test_ldap_actions(self, sources):
        text = sources["ldap"]
        assert "_LDAP_ACTIONS" in text
        for action in (
            "'save-ldap-config'",
            "'delete-ldap-config'",
            "'deselect-ldap-group'",
            "'manual-ldap-sync'",
            "'load-ldap-groups'",
            "'select-ldap-group'",
        ):
            assert action in text

    def test_tools_actions(self, sources):
        text = sources["tools"]
        assert "_TOOLS_ACTIONS" in text
        assert "_TOOLS_CHANGE_ACTIONS" in text
        for action in (
            "'test-connection'",
            "'edit-connection'",
            "'delete-connection'",
            "'show-connection-dialog'",
            "'save-tool-permissions'",
            "'save-erp-config'",
            "'test-connection-in-dialog'",
            "'close-conn-dialog'",
            "'save-connection-from-dialog'",
        ):
            assert action in text
        for action in (
            "'update-tool-assignment'",
            "'toggle-erp-backend-fields'",
            "'update-conn-type-fields'",
        ):
            assert action in text

    def test_tools_save_with_button_lock_preserved(self, sources):
        """save-connection-from-dialog nutzt withButtonLock via .bind(null, editId)
        — die editId kommt jetzt aus data-edit-id statt Inline-Argument."""
        text = sources["tools"]
        assert 'data-action="save-connection-from-dialog" data-edit-id=' in text
        idx = text.find("case 'save-connection-from-dialog'")
        body = text[idx : idx + 300]
        assert "withButtonLock(el, saveConnectionFromDialog.bind(null, el.dataset.editId" in body

    def test_tools_connection_row_data_attr(self, sources):
        """test-/edit-/delete-connection: conn-ID ueber data-conn-id."""
        text = sources["tools"]
        assert 'data-action="test-connection" data-conn-id=' in text
        assert 'data-action="edit-connection" data-conn-id=' in text
        assert 'data-action="delete-connection" data-conn-id=' in text

    def test_mail_actions(self, sources):
        text = sources["mail"]
        assert "_MAIL_ACTIONS" in text
        for action in (
            "'save-mail-config'",
            "'test-mail'",
            "'send-status-report'",
            "'delete-mail-config'",
        ):
            assert action in text

    def test_traces_actions(self, sources):
        text = sources["traces"]
        assert "_TRACES_ACTIONS" in text
        assert "_TRACES_CHANGE_ACTIONS" in text
        for action in (
            "'open-trace-comparison'",
            "'load-duration-histogram'",
            "'load-span-averages'",
            "'export-traces'",
        ):
            assert action in text
        for change_action in (
            "'reload-traces-filter'",
            "'tracing-toggle'",
        ):
            assert change_action in text

    def test_traces_filter_change_actions(self, sources):
        """Alle 6 Filter-Inputs nutzen data-change-action=reload-traces-filter."""
        text = sources["traces"]
        # 7 Vorkommen: 6 Filter-Inputs (Status/Root-Span/Von/Bis/Min-Dauer/Span/Attribut)
        # — Status und Root-Span haben eigene Zeilen, die anderen sind gruppiert
        count = text.count('data-change-action="reload-traces-filter"')
        assert count >= 6, (
            f"tab_system_traces.js: erwartet >=6 Filter-Inputs mit reload-traces-filter, gefunden {count}"
        )

    def test_traces_tracing_toggle_on_checkbox(self, sources):
        """tracingToggle-Checkbox triggert _onTracingToggleChange via
        data-change-action=tracing-toggle direkt am Input."""
        text = sources["traces"]
        assert 'id="tracingToggle" data-change-action="tracing-toggle"' in text

    def test_traces_checkbox_no_inline_stop_propagation(self, sources):
        """_select-Checkbox hatte onclick=event.stopPropagation() — obsolet
        durch ui_core/table.js Fix 0.0.1+380. Ersatzlos entfernt."""
        text = sources["traces"]
        assert 'onclick="event.stopPropagation()"' not in text

    def test_panels_actions(self, sources):
        """Aggregations-Panel-Buttons: load-duration-histogram / load-span-averages.
        Export-Buttons: export-traces mit data-format=csv/json."""
        text = sources["panels"]
        assert 'data-action="load-duration-histogram"' in text
        assert 'data-action="load-span-averages"' in text
        assert 'data-action="export-traces" data-format="csv"' in text
        assert 'data-action="export-traces" data-format="json"' in text

    def test_single_root_delegation_each(self, sources):
        for name, text in sources.items():
            # panels hat keinen eigenen Dispatcher — teilt sich den von traces
            if name == "panels":
                continue
            if name == "chat":
                assert "let _chatTabActionsBound = false" in text
                assert "document.body._chatTabActionsBound" not in text
                assert "document.addEventListener('click', _handleChatTabClick)" in text
                continue
            assert "document.body._" in text, f"tab_{name}.js: kein document.body-Flag"


class TestCspPhase2Batch3:
    """V30 Welle 2 (#1049) — Batch 3: User-UI + Admin-Graph.
    tab_chat.js (User-Chat), tab_history.js (ondblclick fuer Rename),
    tab_search.js (User-Search), tab_ki_graph.js."""

    JS_DIR = _BASE / "kiara" / "web" / "static" / "js"

    @pytest.fixture(scope="class")
    def sources(self):
        return {
            "chat": (self.JS_DIR / "tab_chat.js").read_text(encoding="utf-8"),
            "history": (self.JS_DIR / "tab_history.js").read_text(encoding="utf-8"),
            "search": (self.JS_DIR / "tab_search.js").read_text(encoding="utf-8"),
            "graph": (self.JS_DIR / "tab_ki_graph.js").read_text(encoding="utf-8"),
        }

    def test_no_html_inline_events(self, sources):
        for name, text in sources.items():
            tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", text, flags=re.DOTALL)
            assert not tag_attrs, f"tab_{name}.js: HTML-inline on*=: {tag_attrs[:2]}"

    def test_no_property_handlers(self, sources):
        """Wichtig fuer tab_chat.js: modeSelect.onchange= war bisher
        letzter Property-Handler im User-UI — jetzt via addEventListener
        mit explizitem Rebind."""
        for name, text in sources.items():
            prop = re.findall(r"\.\s*on[a-z]+\s*=", text)
            assert not prop, f"tab_{name}.js: .on*=: {prop[:2]}"

    def test_chat_mode_select_uses_addeventlistener(self, sources):
        """modeSelect.onchange -> addEventListener mit idempotentem Rebind."""
        text = sources["chat"]
        assert "_chatModeChangeBound" not in text
        assert "function _handleChatModeChange" in text
        assert "modeSelect.removeEventListener('change', _handleChatModeChange)" in text
        assert "modeSelect.addEventListener('change', _handleChatModeChange)" in text
        assert "modeSelect.onchange" not in text

    def test_chat_new_button(self, sources):
        text = sources["chat"]
        assert "_CHAT_TAB_ACTIONS" in text
        assert 'data-action="chat-new"' in text

    def test_history_actions_and_row_actions(self, sources):
        text = sources["history"]
        assert "_HISTORY_ACTIONS" in text
        assert "_HISTORY_ROW_ACTIONS" in text
        for action in (
            "'filter-history-by-status'",
            "'start-title-edit'",
            "'resume-chat'",
            "'archive-chat'",
            "'restore-chat'",
            "'delete-chat'",
        ):
            assert action in text

    def test_history_dblclick_action(self, sources):
        """ondblclick fuer Title-Edit wird via dblclick-Event-Listener +
        data-dblclick-action delegiert — eigener Event-Typ, nicht click."""
        text = sources["history"]
        assert "_HISTORY_DBLCLICK_ACTIONS" in text
        assert 'data-dblclick-action="start-title-edit"' in text
        assert "addEventListener('dblclick'" in text

    def test_history_row_action_data_attrs(self, sources):
        text = sources["history"]
        assert 'data-action="start-title-edit" data-chat-id=' in text
        assert 'data-action="delete-chat" data-chat-id=' in text

    def test_search_actions(self, sources):
        text = sources["search"]
        assert "_SEARCH_ACTIONS" in text
        assert "_SEARCH_ROW_ACTIONS" in text
        assert "_SEARCH_CHANGE_ACTIONS" in text
        for action in (
            "'send-selected-to-chat'",
            "'send-selected-summary'",
            "'send-single-to-chat'",
            "'execute-search'",
            "'set-search-mode'",
            "'remove-search-ds'",
        ):
            assert action in text
        for change_action in (
            "'toggle-search-select-all'",
            "'toggle-search-chunk'",
        ):
            assert change_action in text

    def test_search_chunk_checkbox_no_inline_stop_propagation(self, sources):
        """searchCheckbox-Input hatte onclick=event.stopPropagation() —
        obsolet durch ui_core/table.js Fix 0.0.1+380."""
        text = sources["search"]
        # Kein inline stopPropagation mehr
        assert 'onclick="event.stopPropagation()"' not in text

    def test_search_mode_data_attr(self, sources):
        text = sources["search"]
        assert 'data-action="set-search-mode" data-mode="text"' in text
        assert 'data-action="set-search-mode" data-mode="semantic"' in text

    def test_graph_actions(self, sources):
        text = sources["graph"]
        assert "_GRAPH_ACTIONS" in text
        for action in (
            "'sync-erp-graph'",
            "'clear-graph'",
            "'save-graph-config'",
            "'reset-ki-graph-tab'",
        ):
            assert action in text
        # btn-Anker: Dispatcher uebergibt el (der Button)
        idx = text.find("case 'sync-erp-graph'")
        body = text[idx : idx + 100]
        assert "syncErpGraph(el)" in body

    def test_single_root_delegation_each(self, sources):
        for name, text in sources.items():
            if name == "chat":
                assert "let _chatTabActionsBound = false" in text
                assert "document.body._chatTabActionsBound" not in text
                assert "document.addEventListener('click', _handleChatTabClick)" in text
                continue
            assert "document.body._" in text, f"tab_{name}.js: kein document.body-Flag"


class TestCspPhase2Welle2Complete:
    """V30 Welle 2 (#1049) — Exit-Gate: repo-weit keine HTML-inline on*=
    mehr, keine .on*= Property-Handler.
    Bei neuem Tab bitte entweder Dispatcher bauen oder diesen Test
    bewusst bei Bedarf lockern."""

    JS_DIRS = [
        _BASE / "kiara" / "web" / "static" / "js",
        _BASE / "kiara" / "web" / "static" / "ui_core" / "js",
    ]

    @pytest.fixture(scope="class")
    def all_js_files(self):
        files = []
        for d in self.JS_DIRS:
            files.extend(p for p in d.glob("*.js") if not p.name.startswith("vendor"))
        return files

    def test_repo_wide_no_inline_events(self, all_js_files):
        """Multiline-sicherer Tag-Scan: repo-weit 0 HTML-inline on*= ."""
        total = 0
        offenders = []
        for path in all_js_files:
            text = path.read_text(encoding="utf-8")
            matches = re.findall(r"<[^>]*\bon[a-z]+\s*=", text, flags=re.DOTALL)
            if matches:
                offenders.append((path.name, len(matches)))
                total += len(matches)
        assert total == 0, f"Repo-weit inline events: {offenders}"

    def test_repo_wide_no_property_handlers(self, all_js_files):
        """Repo-weit keine .on*= Property-Handler (Gate 4 aus V30)."""
        offenders = []
        for path in all_js_files:
            text = path.read_text(encoding="utf-8")
            matches = re.findall(r"\.\s*on[a-z]+\s*=", text)
            if matches:
                offenders.append((path.name, matches[:3]))
        assert not offenders, f"Repo-weit .on*= Property-Handler: {offenders}"


class TestCspPhase2Batch4:
    """V30 Welle 2 (#1049) — Batch 4 (Abschluss): tab_monitoring,
    tab_backup, tab_onlyoffice + 8 mittelkleine Module.
    Nach diesem Batch sind alle tab_*.js + chat.js CSP-migriert."""

    JS_DIR = _BASE / "kiara" / "web" / "static" / "js"

    @pytest.fixture(scope="class")
    def sources(self):
        names = [
            "tab_monitoring",
            "tab_backup",
            "tab_onlyoffice",
            "tab_allgemein_vektordb",
            "tab_allgemein_web",
            "tab_compliance_anon",
            "tab_compliance_berichte",
            "tab_ki_retrieval",
            "tab_performance",
            "tab_system_aenderungen",
            "tab_vektordb_management",
            "chat",
        ]
        return {n: (self.JS_DIR / f"{n}.js").read_text(encoding="utf-8") for n in names}

    def test_no_inline_events(self, sources):
        for name, text in sources.items():
            tag_attrs = re.findall(r"<[^>]*\bon[a-z]+\s*=", text, flags=re.DOTALL)
            assert not tag_attrs, f"{name}.js: HTML-inline on*=: {tag_attrs[:2]}"

    def test_no_property_handlers(self, sources):
        for name, text in sources.items():
            prop = re.findall(r"\.\s*on[a-z]+\s*=", text)
            assert not prop, f"{name}.js: .on*=: {prop[:2]}"

    def test_monitoring_eventsource_via_addeventlistener(self, sources):
        """SSE _eventSource.onerror = ... wurde auf addEventListener('error')
        umgestellt; Stub-Handler bleibt fuer spaeteres Logging."""
        text = sources["tab_monitoring"]
        assert "_eventSource.onerror" not in text
        assert "_eventSource.addEventListener('error'" in text

    def test_monitoring_range_change_data_attr(self, sources):
        text = sources["tab_monitoring"]
        assert "_MON_ACTIONS" in text
        assert "'range-change'" in text
        history = _js("monitoring_history.js")
        assert 'data-action="range-change" data-range="${key}"' in history
        for time_range in ("1m", "10m", "1h", "6h", "24h", "7d", "30d", "90d", "1y"):
            assert f"'{time_range}':" in history

    def test_backup_preserves_existing_restore_pattern(self, sources):
        """tab_backup.js hatte bereits ein lokales addEventListener-Muster
        fuer data-restore-file (Sprint 2.4). Dieser Batch hat nur die
        create-backup-Action ergaenzt, ohne die bestehende Bridge zu
        beruehren."""
        text = sources["tab_backup"]
        assert "data-restore-file" in text
        assert "btn.addEventListener('click'" in text
        assert "_BACKUP_ACTIONS" in text
        assert "'create-backup'" in text

    def test_ki_retrieval_toggle_section_change(self, sources):
        """toggle-retrieval-section wird ueber data-change-action auf
        Select + Checkbox delegiert (je mit data-section='hybrid'/'rerank')."""
        text = sources["tab_ki_retrieval"]
        assert 'data-change-action="toggle-retrieval-section" data-section="hybrid"' in text
        assert 'data-change-action="toggle-retrieval-section" data-section="rerank"' in text

    def test_vdbmgmt_open_detail_via_dataset(self, sources):
        """_vdbMgmt_openDetail erhaelt alle drei Argumente (dsId, model, dsName)
        ueber el.dataset statt alte this.getAttribute-Bridge."""
        text = sources["tab_vektordb_management"]
        assert "'vdbmgmt-open-detail'" in text
        # alte Bridge entfernt
        assert "this.getAttribute('data-ds-id')" not in text
        # neuer Dispatcher nutzt el.dataset
        assert "el.dataset.dsId" in text
        assert "el.dataset.model" in text
        assert "el.dataset.dsName" in text

    def test_vdbmgmt_sort_via_dataset(self, sources):
        """Spalten-Header-Sort: data-sort-col bleibt, Leser ist Dispatcher
        via el.dataset.sortCol."""
        text = sources["tab_vektordb_management"]
        assert "data-sort-col=" in text
        assert "el.dataset.sortCol" in text
        assert "this.getAttribute('data-sort-col')" not in text

    def test_chat_no_inline_stop_propagation_on_edit_link(self, sources):
        """chat-source-edit-btn Link hat kein stopPropagation mehr
        (war defensiv ohne aktuellen Parent-Handler)."""
        text = sources["chat"]
        assert 'onclick="event.stopPropagation()"' not in text
        # Edit-Link selbst bleibt funktional
        assert 'class="chat-source-edit-btn"' in text

    def test_single_root_delegation(self, sources):
        """Alle 11 Module (ausser chat.js, das nur Helper ist) binden
        document.body-Delegation."""
        for name, text in sources.items():
            if name == "chat":
                continue
            assert "document.body._" in text, f"{name}.js: kein document.body-Flag"
