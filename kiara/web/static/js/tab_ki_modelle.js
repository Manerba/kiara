/**
 * KIara Admin - KI > Modelle & Backend Sub-Tab
 * Multi-Backend-Verwaltung mit CRUD-Tabelle (Phase 13)
 */

const KI_PASSWORD_MASK = '********';  // Muss mit PASSWORD_MASK aus constants.py uebereinstimmen

let backendsTable = null;
let backendsData = [];
let backendsCategories = [];
let backendsFilterCategory = '';
let backendsFilterSearch = '';
let embeddingContractDiagnostics = { contract: {}, servers: [], backends: [] };
let embeddingContractDiagnosticsError = '';

// -- Lifecycle State (Init-Seq-Counter analog tab_ki_ingestion) --
// adminCurrentSubTab-Check allein reicht nicht: bei schnellem Tab-Wechsel
// (ki-modelle -> anderer -> ki-modelle) wird der Tab-Check fuer beide
// Instanzen gleich beantwortet; nur ein Generation-Counter erkennt das
// Re-Init und entwertet den stale Lauf, bevor er backendsData ueberschreibt.
let _kiModelleInitGen = 0;

// -- Shared form state (one modal at a time) --
let _beFormMode = null;   // 'add' | 'edit'
let _beFormEditId = null;  // backend ID when editing
let _beFormGeneration = 0;  // Instanz-Counter: bricht stale Test-/Save-Responses ab, die das Modal nach Close/Reopen verseuchen wuerden

// -- Table-Action-Races: row-lokale In-Flight-Sets --
const _toggleBackendInFlight = new Set();
const _deleteBackendInFlight = new Set();

// #1051: Snapshot der Cloud-Felder beim Modal-Open im Edit-Modus.
// Dient als Vergleichsbasis fuer _cloudFieldsUnchanged(): nur wenn provider/
// model/base_url unveraendert sind, darf _testCloudKey() bei maskiertem Key
// den persisted Backend-Test wiederverwenden. Sonst wuerde ein geaenderter
// Dialog gegen den alten Persistenzzustand getestet — irrefuehrend.
let _beFormBaseline = null;

// #1052: Probe-Dimension fuer unbekannte Cloud-Embedding-Modelle. Wird nach
// erfolgreichem _testEmbeddingCloud() gesetzt und im Save-Pfad ins Payload
// uebernommen, falls Modell/Provider/Base-URL/API-Key unveraendert sind.
// Server-seitiger Guard bleibt _normalize_embedding_dim().
let _embeddingDimProbe = null;

// -- Event-Delegation-Whitelists --
const _KI_MODELLE_ACTIONS = new Set([
    'open-add-backend-dialog',
    'set-backend-category-filter',
    'test-backend',
    'open-edit-backend-modal',
    'delete-backend',
    'open-backend-form-add',
    'close-backend-modal',
    'save-backend',
    'test-url-or-docling',
    'test-cloud-key',
]);

const _KI_MODELLE_CHANGE_ACTIONS = new Set([
    'toggle-backend',
    'backend-provider-change',
    'clear-manual-model-on-select',
]);

const _KI_MODELLE_INPUT_ACTIONS = new Set([
    'update-backend-search',
]);

function _invalidateBackendForm() {
    _beFormGeneration++;
    _beFormMode = null;
    _beFormEditId = null;
    _beFormBaseline = null;
    _embeddingDimProbe = null;
}

function _closeBackendModal() {
    _invalidateBackendForm();
    SharedUi.modal.close();
}

function _beFormCurrentModel() {
    const manual = (document.getElementById('beFormModelManual')?.value || '').trim();
    const select = document.getElementById('beFormModelSelect')?.value || '';
    return manual || select;
}

function _providerNeedsApiKey(provider) {
    return provider === 'anthropic' || provider === 'openai' || provider === 'voyage';
}

function _cloudFieldsUnchanged() {
    if (!_beFormBaseline) return false;
    const provider = document.getElementById('beFormProvider')?.value || '';
    const baseUrl  = document.getElementById('beFormBaseUrl')?.value || '';
    return (provider === _beFormBaseline.provider
         && _beFormCurrentModel() === _beFormBaseline.model
         && baseUrl === _beFormBaseline.base_url);
}

// #1619: Probe-Snapshot speichert Fingerprint statt Klartext-Key. Modul-Global
// kann ESC/Backdrop-Close ueberleben bis zum naechsten Modal-Open; ein Hash
// macht den Speicher-Inhalt nutzlos fuer Memory-Inspect via DevTools.
function _apiKeyFingerprint(key) {
    if (!key) return '';
    let h = 0x811c9dc5 >>> 0;
    for (let i = 0; i < key.length; i++) {
        h ^= key.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return 'fp:' + key.length + ':' + h.toString(16);
}

function _embeddingProbeMatchesCurrent() {
    if (!_embeddingDimProbe) return false;
    const provider = document.getElementById('beFormProvider')?.value || '';
    const baseUrl  = document.getElementById('beFormBaseUrl')?.value || '';
    const apiKey   = document.getElementById('beFormApiKey')?.value || '';
    // Probe gilt nur, wenn der Key zu dem passt, mit dem getestet wurde.
    // Erlaubt: identischer Klartext-Key (Add oder neu eingegebener Key) ODER
    // Maske (Edit-Modus, persisted Key wird wiederverwendet).
    const currentFp = _apiKeyFingerprint(apiKey);
    const apiKeyChanged = apiKey && apiKey !== KI_PASSWORD_MASK
                       && currentFp !== _embeddingDimProbe.api_key_fingerprint;
    if (apiKeyChanged) return false;
    const apiKeyMatches = (currentFp === _embeddingDimProbe.api_key_fingerprint)
                       || (apiKey === KI_PASSWORD_MASK);
    return (apiKeyMatches
         && _embeddingDimProbe.provider === provider
         && _embeddingDimProbe.model    === _beFormCurrentModel()
         && _embeddingDimProbe.base_url === baseUrl);
}

function _handleKiModelleClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_KI_MODELLE_ACTIONS.has(action)) return;

    switch (action) {
        case 'open-add-backend-dialog':
            openAddBackendDialog();
            break;
        case 'set-backend-category-filter':
            backendsFilterCategory = el.dataset.catId || '';
            _updateCatBtns();
            renderBackendsTable();
            break;
        case 'test-backend':
            testBackend(Number(el.dataset.backendId), el);
            break;
        case 'open-edit-backend-modal':
            openEditBackendModal(Number(el.dataset.backendId));
            break;
        case 'delete-backend':
            deleteBackend(Number(el.dataset.backendId), el.dataset.backendName || '');
            break;
        case 'open-backend-form-add':
            _openBackendForm('add', el.dataset.category);
            break;
        case 'close-backend-modal':
            _closeBackendModal();
            break;
        case 'save-backend':
            SharedUi.runtime.withButtonLock(el, _saveBackend);
            break;
        case 'test-url-or-docling':
            _testUrlOrDocling();
            break;
        case 'test-cloud-key':
            _testCloudKey();
            break;
    }
}

function _handleKiModelleChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el) return;
    const action = el.dataset.changeAction;
    if (!_KI_MODELLE_CHANGE_ACTIONS.has(action)) return;

    switch (action) {
        case 'toggle-backend':
            toggleBackend(Number(el.dataset.backendId), el.checked);
            break;
        case 'backend-provider-change':
            _onBeFormProviderChange();
            break;
        case 'clear-manual-model-on-select':
            if (el.value) {
                const manual = document.getElementById('beFormModelManual');
                if (manual) manual.value = '';
            }
            break;
    }
}

function _handleKiModelleInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el) return;
    const action = el.dataset.inputAction;
    if (!_KI_MODELLE_INPUT_ACTIONS.has(action)) return;

    if (action === 'update-backend-search') {
        backendsFilterSearch = el.value.toLowerCase();
        renderBackendsTable();
    }
}

function _bindKiModelleActions() {
    // Nur EINEN delegierenden Root verwenden, damit der Event nicht durch
    // mehrere bubbling-Ebenen denselben Dispatch mehrfach ausloest.
    // document.body deckt alle drei Scopes ab: filterBar, contentContainer
    // und das von SharedUi.modal an body gehaengte .modal-overlay.
    if (document.body._kiModelleActionsBound) return;
    document.body._kiModelleActionsBound = true;
    document.body.addEventListener('click', _handleKiModelleClick);
    document.body.addEventListener('change', _handleKiModelleChange);
    document.body.addEventListener('input', _handleKiModelleInput);

    // Test-Anker: informatives Attribut auf contentContainer, damit
    // Laufzeit-Tests die Delegation verifizieren koennen, ohne auf
    // document.body-Properties zugreifen zu muessen.
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-ki-modelle-actions-bound', 'true');
}

function _cleanupKiModelle() {
    _kiModelleInitGen++;
}

async function _loadEmbeddingContractDiagnostics() {
    try {
        const response = await SharedUi.fetch.safeFetch('/api/admin/embedding-contracts');
        if (!response.ok) return null;
        const data = await response.json();
        return data && typeof data === 'object' && !Array.isArray(data) ? data : null;
    } catch (_error) {
        return null;
    }
}

async function initKiModelleTab() {
    // Re-Init-Sicherheit: alte Cleanups ausfuehren (bumpt _kiModelleInitGen,
    // entwertet damit jede noch in-flight initKiModelleTab-Instanz).
    if (typeof _runTabCleanups === 'function') _runTabCleanups('ki-modelle');

    const myGen = ++_kiModelleInitGen;

    const container = document.getElementById('contentContainer');
    if (!container) return;

    _bindKiModelleActions();

    // Eine offene Backend-Form entwerten, damit stale Responses kein
    // neueres Modal/Tab-State ueberfahren. Kein No-Op-Bump wenn kein
    // Form offen ist (_beFormMode ist dann null).
    if (_beFormMode !== null) {
        _invalidateBackendForm();
    }

    // Filter-Bar
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.style.display = '';
        filterBar.classList.remove('is-hidden', 'u-hidden');
        filterBar.classList.add('u-flex');

        const searchGroup = document.createElement('div');
        searchGroup.className = 'filter-group';
        searchGroup.innerHTML = `<span class="filter-label">Suche:</span>
            <input type="text" class="filter-input" id="backendSearch" placeholder="Name/Modell..."
                data-input-action="update-backend-search">`;
        filterBar.appendChild(searchGroup);

        const catGroup = document.createElement('div');
        catGroup.className = 'filter-group';
        catGroup.id = 'backendCategoryFilters';
        catGroup.innerHTML = '<span class="filter-label">Kategorie:</span>';
        filterBar.appendChild(catGroup);

        const addGroup = document.createElement('div');
        addGroup.className = 'filter-group u-ml-auto';
        addGroup.innerHTML = '<button class="action-btn primary" data-action="open-add-backend-dialog">+ Backend</button>';
        filterBar.appendChild(addGroup);
    }

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    // Die Diagnose ist additiv: parallel starten, aber Netzwerk-, HTTP- und
    // JSON-Fehler innerhalb des optionalen Pfads abfangen. Pflichtdaten und
    // Backend-CRUD duerfen davon weder blockiert noch ausgeblendet werden.
    const contractDiagnosticsPromise = _loadEmbeddingContractDiagnostics();

    try {
        const [catResp, listResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/ki/backends/categories'),
            SharedUi.fetch.safeFetch('/api/admin/ki/backends/'),
        ]);
        // Guard #1: nach Promise.all-await. Stale Re-Init -> raus, bevor wir
        // den Container der neuen Instanz ueberschreiben.
        if (_kiModelleInitGen !== myGen) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-modelle') return;
        if (!catResp.ok || !listResp.ok) {
            container.innerHTML = '<div class="no-data">Backend-Daten konnten nicht geladen werden.</div>';
            return;
        }
        const [catData, listData] = await Promise.all([
            catResp.json(),
            listResp.json(),
        ]);
        // Guard #2: nach json-awaits. Re-Init kann zwischen Promise.all und
        // json passiert sein; backendsCategories/backendsData sind modul-global
        // und wuerden sonst von einem stale Lauf geclobbered.
        if (_kiModelleInitGen !== myGen) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-modelle') return;
        backendsCategories = catData.categories || [];
        backendsData = listData.backends || [];
        embeddingContractDiagnostics = { contract: {}, servers: [], backends: [] };
        embeddingContractDiagnosticsError = 'Vertragsdiagnose wird geladen ...';
        _renderCategoryFilters();
        renderBackendsTable();
        registerTabCleanup('ki-modelle', _cleanupKiModelle);
    } catch (e) {
        if (_kiModelleInitGen !== myGen) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-modelle') return;
        container.innerHTML = '<div class="no-data">Backend-Daten konnten nicht geladen werden.</div>';
        return;
    }

    const contractData = await contractDiagnosticsPromise;
    // Guard #3: Diagnose kann langsamer als die Pflichtdaten sein. Nach Tab-
    // Wechsel/Re-Init darf sie weder globale Daten noch ein neues DOM ersetzen.
    if (_kiModelleInitGen !== myGen) return;
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-modelle') return;
    embeddingContractDiagnostics = contractData || { contract: {}, servers: [], backends: [] };
    embeddingContractDiagnosticsError = contractData
        ? ''
        : 'Vertragsdiagnose konnte nicht geladen werden.';
    renderBackendsTable();
}

function _renderCategoryFilters() {
    const catGroup = document.getElementById('backendCategoryFilters');
    if (!catGroup) return;
    let html = '<span class="filter-label">Kategorie:</span>';
    html += `<button class="filter-btn${backendsFilterCategory === '' ? ' active' : ''}" data-action="set-backend-category-filter" data-cat-id="">Alle</button>`;
    backendsCategories.forEach(c => {
        html += `<button class="filter-btn${backendsFilterCategory === c.id ? ' active' : ''}" data-action="set-backend-category-filter" data-cat-id="${SharedUi.runtime.escapeAttr(c.id)}">${SharedUi.runtime.escapeHtml(c.label)}</button>`;
    });
    catGroup.innerHTML = html;
}

function _updateCatBtns() {
    const btns = document.querySelectorAll('#backendCategoryFilters .filter-btn');
    btns.forEach(btn => {
        const catId = btn.dataset.catId || '';
        btn.classList.toggle('active', catId === backendsFilterCategory);
    });
}

function _categoryLabel(cat) {
    const c = backendsCategories.find(x => x.id === cat);
    return c ? c.label : cat;
}

function _categoryBadgeClass(cat) {
    const map = { embedding: 'badge-info', llm: 'badge-success', cloud: 'badge-purple', vlm: 'badge-warning', ddu: 'badge-danger' };
    return map[cat] || 'badge-default';
}

function _providerBadgeClass(provider) {
    const map = { ollama: 'badge-default', anthropic: 'badge-purple', openai: 'badge-success', voyage: 'badge-info', docling: 'badge-danger' };
    return map[provider] || 'badge-default';
}

// Statische Embedding-Modelllisten fuer Cloud-Provider (statt models.list())
const _EMBEDDING_CLOUD_MODELS = {
    openai: ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'],
    voyage: ['voyage-3', 'voyage-3-lite', 'voyage-3-large', 'voyage-code-3'],
};

function _testStatusHtml(row) {
    if (row.last_test_ok === null || row.last_test_at === null) {
        return '<span class="badge badge-default">Nicht getestet</span>';
    }
    if (row.last_test_ok) {
        return '<span class="badge badge-success">OK</span>';
    }
    const err = row.last_test_error ? ` title="${SharedUi.runtime.escapeAttr(row.last_test_error)}"` : '';
    return `<span class="badge badge-danger"${err}>Fehler</span>`;
}

function _contractStateBadge(state) {
    const classByState = {
        valid: 'badge-success',
        unverified: 'badge-warning',
        not_applicable: 'badge-default',
        missing: 'badge-warning',
        unknown_schema_version: 'badge-danger',
        malformed: 'badge-danger',
        inconsistent: 'badge-danger',
        ambiguous: 'badge-danger',
        discovery_error: 'badge-danger',
        compatible: 'badge-success',
        mismatch: 'badge-danger',
        evidence_failed: 'badge-danger',
    };
    const cssClass = classByState[state] || 'badge-default';
    return `<span class="badge ${cssClass}">${SharedUi.runtime.escapeHtml(state || 'unknown')}</span>`;
}

function _contractCode(value) {
    return value
        ? `<code>${SharedUi.runtime.escapeHtml(value)}</code>`
        : '<span class="u-c-secondary">null</span>';
}

function _contractReasonText(...sources) {
    const reasons = [];
    sources.forEach(source => {
        (source || []).forEach(reason => {
            if (reason && !reasons.includes(reason)) reasons.push(reason);
        });
    });
    return reasons.length ? reasons.join(', ') : '-';
}

function _embeddingContractDiagnosticsHtml() {
    let html = '<section class="u-mt-24" id="embeddingContractDiagnostics">';
    html += '<h3>Embedding-Vertragsdiagnose</h3>';
    if (embeddingContractDiagnosticsError) {
        html += `<div class="no-data">${SharedUi.runtime.escapeHtml(embeddingContractDiagnosticsError)}</div></section>`;
        return html;
    }

    const contract = embeddingContractDiagnostics.contract || {};
    html += '<p class="u-text-xs-secondary">';
    html += `Schema V${SharedUi.runtime.escapeHtml(String(contract.schema_version ?? '-'))}`;
    html += contract.schema_sha256
        ? ` · SHA-256 ${_contractCode(contract.schema_sha256)}`
        : '';
    html += ' · read-only via /api/tags + /api/show</p>';

    const backendRows = embeddingContractDiagnostics.backends || [];
    html += '<h4>Konfigurierte Embedding-Backends</h4>';
    if (backendRows.length === 0) {
        html += '<div class="no-data">Keine aktiven Embedding-Backends.</div>';
    } else {
        html += '<div class="table-responsive"><table class="data-table u-w-full">';
        html += '<thead><tr><th>Backend</th><th>Angefordert</th><th>Status</th><th>Canonical/Profile</th><th>RAG</th><th>Grund</th></tr></thead><tbody>';
        backendRows.forEach(backend => {
            const resolution = backend.resolution || {};
            const profile = resolution.profile || {};
            const canonical = resolution.canonical_model_id || '-';
            const profileId = profile.profile_id || '-';
            const reasons = _contractReasonText(backend.reason_codes, resolution.reason_codes, profile.reason_codes);
            html += '<tr>';
            html += `<td>${SharedUi.runtime.escapeHtml(backend.backend_name || '-')}<br><small>${SharedUi.runtime.escapeHtml(backend.provider || '-')}</small></td>`;
            html += `<td>${_contractCode(backend.requested_model)}</td>`;
            html += `<td>${_contractStateBadge(backend.state)}</td>`;
            html += `<td>${_contractCode(canonical)}<br>${_contractCode(profileId)}</td>`;
            html += `<td>${backend.rag_verified ? '<span class="badge badge-success">verified</span>' : '<span class="badge badge-warning">nein</span>'}</td>`;
            html += `<td>${SharedUi.runtime.escapeHtml(reasons)}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    }

    html += '<h4 class="u-mt-24">Persistierte Index-Kompatibilität</h4>';
    const indexRows = embeddingContractDiagnostics.indexes || [];
    const indexError = embeddingContractDiagnostics.index_diagnostics_error;
    if (indexError) {
        html += `<div class="no-data">Indexdiagnose nicht verfügbar: ${SharedUi.runtime.escapeHtml(indexError)}</div>`;
    } else if (indexRows.length === 0) {
        html += '<div class="no-data">Noch keine persistierten Index-Vertragsmetadaten.</div>';
    } else {
        html += '<div class="table-responsive"><table class="data-table u-w-full embedding-contract-table">';
        html += '<thead><tr><th>Collection / Art</th><th>Profil / Endpoint</th><th>Status</th><th>Index-ID gespeichert / aktuell</th><th>Query-ID</th><th>Evidence</th><th>Grund / Aktion</th></tr></thead><tbody>';
        indexRows.forEach(indexRow => {
            const stored = indexRow.stored || {};
            const current = indexRow.current || {};
            const evidence = indexRow.evidence || {};
            html += '<tr>';
            html += `<td>${_contractCode(indexRow.collection_name)}<br><small>${SharedUi.runtime.escapeHtml(indexRow.vector_kind || '-')}</small></td>`;
            html += `<td>${_contractCode(stored.canonical_model_id)}<br>${_contractCode(stored.profile_id)} · ${_contractCode(stored.profile_endpoint)}</td>`;
            html += `<td>${_contractStateBadge(indexRow.compatibility_state)}</td>`;
            html += `<td>${_contractCode(stored.index_compatibility_id)}<br><small>aktuell: ${_contractCode(current.index_compatibility_id)}</small></td>`;
            html += `<td>${_contractCode(stored.query_compatibility_id)}</td>`;
            html += `<td>${SharedUi.runtime.escapeHtml(evidence.type || '-')}<br><small>n=${SharedUi.runtime.escapeHtml(String(evidence.sample_count ?? 0))} · ${SharedUi.runtime.escapeHtml(evidence.created_at || '-')}</small></td>`;
            html += `<td>${SharedUi.runtime.escapeHtml(indexRow.reason_code || '-')}<br><small>${SharedUi.runtime.escapeHtml(indexRow.operator_action || '-')}</small></td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    }

    html += '<h4 class="u-mt-24">Entdeckte Modellgruppen und Profile</h4>';
    const servers = embeddingContractDiagnostics.servers || [];
    if (servers.length === 0) {
        html += '<div class="no-data">Kein KIron-Vertragsserver entdeckt.</div>';
        html += '</section>';
        return html;
    }
    html += '<div class="table-responsive"><table class="data-table u-w-full embedding-contract-table">';
    html += '<thead><tr><th>Server / Schema</th><th>Canonical / Profil</th><th>Art / Endpoint</th><th>Verifikation</th><th>Rollenphase</th><th>Index-ID</th><th>Query-ID</th><th>Grund / Blocker</th></tr></thead><tbody>';
    servers.forEach(server => {
        const groups = server.groups || [];
        if (groups.length === 0) {
            html += '<tr>';
            html += `<td>${SharedUi.runtime.escapeHtml(server.backend_url || '-')}</td>`;
            html += '<td colspan="6">Kein gültiger Capability-Vertrag</td>';
            html += `<td>${SharedUi.runtime.escapeHtml(_contractReasonText(server.reason_codes))}</td>`;
            html += '</tr>';
            return;
        }
        groups.forEach(group => {
            const profiles = group.profiles || [];
            if (profiles.length === 0) {
                html += '<tr>';
                html += `<td>${SharedUi.runtime.escapeHtml(server.backend_url || '-')}<br>V${SharedUi.runtime.escapeHtml(String(group.schema_version ?? '-'))}</td>`;
                html += `<td>${_contractCode(group.reported_canonical_model_id || '-')}</td>`;
                html += '<td>-</td>';
                html += `<td>${_contractStateBadge(group.state)}</td>`;
                html += '<td>-</td><td>-</td><td>-</td>';
                html += `<td>${SharedUi.runtime.escapeHtml(_contractReasonText(group.reason_codes))}</td>`;
                html += '</tr>';
                return;
            }
            profiles.forEach(profile => {
                const inputType = profile.input_type || {};
                const verification = profile.verification || {};
                const rolePhase = inputType.role_sensitive === null
                    ? 'unknown'
                    : `sensitive=${String(inputType.role_sensitive)}, required=${String(inputType.required)}`;
                const reasons = _contractReasonText(
                    group.reason_codes,
                    profile.reason_codes,
                    verification.blocking_reasons,
                );
                html += '<tr>';
                html += `<td>${SharedUi.runtime.escapeHtml(server.backend_url || '-')}<br>V${SharedUi.runtime.escapeHtml(String(group.schema_version ?? '-'))}</td>`;
                html += `<td>${_contractCode(group.canonical_model_id)}<br>${_contractCode(profile.profile_id)}</td>`;
                html += `<td>${SharedUi.runtime.escapeHtml(profile.kind || '-')}<br>${_contractCode(profile.endpoint)}</td>`;
                html += `<td>${_contractStateBadge(verification.status || group.state)}<br>${profile.rag_verified ? '<span class="badge badge-success">rag_verified</span>' : '<span class="badge badge-warning">nicht RAG-verifiziert</span>'}</td>`;
                html += `<td>${SharedUi.runtime.escapeHtml(rolePhase)}</td>`;
                html += `<td>${_contractCode(profile.index_compatibility_id)}</td>`;
                html += `<td>${_contractCode(profile.query_compatibility_id)}</td>`;
                html += `<td>${SharedUi.runtime.escapeHtml(reasons)}</td>`;
                html += '</tr>';
            });
        });
    });
    html += '</tbody></table></div></section>';
    return html;
}

function renderBackendsTable() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    let filtered = backendsData;
    if (backendsFilterCategory) {
        filtered = filtered.filter(b => b.category === backendsFilterCategory);
    }
    if (backendsFilterSearch) {
        filtered = filtered.filter(b =>
            (b.name || '').toLowerCase().includes(backendsFilterSearch) ||
            (b.model || '').toLowerCase().includes(backendsFilterSearch) ||
            (b.url || '').toLowerCase().includes(backendsFilterSearch) ||
            (b.provider || '').toLowerCase().includes(backendsFilterSearch)
        );
    }

    if (filtered.length === 0) {
        container.innerHTML = `<div class="no-data">
            ${backendsData.length === 0 ? 'Keine Backends konfiguriert.' : 'Keine Treffer fuer den aktuellen Filter.'}
            <br><br><button class="action-btn primary" data-action="open-add-backend-dialog">+ Backend hinzufuegen</button>
        </div>` + _embeddingContractDiagnosticsHtml();
        return;
    }

    let html = '<div class="table-responsive"><table class="data-table ki-modelle-table u-w-full">';
    html += '<thead><tr>';
    html += '<th>Name</th><th>Kategorie</th><th>Provider</th><th>Modell</th>';
    html += '<th>URL</th><th>Prio</th><th>Status</th><th>Aktiv</th><th class="u-width-100">Aktionen</th>';
    html += '</tr></thead><tbody>';

    filtered.forEach(b => {
        const catBadge = `<span class="badge ${_categoryBadgeClass(b.category)}">${SharedUi.runtime.escapeHtml(_categoryLabel(b.category))}</span>`;
        const provBadge = `<span class="badge ${_providerBadgeClass(b.provider)}">${SharedUi.runtime.escapeHtml(b.provider || '-')}</span>`;
        const displayUrl = _providerNeedsApiKey(b.provider) ? (b.base_url || '-') : (b.url || '-');
        const toggleChecked = b.is_active ? 'checked' : '';
        const rowClass = b.is_active ? '' : ' class="u-dim"';

        html += `<tr${rowClass}>`;
        html += `<td>${SharedUi.runtime.escapeHtml(b.name || '-')}</td>`;
        html += `<td>${catBadge}</td>`;
        html += `<td>${provBadge}</td>`;
        html += `<td><code>${SharedUi.runtime.escapeHtml(b.model || '-')}</code></td>`;
        html += `<td class="u-mw-180" title="${SharedUi.runtime.escapeAttr(displayUrl)}">${SharedUi.runtime.escapeHtml(displayUrl)}</td>`;
        html += `<td class="u-text-center">${b.priority}</td>`;
        html += `<td>${_testStatusHtml(b)}</td>`;
        html += `<td class="u-text-center"><label class="toggle-switch"><input type="checkbox" ${toggleChecked} data-change-action="toggle-backend" data-backend-id="${b.id}"><span class="toggle-slider"></span></label></td>`;
        html += '<td><div class="action-buttons">';
        html += `<button class="action-btn-sm" title="Testen" data-action="test-backend" data-backend-id="${b.id}">&#x21BB;</button>`;
        html += `<button class="action-btn-sm" title="Bearbeiten" data-action="open-edit-backend-modal" data-backend-id="${b.id}">&#x270E;</button>`;
        html += `<button class="action-btn-sm danger" title="Loeschen" data-action="delete-backend" data-backend-id="${b.id}" data-backend-name="${SharedUi.runtime.escapeAttr(b.name)}">&#x2716;</button>`;
        html += '</div></td>';
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    html += _embeddingContractDiagnosticsHtml();
    container.innerHTML = html;
}

// =========================================================================
//  Category picker (step 1 of Add)
// =========================================================================

function openAddBackendDialog() {
    SharedUi.modal.open({
        title: 'Neues Backend hinzufuegen',
        body: _buildCategoryGrid(),
        footer: '<button class="action-btn" data-action="close-backend-modal">Abbrechen</button>',
        size: 'medium',
    });
}

function _buildCategoryGrid() {
    let html = '<p class="u-text-sm-secondary u-mb-16">Kategorie waehlen:</p>';
    html += '<div class="backend-category-grid">';
    backendsCategories.forEach(c => {
        html += `<div class="backend-category-card" data-action="open-backend-form-add" data-category="${SharedUi.runtime.escapeAttr(c.id)}">`;
        html += `<strong>${SharedUi.runtime.escapeHtml(c.label)}</strong>`;
        html += `<span class="u-text-xs-secondary u-mt-4">${SharedUi.runtime.escapeHtml(c.description || '')}</span>`;
        html += '</div>';
    });
    html += '</div>';
    return html;
}

// =========================================================================
//  Unified backend form (Add + Edit)
// =========================================================================

/**
 * Opens the backend form modal.
 *
 * @param {'add'|'edit'} mode
 * @param {string}       category  - backend category id
 * @param {object}       [data]    - existing backend row (edit mode)
 */
function _openBackendForm(mode, category, data) {
    _beFormMode = mode;
    _beFormEditId = data ? data.id : null;
    _beFormGeneration++;

    // #1051: Baseline nur im Edit-Modus snapshotten — Add hat keinen
    // "unveraendert"-Vergleichswert. Source ist das uebergebene Backend-Row,
    // nicht das DOM (DOM ist zu diesem Zeitpunkt noch nicht gerendert).
    if (mode === 'edit' && data) {
        _beFormBaseline = {
            provider: data.provider || '',
            model:    data.model || '',
            base_url: data.base_url || '',
        };
    } else {
        _beFormBaseline = null;
    }
    // #1052: Bei jedem Modal-Open frische Probe verlangen.
    _embeddingDimProbe = null;

    const catInfo = backendsCategories.find(c => c.id === category) || {};
    const providers = catInfo.providers || [];
    const currentProvider = data ? data.provider : (providers[0] || 'ollama');
    const needsApiKey = _providerNeedsApiKey(currentProvider);
    const supportsBaseUrl = currentProvider === 'anthropic' || currentProvider === 'openai';
    const isEdit = mode === 'edit';

    let body = `<input type="hidden" id="beFormCategory" value="${SharedUi.runtime.escapeAttr(category)}">`;

    // Provider
    if (providers.length > 1) {
        body += '<div class="modal-field"><label>Provider</label><select class="config-input" id="beFormProvider" data-change-action="backend-provider-change">';
        providers.forEach(p => {
            const sel = p === currentProvider ? ' selected' : '';
            body += `<option value="${p}"${sel}>${SharedUi.runtime.escapeHtml(p)}</option>`;
        });
        body += '</select></div>';
    } else {
        body += `<input type="hidden" id="beFormProvider" value="${SharedUi.runtime.escapeAttr(currentProvider)}">`;
    }

    // Name
    body += `<div class="modal-field"><label>Name</label><input type="text" class="config-input" id="beFormName" value="${SharedUi.runtime.escapeAttr((data && data.name) || '')}" placeholder="z.B. GPU-Server 1"></div>`;

    // URL (Ollama)
    body += `<div class="modal-field${needsApiKey ? ' u-hidden' : ''}" id="beFormUrlField">`;
    body += '<label>Ollama URL</label>';
    body += '<div class="u-flex-gap-8">';
    body += `<input type="text" class="config-input u-flex-1" id="beFormUrl" value="${SharedUi.runtime.escapeAttr((data && data.url) || '')}" placeholder="http://localhost:11434">`;
    body += '<button class="action-btn" id="beFormTestOllamaBtn" data-action="test-url-or-docling">Testen</button>';
    body += '</div>';
    body += '<div id="beFormOllamaStatus" class="u-mt-4"></div>';
    body += '<small class="field-hint u-hidden u-text-xs-secondary u-mt-4" id="beFormUrlHint">DDU laeuft in KIara nur als Remote-Service, z.B. ueber docling-serve oder docling-proxy.</small>';
    body += '</div>';

    // Base URL (Cloud — nur Anthropic/OpenAI, nicht Voyage)
    const _defaultBaseUrls = { anthropic: 'https://api.anthropic.com', openai: 'https://api.openai.com/v1' };
    const _baseUrlValue = (data && data.base_url) || (supportsBaseUrl ? _defaultBaseUrls[currentProvider] || '' : '');
    body += `<div class="modal-field${supportsBaseUrl ? '' : ' u-hidden'}" id="beFormBaseUrlField">`;
    body += '<label>Base URL (optional)</label>';
    body += `<input type="text" class="config-input" id="beFormBaseUrl" value="${SharedUi.runtime.escapeAttr(_baseUrlValue)}" placeholder="${SharedUi.runtime.escapeAttr(_defaultBaseUrls[currentProvider] || 'https://api.anthropic.com')}">`;
    body += '</div>';

    // API-Key (Cloud — Anthropic/OpenAI/Voyage)
    body += `<div class="modal-field${needsApiKey ? '' : ' u-hidden'}" id="beFormApiKeyField">`;
    body += '<label>API-Key</label>';
    body += '<div class="u-flex-gap-8">';
    const _apiKeyInitial = (isEdit && data && data.api_key === KI_PASSWORD_MASK) ? KI_PASSWORD_MASK : '';
    body += `<input type="password" class="config-input u-flex-1" id="beFormApiKey" value="${_apiKeyInitial}" placeholder="${isEdit ? 'Leer = beibehalten' : 'sk-...'}">`;
    body += '<button class="action-btn" id="beFormTestCloudBtn" data-action="test-cloud-key">Testen</button>';
    body += '</div>';
    body += '<div id="beFormCloudStatus" class="u-mt-4"></div>';
    body += '</div>';

    // Model - select (filled by test) + manual input
    body += '<div class="modal-field"><label>Modell</label>';
    body += '<select class="config-input" id="beFormModelSelect"><option value="">-- Erst Verbindung testen oder eingeben --</option></select>';
    body += `<input type="text" class="config-input" id="beFormModelManual" value="${SharedUi.runtime.escapeAttr((data && data.model) || '')}" placeholder="Modell manuell eingeben" class="u-mt-4">`;
    body += '</div>';

    // Priority
    body += `<div class="modal-field"><label>Prioritaet (niedriger = bevorzugt)</label>`;
    body += `<input type="number" class="config-input" id="beFormPriority" value="${SharedUi.runtime.escapeAttr(String(data?.priority ?? 10))}" min="1" max="100"></div>`;

    // Trusted (nur fuer Cloud-Provider anthropic/openai, sonst implizit trusted/nicht relevant)
    // Beide Felder immer rendern — _onBeFormProviderChange() toggelt sie per u-hidden,
    // damit ein Provider-Wechsel im laufenden Modal die Sichtbarkeit korrekt aktualisiert.
    const trustedChecked = (data && data.trusted) ? 'checked' : '';
    const _trustedHidden = (currentProvider === 'anthropic' || currentProvider === 'openai') ? '' : ' u-hidden';
    body += `<div class="modal-field${_trustedHidden}" id="beFormTrustedField"><label><input type="checkbox" id="beFormTrusted" ${trustedChecked}> Vertrauenswuerdig (Restriction-Level-Pruefung ueberspringen)</label></div>`;
    const _trustedHintHidden = (currentProvider === 'ollama') ? '' : ' u-hidden';
    body += `<div class="modal-field u-text-xs-secondary${_trustedHintHidden}" id="beFormTrustedOllamaHint">Lokal / implizit vertrauenswuerdig — keine Restriction-Level-Pruefung noetig.</div>`;

    // Active (edit only)
    if (isEdit) {
        body += `<div class="modal-field"><label><input type="checkbox" id="beFormActive" ${data.is_active ? 'checked' : ''}> Aktiv</label></div>`;
    }

    const title = isEdit
        ? `Backend bearbeiten: ${SharedUi.runtime.escapeHtml((data && data.name) || 'Backend ' + data.id)}`
        : `Backend hinzufuegen: ${catInfo.label || category}`;
    const saveLabel = isEdit ? 'Speichern' : 'Erstellen';

    if (isEdit) {
        // Fresh modal
        SharedUi.modal.open({
            title,
            body,
            footer: `<button class="action-btn" data-action="close-backend-modal">Abbrechen</button>
                     <button class="action-btn primary" data-action="save-backend">Speichern</button>`,
        });
    } else {
        // Rewrite existing category-picker modal
        const modalBody = document.querySelector('.modal-body');
        const modalFooter = document.querySelector('.modal-footer');
        const modalHeader = document.querySelector('.modal-header');
        if (modalBody && modalFooter && modalHeader) {
            modalHeader.textContent = title;
            modalBody.innerHTML = body;
            modalFooter.innerHTML = `<button class="action-btn" data-action="close-backend-modal">Abbrechen</button>
                <button class="action-btn primary" data-action="save-backend">Erstellen</button>`;
        }
    }
    // Provider-abhaengige Felder aktualisieren
    _onBeFormProviderChange();
    // DDU: Modell-Default vorausfuellen wenn leer
    if (category === 'ddu') {
        const modelInput = document.getElementById('beFormModelManual');
        if (modelInput && !modelInput.value) modelInput.value = 'docling';
    }
}

// Legacy entry points
function openProviderForm(category) { _openBackendForm('add', category); }
function openEditBackendModal(backendId) {
    const b = backendsData.find(x => x.id === backendId);
    if (!b) return;
    _openBackendForm('edit', b.category, b);
}

// =========================================================================
//  Provider toggle (shared)
// =========================================================================

function _onBeFormProviderChange() {
    const provider = document.getElementById('beFormProvider')?.value || 'ollama';
    const category = document.getElementById('beFormCategory')?.value || '';
    const needsApiKey = _providerNeedsApiKey(provider);
    const supportsBaseUrl = provider === 'anthropic' || provider === 'openai';
    const isDocling = provider === 'docling';
    const urlField = document.getElementById('beFormUrlField');
    const baseUrlField = document.getElementById('beFormBaseUrlField');
    const apiKeyField = document.getElementById('beFormApiKeyField');
    const urlInput = document.getElementById('beFormUrl');
    const urlHint = document.getElementById('beFormUrlHint');
    const baseUrlInput = document.getElementById('beFormBaseUrl');
    if (urlField) urlField.classList.toggle('u-hidden', needsApiKey);
    if (baseUrlField) baseUrlField.classList.toggle('u-hidden', !supportsBaseUrl);
    if (apiKeyField) apiKeyField.classList.toggle('u-hidden', !needsApiKey);
    // Trusted-Feld konsistent mit URL/BaseURL/ApiKey toggeln, damit ein
    // Provider-Wechsel die Checkbox sichtbar/unsichtbar schaltet.
    const trustedField = document.getElementById('beFormTrustedField');
    const trustedHint = document.getElementById('beFormTrustedOllamaHint');
    const showTrusted = (provider === 'anthropic' || provider === 'openai');
    if (trustedField) trustedField.classList.toggle('u-hidden', !showTrusted);
    if (trustedHint) trustedHint.classList.toggle('u-hidden', provider !== 'ollama');
    // Base URL vorausfuellen bei Provider-Wechsel
    if (supportsBaseUrl && baseUrlInput) {
        const defaults = {
            anthropic: 'https://api.anthropic.com',
            openai: 'https://api.openai.com/v1',
        };
        const allDefaults = Object.values(defaults);
        const currentVal = baseUrlInput.value;
        if (!currentVal || allDefaults.includes(currentVal)) {
            baseUrlInput.value = defaults[provider] || '';
        }
        baseUrlInput.placeholder = defaults[provider] || '';
    }
    // DDU/Docling ist remote-only und braucht eine URL.
    if (urlInput) {
        if (isDocling) {
            urlInput.placeholder = 'http://docling-proxy:5001';
            urlInput.required = true;
        } else if (!needsApiKey) {
            urlInput.placeholder = 'http://localhost:11434';
            urlInput.required = true;
        }
    }
    if (urlHint) urlHint.classList.toggle('u-hidden', !isDocling);
    // Label anpassen
    const urlLabel = urlField?.querySelector('label');
    if (urlLabel) {
        urlLabel.textContent = isDocling ? 'DDU URL' : 'Ollama URL';
    }
    // Embedding-Cloud-Provider: statische Modellliste setzen
    if (category === 'embedding' && _EMBEDDING_CLOUD_MODELS[provider]) {
        _fillModelSelect(_EMBEDDING_CLOUD_MODELS[provider]);
        // Vorhandenes Modell vorselektieren
        const manualInput = document.getElementById('beFormModelManual');
        const select = document.getElementById('beFormModelSelect');
        if (manualInput && select) {
            const currentModel = manualInput.value;
            const opts = Array.from(select.options).map(o => o.value);
            if (opts.includes(currentModel)) {
                select.value = currentModel;
            }
        }
    }
}

// =========================================================================
//  Connection tests (shared)
// =========================================================================

function _fillModelSelect(models) {
    const select = document.getElementById('beFormModelSelect');
    if (!select) return;
    select.innerHTML = '';
    if (models.length === 0) {
        select.innerHTML = '<option value="">-- Keine Modelle gefunden --</option>';
        return;
    }
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        select.appendChild(opt);
    });
    // Change-Handler ueber den zentralen _handleKiModelleChange-Dispatcher:
    // data-change-action="clear-manual-model-on-select" triggert, wenn der
    // User eine Option waehlt, das Leeren des manuellen Modellfelds.
    select.setAttribute('data-change-action', 'clear-manual-model-on-select');
}

function _testUrlOrDocling() {
    const provider = document.getElementById('beFormProvider')?.value || 'ollama';
    if (provider === 'docling') {
        _testDoclingBackend();
    } else {
        _testOllamaUrl();
    }
}

async function _testDoclingBackend() {
    const url = document.getElementById('beFormUrl')?.value || '';
    const btn = document.getElementById('beFormTestOllamaBtn');
    const statusEl = document.getElementById('beFormOllamaStatus');
    const myGen = _beFormGeneration;
    if (btn) { btn.disabled = true; btn.textContent = 'Teste...'; }
    if (statusEl) statusEl.innerHTML = '<span class="u-c-secondary">Wird geprueft...</span>';

    try {
        if (_beFormMode === 'edit' && _beFormEditId) {
            // Gespeichertes Backend: API-Test-Endpoint nutzen
            const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/backends/${_beFormEditId}/test`, { method: 'POST' });
            const data = await resp.json();
            if (myGen !== _beFormGeneration) return;
            if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Test fehlgeschlagen');
            if (statusEl) statusEl.innerHTML = '<span class="u-c-success">Docling verfuegbar</span>';
        } else if (url.trim()) {
            // Neues Backend mit URL: Health-Check per ddu-available
            const resp = await SharedUi.fetch.safeFetch('/api/admin/ddu-available?url=' + encodeURIComponent(url.trim()));
            const data = await resp.json();
            if (myGen !== _beFormGeneration) return;
            if (data.available) {
                if (statusEl) statusEl.innerHTML = '<span class="u-c-success">DDU verfuegbar (' + SharedUi.runtime.escapeHtml(data.mode || 'ok') + ')</span>';
            } else {
                throw new Error(data.message || 'DDU nicht verfuegbar');
            }
        } else {
            SharedUi.notify.show('Bitte Remote-DDU-URL eingeben', 'error');
            throw new Error('Remote-DDU-URL ist erforderlich');
        }
    } catch (e) {
        if (myGen !== _beFormGeneration) return;
        if (statusEl) statusEl.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    } finally {
        if (myGen === _beFormGeneration && btn) { btn.disabled = false; btn.textContent = 'Testen'; }
    }
}

async function _testOllamaUrl() {
    const url = document.getElementById('beFormUrl')?.value;
    if (!url) { SharedUi.notify.show('Bitte URL eingeben', 'error'); return; }

    const category = document.getElementById('beFormCategory')?.value || null;
    const categoryLabels = { embedding: 'Embedding', vlm: 'VLM', llm: 'LLM' };
    const catLabel = categoryLabels[category] || '';

    const btn = document.getElementById('beFormTestOllamaBtn');
    const statusEl = document.getElementById('beFormOllamaStatus');
    const myGen = _beFormGeneration;
    if (btn) { btn.disabled = true; btn.textContent = 'Teste...'; }
    if (statusEl) statusEl.innerHTML = '<span class="u-c-secondary">Verbindung wird getestet...</span>';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, category }),
        });
        const data = await resp.json();
        if (myGen !== _beFormGeneration) return;
        if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Verbindung fehlgeschlagen');

        const models = data.models || [];
        _fillModelSelect(models);
        const countLabel = catLabel ? `${models.length} ${catLabel}-Modelle` : `${models.length} Modelle`;
        if (statusEl) {
            if (models.length > 0) {
                statusEl.innerHTML = `<span class="u-c-success">Erreichbar (${countLabel})</span>`;
            } else {
                statusEl.innerHTML = `<span class="u-c-warning">Erreichbar, aber keine passenden ${catLabel}-Modelle gefunden</span>`;
            }
        }
    } catch (e) {
        if (myGen !== _beFormGeneration) return;
        if (statusEl) statusEl.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    } finally {
        if (myGen === _beFormGeneration && btn) { btn.disabled = false; btn.textContent = 'Testen'; }
    }
}

async function _testCloudKey() {
    const provider = document.getElementById('beFormProvider')?.value || 'anthropic';
    const category = document.getElementById('beFormCategory')?.value || '';
    const apiKey = document.getElementById('beFormApiKey')?.value;
    const baseUrl = document.getElementById('beFormBaseUrl')?.value || '';
    const isMasked = !apiKey || apiKey === KI_PASSWORD_MASK;

    // #1051: Edit-Modus mit unveraendertem Key — nur dann sicher den
    // gespeicherten Backend-Test wiederzuverwenden, wenn auch Provider/
    // Modell/Base-URL unveraendert sind. Andernfalls testet die UI den
    // alten Persistenzzustand gegen einen frisch geaenderten Dialog und
    // liefert irrefuehrende Ergebnisse.
    if (isMasked && _beFormMode === 'edit' && _beFormEditId) {
        if (_cloudFieldsUnchanged()) {
            return _testExistingBackend(_beFormEditId);
        }
        SharedUi.notify.show(
            'Cloud-Felder (Provider/Modell/Base-URL) geaendert — bitte API-Key erneut eingeben',
            'error');
        return;
    }

    if (!apiKey) { SharedUi.notify.show('Bitte API-Key eingeben', 'error'); return; }

    // Embedding-Cloud-Provider: Probe-Embedding statt models.list()
    if (category === 'embedding' && (provider === 'openai' || provider === 'voyage')) {
        return _testEmbeddingCloud(provider, apiKey, baseUrl);
    }

    const btn = document.getElementById('beFormTestCloudBtn');
    const statusEl = document.getElementById('beFormCloudStatus');
    const myGen = _beFormGeneration;
    if (btn) { btn.disabled = true; btn.textContent = 'Teste...'; }
    if (statusEl) statusEl.innerHTML = '<span class="u-c-secondary">Verbindung wird getestet...</span>';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/test-cloud', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, api_key: apiKey, base_url: baseUrl }),
        });
        const data = await resp.json();
        if (myGen !== _beFormGeneration) return;
        if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Verbindung fehlgeschlagen');

        const models = data.models || [];
        _fillModelSelect(models);
        if (statusEl) {
            if (models.length > 0) {
                statusEl.innerHTML = `<span class="u-c-success">Erreichbar (${models.length} Modelle)</span>`;
            } else {
                statusEl.innerHTML = `<span class="u-c-warning">Erreichbar, aber keine Modelle gefunden</span>`;
            }
        }
    } catch (e) {
        if (myGen !== _beFormGeneration) return;
        if (statusEl) statusEl.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    } finally {
        if (myGen === _beFormGeneration && btn) { btn.disabled = false; btn.textContent = 'Testen'; }
    }
}

async function _testEmbeddingCloud(provider, apiKey, baseUrl) {
    const manualModel = (document.getElementById('beFormModelManual')?.value || '').trim();
    const selectModel = document.getElementById('beFormModelSelect')?.value || '';
    const model = manualModel || selectModel;

    if (!model) { SharedUi.notify.show('Bitte Embedding-Modell auswaehlen oder eingeben', 'error'); return; }

    const btn = document.getElementById('beFormTestCloudBtn');
    const statusEl = document.getElementById('beFormCloudStatus');
    const myGen = _beFormGeneration;
    if (btn) { btn.disabled = true; btn.textContent = 'Teste...'; }
    if (statusEl) statusEl.innerHTML = '<span class="u-c-secondary">Probe-Embedding wird getestet...</span>';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/test-embedding-cloud', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, api_key: apiKey, model, base_url: baseUrl }),
        });
        const data = await resp.json();
        if (myGen !== _beFormGeneration) return;
        if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Probe-Embedding fehlgeschlagen');

        const dim = data.dimension;
        if (statusEl) {
            statusEl.innerHTML = `<span class="u-c-success">Erfolgreich — Dimension: ${dim}</span>`;
        }
        // #1052: Probe-Snapshot fuer den spaeteren Save-Pfad. Wird ueber
        // _embeddingProbeMatchesCurrent() lazy validiert — kein eager
        // Invalidator-Listener noetig, weil Save-Pfad ohnehin gegen die
        // aktuellen DOM-Werte vergleicht. Fingerprint statt Klartext-Key
        // (#1619), sonst invalidiert ein noch im Feld stehender Klartext-Key
        // (Add) die Probe, obwohl dasselbe Credential getestet wurde (#1103).
        _embeddingDimProbe = {
            dim,
            model,
            provider,
            base_url: baseUrl || '',
            api_key_fingerprint: _apiKeyFingerprint(apiKey),
        };
    } catch (e) {
        if (myGen !== _beFormGeneration) return;
        if (statusEl) statusEl.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    } finally {
        if (myGen === _beFormGeneration && btn) { btn.disabled = false; btn.textContent = 'Testen'; }
    }
}

async function _testExistingBackend(backendId) {
    const btn = document.getElementById('beFormTestCloudBtn') || document.getElementById('beFormTestOllamaBtn');
    const statusEl = document.getElementById('beFormCloudStatus') || document.getElementById('beFormOllamaStatus');
    const myGen = _beFormGeneration;
    if (btn) { btn.disabled = true; btn.textContent = 'Teste...'; }
    if (statusEl) statusEl.innerHTML = '<span class="u-c-secondary">Verbindung wird getestet...</span>';

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/backends/${backendId}/test`, { method: 'POST' });
        const data = await resp.json();
        if (myGen !== _beFormGeneration) return;
        if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Verbindung fehlgeschlagen');

        // Embedding-Cloud-Backends geben dimension statt models zurueck
        if (data.dimension) {
            if (statusEl) {
                statusEl.innerHTML = `<span class="u-c-success">Erfolgreich — Dimension: ${data.dimension}</span>`;
            }
        } else {
            const models = data.models || [];
            _fillModelSelect(models);
            if (statusEl) {
                if (models.length > 0) {
                    statusEl.innerHTML = `<span class="u-c-success">Erreichbar (${models.length} Modelle)</span>`;
                } else {
                    statusEl.innerHTML = `<span class="u-c-warning">Erreichbar, aber keine Modelle gefunden</span>`;
                }
            }
        }
    } catch (e) {
        if (myGen !== _beFormGeneration) return;
        if (statusEl) statusEl.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    } finally {
        if (myGen === _beFormGeneration && btn) { btn.disabled = false; btn.textContent = 'Testen'; }
    }
}

// =========================================================================
//  Save (shared)
// =========================================================================

async function _saveBackend() {
    // Snapshot vor jeglichen awaits — alle spaeteren Guards vergleichen gegen
    // diese Generation. Ein Re-Open oder Modal-Close inkrementiert
    // _beFormGeneration und entwertet damit automatisch diesen Save-Lauf.
    const mySaveGen  = _beFormGeneration;
    const mySaveMode = _beFormMode;
    const mySaveId   = _beFormEditId;

    const category = document.getElementById('beFormCategory')?.value;
    const provider = document.getElementById('beFormProvider')?.value || 'ollama';
    const name = document.getElementById('beFormName')?.value || '';
    const url = (document.getElementById('beFormUrl')?.value || '').trim();
    const baseUrl = (document.getElementById('beFormBaseUrl')?.value || '').trim();
    const apiKey = document.getElementById('beFormApiKey')?.value || '';
    const manualModel = (document.getElementById('beFormModelManual')?.value || '').trim();
    const selectModel = document.getElementById('beFormModelSelect')?.value || '';
    const model = manualModel || selectModel;
    const priority = parseInt(document.getElementById('beFormPriority')?.value || '10', 10);

    if (!model) { SharedUi.notify.show('Bitte Modell angeben', 'error'); return; }

    const isEdit = mySaveMode === 'edit';
    const providerChanged = isEdit && _beFormBaseline && provider !== _beFormBaseline.provider;
    const needsApiKey = _providerNeedsApiKey(provider);
    if (!needsApiKey && !url) {
        SharedUi.notify.show(provider === 'docling' ? 'Bitte Remote-DDU-URL eingeben' : 'Bitte URL eingeben', 'error');
        return;
    }
    if (providerChanged && needsApiKey && (!apiKey || apiKey === KI_PASSWORD_MASK)) {
        SharedUi.notify.show('Provider geaendert — bitte neuen API-Key eingeben', 'error');
        return;
    }

    const trustedEl = document.getElementById('beFormTrusted');
    const trusted = trustedEl ? trustedEl.checked : false;
    const payload = { category, provider, name, base_url: baseUrl, model, priority, trusted };
    if (!needsApiKey) payload.url = url;
    if (apiKey && apiKey !== KI_PASSWORD_MASK) payload.api_key = apiKey;
    if (providerChanged && !needsApiKey) {
        const confirmed = await SharedUi.modal.confirm({
            message: 'Provider gewechselt - gespeicherter API-Key wird geloescht. Fortfahren?'
        });
        // Generation-Guard: Re-Open zwischen Dialog-await und Save entwertet diesen Lauf.
        if (mySaveGen !== _beFormGeneration) return;
        if (!confirmed) return;
        payload.api_key = '';
    }

    if (isEdit) {
        payload.is_active = document.getElementById('beFormActive')?.checked || false;
    }

    // #1052: Probe-Dimension fuer Cloud-Embedding-Backends ins Payload
    // uebernehmen, falls sie noch zum aktuellen Dialog passt. Server-seitig
    // ueberschreibt _normalize_embedding_dim() bekannte Modelle mit hardcoded
    // Werten — fuer unbekannte Modelle ist die Probe der einzige Weg, die
    // Save-Validation positiv zu bestehen.
    const isCloudEmbedding = category === 'embedding' && (provider === 'openai' || provider === 'voyage');
    if (isCloudEmbedding && _embeddingProbeMatchesCurrent()) {
        payload.extra_config = Object.assign(
            {},
            payload.extra_config || {},
            { embedding_dim: _embeddingDimProbe.dim },
        );
    }

    try {
        const fetchUrl = isEdit ? `/api/admin/ki/backends/${mySaveId}` : '/api/admin/ki/backends/';
        const resp = await SharedUi.fetch.safeFetch(fetchUrl, {
            method: isEdit ? 'PUT' : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        // Guard #1: nach Transport-await. Stale Response -> keine UI-Semantik.
        if (mySaveGen !== _beFormGeneration) return;

        const data = await resp.json();
        // Guard #2: nach json-await. Re-Open kann dazwischen passiert sein.
        if (mySaveGen !== _beFormGeneration) return;

        if (!resp.ok) throw new Error(data.detail || data.error || data.message || (isEdit ? 'Fehler beim Speichern' : 'Fehler beim Erstellen'));
        SharedUi.notify.show(isEdit ? 'Backend aktualisiert' : 'Backend erstellt', 'success');
        _closeBackendModal();    // Bumpt Generation (harmlos doppelt gegen initKiModelleTab-Reload unten)
        initKiModelleTab();
    } catch (e) {
        if (mySaveGen !== _beFormGeneration) return;   // stale error still halten
        SharedUi.notify.show(e.message, 'error');
    }
}

// =========================================================================
//  Toggle / Test / Delete (table actions)
// =========================================================================

async function toggleBackend(backendId, isActive) {
    // Row-lokaler Lock: Doppel-/Race-Klicks auf dieselbe Zeile blocken.
    if (_toggleBackendInFlight.has(backendId)) return;
    _toggleBackendInFlight.add(backendId);
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/backends/${backendId}/toggle`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_active: isActive }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Fehler');
        // Update local data
        const b = backendsData.find(x => x.id === backendId);
        if (b) b.is_active = isActive;
        renderBackendsTable();
    } catch (e) {
        // Lokal revertieren statt Full-Reload: ein offenes Modal/Dialog
        // darf durch Toggle-Fail nicht ueberfahren werden (V30 Risiken:
        // stale Responses duerfen nicht per Tab-Reload juengeren
        // Dialogzustand zerstoeren).
        const b = backendsData.find(x => x.id === backendId);
        if (b) b.is_active = !isActive;
        SharedUi.notify.show(e.message, 'error');
        renderBackendsTable();
    } finally {
        _toggleBackendInFlight.delete(backendId);
    }
}

async function testBackend(backendId, btn) {
    return SharedUi.runtime.withButtonLock(btn, async () => {
        btn.innerHTML = '...';
        try {
            const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/backends/${backendId}/test`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Test fehlgeschlagen');
            SharedUi.notify.show('Verbindung erfolgreich', 'success');
            // Update local
            const b = backendsData.find(x => x.id === backendId);
            if (b) { b.last_test_ok = true; b.last_test_at = new Date().toISOString(); b.last_test_error = ''; }
            renderBackendsTable();
        } catch (e) {
            SharedUi.notify.show('Test fehlgeschlagen: ' + e.message, 'error');
            const b = backendsData.find(x => x.id === backendId);
            if (b) { b.last_test_ok = false; b.last_test_at = new Date().toISOString(); b.last_test_error = e.message; }
            renderBackendsTable();
        }
    });
}

async function deleteBackend(backendId, name) {
    // Row-lokaler Lock: SharedUi.modal.confirm() ist fuer den User natuerlicher
    // Doppelklick-Schutz, aber ein programmatischer Re-Trigger (z.B. zwei
    // Confirm-Events in Folge) darf keine parallelen DELETE-Requests erzeugen.
    if (_deleteBackendInFlight.has(backendId)) return;
    if (!await SharedUi.modal.confirm({ message: `Backend "${name}" wirklich loeschen?` })) return;
    _deleteBackendInFlight.add(backendId);
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/backends/${backendId}`, { method: 'DELETE' });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.error || data.message || 'Fehler');
        SharedUi.notify.show('Backend geloescht', 'success');
        // Tab-Guard: User kann waehrend des DELETE-await gewechselt haben.
        // initKiModelleTab() resettet filterBar/container.innerHTML SOFORT
        // (vor dem eigenen adminCurrentSubTab-Check nach den Daten-awaits),
        // das wuerde sonst den neuen Tab-Inhalt ueberschreiben.
        if (typeof adminCurrentSubTab === 'undefined' || adminCurrentSubTab === 'ki-modelle') {
            initKiModelleTab();
        }
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    } finally {
        _deleteBackendInFlight.delete(backendId);
    }
}
