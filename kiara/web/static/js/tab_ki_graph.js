/**
 * KIara Admin - KI > Wissensgraph Sub-Tab
 */

let _graphInitSeq = 0;

const _GRAPH_ACTIONS = new Set([
    'sync-erp-graph',
    'clear-graph',
    'save-graph-config',
    'reset-ki-graph-tab',
]);

function _handleGraphClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_GRAPH_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'sync-erp-graph':       syncErpGraph(el); break;
        case 'clear-graph':          clearGraph(el); break;
        case 'save-graph-config':    saveGraphConfig(el); break;
        case 'reset-ki-graph-tab':   initKiGraphTab(); break;
    }
}

function _bindGraphActions() {
    if (document.body._graphActionsBound) return;
    document.body._graphActionsBound = true;
    document.body.addEventListener('click', _handleGraphClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-graph-actions-bound', 'true');
}

async function initKiGraphTab() {
    _bindGraphActions();
    // Re-Init-Schutz
    if (typeof _runTabCleanups === 'function') _runTabCleanups('ki-graph');
    _graphInitSeq++;
    const mySeq = _graphInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.classList.remove('u-flex', 'u-visible', 'u-block');
        filterBar.classList.add('u-hidden');
    }

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    // Cleanup registrieren
    registerTabCleanup('ki-graph', () => { _graphInitSeq++; kiaraPolling.stop('ki-graph:stats'); });

    try {
        const [statsResp, settingsResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/graph/stats'),
            SharedUi.fetch.safeFetch('/api/admin/settings'),
        ]);
        if (mySeq !== _graphInitSeq) return;
        if (!statsResp.ok || !settingsResp.ok) throw new Error('Laden fehlgeschlagen');
        const stats = await statsResp.json();
        const settingsData = await settingsResp.json();
        const config = (settingsData && settingsData.config !== undefined) ? settingsData.config : settingsData;
        renderKiGraphTab(container, stats, config);

        // PollingManager nur nach erfolgreichem Init
        kiaraPolling.start('ki-graph:stats', () => refreshGraphStats(), 15000);
    } catch (e) {
        if (mySeq !== _graphInitSeq) return;
        container.innerHTML = '<div class="no-data">Graph-Statistiken konnten nicht geladen werden.</div>';
    }
}

async function refreshGraphStats() {
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-graph') return;
    const mySeq = _graphInitSeq;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/graph/stats');
        if (mySeq !== _graphInitSeq) return;
        if (!resp.ok) return;
        const stats = await resp.json();
        if (mySeq !== _graphInitSeq) return;
        updateGraphKpis(stats);
    } catch (e) { /* ignore */ }
}

function updateGraphKpis(stats) {
    const el = document.getElementById('graphKpiContainer');
    if (!el) return;
    el.innerHTML = buildGraphKpis(stats);
}

function buildGraphKpis(stats) {
    let html = '<div class="widget-row u-mb-20">';
    html += _graphKpi('Knoten gesamt', stats.total_nodes || 0);
    html += _graphKpi('Kanten gesamt', stats.total_edges || 0);
    html += _graphKpi('ERP-Import', stats.sources?.erp_import || 0);
    html += _graphKpi('Extraction', stats.sources?.extraction || 0);
    html += '</div>';

    // Nodes by type
    if (stats.nodes_by_type && Object.keys(stats.nodes_by_type).length > 0) {
        html += '<h4 class="u-my-16-8">Knoten nach Typ</h4>';
        html += '<div class="widget-row u-mb-16">';
        const typeLabels = {
            customer: 'Kunden', supplier: 'Lieferanten', article: 'Artikel',
            order: 'Auftraege', production_order: 'Fertigungsauftraege',
            employee: 'Mitarbeiter', document: 'Dokumente', topic: 'Themen', chunk: 'Chunks'
        };
        for (const [type, count] of Object.entries(stats.nodes_by_type)) {
            html += _graphKpi(typeLabels[type] || type, count);
        }
        html += '</div>';
    }

    // Edges by relation
    if (stats.edges_by_relation && Object.keys(stats.edges_by_relation).length > 0) {
        html += '<h4 class="u-my-16-8">Kanten nach Beziehung</h4>';
        html += '<div class="widget-row u-mb-16">';
        for (const [rel, count] of Object.entries(stats.edges_by_relation)) {
            html += _graphKpi(rel, count);
        }
        html += '</div>';
    }

    return html;
}

function _graphKpi(label, value) {
    return `<div class="widget-card u-mw-120-min"><div class="u-pad-16 u-text-center">
        <div class="u-fs-11 u-c-secondary u-uppercase u-letter-sp-05 u-mb-4">${SharedUi.runtime.escapeHtml(label)}</div>
        <div class="u-fs-20 u-fw-700">${SharedUi.runtime.escapeHtml(String(value))}</div>
    </div></div>`;
}

function renderKiGraphTab(container, stats, config) {
    const graphCfg = config?.graph || {};
    const enabled = graphCfg.enabled || false;

    let html = '<div class="config-form">';
    html += '<h3>Wissensgraph (GraphRAG)</h3>';

    // Activation toggle
    html += '<div class="config-section"><h4>Aktivierung</h4><div class="config-fields">';
    html += `<div class="config-field">
        <label><input type="checkbox" id="graphEnabled" ${enabled ? 'checked' : ''}> Wissensgraph aktivieren</label>
    </div>`;
    html += '</div></div>';

    // KPIs
    html += '<div id="graphKpiContainer">';
    html += buildGraphKpis(stats);
    html += '</div>';

    // Actions
    html += '<div class="config-section"><h4>Aktionen</h4>';
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" id="btnSyncErp" data-action="sync-erp-graph">ERP-Stammdaten importieren</button>';
    html += '<button class="action-btn danger" id="btnClearGraph" data-action="clear-graph">Graph leeren</button>';
    html += '<span id="graphActionStatus" class="u-ml-12 u-text-sm-secondary"></span>';
    html += '</div></div>';

    // Config options
    html += '<div class="config-section"><h4>Einstellungen</h4><div class="config-fields">';
    html += SharedUi.forms.configField({type: 'number', label: 'Max. Traversierungstiefe', key: 'graph.max_traversal_depth', id: 'graphMaxDepth', value: graphCfg.max_traversal_depth || 2, min: 1, max: 4});
    html += SharedUi.forms.configField({type: 'number', label: 'Max. Chunks pro Graph-Query', key: 'graph.max_chunks', id: 'graphMaxChunks', value: graphCfg.max_chunks || 20, min: 5, max: 50});
    html += `<div class="config-field">
        <label><input type="checkbox" id="graphExtractOnIngestion"
               ${graphCfg.extract_on_ingestion !== false ? 'checked' : ''}>
               Entity-Extraction bei Indexierung</label>
    </div>`;
    html += '</div></div>';

    // Save button
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" id="btnSaveGraph" data-action="save-graph-config">Speichern</button>';
    html += '<button class="action-btn" data-action="reset-ki-graph-tab">Zuruecksetzen</button>';
    html += '</div>';

    html += '<p class="u-text-xxs-secondary u-text-center u-mt-12">Auto-Refresh alle 15 Sekunden</p>';
    html += '</div>';

    container.innerHTML = html;
}

function saveGraphConfig(btn) {
    // Formularwerte VOR dem ersten await snapshotten
    const enabled = document.getElementById('graphEnabled')?.checked || false;
    const _maxDepthRaw = parseInt(document.getElementById('graphMaxDepth')?.value, 10);
    const _maxChunksRaw = parseInt(document.getElementById('graphMaxChunks')?.value, 10);
    const maxDepth = Number.isFinite(_maxDepthRaw) ? _maxDepthRaw : 2;
    const maxChunks = Number.isFinite(_maxChunksRaw) ? _maxChunksRaw : 20;
    const extractOnIngestion = document.getElementById('graphExtractOnIngestion')?.checked !== false;
    const mySeq = _graphInitSeq;

    SharedUi.runtime.withButtonLock(btn, async () => {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/settings');
        if (mySeq !== _graphInitSeq) return;
        if (!resp.ok) throw new Error('Einstellungen konnten nicht geladen werden');
        const data = await resp.json();
        const config = (data && data.config !== undefined) ? data.config : data;

        if (!config.graph) config.graph = {};
        config.graph.enabled = enabled;
        config.graph.max_traversal_depth = maxDepth;
        config.graph.max_chunks = maxChunks;
        config.graph.extract_on_ingestion = extractOnIngestion;

        await saveConfigToAPI({
            endpoint: '/api/admin/settings',
            data: { config },
            successMessage: 'Graph-Einstellungen gespeichert',
        });
    }).catch(e => {
        if (mySeq !== _graphInitSeq) return;
        SharedUi.notify.show('Fehler beim Speichern: ' + e.message, 'error');
    });
}

function syncErpGraph(btn) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        if (!await SharedUi.modal.confirm({ message: 'ERP-Stammdaten in den Wissensgraphen importieren?' })) return;
        const statusEl = document.getElementById('graphActionStatus');
        if (statusEl) statusEl.textContent = 'Import laeuft...';
        const resp = await SharedUi.fetch.safeFetch('/api/admin/graph/sync-erp', { method: 'POST' });
        if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || 'Fehler');
        const result = await resp.json();
        if (statusEl) statusEl.textContent = `${result.nodes_created || 0} Knoten, ${result.edges_created || 0} Kanten importiert`;
        SharedUi.notify.show('ERP-Import abgeschlossen', 'success');
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab === 'ki-graph') refreshGraphStats();
    }).catch(e => {
        const statusEl = document.getElementById('graphActionStatus');
        if (statusEl) statusEl.textContent = '';
        SharedUi.notify.show(e.message, 'error');
    });
}

function clearGraph(btn) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        if (!await SharedUi.modal.confirm({ message: 'Gesamten Wissensgraphen loeschen? Diese Aktion kann nicht rueckgaengig gemacht werden.' })) return;
        const statusEl = document.getElementById('graphActionStatus');
        const resp = await SharedUi.fetch.safeFetch('/api/admin/graph/clear?confirm=clear_graph', { method: 'POST' });
        if (!resp.ok) throw new Error('Fehler');
        const result = await resp.json();
        if (statusEl) statusEl.textContent = result.message || 'Graph geleert';
        SharedUi.notify.show('Graph geleert', 'success');
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab === 'ki-graph') refreshGraphStats();
    }).catch(e => SharedUi.notify.show(e.message, 'error'));
}
