/**
 * KIara Admin - System > Performance Sub-Tab
 */

let _perfInitSeq = 0;

const _PERF_ACTIONS = new Set(['clear-query-cache']);

function _handlePerfClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_PERF_ACTIONS.has(el.dataset.action)) return;
    if (el.dataset.action === 'clear-query-cache') clearQueryCache(el);
}

function _bindPerfActions() {
    if (document.body._perfActionsBound) return;
    document.body._perfActionsBound = true;
    document.body.addEventListener('click', _handlePerfClick);
}

async function initPerformanceTab() {
    _bindPerfActions();
    // Re-Init-Schutz
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-performance');
    _perfInitSeq++;
    const mySeq = _perfInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    // Cleanup registrieren
    registerTabCleanup('system-performance', () => { _perfInitSeq++; kiaraPolling.stop('system-performance:stats'); });

    await loadPerformanceData(container, mySeq);

    // PollingManager fuer Auto-Refresh
    kiaraPolling.start('system-performance:stats', () => loadPerformanceData(container, _perfInitSeq), 10000);
}

async function loadPerformanceData(container, expectedSeq) {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/cache/stats');
        if (!resp.ok) throw new Error('Laden fehlgeschlagen');
        const data = await resp.json();
        if (expectedSeq !== _perfInitSeq) return;
        renderPerformanceTab(container, data);
    } catch (e) {
        if (expectedSeq !== _perfInitSeq) return;
        container.innerHTML = '<div class="no-data">Cache-Statistiken konnten nicht geladen werden.</div>';
    }
}

function renderPerformanceTab(container, data) {
    let html = '<div class="config-form">';
    html += '<h3>Performance & Cache</h3>';

    // KPI grid
    html += '<div class="widget-row u-mb-20">';
    html += _perfKpi('Cache aktiv', data.enabled ? 'Ja' : 'Nein');
    html += _perfKpi('Eintraege', `${data.size || 0} / ${data.max_size || 0}`);
    html += _perfKpi('Hit-Rate', `${((data.hit_rate || 0) * 100).toFixed(1)}%`);
    html += _perfKpi('Hits', data.hits || 0);
    html += _perfKpi('Misses', data.misses || 0);
    html += _perfKpi('TTL', `${data.ttl_seconds || 0}s`);
    html += '</div>';

    // Clear button
    html += '<div class="config-section">';
    html += '<div class="config-actions">';
    html += '<button class="action-btn danger" id="btnClearCache" data-action="clear-query-cache">Cache leeren</button>';
    html += '</div></div>';

    html += '<p class="u-text-xxs-secondary u-text-center">Auto-Refresh alle 10 Sekunden</p>';
    html += '</div>';
    container.innerHTML = html;
}

function _perfKpi(label, value) {
    return `<div class="widget-card perf-kpi-card"><div class="perf-kpi-inner">
        <div class="perf-kpi-label u-xs-secondary-mb-4 u-uppercase u-letter-sp-05">${SharedUi.runtime.escapeHtml(label)}</div>
        <div class="perf-kpi-value">${SharedUi.runtime.escapeHtml(String(value))}</div>
    </div></div>`;
}

function clearQueryCache(btn) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        if (!await SharedUi.modal.confirm({ message: 'Query-Cache wirklich leeren?' })) return;
        const resp = await SharedUi.fetch.safeFetch('/api/admin/cache/clear', { method: 'POST' });
        if (!resp.ok) throw new Error('Fehler');
        SharedUi.notify.show('Cache geleert', 'success');
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab === 'system-performance') {
            const container = document.getElementById('contentContainer');
            if (container) await loadPerformanceData(container, _perfInitSeq);
        }
    }).catch(e => SharedUi.notify.show(e.message, 'error'));
}
