/**
 * System-Monitor Tab (Ersetzt Health & Monitoring)
 *
 * Systemdetails via SSE (1/s), Kiron-Verlaufsdiagramme mit 250-ms-RAM-Puffer,
 * historische Zeitreihen aus DB, Service-Management, journalctl.
 */

/* globals adminCurrentSubTab, KiaraMonitoringHistory */

let _eventSource = null;
let _monInitSeq = 0;
let _monServiceTimer = null;
let _monServicesTable = null;
let _monServicesRequestSeq = 0;
const _pendingServiceActions = new Map();
const _MON_SERVICE_TRANSITIONS = new Set(['activating', 'deactivating', 'reloading', 'refreshing']);

const _MON_ACTIONS = new Set(['range-change', 'monitor-service-action']);

function _handleMonClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_MON_ACTIONS.has(el.dataset.action)) return;
    if (el.dataset.action === 'range-change') {
        KiaraMonitoringHistory.setRange(el.dataset.range || '1m');
    } else if (!el.disabled) {
        _changeMonService(el.dataset.service, el.dataset.serviceAction);
    }
}

function _bindMonActions() {
    if (document.body._monActionsBound) return;
    document.body._monActionsBound = true;
    document.body.addEventListener('click', _handleMonClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-mon-actions-bound', 'true');
}

// ──────────────────── Init ────────────────────

async function initMonitoringTab() {
    _bindMonActions();

    // Re-Init-Schutz
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-monitoring');
    _monInitSeq++;
    const mySeq = _monInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    _cleanup();

    // Cleanup frueh registrieren — seq-Invalidation gehoert hierher, nicht in shared _cleanup()
    registerTabCleanup('system-monitoring', () => {
        _monInitSeq++;
        _cleanup();
        kiaraPolling.stop('system-monitoring:services');
    });

    _renderLayout(container);
    _initServicesTable();
    KiaraMonitoringHistory.init();
    _connectSSE();

    // Services + Journal initial laden, dann PollingManager
    // allSettled: ein Endpoint-Fehler darf den anderen nicht abbrechen
    await Promise.allSettled([_fetchServices(), _fetchJournalErrors()]);
    if (mySeq !== _monInitSeq) return;

    kiaraPolling.start('system-monitoring:services', () => {
        _fetchServices();
        _fetchJournalErrors();
    }, 60000);
}

function _cleanup() {
    if (_eventSource) { _eventSource.close(); _eventSource = null; }
    KiaraMonitoringHistory.destroy();
    if (_monServiceTimer) { clearTimeout(_monServiceTimer); _monServiceTimer = null; }
    _pendingServiceActions.forEach(controller => controller.abort());
    _pendingServiceActions.clear();
    if (_monServicesTable) { _monServicesTable.destroy(); _monServicesTable = null; }
}


// ──────────────────── SSE ────────────────────

function _connectSSE() {
    _eventSource = new EventSource('/api/monitoring/stream');

    _eventSource.addEventListener('metrics', (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            console.warn('[monitoring] korruptes SSE-metrics-Event verworfen', e);
            return;
        }
        _updateTopProcesses(data.top_processes || []);
        _updateDbSizes(data.db_sizes || {});
        _updateSystemInfo(data);
    });

    // addEventListener statt Property-Binding (CSP-sauberer Gate-Anker;
    // EventSource reconnected automatisch nach ~3s — Handler bleibt als
    // Stub fuer spaeteres Logging).
    _eventSource.addEventListener('error', () => {});
}


// ──────────────────── Layout ────────────────────

function _renderLayout(container) {
    container.innerHTML = `
        ${KiaraMonitoringHistory.rangeHtml()}

        ${KiaraMonitoringHistory.chartsHtml()}

        <div class="monitor-details">
            <div class="detail-card">
                <h4>Speicher &amp; System</h4>
                <div id="db-sizes">&mdash;</div>
                <div id="system-info">&mdash;</div>
            </div>
            <div class="detail-card">
                <h4>Top-5 Prozesse (RAM)</h4>
                <table class="top-procs-table" id="top-procs">
                    <tbody></tbody>
                </table>
            </div>
        </div>

        <h4>Dienste</h4>
        <div class="monitor-services-table table-container" id="monitor-services-table"></div>

        <h4>Letzte Fehler (journalctl)</h4>
        <table class="error-log-table" id="error-log">
            <thead><tr><th>Zeitstempel</th><th>Dienst</th><th>Nachricht</th></tr></thead>
            <tbody></tbody>
        </table>
    `;
}


// ──────────────────── Systemdetails ────────────────────

function _updateTopProcesses(procs) {
    const tbody = document.querySelector('#top-procs tbody');
    if (!tbody) return;
    tbody.innerHTML = procs.map(p =>
        `<tr><td>${SharedUi.runtime.escapeHtml(p.name)}</td><td>${_formatBytes(p.rss_bytes)}</td><td class="text-muted">${p.pid}</td></tr>`
    ).join('');
}

function _updateDbSizes(sizes) {
    KiaraStorage.update(document.getElementById('db-sizes'), sizes);
}

function _updateSystemInfo(data) {
    const el = document.getElementById('system-info');
    if (!el || !data.os_version) return;
    el.innerHTML = `OS: ${SharedUi.runtime.escapeHtml(data.os_version)}<br>Kernel: ${SharedUi.runtime.escapeHtml(data.kernel || '\u2014')}`;
}


// ──────────────────── Services ────────────────────

function _initServicesTable() {
    _monServicesTable = new SharedUi.table.ExpandableTable({
        id: 'monitoring-services',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px minmax(210px, 1fr) 150px 180px 170px',
        columns: [
            { field: 'name', label: 'Name', sortable: true, align: 'left' },
            { field: 'status', label: 'Status', sortable: true, align: 'left', renderer: _renderServiceStatus },
            { field: 'active_since', label: 'Aktiv seit', sortable: true, align: 'left', renderer: _renderServiceSince },
            { field: 'actions', label: 'Aktionen', align: 'left', renderer: _renderServiceActions },
        ],
        detailRenderer: row => `<p class="monitor-service-description">${SharedUi.runtime.escapeHtml(row.description || '')}</p>`,
    }, 'monitor-services-table');
    _monServicesTable.showLoading();
}

async function _fetchServices() {
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
    const table = _monServicesTable;
    if (!table) return;
    const requestSeq = ++_monServicesRequestSeq;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/monitoring/services', {
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (table !== _monServicesTable || requestSeq !== _monServicesRequestSeq) return;
        const services = data.data.services;
        table.setData(services.map(service => ({ ...service, id: service.name })), services.length);
        if (services.some(service => service.job_pending || _MON_SERVICE_TRANSITIONS.has(service.status))) {
            _scheduleServiceRefresh();
        }
    } catch (err) {
        if (table !== _monServicesTable || requestSeq !== _monServicesRequestSeq) return;
        table.showError('Dienste konnten nicht geladen werden. Die Abfrage wird automatisch wiederholt.');
        console.warn('Services laden fehlgeschlagen:', err);
    }
}

function _renderServiceStatus(value, row) {
    const labels = {
        active: 'Aktiv', inactive: 'Inaktiv', failed: 'Fehlgeschlagen',
        activating: 'Startet …', deactivating: 'Stoppt …', reloading: 'Lädt neu …',
        refreshing: 'Lädt neu …', maintenance: 'Wartung', unknown: 'Unbekannt',
    };
    const statusClass = value === 'active' ? 'active' : value === 'failed' ? 'failed' : 'inactive';
    const label = row.load_state === 'not-found' ? 'Nicht installiert'
        : row.load_state === 'masked' ? 'Gesperrt'
        : row.job_pending && !_MON_SERVICE_TRANSITIONS.has(value) ? 'Aktion läuft …'
        : (labels[value] || 'Unbekannt');
    return `<span class="status-dot ${statusClass}" aria-hidden="true"></span> ${label}`;
}

function _renderServiceSince(value, row) {
    if (row.status !== 'active' || !value) return '&mdash;';
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return '&mdash;';
    return `<time datetime="${SharedUi.runtime.escapeAttr(value)}">${date.toLocaleString('de-DE')}</time>`;
}

function _serviceActionsDisabled(row) {
    return _pendingServiceActions.has(row.name) || row.job_pending
        || row.load_state !== 'loaded' || !['active', 'inactive', 'failed'].includes(row.status);
}

function _renderServiceActions(value, row) {
    const service = SharedUi.runtime.escapeAttr(row.name);
    const action = row.status === 'active' || row.status === 'deactivating' ? 'stop' : 'start';
    const disabled = _serviceActionsDisabled(row) ? 'disabled' : '';
    return `<span class="action-buttons">
        <button type="button" class="action-btn-sm ${action === 'stop' ? 'danger' : ''}"
            data-action="monitor-service-action" data-service="${service}" data-service-action="${action}"
            aria-label="${service} ${action === 'stop' ? 'stoppen' : 'starten'}" ${disabled}>${action === 'stop' ? 'Stopp' : 'Start'}</button>
        <button type="button" class="action-btn-sm" data-action="monitor-service-action" data-service="${service}"
            data-service-action="restart" aria-label="${service} neu starten" ${disabled}>Neustart</button>
    </span>`;
}

function _scheduleServiceRefresh() {
    if (_monServiceTimer) clearTimeout(_monServiceTimer);
    const mySeq = _monInitSeq;
    _monServiceTimer = setTimeout(() => {
        _monServiceTimer = null;
        if (mySeq === _monInitSeq) _fetchServices();
    }, 2000);
}

async function _changeMonService(serviceName, action) {
    const table = _monServicesTable;
    const row = table?.data.find(service => service.name === serviceName);
    if (!row || _serviceActionsDisabled(row) || !['start', 'stop', 'restart'].includes(action)) return;

    // Vor safeFetch sperren: dessen serverseitiger Bestaetigungsdialog darf sich
    // auch bei Doppelklick oder Tabellen-Refresh nur einmal oeffnen.
    const controller = new AbortController();
    _pendingServiceActions.set(serviceName, controller);
    table.silentRender();

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/monitoring/services/${encodeURIComponent(serviceName)}/${action}`, {
            method: 'POST',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
            signal: controller.signal,
        });
        const data = await resp.json();
        if (table !== _monServicesTable) return;
        if (!resp.ok || !data.success) {
            throw new Error(data.message || `HTTP ${resp.status}`);
        }
        await _fetchServices();
        if (table === _monServicesTable) _scheduleServiceRefresh();
    } catch (err) {
        if (table !== _monServicesTable || err.name === 'AbortError' || err.kind === 'confirmation_cancelled') return;
        await SharedUi.modal.alert({ message: 'Dienstaktion fehlgeschlagen: ' + err.message });
        if (table === _monServicesTable) await _fetchServices();
    } finally {
        if (_pendingServiceActions.get(serviceName) === controller) {
            _pendingServiceActions.delete(serviceName);
            if (table === _monServicesTable && document.querySelector('#monitor-services-table .table-row-wrapper')) {
                table.silentRender();
            }
        }
    }
}


// ──────────────────── Journal ────────────────────

async function _fetchJournalErrors() {
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/monitoring/journal', {
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
        // Admin-Function: {success, message, data: {errors: [...]}}
        const errors = (data.data && data.data.errors) || data.errors || [];
        _renderErrorLog(errors);
    } catch (err) {
        console.warn('Journal laden fehlgeschlagen:', err);
    }
}

function _renderErrorLog(errors) {
    const tbody = document.querySelector('#error-log tbody');
    if (!tbody) return;
    if (errors.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="text-muted">Keine Fehler</td></tr>';
        return;
    }
    // journalctl short-iso Format: "2026-03-29T10:12:05+0100 hostname unit[pid]: message"
    tbody.innerHTML = errors.map(line => {
        const parts = line.match(/^(\S+)\s+\S+\s+(\S+?)(?:\[\d+\])?:\s*(.*)$/);
        if (!parts) return `<tr><td colspan="3">${SharedUi.runtime.escapeHtml(line)}</td></tr>`;
        return `<tr><td>${SharedUi.runtime.escapeHtml(parts[1])}</td><td>${SharedUi.runtime.escapeHtml(parts[2])}</td><td>${SharedUi.runtime.escapeHtml(parts[3])}</td></tr>`;
    }).join('');
}


// ──────────────────── Hilfsfunktionen ────────────────────

function _formatBytes(bytes) {
    if (bytes === 0 || bytes == null) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
}
