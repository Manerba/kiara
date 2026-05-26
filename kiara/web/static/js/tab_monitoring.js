/**
 * System-Monitor Tab (Ersetzt Health & Monitoring)
 *
 * Live-Metriken via SSE (1/s), Chart.js mit date-fns Time-Achse,
 * historische Zeitreihen aus DB, Service-Management, journalctl.
 */

/* globals adminCurrentSubTab, Chart */

let _eventSource = null;
let _cpuRamChart = null;
let _netLoadChart = null;
let _currentRange = 'live';
let _rangeSwitching = false;     // Guard gegen parallele _onRangeChange-Aufrufe
let _monInitSeq = 0;
let _monRestartTimer = null;     // 3s-Restart-Refresh-Timer
const _restartingServices = new Set();

const _MON_ACTIONS = new Set(['range-change']);

function _handleMonClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_MON_ACTIONS.has(el.dataset.action)) return;
    if (el.dataset.action === 'range-change') {
        _onRangeChange(el.dataset.range || 'live');
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
    _initCharts();
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
    if (_cpuRamChart) { _cpuRamChart.destroy(); _cpuRamChart = null; }
    if (_netLoadChart) { _netLoadChart.destroy(); _netLoadChart = null; }
    if (_monRestartTimer) { clearTimeout(_monRestartTimer); _monRestartTimer = null; }
    _currentRange = 'live';
    _rangeSwitching = false;
    _restartingServices.clear();
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
        _updateKpiCards(data);
        _updateTopProcesses(data.top_processes || []);
        _updateDbSizes(data.db_sizes || {});
        _updateSystemInfo(data);

        // Charts nur im Live-Modus per SSE befuellen
        if (_currentRange === 'live') {
            _appendToLiveChart(data);
        }
    });

    // addEventListener statt Property-Binding (CSP-sauberer Gate-Anker;
    // EventSource reconnected automatisch nach ~3s — Handler bleibt als
    // Stub fuer spaeteres Logging).
    _eventSource.addEventListener('error', () => {});
}


// ──────────────────── Zeitraum-Wechsel ────────────────────

async function _onRangeChange(range) {
    if (_rangeSwitching) return;
    _rangeSwitching = true;
    const mySeq = _monInitSeq;

    try {
        _currentRange = range;

        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.range === range);
        });

        if (range === 'live') {
            _clearCharts();
        } else {
            const [cpuData, ramData, loadData, netData] = await Promise.all([
                _fetchHistory(range, 'cpu'),
                _fetchHistory(range, 'ram'),
                _fetchHistory(range, 'load'),
                _fetchHistory(range, 'net'),
            ]);
            if (mySeq !== _monInitSeq) return;
            _replaceChartData(cpuData, ramData, loadData, netData);
        }
    } catch(e) {
        if (mySeq !== _monInitSeq) return;
        console.error('Range-Wechsel Fehler:', e);
        // Fallback: zurueck auf Live-Modus
        _currentRange = 'live';
        document.querySelectorAll('.range-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.range === 'live');
        });
        _clearCharts();
    } finally {
        if (mySeq === _monInitSeq) _rangeSwitching = false;
    }
}

async function _fetchHistory(range, metric) {
    const resp = await SharedUi.fetch.safeFetch(
        `/api/monitoring/history?range=${range}&metric=${metric}`,
        { headers: { 'X-Requested-With': 'XMLHttpRequest' } }
    );
    if (!resp.ok) throw new Error(`History ${metric}: HTTP ${resp.status}`);
    return await resp.json();
}


// ──────────────────── Charts ────────────────────

function _initCharts() {
    // --- CPU & RAM Chart ---
    const ctx1 = document.getElementById('chart-cpu-ram')?.getContext('2d');
    if (!ctx1) return;
    _cpuRamChart = new Chart(ctx1, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: '#3b82f6',
                    yAxisID: 'y-percent',
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: 'RAM (GB)',
                    data: [],
                    borderColor: '#10b981',
                    yAxisID: 'y-bytes',
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 1.5,
                    fill: true,
                    backgroundColor: 'rgba(16, 185, 129, 0.08)',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                'y-percent': {
                    type: 'linear', position: 'left',
                    min: 0, max: 100,
                    title: { display: true, text: '%' },
                },
                'y-bytes': {
                    type: 'linear', position: 'right',
                    min: 0,
                    title: { display: true, text: 'GB' },
                    ticks: { callback: v => (v / 1073741824).toFixed(1) },
                    grid: { drawOnChartArea: false },
                },
                x: {
                    type: 'time',
                    time: {
                        tooltipFormat: 'dd.MM.yyyy HH:mm:ss',
                        displayFormats: {
                            second: 'HH:mm:ss',
                            minute: 'HH:mm',
                            hour: 'HH:mm',
                            day: 'dd.MM.',
                            month: 'MMM yyyy',
                        },
                    },
                    ticks: { maxTicksLimit: 10 },
                },
            },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.dataset.yAxisID === 'y-bytes')
                                return `${ctx.dataset.label}: ${(ctx.raw.y / 1073741824).toFixed(2)} GB`;
                            return `${ctx.dataset.label}: ${ctx.raw.y.toFixed(1)}%`;
                        },
                    },
                },
            },
        },
    });

    // --- Netzwerk & Load Chart ---
    const ctx2 = document.getElementById('chart-net-load')?.getContext('2d');
    if (!ctx2) return;
    _netLoadChart = new Chart(ctx2, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Net \u2191 (KB/s)',
                    data: [],
                    borderColor: '#f59e0b',
                    yAxisID: 'y-net',
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: 'Net \u2193 (KB/s)',
                    data: [],
                    borderColor: '#ef4444',
                    yAxisID: 'y-net',
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: 'Load 1m',
                    data: [],
                    borderColor: '#8b5cf6',
                    yAxisID: 'y-load',
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 1.5,
                    borderDash: [4, 2],
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                'y-net': {
                    type: 'linear', position: 'left',
                    min: 0,
                    title: { display: true, text: 'KB/s' },
                    ticks: { callback: v => (v / 1024).toFixed(0) },
                },
                'y-load': {
                    type: 'linear', position: 'right',
                    min: 0,
                    title: { display: true, text: 'Load' },
                    grid: { drawOnChartArea: false },
                },
                x: {
                    type: 'time',
                    time: {
                        tooltipFormat: 'dd.MM.yyyy HH:mm:ss',
                        displayFormats: {
                            second: 'HH:mm:ss',
                            minute: 'HH:mm',
                            hour: 'HH:mm',
                            day: 'dd.MM.',
                            month: 'MMM yyyy',
                        },
                    },
                    ticks: { maxTicksLimit: 10 },
                },
            },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true } },
            },
        },
    });
}

const LIVE_MAX_POINTS = 300;   // 5 Minuten bei 1s Takt

function _appendToLiveChart(data) {
    if (!_cpuRamChart || !_netLoadChart) return;
    const ts = data.ts ? new Date(data.ts) : null;
    if (!ts || Number.isNaN(ts.getTime())) return;

    // CPU/RAM Chart
    _cpuRamChart.data.datasets[0].data.push({ x: ts, y: data.cpu_percent });
    _cpuRamChart.data.datasets[1].data.push({ x: ts, y: data.ram_used_bytes });
    _cpuRamChart.data.datasets.forEach(ds => {
        if (ds.data.length > LIVE_MAX_POINTS) ds.data.shift();
    });
    _cpuRamChart.update('none');

    // Net/Load Chart
    _netLoadChart.data.datasets[0].data.push({ x: ts, y: data.net_sent_bps });
    _netLoadChart.data.datasets[1].data.push({ x: ts, y: data.net_recv_bps });
    _netLoadChart.data.datasets[2].data.push({ x: ts, y: data.load_1m });
    _netLoadChart.data.datasets.forEach(ds => {
        if (ds.data.length > LIVE_MAX_POINTS) ds.data.shift();
    });
    _netLoadChart.update('none');
}

function _replaceChartData(cpuData, ramData, loadData, netData) {
    // CPU/RAM
    _cpuRamChart.data.datasets[0].data = cpuData.timestamps.map((ts, i) => ({
        x: new Date(ts), y: cpuData.series.cpu_percent[i],
    }));
    _cpuRamChart.data.datasets[1].data = ramData.timestamps.map((ts, i) => ({
        x: new Date(ts), y: ramData.series.ram_used_bytes[i],
    }));
    _cpuRamChart.update('none');

    // Net/Load
    _netLoadChart.data.datasets[0].data = netData.timestamps.map((ts, i) => ({
        x: new Date(ts), y: netData.series.net_sent_bps[i],
    }));
    _netLoadChart.data.datasets[1].data = netData.timestamps.map((ts, i) => ({
        x: new Date(ts), y: netData.series.net_recv_bps[i],
    }));
    _netLoadChart.data.datasets[2].data = loadData.timestamps.map((ts, i) => ({
        x: new Date(ts), y: loadData.series.load_1m[i],
    }));
    _netLoadChart.update('none');
}

function _clearCharts() {
    [_cpuRamChart, _netLoadChart].forEach(chart => {
        if (!chart) return;
        chart.data.datasets.forEach(ds => { ds.data = []; });
        chart.update('none');
    });
}


// ──────────────────── Layout ────────────────────

function _renderLayout(container) {
    container.innerHTML = `
        <div class="monitor-range-bar">
            <button class="range-btn active" data-action="range-change" data-range="live">Live</button>
            <button class="range-btn" data-action="range-change" data-range="1h">1h</button>
            <button class="range-btn" data-action="range-change" data-range="24h">24h</button>
            <button class="range-btn" data-action="range-change" data-range="30d">30d</button>
            <button class="range-btn" data-action="range-change" data-range="1y">1y</button>
        </div>

        <div class="kpi-grid" id="kpi-grid">
            <div class="kpi-card" id="kpi-cpu"><span class="kpi-label">CPU</span><span class="kpi-value">&mdash;</span></div>
            <div class="kpi-card" id="kpi-ram"><span class="kpi-label">RAM</span><span class="kpi-value">&mdash;</span></div>
            <div class="kpi-card" id="kpi-swap"><span class="kpi-label">Swap</span><span class="kpi-value">&mdash;</span></div>
            <div class="kpi-card" id="kpi-load"><span class="kpi-label">Load</span><span class="kpi-value">&mdash;</span></div>
            <div class="kpi-card" id="kpi-disk"><span class="kpi-label">Disk</span><span class="kpi-value">&mdash;</span></div>
            <div class="kpi-card" id="kpi-net"><span class="kpi-label">Netzwerk</span><span class="kpi-value">&mdash;</span></div>
            <div class="kpi-card" id="kpi-uptime"><span class="kpi-label">Uptime</span><span class="kpi-value">&mdash;</span></div>
        </div>

        <div class="monitor-charts">
            <div class="monitor-chart-card"><canvas id="chart-cpu-ram"></canvas></div>
            <div class="monitor-chart-card"><canvas id="chart-net-load"></canvas></div>
        </div>

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
        <div class="service-grid" id="service-grid"></div>

        <h4>Letzte Fehler (journalctl)</h4>
        <table class="error-log-table" id="error-log">
            <thead><tr><th>Zeitstempel</th><th>Dienst</th><th>Nachricht</th></tr></thead>
            <tbody></tbody>
        </table>
    `;
}


// ──────────────────── KPI-Updates ────────────────────

function _updateKpiCards(data) {
    const v = (id, html) => {
        const el = document.querySelector(`#${id} .kpi-value`);
        if (el) el.innerHTML = html;
    };
    v('kpi-cpu', (data.cpu_percent ?? 0).toFixed(1) + '%');
    v('kpi-ram', _formatBytes(data.ram_used_bytes) + ' / ' + _formatBytes(data.ram_total_bytes));
    v('kpi-swap', _formatBytes(data.swap_used_bytes) + ' / ' + _formatBytes(data.swap_total_bytes));
    v('kpi-load', `${data.load_1m ?? '\u2014'} / ${data.load_5m ?? '\u2014'} / ${data.load_15m ?? '\u2014'}`);
    v('kpi-disk', _formatBytes(data.disk_used_bytes) + ' / ' + _formatBytes(data.disk_total_bytes));
    v('kpi-net', '\u2191 ' + _formatRate(data.net_sent_bps ?? 0) + ' \u2193 ' + _formatRate(data.net_recv_bps ?? 0));
    if (data.uptime_seconds != null) v('kpi-uptime', _formatUptime(data.uptime_seconds));
}

function _updateTopProcesses(procs) {
    const tbody = document.querySelector('#top-procs tbody');
    if (!tbody) return;
    tbody.innerHTML = procs.map(p =>
        `<tr><td>${SharedUi.runtime.escapeHtml(p.name)}</td><td>${_formatBytes(p.rss_bytes)}</td><td class="text-muted">${p.pid}</td></tr>`
    ).join('');
}

function _updateDbSizes(sizes) {
    const el = document.getElementById('db-sizes');
    if (!el) return;
    el.innerHTML = `
        PostgreSQL: ${_formatBytes(sizes.postgresql_bytes || 0)}<br>
        Qdrant: ${_formatBytes(sizes.qdrant_bytes || 0)}
    `;
}

function _updateSystemInfo(data) {
    const el = document.getElementById('system-info');
    if (!el || !data.os_version) return;
    el.innerHTML = `OS: ${SharedUi.runtime.escapeHtml(data.os_version)}<br>Kernel: ${SharedUi.runtime.escapeHtml(data.kernel || '\u2014')}`;
}


// ──────────────────── Services ────────────────────

async function _fetchServices() {
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/monitoring/services', {
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-monitoring') return;
        _renderServiceCards(data.data || data);
    } catch (err) {
        console.warn('Services laden fehlgeschlagen:', err);
    }
}

function _renderServiceCards(services) {
    const grid = document.getElementById('service-grid');
    if (!grid) return;
    const items = Array.isArray(services) ? services : (services.services || []);
    grid.innerHTML = items.map(svc => {
        const statusClass = svc.status === 'active' ? 'active'
            : svc.status === 'failed' ? 'failed' : 'inactive';
        const displayName = svc.display_name || svc.name;
        return `
            <div class="service-card" data-service="${SharedUi.runtime.escapeAttr(svc.name)}" data-display="${SharedUi.runtime.escapeAttr(displayName)}">
                <div><span class="status-dot ${statusClass}"></span> <strong>${SharedUi.runtime.escapeHtml(displayName)}</strong></div>
                <div class="text-muted">${svc.status ? SharedUi.runtime.escapeHtml(svc.status) : '\u2014'}</div>
                ${svc.uptime ? `<div class="text-muted">${SharedUi.runtime.escapeHtml(svc.uptime)}</div>` : ''}
                ${svc.memory ? `<div class="text-muted">${_formatBytes(svc.memory)}</div>` : ''}
                <button class="restart-btn">Neustart</button>
            </div>
        `;
    }).join('');
    // Event-Delegation statt inline onclick (XSS-sicher)
    grid.querySelectorAll('.restart-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const card = btn.closest('.service-card');
            _restartService(card.dataset.service, card.dataset.display);
        });
    });
}

async function _restartService(serviceName, displayName) {
    // Service-keyed Lock: VOR await setzen, sonst koennen Confirm-Modale stapeln
    if (_restartingServices.has(serviceName)) return;
    _restartingServices.add(serviceName);

    const confirmed = await SharedUi.modal.confirm({ message:
        `Dienst "${displayName}" (${serviceName}) wirklich neu starten?`
    });
    if (!confirmed) {
        _restartingServices.delete(serviceName);
        return;
    }

    const mySeq = _monInitSeq;
    const btn = document.querySelector(`[data-service="${CSS.escape(String(serviceName))}"] .restart-btn`);
    if (btn) { btn.disabled = true; btn.textContent = 'Wird neu gestartet...'; }

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/monitoring/services/${serviceName}/restart`, {
            method: 'POST',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        const data = await resp.json();

        if (data.success) {
            if (mySeq === _monInitSeq) {
                _monRestartTimer = setTimeout(() => {
                    _monRestartTimer = null;
                    if (_monInitSeq === mySeq) _fetchServices();
                }, 3000);
            }
        } else {
            await SharedUi.modal.alert({ message: 'Neustart fehlgeschlagen: ' + data.message });
        }
    } catch (err) {
        await SharedUi.modal.alert({ message: 'Fehler: ' + err.message });
    } finally {
        _restartingServices.delete(serviceName);
        if (btn) { btn.disabled = false; btn.textContent = 'Neustart'; }
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

function _formatUptime(seconds) {
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
}

function _formatRate(bytesPerSec) {
    if (bytesPerSec < 1024) return bytesPerSec.toFixed(0) + ' B/s';
    if (bytesPerSec < 1048576) return (bytesPerSec / 1024).toFixed(1) + ' KB/s';
    return (bytesPerSec / 1048576).toFixed(1) + ' MB/s';
}
