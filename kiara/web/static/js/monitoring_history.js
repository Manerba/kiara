/* globals Chart, TimeSeriesChart, SharedUi, kiaraPolling */
const KiaraMonitoringHistory = (() => {
/**
 * Verlauf-Tab - Historische Metriken mit Zeitreihen-Charts.
 * In KIara: CPU, RAM, Disk und Netzwerk mit gemeinsamem Live-Puffer.
 */

// State
let historyCharts = [];
let historyRequest = null;
const historyCache = new Map();
let historyActiveRange = '1m';
let historyLoadGeneration = 0;

// Eigenstaendiger Livepfad mit serverseitigem RAM-Puffer (ohne PostgreSQL-Zugriff).
let historyLiveSocket = null;
let historyLiveRenderFrame = null;
let historyLiveGeneration = 0;
let historyLiveSamples = [];

const HISTORY_LIVE_PROTOCOL_VERSION = 1;

const HISTORY_RANGES = {
    '1m':  { seconds: 60,       label: '1m',  refreshMs: 0, live: true, maxPoints: 256 },
    '10m': { seconds: 600,      label: '10m', refreshMs: 0, live: true, maxPoints: 2416 },
    '1h':  { seconds: 3600,     label: '1h',  refreshMs: 10000 },
    '6h':  { seconds: 21600,    label: '6h',  refreshMs: 60000 },
    '24h': { seconds: 86400,    label: '24h', refreshMs: 60000 },
    '7d':  { seconds: 604800,   label: '7d',  refreshMs: 0 },
    '30d': { seconds: 2592000,  label: '30d', refreshMs: 0 },
    '90d': { seconds: 7776000,  label: '90d', refreshMs: 0 },
    '1y':  { seconds: 31536000, label: '1y',  refreshMs: 0 },
};

const CHART_CONFIGS = [
    {
        id: 'chart-cpu',
        title: 'CPU & Load',
        series: [
            { key: 'cpu_usage', label: 'CPU %', color: '#3498db', yAxisID: 'y' },
            { key: 'cpu_load_1m', label: 'Load 1m', color: '#e67e22', yAxisID: 'y2' },
        ],
        yAxes: [
            { id: 'y', label: '%', position: 'left', min: 0, max: 100 },
            { id: 'y2', label: 'Load', position: 'right', min: 0 },
        ],
    },
    {
        id: 'chart-ram',
        title: 'RAM',
        series: [
            { key: 'memory_usage', label: 'RAM %', color: '#2ecc71', yAxisID: 'y' },
            { key: 'memory_used_gb', label: 'Belegt (GiB)', color: '#27ae60', yAxisID: 'y2' },
        ],
        yAxes: [
            { id: 'y', label: '%', position: 'left', min: 0, max: 100 },
            { id: 'y2', label: 'GiB', position: 'right', min: 0 },
        ],
    },
    {
        id: 'chart-disk',
        title: 'Disk I/O',
        series: [
            { key: 'disk_read_mb_s', label: 'Lesen MiB/s', color: '#1abc9c', yAxisID: 'y' },
            { key: 'disk_write_mb_s', label: 'Schreiben MiB/s', color: '#e74c3c', yAxisID: 'y' },
        ],
        yAxes: [
            { id: 'y', label: 'MiB/s', position: 'left', min: 0 },
        ],
    },
    {
        id: 'chart-network',
        title: 'Netzwerk',
        series: [
            { key: 'net_sent_mb_s', label: 'Senden ↑ MiB/s', color: '#f59e0b', yAxisID: 'y' },
            { key: 'net_recv_mb_s', label: 'Empfangen ↓ MiB/s', color: '#8b5cf6', yAxisID: 'y' },
        ],
        yAxes: [{ id: 'y', label: 'MiB/s', position: 'left', min: 0 }],
    },
];

const HISTORY_LIVE_METRIC_KEYS = [
    ...new Set(CHART_CONFIGS.flatMap(config => config.series.map(series => series.key))),
];

/**
 * Tab initialisieren (wird von app_core.js aufgerufen).
 */
function rangeHtml() {
    return '<div class="monitor-range-bar"><span class="range-label">Zeitraum:</span>'
        + Object.entries(HISTORY_RANGES).map(([key, config]) =>
            `<button class="range-btn${key === historyActiveRange ? ' active' : ''}" data-action="range-change" data-range="${key}" aria-pressed="${key === historyActiveRange}">${config.label}</button>`
        ).join('') + '</div>';
}

function chartsHtml() {
    return '<div class="monitor-charts">'
        + CHART_CONFIGS.map(config => `<div class="monitor-chart-card"><h4>${config.title}</h4><div class="monitor-chart-canvas"><canvas id="${config.id}" aria-label="${config.title}"></canvas></div></div>`).join('')
        + '</div>';
}

function initHistoryTab() {
    destroyHistoryCharts();
    historyCharts = CHART_CONFIGS.map(config => new TimeSeriesChart(config.id, config));
    document.addEventListener('themeChanged', refreshHistoryTheme);
    setHistoryRange(historyActiveRange);
}

/**
 * Zeitraum wechseln.
 */
function setHistoryRange(range) {
    if (!HISTORY_RANGES[range]) return;
    historyLoadGeneration += 1;
    stopHistoryRefresh();
    stopHistoryLive();
    historyActiveRange = range;

    // Button-Styling
    document.querySelectorAll('.monitor-range-bar .range-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.range === range);
        btn.setAttribute('aria-pressed', String(btn.dataset.range === range));
    });

    if (HISTORY_RANGES[range].live) {
        startHistoryLive();
    } else {
        loadHistoryData();
        startHistoryRefresh();
    }
}

/**
 * Historische Daten vom API laden und Charts aktualisieren.
 */
async function _fetchHistory(range, signal) {
    const resp = await SharedUi.fetch.safeFetch(`/api/monitoring/history?range=${range}`, { signal });
    if (!resp.ok) throw new Error(`Verlauf: HTTP ${resp.status}`);
    return resp.json();
}

function showHistoryData(data) {
    historyCharts.forEach(chart => chart.update(data.timestamps, data.series));
}

async function loadHistoryData() {
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || range.live) return;
    const requestedRange = historyActiveRange;
    const generation = ++historyLoadGeneration;
    const cached = historyCache.get(requestedRange);
    if (cached) showHistoryData(cached.data);
    else {
        historyCharts.forEach(chart => chart.update([], {}));
    }
    const ttl = range.refreshMs || 300000;
    if (cached && Date.now() - cached.savedAt < ttl) return;
    if (historyRequest) historyRequest.abort();
    const request = new AbortController();
    historyRequest = request;
    try {
        const data = await _fetchHistory(requestedRange, request.signal);
        if (generation !== historyLoadGeneration || historyActiveRange !== requestedRange) return;
        if (!Array.isArray(data.timestamps) || !data.series || !Number.isSafeInteger(data.count)) {
            throw new Error('Ungültige Verlaufsdaten');
        }
        historyCache.set(requestedRange, { data, savedAt: Date.now() });
        showHistoryData(data);
    } catch (error) {
        if (generation !== historyLoadGeneration || request.signal.aborted) return;
        console.warn('Monitoring-Verlauf:', error);
    } finally {
        if (historyRequest === request) historyRequest = null;
    }
}

function startHistoryRefresh() {
    const range = HISTORY_RANGES[historyActiveRange];
    if (range && range.refreshMs) {
        kiaraPolling.start('system-monitoring:history', loadHistoryData, range.refreshMs);
    }
}

function stopHistoryRefresh() {
    kiaraPolling.stop('system-monitoring:history');
    if (historyRequest) historyRequest.abort();
    historyRequest = null;
}

function normalizeHistoryLiveSample(raw) {
    if (!raw || typeof raw !== 'object') return null;
    if (!Number.isSafeInteger(raw.sequence) || raw.sequence <= 0) return null;
    if (typeof raw.timestamp !== 'number' || !Number.isFinite(raw.timestamp)) return null;
    if (!raw.values || typeof raw.values !== 'object') return null;

    const values = {};
    for (const key of HISTORY_LIVE_METRIC_KEYS) {
        const value = raw.values[key];
        values[key] = typeof value === 'number' && Number.isFinite(value) ? value : null;
    }

    return { sequence: raw.sequence, timestamp: raw.timestamp, values };
}

function pruneHistoryLiveSamples() {
    if (!historyLiveSamples.length) return;
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || !range.live) return;
    const latestTimestamp = historyLiveSamples[historyLiveSamples.length - 1].timestamp;
    const cutoff = latestTimestamp - range.seconds;
    let firstVisible = 0;
    while (
        firstVisible < historyLiveSamples.length
        && historyLiveSamples[firstVisible].timestamp <= cutoff
    ) {
        firstVisible += 1;
    }
    if (firstVisible > 0) historyLiveSamples.splice(0, firstVisible);
    if (historyLiveSamples.length > range.maxPoints) {
        historyLiveSamples.splice(0, historyLiveSamples.length - range.maxPoints);
    }
}

function replaceHistoryLiveSamples(rawSamples) {
    const bySequence = new Map();
    for (const raw of Array.isArray(rawSamples) ? rawSamples : []) {
        const sample = normalizeHistoryLiveSample(raw);
        if (sample) bySequence.set(sample.sequence, sample);
    }
    historyLiveSamples = [...bySequence.values()].sort((a, b) => a.sequence - b.sequence);
    pruneHistoryLiveSamples();
}

function appendHistoryLiveSample(raw) {
    const sample = normalizeHistoryLiveSample(raw);
    if (!sample) return false;
    const latest = historyLiveSamples[historyLiveSamples.length - 1];
    if (latest && sample.sequence <= latest.sequence) return false;
    historyLiveSamples.push(sample);
    pruneHistoryLiveSamples();
    return true;
}

function renderHistoryLiveCharts() {
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || !range.live) return;
    const timestamps = historyLiveSamples.map(sample => sample.timestamp);
    const series = {};
    for (const cfg of CHART_CONFIGS) {
        for (const item of cfg.series) {
            if (!series[item.key]) series[item.key] = [];
        }
    }
    for (const sample of historyLiveSamples) {
        for (const key of Object.keys(series)) {
            series[key].push(sample.values[key]);
        }
    }
    historyCharts.forEach(chart => chart.update(timestamps, series, {
        relativeWindowSeconds: range.seconds,
    }));
}

function scheduleHistoryLiveRender() {
    if (historyLiveRenderFrame !== null) return;
    historyLiveRenderFrame = window.requestAnimationFrame(() => {
        historyLiveRenderFrame = null;
        renderHistoryLiveCharts();
    });
}

function startHistoryLive(resetSamples = true) {
    stopHistoryLive(false);
    const rangeKey = historyActiveRange;
    const range = HISTORY_RANGES[rangeKey];
    if (!range || !range.live) return;
    if (resetSamples) historyLiveSamples = [];
    const generation = ++historyLiveGeneration;
    renderHistoryLiveCharts();

    const socket = new EventSource(`/api/monitoring/live?range=${rangeKey}`);
    historyLiveSocket = socket;
    socket.addEventListener('history', event => {
        if (generation !== historyLiveGeneration || historyActiveRange !== rangeKey) return;
        let message;
        try { message = JSON.parse(event.data); } catch (error) { return; }
        if (message.version !== HISTORY_LIVE_PROTOCOL_VERSION) {
            console.warn('Monitoring-Verlauf: Protokoll nicht kompatibel.');
            socket.close();
            return;
        }
        if (message.type === `history_${rangeKey}_snapshot`) {
            replaceHistoryLiveSamples(message.samples);
            scheduleHistoryLiveRender();
        } else if (message.type === `history_${rangeKey}_point`) {
            if (appendHistoryLiveSample(message.sample)) {
                scheduleHistoryLiveRender();
            }
        }
    });
    socket.addEventListener('error', () => {
        if (generation !== historyLiveGeneration) return;
        console.warn('Monitoring-Verlauf: Live-Verbindung unterbrochen.');
    });
}

function stopHistoryLive(clearSamples = true) {
    historyLiveGeneration += 1;
    if (historyLiveRenderFrame !== null) {
        window.cancelAnimationFrame(historyLiveRenderFrame);
        historyLiveRenderFrame = null;
    }
    if (historyLiveSocket) {
        const socket = historyLiveSocket;
        historyLiveSocket = null;
        socket.close();
    }
    if (clearSamples) historyLiveSamples = [];
}

/**
 * Alle Chart-Instanzen aufraeumen.
 */
function destroyHistoryCharts() {
    document.removeEventListener('themeChanged', refreshHistoryTheme);
    historyLoadGeneration += 1;
    stopHistoryRefresh();
    stopHistoryLive();
    historyCharts.forEach(c => c.destroy());
    historyCharts = [];
}

/**
 * Theme-Refresh fuer alle Charts (nach Theme-Wechsel).
 */
function refreshHistoryTheme() {
    historyCharts.forEach(c => c.refreshTheme());
}

return { init: initHistoryTab, destroy: destroyHistoryCharts, setRange: setHistoryRange, rangeHtml, chartsHtml };
})();
