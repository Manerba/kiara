"""Node-basierte Vertragstests fuer die 1m-/10m-Logik im Verlauf-Tab."""

import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
SOURCE = Path(__file__).resolve().parents[1] / "kiara/web/static/js/monitoring_history.js"
CHART_SOURCE = SOURCE.with_name("monitoring_chart.js")


@unittest.skipUnless(NODE, "node ist fuer Frontend-Vertragstest nicht installiert")
class HistoryLiveFrontendTests(unittest.TestCase):
    def test_buffer_reconnect_range_races_cache_and_cleanup(self):
        harness = r"""
const assert = require('assert');
const vm = require('vm');
const fs = require('fs');
const sockets = [], charts = [], requests = [], frames = new Map(), polls = new Map();
const listeners = new Map();
let nextFrame = 1;
function node() { return {textContent: '', appendChild() {}, replaceChildren() {}}; }
class EventSource {
    constructor(url) { this.url = url; this.listeners = {}; sockets.push(this); }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    close() { this.closed = true; }
    emit(frame) { this.listeners.history({data: JSON.stringify(frame)}); }
}
class TimeSeriesChart {
    constructor(id, config) { this.config = config; charts.push(this); }
    update(timestamps, series, options) { this.last = {timestamps, series, options}; }
    refreshTheme() { this.themed = true; }
    destroy() { this.destroyed = true; }
}
const context = {
    console, Date, AbortController, EventSource, TimeSeriesChart,
    SharedUi: {fetch: {safeFetch: (url, options) => new Promise(resolve => requests.push({url, options, resolve}))}},
    kiaraPolling: {start: (key, fn) => polls.set(key, fn), stop: key => polls.delete(key)},
    document: {
        getElementById: () => node(), querySelectorAll: () => [],
        createElement: node, createTextNode: node,
        addEventListener: (name, fn) => listeners.set(name, fn),
        removeEventListener: name => listeners.delete(name),
    },
    window: {
        requestAnimationFrame: fn => { const key = nextFrame++; frames.set(key, fn); return key; },
        cancelAnimationFrame: key => frames.delete(key),
    },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8') + '\nglobalThis.monitor = KiaraMonitoringHistory;', context);
const monitor = context.monitor;
const sample = (sequence, timestamp) => ({sequence, timestamp, values: {cpu_usage: 42, net_sent_mb_s: 0.001}, states: {network: 'ok'}});
const snapshot = (range, samples) => ({type: `history_${range}_snapshot`, version: 1, interval_ms: 250, samples});
const flushFrames = () => { for (const [key, fn] of frames) { frames.delete(key); fn(); } };
const tick = () => new Promise(resolve => setImmediate(resolve));
const history = (value) => ({timestamps: [100], series: {cpu_usage: [value]}, count: 1, to: 100});
(async () => {
    monitor.init();
    assert.equal(charts.length, 4);
    assert.deepEqual(charts.map(chart => chart.config.title), ['CPU & Load', 'RAM', 'Disk I/O', 'Netzwerk']);
    assert.equal(requests.length, 0, 'live must not request DB history');
    sockets[0].emit(snapshot('1m', Array.from({length: 2400}, (_, i) => sample(i + 1, i / 4))));
    flushFrames();
    assert.equal(charts[0].last.timestamps.length, 240);
    assert.equal(charts[0].last.options.relativeWindowSeconds, 60);

    monitor.setRange('10m');
    assert(sockets[0].closed);
    sockets[1].emit(snapshot('10m', Array.from({length: 2400}, (_, i) => sample(i + 1, i / 4))));
    flushFrames();
    assert.equal(charts[0].last.timestamps.length, 2400);
    sockets[0].emit(snapshot('1m', [sample(1, 1)]));
    flushFrames();
    assert.equal(charts[0].last.timestamps.length, 2400, 'closed range must not overwrite current data');

    sockets[1].listeners.error();
    sockets[1].emit(snapshot('10m', [sample(1, 1000), sample(2, 1000.25)]));
    flushFrames();
    assert.equal(charts[0].last.timestamps.length, 2, 'reconnect after restart replaces old sequences');
    sockets[1].emit({version: 1, type: 'history_10m_point', sample: sample(2, 1000.25)});
    flushFrames();
    assert.equal(charts[0].last.timestamps.length, 2, 'duplicates must not grow buffer');

    monitor.setRange('1h');
    monitor.setRange('6h');
    assert(requests[0].options.signal.aborted);
    requests[0].resolve({ok: true, json: async () => history(99)});
    await tick();
    assert.equal(charts[0].last.timestamps.length, 0, 'stale request must not overwrite pending range');
    requests[1].resolve({ok: true, json: async () => history(12)});
    await tick();
    assert.equal(charts[0].last.series.cpu_usage[0], 12);
    monitor.setRange('1m');
    monitor.setRange('6h');
    assert.equal(requests.length, 2, 'cached range must be reused');
    assert.equal(charts[0].last.series.cpu_usage[0], 12);
    listeners.get('themeChanged')();
    assert(charts.every(chart => chart.themed));
    monitor.destroy();
    assert.equal(frames.size, 0);
    assert.equal(polls.size, 0);
    assert.equal(listeners.size, 0);
    assert(charts.every(chart => chart.destroyed));
    assert(sockets.every(socket => socket.closed));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
        completed = subprocess.run(
            [NODE, "-e", harness, str(SOURCE)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_live_charts_use_static_relative_ticks(self):
        harness = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');

class FakeChart {
    constructor(_context, config) {
        this.data = config.data;
        this.options = config.options;
        this.updateModes = [];
    }
    update(mode) { this.updateModes.push(mode); }
    destroy() {}
}

const context = {
    Chart: FakeChart,
    Date,
    Number,
    document: {
        documentElement: {},
        getElementById: () => ({ getContext: () => ({}) }),
    },
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
};
vm.createContext(context);
vm.runInContext(source + '\nglobalThis.__TimeSeriesChart = TimeSeriesChart;', context);

const chart = new context.__TimeSeriesChart('chart', {
    series: [{ key: 'cpu_usage', label: 'CPU', color: '#123456', yAxisID: 'y' }],
    yAxes: [{ id: 'y', label: '%', min: 0, max: 100 }],
});

const firstTimestamps = [100, 110, 120, 130, 140, 150, 160];
chart.update(firstTimestamps, { cpu_usage: [1, 2, 3, 4, 5, 6, 7] }, {
    relativeWindowSeconds: 60,
});
let xAxis = chart.chart.options.scales.x;
assert.strictEqual(xAxis.type, 'linear');
assert.strictEqual(xAxis.min, 0);
assert.strictEqual(xAxis.max, 60);
assert.strictEqual(xAxis.reverse, true);
assert.strictEqual(xAxis.ticks.stepSize, 10);
assert.strictEqual(xAxis.ticks.autoSkip, false);
assert.deepStrictEqual(
    [0, 10, 20, 30, 40, 50, 60].map(xAxis.ticks.callback),
    ['0 s', '10 s', '20 s', '30 s', '40 s', '50 s', '60 s'],
);
assert.deepStrictEqual(
    Array.from(chart.chart.data.datasets[0].data, point => point.x),
    [60, 50, 40, 30, 20, 10, 0],
);

chart.update(firstTimestamps.map(value => value + 0.25), {
    cpu_usage: [1, 2, 3, 4, 5, 6, 7],
}, { relativeWindowSeconds: 60 });
assert.deepStrictEqual(
    Array.from(chart.chart.data.datasets[0].data, point => point.x),
    [60, 50, 40, 30, 20, 10, 0],
    'relative labels and positions must not drift with wall-clock time',
);

const tenMinuteTimestamps = [100, 220, 340, 460, 580, 700];
chart.update(tenMinuteTimestamps, { cpu_usage: [1, 2, 3, 4, 5, 6] }, {
    relativeWindowSeconds: 600,
});
xAxis = chart.chart.options.scales.x;
assert.strictEqual(xAxis.type, 'linear');
assert.strictEqual(xAxis.min, 0);
assert.strictEqual(xAxis.max, 600);
assert.strictEqual(xAxis.reverse, true);
assert.strictEqual(xAxis.ticks.stepSize, 120);
assert.deepStrictEqual(
    [0, 120, 240, 360, 480, 600].map(xAxis.ticks.callback),
    ['0 min', '2 min', '4 min', '6 min', '8 min', '10 min'],
);
assert.deepStrictEqual(
    Array.from(chart.chart.data.datasets[0].data, point => point.x),
    [600, 480, 360, 240, 120, 0],
);

chart.update([170], { cpu_usage: [8] });
xAxis = chart.chart.options.scales.x;
assert.strictEqual(xAxis.type, 'time');
assert.ok(chart.chart.data.datasets[0].data[0].x instanceof Date);
"""
        completed = subprocess.run(
            [NODE, "-e", harness, str(CHART_SOURCE)],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"node stdout:\n{completed.stdout}\nnode stderr:\n{completed.stderr}",
        )
