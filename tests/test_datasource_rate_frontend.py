"""Execute the datasource throughput calculation and table lifecycle in Node."""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kiara/web/static/js/tab_datenquellen.js"
TABLE_SOURCE = ROOT / "kiara/web/static/ui_core/js/table.js"
pytestmark = pytest.mark.skipif(not NODE, reason="Node is required for frontend tests")

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let now = 0, rows = [], fetchOverride = null;
const polls = new Map(), cleanups = [];
const content = {innerHTML: '', setAttribute() {}};
const base = {
    id: 12, job_id: 455, started_at: '2026-09-12 15:29:01+00',
    ingestion_status: 'running', phase: 'Verarbeitung...',
    progress_processed: 312683, progress_total: 377462,
};
const response = data => ({ok: true, json: async () => ({data})});
const context = {
    console, URLSearchParams, performance: {now: () => now},
    document: {
        getElementById: id => id === 'contentContainer' ? content : null,
        addEventListener() {}, body: {addEventListener() {}},
    },
    SharedUi: {
        configure() {}, runtime: {escapeHtml: String},
        fetch: {safeFetch: async url => {
            if (url !== '/api/admin/datasources') return response([]);
            return fetchOverride ? fetchOverride() : response(rows);
        }},
    },
    kiaraPolling: {
        has: key => polls.has(key), start: (key, fn) => polls.set(key, fn),
        stop: key => polls.delete(key),
    },
    registerTabCleanup: (key, fn) => cleanups.push(fn),
    _runTabCleanups: () => { while (cleanups.length) cleanups.pop()(); },
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const api = vm.runInContext('({update: _updateDsRates, get: _getDsRate, '
    + 'clear: () => _dsRateHistory.clear(), table: () => dsTable, '
    + 'init: initDatenquellenTab})', context);
const tick = () => new Promise(resolve => setImmediate(resolve));
const poll = () => polls.get('datenquellen:refresh')();
const observe = (at, count, changes = {}) => {
    now = at;
    const row = {...base, ...changes, progress_processed: count};
    api.update([row]);
    return api.get(row);
};
"""


def _run(script):
    result = subprocess.run(
        [NODE, "-e", HARNESS + script, str(SOURCE), str(TABLE_SOURCE)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_progress_delta_wait_times_and_bounded_window():
    _run(r"""
assert.equal(observe(0, 312683), null, 'old progress is only the baseline');
assert.equal(observe(3000, 312695), null, 'wait for a useful observation window');
assert.equal(observe(6000, 312707), 4);
for (let at = 9000; at <= 15000; at += 3000) {
    assert.equal(observe(at, 312707), 24000 / at, 'checkpoint waits count as elapsed time');
}
for (let at = 18000; at <= 36000; at += 3000) observe(at, 312707);
assert.equal(api.get({...base, progress_processed: 312707}), 0, '30 seconds without progress');
assert.equal(observe(39000, 312737), 1, 'only progress in the recent window counts');
assert.equal(observe(70000, 313000), null, 'background/sleep gaps require a new baseline');
assert.equal(observe(76000, 313018), 3);
const row = {...base, progress_processed: 313018};
now = 80000;
assert.equal(api.get(row), 3);
assert.equal(api.get(row), 3, 'rendering must not add observations');
assert.equal(observe(82000, 313030), 2.5);
now = 112001;
assert.equal(api.get(row), null, 'stale observations must not display a rate');
""")


def test_resets_invalid_data_and_independent_datasources():
    _run(r"""
const resetCases = [
    {job_id: 456}, {started_at: '2026-09-12T17:30:00+02:00'},
    {progress_total: 400000}, {phase: 'Synchronisierung...'},
    {ingestion_status: 'paused'}, {ingestion_status: 'success'},
    {job_id: null}, {started_at: null}, {started_at: 'invalid'},
    {progress_total: null}, {progress_total: 0}, {progress_total: 2},
];
for (const change of resetCases) {
    api.clear();
    observe(0, 312683);
    assert.equal(observe(6000, 312707), 4);
    assert.equal(observe(9000, 312719, change), null, JSON.stringify(change));
    assert.equal(observe(12000, 312730), null, 'resume starts with a new baseline');
    assert.equal(observe(18000, 312748), 3);
}
for (const count of [null, undefined, -1, 1.5, '312731', NaN, Infinity, 400000]) {
    api.clear();
    observe(0, 312683);
    observe(6000, 312707);
    assert.equal(observe(9000, count), null, `invalid counter: ${count}`);
}
api.clear();
observe(0, 312683);
observe(6000, 312707);
assert.equal(observe(9000, 312680), null, 'counter rollback resets the sample');
assert.equal(observe(8000, 312683), null, 'monotonic clock rollback resets the sample');
api.clear();
const second = {...base, id: 13, job_id: 456, progress_processed: 100};
api.update([base, second], 0);
api.update([{...base, progress_processed: 312707}, {...second, progress_processed: 112}], 6000);
assert.equal(api.get(base, 6000), 4);
assert.equal(api.get(second, 6000), 2);
api.update([second], 9000);
assert.equal(api.get(base, 9000), null, 'removed datasource is forgotten');
""")


def test_table_polling_render_pause_failure_and_late_response():
    _run(r"""
// Keep actual loading, request ordering and callbacks; replace only DOM rendering.
const proto = context.SharedUi.table.ExpandableTable.prototype;
proto.showLoading = proto.showError = () => {};
proto.render = proto.silentRender = function () {
    this.html = this.filteredData.map(row => this.renderers.ingestionStatus(row.ingestion_status, {}, row)).join('');
};
(async () => {
    rows = [base];
    await api.init();
    await tick();
    let table = api.table();
    assert.match(table.html, /\(– D\/s\)/);
    now = 6000;
    rows = [{...base, progress_processed: 312707}];
    await poll();
    assert.match(table.html, /\(4\.0 D\/s\)/);
    assert.match(table.html, /Wartezeiten sind enthalten/);
    now = 9000;
    table.render(); table.render();
    await poll();
    assert.match(table.html, /\(2\.7 D\/s\)/);
    now = 12000;
    rows = [{...rows[0], ingestion_status: 'paused'}];
    await poll();
    assert.doesNotMatch(table.html, /D\/s/);
    now = 15000;
    rows = [{...base, progress_processed: 312800}];
    await poll();
    assert.match(table.html, /\(– D\/s\)/);
    now = 21000;
    rows = [{...base, progress_processed: 312818}];
    await poll();
    assert.match(table.html, /\(3\.0 D\/s\)/);
    now = 24000;
    fetchOverride = () => ({ok: false, status: 503});
    await poll();
    assert.match(table.html, /\(– D\/s\)/, 'HTTP failure hides stale rate');
    fetchOverride = null;
    now = 27000;
    await poll();
    now = 33000;
    rows = [{...base, progress_processed: 312842}];
    await poll();
    assert.match(table.html, /\(4\.0 D\/s\)/);

    let resolveOld;
    fetchOverride = () => new Promise(resolve => { resolveOld = resolve; });
    const oldPoll = poll();
    fetchOverride = null;
    now = 36000;
    rows = [{...base, job_id: 457}];
    await api.init();
    await tick();
    table = api.table();
    now = 42000;
    rows = [{...rows[0], progress_processed: 312707}];
    await poll();
    assert.match(table.html, /\(4\.0 D\/s\)/);
    resolveOld(response([{...base, progress_processed: 350000}]));
    await oldPoll;
    assert.equal(api.get(rows[0]), 4, 'old table response must not change the new baseline');
    context._runTabCleanups();
    assert.equal(api.get(rows[0]), null);
    assert.equal(polls.size, 0);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")
