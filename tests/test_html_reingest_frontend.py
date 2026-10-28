"""Execute HTML re-ingest dialog and existing job presentation in Node."""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not NODE, reason="Node is required for frontend tests")

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let model = 'nomic-embed-text', mode = 'outdated', modal = null, calls = [], notices = [];
const escapeHtml = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const context = {
    console, URLSearchParams, setTimeout: () => {},
    document: {getElementById: id => id === 'dsEmbedModelFilter' ? {value: model}
        : id === 'htmlReingestMode-12' ? {value: mode} : null},
    SharedUi: {
        runtime: {escapeHtml, escapeAttr: escapeHtml},
        notify: {show: (...args) => notices.push(args)},
        modal: {open: options => {modal = options; return 'modal-1';}, close: () => {}},
        fetch: {safeFetch: async (url, options) => {
            calls.push({url, body: JSON.parse(options.body)});
            return {ok: true, json: async () => ({job_id: 478})};
        }},
    },
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
"""


def _run(script):
    result = subprocess.run(
        [
            NODE,
            "-e",
            HARNESS + script,
            str(ROOT / "kiara/web/static/js/tab_datenquellen.js"),
            str(ROOT / "kiara/web/static/js/tab_jobs.js"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dialog_queues_selected_model_and_mode_without_a_preview():
    _run(r"""
(async () => {
    for (const selectedMode of ['outdated', 'all']) {
        mode = selectedMode;
        const started = context.triggerDsHtmlReingest(12, '<img src=x onerror=1>');
        assert.match(modal.body, /&lt;img/);
        assert.doesNotMatch(modal.body, /<img/);
        assert.match(modal.body, /nomic-embed-text/);
        assert.match(modal.body, /value="outdated" selected/);
        assert.match(modal.body, /HTML-Mailbodies/);
        modal.buttons[1].onClick();
        await started;
        assert.equal(calls.at(-1).url, '/api/admin/datasources/12/reingest-html');
        assert.deepEqual(calls.at(-1).body, {embedding_model: model, mode: selectedMode});
    }
    assert.equal(calls.length, 2, 'one enqueue request for each confirmed job');
})().catch(error => {console.error(error); process.exitCode = 1;});
""")


def test_cancel_close_and_excluded_model_never_enqueue():
    _run(r"""
(async () => {
    let started = context.triggerDsHtmlReingest(12, 'GMX');
    modal.buttons[0].onClick();
    await started;
    started = context.triggerDsHtmlReingest(12, 'GMX');
    modal.onClose();
    await started;
    model = 'text-embedding-3-small';
    modal = null;
    await context.triggerDsHtmlReingest(12, 'GMX');
    assert.equal(modal, null);
    assert.equal(calls.length, 0);
    assert.match(notices.at(-1)[0], /Nomic und Hellord/);
})().catch(error => {console.error(error); process.exitCode = 1;});
""")


def test_existing_job_list_distinguishes_html_and_its_classification_results():
    _run(r"""
assert.match(context._jobTypeBadge('incremental', {}), /Inkr\./);
assert.match(context._jobTypeBadge('full', {}), /Voll/);
const stats = {html_reingest: {mode: 'outdated'}, html_reingest_summary: {
    inspected: 80, selected: 50, current: 12, not_html: 8,
    missing_original: 3, empty_body: 2, errors: 5,
}};
assert.match(context._jobTypeBadge('incremental', {stats}), /HTML/);
const html = context._renderHtmlReingestSummary(stats);
for (const text of ['HTML-Neuaufbereitung', 'Nur veraltete', 'Bestand geprüft',
    'Ausgewählt', 'Bereits aktuell', 'Kein HTML', 'davon Original fehlt', 'davon Body leer',
    'Fehler insgesamt']) assert.ok(html.includes(text), text);
for (const count of [80, 50, 12, 8, 3, 2, 5]) assert.ok(html.includes(`>${count}</td>`));
assert.equal(context._renderHtmlReingestSummary({}), '');
assert.match(context._renderHtmlReingestSummary({html_reingest: {mode: 'all'}}), /Alle bestätigten/);
assert.match(context._renderHtmlReingestSummary({html_reingest: {mode: 'all'}}), /noch nicht abgeschlossen/);
""")


def test_only_failed_and_cancelled_html_jobs_offer_retry():
    _run(r"""
for (const status of ['error', 'cancelled']) {
    const buttons = context._jobActionButtons(478, {status, stats: {html_reingest: {mode: 'all'}}});
    assert.match(buttons, /data-action="retry-html-job"/);
    assert.match(buttons, /title="Wiederholen"/);
    assert.match(buttons, /data-job-id="478"/);
    assert.match(buttons, /data-action="delete-job"/);
    assert.doesNotMatch(context._jobActionButtons(478, {status}), /retry-html-job/);
}
for (const status of ['running', 'pending', 'success']) {
    const buttons = context._jobActionButtons(478, {status, stats: {html_reingest: {mode: 'all'}}});
    assert.doesNotMatch(buttons, /retry-html-job/);
    if (status !== 'success') assert.match(buttons, /cancel-job/);
}
assert.match(context._jobActionButtons(478, {status: 'running'}), /toggle-job-pause/);
""")


def test_retry_uses_existing_job_and_displays_revision_conflict():
    _run(r"""
(async () => {
    let locks = 0, refreshes = 0;
    const button = {};
    context.SharedUi.runtime.withButtonLock = async (btn, action) => {
        assert.equal(btn, button); locks++; return action();
    };
    context.SharedUi.fetch.safeFetch = async (url, options) => {
        calls.push({url, options});
        return {ok: true, json: async () => ({job_id: 479})};
    };
    context.refresh = () => {refreshes++;};
    vm.runInContext('jobsTable = {loadData: refresh}', context);
    await context._retryHtmlJob(button, 478);
    assert.equal(locks, 1);
    assert.equal(refreshes, 1);
    assert.equal(calls[0].url, '/api/admin/ingestion-jobs/478/retry-html');
    assert.equal(calls[0].options.method, 'POST');
    assert.equal(calls[0].options.body, undefined, 'retry cannot replace model or mode');
    assert.match(notices.at(-1)[0], /479/);
    context.SharedUi.fetch.safeFetch = async () => {
        throw {message: 'HTTP 409', body: {detail: 'HTML-Revision geaendert; neuen HTML-Job starten'}};
    };
    await context._retryHtmlJob(button, 478);
    assert.match(notices.at(-1)[0], /HTML-Revision geaendert/);
    assert.equal(notices.at(-1)[1], 'error');
    assert.equal(refreshes, 1, 'conflict must not pretend a retry was queued');
})().catch(error => {console.error(error); process.exitCode = 1;});
""")


def test_html_resume_chain_includes_manual_retries_without_restart_claim():
    _run(r"""
const row = {id: 479, status: 'error', resume_chain_length: 2,
    resume_chain_original_job_id: 478, stats: {html_reingest: {mode: 'all'}},
    resume_chain: [{id: 478, status: 'cancelled'}],
};
const badge = context._jobStatusBadge(row.status, row);
assert.match(badge, /Wiederaufnahmen/);
assert.match(badge, /wiederaufgenommen/);
assert.doesNotMatch(badge, /Service-Restart/);
const detail = context._renderJobDetail(row);
assert.match(detail, /Wiederaufnahmen/);
assert.doesNotMatch(detail, /Neustarts/);
assert.match(context._jobStatusBadge(row.status, {...row, stats: {}}), /Service-Restart/);
assert.match(context._renderJobDetail({...row, stats: {}}), /Neustarts/);
""")


def test_action_is_available_only_for_idle_supported_active_datasources():
    _run(r"""
const content = {innerHTML: '', setAttribute() {}};
context.document = {
    getElementById: id => id === 'contentContainer' ? content : null,
    body: {addEventListener() {}},
};
let table;
context.SharedUi.table = {ExpandableTable: class {
    constructor() {this.renderers = {}; table = this;}
    loadData() {}
}};
context.SharedUi.fetch.safeFetch = async () => ({ok: true, json: async () => ({data: []})});
context.registerTabCleanup = () => {};
(async () => {
    await context.initDatenquellenTab();
    const base = {id: 12, name: 'GMX', type: 'imap', is_active: true, ingestion_status: 'idle'};
    const render = overrides => table.renderers.actions(null, {}, {...base, ...overrides});
    for (const type of ['local', 'smb', 'webdav', 'imap', 'exchange_online'])
        assert.match(render({type}), /data-action="trigger-ds-html-reingest"/);
    for (const type of ['sql', 'sqlds', 'unknown'])
        assert.doesNotMatch(render({type}), /data-action="trigger-ds-html-reingest"/);
    for (const ingestion_status of ['pending', 'running', 'paused'])
        assert.doesNotMatch(render({ingestion_status}), /data-action="trigger-ds-html-reingest"/);
    assert.doesNotMatch(render({is_active: false}), /data-action="trigger-ds-html-reingest"/);
    assert.match(render({}), /data-action="trigger-ds-ingest"/);
    assert.match(render({}), /data-action="trigger-ds-reingest"/);
})().catch(error => {console.error(error); process.exitCode = 1;});
""")
