/**
 * KIara Admin - System > System-Log Sub-Tab
 * Strukturierte Security-Events aus system_events mit Filter, Detail-View,
 * CSV-Export und opt-in Auto-Refresh (15s).
 *
 * Produktneutrale Bausteine (SystemLogFilterBar, renderJsonDetail) sind als
 * freistehende Top-Level-Klassen/Funktionen organisiert, damit eine spaetere
 * Extraktion nach ui_core/filters.js bzw. ui_core/jsonview.js moeglich ist,
 * ohne den Body neu zu schreiben.
 */

let slTable = null;
let slFilterBar = null;
let _slInitSeq = 0;
let _slDomGeneration = 0;

/* ──────────────── Produktneutrale Bausteine ──────────────── */

class SystemLogFilterBar {
    constructor(containerId, { filters, onChange }) {
        this._containerId = containerId;
        this._filters = filters || [];
        this._onChange = typeof onChange === 'function' ? onChange : () => {};
        this._values = {};
        this._selects = {};
        this._inputs = {};
        this._toggles = {};
    }

    render() {
        const container = document.getElementById(this._containerId);
        if (!container) return;
        for (const f of this._filters) {
            const group = document.createElement('div');
            group.className = 'filter-group';
            const label = document.createElement('span');
            label.className = 'filter-label';
            label.textContent = f.label + ':';
            group.appendChild(label);

            if (f.type === 'select' || f.type === 'select-async') {
                const select = document.createElement('select');
                select.className = 'filter-input';
                select.dataset.filterKey = f.key;
                (f.options || []).forEach(opt => {
                    const o = document.createElement('option');
                    o.value = opt.value;
                    o.textContent = opt.label;
                    if (f.default !== undefined && f.default === opt.value) {
                        o.selected = true;
                        this._values[f.key] = opt.value;
                    }
                    select.appendChild(o);
                });
                select.addEventListener('change', () => {
                    this._values[f.key] = select.value;
                    this._onChange(this.getValues());
                });
                group.appendChild(select);
                this._selects[f.key] = select;
                if (f.type === 'select-async' && typeof f.load === 'function') {
                    this._loadAsyncOptions(f, select);
                }
            } else if (f.type === 'search') {
                const input = document.createElement('input');
                input.type = 'text';
                input.className = 'filter-input';
                input.placeholder = f.placeholder || '';
                input.dataset.filterKey = f.key;
                let debounceTimer = null;
                input.addEventListener('input', () => {
                    this._values[f.key] = input.value;
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(() => this._onChange(this.getValues()), 300);
                });
                group.appendChild(input);
                this._inputs[f.key] = input;
            } else if (f.type === 'toggle') {
                const wrap = document.createElement('label');
                wrap.className = 'filter-toggle';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                if (f.default === true) {
                    cb.checked = true;
                    this._values[f.key] = true;
                }
                cb.dataset.filterKey = f.key;
                cb.addEventListener('change', () => {
                    this._values[f.key] = cb.checked;
                    this._onChange(this.getValues());
                });
                wrap.appendChild(cb);
                const span = document.createElement('span');
                span.textContent = f.toggleLabel || '';
                wrap.appendChild(span);
                group.appendChild(wrap);
                this._toggles[f.key] = cb;
            }
            container.appendChild(group);
        }
    }

    async _loadAsyncOptions(f, select) {
        try {
            const items = await f.load();
            if (!Array.isArray(items)) return;
            items.forEach(val => {
                const o = document.createElement('option');
                o.value = val;
                o.textContent = val;
                select.appendChild(o);
            });
        } catch (_err) { /* ignore */ }
    }

    getValues() {
        return Object.assign({}, this._values);
    }

    setValues(obj) {
        if (!obj || typeof obj !== 'object') return;
        for (const [key, val] of Object.entries(obj)) {
            this._values[key] = val;
            if (this._selects[key]) this._selects[key].value = val;
            else if (this._inputs[key]) this._inputs[key].value = val;
            else if (this._toggles[key]) this._toggles[key].checked = !!val;
        }
    }

    destroy() {
        this._selects = {};
        this._inputs = {};
        this._toggles = {};
        this._values = {};
    }
}

function renderJsonDetail(data, opts) {
    const options = Object.assign({ collapsible: true, maxDepth: 4, pretty: true }, opts || {});
    const escape = (typeof SharedUi !== 'undefined' && SharedUi.runtime)
        ? SharedUi.runtime.escapeHtml
        : (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    function renderValue(value, depth) {
        if (value === null) return '<span class="json-null">null</span>';
        if (typeof value === 'boolean') return `<span class="json-bool">${value}</span>`;
        if (typeof value === 'number') return `<span class="json-num">${value}</span>`;
        if (typeof value === 'string') {
            if (value.includes('\n') && value.startsWith('Traceback')) {
                return `<pre class="json-traceback">${escape(value)}</pre>`;
            }
            return `<span class="json-str">"${escape(value)}"</span>`;
        }
        if (Array.isArray(value)) {
            if (depth >= options.maxDepth) return `<span class="json-trunc">[...]</span>`;
            if (value.length === 0) return '<span class="json-empty">[]</span>';
            const items = value.map(v => `<li>${renderValue(v, depth + 1)}</li>`).join('');
            if (!options.collapsible) return `<ul class="json-array">${items}</ul>`;
            return `<details class="json-details" open><summary>Array (${value.length})</summary><ul class="json-array">${items}</ul></details>`;
        }
        if (typeof value === 'object') {
            const keys = Object.keys(value);
            if (keys.length === 0) return '<span class="json-empty">{}</span>';
            if (depth >= options.maxDepth) return `<span class="json-trunc">{...}</span>`;
            const body = keys.map(k =>
                `<li><span class="json-key">${escape(k)}</span>: ${renderValue(value[k], depth + 1)}</li>`
            ).join('');
            if (!options.collapsible) return `<ul class="json-obj">${body}</ul>`;
            return `<details class="json-details" open><summary>Object (${keys.length})</summary><ul class="json-obj">${body}</ul></details>`;
        }
        return `<span class="json-str">${escape(String(value))}</span>`;
    }

    return renderValue(data, 0);
}

/* ──────────────── Log-spezifisches Wiring ──────────────── */

const SL_DEFAULT_EXCLUDE_EVENT_TYPES = 'SYSLOG_VIEW';

function _slSinceForRange(range) {
    const now = Date.now();
    const map = { '1h': 3600e3, '24h': 24 * 3600e3, '7d': 7 * 24 * 3600e3 };
    if (!map[range]) return null;
    return new Date(now - map[range]).toISOString();
}

function mapFilterValuesToApi(values) {
    const out = {};
    if (!values || typeof values !== 'object') return out;

    const since = _slSinceForRange(values.range);
    if (since) out.since = since;

    if (values.min_level && values.min_level !== '') out.min_level = values.min_level;

    if (values.reason && values.reason !== '') out.reason = values.reason;

    if (values.search && values.search.trim() !== '') out.search = values.search.trim();

    // Default-Filter fuer Feedback-Rauschen (SYSLOG_VIEW-Self-Audit)
    if (values.hideSelfAudit !== false) {
        out.exclude_event_types = SL_DEFAULT_EXCLUDE_EVENT_TYPES;
    }

    return out;
}

async function initSystemLogsTab() {
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-logs');
    _slInitSeq++;
    const mySeq = _slInitSeq;

    registerTabCleanup('system-logs', () => {
        _slInitSeq++;
        if (typeof kiaraPolling !== 'undefined' && kiaraPolling) {
            kiaraPolling.stop('system-logs:refresh');
        }
        if (slFilterBar) {
            slFilterBar.destroy();
            slFilterBar = null;
        }
        slTable = null;
        if (SharedUi && SharedUi.table && SharedUi.table.instances) {
            delete SharedUi.table.instances['system-logs'];
        }
    });

    const container = document.getElementById('contentContainer');
    const filterBarEl = document.getElementById('filterBar');
    if (!container) return;

    _slDomGeneration++;
    const tableContainerId = `syslogTableContainer-${_slDomGeneration}`;
    container.innerHTML = `<div id="${tableContainerId}"></div>`;

    if (filterBarEl) {
        filterBarEl.innerHTML = '';
        filterBarEl.classList.remove('u-hidden');
        filterBarEl.classList.add('u-flex');
    }

    slFilterBar = new SystemLogFilterBar('filterBar', {
        filters: [
            { key: 'range', label: 'Zeitraum', type: 'select', default: '24h', options: [
                { value: '1h', label: 'Letzte Stunde' },
                { value: '24h', label: 'Letzte 24h' },
                { value: '7d', label: 'Letzte 7 Tage' },
                { value: '', label: 'Alle' },
            ]},
            { key: 'min_level', label: 'Level', type: 'select', default: 'WARNING', options: [
                { value: 'INFO', label: 'INFO+' },
                { value: 'WARNING', label: 'WARNING+' },
                { value: 'ERROR', label: 'ERROR+' },
                { value: 'CRITICAL', label: 'CRITICAL' },
            ]},
            {
                key: 'reason', label: 'Reason', type: 'select-async',
                options: [{ value: '', label: 'Alle' }],
                load: async () => {
                    try {
                        const resp = await SharedUi.fetch.safeFetch('/api/admin/system-events/distinct-reasons');
                        if (!resp.ok) return [];
                        const data = await resp.json();
                        return Array.isArray(data.reasons) ? data.reasons : [];
                    } catch (_e) { return []; }
                },
            },
            { key: 'search', label: 'Suche', type: 'search', placeholder: 'message enthaelt ...' },
            { key: 'hideSelfAudit', label: '', type: 'toggle',
              default: true, toggleLabel: 'SYSLOG_VIEW ausblenden' },
            { key: 'autoRefresh', label: '', type: 'toggle',
              default: false, toggleLabel: 'Auto-Refresh (15s)' },
        ],
        onChange: (values) => {
            if (mySeq !== _slInitSeq) return;
            if (!slTable) return;
            // Manueller Filterwechsel → Seite 1
            slTable.pagination.currentPage = 1;
            slTable.loadData(mapFilterValuesToApi(values));
            _slUpdateAutoRefresh(values, mySeq);
        },
    });
    slFilterBar.render();

    const config = {
        id: 'system-logs',
        apiEndpoint: '/api/admin/system-events',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 170px 90px 110px 150px 130px 120px 1fr',
        defaultSort: { field: 'ts', direction: 'desc' },
        pagination: {
            enabled: true,
            defaultPageSize: 100,
            options: [50, 100, 200, 500],
        },
        columns: [
            {
                field: 'ts', label: 'Zeitstempel', sortable: true, align: 'left',
                renderer: (v) => {
                    if (!v) return '<span class="u-c-secondary">–</span>';
                    const d = new Date(v.includes('T') ? v : v.replace(' ', 'T'));
                    if (isNaN(d.getTime())) return SharedUi.runtime.escapeHtml(v);
                    const pad = n => String(n).padStart(2, '0');
                    return `<span class="u-fs-12">${pad(d.getDate())}.${pad(d.getMonth()+1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}</span>`;
                },
            },
            {
                field: 'level', label: 'Level', sortable: true, align: 'center',
                renderer: (v) => {
                    const cls = {
                        INFO: 'info', WARNING: 'warning', ERROR: 'danger', CRITICAL: 'danger',
                    }[String(v || '').toUpperCase()] || 'default';
                    return `<span class="badge badge-${cls}">${SharedUi.runtime.escapeHtml(String(v || ''))}</span>`;
                },
            },
            { field: 'event_type', label: 'Event', sortable: true, align: 'left' },
            { field: 'reason', label: 'Reason', sortable: true, align: 'left' },
            { field: 'user_name', label: 'User', sortable: true, align: 'left' },
            { field: 'ip', label: 'IP', sortable: true, align: 'left' },
            { field: 'message', label: 'Message', sortable: false, align: 'left', truncate: 80 },
        ],
        detailRenderer: (row) => {
            let html = '<div class="detail-fields">';
            const fields = ['id', 'ts', 'level', 'source', 'event_type', 'reason', 'user_name', 'ip'];
            for (const f of fields) {
                if (row[f] !== undefined && row[f] !== null) {
                    html += `<div class="detail-field"><span class="detail-label">${SharedUi.runtime.escapeHtml(f)}</span><span class="detail-value">${SharedUi.runtime.escapeHtml(String(row[f]))}</span></div>`;
                }
            }
            html += '</div>';
            html += `<div class="detail-section"><div class="detail-section-label">message</div><pre class="detail-longtext">${SharedUi.runtime.escapeHtml(String(row.message || ''))}</pre></div>`;
            if (row.details && typeof row.details === 'object') {
                html += `<div class="detail-section"><div class="detail-section-label">details</div>${renderJsonDetail(row.details)}</div>`;
            }
            return html;
        },
        detailFields: [],
        headerActions: [
            {
                label: 'CSV-Export',
                onClick: () => {
                    const params = mapFilterValuesToApi(slFilterBar.getValues());
                    const qs = new URLSearchParams(params).toString();
                    window.open('/api/admin/system-events/export.csv' + (qs ? ('?' + qs) : ''), '_blank');
                },
            },
        ],
    };

    if (mySeq !== _slInitSeq) return;
    slTable = new SharedUi.table.ExpandableTable(config, tableContainerId);
    slTable.loadData(mapFilterValuesToApi(slFilterBar.getValues()));
    _slUpdateAutoRefresh(slFilterBar.getValues(), mySeq);
}

function _slUpdateAutoRefresh(values, mySeq) {
    if (typeof kiaraPolling === 'undefined' || !kiaraPolling) return;
    if (values && values.autoRefresh) {
        kiaraPolling.start('system-logs:refresh', () => {
            if (mySeq !== _slInitSeq) {
                kiaraPolling.stop('system-logs:refresh');
                return;
            }
            if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'system-logs') {
                kiaraPolling.stop('system-logs:refresh');
                return;
            }
            if (!slTable) {
                kiaraPolling.stop('system-logs:refresh');
                return;
            }
            slTable.loadData(mapFilterValuesToApi(slFilterBar ? slFilterBar.getValues() : {}));
        }, 15000);
    } else {
        kiaraPolling.stop('system-logs:refresh');
    }
}
