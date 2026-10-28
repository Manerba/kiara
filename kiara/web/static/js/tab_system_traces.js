/**
 * tab_system_traces.js — Trace Explorer (Sprint 17.6)
 *
 * Filter, paginierte Tabelle, Wasserfall-Renderer, Vergleichsmodus,
 * Aggregations-Panel (Histogramm + Span-Durchschnitte), Export.
 */

/* -----------------------------------------------------------------------
   Modul-State
   ----------------------------------------------------------------------- */
let traceTable = null;
let _selectedTraces = new Set();
let _traceHistogramChart = null;
let _traceDomGeneration = 0;
let _traceTableContainerId = null;
let _traceTableRequestSeq = 0;
let _traceTableAbortController = null;
let _tracePanelAbortController = null;
let _tracePanelRequestSeq = 0;
let _traceDetailAbortControllers = new Set();
let _traceDetailElementControllers = new WeakMap();
let _traceTableLoadInFlight = false;
let _traceTableHasFreshData = false;
let _lastLoadedTraceFilters = new URLSearchParams();
let _traceDiagnostics = null;

/* -----------------------------------------------------------------------
   Event-Delegation (CSP-Strict, V30 Welle 2)
   Geteilt mit tab_system_traces_panels.js (gleicher globaler Scope).
   ----------------------------------------------------------------------- */
const _TRACES_ACTIONS = new Set([
    'open-trace-comparison',
    'load-duration-histogram',
    'load-span-averages',
    'export-traces',
]);

const _TRACES_CHANGE_ACTIONS = new Set([
    'reload-traces-filter',
    'tracing-toggle',
]);

function _handleTracesClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_TRACES_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'open-trace-comparison': openTraceComparison(); break;
        case 'load-duration-histogram': loadDurationHistogram(); break;
        case 'load-span-averages': loadSpanAverages(); break;
        case 'export-traces': exportTraces(el.dataset.format || ''); break;
    }
}

function _handleTracesChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_TRACES_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    switch (el.dataset.changeAction) {
        case 'reload-traces-filter': loadTracesFiltered(); break;
        case 'tracing-toggle': _onTracingToggleChange(el.checked); break;
    }
}

function _bindTracesActions() {
    if (document.body._tracesActionsBound) return;
    document.body._tracesActionsBound = true;
    document.body.addEventListener('click', _handleTracesClick);
    document.body.addEventListener('change', _handleTracesChange);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-traces-actions-bound', 'true');
}

/* -----------------------------------------------------------------------
   Lifecycle-Helfer
   ----------------------------------------------------------------------- */
function _destroyTraceHistogram() {
    if (_traceHistogramChart) {
        _traceHistogramChart.destroy();
        _traceHistogramChart = null;
    }
}

function _abortTracePanelRequests() {
    _tracePanelRequestSeq += 1;
    if (_tracePanelAbortController) {
        _tracePanelAbortController.abort();
        _tracePanelAbortController = null;
    }
}

function _abortTraceTableRequests() {
    _traceTableRequestSeq += 1;
    if (_traceTableAbortController) {
        _traceTableAbortController.abort();
        _traceTableAbortController = null;
    }
}

function _abortTraceDetailRequests() {
    _traceDetailAbortControllers.forEach(controller => controller.abort());
    _traceDetailAbortControllers.clear();
    _traceDetailElementControllers = new WeakMap();
}

function _beginTraceTableRequest() {
    _abortTraceTableRequests();
    _traceTableAbortController = new AbortController();
    return {
        requestSeq: _traceTableRequestSeq,
        signal: _traceTableAbortController.signal,
    };
}

function _isCurrentTraceTableRequest(requestSeq) {
    return requestSeq === _traceTableRequestSeq && adminCurrentSubTab === 'system-traces';
}

function _beginTracePanelRequest() {
    _abortTracePanelRequests();
    _tracePanelAbortController = new AbortController();
    return {
        requestSeq: _tracePanelRequestSeq,
        signal: _tracePanelAbortController.signal,
    };
}

function _isCurrentTracePanelRequest(requestSeq) {
    return requestSeq === _tracePanelRequestSeq && adminCurrentSubTab === 'system-traces';
}

function _disposeTraceExplorerState() {
    _abortTraceTableRequests();
    _abortTracePanelRequests();
    _abortTraceDetailRequests();
    _destroyTraceHistogram();
    _clearTraceSelection();
    traceTable = null;
    _traceTableContainerId = null;
    _traceTableLoadInFlight = false;
    _traceTableHasFreshData = false;
    _lastLoadedTraceFilters = new URLSearchParams();
    _traceDiagnostics = null;
    if (SharedUi.table && SharedUi.table.instances) delete SharedUi.table.instances['system-traces'];
}

function _traceFilterParamsToSearchParams(params = {}) {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null && value !== '') {
            searchParams.set(key, String(value));
        }
    }
    return searchParams;
}

/* -----------------------------------------------------------------------
   ExpandableTable Load-Guards (Instanz-Override)
   ----------------------------------------------------------------------- */
function _installTraceTableLoadGuards(table) {
    table.loadData = async function(params = {}) {
        this.showLoading();
        const { requestSeq, signal } = _beginTraceTableRequest();
        _traceTableLoadInFlight = true;
        _traceTableHasFreshData = false;
        _traceDiagnostics = null;
        try {
            const queryParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                if (value !== undefined && value !== '') queryParams.append(key, value);
            }
            if (this.pagination.enabled) {
                queryParams.set('page', this.pagination.currentPage);
                queryParams.set('page_size', this.pagination.pageSize);
                if (this.sortColumn) {
                    queryParams.set('sort_field', this.sortColumn);
                    queryParams.set('sort_dir', this.sortDirection);
                }
            }
            this.lastParams = { ...params };

            const url = queryParams.toString()
                ? `${this.config.apiEndpoint}?${queryParams}`
                : this.config.apiEndpoint;

            const response = await SharedUi.fetch.safeFetch(url, { signal });
            if (!_isCurrentTraceTableRequest(requestSeq)) return false;
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }
            const result = await response.json();
            if (!_isCurrentTraceTableRequest(requestSeq)) return false;

            this.data = Array.isArray(result) ? result : (result.data || result.items || []);
            if (this.pagination.enabled && result.pagination) {
                this.pagination.currentPage = result.pagination.page;
                this.pagination.pageSize = result.pagination.page_size;
                this.pagination.totalItems = result.pagination.total_items;
                this.pagination.totalPages = result.pagination.total_pages;
            }

            _traceTableHasFreshData = true;
            _lastLoadedTraceFilters = _traceFilterParamsToSearchParams(this.lastParams || {});
            if (this.options.onDataLoaded) this.options.onDataLoaded(result);
            this.applyFiltersAndRender();
            _renderTraceEmptyState(requestSeq);
            return true;
        } catch (error) {
            if (error.name === 'AbortError') return false;
            if (!_isCurrentTraceTableRequest(requestSeq)) return false;
            _traceDiagnostics = null;
            console.error('Fehler beim Laden:', error);
            this.showError('Fehler beim Laden der Daten');
            return false;
        } finally {
            if (_isCurrentTraceTableRequest(requestSeq)) {
                _traceTableLoadInFlight = false;
                _traceTableAbortController = null;
            }
        }
    };

    table.silentRefresh = async function() {
        const params = this.lastParams || {};
        const { requestSeq, signal } = _beginTraceTableRequest();
        _traceTableLoadInFlight = true;
        _traceTableHasFreshData = false;
        _traceDiagnostics = null;
        try {
            const queryParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                if (value !== undefined && value !== '') queryParams.append(key, value);
            }
            if (this.pagination.enabled) {
                queryParams.set('page', this.pagination.currentPage);
                queryParams.set('page_size', this.pagination.pageSize);
                if (this.sortColumn) {
                    queryParams.set('sort_field', this.sortColumn);
                    queryParams.set('sort_dir', this.sortDirection);
                }
            }

            const url = queryParams.toString()
                ? `${this.config.apiEndpoint}?${queryParams}`
                : this.config.apiEndpoint;

            const response = await SharedUi.fetch.safeFetch(url, { signal });
            if (!_isCurrentTraceTableRequest(requestSeq)) return false;
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }
            const result = await response.json();
            if (!_isCurrentTraceTableRequest(requestSeq)) return false;

            this.data = Array.isArray(result) ? result : (result.data || result.items || []);
            if (this.pagination.enabled && result.pagination) {
                this.pagination.currentPage = result.pagination.page;
                this.pagination.pageSize = result.pagination.page_size;
                this.pagination.totalItems = result.pagination.total_items;
                this.pagination.totalPages = result.pagination.total_pages;
            }

            _traceTableHasFreshData = true;
            _lastLoadedTraceFilters = _traceFilterParamsToSearchParams(this.lastParams || {});
            if (this.options.onDataLoaded) this.options.onDataLoaded(result);
            this._silentMode = true;
            this.applyFiltersAndRender();
            _renderTraceEmptyState(requestSeq);
            this._silentMode = false;
            return true;
        } catch (error) {
            this._silentMode = false;
            if (error.name === 'AbortError') return false;
            if (!_isCurrentTraceTableRequest(requestSeq)) return false;
            _traceDiagnostics = null;
            throw error;
        } finally {
            if (_isCurrentTraceTableRequest(requestSeq)) {
                _traceTableLoadInFlight = false;
                _traceTableAbortController = null;
            }
        }
    };
}

/* -----------------------------------------------------------------------
   Formatierung
   ----------------------------------------------------------------------- */
function _formatTraceTimestamp(value) {
    if (!value) return '<span class="text-muted">-</span>';
    const d = new Date(String(value).includes('T') ? value : String(value).replace(' ', 'T'));
    if (Number.isNaN(d.getTime())) return SharedUi.runtime.escapeHtml(String(value));
    const pad = (n) => String(n).padStart(2, '0');
    return `<span class="u-fs-12">${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}</span>`;
}

function _renderTraceAttributes(value) {
    const text = JSON.stringify(value || {});
    const shortText = text.length > 80 ? text.slice(0, 77) + '...' : text;
    return `<code title="${SharedUi.runtime.escapeAttr(text)}">${SharedUi.runtime.escapeHtml(shortText)}</code>`;
}

/* -----------------------------------------------------------------------
   ExpandableTable-Config
   ----------------------------------------------------------------------- */
const traceTableConfig = {
    id: 'system-traces',
    apiEndpoint: '/api/traces',
    showHeader: true,
    expandable: true,
    pagination: {
        enabled: true,
        defaultPageSize: 100,
        options: [50, 100, 200, 500],
    },
    gridTemplate: '30px 40px 200px 180px 140px 110px 80px 90px 1.3fr',
    defaultSort: { field: 'start_time', direction: 'desc' },
    columns: [
        {
            field: '_select',
            label: '',
            renderer: (val, row) => `<input type="checkbox"
                data-trace-select="${SharedUi.runtime.escapeAttr(row.trace_id)}"
                ${_selectedTraces.has(row.trace_id) ? 'checked' : ''}>`,
        },
        { field: 'trace_id', label: 'Trace ID', sortable: true, truncate: 24 },
        { field: 'root_span', label: 'Root-Span', sortable: true, truncate: 36 },
        {
            field: 'start_time',
            label: 'Start',
            sortable: true,
            renderer: (v) => _formatTraceTimestamp(v),
        },
        {
            field: 'duration_ms',
            label: 'Dauer (ms)',
            sortable: true,
            align: 'right',
            renderer: (v) => v == null ? '\u2013' : `${Number(v).toFixed(1)} ms`,
        },
        { field: 'span_count', label: 'Spans', sortable: true, align: 'right' },
        {
            field: 'status',
            label: 'Status',
            sortable: true,
            align: 'center',
            renderer: (v) => {
                if (v === 'ERROR') return '<span class="badge badge-danger">ERROR</span>';
                if (v === 'OK') return '<span class="badge badge-success">OK</span>';
                return '<span class="badge badge-default">UNSET</span>';
            },
        },
        {
            field: 'attributes',
            label: 'Attribute',
            sortable: false,
            renderer: (v) => _renderTraceAttributes(v),
        },
    ],
    detailFields: [],
};

/* -----------------------------------------------------------------------
   Filter
   ----------------------------------------------------------------------- */
function _initTraceFilters() {
    const filterBar = document.getElementById('filterBar');
    if (!filterBar) return;
    filterBar.innerHTML = '';
    filterBar.classList.remove('u-hidden');
    filterBar.classList.add('u-flex');

    // 1. Status-Filter
    const statusGroup = document.createElement('div');
    statusGroup.className = 'filter-group';
    statusGroup.innerHTML = `<span class="filter-label">Status:</span>
        <select class="filter-input" id="traceStatusFilter"
                data-change-action="reload-traces-filter">
            <option value="">Alle</option>
            <option value="UNSET">UNSET</option>
            <option value="OK">OK</option>
            <option value="ERROR">Fehler</option>
        </select>`;
    filterBar.appendChild(statusGroup);

    // 2. Root-Span-Filter
    const typeGroup = document.createElement('div');
    typeGroup.className = 'filter-group';
    typeGroup.innerHTML = `<span class="filter-label">Root-Span:</span>
        <input type="text" class="filter-input u-width-190" id="traceTypeFilter"
               placeholder="z.B. rag_query oder hybrid_retrieve"
               data-change-action="reload-traces-filter">`;
    filterBar.appendChild(typeGroup);

    // 3. Zeitraum Von/Bis
    const dateGroup = document.createElement('div');
    dateGroup.className = 'filter-group';
    dateGroup.innerHTML = `<span class="filter-label">Zeitraum:</span>
        <input type="date" class="filter-input" id="traceFromDate"
               data-change-action="reload-traces-filter">
        <span class="u-mx-4">\u2013</span>
        <input type="date" class="filter-input" id="traceToDate"
               data-change-action="reload-traces-filter">`;
    filterBar.appendChild(dateGroup);

    // 4. Min-Dauer (ms)
    const durGroup = document.createElement('div');
    durGroup.className = 'filter-group';
    durGroup.innerHTML = `<span class="filter-label">Min. Dauer:</span>
        <input type="number" class="filter-input u-width-80" id="traceMinDuration"
               placeholder="ms"
               data-change-action="reload-traces-filter">`;
    filterBar.appendChild(durGroup);

    // 5. Span-Name-Suche
    const spanGroup = document.createElement('div');
    spanGroup.className = 'filter-group';
    spanGroup.innerHTML = `<span class="filter-label">Span:</span>
        <input type="text" class="filter-input u-width-140" id="traceSpanFilter"
               placeholder="z.B. graph_retrieve"
               data-change-action="reload-traces-filter">`;
    filterBar.appendChild(spanGroup);

    // 6. Attribut-Suche
    const attrGroup = document.createElement('div');
    attrGroup.className = 'filter-group';
    attrGroup.innerHTML = `<span class="filter-label">Attribut:</span>
        <input type="text" class="filter-input u-width-220" id="traceAttrFilter"
               placeholder="key=value, z.B. final_status=success"
               data-change-action="reload-traces-filter">`;
    filterBar.appendChild(attrGroup);
}

function _collectTraceFilters() {
    const params = new URLSearchParams();
    const status = document.getElementById('traceStatusFilter')?.value;
    const name = document.getElementById('traceTypeFilter')?.value;
    const fromDate = document.getElementById('traceFromDate')?.value;
    const toDate = document.getElementById('traceToDate')?.value;
    const minDur = document.getElementById('traceMinDuration')?.value;
    const spanName = document.getElementById('traceSpanFilter')?.value;
    const attrSearch = document.getElementById('traceAttrFilter')?.value;

    if (status) params.set('status', status);
    if (name) params.set('name', name);
    if (fromDate) params.set('from_date', fromDate);
    if (toDate) params.set('to_date', toDate);
    if (minDur) params.set('min_duration_ms', minDur);
    if (spanName) params.set('span_name', spanName);
    if (attrSearch) params.set('attr_search', attrSearch);
    return params;
}

async function loadTracesFiltered() {
    if (!traceTable) return;
    const params = Object.fromEntries(_collectTraceFilters().entries());

    _clearTraceSelection();
    traceTable.pagination.currentPage = 1;
    await traceTable.loadData(params);
}

/* -----------------------------------------------------------------------
   Selektion
   ----------------------------------------------------------------------- */
function _clearTraceSelection() {
    _selectedTraces.clear();
    const btn = document.getElementById('compareBtn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Vergleichen (0/2)';
    }
}

function toggleTraceSelection(traceId, checkbox) {
    if (checkbox.checked) {
        if (!_selectedTraces.has(traceId) && _selectedTraces.size >= 2) {
            checkbox.checked = false;
            SharedUi.notify.show('Bitte genau zwei Traces auswaehlen.', 'error');
            return;
        }
        _selectedTraces.add(traceId);
    } else {
        _selectedTraces.delete(traceId);
    }
    const btn = document.getElementById('compareBtn');
    if (btn) {
        btn.disabled = _selectedTraces.size !== 2;
        btn.textContent = `Vergleichen (${_selectedTraces.size}/2)`;
    }
}

/* -----------------------------------------------------------------------
   Diagnostics-Validierung
   ----------------------------------------------------------------------- */
function _isValidTraceDiagnostics(diag) {
    if (!diag || typeof diag !== 'object') return false;
    var keys = ['tracing_enabled', 'trace_span_count', 'trace_root_count',
                'query_log_count', 'query_log_trace_id_count'];
    for (var i = 0; i < keys.length; i++) {
        if (!(keys[i] in diag)) return false;
    }
    if (diag.tracing_enabled !== true && diag.tracing_enabled !== false
        && diag.tracing_enabled !== null) return false;
    var countKeys = ['trace_span_count', 'trace_root_count',
                     'query_log_count', 'query_log_trace_id_count'];
    for (var j = 0; j < countKeys.length; j++) {
        if (typeof diag[countKeys[j]] !== 'number') return false;
    }
    return true;
}

/* -----------------------------------------------------------------------
   Trace-spezifischer Empty-State (AP2, Sprint 4)
   ----------------------------------------------------------------------- */
function _renderTraceEmptyState(requestSeq) {
    if (!_isCurrentTraceTableRequest(requestSeq)) return;
    if (!_traceDiagnostics) return;
    if (!_traceTableContainerId) return;
    const container = document.getElementById(_traceTableContainerId);
    if (!container) return;

    const emptyEl = container.querySelector('.table-no-data');
    if (!emptyEl) return;

    const d = _traceDiagnostics;
    const hasFilters = _lastLoadedTraceFilters.toString() !== '';
    let html = '';

    if (hasFilters && d.trace_root_count > 0) {
        html = '<p>Die aktiven Filter ergeben keine Treffer. Es sind Traces vorhanden \u2014 bitte Filter anpassen.</p>';
    } else if (d.tracing_enabled === false) {
        html = '<p><strong>Tracing ist deaktiviert.</strong></p>'
            + '<p>Zum Aktivieren in <code>config/kiara.yaml</code> setzen:</p>'
            + '<pre>tracing:\n  enabled: true\n  exporter: postgresql</pre>'
            + '<p>Nach Aktivierung und Neustart eine neue Chat-Session starten.</p>';
    } else if (d.tracing_enabled === null) {
        html = '<p><strong>Tracing-Status konnte nicht ermittelt werden.</strong></p>'
            + '<p>Bitte <code>config/kiara.yaml</code> pr\u00fcfen und sicherstellen, '
            + 'dass Tracing aktiviert ist (<code>tracing.enabled: true</code>, '
            + '<code>exporter: postgresql</code>).</p>';
    } else if (d.query_log_count > 0 && d.query_log_trace_id_count === 0 && d.trace_span_count === 0) {
        html = '<p>' + SharedUi.runtime.escapeHtml(String(d.query_log_count))
            + ' Chat-Logs vorhanden, aber keine davon mit Trace-ID.</p>'
            + '<p>Vorhandene Chat-Logs wurden ohne Tracing gespeichert und k\u00f6nnen '
            + 'nicht als Trace-Wasserfall angezeigt werden.</p>'
            + '<p>Starten Sie eine neue Chat-Session, um Traces zu erzeugen.</p>';
    } else if (d.query_log_count > 0 && d.query_log_trace_id_count > 0 && d.trace_span_count === 0) {
        html = '<p>' + SharedUi.runtime.escapeHtml(String(d.query_log_trace_id_count))
            + ' Chat-Logs mit Trace-ID vorhanden, aber keine persistierten Spans.</p>'
            + '<p>Bitte Tracing-Exporter und Persistenz-Pipeline pr\u00fcfen '
            + '(<code>tracing.exporter: postgresql</code> in <code>config/kiara.yaml</code>).</p>';
    } else if (d.trace_span_count > 0 && d.trace_root_count === 0) {
        html = '<p><strong>Datenkonsistenz-Hinweis:</strong> '
            + SharedUi.runtime.escapeHtml(String(d.trace_span_count))
            + ' Spans vorhanden, aber keine Root-Spans (parent_span_id IS NULL).</p>';
    } else {
        html = '<p>Keine Traces und keine Chat-Logs vorhanden.</p>'
            + '<p>Starten Sie eine neue Chat-Session, um Traces zu erzeugen.</p>';
    }

    emptyEl.innerHTML = '<div class="trace-empty-diagnostic">' + html + '</div>';
}

/* -----------------------------------------------------------------------
   Tracing-Toggle (Config ein-/ausschalten + Service-Restart)
   ----------------------------------------------------------------------- */
async function _loadTracingStatus() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/traces/status');
        if (adminCurrentSubTab !== 'system-traces') return;
        if (!resp.ok) return;
        const data = await resp.json();
        if (adminCurrentSubTab !== 'system-traces') return;
        const toggle = document.getElementById('tracingToggle');
        const label = document.getElementById('tracingToggleLabel');
        const text = document.getElementById('tracingToggleText');
        if (toggle) {
            toggle.checked = data.enabled;
        }
        if (label) {
            label.classList.remove('u-toggle-disabled');
        }
        if (text) text.textContent = data.enabled ? 'Tracing aktiv' : 'Tracing inaktiv';
    } catch (e) {
        console.warn('Tracing-Status konnte nicht geladen werden:', e);
    }
}

async function _onTracingToggleChange(checked) {
    const toggle = document.getElementById('tracingToggle');
    const label = document.getElementById('tracingToggleLabel');
    const text = document.getElementById('tracingToggleText');

    // Deaktivieren waehrend Request
    if (label) { label.classList.add('u-toggle-disabled'); }
    if (text) text.textContent = 'Wird umgeschaltet\u2026';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/traces/toggle', { method: 'POST' });
        if (adminCurrentSubTab !== 'system-traces') return;
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        if (toggle) toggle.checked = data.enabled;
        if (text) text.textContent = data.enabled ? 'Tracing aktiv' : 'Tracing inaktiv';

        if (data.service_restarted) {
            SharedUi.notify.show(
                data.enabled ? 'Tracing aktiviert — Service neu gestartet' : 'Tracing deaktiviert — Service neu gestartet',
                'success'
            );
        } else if (data.error) {
            SharedUi.notify.show('Config gespeichert, aber Service-Restart fehlgeschlagen: ' + data.error, 'error');
        }

        // Tabelle neu laden damit Empty-State aktualisiert wird
        if (traceTable) await traceTable.loadData(traceTable.lastParams || {});
    } catch (e) {
        if (adminCurrentSubTab !== 'system-traces') return;
        if (toggle) toggle.checked = !checked;
        SharedUi.notify.show('Fehler beim Umschalten: ' + e.message, 'error');
    } finally {
        if (label) { label.classList.remove('u-toggle-disabled'); }
    }
}

/* -----------------------------------------------------------------------
   Init (Haupteinsprungpunkt)
   ----------------------------------------------------------------------- */
async function initSystemTracesTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    _bindTracesActions();

    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-traces');
    _disposeTraceExplorerState();
    _traceDomGeneration += 1;
    _traceTableContainerId = `traceTableContainer-${_traceDomGeneration}`;

    container.innerHTML = `
        <div id="traceExplorerLayout">
            <div id="traceToolbar" class="u-flex-center-gap-8 u-my-12"></div>
            <div id="${_traceTableContainerId}"></div>
            <div id="traceAggregationHost"></div>
        </div>`;

    _initTraceFilters();

    traceTable = new SharedUi.table.ExpandableTable(traceTableConfig, _traceTableContainerId, {
        onDataLoaded: (result) => {
            if (result && result.diagnostics
                && (!result.data || result.data.length === 0)
                && _isValidTraceDiagnostics(result.diagnostics)) {
                _traceDiagnostics = result.diagnostics;
            } else {
                _traceDiagnostics = null;
            }
        },
        onRowExpanded: async (traceId, detailEl) => {
            await loadTraceDetail(traceId, detailEl);
        },
        onAfterRestore: async (rowId, detailEl) => {
            // Rehydrate: Detail nach Re-Render neu laden statt leere Huelle
            if (adminCurrentSubTab === 'system-traces') {
                await loadTraceDetail(rowId, detailEl);
            }
        },
    });
    _installTraceTableLoadGuards(traceTable);

    // Event-Delegation: Checkbox-Selektion + Export auf stabilem Container
    const tableContainer = document.getElementById(_traceTableContainerId);
    if (tableContainer) {
        // Checkbox: change-Event. Row-Expand-Block wird seit 0.0.1+380 von
        // ui_core/table.js _interactiveCellMatch abgedeckt (kein Inline-
        // stopPropagation mehr noetig).
        tableContainer.addEventListener('change', (e) => {
            const checkbox = e.target.closest('[data-trace-select]');
            if (checkbox && checkbox.tagName === 'INPUT') {
                toggleTraceSelection(checkbox.dataset.traceSelect, checkbox);
            }
        });
        // Export-Button
        tableContainer.addEventListener('click', (e) => {
            const exportBtn = e.target.closest('[data-trace-export]');
            if (exportBtn) {
                e.stopPropagation();
                window.open('/api/traces/' + encodeURIComponent(exportBtn.dataset.traceExport) + '/export', '_blank');
            }
        });
    }

    // Tab-Cleanup registrieren
    registerTabCleanup('system-traces', () => {
        _disposeTraceExplorerState();
    });

    document.getElementById('traceToolbar').innerHTML = `
        <div class="u-flex-center-gap-6 u-mr-16">
            <label class="toggle-switch u-toggle-disabled" id="tracingToggleLabel">
                <input type="checkbox" id="tracingToggle" data-change-action="tracing-toggle">
                <span class="toggle-slider"></span>
            </label>
            <span id="tracingToggleText" class="u-text-sm-secondary">Tracing</span>
        </div>
        <button class="btn-sm btn-primary" id="compareBtn" disabled
                data-action="open-trace-comparison">
            Vergleichen (0/2)
        </button>
        ${_renderExportButtons()}`;
    _loadTracingStatus();
    document.getElementById('traceAggregationHost').innerHTML =
        _renderAggregationPanel();

    await loadTracesFiltered();
}
