/**
 * tab_system_traces_panels.js — Trace Explorer Panels, Detail & Export
 *
 * Ausgelagert aus tab_system_traces.js (Phase 2, Sprint 4).
 * Nutzt Modul-State und Request-Guards aus dem Hauptfile
 * (gleicher globaler Scope via klassische Script-Tags).
 */

/* -----------------------------------------------------------------------
   Wasserfall-Renderer
   ----------------------------------------------------------------------- */
function renderWaterfall(spans, traceId) {
    if (!Array.isArray(spans) || spans.length === 0) {
        return '<p class="text-muted">Keine Span-Daten vorhanden.</p>';
    }

    const byId = new Map();
    const childrenByParent = new Map();
    for (const span of spans) {
        byId.set(span.span_id, span);
        const parentKey = span.parent_span_id || '__root__';
        if (!childrenByParent.has(parentKey)) childrenByParent.set(parentKey, []);
        childrenByParent.get(parentKey).push(span);
    }

    const roots = spans
        .filter((span) => !span.parent_span_id || !byId.has(span.parent_span_id))
        .sort((a, b) => (a.start_time_ns || 0) - (b.start_time_ns || 0));

    const ordered = [];
    const walk = (span, depth) => {
        ordered.push({ span, depth });
        const children = (childrenByParent.get(span.span_id) || [])
            .slice()
            .sort((a, b) => (a.start_time_ns || 0) - (b.start_time_ns || 0));
        children.forEach((child) => walk(child, depth + 1));
    };

    if (roots.length) {
        roots.forEach((root) => walk(root, 0));
    } else {
        spans
            .slice()
            .sort((a, b) => (a.start_time_ns || 0) - (b.start_time_ns || 0))
            .forEach((span) => ordered.push({ span, depth: 0 }));
    }

    let traceStart = null;
    let traceEnd = null;
    for (const { span } of ordered) {
        const startNs = Number(span.start_time_ns);
        const endNs = Number(span.end_time_ns ?? span.start_time_ns);
        if (Number.isFinite(startNs)) {
            traceStart = traceStart === null ? startNs : Math.min(traceStart, startNs);
        }
        if (Number.isFinite(endNs)) {
            traceEnd = traceEnd === null ? endNs : Math.max(traceEnd, endNs);
        }
    }
    if (traceStart === null) traceStart = 0;
    if (traceEnd === null || traceEnd <= traceStart) traceEnd = traceStart + 1;
    const totalNs = Math.max(traceEnd - traceStart, 1);

    let html = `<div class="trace-waterfall" data-trace-id="${SharedUi.runtime.escapeAttr(traceId)}">`;
    html += `
        <div class="trace-waterfall-header">
            <span>Span</span>
            <span>Timeline</span>
            <span>Dauer</span>
            <span>Status</span>
        </div>`;

    for (const { span, depth } of ordered) {
        const startNs = Number(span.start_time_ns);
        const endNs = Number(span.end_time_ns ?? span.start_time_ns);
        const safeStart = Number.isFinite(startNs) ? startNs : traceStart;
        const safeEnd = Number.isFinite(endNs) ? Math.max(endNs, safeStart) : safeStart;
        const leftPct = Math.max(0, ((safeStart - traceStart) / totalNs) * 100);
        const rawWidthPct = ((safeEnd - safeStart) / totalNs) * 100;
        const widthPct = Math.max(0.8, Math.min(rawWidthPct || 0.8, 100 - leftPct));
        const durationText = span.duration_ms == null ? '\u2013' : `${Number(span.duration_ms).toFixed(1)} ms`;
        const statusClass = span.status === 'ERROR'
            ? 'trace-waterfall-bar-error'
            : 'trace-waterfall-bar-ok';

        html += `
            <div class="trace-waterfall-row">
                <div class="trace-waterfall-name"
                     data-depth-pad="${depth * 16}">
                    <code>${SharedUi.runtime.escapeHtml(span.name || '(ohne Name)')}</code>
                </div>
                <div class="trace-waterfall-timeline">
                    <div class="trace-waterfall-track">
                        <div class="trace-waterfall-bar ${statusClass}"
                             data-bar-left="${leftPct.toFixed(3)}"
                             data-bar-width="${widthPct.toFixed(3)}"
                             title="${SharedUi.runtime.escapeAttr(`${span.name || ''} | ${durationText}`)}"></div>
                    </div>
                </div>
                <div class="trace-waterfall-duration num">${SharedUi.runtime.escapeHtml(durationText)}</div>
                <div class="trace-waterfall-status">${SharedUi.runtime.escapeHtml(span.status || 'OK')}</div>
            </div>`;
    }

    html += '</div>';
    return html;
}

/* Wendet dynamische Styles (Waterfall-Bar-Positionen, Depth-Padding) nach
   innerHTML-Assign an — CSP-Strict-kompatibel ueber SharedUi.runtimeStyle. */
function _applyWaterfallStyles(rootEl) {
    if (!rootEl || !window.SharedUi || !SharedUi.runtimeStyle) return;
    rootEl.querySelectorAll('.trace-waterfall-name[data-depth-pad]').forEach((el) => {
        const pad = Number(el.dataset.depthPad || 0);
        SharedUi.runtimeStyle.applyStyle(el, { paddingLeft: pad + 'px' });
    });
    rootEl.querySelectorAll('.trace-waterfall-bar[data-bar-left]').forEach((el) => {
        const left = Number(el.dataset.barLeft || 0);
        const width = Number(el.dataset.barWidth || 0);
        SharedUi.runtimeStyle.applyStyle(el, {
            left: left.toFixed(3) + '%',
            width: width.toFixed(3) + '%',
        });
    });
}

/* -----------------------------------------------------------------------
   Detail-View (Expand)
   ----------------------------------------------------------------------- */
async function loadTraceDetail(traceId, detailEl) {
    const previousController = _traceDetailElementControllers.get(detailEl);
    if (previousController) previousController.abort();
    const controller = new AbortController();
    _traceDetailElementControllers.set(detailEl, controller);
    _traceDetailAbortControllers.add(controller);
    const signal = controller.signal;

    detailEl.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/traces/${encodeURIComponent(traceId)}`, { signal });
        if (!resp.ok) {
            if (document.contains(detailEl)) {
                detailEl.innerHTML = '<div class="table-error">Fehler beim Laden des Trace-Details</div>';
            }
            return;
        }

        const data = await resp.json();
        // DOM-Liveness: nur rendern wenn Element noch im DOM lebt
        if (document.contains(detailEl) && adminCurrentSubTab === 'system-traces') {
            detailEl.innerHTML =
                '<div class="detail-content">'
                + renderWaterfall(data.spans, traceId)
                + _renderTraceDetailExport(traceId)
                + '</div>';
            _applyWaterfallStyles(detailEl);
        }
    } catch (error) {
        if (error.name === 'AbortError') return;
        if (document.contains(detailEl)) {
            detailEl.innerHTML = '<div class="table-error">Fehler beim Laden des Trace-Details</div>';
        }
        SharedUi.notify.show(error.message || 'Trace-Detail konnte nicht geladen werden', 'error');
    } finally {
        _traceDetailAbortControllers.delete(controller);
        if (_traceDetailElementControllers.get(detailEl) === controller) {
            _traceDetailElementControllers.delete(detailEl);
        }
    }
}

function _renderTraceDetailExport(traceId) {
    return `<button class="btn-sm btn-outline u-mt-8"
                data-trace-export="${SharedUi.runtime.escapeAttr(traceId)}">
        JSON exportieren
    </button>`;
}

/* -----------------------------------------------------------------------
   Aggregation: Histogramm + Span-Durchschnitte
   ----------------------------------------------------------------------- */
function _renderAggregationPanel() {
    return `
    <div class="trace-aggregation-panel u-mt-20">
        <div class="u-flex-gap-8 u-mb-12">
            <button class="btn-sm btn-primary" data-action="load-duration-histogram">
                Dauer-Histogramm
            </button>
            <button class="btn-sm btn-primary" data-action="load-span-averages">
                Span-Durchschnitte
            </button>
        </div>
        <div id="traceAggregationContent"></div>
    </div>`;
}

async function loadDurationHistogram() {
    const container = document.getElementById('traceAggregationContent');
    if (!container) return;

    const { requestSeq, signal } = _beginTracePanelRequest();
    _destroyTraceHistogram();
    container.innerHTML = '<canvas id="traceHistogramCanvas" height="250"></canvas>'
        + '<div id="traceHistogramStats" class="trace-stats-cards"></div>';

    try {
        const params = _collectTraceFilters();

        const resp = await SharedUi.fetch.safeFetch(`/api/traces/stats/duration-histogram?${params}`, { signal });
        if (!_isCurrentTracePanelRequest(requestSeq)) return;
        if (!resp.ok) {
            container.innerHTML = '<p class="text-muted">Fehler beim Laden des Histogramms.</p>';
            return;
        }
        const data = await resp.json();
        if (!_isCurrentTracePanelRequest(requestSeq)) return;

        if (data.total === 0) {
            container.innerHTML = '<p class="text-muted">Keine Traces im gewaehlten Zeitraum.</p>';
            return;
        }

        // Stats-Karten
        const statsEl = document.getElementById('traceHistogramStats');
        statsEl.innerHTML = `
            <div class="stat-card"><div class="stat-value">${data.total}</div>
                <div class="stat-label">Traces</div></div>
            <div class="stat-card"><div class="stat-value">${data.avg_ms} ms</div>
                <div class="stat-label">Durchschnitt</div></div>
            <div class="stat-card"><div class="stat-value">${data.p50_ms} ms</div>
                <div class="stat-label">Median (P50)</div></div>
            <div class="stat-card"><div class="stat-value">${data.p95_ms} ms</div>
                <div class="stat-label">P95</div></div>
            <div class="stat-card"><div class="stat-value">${data.p99_ms} ms</div>
                <div class="stat-label">P99</div></div>`;

        // Chart
        if (_traceHistogramChart) _traceHistogramChart.destroy();
        if (!Array.isArray(data.buckets) || data.buckets.length === 0) {
            container.innerHTML = '<p class="text-muted">Keine Histogramm-Daten verfuegbar.</p>';
            return;
        }
        const ctx = document.getElementById('traceHistogramCanvas').getContext('2d');
        _traceHistogramChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.buckets.map(b => `${b.from_ms}\u2013${b.to_ms}`),
                datasets: [{
                    label: 'Anzahl Traces',
                    data: data.buckets.map(b => b.count),
                    backgroundColor: 'rgba(52, 152, 219, 0.6)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Dauer (ms)' } },
                    y: { title: { display: true, text: 'Anzahl' }, beginAtZero: true },
                },
            },
        });
    } catch (error) {
        if (error.name === 'AbortError') return;
        if (!_isCurrentTracePanelRequest(requestSeq)) return;
        container.innerHTML = '<p class="text-muted">Fehler beim Laden des Histogramms.</p>';
        SharedUi.notify.show(error.message || 'Histogramm konnte nicht geladen werden', 'error');
    }
}

async function loadSpanAverages() {
    const container = document.getElementById('traceAggregationContent');
    if (!container) return;

    const { requestSeq, signal } = _beginTracePanelRequest();
    _destroyTraceHistogram();
    try {
        const params = _collectTraceFilters();

        const resp = await SharedUi.fetch.safeFetch(`/api/traces/stats/span-averages?${params}`, { signal });
        if (!_isCurrentTracePanelRequest(requestSeq)) return;
        if (!resp.ok) {
            container.innerHTML = '<p class="text-muted">Fehler beim Laden der Span-Daten.</p>';
            return;
        }
        const data = await resp.json();
        if (!_isCurrentTracePanelRequest(requestSeq)) return;

        if (!data.spans.length) {
            container.innerHTML = '<p class="text-muted">Keine Span-Daten vorhanden.</p>';
            return;
        }

        let html = '<table class="data-table"><thead><tr>'
            + '<th>Span</th><th>Anzahl</th><th>Avg (ms)</th>'
            + '<th>Min</th><th>Max</th><th>P95</th>'
            + '</tr></thead><tbody>';
        for (const s of data.spans) {
            html += `<tr>
                <td><code>${SharedUi.runtime.escapeHtml(s.name)}</code></td>
                <td class="num">${s.count}</td>
                <td class="num">${s.avg_ms}</td>
                <td class="num">${s.min_ms}</td>
                <td class="num">${s.max_ms}</td>
                <td class="num">${s.p95_ms}</td>
            </tr>`;
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        if (error.name === 'AbortError') return;
        if (!_isCurrentTracePanelRequest(requestSeq)) return;
        container.innerHTML = '<p class="text-muted">Fehler beim Laden der Span-Daten.</p>';
        SharedUi.notify.show(error.message || 'Span-Daten konnten nicht geladen werden', 'error');
    }
}

/* -----------------------------------------------------------------------
   Vergleich
   ----------------------------------------------------------------------- */
async function openTraceComparison() {
    if (_selectedTraces.size !== 2) {
        SharedUi.notify.show('Bitte genau zwei Traces auswaehlen.', 'error');
        return;
    }
    const { requestSeq, signal } = _beginTracePanelRequest();
    _destroyTraceHistogram();
    try {
        const [a, b] = [..._selectedTraces];
        const resp = await SharedUi.fetch.safeFetch(
            `/api/traces/compare?trace_a=${encodeURIComponent(a)}&trace_b=${encodeURIComponent(b)}`,
            { signal },
        );
        if (!_isCurrentTracePanelRequest(requestSeq)) return;
        if (!resp.ok) {
            SharedUi.notify.show('Vergleich fehlgeschlagen: ' + resp.statusText, 'error');
            return;
        }
        const data = await resp.json();
        if (!_isCurrentTracePanelRequest(requestSeq)) return;

        const container = document.getElementById('traceAggregationContent');
        if (!container) return;
        container.innerHTML = _renderComparisonView(data);
        _applyWaterfallStyles(container);
    } catch (error) {
        if (error.name === 'AbortError') return;
        if (!_isCurrentTracePanelRequest(requestSeq)) return;
        SharedUi.notify.show(error.message || 'Vergleich fehlgeschlagen', 'error');
    }
}

function _renderComparisonView(data) {
    let diffHtml = '<table class="data-table"><thead><tr>'
        + '<th>Span</th><th>A (ms)</th><th>A #</th>'
        + '<th>B (ms)</th><th>B #</th>'
        + '<th>Delta</th><th>Delta %</th></tr></thead><tbody>';
    for (const d of data.diff) {
        const cls = d.delta_ms > 0 ? 'text-danger' : (d.delta_ms < 0 ? 'text-success' : '');
        diffHtml += `<tr>
            <td><code>${SharedUi.runtime.escapeHtml(d.span_name)}</code></td>
            <td class="num">${d.a_ms ?? '\u2013'}</td>
            <td class="num">${d.a_count || '\u2013'}</td>
            <td class="num">${d.b_ms ?? '\u2013'}</td>
            <td class="num">${d.b_count || '\u2013'}</td>
            <td class="num ${cls}">${d.delta_ms > 0 ? '+' : ''}${d.delta_ms}</td>
            <td class="num ${cls}">${d.delta_pct != null ? d.delta_pct + '%' : '\u2013'}</td>
        </tr>`;
    }
    diffHtml += '</tbody></table>';

    return `
    <h4>Trace-Vergleich</h4>
    <div class="trace-compare-header">
        <span>A: ${SharedUi.runtime.escapeHtml(data.trace_a.root_span)} \u2014 ${SharedUi.runtime.escapeHtml(String(data.trace_a.duration_ms ?? '\u2013'))} ms</span>
        <span>B: ${SharedUi.runtime.escapeHtml(data.trace_b.root_span)} \u2014 ${SharedUi.runtime.escapeHtml(String(data.trace_b.duration_ms ?? '\u2013'))} ms</span>
    </div>
    <div class="trace-compare-grid">
        <div class="trace-compare-side">
            ${renderWaterfall(data.trace_a.spans, data.trace_a.trace_id)}
        </div>
        <div class="trace-compare-side">
            ${renderWaterfall(data.trace_b.spans, data.trace_b.trace_id)}
        </div>
    </div>
    <h4 class="u-mt-16">Span-Differenz</h4>
    ${diffHtml}`;
}

/* -----------------------------------------------------------------------
   Export
   ----------------------------------------------------------------------- */
function _renderExportButtons() {
    return `
    <div class="u-flex-gap-8 u-ml-auto">
        <button class="btn-sm btn-outline" data-action="export-traces" data-format="csv">
            CSV-Export
        </button>
        <button class="btn-sm btn-outline" data-action="export-traces" data-format="json">
            JSON-Export
        </button>
    </div>`;
}

function exportTraces(fmt) {
    if (_traceTableLoadInFlight) {
        SharedUi.notify.show('Bitte warten, bis die Tabelle mit den aktuellen Filtern geladen ist.', 'error');
        return;
    }
    if (!_traceTableHasFreshData) {
        SharedUi.notify.show('Export ist erst nach erfolgreichem Laden der aktuellen Tabellenansicht verfuegbar.', 'error');
        return;
    }
    const totalItems = traceTable?.pagination?.totalItems || 0;
    if (!totalItems) {
        SharedUi.notify.show('Keine Traces zum Exportieren.', 'error');
        return;
    }
    if (totalItems > 5000) {
        SharedUi.notify.show(
            `Export ist auf 5000 Traces begrenzt; aktuelle Filter matchen ${totalItems}. Bitte Filter weiter einschraenken.`,
            'error',
        );
        return;
    }

    const params = new URLSearchParams(_lastLoadedTraceFilters.toString());
    params.set('fmt', fmt);
    params.set('limit', String(totalItems));
    if (traceTable?.sortColumn) params.set('sort_field', traceTable.sortColumn);
    if (traceTable?.sortDirection) params.set('sort_dir', traceTable.sortDirection);

    window.open(`/api/traces/export?${params}`, '_blank');
}
