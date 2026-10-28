/**
 * KIara Admin - Benchmark Tabs
 */

let _benchmarksInitSeq = 0;
let _benchmarksData = null;
let _benchmarksSelectedSummaryId = null;
let _beirInitSeq = 0;
let _beirData = null;
let _beirSelectedSummaryId = null;
let _beirStartInFlight = false;

const _BENCHMARK_ACTIONS = new Set([
    'refresh-crag-benchmarks',
    'select-crag-summary',
    'refresh-beir-benchmarks',
    'start-beir-run',
    'select-beir-summary',
]);

function _handleBenchmarkClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_BENCHMARK_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'refresh-crag-benchmarks':
            initBenchmarkCragTab();
            break;
        case 'select-crag-summary':
            _benchmarksSelectedSummaryId = el.dataset.summaryId || null;
            renderKiBenchmarksTab(document.getElementById('contentContainer'), _benchmarksData || {});
            break;
        case 'refresh-beir-benchmarks':
            initBenchmarkBeirTab();
            break;
        case 'start-beir-run':
            startBeirRun();
            break;
        case 'select-beir-summary':
            _beirSelectedSummaryId = el.dataset.summaryId || null;
            renderBenchmarkBeirTab(document.getElementById('contentContainer'), _beirData || {});
            break;
    }
}

function _bindBenchmarkActions() {
    if (document.body._benchmarkActionsBound) return;
    document.body._benchmarkActionsBound = true;
    document.body.addEventListener('click', _handleBenchmarkClick);
}

async function initBenchmarkCragTab() {
    _bindBenchmarkActions();
    if (typeof _runTabCleanups === 'function') _runTabCleanups('benchmark-crag');
    _benchmarksInitSeq++;
    const mySeq = _benchmarksInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.classList.remove('u-flex', 'u-visible', 'u-block');
        filterBar.classList.add('u-hidden');
    }

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    registerTabCleanup('benchmark-crag', () => { _benchmarksInitSeq++; });

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/benchmarks/crag');
        if (mySeq !== _benchmarksInitSeq) return;
        if (!resp.ok) throw new Error('Benchmark-Daten konnten nicht geladen werden');
        const data = await resp.json();
        if (mySeq !== _benchmarksInitSeq) return;
        _benchmarksData = data || {};
        if (!_benchmarksSelectedSummaryId && (_benchmarksData.summaries || []).length > 0) {
            _benchmarksSelectedSummaryId = _benchmarksData.summaries[0].id;
        }
        renderKiBenchmarksTab(container, _benchmarksData);
    } catch (e) {
        if (mySeq !== _benchmarksInitSeq) return;
        container.innerHTML = '<div class="no-data">Benchmark-Ergebnisse konnten nicht geladen werden.</div>';
    }
}

async function initKiBenchmarksTab() {
    return initBenchmarkCragTab();
}

function renderKiBenchmarksTab(container, data) {
    if (!container) return;
    const summaries = Array.isArray(data.summaries) ? data.summaries : [];
    const comparisons = Array.isArray(data.comparisons) ? data.comparisons : [];
    const selected = summaries.find(item => item.id === _benchmarksSelectedSummaryId) || summaries[0] || null;
    if (selected) _benchmarksSelectedSummaryId = selected.id;

    let html = '<div class="config-form">';
    html += '<div class="u-flex-center-gap-12 u-flex-wrap u-mb-16">';
    html += '<h3 class="u-m-0-mb-8">CRAG Benchmarks</h3>';
    html += '<button class="action-btn" data-action="refresh-crag-benchmarks">Aktualisieren</button>';
    html += '</div>';

    html += renderBenchmarkKpis(summaries, comparisons);
    html += renderBenchmarkSummaries(summaries, selected ? selected.id : null);
    html += renderBenchmarkComparison(comparisons);
    html += renderBenchmarkBreakdowns(selected);

    const errors = Array.isArray(data.errors) ? data.errors : [];
    if (errors.length > 0) {
        html += '<div class="config-section"><h4>Lesefehler</h4>';
        html += '<table class="data-table"><thead><tr><th>Datei</th><th>Fehler</th></tr></thead><tbody>';
        errors.forEach(err => {
            html += '<tr>';
            html += `<td class="u-mono u-fs-11">${esc(err.path || '')}</td>`;
            html += `<td>${esc(err.error || '')}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

const BENCHMARK_PLACEHOLDERS = {
    beir: {
        title: 'BEIR',
        status: 'Implementiert',
        subtitle: 'Corpus-Retrieval ueber etablierte IR-Datasets',
        description: 'BEIR liefert Korpora, Queries und Relevanzurteile fuer klassische Retrieval-Metriken wie Recall@k, nDCG@10 und MRR. KIara kann vorbereitete BEIR-Datenquellen ueber die Admin-UI benchmarken und historische Runs anzeigen.',
    },
    mteb: {
        title: 'MTEB',
        status: 'Noch nicht implementiert',
        subtitle: 'Breites Benchmark-Framework fuer Embedding-Modelle',
        description: 'MTEB bewertet Embeddings ueber viele Aufgabentypen, darunter Retrieval. In KIara waere dieser Tab sinnvoll fuer Modellvergleiche, weniger fuer einen vollstaendigen RAG-End-to-End-Test.',
    },
    miracl: {
        title: 'MIRACL',
        status: 'Noch nicht implementiert',
        subtitle: 'Mehrsprachiges Retrieval',
        description: 'MIRACL testet Retrieval in vielen Sprachen mit getrennten Korpora, Queries und Relevanzurteilen. Das passt gut, um KIaras deutsch- und mehrsprachige Retrieval-Qualitaet systematisch zu messen.',
    },
    msmarco: {
        title: 'MS MARCO',
        status: 'Noch nicht implementiert',
        subtitle: 'Web-Passage-Retrieval in grossem Massstab',
        description: 'MS MARCO ist ein grosser Passage-Retrieval-Benchmark mit realen Suchfragen. Fuer KIara waere er vor allem ein Stresstest fuer Ranking, Top-k-Verhalten und Skalierung.',
    },
    nqHotpotqa: {
        title: 'Natural Questions / HotpotQA',
        status: 'Noch nicht implementiert',
        subtitle: 'QA-orientiertes Retrieval',
        description: 'Natural Questions und HotpotQA testen, ob die richtigen Belege fuer konkrete Fragen gefunden werden. HotpotQA ist besonders interessant fuer Multi-Hop-Fragen, bei denen mehrere Textstellen kombiniert werden muessen.',
    },
};

function initBenchmarkBeirTab() {
    _bindBenchmarkActions();
    if (typeof _runTabCleanups === 'function') _runTabCleanups('benchmark-beir');
    _beirInitSeq++;
    const mySeq = _beirInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.classList.remove('u-flex', 'u-visible', 'u-block');
        filterBar.classList.add('u-hidden');
    }

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    registerTabCleanup('benchmark-beir', () => {
        _beirInitSeq++;
        kiaraPolling.stop('benchmark-beir:status');
    });

    loadBeirData(mySeq);
}

function initBenchmarkMtebTab() {
    renderBenchmarkPlaceholder('mteb');
}

function initBenchmarkMiraclTab() {
    renderBenchmarkPlaceholder('miracl');
}

function initBenchmarkMsMarcoTab() {
    renderBenchmarkPlaceholder('msmarco');
}

function initBenchmarkNqHotpotqaTab() {
    renderBenchmarkPlaceholder('nqHotpotqa');
}

function renderBenchmarkPlaceholder(key) {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.classList.remove('u-flex', 'u-visible', 'u-block');
        filterBar.classList.add('u-hidden');
    }

    const item = BENCHMARK_PLACEHOLDERS[key];
    if (!item) {
        container.innerHTML = '<div class="no-data">Benchmark nicht gefunden.</div>';
        return;
    }

    container.innerHTML = `
        <div class="config-form">
            <div class="config-section">
                <div class="u-flex-center-gap-12 u-flex-wrap u-mb-16">
                    <h3 class="u-m-0">${esc(item.title)}</h3>
                    <span class="badge badge-default">${esc(item.status || 'Noch nicht implementiert')}</span>
                </div>
                <h4>${esc(item.subtitle)}</h4>
                <p class="u-text-sm-secondary u-mb-0">${esc(item.description)}</p>
            </div>
        </div>
    `;
}

async function loadBeirData(mySeq) {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/benchmarks/beir');
        if (mySeq !== _beirInitSeq) return;
        if (!resp.ok) throw new Error('BEIR-Daten konnten nicht geladen werden');
        const data = await resp.json();
        if (mySeq !== _beirInitSeq) return;
        _beirData = data || {};
        if (!_beirSelectedSummaryId && (_beirData.summaries || []).length > 0) {
            _beirSelectedSummaryId = _beirData.summaries[0].id;
        }
        renderBenchmarkBeirTab(document.getElementById('contentContainer'), _beirData);
        if ((_beirData.task || {}).running) startBeirPolling(mySeq);
    } catch (e) {
        if (mySeq !== _beirInitSeq) return;
        const container = document.getElementById('contentContainer');
        if (container) container.innerHTML = '<div class="no-data">BEIR-Daten konnten nicht geladen werden.</div>';
    }
}

function renderBenchmarkBeirTab(container, data) {
    if (!container) return;
    const datasets = Array.isArray(data.datasets) ? data.datasets : [];
    const datasources = Array.isArray(data.datasources) ? data.datasources : [];
    const summaries = Array.isArray(data.summaries) ? data.summaries : [];
    const task = data.task || {};
    const selected = summaries.find(item => item.id === _beirSelectedSummaryId) || summaries[0] || null;
    if (selected) _beirSelectedSummaryId = selected.id;

    let html = '<div class="config-form">';
    html += '<div class="u-flex-center-gap-12 u-flex-wrap u-mb-16">';
    html += '<h3 class="u-m-0">BEIR Benchmarks</h3>';
    html += '<span class="badge badge-default">Persistentes Retrieval</span>';
    html += '<button class="action-btn" data-action="refresh-beir-benchmarks">Aktualisieren</button>';
    html += '</div>';

    html += renderBeirRunForm(datasets, datasources, task);
    html += renderBeirTaskStatus(task);
    html += renderBeirKpis(summaries);
    html += renderBeirHistory(summaries, selected ? selected.id : null);
    html += renderBeirSelectedDetails(selected);

    const errors = Array.isArray(data.errors) ? data.errors : [];
    if (errors.length > 0) {
        html += '<div class="config-section"><h4>Lesefehler</h4>';
        html += '<table class="data-table"><thead><tr><th>Quelle</th><th>Fehler</th></tr></thead><tbody>';
        errors.forEach(err => {
            html += '<tr>';
            html += `<td class="u-mono u-fs-11">${esc(err.path || '')}</td>`;
            html += `<td>${esc(err.error || '')}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

function renderBeirRunForm(datasets, datasources, task) {
    const datasetOptions = datasets.map(ds => (
        `<option value="${escAttr(ds.dataset_dir || '')}">${esc(ds.dataset || ds.dataset_dir || '')} (${fmtInt(ds.queries || 0)} Queries)</option>`
    )).join('');
    const datasourceOptions = datasources.map(ds => {
        const embeddings = Array.isArray(ds.embeddings) ? ds.embeddings : [];
        const chunkInfo = embeddings.map(item => `${item.embedding_model}: ${fmtInt(item.chunk_count)}`).join(', ');
        const label = `${ds.id} - ${ds.name || ds.path || ''}${ds.is_active ? '' : ' [deaktiviert]'}${chunkInfo ? ' - ' + chunkInfo : ''}`;
        return `<option value="${escAttr(ds.id)}">${esc(label)}</option>`;
    }).join('');
    const disabled = task.running || _beirStartInFlight ? ' disabled' : '';
    let html = '<div class="config-section">';
    html += '<h4>Neuen Lauf starten</h4>';
    html += '<div class="config-fields">';
    html += `<div class="config-field"><label>Dataset</label><select id="beir-dataset-dir" class="config-input"${disabled}>${datasetOptions || '<option value="">Kein Dataset vorbereitet</option>'}</select></div>`;
    html += `<div class="config-field"><label>Datenquelle</label><select id="beir-datasource-id" class="config-input"${disabled}>${datasourceOptions || '<option value="">Keine BEIR-Datenquelle gefunden</option>'}</select><span class="field-hint">Die Datenquelle darf deaktiviert sein; der Benchmark filtert explizit per Datasource-ID.</span></div>`;
    html += `<div class="config-field"><label>Retriever</label><select id="beir-retriever" class="config-input"${disabled}><option value="hybrid">Hybrid</option><option value="dense">Dense</option></select></div>`;
    html += `<div class="config-field"><label>k-Werte</label><input id="beir-k-values" class="config-input" value="1,3,5,10,100"${disabled}></div>`;
    html += `<div class="config-field"><label>Max Queries</label><input id="beir-max-queries" class="config-input" type="number" min="1" placeholder="leer = alle"${disabled}></div>`;
    html += `<div class="config-field"><label>Embedding-Modell</label><input id="beir-embedding-model" class="config-input" placeholder="leer = aktuelle Config"${disabled}></div>`;
    html += `<div class="config-field"><label>Run Label</label><input id="beir-run-label" class="config-input" placeholder="leer = automatisch"${disabled}></div>`;
    html += `<div class="config-field"><label>Chunk-Overfetch</label><input id="beir-overfetch" class="config-input" type="number" min="1" value="3"${disabled}></div>`;
    html += `<div class="config-field"><label>RRF k</label><input id="beir-rrf-k" class="config-input" type="number" min="1" placeholder="leer = aktuelle Config"${disabled}></div>`;
    html += `<div class="config-field"><label>ES Fuzziness</label><input id="beir-es-fuzziness" class="config-input" placeholder="leer = aktuelle Config, z.B. none/1/AUTO"${disabled}></div>`;
    html += `<div class="config-field"><label>Doc-Aggregation</label><select id="beir-doc-aggregation" class="config-input"${disabled}><option value="first">First Chunk</option><option value="best_plus_bonus">Best + Bonus</option><option value="rank_rrf">Document RRF</option></select></div>`;
    html += `<div class="config-field"><label>Doc-RRF k</label><input id="beir-doc-aggregation-rrf-k" class="config-input" type="number" min="1" value="60"${disabled}></div>`;
    html += `<div class="config-field"><label>Doc-Bonus</label><input id="beir-doc-aggregation-bonus" class="config-input" type="number" min="0" step="0.01" value="0.01"${disabled}></div>`;
    html += `<div class="config-field"><label>Doc-Bonus Cap</label><input id="beir-doc-aggregation-max-bonus-chunks" class="config-input" type="number" min="0" value="3"${disabled}></div>`;
    html += `<div class="config-field"><label>Reranking</label><label class="u-flex-center-gap-8"><input id="beir-reranking-enabled" type="checkbox"${disabled}> Cross-Encoder aktivieren</label></div>`;
    html += `<div class="config-field"><label>Rerank-Modell</label><input id="beir-rerank-model" class="config-input" placeholder="leer = aktuelle Config"${disabled}></div>`;
    html += `<div class="config-field"><label>Rerank Batch</label><input id="beir-rerank-batch-size" class="config-input" type="number" min="1" placeholder="leer = aktuelle Config"${disabled}></div>`;
    html += `<div class="config-field"><label>Rerank Kandidaten</label><input id="beir-rerank-candidate-limit" class="config-input" type="number" min="1" placeholder="leer = alle Kandidaten"${disabled}></div>`;
    html += `<div class="config-field"><label>Filter</label><label class="u-flex-center-gap-8"><input id="beir-use-filter-dir" type="checkbox" checked${disabled}> Dokumentverzeichnis filtern</label></div>`;
    html += '</div>';
    html += `<button class="action-btn primary" data-action="start-beir-run"${disabled}>BEIR starten</button>`;
    html += '</div>';
    return html;
}

function renderBeirTaskStatus(task) {
    if (!task || task.status === 'idle') return '';
    let html = '<div class="config-section">';
    html += '<h4>Status</h4>';
    if (task.running) {
        const total = Number(task.total || 0);
        const progress = Number(task.progress || 0);
        const pct = total > 0 ? Math.round((progress / total) * 100) : 0;
        html += '<div class="u-testbar-track">';
        html += `<div class="u-testbar-fill u-pw-${Math.max(0, Math.min(100, pct))}"></div>`;
        html += `<span class="u-testbar-label">${fmtInt(progress)}/${fmtInt(total)} (${pct}%)</span>`;
        html += '</div>';
        const rank = task.last_first_relevant_rank ? ` · relevanter Rang: ${fmtInt(task.last_first_relevant_rank)}` : '';
        const rerank = Number(task.last_rerank_latency_ms || 0) > 0 ? ` · Rerank: ${fmtFloat(task.last_rerank_latency_ms, 0)} ms` : '';
        html += `<div class="u-text-sm-secondary u-mt-8">Gestartet: ${esc(fmtTimestamp(task.started_at))} · Query: ${esc(task.current_query_id || '-')} · letzte Latenz: ${fmtFloat(task.last_latency_ms, 0)} ms${rerank}${rank}</div>`;
    } else if (task.status === 'success') {
        html += '<div class="badge badge-success">Abgeschlossen</div>';
        html += `<div class="u-text-sm-secondary u-mt-8">Gestartet: ${esc(fmtTimestamp(task.started_at))} · Fertig: ${esc(fmtTimestamp(task.finished_at))}</div>`;
    } else if (task.status === 'partial') {
        html += '<div class="badge badge-warning">Teilgueltig</div>';
        html += `<div class="u-text-sm-secondary u-mt-8">Gestartet: ${esc(fmtTimestamp(task.started_at))} · Fertig: ${esc(fmtTimestamp(task.finished_at))}</div>`;
    } else if (task.status === 'invalid') {
        html += '<div class="badge badge-danger">Ungueltig</div>';
        html += `<div class="u-text-sm-secondary u-mt-8">Gestartet: ${esc(fmtTimestamp(task.started_at))} · Fertig: ${esc(fmtTimestamp(task.finished_at))}</div>`;
    } else if (task.status === 'error') {
        html += `<div class="widget-error u-pad-12 u-pre-wrap">${esc(task.error || 'Unbekannter Fehler')}</div>`;
        html += `<div class="u-text-sm-secondary u-mt-8">Gestartet: ${esc(fmtTimestamp(task.started_at))} · Fertig: ${esc(fmtTimestamp(task.finished_at))}</div>`;
    }
    html += '</div>';
    return html;
}

function renderBeirKpis(summaries) {
    const latest = summaries[0] || null;
    const best = summaries.reduce((current, item) => {
        if (!current) return item;
        return metric(item, 'ndcg@10') > metric(current, 'ndcg@10') ? item : current;
    }, null);
    const bestRecall = summaries.reduce((current, item) => {
        if (!current) return item;
        return metric(item, 'recall@10') > metric(current, 'recall@10') ? item : current;
    }, null);
    let html = '<div class="widget-row u-mb-20">';
    html += benchmarkKpi('Runs', fmtInt(summaries.length));
    html += benchmarkKpi('Best nDCG@10', best ? fmtFloat(metric(best, 'ndcg@10'), 3) : '-');
    html += benchmarkKpi('Best Recall@10', bestRecall ? fmtFloat(metric(bestRecall, 'recall@10'), 3) : '-');
    html += benchmarkKpi('Letzter Run', latest ? (latest.label || latest.relative_path || '') : '-');
    html += '</div>';
    return html;
}

function renderBeirHistory(summaries, selectedId) {
    let html = '<div class="config-section"><h4>Historie</h4>';
    if (!summaries.length) {
        html += '<div class="no-data">Keine BEIR-Runs gefunden.</div></div>';
        return html;
    }
    html += '<div class="table-responsive"><table class="data-table"><thead><tr>';
    ['Run', 'Status', 'Zeit', 'Dataset', 'Retriever', 'Rerank', 'Embedding', 'DS', 'Queries', 'Recall@10', 'nDCG@10', 'MRR@10', 'Hit@10', 'Recall@100', 'Avg ms', 'Datei'].forEach(label => {
        html += `<th>${esc(label)}</th>`;
    });
    html += '</tr></thead><tbody>';
    summaries.forEach(item => {
        const selectedClass = item.id === selectedId ? ' class="u-row-bg-secondary-soft"' : '';
        const run = item.run || {};
        const settings = item.settings || {};
        html += `<tr${selectedClass}>`;
        html += '<td>';
        html += `<button class="action-btn-sm" data-action="select-beir-summary" data-summary-id="${escAttr(item.id)}">${esc(item.label || '')}</button>`;
        html += '</td>';
        html += `<td>${renderBeirRunStatusBadge(item)}</td>`;
        html += `<td class="u-tab-nums">${esc(fmtTimestamp(benchmarkTimestamp(item)))}</td>`;
        html += `<td>${esc(item.dataset || '')}</td>`;
        html += `<td><span class="badge badge-default">${esc(run.retriever || settings.retriever || '')}</span></td>`;
        html += `<td>${renderRerankBadge(item)}</td>`;
        html += `<td>${esc(run.effective_embedding_model || settings.embedding_model_effective || run.embedding_model || '')}</td>`;
        html += `<td class="u-text-right u-tab-nums">${esc(String(run.datasource_id || settings.datasource_id || ''))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtInt(metric(item, 'queries_evaluated'))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'recall@10'), 3)}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'ndcg@10'), 3)}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'mrr@10'), 3)}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'hitrate@10'), 3)}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'recall@100'), 3)}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'avg_latency_ms'), 1)}</td>`;
        html += `<td class="u-mono u-fs-11">${esc(item.relative_path || '')}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table></div></div>';
    return html;
}

function renderBeirSelectedDetails(summary) {
    let html = '<div class="config-section"><h4>Run-Settings</h4>';
    if (!summary) {
        html += '<div class="no-data">Kein Run ausgewaehlt.</div></div>';
        return html;
    }
    const run = summary.run || {};
    const settings = summary.settings || {};
    const reranking = settings.reranking || {};
    const overrides = settings.retrieval_overrides || {};
    const docAggregation = settings.doc_aggregation || {};
    const diagnostics = summary.ranking_diagnostics || {};
    const retrievalFailures = summary.retrieval_failures || {};
    const colbert = summary.colbert_diagnostics || {};
    html += `<div class="u-text-sm-secondary u-mb-12">${esc(summary.label || '')}</div>`;
    if (diagnostics.legacy_derived) {
        html += '<div class="u-text-sm-secondary u-mb-12">Ranking-Diagnose aus alter Details-Datei nachberechnet; exakte Duplicate-Chunk-Werte wurden erst spaeter geloggt.</div>';
    }
    html += '<div class="table-responsive u-mb-12"><table class="data-table"><tbody>';
    html += `<tr><td>Status</td><td>${renderBeirRunStatusBadge(summary)}</td></tr>`;
    html += `<tr><td>Gestartet</td><td class="u-tab-nums">${esc(fmtTimestamp(run.started_at || summary.created_at || summary.timestamp))}</td></tr>`;
    html += `<tr><td>Fertig</td><td class="u-tab-nums">${esc(fmtTimestamp(run.finished_at || summary.created_at || summary.modified))}</td></tr>`;
    html += `<tr><td>Dauer</td><td class="u-tab-nums">${fmtDurationMs(run.duration_ms)}</td></tr>`;
    html += `<tr><td>RRF k Override</td><td class="u-tab-nums">${esc(formatOptional(overrides.rrf_k || run.rrf_k_override))}</td></tr>`;
    html += `<tr><td>ES Fuzziness Override</td><td>${esc(formatOptional(overrides.elasticsearch_fuzziness || run.elasticsearch_fuzziness_override))}</td></tr>`;
    html += `<tr><td>Doc-Aggregation</td><td>${esc(docAggregation.strategy || run.doc_aggregation || '-')}</td></tr>`;
    html += `<tr><td>Doc-RRF k</td><td class="u-tab-nums">${esc(formatOptional(docAggregation.rrf_k || run.doc_aggregation_rrf_k))}</td></tr>`;
    html += `<tr><td>Doc-Bonus</td><td class="u-tab-nums">${esc(formatOptional(docAggregation.bonus || run.doc_aggregation_bonus))}</td></tr>`;
    html += `<tr><td>Doc-Bonus Cap</td><td class="u-tab-nums">${esc(formatOptional(docAggregation.max_bonus_chunks || run.doc_aggregation_max_bonus_chunks))}</td></tr>`;
    html += `<tr><td>Reranking</td><td>${renderRerankBadge(summary)}</td></tr>`;
    html += `<tr><td>Rerank-Modell</td><td>${esc(reranking.model_effective || run.rerank_model || '-')}</td></tr>`;
    html += `<tr><td>Ø Retrieval ms</td><td class="u-tab-nums">${fmtFloat(metric(summary, 'avg_retrieval_latency_ms'), 1)}</td></tr>`;
    html += `<tr><td>Ø Rerank ms</td><td class="u-tab-nums">${fmtFloat(metric(summary, 'avg_rerank_latency_ms'), 1)}</td></tr>`;
    html += `<tr><td>Retrieval-Fehler</td><td class="u-tab-nums">${fmtInt(retrievalFailures.failed_queries)}</td></tr>`;
    html += `<tr><td>ColBERT-Status</td><td>${esc(colbert.status || '-')}</td></tr>`;
    html += `<tr><td>ColBERT-Fehler</td><td class="u-tab-nums">${fmtInt(colbert.failed_queries)}</td></tr>`;
    html += `<tr><td>ColBERT-Nulltreffer</td><td class="u-tab-nums">${fmtInt(colbert.zero_result_queries)}</td></tr>`;
    html += `<tr><td>First Relevant Median</td><td class="u-tab-nums">${fmtFloat(diagnostics.first_relevant_rank_median, 1)}</td></tr>`;
    html += `<tr><td>First Relevant P90</td><td class="u-tab-nums">${fmtFloat(diagnostics.first_relevant_rank_p90, 1)}</td></tr>`;
    html += `<tr><td>Ohne relevanten Treffer @max_k</td><td class="u-tab-nums">${fmtInt(diagnostics.queries_without_relevant_at_max_k)}</td></tr>`;
    html += `<tr><td>Ø Duplicate-Doc-Chunks</td><td class="u-tab-nums">${fmtDiagnosticFloat(diagnostics, 'avg_duplicate_doc_chunks', 1)}</td></tr>`;
    html += '</tbody></table></div>';
    html += `<pre class="u-pre-wrap u-mono u-fs-12">${esc(JSON.stringify(summary.settings || {}, null, 2))}</pre>`;
    html += '</div>';
    return html;
}

async function startBeirRun() {
    if (_beirStartInFlight) return;
    const datasetEl = document.getElementById('beir-dataset-dir');
    const datasourceEl = document.getElementById('beir-datasource-id');
    if (!datasetEl || !datasourceEl || !datasetEl.value || !datasourceEl.value) {
        SharedUi.notify.show('Dataset und Datenquelle sind erforderlich', 'error');
        return;
    }
    _beirStartInFlight = true;
    const payload = {
        dataset_dir: datasetEl.value,
        datasource_id: Number(datasourceEl.value),
        retriever: document.getElementById('beir-retriever')?.value || 'hybrid',
        k_values: document.getElementById('beir-k-values')?.value || '1,3,5,10,100',
        max_queries: document.getElementById('beir-max-queries')?.value || null,
        embedding_model: document.getElementById('beir-embedding-model')?.value || null,
        run_label: document.getElementById('beir-run-label')?.value || null,
        use_filter_dir: !!document.getElementById('beir-use-filter-dir')?.checked,
        chunk_overfetch_factor: Number(document.getElementById('beir-overfetch')?.value || 3),
        rrf_k: document.getElementById('beir-rrf-k')?.value || null,
        elasticsearch_fuzziness: document.getElementById('beir-es-fuzziness')?.value || null,
        doc_aggregation: document.getElementById('beir-doc-aggregation')?.value || 'first',
        doc_aggregation_rrf_k: document.getElementById('beir-doc-aggregation-rrf-k')?.value || 60,
        doc_aggregation_bonus: document.getElementById('beir-doc-aggregation-bonus')?.value || 0.01,
        doc_aggregation_max_bonus_chunks: document.getElementById('beir-doc-aggregation-max-bonus-chunks')?.value || 3,
        reranking_enabled: !!document.getElementById('beir-reranking-enabled')?.checked,
        rerank_model: document.getElementById('beir-rerank-model')?.value || null,
        rerank_batch_size: document.getElementById('beir-rerank-batch-size')?.value || null,
        rerank_candidate_limit: document.getElementById('beir-rerank-candidate-limit')?.value || null,
    };
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/benchmarks/beir/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'BEIR-Start fehlgeschlagen');
        }
        SharedUi.notify.show('BEIR-Benchmark gestartet', 'success');
        _beirStartInFlight = false;
        initBenchmarkBeirTab();
    } catch (e) {
        _beirStartInFlight = false;
        SharedUi.notify.show(e.message || 'BEIR-Start fehlgeschlagen', 'error');
        initBenchmarkBeirTab();
    }
}

function startBeirPolling(mySeq) {
    kiaraPolling.start('benchmark-beir:status', async () => {
        if (mySeq !== _beirInitSeq) {
            kiaraPolling.stop('benchmark-beir:status');
            return;
        }
        const resp = await SharedUi.fetch.safeFetch('/api/admin/benchmarks/beir');
        if (mySeq !== _beirInitSeq) return;
        if (!resp.ok) return;
        const data = await resp.json();
        _beirData = data || {};
        renderBenchmarkBeirTab(document.getElementById('contentContainer'), _beirData);
        if (!((_beirData.task || {}).running)) {
            kiaraPolling.stop('benchmark-beir:status');
            if ((_beirData.task || {}).status === 'success') {
                SharedUi.notify.show('BEIR-Benchmark abgeschlossen', 'success');
            } else if ((_beirData.task || {}).status === 'partial') {
                SharedUi.notify.show('BEIR-Benchmark teilgueltig', 'warning');
            } else if ((_beirData.task || {}).status === 'invalid') {
                SharedUi.notify.show('BEIR-Benchmark ungueltig', 'error');
            } else if ((_beirData.task || {}).status === 'error') {
                SharedUi.notify.show('BEIR-Benchmark fehlgeschlagen', 'error');
            }
        }
    }, 2000);
}

function renderBenchmarkKpis(summaries, comparisons) {
    const latest = summaries[0] || null;
    const best = summaries.reduce((current, item) => {
        if (!current) return item;
        return metric(item, 'score') > metric(current, 'score') ? item : current;
    }, null);
    let html = '<div class="widget-row u-mb-20">';
    html += benchmarkKpi('Runs', fmtInt(summaries.length));
    html += benchmarkKpi('Vergleiche', fmtInt(comparisons.length));
    html += benchmarkKpi('Bester Score', best ? fmtFloat(metric(best, 'score'), 3) : '-');
    html += benchmarkKpi('Letzter Run', latest ? (latest.label || latest.relative_path || '') : '-');
    html += '</div>';
    return html;
}

function benchmarkKpi(label, value) {
    return `<div class="widget-card u-mw-120-min"><div class="u-pad-16 u-text-center">
        <div class="u-fs-11 u-c-secondary u-uppercase u-letter-sp-05 u-mb-4">${esc(label)}</div>
        <div class="u-fs-20 u-fw-700">${esc(value)}</div>
    </div></div>`;
}

function renderRerankBadge(item) {
    const run = (item && item.run) || {};
    const settings = (item && item.settings) || {};
    const reranking = settings.reranking || {};
    const enabled = Boolean(
        run.reranking_enabled ||
        reranking.enabled_effective ||
        reranking.enabled_requested
    );
    if (!enabled) return '<span class="badge badge-default">aus</span>';
    if (reranking.enabled_effective === false) {
        return '<span class="badge badge-warning">Fallback</span>';
    }
    return '<span class="badge badge-success">an</span>';
}

function renderBeirRunStatusBadge(item) {
    const run = (item && item.run) || {};
    const status = item && (item.status || run.status || (item.valid === false || run.valid === false ? 'invalid' : 'success'));
    if (status === 'partial') return '<span class="badge badge-warning">partial</span>';
    if (status === 'invalid') return '<span class="badge badge-danger">invalid</span>';
    if (status === 'error') return '<span class="badge badge-danger">error</span>';
    return '<span class="badge badge-success">success</span>';
}

function renderBenchmarkSummaries(summaries, selectedId) {
    let html = '<div class="config-section"><h4>Runs</h4>';
    if (!summaries.length) {
        html += '<div class="no-data">Keine CRAG-Summaries gefunden.</div></div>';
        return html;
    }
    html += '<div class="table-responsive"><table class="data-table"><thead><tr>';
    ['Run', 'Zeit', 'Dataset', 'Mode', 'k', 'n', 'Score', 'Accuracy', 'Missing', 'Hallucination', 'GoldCtx', 'Avg ms', 'Datei'].forEach(label => {
        html += `<th>${esc(label)}</th>`;
    });
    html += '</tr></thead><tbody>';
    summaries.forEach(item => {
        const selectedClass = item.id === selectedId ? ' class="u-row-bg-secondary-soft"' : '';
        const run = item.run || {};
        html += `<tr${selectedClass}>`;
        html += '<td>';
        html += `<button class="action-btn-sm" data-action="select-crag-summary" data-summary-id="${escAttr(item.id)}">${esc(item.label || '')}</button>`;
        html += '</td>';
        html += `<td class="u-tab-nums">${esc(fmtTimestamp(benchmarkTimestamp(item)))}</td>`;
        html += `<td>${esc(item.dataset || '')}</td>`;
        html += `<td><span class="badge badge-default">${esc(run.retrieval_mode || '')}</span></td>`;
        html += `<td class="u-text-right u-tab-nums">${esc(String(run.k || ''))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtInt(metric(item, 'total'))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'score'), 3)}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtPct(metric(item, 'accuracy'))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtPct(metric(item, 'missing_rate'))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtPct(metric(item, 'hallucination_rate'))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtPct(metric(item, 'gold_answer_in_context_rate'))}</td>`;
        html += `<td class="u-text-right u-tab-nums">${fmtFloat(metric(item, 'avg_latency_ms'), 1)}</td>`;
        html += `<td class="u-mono u-fs-11">${esc(item.relative_path || '')}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table></div></div>';
    return html;
}

function renderBenchmarkComparison(comparisons) {
    let html = '<div class="config-section"><h4>Ablation-Vergleiche</h4>';
    if (!comparisons.length) {
        html += '<div class="no-data">Keine Vergleichsdateien gefunden.</div></div>';
        return html;
    }
    comparisons.forEach(comparison => {
        const runs = Array.isArray(comparison.runs) ? comparison.runs : [];
        html += `<h5 class="u-my-12-0">${esc(comparison.label || comparison.relative_path || '')}</h5>`;
        html += '<div class="table-responsive"><table class="data-table"><thead><tr>';
        ['Run', 'Mode', 'k', 'Score', 'Accuracy', 'Missing', 'Hallucination', 'GoldCtx', 'Avg ms', 'Delta'].forEach(label => {
            html += `<th>${esc(label)}</th>`;
        });
        html += '</tr></thead><tbody>';
        runs.forEach(run => {
            const metrics = run.metrics || {};
            const deltas = run.delta_vs_baseline || {};
            const baseline = run.is_baseline ? ' <span class="badge badge-info">Baseline</span>' : '';
            html += '<tr>';
            html += `<td>${esc(run.label || run.summary_path || '')}${baseline}</td>`;
            html += `<td><span class="badge badge-default">${esc(run.retrieval_mode || '')}</span></td>`;
            html += `<td class="u-text-right u-tab-nums">${esc(String(run.k || ''))}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtFloat(metrics.score, 3)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.accuracy)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.missing_rate)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.hallucination_rate)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.gold_answer_in_context_rate)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtFloat(metrics.avg_latency_ms, 1)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtDelta(deltas.score)}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    });
    html += '</div>';
    return html;
}

function renderBenchmarkBreakdowns(summary) {
    let html = '<div class="config-section"><h4>Breakdowns</h4>';
    if (!summary) {
        html += '<div class="no-data">Kein Run ausgewaehlt.</div></div>';
        return html;
    }
    const fields = ['domain', 'question_type', 'static_or_dynamic', 'retrieval_mode', 'embedding_model', 'k'];
    const breakdowns = summary.breakdowns || {};
    html += `<div class="u-text-sm-secondary u-mb-12">${esc(summary.label || '')}</div>`;
    fields.forEach(field => {
        const values = breakdowns[field] || {};
        const rows = Object.entries(values).sort((a, b) => metric({ overall: b[1] }, 'total') - metric({ overall: a[1] }, 'total'));
        if (!rows.length) return;
        html += `<h5 class="u-my-12-0">${esc(breakdownLabel(field))}</h5>`;
        html += '<div class="table-responsive"><table class="data-table"><thead><tr>';
        ['Wert', 'n', 'Score', 'Accuracy', 'Missing', 'Hallucination', 'GoldCtx', 'Avg ms'].forEach(label => {
            html += `<th>${esc(label)}</th>`;
        });
        html += '</tr></thead><tbody>';
        rows.forEach(([value, metrics]) => {
            html += '<tr>';
            html += `<td>${esc(value)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtInt(metrics.total)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtFloat(metrics.score, 3)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.accuracy)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.missing_rate)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.hallucination_rate)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtPct(metrics.gold_answer_in_context_rate)}</td>`;
            html += `<td class="u-text-right u-tab-nums">${fmtFloat(metrics.avg_latency_ms, 1)}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
    });
    html += '</div>';
    return html;
}

function breakdownLabel(field) {
    const labels = {
        domain: 'Domain',
        question_type: 'Fragetyp',
        static_or_dynamic: 'Static/Dynamic',
        retrieval_mode: 'Retrieval Mode',
        embedding_model: 'Embedding-Modell',
        k: 'Top-k',
    };
    return labels[field] || field;
}

function metric(item, key) {
    const overall = item && item.overall ? item.overall : {};
    const value = overall[key];
    return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function benchmarkTimestamp(item) {
    const run = (item && item.run) || {};
    return run.started_at || item.created_at || item.timestamp || run.finished_at || item.modified || '';
}

function fmtInt(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    return num.toLocaleString('de-DE', { maximumFractionDigits: 0 });
}

function fmtFloat(value, digits) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    return num.toLocaleString('de-DE', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtDiagnosticFloat(diagnostics, key, digits) {
    if (diagnostics && diagnostics.legacy_derived && diagnostics[`${key}_available`] === false) {
        return 'nicht geloggt';
    }
    if (diagnostics && key === 'avg_duplicate_doc_chunks' && diagnostics.duplicate_doc_chunks_available === false) {
        return 'nicht geloggt';
    }
    return fmtFloat(diagnostics ? diagnostics[key] : null, digits);
}

function fmtPct(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    return (num * 100).toLocaleString('de-DE', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%';
}

function fmtDelta(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    const sign = num > 0 ? '+' : '';
    return sign + fmtFloat(num, 3);
}

function fmtDurationMs(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    if (num < 1000) return fmtFloat(num, 0) + ' ms';
    return fmtFloat(num / 1000, 1) + ' s';
}

function fmtTimestamp(value) {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString('de-DE', {
        day: '2-digit',
        month: '2-digit',
        year: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function formatOptional(value) {
    return value === undefined || value === null || value === '' ? '-' : String(value);
}

function esc(value) {
    return SharedUi.runtime.escapeHtml(value);
}

function escAttr(value) {
    return SharedUi.runtime.escapeAttr(value);
}
