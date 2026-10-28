/**
 * KIara Admin - Jobs Tab
 * Ingestion Job-Queue: Liste, Details, Filter, Polling.
 */

let jobsTable = null;
let _jobsCurrentFilter = new Set();  // leer = alle
let _jobsInitSeq = 0;

function _jobsApiParams() {
    return _jobsCurrentFilter.size > 0 ? {status: [..._jobsCurrentFilter].join(',')} : {};
}

const _JOBS_ACTIONS = new Set([
    'filter-jobs',
    'cleanup-job-history',
    'toggle-job-pause',
    'cancel-job',
    'retry-html-job',
    'delete-job',
]);

const _JOBS_ROW_ACTIONS = new Set([
    'toggle-job-pause',
    'cancel-job',
    'retry-html-job',
    'delete-job',
]);

function _handleJobsAction(e) {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    if (!_JOBS_ACTIONS.has(action)) return;

    if (_JOBS_ROW_ACTIONS.has(action)) {
        e.stopPropagation();
    }

    switch (action) {
        case 'filter-jobs':         _filterJobs(btn.dataset.filter); break;
        case 'cleanup-job-history': _cleanupJobHistory(btn); break;
        case 'toggle-job-pause':    _toggleJobPause(btn, Number(btn.dataset.jobId)); break;
        case 'cancel-job':          _cancelJob(btn, Number(btn.dataset.jobId)); break;
        case 'retry-html-job':      _retryHtmlJob(btn, Number(btn.dataset.jobId)); break;
        case 'delete-job':          _deleteJob(btn, Number(btn.dataset.jobId)); break;
    }
}

function _bindJobsActions() {
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (container && !container._jobsActionsBound) {
        container._jobsActionsBound = true;
        container.setAttribute('data-jobs-actions-bound', 'true');
        container.addEventListener('click', _handleJobsAction);
    }
    if (filterBar && !filterBar._jobsActionsBound) {
        filterBar._jobsActionsBound = true;
        filterBar.setAttribute('data-jobs-actions-bound', 'true');
        filterBar.addEventListener('click', _handleJobsAction);
    }
}

async function initJobsTab() {
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Re-Init-Schutz: altes Cleanup ausfuehren
    if (typeof _runTabCleanups === 'function') _runTabCleanups('jobs');
    _jobsInitSeq++;

    _bindJobsActions();

    // Filter Bar (Mehrfachauswahl)
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.remove('u-hidden');
        filterBar.classList.add('u-flex');

        const filterGroup = document.createElement('div');
        filterGroup.className = 'filter-group';
        filterGroup.innerHTML = `
            <button class="action-btn small ${_jobsCurrentFilter.size === 0 ? 'primary' : ''}" data-action="filter-jobs" data-filter="all">Alle</button>
            <button class="action-btn small ${_jobsCurrentFilter.has('running') ? 'primary' : ''}" data-action="filter-jobs" data-filter="running">Laufend</button>
            <button class="action-btn small ${_jobsCurrentFilter.has('pending') ? 'primary' : ''}" data-action="filter-jobs" data-filter="pending">Ausstehend</button>
            <button class="action-btn small ${_jobsCurrentFilter.has('success') ? 'primary' : ''}" data-action="filter-jobs" data-filter="success">Erfolgreich</button>
            <button class="action-btn small ${_jobsCurrentFilter.has('error') ? 'primary' : ''}" data-action="filter-jobs" data-filter="error">Fehlgeschlagen</button>
            <button class="action-btn small ${_jobsCurrentFilter.has('cancelled') ? 'primary' : ''}" data-action="filter-jobs" data-filter="cancelled">Abgebrochen</button>
        `;
        filterBar.appendChild(filterGroup);

        const rightGroup = document.createElement('div');
        rightGroup.className = 'filter-group u-ml-auto';
        rightGroup.innerHTML = `<button class="action-btn small danger" data-action="cleanup-job-history">Historie bereinigen</button>`;
        filterBar.appendChild(rightGroup);
    }

    container.innerHTML = '<div id="jobsTableContainer"></div>';

    const config = {
        id: 'ingestion-jobs',
        apiEndpoint: '/api/admin/ingestion-jobs',
        apiParams: _jobsApiParams(),
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 2fr 1.5fr 70px 90px 1.5fr 80px 1.2fr 80px 80px 100px',
        defaultSort: { field: 'id', direction: 'desc' },
        columns: [
            { field: 'datasource_name', label: 'Datenquelle', sortable: true, align: 'left', truncate: 25 },
            { field: 'embedding_model', label: 'Embedding', sortable: true, align: 'left', truncate: 20 },
            { field: 'job_type', label: 'Typ', sortable: true, align: 'center', renderer: _jobTypeBadge },
            { field: 'status', label: 'Status', sortable: true, align: 'center', renderer: _jobStatusBadge },
            { field: 'progress', label: 'Dateien', sortable: false, align: 'center', renderer: (v, row) => {
                if (row.status === 'pending') return '-';
                const stats = row.stats || {};
                const processed = row.progress_processed || 0;
                const total = row.progress_total || 0;

                // --- Parallel-Scan: Dual-Progress ---
                const scanProc = stats.scan_processed || 0;
                const scanTotal = stats.scan_total || 0;
                if (row.status === 'running' && scanTotal > 0) {
                    const scanPct = Math.round(scanProc / scanTotal * 100);
                    let html = `<div class="u-progress-row"><small class="u-c-secondary">Download</small>`
                        + `<div class="progress-bar-mini"><div class="progress-fill progress-fill--warning u-pw-${scanPct}"></div></div>`
                        + `<small>${scanProc}/${scanTotal}</small></div>`;
                    if (total > 0) {
                        const pct = Math.round(processed / total * 100);
                        html += `<div><small class="u-c-secondary">Ingestion</small>`
                            + `<div class="progress-bar-mini"><div class="progress-fill u-pw-${pct}"></div></div>`
                            + `<small>${processed}/${total}</small></div>`;
                    } else {
                        const phase = stats.phase;
                        if (phase) html += `<small class="u-c-secondary">${SharedUi.runtime.escapeHtml(phase)}</small>`;
                    }
                    const curFile = stats.current_file;
                    if (curFile) {
                        html += `<div class="u-fs-075em u-c-secondary u-mw-220 u-ellipsis" title="${SharedUi.runtime.escapeAttr(curFile)}">${SharedUi.runtime.escapeHtml(curFile)}</div>`;
                    }
                    return html;
                }

                // --- Standard: Single-Progress ---
                if (total === 0) {
                    const phase = stats.phase;
                    return phase ? `<small class="u-c-secondary">${SharedUi.runtime.escapeHtml(phase)}</small>` : '-';
                }
                const pct = Math.round(processed / total * 100);
                let progressHtml = `<div class="progress-bar-mini"><div class="progress-fill u-pw-${pct}"></div></div><small>${processed}/${total}</small>`;
                const curFile = stats.current_file;
                if (row.status === 'running' && curFile) {
                    progressHtml += `<div class="u-fs-075em u-c-secondary u-mw-220 u-ellipsis" title="${SharedUi.runtime.escapeAttr(curFile)}">${SharedUi.runtime.escapeHtml(curFile)}</div>`;
                }
                return progressHtml;
            }},
            { field: 'eta', label: 'Restzeit', sortable: false, align: 'center', renderer: (v, row) => {
                if (row.status !== 'running' || !row.started_at) return '-';
                const processed = row.progress_processed || 0;
                const total = row.progress_total || 0;
                if (total === 0 || processed === 0) return '-';
                const elapsed = (Date.now() - _parseJobDate(row.started_at).getTime()) / 1000;
                if (!isFinite(elapsed) || elapsed < 5) return '-';
                const rate = processed / elapsed;
                const remaining = (total - processed) / rate;
                if (remaining < 60) return `~${Math.round(remaining)}s`;
                if (remaining < 3600) return `~${Math.floor(remaining/60)}m ${Math.round(remaining%60)}s`;
                return `~${Math.floor(remaining/3600)}h ${Math.floor((remaining%3600)/60)}m`;
            }},
            { field: 'started_at', label: 'Gestartet', sortable: true, align: 'center', renderer: _formatJobTime },
            { field: 'duration', label: 'Dauer', sortable: false, align: 'center', renderer: (v, row) => {
                if (!row.started_at) return '-';
                const start = _parseJobDate(row.started_at);
                const end = row.finished_at ? _parseJobDate(row.finished_at) : new Date();
                const sec = Math.round((end - start) / 1000);
                if (!isFinite(sec)) return '-';
                const fmtDur = (s) => {
                    if (s < 60) return `${s}s`;
                    if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
                    return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
                };
                let html = fmtDur(sec);
                if (row.resume_chain_ancestor_duration > 0) {
                    const total = sec + row.resume_chain_ancestor_duration;
                    html = `<span title="Dieser Job: ${fmtDur(sec)}, Vorfahren: ${fmtDur(row.resume_chain_ancestor_duration)}">${fmtDur(total)} (ges.)</span>`;
                }
                return html;
            }},
            { field: 'triggered_by', label: 'Ausloser', sortable: true, align: 'center', renderer: (v) => {
                const labels = { manual: 'Manuell', cron: 'Cron', api: 'API' };
                return labels[v] || SharedUi.runtime.escapeHtml(v);
            }},
            { field: 'id', label: 'Aktionen', sortable: false, align: 'right', renderer: _jobActionButtons },
        ],
        detailFields: [
            { field: 'stats_overview', label: 'Ergebnis', renderer: (v, row) => _renderJobStats(row) },
        ],
        detailRenderer: (row) => _renderJobDetail(row),
    };

    jobsTable = new SharedUi.table.ExpandableTable(config, 'jobsTableContainer', {});
    jobsTable.loadData(_jobsApiParams());

    // Cleanup registrieren + Polling starten
    registerTabCleanup('jobs', () => { kiaraPolling.stop('jobs:refresh'); jobsTable = null; });
    _startJobsPolling();
}

function _jobTypeBadge(type, row) {
    if (row?.stats?.html_reingest) {
        return '<span class="badge info" title="HTML neu aufbereiten">HTML</span>';
    }
    return type === 'full' ? '<span class="badge warning">Voll</span>' : '<span class="badge info">Inkr.</span>';
}

function _jobActionButtons(jobId, row) {
    let html = '<div class="action-buttons">';
    if (row.status === 'running') {
        const pauseLabel = row.is_paused ? '&#x25B6;' : '&#x23F8;';
        html += `<button class="action-btn-sm" title="${row.is_paused ? 'Fortsetzen' : 'Pausieren'}" data-action="toggle-job-pause" data-job-id="${jobId}">${pauseLabel}</button>`;
        html += `<button class="action-btn-sm danger" title="Abbrechen" data-action="cancel-job" data-job-id="${jobId}">&#x2716;</button>`;
    } else if (row.status === 'pending') {
        html += `<button class="action-btn-sm danger" title="Abbrechen" data-action="cancel-job" data-job-id="${jobId}">&#x2716;</button>`;
    } else {
        if (row.stats?.html_reingest && ['error', 'cancelled'].includes(row.status)) {
            html += `<button class="action-btn-sm" title="Wiederholen" data-action="retry-html-job" data-job-id="${jobId}">&#x21BB;</button>`;
        }
        html += `<button class="action-btn-sm danger" title="Loeschen" data-action="delete-job" data-job-id="${jobId}">&#x1F5D1;</button>`;
    }
    return html + '</div>';
}

function _jobStatusBadge(status, row) {
    const map = {
        pending: '<span class="badge">Ausstehend</span>',
        running: '<span class="badge success">Laufend</span>',
        success: '<span class="badge success">Erfolgreich</span>',
        error: '<span class="badge danger">Fehler</span>',
        cancelled: '<span class="badge warning">Abgebrochen</span>',
    };
    if (status === 'running' && row && row.is_paused) {
        return '<span class="badge warning">Pausiert</span>';
    }
    if (status === 'running' && row && row.is_ghost) {
        return '<span class="badge danger" title="Job steht als laufend in der DB, aber kein Worker-Prozess verarbeitet ihn. Wird beim naechsten Service-Restart automatisch bereinigt.">Ghost</span>';
    }
    let html = map[status] || ('<span class="badge">' + SharedUi.runtime.escapeHtml(status) + '</span>');
    if (row && row.resume_chain_length > 1) {
        const restarts = row.resume_chain_length - 1;
        if (row.stats?.html_reingest) {
            html += ` <span class="badge u-fs-075em" title="Job wurde ${restarts}× wiederaufgenommen (Original: #${row.resume_chain_original_job_id})">${restarts}\u00D7 Wiederaufnahmen</span>`;
        } else {
            html += ` <span class="badge u-fs-075em" title="Job wurde ${restarts}× nach Service-Restart fortgesetzt (Original: #${row.resume_chain_original_job_id})">${restarts}\u00D7 Restart</span>`;
        }
    }
    return html;
}

function _parseJobDate(v) {
    if (!v) return null;
    // PostgreSQL: "2026-03-31 15:13:22.992373+02" → JS braucht "+02:00"
    let s = v.replace(' ', 'T');
    s = s.replace(/([+-]\d{2})$/, '$1:00');
    return new Date(s);
}

function _formatJobTime(v) {
    if (!v) return '-';
    try {
        const d = _parseJobDate(v);
        return d.toLocaleString('de-DE', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
    } catch(e) { return v; }
}

function _renderJobDetail(row) {
    const stats = row.stats || {};
    let html = _renderHtmlReingestSummary(stats);
    html += '<div class="detail-grid u-grid-2col-16">';

    // Links: Dateien + Chunks
    html += '<div>';
    html += '<h4 class="u-my-0-8">Dateien & Chunks</h4>';
    html += '<table class="detail-table">';
    html += `<tr><td>Dateien gesamt</td><td>${stats.total_files ?? '-'}</td></tr>`;
    html += `<tr><td>Verarbeitet</td><td>${stats.processed_files ?? '-'}</td></tr>`;
    html += `<tr><td>Uebersprungen</td><td>${stats.skipped_files ?? '-'}</td></tr>`;
    html += `<tr><td>Geloescht</td><td>${stats.deleted_files ?? '-'}</td></tr>`;
    html += `<tr><td>Chunks erzeugt</td><td>${stats.total_chunks ?? '-'}</td></tr>`;
    html += `<tr><td>Fehler</td><td>${stats.errors ?? 0}</td></tr>`;
    html += '</table>';
    html += '</div>';

    // Rechts: Features
    html += '<div>';
    html += '<h4 class="u-my-0-8">Feature-Details</h4>';
    html += '<table class="detail-table">';
    if (stats.ddu_files !== undefined) {
        html += `<tr><td>DDU (Docling)</td><td>${stats.ddu_files} Dateien</td></tr>`;
        if (stats.ddu_fallback_files) html += `<tr><td>DDU Fallback</td><td>${stats.ddu_fallback_files}</td></tr>`;
    }
    if (stats.vlm_chunks !== undefined) {
        html += `<tr><td>VLM Chunks</td><td>${stats.vlm_chunks}</td></tr>`;
        if (stats.vlm_errors) html += `<tr><td>VLM Fehler</td><td>${stats.vlm_errors}</td></tr>`;
    }
    if (stats.graph_entities !== undefined) {
        html += `<tr><td>Graph Entitaeten</td><td>${stats.graph_entities}</td></tr>`;
        html += `<tr><td>Graph Relationen</td><td>${stats.graph_relations || 0}</td></tr>`;
        if (stats.graph_errors) html += `<tr><td>Graph Fehler</td><td>${stats.graph_errors}</td></tr>`;
    }
    if (stats.vectordb_points_ds !== undefined) {
        html += `<tr><td>Vektoren (DS)</td><td>${stats.vectordb_points_ds.toLocaleString('de-DE')}</td></tr>`;
        if (stats.vectordb_points_total !== undefined) {
            html += `<tr><td>Vektoren (gesamt)</td><td>${stats.vectordb_points_total.toLocaleString('de-DE')} <small class="u-c-secondary">${SharedUi.runtime.escapeHtml(stats.vectordb_collection || '')}</small></td></tr>`;
        }
    }
    if (stats.colbert_chunks_written) {
        html += `<tr><td>ColBERT Chunks</td><td>${stats.colbert_chunks_written.toLocaleString('de-DE')}`;
        if (stats.colbert_collection) html += ` <small class="u-c-secondary">${SharedUi.runtime.escapeHtml(stats.colbert_collection)}</small>`;
        html += `</td></tr>`;
    }
    if (stats.colbert_errors) {
        html += `<tr><td>ColBERT Fehler</td><td>${stats.colbert_errors}</td></tr>`;
    }
    if (stats.vectordb_errors) {
        html += `<tr><td>VectorDB Fehler</td><td>${stats.vectordb_errors}</td></tr>`;
    }
    if (stats.vectordb_recoveries) {
        html += `<tr><td>VectorDB Recoveries</td><td>${stats.vectordb_recoveries}</td></tr>`;
    }
    if (stats.duration_seconds !== undefined) {
        const dur = stats.duration_seconds;
        const durStr = dur < 60 ? `${dur}s` : `${Math.floor(dur/60)}m ${dur%60}s`;
        html += `<tr><td>Dauer</td><td>${durStr}</td></tr>`;
    }
    html += '</table>';
    html += '</div>';

    html += '</div>';

    // Resume-Kette
    if (row.resume_chain && row.resume_chain.length > 0) {
        html += '<h4 class="u-my-12-8">Resume-Kette</h4>';
        html += '<table class="detail-table u-w-full">';
        html += `<tr><td>Original-Job</td><td>#${row.resume_chain_original_job_id}</td></tr>`;
        html += `<tr><td>${stats.html_reingest ? 'Wiederaufnahmen' : 'Neustarts'}</td><td>${row.resume_chain_length - 1}</td></tr>`;
        const fmtD = (s) => { if (!s) return '-'; if (s < 60) return `${s}s`; if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`; return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`; };
        const totalDur = (row.resume_chain_ancestor_duration || 0) + (stats.duration_seconds || 0);
        html += `<tr><td>Gesamtdauer (alle Jobs)</td><td>${fmtD(totalDur)}</td></tr>`;
        html += '</table>';
        html += '<h4 class="u-my-12-8">Verlauf</h4>';
        html += '<table class="detail-table u-w-full">';
        html += '<tr><th>Job</th><th>Status</th><th>Gestartet</th><th>Beendet</th><th>Dauer</th><th>Fehler</th></tr>';
        // Vorfahren chronologisch (aeltester zuerst)
        const ancestors = [...row.resume_chain].reverse();
        for (const a of ancestors) {
            html += `<tr>`;
            html += `<td>#${a.id}</td>`;
            html += `<td>${_jobStatusBadge(a.status, {})}</td>`;
            html += `<td>${_formatJobTime(a.started_at)}</td>`;
            html += `<td>${_formatJobTime(a.finished_at)}</td>`;
            html += `<td>${fmtD(a.duration_seconds)}</td>`;
            html += `<td class="u-mw-300 u-ellipsis">${SharedUi.runtime.escapeHtml(a.error_message)}</td>`;
            html += `</tr>`;
        }
        // Aktueller Job
        html += `<tr class="u-bold">`;
        html += `<td>#${row.id} (aktuell)</td>`;
        html += `<td>${_jobStatusBadge(row.status, row)}</td>`;
        html += `<td>${_formatJobTime(row.started_at)}</td>`;
        html += `<td>${_formatJobTime(row.finished_at)}</td>`;
        html += `<td>${fmtD(stats.duration_seconds)}</td>`;
        html += `<td>${SharedUi.runtime.escapeHtml(row.error_message)}</td>`;
        html += `</tr>`;
        html += '</table>';
    }

    // Error-Message (bei Job-Level-Fehler)
    if (row.error_message) {
        html += `<div class="alert danger u-mt-12">${SharedUi.runtime.escapeHtml(row.error_message)}</div>`;
    }

    // Datei-Protokoll (lazy-loaded)
    html += `<div id="fileLogSection-${row.id}" class="u-mt-16"></div>`;
    // Lazy-Load nach Render
    setTimeout(() => _renderFileLogSection(row.id, document.getElementById(`fileLogSection-${row.id}`), row.status), 0);

    return html;
}

function _renderHtmlReingestSummary(stats) {
    if (!stats.html_reingest) return '';
    const mode = stats.html_reingest.mode === 'all'
        ? 'Alle bestätigten HTML-Inhalte' : 'Nur veraltete Aufbereitungsstände';
    let html = '<h4 class="u-my-0-8">HTML-Neuaufbereitung</h4>';
    html += `<p>${mode}</p>`;
    const summary = stats.html_reingest_summary;
    if (!summary) return html + '<p class="u-c-secondary">Bestandsermittlung noch nicht abgeschlossen.</p>';
    html += '<table class="detail-table u-mb-12">';
    for (const [key, label] of [
        ['inspected', 'Bestand geprüft'], ['selected', 'Ausgewählt'],
        ['current', 'Bereits aktuell'], ['not_html', 'Kein HTML'],
        ['errors', 'Fehler insgesamt'],
        ['missing_original', 'davon Original fehlt'], ['empty_body', 'davon Body leer'],
    ]) {
        html += `<tr><td>${label}</td><td>${SharedUi.runtime.escapeHtml(summary[key] ?? '-')}</td></tr>`;
    }
    return html + '</table>';
}

function _renderJobStats(row) {
    const stats = row.stats || {};
    if (!stats.total_files && !row.error_message) return '-';
    let parts = [];
    if (stats.total_files !== undefined) parts.push(`${stats.processed_files || 0}/${stats.total_files} Dateien`);
    if (stats.total_chunks) parts.push(`${stats.total_chunks} Chunks`);
    if (stats.errors) parts.push(`${stats.errors} Fehler`);
    return parts.join(', ') || '-';
}

async function _filterJobs(status) {
    if (status === 'all') {
        _jobsCurrentFilter.clear();
    } else if (_jobsCurrentFilter.has(status)) {
        _jobsCurrentFilter.delete(status);
    } else {
        _jobsCurrentFilter.add(status);
    }
    await initJobsTab();
}

function _jobsErrorMessage(e) {
    const b = (e && e.body) || {};
    return b.detail || b.error || b.message || (e && e.message) || 'Fehler';
}

function _cancelJob(btn, jobId) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        if (!await SharedUi.modal.confirm({ message: 'Job wirklich abbrechen?' })) return;
        try {
            await SharedUi.fetch.safeFetch(`/api/admin/ingestion-jobs/${jobId}/cancel`, {
                method: 'POST', headers: {'X-Requested-With': 'XMLHttpRequest'},
            });
        } catch (e) {
            SharedUi.notify.show(_jobsErrorMessage(e), 'error');
            return;
        }
        if (jobsTable) await jobsTable.loadData(_jobsApiParams());
    });
}

function _deleteJob(btn, jobId) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        if (!await SharedUi.modal.confirm({ message: 'Job #' + jobId + ' unwiderruflich loeschen?' })) return;
        try {
            await SharedUi.fetch.safeFetch(`/api/admin/ingestion-jobs/${jobId}`, {
                method: 'DELETE', headers: {'X-Requested-With': 'XMLHttpRequest'},
            });
        } catch (e) {
            SharedUi.notify.show(_jobsErrorMessage(e), 'error');
            return;
        }
        SharedUi.notify.show('Job geloescht', 'success');
        if (jobsTable) await jobsTable.loadData(_jobsApiParams());
    });
}

function _retryHtmlJob(btn, jobId) {
    return SharedUi.runtime.withButtonLock(btn, async () => {
        try {
            const response = await SharedUi.fetch.safeFetch(`/api/admin/ingestion-jobs/${jobId}/retry-html`, {
                method: 'POST', headers: {'X-Requested-With': 'XMLHttpRequest'},
            });
            const result = await response.json();
            SharedUi.notify.show(`HTML-Job #${result.job_id} erneut eingestellt.`, 'info');
        } catch (e) {
            SharedUi.notify.show(_jobsErrorMessage(e), 'error');
            return;
        }
        if (jobsTable) await jobsTable.loadData(_jobsApiParams());
    });
}

function _toggleJobPause(btn, jobId) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        try {
            await SharedUi.fetch.safeFetch(`/api/admin/ingestion-jobs/${jobId}/pause`, {
                method: 'POST', headers: {'X-Requested-With': 'XMLHttpRequest'},
            });
        } catch (e) {
            SharedUi.notify.show(_jobsErrorMessage(e), 'error');
            return;
        }
        if (jobsTable) await jobsTable.loadData(_jobsApiParams());
    });
}

function _promptCleanupDays() {
    return new Promise((resolve) => {
        const wrap = document.createElement('div');
        const label = document.createElement('label');
        label.className = 'u-fs-14 u-d-block u-mb-8';
        label.textContent = 'Jobs aelter als X Tage loeschen:';
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.value = '30';
        input.className = 'u-w-100p';
        wrap.appendChild(label);
        wrap.appendChild(input);

        let settled = false;
        let modalId = null;
        const settle = (value) => {
            if (settled) return;
            settled = true;
            if (modalId !== null) SharedUi.modal.close(modalId);
            resolve(value);
        };

        modalId = SharedUi.modal.open({
            title: 'Job-Historie bereinigen',
            element: wrap,
            maxWidth: '400px',
            stack: true,
            escapeClose: false,
            buttons: [
                { label: 'Abbrechen', onClick: () => settle(null) },
                { label: 'Loeschen', className: 'primary', onClick: () => settle(input.value) },
            ],
            onOpen: (overlay) => {
                try { input.focus(); input.select(); } catch (_) {}
                overlay.addEventListener('keydown', (ev) => {
                    if (ev.key === 'Enter') { ev.preventDefault(); settle(input.value); }
                    else if (ev.key === 'Escape') { ev.preventDefault(); settle(null); }
                });
            },
            onClose: () => { if (!settled) { settled = true; resolve(null); } },
        });
    });
}

function _cleanupJobHistory(btn) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        const days = await _promptCleanupDays();
        if (days === null || days === '') return;
        const daysInt = parseInt(days);
        if (isNaN(daysInt) || daysInt < 1) { SharedUi.notify.show('Ungueltige Eingabe', 'error'); return; }
        let data;
        try {
            const resp = await SharedUi.fetch.safeFetch(`/api/admin/ingestion-jobs/history?days=${daysInt}`, {
                method: 'DELETE', headers: {'X-Requested-With': 'XMLHttpRequest'},
            });
            data = await resp.json().catch(() => ({}));
        } catch (e) {
            SharedUi.notify.show(_jobsErrorMessage(e), 'error');
            return;
        }
        SharedUi.notify.show(data.message || 'Fertig', 'success');
        if (jobsTable) await jobsTable.loadData(_jobsApiParams());
    });
}

// --- Datei-Protokoll (Per-File-Logs) ---

async function _renderFileLogSection(jobId, container, jobStatus) {
    if (!container) return;
    const myInitSeq = _jobsInitSeq;
    let loadSeq = 0;

    container.innerHTML = '<h4 class="u-my-0-8">Datei-Protokoll</h4><div class="u-c-secondary">Laden...</div>';

    // State fuer Filter/Pagination
    const state = { status: null, search: '', sort: 'file_name', direction: 'asc', offset: 0, limit: 100 };

    async function loadAndRender() {
        if (myInitSeq !== _jobsInitSeq) return;
        const myToken = ++loadSeq;
        const isStale = () => myToken !== loadSeq || myInitSeq !== _jobsInitSeq;
        const params = new URLSearchParams({ sort: state.sort, direction: state.direction, limit: state.limit, offset: state.offset });
        if (state.status) params.set('status', state.status);
        if (state.search) params.set('search', state.search);

        let data;
        try {
            const resp = await SharedUi.fetch.safeFetch(`/api/admin/ingestion-jobs/${jobId}/file-logs?${params}`, {
                headers: {'X-Requested-With': 'XMLHttpRequest'},
            });
            if (isStale()) return;
            data = await resp.json();
            if (isStale()) return;
        } catch(e) {
            if (isStale()) return;
            if (e && e.status === 404) {
                container.innerHTML = '<h4 class="u-my-0-8">Datei-Protokoll</h4><div class="u-c-secondary">Keine Daten verfuegbar</div>';
            } else {
                container.innerHTML = '<h4 class="u-my-0-8">Datei-Protokoll</h4><div class="u-c-secondary">Fehler beim Laden</div>';
            }
            return;
        }

        const summary = data.summary || {};
        const total = data.total || 0;

        let html = '<h4 class="u-my-0-8">Datei-Protokoll</h4>';

        if (jobStatus === 'running') {
            html += '<div class="u-c-secondary u-mb-8 u-italic">Protokoll wird noch geschrieben...</div>';
        }

        // Summary-Badges
        html += '<div class="u-flex-gap-8 u-items-center u-flex-wrap u-mb-8">';
        const badgeStyle = (active) => active ? 'primary' : '';
        html += `<button class="action-btn small ${badgeStyle(state.status === null)}" data-fl-filter="">Alle (${(summary.success||0)+(summary.error||0)+(summary.skipped||0)})</button>`;
        html += `<button class="action-btn small ${badgeStyle(state.status === 'success')} u-c-bullish" data-fl-filter="success">Erfolg: ${summary.success||0}</button>`;
        html += `<button class="action-btn small ${badgeStyle(state.status === 'error')} u-c-danger" data-fl-filter="error">Fehler: ${summary.error||0}</button>`;
        html += `<button class="action-btn small ${badgeStyle(state.status === 'skipped')} u-c-secondary" data-fl-filter="skipped">Uebersprungen: ${summary.skipped||0}</button>`;
        html += `<span class="u-ml-auto u-c-secondary u-fs-085em">Chunks: ${summary.total_chunks||0} | Avg: ${summary.avg_duration_ms||0}ms</span>`;
        html += '</div>';

        // Suchfeld
        html += '<div class="u-mb-8">';
        html += `<input type="text" placeholder="Dateiname suchen..." value="${SharedUi.runtime.escapeAttr(state.search)}" data-fl-search class="jobs-fl-search">`;
        html += '</div>';

        // Tabelle
        const rows = data.data || [];
        if (rows.length === 0) {
            html += '<div class="u-c-secondary u-pad-12">Keine Eintraege</div>';
        } else {
            html += '<div class="u-max-h-400">';
            html += '<table class="detail-table u-w-full u-fs-09em">';
            const sortIcon = (field) => {
                if (state.sort !== field) return '';
                return state.direction === 'asc' ? ' &#x25B2;' : ' &#x25BC;';
            };
            html += '<tr>';
            html += `<th class="u-clickable" data-fl-sort="file_name">Datei${sortIcon('file_name')}</th>`;
            html += `<th class="u-clickable u-text-right" data-fl-sort="file_size">Groesse${sortIcon('file_size')}</th>`;
            html += `<th class="u-clickable" data-fl-sort="parser">Parser${sortIcon('parser')}</th>`;
            html += `<th class="u-clickable u-text-right" data-fl-sort="chunk_count">Chunks${sortIcon('chunk_count')}</th>`;
            html += '<th>Strategie</th>';
            html += `<th class="u-clickable u-text-right" data-fl-sort="duration_ms">Dauer${sortIcon('duration_ms')}</th>`;
            html += `<th class="u-clickable u-text-center" data-fl-sort="status">Status${sortIcon('status')}</th>`;
            html += '</tr>';

            for (const f of rows) {
                const isError = f.status === 'error';
                const rowClass = isError ? ' jobs-fl-row-error' : '';
                const sizeStr = _fmtFileSize(f.file_size);
                const durStr = f.duration_ms == null ? '-'
                    : f.duration_ms < 1000 ? `${f.duration_ms}ms`
                    : `${(f.duration_ms/1000).toFixed(1)}s`;
                const statusBadge = f.status === 'success' ? '<span class="badge success">OK</span>'
                    : f.status === 'error' ? '<span class="badge danger">Fehler</span>'
                    : '<span class="badge">Uebersprungen</span>';
                const parserStr = (f.parser ?? '-') + (f.parser_fallback ? ' (FB)' : '');

                html += `<tr class="${rowClass.trim()}">`;
                html += `<td class="u-mw-280 u-ellipsis" title="${SharedUi.runtime.escapeAttr(f.file_path)}">${SharedUi.runtime.escapeHtml(f.file_name)}</td>`;
                html += `<td class="u-text-right">${sizeStr}</td>`;
                html += `<td>${SharedUi.runtime.escapeHtml(parserStr)}</td>`;
                html += `<td class="u-text-right">${f.chunk_count ?? '-'}</td>`;
                html += `<td>${SharedUi.runtime.escapeHtml(f.chunk_strategy)}</td>`;
                html += `<td class="u-text-right">${durStr}</td>`;
                html += `<td class="u-text-center">${statusBadge}</td>`;
                html += '</tr>';
                if (isError && f.error_message) {
                    html += `<tr class="${rowClass.trim()}"><td colspan="7" class="jobs-fl-error-detail">[${SharedUi.runtime.escapeHtml(f.error_phase)}] ${SharedUi.runtime.escapeHtml(f.error_message)}</td></tr>`;
                }
            }
            html += '</table>';
            html += '</div>';

            // Pagination
            if (total > state.limit) {
                const page = Math.floor(state.offset / state.limit) + 1;
                const pages = Math.ceil(total / state.limit);
                html += `<div class="u-flex-gap-8 u-items-center u-mt-8">`;
                html += `<span class="u-c-secondary u-fs-085em">${total} Dateien, Seite ${page}/${pages}</span>`;
                if (state.offset > 0) {
                    html += `<button class="action-btn small" data-fl-page="prev">Zurueck</button>`;
                }
                if (state.offset + state.limit < total) {
                    html += `<button class="action-btn small" data-fl-page="next">Mehr laden</button>`;
                }
                html += '</div>';
            }
        }

        container.innerHTML = html;

        // Event-Handler
        container.querySelectorAll('[data-fl-filter]').forEach(btn => {
            btn.addEventListener('click', () => {
                const v = btn.getAttribute('data-fl-filter');
                state.status = v || null;
                state.offset = 0;
                loadAndRender();
            });
        });
        container.querySelectorAll('[data-fl-sort]').forEach(th => {
            th.addEventListener('click', () => {
                const field = th.getAttribute('data-fl-sort');
                if (state.sort === field) {
                    state.direction = state.direction === 'asc' ? 'desc' : 'asc';
                } else {
                    state.sort = field;
                    state.direction = 'asc';
                }
                state.offset = 0;
                loadAndRender();
            });
        });
        const searchInput = container.querySelector('[data-fl-search]');
        if (searchInput) {
            let searchTimer = null;
            searchInput.addEventListener('input', () => {
                clearTimeout(searchTimer);
                searchTimer = setTimeout(() => {
                    state.search = searchInput.value.trim();
                    state.offset = 0;
                    loadAndRender();
                }, 300);
            });
        }
        container.querySelectorAll('[data-fl-page]').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.getAttribute('data-fl-page') === 'next') state.offset += state.limit;
                else state.offset = Math.max(0, state.offset - state.limit);
                loadAndRender();
            });
        });
    }

    loadAndRender();
}

function _fmtFileSize(bytes) {
    if (!bytes || bytes === 0) return '-';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function _startJobsPolling() {
    let idleCycles = 0;
    kiaraPolling.start('jobs:refresh', async () => {
        if (!jobsTable) { kiaraPolling.stop('jobs:refresh'); return; }
        try {
            await jobsTable.silentRefresh();
            const rows = jobsTable.getAllData ? jobsTable.getAllData() : [];
            const hasActive = rows.some(r =>
                r.status === 'running' || r.status === 'paused' || r.status === 'pending');
            if (!hasActive) {
                idleCycles++;
                if (idleCycles >= 3) kiaraPolling.stop('jobs:refresh');
            } else {
                idleCycles = 0;
            }
        } catch(e) {
            kiaraPolling.stop('jobs:refresh');
        }
    }, 3000);
}
