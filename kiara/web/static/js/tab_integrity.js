/**
 * KIara Admin - System > Integritaetspruefung Sub-Tab
 * Redesign V8: ExpandableTable + generische startRepair + Collection-Rebuild
 */

let integrityTable = null;
let _integrityInitSeq = 0;
let _integrityRepairLocked = false;

const _INTEGRITY_ACTIONS = new Set([
    'start-integrity-check',
    'run-quick-check',
    'start-enrichment-reindex',
    'start-repair',
    'switch-to-retrieval',
]);

const _INTEGRITY_ROW_ACTIONS = new Set([
    'start-enrichment-reindex',
    'start-repair',
]);

function _handleIntegrityClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_INTEGRITY_ACTIONS.has(action)) return;

    if (_INTEGRITY_ROW_ACTIONS.has(action)) {
        e.stopPropagation();
    }

    switch (action) {
        case 'start-integrity-check':
            startIntegrityCheck();
            break;
        case 'run-quick-check':
            runQuickCheck();
            break;
        case 'start-enrichment-reindex':
            startEnrichmentReindex(Number(el.dataset.dsId));
            break;
        case 'start-repair':
            startRepair(el.dataset.repairAction || '');
            break;
        case 'switch-to-retrieval':
            // <a href="#"> mit preventDefault, navigiert via switchAdminTab
            e.preventDefault();
            switchAdminTab('ki', 'ki-retrieval');
            break;
    }
}

function _bindIntegrityActions() {
    if (document.body._integrityActionsBound) return;
    document.body._integrityActionsBound = true;
    document.body.addEventListener('click', _handleIntegrityClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-integrity-actions-bound', 'true');
}

// --- Check-Definitionen (ID = Key aus run_full_check().checks) ---
const CHECK_DEFS = [
    { id: 'layer_counts',         label: 'Schicht-Zaehlung',      repairActions: ['repair_reconstruct', 'repair_reset_meta'] },
    { id: 'ds_coverage',          label: 'DS-Abdeckung',           repairActions: ['repair_ds_ids', 'repair_stats'] },
    { id: 'es_duplicates',        label: 'ES-Duplikate',           repairActions: [] },
    { id: 'orphan_chunks',        label: 'Verwaiste Chunks',       repairActions: ['repair_orphans'] },
    { id: 'overlapping_ds',       label: 'Ueberlappende DS-Pfade', repairActions: [] },
    { id: 'qdrant_fragmentation', label: 'Qdrant-Fragmentierung',  repairActions: ['repair_rebuild'] },
    { id: 'enrichment_prompt_consistency', label: 'Enrichment-Prompt-Konsistenz', repairActions: [] },
    { id: 'stale_colbert', label: 'Stale ColBERT-Collections', repairActions: ['repair_cleanup_colbert'] },
];

const repairLabels = {
    repair_ds_ids: 'ds_ids-Synchronisation',
    repair_orphans: 'Orphan-Cleanup',
    repair_stats: 'Stats-Korrektur',
    repair_reset_meta: 'Metadaten-Reset',
    repair_reconstruct: 'Metadaten-Rekonstruktion',
    repair_rebuild: 'Collection-Rebuild',
    repair_cleanup_colbert: 'ColBERT-Cleanup',
};

// --- Mapping action -> API-Endpoint + Bestaetigungstext ---
const REPAIR_CONFIG = {
    repair_ds_ids:      { endpoint: '/api/admin/integrity/repair/ds-ids',      confirm: 'ds_ids in Qdrant synchronisieren?' },
    repair_orphans:     { endpoint: '/api/admin/integrity/repair/orphans',     confirm: 'Verwaiste Chunks aus Qdrant loeschen?', danger: true },
    repair_stats:       { endpoint: '/api/admin/integrity/repair/stats',       confirm: 'Chunk-Statistiken korrigieren?' },
    repair_reset_meta:  { endpoint: '/api/admin/integrity/repair/reset-meta',  confirm: 'ALLE Metadaten loeschen (chunk_datasources, file_hashes, datasource_embeddings)?', danger: true },
    repair_reconstruct: { endpoint: '/api/admin/integrity/repair/reconstruct', confirm: 'Metadaten aus Qdrant-Daten rekonstruieren?' },
    repair_rebuild:     { endpoint: '/api/admin/integrity/repair/rebuild',     confirm: 'Fragmentierte Collections neu aufbauen (Snapshot/Restore)?', danger: true },
    repair_cleanup_colbert: { endpoint: '/api/admin/integrity/cleanup/colbert', confirm: 'Stale ColBERT-Collections bereinigen?', danger: true },
};

// --- Transformation: checks-Dict -> ExpandableTable-Rows ---

function checksToRows(checks) {
    return CHECK_DEFS.map(def => ({
        id: def.id,
        name: def.label,
        status: checks[def.id]?.status || 'pending',
        details: checks[def.id]?.details || checks[def.id]?.error || '',
        repairActions: def.repairActions,
        _raw: checks[def.id],
    }));
}

// --- ExpandableTable Config ---

const integrityTableConfig = {
    id: 'integrity',
    showHeader: true,
    expandable: true,
    gridTemplate: '30px 1fr 120px 200px',
    columns: [
        { field: 'name', label: 'Pruefung', sortable: false, align: 'left' },
        { field: 'status', label: 'Status', sortable: false, align: 'center',
          renderer: (v) => {
              const map = {
                  ok: ['badge-healthy', 'OK'],
                  warning: ['badge-degraded', 'Warning'],
                  critical: ['badge-unhealthy', 'Critical'],
                  error: ['badge-unhealthy', 'Fehler'],
                  pending: ['badge-default', 'Ausstehend'],
              };
              const [cls, text] = map[v] || ['badge-default', v];
              return `<span class="badge ${cls}">${text}</span>`;
          }
        },
        { field: 'id', label: 'Aktionen', sortable: false, align: 'center',
          renderer: (v, row) => renderRepairButtons(row)
        },
    ],
    detailRenderer: (row) => renderCheckDetail(row),
};


// --- Detail-Renderer pro Check-Typ ---

function renderCheckDetail(row) {
    const raw = row._raw || {};
    if (!raw || row.status === 'pending') return '<div class="no-data">Noch keine Daten</div>';

    switch (row.id) {
        case 'layer_counts': return renderLayerCountsDetail(raw);
        case 'ds_coverage': return renderDsCoverageDetail(raw);
        case 'es_duplicates': return renderEsDuplicatesDetail(raw);
        case 'orphan_chunks': return renderOrphanChunksDetail(raw);
        case 'overlapping_ds': return renderOverlappingDsDetail(raw);
        case 'qdrant_fragmentation': return renderQdrantFragDetail(raw);
        case 'enrichment_prompt_consistency': return renderEnrichmentPromptConsistency(raw);
        case 'stale_colbert': return renderStaleColbertDetail(raw);
        default: return `<pre class="u-fs-11">${SharedUi.runtime.escapeHtml(JSON.stringify(raw, null, 2))}</pre>`;
    }
}

function renderLayerCountsDetail(lc) {
    let html = '';
    // Collections-Detail
    if (lc.collections && lc.collections.length > 0) {
        html += '<table class="u-detail-table-sm u-mb-8">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-6">Qdrant-Collection</th>';
        html += '<th class="u-cell-left-6">Modell</th>';
        html += '<th class="u-cell-right-6">Chunks</th></tr>';
        lc.collections.forEach(c => {
            html += '<tr class="u-cell-border">';
            html += `<td class="u-pad-6 u-mono u-fs-11">${SharedUi.runtime.escapeHtml(c.name)}</td>`;
            html += `<td class="u-cell-fs-11">${SharedUi.runtime.escapeHtml(c.model)}</td>`;
            html += `<td class="u-cell-right-6-mono">${(c.chunks || 0).toLocaleString('de-DE')}</td></tr>`;
        });
        html += '</table>';
    }
    // Schicht-Tabelle
    html += '<table class="u-detail-table">';
    html += '<tr class="u-cell-border-strong">';
    html += '<th class="u-cell-left-8">Schicht</th>';
    html += '<th class="u-cell-right-8">Anzahl</th></tr>';
    const layers = [
        ['Qdrant (alle Collections)', lc.qdrant_total],
        ['  davon Dense (kb_*)', lc.qdrant_dense_total],
        ['  davon ColBERT (colbert_*)', lc.qdrant_colbert_total],
        ['chunk_datasources (distinct)', lc.chunk_ds_count],
        ['chunk_datasources (Keyspace)', lc.chunk_ds_keyspace_distinct_chunks],
        ['chunk_datasources (Zuordnungen)', lc.chunk_ds_total_rows],
        ['  davon Dense', lc.dense_chunk_ds_total_rows],
        ['  davon ColBERT', lc.colbert_chunk_ds_total_rows],
        ['datasource_embeddings (Summe)', lc.ds_embeddings_total],
        ['  davon Dense', lc.dense_ds_embeddings_total],
        ['  davon ColBERT', lc.colbert_ds_embeddings_total],
        ['file_hashes', lc.file_hashes_count],
    ];
    if (lc.es_configured) {
        layers.splice(4, 0, ['ES-Primärdokumente (physische Kopien)', lc.es_count]);
    }
    layers.forEach(([label, count]) => {
        html += `<tr class="u-cell-border">`;
        html += `<td class="u-pad-8">${SharedUi.runtime.escapeHtml(label)}</td>`;
        html += `<td class="u-cell-right-mono">${count == null ? 'nicht verfügbar' : count.toLocaleString('de-DE')}</td></tr>`;
    });
    html += '</table>';
    if (lc.details) html += `<p class="u-text-xs-secondary u-mt-8">${SharedUi.runtime.escapeHtml(lc.details)}</p>`;
    return html;
}

function renderDsCoverageDetail(dc) {
    let html = '';
    if (dc.es_total > 0) {
        const withDs = dc.es_total - dc.es_without_ds_ids;
        const pct = Math.round(withDs / dc.es_total * 100);
        html += `<p class="u-fs-13">ES: <strong>${pct}%</strong> der Dokumente haben ds_ids `;
        html += `(${withDs.toLocaleString('de-DE')} von ${dc.es_total.toLocaleString('de-DE')})</p>`;
    }
    if (dc.ds_warnings > 0) {
        html += `<p class="u-fs-13 u-c-warning u-mt-4">${dc.ds_warnings} Datenquelle(n) mit Inkonsistenzen</p>`;
    }
    if (dc.colbert_warning_count > 0) {
        html += `<p class="u-fs-13 u-c-warning u-mt-4">`;
        html += `${dc.colbert_warning_count} Datenquelle(n) mit ColBERT aktiviert aber ohne ColBERT-Chunks`;
        if (dc.colbert_warnings && dc.colbert_warnings.length > 0) {
            const names = dc.colbert_warnings.map(d => `DS ${d.id} (${SharedUi.runtime.escapeHtml(d.name)})`).join(', ');
            html += `: ${names}`;
        }
        html += '</p>';
    }
    if (dc.datasources && dc.datasources.length > 0) {
        html += '<table class="u-detail-table u-mt-8">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-8">DS</th>';
        html += '<th class="u-cell-left-8">Name</th>';
        html += '<th class="u-cell-left-8">Modell</th>';
        html += '<th class="u-cell-right-8" title="Tatsaechliche Chunk-Zuordnungen in chunk_datasources">Chunks (Ist)</th>';
        html += '<th class="u-cell-right-8" title="Chunk-Count aus datasource_embeddings (Ingestion-Statistik)">Chunks (Soll)</th>';
        html += '<th class="u-cell-right-8" title="Differenz Ist - Soll">Diff</th>';
        html += '<th class="u-cell-right-8" title="Indexierte Dateien in file_hashes">Dateien</th>';
        html += '<th class="u-cell-center-8">Status</th></tr>';
        dc.datasources.forEach(ds => {
            const isColbert = ds.kind === 'colbert';
            const isWarn = ds.status === 'warning';
            const rowCls = isWarn ? 'u-cell-border u-row-warn' : 'u-cell-border';
            let diffStr, diffCls, deStr, fhStr;
            if (isColbert) {
                diffStr = 'n/a';
                diffCls = 'u-cell-right-mono';
                deStr = 'n/a';
                fhStr = 'n/a';
            } else {
                const diff = (ds.chunk_ds_count || 0) - (ds.de_chunk_count || 0);
                diffStr = diff === 0 ? '-' : (diff > 0 ? '+' : '') + diff.toLocaleString('de-DE');
                diffCls = diff !== 0 ? 'u-cell-right-mono u-c-warning u-bold' : 'u-cell-right-mono';
                deStr = (ds.de_chunk_count || 0).toLocaleString('de-DE');
                fhStr = (ds.fh_count || 0).toLocaleString('de-DE');
            }
            const badge = isColbert
                ? '<span class="badge badge-default">ColBERT</span>'
                : (isWarn ? '<span class="badge badge-degraded">Warning</span>' : '<span class="badge badge-healthy">OK</span>');
            html += `<tr class="${rowCls}">`;
            html += `<td class="u-pad-8">${ds.id}</td>`;
            html += `<td class="u-pad-8">${SharedUi.runtime.escapeHtml(ds.name)}</td>`;
            html += `<td class="u-cell-fs-11-pad8">${SharedUi.runtime.escapeHtml(ds.embedding_model || '')}</td>`;
            html += `<td class="u-cell-right-mono">${(ds.chunk_ds_count || 0).toLocaleString('de-DE')}</td>`;
            html += `<td class="u-cell-right-mono">${deStr}</td>`;
            html += `<td class="${diffCls}">${diffStr}</td>`;
            html += `<td class="u-cell-right-mono">${fhStr}</td>`;
            html += `<td class="u-cell-center-pad">${badge}</td>`;
            html += '</tr>';
            if (isWarn && ds.issues && ds.issues.length > 0) {
                html += `<tr class="${rowCls}"><td colspan="8" class="u-cell-issue-indent">`;
                ds.issues.forEach(issue => { html += `&bull; ${SharedUi.runtime.escapeHtml(issue)}<br>`; });
                html += '</td></tr>';
            }
        });
        html += '</table>';
        html += '<p class="u-text-xxs-secondary u-mt-6">';
        html += 'Ist = chunk_datasources (tatsaechlich), Soll = datasource_embeddings (Ingestion-Statistik), Dateien = file_hashes';
        html += '</p>';
    }
    return html;
}

function renderEsDuplicatesDetail(ed) {
    let html = '';
    if (!ed.es_configured) return '<p class="u-text-sm-secondary">Elasticsearch nicht konfiguriert</p>';
    if (!ed.top_sources || ed.top_sources.length === 0) return '<p class="u-fs-13">Keine Quellen gefunden</p>';

    if (ed.suspected_duplicates > 0) {
        html += `<p class="u-c-warning u-fs-13">${ed.suspected_duplicates} Dateien mit vielen Einträgen je Datenquelle (&gt;5000; Diagnosehinweis)</p>`;
    }
    html += '<table class="u-detail-table-sm">';
    html += '<tr class="u-cell-border-strong">';
    html += '<th class="u-cell-left-6">Quelle</th>';
    html += '<th class="u-cell-right-6">Eintraege</th></tr>';
    ed.top_sources.forEach(s => {
        const warnCls = s.count > 5000 ? ' u-c-warning u-bold' : '';
        html += `<tr class="u-cell-border">`;
        html += `<td class="u-cell-source-break">DS ${SharedUi.runtime.escapeHtml(s.datasource_id)}: ${SharedUi.runtime.escapeHtml(s.source)}</td>`;
        html += `<td class="u-cell-right-6-mono${warnCls}">${s.count.toLocaleString('de-DE')}</td></tr>`;
    });
    html += '</table>';
    html += `<p class="u-text-xxs-secondary u-mt-4">${ed.total_unique_sources.toLocaleString('de-DE')} geschätzte Datenquelle/Datei-Paare, ${ed.total_docs.toLocaleString('de-DE')} physische ES-Primärdokumente</p>`;
    if (ed.suspected_duplicates > 0) {
        html += '<div class="u-divider-top-8">';
        html += '<span class="u-text-xs-secondary">Hohe Zahlen allein sind kein Duplikatnachweis. Rebuild nur unter Offline-Wartung: </span>';
        html += '<a href="#" data-action="switch-to-retrieval" class="u-fs-12">siehe Elasticsearch-Verwaltung</a>';
        html += '</div>';
    }
    return html;
}

function renderOrphanChunksDetail(oc) {
    let html = '';
    if (oc.status === 'error') {
        return `<p class="u-fs-13 u-c-warning">Pruefung fehlgeschlagen: ${SharedUi.runtime.escapeHtml(oc.error || 'Unbekannter Fehler')}</p>`;
    }
    if (oc.sampled != null) {
        html += `<p class="u-fs-13">Stichprobe: ${oc.sampled.toLocaleString('de-DE')} Chunks, `;
        html += `davon <strong>${oc.orphans_found.toLocaleString('de-DE')}</strong> ohne DS-Zuordnung (${oc.orphan_pct}%)</p>`;
        if (oc.sample_orphans && oc.sample_orphans.length > 0) {
            html += '<details><summary class="u-clickable u-fs-12">Beispiele anzeigen</summary>';
            html += '<ul class="u-fs-11 u-mono u-mt-4">';
            oc.sample_orphans.forEach(o => {
                html += `<li>${SharedUi.runtime.escapeHtml(o.source)} (${SharedUi.runtime.escapeHtml(o.chunk_id.substring(0, 12))}...)</li>`;
            });
            html += '</ul></details>';
        }
    }
    return html;
}

function renderOverlappingDsDetail(od) {
    if (!od.overlaps || od.overlaps.length === 0) return '<p class="u-fs-13">Keine Ueberlappungen</p>';
    let html = '<table class="u-detail-table">';
    html += '<tr class="u-cell-border-strong">';
    html += '<th class="u-cell-left-8">DS A</th>';
    html += '<th class="u-cell-left-8">DS B</th>';
    html += '<th class="u-cell-left-8">Beziehung</th></tr>';
    od.overlaps.forEach(o => {
        const rel = o.relationship === 'identical' ? 'Identisch' : 'Unterverzeichnis';
        html += `<tr class="u-cell-border">`;
        html += `<td class="u-pad-8">${o.ds_a.id}: ${SharedUi.runtime.escapeHtml(o.ds_a.name)}</td>`;
        html += `<td class="u-pad-8">${o.ds_b.id}: ${SharedUi.runtime.escapeHtml(o.ds_b.name)}</td>`;
        html += `<td class="u-pad-8">${rel}</td></tr>`;
    });
    html += '</table>';
    return html;
}

function renderQdrantFragDetail(qf) {
    let html = '';
    if (qf.status === 'error') {
        return `<p class="u-fs-13 u-c-warning">Pruefung fehlgeschlagen: ${SharedUi.runtime.escapeHtml(qf.error || 'Unbekannter Fehler')}</p>`;
    }
    if (!qf.collections || qf.collections.length === 0) return '<p class="u-fs-13">Keine Collections</p>';

    html += '<table class="u-detail-table">';
    html += '<tr class="u-cell-border-strong">';
    html += '<th class="u-cell-left-8">Collection</th>';
    html += '<th class="u-cell-right-8">Chunks</th>';
    html += '<th class="u-cell-right-8">Segmente</th>';
    html += '<th class="u-cell-right-8">Speicher</th>';
    html += '<th class="u-cell-center-8">Status</th></tr>';
    qf.collections.forEach(c => {
        const frag = c.fragmented;
        const rowCls = frag ? 'u-cell-border u-row-warn' : 'u-cell-border';
        const badge = frag ? '<span class="badge badge-degraded">Fragmentiert</span>' : '<span class="badge badge-healthy">OK</span>';
        const diskStr = c.disk_mb > 1024 ? (c.disk_mb / 1024).toFixed(1) + ' GB' : c.disk_mb + ' MB';
        html += `<tr class="${rowCls}">`;
        html += `<td class="u-cell-fs-12 u-mono">${SharedUi.runtime.escapeHtml(c.name)}</td>`;
        html += `<td class="u-cell-right-mono">${(c.chunks || 0).toLocaleString('de-DE')}</td>`;
        html += `<td class="u-cell-right-mono">${c.segments}</td>`;
        html += `<td class="u-cell-right-mono">${diskStr}</td>`;
        html += `<td class="u-cell-center-pad">${badge}</td></tr>`;
    });
    html += '</table>';
    return html;
}


function renderEnrichmentPromptConsistency(raw) {
    let html = '';
    if (raw.status === 'error') {
        return `<p class="u-fs-13 u-c-warning">Pruefung fehlgeschlagen: ${SharedUi.runtime.escapeHtml(raw.error || 'Unbekannter Fehler')}</p>`;
    }

    // Summary stats
    const total = raw.total_checked || 0;
    const consistent = raw.consistent || 0;
    const inconsistent = raw.inconsistent || 0;
    const notEnriched = raw.not_enriched || 0;

    const inconsistentCls = inconsistent > 0 ? 'u-c-warning' : '';
    html += '<div class="u-flex-wrap u-flex-gap-24 u-mb-12 u-fs-13">';
    html += `<span>Geprueft: <strong>${total.toLocaleString('de-DE')}</strong></span>`;
    html += `<span>Konsistent: <strong class="u-c-success">${consistent.toLocaleString('de-DE')}</strong></span>`;
    html += `<span>Inkonsistent: <strong class="${inconsistentCls}">${inconsistent.toLocaleString('de-DE')}</strong></span>`;
    html += `<span>Nicht angereichert: <strong>${notEnriched.toLocaleString('de-DE')}</strong></span>`;
    html += '</div>';

    // Per-DS detail table
    if (raw.details && raw.details.length > 0) {
        html += '<table class="u-detail-table">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-8">Datenquelle</th>';
        html += '<th class="u-cell-right-8">Konsistent</th>';
        html += '<th class="u-cell-right-8">Inkonsistent</th>';
        html += '<th class="u-cell-right-8">Nicht angereichert</th>';
        html += '<th class="u-cell-left-8">Prompt-Hashes</th>';
        html += '<th class="u-cell-center-8">Aktion</th></tr>';
        raw.details.forEach(ds => {
            const dsInconsistent = ds.inconsistent || 0;
            const rowCls = dsInconsistent > 0 ? 'u-cell-border u-row-warn' : 'u-cell-border';
            const diffCls = dsInconsistent > 0 ? 'u-cell-right-mono u-c-warning u-bold' : 'u-cell-right-mono';
            html += `<tr class="${rowCls}">`;
            html += `<td class="u-pad-8">${SharedUi.runtime.escapeHtml(ds.datasource_name || ('DS ' + ds.datasource_id))}</td>`;
            html += `<td class="u-cell-right-mono">${(ds.consistent || 0).toLocaleString('de-DE')}</td>`;
            html += `<td class="${diffCls}">${dsInconsistent.toLocaleString('de-DE')}</td>`;
            html += `<td class="u-cell-right-mono">${(ds.not_enriched || 0).toLocaleString('de-DE')}</td>`;
            // Prompt hashes breakdown
            let hashInfo = '-';
            if (ds.found_hashes && ds.found_hashes.length > 0) {
                hashInfo = ds.found_hashes.map(h =>
                    `<span class="u-mono u-fs-11" title="${SharedUi.runtime.escapeAttr(h.hash)}">${SharedUi.runtime.escapeHtml(h.hash.substring(0, 8))}...</span> (${h.count})`
                ).join(', ');
            }
            html += `<td class="u-cell-fs-12">${hashInfo}</td>`;
            // Reindex button if inconsistent
            if (dsInconsistent > 0) {
                html += `<td class="u-cell-center-pad">`;
                html += `<button class="action-btn-sm u-c-warning" `
                     + `data-action="start-enrichment-reindex" data-ds-id="${ds.datasource_id}">Full-Reindex starten</button>`;
                html += `</td>`;
            } else {
                html += `<td class="u-cell-center-pad">-</td>`;
            }
            html += '</tr>';
        });
        html += '</table>';
    } else {
        html += '<p class="u-text-sm-secondary">Keine Datenquellen-Details verfuegbar</p>';
    }
    return html;
}

function renderStaleColbertDetail(data) {
    const stale = data.stale_collections || [];
    const orphaned = data.orphaned_db_entries || [];

    if (stale.length === 0 && orphaned.length === 0) {
        if (data.status === 'warning' || data.status === 'error') {
            return `<p class="u-text-sm-secondary">${SharedUi.runtime.escapeHtml(data.details || data.error || 'Stale-ColBERT-Pruefung nicht moeglich')}</p>`;
        }
        return '<p class="u-text-sm-secondary">Keine stale ColBERT-Daten</p>';
    }

    let html = '';
    if (stale.length > 0) {
        html += '<h5 class="u-heading-tight">Stale Collections</h5>';
        html += '<table class="u-detail-table-sm u-mb-10">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-6">Collection</th>';
        html += '<th class="u-cell-left-6">Modell</th>';
        html += '<th class="u-cell-right-6">Chunks</th>';
        html += '<th class="u-cell-left-6">Reason</th>';
        html += '<th class="u-cell-left-6">Datasources</th></tr>';
        stale.forEach(c => {
            const dsNames = (c.datasources || []).map(d =>
                `${SharedUi.runtime.escapeHtml(d.name || '')} (${d.id})`
            ).join(', ') || '-';
            html += '<tr class="u-cell-border">';
            html += `<td class="u-pad-6 u-mono u-fs-11">${SharedUi.runtime.escapeHtml(c.name)}</td>`;
            html += `<td class="u-cell-fs-11">${SharedUi.runtime.escapeHtml(c.model)}</td>`;
            html += `<td class="u-cell-right-6-mono">${(c.chunks || 0).toLocaleString('de-DE')}</td>`;
            html += `<td class="u-pad-6"><span class="badge badge-degraded">${SharedUi.runtime.escapeHtml(c.reason)}</span></td>`;
            html += `<td class="u-cell-fs-11">${dsNames}</td></tr>`;
        });
        html += '</table>';
    }
    if (orphaned.length > 0) {
        html += '<h5 class="u-heading-tight">Verwaiste DB-Eintraege</h5>';
        html += '<table class="u-detail-table-sm u-mb-10">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-6">Modell</th>';
        html += '<th class="u-cell-left-6">Reason</th>';
        html += '<th class="u-cell-left-6">Datasources</th></tr>';
        orphaned.forEach(e => {
            const dsNames = (e.datasources || []).map(d =>
                `${SharedUi.runtime.escapeHtml(d.name || '')} (${d.id})`
            ).join(', ') || '-';
            html += '<tr class="u-cell-border">';
            html += `<td class="u-cell-fs-11">${SharedUi.runtime.escapeHtml(e.model)}</td>`;
            html += `<td class="u-pad-6"><span class="badge badge-degraded">${SharedUi.runtime.escapeHtml(e.reason)}</span></td>`;
            html += `<td class="u-cell-fs-11">${dsNames}</td></tr>`;
        });
        html += '</table>';
    }
    return html;
}

async function startEnrichmentReindex(datasourceId) {
    const confirmed = await SharedUi.modal.confirm({
        message: `Full-Reindex fuer Datenquelle ${datasourceId} starten? Alle Chunks werden neu indexiert und angereichert.`,
    });
    if (!confirmed) return;

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${datasourceId}/reingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.error || data.detail || 'Fehler');
        }
        SharedUi.notify.show(`Full-Reindex fuer DS ${datasourceId} gestartet`, 'success');
    } catch (err) {
        SharedUi.notify.show(err.message, 'error');
    }
}


// --- Repair-Buttons (conditional pro Check) ---

function renderRepairButtons(row) {
    if (row.status === 'ok' || row.status === 'pending') return '';
    const raw = row._raw || {};
    const buttons = [];

    if (row.id === 'layer_counts') {
        if (raw.qdrant_empty) {
            buttons.push({label: 'Metadaten-Reset', action: 'repair_reset_meta', danger: true});
        } else if ((raw.qdrant_dense_total || raw.qdrant_total) > 0 && raw.chunk_ds_count === 0) {
            buttons.push({label: 'Rekonstruieren', action: 'repair_reconstruct'});
        }
    }
    if (row.id === 'ds_coverage') {
        // Dense-only Repair-Buttons: nur bei Dense-Warnings oder ES-Gaps
        if (raw.ds_warnings > 0 || (raw.es_without_ds_ids > 0 && raw.es_total > 0)) {
            buttons.push({label: 'ds_ids sync', action: 'repair_ds_ids'});
            buttons.push({label: 'Stats fix', action: 'repair_stats'});
        }
        // Keine Repair-Buttons fuer reine ColBERT-Warnings (dense-only Repair-Pfade)
    }
    if (row.id === 'orphan_chunks' && raw.orphans_found > 0) {
        buttons.push({label: 'Cleanup', action: 'repair_orphans', danger: true});
    }
    if (row.id === 'qdrant_fragmentation' && raw.fragmented_count > 0) {
        buttons.push({label: 'Rebuild', action: 'repair_rebuild', danger: true});
    }
    if (row.id === 'stale_colbert'
        && ((raw.stale_collections || []).length > 0
            || (raw.orphaned_db_entries || []).length > 0)) {
        buttons.push({label: 'ColBERT-Cleanup', action: 'repair_cleanup_colbert', danger: true});
    }

    return buttons.map(b => {
        const cls = b.danger ? 'action-btn-sm u-c-warning' : 'action-btn-sm';
        return `<button class="${cls}" `
             + `data-action="start-repair" data-repair-action="${SharedUi.runtime.escapeAttr(b.action)}">${SharedUi.runtime.escapeHtml(b.label)}</button>`;
    }).join(' ');
}


// --- Generische startRepair Funktion ---

async function startRepair(action) {
    if (_integrityRepairLocked) return;
    const cfg = REPAIR_CONFIG[action];
    if (!cfg) { SharedUi.notify.show(`Unbekannte Repair-Aktion: ${action}`, 'error'); return; }

    _integrityRepairLocked = true;
    let pollingStarted = false;
    try {
        const confirmed = await SharedUi.modal.confirm({
            message: cfg.confirm,
        });
        if (!confirmed) return;

        const resp = await SharedUi.fetch.safeFetch(cfg.endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.error || data.detail || 'Fehler');
        }
        SharedUi.notify.show(`${repairLabels[action] || action} gestartet`, 'success');
        pollIntegrityRepair();
        pollingStarted = true;
    } catch (err) {
        SharedUi.notify.show(err.message, 'error');
    } finally {
        // Lock nur freigeben, wenn Polling NICHT gestartet wurde — pollIntegrityRepair
        // verwaltet den Lock waehrend des Repair-Laufs selbst.
        if (!pollingStarted) _integrityRepairLocked = false;
    }
}


// --- Init (Einstiegsfunktion, Pattern: tab_datenquellen.js) ---

async function initIntegrityTab() {
    // Re-Init-Schutz
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-integrity');
    _integrityInitSeq++;
    _integrityRepairLocked = false;
    const mySeq = _integrityInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;

    _bindIntegrityActions();
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    // Cleanup registrieren
    registerTabCleanup('system-integrity', () => {
        _integrityInitSeq++;
        kiaraPolling.stop('system-integrity:status');
        kiaraPolling.stop('system-integrity:repair');
        _integrityRepairLocked = false;
        integrityTable = null;
    });

    container.innerHTML = `
        <div class="config-form">
            <h3>Integritaetspruefung</h3>
            <div class="config-section"><h4>Pruefung</h4>
                <div class="u-flex u-items-center u-flex-gap-12 u-flex-wrap">
                    <button class="action-btn primary" data-action="start-integrity-check" id="btnIntegrity">
                        Vollpruefung starten</button>
                    <button class="action-btn" data-action="run-quick-check" id="btnQuickCheck">
                        Schnell-Check</button>
                    <span id="integrityProgress" class="u-text-sm-secondary"></span>
                </div>
            </div>
            <div id="integrityTableContainer"></div>
            <div id="quickCheckResult"></div>
        </div>`;

    integrityTable = new SharedUi.table.ExpandableTable(integrityTableConfig, 'integrityTableContainer');

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/integrity/status');
        if (mySeq !== _integrityInitSeq) return;
        if (!resp.ok) {
            container.innerHTML = '<div class="no-data">Integritaetsstatus konnte nicht geladen werden.</div>';
            return;
        }
        const data = await resp.json();
        const taskData = data.data || data;
        if (taskData.running) {
            document.getElementById('btnIntegrity').disabled = true;
            document.getElementById('btnQuickCheck').disabled = true;
            if (taskData.action?.startsWith('repair_')) {
                _integrityRepairLocked = true;
                pollIntegrityRepair();
            } else {
                pollIntegrityStatus();
            }
        }
        if (taskData.result?.checks) {
            const rows = checksToRows(taskData.result.checks);
            integrityTable.setData(rows, rows.length);
        } else if (taskData.result?.action) {
            const progress = document.getElementById('integrityProgress');
            if (progress) {
                const statusIcon = taskData.result.status === 'success' ? 'success' : 'error';
                SharedUi.notify.show(taskData.result.message || taskData.result.error || '', statusIcon);
            }
        }
    } catch (e) {
        if (mySeq !== _integrityInitSeq) return;
        container.innerHTML = '<div class="no-data">Integritaetsdaten konnten nicht geladen werden.</div>';
    }
}


// --- Actions ---

async function startIntegrityCheck() {
    const btn = document.getElementById('btnIntegrity');
    const progress = document.getElementById('integrityProgress');
    if (btn) { btn.disabled = true; btn.textContent = 'Pruefung laeuft...'; }
    if (progress) progress.textContent = 'Wird gestartet...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/integrity/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.error || data.detail || 'Fehler');
        }
        pollIntegrityStatus();
    } catch (e) {
        if (btn) { btn.disabled = false; btn.textContent = 'Vollpruefung starten'; }
        if (progress) progress.textContent = '';
        SharedUi.notify.show(e.message, 'error');
    }
}

async function runQuickCheck() {
    const btn = document.getElementById('btnQuickCheck');
    const resultDiv = document.getElementById('quickCheckResult');
    if (btn) btn.disabled = true;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/integrity/quick');
        const wrapper = await resp.json();
        if (!resp.ok) {
            throw new Error(wrapper.detail || wrapper.error || wrapper.message || 'Schnell-Check fehlgeschlagen');
        }
        const data = wrapper.data || wrapper;

        let html = '<div class="config-section"><h4>Schnell-Check: Schicht-Zaehlung</h4>';
        html += '<table class="u-detail-table">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-8">Schicht</th>';
        html += '<th class="u-cell-right-8">Anzahl</th></tr>';
        const rows = [
            ['Qdrant (alle Collections)', data.qdrant_total],
            ['  davon Dense (kb_*)', data.qdrant_dense_total],
            ['  davon ColBERT (colbert_*)', data.qdrant_colbert_total],
            ['chunk_datasources (distinct)', data.chunk_ds_count],
            ['chunk_datasources (Keyspace)', data.chunk_ds_keyspace_distinct_chunks],
            ['chunk_datasources (Zuordnungen)', data.chunk_ds_total_rows],
            ['  davon Dense', data.dense_chunk_ds_total_rows],
            ['  davon ColBERT', data.colbert_chunk_ds_total_rows],
            ['file_hashes', data.file_hashes_count],
            ['datasource_embeddings (Summe)', data.ds_embeddings_total],
            ['  davon Dense', data.dense_ds_embeddings_total],
            ['  davon ColBERT', data.colbert_ds_embeddings_total],
        ];
        if (data.es_configured) rows.splice(4, 0, ['ES-Primärdokumente (physische Kopien)', data.es_count]);
        rows.forEach(([label, count]) => {
            html += `<tr class="u-cell-border">`;
            html += `<td class="u-pad-8">${SharedUi.runtime.escapeHtml(label)}</td>`;
            html += `<td class="u-cell-right-mono">${count == null ? 'nicht verfügbar' : count.toLocaleString('de-DE')}</td></tr>`;
        });
        html += '</table>';
        const statusValue = data.status || 'unknown';
        const statusBadge = statusValue === 'ok' ? 'badge-healthy' : 'badge-degraded';
        html += `<p class="u-mt-8"><span class="badge ${statusBadge}">${statusValue.toUpperCase()}</span> ${SharedUi.runtime.escapeHtml(data.details || '')}</p>`;
        html += '</div>';
        if (resultDiv) resultDiv.innerHTML = html;
    } catch (e) {
        SharedUi.notify.show('Schnell-Check fehlgeschlagen: ' + e.message, 'error');
    } finally {
        if (btn) btn.disabled = false;
    }
}


// --- Polling ---

function pollIntegrityStatus() {
    const mySeq = _integrityInitSeq;
    kiaraPolling.stop('system-integrity:status');
    kiaraPolling.start('system-integrity:status', async () => {
        if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:status'); return; }
        let resp;
        try {
            resp = await SharedUi.fetch.safeFetch('/api/admin/integrity/status');
        } catch (err) {
            if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:status'); return; }
            // Poll-Fehler: sauber stoppen, Buttons/Progress resetten
            kiaraPolling.stop('system-integrity:status');
            const btn = document.getElementById('btnIntegrity');
            const btnQ = document.getElementById('btnQuickCheck');
            const progress = document.getElementById('integrityProgress');
            if (btn) { btn.disabled = false; btn.textContent = 'Vollpruefung starten'; }
            if (btnQ) btnQ.disabled = false;
            if (progress) progress.textContent = '';
            SharedUi.notify.show('Statusabfrage fehlgeschlagen (HTTP ' + ((err && err.status) || 0) + ')', 'error');
            return;
        }
        if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:status'); return; }
        const data = await resp.json();
        if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:status'); return; }
        const taskData = data.data || data;
        const progress = document.getElementById('integrityProgress');
        if (!taskData.running) {
            kiaraPolling.stop('system-integrity:status');
            if (mySeq !== _integrityInitSeq) return;
            if (taskData.result && taskData.result.status === 'error') {
                SharedUi.notify.show('Pruefung fehlgeschlagen: ' + taskData.result.error, 'error');
            } else {
                SharedUi.notify.show('Integritaetspruefung abgeschlossen', 'success');
            }
            const btn = document.getElementById('btnIntegrity');
            const btnQ = document.getElementById('btnQuickCheck');
            if (btn) { btn.disabled = false; btn.textContent = 'Vollpruefung starten'; }
            if (btnQ) btnQ.disabled = false;
            if (progress) progress.textContent = '';
            if (taskData.result?.checks && integrityTable) {
                const rows = checksToRows(taskData.result.checks);
                integrityTable.setData(rows, rows.length);
            }
        } else if (progress && taskData.progress) {
            progress.textContent = `Schritt ${taskData.progress.step}/${taskData.progress.total}: ${taskData.progress.label}`;
        }
    }, 2000);
}

function pollIntegrityRepair() {
    const mySeq = _integrityInitSeq;
    kiaraPolling.stop('system-integrity:repair');

    const btn = document.getElementById('btnIntegrity');
    const btnQ = document.getElementById('btnQuickCheck');
    if (btn) btn.disabled = true;
    if (btnQ) btnQ.disabled = true;

    const progress = document.getElementById('integrityProgress');
    if (progress) progress.textContent = 'Repair laeuft...';

    kiaraPolling.start('system-integrity:repair', async () => {
        if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:repair'); return; }
        let resp;
        try {
            resp = await SharedUi.fetch.safeFetch('/api/admin/integrity/status');
        } catch (err) {
            if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:repair'); return; }
            kiaraPolling.stop('system-integrity:repair');
            _integrityRepairLocked = false;
            if (btn) { btn.disabled = false; btn.textContent = 'Vollpruefung starten'; }
            if (btnQ) btnQ.disabled = false;
            if (progress) progress.textContent = '';
            SharedUi.notify.show('Repair-Statusabfrage fehlgeschlagen (HTTP ' + ((err && err.status) || 0) + ')', 'error');
            return;
        }
        if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:repair'); return; }
        const data = await resp.json();
        if (mySeq !== _integrityInitSeq) { kiaraPolling.stop('system-integrity:repair'); return; }
        const taskData = data.data || data;

        if (progress && taskData.progress) {
            progress.textContent = taskData.progress.label || 'Repair laeuft...';
        }

        if (!taskData.running) {
            kiaraPolling.stop('system-integrity:repair');
            _integrityRepairLocked = false;
            if (mySeq !== _integrityInitSeq) return;
            if (btn) { btn.disabled = false; btn.textContent = 'Vollpruefung starten'; }
            if (btnQ) btnQ.disabled = false;
            if (progress) progress.textContent = '';

            if (taskData.result && taskData.result.status === 'error') {
                SharedUi.notify.show('Repair fehlgeschlagen: ' + taskData.result.error, 'error');
            } else if (taskData.result) {
                SharedUi.notify.show(taskData.result.message || 'Repair abgeschlossen', 'success');
            }
            try {
                const statusResp = await SharedUi.fetch.safeFetch('/api/admin/integrity/status');
                if (mySeq !== _integrityInitSeq) return;
                const statusData = await statusResp.json();
                const freshData = statusData.data || statusData;
                if (freshData.result?.checks && integrityTable) {
                    const rows = checksToRows(freshData.result.checks);
                    integrityTable.setData(rows, rows.length);
                }
            } catch(e) { /* ignore fresh-data fetch errors */ }
        }
    }, 2000);
}
