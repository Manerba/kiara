/**
 * KIara Admin - Datenquellen Tab (Multi-Datasource)
 * CRUD table, Bearbeiten-Modal, Gruppen-Zuweisung, Ingestion (async + Polling).
 */

let dsTable = null;
let _dsInitSeq = 0;
let _connectorTypes = null;  // Cache: Konnektor-Schemas vom Backend
let _currentDsId = null;     // DS-ID des aktuell geoeffneten Modals (null = Create)
let _selectedEmbedModel = '';  // wird in _loadEmbeddingModels() auf Default gesetzt
let _defaultEmbedModel = '';   // Standard-Modell aus ki_backends (Fallback fuer Ingestion)
let _embedModelLateChunking = {};  // { modelName: bool } — Late-Chunking-Support pro Modell
const DS_PASSWORD_MASK = '********';  // Muss mit PASSWORD_MASK aus constants.py uebereinstimmen
let _editOpenToken = 0;  // Invalidation-Token fuer spaete async-Responses im Edit-Dialog
const _dsActiveToggleInFlight = new Set();
const _DS_RATE_WINDOW_MS = 30_000;
const _DS_RATE_MIN_MS = 6_000;
const _dsRateHistory = new Map();
const _HTML_REINGEST_CONNECTORS = new Set(['local', 'smb', 'webdav', 'imap', 'exchange_online']);
const _HTML_REINGEST_MODELS = new Set(['nomic-embed-text', 'hellord/e5-mistral-7b-instruct:q4_0']);

// Nur akzeptierte Datenantworten zaehlen als Beobachtung, niemals Render-Aufrufe.
// Der erste Fortschrittsstand enthaelt ggf. alte/uebersprungene Dateien und
// bildet ausschliesslich die Basis fuer nachfolgende Differenzen.
function _updateDsRates(rows, now = performance.now()) {
    const present = new Set(rows.map(row => row.id));
    for (const id of _dsRateHistory.keys()) {
        if (!present.has(id)) _dsRateHistory.delete(id);
    }
    for (const row of rows) {
        const count = row.progress_processed;
        const total = row.progress_total;
        const startedAt = _parseDsTimestamp(row.started_at)?.getTime();
        if (row.ingestion_status !== 'running'
                || !Number.isSafeInteger(row.job_id) || row.job_id <= 0
                || !Number.isSafeInteger(count) || count < 0
                || !Number.isSafeInteger(total) || total <= 0 || count > total
                || !Number.isFinite(startedAt) || !Number.isFinite(now)) {
            _dsRateHistory.delete(row.id);
            continue;
        }
        const key = `${row.job_id}:${startedAt}:${total}:${row.phase || ''}`;
        let state = _dsRateHistory.get(row.id);
        const last = state?.samples[state.samples.length - 1];
        if (!state || state.key !== key || count < last.count
                || now < last.at || now - last.at > _DS_RATE_WINDOW_MS) {
            state = {key, samples: [{at: now, count}], rate: null};
            _dsRateHistory.set(row.id, state);
            continue;
        }
        if (now === last.at) continue;
        state.samples.push({at: now, count});
        state.samples = state.samples.filter(sample => now - sample.at <= _DS_RATE_WINDOW_MS);
        const first = state.samples[0];
        const elapsed = now - first.at;
        state.rate = elapsed >= _DS_RATE_MIN_MS ? (count - first.count) * 1000 / elapsed : null;
    }
}

function _getDsRate(row, now = performance.now()) {
    const state = _dsRateHistory.get(row.id);
    const startedAt = _parseDsTimestamp(row.started_at)?.getTime();
    const key = `${row.job_id}:${startedAt}:${row.progress_total}:${row.phase || ''}`;
    if (row.ingestion_status !== 'running' || !state || state.key !== key
            || now - state.samples[state.samples.length - 1].at > _DS_RATE_WINDOW_MS) return null;
    return state.rate;
}

function _parseDsTimestamp(value) {
    if (!value) return null;
    let s = String(value).replace(' ', 'T');
    s = s.replace(/([+-]\d{2})$/, '$1:00');
    return new Date(s);
}

// =========================================================================
//  Event-Delegation (CSP-Strict, V30 Welle 2)
// =========================================================================

const _DS_ACTIONS = new Set([
    'open-create-datasource-modal',
    'pause-ds-ingestion',
    'cancel-ds-ingestion',
    'trigger-ds-ingest',
    'trigger-ds-reingest',
    'trigger-ds-html-reingest',
    'open-edit-datasource-modal',
    'open-ds-groups-modal',
    'delete-datasource',
]);

// Row-Actions: stopPropagation im ExpandableTable verhindert Row-Expand/
// Row-Click-Collision (V30: "stopPropagation nur dort behalten wo heute
// Row-Click, Expand oder Link-Navigation bewusst geblockt wird").
const _DS_ROW_ACTIONS = new Set([
    'pause-ds-ingestion',
    'cancel-ds-ingestion',
    'trigger-ds-ingest',
    'trigger-ds-reingest',
    'trigger-ds-html-reingest',
    'open-edit-datasource-modal',
    'open-ds-groups-modal',
    'delete-datasource',
]);

const _DS_CHANGE_ACTIONS = new Set([
    'select-embed-model',
    'ds-type-change',
    'ds-chunk-strategy-change',
    'ds-sync-interval-change',
    'toggle-datasource-active',
]);

const _DS_INPUT_ACTIONS = new Set([
    'update-ds-search',
]);

function _handleDsClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_DS_ACTIONS.has(action)) return;

    if (_DS_ROW_ACTIONS.has(action)) {
        e.stopPropagation();
    }

    switch (action) {
        case 'open-create-datasource-modal':
            openCreateDatasourceModal();
            break;
        case 'pause-ds-ingestion':
            pauseDsIngestion(el, Number(el.dataset.dsId));
            break;
        case 'cancel-ds-ingestion':
            cancelDsIngestion(el, Number(el.dataset.dsId), el.dataset.dsName || '');
            break;
        case 'trigger-ds-ingest':
            triggerDsIngest(Number(el.dataset.dsId), el.dataset.dsName || '');
            break;
        case 'trigger-ds-reingest':
            triggerDsReingest(Number(el.dataset.dsId), el.dataset.dsName || '');
            break;
        case 'trigger-ds-html-reingest':
            triggerDsHtmlReingest(Number(el.dataset.dsId), el.dataset.dsName || '');
            break;
        case 'open-edit-datasource-modal':
            openEditDatasourceModal(Number(el.dataset.dsId));
            break;
        case 'open-ds-groups-modal':
            openDsGroupsModal(Number(el.dataset.dsId), el.dataset.dsName || '');
            break;
        case 'delete-datasource':
            deleteDatasource(
                Number(el.dataset.dsId),
                el.dataset.dsName || '',
                el.dataset.dsActive === '1'
            );
            break;
    }
}

function _handleDsChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el) return;
    const action = el.dataset.changeAction;
    if (!_DS_CHANGE_ACTIONS.has(action)) return;

    switch (action) {
        case 'select-embed-model':
            _selectedEmbedModel = el.value;
            if (dsTable) dsTable.render();
            break;
        case 'ds-type-change':
            _onTypeChange(el.value);
            break;
        case 'ds-chunk-strategy-change':
            toggleChunkFields(el.value);
            break;
        case 'ds-sync-interval-change':
            _onSyncIntervalChange(el.value);
            break;
        case 'toggle-datasource-active':
            _toggleDatasourceActive(el, Number(el.dataset.dsId), el.checked);
            break;
    }
}

function _handleDsInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el) return;
    const action = el.dataset.inputAction;
    if (!_DS_INPUT_ACTIONS.has(action)) return;

    if (action === 'update-ds-search') {
        if (dsTable) dsTable.setFilter('search', el.value);
    }
}

function _bindDsActions() {
    // Einziger Delegations-Root: document.body. Deckt contentContainer,
    // filterBar UND von SharedUi.modal an body gehaengte .modal-overlay ab.
    // Kein Mehrfach-Dispatching durch bubbling-Ebenen (siehe Fix in
    // tab_ki_modelle.js, 0.0.1+376).
    if (document.body._dsActionsBound) return;
    document.body._dsActionsBound = true;
    document.body.addEventListener('click', _handleDsClick);
    document.body.addEventListener('change', _handleDsChange);
    document.body.addEventListener('input', _handleDsInput);

    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-ds-actions-bound', 'true');
}

// Gueltige UI-Presets fuer sync_interval
const _VALID_SYNC_PRESETS = new Set(['global', 'manual', '30min', '1h', '2h', '6h', '12h', '24h', 'custom']);

// Legacy-/Fremdwert-Mapping auf gueltige UI-Presets
const _LEGACY_INTERVAL_MAP = { hourly: '1h', daily: '24h' };

/**
 * Normalisiert einen sync_interval-Wert auf einen gueltigen UI-Preset.
 * Bekannte Legacy-Werte (hourly, daily) werden gemappt.
 * Unbekannte Werte liefern { value: 'custom', needsReview: true, original }.
 */
/**
 * Extrahiert den besten Fehlertext aus einer JSON-Antwort.
 * Admin-Router kann { detail }, { error }, { message } oder Kombinationen liefern.
 */
function _dsResolveError(obj, fallback) {
    if (!obj) return fallback || 'Fehler';
    return obj.detail || obj.error || obj.message || fallback || 'Fehler';
}

/**
 * Remaskiert Secret-Felder in einem Draft-Objekt auf DS_PASSWORD_MASK.
 * Verhindert, dass Klartext-Secrets in den Reopen-DOM gelangen.
 */
function _remaskDraftSecrets(draft) {
    if (!draft || !draft.connector_config || !draft.type) return;
    const types = _connectorTypes || {};
    const schema = types[draft.type]?.schema;
    // Fail-Safe: Schema fehlt (z.B. _loadConnectorTypes lieferte leeres Cache)
    // — pauschal alle Werte maskieren, da wir nicht wissen, welche Secrets sind.
    if (!schema || Object.keys(schema).length === 0) {
        for (const fieldName of Object.keys(draft.connector_config)) {
            const val = draft.connector_config[fieldName];
            if (typeof val === 'string' && val && val !== DS_PASSWORD_MASK) {
                draft.connector_config[fieldName] = DS_PASSWORD_MASK;
            }
        }
        return;
    }
    for (const [fieldName, spec] of Object.entries(schema)) {
        if (spec.secret && draft.connector_config[fieldName] && draft.connector_config[fieldName] !== DS_PASSWORD_MASK) {
            draft.connector_config[fieldName] = DS_PASSWORD_MASK;
        }
    }
}

function _normalizeSyncInterval(raw) {
    if (!raw) return { value: 'global', needsReview: false };
    const v = String(raw).trim().toLowerCase();
    if (_VALID_SYNC_PRESETS.has(v)) return { value: v, needsReview: false };
    if (_LEGACY_INTERVAL_MAP[v]) return { value: _LEGACY_INTERVAL_MAP[v], needsReview: false };
    // Nicht-mapbare Fremdwerte -> Review
    return { value: 'custom', needsReview: true, original: raw };
}

async function _loadConnectorTypes() {
    if (_connectorTypes) return _connectorTypes;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/connector-types',
            {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        const data = await resp.json();
        _connectorTypes = {};
        (data.data || []).forEach(ct => { _connectorTypes[ct.name] = ct; });
    } catch (e) {
        console.error('Konnektor-Typen konnten nicht geladen werden:', e);
        _connectorTypes = {};
    }
    return _connectorTypes;
}

async function checkDduAvailable() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ddu-available', {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        const data = await resp.json();
        window._dduAvailable = data.available;
    } catch(e) {
        window._dduAvailable = false;
    }
}

async function checkVlmAvailable() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/vlm-available', {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        const data = await resp.json();
        window._vlmAvailable = data.available;
        window._vlmModel = data.model || '';
        window._vlmMessage = data.message || '';
    } catch(e) {
        window._vlmAvailable = false;
        window._vlmModel = '';
        window._vlmMessage = '';
    }
}

async function _loadEmbeddingModels() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/embedding-models',
            {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        const data = await resp.json();
        const sel = document.getElementById('dsEmbedModelFilter');
        if (!sel) return;
        _defaultEmbedModel = data.default_model || '';
        _embedModelLateChunking = {};
        (data.models || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name + (m.is_default ? ' (Standard)' : '');
            sel.appendChild(opt);
            _embedModelLateChunking[m.name] = !!m.supports_late_chunking;
        });
        if (!_selectedEmbedModel) {
            _selectedEmbedModel = _defaultEmbedModel;
        }
        if (_selectedEmbedModel) sel.value = _selectedEmbedModel;
        if (dsTable) dsTable.render();
    } catch (e) {
        console.error('Embedding-Modelle konnten nicht geladen werden:', e);
    }
}

async function initDatenquellenTab() {
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Re-Init-Schutz: altes Cleanup ausfuehren
    if (typeof _runTabCleanups === 'function') _runTabCleanups('datenquellen');
    _dsInitSeq++;

    _bindDsActions();

    // DDU- und VLM-Verfuegbarkeit pruefen (einmalig, await damit Guards korrekt greifen)
    const initChecks = await Promise.allSettled([
        checkDduAvailable(),
        checkVlmAvailable(),
        _loadConnectorTypes(),
    ]);
    initChecks.forEach(result => {
        if (result.status === 'rejected') {
            console.warn('Datenquellen-Initialisierung teilweise fehlgeschlagen:', result.reason);
        }
    });

    // Filter bar: search + add button
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.add('u-flex');
        filterBar.classList.remove('u-hidden');

        const searchGroup = document.createElement('div');
        searchGroup.className = 'filter-group';
        searchGroup.innerHTML = `<span class="filter-label">Suche:</span>
            <input type="text" class="filter-input" id="dsSearch" placeholder="Name/Typ..."
                data-input-action="update-ds-search">`;
        filterBar.appendChild(searchGroup);

        const embedGroup = document.createElement('div');
        embedGroup.className = 'filter-group';
        embedGroup.innerHTML = `<span class="filter-label">Modell:</span>
            <select class="filter-input u-min-w-180" id="dsEmbedModelFilter"
                data-change-action="select-embed-model">
            </select>`;
        filterBar.appendChild(embedGroup);
        _loadEmbeddingModels();

        const addGroup = document.createElement('div');
        addGroup.className = 'filter-group u-ml-auto';
        addGroup.innerHTML = `<button class="action-btn primary" data-action="open-create-datasource-modal">+ Datenquelle</button>`;
        filterBar.appendChild(addGroup);
    }

    container.innerHTML = '<div id="dsTableContainer"></div>';

    const config = {
        id: 'datasources',
        apiEndpoint: '/api/admin/datasources',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 150px 70px 1fr 100px 80px 80px 130px 70px 160px',
        defaultSort: { field: 'name', direction: 'asc' },
        columns: [
            { field: 'name', label: 'Name', sortable: true, align: 'left' },
            { field: 'type', label: 'Typ', sortable: true, align: 'center', renderer: 'badge' },
            { field: 'source_display', label: 'Quelle', sortable: false, align: 'left', truncate: 40 },
            { field: 'ingestion_status', label: 'Status', sortable: true, align: 'center', renderer: 'ingestionStatus' },
            { field: 'source_size_bytes', label: 'Größe', sortable: true, align: 'center',
              renderer: (v) => `<span class="u-tab-nums">${formatFileSize(v)}</span>` },
            { field: 'chunk_count', label: 'Chunks', sortable: true, align: 'center',
              renderer: (v, row) => {
                  let count = 0;
                  if (_selectedEmbedModel && row && row.embeddings) {
                      const emb = row.embeddings.find(e => e.embedding_model === _selectedEmbedModel);
                      count = emb ? (emb.chunk_count || 0) : 0;
                  }
                  return `<span class="u-tab-nums">${count.toLocaleString('de-DE')}</span>`;
              },
              sortValue: (row) => {
                  if (!_selectedEmbedModel || !row || !row.embeddings) return 0;
                  const emb = row.embeddings.find(e => e.embedding_model === _selectedEmbedModel);
                  return emb ? (emb.chunk_count || 0) : 0;
              }
            },
            { field: 'last_ingested_at', label: 'Letzte Ingestion', sortable: true, align: 'center',
              renderer: (v, row) => {
                  let ts = null;
                  if (_selectedEmbedModel && row && row.embeddings) {
                      const emb = row.embeddings.find(e => e.embedding_model === _selectedEmbedModel);
                      ts = emb ? emb.last_ingested_at : null;
                  }
                  if (!ts) return '<span class="u-c-secondary">–</span>';
                  const d = _parseDsTimestamp(ts);
                  if (isNaN(d.getTime())) return SharedUi.runtime.escapeHtml(ts);
                  const pad = n => String(n).padStart(2, '0');
                  return `<span class="u-fs-12">${pad(d.getDate())}.${pad(d.getMonth()+1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}</span>`;
              },
              sortValue: (row) => {
                  if (!_selectedEmbedModel || !row || !row.embeddings) return '';
                  const emb = row.embeddings.find(e => e.embedding_model === _selectedEmbedModel);
                  return emb ? (emb.last_ingested_at || '') : '';
              }
            },
            { field: 'is_active', label: 'Aktiv', sortable: true, align: 'center', renderer: 'toggle' },
            { field: 'id', label: 'Aktionen', sortable: false, align: 'center', renderer: 'actions' },
        ],
        detailFields: [
            { field: 'ds_chunk_strategy', label: 'Chunk-Strategie', renderer: (v) => {
                const labels = { auto: 'Automatisch', prose: 'Fliesstext (768/150)', tabular: 'Tabellarisch (256/0)', structured: 'Strukturiert (384/50)', code: 'Code (512/100)', email: 'E-Mail (640/100)', custom: 'Benutzerdefiniert' };
                return labels[v] || v || 'Automatisch';
            }},
            { field: 'last_ingested_at', label: 'Letzte Ingestion' },
            { field: 'file_count', label: 'Dateien' },
            { field: 'source_size_bytes', label: 'Datenmenge', renderer: (v) => formatFileSize(v) },
            { field: 'embeddings', label: 'Embeddings nach Modell', renderer: (v) => {
                if (!v || v.length === 0) return '<span class="u-c-secondary">Noch nicht indexiert</span>';
                let html = '<table class="u-detail-table-sm u-mt-4">';
                html += '<tr class="u-cell-border-strong u-row-bg-secondary-soft">'
                    + '<th class="u-cell-left-4-8-bold">Modell</th>'
                    + '<th class="u-cell-right-4-8-bold">Chunks</th>'
                    + '<th class="u-cell-right-4-8-bold">Dateien</th>'
                    + '<th class="u-cell-left-4-8-bold">Collection</th>'
                    + '<th class="u-cell-right-4-8-bold">Letzte Ingestion</th>'
                    + '</tr>';
                v.forEach(e => {
                    const hasData = (e.chunk_count || 0) > 0;
                    const badge = hasData
                        ? '<span class="badge badge-success u-fs-10 u-ml-4">aktiv</span>'
                        : '<span class="badge badge-default u-fs-10 u-ml-4">leer</span>';
                    const ts = e.last_ingested_at
                        ? e.last_ingested_at.replace('T', ' ')
                        : '-';
                    html += '<tr class="u-cell-border">'
                        + `<td class="u-cell-left-4-8">${SharedUi.runtime.escapeHtml(e.embedding_model)}${badge}</td>`
                        + `<td class="u-cell-right-4-8-tab">${(e.chunk_count || 0).toLocaleString('de-DE')}</td>`
                        + `<td class="u-cell-right-4-8-tab">${e.file_count || 0}</td>`
                        + `<td class="u-cell-left-4-8 u-mono u-fs-11 u-c-secondary">${SharedUi.runtime.escapeHtml(e.collection || '')}</td>`
                        + `<td class="u-cell-right-4-8 u-c-secondary">${SharedUi.runtime.escapeHtml(ts)}</td>`
                        + '</tr>';
                });
                // Summenzeile
                const totalChunks = v.reduce((s, e) => s + (e.chunk_count || 0), 0);
                const totalFiles = v.reduce((s, e) => s + (e.file_count || 0), 0);
                if (v.length > 1) {
                    html += '<tr class="u-row-border-top-strong u-bold">'
                        + '<td class="u-cell-left-4-8">Gesamt</td>'
                        + `<td class="u-cell-right-4-8-tab">${totalChunks.toLocaleString('de-DE')}</td>`
                        + `<td class="u-cell-right-4-8-tab">${totalFiles}</td>`
                        + '<td></td><td></td></tr>';
                }
                html += '</table>';
                return html;
            }},
            { field: 'sync_interval', label: 'Sync-Intervall', renderer: (v) => {
                const labels = { global: 'Globaler Standard', manual: 'Nur manuell', '30min': 'Alle 30 Min', '1h': 'Stuendlich', '2h': 'Alle 2 Stunden', '6h': 'Alle 6 Stunden', '12h': 'Alle 12 Stunden', '24h': 'Taeglich (03:00)', custom: 'Benutzerdefiniert' };
                const norm = _normalizeSyncInterval(v);
                return labels[norm.value] || SharedUi.runtime.escapeHtml(v) || 'Globaler Standard';
            }},
            { field: 'last_error', label: 'Letzter Fehler' },
            { field: 'created_at', label: 'Erstellt am' },
        ],
        filters: [
            { id: 'search', field: 'name', wildcard: true, wildcardFields: ['type', 'source_display'], type: 'text' },
        ],
    };

    const table = new SharedUi.table.ExpandableTable(config, 'dsTableContainer', {
        onDataLoaded: (result) => {
            if (dsTable !== table) return;
            const rows = result.data || result.items || result || [];
            if (!Array.isArray(rows)) return;
            _updateDsRates(rows);
            const hasActive = rows.some(r =>
                r.ingestion_status === 'running' || r.ingestion_status === 'paused' || r.ingestion_status === 'pending');
            if (hasActive && !kiaraPolling.has('datenquellen:refresh')) {
                _startDsPolling();
            }
        },
    });
    dsTable = table;

    // Custom renderers
    dsTable.renderers['badge'] = (value) => {
        const types = _connectorTypes || {};
        const ct = types[value];
        const label = ct?.label || value;
        const cls = ct?.badge_class || 'default';
        return `<span class="badge badge-${cls}">${SharedUi.runtime.escapeHtml(label)}</span>`;
    };

    dsTable.renderers['ingestionStatus'] = (value, col, row) => {
        const statusMap = {
            idle:    { label: 'Bereit',    cls: 'default' },
            pending: { label: 'Wartend',   cls: 'default' },
            running: { label: 'Laeuft...', cls: 'warning' },
            paused:  { label: 'Pausiert',  cls: 'info'    },
            success: { label: 'OK',        cls: 'success' },
            error:   { label: 'Fehler',    cls: 'danger'  },
        };
        const s = statusMap[value] || statusMap.idle;
        const pulseCls = value === 'running' ? ' u-anim-pulse' : '';
        let html = `<span class="badge badge-${s.cls}${pulseCls}">${SharedUi.runtime.escapeHtml(s.label)}</span>`;
        if ((value === 'running' || value === 'paused') && row) {
            const total = row.progress_total || 0;
            const processed = row.progress_processed || 0;
            if (total > 0) {
                let progressText = `${processed}/${total}`;
                if (value === 'running') {
                    const rate = _getDsRate(row);
                    const rateText = rate === null ? '\u2013' : rate.toFixed(1);
                    const title = rate === null ? 'Durchsatz wird gemessen.'
                        : 'Durchschnitt ueber bis zu 30 Sekunden; Wartezeiten sind enthalten.';
                    progressText += ` <span title="${title}">(${rateText} D/s)</span>`;
                }
                html += `<div class="u-text-xxs-secondary u-mt-2">${progressText}</div>`;
            } else if (row.phase) {
                html += `<div class="u-text-xxs-secondary u-mt-2">${SharedUi.runtime.escapeHtml(row.phase)}</div>`;
            }
        }
        return html;
    };

    dsTable.renderers['toggle'] = (value, col, row) => {
        if (col.field !== 'is_active') {
            return `<label class="toggle-switch"><input type="checkbox" ${value ? 'checked' : ''}><span class="toggle-slider"></span></label>`;
        }
        return `<label class="toggle-switch">
            <input type="checkbox" ${value ? 'checked' : ''}
                data-change-action="toggle-datasource-active"
                data-ds-id="${row.id}">
            <span class="toggle-slider"></span></label>`;
    };

    dsTable.renderers['actions'] = (value, col, row) => {
        const isRunning = row.ingestion_status === 'running';
        const isPaused = row.ingestion_status === 'paused';
        const isPending = row.ingestion_status === 'pending';
        const isActive = isRunning || isPaused || isPending;
        let html = '<div class="action-buttons">';
        if (isActive) {
            if (isPaused) {
                html += `<button class="action-btn-sm" title="Fortsetzen" data-action="pause-ds-ingestion" data-ds-id="${row.id}">&#x25B6;</button>`;
            } else if (isRunning) {
                html += `<button class="action-btn-sm" title="Pausieren" data-action="pause-ds-ingestion" data-ds-id="${row.id}">&#x23F8;</button>`;
            }
            html += `<button class="action-btn-sm danger" title="Abbrechen" data-action="cancel-ds-ingestion" data-ds-id="${row.id}" data-ds-name="${SharedUi.runtime.escapeAttr(row.name)}">&#x25A0;</button>`;
        } else {
            html += `<button class="action-btn-sm" title="Indexieren" data-action="trigger-ds-ingest" data-ds-id="${row.id}" data-ds-name="${SharedUi.runtime.escapeAttr(row.name)}">&#x1F504;</button>`;
            html += `<button class="action-btn-sm" title="Voll-Indexieren" data-action="trigger-ds-reingest" data-ds-id="${row.id}" data-ds-name="${SharedUi.runtime.escapeAttr(row.name)}">&#x1F501;</button>`;
            if (row.is_active && _HTML_REINGEST_CONNECTORS.has(row.type)) {
                html += `<button class="action-btn-sm" title="HTML neu aufbereiten" data-action="trigger-ds-html-reingest" data-ds-id="${row.id}" data-ds-name="${SharedUi.runtime.escapeAttr(row.name)}">HTML</button>`;
            }
        }
        html += `<button class="action-btn-sm" title="Bearbeiten" data-action="open-edit-datasource-modal" data-ds-id="${row.id}">&#x270E;</button>`;
        html += `<button class="action-btn-sm" title="Berechtigungen" data-action="open-ds-groups-modal" data-ds-id="${row.id}" data-ds-name="${SharedUi.runtime.escapeAttr(row.name)}">&#x1F465;</button>`;
        html += `<button class="action-btn-sm danger" title="Loeschen" data-action="delete-datasource" data-ds-id="${row.id}" data-ds-name="${SharedUi.runtime.escapeAttr(row.name)}" data-ds-active="${isActive ? 1 : 0}">&#x2716;</button>`;
        html += '</div>';
        return html;
    };

    // Cleanup registrieren
    registerTabCleanup('datenquellen', () => {
        kiaraPolling.stop('datenquellen:refresh');
        _dsRateHistory.clear();
        dsTable = null;
    });

    dsTable.loadData();
}

async function _toggleDatasourceActive(input, dsId, isActive) {
    if (!input || !dsId) return;
    if (_dsActiveToggleInFlight.has(dsId)) {
        input.checked = !isActive;
        return;
    }

    _dsActiveToggleInFlight.add(dsId);
    input.disabled = true;
    try {
        await SharedUi.fetch.safeFetchJson(`/api/admin/datasources/${dsId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ is_active: isActive }),
        });
        SharedUi.notify.show(isActive ? 'Datenquelle aktiviert' : 'Datenquelle deaktiviert', 'success');
        if (dsTable) await dsTable.loadData();
    } catch (e) {
        SharedUi.notify.show(e.message || 'Status konnte nicht aktualisiert werden', 'error');
        if (dsTable) {
            await dsTable.loadData();
        } else {
            input.checked = !isActive;
        }
    } finally {
        _dsActiveToggleInFlight.delete(dsId);
        input.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Create / Edit Modal
// ---------------------------------------------------------------------------

/**
 * Erzeugt Formular-HTML fuer Konnektor-Felder aus dem Schema.
 * @param {Object} schema - config_schema() des Konnektors
 * @param {Object} config - Aktuelle Werte (connector_config)
 * @returns {string} HTML-String
 */
function _connectorFieldDomId(prefix, fieldName) {
    const raw = String(fieldName ?? '');
    if (/^[A-Za-z0-9_-]+$/.test(raw)) return `${prefix}_${raw}`;
    const encoded = Array.from(raw).map(ch => ch.charCodeAt(0).toString(16)).join('_') || 'empty';
    return `${prefix}_${encoded}`;
}

function _connFieldId(fieldName) { return _connectorFieldDomId('connField', fieldName); }
function _folderPickerWrapId(fieldName) { return _connectorFieldDomId('folderPickerWrap', fieldName); }
function _loadFoldersBtnId(fieldName) { return _connectorFieldDomId('btnLoadFolders', fieldName); }
function _allFoldersCheckboxId(fieldName) { return _connectorFieldDomId('chkAllFolders', fieldName); }
function _folderTreeId(fieldName) { return _connectorFieldDomId('folderTree', fieldName); }

function _buildConnectorFields(schema, config) {
    if (!schema || Object.keys(schema).length === 0) return '';
    let html = '';

    for (const [fieldName, spec] of Object.entries(schema)) {
        const value = config[fieldName] ?? spec.default ?? '';
        const label = spec.label || fieldName;
        const ph = spec.placeholder ? ` placeholder="${SharedUi.runtime.escapeAttr(spec.placeholder)}"` : '';
        const req = spec.required ? ' *' : '';
        const id = _connFieldId(fieldName);

        html += '<div class="modal-field">';
        html += `<label>${SharedUi.runtime.escapeHtml(label)}${req}</label>`;

        if (spec.browse) {
            // Text-Input mit Browse-Button — data-field-name statt Inline-Handler
            html += `<div class="u-flex-gap-6">`;
            html += `<input type="text" id="${SharedUi.runtime.escapeAttr(id)}" value="${SharedUi.runtime.escapeAttr(value)}"${ph} class="u-flex-1">`;
            html += `<button type="button" class="action-btn ds-browse-btn u-nowrap" data-field-name="${SharedUi.runtime.escapeAttr(fieldName)}">Durchsuchen...</button>`;
            html += `</div>`;
        } else if (spec.folder_tree) {
            // Folder-Picker: Textfeld + Lade-Button + Baumansicht — data-field-name statt Inline-Handler
            html += `<div id="${SharedUi.runtime.escapeAttr(_folderPickerWrapId(fieldName))}">`;
            html += `<div class="u-flex-center-gap-8 u-mb-6">`;
            html += `<button type="button" class="action-btn ds-load-folders-btn u-nowrap" id="${SharedUi.runtime.escapeAttr(_loadFoldersBtnId(fieldName))}"
                      data-field-name="${SharedUi.runtime.escapeAttr(fieldName)}">Ordner laden</button>`;
            html += `<label class="u-flex-center-gap-4 u-fs-13 u-clickable">`;
            const allChecked = String(value) === '*';
            html += `<input type="checkbox" id="${SharedUi.runtime.escapeAttr(_allFoldersCheckboxId(fieldName))}"
                      class="ds-toggle-all-folders u-size-14" data-field-name="${SharedUi.runtime.escapeAttr(fieldName)}"
                      ${allChecked ? ' checked' : ''}>`;
            html += ` Alle synchronisieren</label>`;
            html += `</div>`;
            html += `<div id="${SharedUi.runtime.escapeAttr(_folderTreeId(fieldName))}" class="u-folder-tree-box u-hidden"></div>`;
            html += `<input type="text" id="${SharedUi.runtime.escapeAttr(id)}" value="${SharedUi.runtime.escapeAttr(value)}"${ph}${allChecked ? ' readonly' : ''}>`;
            html += `</div>`;
        } else if (spec.options) {
            // Select/Dropdown
            html += `<select class="config-input" id="${SharedUi.runtime.escapeAttr(id)}">`;
            spec.options.forEach(opt => {
                const sel = String(value) === String(opt.value) ? ' selected' : '';
                html += `<option value="${SharedUi.runtime.escapeAttr(opt.value)}"${sel}>${SharedUi.runtime.escapeHtml(opt.label)}</option>`;
            });
            html += '</select>';
        } else if (spec.type === 'bool') {
            // Checkbox — V2-Fix H8: Defensiver Check statt Truthy
            const checked = (value === true || value === 1) ? ' checked' : '';
            html += `<input type="checkbox" id="${SharedUi.runtime.escapeAttr(id)}"${checked} class="u-size-16">`;
        } else if (spec.multiline) {
            // Textarea
            html += `<textarea id="${SharedUi.runtime.escapeAttr(id)}" rows="3"${ph} class="u-textarea-mono-xs">${SharedUi.runtime.escapeHtml(String(value))}</textarea>`;
        } else if (spec.secret) {
            // Passwort: Gespeicherte Secrets mit Maske anzeigen, neue leer
            const hasSaved = !!value;
            const pwVal = hasSaved ? DS_PASSWORD_MASK : '';
            const pwPh = hasSaved ? ' placeholder="(unveraendert wenn ********)"' : ph;
            html += `<input type="password" id="${SharedUi.runtime.escapeAttr(id)}" value="${SharedUi.runtime.escapeAttr(pwVal)}"${pwPh}>`;
        } else if (spec.type === 'int') {
            // Zahl — min/max aus Schema uebernehmen
            const numMin = spec.minimum != null ? ` min="${spec.minimum}"` : '';
            const numMax = spec.maximum != null ? ` max="${spec.maximum}"` : '';
            html += `<input type="number" id="${SharedUi.runtime.escapeAttr(id)}" value="${SharedUi.runtime.escapeAttr(value)}"${ph}${numMin}${numMax}>`;
        } else {
            // Text (Standard)
            html += `<input type="text" id="${SharedUi.runtime.escapeAttr(id)}" value="${SharedUi.runtime.escapeAttr(value)}"${ph}>`;
        }

        // Hilfetext
        if (spec.hint) {
            html += `<div class="u-text-xs-secondary u-mt-4">${SharedUi.runtime.escapeHtml(spec.hint)}</div>`;
        }

        html += '</div>';
    }
    return html;
}

function _dsFormHtml(ds) {
    const cfg = ds || {};
    const cc = cfg.connector_config || {};
    const dsType = cfg.type || 'local';
    const types = _connectorTypes || {};

    let html = '';

    // 1. Verbindungs-Einstellungen (Name, Typ, Konnektor-Felder)
    let typeOpts = '';
    for (const [name, ct] of Object.entries(types)) {
        const sel = dsType === name ? ' selected' : '';
        typeOpts += `<option value="${SharedUi.runtime.escapeAttr(name)}"${sel}>${SharedUi.runtime.escapeHtml(ct.label)}</option>`;
    }
    const schema = types[dsType]?.schema || {};
    html += `<details id="dsConnectorDetails" class="u-details-section">
        <summary class="u-details-summary">Verbindungs-Einstellungen</summary>
        <div class="u-mt-12">
            <div class="modal-field"><label>Name *</label><input type="text" id="dsName" value="${SharedUi.runtime.escapeAttr(cfg.name || '')}"></div>
            <div class="modal-field"><label>Typ</label><select class="config-input" id="dsType" data-change-action="ds-type-change">${typeOpts}</select></div>
            <div id="dsConnectorFields">${_buildConnectorFields(schema, cc)}</div>
        </div>
    </details>`;

    // 4. Erweiterte Einstellungen (per-DS, nicht konnektor-spezifisch)
    html += _dsAdvancedSettingsHtml(cfg);

    return html;
}

function _dsAdvancedSettingsHtml(cfg) {
    const rawSync = cfg.sync_interval || 'global';
    const normSync = _normalizeSyncInterval(rawSync);
    const syncInterval = normSync.value;
    const chunkStrategy = cfg.ds_chunk_strategy || 'auto';
    const dsType = cfg.type || 'local';
    const indexAtt = cfg.ds_index_attachments !== undefined ? cfg.ds_index_attachments : true;
    const fileTypesStr = Array.isArray(cfg.ds_file_types) ? cfg.ds_file_types.join(', ') : (cfg.ds_file_types || '');
    const excludeStr = Array.isArray(cfg.ds_exclude_patterns) ? cfg.ds_exclude_patterns.join(', ') : (cfg.ds_exclude_patterns || '');

    let html = '';
    html += `<details class="u-details-section u-mt-16">
        <summary class="u-details-summary">Embedding-Einstellungen</summary>
        <div class="u-mt-12">
            <div class="modal-field">
                <label>Chunk-Strategie</label>
                <select class="config-input" id="dsChunkStrategy" data-change-action="ds-chunk-strategy-change">
                    <option value="auto" ${chunkStrategy === 'auto' ? 'selected' : ''}>Automatisch (empfohlen)</option>
                    <option value="prose" ${chunkStrategy === 'prose' ? 'selected' : ''}>Fliesstext (768/150)</option>
                    <option value="tabular" ${chunkStrategy === 'tabular' ? 'selected' : ''}>Tabellarisch (256/0)</option>
                    <option value="structured" ${chunkStrategy === 'structured' ? 'selected' : ''}>Strukturiert (384/50)</option>
                    <option value="code" ${chunkStrategy === 'code' ? 'selected' : ''}>Code (512/100)</option>
                    <option value="email" ${chunkStrategy === 'email' ? 'selected' : ''}>E-Mail (640/100)</option>
                    <option value="custom" ${chunkStrategy === 'custom' ? 'selected' : ''}>Benutzerdefiniert...</option>
                </select>
            </div>
            <details class="u-my-8-12">
                <summary class="u-details-summary-sm">
                    Wie funktioniert die automatische Chunk-Groesse?
                </summary>
                <div class="u-info-panel">
                    <p class="u-my-0-8">
                        <strong>Warum ist die Chunk-Groesse wichtig?</strong><br>
                        Dokumente werden fuer die Suche in Abschnitte (Chunks) zerlegt.
                        Die optimale Groesse haengt vom Dokumenttyp ab:
                    </p>
                    <table class="u-detail-table-xs">
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Fliesstext</strong> (Berichte, Mails)</td>
                            <td class="u-cell-kv-right">768 Zeichen</td>
                            <td class="u-cell-kv-label">Zusammenhaengende Absaetze bleiben erhalten</td>
                        </tr>
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>E-Mail</strong></td>
                            <td class="u-cell-kv-right">640 Zeichen</td>
                            <td class="u-cell-kv-label">Header + Nachrichtentext als Einheit</td>
                        </tr>
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Code</strong></td>
                            <td class="u-cell-kv-right">512 Zeichen</td>
                            <td class="u-cell-kv-label">Funktionen/Klassen als Einheit</td>
                        </tr>
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Strukturiert</strong> (Formulare)</td>
                            <td class="u-cell-kv-right">384 Zeichen</td>
                            <td class="u-cell-kv-label">Key-Value-Paare praezise trennbar</td>
                        </tr>
                        <tr>
                            <td class="u-cell-kv-value"><strong>Tabellarisch</strong> (CSV, SQL)</td>
                            <td class="u-cell-kv-right">256 Zeichen</td>
                            <td class="u-cell-kv-label">Jede Zeile ist eine eigenstaendige Info</td>
                        </tr>
                    </table>
                    <p class="u-mt-8">
                        Im Modus <em>Automatisch</em> erkennt KIara den Dokumenttyp anhand
                        von Dateiendung und Textstruktur (Zeilenlaenge, Tabellenanteil, Key-Value-Muster)
                        und waehlt die passende Strategie. Wenn Sie einen festen Wert bevorzugen,
                        waehlen Sie eine Strategie oder <em>Benutzerdefiniert</em>.
                    </p>
                </div>
            </details>
            <div id="dsChunkCustomFields" class="${chunkStrategy === 'custom' ? 'u-block' : 'u-hidden'}">
                <div class="modal-field"><label>Chunk Size</label>
                    <input type="number" id="dsChunkSize" value="${cfg.ds_chunk_size || ''}" placeholder="z.B. 1000" min="100" max="10000">
                </div>
                <div class="modal-field"><label>Chunk Overlap</label>
                    <input type="number" id="dsChunkOverlap" value="${cfg.ds_chunk_overlap || ''}" placeholder="z.B. 200" min="0" max="5000">
                </div>
            </div>
            <div class="modal-field"><label>Dateitypen</label>
                <input type="text" id="dsFileTypes" value="${SharedUi.runtime.escapeAttr(fileTypesStr)}" placeholder="Leer = Global (z.B. .pdf, .docx, .txt)">
            </div>
            <div class="modal-field"><label>Ausschlussmuster</label>
                <input type="text" id="dsExcludePatterns" value="${SharedUi.runtime.escapeAttr(excludeStr)}" placeholder="Leer = Global (z.B. */tmp/*, *.bak)">
            </div>
            <div class="modal-field u-field-divider-top">
                <label class="u-flex-center-gap-8 u-clickable">
                    <input type="checkbox" id="dsDduEnabled"
                           ${cfg.ds_ddu_enabled ? 'checked' : ''}
                           class="u-size-16">
                    Erweiterte Dokumentanalyse (Docling)
                </label>
            </div>
            <details class="u-my-8-12">
                <summary class="u-details-summary-sm">
                    Was ist die erweiterte Dokumentanalyse?
                </summary>
                <div class="u-info-panel">
                    <p class="u-my-0-8">
                        <strong>Standardmaessig</strong> extrahiert KIara Text aus PDFs ueber die
                        interne Textschicht (pypdf). Das funktioniert gut bei digital erstellten
                        Dokumenten, stoesst aber an Grenzen bei:
                    </p>
                    <table class="u-detail-table-xs">
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Gescannte PDFs</strong></td>
                            <td class="u-cell-kv-label">Enthalten nur Bilder, kein extrahierbarer Text &rarr; OCR noetig</td>
                        </tr>
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Komplexe Tabellen</strong></td>
                            <td class="u-cell-kv-label">Ohne sichtbare Linien oder mit mehrstufigen Headern &rarr; werden als Textbrei extrahiert</td>
                        </tr>
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Mehrspaltige Layouts</strong></td>
                            <td class="u-cell-kv-label">Spalten werden vermischt, Saetze aus verschiedenen Spalten verschmelzen</td>
                        </tr>
                        <tr>
                            <td class="u-cell-kv-value"><strong>Bilddateien</strong></td>
                            <td class="u-cell-kv-label">PNG, TIFF, JPEG &mdash; koennen nur mit OCR verarbeitet werden</td>
                        </tr>
                    </table>
                    <p class="u-mt-8">
                        <strong>Mit Docling</strong> analysiert KIara jede Seite visuell und erkennt
                        automatisch Ueberschriften, Fliesstext, Tabellen und Abbildungen. Tabellen
                        werden mit korrekter Zeilen-/Spaltenstruktur extrahiert, mehrspaltige Layouts
                        korrekt gelesen und gescannte Dokumente per OCR erschlossen.<br><br>
                        Docling arbeitet <strong>unabhaengig von der Chunk-Strategie</strong>: Es verbessert
                        die Qualitaet des extrahierten Textes, waehrend die Chunk-Groesse weiterhin
                        ueber die gewaehlte Strategie gesteuert wird.<br><br>
                        <em>Hinweis: Die erste Verarbeitung mit Docling ist langsamer als die
                        Standard-Extraktion, da jede Seite visuell analysiert wird.
                        Bei grossen Dokumentenbestaenden kann die Ingestion deutlich laenger dauern.</em>
                    </p>
                </div>
            </details>
            <div class="modal-field u-mt-4 u-ml-24">
                <label class="u-flex-center-gap-8 u-clickable"
                       id="lblDduOcrImages">
                    <input type="checkbox" id="dsDduOcrImages"
                           ${cfg.ds_ddu_ocr_images !== false && cfg.ds_ddu_ocr_images !== 0 ? 'checked' : ''}
                           ${cfg.ds_ddu_enabled ? '' : 'disabled'}
                           class="u-size-16">
                    OCR fuer Bilddateien (JPG, PNG, TIFF)
                </label>
                <div class="u-text-xxs-secondary u-mt-2 u-ml-24">
                    Erkennt Text in Bilddateien per OCR. Deaktivieren, wenn die Datenquelle
                    hauptsaechlich Fotos ohne Text enthaelt.
                </div>
            </div>
            <div class="modal-field u-mt-12">
                <label class="u-flex-center-gap-8 u-clickable">
                    <input type="checkbox" id="dsVlmEnabled"
                           ${cfg.ds_vlm_enabled ? 'checked' : ''}
                           class="u-size-16">
                    VLM-Bildbeschreibung
                </label>
            </div>
            <details class="u-my-8-12">
                <summary class="u-details-summary-sm">
                    Was ist die VLM-Bildbeschreibung?
                </summary>
                <div class="u-info-panel">
                    <p class="u-my-0-8">
                        Ein Vision Language Model (VLM) analysiert Bilddateien und erzeugt eine
                        natuerlichsprachliche Beschreibung des Inhalts. Im Gegensatz zur reinen
                        OCR-Textextraktion (Docling) versteht das VLM den <strong>Kontext</strong>
                        des Bildes:
                    </p>
                    <table class="u-detail-table-xs">
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Technische Zeichnungen</strong></td>
                            <td class="u-cell-kv-label">Masse, Werkstoffe, Toleranzen, Bauteilbeschreibung</td>
                        </tr>
                        <tr class="u-cell-border">
                            <td class="u-cell-kv-value"><strong>Typenschilder</strong></td>
                            <td class="u-cell-kv-label">Hersteller, Artikelnummer, Leistungsdaten, Schutzart</td>
                        </tr>
                        <tr>
                            <td class="u-cell-kv-value"><strong>Fotos</strong></td>
                            <td class="u-cell-kv-label">Bauteilbeschreibung, Fehlerdokumentation, Montagezustand</td>
                        </tr>
                    </table>
                    <p class="u-mt-8">
                        Die Beschreibung wird als eigener Text-Chunk gespeichert und ist ueber die
                        normale Suche auffindbar. VLM kann <strong>zusaetzlich zu Docling</strong>
                        aktiviert werden &mdash; Bilder erhalten dann sowohl OCR-Text als auch eine
                        semantische Beschreibung.<br><br>
                        <em>Hinweis: Die VLM-Beschreibung ist langsamer als OCR, da jedes Bild
                        einzeln an das Vision-Modell geschickt wird. Das VLM teilt sich die GPU
                        mit dem Chat-LLM.</em>
                    </p>
                </div>
            </details>
            <div class="modal-field u-field-divider-top">
                <label class="u-flex-center-gap-8 u-clickable">
                    <input type="checkbox" id="dsContextualRetrieval"
                           ${cfg.ds_contextual_retrieval ? 'checked' : ''}
                           class="u-size-16">
                    Contextual Retrieval (Enrichment)
                </label>
                <div class="u-text-xxs-secondary u-mt-4">
                    Chunks werden vor der Indexierung mit Kontext angereichert (LLM-basiert).
                </div>
            </div>
            <div id="crSettings" class="u-ml-24 ${cfg.ds_contextual_retrieval ? 'u-block' : 'u-hidden'}">
                <div class="modal-field">
                    <label>Prompt-Template</label>
                    <select class="config-input" id="dsEnrichmentPromptId"><option value="">(Standard)</option></select>
                </div>
                <div class="modal-field">
                    <label>Enrichment-Backend</label>
                    <select class="config-input" id="dsEnrichmentBackendId"><option value="">— Backend waehlen —</option></select>
                </div>
            </div>
            <div class="modal-field u-mt-10">
                <label class="u-flex-center-gap-8 u-clickable">
                    <input type="checkbox" id="dsColbertEnabled"
                           ${cfg.ds_colbert_enabled ? 'checked' : ''}
                           class="u-size-16">
                    ColBERT Multi-Vector-Index
                </label>
                <div class="u-text-xxs-secondary u-mt-4">
                    Erzeugt zusaetzlich Token-Level-Vektoren fuer praeziseres Multi-Aspekt-Matching (erhoehter Speicherbedarf).
                </div>
            </div>
            <div class="modal-field u-mt-10">
                <label class="u-flex-center-gap-8 u-clickable">
                    <input type="checkbox" id="dsAcceptAllText"
                           ${cfg.ds_accept_all_text ? 'checked' : ''}
                           class="u-size-16">
                    Alle Textdateien akzeptieren (Content-Detection)
                </label>
                <div class="u-text-xxs-secondary u-mt-4">
                    Dateien mit unbekannter Endung werden anhand ihres Inhalts geprueft und als Plaintext indexiert, wenn sie lesbaren Text enthalten.
                </div>
            </div>
            <div id="dsIndexAttachmentsField" class="modal-field u-mt-12 ${dsType === 'imap' ? 'u-block' : 'u-hidden'}">
                <label class="u-flex-center-gap-8 u-clickable">
                    <input type="checkbox" id="dsIndexAttachments"
                           ${indexAtt ? 'checked' : ''}
                           class="u-size-16">
                    E-Mail-Anhaenge indexieren
                </label>
                <div class="config-hint u-text-xs-secondary u-mt-4">
                    Wenn deaktiviert, werden Anhaenge (PDFs etc.) zwar heruntergeladen, aber nicht in die Suche aufgenommen.
                </div>
            </div>
        </div>
    </details>
    <details class="u-details-section u-mt-16">
        <summary class="u-details-summary">Erweiterte Einstellungen</summary>
        <div class="u-mt-12">
            <div class="modal-field"><label>Cloud-Restriction-Level</label>
                <select class="config-input" id="dsRestrictionLevel">
                    <option value="anonymize" ${(cfg.restriction_level || 'anonymize') === 'anonymize' ? 'selected' : ''}>Anonymisieren (Standard)</option>
                    <option value="public" ${cfg.restriction_level === 'public' ? 'selected' : ''}>Nicht anonymisieren (oeffentlich)</option>
                    <option value="blocked" ${cfg.restriction_level === 'blocked' ? 'selected' : ''}>Fuer Cloud gesperrt</option>
                </select>
                <div class="config-hint u-text-xs-secondary u-mt-4">
                    Steuert ob Daten dieser Quelle an Cloud-Backends gesendet werden duerfen.
                    „Anonymisieren“ wirkt nur bei global aktivierter Anonymisierung.
                </div>
            </div>
            <div class="modal-field"><label>Sync-Intervall</label>
                <select class="config-input" id="dsSyncInterval" data-change-action="ds-sync-interval-change">
                    <option value="global" ${syncInterval === 'global' ? 'selected' : ''}>Globaler Standard</option>
                    <option value="manual" ${syncInterval === 'manual' ? 'selected' : ''}>Nur manuell</option>
                    <option value="30min" ${syncInterval === '30min' ? 'selected' : ''}>Alle 30 Minuten</option>
                    <option value="1h" ${syncInterval === '1h' ? 'selected' : ''}>Stuendlich</option>
                    <option value="2h" ${syncInterval === '2h' ? 'selected' : ''}>Alle 2 Stunden</option>
                    <option value="6h" ${syncInterval === '6h' ? 'selected' : ''}>Alle 6 Stunden</option>
                    <option value="12h" ${syncInterval === '12h' ? 'selected' : ''}>Alle 12 Stunden</option>
                    <option value="24h" ${syncInterval === '24h' ? 'selected' : ''}>Taeglich (03:00)</option>
                    <option value="custom" ${syncInterval === 'custom' ? 'selected' : ''}>Benutzerdefiniert (Cron)</option>
                </select>
                ${normSync.needsReview ? `<div id="dsSyncReviewWarning" class="u-warn-box-warn">Nicht unterstuetzter Intervall-Wert &bdquo;${SharedUi.runtime.escapeHtml(normSync.original)}&ldquo; wurde auf &bdquo;Benutzerdefiniert&ldquo; gesetzt. Bitte pruefen und ggf. anpassen.</div>` : ''}
            </div>
            <div id="dsCronCustomField" class="${syncInterval === 'custom' ? 'u-block' : 'u-hidden'}">
                <div class="modal-field"><label>Cron-Ausdruck (5 Felder)</label>
                    <input type="text" id="dsCronExpr" value="${SharedUi.runtime.escapeAttr(cfg._cron_expr || '')}" placeholder="z.B. 0 */2 * * *"
                           class="u-mono">
                    <small class="u-c-secondary">Format: Minute Stunde Tag Monat Wochentag</small>
                </div>
            </div>
        </div>
    </details>`;
    return html;
}

/**
 * Bindet data-field-name-Buttons an ihre Handler (Browse, Folder-Load, Toggle-All).
 * Wird nach _buildConnectorFields / _onTypeChange aufgerufen.
 */
function _bindConnectorFieldEvents() {
    document.querySelectorAll('.ds-browse-btn[data-field-name]').forEach(btn => {
        btn.addEventListener('click', () => {
            SharedUi.runtime.withButtonLock(btn, () => _openBrowsePicker(btn.dataset.fieldName));
        });
    });
    document.querySelectorAll('.ds-load-folders-btn[data-field-name]').forEach(btn => {
        btn.addEventListener('click', () => _loadFolderTree(btn.dataset.fieldName));
    });
    document.querySelectorAll('.ds-toggle-all-folders[data-field-name]').forEach(cb => {
        cb.addEventListener('change', () => _toggleAllFolders(cb.dataset.fieldName));
    });
}

function _onTypeChange(newType) {
    const types = _connectorTypes || {};
    const ct = types[newType];
    const schema = ct?.schema || {};
    const defaults = ct?.default_config || {};
    const container = document.getElementById('dsConnectorFields');
    if (container) {
        container.innerHTML = _buildConnectorFields(schema, defaults);
        _bindConnectorFieldEvents();
    }
    const connDetails = document.getElementById('dsConnectorDetails');
    if (connDetails) connDetails.open = true;
    // Anhaenge-indexieren-Toggle nur fuer IMAP sichtbar
    const attField = document.getElementById('dsIndexAttachmentsField');
    if (attField) {
        if (newType === 'imap') {
            attField.classList.remove('u-hidden');
            attField.classList.add('u-block');
        } else {
            attField.classList.add('u-hidden');
            attField.classList.remove('u-block');
        }
    }
    // Test-Button nur fuer Remote-Konnektoren anzeigen
    _updateTestBtnVisibility();
}

function _bindDduOcrImagesToggle() {
    const dduCb = document.getElementById('dsDduEnabled');
    const ocrCb = document.getElementById('dsDduOcrImages');
    if (!dduCb || !ocrCb) return;
    const update = () => {
        ocrCb.disabled = !dduCb.checked;
        if (!dduCb.checked) ocrCb.checked = false;
    };
    dduCb.addEventListener('change', update);
}

function _bindContextualRetrievalToggle() {
    const crCb = document.getElementById('dsContextualRetrieval');
    const crSettings = document.getElementById('crSettings');
    if (!crCb || !crSettings) return;
    crCb.addEventListener('change', () => {
        if (crCb.checked) {
            crSettings.classList.remove('u-hidden');
            crSettings.classList.add('u-block');
        } else {
            crSettings.classList.add('u-hidden');
            crSettings.classList.remove('u-block');
        }
    });
}

async function _loadCrDropdowns(ds, openToken) {
    // Prompt-Templates fuer Enrichment laden
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/prompt-templates?chain_type=enrichment',
            {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        if (openToken !== undefined && openToken !== _editOpenToken) return; // Modal ersetzt
        const data = await resp.json();
        if (openToken !== undefined && openToken !== _editOpenToken) return; // Guard nach JSON-Parse
        const sel = document.getElementById('dsEnrichmentPromptId');
        if (sel && data.data) {
            data.data.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t.id;
                opt.textContent = t.name;
                sel.appendChild(opt);
            });
            if (ds && ds.ds_enrichment_prompt_id) {
                sel.value = ds.ds_enrichment_prompt_id;
            }
        }
    } catch (e) {
        console.error('Enrichment-Templates konnten nicht geladen werden:', e);
    }
    // Backends laden (LLM + Cloud)
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/backends/',
            {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        if (openToken !== undefined && openToken !== _editOpenToken) return; // Modal ersetzt
        const data = await resp.json();
        if (openToken !== undefined && openToken !== _editOpenToken) return; // Guard nach JSON-Parse
        const sel = document.getElementById('dsEnrichmentBackendId');
        const backends = data.data || data.backends || [];
        if (sel) {
            if (backends.length) {
                backends
                    .filter(b => b.category === 'llm' || b.category === 'cloud')
                    .forEach(b => {
                        const opt = document.createElement('option');
                        opt.value = b.id;
                        opt.textContent = b.name;
                        sel.appendChild(opt);
                    });
            }
            if (ds && ds.ds_enrichment_backend_id) {
                sel.value = ds.ds_enrichment_backend_id;
                // Mismatch-Check auch bei leerer Liste — sonst bleibt eine
                // verwaiste Backend-ID stumm bestehen, wenn alle LLM/Cloud-
                // Backends deaktiviert wurden
                if (String(sel.value) !== String(ds.ds_enrichment_backend_id)) {
                    SharedUi.notify.show(
                        `Gespeichertes Enrichment-Backend (ID ${ds.ds_enrichment_backend_id}) nicht mehr verfuegbar — bitte neu auswaehlen`,
                        'warning'
                    );
                }
            }
        }
    } catch (e) {
        console.error('Backends konnten nicht geladen werden:', e);
    }
}

function _updateTestBtnVisibility() {
    const dsTestBtn = document.getElementById('dsTestBtn');
    if (!dsTestBtn) return;
    const type = document.getElementById('dsType')?.value || 'local';
    if (type !== 'local') {
        dsTestBtn.classList.remove('u-hidden');
    } else {
        dsTestBtn.classList.add('u-hidden');
    }
}

async function _testDsConnection(dsId) {
    const formData = _getDsFormData();
    if (!formData) return;

    const btn = document.getElementById('dsTestBtn');
    btn.textContent = 'Teste...';
    btn.disabled = true;
    try {
        const payload = { type: formData.type, config: formData.connector_config };
        if (dsId) payload.ds_id = dsId;

        const resp = await SharedUi.fetch.safeFetch('/api/admin/datasources/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const result = await resp.json().catch(() => ({}));
        if (resp.ok) {
            SharedUi.notify.show(result.message || 'Verbindung erfolgreich', 'success');
        } else {
            SharedUi.notify.show(_dsResolveError(result, 'Verbindungstest fehlgeschlagen'), 'error');
        }
    } catch (e) {
        SharedUi.notify.show('Verbindungstest fehlgeschlagen', 'error');
    } finally {
        btn.textContent = 'Verbindung testen';
        btn.disabled = false;
    }
}

function toggleChunkFields(value) {
    const customFields = document.getElementById('dsChunkCustomFields');
    if (!customFields) return;
    if (value === 'custom') {
        customFields.classList.remove('u-hidden');
        customFields.classList.add('u-block');
    } else {
        customFields.classList.add('u-hidden');
        customFields.classList.remove('u-block');
        // Bei fester Strategie: chunk_size/overlap leeren (werden serverseitig aus STRATEGIES genommen)
        const sizeInput = document.getElementById('dsChunkSize');
        const overlapInput = document.getElementById('dsChunkOverlap');
        if (value !== 'auto') {
            // Vordefinierte Strategie: Werte anzeigen (read-only Info)
            const strategies = { prose: [768, 150], tabular: [256, 0], code: [512, 100], structured: [384, 50], email: [640, 100] };
            if (strategies[value]) {
                if (sizeInput) sizeInput.value = '';
                if (overlapInput) overlapInput.value = '';
            }
        } else {
            if (sizeInput) sizeInput.value = '';
            if (overlapInput) overlapInput.value = '';
        }
    }
}

function _parseCommaSeparated(val) {
    if (!val || !val.trim()) return null;
    return val.split(',').map(s => s.trim()).filter(Boolean);
}

function _collectConnectorConfig(schema) {
    const config = {};
    for (const [fieldName, spec] of Object.entries(schema)) {
        const el = document.getElementById(_connFieldId(fieldName));
        if (!el) continue;
        if (spec.type === 'bool') {
            config[fieldName] = el.checked;
        } else if (spec.type === 'int') {
            const parsed = parseInt(el.value);
            config[fieldName] = isNaN(parsed) ? (spec.default ?? 0) : parsed;
        } else if (spec.secret) {
            // PASSWORD_MASK durchreichen — Backend erkennt Maske und behaelt gespeicherten Wert
            config[fieldName] = el.value || '';
        } else {
            config[fieldName] = el.value ?? '';
        }
    }
    return config;
}

function _getDsFormData() {
    const chunkStrategy = document.getElementById('dsChunkStrategy')?.value || 'auto';
    const chunkSize = document.getElementById('dsChunkSize')?.value;
    const chunkOverlap = document.getElementById('dsChunkOverlap')?.value;
    const type = document.getElementById('dsType')?.value || 'local';

    // Konnektor-Config dynamisch aus Schema auslesen
    const types = _connectorTypes || {};
    const schema = types[type]?.schema || {};
    const connector_config = _collectConnectorConfig(schema);

    // Pflichtfeld-Validierung
    for (const [fieldName, spec] of Object.entries(schema)) {
        const val = connector_config[fieldName];
        if (spec.required && (val === undefined || val === null || val === '')) {
            SharedUi.notify.show(`"${spec.label || fieldName}" ist ein Pflichtfeld`, 'error');
            document.getElementById(_connFieldId(fieldName))?.focus();
            return null;
        }
    }

    return {
        name: document.getElementById('dsName')?.value || '',
        type: type,
        connector_config: connector_config,
        // Per-DS Config (bleibt Top-Level)
        ds_chunk_strategy: chunkStrategy,
        ds_chunk_size: chunkSize ? parseInt(chunkSize) : null,
        ds_chunk_overlap: chunkOverlap ? parseInt(chunkOverlap) : null,
        ds_file_types: _parseCommaSeparated(document.getElementById('dsFileTypes')?.value),
        ds_exclude_patterns: _parseCommaSeparated(document.getElementById('dsExcludePatterns')?.value),
        ds_ddu_enabled: document.getElementById('dsDduEnabled')?.checked || false,
        ds_ddu_ocr_images: document.getElementById('dsDduOcrImages')?.checked ?? true,
        ds_vlm_enabled: document.getElementById('dsVlmEnabled')?.checked || false,
        ds_contextual_retrieval: document.getElementById('dsContextualRetrieval')?.checked || false,
        ds_enrichment_prompt_id: parseInt(document.getElementById('dsEnrichmentPromptId')?.value) || null,
        ds_enrichment_backend_id: parseInt(document.getElementById('dsEnrichmentBackendId')?.value) || null,
        ds_colbert_enabled: document.getElementById('dsColbertEnabled')?.checked || false,
        ds_accept_all_text: document.getElementById('dsAcceptAllText')?.checked || false,
        ds_index_attachments: document.getElementById('dsIndexAttachments')?.checked ?? true,
        restriction_level: document.getElementById('dsRestrictionLevel')?.value || 'anonymize',
        sync_interval: document.getElementById('dsSyncInterval')?.value || 'global',
        _cron_expr: document.getElementById('dsCronExpr')?.value || '',
    };
}

function _onSyncIntervalChange(value) {
    const customField = document.getElementById('dsCronCustomField');
    if (!customField) return;
    if (value === 'custom') {
        customField.classList.remove('u-hidden');
        customField.classList.add('u-block');
    } else {
        customField.classList.add('u-hidden');
        customField.classList.remove('u-block');
    }
}

// ---------------------------------------------------------------------------
// Lokale Helper (Sprint 3): Warning, Availability, Modal-Common, Recovery
// ---------------------------------------------------------------------------

/**
 * Erzeugt ein Warning-DOM-Element mit einheitlichem Styling.
 * severity 'error' = rot, 'warn' = gelb. Setzt Text via textContent.
 */
function _createDsWarning(message, severity, id) {
    const el = document.createElement('div');
    if (id) el.id = id;
    if (severity === 'error') {
        el.className = 'u-warn-box-error';
    } else {
        el.className = 'u-warn-box-warn';
    }
    el.textContent = message;
    return el;
}

/**
 * Deaktiviert eine Feature-Checkbox und zeigt eine rote Availability-Warning,
 * wenn das Feature nicht verfuegbar ist. Falsy-Semantik: undefined = nicht verfuegbar.
 */
function _applyAvailabilityGuard(checkboxId, available, message) {
    const cb = document.getElementById(checkboxId);
    if (!cb || available) return;
    cb.disabled = true;
    cb.checked = false;
    cb.parentElement.parentElement.appendChild(_createDsWarning(message, 'error'));
}

/**
 * Gemeinsame Post-Render-Schritte fuer Create- und Edit-Modal.
 * Bindet Connector-Events, Test-Button, Availability-Guards, Toggles und CR-Dropdowns.
 */
function _bindDatasourceModalCommon({ dsId, ds, openToken }) {
    _bindConnectorFieldEvents();
    _updateTestBtnVisibility();
    document.getElementById('dsTestBtn')?.addEventListener('click', () => _testDsConnection(dsId));
    _applyAvailabilityGuard('dsDduEnabled', window._dduAvailable,
        'DDU nicht verfuegbar: Kein erreichbares Remote-DDU-Backend konfiguriert.');
    _applyAvailabilityGuard('dsVlmEnabled', window._vlmAvailable,
        window._vlmMessage || 'VLM nicht verfuegbar: Kein VLM-Modell konfiguriert (ollama.vlm_model in kiara.yaml)');
    _bindDduOcrImagesToggle();
    _bindContextualRetrievalToggle();
    _loadCrDropdowns(ds || null, openToken);
}

/**
 * Behandelt Schedule-Partial-Failure nach erfolgreicher DS-Persistierung.
 * Schliesst Modal, laedt Tabelle, zeigt Warning und oeffnet Edit mit remaskierten Secrets.
 */
async function _recoverDatasourceSchedulePartialFailure({ warningText, dsId, fallbackDraft }) {
    SharedUi.modal.close();
    if (dsTable) await dsTable.loadData();
    SharedUi.notify.show(warningText, 'warning');
    _remaskDraftSecrets(fallbackDraft);
    openEditDatasourceModal(dsId, fallbackDraft);
}

function openCreateDatasourceModal() {
    _editOpenToken++;
    const myToken = _editOpenToken;
    _currentDsId = null;
    SharedUi.modal.open({
        title: 'Neue Datenquelle',
        body: _dsFormHtml(null),
        maxWidth: '540px',
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn" id="dsTestBtn">Verbindung testen</button>
                 <button class="action-btn primary" id="dsCreateBtn">Erstellen</button>`,
        onOpen: () => {
            document.getElementById('dsName')?.focus();
            _bindDatasourceModalCommon({ dsId: null, ds: null, openToken: myToken });
            document.getElementById('dsCreateBtn').addEventListener('click', function() {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const data = _getDsFormData();
                    if (!data) return;
                    if (!data.name.trim()) { SharedUi.notify.show('Name ist erforderlich', 'error'); return; }
                    // Custom-Cron Validierung
                    if (data.sync_interval === 'custom' && !(data._cron_expr || '').trim()) {
                        SharedUi.notify.show('Benutzerdefinierter Cron-Ausdruck darf nicht leer sein', 'error');
                        return;
                    }
                    // Phase 1: DS persistieren — _cron_expr ist Frontend-only, nicht Teil des DS-Bodys
                    let dsResult;
                    try {
                        const { _cron_expr: _omitCron, ...dsPayload } = data;
                        const resp = await SharedUi.fetch.safeFetch('/api/admin/datasources', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(dsPayload),
                        });
                        const body = await resp.json();
                        if (!resp.ok) throw new Error(_dsResolveError(body, 'Datenquelle konnte nicht erstellt werden'));
                        dsResult = body;
                    } catch (e) { SharedUi.notify.show(e.message, 'error'); return; }

                    // Phase 2: Schedule-Write (Partial-Success moeglich)
                    if (dsResult.id) {
                        try {
                            await _saveDsSchedule(dsResult.id, data.sync_interval, data._cron_expr, {
                                deleteScheduleless: false,
                            });
                        } catch (schedErr) {
                            // DS ist bereits persistiert — Partial-Success-Recovery
                            await _recoverDatasourceSchedulePartialFailure({
                                warningText: 'Datenquelle erstellt, aber Zeitplan fehlgeschlagen: ' + (schedErr.message || 'Unbekannter Fehler'),
                                dsId: dsResult.id,
                                fallbackDraft: { ...data, id: dsResult.id },
                            });
                            return;
                        }
                    }
                    SharedUi.notify.show('Datenquelle erstellt', 'success');
                    SharedUi.modal.close();
                    if (dsTable) await dsTable.loadData();
                });
            });
        },
    });
}

function openEditDatasourceModal(dsId, fallbackDs = null) {
    // Invalidiere vorherige Open-Tokens BEVOR SharedUi.modal.open den alten Overlay entfernt
    _editOpenToken++;
    const myToken = _editOpenToken;

    // Daten: fallbackDs hat Vorrang fuer editierte Felder, Table-Payload ist Fallback
    const tableDs = dsTable ? dsTable.getAllData().find(d => d.id === dsId) : null;
    if (!tableDs && !fallbackDs) return;

    // Gemergter Modal-State: fallbackDs uebersteuert fuer user-editable Felder
    const ds = fallbackDs ? { ...(tableDs || {}), ...fallbackDs, id: dsId } : { ...tableDs };
    // Bei bestehenden DS: type immer aus Table-/Original-Wert (nicht aus Draft)
    if (tableDs && fallbackDs) ds.type = tableDs.type || fallbackDs.type;

    _currentDsId = dsId;

    // Schedule-Feld-Dirty-Tracking
    let _scheduleFieldsDirty = false;

    SharedUi.modal.open({
        title: `Datenquelle bearbeiten: ${SharedUi.runtime.escapeHtml(ds.name)}`,
        body: _dsFormHtml(ds),
        maxWidth: '540px',
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn" id="dsTestBtn">Verbindung testen</button>
                 <button class="action-btn primary" id="dsSaveBtn">Speichern</button>`,
        onOpen: () => {
            // Bestehende DS: Typ-Feld deaktivieren (kein scheinbarer Typwechsel)
            const dsTypeEl = document.getElementById('dsType');
            if (dsTypeEl) {
                dsTypeEl.disabled = true;
                dsTypeEl.value = ds.type || 'local';
            }

            // Secret-Felder mit PASSWORD_MASK rendern
            const types = _connectorTypes || {};
            const schema = types[ds.type]?.schema || {};
            for (const [fieldName, spec] of Object.entries(schema)) {
                if (spec.secret) {
                    const el = document.getElementById(_connFieldId(fieldName));
                    if (el && ds.connector_config && ds.connector_config[fieldName]) {
                        el.value = DS_PASSWORD_MASK;
                        el.placeholder = '(unveraendert wenn ********)';
                    }
                }
            }

            _bindDatasourceModalCommon({ dsId, ds, openToken: myToken });

            // Dirty-Tracking fuer Schedule-Felder
            const syncSel = document.getElementById('dsSyncInterval');
            const cronInput = document.getElementById('dsCronExpr');
            if (syncSel) syncSel.addEventListener('change', () => { _scheduleFieldsDirty = true; });
            if (cronInput) cronInput.addEventListener('input', () => { _scheduleFieldsDirty = true; });

            // Asynchroner Schedule-Read (nur ohne fallbackDs)
            if (!fallbackDs) {
                _loadScheduleForEdit(dsId, ds, myToken, () => _scheduleFieldsDirty);
            }

            document.getElementById('dsSaveBtn').addEventListener('click', function() {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const data = _getDsFormData();
                    if (!data) return;
                    if (!data.name.trim()) { SharedUi.notify.show('Name ist erforderlich', 'error'); return; }
                    // Custom-Cron Validierung
                    if (data.sync_interval === 'custom' && !(data._cron_expr || '').trim()) {
                        SharedUi.notify.show('Benutzerdefinierter Cron-Ausdruck darf nicht leer sein', 'error');
                        return;
                    }
                    // Type fuer bestehende DS fixieren
                    data.type = ds.type || data.type;

                    // Phase 1: DS persistieren — _cron_expr ist Frontend-only, nicht Teil des DS-Bodys
                    try {
                        const { _cron_expr: _omitCron, ...dsPayload } = data;
                        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
                            body: JSON.stringify(dsPayload),
                        });
                        const body = await resp.json();
                        if (!resp.ok) throw new Error(_dsResolveError(body, 'Datenquelle konnte nicht aktualisiert werden'));
                    } catch (e) { SharedUi.notify.show(e.message, 'error'); return; }

                    // Phase 2: Schedule-Write (Partial-Success moeglich)
                    try {
                        await _saveDsSchedule(dsId, data.sync_interval, data._cron_expr);
                    } catch (schedErr) {
                        await _recoverDatasourceSchedulePartialFailure({
                            warningText: 'Datenquelle aktualisiert, aber Zeitplan fehlgeschlagen: ' + (schedErr.message || 'Unbekannter Fehler'),
                            dsId: dsId,
                            fallbackDraft: { ...data, id: dsId, type: ds.type },
                        });
                        return;
                    }
                    SharedUi.notify.show('Datenquelle aktualisiert', 'success');
                    SharedUi.modal.close();
                    if (dsTable) await dsTable.loadData();
                });
            });
        },
        onClose: () => {
            // Zweiter Invalidation-Pfad fuer echte Close-Vorgaenge — Token muss
            // unbedingt monoton steigen, sonst koennen stale fetch-Antworten in
            // ein neueres Modal schreiben.
            _editOpenToken++;
        },
    });
}

let _deleteInProgress = false;
async function deleteDatasource(dsId, name, isActive) {
    if (_deleteInProgress) return;
    if (isActive) {
        await SharedUi.modal.alert({ message: `Datenquelle "${name}" kann waehrend einer laufenden Ingestion nicht geloescht werden. Bitte zuerst die Indexierung abbrechen.` });
        return;
    }
    _deleteInProgress = true;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}`, { method: 'DELETE' });
        if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(_dsResolveError(e, 'Datenquelle konnte nicht geloescht werden')); }
        SharedUi.notify.show('Datenquelle geloescht', 'success');
        if (dsTable) dsTable.loadData();
    } catch (e) { SharedUi.notify.show(e.message, 'error'); } finally { _deleteInProgress = false; }
}

// ---------------------------------------------------------------------------
// Groups Modal
// ---------------------------------------------------------------------------

async function openDsGroupsModal(dsId, dsName) {
    // Load groups, users and current assignments in parallel
    let allGroups, assignedGroupIds, allUsers, assignedUserIds, assignedUsersRaw;
    try {
        const [groupsResp, assignedGroupsResp, usersResp, assignedUsersResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/groups'),
            SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/groups`),
            SharedUi.fetch.safeFetch('/api/admin/users'),
            SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/users`),
        ]);
        // Non-2xx als harten Fehler behandeln (nicht mit leeren Listen weiterrendern)
        if (!groupsResp.ok || !assignedGroupsResp.ok || !usersResp.ok || !assignedUsersResp.ok) {
            const failedResp = [groupsResp, assignedGroupsResp, usersResp, assignedUsersResp].find(r => !r.ok);
            const errBody = await failedResp.json().catch(() => ({}));
            throw new Error(_dsResolveError(errBody, 'Berechtigungsdaten konnten nicht geladen werden'));
        }
        const groupsData = await groupsResp.json();
        allGroups = groupsData.data || [];
        const assignedGroupsData = await assignedGroupsResp.json();
        assignedGroupIds = new Set((assignedGroupsData.groups || []).map(g => g.id));
        const usersData = await usersResp.json();
        const activeUsers = (usersData.data || []).filter(u => u.is_active);
        const assignedUsersData = await assignedUsersResp.json();
        assignedUsersRaw = assignedUsersData.users || [];
        assignedUserIds = new Set(assignedUsersRaw.map(u => u.id));

        // Inaktive, aber zugewiesene Benutzer in die Liste zurueckmischen
        const activeUserIds = new Set(activeUsers.map(u => u.id));
        const inactiveAssigned = assignedUsersRaw
            .filter(u => !activeUserIds.has(u.id))
            .map(u => ({ ...u, _inactiveAssigned: true }));
        allUsers = [...activeUsers, ...inactiveAssigned];
    } catch (e) {
        SharedUi.notify.show(e.message || 'Berechtigungen konnten nicht geladen werden', 'error');
        return;
    }

    function _renderPermBody() {
        // Tab-Header
        let body = `<div class="ds-perm-tabs">
            <button type="button" class="ds-perm-tab active" data-tab="groups">Gruppen</button>
            <button type="button" class="ds-perm-tab" data-tab="users">Benutzer</button>
        </div>`;

        body += '<p class="u-text-sm-secondary u-mb-12">Datenquellen ohne Gruppen- oder Benutzerzuweisung sind fuer alle Benutzer sichtbar.</p>';

        // Groups Tab Content
        body += '<div id="dsPermGroupsTab">';
        if (allGroups.length === 0) {
            body += '<p class="u-text-sm-secondary">Keine Gruppen importiert. Bitte zuerst LDAP-Gruppen importieren.</p>';
        } else {
            body += '<div class="u-perm-grid">';
            allGroups.forEach(g => {
                const checked = assignedGroupIds.has(g.id) ? 'checked' : '';
                body += `<input type="checkbox" class="ds-group-cb" value="${g.id}" ${checked}>`;
                body += `<span class="u-nowrap">${SharedUi.runtime.escapeHtml(g.cn)}</span>`;
                body += `<span class="u-text-xxs-secondary u-nowrap u-text-right">${g.member_count || 0} Mitglieder</span>`;
            });
            body += '</div>';
        }
        body += '</div>';

        // Users Tab Content (hidden initially)
        body += '<div id="dsPermUsersTab" class="u-hidden">';
        if (allUsers.length === 0) {
            body += '<p class="u-text-sm-secondary">Keine Benutzer vorhanden.</p>';
        } else {
            body += '<div class="u-perm-grid">';
            allUsers.forEach(u => {
                const checked = assignedUserIds.has(u.id) ? 'checked' : '';
                const isInactive = u._inactiveAssigned || !u.is_active;
                const displayName = (u.vorname || u.nachname)
                    ? `${SharedUi.runtime.escapeHtml(u.username)} (${SharedUi.runtime.escapeHtml([u.vorname, u.nachname].filter(Boolean).join(' '))})`
                    : SharedUi.runtime.escapeHtml(u.username);
                const source = u.auth_source === 'ldap' ? 'LDAP' : (u.auth_source ? SharedUi.runtime.escapeHtml(u.auth_source) : '');
                const statusHint = isInactive ? ' <span class="u-c-danger u-fs-10">(inaktiv)</span>' : '';
                const assignedHint = u._inactiveAssigned ? ' <span class="u-c-warning u-fs-10">(bereits zugewiesen)</span>' : '';
                body += `<input type="checkbox" class="ds-user-cb" value="${u.id}" ${checked}>`;
                body += `<span class="u-nowrap">${displayName}${statusHint}${assignedHint}</span>`;
                body += `<span class="u-text-xxs-secondary u-nowrap u-text-right">${source}</span>`;
            });
            body += '</div>';
        }
        body += '</div>';
        return body;
    }

    function _bindPermTabs() {
        document.querySelectorAll('.ds-perm-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.ds-perm-tab').forEach(t => {
                    t.classList.remove('active');
                });
                tab.classList.add('active');

                const target = tab.dataset.tab;
                const groupsTab = document.getElementById('dsPermGroupsTab');
                const usersTab = document.getElementById('dsPermUsersTab');
                if (groupsTab) {
                    if (target === 'groups') groupsTab.classList.remove('u-hidden');
                    else groupsTab.classList.add('u-hidden');
                }
                if (usersTab) {
                    if (target === 'users') usersTab.classList.remove('u-hidden');
                    else usersTab.classList.add('u-hidden');
                }
            });
        });
    }

    SharedUi.modal.open({
        title: `Berechtigungen: ${SharedUi.runtime.escapeHtml(dsName)}`,
        body: _renderPermBody(),
        maxWidth: '700px',
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn primary" id="dsPermSaveBtn">Speichern</button>`,
        onOpen: () => {
            _bindPermTabs();

            // Serieller, button-gelockter Save
            document.getElementById('dsPermSaveBtn').addEventListener('click', function() {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const groupCbs = document.querySelectorAll('.ds-group-cb:checked');
                    const group_ids = Array.from(groupCbs).map(cb => parseInt(cb.value));
                    const userCbs = document.querySelectorAll('.ds-user-cb:checked');
                    const user_ids = Array.from(userCbs).map(cb => parseInt(cb.value));

                    // Schritt 1: Gruppen speichern (safeFetch wirft bei Nicht-2xx)
                    try {
                        await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/groups`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ group_ids }),
                        });
                    } catch (gErr) {
                        SharedUi.notify.show(_dsResolveError(gErr && gErr.body, 'Fehler beim Speichern der Gruppen'), 'error');
                        return; // Modal bleibt offen, kein Re-Sync
                    }

                    // Schritt 2: Benutzer speichern
                    try {
                        await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/users`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ user_ids }),
                        });
                    } catch (uErr) {
                        // Corrective Re-Sync: Gruppen sind gespeichert, User nicht
                        try {
                            const [rGroups, rUsers] = await Promise.all([
                                SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/groups`),
                                SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/users`),
                            ]);
                            const rGroupsData = await rGroups.json();
                            const rUsersData = await rUsers.json();
                            assignedGroupIds = new Set((rGroupsData.groups || []).map(g => g.id));
                            assignedUserIds = new Set((rUsersData.users || []).map(u => u.id));
                            // Modal-Body mit Server-State neu aufbauen
                            const modalBody = document.querySelector('.modal-body');
                            if (modalBody) {
                                const activeTab = document.querySelector('.ds-perm-tab.active')?.dataset.tab;
                                modalBody.innerHTML = _renderPermBody();
                                _bindPermTabs();
                                if (activeTab && activeTab !== 'groups') {
                                    const tabBtn = document.querySelector(`.ds-perm-tab[data-tab="${activeTab}"]`);
                                    if (tabBtn) tabBtn.click();
                                }
                            }
                            SharedUi.notify.show(_dsResolveError(uErr && uErr.body, 'Fehler beim Speichern der Benutzer. Checkboxen wurden auf den aktuellen Server-Stand zurueckgesetzt.'), 'error');
                        } catch (_reSyncErr) {
                            // Re-Sync fehlgeschlagen — Modal hart schliessen
                            SharedUi.modal.close();
                            SharedUi.notify.show('Berechtigungen teilweise gespeichert. Neulade-Versuch fehlgeschlagen — bitte erneut oeffnen.', 'warning');
                            if (dsTable) dsTable.loadData();
                        }
                        return;
                    }

                    SharedUi.notify.show('Berechtigungen gespeichert', 'success');
                    SharedUi.modal.close();
                    if (dsTable) dsTable.loadData();
                });
            });
        },
    });
}

// ---------------------------------------------------------------------------
// Ingestion trigger (async with polling)
// ---------------------------------------------------------------------------

function _getIngestEmbedModel() {
    const sel = document.getElementById('dsEmbedModelFilter');
    const model = (sel && sel.value) ? sel.value : _defaultEmbedModel;
    return model || _defaultEmbedModel;
}

async function _triggerIngest(dsId, dsName, reingest) {
    const embModel = _getIngestEmbedModel();
    if (!embModel) {
        SharedUi.notify.show('Kein Embedding-Modell verfuegbar. Bitte unter KI-Backends konfigurieren.', 'error');
        return;
    }
    // Late-Chunking-Status fuer gewaehltes Modell
    const lcSupported = !!_embedModelLateChunking[embModel];
    const lcBoxCls = lcSupported ? 'u-lc-box-success' : 'u-lc-box-danger';
    const lcText = lcSupported
        ? '<strong>' + SharedUi.runtime.escapeHtml(embModel) + '</strong> unterstuetzt Late Chunking'
        : '<strong>' + SharedUi.runtime.escapeHtml(embModel) + '</strong> unterstuetzt kein Late Chunking';
    const lcInfoHtml = '<div class="' + lcBoxCls + '">' + lcText + '</div>';

    const mainMsg = reingest
        ? SharedUi.runtime.escapeHtml('"' + dsName + '" voll-indexieren mit Modell "' + embModel + '"?')
        : SharedUi.runtime.escapeHtml('"' + dsName + '" indexieren mit Modell "' + embModel + '"?');
    const subMsg = reingest
        ? 'Alle bestehenden Chunks werden geloescht und neu eingelesen.'
        : 'Nur neue/geaenderte Dateien werden verarbeitet.';

    const confirmed = await new Promise(resolve => {
        let settled = false;
        const settle = (result) => {
            if (settled) return;
            settled = true;
            SharedUi.modal.close(modalId);
            resolve(result);
        };
        const bodyHtml = `
            <p class="u-m-0 u-fs-14">${mainMsg}</p>
            <p class="u-mt-8-0 u-fs-13 u-c-secondary">${SharedUi.runtime.escapeHtml(subMsg)}</p>
            ${lcInfoHtml}`;
        const modalId = SharedUi.modal.open({
            ariaLabel: 'Bestaetigung Ingestion',
            body: bodyHtml,
            maxWidth: '440px',
            stack: true,
            backdropClose: true,
            buttons: [
                { label: 'Abbrechen', onClick: () => settle(false) },
                { label: 'Ja', className: 'primary', onClick: () => settle(true) },
            ],
            onClose: () => { if (!settled) { settled = true; resolve(false); } },
        });
    });
    if (!confirmed) return;

    const endpoint = reingest
        ? `/api/admin/datasources/${dsId}/reingest`
        : `/api/admin/datasources/${dsId}/ingest`;
    try {
        const resp = await SharedUi.fetch.safeFetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embedding_model: embModel }),
        });
        const result = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(result.detail || result.error || result.message || 'Fehler');

        SharedUi.notify.show(reingest ? 'Voll-Ingestion gestartet...' : 'Ingestion gestartet...', 'info');
        if (dsTable) dsTable.loadData();
        // Polling startet automatisch via onDataLoaded wenn running-Status erkannt
    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
}

async function triggerDsIngest(dsId, dsName) {
    await _triggerIngest(dsId, dsName, false);
}

async function triggerDsReingest(dsId, dsName) {
    await _triggerIngest(dsId, dsName, true);
}

async function triggerDsHtmlReingest(dsId, dsName) {
    const embModel = _getIngestEmbedModel();
    if (!_HTML_REINGEST_MODELS.has(embModel)) {
        SharedUi.notify.show('HTML-Neuaufbereitung ist für Nomic und Hellord freigegeben. Bitte das gewünschte Modell auswählen.', 'error');
        return;
    }
    const modeId = `htmlReingestMode-${dsId}`;
    const mode = await new Promise(resolve => {
        let settled = false;
        const settle = (value) => {
            if (settled) return;
            settled = true;
            SharedUi.modal.close(modalId);
            resolve(value);
        };
        const modalId = SharedUi.modal.open({
            ariaLabel: 'HTML neu aufbereiten',
            body: `
                <p class="u-m-0 u-fs-14">Bestehende HTML-Inhalte von <strong>${SharedUi.runtime.escapeHtml(dsName)}</strong> mit Modell <strong>${SharedUi.runtime.escapeHtml(embModel)}</strong> neu aufbereiten?</p>
                <p class="u-mt-8-0 u-fs-13 u-c-secondary">Der Ingestion-Job ermittelt HTML-Dateien und tatsächlich verwendete HTML-Mailbodies. Für alte Mailcaches liest er bei Bedarf das Original aus dem Postfach. Fehlende Originale, leere Bodies und Fehler erscheinen getrennt im Jobergebnis.</p>
                <label for="${modeId}">Auswahl</label>
                <select id="${modeId}" class="form-input u-w-full">
                    <option value="outdated" selected>Nur veraltete Aufbereitungsstände</option>
                    <option value="all">Alle bestätigten HTML-Inhalte</option>
                </select>
                <p class="u-mt-8-0 u-fs-13 u-c-secondary">Es werden nur vorhandene Bestände des gewählten Modells bearbeitet. Fortschritt und Abbruch stehen unter Jobs zur Verfügung.</p>`,
            maxWidth: '540px',
            stack: true,
            backdropClose: true,
            buttons: [
                { label: 'Abbrechen', onClick: () => settle(null) },
                { label: 'Job starten', className: 'primary', onClick: () => {
                    const selected = document.getElementById(modeId)?.value;
                    if (selected === 'outdated' || selected === 'all') settle(selected);
                } },
            ],
            onClose: () => { if (!settled) { settled = true; resolve(null); } },
        });
    });
    if (mode === null) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/reingest-html`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embedding_model: embModel, mode }),
        });
        const result = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(result.detail || result.error || result.message || 'Fehler');
        SharedUi.notify.show(`HTML-Neuaufbereitung als Job #${result.job_id} eingestellt.`, 'info');
        if (dsTable) dsTable.loadData();
    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
}

function _startDsPolling() {
    if (kiaraPolling.has('datenquellen:refresh')) return;
    let errorCount = 0;
    kiaraPolling.start('datenquellen:refresh', async () => {
        if (!dsTable) { kiaraPolling.stop('datenquellen:refresh'); return; }
        const table = dsTable;
        try {
            const refreshed = await table.silentRefresh();
            if (dsTable !== table) return;
            if (refreshed === false) {
                _dsRateHistory.clear();
                table.render();
                return;
            }
            errorCount = 0;
            const rows = dsTable.getAllData ? dsTable.getAllData() : [];
            const hasActive = rows.some(r =>
                r.ingestion_status === 'running' || r.ingestion_status === 'paused' || r.ingestion_status === 'pending');
            if (!hasActive) kiaraPolling.stop('datenquellen:refresh');
        } catch (e) {
            if (dsTable !== table) return;
            _dsRateHistory.clear();
            table.render();
            errorCount++;
            if (errorCount >= 3) kiaraPolling.stop('datenquellen:refresh');
        }
    }, 3000);
}

// ---------------------------------------------------------------------------
// Pause / Cancel Ingestion
// ---------------------------------------------------------------------------

function pauseDsIngestion(btn, dsId) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/pause-ingestion`, { method: 'POST' });
        const result = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(result.detail || result.error || result.message || 'Fehler');
        SharedUi.notify.show(result.message || 'Ingestion-Status geaendert', 'info');
        if (dsTable) await dsTable.loadData();
    }).catch(e => SharedUi.notify.show(e.message, 'error'));
}

function cancelDsIngestion(btn, dsId, dsName) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        if (!await SharedUi.modal.confirm({ message: `Ingestion von "${dsName}" wirklich abbrechen?` })) return;
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/cancel-ingestion`, { method: 'POST' });
        const result = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(result.detail || result.error || result.message || 'Fehler');
        SharedUi.notify.show('Ingestion wird abgebrochen...', 'info');
        if (dsTable) await dsTable.loadData();
    }).catch(e => SharedUi.notify.show(e.message, 'error'));
}

// ---------------------------------------------------------------------------
// SMB Folder Picker
// ---------------------------------------------------------------------------

async function _openBrowsePicker(fieldName) {
    const existingPicker = document.getElementById('fpOverlay');
    if (existingPicker) existingPicker.remove();

    const type = document.getElementById('dsType')?.value || '';
    const types = _connectorTypes || {};
    const ct = types[type];
    const schema = ct?.schema || {};
    const browseSpec = schema[fieldName]?.browse;
    if (!browseSpec) return;

    // Abhaengige Felder auslesen
    const browseData = {};
    for (const dep of (browseSpec.depends_on || [])) {
        const el = document.getElementById(_connFieldId(dep));
        browseData[dep] = el?.value || '';
    }
    browseData.path = '';
    if (_currentDsId) browseData.ds_id = _currentDsId;

    // Pflichtfelder pruefen
    const missingRequired = (browseSpec.depends_on || []).filter(dep => {
        return schema[dep]?.required && !browseData[dep];
    });
    if (missingRequired.length > 0) {
        const labels = missingRequired.map(d => schema[d]?.label || d).join(', ');
        SharedUi.notify.show(`Bitte zuerst ausfuellen: ${labels}`, 'error');
        return;
    }

    // V2-Fix M4: Generischer Titel aus Connector-Label + erstem Feldwert
    const connLabel = ct?.label || type;
    const firstDepValue = browseData[(browseSpec.depends_on || [])[0]] || '';
    const titleStr = firstDepValue ? `${connLabel}: ${firstDepValue}` : connLabel;

    // Verzeichnis laden
    async function loadDir(path) {
        browseData.path = path;
        try {
            const resp = await SharedUi.fetch.safeFetch(browseSpec.endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(browseData),
            });
            const result = await resp.json().catch(() => ({}));
            // Non-2xx oder explizites error-Feld: harter Abbruch
            if (!resp.ok) {
                SharedUi.notify.show(_dsResolveError(result, 'Verzeichnis konnte nicht geladen werden'), 'error');
                return null;
            }
            if (result.error) {
                SharedUi.notify.show(_dsResolveError(result, 'Verzeichnis konnte nicht geladen werden'), 'error');
                return null;
            }
            return result;
        } catch (e) {
            SharedUi.notify.show(
                _dsResolveError(e.body || e, e.message || 'Verbindung fehlgeschlagen'),
                'error',
            );
            return null;
        }
    }

    function renderBrowser(result) {
        const pathDisplay = result.path || '/';
        let html = `<div class="u-flex-center-gap-8 u-mb-12">`;
        html += `<span class="u-text-sm-secondary">Pfad:</span>`;
        html += `<code class="u-fp-path">/${SharedUi.runtime.escapeHtml(pathDisplay)}</code>`;
        if (result.parent !== null && result.parent !== undefined) {
            html += `<button class="action-btn-sm" id="fpUpBtn" title="Uebergeordnetes Verzeichnis">&#x2B06;</button>`;
        }
        html += `</div>`;
        html += `<div class="u-fp-entries">`;
        const entries = result.entries || [];
        if (entries.length === 0) {
            html += `<div class="u-fp-empty">Keine Unterverzeichnisse</div>`;
        }
        entries.forEach(e => {
            const childPath = result.path ? result.path + '/' + e.name : e.name;
            html += `<div class="fp-entry u-fp-entry" data-path="${SharedUi.runtime.escapeAttr(childPath)}">
                <span class="u-fs-16">&#x1F4C1;</span>
                <span>${SharedUi.runtime.escapeHtml(e.name)}</span>
            </div>`;
        });
        html += `</div>`;
        return html;
    }

    // Initial laden
    const initial = await loadDir('');
    if (!initial) return;
    let currentPath = '';

    // Sekundaeres Modal (ueber dem DS-Modal)
    const overlay = document.createElement('div');
    overlay.id = 'fpOverlay';
    overlay.className = 'u-fp-overlay';
    const modal = document.createElement('div');
    modal.className = 'u-fp-modal';
    modal.innerHTML = `
        <h3 class="u-my-0-12">Verzeichnis auswaehlen</h3>
        <p class="u-text-sm-secondary u-mb-12">${SharedUi.runtime.escapeHtml(titleStr)}</p>
        <div id="fpBrowserContent">${renderBrowser(initial)}</div>
        <div class="u-fp-modal-footer">
            <button class="action-btn" id="fpCancelBtn">Abbrechen</button>
            <button class="action-btn primary" id="fpSelectBtn">Auswaehlen</button>
        </div>`;
    overlay.appendChild(modal);
    // Picker als Child des aktuellen Modal-Overlay einhaengen (Fallback: body),
    // damit beim Entfernen des Parent-Modals der Picker als Subtree mit verschwindet.
    const parentOverlay = document.querySelector('.modal-overlay');
    (parentOverlay || document.body).appendChild(overlay);

    function attachEvents() {
        document.getElementById('fpUpBtn')?.addEventListener('click', async () => {
            const parent = currentPath.split('/').slice(0, -1).join('/');
            const result = await loadDir(parent);
            if (result) {
                currentPath = parent;
                document.getElementById('fpBrowserContent').innerHTML = renderBrowser(result);
                attachEvents();
            }
        });
        document.querySelectorAll('.fp-entry').forEach(el => {
            el.addEventListener('click', async () => {
                const path = el.dataset.path;
                const result = await loadDir(path);
                if (result) {
                    currentPath = path;
                    document.getElementById('fpBrowserContent').innerHTML = renderBrowser(result);
                    attachEvents();
                }
            });
        });
    }
    attachEvents();

    function closePicker() {
        document.removeEventListener('keydown', escHandler);
        if (overlay.parentNode) overlay.remove();
    }
    const escHandler = (e) => {
        if (!document.body.contains(overlay)) {
            document.removeEventListener('keydown', escHandler);
            return;
        }
        if (e.key === 'Escape') closePicker();
    };
    document.addEventListener('keydown', escHandler);

    document.getElementById('fpCancelBtn').addEventListener('click', closePicker);
    document.getElementById('fpSelectBtn').addEventListener('click', () => {
        const input = document.getElementById(_connFieldId(fieldName));
        if (input) input.value = currentPath;
        closePicker();
    });
}

// ---------------------------------------------------------------------------
// Folder-Picker (IMAP / Exchange)
// ---------------------------------------------------------------------------

async function _loadFolderTree(fieldName) {
    const btn = document.getElementById(_loadFoldersBtnId(fieldName));
    const treeDiv = document.getElementById(_folderTreeId(fieldName));
    if (!btn || !treeDiv) return;

    btn.textContent = 'Lade...';
    btn.disabled = true;

    try {
        const formData = _getDsFormData();
        if (!formData) { btn.textContent = 'Ordner laden'; btn.disabled = false; return; }

        const resp = await SharedUi.fetch.safeFetch('/api/admin/datasources/list-folders', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify({
                type: formData.type,
                config: formData.connector_config,
                ds_id: _currentDsId,
            }),
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            SharedUi.notify.show(_dsResolveError(errData, `Fehler: HTTP ${resp.status}`), 'error');
            btn.textContent = 'Ordner laden';
            btn.disabled = false;
            return;
        }

        const result = await resp.json();

        const input = document.getElementById(_connFieldId(fieldName));
        const currentValue = (input?.value || '').trim();
        const isAll = currentValue === '*';
        const selected = isAll
            ? []
            : currentValue.split(',').map(s => s.trim()).filter(Boolean);

        const folders = result.folders || [];
        if (folders.length === 0) {
            treeDiv.innerHTML = '<div class="u-pad-8 u-text-sm-secondary">Keine Ordner gefunden</div>';
        } else {
            treeDiv.innerHTML = _renderFolderTree(folders, selected, isAll, 0);
            _applyFolderTreeIndent(treeDiv);
        }
        treeDiv.classList.remove('u-hidden');
        treeDiv.classList.add('u-block');

        treeDiv.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', () => _syncFolderTreeToInput(fieldName));
        });

    } catch (e) {
        SharedUi.notify.show(
            _dsResolveError(e.body || e, e.message || 'Verbindung fehlgeschlagen'),
            'error',
        );
    }
    btn.textContent = 'Ordner laden';
    btn.disabled = false;
}

function _renderFolderTree(folders, selected, allChecked, depth) {
    let html = '';
    for (const folder of folders) {
        const indent = depth * 20;
        const rawPath = folder.raw_path || folder.path;
        const isSelected = allChecked || selected.includes(rawPath);
        const checkedAttr = isSelected ? ' checked' : '';
        const selectable = folder.selectable !== false;
        const disabledAttr = (!selectable || allChecked) ? ' disabled' : '';
        const dimCls = selectable ? '' : ' u-dim';

        html += `<div class="u-ft-row${dimCls}" data-ft-indent="${indent}">`;
        html += `<input type="checkbox" data-path="${SharedUi.runtime.escapeAttr(rawPath)}"
                  data-selectable="${selectable}"
                  class="u-size-14"${checkedAttr}${disabledAttr}>`;
        html += `<span class="u-fs-13">${SharedUi.runtime.escapeHtml(folder.name)}</span>`;
        html += `</div>`;

        if (folder.children && folder.children.length > 0) {
            html += _renderFolderTree(folder.children, selected, allChecked, depth + 1);
        }
    }
    return html;
}

/** Wendet dynamische padding-left-Werte auf Folder-Tree-Zeilen an (ersetzt inline padding-left:Npx) */
function _applyFolderTreeIndent(container) {
    if (!container) return;
    container.querySelectorAll('.u-ft-row[data-ft-indent]').forEach(row => {
        const indent = parseInt(row.dataset.ftIndent, 10) || 0;
        SharedUi.runtimeStyle.applyStyle(row, { paddingLeft: indent + 'px' });
    });
}

function _toggleAllFolders(fieldName) {
    const chk = document.getElementById(_allFoldersCheckboxId(fieldName));
    const input = document.getElementById(_connFieldId(fieldName));
    const treeDiv = document.getElementById(_folderTreeId(fieldName));

    if (chk.checked) {
        input.value = '*';
        input.readOnly = true;
        if (treeDiv) {
            treeDiv.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = true;
                cb.disabled = true;
            });
        }
    } else {
        input.readOnly = false;
        if (treeDiv) {
            treeDiv.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                const selectable = cb.dataset.selectable === 'true';
                cb.disabled = !selectable;
                if (!selectable) cb.checked = false;
            });
        }
        _syncFolderTreeToInput(fieldName);
    }
}

function _syncFolderTreeToInput(fieldName) {
    const treeDiv = document.getElementById(_folderTreeId(fieldName));
    const input = document.getElementById(_connFieldId(fieldName));
    if (!treeDiv || !input || input.readOnly) return;

    const checked = treeDiv.querySelectorAll(
        'input[type="checkbox"]:checked:not([disabled])'
    );
    const paths = Array.from(checked)
        .map(cb => cb.dataset.path)
        .filter(Boolean);
    input.value = paths.join(', ');
}

// ---------------------------------------------------------------------------
// Schedule-API Anbindung
// ---------------------------------------------------------------------------

// Reverse-Lookup: cron_expr -> Preset-Key
const _CRON_TO_PRESET = {
    '*/30 * * * *': '30min',
    '0 * * * *': '1h',
    '0 */2 * * *': '2h',
    '0 */6 * * *': '6h',
    '0 */12 * * *': '12h',
    '0 3 * * *': '24h',
};

/**
 * Laedt den Schedule fuer eine bestehende DS und patcht die UI-Felder.
 * Guards: Open-Token + Dirty-Field + DOM-Existenz.
 */
async function _loadScheduleForEdit(dsId, ds, openToken, isDirtyFn) {
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/schedule`,
            {headers: {'X-Requested-With': 'XMLHttpRequest'}});
        // Guard: Modal noch aktuell?
        if (openToken !== _editOpenToken) return;

        const schedule = await resp.json();
        // Guard: nochmal pruefen nach await
        if (openToken !== _editOpenToken) return;
        // Guard: Dirty-Fields nicht ueberschreiben
        if (isDirtyFn && isDirtyFn()) return;

        const syncSel = document.getElementById('dsSyncInterval');
        const cronInput = document.getElementById('dsCronExpr');
        const cronField = document.getElementById('dsCronCustomField');
        if (!syncSel) return;

        const _showScheduleLoadWarning = () => {
            const syncWarn = document.getElementById('dsSyncReviewWarning');
            if (syncWarn) return;
            if (!cronField) return;
            const warnDiv = _createDsWarning('Zeitplan konnte nicht geladen werden. Angezeigte Werte basieren auf gespeicherten Fallback-Daten.', 'warn', 'dsSyncReviewWarning');
            cronField.parentElement.insertBefore(warnDiv, cronField);
        };

        const _setCronFieldVisible = (visible) => {
            if (!cronField) return;
            if (visible) {
                cronField.classList.remove('u-hidden');
                cronField.classList.add('u-block');
            } else {
                cronField.classList.add('u-hidden');
                cronField.classList.remove('u-block');
            }
        };

        if (schedule.active && schedule.cron_expr) {
            // Reverse-Mapping: cron_expr -> Preset oder custom
            const preset = _CRON_TO_PRESET[schedule.cron_expr];
            if (preset) {
                syncSel.value = preset;
                _setCronFieldVisible(false);
            } else {
                syncSel.value = 'custom';
                if (cronInput) cronInput.value = schedule.cron_expr;
                _setCronFieldVisible(true);
            }
        } else if (!schedule.active) {
            // Kein aktiver Schedule — sync_interval aus DS bestimmt den Zustand
            const normDs = _normalizeSyncInterval(ds.sync_interval);
            if (normDs.value === 'global' || normDs.value === 'manual') {
                syncSel.value = normDs.value;
                _setCronFieldVisible(false);
            }
            // Fuer custom ohne aktiven Schedule: Review-Zustand
            if (normDs.value === 'custom') {
                syncSel.value = 'custom';
                _setCronFieldVisible(true);
                if (cronInput && !cronInput.value.trim()) {
                    _markScheduleReviewState();
                }
            }
        }
    } catch (e) {
        console.error('Schedule-Read fehlgeschlagen:', e);
        // Guard: Modal noch aktuell? (await in safeFetch koennte zwischenzeitlich geschlossen haben)
        if (openToken !== _editOpenToken) return;
        // safeFetch wirft FetchError bei Nicht-2xx — Warning + Fallback-Werte behalten
        _showScheduleLoadWarning();
        // Review-Zustand: custom ohne autoritativen cron_expr
        const normDs = _normalizeSyncInterval(ds.sync_interval);
        if (normDs.value === 'custom') {
            const cronInput = document.getElementById('dsCronExpr');
            if (cronInput && !cronInput.value.trim()) {
                // Kein autoritativer cron_expr — Review blockiert Save
                _markScheduleReviewState();
            }
        }
    }
}

/**
 * Markiert den Schedule als Review-pflichtig: leerer Cron, Warning, Save blockiert.
 */
function _markScheduleReviewState() {
    const cronInput = document.getElementById('dsCronExpr');
    if (cronInput) {
        cronInput.value = '';
        cronInput.placeholder = 'Cron-Ausdruck eingeben oder Intervall wechseln';
    }
    const existing = document.getElementById('dsSyncReviewWarning');
    if (!existing) {
        const cronField = document.getElementById('dsCronCustomField');
        if (cronField) {
            const warnDiv = _createDsWarning('Kein gespeicherter Cron-Ausdruck vorhanden. Bitte eingeben oder ein anderes Intervall waehlen.', 'warn', 'dsSyncReviewWarning');
            cronField.parentElement.insertBefore(warnDiv, cronField.nextSibling);
        }
    }
}

/**
 * Speichert den Schedule fuer eine Datenquelle.
 * Wirft bei HTTP-Fehlern — Caller muss Partial-Success behandeln.
 */
async function _saveDsSchedule(dsId, interval, customCronExpr, options = {}) {
    const norm = _normalizeSyncInterval(interval);
    const iv = norm.value;

    if (iv === 'global' || iv === 'manual') {
        if (options.deleteScheduleless === false) return;
        // Scheduleless: bestehenden Schedule entfernen
        await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/schedule`, {
            method: 'DELETE',
            headers: {'X-Requested-With': 'XMLHttpRequest'},
        });
        return;
    }

    if (iv === 'custom') {
        const cron = (customCronExpr || '').trim();
        if (!cron) throw new Error('Benutzerdefinierter Cron-Ausdruck darf nicht leer sein');
        await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/schedule`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest'},
            body: JSON.stringify({cron_expr: cron}),
        });
        return;
    }

    // Preset-basiert (30min, 1h, 2h, 6h, 12h, 24h)
    await SharedUi.fetch.safeFetch(`/api/admin/datasources/${dsId}/schedule`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest'},
        body: JSON.stringify({cron_expr: iv, preset: iv}),
    });
}
