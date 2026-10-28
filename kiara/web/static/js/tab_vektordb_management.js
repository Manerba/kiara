/**
 * KIara Admin - VektorDB > DB-Management Sub-Tab
 * Matrix (DS x Modell x Chunks), Datei-Ansicht, selektive Loeschung.
 */

let _vdbMgmt = {
    datasources: [],
    models: [],
    detailDsId: null,
    detailModel: null,
    detailDsName: null,
    files: [],
    selected: new Set(),
    sortCol: 'source',
    sortAsc: true,
    totalChunks: 0,
    totalFiles: 0,
    pageChunks: 0,
    page: 1,
    pageSize: 50,
    totalItems: 0,
    totalPages: 0,
    _searchTerm: '',
    _loadSeq: 0,
    _initSeq: 0,
    storage: null,
};

const _VDBM_ACTIONS = new Set([
    'vdbmgmt-delete-collection',
    'vdbmgmt-open-detail',
    'vdbmgmt-select-all',
    'vdbmgmt-delete-selected',
    'vdbmgmt-close-detail',
    'vdbmgmt-sort',
    'vdbmgmt-page',
]);

const _VDBM_ROW_ACTIONS = new Set(['vdbmgmt-open-detail']);

const _VDBM_CHANGE_ACTIONS = new Set([
    'vdbmgmt-collection-select',
    'vdbmgmt-select-all-checkbox',
    'vdbmgmt-toggle-select',
    'vdbmgmt-page-size',
]);

const _VDBM_INPUT_ACTIONS = new Set(['vdbmgmt-search']);

const _vdbMgmt_onSearchDebounced = debounce((value) => _vdbMgmt_onSearch(value), 300);

function _handleVdbmClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_VDBM_ACTIONS.has(el.dataset.action)) return;
    if (_VDBM_ROW_ACTIONS.has(el.dataset.action)) e.stopPropagation();
    switch (el.dataset.action) {
        case 'vdbmgmt-delete-collection': _vdbMgmt_deleteCollection(); break;
        case 'vdbmgmt-open-detail':
            _vdbMgmt_openDetail(
                Number(el.dataset.dsId),
                el.dataset.model || '',
                el.dataset.dsName || ''
            );
            break;
        case 'vdbmgmt-select-all':      _vdbMgmt_selectAll(); break;
        case 'vdbmgmt-delete-selected': _vdbMgmt_deleteSelected(); break;
        case 'vdbmgmt-close-detail':    _vdbMgmt_closeDetail(); break;
        case 'vdbmgmt-sort':            _vdbMgmt_sort(el.dataset.sortCol || ''); break;
        case 'vdbmgmt-page':            _vdbMgmt_goToPage(Number(el.dataset.page)); break;
    }
}

function _handleVdbmChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_VDBM_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    switch (el.dataset.changeAction) {
        case 'vdbmgmt-collection-select':    _vdbMgmt_onCollectionSelect(); break;
        case 'vdbmgmt-select-all-checkbox':  _vdbMgmt_selectAll(el.checked); break;
        case 'vdbmgmt-toggle-select':        _vdbMgmt_toggleSelect(el); break;
        case 'vdbmgmt-page-size':            _vdbMgmt_setPageSize(Number(el.value)); break;
    }
}

function _handleVdbmInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el || !_VDBM_INPUT_ACTIONS.has(el.dataset.inputAction)) return;
    if (el.dataset.inputAction === 'vdbmgmt-search') {
        // Eine laufende Antwort fuer einen aelteren Suchtext darf das gerade
        // bearbeitete Eingabefeld nicht wieder ueberschreiben.
        _vdbMgmt._loadSeq += 1;
        _vdbMgmt_onSearchDebounced(el.value);
    }
}

function _bindVdbmActions() {
    if (document.body._vdbmActionsBound) return;
    document.body._vdbmActionsBound = true;
    document.body.addEventListener('click', _handleVdbmClick);
    document.body.addEventListener('change', _handleVdbmChange);
    document.body.addEventListener('input', _handleVdbmInput);
}

async function initVektordbManagementTab() {
    _bindVdbmActions();
    _runTabCleanups('vektordb-management');
    const mySeq = ++_vdbMgmt._initSeq;
    _vdbMgmt.storage = null;
    registerTabCleanup('vektordb-management', () => {
        _vdbMgmt._initSeq++;
        kiaraPolling.stop('vektordb-management:storage');
    });
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/datasources');
        if (!resp.ok) throw new Error('Fehler beim Laden');
        const data = await resp.json();
        if (mySeq !== _vdbMgmt._initSeq) return;
        _vdbMgmt.datasources = data.data || data.datasources || data || [];
        _vdbMgmt_extractModels();
        _vdbMgmt_render(container);
        await _vdbMgmt_refreshStorage(mySeq);
        if (mySeq !== _vdbMgmt._initSeq) return;
        kiaraPolling.start('vektordb-management:storage', () => _vdbMgmt_refreshStorage(mySeq), 10000);
    } catch (e) {
        if (mySeq !== _vdbMgmt._initSeq) return;
        container.innerHTML = '<div class="no-data">Daten konnten nicht geladen werden.</div>';
    }
}

async function _vdbMgmt_refreshStorage(seq) {
    let data;
    try {
        const response = await SharedUi.fetch.safeFetch('/api/monitoring/storage');
        if (!response.ok) throw new Error('Speichermessung nicht verfügbar');
        data = await response.json();
    } catch (e) {
        data = { qdrant: { status: 'unavailable', size_bytes: null } };
    }
    if (seq !== _vdbMgmt._initSeq) return;
    _vdbMgmt.storage = data;
    KiaraStorage.update(document.getElementById('vdbMgmtStorage'), data, KiaraStorage.collections);
}

function _vdbMgmt_extractModels() {
    const modelSet = new Set();
    for (const ds of _vdbMgmt.datasources) {
        for (const emb of (ds.embeddings || [])) {
            if (emb.embedding_model) modelSet.add(emb.embedding_model);
        }
    }
    _vdbMgmt.models = [...modelSet].sort();
}

function _vdbMgmt_modelStats(model) {
    let chunks = 0;
    let datasources = 0;
    let collection = '';
    for (const d of _vdbMgmt.datasources) {
        let dsHasModel = false;
        for (const emb of (d.embeddings || [])) {
            if (emb.embedding_model === model) {
                dsHasModel = true;
                chunks += (emb.chunk_count || 0);
                if (!collection && emb.collection) collection = emb.collection;
            }
        }
        if (dsHasModel) {
            datasources += 1;
        }
    }
    return { chunks, datasources, collection: collection || model };
}

function _vdbMgmt_renderCollectionDelete() {
    if (!_vdbMgmt.models.length) return '';

    let html = '<div class="vdbmgmt-section">';
    html += '<h3>Collection loeschen</h3>';
    html += '<div class="u-flex-center-gap-8 u-flex-wrap">';
    html += '<select id="vdbMgmtCollectionSelect" data-change-action="vdbmgmt-collection-select" '
        + 'class="vdbmgmt-select">';
    html += '<option value="">Embedding-Modell waehlen...</option>';
    for (const model of _vdbMgmt.models) {
        const stats = _vdbMgmt_modelStats(model);
        html += `<option value="${SharedUi.runtime.escapeAttr(model)}">${SharedUi.runtime.escapeHtml(model)} `
            + `(${stats.chunks.toLocaleString('de-DE')} Chunks)</option>`;
    }
    html += '</select>';
    html += '<button id="vdbMgmtDeleteCollectionBtn" class="action-btn danger" '
        + 'data-action="vdbmgmt-delete-collection" disabled>Collection loeschen</button>';
    html += '</div></div>';
    return html;
}

function _vdbMgmt_onCollectionSelect() {
    const select = document.getElementById('vdbMgmtCollectionSelect');
    const btn = document.getElementById('vdbMgmtDeleteCollectionBtn');
    if (btn) btn.disabled = !select || !select.value;
}

async function _vdbMgmt_deleteCollection() {
    const select = document.getElementById('vdbMgmtCollectionSelect');
    const model = select ? select.value : '';
    if (!model) return;

    const stats = _vdbMgmt_modelStats(model);
    const collection = stats.collection;
    const confirmed = await SharedUi.modal.confirm({ message:
        `Collection "${collection}" (Modell: ${model}) mit `
        + `${stats.chunks.toLocaleString('de-DE')} Chunks aus `
        + `${stats.datasources.toLocaleString('de-DE')} Datenquellen vollstaendig loeschen? `
        + 'Alle Dense-Embeddings dieses Modells werden unwiderruflich entfernt. '
        + 'Ein Re-Index ist erforderlich um die Daten wiederherzustellen.'
    });
    if (!confirmed) return;

    const btn = document.getElementById('vdbMgmtDeleteCollectionBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Loeschen...'; }

    try {
        const resp = await SharedUi.fetch.safeFetch(
            `/api/admin/vectordb/collections/${encodeURIComponent(model)}`,
            { method: 'DELETE', headers: { 'X-Requested-With': 'XMLHttpRequest' } }
        );
        const result = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            throw new Error(result.error || result.message || result.detail || `HTTP ${resp.status}`);
        }
        const successMsg = result.message || `Collection ${result.collection_name || collection} geloescht`;
        await SharedUi.modal.alert({ message:
            `${successMsg}. `
            + `${(result.affected_datasources ?? stats.datasources).toLocaleString('de-DE')} `
            + 'Datenquellen betroffen.'
        });

        if (_vdbMgmt.detailModel === model) {
            _vdbMgmt_closeDetail();
        }
        await _vdbMgmt_refreshAll();
    } catch (e) {
        await SharedUi.modal.alert({ message: 'Fehler: ' + e.message });
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Collection loeschen'; }
    }
}

function _vdbMgmt_render(container) {
    if (!container) container = document.getElementById('contentContainer');
    let html = '<div id="vdbMgmtRoot">';
    html += '<h3>Speicherbedarf der Collections</h3><div id="vdbMgmtStorage">'
        + KiaraStorage.collections(_vdbMgmt.storage) + '</div>';
    html += _vdbMgmt_renderMatrix();
    html += _vdbMgmt_renderCollectionDelete();
    html += '<div id="vdbMgmtDetail"></div>';
    html += '</div>';
    container.innerHTML = html;
}

function _vdbMgmt_renderMatrix() {
    const ds = _vdbMgmt.datasources;
    const models = _vdbMgmt.models;

    if (!ds.length || !models.length) {
        return '<div class="no-data">Keine Datenquellen mit Embeddings vorhanden.</div>';
    }

    let html = '<h3>Datenquellen-Matrix</h3>';
    html += '<div class="table-responsive"><table class="data-table"><thead><tr>';
    html += '<th>Datenquelle</th>';
    for (const m of models) {
        html += `<th class="u-text-center">${SharedUi.runtime.escapeHtml(m)}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const d of ds) {
        html += '<tr>';
        html += `<td>${SharedUi.runtime.escapeHtml(d.name || '(unbenannt)')}</td>`;
        const embMap = {};
        for (const emb of (d.embeddings || [])) {
            embMap[emb.embedding_model] = emb;
        }
        for (const m of models) {
            const emb = embMap[m];
            const cnt = emb ? (emb.chunk_count || 0) : 0;
            if (cnt > 0) {
                html += `<td class="u-text-center u-clickable cell-clickable"
                    data-ds-id="${d.id}" data-model="${SharedUi.runtime.escapeAttr(m)}" data-ds-name="${SharedUi.runtime.escapeAttr(d.name || '')}"
                    data-action="vdbmgmt-open-detail"
                    title="Dateien anzeigen">${cnt.toLocaleString('de-DE')}</td>`;
            } else {
                html += `<td class="u-text-center vdbmgmt-cell-muted">0</td>`;
            }
        }
        html += '</tr>';
    }

    // Summenzeile
    html += '<tr class="vdbmgmt-sum-row">';
    html += '<td>Gesamt</td>';
    for (const m of models) {
        let colSum = 0;
        for (const d of ds) {
            for (const emb of (d.embeddings || [])) {
                if (emb.embedding_model === m) colSum += (emb.chunk_count || 0);
            }
        }
        html += `<td class="u-text-center">${colSum.toLocaleString('de-DE')}</td>`;
    }
    html += '</tr>';

    html += '</tbody></table></div>';
    return html;
}

async function _vdbMgmt_openDetail(dsId, model, dsName) {
    _vdbMgmt.detailDsId = dsId;
    _vdbMgmt.detailModel = model;
    _vdbMgmt.detailDsName = dsName;
    _vdbMgmt.selected.clear();
    _vdbMgmt._searchTerm = '';
    _vdbMgmt.sortCol = 'source';
    _vdbMgmt.sortAsc = true;
    _vdbMgmt.page = 1;
    _vdbMgmt.totalItems = 0;
    _vdbMgmt.totalPages = 0;

    const detail = document.getElementById('vdbMgmtDetail');
    if (!detail) return;
    detail.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    await _vdbMgmt_loadFiles();
}

async function _vdbMgmt_loadFiles(options = {}) {
    const dsId = _vdbMgmt.detailDsId;
    const model = _vdbMgmt.detailModel;
    if (dsId === null || !model) return;

    const detail = document.getElementById('vdbMgmtDetail');
    if (!detail) return;
    if (options.showLoading) {
        detail.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    }

    const loadSeq = ++_vdbMgmt._loadSeq;
    const params = new URLSearchParams({
        model,
        page: String(_vdbMgmt.page),
        page_size: String(_vdbMgmt.pageSize),
        direction: _vdbMgmt.sortAsc ? 'asc' : 'desc',
    });
    if (_vdbMgmt._searchTerm) params.set('search', _vdbMgmt._searchTerm);

    try {
        const resp = await SharedUi.fetch.safeFetch(
            `/api/admin/vectordb/datasources/${dsId}/files?${params.toString()}`
        );
        const data = await resp.json();
        if (loadSeq !== _vdbMgmt._loadSeq
            || _vdbMgmt.detailDsId !== dsId
            || _vdbMgmt.detailModel !== model) return;

        _vdbMgmt.files = data.files || [];
        _vdbMgmt.totalChunks = Number(data.total_chunks || 0);
        _vdbMgmt.totalFiles = Number(data.total_files || 0);
        _vdbMgmt.pageChunks = Number(data.page_chunks || 0);
        const pagination = data.pagination || {};
        _vdbMgmt.page = Number(pagination.page || 1);
        _vdbMgmt.pageSize = Number(pagination.page_size || _vdbMgmt.pageSize);
        _vdbMgmt.totalItems = Number(pagination.total_items || 0);
        _vdbMgmt.totalPages = Number(pagination.total_pages || 0);
        _vdbMgmt_renderDetail();
    } catch (e) {
        if (loadSeq !== _vdbMgmt._loadSeq
            || _vdbMgmt.detailDsId !== dsId
            || _vdbMgmt.detailModel !== model) return;
        const reason = e && e.message ? ` (${SharedUi.runtime.escapeHtml(e.message)})` : '';
        detail.innerHTML = `<div class="no-data">Dateiliste konnte nicht geladen werden.${reason}</div>`;
    }
}

function _vdbMgmt_renderDetail() {
    const detail = document.getElementById('vdbMgmtDetail');
    if (!detail) return;

    const files = _vdbMgmt.files;
    const selCount = _vdbMgmt.selected.size;
    const allPageSelected = files.length > 0
        && files.every(f => _vdbMgmt.selected.has(f.source));

    let html = '<div class="vdbmgmt-section">';
    const summary = _vdbMgmt._searchTerm
        ? `${_vdbMgmt.totalItems.toLocaleString('de-DE')} von ${_vdbMgmt.totalFiles.toLocaleString('de-DE')} Dateien gefunden; `
            + `${_vdbMgmt.totalChunks.toLocaleString('de-DE')} Chunks insgesamt`
        : `${_vdbMgmt.totalChunks.toLocaleString('de-DE')} Chunks in `
            + `${_vdbMgmt.totalFiles.toLocaleString('de-DE')} Dateien`;
    html += `<h3>${SharedUi.runtime.escapeHtml(_vdbMgmt.detailDsName)} &mdash; ${SharedUi.runtime.escapeHtml(_vdbMgmt.detailModel)}`
         + ` &mdash; ${summary}</h3>`;

    // Toolbar
    html += '<div class="vdbmgmt-toolbar">';
    html += `<input type="text" id="vdbMgmtSearch" placeholder="Dateipfad suchen..." maxlength="500"
        data-input-action="vdbmgmt-search" class="vdbmgmt-search-input"
        value="${SharedUi.runtime.escapeAttr(_vdbMgmt._searchTerm || '')}">`;
    html += `<button class="action-btn" data-action="vdbmgmt-select-all"
        title="Alle Dateien dieser Seite auswaehlen/abwaehlen">Seite auswaehlen</button>`;
    html += `<button class="action-btn danger" data-action="vdbmgmt-delete-selected"
        ${selCount === 0 ? 'disabled' : ''}>Ausgewaehlte loeschen (${selCount})</button>`;
    html += `<button class="action-btn" data-action="vdbmgmt-close-detail">Schliessen</button>`;
    html += '</div>';

    // Table
    html += '<div class="table-responsive"><table class="data-table"><thead><tr>';
    html += '<th class="vdbmgmt-cb-col"><input type="checkbox" data-change-action="vdbmgmt-select-all-checkbox"'
         + (allPageSelected ? ' checked' : '') + '></th>';
    html += _vdbMgmt_sortHeader('source', 'Dateipfad');
    html += '<th>Typ</th>';
    html += '<th>Chunks</th>';
    html += '<th>Geaendert</th>';
    html += '</tr></thead><tbody>';

    if (files.length === 0) {
        html += '<tr><td colspan="5" class="no-data">Keine Dateien gefunden.</td></tr>';
    }

    for (const f of files) {
        const checked = _vdbMgmt.selected.has(f.source) ? ' checked' : '';
        const basename = f.source.split('/').pop() || f.source;
        const dir = f.source.substring(0, f.source.length - basename.length);
        const mod = f.modified ? _vdbMgmt_formatDate(f.modified) : '';
        html += `<tr>`;
        html += `<td><input type="checkbox" value="${SharedUi.runtime.escapeAttr(f.source)}"
            data-change-action="vdbmgmt-toggle-select"${checked}></td>`;
        html += `<td title="${SharedUi.runtime.escapeAttr(f.source)}"><span class="vdbmgmt-cell-muted">${SharedUi.runtime.escapeHtml(dir)}</span>${SharedUi.runtime.escapeHtml(basename)}</td>`;
        html += `<td>${SharedUi.runtime.escapeHtml(f.file_type || '')}</td>`;
        html += `<td class="u-text-right">${(f.chunk_count || 0).toLocaleString('de-DE')}</td>`;
        html += `<td>${SharedUi.runtime.escapeHtml(mod)}</td>`;
        html += '</tr>';
    }

    html += '</tbody></table></div>';
    html += _vdbMgmt_renderPagination();
    html += '</div>';

    const active = document.activeElement;
    const wasSearchFocused = !!(active && active.id === 'vdbMgmtSearch');
    const selStart = wasSearchFocused ? active.selectionStart : 0;
    const selEnd = wasSearchFocused ? active.selectionEnd : 0;

    detail.innerHTML = html;

    if (wasSearchFocused) {
        const newInput = document.getElementById('vdbMgmtSearch');
        if (newInput) {
            newInput.focus();
            try { newInput.setSelectionRange(selStart, selEnd); } catch (e) {}
        }
    }
}

function _vdbMgmt_renderPagination() {
    const current = _vdbMgmt.page;
    const total = _vdbMgmt.totalPages;
    if (total <= 1 && _vdbMgmt.totalItems <= 25) return '';

    const start = Math.max(1, Math.min(current - 2, Math.max(1, total - 4)));
    const end = Math.min(total, start + 4);
    let html = '<div class="pagination-wrapper"><div class="pagination-container">';
    html += '<div class="pagination-nav">';
    html += `<button type="button" class="pagination-btn${current <= 1 ? ' disabled' : ''}" `
        + `data-action="vdbmgmt-page" data-page="1" aria-label="Erste Seite"${current <= 1 ? ' disabled' : ''}>&laquo;</button>`;
    html += `<button type="button" class="pagination-btn${current <= 1 ? ' disabled' : ''}" `
        + `data-action="vdbmgmt-page" data-page="${current - 1}" aria-label="Vorherige Seite"${current <= 1 ? ' disabled' : ''}>&lt;</button>`;
    html += '<div class="pagination-pages">';
    if (start > 1) html += '<span class="pagination-ellipsis">...</span>';
    for (let page = start; page <= end; page += 1) {
        const active = page === current;
        html += `<button type="button" class="pagination-btn${active ? ' active' : ''}" `
            + `data-action="vdbmgmt-page" data-page="${page}"${active ? ' aria-current="page"' : ''}>${page}</button>`;
    }
    if (end < total) html += '<span class="pagination-ellipsis">...</span>';
    html += '</div>';
    html += `<button type="button" class="pagination-btn${current >= total ? ' disabled' : ''}" `
        + `data-action="vdbmgmt-page" data-page="${current + 1}" aria-label="Naechste Seite"${current >= total ? ' disabled' : ''}>&gt;</button>`;
    html += `<button type="button" class="pagination-btn${current >= total ? ' disabled' : ''}" `
        + `data-action="vdbmgmt-page" data-page="${total}" aria-label="Letzte Seite"${current >= total ? ' disabled' : ''}>&raquo;</button>`;
    html += '</div>';
    html += `<div class="pagination-info">Seite ${current} / ${Math.max(total, 1)} &middot; `
        + `${_vdbMgmt.totalItems.toLocaleString('de-DE')} Dateien</div></div>`;
    html += '<div class="pagination-page-size"><label for="vdbMgmtPageSize">Pro Seite</label>';
    html += '<select id="vdbMgmtPageSize" data-change-action="vdbmgmt-page-size" aria-label="Dateien pro Seite">';
    for (const size of [25, 50, 100, 200]) {
        html += `<option value="${size}"${size === _vdbMgmt.pageSize ? ' selected' : ''}>${size}</option>`;
    }
    html += '</select></div></div>';
    return html;
}

function _vdbMgmt_sortHeader(col, label) {
    const arrow = _vdbMgmt.sortCol === col ? (_vdbMgmt.sortAsc ? ' ▲' : ' ▼') : '';
    return `<th class="u-clickable" data-action="vdbmgmt-sort" data-sort-col="${SharedUi.runtime.escapeAttr(col)}">${SharedUi.runtime.escapeHtml(label)}${arrow}</th>`;
}

function _vdbMgmt_sort(col) {
    if (col !== 'source') return;
    if (_vdbMgmt.sortCol === col) {
        _vdbMgmt.sortAsc = !_vdbMgmt.sortAsc;
    } else {
        _vdbMgmt.sortCol = col;
        _vdbMgmt.sortAsc = true;
    }
    _vdbMgmt.page = 1;
    _vdbMgmt_loadFiles();
}

function _vdbMgmt_onSearch(term) {
    _vdbMgmt._searchTerm = String(term || '').trim();
    _vdbMgmt.page = 1;
    _vdbMgmt.selected.clear();
    _vdbMgmt_loadFiles();
}

function _vdbMgmt_goToPage(page) {
    if (!Number.isInteger(page) || page < 1 || page > _vdbMgmt.totalPages
        || page === _vdbMgmt.page) return;
    _vdbMgmt.page = page;
    _vdbMgmt_loadFiles();
}

function _vdbMgmt_setPageSize(pageSize) {
    if (![25, 50, 100, 200].includes(pageSize) || pageSize === _vdbMgmt.pageSize) return;
    _vdbMgmt.pageSize = pageSize;
    _vdbMgmt.page = 1;
    _vdbMgmt_loadFiles();
}

function _vdbMgmt_toggleSelect(checkbox) {
    if (checkbox.checked) {
        _vdbMgmt.selected.add(checkbox.value);
    } else {
        _vdbMgmt.selected.delete(checkbox.value);
    }
    _vdbMgmt_renderDetail();
}

function _vdbMgmt_selectAll(forceState) {
    const files = _vdbMgmt.files;
    const allPageSelected = files.length > 0
        && files.every(f => _vdbMgmt.selected.has(f.source));
    if (forceState === false || (forceState === undefined && allPageSelected)) {
        for (const f of files) _vdbMgmt.selected.delete(f.source);
    } else {
        for (const f of files) _vdbMgmt.selected.add(f.source);
    }
    _vdbMgmt_renderDetail();
}

async function _vdbMgmt_deleteSelected() {
    const paths = [..._vdbMgmt.selected];
    if (paths.length === 0) return;

    const confirmed = await SharedUi.modal.confirm({ message:
        `${paths.length} Datei(en) aus der VektorDB loeschen? Dieser Vorgang kann nicht rueckgaengig gemacht werden.`
    });
    if (!confirmed) return;

    const detail = document.getElementById('vdbMgmtDetail');
    const btn = detail ? detail.querySelector('.action-btn.danger') : null;
    const originalBtnText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = 'Loeschen...'; }

    try {
        const resp = await SharedUi.fetch.safeFetch(
            `/api/admin/vectordb/datasources/${_vdbMgmt.detailDsId}/delete-files`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                },
                body: JSON.stringify({
                    embedding_model: _vdbMgmt.detailModel,
                    file_paths: paths,
                }),
            }
        );
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || `HTTP ${resp.status}`);
        }
        const result = await resp.json();
        await SharedUi.modal.alert({ message:
            `${result.deleted_files} Datei(en) geloescht, `
            + `${result.deleted_chunks} Chunks entfernt `
            + `(${result.orphan_chunks_deleted} Waisen, ${result.shared_chunks_updated} Shared aktualisiert).`
        });
        // Matrix + Detail aktualisieren
        _vdbMgmt.selected.clear();
        await _vdbMgmt_refreshAll();
    } catch (e) {
        await SharedUi.modal.alert({ message: 'Fehler: ' + e.message });
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = originalBtnText; }
    }
}

async function _vdbMgmt_refreshAll() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/datasources');
        if (resp.ok) {
            const data = await resp.json();
            _vdbMgmt.datasources = data.data || data.datasources || data || [];
            _vdbMgmt_extractModels();
        }
    } catch (e) { /* ignore */ }

    const container = document.getElementById('contentContainer');
    const root = document.getElementById('vdbMgmtRoot');
    if (root) {
        _vdbMgmt_render(container);
        // Detail neu laden wenn offen
        if (_vdbMgmt.detailDsId !== null) {
            await _vdbMgmt_loadFiles({ showLoading: true });
        }
    }
}

function _vdbMgmt_closeDetail() {
    _vdbMgmt.detailDsId = null;
    _vdbMgmt.detailModel = null;
    _vdbMgmt.files = [];
    _vdbMgmt.selected.clear();
    _vdbMgmt.totalChunks = 0;
    _vdbMgmt.totalFiles = 0;
    _vdbMgmt.pageChunks = 0;
    _vdbMgmt.page = 1;
    _vdbMgmt.totalItems = 0;
    _vdbMgmt.totalPages = 0;
    _vdbMgmt._loadSeq += 1;
    const detail = document.getElementById('vdbMgmtDetail');
    if (detail) detail.innerHTML = '';
}

function _vdbMgmt_formatDate(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return isoStr;
        return d.toLocaleDateString('de-DE') + ' ' + d.toLocaleTimeString('de-DE', {hour:'2-digit', minute:'2-digit'});
    } catch (e) {
        return isoStr;
    }
}
