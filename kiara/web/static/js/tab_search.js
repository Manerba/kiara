/**
 * KIara - Search Tab
 * Textsuche (Elasticsearch) + Semantische Suche (Qdrant)
 */

let searchTable = null;
const _searchSelectedChunks = new Map();  // chunk_id -> chunk_data
let _searchDsIds = new Set();
let _searchMode = 'text';
let _searchRequestSeq = 0;
let _lastSearchBody = null;

const _SEARCH_ACTIONS = new Set([
    'send-selected-to-chat',
    'send-selected-summary',
    'send-single-to-chat',
    'execute-search',
    'set-search-mode',
    'remove-search-ds',
]);

const _SEARCH_ROW_ACTIONS = new Set([
    'send-single-to-chat',
]);

const _SEARCH_CHANGE_ACTIONS = new Set([
    'toggle-search-select-all',
    'toggle-search-chunk',
]);

function _handleSearchClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_SEARCH_ACTIONS.has(el.dataset.action)) return;
    if (_SEARCH_ROW_ACTIONS.has(el.dataset.action)) e.stopPropagation();
    switch (el.dataset.action) {
        case 'send-selected-to-chat':  sendSelectedToChat(); break;
        case 'send-selected-summary':  sendSelectedSummary(); break;
        case 'send-single-to-chat':    sendSingleToChat(el.dataset.chunkId || ''); break;
        case 'execute-search':         executeSearch(); break;
        case 'set-search-mode':        setSearchMode(el.dataset.mode || 'text'); break;
        case 'remove-search-ds':       _removeSearchDs(el.dataset.dsId || ''); break;
    }
}

function _handleSearchChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_SEARCH_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    switch (el.dataset.changeAction) {
        case 'toggle-search-select-all':
            toggleSearchSelectAll(el.checked);
            break;
        case 'toggle-search-chunk':
            toggleSearchChunk(el.dataset.chunkId || '', el.checked);
            break;
    }
}

function _bindSearchActions() {
    if (document.body._searchActionsBound) return;
    document.body._searchActionsBound = true;
    document.body.addEventListener('click', _handleSearchClick);
    document.body.addEventListener('change', _handleSearchChange);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-search-actions-bound', 'true');
}

async function initSearchTab() {
    _bindSearchActions();
    const container = document.getElementById('contentContainer');
    if (!container) return;

    _searchSelectedChunks.clear();
    _searchDsIds.clear();

    container.innerHTML = '';

    // Filter-Bar innerhalb des Containers (Search hat eigene)
    const searchBar = document.createElement('div');
    searchBar.className = 'search-filter-bar';
    searchBar.innerHTML = _buildSearchFilterBar();
    container.appendChild(searchBar);

    // Aktions-Leiste (versteckt bis Selektion)
    const actionBar = document.createElement('div');
    actionBar.className = 'search-action-bar u-hidden';
    actionBar.id = 'searchActionBar';
    actionBar.innerHTML = `
        <span class="search-action-count" id="searchSelectedCount">0 ausgewaehlt</span>
        <button class="btn btn-primary btn-sm" data-action="send-selected-to-chat">Zum Chat senden &#x25B8;</button>
        <button class="btn btn-secondary btn-sm" data-action="send-selected-summary">KI-Zusammenfassung</button>
    `;
    container.appendChild(actionBar);

    // Tabellen-Container
    const tableDiv = document.createElement('div');
    tableDiv.id = 'searchResults';
    container.appendChild(tableDiv);

    // ExpandableTable konfigurieren
    const tableConfig = {
        id: 'searchTable',
        columns: [
            { field: '_select', label: '<input type="checkbox" id="searchSelectAll" data-change-action="toggle-search-select-all">', rawLabel: true, sortable: false, renderer: 'searchCheckbox' },
            { field: 'score', label: 'Score', sortable: true, renderer: 'searchScore' },
            { field: 'source', label: 'Dateiname', sortable: true },
            { field: 'directory', label: 'Verzeichnis', sortable: true, truncate: 30 },
            { field: 'file_type', label: 'Typ', sortable: true },
            { field: 'modified', label: 'Datum', sortable: true },
            { field: 'text', label: 'Auszug', sortable: false, truncate: 80 },
            { field: '_action', label: '', sortable: false, renderer: 'searchAction' },
        ],
        filters: [],
        pagination: { enabled: true, defaultPageSize: 10, options: [10, 50, 200, 1000] },
        defaultSort: { field: 'score', direction: 'desc' },
        expandable: true,
        showHeader: true,
        gridTemplate: '30px 30px 70px 200px 180px 60px 120px 1fr 40px',
        detailRenderer: function(row) {
            const source = SharedUi.runtime.escapeHtml(row._source_full || row.metadata?.source || '');
            const score = row._raw_score !== undefined ? (row._raw_score * 100).toFixed(1) + '%' : '';
            const text = SharedUi.runtime.escapeHtml(row._full_text || row.text || '');
            return `<div class="search-detail">
                <div class="search-detail-meta"><strong>Quelle:</strong> ${source} &nbsp; <strong>Score:</strong> ${score}</div>
                <pre class="search-detail-text">${text}</pre>
            </div>`;
        },
    };

    searchTable = new SharedUi.table.ExpandableTable(tableConfig, 'searchResults', {
        onPageChange: (page, pageSize) => {
            executeSearch({ page, page_size: pageSize, useSnapshot: true });
        },
    });

    // Custom Renderer registrieren
    searchTable.renderers['searchCheckbox'] = (val, col, row) => {
        const checked = _searchSelectedChunks.has(row.id) ? 'checked' : '';
        return `<input type="checkbox" ${checked} data-change-action="toggle-search-chunk" data-chunk-id="${SharedUi.runtime.escapeAttr(row.id)}">`;
    };
    searchTable.renderers['searchScore'] = (val) => {
        const pct = typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val || '';
        return `<span class="search-score">${pct}</span>`;
    };
    searchTable.renderers['searchAction'] = (val, col, row) => {
        return `<button class="btn-icon" data-action="send-single-to-chat" data-chunk-id="${SharedUi.runtime.escapeAttr(row.id)}" title="Zum Chat senden">&#x25B6;</button>`;
    };

    // Event-Listener
    _initSearchEvents();
    await _loadSearchFilters();
    _updateModeVisibility();
}

function _buildSearchFilterBar() {
    return `
        <div class="search-filter-row">
            <div class="filter-group search-input-group">
                <input type="text" id="searchQuery" class="filter-input search-query-input" placeholder="Suchbegriff eingeben..." autocomplete="off">
                <button class="btn btn-primary btn-sm" id="searchBtn" data-action="execute-search">Suchen</button>
            </div>
            <div class="filter-group">
                <span class="filter-label">Modus:</span>
                <div class="search-mode-toggle">
                    <button class="search-mode-btn active" id="searchModeText" data-action="set-search-mode" data-mode="text">Text</button>
                    <button class="search-mode-btn" id="searchModeSemantic" data-action="set-search-mode" data-mode="semantic">Semantisch</button>
                </div>
            </div>
        </div>
        <div class="search-filter-row">
            <div class="filter-group">
                <span class="filter-label">Datenquelle:</span>
                <select class="filter-select u-hidden" id="searchFilterDs" multiple></select>
                <div class="ds-chips-container" id="searchDsChips"></div>
                <select class="filter-select" id="searchFilterDsSelect"><option value="">Alle</option></select>
            </div>
            <div class="filter-group">
                <span class="filter-label">Dateityp:</span>
                <select class="filter-select" id="searchFilterType"><option value="">Alle</option></select>
            </div>
            <div class="filter-group">
                <span class="filter-label">Verzeichnis:</span>
                <input type="text" id="searchFilterDir" class="filter-input u-width-140" placeholder="Pfad...">
            </div>
            <div class="filter-group">
                <span class="filter-label">Datum:</span>
                <input type="date" id="searchFilterDateFrom" class="filter-input search-date-input">
                <span>-</span>
                <input type="date" id="searchFilterDateTo" class="filter-input search-date-input">
            </div>
            <div class="filter-group">
                <span class="filter-label">k:</span>
                <input type="number" id="searchFilterK" class="filter-input search-k-input" value="10" min="1" max="1000" title="Anzahl Ergebnisse pro Seite">
            </div>
            <div class="filter-group u-hidden" id="searchEmbeddingGroup">
                <span class="filter-label">Modell:</span>
                <select class="filter-select" id="searchFilterModel"></select>
            </div>
        </div>
    `;
}

function _initSearchEvents() {
    const queryInput = document.getElementById('searchQuery');
    if (queryInput) {
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); executeSearch(); }
        });
        queryInput.focus();
    }

    // DS-MultiSelect
    const dsSelect = document.getElementById('searchFilterDsSelect');
    if (dsSelect) {
        dsSelect.addEventListener('change', () => {
            const val = parseInt(dsSelect.value);
            if (val) {
                _searchDsIds.add(val);
                _renderSearchDsChips();
                dsSelect.value = '';
            }
        });
    }
}

async function _loadSearchFilters() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/chat/filters');
        if (!resp.ok) return;
        const data = await resp.json();

        // Datenquellen
        const dsSelect = document.getElementById('searchFilterDsSelect');
        if (dsSelect && data.datasources) {
            data.datasources.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.id;
                opt.textContent = ds.name;
                dsSelect.appendChild(opt);
            });
        }

        // Dateitypen
        const typeSelect = document.getElementById('searchFilterType');
        if (typeSelect && data.file_types) {
            data.file_types.forEach(ft => {
                const opt = document.createElement('option');
                opt.value = ft;
                opt.textContent = ft;
                typeSelect.appendChild(opt);
            });
        }

        // Embedding-Modelle
        const modelSelect = document.getElementById('searchFilterModel');
        if (modelSelect && data.embedding_models) {
            data.embedding_models.forEach(em => {
                const opt = document.createElement('option');
                opt.value = em.model;
                opt.textContent = em.model + (em.has_backend ? '' : ' (offline)');
                modelSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error('Fehler beim Laden der Filter:', err);
    }
}

function _renderSearchDsChips() {
    const container = document.getElementById('searchDsChips');
    const dsSelect = document.getElementById('searchFilterDsSelect');
    if (!container) return;

    container.innerHTML = '';
    _searchDsIds.forEach(id => {
        // Name aus Select holen
        const opt = dsSelect?.querySelector(`option[value="${id}"]`);
        const name = opt ? opt.textContent : `DS ${id}`;
        const chip = document.createElement('span');
        chip.className = 'ds-chip';
        chip.innerHTML = `${SharedUi.runtime.escapeHtml(name)} <span class="ds-chip-remove" data-action="remove-search-ds" data-ds-id="${SharedUi.runtime.escapeAttr(String(id))}">&#x2715;</span>`;
        container.appendChild(chip);
    });
}

function _removeSearchDs(id) {
    _searchDsIds.delete(parseInt(id));
    _renderSearchDsChips();
}

function setSearchMode(mode) {
    _searchMode = mode;
    document.getElementById('searchModeText')?.classList.toggle('active', mode === 'text');
    document.getElementById('searchModeSemantic')?.classList.toggle('active', mode === 'semantic');
    _updateModeVisibility();
}

function _updateModeVisibility() {
    const embGroup = document.getElementById('searchEmbeddingGroup');
    if (embGroup) embGroup.classList.toggle('u-hidden', _searchMode !== 'semantic');
}

async function executeSearch(overrides = {}) {
    const query = document.getElementById('searchQuery')?.value?.trim();
    const hasSnapshot = !!(overrides.useSnapshot && _lastSearchBody);
    if (!query && !hasSnapshot) {
        SharedUi.notify.show('Bitte einen Suchbegriff eingeben.', 'error');
        return;
    }

    const page = overrides.page || 1;
    const kVal = parseInt(document.getElementById('searchFilterK')?.value) || 10;
    const page_size = overrides.page_size || kVal;
    // Pagination-Groesse synchron halten
    if (searchTable && searchTable.pagination.pageSize !== page_size) {
        searchTable.pagination.pageSize = page_size;
    }

    const body = hasSnapshot
        ? { ..._lastSearchBody, page, page_size }
        : {
            query,
            mode: _searchMode,
            page,
            page_size,
            filter_datasource_ids: _searchDsIds.size > 0 ? Array.from(_searchDsIds) : null,
            filter_type: document.getElementById('searchFilterType')?.value || null,
            filter_dir: document.getElementById('searchFilterDir')?.value || null,
            filter_date_from: document.getElementById('searchFilterDateFrom')?.value || null,
            filter_date_to: document.getElementById('searchFilterDateTo')?.value || null,
        };

    if (!overrides.useSnapshot && _searchMode === 'semantic') {
        body.embedding_model = document.getElementById('searchFilterModel')?.value || null;
    }

    const thisSeq = ++_searchRequestSeq;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify(body),
        });

        if (thisSeq !== _searchRequestSeq) return;  // stale response

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            SharedUi.notify.show(err.detail || `Fehler ${resp.status}`, 'error');
            return;
        }

        const result = await resp.json();

        if (thisSeq !== _searchRequestSeq) return;  // stale response
        _lastSearchBody = { ...body };

        // Ergebnisse in Tabellen-Format konvertieren
        const rows = result.results.map(r => ({
            id: r.id,
            _select: '',
            score: r.score,
            source: _extractFilename(r.metadata?.source || ''),
            directory: r.metadata?.directory || '',
            file_type: r.metadata?.file_type || '',
            modified: _formatDate(r.metadata?.modified),
            text: (r.text || '').substring(0, 80),
            _action: '',
            // Vollstaendige Daten fuer Detail/Chat
            _full_text: r.text,
            _source_full: r.metadata?.source || '',
            _raw_score: r.score,
            metadata: r.metadata,
        }));

        // Pagination-State aktualisieren
        searchTable.pagination.currentPage = result.page;
        searchTable.setData(rows, result.total);

        // Info anzeigen
        const infoText = `${result.total} Treffer in ${result.query_time_ms}ms (${result.mode === 'text' ? 'Textsuche' : 'Semantische Suche'})`;
        _showSearchInfo(infoText);

    } catch (err) {
        console.error('Suche fehlgeschlagen:', err);
        SharedUi.notify.show('Suche fehlgeschlagen: ' + err.message, 'error');
    }
}

function _showSearchInfo(text) {
    let info = document.getElementById('searchInfo');
    if (!info) {
        info = document.createElement('div');
        info.id = 'searchInfo';
        info.className = 'search-info';
        const actionBar = document.getElementById('searchActionBar');
        if (actionBar) actionBar.parentNode.insertBefore(info, actionBar);
    }
    info.textContent = text;
}

function _extractFilename(path) {
    if (!path) return '';
    return path.split('/').pop() || path;
}

function _formatDate(dateStr) {
    if (!dateStr) return '';
    try {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;
        const pad = n => String(n).padStart(2, '0');
        return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`;
    } catch { return dateStr; }
}

// Checkbox-Logik
function toggleSearchChunk(id, checked) {
    if (checked) {
        // Chunk-Daten aus Tabelle holen
        const row = searchTable?.data?.find(r => r.id === id);
        if (row) _searchSelectedChunks.set(id, row);
    } else {
        _searchSelectedChunks.delete(id);
    }
    _updateSearchActionBar();
}

function toggleSearchSelectAll(checked) {
    if (!searchTable?.filteredData) return;
    searchTable.filteredData.forEach(row => {
        if (checked) _searchSelectedChunks.set(row.id, row);
        else _searchSelectedChunks.delete(row.id);
    });
    _updateSearchActionBar();
    searchTable.applyFiltersAndRender();
}

function _updateSearchActionBar() {
    const bar = document.getElementById('searchActionBar');
    const count = document.getElementById('searchSelectedCount');
    const n = _searchSelectedChunks.size;
    if (bar) bar.classList.toggle('u-hidden', n === 0);
    if (count) count.textContent = `${n} ausgewaehlt`;
}

// Chat-Uebergabe
function sendSelectedToChat() {
    if (_searchSelectedChunks.size === 0) return;
    const chunks = Array.from(_searchSelectedChunks.values()).map(row => ({
        text: row._full_text || row.text,
        source: row._source_full || row.metadata?.source || '',
        score: row._raw_score ?? row.score ?? 0,
        id: row.id,
    }));
    _sendToChat(chunks);
}

function sendSingleToChat(id) {
    const row = searchTable?.data?.find(r => r.id === id);
    if (!row) return;
    const chunks = [{
        text: row._full_text || row.text,
        source: row._source_full || row.metadata?.source || '',
        score: row._raw_score ?? row.score ?? 0,
        id: row.id,
    }];
    _sendToChat(chunks);
}

function sendSelectedSummary() {
    if (_searchSelectedChunks.size === 0) return;
    window._searchAutoPrompt = 'Fasse die folgenden Dokumente zusammen und hebe die wichtigsten Punkte hervor.';
    sendSelectedToChat();
}

function _sendToChat(chunks) {
    window._searchContext = chunks;
    invalidateTabCache('chat');
    navigateTo('chat');
}
