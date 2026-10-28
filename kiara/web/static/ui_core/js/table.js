/**
 * SharedUi.table — produktneutrale ExpandableTable (Core-Baustein 10).
 *
 * Abhaengigkeiten (Reihenfolge im Template):
 *   core.js -> fetch.js -> runtime.js -> table.js
 *
 * Public API:
 *   SharedUi.table.ExpandableTable  — Klasse
 *   SharedUi.table.instances[id]    — Registry (Auto-Register im Konstruktor)
 *
 * Registry-Semantik: Der Konstruktor registriert sich nach `this.init()`
 * synchron ueber `this.config.id` in `_instances`. Consumer setzen die
 * Registry NICHT mehr selbst; Lese-/Delete-Zugriffe laufen ueber
 * `SharedUi.table.instances`.
 *
 * CSP-Stufe 1: Inline-Handler wurden durch `data-table-action`/
 * `data-table-id`/`data-table-page`/`data-table-column` + Document-Level-
 * Event-Delegation ersetzt (analog Sprint-4/5-`data-copy-*`-Delegation).
 */
(function () {
    'use strict';

    if (typeof window.SharedUi === 'undefined' || typeof window.SharedUi.configure !== 'function') {
        console.error('SharedUi.table: core.js muss vor table.js laden.');
        return;
    }
    if (!window.SharedUi.runtime || typeof window.SharedUi.runtime.escapeHtml !== 'function') {
        console.error('SharedUi.table: runtime.js muss vor table.js laden.');
        return;
    }
    if (!window.SharedUi.fetch || typeof window.SharedUi.fetch.safeFetch !== 'function') {
        console.error('SharedUi.table: fetch.js muss vor table.js laden.');
        return;
    }
    if (window.SharedUi.table) {
        return;
    }

    const _instances = {};
    let _detailIdSeq = 0;

    class ExpandableTable {
        constructor(config, containerId, options = {}) {
            this.config = config;
            this.containerId = containerId;
            this.options = options;
            this.data = [];
            this.filteredData = [];
            this.sortColumn = config.defaultSort?.field || '';
            this.sortDirection = config.defaultSort?.direction || 'asc';
            this.activeFilters = {};
            this.filterValues = {};
            this.lastParams = {};
            this.filterDebounceTimer = null;
            this._requestSeq = 0;
            this._rowStableIds = new WeakMap();
            this._stableIdSeq = 0;

            this.pagination = {
                enabled: config.pagination?.enabled || false,
                currentPage: 1,
                pageSize: config.pagination?.defaultPageSize || 100,
                totalItems: 0,
                totalPages: 0,
                options: config.pagination?.options || [20, 100, 300, 1000]
            };

            this.renderers = {
                'badge': this.renderBadge.bind(this),
                'actions': this.renderActions.bind(this),
                'toggle': this.renderToggle.bind(this),
                'default': this.renderDefault.bind(this),
            };

            this.init();

            if (this.config && this.config.id) {
                if (_instances[this.config.id] && _instances[this.config.id] !== this
                        && typeof _instances[this.config.id].destroy === 'function') {
                    _instances[this.config.id].destroy();
                }
                _instances[this.config.id] = this;
            }
        }

        destroy() {
            if (this.filterDebounceTimer) {
                clearTimeout(this.filterDebounceTimer);
                this.filterDebounceTimer = null;
            }
            if (this.config && this.config.id && _instances[this.config.id] === this) {
                delete _instances[this.config.id];
            }
            this.data = [];
            this.filteredData = [];
        }

        init() {
            this.config.filters?.forEach(filter => {
                if (filter.type === 'buttons') this.activeFilters[filter.field] = new Set();
                else this.filterValues[filter.id] = filter.default || '';
            });
        }

        async loadData(params = {}) {
            this.showLoading();
            const seq = ++this._requestSeq;
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

                const response = await SharedUi.fetch.safeFetch(url);
                if (seq !== this._requestSeq) return false;
                const result = await response.json();
                if (seq !== this._requestSeq) return false;

                this.data = Array.isArray(result) ? result : (result.data || result.items || []);

                if (this.pagination.enabled && result.pagination) {
                    this.pagination.currentPage = result.pagination.page;
                    this.pagination.pageSize = result.pagination.page_size;
                    this.pagination.totalItems = result.pagination.total_items;
                    this.pagination.totalPages = result.pagination.total_pages;
                }

                if (this.options.onDataLoaded) this.options.onDataLoaded(result);
                this.applyFiltersAndRender();
                return true;
            } catch (error) {
                if (seq !== this._requestSeq) return false;
                console.error('Fehler beim Laden:', error);
                this.showError('Fehler beim Laden der Daten');
                return false;
            }
        }

        setData(data, totalItems) {
            this.data = data;
            this.pagination.totalItems = totalItems;
            this.pagination.totalPages = Math.ceil(totalItems / this.pagination.pageSize);
            this.applyFiltersAndRender();
        }

        goToPage(page) {
            if (page < 1 || page > this.pagination.totalPages || page === this.pagination.currentPage) return;
            this.pagination.currentPage = page;
            if (this.options.onPageChange) this.options.onPageChange(page, this.pagination.pageSize);
            else this.loadData(this.lastParams);
        }

        setPageSize(size) {
            if (size === this.pagination.pageSize) return;
            this.pagination.pageSize = size;
            this.pagination.currentPage = 1;
            if (this.options.onPageChange) this.options.onPageChange(1, size);
            else this.loadData(this.lastParams);
        }

        getPageNumbers() {
            const total = this.pagination.totalPages;
            const current = this.pagination.currentPage;
            const windowSize = 11;
            if (total <= windowSize) return Array.from({ length: total }, (_, i) => i + 1);
            const halfWindow = Math.floor(windowSize / 2);
            let start, end;
            if (current <= halfWindow + 1) { start = 1; end = windowSize; }
            else if (current >= total - halfWindow) { start = total - windowSize + 1; end = total; }
            else { start = current - halfWindow; end = current + halfWindow; }
            return Array.from({ length: end - start + 1 }, (_, i) => start + i);
        }

        renderPagination() {
            if (!this.pagination.enabled || this.pagination.totalPages <= 1) return '';
            const tableId = SharedUi.runtime.escapeAttr(this.config.id);
            const current = this.pagination.currentPage;
            const total = this.pagination.totalPages;
            const pageNumbers = this.getPageNumbers();
            const showStartEllipsis = pageNumbers[0] > 1;
            const showEndEllipsis = pageNumbers[pageNumbers.length - 1] < total;
            const pageSizeSelectId = `${this.containerId || tableId}-pagesize`;
            const safePageSizeSelectId = SharedUi.runtime.escapeAttr(pageSizeSelectId);

            let html = '<div class="pagination-wrapper"><div class="pagination-container"><div class="pagination-nav">';
            html += `<button type="button" class="pagination-btn ${current === 1 ? 'disabled' : ''}" `
                 + `data-table-action="goToPage" data-table-id="${tableId}" data-table-page="1" `
                 + `aria-label="Erste Seite" ${current === 1 ? 'disabled' : ''}>&laquo;</button>`;
            html += `<button type="button" class="pagination-btn ${current === 1 ? 'disabled' : ''}" `
                 + `data-table-action="goToPage" data-table-id="${tableId}" data-table-page="${current - 1}" `
                 + `aria-label="Vorherige Seite" ${current === 1 ? 'disabled' : ''}>&lt;</button>`;
            html += '<div class="pagination-pages">';
            if (showStartEllipsis) html += '<span class="pagination-ellipsis">...</span>';
            pageNumbers.forEach(page => {
                const isActive = page === current;
                const ariaCurrent = isActive ? ' aria-current="page"' : '';
                html += `<button type="button" class="pagination-btn ${isActive ? 'active' : ''}" `
                     + `data-table-action="goToPage" data-table-id="${tableId}" data-table-page="${page}" `
                     + `aria-label="Seite ${page}"${ariaCurrent}>${page}</button>`;
            });
            if (showEndEllipsis) html += '<span class="pagination-ellipsis">...</span>';
            html += '</div>';
            html += `<button type="button" class="pagination-btn ${current === total ? 'disabled' : ''}" `
                 + `data-table-action="goToPage" data-table-id="${tableId}" data-table-page="${current + 1}" `
                 + `aria-label="Naechste Seite" ${current === total ? 'disabled' : ''}>&gt;</button>`;
            html += `<button type="button" class="pagination-btn ${current === total ? 'disabled' : ''}" `
                 + `data-table-action="goToPage" data-table-id="${tableId}" data-table-page="${total}" `
                 + `aria-label="Letzte Seite" ${current === total ? 'disabled' : ''}>&raquo;</button>`;
            html += `</div><div class="pagination-info">${current} / ${total}</div></div>`;
            html += `<div class="pagination-page-size"><label for="${safePageSizeSelectId}">Pro Seite</label>`
                 + `<select id="${safePageSizeSelectId}" data-table-action="setPageSize" data-table-id="${tableId}" `
                 + `aria-label="Eintraege pro Seite">`;
            this.pagination.options.forEach(size => {
                html += `<option value="${size}" ${size === this.pagination.pageSize ? 'selected' : ''}>${size}</option>`;
            });
            html += '</select></div></div>';
            return html;
        }

        setFilter(filterId, value) {
            this.filterValues[filterId] = value;
            if (this.pagination.enabled) {
                clearTimeout(this.filterDebounceTimer);
                this.filterDebounceTimer = setTimeout(() => { this.pagination.currentPage = 1; this.loadData(this.lastParams); }, 300);
            } else this.applyFiltersAndRender();
        }

        toggleButtonFilter(field, value) {
            if (!this.activeFilters[field]) this.activeFilters[field] = new Set();
            if (this.activeFilters[field].has(value)) this.activeFilters[field].delete(value);
            else this.activeFilters[field].add(value);
            if (this.pagination.enabled) { this.pagination.currentPage = 1; this.loadData(this.lastParams); }
            else this.applyFiltersAndRender();
        }

        resetButtonFilter(field) {
            if (this.activeFilters[field]) this.activeFilters[field].clear();
            if (this.pagination.enabled) { this.pagination.currentPage = 1; this.loadData(this.lastParams); }
            else this.applyFiltersAndRender();
        }

        applyFiltersAndRender() {
            let filtered = this.data.slice();

            if (!this.pagination.enabled) {
                for (const [field, values] of Object.entries(this.activeFilters)) {
                    if (values.size > 0) filtered = filtered.filter(row => values.has(row[field] || 'none'));
                }
            }

            this.config.filters?.forEach(filter => {
                if (filter.type === 'buttons') return;
                const value = this.filterValues[filter.id];
                if (!value) return;
                if (filter.wildcard) {
                    const searchTerm = value.toLowerCase();
                    const fields = [filter.field, ...(filter.wildcardFields || [])];
                    filtered = filtered.filter(row => fields.some(f => (row[f] != null ? String(row[f]) : '').toLowerCase().includes(searchTerm)));
                } else {
                    filtered = filtered.filter(row => row[filter.field] == value);
                }
            });

            if (this.sortColumn && (!this.pagination.enabled || !this.config.apiEndpoint)) filtered = this.sortData(filtered);
            this.filteredData = filtered;
            if (this._silentMode) this.silentRender();
            else this.render();
        }

        sortBy(column) {
            if (this.sortColumn === column) this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
            else { this.sortColumn = column; this.sortDirection = column.includes('time') || column.includes('date') || column.includes('created') ? 'desc' : 'asc'; }
            if (this.pagination.enabled) {
                this.pagination.currentPage = 1;
                if (this.config.apiEndpoint) this.loadData(this.lastParams);
                else this.applyFiltersAndRender();
            } else this.applyFiltersAndRender();
        }

        sortData(data) {
            const colConfig = this.config.columns.find(c => c.field === this.sortColumn);
            const hasSortValue = colConfig && typeof colConfig.sortValue === 'function';
            return [...data].sort((a, b) => {
                let aVal = hasSortValue ? colConfig.sortValue(a) : a[this.sortColumn];
                let bVal = hasSortValue ? colConfig.sortValue(b) : b[this.sortColumn];
                if (typeof aVal === 'number') return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
                aVal = String(aVal || ''); bVal = String(bVal || '');
                const comparison = aVal.localeCompare(bVal, 'de');
                return this.sortDirection === 'asc' ? comparison : -comparison;
            });
        }

        render() {
            const container = document.getElementById(this.containerId);
            if (!container) return;

            if (!container.classList.contains('table-container')) {
                container.classList.add('table-container');
            }

            const expandedIds = new Set();
            if (this.config.expandable) {
                container.querySelectorAll('.table-row-wrapper.expanded').forEach(el => {
                    expandedIds.add(el.dataset.rowId);
                });
            }

            if (this.filteredData.length === 0) {
                let html = '<div class="table-no-data">Keine Daten vorhanden</div>';
                if (this.pagination.enabled) html += this.renderPagination();
                container.innerHTML = html;
                return;
            }
            let html = '';
            if (this.config.showHeader) html += this.renderHeader();
            html += '<div class="table-body">';
            this.filteredData.forEach((row, index) => { html += this.renderRow(row, index); });
            html += '</div>';
            if (this.pagination.enabled) html += this.renderPagination();
            container.innerHTML = html;

            container.querySelectorAll('.table-row > .table-cell').forEach(cell => {
                cell._renderFingerprint = cell.innerHTML;
            });

            if (expandedIds.size > 0) {
                expandedIds.forEach(id => {
                    const wrapper = container.querySelector(`[data-row-id="${CSS.escape(id)}"]`);
                    if (wrapper) {
                        wrapper.classList.add('expanded');
                        const toggle = wrapper.querySelector('[data-table-action="toggleRow"]');
                        if (toggle) toggle.setAttribute('aria-expanded', 'true');
                        const detail = wrapper.querySelector('.table-row-detail');
                        if (detail) {
                            window.SharedUi.runtimeStyle.applyStyle(detail, { maxHeight: 'none' });
                            detail.removeAttribute('hidden');
                            detail.removeAttribute('aria-hidden');
                            if (this.options.onAfterRestore) {
                                this.options.onAfterRestore(id, detail);
                            }
                        }
                    }
                });
            }
        }

        _gridTemplateClass() {
            // CSP-strict: grid-template-columns ist pro Tabelle konstant;
            // wir erzeugen einmalig eine u-rt-*-Klasse via runtimeStyle und cachen sie.
            if (this._gridClass) return this._gridClass;
            const tpl = this.config.gridTemplate || '';
            if (!tpl) return '';
            if (window.SharedUi && window.SharedUi.runtimeStyle && typeof window.SharedUi.runtimeStyle.classForProps === 'function') {
                this._gridClass = window.SharedUi.runtimeStyle.classForProps({ gridTemplateColumns: tpl });
                return this._gridClass;
            }
            return '';
        }

        renderHeader() {
            const tableId = SharedUi.runtime.escapeAttr(this.config.id);
            const gridCls = this._gridTemplateClass();
            let html = `<div class="table-header ${gridCls}">`;
            if (this.config.expandable) html += '<span class="header-cell header-expand"></span>';
            this.config.columns.forEach(col => {
                const sortIcon = this.getSortIcon(col.field);
                const labelHtml = col.rawLabel ? col.label : SharedUi.runtime.escapeHtml(col.label);
                if (col.sortable) {
                    const field = SharedUi.runtime.escapeAttr(col.field);
                    html += `<button type="button" class="header-cell sortable" `
                         + `data-table-action="sortBy" data-table-id="${tableId}" data-table-column="${field}">`
                         + `${labelHtml} ${sortIcon}</button>`;
                } else {
                    html += `<span class="header-cell">${labelHtml}</span>`;
                }
            });
            html += '</div>';
            return html;
        }

        _rowKey(row) {
            if (row && row.id != null) return String(row.id);
            if (row == null || typeof row !== 'object') return '';
            let key = this._rowStableIds.get(row);
            if (key == null) {
                key = `__r${++this._stableIdSeq}`;
                this._rowStableIds.set(row, key);
            }
            return key;
        }

        renderRow(row, index) {
            const tableId = SharedUi.runtime.escapeAttr(this.config.id);
            const rowId = this._rowKey(row);
            const safeRowId = SharedUi.runtime.escapeAttr(rowId);
            // Stabile ID pro Row (nicht ++_detailIdSeq) — sonst wechselt aria-controls bei jedem Re-Render
            const detailId = `tbl-detail-${this.config.id}-${safeRowId}`;
            const safeDetailId = SharedUi.runtime.escapeAttr(detailId);
            const gridCls = this._gridTemplateClass();
            let html = `<div class="table-row-wrapper" data-row-id="${safeRowId}" data-table-id="${tableId}">`;
            const cursorClass = this.config.expandable ? 'clickable' : '';
            if (this.config.expandable) {
                html += `<div class="table-row ${cursorClass} ${gridCls}" `
                     + `data-table-action="toggleRow" data-table-id="${tableId}" `
                     + `role="button" tabindex="0" aria-expanded="false" aria-controls="${safeDetailId}">`;
                html += '<span class="expand-icon" aria-hidden="true">\u25B6</span>';
            } else {
                html += `<div class="table-row ${gridCls}">`;
            }
            this.config.columns.forEach(col => {
                const value = row[col.field];
                const rendered = this.renderCell(value, col, row);
                const alignClass = col.align !== 'left' ? `align-${col.align}` : '';
                html += `<span class="table-cell ${alignClass} ${col.cssClass || ''}">${rendered}</span>`;
            });
            html += '</div>';
            if (this.config.expandable) {
                html += `<div class="table-row-detail" id="${safeDetailId}" hidden aria-hidden="true">${this.renderDetail(row)}</div>`;
            }
            html += '</div>';
            return html;
        }

        renderCell(value, col, row) {
            if (col.truncate > 0 && typeof value === 'string' && value.length > col.truncate) {
                return `<span title="${SharedUi.runtime.escapeAttr(value)}">${SharedUi.runtime.escapeHtml(value.substring(0, col.truncate) + '...')}</span>`;
            }
            if (typeof col.renderer === 'function') {
                return col.renderer(value, row);
            }
            const renderer = this.renderers[col.renderer] || this.renderers['default'];
            return renderer(value, col, row);
        }

        renderDetail(row) {
            if (typeof this.config.detailRenderer === 'function') {
                return '<div class="detail-content">' + this.config.detailRenderer(row) + '</div>';
            }
            let html = '<div class="detail-content">';
            const compactFields = this.config.detailFields.filter(f => !f.section);
            if (compactFields.length > 0) {
                html += '<div class="detail-fields">';
                compactFields.forEach(field => {
                    const value = row[field.field];
                    let rendered;
                    if (typeof field.renderer === 'function') {
                        rendered = field.renderer(value, field, row);
                    } else if (field.renderer) {
                        rendered = (this.renderers[field.renderer] || this.renderers['default'])(value, field, row);
                    } else {
                        rendered = SharedUi.runtime.escapeHtml(value != null ? String(value) : '-');
                    }
                    html += `<div class="detail-field"><span class="detail-label">${SharedUi.runtime.escapeHtml(field.label)}</span><span class="detail-value">${rendered}</span></div>`;
                });
                html += '</div>';
            }
            const sectionFields = this.config.detailFields.filter(f => f.section);
            sectionFields.forEach(field => {
                const value = row[field.field];
                if (!value) return;
                html += `<div class="detail-section"><div class="detail-section-label">${SharedUi.runtime.escapeHtml(field.label)}</div>`;
                html += `<div class="detail-text">${SharedUi.runtime.escapeHtml(value)}</div></div>`;
            });
            if (this.config.detailTextField && row[this.config.detailTextField]) {
                let text = row[this.config.detailTextField];
                if (text.length > this.config.detailTextMaxLength) text = text.substring(0, this.config.detailTextMaxLength) + '\n\n[Gekuerzt]';
                html += `<div class="detail-section">`;
                if (this.config.detailTextLabel) html += `<div class="detail-section-label">${SharedUi.runtime.escapeHtml(this.config.detailTextLabel)}</div>`;
                html += `<div class="detail-longtext">${SharedUi.runtime.escapeHtml(text)}</div></div>`;
            }
            html += '</div>';
            return html;
        }

        renderDefault(value) { return SharedUi.runtime.escapeHtml(value ?? '-'); }
        renderBadge(value, col) { const cls = col.rendererOptions?.class || value?.toLowerCase() || ''; return `<span class="badge badge-${cls}">${SharedUi.runtime.escapeHtml(value || '-')}</span>`; }
        renderActions(value, col, row) { return ''; }
        renderToggle(value, col, row) { return `<label class="toggle-switch"><input type="checkbox" ${value ? 'checked' : ''}><span class="toggle-slider"></span></label>`; }

        toggleRow(rowId) {
            const container = document.getElementById(this.containerId);
            if (!container) return;
            const wrapper = container.querySelector(`[data-row-id="${CSS.escape(rowId)}"]`);
            if (!wrapper) return;

            const detail = wrapper.querySelector('.table-row-detail');
            if (!detail) return;

            const toggle = wrapper.querySelector('[data-table-action="toggleRow"]');

            const clearExpandHandler = () => {
                if (detail._expandTransitionHandler) {
                    detail.removeEventListener('transitionend', detail._expandTransitionHandler);
                    detail._expandTransitionHandler = null;
                }
            };
            const clearCollapseHandler = () => {
                if (detail._collapseTransitionHandler) {
                    detail.removeEventListener('transitionend', detail._collapseTransitionHandler);
                    detail._collapseTransitionHandler = null;
                }
            };
            const heightsEqual = (a, b) => Math.abs(a - b) < 1;

            const wasExpanded = wrapper.classList.contains('expanded');
            const rowTop = wrapper.getBoundingClientRect().top;

            if (!wasExpanded) {
                detail.removeAttribute('hidden');
                detail.removeAttribute('aria-hidden');
            }

            const applyMaxHeight = (value) => {
                window.SharedUi.runtimeStyle.applyStyle(detail, { maxHeight: value });
            };

            const startHeight = detail.getBoundingClientRect().height;
            applyMaxHeight(startHeight + 'px');
            void detail.offsetHeight;

            if (wasExpanded) {
                clearExpandHandler();
                clearCollapseHandler();
                wrapper.classList.remove('expanded');
                if (toggle) toggle.setAttribute('aria-expanded', 'false');

                if (heightsEqual(startHeight, 0)) {
                    applyMaxHeight('');
                    detail.setAttribute('hidden', '');
                    detail.setAttribute('aria-hidden', 'true');
                    if (this.options.onRowCollapsed) this.options.onRowCollapsed(rowId, wrapper);
                    const newRowTop = wrapper.getBoundingClientRect().top;
                    window.scrollBy(0, newRowTop - rowTop);
                    return;
                }

                applyMaxHeight('0');

                const collapseHandler = (e) => {
                    if (e.propertyName !== 'max-height') return;
                    applyMaxHeight('');
                    detail.setAttribute('hidden', '');
                    detail.setAttribute('aria-hidden', 'true');
                    clearCollapseHandler();
                };
                detail._collapseTransitionHandler = collapseHandler;
                detail.addEventListener('transitionend', collapseHandler);

                if (this.options.onRowCollapsed) this.options.onRowCollapsed(rowId, wrapper);
                const newRowTop = wrapper.getBoundingClientRect().top;
                window.scrollBy(0, newRowTop - rowTop);
                return;
            }

            clearCollapseHandler();
            clearExpandHandler();
            wrapper.classList.add('expanded');
            if (toggle) toggle.setAttribute('aria-expanded', 'true');

            if (this.options.onRowExpanded) {
                this.options.onRowExpanded(rowId, detail);
            }

            const targetHeight = detail.scrollHeight;
            if (heightsEqual(targetHeight, startHeight)) {
                applyMaxHeight('none');
                return;
            }

            applyMaxHeight(targetHeight + 'px');

            const expandHandler = (e) => {
                if (e.propertyName !== 'max-height') return;
                applyMaxHeight('none');
                clearExpandHandler();
            };
            detail._expandTransitionHandler = expandHandler;
            detail.addEventListener('transitionend', expandHandler);
        }

        getSortIcon(field) {
            if (this.sortColumn !== field) return '<span class="sort-icon" aria-hidden="true">\u21C5</span>';
            return this.sortDirection === 'asc'
                ? '<span class="sort-icon active" aria-hidden="true">\u2191</span>'
                : '<span class="sort-icon active" aria-hidden="true">\u2193</span>';
        }

        showLoading() {
            const container = document.getElementById(this.containerId);
            if (container) container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
        }

        showError(message) {
            const container = document.getElementById(this.containerId);
            if (container) container.innerHTML = `<div class="table-error">${SharedUi.runtime.escapeHtml(message)}</div>`;
        }

        getData() { return this.filteredData; }
        getAllData() { return this.data; }
        refresh() { this.loadData(this.lastParams || {}); }

        async silentRefresh() {
            const seq = ++this._requestSeq;
            const params = this.lastParams || {};
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
            let response;
            try {
                response = await SharedUi.fetch.safeFetch(url);
            } catch (err) {
                if (seq !== this._requestSeq) return false;
                console.warn('silentRefresh fetch fehlgeschlagen:', err);
                return false;
            }
            if (seq !== this._requestSeq) return false;
            if (!response.ok) {
                console.warn(`silentRefresh HTTP ${response.status}`);
                return false;
            }
            const result = await response.json();
            if (seq !== this._requestSeq) return false;
            this.data = Array.isArray(result) ? result : (result.data || result.items || []);
            if (this.pagination.enabled && result.pagination) {
                this.pagination.currentPage = result.pagination.page;
                this.pagination.pageSize = result.pagination.page_size;
                this.pagination.totalItems = result.pagination.total_items;
                this.pagination.totalPages = result.pagination.total_pages;
            }
            if (this.options.onDataLoaded) this.options.onDataLoaded(result);
            this._silentMode = true;
            try { this.applyFiltersAndRender(); } finally { this._silentMode = false; }
            return true;
        }

        silentRender() {
            const container = document.getElementById(this.containerId);
            if (!container) return;

            const existingWrappers = container.querySelectorAll('.table-row-wrapper');
            const existingById = new Map();
            existingWrappers.forEach(el => existingById.set(el.dataset.rowId, el));

            const newIds = this.filteredData.map(row => this._rowKey(row));
            const oldIds = Array.from(existingById.keys());

            const sameStructure = newIds.length === oldIds.length &&
                                  newIds.every((id, i) => id === oldIds[i]);

            if (!sameStructure) {
                this.render();
                return;
            }

            const renderProbe = this._renderProbe || (this._renderProbe = document.createElement('div'));

            this.filteredData.forEach((row) => {
                const rowId = this._rowKey(row);
                const wrapper = existingById.get(rowId);
                if (!wrapper) return;

                const cells = wrapper.querySelectorAll('.table-row > .table-cell');
                this.config.columns.forEach((col, colIdx) => {
                    const cell = cells[colIdx];
                    if (!cell) return;

                    const value = row[col.field];
                    const rendered = this.renderCell(value, col, row);

                    renderProbe.innerHTML = rendered;
                    const normalized = renderProbe.innerHTML;
                    const previous = cell._renderFingerprint ?? cell.innerHTML;

                    if (previous !== normalized) {
                        cell.innerHTML = rendered;
                        cell._renderFingerprint = cell.innerHTML;
                    } else {
                        cell._renderFingerprint = previous;
                    }
                });
            });
        }
    }

    // ── Document-Level Event-Delegation ──────────────────────────────
    // Whitelist: goToPage / sortBy / toggleRow (click) und setPageSize (change).
    // Guards: Missing-Instance, Missing-Data-ID, nicht-`[data-table-action]`-Klicks
    // aus interaktiven Zell-Elementen brechen ab (ersetzt den alten
    // renderToggle()-`event.stopPropagation()`-Effekt ohne Pagination-Blockade).

    function _interactiveCellMatch(target) {
        return target.closest('.toggle-switch, input, select, a, button, [contenteditable="true"]');
    }

    function _handleAction(instance, action, targetEl, ev) {
        if (!instance) return;
        if (action === 'goToPage') {
            const page = parseInt(targetEl.getAttribute('data-table-page'), 10);
            if (Number.isFinite(page)) instance.goToPage(page);
            return;
        }
        if (action === 'sortBy') {
            const column = targetEl.getAttribute('data-table-column');
            if (column) instance.sortBy(column);
            return;
        }
        if (action === 'toggleRow') {
            const wrapper = targetEl.closest('[data-row-id]');
            if (!wrapper) return;
            const rowId = wrapper.dataset.rowId;
            if (rowId != null) instance.toggleRow(rowId);
            if (ev) ev.preventDefault();
            return;
        }
    }

    document.addEventListener('click', function (ev) {
        const actionEl = ev.target.closest('[data-table-action]');
        if (!actionEl) return;
        const action = actionEl.getAttribute('data-table-action');
        if (action !== 'goToPage' && action !== 'sortBy' && action !== 'toggleRow') return;

        if (action === 'toggleRow') {
            const interactive = _interactiveCellMatch(ev.target);
            // Klick landete auf einem interaktiven Zellen-Element
            // (.toggle-switch, input, select, a, button, [contenteditable]).
            // In diesem Fall kein Row-Expand ausloesen, damit der Click-
            // Default (Checkbox kippen, Link oeffnen, Button-Action) greift.
            // Der frueher pruefende !closest('[data-table-action="toggleRow"]')
            // war immer false (Zellen liegen per Struktur innerhalb der Row),
            // deshalb hat der Guard nie gegriffen — tab_ki_chat.js Toggle-Chat/
            // Aktiv-Bug vom 2026-04-24.
            if (interactive && interactive !== actionEl) {
                return;
            }
        }

        if (actionEl.hasAttribute('disabled') || actionEl.classList.contains('disabled')) return;

        const tableId = actionEl.getAttribute('data-table-id');
        if (!tableId) return;
        const instance = _instances[tableId];
        if (!instance) return;

        _handleAction(instance, action, actionEl, ev);
    });

    document.addEventListener('keydown', function (ev) {
        if (ev.key !== 'Enter' && ev.key !== ' ' && ev.key !== 'Spacebar') return;
        const actionEl = ev.target.closest('[data-table-action]');
        if (!actionEl) return;
        const action = actionEl.getAttribute('data-table-action');
        if (action !== 'sortBy' && action !== 'toggleRow') return;
        if (ev.target !== actionEl && actionEl.tagName !== 'BUTTON') return;
        if (actionEl.tagName === 'BUTTON' && ev.key === 'Enter') return;

        if (action === 'toggleRow') {
            const interactive = _interactiveCellMatch(ev.target);
            // Siehe Click-Variante oben: falscher !closest()-Check entfernt,
            // damit Enter/Space auf einem Input/Button innerhalb der Row
            // nicht versehentlich den Row-Expand triggert.
            if (interactive && interactive !== actionEl) {
                return;
            }
        }

        const tableId = actionEl.getAttribute('data-table-id');
        if (!tableId) return;
        const instance = _instances[tableId];
        if (!instance) return;

        ev.preventDefault();
        _handleAction(instance, action, actionEl, ev);
    });

    document.addEventListener('change', function (ev) {
        const actionEl = ev.target.closest('[data-table-action="setPageSize"]');
        if (!actionEl) return;
        const tableId = actionEl.getAttribute('data-table-id');
        if (!tableId) return;
        const instance = _instances[tableId];
        if (!instance) return;
        const size = parseInt(actionEl.value, 10);
        if (Number.isFinite(size)) instance.setPageSize(size);
    });

    SharedUi.table = {
        ExpandableTable: ExpandableTable,
        instances: _instances
    };
})();
