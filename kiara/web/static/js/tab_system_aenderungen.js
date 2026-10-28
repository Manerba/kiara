/**
 * KIara Admin - System > Aenderungen Sub-Tab
 * Config-Changelog mit ExpandableTable-Komponente.
 */

let clTable = null;
let clAreaFilter = '';
let clSourceFilter = '';
let _saInitSeq = 0;

const _SA_CHANGE_ACTIONS = new Set(['cl-set-area', 'cl-set-source']);

function _handleSaChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_SA_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    switch (el.dataset.changeAction) {
        case 'cl-set-area':   clSetArea(el.value); break;
        case 'cl-set-source': clSetSource(el.value); break;
    }
}

function _bindSaActions() {
    if (document.body._saActionsBound) return;
    document.body._saActionsBound = true;
    document.body.addEventListener('change', _handleSaChange);
}

async function initSystemAenderungenTab() {
    _bindSaActions();
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-aenderungen');
    _saInitSeq++;
    const mySeq = _saInitSeq;

    registerTabCleanup('system-aenderungen', () => {
        _saInitSeq++;
        clTable = null;
        delete SharedUi.table.instances['changelog'];
    });

    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Filter bar
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.remove('u-hidden');
        filterBar.classList.add('u-flex');

        const areaGroup = document.createElement('div');
        areaGroup.className = 'filter-group';
        areaGroup.innerHTML = `<span class="filter-label">Bereich:</span>
            <select class="filter-input" id="clAreaFilter" data-change-action="cl-set-area">
                <option value="">Alle</option>
            </select>`;
        filterBar.appendChild(areaGroup);

        const sourceGroup = document.createElement('div');
        sourceGroup.className = 'filter-group';
        sourceGroup.innerHTML = `<span class="filter-label">Quelle:</span>
            <select class="filter-input" id="clSourceFilter" data-change-action="cl-set-source">
                <option value="">Alle</option>
                <option value="gui">GUI</option>
                <option value="chat">Chat</option>
            </select>`;
        filterBar.appendChild(sourceGroup);
    }

    // Areas laden (Soft-Fail erlaubt, aber stale Fortsetzung guardieren)
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/system/changelog/areas');
        if (mySeq !== _saInitSeq) return;
        if (resp.ok) {
            const data = await resp.json();
            if (mySeq !== _saInitSeq) return;
            const sel = document.getElementById('clAreaFilter');
            if (sel && data.areas) {
                data.areas.forEach(a => {
                    const opt = document.createElement('option');
                    opt.value = a;
                    opt.textContent = a;
                    sel.appendChild(opt);
                });
            }
        }
    } catch (e) { /* ignore */ }

    if (mySeq !== _saInitSeq) return;
    container.innerHTML = '<div id="clTableContainer"></div>';

    const actionLabels = {
        'create': 'Erstellt',
        'update': 'Geaendert',
        'delete': 'Geloescht',
    };

    const config = {
        id: 'changelog',
        apiEndpoint: '/api/admin/system/changelog',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 150px 100px 120px 90px 70px 1fr 120px',
        defaultSort: { field: 'created_at', direction: 'desc' },
        pagination: {
            enabled: true,
            defaultPageSize: 50,
            options: [20, 50, 100, 200],
        },
        columns: [
            {
                field: 'created_at', label: 'Zeitstempel', sortable: true, align: 'left',
                renderer: (v) => {
                    if (!v) return '<span class="u-c-secondary">–</span>';
                    const d = new Date(v.includes('T') ? v : v.replace(' ', 'T'));
                    if (isNaN(d.getTime())) return SharedUi.runtime.escapeHtml(v);
                    const pad = n => String(n).padStart(2, '0');
                    return `<span class="u-fs-12">${pad(d.getDate())}.${pad(d.getMonth()+1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}</span>`;
                }
            },
            { field: 'username', label: 'Benutzer', sortable: true, align: 'left' },
            { field: 'area', label: 'Bereich', sortable: true, align: 'left' },
            {
                field: 'action', label: 'Aktion', sortable: true, align: 'center',
                renderer: (v) => {
                    const label = actionLabels[v] || v;
                    const colors = { create: 'success', update: 'info', delete: 'danger' };
                    const cls = colors[v] || 'default';
                    return `<span class="badge badge-${cls}">${SharedUi.runtime.escapeHtml(label)}</span>`;
                }
            },
            {
                field: 'source', label: 'Quelle', sortable: true, align: 'center',
                renderer: (v) => {
                    const src = v || 'gui';
                    if (src === 'chat') return '<span class="badge badge-info">Chat</span>';
                    return '<span class="badge badge-default">GUI</span>';
                }
            },
            { field: 'target_label', label: 'Ziel', sortable: true, align: 'left', truncate: 40 },
            { field: 'field_name', label: 'Feld', sortable: true, align: 'left' },
        ],
        detailRenderer: (row) => {
            let html = '<div class="detail-fields">';
            if (row.target_id) {
                html += `<div class="detail-field"><span class="detail-label">Ziel-ID</span><span class="detail-value">${SharedUi.runtime.escapeHtml(row.target_id)}</span></div>`;
            }
            html += '</div>';

            const hasOld = row.old_value !== null && row.old_value !== undefined;
            const hasNew = row.new_value !== null && row.new_value !== undefined;

            if (hasOld || hasNew) {
                html += '<div class="u-grid-2col-12 u-mt-8">';
                if (hasOld) {
                    html += `<div class="detail-section"><div class="detail-section-label">Alter Wert</div>
                        <div class="detail-longtext u-max-h-200">${SharedUi.runtime.escapeHtml(row.old_value)}</div></div>`;
                } else {
                    html += '<div></div>';
                }
                if (hasNew) {
                    html += `<div class="detail-section"><div class="detail-section-label">Neuer Wert</div>
                        <div class="detail-longtext u-max-h-200">${SharedUi.runtime.escapeHtml(row.new_value)}</div></div>`;
                } else {
                    html += '<div></div>';
                }
                html += '</div>';
            }
            return html;
        },
        detailFields: [],
    };

    clTable = new SharedUi.table.ExpandableTable(config, 'clTableContainer');

    clAreaFilter = '';
    clSourceFilter = '';
    clTable.loadData();
}

function _clLoadFiltered() {
    if (!clTable) return;
    clTable.pagination.currentPage = 1;
    const params = {};
    if (clAreaFilter) params.area = clAreaFilter;
    if (clSourceFilter) params.source = clSourceFilter;
    clTable.loadData(params);
}

function clSetArea(value) {
    clAreaFilter = value;
    _clLoadFiltered();
}

function clSetSource(value) {
    clSourceFilter = value;
    _clLoadFiltered();
}
