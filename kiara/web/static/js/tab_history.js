/**
 * KIara - History Tab
 * Chat-Historie mit ExpandableTable.
 */

let historyTable = null;
let currentStatusFilter = '';

const _HISTORY_ACTIONS = new Set([
    'filter-history-by-status',
    'start-title-edit',
    'resume-chat',
    'archive-chat',
    'restore-chat',
    'delete-chat',
]);

const _HISTORY_ROW_ACTIONS = new Set([
    'start-title-edit',
    'resume-chat',
    'archive-chat',
    'restore-chat',
    'delete-chat',
]);

const _HISTORY_DBLCLICK_ACTIONS = new Set(['start-title-edit']);
const _HISTORY_INPUT_ACTIONS = new Set(['update-history-search']);

function _handleHistoryClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_HISTORY_ACTIONS.has(el.dataset.action)) return;
    if (_HISTORY_ROW_ACTIONS.has(el.dataset.action)) e.stopPropagation();
    const chatId = Number(el.dataset.chatId);
    switch (el.dataset.action) {
        case 'filter-history-by-status':
            filterHistoryByStatus(el, el.dataset.value || '');
            break;
        case 'start-title-edit': startTitleEdit(chatId); break;
        case 'resume-chat':      resumeChat(chatId); break;
        case 'archive-chat':     archiveChat(chatId); break;
        case 'restore-chat':     restoreChat(chatId); break;
        case 'delete-chat':      deleteChat(chatId); break;
    }
}

function _handleHistoryDblClick(e) {
    const el = e.target.closest('[data-dblclick-action]');
    if (!el || !_HISTORY_DBLCLICK_ACTIONS.has(el.dataset.dblclickAction)) return;
    e.stopPropagation();
    if (el.dataset.dblclickAction === 'start-title-edit') {
        startTitleEdit(Number(el.dataset.chatId));
    }
}

function _handleHistoryInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el || !_HISTORY_INPUT_ACTIONS.has(el.dataset.inputAction)) return;
    if (el.dataset.inputAction === 'update-history-search' && historyTable) {
        historyTable.setFilter('search', el.value);
    }
}

function _bindHistoryActions() {
    if (document.body._historyActionsBound) return;
    document.body._historyActionsBound = true;
    document.body.addEventListener('click', _handleHistoryClick);
    document.body.addEventListener('dblclick', _handleHistoryDblClick);
    document.body.addEventListener('input', _handleHistoryInput);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-history-actions-bound', 'true');
}

async function initHistoryTab() {
    _bindHistoryActions();
    currentStatusFilter = '';
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Build filter bar
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.remove('u-hidden');
        filterBar.classList.add('u-flex');

        // Search input
        const searchGroup = document.createElement('div');
        searchGroup.className = 'filter-group';
        searchGroup.innerHTML = `<span class="filter-label">Suche:</span>
            <input type="text" class="filter-input" id="historySearch" placeholder="Titel durchsuchen..."
                data-input-action="update-history-search">`;
        filterBar.appendChild(searchGroup);

        // Status filter buttons
        const statusGroup = document.createElement('div');
        statusGroup.className = 'filter-group';
        statusGroup.innerHTML = `<span class="filter-label">Status:</span>
            <div class="filter-buttons" id="historyStatusFilter">
                <button class="filter-btn active" data-action="filter-history-by-status" data-value="">Alle</button>
                <button class="filter-btn" data-action="filter-history-by-status" data-value="active">Aktiv</button>
                <button class="filter-btn" data-action="filter-history-by-status" data-value="archived">Archiviert</button>
            </div>`;
        filterBar.appendChild(statusGroup);
    }

    container.innerHTML = '<div id="historyTableContainer"></div>';

    const config = {
        id: 'history',
        apiEndpoint: '/api/chats',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 150px 1fr 80px 80px 120px',
        defaultSort: { field: 'updated_at', direction: 'desc' },
        columns: [
            { field: 'updated_at', label: 'Datum', sortable: true, align: 'left', renderer: 'default',
              truncate: 16 },
            { field: 'title', label: 'Titel', sortable: true, align: 'left', truncate: 60,
              renderer: 'editableTitle' },
            { field: 'message_count', label: 'Nachr.', sortable: true, align: 'center' },
            { field: 'status', label: 'Status', sortable: true, align: 'center', renderer: 'badge',
              rendererOptions: { class: '' } },
            { field: 'id', label: 'Aktionen', sortable: false, align: 'center', renderer: 'actions' },
        ],
        detailFields: [
            { field: 'created_at', label: 'Erstellt' },
            { field: 'updated_at', label: 'Aktualisiert' },
        ],
        filters: [
            { id: 'search', field: 'title', wildcard: true, type: 'text' },
        ],
    };

    historyTable = new SharedUi.table.ExpandableTable(config, 'historyTableContainer', {
        onRowExpanded: async (rowId, detailEl) => {
            await loadChatDetail(rowId, detailEl);
        },
    });

    // Editable title renderer (Doppelklick zum Bearbeiten)
    historyTable.renderers['editableTitle'] = (value, col, row) => {
        const display = value && value.length > 60 ? value.substring(0, 60) + '...' : (value || '');
        return `<span class="editable-title u-cursor-text" data-chat-id="${SharedUi.runtime.escapeAttr(String(row.id))}" data-full-title="${SharedUi.runtime.escapeAttr(value || '')}"
            data-dblclick-action="start-title-edit"
            title="Doppelklick zum Umbenennen">${SharedUi.runtime.escapeHtml(display)}</span>`;
    };

    // Override actions renderer
    historyTable.renderers['actions'] = (value, col, row) => {
        let html = '<div class="action-buttons">';
        html += `<button class="action-btn-sm" title="Umbenennen" data-action="start-title-edit" data-chat-id="${SharedUi.runtime.escapeAttr(String(row.id))}">&#x270E;</button>`;
        if (row.status === 'active') {
            html += `<button class="action-btn-sm" title="Fortsetzen" data-action="resume-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(row.id))}">&#x25B6;</button>`;
            html += `<button class="action-btn-sm" title="Archivieren" data-action="archive-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(row.id))}">&#x1F4E6;</button>`;
        } else {
            html += `<button class="action-btn-sm" title="Wiederherstellen" data-action="restore-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(row.id))}">&#x21A9;</button>`;
        }
        html += `<button class="action-btn-sm danger" title="Loeschen" data-action="delete-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(row.id))}">&#x2716;</button>`;
        html += '</div>';
        return html;
    };

    // Override badge renderer for status
    historyTable.renderers['badge'] = (value, col, row) => {
        const cls = value === 'active' ? 'success' : value === 'archived' ? 'warning' : 'default';
        const text = value === 'active' ? 'Aktiv' : value === 'archived' ? 'Archiviert' : (value || '-');
        return `<span class="badge badge-${cls}">${SharedUi.runtime.escapeHtml(text)}</span>`;
    };

    // Wrap applyFiltersAndRender to include custom status filter
    const origApplyFilters = historyTable.applyFiltersAndRender.bind(historyTable);
    historyTable.applyFiltersAndRender = function() {
        origApplyFilters();
        if (currentStatusFilter) {
            this.filteredData = this.filteredData.filter(row => row.status === currentStatusFilter);
            this.render();
        }
    };

    historyTable.loadData();
}

function filterHistoryByStatus(btn, status) {
    currentStatusFilter = status;
    // Update button states
    document.querySelectorAll('#historyStatusFilter .filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // Re-filter
    if (historyTable) {
        historyTable.pagination.currentPage = 1;
        historyTable.applyFiltersAndRender();
    }
}

async function loadChatDetail(chatId, detailEl) {
    if (!detailEl) return;
    detailEl.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}`);
        if (!resp.ok) throw new Error('Laden fehlgeschlagen');
        const chat = await resp.json();

        let html = '<div class="chat-detail-preview">';
        if (chat.messages && chat.messages.length > 0) {
            html += '<div class="detail-messages">';
            chat.messages.forEach(msg => {
                const roleLabel = msg.role === 'user' ? 'Du' : 'AI';
                const roleClass = msg.role === 'user' ? 'user' : 'assistant';
                const content = msg.content.length > 500
                    ? msg.content.substring(0, 500) + '...'
                    : msg.content;
                html += `<div class="detail-message ${roleClass}">`;
                html += `<strong>${roleLabel}:</strong> ${SharedUi.runtime.escapeHtml(content)}`;
                if (msg.sources && msg.sources.length > 0) {
                    html += `<div class="detail-sources">${msg.sources.length} Quelle(n)</div>`;
                }
                html += '</div>';
            });
            html += '</div>';
        } else {
            html += '<div class="no-data">Keine Nachrichten</div>';
        }

        html += '<div class="detail-actions-bar">';
        html += `<button class="action-btn primary" data-action="resume-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(chatId))}">Chat fortsetzen</button>`;
        html += '</div>';
        html += '</div>';

        detailEl.innerHTML = html;
    } catch (e) {
        detailEl.innerHTML = '<div class="table-error">Fehler beim Laden des Chat-Verlaufs</div>';
    }
}

async function resumeChat(chatId) {
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}`);
        if (!resp.ok) throw new Error('Chat laden fehlgeschlagen');
        const chat = await resp.json();

        // Pending-Load-Intent: wird von initChatTab oder handleChatTabRestore
        // angewendet — robust gegen Doppelklick / Out-of-order-Aufrufe.
        if (typeof _setPendingChatLoad === 'function') {
            _setPendingChatLoad(chatId, chat.messages || []);
        }
        navigateTo('chat');

        // Hot-Path fuer den Fall, dass chatUI bereits bereit ist.
        if (typeof chatUI !== 'undefined' && chatUI && typeof _applyPendingChatLoad === 'function') {
            _applyPendingChatLoad();
        }
    } catch (e) {
        SharedUi.notify.show('Chat konnte nicht geladen werden', 'error');
    }
}

async function archiveChat(chatId) {
    if (!await SharedUi.modal.confirm({ message: 'Chat archivieren?' })) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}/archive`, { method: 'PUT' });
        if (!resp.ok) throw new Error('Archivieren fehlgeschlagen');
        SharedUi.notify.show('Chat archiviert', 'success');
        if (historyTable) historyTable.loadData();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function restoreChat(chatId) {
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}/restore`, { method: 'PUT' });
        if (!resp.ok) throw new Error('Wiederherstellen fehlgeschlagen');
        SharedUi.notify.show('Chat wiederhergestellt', 'success');
        if (historyTable) historyTable.loadData();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

function startTitleEdit(chatId) {
    const span = document.querySelector(`.editable-title[data-chat-id="${chatId}"]`);
    if (!span) return;
    const currentTitle = (span.dataset.fullTitle !== undefined && span.dataset.fullTitle !== '') ? span.dataset.fullTitle : span.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentTitle;
    input.className = 'filter-input history-title-edit';

    let saved = false;
    const save = async () => {
        if (saved) return;
        saved = true;
        const newTitle = input.value.trim();
        if (!newTitle || newTitle === currentTitle) {
            if (historyTable) historyTable.loadData();
            return;
        }
        try {
            const resp = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle }),
            });
            if (!resp.ok) throw new Error('Umbenennen fehlgeschlagen');
            SharedUi.notify.show('Titel aktualisiert', 'success');
        } catch (e) {
            SharedUi.notify.show(e.message, 'error');
        }
        if (historyTable) historyTable.loadData();
    };

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); save(); }
        if (e.key === 'Escape') { saved = true; if (historyTable) historyTable.loadData(); }
    });
    input.addEventListener('blur', save);

    span.replaceWith(input);
    input.focus();
    input.select();
}

async function deleteChat(chatId) {
    if (!await SharedUi.modal.confirm({ message: 'Chat unwiderruflich loeschen?' })) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Loeschen fehlgeschlagen');
        SharedUi.notify.show('Chat geloescht', 'success');
        if (historyTable) historyTable.loadData();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}
