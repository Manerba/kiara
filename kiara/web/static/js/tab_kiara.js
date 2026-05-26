/**
 * KIara - Admin-Chat Tab (Tab "KIara" in der Admin-UI)
 * Initialisiert den Admin-Chat mit Sidebar (Chat-Liste) und Chat-Bereich.
 *
 * Draft-Modus: Neue Chats starten als lokaler Draft (kein DB-Eintrag).
 * Der Chat wird erst beim ersten User-Prompt persistiert (Lazy-Creation),
 * analog zu etablierten Web-Chats (ChatGPT, Claude).
 *
 * Hinweis: Dieser Service nutzt bewusst direktes fetch() statt SharedUi.fetch.safeFetch,
 * weil der SSE-Stream über ReadableStream + manuelles Event-Parsing läuft und die
 * Cross-Origin-URL (Port 8503) dynamisch konstruiert wird. 401-Redirect wird lokal
 * nachgebildet, weil der globale fetch-Interceptor entfernt wurde. Siehe Sprint 2 Brief.
 */

// Port wird von admin_app.py als window.ADMIN_CHAT_PORT injiziert (Fallback: 8503)
const ADMIN_CHAT_PORT = window.ADMIN_CHAT_PORT || 8503;
const ADMIN_CHAT_BASE = `${window.location.protocol}//${window.location.hostname}:${ADMIN_CHAT_PORT}`;

let _adminChatUI = null;
let _adminChats = [];
let _chatRequestSeq = 0;
let _manualTitleChats = new Set();
let _adminChatRenameCleanup = null;

function _kiaraAdminChatRedirect401(resp) {
    if (resp && resp.status === 401) {
        const cfg = (window.SharedUi && typeof SharedUi._config === 'function') ? SharedUi._config() : null;
        window.location.href = (cfg && cfg.loginUrl) || '/login';
        return true;
    }
    return false;
}

function initAdminChat() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    // Re-Init-Schutz: altes Cleanup ausfuehren (Self-Click auf aktiven Tab
    // ueberspringt switchAdminTab._runTabCleanups, ruft aber callInitFunction).
    if (typeof _runTabCleanups === 'function') _runTabCleanups('kiara');

    container.innerHTML = `
        <div class="admin-chat-layout">
            <aside class="admin-chat-sidebar" id="adminChatSidebar">
                <div class="admin-chat-sidebar-header">
                    <button class="admin-chat-new-btn" id="adminChatNewBtn" title="Neuer Chat">+ Neuer Chat</button>
                    <button class="admin-chat-toggle-btn" id="adminChatToggleBtn" title="Sidebar ein-/ausklappen">&#x2630;</button>
                </div>
                <div class="admin-chat-list" id="adminChatList">
                    <div class="admin-chat-list-loading">Lade Chats...</div>
                </div>
            </aside>
            <section class="admin-chat-main" id="adminChatMain">
                <div class="chat-empty">
                    <div class="chat-empty-icon">&#x1F916;</div>
                    <div class="chat-empty-text">KIara Admin-Chat</div>
                    <div class="chat-empty-hint">Klicke "Neuer Chat" oder waehle einen bestehenden Chat.</div>
                </div>
            </section>
        </div>`;

    const unbindListActions = _bindAdminChatListActions();
    _adminChatUI = new AdminChatUI('adminChatMain', ADMIN_CHAT_BASE);

    // Lazy-Creation Callback: wird von AdminChatUI.sendMessage() aufgerufen
    _adminChatUI._onCreateChat = async (greeting, greetingType) => {
        const chat = await _persistNewChat(greeting, greetingType);
        if (chat) await _loadChatList();
        return chat;
    };

    // Events
    const newBtn = document.getElementById('adminChatNewBtn');
    const onNewChatClick = () => _startDraftChat();
    if (newBtn) newBtn.addEventListener('click', onNewChatClick);
    const toggleBtn = document.getElementById('adminChatToggleBtn');
    const onToggleSidebarClick = () => {
        const sidebar = document.getElementById('adminChatSidebar');
        if (sidebar) sidebar.classList.toggle('collapsed');
    };
    if (toggleBtn) toggleBtn.addEventListener('click', onToggleSidebarClick);

    // Cleanup bei Tab-Wechsel: SSE-Stream beenden, UI-Instanz verwerfen,
    // pending fetches per Seq-Bump self-cancel (siehe _startDraftChat/_selectChat).
    // destroy() setzt das Lifecycle-Flag, sodass laufende sendMessage-Promises
    // nach jedem await abbrechen und keinen Phantom-Chat-Stream mehr starten.
    registerTabCleanup('kiara', () => {
        if (_adminChatUI) {
            if (_adminChatUI.isStreaming) _adminChatUI.abortStream();
            _adminChatUI.destroy();
            _adminChatUI = null;
        }
        if (newBtn) newBtn.removeEventListener('click', onNewChatClick);
        if (toggleBtn) toggleBtn.removeEventListener('click', onToggleSidebarClick);
        if (typeof unbindListActions === 'function') unbindListActions();
        if (_adminChatRenameCleanup) _adminChatRenameCleanup();
        _chatRequestSeq++;
    });

    // Chat-Liste laden, dann Draft starten
    _loadChatList().then(() => _startDraftChat());
}

function _bindAdminChatListActions() {
    const listEl = document.getElementById('adminChatList');
    if (!listEl || listEl.dataset.adminChatBound === '1') return null;
    listEl.dataset.adminChatBound = '1';
    const onListClick = function(event) {
        const actionEl = event.target.closest('[data-admin-chat-action]');
        if (!actionEl) return;
        const action = actionEl.getAttribute('data-admin-chat-action');
        const chatId = parseInt(actionEl.getAttribute('data-chat-id') || '', 10);
        if (!Number.isFinite(chatId)) return;
        if (action === 'select-chat') {
            _selectChat(chatId);
        } else if (action === 'rename-chat') {
            event.stopPropagation();
            _renameChat(chatId, event);
        } else if (action === 'delete-chat') {
            event.stopPropagation();
            _deleteChat(chatId, event);
        }
    };
    listEl.addEventListener('click', onListClick);
    return () => {
        listEl.removeEventListener('click', onListClick);
        delete listEl.dataset.adminChatBound;
    };
}

async function _loadChatList() {
    const listEl = document.getElementById('adminChatList');
    if (!listEl) return;

    try {
        const resp = await fetch(`${ADMIN_CHAT_BASE}/api/admin-chat/chats`, {
            credentials: 'include',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (_kiaraAdminChatRedirect401(resp)) return;
        if (!resp.ok) {
            listEl.innerHTML = '<div class="admin-chat-list-empty">Keine Verbindung zum Chat-Service</div>';
            return;
        }
        _adminChats = await resp.json();
        _renderChatList();
    } catch (e) {
        listEl.innerHTML = '<div class="admin-chat-list-empty">Chat-Service nicht erreichbar (Port ' + SharedUi.runtime.escapeHtml(String(ADMIN_CHAT_PORT)) + ')</div>';
        console.error('Admin-Chat: Fehler beim Laden der Chat-Liste:', e);
    }
}

/**
 * Draft-Chat starten: Greeting vom Server holen und lokal anzeigen.
 * Kein DB-Eintrag — erst beim ersten Prompt wird der Chat persistiert.
 */
async function _startDraftChat() {
    const seq = ++_chatRequestSeq;
    if (_adminChatUI) {
        if (_adminChatUI.isStreaming) _adminChatUI.abortStream();
        _adminChatUI.currentChatId = null;
        _adminChatUI.draftGreeting = null;
    }
    _renderChatList();

    try {
        const resp = await fetch(`${ADMIN_CHAT_BASE}/api/admin-chat/greeting`, {
            credentials: 'include',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (_kiaraAdminChatRedirect401(resp)) return;
        if (_chatRequestSeq !== seq) return;
        if (!resp.ok) {
            if (_adminChatUI) _adminChatUI.render([]);
            return;
        }
        const data = await resp.json();
        if (_chatRequestSeq !== seq) return;
        if (_adminChatUI) {
            const metaType = data.type || 'greeting';
            _adminChatUI.draftGreeting = data.greeting;
            _adminChatUI.draftGreetingType = metaType;
            _adminChatUI.render([{
                role: 'assistant',
                content: data.greeting,
                created_at: new Date().toISOString(),
                metadata: JSON.stringify({type: metaType}),
            }]);
        }
    } catch (e) {
        console.error('Admin-Chat: Greeting konnte nicht geladen werden:', e);
        if (_chatRequestSeq === seq && _adminChatUI) _adminChatUI.render([]);
    }
}

/**
 * Chat in DB persistieren (Lazy-Creation).
 * Wird von AdminChatUI.sendMessage() via _onCreateChat aufgerufen.
 */
async function _persistNewChat(greeting, greetingType) {
    try {
        const buildOpts = () => {
            const o = {
                method: 'POST',
                credentials: 'include',
                headers: Object.assign(
                    {},
                    (SharedUi && SharedUi.fetch && SharedUi.fetch.csrfHeaders
                        ? SharedUi.fetch.csrfHeaders()
                        : { 'X-Requested-With': 'XMLHttpRequest' }),
                ),
            };
            if (greeting) {
                o.headers['Content-Type'] = 'application/json';
                o.body = JSON.stringify({ greeting, greeting_type: greetingType || 'greeting' });
            }
            return o;
        };
        let resp = await fetch(`${ADMIN_CHAT_BASE}/api/admin-chat/chats`, buildOpts());
        if (resp.status === 403 && SharedUi && SharedUi.fetch && SharedUi.fetch.refreshCsrfToken) {
            let parsed = null;
            try { parsed = await resp.clone().json(); } catch (_) {}
            if (parsed && parsed.code === 'csrf_invalid') {
                const fresh = await SharedUi.fetch.refreshCsrfToken();
                if (fresh) {
                    resp = await fetch(`${ADMIN_CHAT_BASE}/api/admin-chat/chats`, buildOpts());
                }
            }
        }
        if (_kiaraAdminChatRedirect401(resp)) return null;
        if (!resp.ok) {
            console.error('Admin-Chat: Fehler beim Erstellen eines neuen Chats');
            return null;
        }
        if (!_adminChatUI) return null;
        return await resp.json();
    } catch (e) {
        console.error('Admin-Chat: Fehler beim Erstellen:', e);
        return null;
    }
}

function _renderChatList() {
    const listEl = document.getElementById('adminChatList');
    if (!listEl) return;

    if (_adminChats.length === 0) {
        listEl.innerHTML = '<div class="admin-chat-list-empty">Keine Chats vorhanden</div>';
        return;
    }

    let html = '';
    _adminChats.forEach(chat => {
        const isActive = _adminChatUI && _adminChatUI.currentChatId === chat.id;
        const title = chat.title || 'Neuer Chat';
        const date = _formatChatDate(chat.updated_at || chat.created_at);
        const msgCount = chat.message_count || 0;
        html += `
            <div class="admin-chat-list-item${isActive ? ' active' : ''}" data-chat-id="${SharedUi.runtime.escapeAttr(String(chat.id))}">
                <div class="admin-chat-list-item-content" data-admin-chat-action="select-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(chat.id))}">
                    <div class="admin-chat-list-title">${SharedUi.runtime.escapeHtml(title)}</div>
                    <div class="admin-chat-list-meta">${date} &middot; ${msgCount} Nachr.</div>
                </div>
                <div class="admin-chat-list-actions">
                    <button class="admin-chat-list-rename" data-admin-chat-action="rename-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(chat.id))}" title="Umbenennen">&#x270E;</button>
                    <button class="admin-chat-list-delete" data-admin-chat-action="delete-chat" data-chat-id="${SharedUi.runtime.escapeAttr(String(chat.id))}" title="Chat loeschen">&#x2716;</button>
                </div>
            </div>`;
    });
    listEl.innerHTML = html;
}

function _formatChatDate(dateStr) {
    if (!dateStr) return '';
    const d = new Date(typeof dateStr === 'string' && /^\d{4}-\d{2}-\d{2} /.test(dateStr)
        ? dateStr.replace(' ', 'T') : dateStr);
    if (isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    const today = new Date();
    if (d.toDateString() === today.toDateString()) {
        return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }
    return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.`;
}

async function _selectChat(chatId) {
    if (_adminChatUI && _adminChatUI.currentChatId === chatId) return;
    if (_adminChatUI && _adminChatUI.isStreaming) _adminChatUI.abortStream();
    const seq = ++_chatRequestSeq;

    try {
        const resp = await fetch(`${ADMIN_CHAT_BASE}/api/admin-chat/chats/${chatId}`, {
            credentials: 'include',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (_kiaraAdminChatRedirect401(resp)) return;
        if (_chatRequestSeq !== seq) return;
        if (!resp.ok) return;
        const chat = await resp.json();
        if (_chatRequestSeq !== seq) return;
        _adminChatUI.currentChatId = chat.id;
        _adminChatUI._titleSetForChat = chat.id;
        _adminChatUI.draftGreeting = null;
        _adminChatUI.render(chat.messages || []);
        _adminChatUI.hydrateServerState({
            pending_confirmations: chat.pending_confirmations || [],
            active_wizard: chat.active_wizard || null,
        });
        _renderChatList();
    } catch (e) {
        console.error('Admin-Chat: Fehler beim Laden des Chats:', e);
    }
}

async function _deleteChat(chatId, event) {
    if (event) event.stopPropagation();
    if (!confirm('Chat wirklich loeschen?')) return;

    try {
        const deleteUrl = `${ADMIN_CHAT_BASE}/api/admin-chat/chats/${chatId}`;
        const buildDeleteOpts = () => ({
            method: 'DELETE',
            credentials: 'include',
            headers: Object.assign(
                {},
                (SharedUi && SharedUi.fetch && SharedUi.fetch.csrfHeaders
                    ? SharedUi.fetch.csrfHeaders()
                    : { 'X-Requested-With': 'XMLHttpRequest' }),
            ),
        });
        let resp = await fetch(deleteUrl, buildDeleteOpts());
        if (resp.status === 403 && SharedUi && SharedUi.fetch && SharedUi.fetch.refreshCsrfToken) {
            let parsed = null;
            try { parsed = await resp.clone().json(); } catch (_) {}
            if (parsed && parsed.code === 'csrf_invalid') {
                const fresh = await SharedUi.fetch.refreshCsrfToken();
                if (fresh) {
                    resp = await fetch(deleteUrl, buildDeleteOpts());
                }
            }
        }
        if (_kiaraAdminChatRedirect401(resp)) return;
        if (!resp.ok && resp.status !== 204) {
            SharedUi.notify.show('Chat konnte nicht geloescht werden', 'error');
            return;
        }

        const wasActive = _adminChatUI && _adminChatUI.currentChatId === chatId;
        await _loadChatList();

        if (wasActive) {
            await _startDraftChat();
        }
    } catch (e) {
        console.error('Admin-Chat: Fehler beim Loeschen:', e);
        SharedUi.notify.show('Chat konnte nicht geloescht werden', 'error');
    }
}

function _renameChat(chatId, event) {
    if (event) event.stopPropagation();
    if (_adminChatRenameCleanup) _adminChatRenameCleanup();
    const item = document.querySelector(`.admin-chat-list-item[data-chat-id="${CSS.escape(String(chatId))}"]`);
    if (!item) return;
    const titleEl = item.querySelector('.admin-chat-list-title');
    if (!titleEl || titleEl.dataset.editing) return;

    const oldTitle = titleEl.textContent;
    titleEl.dataset.editing = '1';
    titleEl.innerHTML = `<input class="admin-chat-rename-input" type="text" value="${SharedUi.runtime.escapeAttr(oldTitle)}">`;
    const input = titleEl.querySelector('input');
    input.focus();
    input.select();

    const save = async () => {
        if (_adminChatRenameCleanup) _adminChatRenameCleanup();
        const newTitle = input.value.trim();
        delete titleEl.dataset.editing;
        if (!newTitle || newTitle === oldTitle) { _renderChatList(); return; }
        try {
            const renameUrl = `${ADMIN_CHAT_BASE}/api/admin-chat/chats/${chatId}`;
            const buildRenameOpts = () => ({
                method: 'PUT',
                credentials: 'include',
                headers: Object.assign(
                    { 'Content-Type': 'application/json' },
                    (SharedUi && SharedUi.fetch && SharedUi.fetch.csrfHeaders
                        ? SharedUi.fetch.csrfHeaders()
                        : { 'X-Requested-With': 'XMLHttpRequest' }),
                ),
                body: JSON.stringify({ title: newTitle.substring(0, 200) }),
            });
            let resp = await fetch(renameUrl, buildRenameOpts());
            if (resp.status === 403 && SharedUi && SharedUi.fetch && SharedUi.fetch.refreshCsrfToken) {
                let parsed = null;
                try { parsed = await resp.clone().json(); } catch (_) {}
                if (parsed && parsed.code === 'csrf_invalid') {
                    const fresh = await SharedUi.fetch.refreshCsrfToken();
                    if (fresh) {
                        resp = await fetch(renameUrl, buildRenameOpts());
                    }
                }
            }
            if (_kiaraAdminChatRedirect401(resp)) return;
            if (!resp.ok) {
                SharedUi.notify.show('Umbenennen fehlgeschlagen', 'error');
                _renderChatList();
                return;
            }
            // Manuell umbenannt — Flag persistent setzen fuer Auto-Titel-Schutz
            // (ueberlebt _loadChatList, das _adminChats komplett ersetzt)
            _manualTitleChats.add(chatId);
            if (_adminChatUI) _adminChatUI._titleSetForChat = chatId;
            await _loadChatList();
        } catch (e) { console.error('Rename fehlgeschlagen:', e); _renderChatList(); }
    };

    let saved = false;
    const onRenameKeydown = (e) => {
        if (e.key === 'Enter') { e.preventDefault(); saved = true; save(); }
        if (e.key === 'Escape') {
            saved = true;
            if (_adminChatRenameCleanup) _adminChatRenameCleanup();
            _renderChatList();
        }
    };
    const onRenameBlur = () => { if (!saved) save(); };
    _adminChatRenameCleanup = () => {
        input.removeEventListener('keydown', onRenameKeydown);
        input.removeEventListener('blur', onRenameBlur);
        _adminChatRenameCleanup = null;
    };
    input.addEventListener('keydown', onRenameKeydown);
    input.addEventListener('blur', onRenameBlur);
}

// Globale Funktion fuer Sidebar-Refresh (wird von AdminChatUI aufgerufen)
function refreshAdminChatSidebar() {
    _loadChatList();
}

// initKiaraTab wird von admin.html callInitFunction() aufgerufen
function initKiaraTab() {
    if (typeof initAdminChat === 'function') {
        initAdminChat();
    }
}
