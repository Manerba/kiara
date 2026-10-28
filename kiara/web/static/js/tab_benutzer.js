/**
 * KIara Admin - Benutzer Tab
 * User management with ExpandableTable.
 */

let userTable = null;

const _USER_ACTIONS = new Set([
    'open-create-user-modal',
    'open-edit-user-modal',
    'delete-user',
]);

const _USER_ROW_ACTIONS = new Set([
    'open-edit-user-modal',
    'delete-user',
]);

const _USER_CHANGE_ACTIONS = new Set(['toggle-user-field']);
const _USER_INPUT_ACTIONS = new Set(['update-user-search']);

function _handleUserClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_USER_ACTIONS.has(action)) return;

    if (_USER_ROW_ACTIONS.has(action)) e.stopPropagation();

    switch (action) {
        case 'open-create-user-modal':
            openCreateUserModal();
            break;
        case 'open-edit-user-modal':
            openEditUserModal(Number(el.dataset.userId));
            break;
        case 'delete-user':
            deleteUser(Number(el.dataset.userId), el.dataset.username || '');
            break;
    }
}

function _handleUserChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_USER_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    if (el.dataset.changeAction === 'toggle-user-field') {
        _toggleUserField(Number(el.dataset.userId), el.dataset.field || '', el.checked);
    }
}

function _handleUserInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el || !_USER_INPUT_ACTIONS.has(el.dataset.inputAction)) return;
    if (el.dataset.inputAction === 'update-user-search' && userTable) {
        userTable.setFilter('search', el.value);
    }
}

function _bindUserActions() {
    if (document.body._userActionsBound) return;
    document.body._userActionsBound = true;
    document.body.addEventListener('click', _handleUserClick);
    document.body.addEventListener('change', _handleUserChange);
    document.body.addEventListener('input', _handleUserInput);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-user-actions-bound', 'true');
}

async function initBenutzerTab() {
    _bindUserActions();
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.remove('u-hidden');
        filterBar.classList.add('u-flex');

        const searchGroup = document.createElement('div');
        searchGroup.className = 'filter-group';
        searchGroup.innerHTML = `<span class="filter-label">Suche:</span>
            <input type="text" class="filter-input" id="userSearch" placeholder="Name/Benutzername..."
                data-input-action="update-user-search">`;
        filterBar.appendChild(searchGroup);

        const addGroup = document.createElement('div');
        addGroup.className = 'filter-group u-ml-auto';
        addGroup.innerHTML = `<button class="action-btn primary" data-action="open-create-user-modal">Neuer Benutzer</button>`;
        filterBar.appendChild(addGroup);
    }

    container.innerHTML = '<div id="userTableContainer"></div>';

    const config = {
        id: 'users',
        apiEndpoint: '/api/admin/users',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 150px 150px 200px 80px 80px 80px 100px',
        defaultSort: { field: 'username', direction: 'asc' },
        columns: [
            { field: 'username', label: 'Benutzer', sortable: true, align: 'left' },
            { field: 'display_name', label: 'Name', sortable: true, align: 'left' },
            { field: 'email', label: 'E-Mail', sortable: true, align: 'left', truncate: 30 },
            { field: 'is_admin', label: 'Admin', sortable: true, align: 'center', renderer: 'toggle' },
            { field: 'auth_source', label: 'Quelle', sortable: true, align: 'center', renderer: 'badge' },
            { field: 'is_active', label: 'Aktiv', sortable: true, align: 'center', renderer: 'toggle' },
            { field: 'id', label: 'Aktionen', sortable: false, align: 'center', renderer: 'actions' },
        ],
        detailFields: [
            { field: 'created_at', label: 'Erstellt am' },
            { field: 'ldap_dn', label: 'LDAP DN' },
        ],
        filters: [
            { id: 'search', field: 'username', wildcard: true, wildcardFields: ['vorname', 'nachname', 'display_name', 'email'], type: 'text' },
        ],
    };

    userTable = new SharedUi.table.ExpandableTable(config, 'userTableContainer', {
        onDataLoaded: () => {
            userTable.data.forEach(u => {
                u.display_name = [u.vorname, u.nachname].filter(Boolean).join(' ') || '-';
            });
        },
    });

    // Custom actions renderer
    userTable.renderers['actions'] = (value, col, row) => {
        let html = '<div class="action-buttons">';
        html += `<button class="action-btn-sm" title="Bearbeiten" data-action="open-edit-user-modal" data-user-id="${row.id}">&#x270E;</button>`;
        html += `<button class="action-btn-sm danger" title="Loeschen" data-action="delete-user" data-user-id="${row.id}" data-username="${SharedUi.runtime.escapeAttr(row.username)}">&#x2716;</button>`;
        html += '</div>';
        return html;
    };

    userTable.renderers['toggle'] = (value, col, row) => {
        // Label-onclick=stopPropagation obsolet: ui_core/js/table.js
        // _interactiveCellMatch haelt Row-Expand bei Klick auf Input/Label
        // inzwischen sauber ab (Fix 0.0.1+380).
        return `<label class="toggle-switch">
            <input type="checkbox" ${value ? 'checked' : ''}
                data-change-action="toggle-user-field"
                data-user-id="${row.id}" data-field="${SharedUi.runtime.escapeAttr(col.field)}">
            <span class="toggle-slider"></span></label>`;
    };

    userTable.renderers['badge'] = (value) => {
        const cls = value === 'ldap' ? 'info' : 'default';
        return `<span class="badge badge-${cls}">${SharedUi.runtime.escapeHtml(value || 'local')}</span>`;
    };

    userTable.loadData();
}

function openCreateUserModal() {
    SharedUi.modal.open({
        title: 'Neuer Benutzer',
        body: `
            <div class="modal-field"><label>Benutzername *</label><input type="text" id="newUsername"></div>
            <div class="modal-field"><label>Passwort</label><input type="password" id="newPassword" placeholder="Mindestens 8 Zeichen"></div>
            <div class="modal-field"><label>Vorname</label><input type="text" id="newVorname"></div>
            <div class="modal-field"><label>Nachname</label><input type="text" id="newNachname"></div>
            <div class="modal-field"><label>E-Mail</label><input type="email" id="newEmail"></div>
            <div class="modal-field"><label><input type="checkbox" id="newIsAdmin"> Administrator</label></div>`,
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn primary" id="createUserBtn">Erstellen</button>`,
        onOpen: () => {
            document.getElementById('newUsername').focus();
            document.getElementById('createUserBtn').addEventListener('click', function () {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const username = document.getElementById('newUsername').value.trim();
                    if (!username) { SharedUi.notify.show('Benutzername erforderlich', 'error'); return; }
                    const password = document.getElementById('newPassword').value || '';
                    if (password && password.length < 8) { SharedUi.notify.show('Passwort muss mindestens 8 Zeichen haben', 'error'); return; }
                    try {
                        const resp = await SharedUi.fetch.safeFetch('/api/admin/users', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                username,
                                password,
                                vorname: document.getElementById('newVorname').value,
                                nachname: document.getElementById('newNachname').value,
                                email: document.getElementById('newEmail').value,
                                is_admin: document.getElementById('newIsAdmin').checked,
                            }),
                        });
                        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
                        SharedUi.notify.show('Benutzer erstellt', 'success');
                        SharedUi.modal.close();
                        if (userTable) userTable.loadData();
                    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
                });
            });
        },
    });
}

function openEditUserModal(userId) {
    const user = userTable ? userTable.getAllData().find(u => u.id === userId) : null;
    if (!user) return;

    SharedUi.modal.open({
        title: `Benutzer bearbeiten: ${SharedUi.runtime.escapeHtml(user.username)}`,
        body: `
            <div class="modal-field"><label>Vorname</label><input type="text" id="editVorname" value="${SharedUi.runtime.escapeAttr(user.vorname || '')}"></div>
            <div class="modal-field"><label>Nachname</label><input type="text" id="editNachname" value="${SharedUi.runtime.escapeAttr(user.nachname || '')}"></div>
            <div class="modal-field"><label>E-Mail</label><input type="email" id="editEmail" value="${SharedUi.runtime.escapeAttr(user.email || '')}"></div>
            <div class="modal-field"><label>Neues Passwort</label><input type="password" id="editPassword" placeholder="Leer lassen = nicht aendern"></div>
            <div class="modal-field"><label><input type="checkbox" id="editIsAdmin" ${user.is_admin ? 'checked' : ''}> Administrator</label></div>
            <div class="modal-field"><label><input type="checkbox" id="editIsActive" ${user.is_active ? 'checked' : ''}> Aktiv</label></div>`,
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn primary" id="saveUserBtn">Speichern</button>`,
        onOpen: () => {
            document.getElementById('saveUserBtn').addEventListener('click', function () {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const data = {
                        vorname: document.getElementById('editVorname').value,
                        nachname: document.getElementById('editNachname').value,
                        email: document.getElementById('editEmail').value,
                        is_admin: document.getElementById('editIsAdmin').checked,
                        is_active: document.getElementById('editIsActive').checked,
                    };
                    const pw = document.getElementById('editPassword').value;
                    if (pw) data.password = pw;
                    try {
                        const resp = await SharedUi.fetch.safeFetch(`/api/admin/users/${userId}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(data),
                        });
                        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
                        SharedUi.notify.show('Benutzer aktualisiert', 'success');
                        SharedUi.modal.close();
                        if (userTable) userTable.loadData();
                    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
                });
            });
        },
    });
}

async function _toggleUserField(userId, field, value) {
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/users/${userId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ [field]: value }),
        });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
        SharedUi.notify.show(`${field === 'is_active' ? 'Status' : 'Admin-Rolle'} aktualisiert`, 'success');
        if (userTable) userTable.loadData();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
        if (userTable) userTable.loadData();
    }
}

async function deleteUser(userId, username) {
    if (!await SharedUi.modal.confirm({ message: `Benutzer "${username}" wirklich loeschen?` })) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/users/${userId}`, { method: 'DELETE' });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
        SharedUi.notify.show('Benutzer geloescht', 'success');
        if (userTable) userTable.loadData();
    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
}
