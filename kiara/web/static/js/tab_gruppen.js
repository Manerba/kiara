/**
 * KIara Admin - Gruppen Tab
 * Importierte LDAP-Gruppen verwalten, Mitgliedschaften synchronisieren.
 */

let gruppenTable = null;

const _GRUPPEN_ACTIONS = new Set([
    'sync-group-members',
    'open-group-import-modal',
    'delete-group',
    'toggle-all-import-groups',
]);

const _GRUPPEN_ROW_ACTIONS = new Set(['delete-group']);
const _GRUPPEN_INPUT_ACTIONS = new Set(['update-gruppen-search']);

function _handleGruppenClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_GRUPPEN_ACTIONS.has(action)) return;

    if (_GRUPPEN_ROW_ACTIONS.has(action)) e.stopPropagation();

    switch (action) {
        case 'sync-group-members':
            syncGroupMembers(el);
            break;
        case 'open-group-import-modal':
            openGroupImportModal();
            break;
        case 'delete-group':
            deleteGroup(Number(el.dataset.groupId), el.dataset.groupName || '');
            break;
        case 'toggle-all-import-groups':
            toggleAllImportGroups();
            break;
    }
}

function _handleGruppenInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el || !_GRUPPEN_INPUT_ACTIONS.has(el.dataset.inputAction)) return;
    if (el.dataset.inputAction === 'update-gruppen-search' && gruppenTable) {
        gruppenTable.setFilter('search', el.value);
    }
}

function _bindGruppenActions() {
    if (document.body._gruppenActionsBound) return;
    document.body._gruppenActionsBound = true;
    document.body.addEventListener('click', _handleGruppenClick);
    document.body.addEventListener('input', _handleGruppenInput);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-gruppen-actions-bound', 'true');
}

async function initGruppenTab() {
    _bindGruppenActions();
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Filter bar: search + sync + import
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.style.display = '';
        filterBar.classList.remove('is-hidden', 'u-hidden');
        filterBar.classList.add('u-flex');

        const searchGroup = document.createElement('div');
        searchGroup.className = 'filter-group';
        searchGroup.innerHTML = `<span class="filter-label">Suche:</span>
            <input type="text" class="filter-input" id="gruppenSearch" placeholder="Gruppenname/DN..."
                data-input-action="update-gruppen-search">`;
        filterBar.appendChild(searchGroup);

        const btnGroup = document.createElement('div');
        btnGroup.className = 'filter-group u-ml-auto u-flex-gap-8';
        btnGroup.innerHTML = `
            <button class="action-btn" data-action="sync-group-members">Mitglieder synchronisieren</button>
            <button class="action-btn primary" data-action="open-group-import-modal">Gruppen importieren</button>`;
        filterBar.appendChild(btnGroup);
    }

    container.innerHTML = '<div id="gruppenTableContainer"></div>';

    const config = {
        id: 'gruppen',
        apiEndpoint: '/api/admin/groups',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px 180px 1fr 80px 100px 80px',
        defaultSort: { field: 'cn', direction: 'asc' },
        columns: [
            { field: 'cn', label: 'Gruppe', sortable: true, align: 'left' },
            { field: 'dn', label: 'DN', sortable: false, align: 'left', truncate: 50 },
            { field: 'member_count', label: 'Mitglieder', sortable: true, align: 'center' },
            { field: 'datasource_count', label: 'Datenquellen', sortable: true, align: 'center' },
            { field: 'id', label: 'Aktionen', sortable: false, align: 'center', renderer: 'actions' },
        ],
        detailFields: [
            { field: 'dn', label: 'Vollstaendiger DN' },
            { field: 'imported_at', label: 'Importiert am' },
            { field: 'updated_at', label: 'Aktualisiert am' },
            { field: 'user_count', label: 'Zugewiesene Benutzer' },
        ],
        filters: [
            { id: 'search', field: 'cn', wildcard: true, wildcardFields: ['dn'], type: 'text' },
        ],
    };

    gruppenTable = new SharedUi.table.ExpandableTable(config, 'gruppenTableContainer', {
        onRowExpanded: async (rowId, detailEl) => {
            // Mitglieder-Container finden oder erstellen
            let memberSection = detailEl.querySelector('.group-members-section');
            if (!memberSection) {
                memberSection = document.createElement('div');
                memberSection.className = 'group-members-section gruppen-members-section';
                const content = detailEl.querySelector('.detail-content');
                if (content) content.appendChild(memberSection);
                else detailEl.appendChild(memberSection);
            }
            memberSection.innerHTML = '<div class="detail-section-label">Mitglieder</div><div class="table-loading"><div class="spinner"></div></div>';

            try {
                const resp = await SharedUi.fetch.safeFetch(`/api/admin/groups/${rowId}/members`);
                if (!resp.ok) throw new Error('Fehler beim Laden');
                const data = await resp.json();
                const members = data.members || [];

                if (members.length === 0) {
                    memberSection.innerHTML = '<div class="detail-section-label">Mitglieder</div><div class="u-text-sm-secondary">Keine Mitglieder zugewiesen</div>';
                    return;
                }

                let html = `<div class="detail-section-label">Mitglieder (${members.length})</div>`;
                html += '<div class="gruppen-members-grid">';
                members.forEach(m => {
                    const name = [m.vorname, m.nachname].filter(Boolean).join(' ');
                    const displayName = name ? `${SharedUi.runtime.escapeHtml(m.username)} (${SharedUi.runtime.escapeHtml(name)})` : SharedUi.runtime.escapeHtml(m.username);
                    const source = m.auth_source === 'ldap' ? 'LDAP' : 'Lokal';
                    const inactive = m.is_active ? '' : ' class="u-dim"';
                    const tag = m.is_active ? '' : ' <span class="u-text-xxs-secondary">(inaktiv)</span>';
                    html += `<span${inactive}>${displayName}${tag}</span>`;
                    html += `<span class="u-c-secondary">${SharedUi.runtime.escapeHtml(m.email || '')}</span>`;
                    html += `<span class="u-text-xxs-secondary">${source}</span>`;
                });
                html += '</div>';
                memberSection.innerHTML = html;
            } catch (e) {
                memberSection.innerHTML = '<div class="detail-section-label">Mitglieder</div><div class="table-error">Fehler beim Laden der Mitglieder</div>';
            }
        },
    });

    gruppenTable.renderers['actions'] = (value, col, row) => {
        let html = '<div class="action-buttons">';
        html += `<button class="action-btn-sm danger" title="Entfernen" data-action="delete-group" data-group-id="${row.id}" data-group-name="${SharedUi.runtime.escapeAttr(row.cn)}">&#x2716;</button>`;
        html += '</div>';
        return html;
    };

    gruppenTable.loadData();
}

async function deleteGroup(groupId, name) {
    if (!await SharedUi.modal.confirm({ message: `Gruppe "${name}" wirklich entfernen?` })) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/groups/${groupId}`, { method: 'DELETE' });
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
        SharedUi.notify.show('Gruppe entfernt', 'success');
        if (gruppenTable) gruppenTable.loadData();
    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
}

async function syncGroupMembers(btn) {
    await SharedUi.runtime.withButtonLock(btn, async () => {
        SharedUi.notify.show('Synchronisiere Mitgliedschaften...', 'info');
        try {
            const resp = await SharedUi.fetch.safeFetch('/api/admin/groups/sync-members', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            const result = await resp.json();
            if (!resp.ok) throw new Error(result.detail || result.error || result.message || 'Fehler');
            const stats = result.stats || {};
            SharedUi.notify.show(
                `Synchronisiert: ${stats.synced || 0} Benutzer, ${stats.errors || 0} Fehler`,
                stats.errors ? 'warning' : 'success'
            );
            if (gruppenTable) gruppenTable.loadData();
        } catch (e) { SharedUi.notify.show(e.message, 'error'); }
    });
}

// ---------------------------------------------------------------------------
// Import Modal
// ---------------------------------------------------------------------------

async function openGroupImportModal() {
    let body = '<div class="modal-field"><label>Base DN (optional)</label>';
    body += '<input type="text" id="importBaseDn" placeholder="Standard: LDAP Base DN"></div>';
    body += '<div class="u-my-12-0"><button class="action-btn" id="loadLdapGroupsBtn">Gruppen laden</button></div>';
    body += '<div id="ldapGroupList"></div>';

    SharedUi.modal.open({
        title: 'LDAP-Gruppen importieren',
        body: body,
        maxWidth: '700px',
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn primary" id="importGroupsBtn" disabled>Importieren</button>`,
        onOpen: () => {
            document.getElementById('loadLdapGroupsBtn').addEventListener('click', function() {
                SharedUi.runtime.withButtonLock(this, loadLdapGroupsForImport);
            });
            document.getElementById('importGroupsBtn').addEventListener('click', function() {
                SharedUi.runtime.withButtonLock(this, importSelectedGroups);
            });
        },
    });
}

async function loadLdapGroupsForImport() {
    const listEl = document.getElementById('ldapGroupList');
    const btn = document.getElementById('loadLdapGroupsBtn');
    if (!listEl) return;

    btn.textContent = 'Lade...';
    btn.disabled = true;
    listEl.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    const baseDn = document.getElementById('importBaseDn')?.value || '';

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/groups/ldap?base_dn=${encodeURIComponent(baseDn)}`);
        if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Fehler'); }
        const data = await resp.json();
        const groups = data.groups || [];

        if (groups.length === 0) {
            listEl.innerHTML = '<div class="no-data">Keine Gruppen gefunden</div>';
            btn.textContent = 'Gruppen laden';
            btn.disabled = false;
            return;
        }

        let html = `<div class="u-text-sm-secondary u-mb-8">${groups.length} Gruppen gefunden</div>`;
        html += '<div class="u-mb-8"><button class="action-btn-sm" data-action="toggle-all-import-groups">Alle auswaehlen</button></div>';
        html += '<div class="gruppen-import-grid">';
        groups.forEach(g => {
            const disabled = g.imported ? 'disabled' : '';
            const importedTag = g.imported ? ' <span class="u-text-xxs-secondary">(importiert)</span>' : '';
            html += `<input type="checkbox" class="import-group-cb" value="${SharedUi.runtime.escapeAttr(g.dn)}" ${disabled}>`;
            html += `<span class="u-nowrap">${SharedUi.runtime.escapeHtml(g.cn)}${importedTag}</span>`;
            html += `<span class="u-text-xxs-secondary u-nowrap u-text-right">${g.member_count} Mitglieder</span>`;
        });
        html += '</div>';
        listEl.innerHTML = html;

        // Enable import button
        const importBtn = document.getElementById('importGroupsBtn');
        if (importBtn) importBtn.disabled = false;
    } catch (e) {
        listEl.innerHTML = `<div class="table-error">${SharedUi.runtime.escapeHtml(e.message)}</div>`;
    }

    btn.textContent = 'Gruppen laden';
    btn.disabled = false;
}

function toggleAllImportGroups() {
    const checkboxes = document.querySelectorAll('.import-group-cb:not(:disabled)');
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
}

async function importSelectedGroups() {
    const checkboxes = document.querySelectorAll('.import-group-cb:checked');
    const group_dns = Array.from(checkboxes).map(cb => cb.value);
    const baseDn = document.getElementById('importBaseDn')?.value || '';

    if (group_dns.length === 0) {
        SharedUi.notify.show('Keine Gruppen ausgewaehlt', 'error');
        return;
    }

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/groups/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_dn: baseDn, group_dns }),
        });
        const result = await resp.json();
        if (!resp.ok) throw new Error(result.detail || result.error || result.message || 'Fehler');
        SharedUi.notify.show(result.message || 'Gruppen importiert', 'success');
        SharedUi.modal.close();
        if (gruppenTable) gruppenTable.loadData();
    } catch (e) { SharedUi.notify.show(e.message, 'error'); }
}
