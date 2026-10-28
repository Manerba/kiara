/**
 * KIara Admin - LDAP Tab
 * LDAP-Konfiguration und AD-Gruppen-Verwaltung (Tareas-Pattern).
 */

let ldapConfig = null;
let _ldapRequestSeq = 0;

const _LDAP_ACTIONS = new Set([
    'save-ldap-config',
    'delete-ldap-config',
    'deselect-ldap-group',
    'manual-ldap-sync',
    'load-ldap-groups',
    'select-ldap-group',
]);

function _handleLdapClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_LDAP_ACTIONS.has(action)) return;

    switch (action) {
        case 'save-ldap-config':
            SharedUi.runtime.withButtonLock(el, saveLdapConfig);
            break;
        case 'delete-ldap-config':
            deleteLdapConfig();
            break;
        case 'deselect-ldap-group':
            deselectLdapGroup();
            break;
        case 'manual-ldap-sync':
            manualLdapSync();
            break;
        case 'load-ldap-groups':
            loadLdapGroups();
            break;
        case 'select-ldap-group':
            selectLdapGroup(el.dataset.groupDn || '', el.dataset.groupName || '');
            break;
    }
}

function _bindLdapActions() {
    if (document.body._ldapActionsBound) return;
    document.body._ldapActionsBound = true;
    document.body.addEventListener('click', _handleLdapClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-ldap-actions-bound', 'true');
}

// ========================================
// Tab Initialisierung
// ========================================

async function initLdapTab() {
    _bindLdapActions();

    if (typeof _runTabCleanups === 'function') _runTabCleanups('benutzer-ldap');
    const seq = ++_ldapRequestSeq;
    if (typeof registerTabCleanup === 'function') {
        registerTabCleanup('benutzer-ldap', () => { _ldapRequestSeq++; });
    }

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('is-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    await loadLdapConfig();

    if (seq !== _ldapRequestSeq) return;
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'benutzer-ldap') return;

    renderLdapTab();
}

async function loadLdapConfig() {
    ldapConfig = await loadConfigFromAPI('/api/admin/ldap/config');
}

// ========================================
// Rendering
// ========================================

function renderLdapTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    const hasConfig = !!ldapConfig;
    const hasGroup = hasConfig && !!ldapConfig.group_dn;

    let html = '<div class="config-container">';

    // Sektion 1: LDAP-Server Konfiguration
    html += renderLdapConfigSection(hasConfig);

    // Sektion 2: AD-Gruppe (nur wenn Config vorhanden)
    if (hasConfig) {
        html += renderLdapGroupSection(hasGroup);
    }

    html += '</div>';
    container.innerHTML = html;

    // SSL-Checkbox Event
    const sslCheckbox = document.getElementById('ldapUseSSL');
    if (sslCheckbox) {
        sslCheckbox.addEventListener('change', function() {
            const portInput = document.getElementById('ldapPort');
            if (portInput) portInput.value = this.checked ? 636 : 389;
        });
    }
}

function renderLdapConfigSection(hasConfig) {
    const c = ldapConfig || {};
    let html = `
        <div class="config-section">
            <h3>LDAP-Konfiguration</h3>
            <div class="ldap-form">
                <div class="config-form-row">
                    <div class="config-form-field">
                        <label>Server</label>
                        <input type="text" id="ldapServer" value="${SharedUi.runtime.escapeAttr(c.server_url || '')}"
                               placeholder="192.168.1.10 oder dc.firma.local">
                    </div>
                    <div class="config-form-field ldap-field-port">
                        <label>Port</label>
                        <input type="number" id="ldapPort" value="${c.port || 389}">
                    </div>
                    <div class="config-form-field ldap-field-ssl">
                        <label>SSL</label>
                        <label class="config-checkbox">
                            <input type="checkbox" id="ldapUseSSL" ${c.use_ssl ? 'checked' : ''}>
                            SSL
                        </label>
                    </div>
                </div>
                <div class="config-form-row">
                    <div class="config-form-field">
                        <label>Bind DN</label>
                        <input type="text" id="ldapBindDN" value="${SharedUi.runtime.escapeAttr(c.bind_dn || '')}"
                               placeholder="CN=ldapreader,OU=Service,DC=firma,DC=local">
                    </div>
                </div>
                <div class="config-form-row">
                    <div class="config-form-field">
                        <label>Bind Passwort</label>
                        <input type="password" id="ldapBindPassword"
                               value=""
                               placeholder="${hasConfig ? '(unveraendert wenn leer)' : 'Passwort...'}" autocomplete="new-password">
                    </div>
                </div>
                <div class="config-form-row">
                    <div class="config-form-field">
                        <label>Suchpfad (Search Base)</label>
                        <input type="text" id="ldapSearchBase" value="${SharedUi.runtime.escapeAttr(c.search_base || '')}"
                               placeholder="OU=Gruppen,DC=firma,DC=local">
                    </div>
                </div>
                <div class="config-form-actions">
                    <button class="action-btn primary" data-action="save-ldap-config">Testen und Speichern</button>
                    ${hasConfig ? '<button class="action-btn danger" data-action="delete-ldap-config">Loeschen</button>' : ''}
                </div>
            </div>`;

    // Status-Anzeige
    if (hasConfig) {
        const ssl = c.use_ssl ? 'Ja' : 'Nein';
        const lastSync = c.last_sync_at || 'Nie';
        html += `
            <div class="config-status-info">
                <strong>Aktive Konfiguration:</strong>
                ${SharedUi.runtime.escapeHtml(c.server_url)}:${c.port} | SSL: ${ssl} | Suchpfad: ${SharedUi.runtime.escapeHtml(c.search_base)} | Letzter Sync: ${SharedUi.runtime.escapeHtml(lastSync)}
            </div>`;
    }

    html += '</div>';
    return html;
}

function renderLdapGroupSection(hasGroup) {
    let html = '<div class="config-section">';
    html += '<h3>AD-Gruppe</h3>';

    if (hasGroup) {
        // Gruppe ist gewaehlt
        html += `
            <div class="ldap-group-selected">
                <div class="ldap-group-info">
                    <span class="badge badge-ldap">Ausgewaehlt</span>
                    <strong>${SharedUi.runtime.escapeHtml(ldapConfig.group_name)}</strong>
                    <span class="ldap-group-dn">${SharedUi.runtime.escapeHtml(ldapConfig.group_dn)}</span>
                </div>
                <button class="action-btn danger" data-action="deselect-ldap-group">Aufheben</button>
            </div>
            <div class="ldap-sync-bar">
                <button class="action-btn primary" data-action="manual-ldap-sync">Jetzt synchronisieren</button>
                <span class="ldap-last-sync">Letzter Sync: ${SharedUi.runtime.escapeHtml(ldapConfig.last_sync_at || 'Nie')}</span>
            </div>
            <div class="ldap-hint">Mitglieder dieser Gruppe werden als Benutzer importiert und koennen sich anmelden, sofern sie in der Benutzerverwaltung als aktiv markiert sind.</div>`;
    } else {
        // Gruppen-Liste laden
        html += `
            <div id="ldapGroupsList">
                <button class="action-btn" data-action="load-ldap-groups">Gruppen laden</button>
            </div>`;
    }

    html += '</div>';
    return html;
}

// ========================================
// LDAP Config CRUD
// ========================================

async function saveLdapConfig() {
    const server_url = document.getElementById('ldapServer')?.value?.trim();
    const port = parseInt(document.getElementById('ldapPort')?.value) || 389;
    const use_ssl = document.getElementById('ldapUseSSL')?.checked ? 1 : 0;
    const bind_dn = document.getElementById('ldapBindDN')?.value?.trim();
    const bind_password = document.getElementById('ldapBindPassword')?.value?.trim();
    const search_base = document.getElementById('ldapSearchBase')?.value?.trim();

    if (!server_url || !bind_dn || !search_base) {
        SharedUi.notify.show('Bitte alle Pflichtfelder ausfuellen', 'error');
        return;
    }

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ldap/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ server_url, port, use_ssl, bind_dn, bind_password, search_base }),
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || errData.message || 'Speichern fehlgeschlagen');
        }
        await loadLdapConfig();
        renderLdapTab();
        SharedUi.notify.show('LDAP-Konfiguration gespeichert', 'success');
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
    }
}

async function deleteLdapConfig() {
    await deleteConfigFromAPI({
        endpoint: '/api/admin/ldap/config',
        confirmMessage: 'LDAP-Konfiguration loeschen? Alle LDAP-Benutzer werden deaktiviert.',
        successMessage: 'LDAP-Konfiguration geloescht',
        onSuccess: () => {
            ldapConfig = null;
            renderLdapTab();
        },
    });
}

// ========================================
// Gruppen
// ========================================

async function loadLdapGroups() {
    const container = document.getElementById('ldapGroupsList');
    if (!container) return;

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ldap/groups');
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || 'Gruppen laden fehlgeschlagen');
        }

        const data = await resp.json();
        const groups = data.groups || [];

        if (groups.length === 0) {
            container.innerHTML = '<p class="ldap-hint">Keine Gruppen im Suchpfad gefunden.</p>';
            return;
        }

        let html = `
            <table class="ldap-groups-table">
                <thead>
                    <tr><th>Gruppenname</th><th>DN</th><th class="ldap-th-action"></th></tr>
                </thead>
                <tbody>`;

        for (const group of groups) {
            html += `
                <tr>
                    <td>${SharedUi.runtime.escapeHtml(group.name)}</td>
                    <td class="ldap-group-dn">${SharedUi.runtime.escapeHtml(group.dn)}</td>
                    <td><button class="action-btn primary ldap-btn-add"
                            data-group-dn="${SharedUi.runtime.escapeAttr(group.dn)}" data-group-name="${SharedUi.runtime.escapeAttr(group.name)}"
                            data-action="select-ldap-group">+</button></td>
                </tr>`;
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = '<button class="action-btn" data-action="load-ldap-groups">Gruppen laden</button>';
        SharedUi.notify.show(error.message, 'error');
    }
}

async function selectLdapGroup(dn, name) {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ldap/group', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ group_dn: dn, group_name: name }),
        });

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || 'Fehler');
        }

        const data = await resp.json();
        const sync = data.sync || {};
        SharedUi.notify.show(
            `Gruppe '${name}' ausgewaehlt: ${sync.created || 0} erstellt, ${sync.updated || 0} aktualisiert`,
            'success'
        );

        await loadLdapConfig();
        renderLdapTab();
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
    }
}

async function deselectLdapGroup() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ldap/group', { method: 'DELETE' });
        if (!resp.ok) throw new Error('Fehler');

        SharedUi.notify.show('Gruppenauswahl aufgehoben', 'success');
        await loadLdapConfig();
        renderLdapTab();
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
    }
}

// ========================================
// Sync
// ========================================

async function manualLdapSync() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ldap/sync', { method: 'POST' });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || 'Synchronisierung fehlgeschlagen');
        }

        const data = await resp.json();
        const sync = data.sync || {};
        SharedUi.notify.show(
            `Synchronisiert: ${sync.created || 0} erstellt, ${sync.updated || 0} aktualisiert, ${sync.deactivated || 0} deaktiviert`,
            'success'
        );

        await loadLdapConfig();
        renderLdapTab();
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
    }
}
