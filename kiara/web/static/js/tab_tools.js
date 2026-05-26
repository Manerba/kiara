/**
 * KIara Admin - Tools > ERP-Tools Sub-Tab
 */

var _erpConfig = {};
var _connections = [];
var _erpConnectorTypes = [];
var _assignments = {};

const _TOOLS_ACTIONS = new Set([
    'test-connection',
    'edit-connection',
    'delete-connection',
    'show-connection-dialog',
    'save-tool-permissions',
    'save-erp-config',
    'test-connection-in-dialog',
    'close-conn-dialog',
    'save-connection-from-dialog',
]);

const _TOOLS_CHANGE_ACTIONS = new Set([
    'update-tool-assignment',
    'toggle-erp-backend-fields',
    'update-conn-type-fields',
]);

function _handleToolsClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_TOOLS_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'test-connection':
            SharedUi.runtime.withButtonLock(el, () => testConnection(el.dataset.connId || ''));
            break;
        case 'edit-connection':
            SharedUi.runtime.withButtonLock(el, () => editConnection(el.dataset.connId || ''));
            break;
        case 'delete-connection':
            deleteConnection(el.dataset.connId || '');
            break;
        case 'show-connection-dialog':
            SharedUi.runtime.withButtonLock(el, () => showConnectionDialog());
            break;
        case 'save-tool-permissions':
            saveToolPermissions();
            break;
        case 'save-erp-config':
            saveErpConfig();
            break;
        case 'test-connection-in-dialog':
            SharedUi.runtime.withButtonLock(el, testConnectionInDialog);
            break;
        case 'close-conn-dialog':
            closeConnDialog();
            break;
        case 'save-connection-from-dialog':
            SharedUi.runtime.withButtonLock(el, saveConnectionFromDialog.bind(null, el.dataset.editId || ''));
            break;
    }
}

function _handleToolsChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_TOOLS_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    switch (el.dataset.changeAction) {
        case 'update-tool-assignment':
            updateToolAssignment(el.dataset.toolName || '', el.value);
            break;
        case 'toggle-erp-backend-fields':
            toggleErpBackendFields(el.value);
            break;
        case 'update-conn-type-fields':
            updateConnTypeFields();
            break;
    }
}

function _bindToolsActions() {
    if (document.body._toolsActionsBound) return;
    document.body._toolsActionsBound = true;
    document.body.addEventListener('click', _handleToolsClick);
    document.body.addEventListener('change', _handleToolsChange);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-tools-actions-bound', 'true');
}

async function initToolsTab() {
    _bindToolsActions();
    var container = document.getElementById('contentContainer');
    if (!container) return;
    var filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        // Pflicht-Loads (list, permissions, groups) — Fehler bricht Init ab
        var [toolsResp, permsResp, groupsResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/tools/list'),
            SharedUi.fetch.safeFetch('/api/admin/tools/permissions'),
            SharedUi.fetch.safeFetch('/api/admin/groups'),
        ]);
        var toolsData = await toolsResp.json();
        var permsData = await permsResp.json();
        var groupsData = await groupsResp.json();

        // Optional-Loads — Fehler fallen auf Defaults zurueck
        var [erpCfg, connData, connTypes, assignData] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/tools/erp-config').then(function(r) { return r.json(); }).catch(function() { return {}; }),
            SharedUi.fetch.safeFetch('/api/admin/tools/connections').then(function(r) { return r.json(); }).catch(function() { return { connections: [] }; }),
            SharedUi.fetch.safeFetch('/api/admin/tools/connector-types').then(function(r) { return r.json(); }).catch(function() { return { types: [] }; }),
            SharedUi.fetch.safeFetch('/api/admin/tools/assignments').then(function(r) { return r.json(); }).catch(function() { return { assignments: {} }; }),
        ]);
        _erpConfig = erpCfg;
        _connections = (connData.connections) || [];
        _erpConnectorTypes = (connTypes.types) || [];
        _assignments = (assignData.assignments) || {};
        renderToolsTab(container, toolsData, permsData, groupsData);
    } catch (e) {
        console.error('initToolsTab Fehler:', e);
        container.innerHTML = '<div class="no-data">Tools konnten nicht geladen werden: ' + SharedUi.runtime.escapeHtml(e.message || String(e)) + '</div>';
    }
}

function renderToolsTab(container, toolsData, permsData, groupsData) {
    var tools = toolsData.tools || toolsData || [];
    var permissions = permsData.permissions || permsData || {};
    var groups = groupsData.data || groupsData.groups || groupsData || [];

    var html = '<div class="config-form">';
    html += '<h3>ERP-Tools</h3>';

    html += _renderConnectionsSection();
    html += _renderToolsTable(tools);

    var permTools = tools.filter(function(t) { return (t.category || '') !== 'builtin'; });
    if (groups.length > 0 && permTools.length > 0) {
        html += _renderPermissionsSection(permTools, permissions, groups);
    }

    html += _renderLegacySection();
    html += '</div>';
    container.innerHTML = html;

    _attachLegacyEventListeners();
}

function _renderConnectionsSection() {
    var html = '<div class="config-section"><h4>ERP-Verbindungen</h4>';
    html += '<p class="u-text-xs-secondary u-mb-12">Verbindungen zu ERP-Systemen. Jedes ERP-Tool kann einer Verbindung zugewiesen werden.</p>';

    if (_connections.length === 0) {
        html += '<div class="no-data u-mb-12">Keine Verbindungen konfiguriert.</div>';
    } else {
        html += '<table class="u-detail-table" id="connectionsTable">';
        html += '<tr class="u-cell-border-strong">';
        html += '<th class="u-cell-left-8">Name</th>';
        html += '<th class="u-cell-left-8">Typ</th>';
        html += '<th class="u-cell-left-8">URL / Host</th>';
        html += '<th class="u-cell-center-8">Status</th>';
        html += '<th class="u-cell-center-8">Aktionen</th>';
        html += '</tr>';

        _connections.forEach(function(conn) {
            var typeInfo = _erpConnectorTypes.find(function(t) { return t.type_id === conn.type; });
            var typeName = typeInfo ? typeInfo.display_name : conn.type;
            var urlField = conn.config.url || conn.config.dsn || '';
            var urlDisplay = urlField === '********' ? '(verschluesselt)' : (urlField.length > 40 ? urlField.substring(0, 40) + '...' : urlField);
            html += '<tr class="u-cell-border" data-conn-id="' + SharedUi.runtime.escapeAttr(conn.id) + '">';
            html += '<td class="u-cell-left-8-bold">' + SharedUi.runtime.escapeHtml(conn.name) + '</td>';
            html += '<td class="u-pad-8 u-c-secondary">' + SharedUi.runtime.escapeHtml(typeName) + '</td>';
            html += '<td class="u-pad-8 u-c-secondary u-mono u-fs-12">' + SharedUi.runtime.escapeHtml(urlDisplay) + '</td>';
            html += '<td class="u-cell-center-8" id="connStatus-' + SharedUi.runtime.escapeAttr(conn.id) + '"><span class="badge badge-default">Nicht getestet</span></td>';
            html += '<td class="u-cell-center-8">';
            html += '<button class="action-btn small u-tools-btn-space" data-action="test-connection" data-conn-id="' + SharedUi.runtime.escapeAttr(conn.id) + '" title="Verbindung testen">&#x25B6;</button>';
            html += '<button class="action-btn small u-tools-btn-space" data-action="edit-connection" data-conn-id="' + SharedUi.runtime.escapeAttr(conn.id) + '" title="Bearbeiten">&#x270E;</button>';
            html += '<button class="action-btn small danger" data-action="delete-connection" data-conn-id="' + SharedUi.runtime.escapeAttr(conn.id) + '" title="Loeschen">&#x2716;</button>';
            html += '</td></tr>';
        });
        html += '</table>';
    }

    html += '<div class="config-actions u-mt-12">';
    html += '<button class="action-btn primary" data-action="show-connection-dialog">+ Verbindung anlegen</button>';
    html += '</div></div>';
    return html;
}

function _renderToolsTable(tools) {
    if (tools.length === 0) {
        return '<div class="no-data">Keine Tools registriert.</div>';
    }

    var html = '<div class="config-section"><h4>Registrierte Tools</h4>';
    html += '<table class="u-detail-table">';
    html += '<tr class="u-cell-border-strong">';
    html += '<th class="u-cell-left-8">Name</th>';
    html += '<th class="u-cell-left-8">Beschreibung</th>';
    html += '<th class="u-cell-center-8">Typ</th>';
    html += '<th class="u-cell-center-8">Bestaetigung</th>';
    html += '<th class="u-cell-center-8">ERP-Quelle</th>';
    html += '</tr>';

    tools.forEach(function(t) {
        var name = t.name || t;
        var desc = t.description || '';
        var category = t.category || '';
        var catBadge = category === 'builtin'
            ? '<span class="badge badge-info" title="Fest integriert, immer verfuegbar">Builtin</span>'
            : '';
        var confirmBadge = t.requires_confirmation
            ? '<span class="badge badge-warning">Ja</span>'
            : '<span class="badge badge-default">Nein</span>';

        var erpSourceHtml = '<span class="u-c-secondary">&mdash;</span>';
        if (category === 'erp') {
            var currentAssignment = _assignments[name] || '';
            erpSourceHtml = '<select class="config-input u-fs-12 u-pad-2-6 u-w-auto" data-tool-name="' + SharedUi.runtime.escapeAttr(name) + '" data-change-action="update-tool-assignment">';
            erpSourceHtml += '<option value=""' + (!currentAssignment ? ' selected' : '') + '>Standard-Backend</option>';
            _connections.forEach(function(conn) {
                var sel = currentAssignment === conn.id ? ' selected' : '';
                erpSourceHtml += '<option value="' + SharedUi.runtime.escapeAttr(conn.id) + '"' + sel + '>' + SharedUi.runtime.escapeHtml(conn.name) + '</option>';
            });
            erpSourceHtml += '</select>';
        }

        html += '<tr class="u-cell-border">';
        html += '<td class="u-cell-left-8-bold">' + SharedUi.runtime.escapeHtml(name) + '</td>';
        html += '<td class="u-pad-8 u-c-secondary">' + SharedUi.runtime.escapeHtml(desc) + '</td>';
        html += '<td class="u-cell-center-8">' + catBadge + '</td>';
        html += '<td class="u-cell-center-8">' + confirmBadge + '</td>';
        html += '<td class="u-cell-center-8">' + erpSourceHtml + '</td>';
        html += '</tr>';
    });
    html += '</table></div>';
    return html;
}

function _renderPermissionsSection(permTools, permissions, groups) {
    var html = '<div class="config-section"><h4>Berechtigungen</h4>';
    html += '<p class="u-text-xs-secondary u-mb-12">Welche LDAP-Gruppen duerfen welche Tools nutzen? Builtin-Tools sind immer fuer alle verfuegbar.</p>';
    html += '<table class="u-detail-table-sm" id="toolPermsTable">';
    html += '<tr class="u-cell-border-strong"><th class="u-cell-left-6">Gruppe</th>';
    permTools.forEach(function(t) {
        html += '<th class="u-cell-fs-11 u-text-center">' + SharedUi.runtime.escapeHtml(t.name || t) + '</th>';
    });
    html += '</tr>';

    groups.forEach(function(g) {
        var gLabel = g.cn || g.name || g;
        var gName = g.dn || g.cn || g.name || g;
        var gPerms = permissions[gName] || [];
        html += '<tr class="u-cell-border" data-group="' + SharedUi.runtime.escapeAttr(gName) + '">';
        html += '<td class="u-cell-left-6-bold">' + SharedUi.runtime.escapeHtml(gLabel) + '</td>';
        permTools.forEach(function(t) {
            var tName = t.name || t;
            var checked = gPerms.includes(tName) ? 'checked' : '';
            html += '<td class="u-cell-center-6"><input type="checkbox" data-tool="' + SharedUi.runtime.escapeAttr(tName) + '" ' + checked + '></td>';
        });
        html += '</tr>';
    });
    html += '</table>';
    html += '<div class="config-actions u-mt-12">';
    html += '<button class="action-btn primary" data-action="save-tool-permissions">Berechtigungen speichern</button>';
    html += '</div></div>';
    return html;
}

function _renderLegacySection() {
    var erpBackend = _erpConfig.backend || 'mock';
    var html = '<details class="config-section u-details-box u-mt-12">';
    html += '<summary class="u-summary-bold">Legacy ERP-Backend-Konfiguration</summary>';
    html += '<p class="u-text-xs-secondary u-my-8-0">Diese Konfiguration wird verwendet, wenn ein ERP-Tool keiner Verbindung zugewiesen ist.</p>';

    html += '<div class="modal-field"><label>Backend</label>'
        + '<select class="config-input" id="erpBackend" data-change-action="toggle-erp-backend-fields">'
        + '<option value="mock"' + (erpBackend === 'mock' ? ' selected' : '') + '>Mock-Daten (Entwicklung)</option>'
        + '<option value="sql"' + (erpBackend === 'sql' ? ' selected' : '') + '>SQL-Datenbank</option>'
        + '</select></div>';

    var showSql = erpBackend === 'sql';
    html += '<div id="erpSqlFields" class="' + (showSql ? 'u-block' : 'u-hidden') + '">';
    html += '<div class="modal-field"><label>Verbindungsstring (DSN)</label>'
        + '<input type="text" id="erpSqlDsn" value="' + SharedUi.runtime.escapeAttr(_erpConfig.sql_dsn || '') + '" placeholder="DRIVER={ODBC Driver 18...};SERVER=..."></div>';
    html += '<div class="modal-field"><label>Treiber</label>'
        + '<select class="config-input" id="erpSqlDriver">'
        + '<option value="auto"' + ((_erpConfig.sql_driver || 'auto') === 'auto' ? ' selected' : '') + '>Auto-Erkennung</option>'
        + '<option value="mssql"' + (_erpConfig.sql_driver === 'mssql' ? ' selected' : '') + '>MSSQL</option>'
        + '<option value="postgresql"' + (_erpConfig.sql_driver === 'postgresql' ? ' selected' : '') + '>PostgreSQL</option>'
        + '<option value="mysql"' + (_erpConfig.sql_driver === 'mysql' ? ' selected' : '') + '>MySQL</option>'
        + '</select></div>';
    html += '<div class="modal-field"><button class="action-btn" id="erpTestConnBtn">Verbindung testen</button>'
        + '<span id="erpTestResult" class="u-ml-8 u-fs-12"></span></div>';

    var mapping = _erpConfig.mapping || {};
    var entities = [
        { key: 'orders', label: 'Auftraege', idLabel: 'Auftragsnummer' },
        { key: 'articles', label: 'Artikel', idLabel: 'Artikelnummer' },
        { key: 'customers', label: 'Kunden', idLabel: 'Kundennummer' },
        { key: 'production_orders', label: 'Fertigungsauftraege', idLabel: 'FA-Nummer' },
    ];

    html += '<h5 class="u-mt-16">Tabellen-Mapping</h5>';
    html += '<p class="u-text-xxs-secondary u-mb-8">Welche Tabellen/Views enthalten die ERP-Daten? Leer lassen = nicht konfiguriert.</p>';

    entities.forEach(function(e) {
        var m = mapping[e.key] || {};
        html += '<details class="u-details-box u-mb-8">'
            + '<summary class="u-summary-bold-13">' + e.label + '</summary>'
            + '<div class="u-mt-8">'
            + '<div class="modal-field"><label>Tabelle/View</label>'
            + '<input type="text" class="erp-map-table" data-entity="' + e.key + '" value="' + SharedUi.runtime.escapeAttr(m.table || '') + '" placeholder="z.B. AuftragskopfView"></div>'
            + '<div class="modal-field"><label>' + e.idLabel + ' (ID-Spalte)</label>'
            + '<input type="text" class="erp-map-idcol" data-entity="' + e.key + '" value="' + SharedUi.runtime.escapeAttr(m.id_column || '') + '" placeholder="z.B. Auftragsnummer"></div>'
            + '<div class="modal-field"><label>Spalten-Mapping (JSON)</label>'
            + '<textarea class="erp-map-cols u-textarea-mono-xxs" data-entity="' + e.key + '" rows="3" placeholder=\'{"order_id": "Auftragsnummer", ...}\'>' + SharedUi.runtime.escapeHtml(m.columns ? JSON.stringify(m.columns, null, 2) : '') + '</textarea></div>'
            + '</div></details>';
    });
    html += '</div>';

    html += '<div class="config-actions u-mt-12">';
    html += '<button class="action-btn primary" data-action="save-erp-config">ERP-Konfiguration speichern</button>';
    html += '</div></details>';
    return html;
}

function showConnectionDialog(editId) {
    var existing = editId ? _connections.find(function(c) { return c.id === editId; }) : null;
    var title = existing ? 'Verbindung bearbeiten' : 'Verbindung anlegen';

    var html = '<div class="modal-overlay" id="connDialogOverlay">'
        + '<div class="modal-content u-mw-500">'
        + '<div class="modal-header">' + SharedUi.runtime.escapeHtml(title) + '</div>'
        + '<div class="modal-body">';

    html += '<div class="modal-field"><label>Name</label>'
        + '<input type="text" id="connName" value="' + SharedUi.runtime.escapeAttr(existing ? existing.name : '') + '" placeholder="z.B. Odoo Produktion"></div>';

    html += '<div class="modal-field"><label>Typ</label>'
        + '<select class="config-input" id="connType" data-change-action="update-conn-type-fields">';
    _erpConnectorTypes.forEach(function(t) {
        var sel = existing && existing.type === t.type_id ? ' selected' : '';
        html += '<option value="' + SharedUi.runtime.escapeAttr(t.type_id) + '"' + sel + '>' + SharedUi.runtime.escapeHtml(t.display_name) + '</option>';
    });
    html += '</select></div>';

    html += '<div id="connTypeFields"></div>';

    html += '<div class="modal-field u-mt-12">'
        + '<button class="action-btn" id="connTestBtn" data-action="test-connection-in-dialog">Verbindung testen</button>'
        + '<span id="connTestResult" class="u-ml-8 u-fs-12"></span></div>';

    html += '</div>'
        + '<div class="modal-footer">'
        + '<button class="action-btn" data-action="close-conn-dialog">Abbrechen</button>'
        + '<button class="action-btn primary" data-action="save-connection-from-dialog" data-edit-id="' + SharedUi.runtime.escapeAttr(editId || '') + '">Speichern</button>'
        + '</div></div></div>';

    document.body.insertAdjacentHTML('beforeend', html);
    if (existing) {
        window._editConnConfig = existing.config;
        window._editConnId = editId;
    } else {
        window._editConnConfig = null;
        window._editConnId = null;
    }
    updateConnTypeFields();
}

function closeConnDialog() {
    var overlay = document.getElementById('connDialogOverlay');
    if (overlay) overlay.remove();
    window._editConnConfig = null;
    window._editConnId = null;
}

function updateConnTypeFields() {
    var typeEl = document.getElementById('connType');
    var typeId = typeEl ? typeEl.value : '';
    var container = document.getElementById('connTypeFields');
    if (!container || !typeId) return;

    var typeInfo = _erpConnectorTypes.find(function(t) { return t.type_id === typeId; });
    if (!typeInfo) { container.innerHTML = ''; return; }

    var existingConfig = window._editConnConfig || {};
    var html = '<div class="u-conn-fields-box">';

    typeInfo.config_fields.forEach(function(f) {
        if (f.type === 'json') {
            var val = existingConfig[f.name] ? JSON.stringify(existingConfig[f.name], null, 2) : '';
            html += '<div class="modal-field"><label>' + SharedUi.runtime.escapeHtml(f.label) + '</label>'
                + '<textarea class="conn-field u-textarea-mono-xs" data-field="' + SharedUi.runtime.escapeAttr(f.name) + '" data-field-type="json" rows="4" placeholder="' + SharedUi.runtime.escapeAttr(f.placeholder || '') + '">' + SharedUi.runtime.escapeHtml(val) + '</textarea></div>';
        } else if (f.type === 'select') {
            html += '<div class="modal-field"><label>' + SharedUi.runtime.escapeHtml(f.label) + '</label>'
                + '<select class="config-input conn-field" data-field="' + SharedUi.runtime.escapeAttr(f.name) + '">';
            (f.options || []).forEach(function(opt) {
                var sel = existingConfig[f.name] === opt ? ' selected' : '';
                html += '<option value="' + SharedUi.runtime.escapeAttr(opt) + '"' + sel + '>' + SharedUi.runtime.escapeHtml(opt) + '</option>';
            });
            html += '</select></div>';
        } else {
            var inputType = f.type === 'password' ? 'password' : 'text';
            var val2 = existingConfig[f.name] || '';
            html += '<div class="modal-field"><label>' + SharedUi.runtime.escapeHtml(f.label) + '</label>'
                + '<input type="' + inputType + '" class="conn-field" data-field="' + SharedUi.runtime.escapeAttr(f.name) + '" value="' + SharedUi.runtime.escapeAttr(val2) + '" placeholder="' + SharedUi.runtime.escapeAttr(f.placeholder || '') + '"></div>';
        }
    });

    html += '</div>';
    container.innerHTML = html;
}

function _collectConnConfig() {
    var config = {};
    var hasError = false;
    document.querySelectorAll('.conn-field').forEach(function(el) {
        if (hasError) return;
        var name = el.dataset.field;
        var parent = el.closest('.modal-field');
        var label = parent ? parent.querySelector('label') : null;
        var labelText = label ? label.textContent : '';
        if (el.dataset.fieldType === 'json') {
            try { config[name] = JSON.parse(el.value); } catch(e) {
                SharedUi.notify.show(`Ungueltiges JSON in "${labelText.trim()}": ${e.message}`, 'error');
                hasError = true;
                return;
            }
        } else {
            config[name] = el.value;
        }
    });
    if (hasError) return null;
    return config;
}

async function saveConnectionFromDialog(editId) {
    var nameEl = document.getElementById('connName');
    var typeEl = document.getElementById('connType');
    var name = nameEl ? nameEl.value.trim() : '';
    var type = typeEl ? typeEl.value : '';
    if (!name) { SharedUi.notify.show('Name ist erforderlich', 'error'); return; }

    var config = _collectConnConfig();
    if (!config) return;
    var payload = { name: name, type: type, config: config, enabled: true };

    try {
        var resp;
        if (editId) {
            resp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections/' + editId, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
        } else {
            resp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
        }
        if (!resp.ok) { var e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
        SharedUi.notify.show(editId ? 'Verbindung aktualisiert' : 'Verbindung angelegt', 'success');
        closeConnDialog();
        initToolsTab();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

function editConnection(connId) {
    showConnectionDialog(connId);
}

async function deleteConnection(connId) {
    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections/' + connId, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Loeschen fehlgeschlagen');
        SharedUi.notify.show('Verbindung geloescht', 'success');
        initToolsTab();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function testConnection(connId) {
    var statusEl = document.getElementById('connStatus-' + connId);
    if (statusEl) statusEl.innerHTML = '<span class="badge badge-default">Teste...</span>';
    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections/' + connId + '/test', { method: 'POST' });
        var data = await resp.json();
        if (statusEl) {
            if (resp.ok && data.success) {
                statusEl.innerHTML = '<span class="badge badge-success" title="' + SharedUi.runtime.escapeAttr(data.message || '') + '">&check; OK</span>';
            } else {
                statusEl.innerHTML = '<span class="badge badge-danger" title="' + SharedUi.runtime.escapeAttr(data.detail || data.message || '') + '">&times; Fehler</span>';
            }
        }
    } catch (e) {
        if (statusEl) statusEl.innerHTML = '<span class="badge badge-danger">Fehler</span>';
    }
}

async function testConnectionInDialog() {
    var btn = document.getElementById('connTestBtn');
    var resultEl = document.getElementById('connTestResult');
    if (!btn || !resultEl) return;

    var nameEl = document.getElementById('connName');
    var typeEl = document.getElementById('connType');
    var name = (nameEl ? nameEl.value.trim() : '') || 'Test';
    var type = typeEl ? typeEl.value : '';
    var config = _collectConnConfig();
    if (!config) return;

    btn.disabled = true;
    btn.textContent = 'Teste...';
    resultEl.textContent = '';

    // Edit-Modus: Pruefen ob Passwort-Felder noch maskiert sind (********)
    var hasMaskedSecret = false;
    if (window._editConnId) {
        document.querySelectorAll('.conn-field').forEach(function(el) {
            if (el.type === 'password' && el.value === '********') hasMaskedSecret = true;
        });
    }

    if (window._editConnId && hasMaskedSecret) {
        // Bestehende Verbindung direkt testen (maskiertes Passwort kann nicht
        // als Klartext an eine neue temp. Verbindung gesendet werden)
        try {
            var testResp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections/' + window._editConnId + '/test', { method: 'POST' });
            var testData = await testResp.json();
            if (testResp.ok && testData.success) {
                resultEl.innerHTML = '<span class="u-c-success">&check; ' + SharedUi.runtime.escapeHtml(testData.message || 'OK') + '</span>';
            } else {
                resultEl.innerHTML = '<span class="u-c-danger">&times; ' + SharedUi.runtime.escapeHtml(testData.detail || testData.error || testData.message || 'Fehlgeschlagen') + '</span>';
            }
        } catch (e) {
            resultEl.innerHTML = '<span class="u-c-danger">&times; ' + SharedUi.runtime.escapeHtml(e.message) + '</span>';
        }
    } else {
        // Neue Verbindung oder alle Secrets manuell eingegeben: temp. Verbindung anlegen + testen
        var testConnId = null;
        try {
            var saveResp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name + ' (Test)', type: type, config: config, enabled: true }),
            });
            if (!saveResp.ok) throw new Error('Speichern fehlgeschlagen');
            var saveData = await saveResp.json();
            testConnId = saveData.id;

            var testResp = await SharedUi.fetch.safeFetch('/api/admin/tools/connections/' + testConnId + '/test', { method: 'POST' });
            var testData = await testResp.json();

            if (testResp.ok && testData.success) {
                resultEl.innerHTML = '<span class="u-c-success">&check; ' + SharedUi.runtime.escapeHtml(testData.message || 'OK') + '</span>';
            } else {
                resultEl.innerHTML = '<span class="u-c-danger">&times; ' + SharedUi.runtime.escapeHtml(testData.detail || testData.error || testData.message || 'Fehlgeschlagen') + '</span>';
            }
        } catch (e) {
            resultEl.innerHTML = '<span class="u-c-danger">&times; ' + SharedUi.runtime.escapeHtml(e.message) + '</span>';
        } finally {
            // Garantierter Delete-Cleanup der temporaeren Verbindung
            if (testConnId) {
                try { await SharedUi.fetch.safeFetch('/api/admin/tools/connections/' + testConnId, { method: 'DELETE' }); } catch(_) {}
            }
        }
    }

    btn.disabled = false;
    btn.textContent = 'Verbindung testen';
}

async function updateToolAssignment(toolName, connId) {
    var prev = _assignments[toolName];
    _assignments[toolName] = connId;
    var assignments = {};
    var keys = Object.keys(_assignments);
    for (var i = 0; i < keys.length; i++) {
        if (_assignments[keys[i]]) assignments[keys[i]] = _assignments[keys[i]];
    }
    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/tools/assignments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ assignments: assignments }),
        });
        if (!resp.ok) throw new Error('Fehler beim Speichern');
        SharedUi.notify.show('Tool "' + toolName + '" zugewiesen', 'success');
    } catch (e) {
        _assignments[toolName] = prev;
        var escapedName = String(toolName).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
        var selectEl = document.querySelector('select[data-tool-name="' + escapedName + '"]');
        if (selectEl) selectEl.value = prev || '';
        SharedUi.notify.show(e.message, 'error');
    }
}

function _attachLegacyEventListeners() {
    var testBtn = document.getElementById('erpTestConnBtn');
    if (testBtn) {
        testBtn.addEventListener('click', async function() {
            var btn = document.getElementById('erpTestConnBtn');
            var resultEl = document.getElementById('erpTestResult');
            btn.disabled = true;
            btn.textContent = 'Teste...';
            resultEl.textContent = '';
            try {
                await saveErpConfig(true);
                var resp = await SharedUi.fetch.safeFetch('/api/admin/tools/erp-test-connection', { method: 'POST' });
                var data = await resp.json();
                if (resp.ok) {
                    resultEl.innerHTML = '<span class="u-c-success">' + SharedUi.runtime.escapeHtml(data.tables ? data.tables.length + ' Tabellen gefunden' : 'Verbindung OK') + '</span>';
                } else {
                    resultEl.innerHTML = '<span class="u-c-danger">' + SharedUi.runtime.escapeHtml(data.detail || data.error || data.message || 'Fehlgeschlagen') + '</span>';
                }
            } catch (e) {
                resultEl.innerHTML = '<span class="u-c-danger">Fehler</span>';
            }
            btn.disabled = false;
            btn.textContent = 'Verbindung testen';
        });
    }
}

function toggleErpBackendFields(value) {
    var fields = document.getElementById('erpSqlFields');
    if (!fields) return;
    if (value === 'sql') {
        fields.classList.remove('u-hidden');
        fields.classList.add('u-block');
    } else {
        fields.classList.remove('u-block');
        fields.classList.add('u-hidden');
    }
}

async function saveErpConfig(silent) {
    var backendEl = document.getElementById('erpBackend');
    var dsnEl = document.getElementById('erpSqlDsn');
    var driverEl = document.getElementById('erpSqlDriver');
    var backend = backendEl ? backendEl.value : 'mock';
    var sqlDsn = dsnEl ? dsnEl.value : '';
    var sqlDriver = driverEl ? driverEl.value : 'auto';

    var mapping = {};
    var existingMapping = _erpConfig.mapping || {};
    var hasError = false;
    document.querySelectorAll('.erp-map-table').forEach(function(el) {
        var entity = el.dataset.entity;
        var table = el.value.trim();
        if (!table) return;

        var idColEl = document.querySelector('.erp-map-idcol[data-entity="' + entity + '"]');
        var colsEl = document.querySelector('.erp-map-cols[data-entity="' + entity + '"]');
        var idCol = idColEl ? idColEl.value : '';
        var colsJson = colsEl ? colsEl.value : '{}';
        var columns = {};
        try { columns = JSON.parse(colsJson); } catch(e) { SharedUi.notify.show('Ungueltiges JSON im Spalten-Mapping fuer ' + entity, 'error'); hasError = true; return; }

        mapping[entity] = Object.assign({}, existingMapping[entity] || {}, {
            table: table,
            id_column: idCol,
            columns: columns,
        });
    });
    if (hasError) return;

    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/tools/erp-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend: backend, sql_dsn: sqlDsn, sql_driver: sqlDriver, mapping: mapping }),
        });
        if (!resp.ok) { var e = await resp.json(); throw new Error(e.detail || e.error || e.message || 'Fehler'); }
        if (!silent) SharedUi.notify.show('ERP-Konfiguration gespeichert', 'success');
    } catch (e) {
        if (!silent) SharedUi.notify.show(e.message, 'error');
    }
}

async function saveToolPermissions() {
    var table = document.getElementById('toolPermsTable');
    if (!table) return;
    var rows = table.querySelectorAll('tr[data-group]');
    var errors = 0;
    for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        var group = row.dataset.group;
        var tools = [];
        row.querySelectorAll('input[data-tool]:checked').forEach(function(cb) { tools.push(cb.dataset.tool); });
        try {
            var resp = await SharedUi.fetch.safeFetch('/api/admin/tools/permissions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
                body: JSON.stringify({ group: group, tools: tools }),
            });
            if (!resp.ok) errors++;
        } catch (e) { errors++; }
    }
    if (errors === 0) {
        SharedUi.notify.show('Berechtigungen gespeichert', 'success');
    } else {
        SharedUi.notify.show(errors + ' Fehler beim Speichern', 'error');
        await initToolsTab();
    }
}
