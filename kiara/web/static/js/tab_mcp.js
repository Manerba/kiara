/**
 * KIara Admin - Tools > MCP / API-Keys Sub-Tab
 * Sprint 1.4: Lifecycle-Guards, One-Time-Key-Stash, Button-Locks, Fehlerresolver
 */

let _mcpRequestSeq = 0;
let _mcpPendingCreatedKey = null;
let _mcpCurrentConfig = null;

/**
 * Gemeinsamer Fehlerresolver fuer MCP-Mutationen.
 * Kette: detail || error || message || statusText || Fallback.
 * Toleriert Nicht-JSON-Error-Bodies.
 */
async function _mcpResolveError(resp, fallback) {
    const data = await resp.json().catch(() => ({}));
    return data.detail || data.error || data.message || resp.statusText || fallback;
}

/**
 * Rendert den pending One-Time-Key in #mcpNewKeyDisplay.
 * Loescht den Slot NICHT — nur mcpCopyKey() und mcpDismissPendingKey() duerfen das.
 */
function _renderPendingMcpKey() {
    if (!_mcpPendingCreatedKey) return;
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'tools-mcp') return;
    var display = document.getElementById('mcpNewKeyDisplay');
    if (!display) return;
    display.classList.remove('is-hidden');
    display.innerHTML = '<div class="mcp-pending-key">'
        + '<strong>API-Key erstellt \u2014 jetzt kopieren!</strong>'
        + '<div class="mcp-pending-actions">'
        + '<code id="mcpNewKeyValue" class="mcp-pending-value">' + SharedUi.runtime.escapeHtml(_mcpPendingCreatedKey) + '</code>'
        + '<button class="action-btn-sm" data-mcp-action="copy-pending-key">Kopieren</button>'
        + ' <button class="action-btn-sm" data-mcp-action="dismiss-pending-key">Verwerfen</button>'
        + '</div>'
        + '<div class="mcp-pending-hint">Dieser Key wird nur einmal angezeigt. Bitte jetzt kopieren!</div>'
        + '</div>';
}

function _bindMcpActions(container) {
    if (!container || container.dataset.mcpActionsBound === '1') return;
    container.dataset.mcpActionsBound = '1';
    container.addEventListener('click', function(event) {
        var actionEl = event.target.closest('[data-mcp-action]');
        if (!actionEl) return;

        var action = actionEl.getAttribute('data-mcp-action');
        if (action === 'toggle-server') {
            var currentEnabled = actionEl.getAttribute('data-mcp-enabled') === 'true';
            SharedUi.runtime.withButtonLock(actionEl, function() {
                return mcpToggleServer(currentEnabled);
            });
            return;
        }
        if (action === 'create-key') {
            SharedUi.runtime.withButtonLock(actionEl, mcpCreateKey);
            return;
        }
        if (action === 'copy-pending-key') {
            mcpCopyKey();
            return;
        }
        if (action === 'dismiss-pending-key') {
            mcpDismissPendingKey();
            return;
        }
        if (action === 'save-retrieval-settings') {
            SharedUi.runtime.withButtonLock(actionEl, function() {
                return mcpSaveRetrievalSettings(actionEl);
            });
            return;
        }
        if (action === 'reset-retrieval-settings') {
            initMcpTab();
            return;
        }

        var keyId = parseInt(actionEl.getAttribute('data-key-id') || '', 10);
        if (!Number.isFinite(keyId)) return;

        if (action === 'toggle-key') {
            SharedUi.runtime.withButtonLock(actionEl, function() {
                return mcpToggleKey(keyId);
            });
        } else if (action === 'delete-key') {
            var keyName = actionEl.getAttribute('data-key-name') || '';
            SharedUi.runtime.withButtonLock(actionEl, function() {
                return mcpDeleteKey(keyId, keyName);
            });
        }
    });
}

async function initMcpTab() {
    // Cleanup am Init-Anfang: invalidiert aeltere in-flight Laeufe
    if (typeof _runTabCleanups === 'function') _runTabCleanups('tools-mcp');

    var seq = ++_mcpRequestSeq;

    var container = document.getElementById('contentContainer');
    if (!container) return;
    _bindMcpActions(container);

    // Active-Subtab-Guard vor Spinner-Write
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'tools-mcp') return;

    var filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    // Cleanup registrieren: invalidiert aeltere in-flight Requests, behaelt pending Key
    if (typeof registerTabCleanup === 'function') {
        registerTabCleanup('tools-mcp', function() { _mcpRequestSeq++; });
    }

    try {
        // Pflicht-Loads via safeFetch (wirft bei non-2xx), Status bleibt Soft-Fail
        var responses = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/settings', { headers: { 'X-Requested-With': 'XMLHttpRequest' } }),
            SharedUi.fetch.safeFetch('/api/admin/mcp/keys', { headers: { 'X-Requested-With': 'XMLHttpRequest' } }),
            SharedUi.fetch.safeFetch('/api/admin/users', { headers: { 'X-Requested-With': 'XMLHttpRequest' } }),
            SharedUi.fetch.safeFetch('/api/admin/mcp/status', { headers: { 'X-Requested-With': 'XMLHttpRequest' } }).catch(function() { return null; }),
        ]);
        var settingsResp = responses[0], keysResp = responses[1], usersResp = responses[2], statusResp = responses[3];

        // Latest-only + Active-Subtab-Guard nach await
        if (seq !== _mcpRequestSeq) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'tools-mcp') return;

        var config = (await settingsResp.json()).config;
        var keysData = await keysResp.json();
        var usersData = await usersResp.json();
        var statusData = null;
        if (statusResp && statusResp.ok) {
            statusData = await statusResp.json();
        }

        renderMcpTab(container, config, keysData, usersData, statusData);

        // Pending Key nach erfolgreichem Render nachrendern
        _renderPendingMcpKey();
    } catch (e) {
        // Guards im catch
        if (seq !== _mcpRequestSeq) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'tools-mcp') return;
        container.innerHTML = '<div class="no-data">MCP-Daten konnten nicht geladen werden.</div>';
    }
}

function renderMcpTab(container, config, keysData, usersData, statusData) {
    _mcpCurrentConfig = config;
    var mcpEndpointEnabled = statusData?.mcp_endpoint_enabled ?? config?.web?.mcp_enabled ?? false;
    var retrievalEnabled = statusData?.retrieval_enabled ?? config?.web?.mcp_retrieval_enabled ?? false;
    var serviceShouldRun = statusData?.service_should_run ?? (mcpEndpointEnabled || retrievalEnabled);
    var mcpPort = statusData?.port ?? config?.web?.mcp_port ?? 8504;
    var embeddingModel = config?.ollama?.embedding_model || 'unbekannt';
    var serviceRunning = statusData?.service_running ?? false;
    var serviceStatus = statusData?.service_status ?? 'unknown';
    var keys = keysData.data || [];
    var users = usersData.data || usersData.users || [];

    var html = '<div class="config-form">';
    html += '<h3>MCP / API-Keys</h3>';

    html += '<div class="config-section"><h4>MCP-Server Status</h4>';
    if (serviceShouldRun && serviceRunning) {
        html += '<div class="mcp-alert mcp-alert-success">';
        html += '<div class="mcp-alert-row">';
        html += '<strong class="mcp-alert-status">\u25cf kiara-mcp.service aktiv</strong>';
        html += '<div class="mcp-alert-meta">';
        html += '<span>Port <code>' + SharedUi.runtime.escapeHtml(String(mcpPort)) + '</code></span>';
        if (statusData?.pid) html += '<span>PID <code>' + SharedUi.runtime.escapeHtml(String(statusData.pid)) + '</code></span>';
        if (statusData?.uptime) html += '<span>' + SharedUi.runtime.escapeHtml(statusData.uptime) + '</span>';
        html += '</div>';
        html += '</div>';
        html += '<div class="mcp-alert-details">';
        html += '<div>MCP-Endpunkt <code>/mcp</code>: <strong>' + (mcpEndpointEnabled ? 'aktiv' : 'deaktiviert') + '</strong></div>';
        html += '<div>Retrieval-REST <code>/api/retrieval/*</code>: <strong>' + (retrievalEnabled ? 'aktiv' : 'deaktiviert') + '</strong></div>';
        if (mcpEndpointEnabled) {
            html += '<div>MCP-URL: <code>http://' + SharedUi.runtime.escapeHtml(location.hostname) + ':' + mcpPort + '/mcp</code></div>';
        }
        html += '<div>Embedding-Modell: <code>' + SharedUi.runtime.escapeHtml(embeddingModel) + '</code></div>';
        html += '</div>';
        html += '</div>';
    } else if (serviceShouldRun && !serviceRunning) {
        html += '<div class="mcp-alert mcp-alert-warning">';
        html += '<strong class="mcp-alert-status">\u26a0 kiara-mcp.service sollte laufen, ist aber nicht aktiv</strong>';
        html += '<div class="mcp-alert-details">';
        html += '<div>Service-Status: <code>' + SharedUi.runtime.escapeHtml(serviceStatus) + '</code></div>';
        html += '<div>MCP-Endpunkt <code>/mcp</code>: <strong>' + (mcpEndpointEnabled ? 'aktiv' : 'deaktiviert') + '</strong></div>';
        html += '<div>Retrieval-REST <code>/api/retrieval/*</code>: <strong>' + (retrievalEnabled ? 'aktiv' : 'deaktiviert') + '</strong></div>';
        html += '</div>';
        html += '</div>';
    } else {
        html += '<div class="mcp-alert mcp-alert-neutral">';
        html += '<strong class="mcp-alert-status">\u25cb kiara-mcp.service deaktiviert</strong>';
        html += '<div class="mcp-alert-details">';
        html += '<div>MCP-Endpunkt <code>/mcp</code>: <strong>deaktiviert</strong></div>';
        html += '<div>Retrieval-REST <code>/api/retrieval/*</code>: <strong>deaktiviert</strong></div>';
        html += '</div>';
        html += '</div>';
    }

    html += '<div class="mcp-server-actions">';
    html += '<button class="action-btn ' + (mcpEndpointEnabled ? 'danger' : 'primary') + '" data-mcp-action="toggle-server" data-mcp-enabled="' + (mcpEndpointEnabled ? 'true' : 'false') + '">';
    html += mcpEndpointEnabled ? 'MCP-Endpunkt deaktivieren' : 'MCP-Endpunkt aktivieren';
    html += '</button></div>';

    if (mcpEndpointEnabled) {
        html += '<details class="mcp-details">';
        html += '<summary class="mcp-summary">Konfigurationsbeispiel (VS Code)</summary>';
        html += '<pre class="mcp-code-block">';
        html += SharedUi.runtime.escapeHtml(JSON.stringify({
            servers: {
                kiara: {
                    type: "streamable-http",
                    url: "http://" + location.hostname + ":" + mcpPort + "/mcp",
                    headers: { Authorization: "Bearer orag_..." }
                }
            }
        }, null, 2));
        html += '</pre></details>';

        html += '<details class="mcp-details">';
        html += '<summary class="mcp-summary">Konfigurationsbeispiel (Claude Code)</summary>';
        html += '<pre class="mcp-code-block">';
        html += SharedUi.runtime.escapeHtml(JSON.stringify({
            mcpServers: {
                kiara: {
                    type: "streamable-http",
                    url: "http://" + location.hostname + ":" + mcpPort + "/mcp",
                    headers: { Authorization: "Bearer orag_..." }
                }
            }
        }, null, 2));
        html += '</pre></details>';
    }
    html += '</div>';

    html += _mcpRenderRetrievalSettings(config);

    html += '<div class="config-section"><h4>Neuen API-Key erstellen</h4>';
    html += '<div class="mcp-key-create-row">';
    html += '<div class="mcp-field mcp-field-grow"><label class="mcp-field-label">Name</label>';
    html += '<input type="text" id="mcpKeyName" class="mcp-input" placeholder="z.B. VS Code Andreas"></div>';
    html += '<div class="mcp-field"><label class="mcp-field-label">Benutzer</label>';
    html += '<select id="mcpKeyUser" class="mcp-select">';
    users.forEach(function(u) {
        var uid = u.id;
        var uname = u.username || u.vorname || '';
        html += '<option value="' + uid + '">' + SharedUi.runtime.escapeHtml(uname) + '</option>';
    });
    html += '</select></div>';
    html += '<div class="mcp-field"><label class="mcp-field-label">Ablaufdatum (optional)</label>';
    html += '<input type="date" id="mcpKeyExpires" class="mcp-input"></div>';
    html += '<button class="action-btn primary" data-mcp-action="create-key">Key erstellen</button>';
    html += '</div>';
    html += '<div id="mcpNewKeyDisplay" class="is-hidden mcp-new-key-display"></div>';
    html += '</div>';

    html += '<div class="config-section"><h4>API-Keys</h4>';
    if (keys.length === 0) {
        html += '<div class="no-data">Keine API-Keys vorhanden.</div>';
    } else {
        html += '<table class="mcp-table">';
        html += '<tr class="mcp-table-head">';
        html += '<th class="mcp-table-cell">Name</th>';
        html += '<th class="mcp-table-cell">Key</th>';
        html += '<th class="mcp-table-cell">Benutzer</th>';
        html += '<th class="mcp-table-cell">Erstellt</th>';
        html += '<th class="mcp-table-cell">Letzter Zugriff</th>';
        html += '<th class="mcp-table-cell">Ablauf</th>';
        html += '<th class="mcp-table-cell mcp-table-cell-center">Status</th>';
        html += '<th class="mcp-table-cell mcp-table-cell-center">Aktionen</th>';
        html += '</tr>';
        keys.forEach(function(k) {
            var statusBadge = k.is_active
                ? '<span class="badge badge-success">Aktiv</span>'
                : '<span class="badge badge-danger">Gesperrt</span>';
            var toggleLabel = k.is_active ? 'Sperren' : 'Aktivieren';
            var lastUsed = k.last_used_at || '-';
            var expires = k.expires_at || 'unbegrenzt';
            var created = k.created_at ? k.created_at.substring(0, 10) : '-';
            html += '<tr class="mcp-table-row">';
            html += '<td class="mcp-table-cell mcp-table-cell-strong">' + SharedUi.runtime.escapeHtml(k.name) + '</td>';
            html += '<td class="mcp-table-cell mcp-table-cell-mono">' + SharedUi.runtime.escapeHtml(k.key_prefix) + '</td>';
            html += '<td class="mcp-table-cell">' + SharedUi.runtime.escapeHtml(k.username || '') + '</td>';
            html += '<td class="mcp-table-cell mcp-table-cell-small">' + SharedUi.runtime.escapeHtml(created) + '</td>';
            html += '<td class="mcp-table-cell mcp-table-cell-small">' + SharedUi.runtime.escapeHtml(lastUsed) + '</td>';
            html += '<td class="mcp-table-cell mcp-table-cell-small">' + SharedUi.runtime.escapeHtml(expires) + '</td>';
            html += '<td class="mcp-table-cell mcp-table-cell-center">' + statusBadge + '</td>';
            html += '<td class="mcp-table-cell mcp-table-cell-center mcp-nowrap">';
            html += '<button class="action-btn-sm" data-mcp-action="toggle-key" data-key-id="' + k.id + '">' + toggleLabel + '</button> ';
            html += '<button class="action-btn-sm danger" data-mcp-action="delete-key" data-key-id="' + k.id + '" data-key-name="' + SharedUi.runtime.escapeAttr(k.name) + '">Loeschen</button>';
            html += '</td></tr>';
        });
        html += '</table>';
    }
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;
}

function _mcpRenderRetrievalSettings(config) {
    var web = config?.web || {};
    var enabled = web.mcp_retrieval_enabled === true;
    var allowRaw = web.mcp_retrieval_allow_raw_chunks === true;
    var includeText = web.mcp_retrieval_include_text_default === true;

    var html = '<div class="config-section"><h4>Retrieval-only API</h4>';
    html += '<p class="config-hint">Steuert den externen MCP/API-Zugriff auf KIaras Retrieval-Pipeline. Diese Schalter aendern nicht die Chat-/RAG-Retrieval-Policy; sie begrenzen Zugriff, Rohchunk-Ausgabe und Antwortgroesse fuer externe Clients.</p>';
    html += '<div class="config-fields">';
    html += _mcpRetrievalCheckbox(
        'web.mcp_retrieval_enabled',
        enabled,
        'Retrieval-only Tools aktivieren',
        'Erlaubt search_documents, list_datasources und get_document_context fuer gueltige MCP/API-Keys.'
    );
    html += _mcpRetrievalCheckbox(
        'web.mcp_retrieval_allow_raw_chunks',
        allowRaw,
        'Rohchunk-Volltext erlauben',
        'Sicherheitsrelevant: Wenn aktiv, duerfen externe Clients vollstaendige Chunktexte anfordern. Standard bleibt fail-closed aus.'
    );
    html += _mcpRetrievalCheckbox(
        'web.mcp_retrieval_include_text_default',
        includeText,
        'Text standardmaessig mitsenden',
        'Nur wirksam, wenn Rohchunks erlaubt sind. Andernfalls bleiben Excerpts, Quellen und Metadaten verfuegbar.'
    );
    html += _mcpRetrievalNumber(
        'web.mcp_retrieval_max_k',
        web.mcp_retrieval_max_k ?? 50,
        'Max. Treffer k',
        1,
        200,
        'Obergrenze fuer Treffer pro search_documents-Aufruf.'
    );
    html += _mcpRetrievalNumber(
        'web.mcp_retrieval_max_query_length',
        web.mcp_retrieval_max_query_length ?? 10000,
        'Max. Query-Laenge',
        1,
        50000,
        'Maximale Zeichenanzahl der Suchanfrage.'
    );
    html += _mcpRetrievalNumber(
        'web.mcp_retrieval_max_chars_per_result',
        web.mcp_retrieval_max_chars_per_result ?? 4000,
        'Max. Zeichen pro Treffer',
        1,
        50000,
        'Begrenzt Excerpt/Rohtext pro Treffer.'
    );
    html += _mcpRetrievalNumber(
        'web.mcp_retrieval_max_context_window',
        web.mcp_retrieval_max_context_window ?? 5,
        'Max. Kontextfenster',
        1,
        50,
        'Obergrenze fuer Nachbar-Chunks vor/nach einem Treffer.'
    );
    html += _mcpRetrievalNumber(
        'web.mcp_retrieval_max_context_chunks',
        web.mcp_retrieval_max_context_chunks ?? 10,
        'Max. Kontext-Chunks',
        1,
        200,
        'Obergrenze fuer get_document_context.'
    );
    html += _mcpRetrievalNumber(
        'web.mcp_retrieval_max_chars_total',
        web.mcp_retrieval_max_chars_total ?? 20000,
        'Max. Gesamtzeichen',
        1,
        200000,
        'Gesamtlimit fuer Textausgabe eines Retrieval-only Calls.'
    );
    html += '</div>';
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-mcp-action="save-retrieval-settings">Retrieval-only speichern</button>';
    html += '<button class="action-btn" data-mcp-action="reset-retrieval-settings">Zuruecksetzen</button>';
    html += '</div>';
    html += '</div>';
    return html;
}

function _mcpRetrievalCheckbox(key, checked, label, hint) {
    var html = '<div class="config-field">';
    html += '<label><input type="checkbox" class="config-checkbox mcp-retrieval-checkbox" data-key="';
    html += SharedUi.runtime.escapeAttr(key) + '"' + (checked ? ' checked' : '') + '> ';
    html += SharedUi.runtime.escapeHtml(label) + '</label>';
    html += '<span class="field-hint">' + SharedUi.runtime.escapeHtml(hint) + '</span>';
    html += '</div>';
    return html;
}

function _mcpRetrievalNumber(key, value, label, min, max, hint) {
    var html = '<div class="config-field"><label>' + SharedUi.runtime.escapeHtml(label) + '</label>';
    html += '<input type="number" class="config-input mcp-retrieval-input" data-key="';
    html += SharedUi.runtime.escapeAttr(key) + '" value="';
    html += SharedUi.runtime.escapeAttr(String(value)) + '" min="';
    html += SharedUi.runtime.escapeAttr(String(min)) + '" max="';
    html += SharedUi.runtime.escapeAttr(String(max)) + '">';
    html += '<span class="field-hint">' + SharedUi.runtime.escapeHtml(hint) + '</span>';
    html += '</div>';
    return html;
}

async function mcpCreateKey() {
    var name = document.getElementById('mcpKeyName')?.value?.trim();
    var userId = document.getElementById('mcpKeyUser')?.value;
    var expiresRaw = document.getElementById('mcpKeyExpires')?.value;
    var expires = expiresRaw || null;

    if (!name) {
        SharedUi.notify.show('Bitte einen Namen eingeben.', 'error');
        return;
    }
    var parsedUserId = parseInt(userId);
    if (isNaN(parsedUserId)) {
        SharedUi.notify.show('Bitte einen Benutzer auswaehlen.', 'error');
        return;
    }

    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/mcp/keys', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify({ name: name, user_id: parsedUserId, expires_at: expires }),
        });
        if (!resp.ok) {
            throw new Error(await _mcpResolveError(resp, 'Fehler beim Erstellen des API-Keys'));
        }
        var data = await resp.json();

        // Slot SOFORT setzen — vor jedem DOM-Zugriff oder Tab-Check
        _mcpPendingCreatedKey = data.key;

        // Spaete Success-DOM-Writes unter Guards
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'tools-mcp') return;

        _renderPendingMcpKey();

        var nameInput = document.getElementById('mcpKeyName');
        if (nameInput) nameInput.value = '';
        var expiresInput = document.getElementById('mcpKeyExpires');
        if (expiresInput) expiresInput.value = '';
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function mcpSaveRetrievalSettings() {
    if (!_mcpCurrentConfig) return;

    var rawChunksEl = document.querySelector('[data-key="web.mcp_retrieval_allow_raw_chunks"]');
    var includeTextEl = document.querySelector('[data-key="web.mcp_retrieval_include_text_default"]');
    if (includeTextEl && includeTextEl.checked && (!rawChunksEl || !rawChunksEl.checked)) {
        SharedUi.notify.show('Text standardmaessig mitsenden setzt erlaubte Rohchunks voraus.', 'error');
        includeTextEl.focus();
        return;
    }

    var rawChunksWereEnabled = _mcpCurrentConfig?.web?.mcp_retrieval_allow_raw_chunks === true;
    if (rawChunksEl && rawChunksEl.checked && !rawChunksWereEnabled) {
        var confirmed = await SharedUi.modal.confirm({
            message: 'Rohchunk-Volltext fuer externe MCP/API-Clients wirklich erlauben? Diese Einstellung kann vollstaendige Chunktexte ausgeben und sollte nur fuer trusted Agents aktiviert werden.'
        });
        if (!confirmed) return;
    }

    var result = await collectAndSaveConfig(_mcpCurrentConfig, {
        selector: '.mcp-retrieval-input[data-key], .mcp-retrieval-checkbox[data-key]',
        successMessage: 'Retrieval-only Einstellungen gespeichert',
    });
    if (result) _mcpCurrentConfig = result;
}

function mcpCopyKey() {
    if (!_mcpPendingCreatedKey) return;
    var keyText = _mcpPendingCreatedKey;

    var onSuccess = function() {
        _mcpPendingCreatedKey = null;
        var display = document.getElementById('mcpNewKeyDisplay');
        if (display) { display.classList.add('is-hidden'); display.innerHTML = ''; }
        SharedUi.notify.show('Key kopiert!', 'success');
    };

    var onFail = function() {
        SharedUi.notify.show('Key konnte nicht kopiert werden. Bitte manuell kopieren.', 'error');
    };

    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(keyText).then(onSuccess).catch(function() {
            _mcpCopyFallback(keyText, onSuccess, onFail);
        });
    } else {
        _mcpCopyFallback(keyText, onSuccess, onFail);
    }
}

function _mcpCopyFallback(text, onSuccess, onFail) {
    try {
        var ta = document.createElement('textarea');
        ta.className = 'mcp-copy-buffer';
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        var ok = document.execCommand('copy');
        ta.remove();
        if (ok) { onSuccess(); } else { onFail(); }
    } catch (e) {
        onFail();
    }
}

function mcpDismissPendingKey() {
    _mcpPendingCreatedKey = null;
    var display = document.getElementById('mcpNewKeyDisplay');
    if (display) { display.classList.add('is-hidden'); display.innerHTML = ''; }
}

async function mcpToggleServer(currentEnabled) {
    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/mcp/toggle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
            },
            body: JSON.stringify({ enabled: !currentEnabled }),
        });
        if (!resp.ok) {
            throw new Error(await _mcpResolveError(resp, 'Toggle fehlgeschlagen'));
        }
        SharedUi.notify.show(
            !currentEnabled ? 'MCP-Endpunkt aktiviert' : 'MCP-Endpunkt deaktiviert',
            'success'
        );
        // systemctl start ist asynchron — exponentiell pollen, damit der Render
        // nicht "aktiviert, aber Service nicht aktiv" anzeigt, waehrend systemd
        // den Service noch hochfaehrt. Beim Deaktivieren ist Polling unnoetig.
        if (!currentEnabled) {
            var delay = 500;
            for (var i = 0; i < 12; i++) {
                await new Promise(function(r) { setTimeout(r, delay); });
                var statusResp = await SharedUi.fetch.safeFetch(
                    '/api/admin/mcp/status',
                    { headers: { 'X-Requested-With': 'XMLHttpRequest' } }
                ).catch(function() { return null; });
                if (statusResp && statusResp.ok) {
                    var statusData = await statusResp.json();
                    if (statusData && statusData.service_running) break;
                }
                delay = Math.min(Math.round(delay * 1.5), 2000);
            }
        }
        await initMcpTab();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function mcpToggleKey(keyId) {
    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/mcp/keys/' + keyId + '/toggle', {
            method: 'PUT',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) {
            throw new Error(await _mcpResolveError(resp, 'Fehler'));
        }
        await initMcpTab();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function mcpDeleteKey(keyId, keyName) {
    try {
        var resp = await SharedUi.fetch.safeFetch('/api/admin/mcp/keys/' + keyId, {
            method: 'DELETE',
            headers: { 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) {
            throw new Error(await _mcpResolveError(resp, 'Fehler'));
        }
        SharedUi.notify.show('API-Key geloescht', 'success');
        await initMcpTab();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}
