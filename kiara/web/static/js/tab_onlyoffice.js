/**
 * KIara Admin - OnlyOffice Tab
 * OnlyOffice Document Server configuration.
 */

let ooConfig = null;

const _OO_ACTIONS = new Set([
    'save-onlyoffice-config',
    'test-onlyoffice',
    'refresh-onlyoffice-discovery',
    'delete-onlyoffice-config',
]);

function _handleOnlyOfficeClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_OO_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'save-onlyoffice-config':   SharedUi.runtime.withButtonLock(el, saveOnlyOfficeConfig); break;
        case 'test-onlyoffice':          testOnlyOffice(); break;
        case 'refresh-onlyoffice-discovery': refreshOnlyOfficeDiscovery(); break;
        case 'delete-onlyoffice-config': deleteOnlyOfficeConfig(); break;
    }
}

function _bindOnlyOfficeActions() {
    if (document.body._onlyOfficeActionsBound) return;
    document.body._onlyOfficeActionsBound = true;
    document.body.addEventListener('click', _handleOnlyOfficeClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-onlyoffice-actions-bound', 'true');
}

async function initOnlyOfficeTab() {
    _bindOnlyOfficeActions();
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;
    if (filterBar) filterBar.classList.add('u-hidden');

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const data = await loadConfigFromAPI('/api/admin/onlyoffice/config');
        ooConfig = data;
        renderOnlyOfficeForm(container);
    } catch (e) {
        container.innerHTML = '<div class="no-data">OnlyOffice-Konfiguration konnte nicht geladen werden.</div>';
    }
}

function renderOnlyOfficeForm(container) {
    const cfg = ooConfig || {};
    let html = '<div class="config-form">';
    html += '<h3>OnlyOffice Document Server</h3>';

    html += '<div class="config-section"><h4>Verbindung</h4><div class="config-fields">';
    html += `<div class="config-field"><label>Document Server URL</label>
        <input type="text" class="config-input" id="oo_url"
            value="${SharedUi.runtime.escapeAttr(cfg.document_server_url || '')}"
            placeholder="https://onlyoffice.domain.local"></div>`;
    html += `<div class="config-field"><label>JWT Secret</label>
        <input type="password" class="config-input" id="oo_secret"
            value="${SharedUi.runtime.escapeAttr(cfg.jwt_secret || '')}"
            placeholder="Geheimer Schluessel"></div>`;
    html += `<div class="config-field"><label>Externe WOPI-Basis-URL</label>
        <input type="text" class="config-input" id="oo_wopi_external_base_url"
            value="${SharedUi.runtime.escapeAttr(cfg.wopi_external_base_url || '')}"
            placeholder="https://kiara.domain.local"></div>`;
    const proofChecked = (cfg.proof_validation_enabled !== false) ? 'checked' : '';
    html += `<div class="config-field"><label>
        <input type="checkbox" id="oo_proof_validation_enabled" ${proofChecked}> WOPI-Proof-Header validieren
    </label></div>`;
    html += '</div></div>';

    // Actions
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-action="save-onlyoffice-config">Speichern</button>';
    if (ooConfig) {
        html += '<button class="action-btn" data-action="test-onlyoffice">Verbindung testen</button>';
        html += '<button class="action-btn" data-action="refresh-onlyoffice-discovery">Discovery aktualisieren</button>';
        html += '<button class="action-btn danger" data-action="delete-onlyoffice-config">Konfiguration loeschen</button>';
    }
    html += '<span id="ooTestResult" class="oo-test-result"></span>';
    html += '</div>';

    // Info
    html += '<div class="config-section u-mt-24"><h4>Hinweise</h4>';
    html += '<ul class="oo-hints-list">';
    html += '<li>Der OnlyOffice Document Server muss als Docker-Container laufen</li>';
    html += '<li>Standard-Port: 8443 (HTTPS) oder 8080 (HTTP)</li>';
    html += '<li>Das JWT-Secret muss mit der Konfiguration des Document Servers uebereinstimmen</li>';
    html += '<li>Unterstuetzte Formate: DOCX, XLSX, PPTX, ODT, ODS, ODP, TXT, CSV</li>';
    html += '</ul></div>';

    html += '</div>';
    container.innerHTML = html;
}

function _normalizeOnlyOfficeHttpUrl(rawValue, fieldName, required = false) {
    const value = String(rawValue || '').trim();
    if (!value) {
        if (required) throw new Error(`${fieldName} erforderlich`);
        return '';
    }

    let parsed;
    try {
        parsed = new URL(value);
    } catch (e) {
        throw new Error(`${fieldName} muss eine gueltige http(s)-URL sein`);
    }

    if ((parsed.protocol !== 'http:' && parsed.protocol !== 'https:') || !parsed.hostname) {
        throw new Error(`${fieldName} muss eine http(s)-URL mit Host sein`);
    }
    if (parsed.username || parsed.password) {
        throw new Error(`${fieldName} darf keine Credentials enthalten`);
    }
    if (parsed.search || parsed.hash) {
        throw new Error(`${fieldName} darf keine Query/Fragment enthalten`);
    }

    parsed.pathname = parsed.pathname.replace(/\/+$/, '');
    return parsed.toString().replace(/\/$/, '');
}

async function saveOnlyOfficeConfig() {
    const url = document.getElementById('oo_url')?.value || '';
    const secret = document.getElementById('oo_secret')?.value || '';
    const wopiExternalBaseUrl = document.getElementById('oo_wopi_external_base_url')?.value || '';
    const proofValidationEnabled = document.getElementById('oo_proof_validation_enabled')?.checked !== false;

    let normalizedUrl;
    let normalizedWopiExternalBaseUrl;
    try {
        normalizedUrl = _normalizeOnlyOfficeHttpUrl(url, 'Document Server URL', true);
        normalizedWopiExternalBaseUrl = _normalizeOnlyOfficeHttpUrl(wopiExternalBaseUrl, 'Externe WOPI-Basis-URL');
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
        return;
    }
    if (!secret) { SharedUi.notify.show('JWT-Secret erforderlich', 'error'); return; }

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/onlyoffice/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                document_server_url: normalizedUrl,
                jwt_secret: secret,
                wopi_external_base_url: normalizedWopiExternalBaseUrl,
                proof_validation_enabled: proofValidationEnabled,
            }),
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || errData.message || 'Speichern fehlgeschlagen');
        }
        const reloaded = await loadConfigFromAPI('/api/admin/onlyoffice/config');
        ooConfig = reloaded;
        const container = document.getElementById('contentContainer');
        if (container) renderOnlyOfficeForm(container);
        SharedUi.notify.show('OnlyOffice-Konfiguration gespeichert', 'success');
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
    }
}

async function refreshOnlyOfficeDiscovery() {
    const resultEl = document.getElementById('ooTestResult');
    if (resultEl) resultEl.textContent = 'Aktualisiere...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/onlyoffice/refresh-discovery', { method: 'POST' });
        const data = await resp.json().catch(() => ({ detail: resp.statusText }));
        if (resultEl) {
            resultEl.classList.remove('u-c-success', 'u-c-danger');
            if (resp.ok) {
                resultEl.textContent = data.message || 'Discovery aktualisiert';
                resultEl.classList.add('u-c-success');
            } else {
                resultEl.textContent = `Fehler: ${data.detail || data.message || resp.statusText}`;
                resultEl.classList.add('u-c-danger');
            }
        }
    } catch (e) {
        if (resultEl) {
            resultEl.classList.remove('u-c-success');
            resultEl.textContent = 'Discovery-Aktualisierung fehlgeschlagen';
            resultEl.classList.add('u-c-danger');
        }
    }
}

async function testOnlyOffice() {
    const resultEl = document.getElementById('ooTestResult');
    if (resultEl) resultEl.textContent = 'Teste...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/onlyoffice/test', { method: 'POST' });
        const data = await resp.json().catch(() => ({ detail: resp.statusText }));
        if (resultEl) {
            resultEl.classList.remove('u-c-success', 'u-c-danger');
            if (resp.ok) {
                resultEl.textContent = data.message;
                resultEl.classList.add('u-c-success');
            } else {
                resultEl.textContent = `Fehler: ${data.detail || data.message || resp.statusText}`;
                resultEl.classList.add('u-c-danger');
            }
        }
    } catch (e) {
        if (resultEl) {
            resultEl.classList.remove('u-c-success');
            resultEl.textContent = 'Test fehlgeschlagen';
            resultEl.classList.add('u-c-danger');
        }
    }
}

async function deleteOnlyOfficeConfig() {
    await deleteConfigFromAPI({
        endpoint: '/api/admin/onlyoffice/config',
        confirmMessage: 'OnlyOffice-Konfiguration wirklich loeschen?',
        successMessage: 'OnlyOffice-Konfiguration geloescht',
        onSuccess: () => initOnlyOfficeTab(),
    });
}
