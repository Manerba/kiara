/**
 * KIara Admin - Allgemein > Web Sub-Tab
 * Netzwerkeinstellungen: web.host, web.port, web.admin_port
 */

let allgemeinWebConfig = null;

const _ALLG_WEB_ACTIONS = new Set([
    'save-allgemein-web',
    'reset-allgemein-web',
]);

function _handleAllgWebClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_ALLG_WEB_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'save-allgemein-web':  saveAllgemeinWeb(); break;
        case 'reset-allgemein-web': initAllgemeinWebTab(); break;
    }
}

function _bindAllgWebActions() {
    if (document.body._allgWebActionsBound) return;
    document.body._allgWebActionsBound = true;
    document.body.addEventListener('click', _handleAllgWebClick);
}

async function initAllgemeinWebTab() {
    _bindAllgWebActions();
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const data = await loadConfigFromAPI('/api/admin/settings');
        allgemeinWebConfig = data;
        renderAllgemeinWebForm(container);
    } catch (e) {
        container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
    }
}

function renderAllgemeinWebForm(container) {
    if (!allgemeinWebConfig) {
        container.innerHTML = '<div class="no-data">Keine Konfiguration vorhanden.</div>';
        return;
    }

    const cfg = allgemeinWebConfig;
    let html = '<div class="config-form">';
    html += '<h3>Web-Einstellungen</h3>';

    html += '<div class="config-section"><h4>Netzwerk</h4>';
    html += '<p class="config-hint"><b>Aenderungen an Host und Ports erfordern einen Neustart der Dienste (kiara, kiara-admin).</b></p>';
    html += '<div class="config-fields">';
    html += _webField('web.host', 'Host', cfg.web?.host || '0.0.0.0', 'text');
    html += _webField('web.port', 'Port (Chat-UI)', cfg.web?.port || 8501, 'number');
    html += _webField('web.admin_port', 'Admin-Port', cfg.web?.admin_port || 8502, 'number');
    html += '</div></div>';

    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-action="save-allgemein-web">Speichern</button>';
    html += '<button class="action-btn" data-action="reset-allgemein-web">Zuruecksetzen</button>';
    html += '</div>';
    html += '</div>';
    container.innerHTML = html;
}

function _webField(key, label, value, type) {
    return `<div class="config-field">
        <label>${SharedUi.runtime.escapeHtml(label)}</label>
        <input type="${type}" data-key="${SharedUi.runtime.escapeAttr(key)}" value="${SharedUi.runtime.escapeAttr(String(value))}" class="config-input">
    </div>`;
}

async function saveAllgemeinWeb() {
    if (!allgemeinWebConfig) return;
    const result = await collectAndSaveConfig(allgemeinWebConfig, {
        successMessage: 'Web-Einstellungen gespeichert',
    });
    if (result) allgemeinWebConfig = result;
}
