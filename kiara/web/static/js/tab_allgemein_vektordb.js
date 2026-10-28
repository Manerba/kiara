/**
 * KIara Admin - Allgemein > VektorDB Sub-Tab
 * Backend: Qdrant
 */

let allgemeinVektordbConfig = null;
const VDB_PASSWORD_MASK = '********';  // Muss mit PASSWORD_MASK aus constants.py uebereinstimmen

const _ALLG_VDB_ACTIONS = new Set([
    'test-vectordb-connection',
    'save-allgemein-vektordb',
    'reset-allgemein-vektordb',
]);

function _handleAllgVdbClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_ALLG_VDB_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'test-vectordb-connection': testVectordbConnection(); break;
        case 'save-allgemein-vektordb':  saveAllgemeinVektordb(); break;
        case 'reset-allgemein-vektordb': initAllgemeinVektordbTab(); break;
    }
}

function _bindAllgVdbActions() {
    if (document.body._allgVdbActionsBound) return;
    document.body._allgVdbActionsBound = true;
    document.body.addEventListener('click', _handleAllgVdbClick);
}

async function initAllgemeinVektordbTab() {
    _bindAllgVdbActions();
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const data = await loadConfigFromAPI('/api/admin/settings');
        allgemeinVektordbConfig = data;
        renderAllgemeinVektordbForm(container);
    } catch (e) {
        container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
    }
}

function renderAllgemeinVektordbForm(container) {
    if (!allgemeinVektordbConfig) {
        container.innerHTML = '<div class="no-data">Keine Konfiguration vorhanden.</div>';
        return;
    }

    const cfg = allgemeinVektordbConfig;
    let html = '<div class="config-form">';
    html += '<h3>VektorDB-Einstellungen</h3>';

    // Qdrant-spezifische Felder
    html += '<div class="config-section" id="vdb-qdrant">';
    html += '<h4>Qdrant</h4>';
    html += '<p class="config-hint"><b>Aenderungen erfordern einen Neustart des Dienstes.</b> Embedding-Pfade nutzen den bestehenden Connection-Pool weiter, bis kiara/kiara-worker neu gestartet werden.</p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field">
        <label>Server-URL</label>
        <input type="text" data-key="vectordb.qdrant_url" value="${SharedUi.runtime.escapeAttr(cfg.vectordb?.qdrant_url || 'http://localhost:6333')}" class="config-input" placeholder="http://vectordb-server:6333">
    </div>`;
    html += `<div class="config-field">
        <label>API-Key (optional)</label>
        <input type="password" data-key="vectordb.qdrant_api_key" value="${SharedUi.runtime.escapeAttr(cfg.vectordb?.qdrant_api_key || '')}" class="config-input" placeholder="Leer lassen wenn nicht noetig">
    </div>`;
    html += `<div class="config-field">
        <label>gRPC verwenden</label>
        <input type="checkbox" data-key="vectordb.qdrant_grpc" ${cfg.vectordb?.qdrant_grpc ? 'checked' : ''} class="config-checkbox">
        <span class="u-text-xs-secondary">Schneller fuer grosse Batches</span>
    </div>`;
    html += '</div></div>';

    // Gemeinsame Felder
    html += '<div class="config-section"><h4>Allgemein</h4><div class="config-fields">';
    html += `<div class="config-field">
        <label>Vektor-Dimension</label>
        <input type="number" data-key="vectordb.vector_size" value="${cfg.vectordb?.vector_size || 768}" class="config-input" min="1" max="8192">
        <span class="u-text-xs-secondary">Wird vom Embedding-Modell bestimmt (nomic-embed-text: 768)</span>
    </div>`;
    html += '</div></div>';

    html += '<div class="config-actions">';
    html += '<button class="action-btn" data-action="test-vectordb-connection">Verbindung testen</button>';
    html += '<button class="action-btn primary" data-action="save-allgemein-vektordb">Speichern</button>';
    html += '<button class="action-btn" data-action="reset-allgemein-vektordb">Zuruecksetzen</button>';
    html += '</div>';
    html += '<div id="vdb-test-result" class="vdb-test-result"></div>';
    html += '</div>';
    container.innerHTML = html;
}

async function testVectordbConnection() {
    const resultEl = document.getElementById('vdb-test-result');
    if (resultEl) resultEl.innerHTML = '<span class="u-c-secondary">Teste Verbindung...</span>';

    const body = { backend: 'qdrant' };
    const urlInput = document.querySelector('[data-key="vectordb.qdrant_url"]');
    const keyInput = document.querySelector('[data-key="vectordb.qdrant_api_key"]');
    body.qdrant_url = urlInput ? urlInput.value : '';
    // Masken-Wert nicht als echten Key senden — Backend nutzt dann gespeicherten Wert
    const keyValue = keyInput ? keyInput.value : '';
    if (keyValue && keyValue !== VDB_PASSWORD_MASK) body.qdrant_api_key = keyValue;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/vectordb/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify(body),
        });
        const data = await resp.json().catch(() => ({}));
        if (resp.ok) {
            resultEl.innerHTML = `<span class="u-c-success">${SharedUi.runtime.escapeHtml(data.message)}</span>`;
        } else {
            resultEl.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(data.detail || data.message || resp.statusText)}</span>`;
        }
    } catch (e) {
        if (resultEl) resultEl.innerHTML = `<span class="u-c-danger">Verbindungstest fehlgeschlagen: ${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    }
}

async function saveAllgemeinVektordb() {
    if (!allgemeinVektordbConfig) return;
    const result = await collectAndSaveConfig(allgemeinVektordbConfig, {
        successMessage: 'VektorDB-Einstellungen gespeichert. Neustart von kiara/kiara-worker erforderlich, damit Embedding-Pfade die neue Verbindung nutzen.',
    });
    if (result) allgemeinVektordbConfig = result;
}
