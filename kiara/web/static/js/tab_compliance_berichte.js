/**
 * KIara Admin - Compliance > Berichte Sub-Tab
 */

let _cbInitSeq = 0;

const _CB_ACTIONS = new Set(['generate-vvt', 'generate-dsfa']);

function _handleCbClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_CB_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'generate-vvt':  generateVVT(); break;
        case 'generate-dsfa': generateDSFA(); break;
    }
}

function _bindCbActions() {
    if (document.body._cbActionsBound) return;
    document.body._cbActionsBound = true;
    document.body.addEventListener('click', _handleCbClick);
}

async function initComplianceBerichteTab() {
    _bindCbActions();
    _cbInitSeq++;
    const mySeq = _cbInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const dashboard = await SharedUi.fetch.safeFetch('/api/admin/compliance/dashboard').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });
        if (mySeq !== _cbInitSeq) return;
        renderComplianceBerichteTab(container, dashboard);
    } catch (e) {
        if (mySeq !== _cbInitSeq) return;
        container.innerHTML = '<div class="no-data">Compliance-Daten konnten nicht geladen werden.</div>';
    }
}

function renderComplianceBerichteTab(container, dashboard) {
    let html = '<div class="config-form">';
    html += '<h3>Compliance-Berichte</h3>';

    // KPIs
    const d = dashboard || {};
    html += '<div class="widget-row u-mb-20">';
    html += _complianceKpi('Anfragen (30 Tage)', d.total_queries_30d || 0);
    html += _complianceKpi('Cloud-Anteil', (d.cloud_percentage || 0).toFixed(1) + '%');
    html += _complianceKpi('Anonymisierte Entitaeten', d.entities_anonymized_30d || 0);
    html += _complianceKpi('Aktive Nutzer', d.unique_users_30d || 0);
    html += '</div>';

    // VVT
    html += '<div class="config-section"><h4>Verzeichnis der Verarbeitungstaetigkeiten (VVT)</h4>';
    html += '<p class="u-text-xs-secondary u-mb-12">DSGVO Art. 30 — Automatisch generiert aus Systemkonfiguration.</p>';
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-action="generate-vvt">VVT generieren</button>';
    html += '</div>';
    html += '<div id="vvtPreview" class="u-mt-12"></div>';
    html += '</div>';

    // DSFA
    html += '<div class="config-section"><h4>Datenschutz-Folgenabschaetzung (DSFA)</h4>';
    html += '<p class="u-text-xs-secondary u-mb-12">DSGVO Art. 35 — Risikobewertung des RAG-Systems.</p>';
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-action="generate-dsfa">DSFA generieren</button>';
    html += '</div>';
    html += '<div id="dsfaPreview" class="u-mt-12"></div>';
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;
}

function _complianceKpi(label, value) {
    return `<div class="widget-card cmp-kpi-card"><div class="cmp-kpi-inner">
        <div class="cmp-kpi-label">${SharedUi.runtime.escapeHtml(label)}</div>
        <div class="cmp-kpi-value">${SharedUi.runtime.escapeHtml(String(value))}</div>
    </div></div>`;
}

async function generateVVT() {
    const preview = document.getElementById('vvtPreview');
    if (!preview) return;
    preview.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/compliance/vvt');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        let html = '<div class="cmp-preview-panel">';
        html += '<pre class="cmp-preview-pre">' + SharedUi.runtime.escapeHtml(JSON.stringify(data, null, 2)) + '</pre>';
        html += '</div>';
        preview.innerHTML = html;
    } catch (e) {
        preview.innerHTML = '<div class="no-data">VVT konnte nicht generiert werden.</div>';
    }
}

async function generateDSFA() {
    const preview = document.getElementById('dsfaPreview');
    if (!preview) return;
    preview.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/compliance/dsfa');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        let html = '<div class="cmp-preview-panel">';
        html += '<pre class="cmp-preview-pre">' + SharedUi.runtime.escapeHtml(JSON.stringify(data, null, 2)) + '</pre>';
        html += '</div>';
        preview.innerHTML = html;
    } catch (e) {
        preview.innerHTML = '<div class="no-data">DSFA konnte nicht generiert werden.</div>';
    }
}
