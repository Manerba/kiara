/**
 * KIara Admin - Tools > Workflows Sub-Tab
 */

let _workflowsInitSeq = 0;

async function initWorkflowsTab() {
    _workflowsInitSeq++;
    const mySeq = _workflowsInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const [templatesResp, exportsResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/workflows/templates').then(r => { if (!r.ok) throw new Error('Laden fehlgeschlagen'); return r.json(); }),
            SharedUi.fetch.safeFetch('/api/workflows/exports').then(r => { if (!r.ok) throw new Error('Laden fehlgeschlagen'); return r.json(); }),
        ]);
        if (mySeq !== _workflowsInitSeq) return;
        renderWorkflowsTab(container, templatesResp, exportsResp);
    } catch (e) {
        if (mySeq !== _workflowsInitSeq) return;
        container.innerHTML = '<div class="no-data">Workflows konnten nicht geladen werden.</div>';
    }
}

function renderWorkflowsTab(container, templatesData, exportsData) {
    const templates = Array.isArray(templatesData?.templates) ? templatesData.templates : (Array.isArray(templatesData) ? templatesData : []);
    const exports = Array.isArray(exportsData?.exports) ? exportsData.exports : (Array.isArray(exportsData) ? exportsData : []);

    let html = '<div class="config-form">';

    // Templates
    html += '<h3>Dokumenten-Templates</h3>';
    if (templates.length === 0) {
        html += '<div class="no-data">Keine Templates vorhanden.</div>';
    } else {
        html += '<div class="config-section">';
        html += '<table class="u-detail-table">';
        html += '<tr class="u-cell-border-strong"><th class="u-cell-left-8">Name</th><th class="u-cell-left-8">Beschreibung</th><th class="u-cell-left-8">Variablen</th></tr>';
        templates.forEach(t => {
            const vars = (t.variables || []).join(', ') || '-';
            html += `<tr class="u-cell-border"><td class="u-cell-left-8-bold">${SharedUi.runtime.escapeHtml(t.name || '')}</td><td class="u-cell-left-8 u-c-secondary">${SharedUi.runtime.escapeHtml(t.description || '')}</td><td class="u-cell-left-8 u-fs-11 u-mono">${SharedUi.runtime.escapeHtml(vars)}</td></tr>`;
        });
        html += '</table></div>';
    }

    // Exports
    html += '<h3 class="u-mt-24">Exporte</h3>';
    if (exports.length === 0) {
        html += '<div class="no-data">Keine Exporte vorhanden.</div>';
    } else {
        html += '<div class="config-section">';
        html += '<table class="u-detail-table">';
        html += '<tr class="u-cell-border-strong"><th class="u-cell-left-8">Dateiname</th><th class="u-cell-left-8">Format</th><th class="u-cell-right-8">Groesse</th><th class="u-cell-left-8">Erstellt</th><th class="u-cell-center-8">Aktionen</th></tr>';
        exports.forEach(e => {
            const size = e.size_bytes ? (e.size_bytes / 1024).toFixed(1) + ' KB' : '-';
            const created = e.created_at || '-';
            html += `<tr class="u-cell-border">
                <td class="u-cell-left-8">${SharedUi.runtime.escapeHtml(e.filename || '')}</td>
                <td class="u-cell-left-8">${SharedUi.runtime.escapeHtml(e.format || '')}</td>
                <td class="u-cell-right-8">${SharedUi.runtime.escapeHtml(size)}</td>
                <td class="u-cell-left-8 u-fs-12">${SharedUi.runtime.escapeHtml(created)}</td>
                <td class="u-cell-center-pad">
                    <div class="action-buttons">
                        <button class="action-btn-sm" data-wf-download="${SharedUi.runtime.escapeAttr(e.filename)}">Download</button>
                        <button class="action-btn-sm danger" data-wf-delete="${SharedUi.runtime.escapeAttr(e.filename)}">Loeschen</button>
                    </div>
                </td>
            </tr>`;
        });
        html += '</table></div>';
    }

    html += '</div>';
    container.innerHTML = html;

    // Event-Delegation: Export-Buttons (data-* statt Inline-JS)
    container.querySelectorAll('[data-wf-download]').forEach(btn => {
        btn.addEventListener('click', function() { downloadWorkflowExport(this.dataset.wfDownload); });
    });
    container.querySelectorAll('[data-wf-delete]').forEach(btn => {
        btn.addEventListener('click', function() { deleteWorkflowExport(this.dataset.wfDelete); });
    });
}

function downloadWorkflowExport(filename) {
    window.open(`/api/workflows/exports/${encodeURIComponent(filename)}/download`, '_blank');
}

async function deleteWorkflowExport(filename) {
    if (!await SharedUi.modal.confirm({ message: `Export "${filename}" wirklich loeschen?` })) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/workflows/exports/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Loeschen fehlgeschlagen');
        SharedUi.notify.show('Export geloescht', 'success');
        initWorkflowsTab();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}
