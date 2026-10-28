/**
 * KIara Admin - Compliance > Berechtigungsmatrix Sub-Tab
 */

let _cmInitSeq = 0;

async function initComplianceMatrixTab() {
    _cmInitSeq++;
    const mySeq = _cmInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/compliance/permissions-matrix');
        if (mySeq !== _cmInitSeq) return;
        if (!resp.ok) throw new Error('Laden fehlgeschlagen');
        const data = await resp.json();
        if (mySeq !== _cmInitSeq) return;
        renderComplianceMatrixTab(container, data);
    } catch (e) {
        if (mySeq !== _cmInitSeq) return;
        container.innerHTML = '<div class="no-data">Berechtigungsmatrix konnte nicht geladen werden.</div>';
    }
}

function renderComplianceMatrixTab(container, data) {
    const matrix = data.matrix || data || {};
    const groups = matrix.groups || [];

    // Spalten aus groups[].datasources/tools ableiten (Backend liefert keine Top-Level-Listen)
    // Key = ds.id (PK), nicht ds.name — gleichnamige Datenquellen wuerden sonst kollabieren
    const dsMap = new Map();
    groups.forEach(g => (g.datasources || []).forEach(ds => {
        const key = (ds && ds.id != null) ? ds.id : (ds.name || ds);
        if (!dsMap.has(key)) dsMap.set(key, ds);
    }));
    const datasources = Array.from(dsMap.values());

    const toolSet = new Set();
    groups.forEach(g => (g.tools || []).forEach(t => toolSet.add(t.name || t)));
    const tools = Array.from(toolSet);

    let html = '<div class="config-form">';
    html += '<h3>Berechtigungsmatrix</h3>';
    html += '<p class="cmp-intro">Uebersicht: Welche LDAP-Gruppe hat Zugriff auf welche Datenquellen und ERP-Tools? (Nur Anzeige)</p>';

    if (groups.length === 0) {
        html += '<div class="no-data">Keine Gruppen vorhanden.</div>';
    } else {
        html += '<div class="config-section cmp-matrix-wrap">';
        html += '<table class="cmp-matrix-table">';

        // Header
        html += '<tr class="u-cell-border-strong"><th class="u-cell-left-8">Gruppe</th><th class="u-cell-center-8">Mitglieder</th>';
        datasources.forEach(ds => {
            html += `<th class="cmp-matrix-th-vert">${SharedUi.runtime.escapeHtml(ds.name || ds)}</th>`;
        });
        tools.forEach(t => {
            html += `<th class="cmp-matrix-th-vert cmp-matrix-th-tool">${SharedUi.runtime.escapeHtml(t.name || t)}</th>`;
        });
        html += '</tr>';

        // Rows
        groups.forEach(g => {
            html += `<tr class="u-cell-border">`;
            html += `<td class="cmp-matrix-td-group">${SharedUi.runtime.escapeHtml(g.name || '')}</td>`;
            html += `<td class="u-cell-center-pad">${g.member_count || 0}</td>`;
            const gDs = g.datasources || [];
            datasources.forEach(ds => {
                const dsId = (ds && ds.id != null) ? ds.id : null;
                const dsName = ds.name || ds;
                const has = (dsId != null)
                    ? gDs.some(d => d && d.id === dsId)
                    : (gDs.includes(dsName) || gDs.some(d => (d.name || d) === dsName));
                html += `<td class="cmp-matrix-cell">${has ? '<span class="u-c-bullish">&#10003;</span>' : '<span class="u-c-secondary">&#8212;</span>'}</td>`;
            });
            const gTools = g.tools || [];
            tools.forEach(t => {
                const tName = t.name || t;
                const has = gTools.includes(tName) || gTools.some(t => (t.name || t) === tName);
                html += `<td class="cmp-matrix-cell">${has ? '<span class="u-c-bullish">&#10003;</span>' : '<span class="u-c-secondary">&#8212;</span>'}</td>`;
            });
            html += '</tr>';
        });

        html += '</table></div>';
    }

    html += '</div>';
    container.innerHTML = html;
}
