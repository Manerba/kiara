/**
 * KIara Admin - Compliance > Audit-Trail Sub-Tab
 * Nutzt ExpandableTable mit server-seitiger Pagination + Lazy-Load Detail.
 */

let auditTable = null;
let auditFilters = { chain_type: '', routed_to: '' };
let _auditInitSeq = 0;

const _AUDIT_ACTIONS = new Set([
    'set-audit-filter',
    'load-audit-filtered',
    'open-audit-detail',
]);

const _AUDIT_ROW_ACTIONS = new Set([
    'open-audit-detail',
]);

function _handleAuditAction(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_AUDIT_ACTIONS.has(action)) return;

    if (_AUDIT_ROW_ACTIONS.has(action)) {
        e.stopPropagation();
    }

    switch (action) {
        case 'set-audit-filter':
            setAuditFilter(el.dataset.filterField, el.dataset.filterValue, el);
            break;
        case 'load-audit-filtered':
            loadAuditFiltered();
            break;
        case 'open-audit-detail':
            // Link-Default (Navigation via href target=_blank) bleibt aktiv,
            // stopPropagation verhindert nur den Row-Expand darunter.
            break;
    }
}

function _handleAuditKeydown(e) {
    if (e.key !== 'Enter') return;
    const el = e.target.closest('[data-keydown-action]');
    if (!el) return;
    const action = el.dataset.keydownAction;
    if (action === 'load-audit-filtered') loadAuditFiltered();
}

function _bindAuditActions() {
    const container = document.getElementById('contentContainer');
    if (container && !container._auditActionsBound) {
        container._auditActionsBound = true;
        container.setAttribute('data-audit-actions-bound', 'true');
        container.addEventListener('click', _handleAuditAction);
        container.addEventListener('keydown', _handleAuditKeydown);
    }
}

function modeBadge(mode) {
    const map = {
        standard: '<span class="badge badge-default">Standard</span>',
        unchained: '<span class="badge badge-warning">Unchained</span>',
        agent: '<span class="badge badge-purple">Agent</span>',
    };
    if (typeof mode === 'string' && Object.prototype.hasOwnProperty.call(map, mode)) {
        return map[mode];
    }
    return SharedUi.runtime.escapeHtml(mode || "\u2014");
}

function routeBadge(route) {
    const map = {
        local: '<span class="badge badge-default">Lokal</span>',
        cloud: '<span class="badge badge-warning">Cloud</span>',
    };
    if (typeof route === 'string' && route.startsWith('mcp:')) {
        return '<span class="badge badge-default">MCP</span>';
    }
    if (typeof route === 'string' && Object.prototype.hasOwnProperty.call(map, route)) {
        return map[route];
    }
    return SharedUi.runtime.escapeHtml(route || "\u2014");
}

function setAuditFilter(field, value, btn) {
    auditFilters[field] = value;
    if (btn) {
        btn.closest('.filter-buttons')
           .querySelectorAll('.filter-btn')
           .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }
    loadAuditFiltered();
}

function loadAuditFiltered() {
    if (!auditTable) return;
    const params = {};
    if (auditFilters.chain_type) params.chain_type = auditFilters.chain_type;
    if (auditFilters.routed_to) params.routed_to = auditFilters.routed_to;
    const dateFrom = document.getElementById('auditDateFrom')?.value;
    const dateTo = document.getElementById('auditDateTo')?.value;
    const search = document.getElementById('auditSearch')?.value;
    if (dateFrom) params.date_from = dateFrom;
    if (dateTo) params.date_to = dateTo;
    if (search) params.search_text = search;
    auditTable.pagination.currentPage = 1;
    auditTable.loadData(params);
}

// Synchroner Placeholder — gibt sofort HTML zurueck
function auditDetailPlaceholder(row) {
    return `<div class="audit-detail" data-audit-id="${SharedUi.runtime.escapeAttr(String(row.id))}">
        <div class="detail-loading"><div class="spinner"></div> Lade Details...</div>
    </div>`;
}

function renderPromptSection(fullPrompt) {
    if (!fullPrompt) return '<pre class="long-text">Nicht verfuegbar</pre>';

    // Versuch JSON zu parsen (Agent-Modus)
    try {
        const conv = JSON.parse(fullPrompt);

        // Ollama-Agent: Array von {type, prompt}
        if (Array.isArray(conv)) {
            return conv.map((entry, i) => `
                <div class="conversation-step">
                    <span class="badge badge-default">${SharedUi.runtime.escapeHtml(entry.type || 'step')} ${i + 1}</span>
                    <pre class="long-text">${SharedUi.runtime.escapeHtml(entry.prompt || '')}</pre>
                </div>
            `).join('');
        }

        // Cloud-Agent: {system, messages, tools}
        if (conv.messages) {
            let html = '';
            if (conv.system) {
                html += `<div class="conversation-step">
                    <span class="badge badge-purple">system</span>
                    <pre class="long-text">${SharedUi.runtime.escapeHtml(conv.system)}</pre>
                </div>`;
            }
            for (const msg of conv.messages) {
                const role = msg.role || 'unknown';
                const badge = role === 'user' ? 'badge-default'
                            : role === 'assistant' ? 'badge-purple'
                            : 'badge-warning';
                const content = typeof msg.content === 'string'
                    ? msg.content
                    : JSON.stringify(msg.content, null, 2);
                html += `<div class="conversation-step">
                    <span class="badge ${badge}">${SharedUi.runtime.escapeHtml(role)}</span>
                    <pre class="long-text">${SharedUi.runtime.escapeHtml(content)}</pre>
                </div>`;
            }
            return html;
        }
    } catch (e) {
        // Kein JSON — Klartext (Standard/Unchained)
    }

    return `<pre class="long-text">${SharedUi.runtime.escapeHtml(fullPrompt)}</pre>`;
}

// Async Nachladen via onRowExpanded-Callback
async function onAuditRowExpanded(rowId, detailElement) {
    const container = detailElement.querySelector('.audit-detail');
    if (!container || container.dataset.loaded === "true") return;
    if (container.dataset.loading === "true") return;
    container.dataset.loading = "true";
    const mySeq = _auditInitSeq;

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/compliance/audit/${container.dataset.auditId}`);
        if (mySeq !== _auditInitSeq) return;
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const detail = await resp.json();
        if (mySeq !== _auditInitSeq) return;
        container.dataset.loaded = "true";

        container.innerHTML = `
            <div class="detail-section">
                <h4>Prompt an KI-Backend</h4>
                ${renderPromptSection(detail.full_prompt)}
            </div>
            ${detail.tool_calls?.length ? `
            <div class="detail-section">
                <h4>Tool-Calls (${detail.tool_calls.length})</h4>
                <div class="tool-calls-list">
                    ${detail.tool_calls.map(tc => `
                        <div class="tool-call-item">
                            <span class="tool-name">${SharedUi.runtime.escapeHtml(tc.tool)}</span>
                            <span class="tool-iteration">Iteration ${tc.iteration}</span>
                            <pre class="tool-params">${SharedUi.runtime.escapeHtml(JSON.stringify(tc.params, null, 2))}</pre>
                            ${tc.result ? `<pre class="tool-result">${SharedUi.runtime.escapeHtml(tc.result)}</pre>` : ""}
                        </div>
                    `).join("")}
                </div>
            </div>` : ""}
            <div class="detail-section">
                <h4>KI-Output</h4>
                <pre class="long-text">${SharedUi.runtime.escapeHtml(detail.full_answer || "Nicht verfuegbar")}</pre>
            </div>
            <div class="detail-section detail-meta">
                <span>Modell: ${SharedUi.runtime.escapeHtml(detail.model_used || "\u2014")}</span>
                <span>Chunks: ${detail.sources_count || 0}</span>
                <span>Anonymisiert: ${detail.entities_anonymized || 0} Entities</span>
                <span>Dauer: ${detail.duration_ms != null ? detail.duration_ms + 'ms' : '\u2014'}</span>
            </div>
            ${detail.cloud_audit ? `
            <div class="detail-section">
                <h4>Cloud-Audit</h4>
                <div class="detail-meta u-flex-wrap">
                    <span>Provider: ${SharedUi.runtime.escapeHtml(detail.cloud_audit.provider || "\u2014")}</span>
                    <span>Modell: ${SharedUi.runtime.escapeHtml(detail.cloud_audit.model || "\u2014")}</span>
                    <span>Tokens: ${detail.cloud_audit.tokens_in || 0} in / ${detail.cloud_audit.tokens_out || 0} out</span>
                    <span>Entities: ${detail.cloud_audit.cloud_entities || 0}</span>
                </div>
            </div>` : ""}
        `;
    } catch (e) {
        if (mySeq !== _auditInitSeq) return;
        container.dataset.loading = "";
        container.innerHTML = `<div class="detail-error">Fehler beim Laden: ${SharedUi.runtime.escapeHtml(e.message)}</div>`;
    }
}


async function initComplianceAuditTab() {
    _auditInitSeq++;
    const container = document.getElementById('contentContainer');
    if (!container) return;

    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    _bindAuditActions();

    container.innerHTML = `
    <div class="filter-bar">
        <div class="filter-group">
            <span class="filter-label">Von:</span>
            <input type="date" id="auditDateFrom">
        </div>
        <div class="filter-group">
            <span class="filter-label">Bis:</span>
            <input type="date" id="auditDateTo">
        </div>
        <div class="filter-group">
            <span class="filter-label">Modus:</span>
            <div class="filter-buttons">
                <button class="filter-btn active" data-action="set-audit-filter" data-filter-field="chain_type" data-filter-value="">Alle</button>
                <button class="filter-btn" data-action="set-audit-filter" data-filter-field="chain_type" data-filter-value="standard">Standard</button>
                <button class="filter-btn" data-action="set-audit-filter" data-filter-field="chain_type" data-filter-value="unchained">Unchained</button>
                <button class="filter-btn" data-action="set-audit-filter" data-filter-field="chain_type" data-filter-value="agent">Agent</button>
            </div>
        </div>
        <div class="filter-group">
            <span class="filter-label">Route:</span>
            <div class="filter-buttons">
                <button class="filter-btn active" data-action="set-audit-filter" data-filter-field="routed_to" data-filter-value="">Alle</button>
                <button class="filter-btn" data-action="set-audit-filter" data-filter-field="routed_to" data-filter-value="local">Lokal</button>
                <button class="filter-btn" data-action="set-audit-filter" data-filter-field="routed_to" data-filter-value="cloud">Cloud</button>
            </div>
        </div>
        <div class="filter-group">
            <input type="text" placeholder="Suche..." id="auditSearch"
                   data-keydown-action="load-audit-filtered">
            <button class="action-btn primary audit-search-btn" data-action="load-audit-filtered">Suchen</button>
        </div>
    </div>
    <div id="auditTableContainer"></div>`;

    // Reset Filter-State
    auditFilters = { chain_type: '', routed_to: '' };

    const auditTableConfig = {
        id: "compliance-audit",
        apiEndpoint: "/api/admin/compliance/audit",
        showHeader: true,
        expandable: true,
        gridTemplate: "30px 50px 160px 120px 80px 80px 60px 1fr 80px 40px",
        defaultSort: { field: "created_at", direction: "desc" },
        pagination: {
            enabled: true,
            defaultPageSize: 50,
            options: [20, 50, 100, 300],
        },
        columns: [
            { field: "id", label: "ID", sortable: true, align: "right" },
            { field: "created_at", label: "Zeitstempel", sortable: true },
            { field: "username", label: "Benutzer", sortable: true },
            { field: "chain_type", label: "Modus", sortable: true,
              renderer: (val) => modeBadge(val) },
            { field: "routed_to", label: "Route", sortable: true,
              renderer: (val) => routeBadge(val) },
            { field: "chat_id", label: "Chat",
              renderer: (val) => val ? `#${SharedUi.runtime.escapeHtml(String(val))}` : "\u2014" },
            { field: "question_excerpt", label: "Frage", truncate: 60 },
            { field: "duration_ms", label: "Dauer", sortable: true, align: "right",
              renderer: (val) => val != null ? `${SharedUi.runtime.escapeHtml(String(val))}ms` : "\u2014" },
            { field: "_action", label: "",
              renderer: (val, row) => `<a href="/audit/${SharedUi.runtime.escapeAttr(String(row.id))}" target="_blank" rel="noopener noreferrer"
                  class="audit-detail-link" title="Vollansicht"
                  data-action="open-audit-detail">&#x29C9;</a>` },
        ],
        detailRenderer: auditDetailPlaceholder,
    };

    auditTable = new SharedUi.table.ExpandableTable(
        auditTableConfig, 'auditTableContainer',
        { onRowExpanded: onAuditRowExpanded }
    );

    loadAuditFiltered();
}
