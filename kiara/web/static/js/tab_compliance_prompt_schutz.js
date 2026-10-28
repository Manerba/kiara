/**
 * KIara Admin — Compliance > Prompt-Schutz Sub-Tab
 * Zeigt erkannte Prompt-Injection-Events mit Filtern und Trend-Statistiken.
 */

let promptDefenseTable = null;
let pdFilters = { severity: '', source_type: '' };

function formatDateTime(ts) {
    // ISO-Timestamp → lesbares Format (analog formatTimestamp in audit_detail.js)
    if (!ts) return "\u2014";
    return SharedUi.runtime.escapeHtml(String(ts).replace("T", " ").replace(/\.\d+$/, ""));
}

function severityBadge(level) {
    const map = {
        niedrig: '<span class="badge badge-default">Niedrig</span>',
        mittel:  '<span class="badge badge-warning">Mittel</span>',
        hoch:    '<span class="badge badge-danger">Hoch</span>',
    };
    if (typeof level === 'string' && Object.prototype.hasOwnProperty.call(map, level)) {
        return map[level];
    }
    return SharedUi.runtime.escapeHtml(level || "\u2014");
}

function sourceTypeBadge(type) {
    const map = {
        user_input:        '<span class="badge badge-default">User-Eingabe</span>',
        retrieval_context: '<span class="badge badge-warning">Dokument</span>',
        tool_result:       '<span class="badge badge-purple">Tool-Ergebnis</span>',
        chat_history:      '<span class="badge badge-default">Historie</span>',
    };
    if (typeof type === 'string' && Object.prototype.hasOwnProperty.call(map, type)) {
        return map[type];
    }
    return SharedUi.runtime.escapeHtml(type || "\u2014");
}

async function initCompliancePromptSchutzTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');

    // Filter-State explizit zuruecksetzen bei Re-Init
    pdFilters = { severity: '', source_type: '' };

    _bindPdActions();

    container.innerHTML = `
        <h3>Prompt-Injection-Schutz</h3>
        <p class="text-muted">Erkannte und bereinigte Injection-Versuche in User-Eingaben,
           Retrieval-Kontext und Tool-Ergebnissen.</p>

        <!-- Statistik-Karten -->
        <div id="pdStatsCards" class="stats-row"></div>

        <!-- Filter -->
        <div class="filter-bar">
            <div class="filter-group">
                <label>Zeitraum:</label>
                <input type="date" id="pdDateFrom" data-change-action="reload-pd-filter">
                <input type="date" id="pdDateTo" data-change-action="reload-pd-filter">
            </div>
            <div class="filter-group">
                <label>Schweregrad:</label>
                <div class="filter-buttons">
                    <button class="filter-btn active" data-action="set-pd-filter" data-filter-field="severity" data-filter-value="">Alle</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="severity" data-filter-value="hoch">Hoch</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="severity" data-filter-value="mittel">Mittel</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="severity" data-filter-value="niedrig">Niedrig</button>
                </div>
            </div>
            <div class="filter-group">
                <label>Quelle:</label>
                <div class="filter-buttons">
                    <button class="filter-btn active" data-action="set-pd-filter" data-filter-field="source_type" data-filter-value="">Alle</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="source_type" data-filter-value="user_input">User</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="source_type" data-filter-value="retrieval_context">Dokument</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="source_type" data-filter-value="tool_result">Tool</button>
                    <button class="filter-btn" data-action="set-pd-filter" data-filter-field="source_type" data-filter-value="chat_history">Historie</button>
                </div>
            </div>
        </div>

        <!-- Event-Tabelle -->
        <div id="pdTableContainer"></div>
    `;

    // Stats laden
    loadPdStats();

    // ExpandableTable initialisieren (Konstruktor: config, containerId, options)
    promptDefenseTable = new SharedUi.table.ExpandableTable({
        id: 'prompt-defense-events',
        apiEndpoint: '/api/admin/compliance/prompt-defense',
        showHeader: true,
        gridTemplate: '30px 160px 100px 90px 1fr 80px 200px',
        columns: [
            { field: 'timestamp',        label: 'Zeitpunkt',   renderer: v => formatDateTime(v) },
            { field: 'severity',         label: 'Schweregrad', renderer: v => severityBadge(v) },
            { field: 'source_type',      label: 'Quelle',      renderer: v => sourceTypeBadge(v) },
            { field: 'matched_patterns', label: 'Patterns',    renderer: v => formatPatterns(v) },
            { field: 'chars_removed',    label: 'Entfernt',    renderer: v => `${Number(v) || 0} Zeichen` },
            { field: 'document_source',  label: 'Dokument',    renderer: v => v ? SharedUi.runtime.escapeHtml(v) : '\u2014' },
        ],
        expandable: true,
        detailRenderer: renderPdDetail,
        pagination: { enabled: true, defaultPageSize: 50 },
    }, 'pdTableContainer');
    promptDefenseTable.loadData();
}

function formatPatterns(patternsJson) {
    // JSON-String → lesbare Badge-Liste
    let names;
    try { names = JSON.parse(patternsJson || '[]'); } catch(e) { return '<span class="badge badge-sm badge-danger">Parse-Fehler</span>'; }
    const labels = {
        chatml_special_tokens: 'ChatML',
        llama_markers: 'Llama/Mistral',
        gemini_markers: 'Gemini',
        role_spoofing: 'Role-Spoofing',
        fake_system_tags: 'System-Tags',
        anthropic_tags: 'Anthropic-Tags',
        marker_spoofing: 'Tag-Spoofing',
        invisible_chars: 'Unsichtbare Zeichen',
    };
    return names.map(n => {
        const label = (typeof n === 'string' && Object.prototype.hasOwnProperty.call(labels, n))
            ? labels[n]
            : SharedUi.runtime.escapeHtml(n == null ? "—" : String(n));
        return `<span class="badge badge-sm">${label}</span>`;
    }).join(' ');
}

function renderPdDetail(row) {
    // Expandierte Detailansicht: entfernte Snippets anzeigen
    let snippets;
    try { snippets = JSON.parse(row.removed_snippets || '[]'); } catch(e) { snippets = []; }
    let html = '<div class="detail-section">';
    html += '<strong>Entfernte Inhalte:</strong>';
    if (snippets.length === 0) {
        html += '<p class="text-muted">Nur unsichtbare Zeichen (kein darstellbarer Inhalt)</p>';
    } else {
        html += '<ul class="snippet-list">';
        snippets.forEach(s => { html += `<li><code>${SharedUi.runtime.escapeHtml(s)}</code></li>`; });
        html += '</ul>';
    }
    if (row.question_hash) {
        html += `<p><strong>Korrelation:</strong> <code>${SharedUi.runtime.escapeHtml(row.question_hash)}</code> (query_log)</p>`;
    }
    html += '</div>';
    return html;
}

const _PD_ACTIONS = new Set([
    'set-pd-filter',
]);

function _handlePdAction(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_PD_ACTIONS.has(action)) return;

    if (action === 'set-pd-filter') {
        setPdFilter(el.dataset.filterField, el.dataset.filterValue, el);
    }
}

function _handlePdChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el) return;
    const action = el.dataset.changeAction;
    if (action === 'reload-pd-filter') loadPdFiltered();
}

function _bindPdActions() {
    const container = document.getElementById('contentContainer');
    if (container && !container._pdActionsBound) {
        container._pdActionsBound = true;
        container.setAttribute('data-pd-actions-bound', 'true');
        container.addEventListener('click', _handlePdAction);
        container.addEventListener('change', _handlePdChange);
    }
}

function setPdFilter(field, value, btn) {
    pdFilters[field] = value;
    // Button-Active-State umschalten
    if (btn) {
        const group = btn.closest('.filter-buttons');
        group.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }
    loadPdFiltered();
}

function loadPdFiltered() {
    const params = { ...pdFilters };
    const dateFrom = document.getElementById('pdDateFrom')?.value;
    const dateTo = document.getElementById('pdDateTo')?.value;
    if (dateFrom) params.date_from = dateFrom;
    if (dateTo) params.date_to = dateTo;
    // Leere Werte entfernen
    Object.keys(params).forEach(k => { if (!params[k]) delete params[k]; });
    promptDefenseTable.pagination.currentPage = 1;
    promptDefenseTable.loadData(params);
}

async function loadPdStats() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/compliance/prompt-defense-stats');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const stats = await resp.json();
        const container = document.getElementById('pdStatsCards');
        if (!container) return;
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_30d || 0}</div>
                <div class="stat-label">Events (30 Tage)</div>
            </div>
            <div class="stat-card ${stats.high_count > 0 ? 'stat-danger' : ''}">
                <div class="stat-value">${stats.high_count || 0}</div>
                <div class="stat-label">Schweregrad Hoch</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${SharedUi.runtime.escapeHtml(stats.top_source || '\u2014')}</div>
                <div class="stat-label">H\u00e4ufigste Quelle</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${SharedUi.runtime.escapeHtml(stats.top_pattern || '\u2014')}</div>
                <div class="stat-label">H\u00e4ufigstes Pattern</div>
            </div>
        `;
    } catch (err) {
        console.error('Prompt-Defense-Stats laden fehlgeschlagen:', err);
        const container = document.getElementById('pdStatsCards');
        if (container) container.innerHTML = '<p class="text-muted">Statistiken konnten nicht geladen werden.</p>';
    }
}
