/**
 * KIara Audit Detail View
 * Vollbild-Detailansicht fuer einzelne Audit-Trail-Eintraege.
 */
(function() {
    "use strict";

    let auditData = null;
    let currentView = "form";

    const auditId = extractAuditId();
    if (!auditId) {
        showError("Ungueltige Audit-ID in der URL.");
        return;
    }

    document.title = `KIara - Audit #${auditId}`;
    fetchAuditData(auditId);

    function extractAuditId() {
        const match = window.location.pathname.match(/\/audit\/(\d+)/);
        return match ? parseInt(match[1], 10) : null;
    }

    async function fetchAuditData(id) {
        try {
            const resp = await fetch(`/api/admin/compliance/audit/${id}`, {  // uses raw fetch because: standalone audit_detail.html does not load SharedUi - keep local 401/403 redirect
                headers: { "X-Requested-With": "XMLHttpRequest" },
                credentials: "same-origin",
            });

            if (resp.status === 401 || resp.status === 403) {
                window.location.href = `/login?next=/audit/${id}`;
                return;
            }
            if (resp.status === 404) {
                showError("Audit-Eintrag nicht gefunden", `Eintrag #${id} existiert nicht.`);
                return;
            }
            if (!resp.ok) {
                showError("Fehler beim Laden", `HTTP ${resp.status}`);
                return;
            }

            auditData = await resp.json();
            renderApp();
        } catch (e) {
            showError("Verbindungsfehler", "Der Server ist nicht erreichbar.");
        }
    }

    function showError(title, detail) {
        document.getElementById("loading").classList.add("audit-hidden");
        const el = document.getElementById("error-container");
        el.classList.remove("audit-hidden");
        el.innerHTML = `<h2>${esc(title)}</h2>${detail ? `<p>${esc(detail)}</p>` : ""}
            <p class="audit-error-back-link"><a href="/">Zurueck zur Uebersicht</a></p>`;
    }

    function renderApp() {
        document.getElementById("loading").classList.add("audit-hidden");
        document.getElementById("app").classList.remove("audit-hidden");
        renderFormView(auditData);
        renderRawView(auditData);
    }

    // ========== FORM VIEW ==========

    function renderFormView(d) {
        const container = document.getElementById("form-view");
        const ts = formatTimestamp(d.created_at);

        let html = `
        <div class="report-header">
            <div>
                <h1>KIara Audit-Bericht</h1>
            </div>
            <div class="report-meta">
                <div>Bericht #${esc(String(d.id))}</div>
                <div>${esc(ts)}</div>
            </div>
        </div>
        <div class="report-body">`;

        // Meta-Block
        html += `<div class="section">
            <div class="section-title">Meta-Daten</div>
            <div class="meta-grid">
                <div class="meta-item"><label>ID</label><span>${esc(String(d.id))}</span></div>
                <div class="meta-item"><label>Zeitstempel</label><span>${esc(ts)}</span></div>
                <div class="meta-item"><label>Benutzer</label><span>${esc(d.username || "Unbekannt")}</span></div>
                <div class="meta-item"><label>Modus</label><span>${modeBadge(d.chain_type)}</span></div>
                <div class="meta-item"><label>Route</label><span>${routeBadge(d.routed_to)}</span></div>
                <div class="meta-item"><label>Modell</label><span>${esc(d.model_used || "\u2014")}</span></div>
                <div class="meta-item"><label>Dauer</label><span>${d.duration_ms != null ? esc(String(d.duration_ms)) + "ms" : "\u2014"}</span></div>
                <div class="meta-item"><label>Chat-ID</label><span>${d.chat_id ? "#" + esc(String(d.chat_id)) : "\u2014"}</span></div>
                ${d.chat_title ? `<div class="meta-item"><label>Chat</label><span>${esc(d.chat_title)}</span></div>` : ""}
                <div class="meta-item"><label>Quellen</label><span>${esc(String(d.sources_count || 0))}</span></div>
                <div class="meta-item"><label>Anonymisiert</label><span>${esc(String(d.entities_anonymized || 0))} Entities</span></div>
            </div>
        </div>`;

        if (d.request_settings && Object.keys(d.request_settings).length) {
            const settings = d.request_settings;
            html += `<div class="section"><div class="section-title">Quellensteuerung</div>
                <div class="meta-grid">
                    <div class="meta-item"><label>Chat-Vorgabe</label><span>${esc(settings.ui_k == null ? 'Admin-Standard' : (settings.ui_k === 0 ? 'Auto' : String(settings.ui_k)))}</span></div>
                    <div class="meta-item"><label>Wirksame Auswahl</label><span>${esc(settings.selection === 'auto' ? 'Auto je Suche' : String(settings.effective_k))}</span></div>
                </div></div>`;
        }

        // Frage
        html += `<div class="section">
            <div class="section-title">Frage</div>
            <div class="question-text">${esc(d.question_excerpt || "\u2014")}</div>
        </div>`;

        // Kontext/Quellen
        if (d.context_sources && d.context_sources.length > 0) {
            html += `<div class="section">
                <div class="section-title">Kontext / Quellen (${d.context_sources.length})</div>
                <div class="sources-grid">`;
            for (const src of d.context_sources) {
                const filename = src.source ? src.source.split("/").pop() : "Unbekannt";
                const score = src.score != null ? Number(src.score).toFixed(3) : "";
                html += `<div class="source-card">
                    <div class="source-header">
                        <span class="source-file">${esc(filename)}</span>
                        ${score ? `<span class="source-score">${score}</span>` : ""}
                    </div>
                    <div class="source-path">${esc(src.source || "")}</div>
                    ${src.excerpt ? `<div class="source-excerpt">${esc(src.excerpt)}</div>` : ""}
                </div>`;
            }
            html += `</div></div>`;
        }

        // Tool-Calls
        if (d.tool_calls && d.tool_calls.length > 0) {
            html += `<div class="section">
                <div class="section-title">Tool-Aufrufe (${d.tool_calls.length})</div>`;
            for (const tc of d.tool_calls) {
                html += `<div class="tool-call">
                    <div class="tool-call-header">${esc(tc.tool || tc.name || "Tool")}</div>
                    <div class="tool-call-body">`;
                if (tc.params || tc.args || tc.arguments || tc.input) {
                    const args = tc.params || tc.args || tc.arguments || tc.input;
                    html += `<div class="tool-call-label">Parameter</div>
                        <pre>${esc(typeof args === "string" ? args : JSON.stringify(args, null, 2))}</pre>`;
                }
                if (tc.source_selection) {
                    const sources = tc.source_selection;
                    html += `<div class="tool-call-label">Quellen: Modell-Vorschlag / wirksames k</div>
                        <pre>${esc(String(sources.model_k ?? 'nicht gesetzt'))} / ${esc(String(sources.effective_k ?? 'abgelehnt'))}${sources.model_k_ignored ? ' (feste Chat-Vorgabe hat Vorrang)' : ''}</pre>`;
                }
                if (tc.result !== undefined && tc.result !== null) {
                    html += `<div class="tool-call-label">Ergebnis</div>
                        <pre>${esc(typeof tc.result === "string" ? tc.result : JSON.stringify(tc.result, null, 2))}</pre>`;
                }
                html += `</div></div>`;
            }
            html += `</div>`;
        }

        // Antwort (Markdown gerendert)
        if (d.full_answer || d.answer_excerpt) {
            const answerText = d.full_answer || d.answer_excerpt || "";
            let rendered;
            try {
                rendered = (typeof marked !== "undefined" && marked.parse && typeof DOMPurify !== "undefined")
                    ? DOMPurify.sanitize(marked.parse(answerText)) : escPre(answerText);
            } catch (_) {
                rendered = escPre(answerText);
            }
            html += `<div class="section">
                <div class="section-title">Antwort</div>
                <div class="answer-content">${rendered}</div>
            </div>`;
        }

        // Cloud-Audit
        if (d.cloud_audit) {
            const ca = d.cloud_audit;
            html += `<div class="section">
                <div class="section-title">Cloud-Audit (DSGVO)</div>
                <table class="cloud-table">
                    <tr><th>Provider</th><td>${esc(ca.provider || "\u2014")}</td></tr>
                    <tr><th>Modell</th><td>${esc(ca.model || "\u2014")}</td></tr>
                    <tr><th>Tokens (In/Out)</th><td>${esc(String(ca.tokens_in || 0))} / ${esc(String(ca.tokens_out || 0))}</td></tr>
                    ${ca.cloud_entities ? `<tr><th>Anonymisiert</th><td>${esc(String(ca.cloud_entities))}</td></tr>` : ""}
                </table>
            </div>`;
        }

        html += `</div>`; // report-body
        container.innerHTML = html;
    }

    // ========== RAW VIEW ==========

    function renderRawView(d) {
        const container = document.getElementById("raw-view");

        const sections = [];

        sections.push({ title: "Meta-Daten (JSON)", content: jsonBlock(filterMeta(d)), open: false });

        if (d.request_settings && Object.keys(d.request_settings).length) {
            sections.push({ title: "Quellensteuerung (Profil zum Anfragezeitpunkt)", content: jsonBlock(d.request_settings), open: false });
        }

        if (d.full_prompt) {
            let formatted;
            try {
                const parsed = JSON.parse(d.full_prompt);
                formatted = JSON.stringify(parsed, null, 2);
            } catch (_) { formatted = d.full_prompt; }
            sections.push({ title: "Full Prompt", content: formatted, open: false });
        }

        if (d.full_answer) {
            sections.push({ title: "Full Answer", content: d.full_answer, open: false });
        }

        if (d.context_sources && d.context_sources.length > 0) {
            sections.push({ title: `Context Sources (${d.context_sources.length})`, content: jsonBlock(d.context_sources), open: false });
        }

        if (d.tool_calls && d.tool_calls.length > 0) {
            for (let i = 0; i < d.tool_calls.length; i++) {
                const tc = d.tool_calls[i];
                const name = tc.tool || tc.name || `Tool #${i + 1}`;
                sections.push({ title: `Tool: ${name}`, content: jsonBlock(tc), open: false });
            }
        }

        if (d.cloud_audit) {
            sections.push({ title: "Cloud-Audit", content: jsonBlock(d.cloud_audit), open: false });
        }

        let html = `<div class="raw-controls">
            <button data-audit-action="toggleAllRaw" id="btn-toggle-all">Alles expandieren</button>
        </div>`;

        for (let i = 0; i < sections.length; i++) {
            const s = sections[i];
            html += `<div class="raw-section">
                <div class="raw-section-header" data-audit-action="toggleRawSection" data-audit-index="${i}">
                    <span>${esc(s.title)}</span>
                    <span class="chevron" id="chevron-${i}">&#x25B6;</span>
                </div>
                <div class="raw-section-content" id="raw-content-${i}">
                    <pre>${esc(typeof s.content === "string" ? s.content : JSON.stringify(s.content, null, 2))}</pre>
                </div>
            </div>`;
        }

        container.innerHTML = html;
        container._sectionCount = sections.length;
    }

    function filterMeta(d) {
        const keys = ["id", "created_at", "username", "chain_type", "routed_to",
            "model_used", "duration_ms", "chat_id", "chat_title", "sources_count",
            "chunks_retrieved", "entities_anonymized", "tools_used", "question_hash"];
        const out = {};
        for (const k of keys) { if (d[k] !== undefined && d[k] !== null) out[k] = d[k]; }
        return out;
    }

    function jsonBlock(obj) {
        return JSON.stringify(obj, null, 2);
    }

    // ========== TOGGLE LOGIC ==========
    // CSP-kompatibel: Document-Level-Delegation ueber data-audit-action
    // (Pattern analog SharedUi.table, siehe ui_core/js/table.js).

    function setView(view) {
        currentView = view;
        document.getElementById("form-view").classList.toggle("audit-hidden", view !== "form");
        document.getElementById("raw-view").classList.toggle("audit-hidden", view !== "raw");
        document.getElementById("btn-form-view").className = "toggle-btn" + (view === "form" ? " active" : "");
        document.getElementById("btn-raw-view").className = "toggle-btn" + (view === "raw" ? " active" : "");
    }

    function toggleRawSection(idx) {
        const content = document.getElementById(`raw-content-${idx}`);
        const chevron = document.getElementById(`chevron-${idx}`);
        content.classList.toggle("open");
        chevron.classList.toggle("open");
    }

    function toggleAllRaw() {
        const container = document.getElementById("raw-view");
        const count = container._sectionCount || 0;
        const allOpen = areAllOpen(count);
        for (let i = 0; i < count; i++) {
            const content = document.getElementById(`raw-content-${i}`);
            const chevron = document.getElementById(`chevron-${i}`);
            if (allOpen) {
                content.classList.remove("open");
                chevron.classList.remove("open");
            } else {
                content.classList.add("open");
                chevron.classList.add("open");
            }
        }
        document.getElementById("btn-toggle-all").textContent =
            allOpen ? "Alles expandieren" : "Alles einklappen";
    }

    document.addEventListener("click", function(ev) {
        const actionEl = ev.target.closest("[data-audit-action]");
        if (!actionEl) return;
        const action = actionEl.getAttribute("data-audit-action");
        if (action === "setView") {
            const view = actionEl.getAttribute("data-audit-view");
            if (view) setView(view);
        } else if (action === "toggleRawSection") {
            const idx = parseInt(actionEl.getAttribute("data-audit-index"), 10);
            if (Number.isFinite(idx)) toggleRawSection(idx);
        } else if (action === "toggleAllRaw") {
            toggleAllRaw();
        }
    });

    function areAllOpen(count) {
        for (let i = 0; i < count; i++) {
            const el = document.getElementById(`raw-content-${i}`);
            if (!el || !el.classList.contains("open")) return false;
        }
        return count > 0;
    }

    // ========== HELPERS ==========

    function esc(str) {
        if (str == null) return "";
        const d = document.createElement("div");
        d.textContent = String(str);
        return d.innerHTML;
    }

    function escPre(str) {
        return `<pre class="audit-pre-wrap">${esc(str)}</pre>`;
    }

    function formatTimestamp(ts) {
        if (!ts) return "\u2014";
        return ts.replace("T", " ").replace(/\.\d+$/, "");
    }

    function modeBadge(mode) {
        const labels = Object.assign(Object.create(null), { standard: "Standard", unchained: "Unchained", agent: "Agent" });
        const cls = Object.assign(Object.create(null), { standard: "badge-standard", unchained: "badge-unchained", agent: "badge-agent" });
        return `<span class="badge ${cls[mode] || "badge-standard"}">${esc(labels[mode] || mode || "\u2014")}</span>`;
    }

    function routeBadge(route) {
        if (!route) return "\u2014";
        const isCloud = route.includes("cloud");
        return `<span class="badge ${isCloud ? "badge-cloud" : "badge-local"}">${esc(route)}</span>`;
    }
})();
