/**
 * KIara - Chat UI Class
 * SSE-Streaming-Wiring um SharedUi.chat.ChatView (ui_core/js/chat_primitives.js).
 * Markdown/Copy/Think kommen aus SharedUi.markdown (ui_core/js/markdown.js);
 * Timestamps aus formatKiaraTimestamp in chat_utils.js.
 */

// Trace-Debug State (S7)
const _TRACE_DEBUG_KEY = "kiara_trace_debug";
let _traceDebugAvailable = false;
let _traceAdminPort = 8502;
let _traceDebugInitialized = false;
let _traceDebugInitPromise = null;
let _traceDebugGeneration = 0;
const _traceDebugRetryTimers = new Set();

function isTraceDebugEnabled() {
    return _traceDebugAvailable && localStorage.getItem(_TRACE_DEBUG_KEY) === "true";
}

function toggleTraceDebug() {
    const current = isTraceDebugEnabled();
    localStorage.setItem(_TRACE_DEBUG_KEY, current ? "false" : "true");
    return !current;
}

function _clearTraceDebugBars() {
    document.querySelectorAll(".trace-debug-bar").forEach(el => el.remove());
}

/**
 * Frontend-Beacon fuer Stream-Abrisse, die der Server nicht sieht
 * (Worker-Tod mid-stream, Connection-Reset, Proxy-Timeout etc).
 * Backend-Logging greift in diesen Faellen nicht — kein Server-Code laeuft
 * mehr, der LLM_BACKEND_ERROR oder vergleichbares schreiben koennte.
 * Beacon schliesst die Diagnose-Luecke ueber CHAT_CLIENT_ERROR-Events.
 *
 * Fail-safe: Beacon-Fehler darf den User nicht stoeren — alles in try/catch,
 * keepalive: true erlaubt Senden auch bei Page-Unload.
 */
// Backend akzeptiert nur typing.Literal-Allowlist fuer mode/error_kind.
// Wir normalisieren clientseitig auf "unknown" bei nicht erkannten Werten
// statt den 422 vom Server zu provozieren — Audit-Run-2 Finding #5.
const _ALLOWED_MODES = new Set(['agent', 'standard', 'rewriter', 'unchained', 'research', 'unknown']);
const _ALLOWED_ERROR_KINDS = new Set(['network', 'http', 'unauthorized', 'unknown']);

function _reportChatClientError({ chatId, errorKind, errorMessage, mode }) {
    const safeMode = _ALLOWED_MODES.has(mode) ? mode : 'unknown';
    const safeKind = _ALLOWED_ERROR_KINDS.has(errorKind) ? errorKind : 'unknown';
    try {
        // safeFetch ist async — try/catch fasst nur synchrone Fehler.
        // Fuer Promise-Rejections (z.B. 422 vom Server, Netzwerk-Reject)
        // braucht es .catch(); ohne diesen landet der Fehler als
        // "Unhandled Promise Rejection" in der DevTools-Konsole.
        // Audit-Run-2 Finding #5.
        const promise = SharedUi.fetch.safeFetch('/api/chat-debug/client-error', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            keepalive: true,
            body: JSON.stringify({
                chat_id: chatId == null ? null : String(chatId),
                error_kind: safeKind,
                error_message: String(errorMessage || '').slice(0, 2000),
                mode: safeMode,
            }),
        });
        if (promise && typeof promise.catch === 'function') {
            promise.catch(() => {
                // fail-safe: Beacon ist Diagnose, kein Pflichtweg
            });
        }
    } catch (_) {
        // fail-safe: Beacon-Aufruf darf den User nicht stoeren
    }
}

function _bumpTraceDebugGeneration() {
    _traceDebugGeneration += 1;
    _traceDebugRetryTimers.forEach(t => clearTimeout(t));
    _traceDebugRetryTimers.clear();
    return _traceDebugGeneration;
}

function _refreshTraceDebugBars() {
    if (!isTraceDebugEnabled()) {
        _clearTraceDebugBars();
        return;
    }
    document.querySelectorAll(".chat-message.assistant[data-trace-id]").forEach(messageEl => {
        const traceId = messageEl.dataset.traceId;
        if (traceId && !messageEl.querySelector(".trace-debug-bar")) {
            _loadTraceDebug(messageEl, traceId, 0, _traceDebugGeneration);
        }
    });
}

function handleChatTabRestore() {
    void _initTraceDebug();
    _refreshTraceDebugBars();
    _applyPendingChatLoad();
}

async function _initTraceDebug() {
    if (_traceDebugInitialized) return;
    if (_traceDebugInitPromise) return _traceDebugInitPromise;

    _traceDebugInitPromise = (async () => {
        try {
            if (document.querySelector(".trace-debug-toggle")) {
                _traceDebugInitialized = true;
                return;
            }

            let data;
            try {
                data = await SharedUi.fetch.safeFetchJson("/api/chat-debug/status", {
                    headers: {"X-Requested-With": "XMLHttpRequest"},
                });
            } catch (err) {
                if (err && (err.kind === 'http' || err.kind === 'unauthorized')) {
                    _traceDebugInitialized = true;
                }
                return;
            }
            _traceDebugInitialized = true;
            if (!data.enabled) return;

            _traceDebugAvailable = true;
            _traceAdminPort = data.admin_port || 8502;

            const btn = _createTraceDebugToggle();
            if (!btn) return;

            const headerControls = document.querySelector(".header-controls");
            if (headerControls && !headerControls.querySelector(".trace-debug-toggle")) {
                headerControls.insertBefore(btn, headerControls.firstChild);
            }

            _refreshTraceDebugBars();
        } catch (e) {
            // Kein Trace-Debug verfuegbar
        } finally {
            _traceDebugInitPromise = null;
        }
    })();

    return _traceDebugInitPromise;
}

async function _loadTraceDebug(bubbleEl, traceId, retryCount = 0, generation = _traceDebugGeneration) {
    if (!isTraceDebugEnabled()) return;
    if (generation !== _traceDebugGeneration) return;
    const bubble = bubbleEl?.querySelector(".chat-bubble");
    if (!bubble || !document.body.contains(bubble)) return;
    try {
        let trace;
        try {
            trace = await SharedUi.fetch.safeFetchJson(`/api/chat-debug/trace/${encodeURIComponent(traceId)}`, {
                headers: {"X-Requested-With": "XMLHttpRequest"},
            });
        } catch (err) {
            if (err && err.status === 404 && retryCount < 2 && generation === _traceDebugGeneration) {
                const timer = setTimeout(() => {
                    _traceDebugRetryTimers.delete(timer);
                    if (isTraceDebugEnabled() && generation === _traceDebugGeneration) {
                        _loadTraceDebug(bubbleEl, traceId, retryCount + 1, generation);
                    }
                }, 2000);
                _traceDebugRetryTimers.add(timer);
            }
            return;
        }
        if (generation !== _traceDebugGeneration) return;
        if (!document.body.contains(bubble)) return;
        if (!isTraceDebugEnabled()) return;
        _renderTraceTimingBar(bubbleEl, trace, traceId, generation);
    } catch (e) {
        console.debug("Trace-Debug fehlgeschlagen:", e);
    }
}

function _renderTraceTimingBar(bubbleEl, trace, traceId, generation = _traceDebugGeneration) {
    if (generation !== _traceDebugGeneration) return;
    const bubble = bubbleEl?.querySelector(".chat-bubble");
    if (!bubble) return;
    if (!document.body.contains(bubble)) return;

    const spans = trace.spans || [];
    if (!spans.length) return;

    const root = spans.find(s => !s.parent_span_id);
    if (!root) return;

    const children = spans.filter(s => s.parent_span_id === root.span_id);

    const segments = children
        .filter(s => s.duration_ms > 0)
        .sort((a, b) => a.start_time_ns - b.start_time_ns)
        .map(s => `${_spanLabel(s.name)}: ${Math.round(s.duration_ms)}ms`)
        .join(" \u00B7 ");

    const totalMs = Math.round(Number(root.duration_ms) || 0);

    const adminUrl = new URL(window.location.href);
    adminUrl.port = String(_traceAdminPort);
    adminUrl.pathname = "/system/traces";
    adminUrl.search = "";
    adminUrl.hash = "";
    const adminTracesUrl = adminUrl.toString();

    const bar = document.createElement("div");
    bar.className = "trace-debug-bar";
    bar.innerHTML =
        `<span class="trace-debug-segments">${SharedUi.runtime.escapeHtml(segments)}</span>` +
        `<span class="trace-debug-total">${totalMs}ms</span>` +
        `<a href="${adminTracesUrl}" rel="noopener noreferrer" ` +
        `   class="trace-debug-link" target="_blank" title="Im Trace-Viewer oeffnen">\u2197</a>`;

    const existing = bubble.querySelector(".trace-debug-bar");
    if (existing) existing.remove();

    const footerAnchor = bubble.querySelector(".chat-usage, .chat-sources");
    if (footerAnchor) {
        bubble.insertBefore(bar, footerAnchor);
    } else {
        bubble.appendChild(bar);
    }
}

function _spanLabel(name) {
    const labels = {
        "query_expansion": "Expansion",
        "hybrid_retrieve": "Retrieval",
        "rerank": "Reranking",
        "sibling_expansion": "Siblings",
        "context_build": "Context",
        "llm_setup": "LLM",
        "llm_generate": "LLM",
        "agent_iteration": "Iteration",
        "final_answer": "Antwort",
    };
    return labels[name] || name;
}

function _createTraceDebugToggle() {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "trace-debug-toggle";
    btn.title = "Trace-Debug ein/aus";
    btn.textContent = "\u23F1";
    btn.addEventListener("click", () => {
        const active = toggleTraceDebug();
        btn.classList.toggle("active", active);
        if (active) _refreshTraceDebugBars();
        else _clearTraceDebugBars();
        SharedUi.notify.show(active ? "Trace-Debug aktiviert" : "Trace-Debug deaktiviert");
    });
    btn.classList.toggle("active", localStorage.getItem(_TRACE_DEBUG_KEY) === "true");
    return btn;
}

// ── History-Resume Token (Pending-Load-Intent) ─────────────────────

let _pendingChatLoad = null;
let _pendingChatLoadToken = 0;

function _setPendingChatLoad(chatId, messages) {
    _pendingChatLoadToken += 1;
    _pendingChatLoad = { token: _pendingChatLoadToken, chatId, messages };
    return _pendingChatLoadToken;
}

function _applyPendingChatLoad() {
    if (!_pendingChatLoad) return;
    if (typeof chatUI === 'undefined' || !chatUI) return;
    const p = _pendingChatLoad;
    _pendingChatLoad = null;
    chatUI.loadChat(p.chatId, p.messages || []);
}

// ── Source-Rendering (KIara-spezifisch: Score + Editor-Link) ───────

const _EDIT_EXTS = ['.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp', '.txt', '.csv', '.doc', '.xls', '.ppt'];

function _finiteScore(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(3) : '';
}

function _kiaraRenderSourceItem(src) {
    const name = SharedUi.runtime.escapeHtml(src && src.source ? src.source : '');
    const score = _finiteScore(src && src.score);
    const excerpt = SharedUi.runtime.escapeHtml(src && src.excerpt ? src.excerpt : '');
    const sourcePath = src && src.source ? String(src.source) : '';
    const srcExt = sourcePath.substring(sourcePath.lastIndexOf('.')).toLowerCase();
    const editHtml = _EDIT_EXTS.includes(srcExt)
        ? `<a class="chat-source-edit-btn" href="/editor?file=${encodeURIComponent(sourcePath)}" target="_blank" rel="noopener noreferrer">Bearbeiten</a>`
        : '';
    return `<div class="chat-source-item">
        <div class="chat-source-header">
            <span class="chat-source-name">${name}</span>
            ${score ? `<span class="chat-source-score">Relevanz: ${score}</span>` : ''}
        </div>
        <div class="chat-source-excerpt">${excerpt}</div>
        ${editHtml}
    </div>`;
}

// ── ChatUI: Wiring um SharedUi.chat.ChatView ─────────────────────

class ChatUI {
    constructor(containerId) {
        this.containerId = containerId;
        this.currentChatId = null;
        this.isStreaming = false;
        this._abortController = null;
        this._streamGeneration = 0;
        this._contextChunks = null;
        this._view = null;
        // SSE sauber abbrechen wenn die Seite verlassen wird (Tab-Close/Reload/Navigation)
        this._onPageHide = () => {
            if (this._abortController) {
                try { this._abortController.abort(); } catch (e) { /* ignore */ }
                this._abortController = null;
            }
        };
        window.addEventListener('pagehide', this._onPageHide);
    }

    _buildView() {
        if (this._view) { try { this._view.destroy(); } catch (e) { /* ignore */ } }
        this._view = new SharedUi.chat.ChatView(this.containerId, {
            renderInput: true,
            inputId: 'chatInput',
            sendButtonId: 'chatSendBtn',
            stopButtonId: 'chatStopBtn',
            formatTimestamp: formatKiaraTimestamp,
            avatarLabels: { user: 'Du', assistant: 'AI' },
            renderSourceItem: _kiaraRenderSourceItem,
            onSend: (text) => this.sendMessage(text),
            onStop: () => this.abortStream(),
            emptyState: `<div class="chat-empty">
                <div class="chat-empty-icon">&#x1F50D;</div>
                <div class="chat-empty-text">Stelle eine Frage an die Wissensbasis</div>
                <div class="chat-empty-hint">Die Antwort basiert auf den indexierten Dokumenten.</div>
            </div>`,
        });
    }

    render(messages = []) {
        if (this._abortController) {
            this._abortController.abort();
            this.isStreaming = false;
            this._abortController = null;
        }
        this._streamGeneration += 1;
        _bumpTraceDebugGeneration();
        this._contextChunks = null;

        this._buildView();
        this._view.render();

        if (messages && messages.length > 0) {
            this._renderMessages(messages);
        }

        if (_traceDebugAvailable) {
            _refreshTraceDebugBars();
        }
    }

    _renderMessages(messages) {
        messages.forEach(msg => {
            if (msg.role === 'context') {
                const messagesEl = document.getElementById('chatMessages');
                if (messagesEl) {
                    messagesEl.insertAdjacentHTML('beforeend', this._renderContextMessage(msg));
                }
                return;
            }
            if (msg.role === 'user') {
                this._view.appendUserMessage(msg.content || '', { createdAt: msg.created_at });
                return;
            }
            const el = this._view.startAssistantMessage({ createdAt: msg.created_at });
            if (!el) return;
            el.dataset.chatTextAnchored = '1';
            if (msg.trace_id) el.dataset.traceId = String(msg.trace_id);
            const bubble = el.querySelector('.chat-bubble');
            const textEl = bubble ? bubble.querySelector('.chat-text') : null;
            if (textEl) textEl.innerHTML = SharedUi.markdown.render(msg.content || '', { thinkBlocks: true });
            this._renderProcessingWarnings(el, msg.warnings || []);
            if (msg.sources && msg.sources.length > 0) {
                this._view.setSources(msg.sources, { target: el });
            }
        });
        this._view.scrollToBottom(true);
    }

    _renderContextMessage(msg) {
        let data;
        try { data = typeof msg.content === 'string' ? JSON.parse(msg.content) : msg.content; }
        catch { data = { text: msg.content || '', source: '', score: 0 }; }
        const filename = (data.source || '').split('/').pop() || 'Dokument';
        const scoreStr = _finiteScore(data.score);
        const excerpt = (data.text || '').substring(0, 150);
        return `<div class="chat-message-context">
            <div class="chat-context-bubble">
                <div class="chat-context-label">Kontext</div>
                <div class="chat-context-title">${SharedUi.runtime.escapeHtml(filename)}${scoreStr ? '<span class="chat-context-score">' + scoreStr + '</span>' : ''}</div>
                <div class="chat-context-text">${SharedUi.runtime.escapeHtml(excerpt)}</div>
            </div>
        </div>`;
    }

    renderContextMessages(chunks) {
        const messagesEl = document.getElementById('chatMessages');
        if (!messagesEl) return;
        const empty = messagesEl.querySelector('.chat-empty');
        if (empty) empty.remove();

        if (chunks.length <= 3) {
            chunks.forEach(c => {
                messagesEl.insertAdjacentHTML('beforeend', this._renderContextMessage({
                    role: 'context', content: c,
                }));
            });
        } else {
            let html = '<div class="chat-context-group"><details class="chat-context-group-inner">';
            html += `<summary>${chunks.length} Kontext-Dokumente</summary>`;
            chunks.forEach(c => {
                html += this._renderContextMessage({ role: 'context', content: c });
            });
            html += '</details></div>';
            messagesEl.insertAdjacentHTML('beforeend', html);
        }
        if (this._view) this._view.scrollToBottom(true);
    }

    async sendMessage(overrideText) {
        if (this.isStreaming) return;
        const input = document.getElementById('chatInput');
        let question;
        if (typeof overrideText === 'string' && overrideText.trim()) {
            question = overrideText.trim();
            if (input) input.value = '';
        } else {
            if (!input) return;
            question = input.value.trim();
            if (!question) return;
            input.value = '';
            SharedUi.runtimeStyle.applyStyle(input, { height: 'auto' });
        }

        const sendBtn = document.getElementById('chatSendBtn');
        if (sendBtn) sendBtn.disabled = true;
        this.isStreaming = true;
        this._abortController = new AbortController();
        const myGeneration = ++this._streamGeneration;

        if (this._view) this._view.showStopButton(true);

        this._view.appendUserMessage(question, { createdAt: new Date().toISOString() });

        // Filter aus Toolbar
        const filterDsIds = (typeof _selectedDsIds !== 'undefined' && _selectedDsIds.size > 0)
            ? Array.from(_selectedDsIds) : null;
        const filterDsExclude = document.getElementById('chatFilterDsNot')?.checked || false;
        const filterType = document.getElementById('chatFilterType')?.value || '';
        const filterModel = document.getElementById('chatFilterModel')?.value || '';
        const filterLlm = document.getElementById('chatFilterLlm')?.value || '';
        const useHistory = document.getElementById('chatFilterHistory')?.checked || false;
        const useThink = document.getElementById('chatFilterThink')?.checked || false;
        const numCtx = (parseInt(document.getElementById('chatFilterCtx')?.value) || 8) * 1024;

        const modeSelect = document.getElementById('chatFilterMode');
        let mode = 'standard';
        let templateId = null;
        if (modeSelect) {
            const selectedOpt = modeSelect.options[modeSelect.selectedIndex];
            if (selectedOpt) {
                mode = selectedOpt.dataset.chainType || 'standard';
                const tid = parseInt(modeSelect.value);
                if (tid > 0) templateId = tid;
            }
        }

        const typingEl = this._view.appendStreamingIndicator({ id: 'chatTyping', label: 'Denke nach\u2026' });
        let bubbleEl = null;
        let earlyCreatedChatId = null;

        try {
            const chatId = this.currentChatId || 'new';
            const tempVal = parseFloat(document.getElementById('chatFilterTemp')?.value);
            const temperature = (!isNaN(tempVal) && mode === 'unchained') ? tempVal : null;
            const maxRepeats = parseInt(document.getElementById('chatFilterMaxRepeats')?.value ?? '5');
            const body = {
                question,
                k: _getChatSourceK(mode),
                num_ctx: numCtx,
                mode,
                filter_datasource_ids: filterDsIds,
                filter_datasource_exclude: filterDsExclude,
                filter_type: filterType || null,
                embedding_model: filterModel || null,
                llm_backend_id: filterLlm ? parseInt(filterLlm) : null,
                use_history: useHistory,
                think: useThink,
                temperature,
                max_tool_repeats: maxRepeats,
            };
            if (templateId) body.template_id = templateId;
            if (this._contextChunks) {
                body.context_chunks = this._contextChunks;
                this._contextChunks = null;
            }

            // Streaming-Request: safeFetch-Auto-Retry deaktivieren (ReadableStream
            // ist nicht erneut konsumierbar). Bei 403+csrf_invalid den Stream
            // manuell neu aufbauen.
            let response;
            const initialController = this._abortController;
            try {
                response = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                    signal: initialController.signal,
                    csrfRetry: false,
                });
            } catch (streamErr) {
                if (streamErr && streamErr.status === 403 && streamErr.body && streamErr.body.code === 'csrf_invalid'
                    && SharedUi.fetch && SharedUi.fetch.refreshCsrfToken) {
                    const fresh = await SharedUi.fetch.refreshCsrfToken();
                    const stillCurrent = myGeneration === this._streamGeneration
                        && this._abortController === initialController
                        && !initialController.signal.aborted;
                    if (!stillCurrent) {
                        if (myGeneration === this._streamGeneration) {
                            this._view.removeStreamingIndicator(typingEl);
                        }
                        return;
                    }
                    if (fresh) {
                        this._abortController = new AbortController();
                        response = await SharedUi.fetch.safeFetch(`/api/chat/${chatId}/query`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(body),
                            signal: this._abortController.signal,
                            csrfRetry: false,
                        });
                    } else {
                        throw streamErr;
                    }
                } else {
                    throw streamErr;
                }
            }

            this._view.removeStreamingIndicator(typingEl);
            bubbleEl = this._view.startAssistantMessage({
                createdAt: new Date().toISOString(),
                showCopyButton: true,
            });
            let fullText = '';
            let pendingSources = [];
            // Buffered Pre-Chunk-Events (werden als Custom-Blocks nach erstem Chunk eingefuegt)
            // Tool-Iteration-Counter pro Stream (raw iteration ist nicht eindeutig)
            let toolOrdinal = 0;
            const toolHandles = [];

            // rAF-Throttle fuer Markdown-Render: vermeidet O(N^2) durch full-text re-render pro Chunk
            let pendingChunkText = null;
            let chunkRafId = null;
            const renderPendingChunk = () => {
                chunkRafId = null;
                if (pendingChunkText === null) return;
                if (!bubbleEl || !document.body.contains(bubbleEl)) return;
                this._view.appendAssistantChunk(pendingChunkText, { target: bubbleEl, thinkBlocks: true });
                pendingChunkText = null;
            };
            const flushChunkRender = () => {
                if (chunkRafId !== null) {
                    cancelAnimationFrame(chunkRafId);
                    renderPendingChunk();
                }
            };

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        currentEvent = line.substring(7).trim();
                    } else if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6);
                        try {
                            const data = JSON.parse(dataStr);
                            if (!bubbleEl || !document.body.contains(bubbleEl)) {
                                try { await reader.cancel(); } catch (e) {}
                                return;
                            }
                            if (currentEvent === 'chat_created') {
                                if (data && data.chat_id && myGeneration === this._streamGeneration) {
                                    earlyCreatedChatId = data.chat_id;
                                    this.currentChatId = data.chat_id;
                                }
                            } else if (currentEvent === 'sources') {
                                pendingSources = Array.isArray(data) ? data : [];
                            } else if (currentEvent === 'chunk') {
                                fullText += data.text || '';
                                pendingChunkText = fullText;
                                if (chunkRafId === null) {
                                    chunkRafId = requestAnimationFrame(renderPendingChunk);
                                }
                            } else if (currentEvent === 'done') {
                                if (data.chat_id && myGeneration === this._streamGeneration) {
                                    this.currentChatId = data.chat_id;
                                }
                                if (data.trace_id) {
                                    bubbleEl.dataset.traceId = data.trace_id;
                                }
                                if (isTraceDebugEnabled() && data.trace_id) {
                                    _loadTraceDebug(bubbleEl, data.trace_id, 0, _traceDebugGeneration);
                                }
                            } else if (currentEvent === 'mode') {
                                this._appendModeBadge(bubbleEl, data.mode || '');
                            } else if (currentEvent === 'rewrite') {
                                this._appendRewriteInfo(bubbleEl, data);
                            } else if (currentEvent === 'expansion') {
                                this._appendExpansionInfo(bubbleEl, data);
                            } else if (currentEvent === 'processing_warnings') {
                                this._renderProcessingWarnings(bubbleEl, data.warnings);
                            } else if (currentEvent === 'research_progress') {
                                const labels = {search: 'Suche nach weiteren Belegen', read: 'Originalstellen lesen', verify: 'Antwort anhand der Belege prüfen'};
                                const text = labels[data.phase] || 'Recherche läuft';
                                const existing = this._view.getCustomBlockEl('research-progress', {target: bubbleEl});
                                if (existing) {
                                    const status = existing.querySelector('[role="status"]') || existing;
                                    status.textContent = text;
                                } else {
                                    this._view.appendCustomBlock(`<div role="status">${SharedUi.runtime.escapeHtml(text)}</div>`,
                                        {target: bubbleEl, slot: 'beforeText', id: 'research-progress', sanitize: false});
                                }
                            } else if (currentEvent === 'research_status') {
                                const progress = this._view.getCustomBlockEl('research-progress', {target: bubbleEl});
                                if (progress) progress.remove();
                            } else if (currentEvent === 'tool_call') {
                                const ord = ++toolOrdinal;
                                const toolId = `u-${data.iteration || 0}-${ord}`;
                                const el = this._view.appendToolBlock({
                                    target: bubbleEl,
                                    id: toolId,
                                    name: data.tool || '',
                                    params: JSON.stringify(data.params || {}),
                                    summary: 'Ausfuehrung...',
                                    status: 'pending',
                                });
                                toolHandles.push({ id: toolId, el });
                            } else if (currentEvent === 'tool_result') {
                                this._updateLastToolResult(bubbleEl, data);
                            } else if (currentEvent === 'usage') {
                                this._view.setUsage(data, { target: bubbleEl });
                            } else if (currentEvent === 'error') {
                                this._appendError(bubbleEl, data.message || 'Unbekannter Fehler');
                            } else if (currentEvent === 'route') {
                                // no-op
                            }
                        } catch (e) { /* parse error, skip */ }
                        currentEvent = '';
                    }
                }
            }

            flushChunkRender();

            if (myGeneration !== this._streamGeneration) {
                return;
            }
            if (!bubbleEl || !document.body.contains(bubbleEl)) {
                return;
            }
            if (pendingSources.length > 0) {
                this._view.setSources(pendingSources, { target: bubbleEl });
            }
            this._view.finishAssistantMessage({ target: bubbleEl });

        } catch (error) {
            this._view.removeStreamingIndicator(typingEl);
            if (error && error.name === 'AbortError') {
                if (mode === 'research' && bubbleEl) {
                    const textEl = bubbleEl.querySelector('.chat-text');
                    if (textEl) textEl.textContent = 'Recherche abgebrochen. Es liegt keine abschließend geprüfte Antwort vor.';
                }
            } else if (error && (error.kind === 'http' || error.kind === 'unauthorized')) {
                _reportChatClientError({
                    chatId: this.currentChatId || earlyCreatedChatId,
                    errorKind: error.kind,
                    errorMessage: String(error?.message || ''),
                    mode,
                });
                const el = bubbleEl || this._view.startAssistantMessage({ createdAt: new Date().toISOString(), showCopyButton: false });
                if (el) {
                    const textEl = el.querySelector('.chat-text');
                    if (textEl) textEl.textContent = 'Fehler bei der Abfrage.';
                }
            } else {
                _reportChatClientError({
                    chatId: this.currentChatId || earlyCreatedChatId,
                    errorKind: 'network',
                    errorMessage: String(error?.message || error || ''),
                    mode,
                });
                const el = bubbleEl || this._view.startAssistantMessage({ createdAt: new Date().toISOString(), showCopyButton: false });
                if (el) {
                    const textEl = el.querySelector('.chat-text');
                    if (textEl) textEl.textContent = 'Netzwerkfehler bei der Abfrage.';
                }
            }
        } finally {
            if (this._streamGeneration === myGeneration) {
                if (this._view && bubbleEl) {
                    const progress = this._view.getCustomBlockEl('research-progress', {target: bubbleEl});
                    if (progress) progress.remove();
                }
                if (sendBtn) sendBtn.disabled = false;
                this.isStreaming = false;
                this._abortController = null;
                if (this._view) this._view.showStopButton(false);
            }
        }
    }

    abortStream() {
        if (this._abortController) {
            this._abortController.abort();
        }
    }

    _appendModeBadge(bubbleEl, mode) {
        if (!bubbleEl || mode === 'standard') return;
        const bubble = bubbleEl.querySelector('.chat-bubble');
        if (!bubble || bubble.querySelector('.chat-mode-badge')) return;
        const modeSelect = document.getElementById('chatFilterMode');
        let label = mode === 'research' ? 'Recherche' : mode === 'unchained' ? 'Unchained' : (mode === 'agent' ? 'Agent' : mode);
        let cls = mode === 'unchained' ? 'unchained' : (mode === 'agent' ? 'agent' : '');
        if (modeSelect) {
            const selectedOpt = modeSelect.options[modeSelect.selectedIndex];
            if (selectedOpt) label = selectedOpt.textContent || label;
        }
        const badgeHtml = `<span class="chat-mode-badge ${cls}">${SharedUi.runtime.escapeHtml(label)}</span>`;
        const timestamp = bubble.querySelector('.chat-timestamp');
        if (timestamp) {
            timestamp.insertAdjacentHTML('afterend', badgeHtml);
        } else {
            bubble.insertAdjacentHTML('afterbegin', badgeHtml);
        }
    }

    _appendRewriteInfo(bubbleEl, data) {
        if (!bubbleEl) return;
        const bubble = bubbleEl.querySelector('.chat-bubble');
        if (!bubble || bubble.querySelector('.chat-rewrite-info')) return;
        const rewritten = SharedUi.runtime.escapeHtml(data.rewritten || '');
        if (!rewritten) return;
        this._view.appendCustomBlock(
            `<div class="chat-rewrite-info">
                <span class="rewrite-label">Suchbegriff:</span> <span class="rewrite-tag">${rewritten}</span>
            </div>`,
            { target: bubbleEl, slot: 'beforeText', id: 'rewrite', sanitize: false }
        );
    }

    _appendExpansionInfo(bubbleEl, data) {
        if (!bubbleEl) return;
        const bubble = bubbleEl.querySelector('.chat-bubble');
        if (!bubble || bubble.querySelector('.chat-expansion-info')) return;
        const variants = (data.variants || []).map(v => SharedUi.runtime.escapeHtml(v));
        if (!variants.length) return;
        const tags = variants.map(v => `<span class="expansion-tag">${v}</span>`).join('');
        this._view.appendCustomBlock(
            `<div class="chat-expansion-info">
                <span class="expansion-label">Query-Expansion:</span> ${tags}
            </div>`,
            { target: bubbleEl, slot: 'beforeText', id: 'expansion', sanitize: false }
        );
    }

    _renderProcessingWarnings(bubbleEl, warnings) {
        if (!bubbleEl || !this._view) return;
        const entries = Array.isArray(warnings) ? warnings.filter(w => w && typeof w.message === 'string') : [];
        const existing = this._view.getCustomBlockEl('processing-warnings', { target: bubbleEl });
        if (!entries.length) {
            if (existing) existing.remove();
            return;
        }
        const items = entries.map(w => {
            const count = Number(w.count);
            const repeats = Number.isSafeInteger(count) && count > 1 ? ` (${count}× aufgetreten)` : '';
            let message = w.code === 'reranking_unavailable'
                ? w.message.replace('Relevanzsortierung', 'Relevanzsortierung (Reranking)')
                : w.message;
            if (w.reason === 'window_estimate_exceeded' && w.details) {
                const estimated = Number(w.details.estimated_tokens);
                const limit = Number(w.details.num_ctx);
                if (Number.isSafeInteger(estimated) && Number.isSafeInteger(limit)) {
                    message += ` Geschätzt: ${estimated.toLocaleString('de-DE')} Tokens; Kontextfenster: ${limit.toLocaleString('de-DE')} Tokens.`;
                }
            }
            if (w.component === 'research' && w.details) {
                const coverage = w.details.coverage || {};
                message += ` Gelesene Bereiche: ${Number(coverage.pages_read || 0)}; vollständig gelesene Dokumente: ${Number(coverage.documents_read_complete || 0)} von ${Number(coverage.documents_found || 0)}. Kontextbudgets basieren auf Tokenschätzungen.`;
            }
            return `${SharedUi.runtime.escapeHtml(message)}${repeats}`;
        }).join(' · ');
        const researchOnly = entries.every(w => w.component === 'research');
        const title = researchOnly ? 'Recherche-Prüfstand:' : 'Eingeschränkte Suchqualität:';
        const html = `<div class="chat-processing-warning${researchOnly ? ' chat-research-warning' : ''}" role="status" aria-live="polite" aria-atomic="true" tabindex="0"><strong>${title}</strong> ${items}</div>`;
        if (existing) {
            existing.innerHTML = html;
        } else {
            this._view.appendCustomBlock(html, {
                target: bubbleEl, slot: 'beforeText', id: 'processing-warnings', sanitize: false,
            });
        }
    }

    _appendError(bubbleEl, message) {
        if (!bubbleEl) return;
        const bubble = bubbleEl.querySelector('.chat-bubble');
        if (!bubble) return;
        const textEl = bubble.querySelector('.chat-text');
        if (textEl) textEl.remove();
        // Copy-Button entfernen — ohne chat-text gibt es nichts zu kopieren
        const copyBtn = bubble.querySelector('.msg-copy-btn');
        if (copyBtn) copyBtn.remove();
        this._view.appendCustomBlock(
            `<div class="chat-context-warning chat-context-warning--error">
                LLM-Backend-Fehler: ${SharedUi.runtime.escapeHtml(message)}
            </div>`,
            { target: bubbleEl, slot: 'beforeText', id: 'error', sanitize: false }
        );
    }

    _updateLastToolResult(bubbleEl, data) {
        if (!bubbleEl) return;
        const calls = bubbleEl.querySelectorAll('.chat-tool-call');
        const lastCall = calls.length > 0 ? calls[calls.length - 1] : null;
        if (!lastCall) return;
        const indicator = lastCall.querySelector('.chat-tool-status');
        const summary = lastCall.querySelector('.chat-tool-summary');
        if (indicator) {
            indicator.classList.remove('pending');
            indicator.classList.add(data.success ? 'success' : 'error');
        }
        if (summary) {
            summary.textContent = data.summary || (data.success ? 'Erledigt' : 'Fehler');
        }
        const container = bubbleEl.querySelector('.chat-tool-container');
        if (container) {
            const countEl = container.querySelector('.chat-tool-count');
            const n = container.querySelectorAll('.chat-tool-call').length;
            if (countEl) countEl.textContent = `${n} Tool-Aufruf${n !== 1 ? 'e' : ''}`;
        }
        if (this._view) this._view.scrollToBottom();
    }

    scrollToBottom(force = false) {
        if (this._view) this._view.scrollToBottom(force);
    }

    loadChat(chatId, messages) {
        this.currentChatId = chatId;
        this.render(messages);
    }

    newChat() {
        this.currentChatId = null;
        this.render([]);
    }
}

// Degradation-Banner: Pollt alle 60s den Backend-Status
(function initDegradationBanner() {
    const bannerId = 'degradation-banner';

    function updateBanner(mode, message) {
        let banner = document.getElementById(bannerId);
        if (mode === 'FULL') {
            if (banner) banner.remove();
            return;
        }
        if (!banner) {
            banner = document.createElement('div');
            banner.id = bannerId;
            banner.className = 'degradation-banner';
            const chatInput = document.querySelector('.chat-input-area');
            if (chatInput) {
                chatInput.parentNode.insertBefore(banner, chatInput);
            } else {
                const chatContainer = document.querySelector('.chat-container') || document.body;
                chatContainer.prepend(banner);
            }
        }
        banner.textContent = message;
    }

    async function pollStatus() {
        try {
            const data = await SharedUi.fetch.safeFetchJson('/api/health/degradation-mode');
            updateBanner(data.mode, data.message);
        } catch (e) { /* Netzwerk-/HTTP-Fehler — Banner behalten */ }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', pollStatus);
    } else {
        pollStatus();
    }
    SharedUi.runtime.polling.start('chat-degradation', pollStatus, 60000);
})();
