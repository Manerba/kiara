/**
 * KIara - Admin Chat UI Class
 * Wiring um SharedUi.chat.ChatView (ui_core/js/chat_primitives.js) fuer Admin-Chat.
 * Markdown/Copy/Think kommen aus SharedUi.markdown (ui_core/js/markdown.js);
 * Timestamps aus formatKiaraTimestamp in chat_utils.js.
 * Unterschiede zu ChatUI: Cross-Origin SSE (credentials: include), kein DS-Filter,
 * Agent-only Mode + produktspezifische Admin-Events (confirmation, wizard, dismissed,
 * user_masked) als Custom-Bubbles / In-place-Wizard-Updates.
 *
 * Hinweis: Dieser Service nutzt bewusst direktes fetch() statt SharedUi.fetch.safeFetch,
 * weil der SSE-Stream ueber ReadableStream + manuelles Event-Parsing laeuft und die
 * Cross-Origin-URL (Port 8503) dynamisch konstruiert wird. 401-Redirect wird lokal
 * nachgebildet, weil der globale fetch-Interceptor entfernt wurde. Siehe Sprint 2 Brief.
 */

function _safeInnerHTML(el, html) {
    if (typeof DOMPurify !== 'undefined') {
        el.innerHTML = DOMPurify.sanitize(html);
    } else {
        console.error('DOMPurify nicht verfuegbar — Fallback auf textContent');
        el.textContent = html;
    }
}

function _adminChatRedirect401(resp) {
    if (resp && resp.status === 401) {
        const cfg = (window.SharedUi && typeof SharedUi._config === 'function') ? SharedUi._config() : null;
        window.location.href = (cfg && cfg.loginUrl) || '/login';
        return true;
    }
    return false;
}

// Secret-verdaechtiger Text fuer optimistischen Leak-Schutz vor user_masked-Antwort
const _SECRET_HINT_RE = /(password|passwort|api[_\s-]?key|token|secret|geheim)/i;

function _isSecretSuspect(text) {
    return typeof text === 'string' && _SECRET_HINT_RE.test(text);
}

// Admin-spezifisches Source-Rendering (ohne Editor-Link, wie bisher)
function _adminRenderSourceItem(src) {
    const name = SharedUi.runtime.escapeHtml(src && src.source ? src.source : '');
    const scoreRaw = Number(src && src.score);
    const score = Number.isFinite(scoreRaw) ? scoreRaw.toFixed(3) : '';
    const excerpt = SharedUi.runtime.escapeHtml(src && src.excerpt ? src.excerpt : '');
    return `<div class="chat-source-item">
        <div class="chat-source-header">
            <span class="chat-source-name">${name}</span>
            ${score ? `<span class="chat-source-score">Relevanz: ${score}</span>` : ''}
        </div>
        <div class="chat-source-excerpt">${excerpt}</div>
    </div>`;
}

class AdminChatUI {
    constructor(containerId, baseUrl) {
        this.containerId = containerId;
        this.baseUrl = baseUrl;  // z.B. http://rag:8503
        this.currentChatId = null;
        this.draftGreeting = null;
        this.draftGreetingType = null;
        this.isStreaming = false;
        this._abortController = null;
        this._creating = false;
        this._onCreateChat = null;
        this._titleSetForChat = null;
        this._titlePutInFlight = new Set();
        this._view = null;
        this._streamGeneration = 0;
        this._backendLoadSeq = 0;
        this._pendingUserBubble = null;
        this._destroyed = false;
        this._inputEl = null;
        this._sendBtnEl = null;
        this._stopBtnEl = null;
        this._onInputKeydown = null;
        this._onInput = null;
        this._onSendClick = null;
        this._onStopClick = null;
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
            renderInput: false, // Admin-Chat rendert eine eigene Input-Row mit Backend-Select
            formatTimestamp: formatKiaraTimestamp,
            avatarLabels: { user: 'Du', assistant: 'KI' },
            renderSourceItem: _adminRenderSourceItem,
            messagesId: 'adminChatMessages',
            messagesSelector: '#adminChatMessages',
            containerClass: 'admin-chat-container',
            messagesClass: 'admin-chat-messages',
            adoptExistingContainer: true,
        });
    }

    render(messages = []) {
        const container = document.getElementById(this.containerId);
        if (!container) return;
        this._unbindEvents();

        // Shell fuer Admin-Chat manuell rendern (wegen Backend-Select in Input-Row).
        let shell = '<div class="admin-chat-container">';
        shell += '<div class="admin-chat-messages" id="adminChatMessages"></div>';
        shell += `
            <div class="admin-chat-input-area">
                <div class="admin-chat-input-row">
                    <textarea class="chat-input" id="adminChatInput" placeholder="Frage eingeben..." rows="1"></textarea>
                    <button type="button" class="chat-send-btn" id="adminChatSendBtn">Senden</button>
                    <button type="button" class="chat-stop-btn u-hidden" id="adminChatStopBtn" title="Abbrechen">
                        <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><rect x="2" y="2" width="12" height="12" rx="2" fill="currentColor"/></svg>
                    </button>
                </div>
                <div class="admin-chat-input-options">
                    <select id="adminChatBackend" class="admin-chat-backend-select" title="LLM-Backend">
                        <option value="">Standard-Backend</option>
                    </select>
                </div>
            </div>`;
        shell += '</div>';
        container.innerHTML = shell;

        this._buildView();
        this._view.render();

        const messagesEl = document.getElementById('adminChatMessages');
        if (messages.length === 0 && messagesEl) {
            messagesEl.insertAdjacentHTML('beforeend', `<div class="chat-empty">
                <div class="chat-empty-icon">&#x1F916;</div>
                <div class="chat-empty-text">KIara Admin-Chat</div>
                <div class="chat-empty-hint">Stelle Fragen zum System oder fuehre Admin-Funktionen per Chat aus.</div>
            </div>`);
        } else {
            this._renderMessages(messages);
        }

        this._bindEvents();
        this._loadBackends();
        this._view.scrollToBottom(true);
    }

    hydrateServerState(state = {}) {
        if (!this._view) return;

        const confirmations = Array.isArray(state.pending_confirmations)
            ? state.pending_confirmations
            : [];
        confirmations.forEach(item => {
            const params = this._parseJsonObject(item.params);
            const confirmString = item.confirm_string || '';
            this._renderConfirmationBubble({
                message: item.message || (confirmString ? `Bitte bestaetigen: ${confirmString}` : ''),
                confirm_string: confirmString,
                function_name: item.function_name || '',
                params,
            });
        });

        const activeWizard = state.active_wizard;
        if (activeWizard) {
            const collected = activeWizard.collected || this._parseJsonObject(activeWizard.collected_params);
            this._renderWizardChecklist({
                wizard_id: activeWizard.id || activeWizard.wizard_id || 'active',
                function_name: activeWizard.function_name || '',
                collected,
                missing_required: Array.isArray(activeWizard.missing_required) ? activeWizard.missing_required : [],
                optional: Array.isArray(activeWizard.optional) ? activeWizard.optional : [],
                parameter_help: activeWizard.parameter_help || {},
            });
        }
    }

    _parseJsonObject(value) {
        if (value && typeof value === 'object' && !Array.isArray(value)) return value;
        if (typeof value !== 'string' || value.length === 0) return {};
        try {
            const parsed = JSON.parse(value);
            return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
        } catch (_) {
            return {};
        }
    }

    _renderMessages(messages) {
        messages.forEach(msg => {
            let metadata = {};
            if (typeof msg.metadata === 'string') {
                try { metadata = JSON.parse(msg.metadata || '{}'); } catch (e) { /* ignore */ }
            } else if (msg.metadata && typeof msg.metadata === 'object') {
                metadata = msg.metadata;
            }
            const isGreeting = metadata.type === 'greeting' || metadata.type === 'setup_wizard';
            const isUser = msg.role === 'user';
            if (isUser) {
                this._view.appendUserMessage(msg.content || '', { createdAt: msg.created_at });
                return;
            }
            const el = this._view.startAssistantMessage({ createdAt: msg.created_at });
            if (!el) return;
            if (isGreeting) el.classList.add('greeting');
            el.dataset.chatTextAnchored = '1';

            // Tool-Calls (aus DB) oberhalb der Antwort
            let toolCalls = msg.tool_calls;
            if (typeof toolCalls === 'string') {
                try { toolCalls = JSON.parse(toolCalls); } catch { toolCalls = null; }
            }
            if (toolCalls && toolCalls.length > 0) {
                toolCalls.forEach((tc, idx) => {
                    const name = tc.name || tc.tool || 'unknown';
                    const params = tc.params || tc.arguments || '';
                    const paramStr = typeof params === 'object' ? JSON.stringify(params) : String(params);
                    const success = tc.success !== false;
                    const summary = tc.summary || tc.result || (success ? 'Erledigt' : 'Fehler');
                    this._view.appendToolBlock({
                        target: el,
                        id: `h-${idx}`,
                        name,
                        params: paramStr,
                        summary,
                        status: success ? 'success' : 'error',
                    });
                });
            }

            const bubble = el.querySelector('.chat-bubble');
            const textEl = bubble ? bubble.querySelector('.chat-text') : null;
            if (textEl) textEl.innerHTML = SharedUi.markdown.render(msg.content || '', { thinkBlocks: true });

            // Sources aus DB (JSON-String)
            let sources = msg.sources;
            if (typeof sources === 'string') {
                try { sources = JSON.parse(sources); } catch { sources = null; }
            }
            if (sources && sources.length > 0) {
                this._view.setSources(sources, { target: el });
            }

            // Usage aus Metadata
            if (metadata.usage) {
                this._view.setUsage(metadata.usage, { target: el });
            }
        });
        this._view.scrollToBottom(true);
    }

    _unbindEvents() {
        if (this._inputEl && this._onInputKeydown) {
            this._inputEl.removeEventListener('keydown', this._onInputKeydown);
        }
        if (this._inputEl && this._onInput) {
            this._inputEl.removeEventListener('input', this._onInput);
        }
        if (this._sendBtnEl && this._onSendClick) {
            this._sendBtnEl.removeEventListener('click', this._onSendClick);
        }
        if (this._stopBtnEl && this._onStopClick) {
            this._stopBtnEl.removeEventListener('click', this._onStopClick);
        }
        this._inputEl = null;
        this._sendBtnEl = null;
        this._stopBtnEl = null;
        this._onInputKeydown = null;
        this._onInput = null;
        this._onSendClick = null;
        this._onStopClick = null;
    }

    _bindEvents() {
        const input = document.getElementById('adminChatInput');
        const sendBtn = document.getElementById('adminChatSendBtn');
        const stopBtn = document.getElementById('adminChatStopBtn');

        if (input) {
            this._inputEl = input;
            this._onInputKeydown = (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            };
            this._onInput = () => {
                // Auto-Resize via nonced runtimeStyle (CSP-strict, V30 Welle 3).
                SharedUi.runtimeStyle.applyStyle(input, { height: 'auto' });
                SharedUi.runtimeStyle.applyStyle(input, { height: Math.min(input.scrollHeight, 120) + 'px' });
            };
            input.addEventListener('keydown', this._onInputKeydown);
            input.addEventListener('input', this._onInput);
        }

        if (sendBtn) {
            this._sendBtnEl = sendBtn;
            this._onSendClick = () => this.sendMessage();
            sendBtn.addEventListener('click', this._onSendClick);
        }
        if (stopBtn) {
            this._stopBtnEl = stopBtn;
            this._onStopClick = () => this.abortStream();
            stopBtn.addEventListener('click', this._onStopClick);
        }
    }

    async _loadBackends() {
        // Eigener Counter, damit ein Chat-Wechsel (der render()->_loadBackends() triggert)
        // nicht den Stream-Counter bumpt und den sendMessage-finally-Check verfehlen laesst
        // (Deadlock: isStreaming=true, Send-Button disabled).
        const myGen = ++this._backendLoadSeq;
        try {
            const resp = await fetch(`${this.baseUrl}/api/admin-chat/backends`, {
                credentials: 'include',
                headers: { 'X-Requested-With': 'XMLHttpRequest' },
            });
            if (_adminChatRedirect401(resp)) return;
            if (!resp.ok) return;
            const backends = await resp.json();
            if (this._destroyed) return;
            if (myGen !== this._backendLoadSeq) return;
            const select = document.getElementById('adminChatBackend');
            if (!select) return;
            // Idempotent: bestehende Optionen (ausser Platzhalter) entfernen
            Array.from(select.options).forEach(opt => {
                if (opt.value !== '') opt.remove();
            });
            const prev = select.value;
            backends.forEach(b => {
                const opt = document.createElement('option');
                opt.value = b.id;
                opt.textContent = `${b.name} (${b.provider})`;
                select.appendChild(opt);
            });
            if (prev && Array.from(select.options).some(o => o.value === prev)) {
                select.value = prev;
            }
        } catch (e) {
            console.warn('Admin-Chat: Backend-Liste konnte nicht geladen werden:', e);
        }
    }

    async sendMessage() {
        const input = document.getElementById('adminChatInput');
        const sendBtn = document.getElementById('adminChatSendBtn');
        if (!input || this.isStreaming || this._destroyed) return;

        const question = input.value.trim();
        if (!question) return;

        // Lazy-Creation: Draft -> persistierter Chat
        if (!this.currentChatId) {
            if (this._creating) return;
            this._creating = true;
            try {
                if (!this._onCreateChat) {
                    console.error('Admin-Chat: Kein _onCreateChat Callback');
                    return;
                }
                const chat = await this._onCreateChat(this.draftGreeting, this.draftGreetingType);
                if (this._destroyed) return;
                if (!chat) return;
                this.currentChatId = chat.id;
                this.draftGreeting = null;
                this.draftGreetingType = null;
            } finally {
                this._creating = false;
            }
        }
        if (this._destroyed) return;

        input.value = '';
        SharedUi.runtimeStyle.applyStyle(input, { height: 'auto' });
        sendBtn.disabled = true;
        this.isStreaming = true;
        this._abortController = new AbortController();
        const streamChatId = this.currentChatId;
        const myGeneration = ++this._streamGeneration;

        const stopBtn = document.getElementById('adminChatStopBtn');
        if (stopBtn) stopBtn.classList.remove('u-hidden');

        // Optimistischer Leak-Schutz: Bei secret-verdaechtigen Prompts neutralen
        // Platzhalter rendern. user_masked-Event ersetzt den Text spaeter.
        const secretSuspect = _isSecretSuspect(question);
        const displayedUserText = secretSuspect ? '\u2026' : question;
        const userEl = this._view.appendUserMessage(displayedUserText, { createdAt: new Date().toISOString() });
        if (userEl) {
            userEl.dataset.pendingMask = secretSuspect ? 'suspect' : 'pending';
            userEl.dataset.streamGen = String(myGeneration);
        }
        this._pendingUserBubble = userEl;

        const backendSelect = document.getElementById('adminChatBackend');
        const llmBackendId = backendSelect && backendSelect.value ? parseInt(backendSelect.value) : null;

        const typingEl = this._view.appendStreamingIndicator({ id: 'adminChatTyping', role: 'assistant', label: 'Denke nach\u2026' });
        let bubbleEl = null;
        let toolOrdinal = 0;

        try {
            const body = {
                question,
                llm_backend_id: llmBackendId,
                mode: 'agent',
                use_history: true,
                think: true,
            };

            const streamUrl = `${this.baseUrl}/api/admin-chat/chats/${streamChatId}/query`;
            const initialController = this._abortController;
            const buildStreamOptions = (controller) => ({
                method: 'POST',
                headers: Object.assign(
                    { 'Content-Type': 'application/json' },
                    (window.SharedUi && window.SharedUi.fetch && window.SharedUi.fetch.csrfHeaders
                        ? window.SharedUi.fetch.csrfHeaders()
                        : { 'X-Requested-With': 'XMLHttpRequest' }),
                ),
                credentials: 'include',
                body: JSON.stringify(body),
                signal: controller.signal,
            });
            let response = await fetch(streamUrl, buildStreamOptions(initialController));

            // Streaming-Retry-Ausschluss aus safeFetch: Bei 403+csrf_invalid
            // manuell refreshen und den kompletten Request neu aufsetzen.
            if (response.status === 403 && window.SharedUi && window.SharedUi.fetch && window.SharedUi.fetch.refreshCsrfToken) {
                let parsed = null;
                try { parsed = await response.clone().json(); } catch (_) {}
                if (parsed && parsed.code === 'csrf_invalid') {
                    const fresh = await window.SharedUi.fetch.refreshCsrfToken();
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
                        response = await fetch(streamUrl, buildStreamOptions(this._abortController));
                    }
                }
            }

            if (_adminChatRedirect401(response)) return;
            if (!response.ok) {
                this._view.removeStreamingIndicator(typingEl);
                const errText = await response.text().catch(() => 'Unbekannter Fehler');
                const errEl = this._view.startAssistantMessage({
                    createdAt: new Date().toISOString(), showCopyButton: false,
                });
                if (errEl) {
                    const textEl = errEl.querySelector('.chat-text');
                    if (textEl) textEl.textContent = `Fehler bei der Abfrage: ${errText}`;
                }
                return;
            }

            this._view.removeStreamingIndicator(typingEl);
            bubbleEl = this._view.startAssistantMessage({
                createdAt: new Date().toISOString(), showCopyButton: true,
            });
            let fullText = '';
            let pendingSources = [];
            let hasVisibleContent = false;

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    buffer += decoder.decode();
                    break;
                }

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
                            if (currentEvent === 'chat_created') {
                                if (data && data.chat_id && myGeneration === this._streamGeneration) {
                                    if (!this.currentChatId || this.currentChatId === streamChatId) {
                                        this.currentChatId = data.chat_id;
                                    }
                                }
                            } else if (currentEvent === 'sources') {
                                pendingSources = Array.isArray(data) ? data : [];
                            } else if (currentEvent === 'chunk') {
                                hasVisibleContent = true;
                                fullText += data.text || '';
                                this._view.appendAssistantChunk(fullText, { target: bubbleEl, thinkBlocks: true });
                            } else if (currentEvent === 'done') {
                                if (data.chat_id && myGeneration === this._streamGeneration
                                    && this.currentChatId === streamChatId) {
                                    this.currentChatId = data.chat_id;
                                }
                            } else if (currentEvent === 'mode') {
                                this._appendModeBadge(bubbleEl, data.mode || '');
                            } else if (currentEvent === 'tool_call') {
                                hasVisibleContent = true;
                                const ord = ++toolOrdinal;
                                this._view.appendToolBlock({
                                    target: bubbleEl,
                                    id: `a-${data.iteration || 0}-${ord}`,
                                    name: data.tool || '',
                                    params: JSON.stringify(data.params || {}),
                                    summary: 'Ausfuehrung...',
                                    status: 'pending',
                                });
                            } else if (currentEvent === 'tool_result') {
                                this._updateLastToolResult(bubbleEl, data);
                            } else if (currentEvent === 'usage') {
                                this._view.setUsage(data, { target: bubbleEl });
                            } else if (currentEvent === 'confirmation_required') {
                                this._renderConfirmationBubble(data);
                            } else if (currentEvent === 'confirmation_result') {
                                this._renderConfirmationResult(data);
                            } else if (currentEvent === 'wizard_update') {
                                this._renderWizardChecklist(data);
                            } else if (currentEvent === 'action_dismissed') {
                                this._renderDismissedNotice(data);
                            } else if (currentEvent === 'user_masked') {
                                this._applyUserMask(data, myGeneration);
                            } else if (currentEvent === 'error') {
                                hasVisibleContent = true;
                                this._appendError(bubbleEl, data.message || 'Unbekannter Fehler');
                            }
                        } catch (e) { console.warn('SSE-Parse-Fehler:', currentEvent, dataStr, e); }
                        currentEvent = '';
                    }
                }
            }

            // Wenn der Stream nur Custom-Events produzierte (keine Chunks/Tool-Calls),
            // entferne die leere Assistant-Bubble.
            if (!hasVisibleContent && bubbleEl && bubbleEl.parentNode) {
                const textEl = bubbleEl.querySelector('.chat-text');
                const hasText = textEl && textEl.textContent.trim().length > 0;
                const hasTool = bubbleEl.querySelector('.chat-tool-container');
                if (!hasText && !hasTool) {
                    bubbleEl.remove();
                    bubbleEl = null;
                }
            }

            if (pendingSources.length > 0 && bubbleEl) {
                this._view.setSources(pendingSources, { target: bubbleEl });
            }
            if (bubbleEl) this._view.finishAssistantMessage({ target: bubbleEl });

            // Auto-Title
            this._maybeUpdateTitle(question, streamChatId);

        } catch (error) {
            if (this._destroyed) return;
            this._view.removeStreamingIndicator(typingEl);
            if (error && error.name !== 'AbortError') {
                const errEl = this._view.startAssistantMessage({
                    createdAt: new Date().toISOString(), showCopyButton: false,
                });
                if (errEl) {
                    const textEl = errEl.querySelector('.chat-text');
                    if (textEl) textEl.textContent = 'Netzwerkfehler bei der Abfrage.';
                }
            }
        } finally {
            if (this._streamGeneration === myGeneration) {
                if (sendBtn) sendBtn.disabled = false;
                this.isStreaming = false;
                this._abortController = null;
                const stopBtnRef = document.getElementById('adminChatStopBtn');
                if (stopBtnRef) stopBtnRef.classList.add('u-hidden');
            }
        }
    }

    abortStream() {
        if (this._abortController) {
            this._abortController.abort();
        }
    }

    destroy() {
        this._destroyed = true;
        this._unbindEvents();
        if (this._abortController) {
            try { this._abortController.abort(); } catch (e) { /* ignore */ }
            this._abortController = null;
        }
        if (this._view) {
            try { this._view.destroy(); } catch (e) { /* ignore */ }
            this._view = null;
        }
        window.removeEventListener('pagehide', this._onPageHide);
    }

    scrollToBottom(force) {
        if (this._view) this._view.scrollToBottom(force);
    }

    _applyUserMask(data, streamGen) {
        // Ziel: User-Bubble dieser Stream-Generation, nicht ein globaler erster Treffer.
        // _pendingUserBubble nur uebernehmen, wenn streamGen passt — sonst koennte
        // ein spaeter eintreffendes user_masked-Event aus einer alten Generation
        // die Bubble eines aktuellen, ueberlappenden Streams ueberschreiben.
        const pendingMatchesGen = this._pendingUserBubble
            && document.body.contains(this._pendingUserBubble)
            && this._pendingUserBubble.dataset.streamGen === String(streamGen);
        const target = pendingMatchesGen
            ? this._pendingUserBubble
            : document.querySelector(`#adminChatMessages .chat-message.user[data-stream-gen="${streamGen}"]`);
        if (!target) return;
        const textEl = target.querySelector('.chat-text');
        if (textEl) textEl.textContent = data.text || '(API-Key eingegeben)';
        target.removeAttribute('data-pending-mask');
        if (this._pendingUserBubble === target) this._pendingUserBubble = null;
    }

    _appendModeBadge(bubbleEl, mode) {
        if (!bubbleEl || mode === 'standard') return;
        const bubble = bubbleEl.querySelector('.chat-bubble');
        if (!bubble || bubble.querySelector('.chat-mode-badge')) return;
        const label = mode === 'agent' ? 'Agent' : mode;
        const cls = mode === 'agent' ? 'agent' : '';
        bubble.insertAdjacentHTML('afterbegin',
            `<span class="chat-mode-badge ${cls}">${SharedUi.runtime.escapeHtml(label)}</span>`);
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

    _appendError(bubbleEl, message) {
        if (!bubbleEl || !this._view) return;
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

    _renderConfirmationBubble(data) {
        const paramsHtml = Object.entries(this._parseJsonObject(data.params))
            .map(([k, v]) => `<span class="param-item">${SharedUi.runtime.escapeHtml(k)} = ${SharedUi.runtime.escapeHtml(String(v))}</span>`)
            .join(', ');
        const html = `<div class="chat-message system confirmation">
            <div class="chat-avatar system-avatar">!</div>
            <div class="chat-bubble confirmation-bubble">
                <div class="confirmation-header">Systembestaetigung</div>
                <div class="confirmation-message">${SharedUi.runtime.escapeHtml(data.message || '')}</div>
                <div class="confirmation-prompt">
                    Tippen Sie zur Bestaetigung: <code class="confirm-string">${SharedUi.runtime.escapeHtml(data.confirm_string || '')}</code>
                </div>
                <div class="confirmation-details">
                    <span class="func-name">${SharedUi.runtime.escapeHtml(data.function_name || '')}</span>
                    ${paramsHtml ? `<br>Parameter: ${paramsHtml}` : ''}
                </div>
            </div>
        </div>`;
        this._view.appendCustomBubble(html, { sanitize: true });
    }

    _renderConfirmationResult(data) {
        const result = data.result || {};
        const html = `<div class="chat-message system confirmation-result">
            <div class="chat-avatar system-avatar">${result.success ? 'OK' : '!'}</div>
            <div class="chat-bubble ${result.success ? 'confirmation-success' : 'confirmation-error'}">
                ${SharedUi.runtime.escapeHtml(result.message || 'Aktion ausgefuehrt')}
            </div>
        </div>`;
        this._view.appendCustomBubble(html, { sanitize: true });
    }

    _renderDismissedNotice(data) {
        const html = `<div class="chat-message system dismissed">
            <div class="chat-avatar system-avatar">i</div>
            <div class="chat-bubble dismissed-bubble">
                Aktion verworfen: ${SharedUi.runtime.escapeHtml(data.reason || '')}
            </div>
        </div>`;
        this._view.appendCustomBubble(html, { sanitize: true });
    }

    _renderWizardChecklist(data) {
        const wizardKey = `wizard-${data.wizard_id}`;
        let wrapper = this._view.getCustomBubbleEl(wizardKey);
        if (!wrapper) {
            const html = `<div class="chat-message system wizard">
                <div class="chat-avatar system-avatar">?</div>
                <div class="chat-bubble wizard-bubble" data-wizard-id="${SharedUi.runtime.escapeAttr(String(data.wizard_id))}"></div>
            </div>`;
            wrapper = this._view.appendCustomBubble(html, { id: wizardKey, sanitize: true });
            if (!wrapper) return;
        }

        const bubble = wrapper.querySelector('.wizard-bubble');
        if (!bubble) return;

        const collected = data.collected || {};
        const missing = data.missing_required || [];
        const optional = data.optional || [];
        const help = data.parameter_help || {};

        let html = `<div class="wizard-header">${SharedUi.runtime.escapeHtml(data.function_name || '')}</div>`;
        html += '<div class="wizard-params">';

        for (const [key, val] of Object.entries(collected)) {
            html += `<div class="wizard-param collected">
                <span class="param-status">OK</span>
                <span class="param-name">${SharedUi.runtime.escapeHtml(key)}:</span>
                <span class="param-value">${SharedUi.runtime.escapeHtml(String(val))}</span>
            </div>`;
        }
        for (const key of missing) {
            const desc = (help[key] && help[key].description) || '';
            html += `<div class="wizard-param missing">
                <span class="param-status">X</span>
                <span class="param-name">${SharedUi.runtime.escapeHtml(key)}:</span>
                <span class="param-desc">${SharedUi.runtime.escapeHtml(desc)}</span>
            </div>`;
        }
        for (const key of optional) {
            if (!(key in collected)) {
                const desc = (help[key] && help[key].description) || 'Standard';
                html += `<div class="wizard-param optional">
                    <span class="param-status">o</span>
                    <span class="param-name">${SharedUi.runtime.escapeHtml(key)}:</span>
                    <span class="param-desc">${SharedUi.runtime.escapeHtml(desc)}</span>
                </div>`;
            }
        }

        html += '</div>';
        html += `<div class="wizard-footer">Noch ${missing.length} Pflichtfeld${missing.length !== 1 ? 'er' : ''} offen.</div>`;
        _safeInnerHTML(bubble, html);
        this._view.scrollToBottom();
    }

    async _maybeUpdateTitle(question, chatId) {
        const targetChatId = chatId || this.currentChatId;
        if (!targetChatId) return;
        // Nur beim ersten User-Input den Titel setzen
        if (this._titleSetForChat === targetChatId) return;
        if (this._titlePutInFlight.has(targetChatId)) return;
        // Manual-Title-Guard (persistent ueber _loadChatList hinweg)
        if (typeof _manualTitleChats !== 'undefined' && _manualTitleChats.has(targetChatId)) return;
        const title = question.substring(0, 80) + (question.length > 80 ? '...' : '');
        this._titlePutInFlight.add(targetChatId);
        const titleUrl = `${this.baseUrl}/api/admin-chat/chats/${targetChatId}`;
        const buildTitleOptions = () => ({
            method: 'PUT',
            headers: Object.assign(
                { 'Content-Type': 'application/json' },
                (window.SharedUi && window.SharedUi.fetch && window.SharedUi.fetch.csrfHeaders
                    ? window.SharedUi.fetch.csrfHeaders()
                    : { 'X-Requested-With': 'XMLHttpRequest' }),
            ),
            credentials: 'include',
            body: JSON.stringify({ title }),
        });
        try {
            let resp = await fetch(titleUrl, buildTitleOptions());
            if (resp.status === 403 && window.SharedUi && window.SharedUi.fetch && window.SharedUi.fetch.refreshCsrfToken) {
                let parsed = null;
                try { parsed = await resp.clone().json(); } catch (_) {}
                if (parsed && parsed.code === 'csrf_invalid') {
                    const fresh = await window.SharedUi.fetch.refreshCsrfToken();
                    if (fresh) {
                        resp = await fetch(titleUrl, buildTitleOptions());
                    }
                }
            }
            if (_adminChatRedirect401(resp)) return;
            if (resp.ok) {
                this._titleSetForChat = targetChatId;
                if (typeof refreshAdminChatSidebar === 'function') {
                    refreshAdminChatSidebar();
                }
            }
        } catch (e) { /* Fire-and-forget */ } finally {
            this._titlePutInFlight.delete(targetChatId);
        }
    }
}
