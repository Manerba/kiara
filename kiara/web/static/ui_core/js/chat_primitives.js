/**
 * SharedUi.chat — produktneutrale Chat-Primitives (Core-Baustein 9).
 *
 * Liefert `SharedUi.chat.ChatView` fuer Container/Messages/Streaming/Sources/
 * Tool-Blocks/Custom-Slots/Input/Scroll/History/Cleanup. Produktspezifisches
 * (SSE-Transport, Tool-/Source-Schema, Trace, Confirmation, Wizard, Setup)
 * bleibt in den jeweiligen Wiring-Layern.
 *
 * Abhaengigkeiten (Reihenfolge im Template):
 *   core.js -> runtime.js -> purify.min.js -> markdown.js -> chat_primitives.js
 *
 * API (19 Methoden): appendUserMessage, startAssistantMessage,
 *   appendAssistantChunk, finishAssistantMessage, setSources, setUsage,
 *   appendToolBlock, getToolBlockEl, appendCustomBlock, getCustomBlockEl,
 *   appendCustomBubble, getCustomBubbleEl, appendStreamingIndicator,
 *   removeStreamingIndicator, clear, loadHistory, scrollToBottom,
 *   showStopButton, destroy.
 */
(function () {
    'use strict';

    if (typeof window.SharedUi === 'undefined') {
        console.error('SharedUi.chat: core.js muss vor chat_primitives.js laden.');
        return;
    }
    if (!SharedUi.runtime || !SharedUi.markdown) {
        console.error('SharedUi.chat: SharedUi.runtime und SharedUi.markdown sind Pflicht.');
        return;
    }

    const DEFAULTS = {
        renderInput: true,
        inputId: 'chatInput',
        sendButtonId: 'chatSendBtn',
        stopButtonId: 'chatStopBtn',
        stopButtonSelector: null,
        onSend: null,
        onStop: null,
        formatTimestamp: null,
        avatarLabels: { user: 'Du', assistant: 'AI' },
        renderAvatar: null,
        renderSourceItem: null,
        onMessageRendered: null,
        adoptExistingContainer: false,
        containerClass: 'chat-container',
        messagesClass: 'chat-messages',
        messagesId: 'chatMessages',
        messagesSelector: null,
        emptyState: null,
        copyButtonLabel: '&#x2398; Kopieren',
        copyButtonTitle: 'Antwort kopieren',
    };

    // Brain-Thinking-SVG (produktneutral; IDs per Instanz eindeutig).
    function _thinkingIndicatorSvg(prefix) {
        const p = String(prefix || 'chat').replace(/[^A-Za-z0-9_-]/g, '_');
        return `
            <svg viewBox="-26 -27 52 53" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <defs>
                    <linearGradient id="${p}G" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#3b82f6"/>
                        <stop offset="100%" stop-color="#8b5cf6"/>
                    </linearGradient>
                    <filter id="${p}F">
                        <feGaussianBlur stdDeviation="2" result="b"/>
                        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                </defs>
                <g stroke="url(#${p}G)" stroke-width="2.5">
                    <animate attributeName="opacity" values="0.25;0.4;0.25" dur="3s" repeatCount="indefinite"/>
                    <path id="${p}P1" d="M-2,-22C-18,-22-22,-12-22,-4C-22,2-18,6-16,8C-20,12-18,20-10,22C-6,23-2,20-2,16"/>
                    <path id="${p}P2" d="M2,-22C18,-22 22,-12 22,-4C22,2 18,6 16,8C20,12 18,20 10,22C6,23 2,20 2,16"/>
                    <path id="${p}P3" d="M0,-22L0,16"/>
                    <path id="${p}P4" d="M-2,-8C-10,-8-14,-4-14,0"/>
                    <path id="${p}P6" d="M2,-8C10,-8 14,-4 14,0"/>
                    <path id="${p}P7" d="M2,4C8,4 12,8 12,12"/>
                </g>
                <g filter="url(#${p}F)">
                    <ellipse rx="1.5" ry="1.5" fill="#60a5fa">
                        <animateMotion dur="2.2s" repeatCount="indefinite" rotate="auto" begin="0s"><mpath href="#${p}P1"/></animateMotion>
                        <animate attributeName="rx" values="1.5;4;1.5" keyTimes="0;0.5;1" dur="2.2s" repeatCount="indefinite" begin="0s"/>
                        <animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.08;0.45;0.88;1" dur="2.2s" repeatCount="indefinite" begin="0s"/>
                    </ellipse>
                    <ellipse rx="1.5" ry="1.5" fill="#a78bfa">
                        <animateMotion dur="2.5s" repeatCount="indefinite" rotate="auto" begin="0.8s"><mpath href="#${p}P2"/></animateMotion>
                        <animate attributeName="rx" values="1.5;4;1.5" keyTimes="0;0.5;1" dur="2.5s" repeatCount="indefinite" begin="0.8s"/>
                        <animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.08;0.45;0.88;1" dur="2.5s" repeatCount="indefinite" begin="0.8s"/>
                    </ellipse>
                    <ellipse rx="1.5" ry="1.5" fill="#93c5fd">
                        <animateMotion dur="1.8s" repeatCount="indefinite" rotate="auto" begin="0.3s" keyPoints="1;0" keyTimes="0;1" calcMode="linear"><mpath href="#${p}P3"/></animateMotion>
                        <animate attributeName="rx" values="1.5;3.5;1.5" keyTimes="0;0.5;1" dur="1.8s" repeatCount="indefinite" begin="0.3s"/>
                        <animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.08;0.45;0.88;1" dur="1.8s" repeatCount="indefinite" begin="0.3s"/>
                    </ellipse>
                    <ellipse rx="1.5" ry="1.5" fill="#60a5fa">
                        <animateMotion dur="1.4s" repeatCount="indefinite" rotate="auto" begin="1.2s"><mpath href="#${p}P4"/></animateMotion>
                        <animate attributeName="rx" values="1.5;3;1.5" keyTimes="0;0.5;1" dur="1.4s" repeatCount="indefinite" begin="1.2s"/>
                        <animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.1;0.45;0.85;1" dur="1.4s" repeatCount="indefinite" begin="1.2s"/>
                    </ellipse>
                    <ellipse rx="1.5" ry="1.5" fill="#a78bfa">
                        <animateMotion dur="1.5s" repeatCount="indefinite" rotate="auto" begin="0.5s" keyPoints="1;0" keyTimes="0;1" calcMode="linear"><mpath href="#${p}P6"/></animateMotion>
                        <animate attributeName="rx" values="1.5;3;1.5" keyTimes="0;0.5;1" dur="1.5s" repeatCount="indefinite" begin="0.5s"/>
                        <animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.1;0.45;0.85;1" dur="1.5s" repeatCount="indefinite" begin="0.5s"/>
                    </ellipse>
                    <ellipse rx="1.5" ry="1.5" fill="#93c5fd">
                        <animateMotion dur="1.6s" repeatCount="indefinite" rotate="auto" begin="1.8s"><mpath href="#${p}P7"/></animateMotion>
                        <animate attributeName="rx" values="1.5;3;1.5" keyTimes="0;0.5;1" dur="1.6s" repeatCount="indefinite" begin="1.8s"/>
                        <animate attributeName="opacity" values="0;1;1;1;0" keyTimes="0;0.1;0.45;0.85;1" dur="1.6s" repeatCount="indefinite" begin="1.8s"/>
                    </ellipse>
                </g>
            </svg>`;
    }

    let _instanceCounter = 0;

    class ChatView {
        constructor(containerId, options) {
            this.containerId = containerId;
            this.options = Object.assign({}, DEFAULTS, options || {});
            this._messagesEl = null;
            this._inputEl = null;
            this._sendBtn = null;
            this._stopBtn = null;
            this._listeners = [];
            this._customBubbles = new Map();
            this._svgPrefix = `cv${++_instanceCounter}_`;
            this._blockIdSeq = 0;
        }

        // ── Helpers ────────────────────────────────────────────

        _getContainer() {
            return document.getElementById(this.containerId);
        }

        _getMessagesEl() {
            if (this._messagesEl && document.body.contains(this._messagesEl)) {
                return this._messagesEl;
            }
            if (this.options.messagesSelector) {
                this._messagesEl = document.querySelector(this.options.messagesSelector);
            } else if (this.options.messagesId) {
                this._messagesEl = document.getElementById(this.options.messagesId);
            }
            return this._messagesEl;
        }

        _fmtTs(dateInput) {
            const fn = this.options.formatTimestamp;
            if (!fn) return '';
            try { return fn(dateInput) || ''; } catch (e) { return ''; }
        }

        _avatarHtml(role) {
            if (this.options.renderAvatar) {
                try {
                    const custom = this.options.renderAvatar(role);
                    if (custom != null) return String(custom);
                } catch (e) { /* fall through */ }
            }
            const labels = this.options.avatarLabels || {};
            const label = labels[role] || (role === 'user' ? 'Du' : 'AI');
            return SharedUi.runtime.escapeHtml(label);
        }

        _removeEmptyState() {
            const el = this._getMessagesEl();
            if (!el) return;
            const empty = el.querySelector('.chat-empty');
            if (empty) empty.remove();
        }

        _notifyRendered(el, info) {
            const cb = this.options.onMessageRendered;
            if (!cb || !el) return;
            try { cb(el, info || {}); } catch (e) { /* swallow */ }
        }

        _sanitize(html) {
            if (typeof DOMPurify !== 'undefined') {
                return DOMPurify.sanitize(html == null ? '' : String(html));
            }
            const tmp = document.createElement('div');
            tmp.textContent = html == null ? '' : String(html);
            return tmp.innerHTML;
        }

        // ── Shell / Lifecycle ──────────────────────────────────

        render() {
            if (this.options.adoptExistingContainer) {
                this._messagesEl = this._getMessagesEl();
            } else {
                this._renderShell();
            }
            if (this._messagesEl && !this._messagesEl.hasAttribute('aria-live')) {
                this._messagesEl.setAttribute('aria-live', 'polite');
            }
            this._bindInputEvents();
            this.scrollToBottom(true);
        }

        _renderShell() {
            const container = this._getContainer();
            if (!container) return;

            const messagesClass = SharedUi.runtime.escapeAttr(this.options.messagesClass);
            const messagesId = SharedUi.runtime.escapeAttr(this.options.messagesId);
            const containerClass = SharedUi.runtime.escapeAttr(this.options.containerClass);

            let html = `<div class="${containerClass}">`;
            html += `<div class="${messagesClass}" id="${messagesId}"></div>`;
            if (this.options.renderInput) {
                const inputId = SharedUi.runtime.escapeAttr(this.options.inputId);
                const sendId = SharedUi.runtime.escapeAttr(this.options.sendButtonId);
                const stopId = SharedUi.runtime.escapeAttr(this.options.stopButtonId);
                html += `
                    <div class="chat-input-area">
                        <textarea class="chat-input" id="${inputId}" placeholder="Frage eingeben..." rows="1"></textarea>
                        <button type="button" class="chat-send-btn" id="${sendId}">Senden</button>
                        <button type="button" class="chat-stop-btn u-hidden" id="${stopId}" title="Abbrechen">
                            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><rect x="2" y="2" width="12" height="12" rx="2" fill="currentColor"/></svg>
                        </button>
                    </div>`;
            }
            html += '</div>';
            container.innerHTML = html;
            this._messagesEl = container.querySelector('.' + this.options.messagesClass);

            if (this.options.emptyState && this._messagesEl) {
                this._messagesEl.insertAdjacentHTML('beforeend', this._sanitize(this.options.emptyState));
            }
        }

        _bindInputEvents() {
            const input = this.options.inputId ? document.getElementById(this.options.inputId) : null;
            const sendBtn = this.options.sendButtonId ? document.getElementById(this.options.sendButtonId) : null;
            const stopBtn = this._resolveStopButton();
            this._inputEl = input;
            this._sendBtn = sendBtn;
            this._stopBtn = stopBtn;

            if (input && this.options.onSend) {
                const keyHandler = (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this._triggerSend();
                    }
                };
                const applyInputHeight = (value) => {
                    window.SharedUi.runtimeStyle.applyStyle(input, { height: value });
                };
                const inputHandler = () => {
                    applyInputHeight('auto');
                    applyInputHeight(Math.min(input.scrollHeight, 120) + 'px');
                };
                input.addEventListener('keydown', keyHandler);
                input.addEventListener('input', inputHandler);
                this._listeners.push({ el: input, event: 'keydown', fn: keyHandler });
                this._listeners.push({ el: input, event: 'input', fn: inputHandler });
            }
            if (sendBtn && this.options.onSend) {
                const fn = () => this._triggerSend();
                sendBtn.addEventListener('click', fn);
                this._listeners.push({ el: sendBtn, event: 'click', fn });
            }
            if (stopBtn && this.options.onStop) {
                const fn = () => {
                    try { this.options.onStop(); } catch (e) { console.error('ChatView onStop:', e); }
                };
                stopBtn.addEventListener('click', fn);
                this._listeners.push({ el: stopBtn, event: 'click', fn });
            }
        }

        _resolveStopButton() {
            if (this.options.stopButtonSelector) return document.querySelector(this.options.stopButtonSelector);
            if (this.options.stopButtonId) return document.getElementById(this.options.stopButtonId);
            return null;
        }

        _triggerSend() {
            if (!this._inputEl || !this.options.onSend) return;
            const text = this._inputEl.value.trim();
            if (!text) return;
            const ctx = {
                clearInput: () => {
                    this._inputEl.value = '';
                    window.SharedUi.runtimeStyle.applyStyle(this._inputEl, { height: 'auto' });
                },
                disableSend: () => { if (this._sendBtn) this._sendBtn.disabled = true; },
                enableSend: () => { if (this._sendBtn) this._sendBtn.disabled = false; },
            };
            let p;
            try { p = this.options.onSend(text, ctx); } catch (e) { console.error('ChatView onSend:', e); }
            if (p && typeof p.catch === 'function') p.catch((err) => console.error('ChatView onSend:', err));
        }

        // ── Message-Rendering ─────────────────────────────────

        appendUserMessage(text, opts) {
            const o = opts || {};
            const messagesEl = this._getMessagesEl();
            if (!messagesEl) return null;
            this._removeEmptyState();

            const ts = this._fmtTs(o.createdAt || null);
            const tsHtml = ts ? `<div class="chat-timestamp">${SharedUi.runtime.escapeHtml(ts)}</div>` : '';
            const avatarHtml = this._avatarHtml('user');
            const textHtml = SharedUi.markdown.render(text || '', { thinkBlocks: false });

            const div = document.createElement('div');
            div.className = 'chat-message user';
            div.innerHTML = `<div class="chat-avatar">${avatarHtml}</div>
                <div class="chat-bubble">${tsHtml}<div class="chat-text">${textHtml}</div></div>`;
            messagesEl.appendChild(div);
            this._notifyRendered(div, { role: 'user', text: text || '' });
            this.scrollToBottom(true);
            return div;
        }

        startAssistantMessage(opts) {
            const o = opts || {};
            const messagesEl = this._getMessagesEl();
            if (!messagesEl) return null;
            this._removeEmptyState();

            const role = o.role || 'assistant';
            const ts = this._fmtTs(o.createdAt || null);
            const tsHtml = ts ? `<div class="chat-timestamp">${SharedUi.runtime.escapeHtml(ts)}</div>` : '';
            const showCopy = o.showCopyButton !== false;
            const title = SharedUi.runtime.escapeAttr(this.options.copyButtonTitle);
            const copyBtn = showCopy
                ? `<button type="button" class="msg-copy-btn" title="${title}" aria-label="${title}">${this.options.copyButtonLabel}</button>`
                : '';
            const avatarHtml = this._avatarHtml(role);

            const div = document.createElement('div');
            div.className = `chat-message ${role}`;
            div.innerHTML = `<div class="chat-avatar">${avatarHtml}</div>
                <div class="chat-bubble">${tsHtml}${copyBtn}<div class="chat-text"></div></div>`;
            messagesEl.appendChild(div);
            this.scrollToBottom(true);
            return div;
        }

        appendAssistantChunk(text, opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target)) return;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble) return;
            const textEl = bubble.querySelector('.chat-text');
            if (!textEl) return;
            if (!target.dataset.chatTextAnchored) {
                target.dataset.chatTextAnchored = '1';
                if (textEl.parentNode) textEl.parentNode.appendChild(textEl);
            }
            const thinkBlocks = o.thinkBlocks !== false;
            textEl.innerHTML = SharedUi.markdown.render(text || '', { thinkBlocks });
            this.scrollToBottom();
        }

        finishAssistantMessage(opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target) return;
            this._notifyRendered(target, { role: 'assistant', finished: true });
        }

        // ── Sources / Usage ────────────────────────────────────

        setSources(sources, opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target)) return;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble) return;
            const existing = bubble.querySelector('.chat-sources');
            if (existing) existing.remove();
            const list = Array.isArray(sources) ? sources : [];
            if (list.length === 0) return;

            const usageAnchor = bubble.querySelector('.chat-usage');
            const block = this._buildSourcesBlock(list);
            if (usageAnchor) {
                bubble.insertBefore(block, usageAnchor);
            } else {
                bubble.appendChild(block);
            }
        }

        _buildSourcesBlock(sources) {
            const wrap = document.createElement('div');
            wrap.className = 'chat-sources';

            const renderer = this.options.renderSourceItem;
            const listId = `chat-src-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
            const n = sources.length;
            const label = `${n} Quelle${n !== 1 ? 'n' : ''}`;

            const items = sources.map((src, i) => {
                if (renderer) {
                    try {
                        const out = renderer(src, i);
                        return (out == null ? '' : String(out));
                    } catch (e) { return ''; }
                }
                return this._defaultSourceItemHtml(src);
            }).join('');

            wrap.innerHTML = `<button type="button" class="chat-sources-toggle" aria-expanded="false" aria-controls="${SharedUi.runtime.escapeAttr(listId)}">
                    <span class="chat-sources-toggle-icon">&#x25B6;</span> ${SharedUi.runtime.escapeHtml(label)}
                </button>
                <div class="chat-sources-list" id="${SharedUi.runtime.escapeAttr(listId)}">${items}</div>`;

            const toggle = wrap.querySelector('.chat-sources-toggle');
            const listEl = wrap.querySelector('.chat-sources-list');
            const icon = wrap.querySelector('.chat-sources-toggle-icon');
            if (toggle && listEl) {
                const toggleFn = () => {
                    const isOpen = listEl.classList.toggle('open');
                    toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                    if (icon) icon.innerHTML = isOpen ? '&#x25BC;' : '&#x25B6;';
                };
                toggle.addEventListener('click', toggleFn);
                toggle.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        toggleFn();
                    }
                });
            }
            return wrap;
        }

        _defaultSourceItemHtml(src) {
            const name = SharedUi.runtime.escapeHtml(src && src.source ? src.source : '');
            const excerpt = SharedUi.runtime.escapeHtml(src && src.excerpt ? src.excerpt : '');
            return `<div class="chat-source-item">
                <div class="chat-source-header">
                    <span class="chat-source-name">${name}</span>
                </div>
                <div class="chat-source-excerpt">${excerpt}</div>
            </div>`;
        }

        setUsage(data, opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target)) return;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble || bubble.querySelector('.chat-usage')) return;

            const prompt = Number(data && data.prompt_tokens) || 0;
            const response = Number(data && data.response_tokens) || 0;
            const total = prompt + response;
            const numCtx = Number(data && data.num_ctx) || 0;
            const durationMs = Number(data && data.total_duration_ms) || 0;

            const durationSec = durationMs / 1000;
            const tps = durationSec > 0 ? (response / durationSec).toFixed(1) : '\u2013';
            let durationLabel = '';
            if (durationSec > 0) {
                durationLabel = durationSec >= 60
                    ? `${Math.floor(durationSec / 60)}m ${(durationSec % 60).toFixed(0)}s`
                    : `${durationSec.toFixed(1)}s`;
            }

            let barHtml = '';
            if (numCtx > 0) {
                const promptPct = Math.min((prompt / numCtx) * 100, 100);
                const responsePct = Math.min((response / numCtx) * 100, 100 - promptPct);
                const promptCls = 'u-pw-' + Math.round(promptPct);
                const responseCls = 'u-pw-' + Math.round(responsePct);
                barHtml = `<div class="chat-usage-bar">
                    <div class="chat-usage-prompt ${promptCls}"></div>
                    <div class="chat-usage-response ${responseCls}"></div>
                </div>`;
            }

            const fmt = (n) => n.toLocaleString('de-DE');
            const model = data && data.model ? String(data.model) : '';
            const ctxLabel = numCtx > 0 ? ` / ${fmt(numCtx)}` : '';
            const tooltipText = `Prompt: ${fmt(prompt)} | Antwort: ${fmt(response)}`;
            const metaParts = [];
            if (model) metaParts.push(SharedUi.runtime.escapeHtml(model));
            if (tps !== '\u2013') metaParts.push(`${tps} t/s`);
            if (durationLabel) metaParts.push(durationLabel);
            const metaHtml = metaParts.length > 0
                ? `<span class="chat-usage-meta">${metaParts.join(' | ')}</span>`
                : '';

            const wasNear = bubble && this._isNearBottom();
            bubble.insertAdjacentHTML('beforeend', `<div class="chat-usage">
                ${barHtml}
                <div class="chat-usage-label">
                    <span title="${SharedUi.runtime.escapeAttr(tooltipText)}">${fmt(total)}${ctxLabel} Tokens</span>
                    ${metaHtml}
                </div>
            </div>`);
            if (wasNear) this.scrollToBottom(true);
        }

        _isNearBottom() {
            const el = this._getMessagesEl();
            if (!el) return true;
            return (el.scrollHeight - el.scrollTop - el.clientHeight) <= 80;
        }

        // ── Tool-Blocks ────────────────────────────────────────

        appendToolBlock(opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target)) return null;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble) return null;

            let container = bubble.querySelector('.chat-tool-container');
            if (!container) {
                const summary = document.createElement('details');
                summary.className = 'chat-tool-container';
                summary.open = true;
                summary.innerHTML = `<summary class="chat-tool-container-summary">
                        <span class="chat-tool-count">1 Tool-Aufruf</span>
                    </summary>
                    <div class="chat-tool-list"></div>`;
                const textEl = bubble.querySelector('.chat-text');
                if (textEl && !target.dataset.chatTextAnchored) {
                    bubble.insertBefore(summary, textEl);
                } else {
                    const sourcesOrUsage = bubble.querySelector('.chat-sources, .chat-usage');
                    if (sourcesOrUsage) bubble.insertBefore(summary, sourcesOrUsage);
                    else if (textEl) bubble.insertBefore(summary, textEl);
                    else bubble.appendChild(summary);
                }
                container = summary;
            }

            const list = container.querySelector('.chat-tool-list');
            if (!list) return null;

            const div = document.createElement('div');
            div.className = 'chat-tool-call';
            if (o.id != null) div.dataset.toolId = String(o.id);

            const name = SharedUi.runtime.escapeHtml(o.name || '');
            const paramsRaw = o.params == null ? '' : String(o.params);
            const paramsHtml = paramsRaw
                ? `(${SharedUi.runtime.escapeHtml(paramsRaw)})`
                : '';
            const summaryText = SharedUi.runtime.escapeHtml(o.summary || 'Ausfuehrung...');
            const statusCls = SharedUi.runtime.escapeAttr(o.status || 'pending');

            div.innerHTML = `<div class="chat-tool-call-header">
                    <span class="chat-tool-status ${statusCls}"></span>
                    <strong>${name}</strong>${paramsHtml}
                </div>
                <div class="chat-tool-summary">${summaryText}</div>`;
            list.appendChild(div);
            this._updateToolContainerSummary(container);
            this.scrollToBottom();
            return div;
        }

        getToolBlockEl(id, opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target) || id == null) return null;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble) return null;
            const safeId = String(id).replace(/"/g, '\\"');
            return bubble.querySelector(`.chat-tool-call[data-tool-id="${safeId}"]`);
        }

        _updateToolContainerSummary(container) {
            if (!container) return;
            const calls = container.querySelectorAll('.chat-tool-call');
            const countEl = container.querySelector('.chat-tool-count');
            if (countEl) {
                const n = calls.length;
                countEl.textContent = `${n} Tool-Aufruf${n !== 1 ? 'e' : ''}`;
            }
        }

        // ── Custom-Slots (target-aware) ────────────────────────

        appendCustomBlock(html, opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target)) return null;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble) return null;

            const slot = o.slot === 'afterText' || o.slot === 'footer' ? o.slot : 'beforeText';
            const doSanitize = o.sanitize !== false;
            const safeHtml = doSanitize ? this._sanitize(html) : (html || '');

            const div = document.createElement('div');
            div.className = 'chat-custom-block';
            div.dataset.slot = slot;
            const id = o.id != null ? String(o.id) : `auto-${++this._blockIdSeq}`;
            div.dataset.customBlockId = id;
            div.innerHTML = safeHtml;

            const textEl = bubble.querySelector('.chat-text');
            const sourcesAnchor = bubble.querySelector('.chat-sources, .chat-usage');

            if (slot === 'beforeText') {
                if (textEl) bubble.insertBefore(div, textEl);
                else if (sourcesAnchor) bubble.insertBefore(div, sourcesAnchor);
                else bubble.appendChild(div);
            } else if (slot === 'afterText') {
                if (sourcesAnchor) bubble.insertBefore(div, sourcesAnchor);
                else bubble.appendChild(div);
            } else {
                bubble.appendChild(div);
            }
            return div;
        }

        getCustomBlockEl(id, opts) {
            const o = opts || {};
            const target = o.target || o.messageEl;
            if (!target || !document.body.contains(target) || id == null) return null;
            const bubble = target.querySelector('.chat-bubble');
            if (!bubble) return null;
            const safeId = String(id).replace(/"/g, '\\"');
            return bubble.querySelector(`.chat-custom-block[data-custom-block-id="${safeId}"]`);
        }

        appendCustomBubble(html, opts) {
            const o = opts || {};
            const messagesEl = this._getMessagesEl();
            if (!messagesEl) return null;
            this._removeEmptyState();

            const doSanitize = o.sanitize === true;
            const safeHtml = doSanitize ? this._sanitize(html) : (html || '');

            const wrapper = document.createElement('div');
            wrapper.innerHTML = safeHtml;
            const el = wrapper.firstElementChild;
            if (!el) return null;
            if (o.id != null) {
                el.dataset.customBubbleId = String(o.id);
                this._customBubbles.set(String(o.id), el);
            }
            messagesEl.appendChild(el);
            this.scrollToBottom();
            return el;
        }

        getCustomBubbleEl(id) {
            if (id == null) return null;
            const key = String(id);
            const el = this._customBubbles.get(key);
            if (el && document.body.contains(el)) return el;
            this._customBubbles.delete(key);
            return null;
        }

        // ── Streaming-Indicator ────────────────────────────────

        appendStreamingIndicator(opts) {
            const o = opts || {};
            const messagesEl = this._getMessagesEl();
            if (!messagesEl) return null;
            this._removeEmptyState();

            const role = o.role || 'assistant';
            const avatarHtml = this._avatarHtml(role);
            const label = o.label != null ? String(o.label) : 'Denke nach\u2026';

            const div = document.createElement('div');
            div.className = `chat-message ${role}`;
            if (o.id) div.id = String(o.id);
            div.innerHTML = `<div class="chat-avatar">${avatarHtml}</div>
                <div class="chat-bubble">
                    <div class="brain-thinking-wrap">${_thinkingIndicatorSvg(this._svgPrefix + (this._blockIdSeq++))}
                        <span class="brain-thinking-label">${SharedUi.runtime.escapeHtml(label)}</span>
                    </div>
                </div>`;
            messagesEl.appendChild(div);
            this.scrollToBottom(true);
            return div;
        }

        removeStreamingIndicator(el) {
            if (el && el.parentNode) el.remove();
        }

        // ── History / Clear / Destroy ──────────────────────────

        clear() {
            const el = this._getMessagesEl();
            if (el) el.innerHTML = '';
            this._customBubbles.clear();
        }

        loadHistory(messages) {
            this.clear();
            if (!Array.isArray(messages) || messages.length === 0) return;
            messages.forEach((msg) => this._renderHistoryMessage(msg));
            this.scrollToBottom(true);
        }

        _renderHistoryMessage(msg) {
            if (!msg) return;
            const role = msg.role === 'user' ? 'user' : 'assistant';
            let el;
            if (role === 'user') {
                el = this.appendUserMessage(msg.content || '', { createdAt: msg.created_at || null });
            } else {
                el = this.startAssistantMessage({ createdAt: msg.created_at || null, showCopyButton: true });
                if (el) {
                    el.dataset.chatTextAnchored = '1';
                    const bubble = el.querySelector('.chat-bubble');
                    const textEl = bubble ? bubble.querySelector('.chat-text') : null;
                    if (textEl) {
                        textEl.innerHTML = SharedUi.markdown.render(msg.content || '', { thinkBlocks: true });
                    }
                }
            }
            this._notifyRendered(el, { role, history: true, message: msg });
        }

        scrollToBottom(force) {
            const el = this._getMessagesEl();
            if (!el) return;
            if (!force) {
                const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
                if (dist > 80) return;
            }
            requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
        }

        showStopButton(visible) {
            const btn = (this._stopBtn && document.body.contains(this._stopBtn))
                ? this._stopBtn
                : this._resolveStopButton();
            if (btn) {
                if (visible) btn.classList.remove('u-hidden');
                else btn.classList.add('u-hidden');
            }
        }

        destroy() {
            this._listeners.forEach(({ el, event, fn }) => {
                try { el.removeEventListener(event, fn); } catch (e) { /* ignore */ }
            });
            this._listeners = [];
            this._customBubbles.clear();
            this._messagesEl = null;
            this._inputEl = null;
            this._sendBtn = null;
            this._stopBtn = null;
        }
    }

    SharedUi.chat = { ChatView };
})();
