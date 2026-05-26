/**
 * SharedUi.markdown — produktneutrales Markdown-Rendering mit zentralem
 * DOMPurify-Sanitizing, Think-Block-Support (inkl. Streaming-Zustand) und
 * delegierter Copy-Button-Handhabung.
 *
 * Abhaengigkeiten (Reihenfolge im Template):
 *   core.js -> runtime.js -> vendor/purify.min.js -> markdown.js
 *
 * Produktneutralitaet: Keine Referenzen auf produktspezifische Symbole
 * oder URLs. Bewusst generisch gehaltene Chat-Klassen im Tabellen-/Copy-
 * Pfad (".markdown-table" im Tabellen-Renderer, ".chat-bubble"/".chat-text"
 * im Message-Copy-Delegationszweig) entsprechen den Core-Chat-Primitives
 * aus ui_core/chat_primitives.js. "Reasoning"-Label im Think-Block bleibt
 * als produktneutrale Bezeichnung.
 *
 * API:
 *   SharedUi.markdown.render(text, { thinkBlocks?: boolean }) -> string
 *   SharedUi.markdown.renderInto(element, text, options?)    -> void
 */
(function() {
    'use strict';

    if (typeof window.SharedUi === 'undefined') {
        console.error('SharedUi.markdown: core.js muss vor markdown.js laden.');
        return;
    }

    const _escape = (s) => (window.SharedUi && SharedUi.runtime && SharedUi.runtime.escapeHtml)
        ? SharedUi.runtime.escapeHtml(s)
        : String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');

    // ── Sanitizer ────────────────────────────────────────────────────

    function _sanitize(html) {
        if (typeof DOMPurify === 'undefined') {
            // Fail-closed: ohne DOMPurify nur escapten Text rendern.
            // Einmalig sichtbar warnen, sonst bleibt der Defekt-Zustand
            // (Markdown als HTML-Quelltext) ohne Erklaerung fuer den User.
            if (!_sanitize._warned) {
                _sanitize._warned = true;
                console.error('SharedUi.markdown: DOMPurify fehlt - Markdown wird als escapter Text angezeigt (vendor/purify.min.js nicht geladen).');
                if (window.SharedUi && SharedUi.notify && typeof SharedUi.notify.show === 'function') {
                    try {
                        SharedUi.notify.show(
                            'Anzeigemodul fehlt: Inhalte werden ohne Formatierung dargestellt. Seite neu laden oder Administrator informieren.',
                            'error',
                            0
                        );
                    } catch (e) { /* Notify-Modul nicht bereit - Console-Error reicht */ }
                }
            }
            const tmp = document.createElement('div');
            tmp.textContent = html == null ? '' : String(html);
            return _escape(tmp.textContent);
        }
        return DOMPurify.sanitize(html, {
            USE_PROFILES: { html: true },
            FORBID_TAGS: ['style', 'script', 'iframe', 'object', 'embed'],
            FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur', 'onsubmit'],
            ADD_ATTR: ['data-copy-target', 'data-copy-text'],
        });
    }

    // ── Think-Block-Extraktion ───────────────────────────────────────

    function _extractThinkBlocks(text) {
        if (!text || !text.includes('<think>')) {
            return { thinkHtml: '', displayText: text || '' };
        }

        // Markdown-Code-Bloecke vor der Think-Extraktion ausblenden, damit
        // literale <think>-Tags in Code-Beispielen erhalten bleiben.
        const codeStash = [];
        const _stashCode = (m) => {
            const tok = '%%CBSTH' + codeStash.length + '%%';
            codeStash.push(m);
            return tok;
        };
        let working = text.replace(/```[\s\S]*?```/g, _stashCode).replace(/`[^`\n]+`/g, _stashCode);

        let thinkHtml = '';
        let displayText = working;

        const closedRe = /<think>([\s\S]*?)<\/think>/g;
        let match;
        while ((match = closedRe.exec(working)) !== null) {
            thinkHtml += '<details class="think-block"><summary><span class="think-toggle-label"></span>'
                + '<button type="button" class="think-copy-btn" title="Reasoning kopieren" aria-label="Reasoning kopieren">&#x2398; Kopieren</button></summary>'
                + '<div class="think-block-content">' + _escape(match[1].trim()) + '</div></details>';
        }
        displayText = displayText.replace(closedRe, '');

        // Offener <think>-Block ohne </think> — Streaming-Zustand
        const openIdx = displayText.indexOf('<think>');
        if (openIdx !== -1) {
            const before = displayText.substring(0, openIdx);
            const thinkContent = displayText.substring(openIdx + 7);
            thinkHtml += '<details class="think-block thinking" open>'
                + '<summary><span class="think-toggle-label"></span></summary>'
                + '<div class="think-block-content">' + _escape(thinkContent.trim()) + '</div>'
                + '</details>';
            displayText = before;
        }

        // Code-Bloecke restaurieren, damit _markdownToHtml die Originalsyntax
        // (```/`) sieht und korrekt tokenisiert.
        displayText = displayText.replace(/%%CBSTH(\d+)%%/g, (_, idx) => codeStash[parseInt(idx, 10)] || '');

        displayText = displayText.trim();
        return { thinkHtml, displayText };
    }

    // ── Mini-Markdown-Engine ─────────────────────────────────────────

    function _markdownToHtml(text) {
        if (!text) return '';
        let html = _escape(text);
        const codeTokens = [];

        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            const tok = `%%CBTK${codeTokens.length}%%`;
            codeTokens.push('<div class="code-block-wrap"><button type="button" class="code-copy-btn" title="Code kopieren" aria-label="Code kopieren">&#x2398;</button><pre><code>' + code + '</code></pre></div>');
            return tok;
        });
        html = html.replace(/`([^`]+)`/g, (_, code) => {
            const tok = `%%CBTK${codeTokens.length}%%`;
            codeTokens.push('<code>' + code + '</code>');
            return tok;
        });
        html = html.replace(/^(#{1,6})\s+(.+)$/gm, (m, hashes, content) => {
            const lvl = Math.min(Math.max(hashes.length, 1), 6);
            return `<h${lvl}>${content}</h${lvl}>`;
        });
        html = html.replace(/(?:^|\n)(\|.+\|)\n(\|[\s:|-]+\|)\n((?:\|.+\|\n?)+)/g, (m, header, sep, body) => {
            const thCells = header.split('|').filter(c => c.trim() !== '').map(c => `<th>${c.trim()}</th>`).join('');
            const alignments = sep.split('|').filter(c => c.trim() !== '').map(c => {
                const t = c.trim();
                if (t.startsWith(':') && t.endsWith(':')) return 'center';
                if (t.endsWith(':')) return 'right';
                return 'left';
            });
            const rows = body.trim().split('\n').map(row => {
                const cells = row.split('|').filter(c => c.trim() !== '');
                const tds = cells.map((c, i) => {
                    const a = alignments[i] || 'left';
                    const cls = a === 'center' ? 'u-text-center' : (a === 'right' ? 'u-text-right' : 'u-text-left');
                    return `<td class="${cls}">${c.trim()}</td>`;
                }).join('');
                return `<tr>${tds}</tr>`;
            }).join('');
            return `<table class="markdown-table"><thead><tr>${thCells}</tr></thead><tbody>${rows}</tbody></table>`;
        });
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/([\s\S]*?)(<pre[\s\S]*?<\/pre>|$)/g, (m, plain, pre) =>
            plain.replace(/\n/g, '<br>') + pre
        );
        // Restore via Callback — sonst werden $1/$$/$& in Code-Bloecken als
        // String.replace-Back-Refs interpretiert.
        html = html.replace(/%%CBTK(\d+)%%/g, (_, idx) => codeTokens[parseInt(idx, 10)] || '');
        return html;
    }

    // ── Public render ────────────────────────────────────────────────

    function render(text, options) {
        const { thinkBlocks = false } = options || {};
        let thinkHtml = '';
        let body = text || '';
        if (thinkBlocks) {
            const parts = _extractThinkBlocks(text);
            thinkHtml = parts.thinkHtml;
            body = parts.displayText;
        }
        const full = thinkHtml + _markdownToHtml(body);
        return _sanitize(full);
    }

    function renderInto(element, text, options) {
        if (!element) return;
        element.innerHTML = render(text, options);
    }

    // ── Copy-Helper ──────────────────────────────────────────────────

    function _clipCopy(text, btn, successLabel) {
        if (!btn.dataset.origLabel) {
            btn.dataset.origLabel = btn.innerHTML;
        }
        const orig = btn.dataset.origLabel;
        const ok = () => {
            btn.innerHTML = successLabel;
            btn.classList.add('copy-ok');
            clearTimeout(btn._copyTimer);
            btn._copyTimer = setTimeout(() => {
                btn.innerHTML = orig;
                btn.classList.remove('copy-ok');
            }, 1500);
        };
        const fallback = () => {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.className = 'u-offscreen-clipboard';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
            ok();
        };
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(ok).catch(fallback);
        } else {
            fallback();
        }
    }

    // ── Delegierter Click-Handler (idempotent) ───────────────────────
    // Erkennt:
    //   1. data-copy-target / data-copy-text (produktneutral)
    //   2. .code-copy-btn (Anker .code-block-wrap)
    //   3. .think-copy-btn (Anker .think-block-content)
    //   4. .msg-copy-btn  (Anker .chat-bubble .chat-text — Sprint 5)
    if (document.body && document.body.dataset.sharedUiCopy !== '1') {
        document.body.dataset.sharedUiCopy = '1';
        document.body.addEventListener('click', (e) => {
            const attrBtn = e.target.closest('[data-copy-target],[data-copy-text]');
            if (attrBtn) {
                let text = attrBtn.getAttribute('data-copy-text');
                if (text == null) {
                    const sel = attrBtn.getAttribute('data-copy-target');
                    const tgt = sel ? document.querySelector(sel) : null;
                    text = tgt ? tgt.innerText : '';
                }
                _clipCopy(text || '', attrBtn, '&#x2714; Kopiert');
                return;
            }
            const codeBtn = e.target.closest('.code-copy-btn');
            if (codeBtn) {
                const wrap = codeBtn.closest('.code-block-wrap') || codeBtn.parentElement;
                const codeEl = wrap && wrap.querySelector('code');
                if (codeEl) _clipCopy(codeEl.textContent, codeBtn, '&#x2714;');
                return;
            }
            const thinkBtn = e.target.closest('.think-copy-btn');
            if (thinkBtn) {
                const content = thinkBtn.closest('.think-block');
                const inner = content && content.querySelector('.think-block-content');
                if (inner) _clipCopy(inner.textContent, thinkBtn, '&#x2714; Kopiert');
                return;
            }
            const msgBtn = e.target.closest('.msg-copy-btn');
            if (msgBtn) {
                const bubble = msgBtn.closest('.chat-bubble');
                const textEl = bubble && bubble.querySelector('.chat-text');
                if (textEl) _clipCopy(textEl.innerText, msgBtn, '&#x2714; Kopiert');
                return;
            }
        });
    }

    // ── Namespace-Export ─────────────────────────────────────────────

    SharedUi.markdown = { render, renderInto };
})();
