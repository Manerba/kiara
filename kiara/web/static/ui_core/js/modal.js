/**
 * Shared Frontend Core — Modal
 *
     * SharedUi.modal API: open / close / closeAll / confirm / alert.
     * Superset der heutigen createModal-Felder; msgbox-Wrapper ueber confirm/alert.
     * Produktneutral: keine API-URLs, keine Produktbegriffe.
     * Benoetigt zuvor geladenes ui_core/js/core.js, runtime.js und DOMPurify.
     */
(function () {
    if (!window.SharedUi || typeof window.SharedUi.configure !== 'function') {
        throw new Error('ui_core/core.js must load before modal.js');
    }
    if (!window.SharedUi.runtime || typeof window.SharedUi.runtime.escapeHtml !== 'function') {
        throw new Error('ui_core/runtime.js must load before modal.js');
    }
    if (!window.DOMPurify || typeof window.DOMPurify.sanitize !== 'function') {
        throw new Error('vendor/purify.min.js must load before modal.js');
    }

    if (window.SharedUi.modal) {
        return;
    }

    var SIZE_MAP = { small: '400px', medium: '700px', large: '900px' };
    var FOCUSABLE_SELECTOR = [
        'input:not([type="hidden"]):not([disabled])',
        'textarea:not([disabled])',
        'select:not([disabled])',
        'button:not([disabled])',
        'a[href]',
        'summary',
        '[tabindex]:not([tabindex="-1"])',
        '[contenteditable]:not([contenteditable="false"])'
    ].join(', ');

    function _collectFocusable(overlay) {
        var nodes = overlay.querySelectorAll(FOCUSABLE_SELECTOR);
        var out = [];
        for (var i = 0; i < nodes.length; i++) {
            var el = nodes[i];
            if (el.hasAttribute('disabled')) continue;
            if (el.getAttribute('aria-hidden') === 'true') continue;
            if (el.offsetParent === null && el !== overlay.querySelector('.modal-content')) continue;
            out.push(el);
        }
        return out;
    }

    var _stack = [];
    var _seq = 0;

    function _labels() {
        var cfg = window.SharedUi._config() || {};
        var ml = cfg.modalLabels || {};
        return {
            ok: ml.ok || 'OK',
            cancel: ml.cancel || 'Abbrechen',
            yes: ml.yes || 'Ja'
        };
    }

    function _topEntry() {
        return _stack.length > 0 ? _stack[_stack.length - 1] : null;
    }

    function _findEntry(id) {
        for (var i = 0; i < _stack.length; i++) {
            if (_stack[i].id === id) return _stack[i];
        }
        return null;
    }

    function _removeEntry(id) {
        for (var i = 0; i < _stack.length; i++) {
            if (_stack[i].id === id) {
                _stack.splice(i, 1);
                return;
            }
        }
    }

    function _closeManagedAll() {
        var ids = _stack.map(function (e) { return e.id; });
        for (var i = ids.length - 1; i >= 0; i--) {
            close(ids[i]);
        }
    }

    function _sanitizeHtml(html) {
        var raw = (html === null || html === undefined) ? '' : String(html);
        return window.DOMPurify.sanitize(raw, {
            USE_PROFILES: { html: true },
            FORBID_TAGS: ['style', 'script', 'iframe', 'object', 'embed'],
            FORBID_ATTR: [
                'style',
                'onerror',
                'onload',
                'onclick',
                'onmouseover',
                'onfocus',
                'onblur',
                'onsubmit',
            ],
        });
    }

    function _setSanitizedHtml(el, html) {
        el.innerHTML = _sanitizeHtml(html);
    }

    function _applyBody(bodyEl, options) {
        if (options.body !== undefined) {
            if (options.body instanceof Node) {
                bodyEl.appendChild(options.body);
            } else {
                _setSanitizedHtml(bodyEl, options.body);
            }
            return;
        }
        if (options.html !== undefined) {
            _setSanitizedHtml(bodyEl, options.html);
            return;
        }
        if (options.text !== undefined) {
            bodyEl.textContent = (options.text === null) ? '' : String(options.text);
            return;
        }
        if (options.element instanceof Node) {
            bodyEl.appendChild(options.element);
        }
    }

    function _renderButtons(footerEl, buttons, id) {
        var withLock = window.SharedUi.runtime.withButtonLock;
        for (var i = 0; i < buttons.length; i++) {
            var spec = buttons[i] || {};
            var btn = document.createElement('button');
            var cls = 'action-btn';
            if (spec.className) cls += ' ' + spec.className;
            btn.className = cls;
            btn.textContent = (spec.label === undefined || spec.label === null) ? '' : String(spec.label);
            (function (clickHandler) {
                btn.addEventListener('click', function () {
                    withLock(btn, function () {
                        if (typeof clickHandler === 'function') {
                            return clickHandler(id);
                        }
                    });
                });
            })(spec.onClick);
            footerEl.appendChild(btn);
        }
    }

    function _renderDefaultFooter(footerEl) {
        var labels = _labels();
        var btn = document.createElement('button');
        btn.className = 'action-btn';
        btn.textContent = labels.cancel;
        btn.setAttribute('data-sharedui-close', '');
        footerEl.appendChild(btn);
    }

    function _autoFocus(overlay) {
        if (overlay.contains(document.activeElement)) {
            return;
        }
        var content = overlay.querySelector('.modal-content');
        var candidate = overlay.querySelector(FOCUSABLE_SELECTOR);
        if (candidate) {
            try { candidate.focus(); } catch (_) {}
            return;
        }
        if (content) {
            try { content.focus(); } catch (_) {}
        }
    }

    function open(options) {
        options = options || {};

        if (options.id && _findEntry(options.id)) {
            throw new Error('SharedUi.modal: duplicate id ' + options.id);
        }

        if (options.stack !== true) {
            _closeManagedAll();
        }

        _seq += 1;
        var id = options.id || ('modal-' + _seq);

        var escape = window.SharedUi.runtime.escapeHtml;

        var overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.setAttribute('data-sharedui-modal-id', id);

        var content = document.createElement('div');
        var contentClass = 'modal-content' + (options.cssClass ? ' ' + options.cssClass : '');
        content.className = contentClass;
        content.tabIndex = -1;

        var maxWidth = options.maxWidth || (options.size && SIZE_MAP[options.size]) || '';
        if (maxWidth) {
            // CSP-strict (V30 Welle 3): nonced runtimeStyle statt style-Attribut.
            window.SharedUi.runtimeStyle.applyStyle(content, { maxWidth: maxWidth });
        }

        var hasTitle = options.title !== undefined && options.title !== null && String(options.title).length > 0;
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-modal', 'true');
        if (hasTitle) {
            var header = document.createElement('div');
            header.className = 'modal-header';
            header.textContent = String(options.title);
            content.appendChild(header);
            overlay.setAttribute('aria-label', String(options.title));
        } else {
            var fallbackLabel = options.ariaLabel || 'Dialog';
            overlay.setAttribute('aria-label', String(fallbackLabel));
        }

        var body = document.createElement('div');
        body.className = 'modal-body';
        _applyBody(body, options);
        content.appendChild(body);

        var hasFooter = false;
        var footer = null;
        if (Array.isArray(options.buttons) && options.buttons.length > 0) {
            footer = document.createElement('div');
            footer.className = 'modal-footer';
            _renderButtons(footer, options.buttons, id);
            content.appendChild(footer);
            hasFooter = true;
        } else if (options.footer === '') {
            // explizit kein Footer
        } else if (typeof options.footer === 'string') {
            footer = document.createElement('div');
            footer.className = 'modal-footer';
            _setSanitizedHtml(footer, options.footer);
            content.appendChild(footer);
            hasFooter = true;
        } else {
            footer = document.createElement('div');
            footer.className = 'modal-footer';
            _renderDefaultFooter(footer);
            content.appendChild(footer);
            hasFooter = true;
        }

        overlay.appendChild(content);

        function defaultCloseDelegation(ev) {
            var target = ev.target;
            if (target && target.hasAttribute && target.hasAttribute('data-sharedui-close')) {
                close(id);
            }
        }
        overlay.addEventListener('click', defaultCloseDelegation);

        function backdropHandler(ev) {
            if (ev.target === overlay) {
                close(id);
            }
        }
        if (options.backdropClose === true) {
            overlay.addEventListener('click', backdropHandler);
        }

        function keyHandler(ev) {
            var top = _topEntry();
            if (!top || top.id !== id) return;
            if (ev.key === 'Tab') {
                var focusable = _collectFocusable(overlay);
                if (focusable.length === 0) {
                    ev.preventDefault();
                    var contentEl = overlay.querySelector('.modal-content');
                    if (contentEl) { try { contentEl.focus(); } catch (_) {} }
                    return;
                }
                var first = focusable[0];
                var last = focusable[focusable.length - 1];
                var active = document.activeElement;
                if (ev.shiftKey) {
                    if (active === first || !overlay.contains(active)) {
                        ev.preventDefault();
                        try { last.focus(); } catch (_) {}
                    }
                } else {
                    if (active === last || !overlay.contains(active)) {
                        ev.preventDefault();
                        try { first.focus(); } catch (_) {}
                    }
                }
                return;
            }
            if (ev.key !== 'Escape') return;
            if (options.escapeClose === false) return;
            ev.stopPropagation();
            close(id);
        }
        document.addEventListener('keydown', keyHandler);

        var returnFocus = (document.activeElement && document.activeElement !== document.body)
            ? document.activeElement
            : null;

        var entry = {
            id: id,
            overlay: overlay,
            keyHandler: keyHandler,
            returnFocus: returnFocus,
            onClose: typeof options.onClose === 'function' ? options.onClose : null
        };
        _stack.push(entry);

        document.body.appendChild(overlay);

        try {
            if (typeof options.onOpen === 'function') {
                options.onOpen(overlay);
            }
        } catch (err) {
            document.removeEventListener('keydown', keyHandler);
            _removeEntry(id);
            try { overlay.remove(); } catch (_) {}
            throw err;
        }

        if (options.autoFocus !== false) {
            _autoFocus(overlay);
        }

        return id;
    }

    function close(id) {
        var entry;
        if (id === undefined || id === null) {
            entry = _topEntry();
        } else {
            entry = _findEntry(id);
        }
        if (!entry) return false;

        document.removeEventListener('keydown', entry.keyHandler);
        _removeEntry(entry.id);

        try {
            if (entry.overlay && entry.overlay.parentNode) {
                entry.overlay.parentNode.removeChild(entry.overlay);
            }
        } catch (_) {}

        if (typeof entry.onClose === 'function') {
            try { entry.onClose(); } catch (e) { console.error('modal.onClose:', e); }
        }

        if (entry.returnFocus && document.body.contains(entry.returnFocus)) {
            var top = _topEntry();
            if (!top) {
                try { entry.returnFocus.focus(); } catch (_) {}
            }
        }

        return true;
    }

    function closeAll() {
        _closeManagedAll();
    }

    function confirm(opts) {
        opts = opts || {};
        var labels = _labels();
        var okLabel = opts.okLabel || labels.yes;
        var cancelLabel = opts.cancelLabel || labels.cancel;
        var msgText = (opts.message === null || opts.message === undefined) ? '' : String(opts.message);

        return new Promise(function (resolve) {
            var settled = false;
            var dialogOverlay = null;
            function settle(result) {
                if (settled) return;
                settled = true;
                close(id);
                resolve(result);
            }

            var bodyP = document.createElement('p');
            bodyP.className = 'u-m-0 u-fs-14';
            bodyP.textContent = msgText;
            var bodyWrap = document.createElement('div');
            bodyWrap.appendChild(bodyP);

            var dialogKeyHandler = function (ev) {
                if (ev.key === 'Enter') { ev.preventDefault(); settle(true); }
                else if (ev.key === 'Escape') { ev.preventDefault(); settle(false); }
            };

            var id = open({
                title: opts.title,
                ariaLabel: opts.ariaLabel || 'Bestaetigung',
                element: bodyWrap,
                maxWidth: '400px',
                stack: true,
                escapeClose: false,
                buttons: [
                    { label: cancelLabel, onClick: function () { settle(false); } },
                    { label: okLabel, className: 'primary', onClick: function () { settle(true); } }
                ],
                onOpen: function (overlay) {
                    dialogOverlay = overlay;
                    overlay.addEventListener('keydown', dialogKeyHandler);
                    var primary = overlay.querySelector('.action-btn.primary');
                    if (primary) { try { primary.focus(); } catch (_) {} }
                },
                onClose: function () {
                    if (dialogOverlay) {
                        dialogOverlay.removeEventListener('keydown', dialogKeyHandler);
                        dialogOverlay = null;
                    }
                    if (!settled) {
                        settled = true;
                        resolve(false);
                    }
                }
            });
        });
    }

    function alert(opts) {
        opts = opts || {};
        var labels = _labels();
        var okLabel = opts.okLabel || labels.ok;
        var msgText = (opts.message === null || opts.message === undefined) ? '' : String(opts.message);

        return new Promise(function (resolve) {
            var settled = false;
            var dialogOverlay = null;
            function settle(result) {
                if (settled) return;
                settled = true;
                close(id);
                resolve(result);
            }

            var bodyP = document.createElement('p');
            bodyP.className = 'u-m-0 u-fs-14';
            bodyP.textContent = msgText;
            var bodyWrap = document.createElement('div');
            bodyWrap.appendChild(bodyP);

            var dialogKeyHandler = function (ev) {
                if (ev.key === 'Enter') { ev.preventDefault(); settle(true); }
                else if (ev.key === 'Escape') { ev.preventDefault(); settle(false); }
            };

            var id = open({
                title: opts.title,
                ariaLabel: opts.ariaLabel || 'Hinweis',
                element: bodyWrap,
                maxWidth: '400px',
                stack: true,
                escapeClose: false,
                buttons: [
                    { label: okLabel, className: 'primary', onClick: function () { settle(true); } }
                ],
                onOpen: function (overlay) {
                    dialogOverlay = overlay;
                    overlay.addEventListener('keydown', dialogKeyHandler);
                    var primary = overlay.querySelector('.action-btn.primary');
                    if (primary) { try { primary.focus(); } catch (_) {} }
                },
                onClose: function () {
                    if (dialogOverlay) {
                        dialogOverlay.removeEventListener('keydown', dialogKeyHandler);
                        dialogOverlay = null;
                    }
                    if (!settled) {
                        settled = true;
                        resolve(false);
                    }
                }
            });
        });
    }

    window.SharedUi.modal = {
        open: open,
        close: close,
        closeAll: closeAll,
        confirm: confirm,
        alert: alert
    };
})();
