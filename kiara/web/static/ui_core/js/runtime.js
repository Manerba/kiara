/**
 * Shared Frontend Core — Runtime Helpers
 *
 * SharedUi.runtime API: escapeHtml / escapeAttr / escapeJSAttr,
 * createAbort / abort / abortAll, withButtonLock,
 * polling (Singleton) + PollingManager (Klasse).
 * Produktneutral: kein Tab-Routing, kein URL-Scope-Wissen.
 * Benoetigt zuvor geladenes ui_core/js/core.js.
 */
(function () {
    if (!window.SharedUi || typeof window.SharedUi.configure !== 'function') {
        throw new Error('ui_core/core.js must load before runtime.js');
    }

    if (window.SharedUi.runtime) {
        return;
    }

    function escapeHtml(text) {
        if (text === null || text === undefined) return '';
        var div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML;
    }

    function escapeAttr(text) {
        if (text === null || text === undefined) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    function escapeJSAttr(text) {
        if (text === null || text === undefined) return '';
        var s = String(text)
            .replace(/\\/g, '\\\\')
            .replace(/'/g, "\\'")
            .replace(/\n/g, '\\n')
            .replace(/\r/g, '\\r');
        return escapeAttr(s);
    }

    var _aborts = new Map();

    function abort(id) {
        var ctrl = _aborts.get(id);
        if (ctrl) {
            try { ctrl.abort(); } catch (_) { /* AbortController ist robust */ }
            _aborts.delete(id);
        }
    }

    function createAbort(id) {
        abort(id);
        var ctrl = new AbortController();
        _aborts.set(id, ctrl);
        return ctrl;
    }

    // Guard-Helper fuer Consumer: Nach einem await pruefen, ob der eigene
    // Controller noch der aktuelle fuer diese ID ist. Wenn false, sollte der
    // Consumer das Resultat verwerfen (stale write gegen neueren Request).
    //   const ctrl = SharedUi.runtime.createAbort('my-save');
    //   const result = await fetch(...);
    //   if (!SharedUi.runtime.isCurrent('my-save', ctrl)) return;
    //   // ... DOM-Update mit result
    function isCurrent(id, ctrl) {
        if (!ctrl) return false;
        return _aborts.get(id) === ctrl;
    }

    function abortAll() {
        var ids = Array.from(_aborts.keys());
        for (var i = 0; i < ids.length; i++) {
            abort(ids[i]);
        }
    }

    async function withButtonLock(btn, fn) {
        if (!btn || btn.disabled) return;
        var prevHTML = btn.innerHTML;
        btn.disabled = true;
        try {
            return await fn();
        } finally {
            btn.disabled = false;
            btn.innerHTML = prevHTML;
        }
    }

    function PollingManager() {
        this._timers = new Map();
    }

    PollingManager.prototype.start = function (id, fn, intervalMs, opts) {
        this.stop(id);
        var isActive = (opts && typeof opts.isActive === 'function') ? opts.isActive : null;
        var running = false;
        var generation = Symbol(id);
        var self = this;
        var tick = async function () {
            var current = self._timers.get(id);
            if (!current || current.generation !== generation) return;
            if (isActive && !isActive(id)) {
                self.stop(id);
                return;
            }
            if (running) return;
            running = true;
            try {
                await fn({
                    isCurrent: function () {
                        var active = self._timers.get(id);
                        return !!active && active.generation === generation;
                    }
                });
            } catch (err) {
                console.error('Poll ' + id + ':', err);
            } finally {
                running = false;
            }
        };
        this._timers.set(id, {
            handle: setInterval(tick, intervalMs),
            generation: generation
        });
    };

    PollingManager.prototype.stop = function (id) {
        var h = this._timers.get(id);
        if (h) {
            clearInterval(h.handle);
            this._timers.delete(id);
        }
    };

    PollingManager.prototype.has = function (id) {
        return this._timers.has(id);
    };

    PollingManager.prototype.stopAll = function () {
        var ids = Array.from(this._timers.keys());
        for (var i = 0; i < ids.length; i++) {
            this.stop(ids[i]);
        }
    };

    var polling = new PollingManager();

    window.SharedUi.runtime = {
        escapeHtml: escapeHtml,
        escapeAttr: escapeAttr,
        escapeJSAttr: escapeJSAttr,
        createAbort: createAbort,
        abort: abort,
        abortAll: abortAll,
        isCurrent: isCurrent,
        withButtonLock: withButtonLock,
        polling: polling,
        PollingManager: PollingManager
    };
})();
