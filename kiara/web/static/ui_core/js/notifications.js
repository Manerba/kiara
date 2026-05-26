/**
 * Shared Frontend Core — Notifications
 *
 * SharedUi.notify API: show / dismiss / dismissAll.
 * Single-Slot-Verhalten: neue Notification verdraengt die aktuelle.
 * Produktneutral: keine API-URLs, keine Produktbegriffe.
 * Benoetigt zuvor geladenes ui_core/js/core.js und runtime.js.
 */
(function () {
    if (!window.SharedUi || typeof window.SharedUi.configure !== 'function') {
        throw new Error('ui_core/core.js must load before notifications.js');
    }
    if (!window.SharedUi.runtime || typeof window.SharedUi.runtime.escapeHtml !== 'function') {
        throw new Error('ui_core/runtime.js must load before notifications.js');
    }

    if (window.SharedUi.notify) {
        return;
    }

    var VALID = new Set(['success', 'error', 'warning', 'info', 'loading']);
    var DEFAULT_DURATION = {
        success: 3000,
        error: 6000,
        warning: 5000,
        info: 3000,
        loading: 0
    };

    var _seq = 0;

    function show(message, type, durationMs) {
        var resolvedType = VALID.has(type) ? type : 'info';
        var duration = (typeof durationMs === 'number' && durationMs >= 0)
            ? durationMs
            : DEFAULT_DURATION[resolvedType];

        var existing = document.querySelector('.notification');
        if (existing) {
            existing.remove();
        }

        _seq += 1;
        var id = 'notify-' + _seq;

        var node = document.createElement('div');
        node.className = 'notification notification-' + resolvedType;
        node.id = id;
        node.setAttribute('data-notify-id', id);
        node.setAttribute('role', 'status');
        node.setAttribute('aria-live', 'polite');
        node.textContent = (message === null || message === undefined) ? '' : String(message);

        document.body.appendChild(node);

        if (duration > 0) {
            setTimeout(function () {
                var current = document.getElementById(id);
                if (!current) return;
                current.classList.add('fade-out');
                setTimeout(function () {
                    var stillThere = document.getElementById(id);
                    if (stillThere) stillThere.remove();
                }, 300);
            }, duration);
        }

        return id;
    }

    function dismiss(id) {
        if (!id) return;
        var node = document.getElementById(id);
        if (node && node.getAttribute('data-notify-id') === id) {
            node.remove();
        }
    }

    function dismissAll() {
        var nodes = document.querySelectorAll('.notification');
        for (var i = 0; i < nodes.length; i++) {
            nodes[i].remove();
        }
    }

    window.SharedUi.notify = {
        show: show,
        dismiss: dismiss,
        dismissAll: dismissAll
    };
})();
