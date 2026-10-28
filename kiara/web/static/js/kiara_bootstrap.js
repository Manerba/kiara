/**
 * KIara-Bootstrap — Produkt-Konfiguration fuer SharedUi
 *
 * Konfiguriert den KIara-Storage-Key `kiara-theme` (Bindestrich!),
 * laedt das Theme und synchronisiert `.theme-option`-Buttons.
 * Muss nach ui_core/core.js + ui_core/theme.js laufen.
 */
(function () {
    if (!window.SharedUi || !window.SharedUi.theme) {
        return;
    }

    window.SharedUi.configure({
        productName: 'KIara',
        themeStorageKey: 'kiara-theme',
        allowedThemes: ['light', 'dark', 'kontrast'],
        loginUrl: '/login',
        logoutUrl: '/logout',
        authMeUrl: '/api/auth/me',
        csrfHeader: 'X-Requested-With',
        csrfValue: 'XMLHttpRequest',
        csrfTokenHeader: 'X-CSRF-Token',
        csrfTokenValue: null,
        csrfRefreshUrl: '/api/auth/csrf',
        defaultCredentials: 'same-origin',
        on401: function (response, url, options) {
            window.location.href = '/login';
        },
        modalLabels: { ok: 'OK', cancel: 'Abbrechen', yes: 'Ja' }
    });

    window.SharedUi.theme.load();

    function _markButton(theme) {
        if (typeof theme !== 'string' || theme.length === 0) {
            return;
        }
        var buttons = document.querySelectorAll('.theme-option');
        for (var i = 0; i < buttons.length; i++) {
            var btn = buttons[i];
            btn.classList.toggle('active', btn.dataset.theme === theme);
        }
    }

    document.addEventListener('themeChanged', function (e) {
        if (!e || !e.detail || typeof e.detail.theme !== 'string') {
            return;
        }
        if (!window.SharedUi.theme.isAllowed(e.detail.theme)) {
            return;
        }
        _markButton(e.detail.theme);
    });

    function _initialSync() {
        _markButton(document.documentElement.getAttribute('data-theme'));
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', _initialSync);
    } else {
        _initialSync();
    }
})();
