/**
 * KIara - Core Module
 * Theme, Settings, Tabs, URL-Routing, Notifications
 *
 * Fetch/Auth/CSRF und Runtime-Helper (escape*, withButtonLock, polling) leben
 * jetzt in ui_core/js/fetch.js und ui_core/js/runtime.js. KIara-spezifische
 * Konfiguration (Login-Redirect, CSRF-Headername) erfolgt in kiara_bootstrap.js.
 */

// Globale Variablen
let currentTab = null;
let currentUser = null;
let settingsDropdownOpen = false;
const tabDOMCache = {};

// --- Sprint 1.1: Zentrale Runtime-Infrastruktur ---

// Tab-Cleanup-Registry
const _tabCleanups = {};

function registerTabCleanup(key, fn) {
    if (!_tabCleanups[key]) _tabCleanups[key] = [];
    _tabCleanups[key].push(fn);
}

function _runTabCleanups(mainTabKey) {
    const fns = _tabCleanups[mainTabKey];
    if (fns) {
        fns.forEach(fn => { try { fn(); } catch(e) { console.error('Tab-Cleanup Fehler:', e); } });
        delete _tabCleanups[mainTabKey];
    }
}

function _runSubTabCleanups(mainTabKey) {
    const prefix = mainTabKey + '-';
    for (const key of Object.keys(_tabCleanups)) {
        if (key.startsWith(prefix)) {
            _tabCleanups[key].forEach(fn => { try { fn(); } catch(e) { console.error('SubTab-Cleanup Fehler:', e); } });
            delete _tabCleanups[key];
        }
    }
}

// KIara-spezifische Polling-Schicht ueber SharedUi.runtime.polling
const _MAIN_TAB_KEYS = new Set(['kiara', 'datenquellen', 'jobs']);

function kiaraPollingIsActive(id) {
    const tabForId = id.split(':')[0];
    if (_MAIN_TAB_KEYS.has(tabForId)) {
        return typeof adminCurrentTab === 'undefined' || adminCurrentTab === tabForId;
    }
    return typeof adminCurrentSubTab === 'undefined' || adminCurrentSubTab === tabForId;
}

const kiaraPolling = {
    start(id, fn, intervalMs) {
        SharedUi.runtime.polling.start(id, fn, intervalMs, { isActive: kiaraPollingIsActive });
    },
    stop(id) { SharedUi.runtime.polling.stop(id); },
    has(id) { return SharedUi.runtime.polling.has(id); },
    stopAll() { SharedUi.runtime.polling.stopAll(); }
};

// Initialisierung
document.addEventListener('DOMContentLoaded', function() {
    initDensity();
    initEventListeners();

    if (!document.body.dataset.adminMode) {
        loadCurrentUser();
        initTabs();
    }
});

function initEventListeners() {
    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleSettingsDropdown();
        });
    }

    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            SharedUi.theme.set(btn.dataset.theme);
        });
    });

    const densityToggle = document.getElementById('densityToggle');
    if (densityToggle) densityToggle.addEventListener('click', toggleDensity);

    const restartBtn = document.getElementById('restartBtn');
    if (restartBtn) restartBtn.addEventListener('click', restartDashboard);

    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) logoutBtn.addEventListener('click', logout);

    const changePasswordBtn = document.getElementById('changePasswordBtn');
    if (changePasswordBtn) changePasswordBtn.addEventListener('click', openPasswordChangeModal);

    // Admin-Guard: User-App-Router nur im User-Modus binden
    if (!document.body.dataset.adminMode) {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                navigateTo(this.dataset.tab);
            });
        });

        window.addEventListener('popstate', function(e) {
            const route = parseRoute();
            switchTab(route.tab);
        });
    }

    document.addEventListener('click', function(e) {
        const settingsWrapper = document.querySelector('.settings-dropdown-wrapper');
        if (settingsWrapper && settingsDropdownOpen && !settingsWrapper.contains(e.target)) {
            toggleSettingsDropdown(false);
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            if (settingsDropdownOpen) toggleSettingsDropdown(false);
        }
    });
}

// User / Auth
async function loadCurrentUser() {
    try {
        const resp = await fetch('/api/auth/me');  // uses raw fetch because: loadCurrentUser does its own 401 inline - safeFetch's on401 would loop
        if (!resp.ok) { window.location.href = '/login'; return; }
        const data = await resp.json();
        currentUser = data.user;

        const headerUsername = document.getElementById('headerUsername');
        if (headerUsername && currentUser) {
            const displayName = currentUser.vorname
                ? `${currentUser.vorname} ${currentUser.nachname}`.trim()
                : currentUser.username;
            headerUsername.textContent = displayName;
        }
        // LDAP-Benutzer koennen ihr Passwort nur im LDAP-System aendern —
        // den "Passwort aendern"-Button im Settings-Dropdown ausblenden.
        if (currentUser && currentUser.auth_source === 'ldap') {
            const changePwdBtn = document.getElementById('changePasswordBtn');
            if (changePwdBtn) changePwdBtn.classList.add('u-hidden');
        }

        // CSRF-Token hydratisieren, falls kein Login-Response vorlag (Page-Reload).
        const cfg = window.SharedUi && window.SharedUi._config && window.SharedUi._config();
        if (cfg && !cfg.csrfTokenValue && window.SharedUi.fetch && window.SharedUi.fetch.refreshCsrfToken) {
            try { await window.SharedUi.fetch.refreshCsrfToken(); } catch (_) { /* tolerant */ }
        }
    } catch (e) { /* Netzwerkfehler */ }
}

async function logout() {
    try { await SharedUi.fetch.safeFetch('/api/auth/logout', { method: 'POST' }); } catch (e) {}
    const cookieName = document.body.dataset.cookie || 'kiara_session';
    document.cookie = cookieName + '=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
    window.location.href = '/login';
}

function openPasswordChangeModal() {
    toggleSettingsDropdown(false);
    SharedUi.modal.open({
        title: 'Passwort aendern',
        body: `
            <div class="modal-field">
                <label>Altes Passwort</label>
                <input type="password" id="pwdOld" autocomplete="current-password">
            </div>
            <div class="modal-field">
                <label>Neues Passwort</label>
                <input type="password" id="pwdNew" autocomplete="new-password">
            </div>
            <div class="modal-field">
                <label>Neues Passwort bestaetigen</label>
                <input type="password" id="pwdConfirm" autocomplete="new-password">
            </div>`,
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>` +
                `<button class="action-btn primary" id="pwdSaveBtn">Aendern</button>`,
        onOpen: () => {
            document.getElementById('pwdOld').focus();
            document.getElementById('pwdSaveBtn').addEventListener('click', function() {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const oldPwd = document.getElementById('pwdOld').value;
                    const newPwd = document.getElementById('pwdNew').value;
                    const confirmPwd = document.getElementById('pwdConfirm').value;

                    if (!oldPwd || !newPwd) { SharedUi.notify.show('Bitte alle Felder ausfuellen', 'error'); return; }
                    if (newPwd !== confirmPwd) { SharedUi.notify.show('Passwoerter stimmen nicht ueberein', 'error'); return; }
                    if (newPwd.length < 4) { SharedUi.notify.show('Passwort muss mindestens 4 Zeichen haben', 'error'); return; }

                    try {
                        const pwdResp = await SharedUi.fetch.safeFetch('/api/auth/password', {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ old_password: oldPwd, new_password: newPwd }),
                        });
                        // Rotierten CSRF-Token direkt uebernehmen — sonst laeuft
                        // der naechste mutierende Call durch den 403-Refresh-Umweg.
                        try {
                            const body = await pwdResp.json();
                            if (body && typeof body.csrf_token === 'string') {
                                const cfg = window.SharedUi._config();
                                cfg.csrfTokenValue = body.csrf_token;
                            }
                        } catch (_) { /* leerer/nicht-JSON-Body ist ok */ }
                        SharedUi.notify.show('Passwort geaendert', 'success');
                        SharedUi.modal.close();
                    } catch (error) {
                        const msg = (error && error.body && error.body.detail) || (error && error.message) || 'Fehler';
                        SharedUi.notify.show(msg, 'error');
                    }
                });
            });
        },
    });
}

// Density
function initDensity() {
    const raw = localStorage.getItem('kiara-density');
    const savedDensity = raw === 'compact' ? 'compact' : 'normal';
    document.documentElement.setAttribute('data-density', savedDensity);
    updateDensityButton(savedDensity);
}

function toggleDensity() {
    const current = document.documentElement.getAttribute('data-density');
    const newDensity = current === 'compact' ? 'normal' : 'compact';
    document.documentElement.setAttribute('data-density', newDensity);
    localStorage.setItem('kiara-density', newDensity);
    updateDensityButton(newDensity);
}

function updateDensityButton(density) {
    const densityStatus = document.getElementById('densityStatus');
    if (densityStatus) densityStatus.textContent = density === 'compact' ? 'An' : 'Aus';
}

// Settings Dropdown
function toggleSettingsDropdown(forceState = null) {
    const dropdown = document.getElementById('settingsDropdown');
    const btn = document.getElementById('settingsBtn');
    settingsDropdownOpen = forceState !== null ? forceState : !settingsDropdownOpen;
    if (dropdown) dropdown.classList.toggle('is-hidden', !settingsDropdownOpen);
    if (btn) btn.classList.toggle('u-rotate-90', settingsDropdownOpen);
}

async function restartDashboard() {
    if (!confirm('Services wirklich neu starten?')) return;
    try { await SharedUi.fetch.safeFetch('/api/dashboard/restart', { method: 'POST' }); } catch (e) {}
    try { if (window.kiaraPolling && kiaraPolling.stopAll) kiaraPolling.stopAll(); } catch (e) {}
    try { if (SharedUi.runtime && SharedUi.runtime.abortAll) SharedUi.runtime.abortAll(); } catch (e) {}
    setTimeout(() => location.reload(), 3000);
}

// URL-Routing
function parseRoute() {
    const path = window.location.pathname.replace(/\/+$/, '') || '/';
    const segments = path.split('/').filter(Boolean);
    if (segments.length === 0) return { tab: 'chat' };
    const tabMap = { 'chat': 'chat', 'search': 'search', 'history': 'history' };
    return { tab: tabMap[segments[0]] || 'chat' };
}

function buildPath(tab) {
    const pathMap = { 'chat': '/chat', 'search': '/search', 'history': '/history' };
    return pathMap[tab] || '/chat';
}

function navigateTo(tab, replaceState) {
    const path = buildPath(tab);
    const state = { tab };
    if (replaceState) history.replaceState(state, '', path);
    else history.pushState(state, '', path);
    switchTab(tab);
}

// Tab-Wechsel
function initTabs() {
    const route = parseRoute();
    switchTab(route.tab);
    const path = buildPath(route.tab);
    history.replaceState({ tab: route.tab }, '', path);
}

function switchTab(tab) {
    const previousTab = currentTab;
    const isSameTab = (previousTab === tab);

    if (isSameTab) invalidateTabCache(tab);
    else if (previousTab) saveTabDOM(previousTab);

    currentTab = tab;

    document.querySelectorAll('.main-tabs .tab').forEach(t => {
        t.classList.remove('active');
        if (t.dataset.tab === tab) t.classList.add('active');
    });

    const filterBar = document.getElementById('filterBar');

    if (!isSameTab && restoreTabDOM(tab)) {
        if (tab === 'chat' && typeof handleChatTabRestore === 'function') {
            handleChatTabRestore();
        }
        return;
    }

    if (tab === 'chat') {
        if (typeof initChatTab !== 'undefined') initChatTab();
        if (filterBar) filterBar.classList.remove('is-hidden');
    } else if (tab === 'search') {
        if (typeof initSearchTab !== 'undefined') initSearchTab();
        if (filterBar) filterBar.classList.add('is-hidden');
    } else if (tab === 'history') {
        if (typeof initHistoryTab !== 'undefined') initHistoryTab();
        if (filterBar) filterBar.classList.remove('is-hidden');
    }
}

// DOM-Caching
function saveTabDOM(tabName) {
    const contentContainer = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!contentContainer || contentContainer.children.length === 0) return;

    const contentFragment = document.createDocumentFragment();
    while (contentContainer.firstChild) contentFragment.appendChild(contentContainer.firstChild);

    const filterFragment = document.createDocumentFragment();
    // Status der filter-bar via Klasse cachen (CSP-strict: kein .style.*-Write).
    let filterBarHidden = true;
    if (filterBar) {
        filterBarHidden = filterBar.classList.contains('is-hidden');
        while (filterBar.firstChild) filterFragment.appendChild(filterBar.firstChild);
    }

    tabDOMCache[tabName] = {
        content: contentFragment, filter: filterFragment,
        filterBarHidden: filterBarHidden, scrollY: window.scrollY,
    };
}

function restoreTabDOM(tabName) {
    const cached = tabDOMCache[tabName];
    if (!cached) return false;

    // Clone im Cache behalten (Backup bis saveTabDOM() ueberschreibt)
    tabDOMCache[tabName] = {
        content: cached.content.cloneNode(true),
        filter: cached.filter.cloneNode(true),
        filterBarHidden: cached.filterBarHidden,
        scrollY: cached.scrollY,
    };

    const contentContainer = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');

    // Original restaurieren — Event-Listener bleiben erhalten!
    if (contentContainer) { contentContainer.innerHTML = ''; contentContainer.appendChild(cached.content); }
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.appendChild(cached.filter);
        filterBar.classList.toggle('is-hidden', cached.filterBarHidden);
    }

    const savedScrollY = cached.scrollY;
    requestAnimationFrame(() => window.scrollTo(0, savedScrollY));
    return true;
}

function invalidateTabCache(tabName) { delete tabDOMCache[tabName]; }

// Helpers
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '\u2013';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Tab-Init Helper
async function initTabGeneric(config) {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.toggle('is-hidden', !config.showFilterBar);
    if (config.loadFn) await config.loadFn();
    if (config.renderFn) config.renderFn();
}

// Config CRUD Helpers
async function loadConfigFromAPI(endpoint) {
    try {
        const data = await SharedUi.fetch.safeFetchJson(endpoint);
        return data.config !== undefined ? data.config : data;
    } catch (e) { return null; }
}

async function saveConfigToAPI(config) {
    try {
        await SharedUi.fetch.safeFetch(config.endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config.data),
        });
        SharedUi.notify.show(config.successMessage, 'success');
        if (config.onSuccess) await config.onSuccess();
        return true;
    } catch (error) {
        const msg = (error && error.body && (error.body.detail || error.body.error || error.body.message)) || (error && error.message) || 'Speichern fehlgeschlagen';
        SharedUi.notify.show(msg, 'error');
        return false;
    }
}

/**
 * Sammelt data-key Felder aus contentContainer, validiert sie,
 * merged sie in eine geklonte Config und speichert via saveConfigToAPI.
 *
 * @param {object} currentConfig - Bestehende Config (wird geklont, nicht mutiert)
 * @param {object} options
 * @param {string} [options.endpoint='/api/admin/settings'] - API-Endpoint
 * @param {string} [options.successMessage='Gespeichert'] - Erfolgsmeldung
 * @param {string} [options.selector='.config-input[data-key], .config-checkbox[data-key]'] - CSS-Selector
 * @param {function} [options.beforeSave] - Hook: (config) => void|false.
 *     Config wird in-place modifiziert. Return false um Save abzubrechen.
 *     Kein Return oder truthy Return → Save wird fortgesetzt.
 * @param {function} [options.onSuccess] - Callback nach erfolgreichem Save
 * @returns {object|false} - Aktualisierte Config oder false bei Fehler
 */
async function collectAndSaveConfig(currentConfig, options = {}) {
    const container = document.getElementById('contentContainer');
    if (!container) return false;

    const endpoint = options.endpoint || '/api/admin/settings';
    const selector = options.selector || '.config-input[data-key], .config-checkbox[data-key]';
    const updated = JSON.parse(JSON.stringify(currentConfig));

    // 1. Felder einsammeln + parsen + validieren
    for (const input of container.querySelectorAll(selector)) {
        const keys = input.dataset.key.split('.');
        let val;

        if (input.type === 'checkbox') {
            val = input.checked;
        } else if (input.type === 'number') {
            const rawNumber = input.value.trim();
            val = Number(rawNumber);
            if (rawNumber === '' || !Number.isFinite(val) || !input.validity.valid) {
                SharedUi.notify.show(`Ungueltiger Wert fuer "${input.dataset.key}"`, 'error');
                input.focus();
                return false;
            }
            // HTML min/max Attribute als Validierung nutzen
            const min = input.hasAttribute('min') ? Number(input.min) : null;
            const max = input.hasAttribute('max') ? Number(input.max) : null;
            if (min !== null && val < min) {
                SharedUi.notify.show(`"${input.dataset.key}" muss mindestens ${min} sein`, 'error');
                input.focus();
                return false;
            }
            if (max !== null && val > max) {
                SharedUi.notify.show(`"${input.dataset.key}" darf hoechstens ${max} sein`, 'error');
                input.focus();
                return false;
            }
        } else if (input.tagName === 'SELECT') {
            val = input.value;
        } else {
            val = input.value;
        }

        // Pflichtfeld-Pruefung
        if (input.required && (val === '' || val === undefined || val === null)) {
            SharedUi.notify.show(`"${input.dataset.key}" ist ein Pflichtfeld`, 'error');
            input.focus();
            return false;
        }

        // Verschachteltes Config-Objekt modifizieren
        let obj = updated;
        for (let i = 0; i < keys.length - 1; i++) {
            if (!obj[keys[i]]) obj[keys[i]] = {};
            obj = obj[keys[i]];
        }
        obj[keys[keys.length - 1]] = val;
    }

    // 2. beforeSave-Hook fuer Sonderfelder (Textareas, Dropdowns via ID, etc.)
    if (options.beforeSave) {
        const result = options.beforeSave(updated);
        if (result === false) return false;
    }

    // 3. Speichern via bestehendem saveConfigToAPI
    const success = await saveConfigToAPI({
        endpoint: endpoint,
        data: { config: updated },
        successMessage: options.successMessage || 'Gespeichert',
        onSuccess: options.onSuccess,
    });

    return success ? updated : false;
}

async function deleteConfigFromAPI(config) {
    if (!await SharedUi.modal.confirm({ message: config.confirmMessage })) return false;
    try {
        await SharedUi.fetch.safeFetch(config.endpoint, { method: 'DELETE' });
        SharedUi.notify.show(config.successMessage, 'success');
        if (config.onSuccess) await config.onSuccess();
        return true;
    } catch (error) {
        const msg = (error && error.body && (error.body.detail || error.body.error || error.body.message)) || (error && error.message) || 'Loeschen fehlgeschlagen';
        SharedUi.notify.show(msg, 'error');
        return false;
    }
}
