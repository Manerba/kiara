/**
 * Shared Frontend Core — Fetch Runtime
 *
 * SharedUi.fetch API: safeFetch / safeFetchJson / FetchError.
 * Produktneutral: kein URL-Scope-Filter, kein Login-Redirect ausser via cfg.on401.
 * Benoetigt zuvor geladenes ui_core/js/core.js.
 */
(function () {
    if (!window.SharedUi || typeof window.SharedUi.configure !== 'function') {
        throw new Error('ui_core/core.js must load before fetch.js');
    }

    if (window.SharedUi.fetch) {
        return;
    }

    var MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

    function FetchError(kind, status, message, body) {
        var err = new Error(message);
        err.name = 'FetchError';
        err.kind = kind;
        err.status = status;
        err.body = body;
        return err;
    }
    FetchError.prototype = Object.create(Error.prototype);
    FetchError.prototype.constructor = FetchError;

    function _setHeader(headers, name, value) {
        if (headers && typeof headers.set === 'function' && typeof headers.append === 'function') {
            headers.set(name, value);
            return headers;
        }
        if (Array.isArray(headers)) {
            headers.push([name, value]);
            return headers;
        }
        if (headers && typeof headers === 'object') {
            headers[name] = value;
            return headers;
        }
        var obj = {};
        obj[name] = value;
        return obj;
    }

    function _cloneOptions(options) {
        var opts = Object.assign({}, options || {});
        if (opts.headers) {
            if (typeof Headers !== 'undefined' && opts.headers instanceof Headers) {
                opts.headers = new Headers(opts.headers);
            } else if (Array.isArray(opts.headers)) {
                opts.headers = opts.headers.slice();
            } else if (typeof opts.headers === 'object') {
                opts.headers = Object.assign({}, opts.headers);
            }
        }
        return opts;
    }

    function csrfHeaders() {
        var cfg = window.SharedUi._config();
        var headers = {};
        if (cfg.csrfHeader && cfg.csrfValue) {
            headers[cfg.csrfHeader] = cfg.csrfValue;
        }
        if (cfg.csrfTokenHeader && cfg.csrfTokenValue) {
            headers[cfg.csrfTokenHeader] = cfg.csrfTokenValue;
        }
        return headers;
    }

    var _refreshInFlight = null;

    async function _doRefreshCsrfToken() {
        var cfg = window.SharedUi._config();
        var url = cfg.csrfRefreshUrl;
        if (!url) {
            return null;
        }
        var headers = {};
        if (cfg.csrfHeader && cfg.csrfValue) {
            headers[cfg.csrfHeader] = cfg.csrfValue;
        }
        var resp;
        try {
            resp = await fetch(url, {
                method: 'GET',
                credentials: cfg.defaultCredentials || 'same-origin',
                headers: headers
            });
        } catch (_) {
            return null;
        }
        if (resp.status === 401) {
            if (typeof cfg.on401 === 'function') {
                try { await Promise.resolve(cfg.on401(resp, url, {})); } catch (_) {}
            } else if (cfg.loginUrl) {
                window.location.href = cfg.loginUrl;
            }
            return null;
        }
        if (!resp.ok) {
            return null;
        }
        var body;
        try { body = await resp.json(); } catch (_) { return null; }
        var token = body && body.csrf_token;
        if (typeof token === 'string' && token.length > 0) {
            cfg.csrfTokenValue = token;
            return token;
        }
        return null;
    }

    function refreshCsrfToken() {
        // Single-Flight: parallele 403/csrf_invalid-Retries warten auf denselben Refresh
        // und vermeiden Races auf cfg.csrfTokenValue.
        if (_refreshInFlight) {
            return _refreshInFlight;
        }
        var inflight = _doRefreshCsrfToken();
        _refreshInFlight = inflight;
        inflight.finally(function () {
            if (_refreshInFlight === inflight) {
                _refreshInFlight = null;
            }
        });
        return inflight;
    }

    async function _parseJsonBody(response) {
        try {
            var parsed = await response.clone().json();
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                return parsed;
            }
        } catch (_) { /* kein JSON */ }
        return null;
    }

    async function _performFetch(url, opts, cfg) {
        var method = (opts.method || 'GET').toUpperCase();
        if (MUTATING.has(method)) {
            if (cfg.csrfHeader) {
                opts.headers = _setHeader(opts.headers, cfg.csrfHeader, cfg.csrfValue);
            }
            if (cfg.csrfTokenHeader && cfg.csrfTokenValue) {
                opts.headers = _setHeader(opts.headers, cfg.csrfTokenHeader, cfg.csrfTokenValue);
            }
        }
        if (!('credentials' in opts) && cfg.defaultCredentials) {
            opts.credentials = cfg.defaultCredentials;
        }
        return fetch(url, opts);
    }

    async function _confirmAndRetry(response, url, options, cfg) {
        if (response.status !== 428 && response.status !== 409) {
            return response;
        }
        options = options || {};
        if (options.confirmationRetry === false) {
            return response;
        }
        var body = await _parseJsonBody(response);
        if (!body || body.requires_confirmation !== true || !body.confirmation) {
            return response;
        }
        if (!window.SharedUi.modal || typeof window.SharedUi.modal.confirm !== 'function') {
            return response;
        }

        var confirmation = body.confirmation || {};
        var confirmString = confirmation.confirm_string || '';
        var message = body.message || 'Bestaetigung erforderlich';
        if (confirmString) {
            message += ': ' + confirmString;
        }
        var accepted = await window.SharedUi.modal.confirm({
            title: 'Bestaetigung',
            message: message,
        });
        if (!accepted) {
            throw FetchError('confirmation_cancelled', response.status, 'Bestaetigung abgebrochen', body);
        }

        var retryOpts = _cloneOptions(options);
        delete retryOpts.confirmationRetry;
        retryOpts.headers = _setHeader(
            retryOpts.headers,
            'X-KIARA-Confirm-Token',
            confirmation.confirm_token || ''
        );
        retryOpts.headers = _setHeader(
            retryOpts.headers,
            'X-KIARA-Confirm-String',
            confirmString
        );
        try {
            return await _performFetch(url, retryOpts, cfg);
        } catch (err) {
            if (err && err.name === 'AbortError') {
                throw err;
            }
            throw FetchError('network', 0, err && err.message ? err.message : String(err), null);
        }
    }

    async function safeFetch(url, options) {
        var cfg = window.SharedUi._config();
        var opts = _cloneOptions(options);
        var csrfRetry = opts.csrfRetry !== false;   // Opt-out fuer Streaming
        delete opts.csrfRetry;                       // nicht an fetch() durchreichen
        delete opts.confirmationRetry;               // nur fuer safeFetch-Steuerung

        var response;
        try {
            response = await _performFetch(url, opts, cfg);
        } catch (err) {
            if (err && err.name === 'AbortError') {
                throw err;
            }
            throw FetchError('network', 0, err && err.message ? err.message : String(err), null);
        }

        if (response.status === 401) {
            if (typeof cfg.on401 === 'function') {
                try {
                    await Promise.resolve(cfg.on401(response, url, opts));
                } catch (_) { /* on401-Fehler werden geschluckt */ }
            } else if (cfg.loginUrl) {
                window.location.href = cfg.loginUrl;
            }
            throw FetchError('unauthorized', 401, 'unauthorized', null);
        }

        if (response.status === 403 && csrfRetry) {
            var body403 = await _parseJsonBody(response);
            if (body403 && body403.code === 'csrf_invalid') {
                var fresh = await refreshCsrfToken();
                if (fresh) {
                    // Eine einmalige Wiederholung mit frischem Token — Opt-out verhindert Rekursion
                    var retryOpts = _cloneOptions(options);
                    retryOpts.csrfRetry = false;
                    try {
                        response = await _performFetch(url, retryOpts, window.SharedUi._config());
                    } catch (err) {
                        if (err && err.name === 'AbortError') {
                            throw err;
                        }
                        throw FetchError('network', 0, err && err.message ? err.message : String(err), null);
                    }
                    if (response.status === 401) {
                        if (typeof cfg.on401 === 'function') {
                            try { await Promise.resolve(cfg.on401(response, url, retryOpts)); } catch (_) {}
                        } else if (cfg.loginUrl) {
                            window.location.href = cfg.loginUrl;
                        }
                        throw FetchError('unauthorized', 401, 'unauthorized', null);
                    }
                }
            }
        }

        response = await _confirmAndRetry(response, url, options, cfg);

        if (response.status === 401) {
            if (typeof cfg.on401 === 'function') {
                try { await Promise.resolve(cfg.on401(response, url, opts)); } catch (_) {}
            } else if (cfg.loginUrl) {
                window.location.href = cfg.loginUrl;
            }
            throw FetchError('unauthorized', 401, 'unauthorized', null);
        }

        if (!response.ok) {
            var body = await _parseJsonBody(response);
            var msg = (body && (body.detail || body.error || body.message)) || response.statusText;
            throw FetchError('http', response.status, msg, body);
        }

        return response;
    }

    async function safeFetchJson(url, options) {
        var resp = await safeFetch(url, options);
        if (resp.status === 204) return null;
        try {
            return await resp.json();
        } catch (e) {
            throw FetchError('parse', resp.status, 'Antwort konnte nicht als JSON gelesen werden', null);
        }
    }

    window.SharedUi.fetch = {
        safeFetch: safeFetch,
        safeFetchJson: safeFetchJson,
        FetchError: FetchError,
        csrfHeaders: csrfHeaders,
        refreshCsrfToken: refreshCsrfToken
    };
})();
