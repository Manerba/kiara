/**
 * KIara UI-Core — Runtime-Stylesheet-Helper (V30 Welle 3, CSP-Strict)
 *
 * Dynamische Styles, die sich nicht auf statische Utility-Klassen
 * reduzieren lassen (Progress-Widths, Grid-Positionen, animierte
 * Hoehen), brauchen unter strict CSP einen nonced Weg. Dieser Helper
 * haengt ein einziges <style nonce> in den <head> und updated CSS-
 * Regeln ueber CSSStyleSheet.insertRule/deleteRule. Dadurch wird weder
 * das style-Attribut noch element.style.* beschrieben.
 *
 * Public API: SharedUi.runtimeStyle
 *   - setRule(selector, props)         Setzt/updated eine Regel fuer einen Selektor.
 *   - applyStyle(element, props)       Weist dem Element eine generierte Klasse zu
 *                                      und setzt die CSS-Regel dafuer.
 *   - classForProps(props)             Liefert einen Klassennamen, unter dem
 *                                      die props als Regel eingefuegt wurden.
 *   - removeRule(selector)             Loescht eine zuvor gesetzte Regel.
 */
(function () {
    'use strict';

    let _sheet = null;
    const _registered = new Set();  // selectors with an active rule
    const _propsCache = new Map();  // serializedProps -> u-rt-Klassenname
    let _classCounter = 0;

    // CSSOM-Indizes verschieben sich nach deleteRule (alle Indizes > i um 1 nach
    // oben). Statt einer brueckigen Index-Map suchen wir den aktuellen Index per
    // selectorText neu — das ist auch resistent gegen externe Stylesheet-Mutation.
    function _findRuleIdx(sheet, selector) {
        const rules = sheet.cssRules;
        for (let i = 0; i < rules.length; i++) {
            if (rules[i].selectorText === selector) return i;
        }
        return -1;
    }

    function _getNonce() {
        const meta = document.querySelector('meta[name="csp-nonce"]');
        return meta ? meta.getAttribute('content') : null;
    }

    function _ensureSheet() {
        if (_sheet) return _sheet;
        const style = document.createElement('style');
        const nonce = _getNonce();
        if (nonce) style.setAttribute('nonce', nonce);
        style.setAttribute('data-sharedui-runtime-style', '');
        document.head.appendChild(style);
        _sheet = style.sheet;
        return _sheet;
    }

    function _kebab(camel) {
        return camel.replace(/[A-Z]/g, function (m) { return '-' + m.toLowerCase(); });
    }

    function _serializeProps(props) {
        const parts = [];
        for (const key in props) {
            if (!Object.prototype.hasOwnProperty.call(props, key)) continue;
            const val = props[key];
            if (val === undefined || val === null || val === '') continue;
            parts.push(_kebab(key) + ': ' + val);
        }
        return parts.join('; ');
    }

    function setRule(selector, props) {
        const sheet = _ensureSheet();
        const body = _serializeProps(props);
        if (!body) {
            removeRule(selector);
            return;
        }
        const rule = selector + ' { ' + body + ' }';
        if (_registered.has(selector)) {
            const oldIdx = _findRuleIdx(sheet, selector);
            if (oldIdx !== -1) {
                try { sheet.deleteRule(oldIdx); } catch (_) { /* ignore */ }
            }
        }
        sheet.insertRule(rule, sheet.cssRules.length);
        _registered.add(selector);
    }

    function removeRule(selector) {
        if (!_registered.has(selector) || !_sheet) return;
        const idx = _findRuleIdx(_sheet, selector);
        if (idx !== -1) {
            try { _sheet.deleteRule(idx); } catch (_) { /* ignore */ }
        }
        _registered.delete(selector);
    }

    function classForProps(props) {
        const key = _serializeProps(props);
        if (key && _propsCache.has(key)) {
            return _propsCache.get(key);
        }
        const cls = 'u-rt-' + (++_classCounter);
        setRule('.' + cls, props);
        if (key) _propsCache.set(key, cls);
        return cls;
    }

    function applyStyle(element, props) {
        if (!element) return;
        let cls = element.getAttribute('data-rt-class');
        if (!cls) {
            cls = 'u-rt-' + (++_classCounter);
            element.setAttribute('data-rt-class', cls);
            element.classList.add(cls);
        }
        setRule('.' + cls, props);
    }

    window.SharedUi = window.SharedUi || {};
    window.SharedUi.runtimeStyle = {
        setRule: setRule,
        removeRule: removeRule,
        applyStyle: applyStyle,
        classForProps: classForProps,
    };
})();
