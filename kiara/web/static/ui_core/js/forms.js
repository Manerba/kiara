/**
 * SharedUi.forms — produktneutrale Form-Helper (Core-Baustein 7).
 *
 * Abhaengigkeiten (Reihenfolge im Template):
 *   core.js -> runtime.js -> forms.js
 *
 * Public API:
 *   SharedUi.forms.configField(opts) -> HTML-String
 *
 * Die Ausgabe ist byte-identisch zum frueheren KIara-Helper (ehem.
 * form_helpers.js) — bestehende Aufrufer und `app_core.js#collectAndSaveConfig`
 * haengen an den `data-key`- und Validierungsattributen. Attribut-Reihenfolge:
 *   type -> data-key -> id -> value -> min -> max -> placeholder -> class -> required
 */
(function () {
    'use strict';

    if (typeof window.SharedUi === 'undefined' || typeof window.SharedUi.configure !== 'function') {
        console.error('SharedUi.forms: core.js muss vor forms.js laden.');
        return;
    }
    if (!window.SharedUi.runtime || typeof window.SharedUi.runtime.escapeHtml !== 'function') {
        console.error('SharedUi.forms: runtime.js muss vor forms.js laden.');
        return;
    }
    if (window.SharedUi.forms) {
        return;
    }

    var ALLOWED_TYPES = { text: true, number: true };

    function configField(opts) {
        if (!opts || typeof opts !== 'object') {
            throw new Error('configField: opts muss ein Objekt sein');
        }
        if (!opts.type || !ALLOWED_TYPES[opts.type]) {
            throw new Error('configField: type muss "text" oder "number" sein, erhalten: ' + String(opts.type));
        }
        if (!opts.label) {
            throw new Error('configField: label ist Pflicht');
        }
        if (!opts.key) {
            throw new Error('configField: key ist Pflicht');
        }

        var html = '<div class="config-field"><label>' + SharedUi.runtime.escapeHtml(opts.label) + '</label>';
        html += '<input type="' + SharedUi.runtime.escapeAttr(opts.type) + '"';
        html += ' data-key="' + SharedUi.runtime.escapeAttr(opts.key) + '"';

        if (opts.id) {
            html += ' id="' + SharedUi.runtime.escapeAttr(opts.id) + '"';
        }
        if (opts.value !== undefined && opts.value !== null) {
            html += ' value="' + SharedUi.runtime.escapeAttr(String(opts.value)) + '"';
        }
        if (opts.min !== undefined && opts.min !== null) {
            html += ' min="' + SharedUi.runtime.escapeAttr(String(opts.min)) + '"';
        }
        if (opts.max !== undefined && opts.max !== null) {
            html += ' max="' + SharedUi.runtime.escapeAttr(String(opts.max)) + '"';
        }
        if (opts.placeholder) {
            html += ' placeholder="' + SharedUi.runtime.escapeAttr(opts.placeholder) + '"';
        }

        var cls = 'config-input';
        if (opts.cls) {
            cls += ' ' + opts.cls;
        }
        html += ' class="' + SharedUi.runtime.escapeAttr(cls) + '"';

        if (opts.required === true) {
            html += ' required';
        }

        html += '>';

        if (opts.hint) {
            html += '<span class="field-hint">' + SharedUi.runtime.escapeHtml(opts.hint) + '</span>';
        }

        html += '</div>';
        return html;
    }

    SharedUi.forms = {
        configField: configField
    };
})();
