/**
 * KIara Admin - Compliance > Anonymisierung Sub-Tab
 * Anonymisierungs-Konfiguration mit Layer-Sektionen und Test
 */

let complianceAnonConfig = null;
let _anonInitSeq = 0;

const _ANON_ACTIONS = new Set([
    'reload-automaton',
    'save-compliance-anon',
    'open-anon-test-dialog',
    'run-anon-test',
    'close-anon-test-dialog',
]);

function _handleAnonClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_ANON_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'reload-automaton':        reloadAutomaton(); break;
        case 'save-compliance-anon':    SharedUi.runtime.withButtonLock(el, saveComplianceAnonConfig); break;
        case 'open-anon-test-dialog':   openAnonymizeTestDialog(); break;
        case 'run-anon-test':           runAnonymizeTest(); break;
        case 'close-anon-test-dialog':  closeAnonymizeTestDialog(); break;
    }
}

function _bindAnonActions() {
    if (document.body._anonActionsBound) return;
    document.body._anonActionsBound = true;
    document.body.addEventListener('click', _handleAnonClick);
}

async function initComplianceAnonTab() {
    _anonInitSeq++;
    const mySeq = _anonInitSeq;

    _bindAnonActions();
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        complianceAnonConfig = await loadConfigFromAPI('/api/admin/ki/config');
        if (mySeq !== _anonInitSeq) return;
        if (complianceAnonConfig === null) {
            container.innerHTML = '<div class="no-data">KI-Konfiguration konnte nicht geladen werden. Bitte pruefen Sie die Backend-Verbindung.</div>';
            return;
        }
        renderComplianceAnonTab(container);
    } catch (e) {
        if (mySeq !== _anonInitSeq) return;
        container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
    }
}

function _sectionHeader(number, title, subtitle) {
    let html = `<h4 class="cmp-section-title">`;
    html += `<span class="cmp-section-badge">${number}</span>`;
    html += `${SharedUi.runtime.escapeHtml(title)}</h4>`;
    if (subtitle) {
        html += `<p class="cmp-section-subtitle">${SharedUi.runtime.escapeHtml(subtitle)}</p>`;
    }
    return html;
}

function renderComplianceAnonTab(container) {
    const cfg = complianceAnonConfig || {};
    let html = '<div class="config-form">';
    html += '<h3>Anonymisierung</h3>';
    html += '<p class="cmp-intro">Vierstufige Anonymisierung: ERP-Stammdaten, LDAP/AD, Regex-Muster und LLM-Erkennung. Sensible Daten werden vor Cloud-Versand maskiert.</p>';

    // =====================================================================
    // Allgemein
    // =====================================================================
    html += '<div class="config-section">';
    html += '<h4 class="cmp-section-title-plain">Allgemeine Einstellungen</h4>';
    html += '<div class="config-fields">';

    html += '<div class="config-field"><label class="u-flex-center-gap-8 u-clickable">';
    html += `<input type="checkbox" id="ki_anon_enabled" ${cfg.anon_enabled !== false ? 'checked' : ''}>`;
    html += 'Anonymisierung global aktivieren</label>';
    html += '<div class="config-hint u-text-xs-secondary u-mt-4">Aus: keine Anonymisierung, auch bei Datenquellen mit der Vorgabe „Anonymisieren“. An: die Vorgaben der Datenquellen gelten. Cloud-Sperren gelten unabhaengig davon.</div></div>';

    html += '<div class="config-field"><label class="u-flex-center-gap-8 u-clickable">';
    html += `<input type="checkbox" id="ki_anon_audit_log" ${cfg.anon_audit_log !== false ? 'checked' : ''}>`;
    html += 'Audit-Log aktivieren</label></div>';

    // Excluded terms
    html += '<div class="config-field"><label>Ausgeschlossene Begriffe (einer pro Zeile)</label>';
    let excludedTerms = '';
    const _excludedTermsRaw = cfg.anon_excluded_terms || '[]';
    try {
        excludedTerms = JSON.parse(_excludedTermsRaw).join('\n');
    } catch(e) {
        excludedTerms = _excludedTermsRaw;
        SharedUi.notify.show('Ausgeschlossene Begriffe: JSON konnte nicht geparst werden — Roh-Inhalt wird angezeigt, bitte korrigieren', 'warning');
    }
    html += `<textarea class="config-input u-textarea-mono" id="ki_anon_excluded_terms" rows="3" placeholder="IT&#10;SAP&#10;HR">${SharedUi.runtime.escapeHtml(excludedTerms)}</textarea></div>`;

    html += '</div></div>';

    // =====================================================================
    // Schicht 1: ERP-Stammdaten (Aho-Corasick)
    // =====================================================================
    html += '<div class="config-section">';
    html += _sectionHeader('1', 'ERP-Stammdaten (Aho-Corasick)', 'Deterministische Erkennung bekannter Kunden, Lieferanten, Mitarbeiter und Produkte aus CSV/JSON-Stammdaten.');
    html += '<div class="config-fields">';

    html += '<div class="config-field"><label class="u-flex-center-gap-8 u-clickable">';
    html += `<input type="checkbox" id="ki_anon_erp_enabled" ${cfg.anon_erp_enabled ? 'checked' : ''}>`;
    html += 'ERP-Import aktiv</label></div>';

    html += '<div class="config-field"><label>Stammdaten-Pfad</label>';
    html += `<input type="text" class="config-input" id="ki_anon_erp_path" value="${SharedUi.runtime.escapeAttr(cfg.anon_erp_path || '/etc/kiara/stammdaten')}" placeholder="/etc/kiara/stammdaten"></div>`;

    html += '<div class="config-field u-flex-gap-8 u-items-center">';
    html += '<button class="action-btn" data-action="reload-automaton" id="btnReloadAuto">Stammdaten laden</button>';
    html += '<span id="automatonStatus" class="u-fs-13"></span>';
    html += '</div>';

    // Word boundaries + min length
    html += '<div class="config-field"><label class="u-flex-center-gap-8 u-clickable">';
    html += `<input type="checkbox" id="ki_anon_word_boundaries" ${cfg.anon_word_boundaries !== false ? 'checked' : ''}>`;
    html += 'Wortgrenzen-Pruefung (empfohlen)</label></div>';

    html += '<div class="config-field"><label>Mindestlaenge fuer Treffer</label>';
    html += `<input type="number" class="config-input u-width-80" id="ki_anon_min_term_length" value="${cfg.anon_min_term_length || 3}" min="1" max="20"></div>`;

    // Custom entries
    html += '<div class="config-field"><label>Benutzerdefinierte Eintraege (JSON-Array)</label>';
    html += '<p class="cmp-field-hint">Format: [{"text":"Firmenname","type":"ORG","key":"K-001"}]</p>';
    html += `<textarea class="config-input u-textarea-mono-xs" id="ki_anon_custom_entries" rows="3">${SharedUi.runtime.escapeHtml(cfg.anon_custom_entries || '[]')}</textarea></div>`;

    html += '</div></div>';

    // =====================================================================
    // Schicht 2: LDAP / Active Directory
    // =====================================================================
    html += '<div class="config-section">';
    html += _sectionHeader('2', 'LDAP / Active Directory', 'Mitarbeiter-Stammdaten aus dem Active Directory ergaenzen den Aho-Corasick-Automaten.');
    html += '<div class="config-fields">';

    html += '<div class="config-field"><label class="u-flex-center-gap-8 u-clickable">';
    html += `<input type="checkbox" id="ki_anon_ldap_enabled" ${cfg.anon_ldap_enabled !== false ? 'checked' : ''}>`;
    html += 'LDAP/AD-Mitarbeiter einbeziehen</label></div>';

    html += '</div></div>';

    // =====================================================================
    // Schicht 3: Regex-Muster
    // =====================================================================
    html += '<div class="config-section">';
    html += _sectionHeader('3', 'Regex-Muster', 'Erkennt strukturierte Daten wie IBAN, Steuer-IDs, E-Mail-Adressen und Telefonnummern per regulaeren Ausdruecken.');
    html += '<div class="config-fields">';

    html += '<div class="config-field"><label>Benutzerdefinierte Regex-Muster (JSON-Array)</label>';
    html += '<p class="cmp-field-hint">Format: [{"name":"Mitarbeiter-Nr","pattern":"MA-\\\\d{5}","type":"INTERNAL_CODE"}]</p>';
    html += `<textarea class="config-input u-textarea-mono-xs" id="ki_anon_custom_patterns" rows="3">${SharedUi.runtime.escapeHtml(cfg.anon_custom_patterns || '[]')}</textarea></div>`;

    html += '</div></div>';

    // =====================================================================
    // Schicht 4: LLM-Erkennung
    // =====================================================================
    html += '<div class="config-section">';
    html += _sectionHeader('4', 'LLM-Erkennung', 'Probabilistische Erkennung unbekannter Personen, Organisationen und Adressen per Sprachmodell. Analysiert nur noch nicht maskierte Textbereiche.');
    html += '<div class="config-fields">';

    html += '<div class="config-field"><label class="u-flex-center-gap-8 u-clickable">';
    html += `<input type="checkbox" id="ki_anon_llm_enabled" ${cfg.anon_llm_enabled ? 'checked' : ''}>`;
    html += 'LLM-Erkennung aktivieren</label>';
    html += '<p class="cmp-field-hint cmp-field-hint--mt-4">Erhoet Latenz um ca. 1-5 Sekunden pro Anfrage. Ergebnisse werden gecacht.</p>';
    html += '</div>';

    html += '<div class="config-field"><label>Modell (leer = Standard-LLM)</label>';
    html += `<input type="text" class="config-input" id="ki_anon_llm_model" value="${SharedUi.runtime.escapeAttr(cfg.anon_llm_model || '')}" placeholder="Standard-LLM verwenden"></div>`;

    html += '<div class="config-field"><label>Confidence-Schwellwert</label>';
    html += '<p class="cmp-field-hint">Entitaeten unter diesem Wert werden verworfen (0.0 - 1.0)</p>';
    html += `<input type="text" class="config-input u-width-100" id="ki_anon_llm_confidence" value="${SharedUi.runtime.escapeAttr(cfg.anon_llm_confidence || '0.7')}" placeholder="0.7"></div>`;

    html += '<div class="config-field"><label>Temperatur</label>';
    html += '<p class="cmp-field-hint">Niedrig = deterministischer (empfohlen: 0.1)</p>';
    html += `<input type="text" class="config-input u-width-100" id="ki_anon_llm_temperature" value="${SharedUi.runtime.escapeAttr(cfg.anon_llm_temperature || '0.1')}" placeholder="0.1"></div>`;

    html += '<div class="config-field"><label>Timeout (Sekunden)</label>';
    html += `<input type="number" class="config-input u-width-100" id="ki_anon_llm_timeout" value="${cfg.anon_llm_timeout || 10}" min="1" max="60"></div>`;

    html += '<div class="config-field"><label>Cache-Groesse</label>';
    html += '<p class="cmp-field-hint">Anzahl gecachter Ergebnisse (LRU)</p>';
    html += `<input type="number" class="config-input u-width-100" id="ki_anon_llm_cache_size" value="${cfg.anon_llm_cache_size || 256}" min="0" max="10000"></div>`;

    html += '<div class="config-field"><label>Context-Window (num_ctx)</label>';
    html += `<input type="number" class="config-input u-width-140" id="ki_anon_llm_num_ctx" value="${cfg.anon_llm_num_ctx || 4096}" min="512" max="131072" step="512"></div>`;

    html += '</div></div>';

    // =====================================================================
    // Test + Speichern
    // =====================================================================
    html += '<div class="u-flex-gap-8 u-items-center u-mt-8">';
    html += '<button class="action-btn primary" data-action="save-compliance-anon">Speichern</button>';
    html += '<button class="action-btn" data-action="open-anon-test-dialog">Anonymisierung testen...</button>';
    html += '</div>';
    html += '</div>';

    // Test modal
    html += `<div id="anonTestModal" class="anon-test-modal u-hidden">
        <div class="anon-test-modal-dialog">
            <h4 class="anon-test-modal-title">Anonymisierung testen</h4>
            <div class="config-field"><label>Beispieltext</label>
            <textarea id="anonTestInput" class="config-input u-textarea-mono" rows="5" placeholder="Schmidt GmbH hat eine Rechnung an Mueller AG geschickt..."></textarea></div>
            <div class="u-flex-gap-8 u-my-12-8">
                <button class="action-btn primary" data-action="run-anon-test" id="btnAnonTest">Anonymisieren</button>
                <button class="action-btn" data-action="close-anon-test-dialog">Schliessen</button>
            </div>
            <div id="anonTestResult" class="u-hidden">
                <div class="config-field"><label>Anonymisierter Text</label>
                <pre id="anonTestOutput" class="anon-test-output"></pre></div>
                <div id="anonTestMatches"></div>
                <div id="anonTestStats" class="anon-test-stats"></div>
            </div>
        </div>
    </div>`;

    container.innerHTML = html;
    _loadAutomatonStats();
}

async function reloadAutomaton() {
    const btn = document.getElementById('btnReloadAuto');
    if (btn) { btn.disabled = true; btn.textContent = 'Lade...'; }
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/reload-automaton', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Fehler');
        const statusEl = document.getElementById('automatonStatus');
        if (statusEl) {
            statusEl.innerHTML = `<span class="u-c-success">${SharedUi.runtime.escapeHtml(data.entry_count)} Eintraege geladen</span>`;
        }
        SharedUi.notify.show(`Automat aktualisiert: ${data.entry_count} Eintraege`, 'success');
    } catch (e) {
        SharedUi.notify.show('Stammdaten laden fehlgeschlagen: ' + e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Stammdaten laden'; }
    }
}

async function _loadAutomatonStats() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/automaton-stats');
        if (!resp.ok) return;
        const data = await resp.json();
        const statusEl = document.getElementById('automatonStatus');
        if (statusEl && data.entry_count > 0) {
            statusEl.innerHTML = `<span class="u-text-xs-secondary">${SharedUi.runtime.escapeHtml(data.entry_count)} Eintraege | Letzter Import: ${SharedUi.runtime.escapeHtml(data.last_built || 'nie')}</span>`;
        }
    } catch (e) { /* ignore */ }
}

function openAnonymizeTestDialog() {
    const modal = document.getElementById('anonTestModal');
    if (modal) {
        modal.classList.remove('u-hidden');
        modal.classList.add('u-flex');
        const result = document.getElementById('anonTestResult');
        if (result) {
            result.classList.add('u-hidden');
            result.classList.remove('u-block');
        }
    }
}

function closeAnonymizeTestDialog() {
    const modal = document.getElementById('anonTestModal');
    if (modal) {
        modal.classList.add('u-hidden');
        modal.classList.remove('u-flex');
    }
}

async function runAnonymizeTest() {
    const text = document.getElementById('anonTestInput')?.value;
    if (!text) { SharedUi.notify.show('Bitte Beispieltext eingeben', 'error'); return; }
    const btn = document.getElementById('btnAnonTest');
    if (btn) { btn.disabled = true; btn.textContent = 'Teste...'; }
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/test-anonymize', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Fehler');
        const result = document.getElementById('anonTestResult');
        if (result) { result.classList.remove('u-hidden'); result.classList.add('u-block'); }
        document.getElementById('anonTestOutput').textContent = data.anonymized || '(keine Aenderungen)';
        const matchesEl = document.getElementById('anonTestMatches');
        if (matchesEl && data.matches && data.matches.length > 0) {
            let mhtml = '<label class="anon-matches-label">Gefundene Entitaeten</label>';
            mhtml += '<table class="anon-matches-table">';
            mhtml += '<tr class="u-cell-border"><th class="u-cell-left-4-8">Original</th><th class="u-cell-left-4-8">Pseudonym</th></tr>';
            data.matches.forEach(m => {
                mhtml += `<tr class="u-cell-border"><td class="u-cell-left-4-8">${SharedUi.runtime.escapeHtml(m.original)}</td><td class="u-cell-left-4-8 u-mono">${SharedUi.runtime.escapeHtml(m.pseudonym)}</td></tr>`;
            });
            mhtml += '</table>';
            matchesEl.innerHTML = mhtml;
        } else if (matchesEl) {
            matchesEl.innerHTML = '<p class="u-text-xs-secondary">Keine Entitaeten gefunden.</p>';
        }
        const statsEl = document.getElementById('anonTestStats');
        if (statsEl && data.stats) {
            statsEl.textContent = `Gesamt: ${data.stats.total_entities} Entitaeten`;
        }
    } catch (e) {
        SharedUi.notify.show('Anonymisierung fehlgeschlagen: ' + e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Anonymisieren'; }
    }
}

function getComplianceAnonFormData() {
    const excludedRaw = document.getElementById('ki_anon_excluded_terms')?.value || '';
    const excludedArr = excludedRaw.split('\n').map(s => s.trim()).filter(Boolean);
    return {
        anon_enabled: document.getElementById('ki_anon_enabled')?.checked !== false,
        anon_erp_enabled: document.getElementById('ki_anon_erp_enabled')?.checked || false,
        anon_erp_path: document.getElementById('ki_anon_erp_path')?.value || '/etc/kiara/stammdaten',
        anon_ldap_enabled: document.getElementById('ki_anon_ldap_enabled')?.checked !== false,
        anon_word_boundaries: document.getElementById('ki_anon_word_boundaries')?.checked !== false,
        anon_min_term_length: parseInt(document.getElementById('ki_anon_min_term_length')?.value || '3', 10),
        anon_excluded_terms: JSON.stringify(excludedArr),
        anon_custom_patterns: document.getElementById('ki_anon_custom_patterns')?.value || '[]',
        anon_audit_log: document.getElementById('ki_anon_audit_log')?.checked !== false,
        anon_custom_entries: document.getElementById('ki_anon_custom_entries')?.value || '[]',
        // LLM-Erkennung (Schicht 4)
        anon_llm_enabled: document.getElementById('ki_anon_llm_enabled')?.checked || false,
        anon_llm_model: document.getElementById('ki_anon_llm_model')?.value || '',
        anon_llm_temperature: document.getElementById('ki_anon_llm_temperature')?.value || '0.1',
        anon_llm_confidence: document.getElementById('ki_anon_llm_confidence')?.value || '0.7',
        anon_llm_timeout: parseInt(document.getElementById('ki_anon_llm_timeout')?.value || '10', 10),
        anon_llm_cache_size: parseInt(document.getElementById('ki_anon_llm_cache_size')?.value || '256', 10),
        anon_llm_num_ctx: parseInt(document.getElementById('ki_anon_llm_num_ctx')?.value || '4096', 10),
    };
}

async function saveComplianceAnonConfig() {
    const formData = getComplianceAnonFormData();
    // Validate JSON fields
    try { JSON.parse(formData.anon_excluded_terms); } catch(e) {
        SharedUi.notify.show('Ausgeschlossene Begriffe: Ungueltiges JSON', 'error'); return;
    }
    try { JSON.parse(formData.anon_custom_patterns); } catch(e) {
        SharedUi.notify.show('Regex-Muster: Ungueltiges JSON', 'error'); return;
    }
    try { JSON.parse(formData.anon_custom_entries); } catch(e) {
        SharedUi.notify.show('Benutzerdefinierte Eintraege: Ungueltiges JSON', 'error'); return;
    }
    // Validate numeric LLM fields (strict: keine Prefix-Werte wie "0.7foo")
    const _isStrictFloat = (s) => /^\d+(\.\d+)?$/.test(s.trim());
    if (!_isStrictFloat(formData.anon_llm_confidence)) {
        SharedUi.notify.show('Confidence-Schwellwert: Ungueltige Zahl', 'error'); return;
    }
    const conf = parseFloat(formData.anon_llm_confidence);
    if (conf < 0 || conf > 1) {
        SharedUi.notify.show('Confidence-Schwellwert muss zwischen 0.0 und 1.0 liegen', 'error'); return;
    }
    if (!_isStrictFloat(formData.anon_llm_temperature)) {
        SharedUi.notify.show('Temperatur: Ungueltige Zahl', 'error'); return;
    }
    const temp = parseFloat(formData.anon_llm_temperature);
    if (temp < 0 || temp > 2) {
        SharedUi.notify.show('Temperatur muss zwischen 0.0 und 2.0 liegen', 'error'); return;
    }
    try {
        // Vollconfig-Preread: null = Fehler/Blockierzustand
        const fullCfg = await loadConfigFromAPI('/api/admin/ki/config');
        if (fullCfg === null) {
            SharedUi.notify.show('Vollkonfiguration konnte nicht geladen werden — Speichern abgebrochen.', 'error');
            return;
        }
        const merged = Object.assign({}, fullCfg, formData);

        // Lokaler POST (nicht ueber saveConfigToAPI, um Toast-Reihenfolge zu kontrollieren)
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(merged),
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || errData.message || 'Speichern fehlgeschlagen');
        }

        // Re-Read: null nach erfolgreichem POST ist ein Fehlerzustand
        const reloaded = await loadConfigFromAPI('/api/admin/ki/config');
        if (reloaded === null) {
            SharedUi.notify.show('Gespeichert, aber Neuladen der Konfiguration fehlgeschlagen.', 'error');
            return;
        }

        // Erst jetzt: Erfolg signalisieren + Tab mit frischen Daten rendern
        complianceAnonConfig = reloaded;
        const container = document.getElementById('contentContainer');
        if (container) renderComplianceAnonTab(container);
        SharedUi.notify.show('Anonymisierungs-Konfiguration gespeichert', 'success');
    } catch (e) {
        SharedUi.notify.show('Speichern fehlgeschlagen: ' + e.message, 'error');
    }
}
