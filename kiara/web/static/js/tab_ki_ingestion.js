/**
 * KIara Admin - KI > Ingestion Sub-Tab
 * Globale Ingestion-Einstellungen.
 */

// --- Config State ---
let kiIngestionConfig = null;
let _embeddingBackends = [];

// --- Lifecycle State ---
let _ingInitGen = 0;

// =========================================================================
//  Event-Delegation (CSP-Strict, V30 Welle 2)
// =========================================================================

const _ING_ACTIONS = new Set([
    'reset-vlm-prompt',
    'reset-accept-all-blocklist',
    'save-ki-ingestion-config',
    'reset-ki-ingestion-tab',
]);

function _handleIngClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_ING_ACTIONS.has(action)) return;

    switch (action) {
        case 'reset-vlm-prompt':
            resetVlmPrompt();
            break;
        case 'reset-accept-all-blocklist':
            resetAcceptAllBlocklist();
            break;
        case 'save-ki-ingestion-config':
            saveKiIngestionConfig(el);
            break;
        case 'reset-ki-ingestion-tab':
            initKiIngestionTab();
            break;
    }
}

function _bindIngActions() {
    // Single-Root-Delegation auf document.body (Lehre aus 0.0.1+376).
    if (document.body._ingActionsBound) return;
    document.body._ingActionsBound = true;
    document.body.addEventListener('click', _handleIngClick);

    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-ing-actions-bound', 'true');
}

// ─────────────────────────────────────────────
// Cleanup
// ─────────────────────────────────────────────

function _cleanupIngestion() {
    _ingInitGen++;
}

// ─────────────────────────────────────────────
// Init + Lifecycle (AP1)
// ─────────────────────────────────────────────

async function initKiIngestionTab() {
    // Re-Init-Sicherheit: alte Cleanups ausfuehren
    if (typeof _runTabCleanups === 'function') _runTabCleanups('ki-ingestion');

    var myGen = ++_ingInitGen;

    var container = document.getElementById('contentContainer');
    if (!container) return;

    _bindIngActions();

    // Guard vor DOM-Write
    if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-ingestion') return;

    var filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.classList.remove('u-flex', 'u-visible', 'u-block');
        filterBar.classList.add('u-hidden');
    }
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        var results = await Promise.all([
            loadConfigFromAPI('/api/admin/settings'),
            SharedUi.fetch.safeFetch('/api/admin/ki/backends/', { headers: { 'X-Requested-With': 'XMLHttpRequest' } })
                .then(function(r) { return r.ok ? r.json() : { backends: [] }; })
                .catch(function() { return { backends: [] }; }),
        ]);

        // Guard nach await
        if (_ingInitGen !== myGen) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-ingestion') return;

        var data = results[0];
        var backendsResp = results[1];

        // loadConfigFromAPI === null explizit behandeln
        if (data === null) {
            container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
            return;
        }

        kiIngestionConfig = data;
        _embeddingBackends = (backendsResp.backends || [])
            .filter(function(b) { return b.category === 'embedding' && b.is_active; });

        renderKiIngestionTab(container);

        // Cleanup registrieren (nach Render, damit alte Registrierung schon via _runTabCleanups geloescht ist)
        registerTabCleanup('ki-ingestion', _cleanupIngestion);

    } catch (e) {
        // Guard im catch
        if (_ingInitGen !== myGen) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-ingestion') return;
        container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
    }
}

// ─────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────

function renderKiIngestionTab(container) {
    if (!kiIngestionConfig) {
        container.innerHTML = '<div class="no-data">Keine Konfiguration vorhanden.</div>';
        return;
    }
    var cfg = kiIngestionConfig;
    var html = '<div class="config-form">';
    html += '<h3>Indexierungs-Einstellungen</h3>';

    // Hinweis bei aktiviertem Wissensgraph
    if (cfg.graph && cfg.graph.enabled) {
        html += '<div class="info-panel-content u-info-banner u-mb-16 u-pad-12 u-fs-13">';
        html += '<strong>Wissensgraph aktiv:</strong> Bei der Indexierung werden zusaetzlich Entitaeten und Beziehungen extrahiert. ';
        html += 'Dies erhoeht die Indexierungszeit pro Chunk um ca. 1-2 Sekunden (LLM-Call).';
        html += '</div>';
    }

    html += '<div class="config-section"><h4>Chunk-Parameter</h4><div class="config-fields">';
    // Fallback-Operatoren bewusst gemischt: || fuer chunk_size/batch_size (0 ist
    // ungueltig, soll Fallback ausloesen), ?? fuer chunk_overlap (0 ist gueltiger
    // Wert fuer "kein Overlap" und darf NICHT durch 200 ueberschrieben werden).
    html += SharedUi.forms.configField({type: 'number', label: 'Chunk-Groesse', key: 'ingestion.chunk_size', value: cfg.ingestion?.chunk_size || 1000, min: 100, max: 10000});
    html += SharedUi.forms.configField({type: 'number', label: 'Chunk-Overlap', key: 'ingestion.chunk_overlap', value: cfg.ingestion?.chunk_overlap ?? 200, min: 0, max: 5000});
    html += SharedUi.forms.configField({type: 'number', label: 'Batch-Groesse', key: 'ingestion.batch_size', value: cfg.ingestion?.batch_size || 100, min: 1, max: 2000});
    html += '</div></div>';

    // --- Late Chunking Toggle ---
    var lateChunkingEnabled = cfg.enrichment && cfg.enrichment.late_chunking || false;
    html += '<div class="config-section"><h4>Late Chunking</h4>';
    html += '<p class="u-text-sm-secondary u-mb-8">';
    html += 'Erzeugt kontextbewusste Embeddings, indem das gesamte Dokument durch den ';
    html += 'Transformer geschickt wird. Verbessert die Suche bei Pronomen und ';
    html += 'dokumentweiten Zusammenhaengen.</p>';
    html += '<div class="config-fields">';
    html += '<div class="config-field">' +
        '<label>Late Chunking aktivieren</label>' +
        '<input type="checkbox" data-key="enrichment.late_chunking" ' + (lateChunkingEnabled ? 'checked' : '') + ' class="config-checkbox">' +
        '</div>';
    html += '</div>';
    html += '<div class="info-panel-content u-info-banner u-mt-8 u-pad-10 u-fs-12">';
    html += '<strong>Hinweis:</strong> Aenderungen wirken sich nur auf zukuenftige Indexierungen aus. ';
    html += 'Fuer bestehende Datenquellen ist ein Re-Index erforderlich.';
    html += '</div>';

    // Embedding-Backends-Tabelle mit Late-Chunking-Support
    if (_embeddingBackends.length > 0) {
        html += '<table class="data-table u-mt-12 u-fs-13 u-w-full">';
        html += '<thead><tr><th>Embedding-Backend</th><th>Modell</th><th>Provider</th><th>Late Chunking</th></tr></thead>';
        html += '<tbody>';
        for (var i = 0; i < _embeddingBackends.length; i++) {
            var b = _embeddingBackends[i];
            var sup = b.supports_late_chunking;
            var badge = sup
                ? '<span class="u-badge-success">unterstuetzt</span>'
                : '<span class="u-badge-danger">nicht unterstuetzt</span>';
            html += '<tr><td>' + SharedUi.runtime.escapeHtml(b.name) + '</td><td><code>' + SharedUi.runtime.escapeHtml(b.model) + '</code></td><td>' + SharedUi.runtime.escapeHtml(b.provider) + '</td><td>' + badge + '</td></tr>';
        }
        html += '</tbody></table>';
    }

    html += '</div>';

    // --- VLM-Bildbeschreibungs-Prompt ---
    var vlmPromptVal = cfg.ingestion && cfg.ingestion.vlm_prompt || '';
    html += '<div class="config-section"><h4>VLM-Bildbeschreibungs-Prompt</h4>';
    html += '<p class="u-text-sm-secondary u-mb-8">';
    html += 'Prompt fuer die KI-Bildbeschreibung bei der Indexierung. ';
    html += 'Leer lassen fuer den Standard-Prompt (Deutsch, technische Zeichnungen, Typenschilder, Fotos).</p>';
    html += '<div class="config-fields">';
    html += '<div class="config-field u-w-full">' +
        '<textarea id="vlmPromptTextarea" rows="5" class="u-textarea-mono" placeholder="Standard: Beschreibe den Inhalt dieses Bildes detailliert auf Deutsch...">' + SharedUi.runtime.escapeHtml(vlmPromptVal) + '</textarea></div>';
    html += '</div>';
    html += '<div class="config-actions u-mt-8">';
    html += '<button class="action-btn" data-action="reset-vlm-prompt">Auf Standard zuruecksetzen</button>';
    html += '</div></div>';

    // --- Blocklist fuer accept_all_text ---
    var blocklist = cfg.ingestion && cfg.ingestion.accept_all_blocklist || [];
    html += '<div class="config-section"><h4>Ausgeschlossene Dateitypen (accept_all_text)</h4>';
    html += '<p class="u-text-sm-secondary u-mb-8">';
    html += 'Dateien mit diesen Endungen werden bei aktiviertem <em>accept_all_text</em> nie indexiert, ';
    html += 'auch wenn sie den Text-Check bestehen (z.B. S/MIME-Signaturen, Archive, Binaerdateien). ';
    html += 'Eine Endung pro Zeile, mit Punkt.</p>';
    html += '<div class="config-fields">';
    html += '<div class="config-field u-w-full">' +
        '<textarea id="acceptAllBlocklist" rows="10" class="u-textarea-mono">' + SharedUi.runtime.escapeHtml(blocklist.join('\n')) + '</textarea></div>';
    html += '</div>';
    html += '<div class="config-actions u-mt-8">';
    html += '<button class="action-btn" data-action="reset-accept-all-blocklist">Auf Standard zuruecksetzen</button>';
    html += '</div></div>';

    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-ing-save data-action="save-ki-ingestion-config">Speichern</button>';
    html += '<button class="action-btn" data-action="reset-ki-ingestion-tab">Zuruecksetzen</button>';
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;
}

// ─────────────────────────────────────────────
// Defaults + Hilfsfunktionen
// ─────────────────────────────────────────────

// Default-VLM-Prompt (identisch mit Python DEFAULT_VLM_PROMPT)
const _DEFAULT_VLM_PROMPT = "Beschreibe den Inhalt dieses Bildes detailliert auf Deutsch. " +
    "Wenn es sich um eine technische Zeichnung handelt, nenne Masse, Werkstoffe, " +
    "Artikelnummern und Fertigungshinweise. " +
    "Wenn es sich um ein Typenschild handelt, nenne alle ablesbaren Daten. " +
    "Wenn es sich um ein Foto handelt, beschreibe was zu sehen ist.";

function resetVlmPrompt() {
    var ta = document.getElementById('vlmPromptTextarea');
    if (ta) ta.value = _DEFAULT_VLM_PROMPT;
}

// Default-Blocklist (identisch mit Python _ACCEPT_ALL_BLOCKLIST)
const _DEFAULT_ACCEPT_ALL_BLOCKLIST = [
    ".p7s", ".p7m", ".p7b", ".p7c", ".pem", ".cer", ".crt", ".der",
    ".key", ".sig", ".asc", ".gpg", ".pgp",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz",
    ".dat", ".winmail", ".dvbvkey",
    ".ics", ".vcf", ".vcs",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
    ".ico", ".webp", ".svg", ".heic", ".heif",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mkv", ".mov", ".flac",
    ".exe", ".dll", ".so", ".dylib", ".msi", ".deb", ".rpm",
    ".db", ".sqlite", ".lock", ".tmp", ".bak", ".swp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
];

function _parseBlocklistTextarea() {
    var ta = document.getElementById('acceptAllBlocklist');
    if (!ta) return [];
    return ta.value.split('\n')
        .map(function(s) { return s.trim(); })
        .filter(function(s) { return s.length > 0; })
        .map(function(s) { return s.startsWith('.') ? s : '.' + s; });
}

function resetAcceptAllBlocklist() {
    var ta = document.getElementById('acceptAllBlocklist');
    if (ta) ta.value = _DEFAULT_ACCEPT_ALL_BLOCKLIST.join('\n');
}

// ─────────────────────────────────────────────
// Save (AP6)
// ─────────────────────────────────────────────

async function saveKiIngestionConfig(btn) {
    if (!kiIngestionConfig) return;
    if (!btn) return;
    var myGen = _ingInitGen;
    await SharedUi.runtime.withButtonLock(btn, async function() {
        var result = await collectAndSaveConfig(kiIngestionConfig, {
            successMessage: 'Indexierungs-Einstellungen gespeichert',
            beforeSave: function(cfg) {
                if (!cfg.ingestion) cfg.ingestion = {};
                var vlmTa = document.getElementById('vlmPromptTextarea');
                cfg.ingestion.vlm_prompt = vlmTa ? vlmTa.value.trim() : '';
                cfg.ingestion.accept_all_blocklist = _parseBlocklistTextarea();
            },
        });
        if (_ingInitGen !== myGen) return;
        if (result) kiIngestionConfig = result;
    });
}
