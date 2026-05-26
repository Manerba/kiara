/**
 * KIara Admin - KI > Retrieval Sub-Tab
 * Hybrid-Suche, Cross-Encoder Reranking, Query-Expansion
 *
 * Sprint 1.2: Elasticsearch State-Machine
 * - Init/Cleanup/latest-only Guards
 * - PollingManager fuer Status-Poll
 * - Generationsbasierter Task-Poll
 * - Start-Guard fuer rebuild/verify
 * - withButtonLock fuer delete/save
 * - Offscreen-Result-Recovery
 */

// _escRet entfernt — SharedUi.runtime.escapeHtml() aus app_core.js wird verwendet

let kiRetrievalConfig = null;

const _RET_ACTIONS = new Set([
    'es-rebuild-index',
    'es-verify-integrity',
    'es-delete-index',
    'save-ki-retrieval-config',
    'reset-ki-retrieval-tab',
]);

const _RET_CHANGE_ACTIONS = new Set([
    'toggle-retrieval-section',
]);

function _handleRetClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_RET_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'es-rebuild-index':         esRebuildIndex(); break;
        case 'es-verify-integrity':      esVerifyIntegrity(); break;
        case 'es-delete-index':          esDeleteIndex(el); break;
        case 'save-ki-retrieval-config': saveKiRetrievalConfig(el); break;
        case 'reset-ki-retrieval-tab':   initKiRetrievalTab(); break;
    }
}

function _handleRetChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el || !_RET_CHANGE_ACTIONS.has(el.dataset.changeAction)) return;
    if (el.dataset.changeAction === 'toggle-retrieval-section') {
        toggleRetrievalSection(el.dataset.section || '');
    }
}

function _bindRetActions() {
    if (document.body._retActionsBound) return;
    document.body._retActionsBound = true;
    document.body.addEventListener('click', _handleRetClick);
    document.body.addEventListener('change', _handleRetChange);
}

// --- Sprint 1.2: Lifecycle + Poll State ---
let _esInitSeq = 0;            // Init-Generation (latest-only)
let _esTaskGen = 0;            // Task-Poll-Generation (Invalidierung)
let _esStartGuard = false;     // Synchroner Start-Guard (rebuild/verify)
let _esTaskPollActive = false;  // Taskpoll laeuft gerade
let _esLastShownResult = null;  // Result-Lifecycle-Anker (Fingerprint)
let _esSaveInFlight = false;    // Save-in-flight-Guard

function _esGetCurrentMode() {
    const el = document.getElementById('retrieval-hybrid-search');
    return el ? el.value : null;
}

function _esSetButtons(enabled) {
    ['es-btn-rebuild', 'es-btn-verify', 'es-btn-delete'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = !enabled;
    });
}

function _esStartPolling() {
    kiaraPolling.start('ki-retrieval:refresh', () => {
        if (_esGetCurrentMode() !== 'elasticsearch') return;
        return esLoadStatus();
    }, 2000);
}

function _esStopPolling() {
    kiaraPolling.stop('ki-retrieval:refresh');
    _esTaskGen++;
    _esTaskPollActive = false;
}

// ---- Init + Lifecycle (AP1) ----

async function initKiRetrievalTab() {
    _bindRetActions();
    // Re-Init-Schutz: alte Polls/Cleanups beenden
    if (typeof _runTabCleanups === 'function') _runTabCleanups('ki-retrieval');
    _esInitSeq++;
    const mySeq = _esInitSeq;
    // Guards NICHT blind freigeben — Save-Pfad resettet _esSaveInFlight selbst im finally

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.classList.remove('u-flex', 'u-visible', 'u-block');
        filterBar.classList.add('u-hidden');
    }
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    // Cleanup registrieren — stoppt BEIDE Poll-Pfade (Status + Task)
    registerTabCleanup('ki-retrieval', () => { _esStopPolling(); });

    try {
        const data = await loadConfigFromAPI('/api/admin/settings');
        // Latest-only guard nach await
        if (mySeq !== _esInitSeq) return;
        if (typeof adminCurrentSubTab !== 'undefined' && adminCurrentSubTab !== 'ki-retrieval') return;

        // Explizites null-Handling
        if (!data) {
            container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
            return;
        }

        kiRetrievalConfig = data;
        renderKiRetrievalTab(container);
    } catch (e) {
        // Async-Fehler final selbst behandeln (Caller awaited nicht)
        if (mySeq !== _esInitSeq) return;
        container.innerHTML = '<div class="no-data">Konfiguration konnte nicht geladen werden.</div>';
    }
}

// ---- Render ----

function renderKiRetrievalTab(container) {
    if (!kiRetrievalConfig) {
        container.innerHTML = '<div class="no-data">Keine Konfiguration vorhanden.</div>';
        return;
    }
    const cfg = kiRetrievalConfig;
    const r = cfg.retrieval || {};

    let html = '<div class="config-form">';
    html += '<h3>Retrieval-Einstellungen</h3>';

    // --- Section 1: Hybrid-Suche ---
    let hybridMode = r.hybrid_search || 'none';

    const showBm25 = hybridMode === 'bm25';
    const showEs = hybridMode === 'elasticsearch';
    const showRrf = showBm25 || showEs;

    html += '<div class="config-section">';
    html += '<h4>Hybrid-Suche</h4>';
    html += '<p class="config-hint">Kombiniert semantische Suche (Vektordatenbank) mit Keyword-Suche und fusioniert die Ergebnisse per Reciprocal Rank Fusion. Semantische Suche findet bedeutungsaehnliche Texte, waehrend Keyword-Suche exakte Begriffe und Fachwoerter besser trifft. Die Fusion beider Verfahren liefert deutlich bessere Ergebnisse als jedes einzeln.</p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field"><label>Methode</label>
        <select id="retrieval-hybrid-search" class="config-input" data-change-action="toggle-retrieval-section" data-section="hybrid">
            <option value="none"${hybridMode === 'none' ? ' selected' : ''}>Keine</option>
            <option value="bm25"${hybridMode === 'bm25' ? ' selected' : ''}>BM25 (In-Memory)</option>
            <option value="elasticsearch"${hybridMode === 'elasticsearch' ? ' selected' : ''}>Elasticsearch</option>
        </select>
        <span class="field-hint">BM25 baut einen In-Memory-Index aus allen Chunks auf (benoetigt RAM, ~90s Startup bei 630k Chunks). Elasticsearch nutzt einen externen Suchserver mit persistentem Index (schnellerer Startup, skaliert besser).</span>
    </div>`;

    // --- BM25-spezifische Optionen ---
    html += `<div id="bm25-deps" class="${showBm25 ? '' : 'u-hidden'}">
        <div class="config-fields">
        </div>
    </div>`;

    // --- Elasticsearch-spezifische Optionen ---
    html += `<div id="es-deps" class="${showEs ? '' : 'u-hidden'}">
        <div class="config-fields">
            ${SharedUi.forms.configField({type: 'text', label: 'Elasticsearch URL', key: 'retrieval.elasticsearch_url', value: r.elasticsearch_url || 'http://localhost:9200', required: true, hint: 'URL des Elasticsearch-Servers. Standard: http://localhost:9200'})}
            ${SharedUi.forms.configField({type: 'text', label: 'Index-Name', key: 'retrieval.elasticsearch_index', value: r.elasticsearch_index || 'kiara', required: true, hint: 'Name des Elasticsearch-Index fuer die Keyword-Suche. Standard: kiara'})}
            <div class="config-field"><label>Fuzziness (unscharfe Suche)</label>
                <select data-key="retrieval.elasticsearch_fuzziness" class="config-input">
                    <option value="none"${(r.elasticsearch_fuzziness || 'none') === 'none' ? ' selected' : ''}>Keine (exakt)</option>
                    <option value="AUTO"${r.elasticsearch_fuzziness === 'AUTO' ? ' selected' : ''}>AUTO</option>
                    <option value="1"${r.elasticsearch_fuzziness === '1' ? ' selected' : ''}>1 (max. 1 Zeichen Abweichung)</option>
                    <option value="2"${r.elasticsearch_fuzziness === '2' ? ' selected' : ''}>2 (max. 2 Zeichen Abweichung)</option>
                </select>
                <span class="field-hint">Steuert, ob Elasticsearch auch aehnlich geschriebene Woerter findet (Levenshtein-Distanz). Kann Schreibfehler in Suchanfragen und Dokumenten abfangen (z.B. "Hetznr" findet "Hetzner"). <b>Achtung:</b> Bei kurzen Suchbegriffen und Zahlen kann Fuzziness zu massenhaft falschen Treffern fuehren — "2025" wuerde dann auch "2024", "2026", "1025" usw. matchen. Empfehlung: <b>Keine</b>, da die semantische Suche bereits unscharfes Matching liefert und die Keyword-Suche gezielt exakte Begriffe treffen soll.</span>
            </div>
        </div>
        <div id="es-index-panel" class="u-panel-bg u-mt-12 u-pad-12">
            <h5 class="u-my-0-8">Index-Verwaltung</h5>
            <div id="es-status-info" class="u-mb-10 u-fs-09em u-c-secondary">Status wird geladen...</div>
            <div class="u-flex-gap-8 u-flex-wrap">
                <button class="action-btn primary" data-action="es-rebuild-index" id="es-btn-rebuild">Index aufbauen</button>
                <button class="action-btn" data-action="es-verify-integrity" id="es-btn-verify">Integritaet pruefen</button>
                <button class="action-btn danger" data-action="es-delete-index" id="es-btn-delete">Index loeschen</button>
            </div>
            <div id="es-task-result" class="u-mt-10 u-hidden"></div>
        </div>
    </div>`;

    // --- RRF-K (fuer beide Methoden) ---
    html += `<div id="hybrid-deps" class="${showRrf ? '' : 'u-hidden'}">
        <div class="config-fields">
            ${SharedUi.forms.configField({type: 'number', label: 'RRF-K (Reciprocal Rank Fusion)', key: 'retrieval.rrf_k', value: r.rrf_k ?? 60, min: 1, hint: 'Glaettungsparameter fuer die Rang-Fusion. Bestimmt, wie stark die Position in den Einzelrankings gewichtet wird. Niedrige Werte (z.B. 1) bevorzugen Top-Treffer stark, hohe Werte (z.B. 60) gewichten die Rankings gleichmaessiger. Standard: 60.'})}
        </div>
    </div>`;
    html += '</div></div>';

    // --- Section 2: Reranking ---
    const rerankChecked = r.reranking ? 'checked' : '';
    const rerankDepClass = r.reranking ? '' : 'u-section-disabled';

    html += '<div class="config-section">';
    html += '<h4>Reranking</h4>';
    html += '<p class="config-hint">Bewertet abgerufene Chunks mit einem Cross-Encoder-Modell neu und gibt nur die relevantesten zurueck. Im Gegensatz zur initialen Vektorsuche (Bi-Encoder) bewertet der Cross-Encoder Anfrage und Chunk gemeinsam, was praezisere Relevanzscores liefert — allerdings auf Kosten der Geschwindigkeit. Deshalb wird zuerst breit gesucht und dann nur die Top-Ergebnisse rerankt.</p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field">
        <label><input type="checkbox" id="retrieval-reranking" ${rerankChecked} data-change-action="toggle-retrieval-section" data-section="rerank"> Cross-Encoder Reranking aktivieren</label>
    </div>`;
    html += `<div id="rerank-deps" class="${rerankDepClass}">
        <div class="config-fields">
            ${SharedUi.forms.configField({type: 'text', label: 'Rerank-Modell', key: 'retrieval.rerank_model', value: r.rerank_model || 'cross-encoder/ms-marco-MiniLM-L-6-v2', hint: 'HuggingFace-Modellname des Cross-Encoders. Das Modell wird beim ersten Aufruf automatisch heruntergeladen und lokal gecacht. Standard-Modell (ms-marco-MiniLM-L-6-v2) ist kompakt (~80 MB) und fuer englische/deutsche Texte optimiert.'})}
            ${SharedUi.forms.configField({type: 'number', label: 'Rerank Top-K', key: 'retrieval.rerank_top_k', value: r.rerank_top_k ?? 5, min: 1, hint: 'Anzahl der Chunks, die nach dem Reranking an das LLM weitergegeben werden. Nur die besten Top-K Ergebnisse werden in den Kontext aufgenommen. Weniger = fokussiertere Antworten, mehr = breiterer Kontext. Standard: 5.'})}
            ${SharedUi.forms.configField({type: 'number', label: 'Fetch-Multiplikator', key: 'retrieval.rerank_fetch_multiplier', value: r.rerank_fetch_multiplier ?? 3, min: 1, hint: 'Bestimmt, wie viele Chunks initial abgerufen werden: Top-K × Multiplikator. Bei Top-K=5 und Multiplikator=3 werden 15 Chunks geholt und auf die besten 5 reduziert. Hoehere Werte erhoehen die Trefferchance, kosten aber mehr Rechenzeit. Standard: 3.'})}
        </div>
    </div>`;
    html += '</div></div>';

    // --- Section 3: Query-Expansion ---
    const qeChecked = r.query_expansion ? 'checked' : '';

    html += '<div class="config-section">';
    html += '<h4>Query-Expansion</h4>';
    html += '<p class="config-hint">Generiert alternative Suchanfragen (Deutsch + Englisch) per LLM, um die Trefferquote zu erhoehen. Das LLM formuliert die Benutzeranfrage in mehreren Varianten um — z.B. Synonyme, Fachbegriffe und eine englische Uebersetzung. Die Ergebnisse aller Varianten werden zusammengefuehrt. Erhoet die Latenz um einen LLM-Call pro Anfrage, verbessert aber den Recall insbesondere bei kurzen oder mehrdeutigen Anfragen deutlich.</p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field">
        <label><input type="checkbox" id="retrieval-query-expansion" ${qeChecked}> Query-Expansion aktivieren</label>
    </div>`;
    html += '</div></div>';

    // --- Section 4: Kontext-Limits ---
    html += '<div class="config-section">';
    html += '<h4>Kontext-Limits</h4>';
    html += '<p class="config-hint">Begrenzt die maximale Anzahl von Retrieval-Chunks (Suchergebnisse aus der VektorDB), die als Kontext an das LLM uebergeben werden. Schuetzt vor Ueberlauf des Kontextfensters — besonders wichtig im Agent-Modus, wo mehrere Suchanfragen Chunks akkumulieren koennen. Chunks sind nach Relevanz sortiert; bei Ueberschreitung werden die am wenigsten relevanten verworfen.</p>';
    html += '<div class="config-fields">';
    html += SharedUi.forms.configField({type: 'number', label: 'Maximale Kontext-Chunks', key: 'retrieval.max_context_chunks', value: r.max_context_chunks ?? 80, min: 5, max: 500, hint: 'Obergrenze fuer die Anzahl der Chunks im LLM-Kontext. Gilt fuer Standard- und Agent-Modus. Faustregel: Bei num_ctx=8192 (~28k Zeichen) passen ca. 25-30 Chunks, bei num_ctx=16384 (~57k Zeichen) ca. 50-80 Chunks. Hoehere Werte ermoeglichen mehr Kontext, riskieren aber Abschneidung durch das LLM. Standard: 80.'});
    html += '</div></div>';

    // --- Section 5: Agent-Deduplizierung ---
    const dedupChecked = (r.agent_dedup !== false) ? 'checked' : '';
    html += '<div class="config-section">';
    html += '<h4>Agent-Deduplizierung</h4>';
    html += '<p class="config-hint">Verhindert, dass der Agent im Tool-Loop dieselben Dokument-Chunks mehrfach erhaelt. Wenn der Agent bei einer Anfrage mehrmals die Wissensbasis durchsucht, werden bereits zurueckgegebene Chunks bei Folge-Aufrufen herausgefiltert. Das reduziert redundanten Kontext und verhindert, dass das LLM durch doppelte Informationen verwirrt wird. Die Deduplizierung arbeitet auf Basis von Dateiname + Chunk-Nummer und wirkt nur innerhalb einer einzelnen Chat-Anfrage.</p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field">
        <label><input type="checkbox" id="retrieval-agent-dedup" ${dedupChecked}> Deduplizierung ueber Tool-Calls hinweg aktivieren</label>
    </div>`;
    html += '</div></div>';

    // --- Section 6: Spezialisierte Optimierungen ---
    const agentCfg = (cfg.chat && cfg.chat.agent) || {};
    const rawCompactionChecked = agentCfg.raw_chunk_compaction !== false ? 'checked' : '';
    const finalRerankChecked = agentCfg.final_reranking !== false ? 'checked' : '';
    const invoiceGroupingChecked = agentCfg.invoice_number_grouping !== false ? 'checked' : '';
    html += '<div class="config-section">';
    html += '<h4>Spezialisierte Optimierungen</h4>';
    html += '<p class="config-hint">Optionale Heuristiken fuer wiederkehrende Dokumenttypen und Agent-Antworten. Diese Schalter ermoeglichen A/B-Tests: Die allgemeine Retrieval-Pipeline bleibt aktiv, einzelne Speziallogiken koennen separat ein- oder ausgeschaltet werden.</p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field">
        <label><input type="checkbox" class="config-checkbox" data-key="chat.agent.raw_chunk_compaction" id="agent-raw-chunk-compaction" ${rawCompactionChecked}> Raw-Chunks source-diverse verdichten</label>
        <span class="field-hint">Verdichtet Raw-Chunks aus mehreren Quellen vor dem finalen Agent-Prompt zu kompakten Gruppen. Ausschalten, um den Legacy-Pfad mit direkter Budget-Kuerzung der Raw-Chunks zu testen.</span>
    </div>`;
    html += `<div class="config-field">
        <label><input type="checkbox" class="config-checkbox" data-key="chat.agent.final_reranking" id="agent-final-reranking" ${finalRerankChecked}> Finales Agent-Reranking aktivieren</label>
        <span class="field-hint">Rerankt die gesammelten Agent-Suchchunks vor der finalen Antwort gegen die Originalfrage und filtert niedrige Cross-Encoder-Scores. Ausschalten, um alle gesammelten Chunks nur per Kontext-Cap zu begrenzen.</span>
    </div>`;
    html += `<div class="config-field">
        <label><input type="checkbox" class="config-checkbox" data-key="chat.agent.invoice_number_grouping" id="agent-invoice-number-grouping" ${invoiceGroupingChecked}> Rechnungsnummern als Beleg-Gruppen verwenden</label>
        <span class="field-hint">Gruppiert Agent-Raw-Chunks mit gleicher Rechnungsnummer zu einem kompakten Beleg und schuetzt diese Rechnungsgruppen im finalen Agent-Kontext. Ausschalten, um nur die allgemeine source-diverse Kontextverdichtung ohne rechnungsnummernspezifische Optimierung zu testen.</span>
    </div>`;
    html += '</div></div>';

    // --- Section 7: Parallelitaet (Worker) ---
    const w = cfg.web || {};
    const workers = w.workers ?? 4;

    html += '<div class="config-section">';
    html += '<h4>Parallelitaet</h4>';
    html += '<p class="config-hint">Steuert, wie viele gleichzeitige Anfragen der KIara-Server verarbeiten kann. Jeder Worker ist ein eigenstaendiger Serverprozess mit eigenem Speicher. Wenn z.B. 4 Worker konfiguriert sind, koennen 4 Benutzer gleichzeitig Anfragen stellen, ohne sich gegenseitig zu blockieren. Mehr Worker erhoehen die Parallelitaet, benoetigen aber mehr RAM (~200-400 MB pro Worker). <b>Aenderungen erfordern einen Neustart des Dienstes.</b></p>';
    html += '<div class="config-fields">';
    html += `<div class="config-field"><label>Anzahl Worker-Prozesse</label>
        <input type="number" id="web-workers" value="${SharedUi.runtime.escapeAttr(String(workers))}" class="config-input" min="1" max="16">
        <span class="field-hint">Anzahl der Uvicorn-Worker-Prozesse fuer die Chat-UI. Bestimmt, wie viele Chat-Sessions gleichzeitig bedient werden koennen. Bei Wert 1 blockieren sich alle Sessions gegenseitig waehrend einer laufenden Anfrage. Empfehlung: 2-4 fuer normale Nutzung, 4-8 fuer Teams. Standard: 4.</span>
    </div>`;
    html += '</div></div>';

    // --- Buttons ---
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-action="save-ki-retrieval-config">Speichern</button>';
    html += '<button class="action-btn" data-action="reset-ki-retrieval-tab">Zuruecksetzen</button>';
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;

    // ES-Status laden wenn Elasticsearch ausgewaehlt
    if (showEs) esLoadStatus();
}

function toggleRetrievalSection(section) {
    if (section === 'hybrid') {
        const mode = document.getElementById('retrieval-hybrid-search').value;
        const bm25Deps = document.getElementById('bm25-deps');
        const esDeps = document.getElementById('es-deps');
        const rrfDeps = document.getElementById('hybrid-deps');
        if (bm25Deps) {
            bm25Deps.classList.toggle('u-hidden', mode !== 'bm25');
            bm25Deps.querySelectorAll('input, select, textarea').forEach(el => {
                el.disabled = mode !== 'bm25';
            });
        }
        if (esDeps) {
            esDeps.classList.toggle('u-hidden', mode !== 'elasticsearch');
            esDeps.querySelectorAll('input, select, textarea').forEach(el => {
                el.disabled = mode !== 'elasticsearch';
            });
        }
        if (rrfDeps) {
            rrfDeps.classList.toggle('u-hidden', mode === 'none');
            rrfDeps.querySelectorAll('input, select, textarea').forEach(el => {
                el.disabled = mode === 'none';
            });
        }
        if (mode === 'elasticsearch') esLoadStatus();
        else _esStopPolling();
    } else if (section === 'rerank') {
        const checked = document.getElementById('retrieval-reranking').checked;
        const deps = document.getElementById('rerank-deps');
        if (deps) {
            deps.classList.toggle('u-section-disabled', !checked);
            // disabled-Attribut auf Eingabefelder setzen — dadurch werden Werte
            // beim Save nicht mehr gesammelt und der Browser rendert sie grau.
            deps.querySelectorAll('input, select, textarea').forEach(el => {
                el.disabled = !checked;
            });
        }
    }
}

// ---- Save (AP4) ----

async function saveKiRetrievalConfig(btn) {
    if (!kiRetrievalConfig) return;
    if (_esSaveInFlight) return;
    const mySeq = _esInitSeq;
    await SharedUi.runtime.withButtonLock(btn, async () => {
        _esSaveInFlight = true;
        try {
            const result = await collectAndSaveConfig(kiRetrievalConfig, {
                selector: '.config-input[data-key^="retrieval."], .config-checkbox[data-key^="chat.agent."]',
                successMessage: 'Retrieval-Einstellungen gespeichert',
                beforeSave: (cfg) => {
                    if (!cfg.retrieval) cfg.retrieval = {};
                    const hybridEl = document.getElementById('retrieval-hybrid-search');
                    const rerankEl = document.getElementById('retrieval-reranking');
                    const qeEl = document.getElementById('retrieval-query-expansion');
                    const dedupEl = document.getElementById('retrieval-agent-dedup');
                    if (hybridEl) cfg.retrieval.hybrid_search = hybridEl.value;
                    if (rerankEl) cfg.retrieval.reranking = rerankEl.checked;
                    if (qeEl) cfg.retrieval.query_expansion = qeEl.checked;
                    if (dedupEl) cfg.retrieval.agent_dedup = dedupEl.checked;
                    // ES-Felder nur bei Modus=elasticsearch behalten
                    if (hybridEl && hybridEl.value !== 'elasticsearch') {
                        delete cfg.retrieval.elasticsearch_url;
                        delete cfg.retrieval.elasticsearch_index;
                        delete cfg.retrieval.elasticsearch_fuzziness;
                    }
                    // RRF-K nur bei aktivem Hybrid-Modus persistieren
                    if (hybridEl && hybridEl.value === 'none') {
                        delete cfg.retrieval.rrf_k;
                    }
                    // Worker-Anzahl
                    const workersEl = document.getElementById('web-workers');
                    if (workersEl) {
                        const rawWorkers = workersEl.value.trim();
                        const workers = Number(rawWorkers);
                        if (rawWorkers === '' || !Number.isInteger(workers) || workers < 1 || workers > 16) {
                            SharedUi.notify.show('Worker-Anzahl muss eine ganze Zahl zwischen 1 und 16 sein', 'error');
                            workersEl.focus();
                            return false;
                        }
                        if (!cfg.web) cfg.web = {};
                        cfg.web.workers = workers;
                    }
                },
            });
            // Stale-Guard: Re-Init waehrend Save
            if (mySeq !== _esInitSeq) return;
            if (result) kiRetrievalConfig = result;
        } finally {
            _esSaveInFlight = false;
        }
    });
}

// ---- Elasticsearch Status-Poll (AP2) ----

async function esLoadStatus() {
    const info = document.getElementById('es-status-info');
    if (!info) return;
    const mySeq = _esInitSeq;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/elasticsearch/status', {credentials: 'same-origin'});
        // Latest-only guards nach await
        if (mySeq !== _esInitSeq) return;
        if (_esGetCurrentMode() !== 'elasticsearch') return;

        // resp.ok-Check mit tolerantem Error-Body
        if (!resp.ok) {
            let errMsg = 'HTTP ' + resp.status;
            try { const d = await resp.json(); errMsg = d.detail || d.error || d.message || errMsg; } catch(_) {}
            if (mySeq !== _esInitSeq) return;
            info.innerHTML = '<span class="u-c-danger">' + SharedUi.runtime.escapeHtml(errMsg) + '</span>';
            return;
        }

        const data = await resp.json();
        if (mySeq !== _esInitSeq) return;

        // available/connected-Normalisierung (Test-Drift)
        const isAvailable = data.available ?? data.connected;
        if (!isAvailable) {
            info.innerHTML = '<span class="u-c-danger">Elasticsearch nicht erreichbar</span>';
            return;
        }

        // Status-HTML zusammenbauen
        let statusHtml = '';
        if (data.index_exists) {
            const pct = data.qdrant_count > 0 ? (data.doc_count / data.qdrant_count * 100).toFixed(1) : '0.0';
            const colorCls = data.doc_count === data.qdrant_count ? 'u-c-success' : 'u-c-warning';
            statusHtml = 'Index <b>' + SharedUi.runtime.escapeHtml(data.index_name || 'kiara') + '</b>: ';
            statusHtml += '<span class="' + colorCls + '">' + (data.doc_count ?? 0).toLocaleString('de') + ' / ' + (data.qdrant_count ?? 0).toLocaleString('de') + ' Chunks (' + pct + '%)</span>';
            statusHtml += ' &mdash; Health: <b>' + SharedUi.runtime.escapeHtml(data.health) + '</b>';
        } else {
            statusHtml = 'Index <b>' + SharedUi.runtime.escapeHtml(data.index_name || 'kiara') + '</b>: nicht vorhanden.';
            if (!data.task_running && !data.error) {
                statusHtml += ' Klicken Sie "Index aufbauen" um den Index zu erstellen.';
            }
        }

        // available:true + error!=null sichtbar machen
        if (data.error) {
            statusHtml += ' &mdash; <span class="u-c-warning">Fehler: ' + SharedUi.runtime.escapeHtml(data.error) + '</span>';
        }

        if (data.task_running) {
            statusHtml += ' &mdash; <span class="u-c-primary">Aufgabe laeuft: ' + SharedUi.runtime.escapeHtml(data.task_action) + '</span>';
            if (data.task_progress && data.task_progress.total > 0) {
                const tPct = (data.task_progress.indexed / data.task_progress.total * 100).toFixed(0);
                statusHtml += ' (' + tPct + '%)';
            }
        }

        info.innerHTML = statusHtml;

        // Task-State: Resume oder Guard-Freigabe
        if (data.task_running) {
            _esSetButtons(false);
            // Genau EINEN Resume-Pfad, kein zweiter Poll-Arm
            if (!_esTaskPollActive) _esResumeTaskPoll();
        } else if (!_esTaskPollActive) {
            // Kein Task, kein Poll — Buttons frei, Guard frei
            _esSetButtons(true);
            _esStartGuard = false;
            // Einmaliger Recovery-Handover fuer offscreen-abgeschlossene Tasks
            _esCheckForOffscreenResult();
        }
        // Wenn _esTaskPollActive: Taskpoll laeuft, nicht eingreifen (F48)

    } catch(e) {
        if (mySeq !== _esInitSeq) return;
        info.innerHTML = '<span class="u-c-danger">Status konnte nicht geladen werden</span>';
    }
}

// ---- Task-Poll (AP3) ----

function _esResumeTaskPoll() {
    if (_esTaskPollActive) return;
    _esStartGuard = true;
    _esWaitForTask();
}

async function _esWaitForTask() {
    // Neue Generation — invalidiert alle aelteren Poll-Arme
    _esTaskGen++;
    const myGen = _esTaskGen;
    const mySeq = _esInitSeq;
    _esTaskPollActive = true;
    _esSetButtons(false);
    _esStartPolling();

    const resultDiv = document.getElementById('es-task-result');
    if (resultDiv) { resultDiv.classList.add('u-hidden'); resultDiv.innerHTML = ''; }

    let pollErrors = 0;
    const poll = async () => {
        // Generations-/Subtab-/Mode-Check VOR fetch
        if (myGen !== _esTaskGen || mySeq !== _esInitSeq) { _esTaskPollActive = false; return; }
        if (_esGetCurrentMode() !== 'elasticsearch') { _esTaskPollActive = false; return; }

        try {
            const resp = await SharedUi.fetch.safeFetch('/api/admin/elasticsearch/task', {credentials: 'same-origin'});

            // Generations-Check NACH await
            if (myGen !== _esTaskGen || mySeq !== _esInitSeq) { _esTaskPollActive = false; return; }

            if (!resp.ok) {
                let errMsg = 'HTTP ' + resp.status;
                try { const d = await resp.json(); errMsg = d.detail || d.error || d.message || errMsg; } catch(_) {}
                throw new Error(errMsg);
            }

            const data = await resp.json();
            if (myGen !== _esTaskGen || mySeq !== _esInitSeq) { _esTaskPollActive = false; return; }

            pollErrors = 0;

            if (!data.running && data.result) {
                // Finaler Handover — Result uebernehmen
                _esStopPolling();
                _esStartGuard = false;
                _esSetButtons(true);
                const fp = JSON.stringify(data.result);
                if (_esLastShownResult !== fp) {
                    _esLastShownResult = fp;
                    _esShowResult(data.result);
                }
                // Status sauber nachziehen (kein zweiter Poll-Arm)
                esLoadStatus();
                return;
            } else if (!data.running) {
                // Backend-State idle ohne Result (Service-Restart waehrend Polling) — fail-soft beenden
                _esStopPolling();
                _esStartGuard = false;
                _esSetButtons(true);
                esLoadStatus();
                return;
            }
        } catch(e) {
            console.error('ES poll error:', e);
            if (myGen !== _esTaskGen) { _esTaskPollActive = false; return; }
            pollErrors++;
            if (pollErrors >= 3) {
                _esStopPolling();
                _esStartGuard = false;
                _esSetButtons(true);
                // Korrekter Selektor: #es-status-info (F15)
                const info = document.getElementById('es-status-info');
                if (info) info.textContent = 'Polling fehlgeschlagen \u2014 bitte manuell aktualisieren';
                return;
            }
        }
        setTimeout(poll, 1500);
    };
    setTimeout(poll, 1000);
}

async function _esCheckForOffscreenResult() {
    const mySeq = _esInitSeq;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/elasticsearch/task', {credentials: 'same-origin'});
        if (mySeq !== _esInitSeq) return;
        if (!resp.ok) return;
        const data = await resp.json();
        if (mySeq !== _esInitSeq) return;
        if (!data.running && data.result) {
            const fp = JSON.stringify(data.result);
            if (_esLastShownResult !== fp) {
                _esLastShownResult = fp;
                _esShowResult(data.result);
            }
        }
    } catch(e) { /* ignore */ }
}

function _esShowResult(result) {
    const div = document.getElementById('es-task-result');
    if (!div) return;
    div.classList.remove('u-hidden');
    div.classList.add('u-block');

    if (result.status === 'error') {
        div.innerHTML = '<div class="u-result-error u-pad-8">Fehler: ' + SharedUi.runtime.escapeHtml(result.error || 'Unbekannt') + '</div>';
        return;
    }

    let html = '<div class="u-result-ok u-pad-8">';
    if (result.indexed !== undefined) {
        html += 'Rebuild abgeschlossen: <b>' + result.indexed.toLocaleString('de') + '</b> / ' + result.total.toLocaleString('de') + ' Chunks indexiert';
        if (result.errors > 0) html += ', <span class="u-c-danger">' + result.errors + ' Fehler</span>';
    } else if (result.integrity_pct !== undefined) {
        const colorCls = result.integrity_pct >= 99.9 ? 'u-c-success' : 'u-c-warning';
        html += 'Integritaet: <b class="' + colorCls + '">' + result.integrity_pct + '%</b>';
        html += ' (' + result.matching + '/' + result.sampled + ' Stichproben identisch)';
        if (result.missing_in_es > 0) html += ', <b>' + result.missing_in_es + '</b> in ES fehlend';
        if (result.mismatches && result.mismatches.length > 0) {
            html += '<br>Abweichungen: ' + result.mismatches.map(m => SharedUi.runtime.escapeHtml(m.source)).join(', ');
        }
    }
    html += '</div>';
    div.innerHTML = html;
}

// ---- User-Aktionen (AP4) ----

async function esRebuildIndex() {
    // Synchroner Start-Guard VOR dem POST (F14)
    if (_esStartGuard) {
        SharedUi.notify.show('Ein Rebuild/Verify laeuft bereits', 'warning');
        return;
    }
    if (!await SharedUi.modal.confirm({ message: 'Index komplett neu aufbauen? Der bestehende Index wird geloescht und alle Chunks aus der Vektordatenbank kopiert.' })) return;

    _esStartGuard = true;
    _esSetButtons(false);
    _esLastShownResult = null;
    const mySeq = _esInitSeq;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/elasticsearch/rebuild', {method: 'POST', credentials: 'same-origin'});
        if (mySeq !== _esInitSeq) return;

        if (!resp.ok) {
            let errMsg = 'HTTP ' + resp.status;
            try {
                const d = await resp.json();
                errMsg = d.detail || d.error || d.message || errMsg;
                // Konflikt: Task laeuft bereits -> Resume statt Unlock (F67)
                if ((d.error || d.message || '').includes('Ein ES-Task laeuft bereits')) {
                    _esResumeTaskPoll();
                    return;
                }
            } catch(_) {}
            SharedUi.notify.show('Rebuild fehlgeschlagen: ' + errMsg, 'error');
            _esStartGuard = false;
            _esSetButtons(true);
            return;
        }

        _esWaitForTask();
    } catch(e) {
        if (mySeq !== _esInitSeq) return;
        // Konflikt: ES-Task laeuft bereits -> Resume statt Unlock (F67)
        // safeFetch wirft bei !ok einen FetchError; Konflikt-Detection muss hier passieren.
        const _bodyMsg = (e && e.body && (e.body.error || e.body.message || e.body.detail)) || '';
        const _errMsg = _bodyMsg || (e && e.message) || '';
        if (_errMsg.includes('Ein ES-Task läuft bereits') || _errMsg.includes('Ein ES-Task laeuft bereits')) {
            _esResumeTaskPoll();
            return;
        }
        SharedUi.notify.show('Rebuild konnte nicht gestartet werden: ' + e.message, 'error');
        _esStartGuard = false;
        _esSetButtons(true);
    }
}

async function esVerifyIntegrity() {
    // Synchroner Start-Guard VOR dem POST (F14)
    if (_esStartGuard) {
        SharedUi.notify.show('Ein Rebuild/Verify laeuft bereits', 'warning');
        return;
    }

    _esStartGuard = true;
    _esSetButtons(false);
    _esLastShownResult = null;
    const mySeq = _esInitSeq;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/elasticsearch/verify', {method: 'POST', credentials: 'same-origin'});
        if (mySeq !== _esInitSeq) return;

        if (!resp.ok) {
            let errMsg = 'HTTP ' + resp.status;
            try {
                const d = await resp.json();
                errMsg = d.detail || d.error || d.message || errMsg;
                // Konflikt: Task laeuft bereits -> Resume statt Unlock (F67)
                if ((d.error || d.message || '').includes('Ein ES-Task laeuft bereits')) {
                    _esResumeTaskPoll();
                    return;
                }
            } catch(_) {}
            SharedUi.notify.show('Integritaetspruefung fehlgeschlagen: ' + errMsg, 'error');
            _esStartGuard = false;
            _esSetButtons(true);
            return;
        }

        _esWaitForTask();
    } catch(e) {
        if (mySeq !== _esInitSeq) return;
        // Konflikt: ES-Task laeuft bereits -> Resume statt Unlock (F67)
        // safeFetch wirft bei !ok einen FetchError; Konflikt-Detection muss hier passieren.
        const _bodyMsg = (e && e.body && (e.body.error || e.body.message || e.body.detail)) || '';
        const _errMsg = _bodyMsg || (e && e.message) || '';
        if (_errMsg.includes('Ein ES-Task läuft bereits') || _errMsg.includes('Ein ES-Task laeuft bereits')) {
            _esResumeTaskPoll();
            return;
        }
        SharedUi.notify.show('Integritaetspruefung konnte nicht gestartet werden: ' + e.message, 'error');
        _esStartGuard = false;
        _esSetButtons(true);
    }
}

async function esDeleteIndex(btn) {
    if (!await SharedUi.modal.confirm({ message: 'Elasticsearch-Index wirklich loeschen? Alle indexierten Chunks gehen verloren.' })) return;
    await SharedUi.runtime.withButtonLock(btn || document.getElementById('es-btn-delete'), async () => {
        try {
            const resp = await SharedUi.fetch.safeFetch('/api/admin/elasticsearch/index', {method: 'DELETE', credentials: 'same-origin'});
            if (!resp.ok) {
                let errMsg = 'HTTP ' + resp.status;
                try { const d = await resp.json(); errMsg = d.detail || d.error || d.message || errMsg; } catch(_) {}
                SharedUi.notify.show('Loeschen fehlgeschlagen: ' + errMsg, 'error');
                return;
            }
            const data = await resp.json();
            SharedUi.notify.show(data.message || 'Index geloescht', 'success');
            esLoadStatus();
        } catch(e) {
            SharedUi.notify.show('Loeschen fehlgeschlagen: ' + e.message, 'error');
        }
    });
}
