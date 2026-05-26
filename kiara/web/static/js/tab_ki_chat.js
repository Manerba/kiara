/**
 * KIara Admin - KI > Chat Sub-Tab
 * Prompt-Template-Management mit ExpandableTable
 */

let ptTable = null;

// Init-Generation analog tab_ki_ingestion / tab_ldap: Re-Init-Safety bei
// schnellem Tab-Switch. Cleanup beim Tab-Wechsel zaehlt _kiChatInitGen hoch,
// sodass Late-Responses (onRowExpanded-Hydration) und alte ExpandableTable-
// Instanzen erkennen koennen, dass sie bereits veraltet sind.
let _kiChatInitGen = 0;

// Row-lokale In-Flight-Sets fuer Toggle-Mutationen (V30 Risiken: spaete
// ptTable.refresh()-Antworten duerfen juengere Checkbox-Interaktionen weder
// doppelt committen noch wieder zurueckrollen).
const _toggleTemplateChatInFlight = new Set();
const _toggleTemplateActiveInFlight = new Set();

const _KI_CHAT_ACTIONS = new Set([
    'filter-templates-by-chain-type',
    'open-create-template-modal',
    'save-template',
    'delete-template',
    'test-enrichment-prompt',
    'create-template',
    'rollback-template',
]);

const _KI_CHAT_CHANGE_ACTIONS = new Set([
    'toggle-template-chat',
    'toggle-template-active',
    'create-chain-type-change',
]);

const _KI_CHAT_INPUT_ACTIONS = new Set([
    'update-pt-search',
]);

function _handleKiChatClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;
    if (!_KI_CHAT_ACTIONS.has(action)) return;

    switch (action) {
        case 'filter-templates-by-chain-type':
            filterTemplatesByChainType(el, el.dataset.chainType || '');
            break;
        case 'open-create-template-modal':
            openCreateTemplateModal();
            break;
        case 'save-template':
            SharedUi.runtime.withButtonLock(el, () => saveTemplate(Number(el.dataset.templateId)));
            break;
        case 'delete-template':
            SharedUi.runtime.withButtonLock(el, () => deleteTemplate(Number(el.dataset.templateId)));
            break;
        case 'test-enrichment-prompt':
            SharedUi.runtime.withButtonLock(el, () => testEnrichmentPrompt(Number(el.dataset.templateId)));
            break;
        case 'create-template':
            SharedUi.runtime.withButtonLock(el, createTemplate);
            break;
        case 'rollback-template':
            SharedUi.runtime.withButtonLock(el, () => rollbackTemplate(
                Number(el.dataset.templateId),
                Number(el.dataset.historyId)
            ));
            break;
    }
}

function _handleKiChatChange(e) {
    const el = e.target.closest('[data-change-action]');
    if (!el) return;
    const action = el.dataset.changeAction;
    if (!_KI_CHAT_CHANGE_ACTIONS.has(action)) return;

    switch (action) {
        case 'toggle-template-chat':
            toggleTemplateChat(Number(el.dataset.templateId), el.checked);
            break;
        case 'toggle-template-active':
            toggleTemplateActive(Number(el.dataset.templateId), el.checked);
            break;
        case 'create-chain-type-change':
            onCreateChainTypeChange(el.value);
            break;
    }
}

function _handleKiChatInput(e) {
    const el = e.target.closest('[data-input-action]');
    if (!el) return;
    const action = el.dataset.inputAction;
    if (!_KI_CHAT_INPUT_ACTIONS.has(action)) return;

    if (action === 'update-pt-search' && ptTable) {
        ptTable.setFilter('search', el.value);
    }
}

function _bindKiChatActions() {
    // Einziger Delegations-Root: document.body (Lehre aus 0.0.1+376).
    // Deckt contentContainer, filterBar und Modal-Overlay ab.
    if (document.body._kiChatActionsBound) return;
    document.body._kiChatActionsBound = true;
    document.body.addEventListener('click', _handleKiChatClick);
    document.body.addEventListener('change', _handleKiChatChange);
    document.body.addEventListener('input', _handleKiChatInput);

    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-ki-chat-actions-bound', 'true');
}

async function initKiChatTab() {
    // Re-Init-Sicherheit: alte Cleanups ausfuehren (invalidiert aeltere
    // Generationen, sodass Late-Responses einer vorherigen Tabellen-Instanz
    // den frisch initialisierten DOM nicht mehr veraendern).
    if (typeof _runTabCleanups === 'function') _runTabCleanups('ki-chat');
    const myGen = ++_kiChatInitGen;

    _bindKiChatActions();
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Cleanup-Hook: naechster Tab-Wechsel zaehlt die Generation hoch und
    // markiert damit alle in-flight onRowExpanded-Hydrations als veraltet.
    if (typeof registerTabCleanup === 'function') {
        registerTabCleanup('ki-chat', () => { _kiChatInitGen++; });
    }

    // Filter bar: Suche + Neu-Button
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.add('u-flex');

        const searchGroup = document.createElement('div');
        searchGroup.className = 'filter-group';
        searchGroup.innerHTML = `<span class="filter-label">Suche:</span>
            <input type="text" class="filter-input" id="ptSearch" placeholder="Name/Label..."
                data-input-action="update-pt-search">`;
        filterBar.appendChild(searchGroup);

        const typeGroup = document.createElement('div');
        typeGroup.className = 'filter-group u-ml-12';
        typeGroup.innerHTML = `<span class="filter-label">Filter:</span>
            <div class="u-flex-gap-4">
                <button class="action-btn small active" id="ptFilterAll" data-action="filter-templates-by-chain-type" data-chain-type="">Alle</button>
                <button class="action-btn small" id="ptFilterEnrichment" data-action="filter-templates-by-chain-type" data-chain-type="enrichment">Enrichment</button>
            </div>`;
        filterBar.appendChild(typeGroup);

        const addGroup = document.createElement('div');
        addGroup.className = 'filter-group u-ml-auto';
        addGroup.innerHTML = `<button class="action-btn primary" data-action="open-create-template-modal">+ Neues Template</button>`;
        filterBar.appendChild(addGroup);
    }

    // Tabellen-Container
    container.innerHTML = '<div id="ptTableContainer"></div>';

    const tableConfig = {
        id: 'promptTemplates',
        apiEndpoint: '/api/admin/ki/prompt-templates',
        gridTemplate: '24px 1fr 100px 90px 70px 60px',
        showHeader: true,
        expandable: true,
        defaultSort: { field: 'sort_order', direction: 'asc' },
        columns: [
            { field: 'label', label: 'Name / Label', sortable: true, align: 'left',
              renderer: (v, row) => {
                  const label = SharedUi.runtime.escapeHtml(v || row.name);
                  const name = SharedUi.runtime.escapeHtml(row.name);
                  const sysHtml = row.is_system ? ' <span class="badge badge-info u-fs-10">System</span>' : '';
                  return `<strong>${label}</strong>${sysHtml}<br><span class="u-text-xxs-secondary">${name}</span>`;
              }
            },
            { field: 'chain_type', label: 'Typ', sortable: true, align: 'left',
              renderer: (v) => {
                  const labels = { standard: 'Standard', unchained: 'Unchained', agent: 'Agent', enrichment: 'Enrichment', admin_agent: 'Admin', rewriter: 'Rewriter' };
                  const colors = { standard: '#3b82f6', unchained: '#f59e0b', agent: '#8b5cf6', enrichment: '#10b981', admin_agent: '#6b7280', rewriter: '#10b981' };
                  const color = colors[v] || '#888';
                  const dynCls = SharedUi.runtimeStyle.classForProps({ background: color, color: '#fff', fontSize: '11px' });
                  return `<span class="badge ${dynCls}">${labels[v] || SharedUi.runtime.escapeHtml(v)}</span>`;
              }
            },
            { field: 'show_in_chat', label: 'Chat', sortable: false, align: 'center',
              renderer: (v, row) => {
                  if (row.is_cloud) return '<span class="u-text-xxs-secondary">intern</span>';
                  const nonChatTypes = ['enrichment', 'admin_agent'];
                  if (nonChatTypes.includes(row.chain_type)) return '<span class="u-text-xxs-secondary">nicht verf\u00fcgbar</span>';
                  const checked = v ? 'checked' : '';
                  // Label-onclick stopPropagation entfaellt — ExpandableTable
                  // nutzt _interactiveCellMatch('.toggle-switch, input, ...')
                  // und laesst Row-Expand bei Toggle-Clicks bereits durchgehen.
                  return `<label class="toggle-switch">
                      <input type="checkbox" ${checked} data-change-action="toggle-template-chat" data-template-id="${row.id}">
                      <span class="toggle-slider"></span></label>`;
              }
            },
            { field: 'is_active', label: 'Aktiv', sortable: false, align: 'center',
              renderer: (v, row) => {
                  const checked = v ? 'checked' : '';
                  return `<label class="toggle-switch">
                      <input type="checkbox" ${checked} data-change-action="toggle-template-active" data-template-id="${row.id}">
                      <span class="toggle-slider"></span></label>`;
              }
            },
        ],
        filters: [
            { id: 'search', field: 'name', type: 'text', wildcard: true, wildcardFields: ['label', 'description'] },
        ],
        detailRenderer: (row) => renderTemplateDetail(row),
    };

    ptTable = new SharedUi.table.ExpandableTable(tableConfig, 'ptTableContainer', {
        onRowExpanded: async (rowId, detailEl) => {
            // Textarea auto-resize
            const ta = detailEl.querySelector('.pt-content-textarea');
            if (ta) {
                SharedUi.runtimeStyle.applyStyle(ta, { height: 'auto' });
                SharedUi.runtimeStyle.applyStyle(ta, { height: Math.max(120, ta.scrollHeight) + 'px' });
            }
            const row = ptTable.getAllData().find(r => String(r.id) === String(rowId));
            // CSP Detail-Load: Listen-Mapper liefert cloud_system_prompt nicht
            const cspEl = document.getElementById(`ptCloudSystemPrompt_${rowId}`);
            if (cspEl) {
                cspEl.dataset.cspLoaded = '0';
                try {
                    const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${rowId}`);
                    // Generation-Check nach await: bricht Late-Hydration ab,
                    // wenn der Tab inzwischen re-initialisiert wurde.
                    if (myGen !== _kiChatInitGen) return;
                    if (resp.ok) {
                        const detail = await resp.json();
                        if (myGen !== _kiChatInitGen) return;
                        cspEl.value = detail.cloud_system_prompt || '';
                        cspEl.placeholder = 'z.B. Du bist ein hilfreicher Unternehmens-Assistent...';
                        cspEl.dataset.cspLoaded = '1';
                        cspEl.disabled = false;
                    } else {
                        cspEl.placeholder = '(Laden fehlgeschlagen)';
                    }
                } catch (e) {
                    if (myGen !== _kiChatInitGen) return;
                    cspEl.placeholder = '(Laden fehlgeschlagen)';
                }
                // Auto-resize nach Wertbefuellung
                SharedUi.runtimeStyle.applyStyle(cspEl, { height: 'auto' });
                SharedUi.runtimeStyle.applyStyle(cspEl, { height: Math.max(80, cspEl.scrollHeight) + 'px' });
            }
            // Enrichment: load backends dropdown + version history
            if (row && row.chain_type === 'enrichment') {
                loadEnrichmentBackends(rowId);
                loadTemplateHistory(rowId);
            }
        }
    });
    ptTable.loadData();
}

function renderTemplateDetail(row) {
    const placeholderHints = {
        standard: '<code>{context}</code> und <code>{question}</code> sind Pflicht.',
        unchained: 'Keine Pflicht-Platzhalter.',
        agent: '<code>{tools_prompt}</code> ist Pflicht.',
        enrichment: '<code>{chunk_text}</code> ist Pflicht. Verfuegbar: <code>{document_context}</code>, <code>{file_type}</code>, <code>{source}</code>, <code>{datasource_name}</code>.',
    };
    const hint = placeholderHints[row.chain_type] || '';

    let html = '<div class="u-pad-v-12">';

    // Template-Inhalt (System-Prompt)
    html += `<div class="u-mb-12">
        <label class="u-bold u-block u-mb-4">System-Prompt${row.chain_type === 'agent' ? ' (Tool-Planung)' : ''}</label>
        <p class="u-text-xs-secondary u-my-0-6">Platzhalter: ${hint}</p>
        <textarea class="config-input pt-content-textarea u-textarea-mono" id="ptContent_${row.id}"
            rows="8">${SharedUi.runtime.escapeHtml(row.content)}</textarea>
    </div>`;

    // Cloud-System-Prompt: nur bei standard/agent (content hat Platzhalter, kann nicht als system= dienen)
    // Bei unchained/is_cloud wird content direkt als Cloud-System-Prompt verwendet.
    // Wert wird per Detail-Load nachgeladen, weil der Listen-Mapper cloud_system_prompt nicht liefert.
    if (!row.is_cloud && (row.chain_type === 'standard' || row.chain_type === 'agent')) {
        html += `<div class="u-mb-12">
            <label class="u-bold u-block u-mb-4">Cloud-System-Prompt
                <span class="u-text-xs-secondary u-fw-400">(Anthropic / OpenAI)</span></label>
            <p class="u-text-xs-secondary u-my-0-6">
                Separater System-Prompt fuer Cloud-AI-Backends. Wird als <code>system</code>-Parameter an die API gesendet.
                Der obige System-Prompt (mit Platzhaltern) wird als User-Message gesendet.</p>
            <textarea class="config-input pt-content-textarea u-textarea-mono" id="ptCloudSystemPrompt_${row.id}"
                rows="4"
                placeholder="Laden..." disabled></textarea>
        </div>`;
    }

    // Final-Prompt (nur Agent-Templates)
    if (row.chain_type === 'agent') {
        html += `<div class="u-mb-12">
            <label class="u-bold u-block u-mb-4">Antwort-Prompt (finale Antwortgenerierung)</label>
            <p class="u-text-xs-secondary u-my-0-6">
                Instruktion fuer die finale Antwort, nachdem der Agent seine Tool-Ergebnisse gesammelt hat.
                Kontext, History und Frage werden automatisch angefuegt. Leer = Standard-Prompt.</p>
            <textarea class="config-input pt-content-textarea u-textarea-mono" id="ptFinalPrompt_${row.id}"
                rows="4"
                placeholder="Leer lassen fuer Standard-Prompt...">${SharedUi.runtime.escapeHtml(row.final_prompt || '')}</textarea>
        </div>`;
    }

    // Zusatzfelder
    html += '<div class="u-grid-3col-12 u-mb-12">';

    html += `<div>
        <label class="u-bold u-block u-mb-4">Beschreibung</label>
        <input type="text" class="config-input u-w-full u-box-border" id="ptDesc_${row.id}" value="${SharedUi.runtime.escapeAttr(row.description)}">
    </div>`;

    html += `<div>
        <label class="u-bold u-block u-mb-4">Badge-Farbe</label>
        <input type="text" class="config-input u-w-full u-box-border" id="ptBadge_${row.id}" value="${SharedUi.runtime.escapeAttr(row.badge_color)}"
            placeholder="#hex oder leer">
    </div>`;

    html += `<div>
        <label class="u-bold u-block u-mb-4">Sortierung</label>
        <input type="number" class="config-input u-w-full u-box-border" id="ptSort_${row.id}" value="${row.sort_order}"
            min="0" max="999">
    </div>`;

    html += '</div>';

    // Label (editierbar)
    html += `<div class="u-mb-12">
        <label class="u-bold u-block u-mb-4">Anzeige-Label</label>
        <input type="text" class="config-input u-width-300 u-box-border" id="ptLabel_${row.id}" value="${SharedUi.runtime.escapeAttr(row.label)}">
    </div>`;

    // Aktionen
    html += '<div class="u-flex-gap-8 u-mt-8">';
    html += `<button class="action-btn primary" data-action="save-template" data-template-id="${row.id}">Speichern</button>`;

    if (!row.is_system) {
        html += `<button class="action-btn danger" data-action="delete-template" data-template-id="${row.id}">Loeschen</button>`;
    }

    html += '</div>';

    // Enrichment-only sections: Prompt-Test + Versionshistorie
    if (row.chain_type === 'enrichment') {
        // Section A: Prompt-Test
        html += `<div class="u-section-divider">
            <h4 class="u-m-0-mb-8">Prompt testen</h4>
            <div class="form-group u-mb-8">
                <label class="u-bold u-block u-mb-4">Backend</label>
                <select class="config-input u-width-300 u-box-border" id="testBackendId_${row.id}">
                    <option value="">(Standard-LLM)</option>
                </select>
            </div>
            <div class="form-group u-mb-8">
                <label class="u-bold u-block u-mb-4">Beispieltext</label>
                <textarea class="config-input u-textarea-mono" id="testSampleText_${row.id}" rows="4"
                    placeholder="Chunk-Text zum Testen eingeben...">Dies ist ein Beispiel-Chunk fuer den Enrichment-Test.</textarea>
            </div>
            <button class="action-btn primary" data-action="test-enrichment-prompt" data-template-id="${row.id}">Test ausfuehren</button>
            <div id="testResult_${row.id}" class="u-mt-8"></div>
        </div>`;

        // Section B: Versionshistorie
        html += `<div class="u-section-divider">
            <h4 class="u-m-0-mb-8">Versionen</h4>
            <div id="historyList_${row.id}" class="u-text-sm-secondary">Laden...</div>
        </div>`;
    }

    html += '</div>';
    return html;
}


async function saveTemplate(id) {
    const content = document.getElementById(`ptContent_${id}`)?.value || '';
    const description = document.getElementById(`ptDesc_${id}`)?.value || '';
    const badge_color = document.getElementById(`ptBadge_${id}`)?.value || '';
    const sort_order = parseInt(document.getElementById(`ptSort_${id}`)?.value) || 0;
    const label = document.getElementById(`ptLabel_${id}`)?.value || '';
    const finalPromptEl = document.getElementById(`ptFinalPrompt_${id}`);
    const cspEl = document.getElementById(`ptCloudSystemPrompt_${id}`);
    const payload = { content, description, badge_color, sort_order, label };
    // cloud_system_prompt nur senden, wenn der Detail-Load abgeschlossen ist.
    if (cspEl) {
        if (cspEl.dataset.cspLoaded !== '1') {
            SharedUi.notify.show('Cloud-System-Prompt ist noch nicht geladen. Speichern abgebrochen.', 'error');
            return;
        }
        payload.cloud_system_prompt = cspEl.value || '';
    }
    if (finalPromptEl) payload.final_prompt = finalPromptEl.value || '';

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Speichern fehlgeschlagen');
        }
        SharedUi.notify.show('Template gespeichert', 'success');
        ptTable.refresh();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function deleteTemplate(id) {
    if (!await SharedUi.modal.confirm({ message: 'Template endgueltig loeschen?' })) return;
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${id}`, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Loeschen fehlgeschlagen');
        }
        SharedUi.notify.show('Template geloescht', 'success');
        ptTable.refresh();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

async function toggleTemplateChat(id, checked) {
    // Row-lokaler Lock: Doppel-/Race-Klicks auf dieselbe Row blocken,
    // sonst koennte ein spaetes ptTable.silentRefresh() juengere Checkbox-
    // Interaktionen doppelt committen oder zuruecksetzen (V30).
    if (_toggleTemplateChatInFlight.has(id)) return;
    _toggleTemplateChatInFlight.add(id);
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ show_in_chat: checked }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Aenderung fehlgeschlagen');
        }
        // silentRefresh statt refresh: Diff-basiertes Update via silentRender,
        // kein Loading-State, kein Flicker der gesamten Tabelle.
        ptTable.silentRefresh();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
        ptTable.silentRefresh();
    } finally {
        _toggleTemplateChatInFlight.delete(id);
    }
}

async function toggleTemplateActive(id, checked) {
    if (_toggleTemplateActiveInFlight.has(id)) return;
    _toggleTemplateActiveInFlight.add(id);
    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_active: checked }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Aenderung fehlgeschlagen');
        }
        ptTable.silentRefresh();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
        ptTable.silentRefresh();
    } finally {
        _toggleTemplateActiveInFlight.delete(id);
    }
}

function openCreateTemplateModal() {
    SharedUi.modal.open({
        title: 'Neues Prompt-Template',
        size: 'medium',
        body: `
            <div class="config-form">
                <div class="config-fields u-grid-2col-12">
                    <div class="config-field">
                        <label>Interner Name</label>
                        <input type="text" class="config-input" id="newPtName" placeholder="z.B. it_helpdesk">
                    </div>
                    <div class="config-field">
                        <label>Anzeige-Label</label>
                        <input type="text" class="config-input" id="newPtLabel" placeholder="z.B. IT-Helpdesk">
                    </div>
                    <div class="config-field">
                        <label>Chain-Typ</label>
                        <select class="config-input" id="newPtChainType" data-change-action="create-chain-type-change">
                            <option value="standard">Standard (RAG)</option>
                            <option value="unchained">Unchained (freies LLM)</option>
                            <option value="agent">Agent (Tool-Use)</option>
                            <option value="enrichment">Enrichment</option>
                        </select>
                    </div>
                    <div class="config-field">
                        <label>Beschreibung</label>
                        <input type="text" class="config-input" id="newPtDesc" placeholder="Kurzbeschreibung...">
                    </div>
                </div>
                <div class="config-field u-mt-12">
                    <label>Template-Inhalt</label>
                    <textarea class="config-input u-textarea-mono" id="newPtContent" rows="8"
                        placeholder="Prompt-Text hier eingeben..."></textarea>
                </div>
                <div class="config-field u-mt-12">
                    <label>Cloud-System-Prompt <span class="u-text-xs-secondary">(optional, fuer Anthropic/OpenAI)</span></label>
                    <textarea class="config-input u-textarea-mono" id="newPtCloudSystemPrompt" rows="3"
                        placeholder="Separater System-Prompt fuer Cloud-Backends..."></textarea>
                </div>
                <div class="config-field u-mt-8">
                    <label class="toggle-switch u-inline-flex-center-gap-8">
                        <input type="checkbox" id="newPtShowInChat" checked>
                        <span class="toggle-slider"></span>
                        <span>Als Chat-Modus anzeigen</span>
                    </label>
                </div>
            </div>
        `,
        footer: `
            <button class="action-btn" data-sharedui-close>Abbrechen</button>
            <button class="action-btn primary" data-action="create-template">Erstellen</button>
        `,
    });
}

async function createTemplate() {
    const name = document.getElementById('newPtName')?.value?.trim();
    const label = document.getElementById('newPtLabel')?.value?.trim();
    const chain_type = document.getElementById('newPtChainType')?.value;
    const description = document.getElementById('newPtDesc')?.value?.trim() || '';
    const content = document.getElementById('newPtContent')?.value || '';
    const show_in_chat = document.getElementById('newPtShowInChat')?.checked || false;
    const cloud_system_prompt = document.getElementById('newPtCloudSystemPrompt')?.value || '';

    if (!name) { SharedUi.notify.show('Name ist erforderlich', 'error'); return; }
    if (!content) { SharedUi.notify.show('Template-Inhalt ist erforderlich', 'error'); return; }

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/prompt-templates', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, label: label || name, description, content, chain_type, show_in_chat, cloud_system_prompt }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Erstellen fehlgeschlagen');
        }
        SharedUi.modal.close();
        SharedUi.notify.show('Template erstellt', 'success');
        ptTable.refresh();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

function filterTemplatesByChainType(buttonEl, value) {
    // CSS-Klasse auf Buttons umschalten
    const parent = buttonEl.closest('.filter-group');
    if (parent) {
        parent.querySelectorAll('.action-btn').forEach(b => b.classList.remove('active'));
    }
    buttonEl.classList.add('active');
    // Tabelle mit chain_type-Filter neu laden
    if (ptTable) ptTable.loadData(value ? { chain_type: value } : {});
}

function onCreateChainTypeChange(value) {
    const cb = document.getElementById('newPtShowInChat');
    if (!cb) return;
    const nonChatTypes = ['enrichment', 'admin_agent'];
    if (nonChatTypes.includes(value)) {
        cb.checked = false;
        cb.disabled = true;
    } else {
        cb.disabled = false;
    }
}

// --- Enrichment: Backend-Dropdown laden ---
async function loadEnrichmentBackends(templateId) {
    const select = document.getElementById(`testBackendId_${templateId}`);
    if (!select) return;
    select.innerHTML = '';
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = '(Standard-LLM)';
    select.appendChild(defaultOpt);
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/backends/');
        if (!resp.ok) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '(Backends konnten nicht geladen werden)';
            opt.disabled = true;
            select.appendChild(opt);
            return;
        }
        const data = await resp.json();
        const items = data.backends || (Array.isArray(data) ? data : []);
        let added = 0;
        items.forEach(b => {
            if ((b.category === 'llm' || b.category === 'cloud') && b.is_active) {
                const opt = document.createElement('option');
                opt.value = b.id;
                opt.textContent = b.name || b.label || `Backend #${b.id}`;
                select.appendChild(opt);
                added++;
            }
        });
        if (added === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '(Keine aktiven Backends verfuegbar)';
            opt.disabled = true;
            select.appendChild(opt);
        }
    } catch (e) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(Laden fehlgeschlagen)';
        opt.disabled = true;
        select.appendChild(opt);
    }
}

// --- Enrichment: Prompt-Test ausfuehren ---
async function testEnrichmentPrompt(templateId) {
    const sampleText = document.getElementById(`testSampleText_${templateId}`)?.value || '';
    const backendId = document.getElementById(`testBackendId_${templateId}`)?.value || '';
    const resultDiv = document.getElementById(`testResult_${templateId}`);
    if (!resultDiv) return;

    if (!sampleText.trim()) {
        SharedUi.notify.show('Bitte Beispieltext eingeben', 'error');
        return;
    }

    resultDiv.innerHTML = '<span class="u-c-secondary">Test laeuft...</span>';

    const payload = { template_id: templateId, sample_text: sampleText };
    if (backendId) {
        const parsed = parseInt(backendId, 10);
        if (Number.isFinite(parsed)) payload.backend_id = parsed;
    }

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/ki/enrichment/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Test fehlgeschlagen');
        }
        const data = await resp.json();
        let html = '<div class="u-test-result-panel">';
        if (data.duration_ms !== undefined) {
            html += `<div class="u-mb-6 u-c-secondary">Dauer: ${data.duration_ms} ms</div>`;
        }
        if (data.prefix !== undefined && data.prefix !== null) {
            html += `<div class="u-mb-6"><strong>Prefix:</strong></div>
                <pre class="u-code-block u-max-h-200 u-mb-8">${SharedUi.runtime.escapeHtml(data.prefix)}</pre>`;
        }
        if (data.enriched_text !== undefined) {
            html += `<div class="u-mb-6"><strong>Enriched Text:</strong></div>
                <pre class="u-code-block u-max-h-300 u-m-0">${SharedUi.runtime.escapeHtml(data.enriched_text)}</pre>`;
        }
        html += '</div>';
        resultDiv.innerHTML = html;
    } catch (e) {
        resultDiv.innerHTML = `<div class="u-c-danger u-mt-4">${SharedUi.runtime.escapeHtml(e.message)}</div>`;
    }
}

// --- Enrichment: Versionshistorie laden ---
async function loadTemplateHistory(templateId) {
    const container = document.getElementById(`historyList_${templateId}`);
    if (!container) return;

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${templateId}/history`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Historie konnte nicht geladen werden');
        }
        const history = await resp.json();
        const items = Array.isArray(history) ? history : (history.data || history.items || []);

        if (items.length === 0) {
            container.innerHTML = '<span class="u-text-sm-secondary">Keine Versionshistorie vorhanden.</span>';
            return;
        }

        let html = '<table class="u-detail-table">';
        html += '<thead><tr class="u-cell-border-strong">'
            + '<th class="u-cell-left-4-8">Datum</th>'
            + '<th class="u-cell-left-4-8">Hash</th>'
            + '<th class="u-cell-right-4-8">Aktion</th>'
            + '</tr></thead><tbody>';

        items.forEach(item => {
            const date = item.created_at ? new Date(item.created_at).toLocaleString('de-DE') : '-';
            const hash = item.content_hash ? SharedUi.runtime.escapeHtml(item.content_hash.substring(0, 12)) : '-';
            html += `<tr class="u-row-border-light">
                <td class="u-pad-4-8">${date}</td>
                <td class="u-pad-4-8"><code class="u-fs-12">${hash}</code></td>
                <td class="u-cell-right-4-8">
                    <button class="action-btn small" data-action="rollback-template" data-template-id="${templateId}" data-history-id="${item.id}">Wiederherstellen</button>
                </td>
            </tr>`;
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = `<span class="u-c-danger">${SharedUi.runtime.escapeHtml(e.message)}</span>`;
    }
}

// --- Enrichment: Rollback auf eine fruehere Version ---
async function rollbackTemplate(templateId, historyId) {
    if (!await SharedUi.modal.confirm({ message: 'Template auf diese Version zuruecksetzen?' })) return;

    try {
        const resp = await SharedUi.fetch.safeFetch(`/api/admin/ki/prompt-templates/${templateId}/rollback/${historyId}`, {
            method: 'POST',
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || err.error || err.message || 'Rollback fehlgeschlagen');
        }
        SharedUi.notify.show('Template wiederhergestellt', 'success');
        ptTable.refresh();
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}
