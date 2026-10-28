/**
 * KIara - Chat Tab
 */

let chatUI = null;
const _selectedDsIds = new Set();
let _chatInitGen = 0;
let _chatTabActionsBound = false;
let _chatSourceProfiles = {};
let _chatSourceMode = null;
const _chatSourceChoices = new Map();

const _CHAT_TAB_ACTIONS = new Set(['chat-new']);

function _handleChatTabClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_CHAT_TAB_ACTIONS.has(el.dataset.action)) return;
    if (el.dataset.action === 'chat-new' && chatUI) chatUI.newChat();
}

function _bindChatTabActions() {
    if (_chatTabActionsBound) return;
    _chatTabActionsBound = true;
    document.addEventListener('click', _handleChatTabClick);
}

function _handleChatModeChange(event) {
    const modeSelect = event.currentTarget;
    const selected = modeSelect ? modeSelect.options[modeSelect.selectedIndex] : null;
    const chainType = selected ? (selected.dataset.chainType || 'standard') : 'standard';
    _updateFilterVisibility(chainType);
    _updateThinkToggle(chainType);
    _applyModeDefaults(chainType);
}

function _bindChatModeSelect(modeSelect) {
    if (!modeSelect) return;
    modeSelect.removeEventListener('change', _handleChatModeChange);
    modeSelect.addEventListener('change', _handleChatModeChange);
}

async function initChatTab() {
    _bindChatTabActions();
    const gen = ++_chatInitGen;
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    // Build filter bar
    if (filterBar) {
        filterBar.innerHTML = '';
        filterBar.classList.remove('u-hidden');
        filterBar.classList.add('u-flex');

        // Chat mode selector (dynamisch populiert)
        const modeGroup = document.createElement('div');
        modeGroup.className = 'filter-group';
        modeGroup.innerHTML = `<span class="filter-label">Modus:</span>
            <select class="filter-select" id="chatFilterMode"></select>`;
        filterBar.appendChild(modeGroup);

        // LLM backend selector (alle Modi)
        const llmGroup = document.createElement('div');
        llmGroup.className = 'filter-group';
        llmGroup.innerHTML = `<span class="filter-label">LLM:</span>
            <select class="filter-select" id="chatFilterLlm"><option value="">Auto</option></select>`;
        filterBar.appendChild(llmGroup);

        // Embedding model filter
        const embGroup = document.createElement('div');
        embGroup.className = 'filter-group';
        embGroup.dataset.ragFilter = 'true';
        embGroup.innerHTML = `<span class="filter-label">Embedding:</span>
            <select class="filter-select" id="chatFilterModel"><option value="">Standard</option></select>`;
        filterBar.appendChild(embGroup);

        // Datasource filter (Multi-Select mit Chips)
        const dsGroup = document.createElement('div');
        dsGroup.className = 'filter-group';
        dsGroup.dataset.ragFilter = 'true';
        dsGroup.innerHTML = `<span class="filter-label">Datenquelle:</span>
            <div class="ds-filter-wrap">
                <select class="filter-select" id="chatFilterDs"><option value="" id="chatFilterDsAll">Alle</option></select>
                <span class="ds-not-label">NOT</span>
                <label class="toggle-switch"><input type="checkbox" id="chatFilterDsNot" disabled><span class="toggle-slider"></span></label>
                <button class="ds-clear-btn" id="chatDsClearAll" disabled title="Alle entfernen">&times;</button>
                <div class="ds-chip-container" id="chatDsChips"></div>
            </div>`;
        filterBar.appendChild(dsGroup);

        // File type filter
        const typeGroup = document.createElement('div');
        typeGroup.className = 'filter-group';
        typeGroup.dataset.ragFilter = 'true';
        typeGroup.innerHTML = `<span class="filter-label">Dateityp:</span>
            <select class="filter-select" id="chatFilterType"><option value="">Alle</option></select>`;
        filterBar.appendChild(typeGroup);

        // Grenzen und Standards kommen ausschliesslich aus KI > Chat.
        const kGroup = document.createElement('div');
        kGroup.className = 'filter-group';
        kGroup.dataset.ragFilter = 'true';
        kGroup.innerHTML = `<span class="filter-label">Quellen:</span>
            <label id="chatSourceAutoGroup" class="u-hidden"><input type="checkbox" id="chatSourceAuto"> Auto</label>
            <input type="range" id="chatFilterKSlider" class="filter-slider" disabled>
            <input type="number" id="chatFilterK" min="1" step="1" class="filter-number" disabled>`;
        filterBar.appendChild(kGroup);
        const kSlider = kGroup.querySelector('#chatFilterKSlider');
        const kInput = kGroup.querySelector('#chatFilterK');
        const autoInput = kGroup.querySelector('#chatSourceAuto');
        kSlider.addEventListener('input', () => {
            kInput.value = kSlider.value;
            _rememberChatSourceChoice();
        });
        kInput.addEventListener('change', () => {
            const value = Number(kInput.value);
            if (kInput.value !== '' && Number.isInteger(value) && value >= Number(kInput.min) && value <= Number(kInput.max)) {
                kSlider.value = value;
                _rememberChatSourceChoice();
            }
        });
        autoInput.addEventListener('change', () => {
            kInput.disabled = autoInput.checked;
            kSlider.disabled = autoInput.checked;
            _rememberChatSourceChoice();
        });

        // Context window (num_ctx) – Slider 1K-32K in 1K-Schritten, alle Modi
        const ctxGroup = document.createElement('div');
        ctxGroup.className = 'filter-group';
        ctxGroup.innerHTML = `<span class="filter-label">Kontext:</span>
            <input type="range" id="chatFilterCtxSlider" min="1" max="32" value="8" class="filter-slider">
            <input type="number" id="chatFilterCtx" min="1" max="32" value="8" class="filter-number">
            <span class="filter-unit">K</span>`;
        filterBar.appendChild(ctxGroup);

        // Sync slider <-> textbox
        const ctxSlider = ctxGroup.querySelector('#chatFilterCtxSlider');
        const ctxInput = ctxGroup.querySelector('#chatFilterCtx');
        ctxSlider.addEventListener('input', () => { ctxInput.value = ctxSlider.value; });
        ctxInput.addEventListener('input', () => {
            let v = parseInt(ctxInput.value) || 1;
            const maxK = parseInt(ctxSlider.max) || 32;
            if (v < 1) v = 1;
            if (v > maxK) v = maxK;
            ctxInput.value = v;
            ctxSlider.value = v;
        });

        // History toggle (Gespraechskontext)
        const histGroup = document.createElement('div');
        histGroup.className = 'filter-group';
        histGroup.innerHTML = `<span class="filter-label">Kontext:</span>
            <label class="toggle-switch">
                <input type="checkbox" id="chatFilterHistory">
                <span class="toggle-slider"></span>
            </label>`;
        filterBar.appendChild(histGroup);

        // Thinking toggle (qwen3 <think>-Protokoll)
        const thinkGroup = document.createElement('div');
        thinkGroup.className = 'filter-group';
        thinkGroup.id = 'chatThinkGroup';
        thinkGroup.innerHTML = `<span class="filter-label">Thinking:</span>
            <label class="toggle-switch">
                <input type="checkbox" id="chatFilterThink">
                <span class="toggle-slider"></span>
            </label>`;
        filterBar.appendChild(thinkGroup);

        // Max Suchen slider (nur Agent-Modus)
        const repeatsGroup = document.createElement('div');
        repeatsGroup.className = 'filter-group u-hidden';
        repeatsGroup.dataset.agentFilter = 'true';
        repeatsGroup.innerHTML = `
            <span class="filter-label">Max Suchen:</span>
            <input type="range" id="chatFilterMaxRepeatsSlider"
                   min="0" max="20" value="5" step="1"
                   class="filter-slider u-width-80">
            <input type="number" id="chatFilterMaxRepeats"
                   min="0" max="20" value="5"
                   class="filter-number chat-filter-number-50">
            <small class="u-c-secondary">0 = unbegrenzt</small>`;
        filterBar.appendChild(repeatsGroup);

        // Slider <-> Number synchronisieren
        const repeatsSlider = document.getElementById('chatFilterMaxRepeatsSlider');
        const repeatsInput = document.getElementById('chatFilterMaxRepeats');
        if (repeatsSlider && repeatsInput) {
            repeatsSlider.addEventListener('input', () => {
                repeatsInput.value = repeatsSlider.value;
            });
            repeatsInput.addEventListener('input', () => {
                repeatsSlider.value = repeatsInput.value;
            });
        }

        // Temperature slider (nur Unchained-Modus)
        const tempGroup = document.createElement('div');
        tempGroup.className = 'filter-group u-hidden';
        tempGroup.dataset.unchainedFilter = 'true';
        tempGroup.innerHTML = `<span class="filter-label">Temp:</span>
            <input type="range" id="chatFilterTempSlider" min="0" max="100" value="70" class="filter-slider">
            <input type="number" id="chatFilterTemp" min="0" max="1" step="0.05" value="0.70" class="filter-number">`;
        filterBar.appendChild(tempGroup);

        // Sync slider (0-100) <-> textbox (0.00-1.00)
        const tempSlider = tempGroup.querySelector('#chatFilterTempSlider');
        const tempInput = tempGroup.querySelector('#chatFilterTemp');
        tempSlider.addEventListener('input', () => {
            tempInput.value = (parseInt(tempSlider.value) / 100).toFixed(2);
        });
        tempInput.addEventListener('input', () => {
            let v = parseFloat(tempInput.value);
            if (isNaN(v) || v < 0) v = 0;
            if (v > 1) v = 1;
            tempInput.value = v.toFixed(2);
            tempSlider.value = Math.round(v * 100);
        });

        // New chat button
        const newChatGroup = document.createElement('div');
        newChatGroup.className = 'filter-group u-ml-auto';
        newChatGroup.innerHTML = `<button class="action-btn primary" data-action="chat-new">Neuer Chat</button>`;
        filterBar.appendChild(newChatGroup);

        // Mode-Selector Event: RAG-Filter ein/ausblenden + Kompatibilitaet.
        const modeSelect = document.getElementById('chatFilterMode');
        _bindChatModeSelect(modeSelect);

        // Modes dynamisch laden, dann Filter + Multi-Select init
        await _loadChatModes();
        // Race-Schutz: Tab-Wechsel zu Search/History invalidiert _chatInitGen NICHT,
        // daher zusaetzlich currentTab pruefen (sonst ueberschreibt chatUI.render()
        // unten den Inhalt des jetzt aktiven Tabs).
        if (gen !== _chatInitGen || currentTab !== 'chat') return;
        // Initial-State basierend auf initial gewaehltem Modus
        const initialChainType = _getSelectedChainType();
        _updateThinkToggle(initialChainType);
        _applyModeDefaults(initialChainType);
        await loadChatFilters(gen);
        if (gen !== _chatInitGen || currentTab !== 'chat') return;
        _initDsMultiSelect();
        _syncFilterBarHeight();
    }

    // Initialize chat UI
    if (!chatUI) chatUI = new ChatUI('contentContainer');
    chatUI.render([]);
    void _initTraceDebug();

    // Pending-Load-Intent anwenden (z.B. aus tab_history.resumeChat vor initChatTab).
    if (typeof _applyPendingChatLoad === 'function') {
        _applyPendingChatLoad();
    }

    // Search-Kontext erkennen (aus Such-Tab uebernommen)
    if (window._searchContext && window._searchContext.length > 0) {
        const chunks = window._searchContext;
        const autoPrompt = window._searchAutoPrompt;
        window._searchContext = null;
        window._searchAutoPrompt = null;
        await startChatWithContext(chunks, autoPrompt);
    }
}

async function startChatWithContext(chunks, autoPrompt = null) {
    const contextData = chunks.map(c => ({
        id: c.id, text: c.text, source: c.source || '', score: c.score || 0,
    }));
    let chatId;
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
            body: JSON.stringify({ title: 'Suche', context_chunks: contextData }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        ({ id: chatId } = await resp.json());
    } catch (err) {
        console.error('Chat-Erstellung fehlgeschlagen:', err);
        SharedUi.notify.show('Fehler beim Erstellen des Chats. Bitte erneut versuchen.', 'error');
        return;
    }

    chatUI.currentChatId = chatId;
    chatUI.render([]);
    chatUI.renderContextMessages(chunks);
    chatUI._contextChunks = contextData;

    if (autoPrompt) {
        const input = document.getElementById('chatInput');
        if (input) {
            input.value = autoPrompt;
            await chatUI.sendMessage();
        }
    }
}

async function _loadChatModes() {
    const modeSelect = document.getElementById('chatFilterMode');
    if (!modeSelect) return;

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/chat/modes');
        if (!resp.ok) throw new Error('Modes laden fehlgeschlagen');
        const data = await resp.json();
        const modes = data.modes || [];
        _chatSourceProfiles = data.source_profiles || {};

        if (modes.length > 0) {
            modeSelect.innerHTML = '';
            modes.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.label || m.name;
                opt.dataset.chainType = m.chain_type;
                opt.dataset.name = m.name;
                modeSelect.appendChild(opt);
            });
            return;
        }
    } catch (e) {
        // Fallback
    }

    // Fallback: hardcoded Optionen
    modeSelect.innerHTML = `
        <option value="0" data-chain-type="standard" data-name="system_standard">Standard</option>
        <option value="0" data-chain-type="rewriter" data-name="system_rewriter">Hoch</option>
        <option value="0" data-chain-type="unchained" data-name="system_unchained">Unchained</option>
        <option value="0" data-chain-type="agent" data-name="system_agent">Agent</option>`;
}

function _getSelectedChainType() {
    const modeSelect = document.getElementById('chatFilterMode');
    if (!modeSelect) return 'standard';
    const selected = modeSelect.options[modeSelect.selectedIndex];
    return selected ? (selected.dataset.chainType || 'standard') : 'standard';
}

let _fbObserver = null;

function _syncFilterBarHeight() {
    const fb = document.getElementById('filterBar');
    if (!fb) return;
    const writeVar = () => {
        SharedUi.runtimeStyle.setRule(':root', {
            '--filter-bar-height': fb.offsetHeight + 'px',
        });
    };
    requestAnimationFrame(writeVar);
    // ResizeObserver einmalig registrieren (reagiert auf Wrap bei Fenster-Resize)
    if (!_fbObserver && typeof ResizeObserver !== 'undefined') {
        _fbObserver = new ResizeObserver(writeVar);
        _fbObserver.observe(fb);
    }
}

function _updateFilterVisibility(chainType) {
    const ragFilters = document.querySelectorAll('[data-rag-filter]');
    const hidden = (chainType === 'unchained');
    ragFilters.forEach(el => { el.classList.toggle('u-hidden', hidden); });

    // Temperature-Slider nur bei Unchained anzeigen
    const unchainedFilters = document.querySelectorAll('[data-unchained-filter]');
    unchainedFilters.forEach(el => { el.classList.toggle('u-hidden', !hidden); });

    // Agent-Only Filter (Max Suchen)
    const agentFilters = document.querySelectorAll('[data-agent-filter]');
    const isAgent = (chainType === 'agent');
    agentFilters.forEach(el => { el.classList.toggle('u-hidden', !isAgent); });

    _syncFilterBarHeight();
}

function _updateThinkToggle(chainType) {
    const thinkGroup = document.getElementById('chatThinkGroup');
    const thinkToggle = document.getElementById('chatFilterThink');
    if (!thinkGroup || !thinkToggle) return;
    // Recherche verwendet budgetierte Leser-/Pruefaufrufe ohne Thinking.
    thinkToggle.disabled = chainType === 'research';
    thinkGroup.classList.toggle('u-dim', chainType === 'research');
}

function _rememberChatSourceChoice() {
    if (!_chatSourceMode || !_chatSourceProfiles[_chatSourceMode]) return;
    const input = document.getElementById('chatFilterK');
    const auto = document.getElementById('chatSourceAuto');
    const k = Number(input.value);
    if (!Number.isInteger(k) || k < 1 || k > Number(input.max)) return;
    const choice = { k, auto: auto.checked };
    _chatSourceChoices.set(_chatSourceMode, choice);
    try { sessionStorage.setItem('kiara-chat-sources:' + _chatSourceMode, JSON.stringify(choice)); } catch (_) {}
}

function _applyChatSourceProfile(mode) {
    _chatSourceMode = mode;
    const profile = _chatSourceProfiles[mode];
    const input = document.getElementById('chatFilterK');
    const slider = document.getElementById('chatFilterKSlider');
    const auto = document.getElementById('chatSourceAuto');
    const group = document.getElementById('chatSourceAutoGroup');
    if (!input || !slider || !auto || !group) return;
    group.classList.toggle('u-hidden', mode !== 'agent' || !profile?.auto_enabled);
    if (!profile) {
        input.disabled = slider.disabled = true;
        auto.checked = false;
        return;
    }
    let saved = _chatSourceChoices.get(mode);
    if (!saved) {
        try { saved = JSON.parse(sessionStorage.getItem('kiara-chat-sources:' + mode) || 'null'); } catch (_) {}
    }
    const k = saved && Number.isInteger(saved.k) && saved.k >= 1 && saved.k <= profile.max_k
        ? saved.k : (profile.default_k || profile.auto_default_k);
    const useAuto = mode === 'agent' && profile.auto_enabled
        && (saved ? saved.auto === true : profile.default_k === 0);
    input.min = slider.min = 1;
    input.max = slider.max = profile.max_k;
    input.value = slider.value = k;
    auto.checked = useAuto;
    input.disabled = slider.disabled = useAuto;
}

function _getChatSourceK(mode) {
    if (mode === 'unchained') return null;
    const profile = _chatSourceProfiles[mode];
    if (!profile) throw new Error('Quelleneinstellungen konnten nicht geladen werden. Bitte den Chat neu laden.');
    if (document.getElementById('chatSourceAuto')?.checked) {
        if (mode !== 'agent' || !profile.auto_enabled) throw new Error('Automatische Quellenwahl ist nicht freigegeben.');
        return 0;
    }
    const input = document.getElementById('chatFilterK');
    const k = Number(input?.value);
    if (!input?.value || !Number.isInteger(k) || k < 1 || k > profile.max_k) {
        throw new Error('Quellen muss eine ganze Zahl zwischen 1 und ' + profile.max_k + ' sein.');
    }
    return k;
}

// --- Agent-Modus Defaults fuer Denken und Kontext ---
let _preAgentSettings = null;

function _applyModeDefaults(chainType) {
    const thinkToggle = document.getElementById('chatFilterThink');
    _applyChatSourceProfile(chainType);
    const ctxSlider = document.getElementById('chatFilterCtxSlider');
    const ctxInput = document.getElementById('chatFilterCtx');

    if (chainType === 'agent') {
        // Aktuelle Werte sichern (nur wenn noch nicht im Agent-Modus)
        if (!_preAgentSettings) {
            _preAgentSettings = {
                think: thinkToggle ? thinkToggle.checked : false,
                ctx: ctxInput ? parseInt(ctxInput.value) || 8 : 8,
            };
        }
        // Agent-Defaults setzen
        if (thinkToggle) thinkToggle.checked = true;
        if (ctxSlider) ctxSlider.value = 16;
        if (ctxInput) ctxInput.value = 16;
    } else if (_preAgentSettings) {
        // Vorherige Werte wiederherstellen
        if (thinkToggle) thinkToggle.checked = _preAgentSettings.think;
        if (ctxSlider) ctxSlider.value = _preAgentSettings.ctx;
        if (ctxInput) ctxInput.value = _preAgentSettings.ctx;
        _preAgentSettings = null;
    }
    if (chainType === 'research' && thinkToggle) thinkToggle.checked = false;
}



async function loadChatFilters(gen = null) {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/chat/filters');
        if (!resp.ok) return;
        const data = await resp.json();
        // Race-Schutz: Tab-Wechsel/Re-Init invalidiert diesen Lauf vor DOM-Mutationen.
        if (gen !== null && (gen !== _chatInitGen || currentTab !== 'chat')) return;

        // LLM backends (mit num_ctx und provider als data-Attribute)
        const llmSelect = document.getElementById('chatFilterLlm');
        if (llmSelect && data.llm_backends) {
            data.llm_backends.forEach(b => {
                const opt = document.createElement('option');
                opt.value = b.id;
                opt.textContent = `${b.name} (${b.model})`;
                if (b.num_ctx) opt.dataset.numCtx = b.num_ctx;
                opt.dataset.provider = b.provider || 'ollama';
                llmSelect.appendChild(opt);
            });
            // Change-Event: Slider-Maximum + Modus-Kompatibilitaet
            llmSelect.addEventListener('change', () => {
                const selected = llmSelect.options[llmSelect.selectedIndex];
                const numCtx = selected && selected.dataset.numCtx ? parseInt(selected.dataset.numCtx) : null;
                const ctxSlider = document.getElementById('chatFilterCtxSlider');
                const ctxInput = document.getElementById('chatFilterCtx');
                if (ctxSlider && ctxInput) {
                    const maxK = numCtx ? Math.ceil(numCtx / 1024) : 32;
                    ctxSlider.max = maxK;
                    ctxInput.max = maxK;
                    if (parseInt(ctxSlider.value) > maxK) {
                        ctxSlider.value = maxK;
                        ctxInput.value = maxK;
                    }
                }
            });
        }

        // Embedding models (V13: Objekte mit has_backend-Flag)
        const modelSelect = document.getElementById('chatFilterModel');
        if (modelSelect && data.embedding_models) {
            if (data.embedding_models.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'Kein Modell verfuegbar';
                opt.disabled = true;
                opt.selected = true;
                modelSelect.appendChild(opt);
                const sendBtn = document.getElementById('chatSendBtn');
                if (sendBtn) {
                    sendBtn.disabled = true;
                    sendBtn.title = 'Kein Embedding-Modell verfuegbar — bitte Backend konfigurieren';
                }
            } else {
                let defaultModel = data.default_embedding_model || '';
                const defaultHasBackend = data.embedding_models.find(
                    m => m.model === defaultModel && m.has_backend
                );
                if (!defaultHasBackend) {
                    const firstAvailable = data.embedding_models.find(m => m.has_backend);
                    if (firstAvailable) defaultModel = firstAvailable.model;
                }

                data.embedding_models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.model;
                    if (m.has_backend) {
                        opt.textContent = m.model;
                    } else {
                        opt.textContent = `${m.model} (kein Backend)`;
                        opt.classList.add('chat-model-option-disabled');
                        opt.disabled = true;
                    }
                    if (m.model === defaultModel) opt.selected = true;
                    modelSelect.appendChild(opt);
                });

                // Alle Modelle ohne Backend — Sende-Button sperren
                if (!data.embedding_models.some(m => m.has_backend)) {
                    const sendBtn = document.getElementById('chatSendBtn');
                    if (sendBtn) {
                        sendBtn.disabled = true;
                        sendBtn.title = 'Kein Embedding-Backend verfuegbar — bitte Backend konfigurieren';
                    }
                }
            }
        }

        const dsSelect = document.getElementById('chatFilterDs');
        if (dsSelect && data.datasources) {
            window._chatDsList = data.datasources;
            data.datasources.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds.id;
                opt.textContent = ds.name;
                dsSelect.appendChild(opt);
            });
        }

        const typeSelect = document.getElementById('chatFilterType');
        if (typeSelect && data.file_types) {
            data.file_types.forEach(ft => {
                const opt = document.createElement('option');
                opt.value = ft;
                opt.textContent = ft;
                typeSelect.appendChild(opt);
            });
        }

        _renderDsChips();
    } catch (e) {
        // Filter nicht verfuegbar
    }
}

function _initDsMultiSelect() {
    const dsSelect = document.getElementById('chatFilterDs');
    const clearBtn = document.getElementById('chatDsClearAll');
    const notToggle = document.getElementById('chatFilterDsNot');
    const notLabel = notToggle ? notToggle.closest('.ds-filter-wrap').querySelector('.ds-not-label') : null;
    const chipContainer = document.getElementById('chatDsChips');
    if (!dsSelect) return;

    dsSelect.addEventListener('change', () => {
        const val = dsSelect.value;
        if (!val) return;
        _selectedDsIds.add(parseInt(val));
        dsSelect.value = '';
        _renderDsChips();
    });

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            _selectedDsIds.clear();
            if (notToggle) notToggle.checked = false;
            if (notLabel) notLabel.classList.remove('active');
            _renderDsChips();
        });
    }

    if (notToggle && notLabel) {
        notToggle.addEventListener('change', () => {
            notLabel.classList.toggle('active', notToggle.checked);
        });
    }

    if (chipContainer) {
        chipContainer.addEventListener('click', (e) => {
            const btn = e.target.closest('.ds-chip-remove');
            if (!btn || !chipContainer.contains(btn)) return;
            _selectedDsIds.delete(parseInt(btn.dataset.id));
            _renderDsChips();
        });
    }
}

function _renderDsChips() {
    const chipContainer = document.getElementById('chatDsChips');
    const dsSelect = document.getElementById('chatFilterDs');
    const allOption = document.getElementById('chatFilterDsAll');
    const clearBtn = document.getElementById('chatDsClearAll');
    const notToggle = document.getElementById('chatFilterDsNot');
    const notLabel = notToggle ? notToggle.closest('.ds-filter-wrap').querySelector('.ds-not-label') : null;
    if (!chipContainer || !dsSelect) return;

    const hasChips = _selectedDsIds.size > 0;
    const dsList = window._chatDsList || [];

    // Chips rendern
    chipContainer.innerHTML = '';
    _selectedDsIds.forEach(id => {
        const ds = dsList.find(d => d.id === id);
        const name = ds ? ds.name : `#${id}`;
        const chip = document.createElement('span');
        chip.className = 'ds-chip';
        chip.innerHTML = `${SharedUi.runtime.escapeHtml(name)}<button class="ds-chip-remove" data-id="${SharedUi.runtime.escapeAttr(String(id))}">&times;</button>`;
        chipContainer.appendChild(chip);
    });

    // Chip-Remove-Events: Delegation am chipContainer (in _initDsMultiSelect)

    // Dropdown-Optionen: ausgewaehlte ausblenden
    Array.from(dsSelect.options).forEach(opt => {
        if (!opt.value) return; // Platzhalter
        opt.classList.toggle('u-hidden', _selectedDsIds.has(parseInt(opt.value)));
    });

    // Platzhalter-Text + "Alle"-Option Sichtbarkeit
    if (allOption) {
        allOption.textContent = hasChips ? 'Auswahl' : 'Alle';
        allOption.classList.toggle('u-hidden', hasChips);
    }
    dsSelect.value = '';

    // NOT-Toggle + Clear-Button
    if (notToggle) {
        notToggle.disabled = !hasChips;
        if (!hasChips) {
            notToggle.checked = false;
            if (notLabel) notLabel.classList.remove('active');
        }
    }
    if (clearBtn) clearBtn.disabled = !hasChips;
}
