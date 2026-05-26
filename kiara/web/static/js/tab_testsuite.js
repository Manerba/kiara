/**
 * KIara Admin - System > Selbsttest Sub-Tab
 * Pattern: Background-Thread + Status-Polling (analog tab_backup.js)
 */

let _testInitSeq = 0;

const _SELBSTTEST_ACTIONS = new Set(['start-tests']);

function _handleSelbsttestClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_SELBSTTEST_ACTIONS.has(el.dataset.action)) return;
    if (el.dataset.action === 'start-tests') startTests();
}

function _bindSelbsttestActions() {
    if (document.body._selbsttestActionsBound) return;
    document.body._selbsttestActionsBound = true;
    document.body.addEventListener('click', _handleSelbsttestClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-selbsttest-actions-bound', 'true');
}

async function initSelbsttestTab() {
    _bindSelbsttestActions();
    // Re-Init-Schutz
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-selbsttest');
    _testInitSeq++;
    const mySeq = _testInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    // Cleanup registrieren — seq-Invalidation hier
    registerTabCleanup('system-selbsttest', () => {
        _testInitSeq++;
        kiaraPolling.stop('system-selbsttest:status');
    });

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/tests/status');
        if (mySeq !== _testInitSeq) return;
        if (!resp.ok) throw new Error('Laden fehlgeschlagen');
        const data = await resp.json();
        renderSelbsttestTab(container, data, mySeq);
    } catch (e) {
        if (mySeq !== _testInitSeq) return;
        container.innerHTML = '<div class="no-data">Selbsttest-Status konnte nicht geladen werden.</div>';
    }
}

function renderSelbsttestTab(container, data, mySeq) {
    if (mySeq !== undefined && mySeq !== _testInitSeq) return;
    const isRunning = data.running || false;
    const hasResults = (data.results && data.results.length > 0) || data.error;

    let html = '<div class="config-form">';
    html += '<h3>Selbsttest</h3>';

    // Start-Button
    html += '<div class="config-section">';
    html += '<div class="u-flex-center-gap-12">';
    html += `<button class="action-btn primary" data-action="start-tests" id="btnStartTests" ${isRunning ? 'disabled' : ''}>Selbsttest starten</button>`;
    html += '<span id="testProgressInfo" class="u-text-sm-secondary">';
    if (isRunning) html += 'Tests laufen...';
    html += '</span>';
    html += '</div>';
    html += '</div>';

    // Fortschrittsbereich
    html += '<div id="testProgressSection">';
    if (isRunning) {
        html += renderProgressBar(data);
    }
    html += '</div>';

    // Ergebnis-Bereich
    html += '<div id="testResultSection">';
    if (!isRunning && hasResults) {
        html += renderResults(data);
    }
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;

    // Event-Delegation: Section-/Detail-Toggles (data-* statt Inline-JS)
    _bindTestToggles(container);

    // Reattach: bei Rueckkehr laufenden Test weiterpollen
    if (isRunning) pollTestStatus();
}

function renderProgressBar(data) {
    const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;

    let html = '<div class="config-section u-mt-8">';
    html += '<h4>Fortschritt</h4>';

    // Balken
    html += '<div class="u-testbar-track">';
    html += `<div id="testBar" class="u-testbar-fill u-pw-${pct}"></div>`;
    html += `<span class="u-testbar-label" id="testBarLabel">${data.progress}/${data.total}  (${pct}%)</span>`;
    html += '</div>';

    // Aktueller Test
    html += `<div class="u-test-current-name" id="testCurrentName">${SharedUi.runtime.escapeHtml(data.current_test || '')}</div>`;

    // Live-Zaehler
    html += '<div class="u-test-live-counters" id="testLiveCounters">';
    html += renderLiveCounters(data);
    html += '</div>';

    html += '</div>';
    return html;
}

function renderLiveCounters(data) {
    let html = '';
    html += `<span class="u-c-bullish">&#10003; ${data.passed}</span>`;
    html += `<span class="u-c-bearish">&#10007; ${data.failed}</span>`;
    if (data.errors > 0) html += `<span class="u-c-bearish">&#9888; ${data.errors}</span>`;
    html += `<span class="u-c-secondary">&#8856; ${data.skipped}</span>`;
    return html;
}

function renderResults(data) {
    let html = '';

    if (data.error) {
        html += '<div class="config-section u-mt-8">';
        html += `<div class="widget-error u-fs-14 u-pad-12 u-pre-wrap">${SharedUi.runtime.escapeHtml(data.error)}</div>`;
        html += '</div>';
    }

    if (!data.results || data.results.length === 0) return html;

    // KPI-Leiste
    html += '<div class="config-section u-mt-8">';
    html += '<div class="u-test-kpi-bar">';

    html += `<span class="u-c-bullish u-bold">&#10003; ${data.passed} bestanden</span>`;
    if (data.failed > 0) {
        html += `<span class="u-c-bearish u-bold">&#10007; ${data.failed} fehlgeschlagen</span>`;
    }
    if (data.errors > 0) {
        html += `<span class="u-c-bearish u-bold">&#9888; ${data.errors} Fehler</span>`;
    }
    if (data.skipped > 0) {
        html += `<span class="u-c-secondary">&#8856; ${data.skipped} uebersprungen</span>`;
    }
    html += `<span class="u-c-secondary">&#9201; ${data.duration}s</span>`;

    html += '</div>';
    html += '</div>';

    // Fehlgeschlagene Tests (aufgeklappt)
    const failed = data.results.filter(r => r.status === 'failed' || r.status === 'error');
    if (failed.length > 0) {
        html += '<div class="config-section u-mt-8">';
        html += `<h4 class="u-test-section-header" data-toggle-section="failedSection">`;
        html += `<span id="failedSectionArrow" class="u-test-arrow">&#9660;</span> Fehlgeschlagene Tests (${failed.length})</h4>`;
        html += '<div id="failedSection">';
        failed.forEach((t, i) => {
            const statusLabel = t.status === 'error' ? 'ERROR' : 'FAILED';
            html += `<div class="u-test-fail-row">`;
            html += `<div class="u-test-fail-head" data-toggle-detail="failDetail${i}" data-toggle-arrow="failArrow${i}">`;
            html += `<span id="failArrow${i}" class="u-test-arrow-sm">&#9654;</span>`;
            html += `<span class="u-c-bearish u-bold u-fs-12">${statusLabel}</span>`;
            html += `<span class="u-mono u-fs-12 u-ellipsis">${SharedUi.runtime.escapeHtml(t.name)}</span>`;
            html += '</div>';
            html += `<div id="failDetail${i}" class="u-test-fail-detail u-hidden">`;
            if (t.details) {
                html += `<pre class="u-test-fail-pre">${SharedUi.runtime.escapeHtml(t.details)}</pre>`;
            } else {
                html += '<span class="u-text-xs-secondary">Keine Details verfuegbar.</span>';
            }
            html += '</div></div>';
        });
        html += '</div></div>';
    }

    // Bestandene Tests (eingeklappt)
    const passed = data.results.filter(r => r.status === 'passed');
    if (passed.length > 0) {
        html += '<div class="config-section u-mt-8">';
        html += `<h4 class="u-test-section-header" data-toggle-section="passedSection">`;
        html += `<span id="passedSectionArrow" class="u-test-arrow u-test-arrow-collapsed">&#9660;</span> Bestanden (${passed.length})</h4>`;
        html += '<div id="passedSection" class="u-hidden">';
        html += '<div class="u-test-list-cols">';
        passed.forEach(t => {
            html += `<div class="u-test-list-row u-c-bullish">&#10003; ${SharedUi.runtime.escapeHtml(t.name)}</div>`;
        });
        html += '</div></div></div>';
    }

    // Uebersprungene Tests (eingeklappt)
    const skipped = data.results.filter(r => r.status === 'skipped');
    if (skipped.length > 0) {
        html += '<div class="config-section u-mt-8">';
        html += `<h4 class="u-test-section-header" data-toggle-section="skippedSection">`;
        html += `<span id="skippedSectionArrow" class="u-test-arrow u-test-arrow-collapsed">&#9660;</span> Uebersprungen (${skipped.length})</h4>`;
        html += '<div id="skippedSection" class="u-hidden">';
        html += '<div class="u-test-list-cols">';
        skipped.forEach(t => {
            html += `<div class="u-test-list-row u-c-secondary">&#8856; ${SharedUi.runtime.escapeHtml(t.name)}</div>`;
        });
        html += '</div></div></div>';
    }

    return html;
}

function _bindTestToggles(root) {
    if (!root) return;
    root.querySelectorAll('[data-toggle-section]').forEach(el => {
        el.addEventListener('click', function() { toggleTestSection(this.dataset.toggleSection); });
    });
    root.querySelectorAll('[data-toggle-detail]').forEach(el => {
        el.addEventListener('click', function() { toggleTestDetail(this.dataset.toggleDetail, this.dataset.toggleArrow); });
    });
}

function toggleTestSection(sectionId) {
    const section = document.getElementById(sectionId);
    const arrow = document.getElementById(sectionId + 'Arrow');
    if (!section) return;
    const isHidden = section.classList.contains('u-hidden');
    section.classList.toggle('u-hidden', !isHidden);
    if (arrow) arrow.classList.toggle('u-test-arrow-collapsed', !isHidden);
}

function toggleTestDetail(detailId, arrowId) {
    const detail = document.getElementById(detailId);
    const arrow = document.getElementById(arrowId);
    if (!detail) return;
    const isHidden = detail.classList.contains('u-hidden');
    detail.classList.toggle('u-hidden', !isHidden);
    if (arrow) arrow.classList.toggle('u-test-arrow-expanded', isHidden);
}

async function startTests() {
    const btn = document.getElementById('btnStartTests');
    const info = document.getElementById('testProgressInfo');
    if (btn) btn.disabled = true;
    if (info) info.textContent = 'Tests werden gestartet...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/tests/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || data.error || data.message || 'Fehler');
        }
        if (info) info.textContent = 'Tests laufen...';

        // Fortschrittsbereich initialisieren
        const progressSection = document.getElementById('testProgressSection');
        if (progressSection) {
            progressSection.innerHTML = renderProgressBar({
                progress: 0, total: 0, passed: 0, failed: 0, errors: 0, skipped: 0, current_test: '',
            });
        }
        // Altes Ergebnis entfernen
        const resultSection = document.getElementById('testResultSection');
        if (resultSection) resultSection.innerHTML = '';

        pollTestStatus();
    } catch (e) {
        if (btn) btn.disabled = false;
        if (info) info.textContent = '';
        SharedUi.notify.show(e.message, 'error');
    }
}

function pollTestStatus() {
    const mySeq = _testInitSeq;
    kiaraPolling.start('system-selbsttest:status', async () => {
        if (mySeq !== _testInitSeq) { kiaraPolling.stop('system-selbsttest:status'); return; }
        const resp = await SharedUi.fetch.safeFetch('/api/admin/tests/status');
        if (!resp.ok) {
            kiaraPolling.stop('system-selbsttest:status');
            if (mySeq !== _testInitSeq) return;
            onTestsFinished({error: `Status-Abfrage fehlgeschlagen (HTTP ${resp.status})`});
            return;
        }
        const data = await resp.json().catch(() => null);
        if (!data) {
            kiaraPolling.stop('system-selbsttest:status');
            if (mySeq !== _testInitSeq) return;
            onTestsFinished({error: 'Status-Antwort konnte nicht geparst werden'});
            return;
        }
        if (mySeq !== _testInitSeq) { kiaraPolling.stop('system-selbsttest:status'); return; }

        if (data.running) {
            updateProgress(data);
        } else {
            kiaraPolling.stop('system-selbsttest:status');
            if (mySeq !== _testInitSeq) return;
            onTestsFinished(data);
        }
    }, 1500);
}

function updateProgress(data) {
    const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;

    const bar = document.getElementById('testBar');
    if (bar) {
        const old = Array.from(bar.classList).filter(c => c.startsWith('u-pw-'));
        old.forEach(c => bar.classList.remove(c));
        bar.classList.add('u-pw-' + pct);
    }

    const label = document.getElementById('testBarLabel');
    if (label) label.textContent = `${data.progress}/${data.total}  (${pct}%)`;

    const name = document.getElementById('testCurrentName');
    if (name) name.textContent = data.current_test || '';

    const counters = document.getElementById('testLiveCounters');
    if (counters) counters.innerHTML = renderLiveCounters(data);
}

function onTestsFinished(data) {
    const btn = document.getElementById('btnStartTests');
    const info = document.getElementById('testProgressInfo');
    if (btn) btn.disabled = false;
    if (info) info.textContent = '';

    // Fortschrittsbalken ausblenden
    const progressSection = document.getElementById('testProgressSection');
    if (progressSection) progressSection.innerHTML = '';

    // Ergebnis rendern
    const resultSection = document.getElementById('testResultSection');
    if (resultSection) {
        resultSection.innerHTML = renderResults(data);
        _bindTestToggles(resultSection);
    }

    // Benachrichtigung
    if (data.error) {
        SharedUi.notify.show('Selbsttest fehlgeschlagen: ' + data.error, 'error');
    } else if (data.progress === 0 && data.total > 0) {
        SharedUi.notify.show('Selbsttest abgebrochen: Kein Test wurde ausgefuehrt (Collection-Fehler?)', 'error');
    } else if (data.failed > 0 || data.errors > 0) {
        SharedUi.notify.show(`Selbsttest: ${data.failed + data.errors} Test(s) fehlgeschlagen`, 'error');
    } else {
        SharedUi.notify.show(`Selbsttest: Alle ${data.passed} Tests bestanden (${data.duration}s)`, 'success');
    }
}
