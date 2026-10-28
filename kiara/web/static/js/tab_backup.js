/**
 * KIara Admin - System > Backup & Restore Sub-Tab
 */

let _backupInitSeq = 0;

const _BACKUP_ACTIONS = new Set(['create-backup']);

function _handleBackupClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_BACKUP_ACTIONS.has(el.dataset.action)) return;
    if (el.dataset.action === 'create-backup') createBackup();
}

function _bindBackupActions() {
    if (document.body._backupActionsBound) return;
    document.body._backupActionsBound = true;
    document.body.addEventListener('click', _handleBackupClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-backup-actions-bound', 'true');
}

async function initBackupTab() {
    _bindBackupActions();
    // Re-Init-Schutz
    if (typeof _runTabCleanups === 'function') _runTabCleanups('system-backup');
    _backupInitSeq++;
    const mySeq = _backupInitSeq;

    const container = document.getElementById('contentContainer');
    if (!container) return;
    const filterBar = document.getElementById('filterBar');
    if (filterBar) filterBar.classList.add('u-hidden');
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    // Cleanup registrieren — seq-Invalidation hier, nicht nur Poller-Stop
    registerTabCleanup('system-backup', () => {
        _backupInitSeq++;
        kiaraPolling.stop('system-backup:status');
    });

    try {
        const [listResp, statusResp, snapResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/backup/list').then(r => { if (!r.ok) throw new Error('Laden fehlgeschlagen'); return r.json(); }),
            SharedUi.fetch.safeFetch('/api/admin/backup/status').then(r => { if (!r.ok) throw new Error('Laden fehlgeschlagen'); return r.json(); }),
            SharedUi.fetch.safeFetch('/api/admin/config-snapshots/list')
                .then(r => { if (!r.ok) throw new Error('snapshots'); return r.json(); })
                .catch(() => ({ snapshots: null })),
        ]);
        if (mySeq !== _backupInitSeq) return;
        renderBackupTab(container, listResp, statusResp, snapResp);
    } catch (e) {
        if (mySeq !== _backupInitSeq) return;
        container.innerHTML = '<div class="no-data">Backup-Daten konnten nicht geladen werden.</div>';
    }
}

function renderBackupTab(container, listData, statusData, snapData) {
    const backups = listData.backups || [];
    const isRunning = statusData.running || false;
    const snapshots = (snapData && snapData.snapshots) || [];
    const snapshotsFailed = !snapData || snapData.snapshots === null;

    let html = '<div class="config-form">';
    html += '<h3>Backup & Restore</h3>';

    // Create section
    html += '<div class="config-section"><h4>Backup erstellen</h4>';
    html += '<div class="u-flex-center-gap-12">';
    html += '<label class="backup-incl-label u-clickable"><input type="checkbox" id="backupIncludeVectordb"> File-backed VectorDB-Daten einschliessen</label>';
    html += `<button class="action-btn primary" data-action="create-backup" id="btnCreateBackup" ${isRunning ? 'disabled' : ''}>Backup erstellen</button>`;
    html += '<span id="backupProgress" class="u-text-sm-secondary">';
    if (isRunning) html += 'Backup laeuft...';
    html += '</span>';
    html += '</div></div>';

    // Backup list
    html += '<div class="config-section"><h4>Vorhandene Backups</h4>';
    if (backups.length === 0) {
        html += '<div class="no-data">Keine Backups vorhanden.</div>';
    } else {
        html += '<table class="u-detail-table">';
        html += '<tr class="u-cell-border-strong"><th class="u-cell-left-8">Dateiname</th><th class="u-cell-right-8">Groesse</th><th class="u-cell-left-8">Erstellt</th><th class="u-cell-center-8">Aktionen</th></tr>';
        backups.forEach(b => {
            const size = (b.size_bytes / (1024 * 1024)).toFixed(1) + ' MB';
            html += `<tr class="u-cell-border">
                <td class="u-cell-left-8 u-mono u-fs-12">${SharedUi.runtime.escapeHtml(b.filename)}</td>
                <td class="u-cell-right-8">${SharedUi.runtime.escapeHtml(size)}</td>
                <td class="u-cell-left-8 u-fs-12">${SharedUi.runtime.escapeHtml(b.created_at || '-')}</td>
                <td class="u-cell-center-pad">
                    <button class="action-btn-sm" data-restore-file="${SharedUi.runtime.escapeAttr(b.filename)}">Wiederherstellen</button>
                </td>
            </tr>`;
        });
        html += '</table>';
    }
    html += '</div>';

    // Config-Snapshots (leichtgewichtig: nur kiara.yaml, mit Benchmark-Verknuepfung)
    html += '<div class="config-section"><h4>Konfigurations-Snapshots</h4>';
    html += '<div class="u-text-sm-secondary u-mb-8">Versionierte Stände von kiara.yaml. '
        + 'Benchmark-Läufe verknüpfen sich automatisch über die SHA256-Prüfsumme.</div>';
    html += '<div class="u-flex-center-gap-12 u-mb-8">';
    html += '<button class="action-btn primary" id="btnCreateConfigSnapshot">Snapshot erstellen</button>';
    html += '</div>';
    if (snapshotsFailed) {
        html += '<div class="no-data">Config-Snapshots konnten nicht geladen werden.</div>';
    } else if (snapshots.length === 0) {
        html += '<div class="no-data">Keine Config-Snapshots vorhanden.</div>';
    } else {
        html += '<table class="u-detail-table">';
        html += '<tr class="u-cell-border-strong"><th class="u-cell-left-8">Zeit (UTC)</th><th class="u-cell-left-8">Label</th><th class="u-cell-left-8">Hash</th><th class="u-cell-left-8">Trigger</th><th class="u-cell-left-8">Version</th><th class="u-cell-left-8">Benchmark-Läufe</th><th class="u-cell-center-8">Aktionen</th></tr>';
        snapshots.forEach(s => {
            const runs = (s.benchmark_runs || []);
            const runsHtml = runs.length
                ? `<span title="${SharedUi.runtime.escapeAttr(runs.join(', '))}">${runs.length} Lauf/Läufe</span>`
                : '<span class="u-text-sm-secondary">—</span>';
            const current = s.is_current ? ' <span class="badge badge-success">aktiv</span>' : '';
            html += `<tr class="u-cell-border">
                <td class="u-cell-left-8 u-mono u-fs-12">${SharedUi.runtime.escapeHtml(s.timestamp || '')}</td>
                <td class="u-cell-left-8">${SharedUi.runtime.escapeHtml(s.label || '')}${current}</td>
                <td class="u-cell-left-8 u-mono u-fs-12" title="${SharedUi.runtime.escapeAttr(s.config_sha256 || '')}">${SharedUi.runtime.escapeHtml((s.config_sha256 || '').slice(0, 8))}</td>
                <td class="u-cell-left-8 u-fs-12">${SharedUi.runtime.escapeHtml(s.trigger || '')}</td>
                <td class="u-cell-left-8 u-fs-12">${SharedUi.runtime.escapeHtml(s.kiara_version || '')}</td>
                <td class="u-cell-left-8 u-fs-12">${runsHtml}</td>
                <td class="u-cell-center-pad">
                    <button class="action-btn-sm" data-snap-diff="${SharedUi.runtime.escapeAttr(s.file)}">Diff</button>
                    <button class="action-btn-sm" data-snap-label="${SharedUi.runtime.escapeAttr(s.file)}">Benennen</button>
                    <button class="action-btn-sm" data-snap-restore="${SharedUi.runtime.escapeAttr(s.file)}" ${s.is_current ? 'disabled' : ''}>Wiederherstellen</button>
                </td>
            </tr>`;
        });
        html += '</table>';
    }
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;

    // Event-Delegation: Restore-Buttons (data-* statt Inline-JS)
    container.querySelectorAll('[data-restore-file]').forEach(btn => {
        btn.addEventListener('click', function() { restoreBackup(this, this.dataset.restoreFile); });
    });
    const snapCreateBtn = document.getElementById('btnCreateConfigSnapshot');
    if (snapCreateBtn) snapCreateBtn.addEventListener('click', createConfigSnapshot);
    container.querySelectorAll('[data-snap-diff]').forEach(btn => {
        btn.addEventListener('click', function() { diffConfigSnapshot(this.dataset.snapDiff); });
    });
    container.querySelectorAll('[data-snap-label]').forEach(btn => {
        btn.addEventListener('click', function() { labelConfigSnapshot(this.dataset.snapLabel); });
    });
    container.querySelectorAll('[data-snap-restore]').forEach(btn => {
        btn.addEventListener('click', function() { restoreConfigSnapshot(this, this.dataset.snapRestore); });
    });

    // Reattach: bei Rueckkehr laufendes Backup weiterpollen
    if (isRunning) pollBackupStatus();
}

async function createBackup() {
    const includeVectordb = document.getElementById('backupIncludeVectordb')?.checked === true;
    const btn = document.getElementById('btnCreateBackup');
    const progress = document.getElementById('backupProgress');
    if (btn) btn.disabled = true;
    if (progress) progress.textContent = 'Backup wird gestartet...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/backup/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ include_vectordb_data: includeVectordb }),
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || data.error || data.message || 'Fehler');
        }
        if (progress) progress.textContent = 'Backup laeuft...';
        pollBackupStatus();
    } catch (e) {
        if (btn) btn.disabled = false;
        if (progress) progress.textContent = '';
        SharedUi.notify.show(e.message, 'error');
    }
}

function pollBackupStatus() {
    const mySeq = _backupInitSeq;
    kiaraPolling.start('system-backup:status', async () => {
        if (mySeq !== _backupInitSeq) { kiaraPolling.stop('system-backup:status'); return; }
        const resp = await SharedUi.fetch.safeFetch('/api/admin/backup/status');
        if (!resp.ok) return;
        const data = await resp.json().catch(() => null);
        if (!data) return;
        if (mySeq !== _backupInitSeq) { kiaraPolling.stop('system-backup:status'); return; }
        const progress = document.getElementById('backupProgress');
        if (!data.running) {
            kiaraPolling.stop('system-backup:status');
            if (data.error) {
                if (progress) progress.textContent = 'Fehler: ' + data.error;
                const btn = document.getElementById('btnCreateBackup');
                if (btn) btn.disabled = false;
                SharedUi.notify.show('Backup fehlgeschlagen', 'error');
            } else if (data.result) {
                SharedUi.notify.show('Backup erstellt: ' + data.result.filename, 'success');
                initBackupTab();
            }
        } else {
            if (progress) progress.textContent = 'Backup laeuft...';
        }
    }, 3000);
}

function restoreBackup(btn, filename) {
    SharedUi.runtime.withButtonLock(btn, async () => {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/backup/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, force: true }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || data.error || 'Fehler');
        SharedUi.notify.show('Backup wiederhergestellt', 'success');
    }).catch(e => SharedUi.notify.show(e.message, 'error'));
}

// --- Konfigurations-Snapshots ---

function createConfigSnapshot() {
    SharedUi.modal.open({
        title: 'Config-Snapshot erstellen',
        body: `
            <div class="modal-field"><label>Label (optional, a-z 0-9 . _ -)</label><input type="text" id="snapLabel" placeholder="z.B. reranker-abstain-live"></div>
            <div class="modal-field"><label>Kommentar (optional)</label><input type="text" id="snapComment" placeholder="Anlass, Messwerte, Referenzen"></div>`,
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn primary" id="snapCreateBtn">Erstellen</button>`,
        onOpen: () => {
            document.getElementById('snapLabel').focus();
            document.getElementById('snapCreateBtn').addEventListener('click', function () {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const label = document.getElementById('snapLabel').value.trim();
                    const comment = document.getElementById('snapComment').value.trim();
                    const resp = await SharedUi.fetch.safeFetch('/api/admin/config-snapshots/create', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ label, comment }),
                    });
                    const data = await resp.json().catch(() => ({}));
                    if (!resp.ok) throw new Error(data.detail || data.error || 'Fehler');
                    SharedUi.modal.close();
                    SharedUi.notify.show('Config-Snapshot angelegt', 'success');
                    initBackupTab();
                }).catch(e => SharedUi.notify.show(e.message, 'error'));
            });
        },
    });
}

function labelConfigSnapshot(file) {
    SharedUi.modal.open({
        title: 'Snapshot benennen',
        body: `
            <div class="modal-field"><label>Neues Label</label><input type="text" id="snapNewLabel"></div>
            <div class="modal-field"><label>Kommentar (optional, ersetzt bestehenden)</label><input type="text" id="snapNewComment"></div>`,
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn primary" id="snapLabelBtn">Speichern</button>`,
        onOpen: () => {
            document.getElementById('snapNewLabel').focus();
            document.getElementById('snapLabelBtn').addEventListener('click', function () {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const label = document.getElementById('snapNewLabel').value.trim();
                    if (!label) { SharedUi.notify.show('Label erforderlich', 'error'); return; }
                    const comment = document.getElementById('snapNewComment').value.trim();
                    const body = { ref: file, label };
                    if (comment) body.comment = comment;
                    const resp = await SharedUi.fetch.safeFetch('/api/admin/config-snapshots/label', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body),
                    });
                    const data = await resp.json().catch(() => ({}));
                    if (!resp.ok) throw new Error(data.detail || data.error || 'Fehler');
                    SharedUi.modal.close();
                    SharedUi.notify.show('Snapshot benannt', 'success');
                    initBackupTab();
                }).catch(e => SharedUi.notify.show(e.message, 'error'));
            });
        },
    });
}

async function diffConfigSnapshot(file) {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/config-snapshots/diff', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ref: file, against: 'current' }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || data.error || 'Fehler');
        const changes = (data.changes || []).slice(1);
        let body;
        if (changes.length === 0) {
            body = '<div class="no-data">Keine Unterschiede zur aktuellen Konfiguration.</div>';
        } else {
            body = '<table class="u-detail-table"><tr class="u-cell-border-strong"><th class="u-cell-left-8">Schlüssel</th><th class="u-cell-left-8">Snapshot</th><th class="u-cell-left-8">Aktuell</th></tr>';
            changes.forEach(c => {
                body += `<tr class="u-cell-border">
                    <td class="u-cell-left-8 u-mono u-fs-12">${SharedUi.runtime.escapeHtml(c.key)}</td>
                    <td class="u-cell-left-8 u-mono u-fs-12">${SharedUi.runtime.escapeHtml(String(c.snapshot))}</td>
                    <td class="u-cell-left-8 u-mono u-fs-12">${SharedUi.runtime.escapeHtml(String(c.against))}</td>
                </tr>`;
            });
            body += '</table>';
        }
        SharedUi.modal.open({
            title: `Diff: ${file} → aktuell`,
            body,
            footer: '<button class="action-btn" data-sharedui-close>Schließen</button>',
        });
    } catch (e) {
        SharedUi.notify.show(e.message, 'error');
    }
}

function restoreConfigSnapshot(btn, file) {
    SharedUi.modal.open({
        title: 'Config-Snapshot wiederherstellen',
        body: `
            <p>Die aktive <span class="u-mono">kiara.yaml</span> wird durch
            <span class="u-mono">${SharedUi.runtime.escapeHtml(file)}</span> ersetzt.
            Der aktuelle Stand wird vorher automatisch gesichert.</p>
            <div class="modal-field"><label class="u-clickable"><input type="checkbox" id="snapRestart" checked> Services nach dem Restore neu starten</label></div>`,
        footer: `<button class="action-btn" data-sharedui-close>Abbrechen</button>
                 <button class="action-btn danger" id="snapRestoreBtn">Wiederherstellen</button>`,
        onOpen: () => {
            document.getElementById('snapRestoreBtn').addEventListener('click', function () {
                SharedUi.runtime.withButtonLock(this, async () => {
                    const restart = document.getElementById('snapRestart').checked === true;
                    const resp = await SharedUi.fetch.safeFetch('/api/admin/config-snapshots/restore', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ref: file, restart }),
                    });
                    const data = await resp.json().catch(() => ({}));
                    if (!resp.ok) throw new Error(data.detail || data.error || 'Fehler');
                    SharedUi.modal.close();
                    SharedUi.notify.show(restart
                        ? 'Konfiguration wiederhergestellt — Services starten neu'
                        : 'Konfiguration wiederhergestellt (Restart ausstehend)', 'success');
                    if (!restart) initBackupTab();
                }).catch(e => SharedUi.notify.show(e.message, 'error'));
            });
        },
    });
}
