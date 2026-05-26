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
        const [listResp, statusResp] = await Promise.all([
            SharedUi.fetch.safeFetch('/api/admin/backup/list').then(r => { if (!r.ok) throw new Error('Laden fehlgeschlagen'); return r.json(); }),
            SharedUi.fetch.safeFetch('/api/admin/backup/status').then(r => { if (!r.ok) throw new Error('Laden fehlgeschlagen'); return r.json(); }),
        ]);
        if (mySeq !== _backupInitSeq) return;
        renderBackupTab(container, listResp, statusResp);
    } catch (e) {
        if (mySeq !== _backupInitSeq) return;
        container.innerHTML = '<div class="no-data">Backup-Daten konnten nicht geladen werden.</div>';
    }
}

function renderBackupTab(container, listData, statusData) {
    const backups = listData.backups || [];
    const isRunning = statusData.running || false;

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

    html += '</div>';
    container.innerHTML = html;

    // Event-Delegation: Restore-Buttons (data-* statt Inline-JS)
    container.querySelectorAll('[data-restore-file]').forEach(btn => {
        btn.addEventListener('click', function() { restoreBackup(this, this.dataset.restoreFile); });
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
