/**
 * KIara Admin - Mail Tab
 * SMTP configuration and test.
 */

let mailConfig = null;

const _MAIL_ACTIONS = new Set([
    'save-mail-config',
    'test-mail',
    'send-status-report',
    'delete-mail-config',
]);

function _handleMailClick(e) {
    const el = e.target.closest('[data-action]');
    if (!el || !_MAIL_ACTIONS.has(el.dataset.action)) return;
    switch (el.dataset.action) {
        case 'save-mail-config':      SharedUi.runtime.withButtonLock(el, saveMailConfig); break;
        case 'test-mail':             SharedUi.runtime.withButtonLock(el, testMail); break;
        case 'send-status-report':    SharedUi.runtime.withButtonLock(el, sendStatusReport); break;
        case 'delete-mail-config':    deleteMailConfig(); break;
    }
}

function _bindMailActions() {
    if (document.body._mailActionsBound) return;
    document.body._mailActionsBound = true;
    document.body.addEventListener('click', _handleMailClick);
    const container = document.getElementById('contentContainer');
    if (container) container.setAttribute('data-mail-actions-bound', 'true');
}

async function initMailTab() {
    _bindMailActions();
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;
    if (filterBar) filterBar.classList.add('u-hidden');

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const data = await loadConfigFromAPI('/api/admin/mail/config');
        mailConfig = data;
        renderMailForm(container);
    } catch (e) {
        container.innerHTML = '<div class="no-data">Mail-Konfiguration konnte nicht geladen werden.</div>';
    }
}

function renderMailForm(container) {
    const cfg = mailConfig || {};
    let html = '<div class="config-form">';
    html += '<h3>Mail-Konfiguration (SMTP)</h3>';

    // Server
    html += '<div class="config-section"><h4>SMTP-Server</h4><div class="config-fields">';
    html += renderMailField('mail_server', 'Server', cfg.smtp_server || '', 'text', 'smtp.example.com');
    html += renderMailField('mail_port', 'Port', cfg.smtp_port || 587, 'number');
    html += `<div class="config-field"><label>Verschluesselung</label>
        <select class="config-input" id="mail_encryption">
            <option value="none" ${cfg.encryption === 'none' ? 'selected' : ''}>Keine</option>
            <option value="starttls" ${(!cfg.encryption || cfg.encryption === 'starttls') ? 'selected' : ''}>StartTLS</option>
            <option value="ssl" ${cfg.encryption === 'ssl' ? 'selected' : ''}>SSL/TLS</option>
        </select></div>`;
    html += '</div></div>';

    // Credentials
    html += '<div class="config-section"><h4>Anmeldung</h4><div class="config-fields">';
    html += renderMailField('mail_user', 'Benutzername', cfg.username || '', 'text');
    const pwPlaceholder = mailConfig ? '(unveraendert wenn leer)' : 'Passwort...';
    html += renderMailField('mail_pw', 'Passwort', '', 'password', pwPlaceholder);
    html += '</div></div>';

    // Addresses
    html += '<div class="config-section"><h4>Adressen</h4><div class="config-fields">';
    html += renderMailField('mail_sender', 'Absender', cfg.sender_address || '', 'text', 'kiara@domain.local');
    html += renderMailField('mail_recipients', 'Empfaenger', cfg.recipient_addresses || '', 'text', 'admin@domain.local, admin2@domain.local');
    html += '</div></div>';

    // Notifications
    html += '<div class="config-section"><h4>Benachrichtigungen</h4><div class="config-fields">';
    html += `<div class="config-field"><label><input type="checkbox" id="mail_notify_ollama" ${cfg.notify_ollama_down !== false ? 'checked' : ''}> Ollama nicht erreichbar</label></div>`;
    html += `<div class="config-field"><label><input type="checkbox" id="mail_notify_ds" ${cfg.notify_datasource_down !== false ? 'checked' : ''}> Datenquelle nicht erreichbar</label></div>`;
    html += `<div class="config-field"><label><input type="checkbox" id="mail_notify_index" ${cfg.notify_index_error !== false ? 'checked' : ''}> Indexierungsfehler</label></div>`;
    html += '</div></div>';

    // Status Reports
    html += '<div class="config-section"><h4>Status-Reports</h4><div class="config-fields">';
    html += `<div class="config-field"><label><input type="checkbox" id="mail_report_enabled" ${cfg.status_report_enabled ? 'checked' : ''}> Automatische Status-Reports</label></div>`;
    html += `<div class="config-field"><label>Intervall</label>
        <select class="config-input" id="mail_report_interval">
            <option value="daily" ${cfg.status_report_interval === 'daily' ? 'selected' : ''}>Taeglich</option>
            <option value="weekly" ${(!cfg.status_report_interval || cfg.status_report_interval === 'weekly') ? 'selected' : ''}>Woechentlich</option>
            <option value="monthly" ${cfg.status_report_interval === 'monthly' ? 'selected' : ''}>Monatlich</option>
        </select></div>`;
    html += '</div></div>';

    // Actions
    html += '<div class="config-actions">';
    html += '<button class="action-btn primary" data-action="save-mail-config">Speichern</button>';
    if (mailConfig) {
        html += '<button class="action-btn" data-action="test-mail">Test-Mail senden</button>';
        html += '<button class="action-btn" data-action="send-status-report">Status-Report senden</button>';
        html += '<button class="action-btn danger" data-action="delete-mail-config">Konfiguration loeschen</button>';
    }
    html += '<span id="mailTestResult" class="mail-test-result"></span>';
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;
}

function renderMailField(id, label, value, type, placeholder) {
    return `<div class="config-field"><label>${SharedUi.runtime.escapeHtml(label)}</label>
        <input type="${type}" class="config-input" id="${id}"
            value="${SharedUi.runtime.escapeAttr(String(value || ''))}"
            placeholder="${SharedUi.runtime.escapeAttr(placeholder || '')}"></div>`;
}

function getMailFormData() {
    return {
        smtp_server: document.getElementById('mail_server')?.value || '',
        smtp_port: parseInt(document.getElementById('mail_port')?.value) || 587,
        encryption: document.getElementById('mail_encryption')?.value || 'starttls',
        username: document.getElementById('mail_user')?.value || '',
        password: document.getElementById('mail_pw')?.value || '',
        sender_address: document.getElementById('mail_sender')?.value || '',
        recipient_addresses: document.getElementById('mail_recipients')?.value || '',
        notify_ollama_down: document.getElementById('mail_notify_ollama')?.checked ?? true,
        notify_datasource_down: document.getElementById('mail_notify_ds')?.checked ?? true,
        notify_index_error: document.getElementById('mail_notify_index')?.checked ?? true,
        status_report_enabled: document.getElementById('mail_report_enabled')?.checked ?? false,
        status_report_interval: document.getElementById('mail_report_interval')?.value || 'weekly',
    };
}

async function saveMailConfig() {
    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/mail/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(getMailFormData()),
        });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || errData.error || errData.message || 'Speichern fehlgeschlagen');
        }
        const reloaded = await loadConfigFromAPI('/api/admin/mail/config');
        mailConfig = reloaded;
        const container = document.getElementById('contentContainer');
        if (container) renderMailForm(container);
        SharedUi.notify.show('Mail-Konfiguration gespeichert', 'success');
    } catch (error) {
        SharedUi.notify.show(error.message, 'error');
    }
}

async function testMail() {
    const resultEl = document.getElementById('mailTestResult');
    if (resultEl) resultEl.textContent = 'Sende Test-Mail...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/mail/test', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Fehler');
        if (resultEl) {
            resultEl.classList.remove('u-c-danger');
            resultEl.textContent = data.message;
            resultEl.classList.add('u-c-success');
        }
        SharedUi.notify.show('Test-Mail gesendet', 'success');
    } catch (e) {
        if (resultEl) {
            resultEl.classList.remove('u-c-success');
            resultEl.textContent = e.message;
            resultEl.classList.add('u-c-danger');
        }
        SharedUi.notify.show(e.message, 'error');
    }
}

async function sendStatusReport() {
    const resultEl = document.getElementById('mailTestResult');
    if (resultEl) resultEl.textContent = 'Sende Status-Report...';

    try {
        const resp = await SharedUi.fetch.safeFetch('/api/admin/mail/status-report', { method: 'POST' });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Fehler');
        if (resultEl) {
            resultEl.classList.remove('u-c-danger');
            resultEl.textContent = data.message;
            resultEl.classList.add('u-c-success');
        }
        SharedUi.notify.show('Status-Report gesendet', 'success');
    } catch (e) {
        if (resultEl) {
            resultEl.classList.remove('u-c-success');
            resultEl.textContent = e.message;
            resultEl.classList.add('u-c-danger');
        }
        SharedUi.notify.show(e.message, 'error');
    }
}

async function deleteMailConfig() {
    await deleteConfigFromAPI({
        endpoint: '/api/admin/mail/config',
        confirmMessage: 'Mail-Konfiguration wirklich loeschen?',
        successMessage: 'Mail-Konfiguration geloescht',
        onSuccess: () => initMailTab(),
    });
}
