/** Shared storage presentation for System-Monitor and VectorDB management. */
const KiaraStorage = (() => {
    const escape = value => SharedUi.runtime.escapeHtml(String(value ?? ''));

    function formatBytes(value) {
        if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return 'nicht verfügbar';
        const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
        const index = value > 0 ? Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1) : 0;
        return (value / Math.pow(1024, index)).toLocaleString('de-DE', { maximumFractionDigits: 2 }) + ' ' + units[index];
    }

    function collectionsTable(data) {
        const collections = data?.collections;
        if (!collections?.length) {
            return `<p class="storage-note">${data?.status === 'ok' ? 'Keine Collections vorhanden.' : 'Collection-Größen nicht verfügbar.'}</p>`;
        }
        const rows = [...collections].sort((a, b) => (b.size_bytes ?? -1) - (a.size_bytes ?? -1));
        return '<div class="storage-table-wrap"><table class="storage-table"><thead><tr>'
            + '<th scope="col">Collection / Modell</th><th scope="col">Typ</th><th scope="col">Größe</th>'
            + '</tr></thead><tbody>'
            + rows.map(item => '<tr><td>' + escape(item.model || item.name)
                + (item.model ? `<small>${escape(item.name)}</small>` : '')
                + '</td><td>' + escape(item.kind === 'colbert' ? 'ColBERT' : item.kind === 'kb' ? 'Dense' : '—')
                + '</td><td>' + formatBytes(item.size_bytes) + '</td></tr>').join('')
            + '</tbody></table></div>';
    }

    function postgresTable(data) {
        if (!Array.isArray(data?.tables)) return '<p class="storage-note">Tabellengrößen nicht verfügbar.</p>';
        if (!data.tables.length) return '<p class="storage-note">Keine Anwendungstabellen vorhanden.</p>';
        return '<div class="storage-table-wrap"><table class="storage-table"><thead><tr>'
            + '<th scope="col">Tabelle</th><th scope="col">Daten</th><th scope="col">SQL-Indizes</th><th scope="col">Gesamt</th>'
            + '</tr></thead><tbody>'
            + data.tables.map(item => '<tr><td>' + escape((item.schema === 'public' ? '' : item.schema + '.') + item.name)
                + '</td><td>' + formatBytes(item.table_bytes)
                + '</td><td>' + formatBytes(item.index_bytes)
                + '</td><td>' + formatBytes(item.size_bytes) + '</td></tr>').join('')
            + '</tbody></table></div>';
    }

    function timestamp(data) {
        if (!data?.measured_at) return '';
        const date = new Date(data.measured_at);
        if (Number.isNaN(date.getTime())) return '';
        return `<p class="storage-note">Stand: ${escape(date.toLocaleString('de-DE'))} · Messung jede Minute</p>`;
    }

    function summary(data) {
        if (!data || data.loading) return '<p class="storage-note">Speicherbedarf wird ermittelt …</p>';
        let html = '<dl class="storage-summary"><div class="storage-total"><dt>Datenbanken &amp; Indizes gesamt</dt>'
            + '<dd>' + formatBytes(data.total_bytes) + '</dd></div>';
        for (const [key, label] of [['postgresql', 'PostgreSQL'], ['qdrant', 'Qdrant'], ['elasticsearch', 'Elasticsearch']]) {
            const item = data[key];
            const value = item?.status === 'not_configured' ? 'nicht konfiguriert' : formatBytes(item?.size_bytes);
            html += '<div><dt>' + label + '</dt><dd title="' + SharedUi.runtime.escapeAttr(item?.reason || '') + '">' + value + '</dd></div>';
        }
        html += '</dl>';
        if (data.total_bytes == null) html += '<p class="storage-note">Gesamtgröße nicht verfügbar, da Messwerte fehlen.</p>';
        html += '<details class="storage-details" data-storage-detail="postgresql"><summary>Größte PostgreSQL-Tabellen</summary>'
            + postgresTable(data.postgresql) + '</details>';
        html += '<details class="storage-details" data-storage-detail="qdrant"><summary>Qdrant-Collections</summary>'
            + collectionsTable(data.qdrant) + '</details>';
        if (data.keyword_mode === 'bm25') {
            html += '<p class="storage-note">Keyword-Suche: BM25 im Arbeitsspeicher, ohne eigene persistente Datenbank.</p>';
        } else if (data.keyword_mode === 'elasticsearch') {
            html += '<p class="storage-note">Keyword-Suche: Elasticsearch'
                + (data.elasticsearch?.indices?.length ? ' (' + escape(data.elasticsearch.indices.join(', ')) + ')' : '') + '. Primär- und Replikatdaten einschließlich Translogs; aufbewahrter Altindex nicht enthalten.</p>';
        }
        html += '<p class="storage-note">Wissensgraph in PostgreSQL enthalten; ColBERT in Qdrant. '
            + 'Ohne Quelldateien, Modelle, Backups und PostgreSQL-Transaktionslogs.</p>';
        return html + timestamp(data);
    }

    function collections(data) {
        if (!data || data.loading) return '<p class="storage-note">Collection-Größen werden ermittelt …</p>';
        return '<p class="storage-collection-total">Qdrant gesamt: <strong>' + formatBytes(data.qdrant?.size_bytes) + '</strong></p>'
            + collectionsTable(data.qdrant)
            + '<p class="storage-note">Belegter Plattenplatz aller Collections, einschließlich ColBERT. '
            + 'Die Gesamtgröße enthält auch gemeinsame Verwaltungsdateien.</p>' + timestamp(data);
    }

    function update(element, data, render = summary) {
        if (!element) return;
        const key = JSON.stringify(data);
        if (element._storageSnapshot === key) return;
        const opened = new Set(Array.from(element.querySelectorAll('details[open]'), item => item.dataset.storageDetail));
        element.innerHTML = render(data);
        element.querySelectorAll('details').forEach(item => { item.open = opened.has(item.dataset.storageDetail); });
        element._storageSnapshot = key;
    }

    return { update, summary, collections, formatBytes };
})();
