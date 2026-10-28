/**
 * KIara - Chat-Lokalisierung (Timestamp).
 *
 * Seit Sprint 4 (Shared Frontend Core) liefert SharedUi.markdown aus
 * ui_core/js/markdown.js Copy/Think/Markdown. Diese Datei behaelt nur
 * noch die deutsche Timestamp-Formatierung als Top-Level-Funktion.
 */

function formatKiaraTimestamp(dateInput) {
    if (!dateInput) return '';
    let d;
    if (typeof dateInput === 'string' && /^\d{4}-\d{2}-\d{2} /.test(dateInput)) {
        d = new Date(dateInput.replace(' ', 'T') + 'Z');
    } else {
        d = new Date(dateInput);
    }
    if (isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
