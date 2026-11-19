"""PST-Scanner: schreibt Mails, Termine und Anhaenge einer PST in den Mail-Cache.

Gleiches Ausgabeformat wie der IMAP-Scanner (Textdatei je Mail, Anhaenge unter
``<Ordner>/attachments/``), damit Chunking (E-Mail-Strategie), Mail-Archiv und
Anhangsfilter unveraendert greifen. Termine erhalten einen eigenen Kopfblock.

PST-Archive aendern sich selten: Ein Scan wird uebersprungen, solange Groesse,
Aenderungszeit und Scan-Optionen gleich sind. Nach einem vollstaendigen Scan
werden Cache-Dateien entfernt, die nicht mehr in der PST vorkommen.
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime

from kiara.ingestion.imap_scanner import IMAPScanner, _MailBody
from kiara.ingestion.mail_archive import validate_mail_archive, write_mail_cache
from kiara.ingestion.pst_reader import PstFile, PstItem, format_local
from kiara.shared.constants import IMAP_DEFAULT_MAX_ATTACHMENT_MB, IMAP_MAX_FILENAME_LENGTH

logger = logging.getLogger(__name__)

_STATE_FILE = ".pst_state.json"
_SCAN_MARKER = ".scan_done"


class PSTScanner:
    """Liest eine PST-Datei und exportiert sie in ein Cache-Verzeichnis."""

    def __init__(self, config: dict):
        """
        Config-Keys:
            pst_path: str - Absoluter Pfad der PST-Datei
            folders: str - "*" oder kommaseparierte Ordnerpfade ("Posteingang, Kalender")
            include_calendar: bool - Termine exportieren (Standard: True)
            download_attachments: bool - Anhaenge exportieren (Standard: True)
            max_attachment_mb: int - Max. Anhangsgroesse in MB (Standard: 25)
            mail_archive: dict | None - Mail-Archiv-Konfiguration
        """
        self._pst_path = config["pst_path"]
        raw_folders = str(config.get("folders") or "*").strip()
        self._all_folders = raw_folders == "*"
        self._folders = {f.strip().strip("/") for f in raw_folders.split(",") if f.strip()}
        self._include_calendar = config.get("include_calendar", True)
        self._download_attachments = config.get("download_attachments", True)
        self._max_attachment_bytes = (
            int(config.get("max_attachment_mb") or IMAP_DEFAULT_MAX_ATTACHMENT_MB) * 1024 * 1024
        )
        self._mail_archive = config.get("mail_archive")

    # --- Zustand ---

    def _fingerprint(self) -> dict:
        stat = os.stat(self._pst_path)
        options = json.dumps(
            {
                "folders": "*" if self._all_folders else sorted(self._folders),
                "calendar": bool(self._include_calendar),
                "attachments": bool(self._download_attachments),
                "max_attachment_bytes": self._max_attachment_bytes,
                "archive": self._mail_archive,
            },
            sort_keys=True,
        )
        return {
            "path": self._pst_path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "options": hashlib.sha256(options.encode()).hexdigest(),
        }

    @staticmethod
    def _load_state(output_dir: str) -> dict:
        try:
            with open(os.path.join(output_dir, _STATE_FILE), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_state(output_dir: str, state: dict) -> None:
        path = os.path.join(output_dir, _STATE_FILE)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, path)

    def is_current(self, output_dir: str) -> bool:
        """True, wenn der Cache bereits vollstaendig zu dieser PST passt."""
        state = self._load_state(output_dir)
        return (
            state.get("complete") is True
            and state.get("fingerprint") == self._fingerprint()
            and os.path.exists(os.path.join(output_dir, _SCAN_MARKER))
        )

    def _selected(self, path: str) -> bool:
        if self._all_folders:
            return True
        return path in self._folders

    # --- Scan ---

    def scan(
        self,
        output_dir: str,
        progress_callback: Callable[[int, int], None] | None = None,
        control_callback: Callable[[], str | None] | None = None,
    ) -> list[str]:
        """Exportiert die PST nach output_dir und gibt die geschriebenen Dateien zurueck.

        Raises:
            InterruptedError: Bei Abbruch/Shutdown ueber control_callback.
            OSError: Wenn die PST nicht lesbar ist.
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.is_current(output_dir):
            logger.info("PST unveraendert (%s) — Scan uebersprungen", self._pst_path)
            return []

        fingerprint = self._fingerprint()
        written: set[str] = set()
        errors = 0
        exported = 0
        with PstFile(self._pst_path) as pst:
            folders = [(p, f) for p, f in pst.iter_folders() if self._selected(p)]
            total = pst.count_messages(folders)
            processed = 0
            for folder_path, folder in folders:
                for index in range(folder.number_of_sub_messages):
                    signal = control_callback() if control_callback else None
                    if signal in ("cancel", "shutdown"):
                        raise InterruptedError(signal)
                    processed += 1
                    try:
                        item = pst.read_item(
                            folder.get_sub_message(index),
                            with_attachments=self._download_attachments,
                            max_attachment_bytes=self._max_attachment_bytes,
                        )
                        if item is None:
                            continue
                        if item.kind == "appointment" and not self._include_calendar:
                            continue
                        written.update(self._save_item(item, folder_path, output_dir))
                        exported += 1
                    except Exception as exc:
                        errors += 1
                        logger.warning(
                            "PST %s: Element %d in '%s' nicht lesbar: %s",
                            self._pst_path,
                            index,
                            folder_path,
                            exc,
                        )
                    if progress_callback and (processed % 25 == 0 or processed == total):
                        progress_callback(processed, total)

        removed = self._remove_stale(output_dir, written) if not errors else 0
        self._save_state(
            output_dir,
            {"fingerprint": fingerprint, "complete": errors == 0, "errors": errors},
        )
        with open(os.path.join(output_dir, _SCAN_MARKER), "w") as f:
            f.write(datetime.now().isoformat())
        logger.info(
            "PST-Scan %s: %d Elemente exportiert, %d Dateien, %d entfernt, %d Fehler",
            self._pst_path,
            exported,
            len(written),
            removed,
            errors,
        )
        return sorted(written)

    @staticmethod
    def _remove_stale(output_dir: str, written: set[str]) -> int:
        """Cache-Dateien entfernen, die nicht mehr aus der PST stammen."""
        removed = 0
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                if root == output_dir and name.startswith("."):
                    continue
                if path not in written:
                    os.unlink(path)
                    removed += 1
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    pass  # nicht leer
        return removed

    # --- Ausgabe ---

    def _save_item(self, item: PstItem, folder_path: str, output_dir: str) -> list[str]:
        """Schreibt ein Element als Textdatei plus Anhaenge (Format wie IMAP-Cache)."""
        moment = item.start if item.kind == "appointment" else item.date
        date_prefix = moment.astimezone().strftime("%Y%m%d") if moment else "00000000"
        uid = str(item.identifier)
        safe_subject = IMAPScanner._safe_filename(item.subject)[:IMAP_MAX_FILENAME_LENGTH]
        folder_key = IMAPScanner._folder_key(folder_path)
        mail_path = os.path.join(output_dir, folder_key, f"{date_prefix}_{uid}_{safe_subject}.txt")

        selected_body = _MailBody(item.body, item.body_mime, item.body_charset)
        validate_mail_archive(self._mail_archive, output_dir, selected_body.payload)
        body, body_metadata = selected_body.convert()

        write_mail_cache(
            mail_path,
            self._header(item, folder_path) + body,
            body=selected_body.payload,
            body_text=body,
            mime_type=selected_body.mime_type,
            charset=selected_body.charset,
            metadata=body_metadata,
            archive_config=self._mail_archive,
        )
        files = [mail_path]

        for index, attachment in enumerate(item.attachments, start=1):
            name = IMAPScanner._safe_filename(attachment.filename)
            att_path = os.path.join(
                output_dir, folder_key, "attachments", f"{date_prefix}_{uid}_{index}_{name}"
            )
            _write_if_changed(att_path, attachment.data)
            files.append(att_path)
        return files

    @staticmethod
    def _header(item: PstItem, folder_path: str) -> str:
        if item.kind == "appointment":
            lines = [
                f"Termin: {item.subject}",
                f"Beginn: {format_local(item.start, date_only=item.all_day)}",
                f"Ende: {format_local(item.end, date_only=item.all_day)}",
            ]
            if item.all_day:
                lines.append("Ganztaegig: ja")
            if item.recurrence:
                lines.append(f"Serie: {item.recurrence}")
            if item.canceled:
                lines.append("Status: abgesagt")
            lines += [
                f"Ort: {item.location}",
                f"Organisator: {item.sender}",
                f"Teilnehmer: {item.attendees or item.to}",
                f"Ordner: {folder_path}",
            ]
            return "\n".join(lines) + "\n---\n"
        lines = [
            f"Betreff: {item.subject}",
            f"Von: {item.sender}",
            f"An: {item.to}",
        ]
        if item.cc:
            lines.append(f"Cc: {item.cc}")
        lines.append(f"Datum: {format_local(item.date)}")
        if item.start is not None:
            # Besprechungsanfragen/-absagen: Terminangaben mit in den Kopf
            lines.append(
                f"Termin: {format_local(item.start, date_only=item.all_day)} bis "
                f"{format_local(item.end, date_only=item.all_day)}"
                + (f", {item.location}" if item.location else "")
            )
        lines.append(f"Ordner: {folder_path}")
        return "\n".join(lines) + "\n---\n"


def _write_if_changed(path: str, data: bytes) -> None:
    """Anhang nur schreiben, wenn er fehlt oder abweicht (mtime bleibt sonst stabil)."""
    try:
        if os.path.getsize(path) == len(data):
            with open(path, "rb") as f:
                if f.read() == data:
                    return
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
