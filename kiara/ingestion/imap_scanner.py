"""IMAP E-Mail-Scanner fuer die Ingestion-Pipeline."""

import email
import hashlib
import imaplib
import json
import logging
import os
import re
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header
from email.utils import parsedate_to_datetime

from kiara.ingestion.mail_archive import validate_mail_archive, write_mail_cache
from kiara.shared.constants import (
    IMAP_DEFAULT_MAX_ATTACHMENT_MB,
    IMAP_DEFAULT_SINCE_DAYS,
    IMAP_FETCH_BATCH_SIZE,
    IMAP_MAX_FILENAME_LENGTH,
    IMAP_PORT_SSL,
)

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_CONN_ERRORS = 5
_MAX_RECONNECT_ATTEMPTS = 5
_UID_REGISTRY_FILE = ".uid_registry.json"


@dataclass(frozen=True)
class _MailBody:
    """Ausgewaehltes Original vor der verlustfreien Cache-Aufbereitung."""

    payload: bytes = b""
    mime_type: str = "text/plain"
    charset: str = "utf-8"

    def convert(self) -> tuple[str, dict]:
        decoded = IMAPScanner._safe_decode(self.payload, self.charset)
        if self.mime_type == "text/html":
            from kiara_document_loaders.html import convert_html

            converted = convert_html(decoded)
            return converted.markdown, converted.metadata()
        digest = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
        return decoded, {
            "parser": "mail-plain",
            "content_format": "text",
            "source_mime_type": "text/plain",
            "mail_parser_revision": "1",
            "mail_profile_revision": "plain-main-body-1",
            "mail_source_hash": digest,
            "mail_output_hash": digest,
        }


class IMAPScanner:
    """Scannt ein IMAP-Postfach und extrahiert E-Mails als Dokumente."""

    def __init__(self, config: dict):
        """
        Config-Keys (aus Datasource-Spalten):
            imap_server: str - IMAP-Server (z.B. "mail.firma.de")
            imap_port: int - Port (993 fuer SSL, 143 fuer STARTTLS)
            imap_ssl: bool - SSL verwenden (Standard: True)
            imap_username: str - Benutzername / E-Mail-Adresse
            imap_password: str - Passwort (verschluesselt in DB)
            imap_folders: str - Kommaseparierte Ordner (Standard: "INBOX")
            imap_since_days: int - Nur Mails der letzten N Tage (Standard: 90)
            imap_download_attachments: bool - Anhaenge herunterladen (Standard: True)
            imap_max_attachment_mb: int - Max. Anhangsgroesse in MB (Standard: 25)
        """
        self._server = config.get("imap_server", "")
        self._port = config.get("imap_port", IMAP_PORT_SSL)
        self._ssl = config.get("imap_ssl", True)
        self._username = config.get("imap_username", "")
        self._password = config.get("imap_password", "")
        raw_folders = config.get("imap_folders", "INBOX").strip()
        self._all_folders = raw_folders == "*"
        self._folders = [] if self._all_folders else [f.strip() for f in raw_folders.split(",")]
        self._since_days = config.get("imap_since_days", IMAP_DEFAULT_SINCE_DAYS)
        self._download_attachments = config.get("imap_download_attachments", True)
        self._max_attachment_bytes = (
            config.get("imap_max_attachment_mb", IMAP_DEFAULT_MAX_ATTACHMENT_MB) * 1024 * 1024
        )
        self._mail_archive = config.get("mail_archive")

    def connect(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        """Verbindung zum IMAP-Server herstellen."""
        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        tls_context = ssl.create_default_context()
        if self._ssl:
            conn = imaplib.IMAP4_SSL(
                self._server, self._port, ssl_context=tls_context, timeout=CONNECTOR_CONNECT_TIMEOUT
            )
        else:
            conn = imaplib.IMAP4(self._server, self._port, timeout=CONNECTOR_CONNECT_TIMEOUT)
        try:
            if not self._ssl:
                conn.starttls(ssl_context=tls_context)
            conn.login(self._username, self._password)
            # Timeout nach Verbindungsaufbau zuruecksetzen: Der Socket soll fuer
            # nachfolgende Operationen (select, search, fetch) im Blocking-Modus
            # laufen — kein per-Operation Timeout, konsistent mit bisherigem Verhalten.
            conn.sock.settimeout(None)
        except Exception:
            try:
                conn.shutdown()
            except Exception:
                pass
            raise
        return conn

    def test_connection(self) -> dict:
        """Verbindungstest. Gibt Ordnerliste zurueck."""
        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        try:
            conn = self.connect()
        except TimeoutError:
            return {
                "success": False,
                "message": (
                    f"Verbindungsaufbau fehlgeschlagen — Server {self._server}:{self._port} "
                    f"nicht erreichbar (Timeout nach {CONNECTOR_CONNECT_TIMEOUT}s)"
                ),
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}
        try:
            conn.sock.settimeout(CONNECTOR_CONNECT_TIMEOUT)
            _status, raw_folders = conn.list()
            parsed = self._parse_imap_list(raw_folders)
            folder_names = [f["name"] for f in parsed]
            return {"success": True, "folders": folder_names}
        except Exception as exc:
            return {"success": False, "message": str(exc)}
        finally:
            try:
                conn.logout()
            except Exception as exc:
                logger.debug("IMAP logout fehlgeschlagen: %s", exc)

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Prueft ob ein Fehler auf eine verlorene IMAP-Verbindung hindeutet."""
        msg = str(exc).lower()
        return any(
            pattern in msg
            for pattern in (
                "eof occurred in violation of protocol",
                "socket error: eof",
                "connection reset",
                "broken pipe",
                "connection refused",
                "timed out",
                "command: fetch =>",
            )
        ) or isinstance(
            exc,
            (
                ConnectionError,
                BrokenPipeError,
                ConnectionResetError,
                ssl.SSLError,
                OSError,
            ),
        )

    def _ensure_connection_ready(self, conn, folder: str | None = None):
        """Hook fuer Scanner mit kurzlebigen Auth-Tokens.

        Passwort-IMAP braucht keine Vorab-Aktion. OAuth2-Scanner koennen hier
        vor IMAP-Operationen proaktiv reconnecten.
        """
        return conn

    def _reconnect(
        self, conn: imaplib.IMAP4_SSL | imaplib.IMAP4, folder: str | None = None
    ) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        """Verbindung neu aufbauen. Gibt neue Connection zurueck.

        Versucht bis zu _MAX_RECONNECT_ATTEMPTS Mal. Wenn ein Ordner
        angegeben ist, wird dieser nach dem Reconnect selektiert.
        """
        # Alte Verbindung sauber beenden
        try:
            conn.logout()
        except Exception:
            pass

        last_exc = None
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            try:
                new_conn = self.connect()
                if folder:
                    new_conn.select(folder, readonly=True)
                logger.info(
                    "IMAP-Reconnect erfolgreich (Versuch %d/%d)", attempt, _MAX_RECONNECT_ATTEMPTS
                )
                return new_conn
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "IMAP-Reconnect fehlgeschlagen (Versuch %d/%d): %s",
                    attempt,
                    _MAX_RECONNECT_ATTEMPTS,
                    exc,
                )
                time.sleep(min(2**attempt, 30))

        raise ConnectionError(
            f"IMAP-Reconnect nach {_MAX_RECONNECT_ATTEMPTS} Versuchen fehlgeschlagen: {last_exc}"
        )

    def _load_uid_registry(self, output_dir: str) -> dict[str, dict]:
        """Laedt die UID-Registry aus dem Cache-Verzeichnis.

        Bei fehlendem oder korruptem JSON wird ein leeres Dict zurueckgegeben
        (Fallback auf vollen Download).
        """
        path = os.path.join(output_dir, _UID_REGISTRY_FILE)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning("UID-Registry korrupt, voller Rescan: %s", exc)
            return {}

    def _save_uid_registry(self, output_dir: str, registry: dict) -> None:
        """Speichert die UID-Registry atomar via os.replace()."""
        path = os.path.join(output_dir, _UID_REGISTRY_FILE)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(registry, f)
        os.replace(tmp, path)

    def _scan_folder_uids(
        self,
        conn,
        folder: str,
        folder_key: str,
        since_str: str,
        registry: dict,
    ) -> tuple[list[bytes], int] | None:
        """Selektiert Ordner, prueft UIDVALIDITY und sucht neue UIDs.

        Kombiniert IMAP SELECT, UIDVALIDITY-Validierung und UID SEARCH
        in einem Schritt. Wird sowohl im Normalpfad als auch nach
        Reconnect aufgerufen — eliminiert duplizierte Logik.

        Args:
            conn: Aktive IMAP-Verbindung.
            folder: IMAP-Ordnername.
            folder_key: Sanitisierter Schluessel fuer die UID-Registry.
            since_str: IMAP-Datumsstring fuer SINCE-Filter.
            registry: UID-Registry (wird in-place aktualisiert).

        Returns:
            (unknown_uids, skipped_count) bei Erfolg.
            None wenn der Ordner nicht selektierbar ist.
        """
        status, _ = conn.select(folder, readonly=True)
        if status != "OK":
            logger.warning("IMAP-Ordner nicht selektierbar: %s", folder)
            return None

        # UIDVALIDITY pruefen — bei Aenderung wird der UID-Cache invalidiert
        _typ, uidval_data = conn.response("UIDVALIDITY")
        uidvalidity = uidval_data[0].decode() if uidval_data and uidval_data[0] else None
        folder_reg = registry.get(folder_key, {})

        if uidvalidity:
            if folder_reg.get("uidvalidity") and folder_reg["uidvalidity"] != uidvalidity:
                logger.warning(
                    "UIDVALIDITY geaendert fuer '%s' (%s -> %s) — voller Rescan",
                    folder,
                    folder_reg.get("uidvalidity"),
                    uidvalidity,
                )
                folder_reg = {"uidvalidity": uidvalidity, "uids": {}}
            elif not folder_reg.get("uidvalidity"):
                folder_reg["uidvalidity"] = uidvalidity
            registry[folder_key] = folder_reg

        # UID SEARCH (liefert UIDs statt Sequenznummern)
        status, msg_data = conn.uid("search", None, f"(SINCE {since_str})")
        if status != "OK" or not msg_data[0]:
            return [], 0

        all_uids = msg_data[0].split()
        known = set(folder_reg.get("uids", {}).keys())
        unknown_uids = [u for u in all_uids if u.decode("ascii", errors="replace") not in known]
        skipped = len(all_uids) - len(unknown_uids)

        return unknown_uids, skipped

    def _discover_folders(self, conn) -> list[str]:
        """Ermittelt die zu scannenden Ordner (Wildcard oder explizite Liste).

        Bei Wildcard (*) werden alle selektierbaren Ordner dynamisch vom
        Server geladen. Sonst wird die konfigurierte Ordnerliste verwendet.

        Returns:
            Liste von IMAP-Ordnernamen (raw_name fuer Wildcard, Strings sonst).
        """
        if not self._all_folders:
            return self._folders

        try:
            conn.sock.settimeout(30)
            status, raw = conn.list()
        except Exception:
            try:
                conn.logout()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.sock.settimeout(None)
            except Exception:
                pass
        status_text = (
            status.decode("ascii", errors="replace") if isinstance(status, bytes) else str(status)
        )
        if status_text.upper() != "OK" or not raw:
            raise RuntimeError(f"IMAP LIST fehlgeschlagen: {status_text}")
        parsed = self._parse_imap_list(raw)
        folders = [f["raw_name"] for f in parsed if "\\noselect" not in f["flags"]]
        logger.info("Wildcard-Modus: %d Ordner gefunden", len(folders))
        return folders

    def _collect_new_uids(
        self,
        conn,
        folders: list[str],
        since_str: str,
        registry: dict,
    ) -> tuple[list[tuple[str, list[bytes], str]], int, object]:
        """Phase 1: UID SEARCH ueber alle Ordner, bekannte UIDs filtern.

        Args:
            conn: Aktive IMAP-Verbindung.
            folders: Liste der zu scannenden Ordner.
            since_str: IMAP-Datumsstring fuer SINCE-Filter.
            registry: UID-Registry (wird in-place aktualisiert).

        Returns:
            (folder_ids, skipped_total, conn) — folder_ids ist Liste von
            (folder, unknown_uids, folder_key), conn ggf. erneuert.
        """
        folder_ids = []
        skipped_total = 0

        for folder in folders:
            folder_key = self._folder_key(folder)
            try:
                conn = self._ensure_connection_ready(conn)
                result = self._scan_folder_uids(conn, folder, folder_key, since_str, registry)
            except Exception as exc:
                if self._is_connection_error(exc):
                    logger.warning("Verbindungsverlust bei Ordner-Scan '%s', reconnecte...", folder)
                    try:
                        conn = self._reconnect(conn)
                    except ConnectionError:
                        logger.error("Reconnect fehlgeschlagen bei Ordner-Scan, breche ab")
                        raise
                    try:
                        result = self._scan_folder_uids(
                            conn, folder, folder_key, since_str, registry
                        )
                    except Exception as e2:
                        logger.warning("Fehler bei '%s' auch nach Reconnect: %s", folder, e2)
                        continue
                else:
                    logger.warning("Fehler bei IMAP-Ordner '%s': %s", folder, exc)
                    continue

            if result is None:
                continue
            unknown_uids, skipped = result
            skipped_total += skipped

            if skipped:
                logger.info(
                    "IMAP '%s': %d/%d uebersprungen (UID-Cache)",
                    folder,
                    skipped,
                    skipped + len(unknown_uids),
                )
            if unknown_uids:
                logger.info("IMAP '%s': %d neue Mails", folder, len(unknown_uids))
                folder_ids.append((folder, unknown_uids, folder_key))

        return folder_ids, skipped_total, conn

    def _fetch_and_save_mails(
        self,
        conn,
        folder_ids: list[tuple[str, list[bytes], str]],
        registry: dict,
        output_dir: str,
        progress_callback: Callable[[int, int], None] | None,
        control_callback: Callable[[], str | None] | None,
    ) -> tuple[list[str], object, str | None]:
        """Phase 2: Neue UIDs batch-weise fetchen, speichern und Registry updaten.

        Returns:
            (files, conn, abort_reason) — Liste geschriebener Dateipfade,
            (ggf. erneuerte) Connection und None bei erfolgreichem Scan.
        """
        files: list[str] = []
        total = sum(len(uids) for _, uids, _ in folder_ids)
        processed = 0
        abort_reason: str | None = None

        for folder, uids, folder_key in folder_ids:
            if abort_reason:
                break
            try:
                conn = self._ensure_connection_ready(conn, folder=folder)
                status, _ = conn.select(folder, readonly=True)
            except Exception as exc:
                if self._is_connection_error(exc):
                    try:
                        conn = self._reconnect(conn, folder=folder)
                    except ConnectionError:
                        logger.error("Reconnect fehlgeschlagen bei '%s', breche ab", folder)
                        break
                else:
                    logger.warning("Fehler bei IMAP-Ordner '%s': %s", folder, exc)
                    processed += len(uids)
                    continue
            else:
                if status != "OK":
                    logger.warning("IMAP-Ordner nicht selektierbar: '%s', ueberspringe", folder)
                    processed += len(uids)
                    continue

            folder_reg = registry.get(folder_key, {})

            for batch_start in range(0, len(uids), IMAP_FETCH_BATCH_SIZE):
                if abort_reason:
                    break
                if control_callback and control_callback() == "cancel":
                    logger.info("IMAP-Scan abgebrochen (Cancel-Signal)")
                    abort_reason = "IMAP-Scan abgebrochen"
                    break

                batch_uids = uids[batch_start : batch_start + IMAP_FETCH_BATCH_SIZE]
                batch_results, conn = self._fetch_batch_with_reconnect(conn, batch_uids, folder)

                if batch_results is None:
                    processed += len(batch_uids)
                    abort_reason = "IMAP-Batch-Fetch nach Reconnect fehlgeschlagen"
                    break

                for uid, raw_mail in batch_results:
                    if raw_mail is None:
                        processed += 1
                        continue
                    try:
                        uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                        mail_files = self._save_mail(raw_mail, folder, output_dir, uid=uid_str)
                        files.extend(mail_files)
                        if uid and mail_files:
                            rel_path = os.path.relpath(mail_files[0], output_dir)
                            folder_reg.setdefault("uids", {})[uid_str] = rel_path
                    except Exception as exc:
                        logger.warning("Fehler bei Mail UID %s/%s: %s", folder, uid, exc)
                    processed += 1

                if progress_callback:
                    progress_callback(processed, total)

            if not abort_reason:
                registry[folder_key] = folder_reg
                self._save_uid_registry(output_dir, registry)

        if progress_callback and total > 0 and not abort_reason:
            progress_callback(total, total)

        return files, conn, abort_reason

    def scan(
        self,
        output_dir: str,
        progress_callback: Callable[[int, int], None] | None = None,
        control_callback: Callable[[], str | None] | None = None,
    ) -> list[str]:
        """Scannt Mails und schreibt sie als .txt-Dateien + Anhaenge in output_dir.

        Orchestriert drei Phasen:
        1. Ordner-Discovery (Wildcard oder explizit)
        2. UID-Sammlung (neue Mails identifizieren via UID-Registry)
        3. Batch-Fetch + Speichern (mit automatischem Reconnect)

        Args:
            output_dir: Zielverzeichnis fuer die exportierten Dateien.
            progress_callback: Optionaler Callback(processed, total).
            control_callback: Optionaler Callback() -> "cancel"|None.

        Returns:
            Liste der geschriebenen Dateipfade.
        """
        conn = self.connect()
        try:
            since_date = datetime.now(timezone.utc) - timedelta(days=self._since_days)
            since_str = since_date.strftime("%d-%b-%Y")
            registry = self._load_uid_registry(output_dir)

            folders = self._discover_folders(conn)
            folder_ids, skipped_total, conn = self._collect_new_uids(
                conn, folders, since_str, registry
            )

            if skipped_total:
                total = sum(len(uids) for _, uids, _ in folder_ids)
                logger.info(
                    "IMAP-Scan: %d neue Mails, %d uebersprungen (UID-Cache)", total, skipped_total
                )

            files, conn, abort_reason = self._fetch_and_save_mails(
                conn, folder_ids, registry, output_dir, progress_callback, control_callback
            )
        finally:
            try:
                conn.logout()
            except Exception:
                pass

        if abort_reason:
            raise InterruptedError(abort_reason)

        marker = os.path.join(output_dir, ".scan_done")
        with open(marker, "w") as f:
            f.write(datetime.now().isoformat())

        self._save_uid_registry(output_dir, registry)
        return files

    def _fetch_batch(self, conn, batch_uids: list[bytes]) -> list[tuple[bytes, bytes | None]]:
        """Holt einen Batch von Mails per UID FETCH.

        Verwendet ein UID-Set (z.B. "45231,45232,...") um alle Mails
        in einem einzigen IMAP-Kommando herunterzuladen.

        Args:
            conn: IMAP-Verbindung.
            batch_uids: Liste von UIDs als bytes (z.B. [b'45231', b'45232']).

        Returns:
            Liste von (uid, raw_bytes | None) Tuples.
            raw_bytes ist None bei Fetch-Fehlern einzelner Mails.
        """
        uid_set = b",".join(batch_uids)
        status, data = conn.uid("fetch", uid_set, "(UID BODY.PEEK[])")

        results = []
        if status != "OK":
            return [(uid, None) for uid in batch_uids]

        # imaplib gibt fuer Batch-Fetch eine flache Liste zurueck:
        # [(b'1 (UID 45231 RFC822 {size}', b'raw_mail'), b')', ...]
        # Jede Mail besteht aus einem Tuple (header, body) gefolgt von b')'.
        fetched = {}
        mismatch_count = 0
        first_mismatch_snippet: bytes | None = None
        for item in data:
            if isinstance(item, tuple) and len(item) == 2:
                header = item[0]
                if isinstance(header, bytes):
                    uid_match = re.search(rb"UID (\d+)", header)
                    if uid_match:
                        fetched[uid_match.group(1)] = item[1]
                    else:
                        mismatch_count += 1
                        if first_mismatch_snippet is None:
                            first_mismatch_snippet = header[:120]

        if mismatch_count:
            logger.warning(
                "IMAP UID-Parsing: %d Header ohne UID-Match in Batch (%d Mails). "
                "Erster Header-Snippet: %r",
                mismatch_count,
                len(batch_uids),
                first_mismatch_snippet,
            )

        for uid in batch_uids:
            results.append((uid, fetched.get(uid)))

        return results

    def _fetch_batch_with_reconnect(
        self,
        conn: imaplib.IMAP4_SSL | imaplib.IMAP4,
        batch_uids: list[bytes],
        folder: str,
    ) -> tuple[list[tuple[bytes, bytes | None]] | None, imaplib.IMAP4_SSL | imaplib.IMAP4]:
        """Batch-Fetch mit automatischem Reconnect bei Verbindungsverlust.

        Returns:
            (results, conn) — results ist Liste von (uid, raw|None),
            conn ist die (ggf. neue) Verbindung.
            results ist None wenn auch Reconnect fehlschlaegt.
        """
        try:
            conn = self._ensure_connection_ready(conn, folder=folder)
            results = self._fetch_batch(conn, batch_uids)
            return results, conn
        except Exception as exc:
            if not self._is_connection_error(exc):
                logger.warning("Batch-Fetch Fehler (kein Connection-Error): %s", exc)
                return [(uid, None) for uid in batch_uids], conn

            logger.warning(
                "Verbindungsverlust bei Batch-Fetch (%d Mails), reconnecte...", len(batch_uids)
            )
            try:
                conn = self._reconnect(conn, folder=folder)
            except ConnectionError:
                logger.error("IMAP-Reconnect endgueltig fehlgeschlagen — Scan wird abgebrochen")
                return None, conn

            # Retry nach Reconnect
            try:
                results = self._fetch_batch(conn, batch_uids)
                return results, conn
            except Exception as e2:
                logger.warning("Batch-Fetch auch nach Reconnect fehlgeschlagen: %s", e2)
                return [(uid, None) for uid in batch_uids], conn

    def reprocess_cached_mails(
        self,
        output_dir: str,
        cache_paths: list[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        control_callback: Callable[[], str | None] | None = None,
    ) -> dict:
        """Bekannte Bestandsmails erneut aufbereiten; Dateien danach ganz neu ingestieren.

        Registry und Scanmarker sind keine Migrationsfortschrittsmarker. Fehler
        bleiben pro Cachepfad offen; Rueckgabe bestaetigt nur die Cacheaufbereitung.
        """
        from kiara.ingestion.mail_migration import reprocess_cached_mails

        return reprocess_cached_mails(
            self, output_dir, cache_paths, progress_callback, control_callback
        )

    def _save_mail(
        self,
        raw: bytes,
        folder: str,
        output_dir: str,
        uid: str = "",
        *,
        cache_path: str | None = None,
        download_attachments: bool = True,
    ) -> list[str]:
        """Verarbeitet rohe Mail-Bytes -> Textdatei + Anhaenge."""
        msg = email.message_from_bytes(raw)

        # Header parsen
        subject = self._decode_header(msg.get("Subject", ""))
        sender = self._decode_header(msg.get("From", ""))
        to = self._decode_header(msg.get("To", ""))
        date_str = msg.get("Date", "")
        try:
            date = parsedate_to_datetime(date_str)
            date_formatted = date.strftime("%Y-%m-%d %H:%M")
            date_prefix = date.strftime("%Y%m%d")
        except (ValueError, TypeError):
            date_formatted = date_str
            date_prefix = "00000000"

        # Auswahl, Archivgrenzen und Konvertierung muessen vor allen Writes gelingen.
        selected_body = self._select_body(msg)
        validate_mail_archive(self._mail_archive, output_dir, selected_body.payload)
        body, body_metadata = selected_body.convert()

        # Mail als Textdatei schreiben
        safe_uid = self._safe_filename(str(uid)) if uid else ""
        safe_subject = self._safe_filename(subject)[:IMAP_MAX_FILENAME_LENGTH]
        mail_filename = (
            f"{date_prefix}_{safe_uid}_{safe_subject}.txt"
            if safe_uid
            else f"{date_prefix}_{safe_subject}.txt"
        )
        folder_key = self._folder_key(folder)
        mail_path = cache_path or os.path.join(output_dir, folder_key, mail_filename)
        mail_text = (
            f"Betreff: {subject}\n"
            f"Von: {sender}\n"
            f"An: {to}\n"
            f"Datum: {date_formatted}\n"
            f"Ordner: {folder}\n"
            f"---\n"
            f"{body}"
        )

        write_mail_cache(
            mail_path,
            mail_text,
            body=selected_body.payload,
            body_text=body,
            mime_type=selected_body.mime_type,
            charset=selected_body.charset,
            metadata=body_metadata,
            archive_config=self._mail_archive,
        )

        files = [mail_path]

        # Anhaenge extrahieren
        att_idx = 0
        if self._download_attachments and download_attachments:
            for part in msg.walk():
                if part.get_content_disposition() != "attachment":
                    continue
                filename = part.get_filename()
                if not filename:
                    continue
                filename = self._decode_header(filename)
                payload = part.get_payload(decode=True)
                if not payload or len(payload) > self._max_attachment_bytes:
                    continue

                att_idx += 1
                att_base = self._safe_filename(filename)
                att_name = (
                    f"{date_prefix}_{safe_uid}_{att_idx}_{att_base}"
                    if safe_uid
                    else f"{date_prefix}_{att_idx}_{att_base}"
                )
                att_path = os.path.join(output_dir, folder_key, "attachments", att_name)
                os.makedirs(os.path.dirname(att_path), exist_ok=True)
                with open(att_path, "wb") as f:
                    f.write(payload)
                files.append(att_path)

        return files

    @staticmethod
    def _safe_decode(payload: bytes, charset: str | None) -> str:
        try:
            return payload.decode(charset or "utf-8", errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_body(msg) -> str:
        """Extrahiert den Hauptbody; inhaltstragender Klartext hat Vorrang vor HTML."""
        body, _metadata = IMAPScanner._select_body(msg).convert()
        return body

    @staticmethod
    def _body_candidates(msg):
        """Nur Hauptbody-Zweige durchlaufen, keine Anhaenge/weitergeleiteten Mails."""
        if msg.get_content_disposition() == "attachment" or msg.get_filename() is not None:
            return
        content_type = msg.get_content_type()
        if msg.get_content_maintype() == "message":
            return
        if msg.is_multipart():
            children = msg.get_payload()
            if content_type == "multipart/related":
                start = msg.get_param("start")
                if start:
                    root_id = start.strip().strip("<>")
                    roots = [
                        part
                        for part in children
                        if str(part.get("Content-ID", "")).strip().strip("<>") == root_id
                    ]
                    if len(roots) != 1:
                        raise ValueError("multipart/related hat keinen eindeutigen start-Root")
                    children = roots
                else:
                    children = children[:1]
            for child in children:
                yield from IMAPScanner._body_candidates(child)
        elif content_type in ("text/plain", "text/html"):
            payload = msg.get_payload(decode=True)
            if payload is not None:
                yield _MailBody(payload, content_type, msg.get_content_charset() or "utf-8")

    @staticmethod
    def _select_body(msg) -> _MailBody:
        html_body = empty_body = None
        for body in IMAPScanner._body_candidates(msg):
            if body.mime_type == "text/plain":
                if IMAPScanner._safe_decode(body.payload, body.charset).strip():
                    return body
                if empty_body is None:
                    empty_body = body
            elif html_body is None:
                html_body = body
        return html_body or empty_body or _MailBody()

    @staticmethod
    def _decode_header(value: str) -> str:
        """Decodiert MIME-kodierte Header (=?UTF-8?Q?...?=)."""
        if not value:
            return ""
        parts = decode_header(value)
        result = []
        for text, charset in parts:
            if isinstance(text, bytes):
                result.append(IMAPScanner._safe_decode(text, charset))
            else:
                result.append(text)
        return " ".join(result)

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Erzeugt einen sicheren Dateinamen."""
        safe = re.sub(r"[^\w\s\-.]", "", name)
        return safe.strip()[:IMAP_MAX_FILENAME_LENGTH] or "unnamed"

    @staticmethod
    def _folder_key(folder: str) -> str:
        """Erzeugt einen kollisionsarmen Ordner-Key fuer Registry und Pfade."""
        raw = str(folder or "")
        safe = re.sub(r"[^\w\s\-.]+", "_", raw).strip(" ._-")[:80] or "folder"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{safe}_{digest}"

    @staticmethod
    def _decode_imap_utf7(name: str) -> str:
        """Dekodiert IMAP Modified UTF-7 Ordnernamen (RFC 3501 Section 5.1.3).

        IMAP Modified UTF-7 unterscheidet sich von Standard-UTF-7:
        - '&' leitet eine kodierte Sequenz ein, '-' beendet sie
        - '&-' ist ein literales '&'
        - Im Base64-Alphabet wird ',' statt '/' verwendet
        - Die kodierten Bytes sind UTF-16-BE

        Beispiele: '&AOQ-' -> 'ae', 'Entw&APw-rfe' -> 'Entwuerfe', '&-' -> '&'

        Bei fehlerhaften Sequenzen wird der Originaltext zurueckgegeben.
        """
        import base64

        result = []
        i = 0
        while i < len(name):
            if name[i] == "&":
                end = name.find("-", i + 1)
                if end == -1:
                    result.append(name[i:])
                    break
                if end == i + 1:
                    result.append("&")
                else:
                    encoded = name[i + 1 : end]
                    b64 = encoded.replace(",", "/")
                    padding = -len(b64) % 4
                    b64 += "=" * padding
                    try:
                        raw = base64.b64decode(b64)
                        result.append(raw.decode("utf-16-be"))
                    except Exception as exc:
                        logger.debug(
                            "UTF-7 Dekodierung fehlgeschlagen fuer %r: %s", name[i : end + 1], exc
                        )
                        result.append(name[i : end + 1])
                i = end + 1
            else:
                result.append(name[i])
                i += 1
        return "".join(result)

    @staticmethod
    def _parse_imap_list(raw_folders: list[bytes]) -> list[dict]:
        """Parst IMAP LIST-Antworten in strukturierte Dicts.

        Verarbeitet beliebige Delimiter (/, .) und NIL-Delimiter (flache
        Namespaces). Extrahiert Flags und gibt sowohl dekodierte als auch
        Original-Ordnernamen zurueck. Flags werden auf Kleinschreibung
        normalisiert fuer case-insensitive Vergleiche (RFC 3501).

        Unterstuetzt auch IMAP Literal-Antworten (Tuple-Format von imaplib),
        bei denen der Ordnername als separates bytes-Element im Tuple steht.

        Format: b'(\\HasChildren) "/" "INBOX"'
                b'(\\Noselect) NIL "FlatFolder"'
                (b'(\\HasChildren) "/" {11}', b'Sent Items')   # Literal
        """
        results = []
        pattern = re.compile(
            r"\(([^)]*)\)"  # (flags)
            r'\s+(?:"(.)"|NIL)'  # "delimiter" oder NIL
            r'\s+"?([^"]*)"?\s*$'  # "name" oder name
        )
        prefix_pattern = re.compile(
            r"\(([^)]*)\)"  # (flags)
            r'\s+(?:"(.)"|NIL)'  # "delimiter" oder NIL
        )
        for raw in raw_folders:
            if isinstance(raw, tuple):
                if len(raw) < 2:
                    continue
                prefix = raw[0].decode("utf-8", errors="replace")
                pm = prefix_pattern.search(prefix)
                if not pm:
                    continue
                flags = [f.lower() for f in pm.group(1).split()]
                delimiter = pm.group(2) or "/"
                raw_name = (
                    raw[1].decode("utf-8", errors="replace").strip()
                    if isinstance(raw[1], bytes)
                    else ""
                )
                if not raw_name:
                    continue
                name = IMAPScanner._decode_imap_utf7(raw_name)
                results.append(
                    {
                        "name": name,
                        "raw_name": raw_name,
                        "flags": flags,
                        "delimiter": delimiter,
                    }
                )
                continue
            elif isinstance(raw, bytes):
                line = raw.decode("utf-8", errors="replace")
            else:
                continue
            m = pattern.search(line)
            if not m:
                continue
            flags = [f.lower() for f in m.group(1).split()]
            delimiter = m.group(2) or "/"
            raw_name = m.group(3).strip()
            name = IMAPScanner._decode_imap_utf7(raw_name)
            results.append(
                {
                    "name": name,
                    "raw_name": raw_name,
                    "flags": flags,
                    "delimiter": delimiter,
                }
            )
        return results

    @staticmethod
    def _build_folder_tree(parsed: list[dict]) -> list[dict]:
        """Baut hierarchische Baumstruktur aus flacher Ordnerliste.

        Erzeugt synthetische Zwischen-Knoten ueber ALLE Ebenen hinweg wenn
        der Server z.B. 'INBOX/A/B/C' liefert, aber 'INBOX', 'INBOX/A' und
        'INBOX/A/B' fehlen.
        """
        if not parsed:
            return []
        delimiter = parsed[0].get("delimiter", "/")
        sorted_parsed = sorted(parsed, key=lambda f: f["name"].lower())

        root = []
        node_map = {}

        def _ensure_parent(path: str, raw_path: str) -> dict | None:
            """Stellt sicher, dass der Parent-Knoten existiert (rekursiv)."""
            parts = path.split(delimiter)
            if len(parts) <= 1:
                return None

            parent_path = delimiter.join(parts[:-1])
            if parent_path in node_map:
                return node_map[parent_path]

            raw_parts = raw_path.split(delimiter)
            raw_parent = delimiter.join(raw_parts[: len(parts) - 1])
            grandparent = _ensure_parent(parent_path, raw_parent)

            parent_parts = parent_path.split(delimiter)
            parent_display = parent_parts[-1] if parent_parts[-1] else parent_path
            parent_node = {
                "name": parent_display,
                "path": parent_path,
                "raw_path": raw_parent,
                "selectable": False,
                "children": [],
            }
            node_map[parent_path] = parent_node

            if grandparent:
                grandparent["children"].append(parent_node)
            else:
                root.append(parent_node)

            return parent_node

        for folder in sorted_parsed:
            path = folder["name"]
            raw_path = folder["raw_name"]
            selectable = "\\noselect" not in folder["flags"]
            parts = path.split(delimiter)
            display_name = parts[-1] if parts[-1] else path

            existing = node_map.get(path)
            if existing:
                existing["name"] = display_name
                existing["raw_path"] = raw_path
                existing["selectable"] = selectable
                continue

            node = {
                "name": display_name,
                "path": path,
                "raw_path": raw_path,
                "selectable": selectable,
                "children": [],
            }
            node_map[path] = node

            parent = _ensure_parent(path, raw_path)
            if parent:
                parent["children"].append(node)
            else:
                root.append(node)

        return root
