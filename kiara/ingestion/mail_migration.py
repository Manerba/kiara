"""Explizite Wiederaufbereitung bekannter Mailcache-Dateien ohne Scanfortschritt."""

import email
import json
import logging
import os
import re
from pathlib import Path

from kiara.ingestion.mail_archive import (
    read_mail_metadata,
    read_mail_original,
    validate_mail_archive,
    write_mail_cache,
)

logger = logging.getLogger(__name__)

_IMAP_FLAG = rb'\\?[^\x00-\x20\x7f-\xff(){%*"\\\]]+'
_FLAG_NOTIFICATION_ATTRIBUTE = re.compile(
    rb"FLAGS \((?P<flags>(?:" + _IMAP_FLAG + rb"(?: " + _IMAP_FLAG + rb")*)?)\)"
    rb"|UID (?P<uid>[1-9][0-9]*)|MODSEQ \((?P<modseq>[1-9][0-9]*)\)",
    re.IGNORECASE,
)


class _MissingMailOriginal(ValueError):
    """Die gepruefte UID wurde vom Server nicht geliefert."""


class _EmptyMailBody(ValueError):
    """Der ausgewaehlte Hauptbody enthaelt keinen lesbaren Text."""


def _number(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"Fehlende oder ungueltige {label}")
    if int(value) > 4294967295:
        raise ValueError(f"Ungueltige {label}")
    return value


def _is_flag_notification(data: bytes) -> bool:
    """Nur vollstaendige FLAGS-Mitteilungen ohne Nachrichteninhalt uebergehen.

    imaplib liefert unangeforderte FETCH-Mitteilungen als Bytes neben den
    Body-Tupeln. UID und MODSEQ sind zulaessige Begleitattribute; andere oder
    doppelte Attribute bleiben ein Fehler.
    """
    response = re.fullmatch(rb"([1-9][0-9]*) \((.*)\)", data)
    if response is None:
        return False
    try:
        _number(response[1].decode("ascii"), "IMAP-Sequenznummer")
        attributes = response[2]
        seen = set()
        position = 0
        while position < len(attributes):
            match = _FLAG_NOTIFICATION_ATTRIBUTE.match(attributes, position)
            if match is None or match.lastgroup in seen:
                return False
            seen.add(match.lastgroup)
            if match.lastgroup == "uid":
                _number(match["uid"].decode("ascii"), "UID der Flagmitteilung")
            elif match.lastgroup == "modseq" and int(match["modseq"]) > 18446744073709551615:
                return False
            position = match.end()
            if position < len(attributes):
                if attributes[position : position + 1] != b" " or position + 1 == len(attributes):
                    return False
                position += 1
        return "flags" in seen
    except ValueError:
        return False


def _path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Ungueltiger Mailcachepfad")
    return Path(os.path.abspath(root / value))


def _registry_entries(root: Path) -> dict:
    """Fehlende/defekte Registry niemals als erfolgreichen leeren Bestand behandeln."""
    registry = json.loads((root / ".uid_registry.json").read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Ungueltige UID-Registry")
    entries = {}
    for folder_key, folder in registry.items():
        if not isinstance(folder, dict) or not isinstance(folder.get("uids", {}), dict):
            raise ValueError("Ungueltiger Ordner in UID-Registry")
        for uid, relative in folder.get("uids", {}).items():
            path = _path(root, relative)
            entries.setdefault(str(path), []).append(
                (folder_key, uid, folder.get("uidvalidity"), relative)
            )
    return entries


def _check_path(root: Path, path: Path, entry: tuple) -> None:
    folder_key, _uid, _validity, relative = entry
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not path.is_relative_to(root)
        or path.resolve() != path
        or not path.is_file()
        or path.suffix.lower() != ".txt"
        or path.parent != root / folder_key
    ):
        raise ValueError("Mailcachepfad ist keine bestehende Registry-Mail unter output_dir")


def _entries_by_uid(entries: dict) -> dict:
    """Nur eindeutige Cachepfade als Originalidentitaetsbeleg indexieren."""
    return {
        (references[0][0], references[0][1]): (Path(path), references[0])
        for path, references in entries.items()
        if len(references) == 1
    }


def _folder_name(
    scanner, path: Path, folder_key: str, *, uid=None, expected_validity=None, registry_by_uid=None
) -> str:
    candidates = set(scanner._folders)
    configured = [name for name in candidates if scanner._folder_key(name) == folder_key]
    if len(configured) == 1:
        return configured[0]
    # Der unveraenderte Mailheader kann auch nicht mehr konfigurierte Ordner belegen.
    header_folders = []
    header_complete = False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.rstrip("\r\n") == "---":
                header_complete = True
                break
            if line.startswith("Ordner: "):
                header_folders.append(line[len("Ordner: ") :].rstrip("\r\n"))
    candidates.update(header_folders)
    matches = [name for name in candidates if scanner._folder_key(name) == folder_key]
    if len(matches) == 1:
        return matches[0]
    # Vor 58f486e7 wurde ausschliesslich '/' durch '_' ersetzt. Der alte
    # Schluessel allein ist wegen Kollisionen kein Originalordnerbeleg.
    if not matches and header_complete and len(header_folders) == 1 and registry_by_uid is not None:
        folder = header_folders[0]
        if folder and folder.replace("/", "_") == folder_key:
            _number(uid, "UID")
            _number(expected_validity, "UIDVALIDITY der Registry")
            canonical = registry_by_uid.get((scanner._folder_key(folder), uid))
            if canonical is not None:
                canonical_path, canonical_entry = canonical
                if canonical_entry[2] == expected_validity:
                    _check_path(path.parent.parent, canonical_path, canonical_entry)
                    if re.match(r"^[0-9]{8}_" + re.escape(uid) + "_", canonical_path.name):
                        return folder
    raise ValueError("Urspruenglicher IMAP-Ordner ist nicht eindeutig belegt")


class _RefetchSession:
    def __init__(self, scanner):
        self.scanner = scanner
        self.conn = None

    def fetch(self, folder: str, uid: str, expected_validity: str) -> bytes:
        scanner = self.scanner
        _number(uid, "UID")
        _number(expected_validity, "UIDVALIDITY der Registry")
        if self.conn is None:
            self.conn = scanner.connect()
        for attempt in range(2):
            try:
                self.conn = scanner._ensure_connection_ready(self.conn)
                status, _ = self.conn.select(folder, readonly=True)
                if status != "OK":
                    raise ValueError(f"IMAP-Ordner nicht selektierbar: {folder}")
                _kind, values = self.conn.response("UIDVALIDITY")
                value = values[0] if values and len(values) == 1 else None
                actual = value.decode("ascii") if isinstance(value, bytes) else value
                if _number(actual, "UIDVALIDITY des Servers") != expected_validity:
                    raise ValueError("UIDVALIDITY stimmt nicht mit der Registry ueberein")
                return self._fetch_original(uid)
            except Exception as exc:
                if attempt or not scanner._is_connection_error(exc):
                    raise
                # Nach jedem Reconnect SELECT und UIDVALIDITY erneut pruefen.
                self.conn = scanner._reconnect(self.conn)
        raise RuntimeError("IMAP-Originalabruf ohne Ergebnis")

    def _fetch_original(self, uid: str) -> bytes:
        """Nur OK ohne Nachricht belegt ein fehlendes Original, kein FETCH-Fehler."""
        encoded_uid = uid.encode("ascii")
        status, data = self.conn.uid("fetch", encoded_uid, "(UID BODY.PEEK[])")
        if status != "OK":
            raise ValueError(f"IMAP-Originalabruf UID {uid} abgewiesen: {status}")
        messages = []
        for item in data or []:
            if not isinstance(item, tuple):
                if item is None or (isinstance(item, bytes) and item.strip() in (b"", b")")):
                    continue
                if isinstance(item, bytes) and _is_flag_notification(item):
                    continue
                raise ValueError("Unvollstaendige Antwort auf IMAP-Originalabruf")
            if len(item) != 2 or not isinstance(item[0], bytes):
                raise ValueError("Ungueltige Antwort auf IMAP-Originalabruf")
            found_uids = re.findall(rb"\bUID ([0-9]+)\b", item[0], re.IGNORECASE)
            if found_uids != [encoded_uid]:
                raise ValueError("UID im Originalabruf stimmt nicht eindeutig mit Auftrag ueberein")
            if not isinstance(item[1], bytes) or not item[1]:
                raise ValueError("Originalabruf enthaelt keinen vollstaendigen Mailinhalt")
            messages.append(item[1])
        if not messages:
            raise _MissingMailOriginal(f"Originalmail UID {uid} nicht abrufbar")
        if len(messages) != 1:
            raise ValueError("Originalabruf enthaelt mehrere Nachrichten fuer dieselbe UID")
        return messages[0]

    def close(self):
        if self.conn is not None:
            try:
                self.conn.logout()
            except Exception:
                pass


def reprocess_cached_mails(
    scanner,
    output_dir: str,
    cache_paths=None,
    progress_callback=None,
    control_callback=None,
) -> dict:
    """Nur Cacheaufbereitung melden; erfolgreicher Ingest bleibt Aufgabe des Aufrufers.

    Fehler sind pro absolutem Cachepfad erneut bearbeitbar. Cancel oder Shutdown
    beendet vor der naechsten Datei; ersetzte Dateien bleiben in ``files`` enthalten.
    Callbackfehler propagieren, damit die Jobsteuerung nicht verschluckt wird.
    """
    from kiara.ingestion.imap_scanner import _MailBody

    root = Path(output_dir).resolve()
    entries = _registry_entries(root)
    registry_by_uid = _entries_by_uid(entries)
    paths = list(entries) if cache_paths is None else [str(_path(root, p)) for p in cache_paths]
    paths = list(dict.fromkeys(paths))
    result = {"files": [], "errors": {}, "cancelled": False}
    session = _RefetchSession(scanner)
    try:
        for index, path_text in enumerate(paths):
            if control_callback and control_callback() in ("cancel", "shutdown"):
                result["cancelled"] = True
                break
            try:
                registered = entries.get(path_text, [])
                if len(registered) != 1:
                    raise ValueError("Mailcachepfad ist nicht eindeutig in der UID-Registry")
                entry = registered[0]
                path = Path(path_text)
                _check_path(root, path, entry)
                validate_mail_archive(scanner._mail_archive, str(root), b"")
                original = read_mail_original(path, scanner._mail_archive)
                if original is not None:
                    selected = _MailBody(
                        original["body"], original["mime_type"], original["charset"]
                    )
                    body, metadata = selected.convert()
                    _check_path(root, path, entry)
                    write_mail_cache(
                        path,
                        original["prefix"] + body,
                        body=selected.payload,
                        body_text=body,
                        mime_type=selected.mime_type,
                        charset=selected.charset,
                        metadata=metadata,
                        archive_config=scanner._mail_archive,
                    )
                else:
                    folder_key, uid, validity, _relative = entry
                    folder = _folder_name(
                        scanner,
                        path,
                        folder_key,
                        uid=uid,
                        expected_validity=validity,
                        registry_by_uid=registry_by_uid,
                    )
                    raw = session.fetch(folder, uid, validity)
                    _check_path(root, path, entry)
                    scanner._save_mail(
                        raw,
                        folder,
                        str(root),
                        uid,
                        cache_path=path_text,
                        download_attachments=False,
                    )
                result["files"].append(path_text)
            except Exception as exc:
                result["errors"][path_text] = str(exc)
                logger.warning("Mail-Wiederaufbereitung fehlgeschlagen fuer %s: %s", path_text, exc)
            if progress_callback:
                progress_callback(index + 1, len(paths))
    finally:
        session.close()
    return result


def _cache_prefix(path: Path, folder: str) -> str:
    """Vorhandene Mailheader erhalten, statt sie aus neuem MIME zu ersetzen."""
    lines = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            lines.append(line)
            if line.rstrip("\r\n") == "---":
                folders = [
                    item[len("Ordner: ") :].rstrip("\r\n")
                    for item in lines[:-1]
                    if item.startswith("Ordner: ")
                ]
                if folders == [folder] and lines[0].startswith("Betreff: "):
                    return "".join(lines)
                break
    raise ValueError("Mailcache hat keinen eindeutigen bestehenden Headerprefix")


def _mail_cache_is_current(
    config, path: Path, archive_config, original_prefix: str, stored_hashes, hash_mode
) -> bool:
    from kiara.ingestion._file_utils import compute_file_hash
    from kiara.ingestion.processing_revision import processing_file_hash

    if str(path) not in stored_hashes:
        return False
    try:
        # Ein geprueftes Original darf einen nach Abbruch defekten Cache reparieren.
        # Ein solcher Cache ist auch bei zufaellig passendem DB-Hash nicht aktuell.
        cache_text = path.read_bytes().decode("utf-8")
        metadata = read_mail_metadata(path, cache_text, archive_config)
    except (ValueError, UnicodeError):
        return False
    if not metadata:
        return False
    if not cache_text[len(original_prefix) :].strip():
        raise _EmptyMailBody("HTML-Hauptbody ergibt keinen lesbaren Text")
    actual = processing_file_hash(config, str(path), compute_file_hash(str(path), hash_mode))
    return stored_hashes[str(path)] == actual


def prepare_html_mails(
    scanner,
    output_dir: str,
    cache_paths,
    config: dict,
    *,
    mode: str = "outdated",
    stored_hashes: dict | None = None,
    hash_mode: str = "content",
    progress_callback=None,
    control_callback=None,
) -> dict:
    """Bekannte Mails aus Originalen klassifizieren und nur HTML neu aufbereiten.

    ``files`` bestaetigt nur erfolgreiche Cacheaufbereitung, keinen Ingest.
    ``html_paths`` enthaelt auch bereits aktuelle oder abgewiesene HTML-Bodies.
    Plaintext bleibt bytegleich, auch wenn seine Klassifizierung Netzabruf brauchte.
    Callbackfehler propagieren; Cancel/Shutdown stoppt vor Abruf oder Schreiben.
    """
    from kiara.config import get_mail_archive_config
    from kiara.ingestion.imap_scanner import _MailBody

    if mode not in ("outdated", "all"):
        raise ValueError("HTML-Mailmodus muss outdated oder all sein")
    if hash_mode not in ("content", "mtime"):
        raise ValueError("Ungueltiger Mail-Hashmodus")
    if scanner._mail_archive != get_mail_archive_config(config):
        raise ValueError("Scanner und Ingest muessen dasselbe Mail-Originalarchiv verwenden")
    root = Path(output_dir).resolve()
    validate_mail_archive(scanner._mail_archive, str(root), b"")
    entries = _registry_entries(root)
    registry_by_uid = _entries_by_uid(entries)
    paths = list(entries) if cache_paths is None else [str(_path(root, p)) for p in cache_paths]
    paths = list(dict.fromkeys(paths))
    stored_hashes = stored_hashes or {}
    result = {
        "files": [],
        "html_paths": [],
        "errors": {},
        "error_kinds": {},
        "cancelled": False,
        "skipped": {"plain": 0, "current": 0},
        "examined": 0,
    }
    session = _RefetchSession(scanner)

    def stopped():
        if control_callback and control_callback() in ("cancel", "shutdown"):
            result["cancelled"] = True
            return True
        return False

    def failed(path_text, exc):
        kind = "preparation"
        if isinstance(exc, _MissingMailOriginal):
            kind = "missing_original"
        elif isinstance(exc, _EmptyMailBody):
            kind = "empty_body"
        result["errors"][path_text] = str(exc)
        result["error_kinds"][path_text] = kind
        logger.warning("HTML-Mailaufbereitung fehlgeschlagen fuer %s: %s", path_text, exc)

    def progress(index):
        result["examined"] = index + 1
        if progress_callback:
            progress_callback(index + 1, len(paths))

    try:
        for index, path_text in enumerate(paths):
            if stopped():
                break
            try:
                registered = entries.get(path_text, [])
                if len(registered) != 1:
                    raise ValueError("Mailcachepfad ist nicht eindeutig in der UID-Registry")
                entry = registered[0]
                path = Path(path_text)
                _check_path(root, path, entry)
                original = read_mail_original(path, scanner._mail_archive)
                if original is None:
                    folder_key, uid, validity, _relative = entry
                    folder = _folder_name(
                        scanner,
                        path,
                        folder_key,
                        uid=uid,
                        expected_validity=validity,
                        registry_by_uid=registry_by_uid,
                    )
                else:
                    selected = _MailBody(
                        original["body"], original["mime_type"], original["charset"]
                    )
            except Exception as exc:
                failed(path_text, exc)
                progress(index)
                continue

            if original is None:
                if stopped():
                    break
                try:
                    raw = session.fetch(folder, uid, validity)
                    selected = scanner._select_body(email.message_from_bytes(raw))
                except Exception as exc:
                    failed(path_text, exc)
                    progress(index)
                    continue

            prepared = None
            try:
                if selected.mime_type == "text/html":
                    result["html_paths"].append(path_text)
                if not scanner._safe_decode(selected.payload, selected.charset).strip():
                    raise _EmptyMailBody("Originalmail hat keinen lesbaren Hauptbody")
                if selected.mime_type == "text/plain":
                    result["skipped"]["plain"] += 1
                elif (
                    original is not None
                    and mode == "outdated"
                    and _mail_cache_is_current(
                        config,
                        path,
                        scanner._mail_archive,
                        original["prefix"],
                        stored_hashes,
                        hash_mode,
                    )
                ):
                    result["skipped"]["current"] += 1
                else:
                    validate_mail_archive(scanner._mail_archive, str(root), selected.payload)
                    body, metadata = selected.convert()
                    if not body.strip():
                        raise _EmptyMailBody("HTML-Hauptbody ergibt keinen lesbaren Text")
                    prefix = (
                        original["prefix"] if original is not None else _cache_prefix(path, folder)
                    )
                    prepared = (prefix + body, body, metadata)
            except Exception as exc:
                failed(path_text, exc)

            if prepared is not None:
                if stopped():
                    break
                try:
                    _check_path(root, path, entry)
                    cache_text, body, metadata = prepared
                    write_mail_cache(
                        path,
                        cache_text,
                        body=selected.payload,
                        body_text=body,
                        mime_type=selected.mime_type,
                        charset=selected.charset,
                        metadata=metadata,
                        archive_config=scanner._mail_archive,
                    )
                    result["files"].append(path_text)
                except Exception as exc:
                    failed(path_text, exc)
            progress(index)
    finally:
        session.close()
    return result
