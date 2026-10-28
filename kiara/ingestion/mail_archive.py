"""Originalbody und Konvertierungsprovenienz ausserhalb des indexierten Mailcaches.

Pro kanonischem Cachepfad existiert genau eine aktuelle Archivdatei. Die Angaben
belegen die Konvertierung; sie behaupten keinen abgeschlossenen Ingest.
"""

import base64
import binascii
import hashlib
import json
import os
import tempfile
from pathlib import Path


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _settings(archive_config: dict) -> tuple[Path, int]:
    root = Path(archive_config.get("path", ""))
    limit = archive_config.get("max_body_bytes")
    if not root.is_absolute():
        raise ValueError("Mail-Archivpfad muss absolut sein")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Mail-Archiv max_body_bytes muss eine positive Ganzzahl sein")
    return root.resolve(), limit


def _archive_path(cache_path: str | Path, archive_root: Path) -> Path:
    key = _hash(os.fsencode(Path(cache_path).resolve()))
    return archive_root / key[:2] / f"{key}.json"


def validate_mail_archive(archive_config: dict | None, output_dir: str, body: bytes) -> None:
    """Konfiguration und Groessenlimit vor Konvertierung und Schreiboperationen pruefen."""
    if archive_config is None:
        return
    root, limit = _settings(archive_config)
    if root.is_relative_to(Path(output_dir).resolve()):
        raise ValueError("Mail-Archiv darf nicht im indexierten Mailcache liegen")
    if len(body) > limit:
        raise ValueError(
            f"Originalbody ueberschreitet Mail-Archiv-Limit ({len(body)} > {limit} bytes)"
        )


def _write_temp(path: Path, data: bytes) -> Path:
    """Vollstaendige Datei im Ziel-Dateisystem vorbereiten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_mail_cache(
    cache_path: str | Path,
    cache_text: str,
    *,
    body: bytes,
    body_text: str,
    mime_type: str,
    charset: str,
    metadata: dict,
    archive_config: dict | None,
) -> None:
    """TXT atomar ersetzen und erst danach seine Konvertierungsprovenienz bestaetigen.

    Beide Ausgaben werden vor dem ersten Ersetzen vorbereitet. Scheitert der
    Archiv-Commit regulaer, wird der vorherige Cache wiederhergestellt. Harte
    Abbrueche koennen einen Teilstand hinterlassen; die UID wird erst vom
    aufrufenden Scanner nach erfolgreicher Rueckkehr bestaetigt.
    """
    cache = Path(cache_path)
    cache_bytes = cache_text.encode("utf-8")
    body_bytes = body_text.encode("utf-8")
    if not cache_bytes.endswith(body_bytes):
        raise ValueError("Konvertierter Body ist nicht das Ende des Mailcaches")
    archive = None
    archive_bytes = None
    if archive_config is not None:
        root, limit = _settings(archive_config)
        if len(body) > limit:
            raise ValueError("Originalbody ueberschreitet Mail-Archiv-Limit")
        archive = _archive_path(cache, root)
        record = {
            "version": 1,
            "cache_path": str(cache.resolve()),
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_hash": _hash(body),
            "mime_type": mime_type,
            "charset": charset,
            "cache_hash": _hash(cache_bytes),
            "cache_body_start": len(cache_bytes) - len(body_bytes),
            "cache_prefix_base64": base64.b64encode(
                cache_bytes[: len(cache_bytes) - len(body_bytes)]
            ).decode("ascii"),
            "cache_prefix_hash": _hash(cache_bytes[: len(cache_bytes) - len(body_bytes)]),
            "metadata": metadata,
        }
        archive_bytes = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")

    staged_cache = staged_archive = rollback_cache = None
    try:
        if archive is not None:
            staged_archive = _write_temp(archive, archive_bytes)
            if cache.exists():
                rollback_cache = _write_temp(cache, cache.read_bytes())
        staged_cache = _write_temp(cache, cache_bytes)
        os.replace(staged_cache, cache)
        if archive is not None:
            try:
                os.replace(staged_archive, archive)
            except Exception:
                if rollback_cache is not None:
                    os.replace(rollback_cache, cache)
                else:
                    cache.unlink(missing_ok=True)
                raise
    finally:
        for temporary in (staged_cache, staged_archive, rollback_cache):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _read_original_record(cache_path: str | Path, archive_config: dict | None):
    """Original und seine Provenienz unabhaengig vom austauschbaren Cache pruefen."""
    if archive_config is None:
        return None
    root, limit = _settings(archive_config)
    archive = _archive_path(cache_path, root)
    if not archive.exists():
        return None
    try:
        record = json.loads(archive.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("version") != 1:
            raise ValueError("Unbekanntes Mail-Archivformat")
        if record.get("cache_path") != str(Path(cache_path).resolve()):
            raise ValueError("Mail-Archiv verweist auf anderen Cachepfad")
        body = base64.b64decode(record["body_base64"], validate=True)
        if len(body) > limit or record.get("body_hash") != _hash(body):
            raise ValueError("Originalbody-Hash oder Mail-Archiv-Limit verletzt")
        metadata = record["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Mail-Provenienz enthaelt keine Metadaten")
        mime_type = record["mime_type"]
        charset = record["charset"]
        if mime_type not in ("text/plain", "text/html") or not isinstance(charset, str):
            raise ValueError("Ungueltige MIME-/Charset-Angaben im Mail-Archiv")
        if metadata.get("source_mime_type") != mime_type:
            raise ValueError("MIME-Typ widerspricht Mail-Provenienz")
        try:
            decoded = body.decode(charset, errors="replace")
        except LookupError:
            decoded = body.decode("utf-8", errors="replace")
        prefix = "html" if mime_type == "text/html" else "mail"
        if metadata.get(f"{prefix}_source_hash") != _hash(decoded.encode("utf-8")):
            raise ValueError("Dekodierter Originalbody-Hash widerspricht Mail-Provenienz")
        expected_parser = "html-to-markdown" if mime_type == "text/html" else "mail-plain"
        expected_format = "markdown" if mime_type == "text/html" else "text"
        if (
            metadata.get("parser") != expected_parser
            or metadata.get("content_format") != expected_format
        ):
            raise ValueError("Parser/Format widerspricht Mail-Provenienz")
        for field in ("parser_revision", "profile_revision", "output_hash"):
            if (
                not isinstance(metadata.get(f"{prefix}_{field}"), str)
                or not metadata[f"{prefix}_{field}"]
            ):
                raise ValueError("Unvollstaendige Mail-Konvertierungsprovenienz")
        return record, body
    except (KeyError, TypeError, UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError(f"Inkonsistente Mail-Provenienz fuer {cache_path}: {exc}") from exc


def _verified_cache_bytes(cache_path: str | Path, record: dict) -> bytes:
    cache_bytes = Path(cache_path).read_bytes()
    if record.get("cache_hash") != _hash(cache_bytes):
        raise ValueError("Mail-Cachehash stimmt nicht mit Provenienz ueberein")
    body_start = record.get("cache_body_start")
    prefix = "html" if record["mime_type"] == "text/html" else "mail"
    if (
        isinstance(body_start, bool)
        or not isinstance(body_start, int)
        or not 0 <= body_start <= len(cache_bytes)
        or record["metadata"][f"{prefix}_output_hash"] != _hash(cache_bytes[body_start:])
    ):
        raise ValueError("Konvertierter Body-Hash widerspricht Mail-Provenienz")
    if "cache_prefix_base64" in record and _verified_prefix(record) != cache_bytes[:body_start]:
        raise ValueError("Mail-Cacheprefix widerspricht Provenienz")
    return cache_bytes


def _verified_prefix(record: dict) -> bytes:
    prefix = base64.b64decode(record["cache_prefix_base64"], validate=True)
    if record.get("cache_prefix_hash") != _hash(prefix) or len(prefix) != record.get(
        "cache_body_start"
    ):
        raise ValueError("Mail-Cacheprefix-Hash widerspricht Provenienz")
    prefix.decode("utf-8")
    return prefix


def read_mail_original(cache_path: str | Path, archive_config: dict | None) -> dict | None:
    """Geprueftes Original fuer explizite Wiederaufbereitung liefern.

    Der aktuelle Cache darf nach einem harten Abbruch inkonsistent sein. Nur
    ein separat gepruefter archivierter Prefix erlaubt dann die Rekonstruktion.
    Fehlt er und ist der Cache unpruefbar, muss die bekannte UID neu gelesen werden.
    """
    original = _read_original_record(cache_path, archive_config)
    if original is None:
        return None
    record, body = original
    try:
        if "cache_prefix_base64" in record:
            prefix = _verified_prefix(record)
        else:
            try:
                cache_bytes = _verified_cache_bytes(cache_path, record)
            except ValueError:
                return None
            prefix = cache_bytes[: record["cache_body_start"]]
        return {
            "body": body,
            "prefix": prefix.decode("utf-8"),
            "mime_type": record["mime_type"],
            "charset": record["charset"],
        }
    except (KeyError, TypeError, UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError(f"Inkonsistente Mail-Provenienz fuer {cache_path}: {exc}") from exc


def read_mail_metadata(
    cache_path: str | Path, cache_text: str, archive_config: dict | None
) -> dict:
    """Nur vollstaendig konsistente Cache-Provenienz zum normalen Ingest zulassen."""
    original = _read_original_record(cache_path, archive_config)
    if original is None:
        return {}
    record, body = original
    try:
        cache_bytes = _verified_cache_bytes(cache_path, record)
        decoded_cache = cache_bytes.decode("utf-8")
        normalized_cache = decoded_cache.replace("\r\n", "\n").replace("\r", "\n")
        if cache_text not in (decoded_cache, normalized_cache):
            raise ValueError("Geladener Mailtext stimmt nicht mit Cachebytes ueberein")
        return {
            **record["metadata"],
            "mail_body_hash": record["body_hash"],
            "mail_body_charset": record["charset"],
            "mail_body_bytes": len(body),
            "mail_cache_hash": record["cache_hash"],
        }
    except (KeyError, TypeError, UnicodeError, binascii.Error, ValueError) as exc:
        raise ValueError(f"Inkonsistente Mail-Provenienz fuer {cache_path}: {exc}") from exc
