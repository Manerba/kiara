"""Browse-Helper fuer Connector-Capabilities."""

from __future__ import annotations

import email.utils
import os
import posixpath
import re
import subprocess
import tempfile
from urllib.parse import quote, unquote, urlparse
from xml.etree import ElementTree as ET

import httpx

from kiara.config import DEFAULT_TIMEOUTS

_MAX_WEBDAV_RESPONSE_BYTES = 2 * 1024 * 1024
_SMB_PATH_RE = re.compile(r"^[a-zA-Z0-9\\\s.\-_()\[\]&+,'@#=äöüÄÖÜß]+$")
_XML_FORBIDDEN_DECL_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


class ConnectorBrowseError(Exception):
    """Fehlervertrag fuer Connector-Browse-Implementierungen."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        reason: str = "bad_input",
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.reason = reason
        super().__init__(message)


def parent_path(path: str | None) -> str | None:
    if not path or path in (".", "/"):
        return None
    parts = path.strip("/").split("/")[:-1]
    return "/".join(parts) if parts else ""


def _contains_control(value: str) -> bool:
    return "\x00" in value or "\r" in value or "\n" in value


def normalize_browse_path(path: str | None) -> str:
    """Normalisiert relative Browse-Pfade fail-closed."""

    if path is None:
        return ""
    raw = str(path).replace("\\", "/")
    if _contains_control(raw):
        raise ConnectorBrowseError("Ungueltiger Pfad", 400, "bad_input")
    if raw in ("", ".", "/"):
        return ""
    if raw.startswith("/"):
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")

    parts = [part for part in raw.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")
    if not parts:
        return ""

    normalized = posixpath.normpath("/".join(parts))
    if normalized in ("", "."):
        return ""
    if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")
    return normalized


def _normalize_config_root(path: str | None) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if _contains_control(raw):
        raise ConnectorBrowseError("Ungueltiger Pfad", 400, "bad_input")
    if raw in ("", ".", "/"):
        return ""

    normalized = posixpath.normpath(raw.strip("/"))
    if normalized in ("", "."):
        return ""
    if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")
    return normalized


def _validate_percent_decoded_path(path: str | None, *, allow_absolute_root: bool) -> None:
    decoded = unquote(str(path or "").replace("\\", "/"))
    if _contains_control(decoded):
        raise ConnectorBrowseError("Ungueltiger Pfad", 400, "bad_input")
    if decoded in ("", ".", "/"):
        return
    if not allow_absolute_root and decoded.startswith("/"):
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")
    decoded_parts = [part for part in decoded.strip("/").split("/") if part and part != "."]
    if any(part == ".." for part in decoded_parts):
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")
    normalized = posixpath.normpath(decoded.strip("/") if allow_absolute_root else decoded)
    if normalized in ("", "."):
        return
    if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")


def _normalize_webdav_config_root(path: str | None) -> str:
    _validate_percent_decoded_path(path, allow_absolute_root=True)
    return _normalize_config_root(path)


def _normalize_webdav_browse_path(path: str | None) -> str:
    _validate_percent_decoded_path(path, allow_absolute_root=False)
    return normalize_browse_path(path)


def _join_logical_paths(root: str, path: str) -> str:
    if root and path:
        return normalize_browse_path(posixpath.join(root, path))
    return normalize_browse_path(root or path)


def _sort_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries, key=lambda item: (not item.get("is_dir", False), item.get("name", "").lower())
    )


def _timestamp_from_http_date(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return parsed.timestamp()


def list_local_directory(
    base_path: str,
    path: str | None,
    *,
    directories_only: bool = False,
    allow_absolute: bool = False,
    max_entries: int | None = None,
) -> dict:
    """Listet ein lokales Verzeichnis innerhalb von ``base_path``."""

    if not base_path:
        raise ConnectorBrowseError("Verzeichnis nicht gefunden", 404, "not_found")

    real_base = os.path.realpath(base_path)
    raw_path = "" if path is None else str(path)
    if _contains_control(raw_path):
        raise ConnectorBrowseError("Ungueltiger Pfad", 400, "bad_input")

    if allow_absolute and os.path.isabs(raw_path):
        target = os.path.realpath(raw_path)
        if not (target == real_base or target.startswith(real_base + os.sep)):
            raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")
        normalized = os.path.relpath(target, real_base)
        normalized = "" if normalized == "." else normalize_browse_path(normalized)
    else:
        normalized = normalize_browse_path(path)
        target = os.path.realpath(os.path.join(real_base, normalized))
        if not (target == real_base or target.startswith(real_base + os.sep)):
            raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")

    if not os.path.isdir(target):
        raise ConnectorBrowseError("Verzeichnis nicht gefunden", 404, "not_found")

    entries: list[dict] = []
    scanned = 0
    truncated = False
    try:
        with os.scandir(target) as iterator:
            for entry in iterator:
                scanned += 1
                if max_entries is not None and scanned > max_entries:
                    truncated = True
                    break
                try:
                    is_dir = entry.is_dir()
                    if directories_only and not is_dir:
                        continue
                    stat = entry.stat()
                    entry_path = (
                        posixpath.join(normalized, entry.name) if normalized else entry.name
                    )
                    entries.append(
                        {
                            "name": entry.name,
                            "path": entry_path,
                            "is_dir": is_dir,
                            "size": 0 if is_dir else stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )
                except (PermissionError, OSError):
                    continue
    except PermissionError as exc:
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden") from exc
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ConnectorBrowseError("Verzeichnis nicht gefunden", 404, "not_found") from exc

    return {"entries": _sort_entries(entries), "truncated": truncated}


def _smb_logical_path(config: dict, path: str | None) -> str:
    sub_path = _normalize_config_root(config.get("sub_path"))
    browse_path = normalize_browse_path(path)
    return _join_logical_paths(sub_path, browse_path)


def _parse_smbclient_stdout(stdout: str, logical_path: str, directories_only: bool) -> list[dict]:
    entries: list[dict] = []
    for line in stdout.splitlines():
        match = re.match(
            r"^\s*(?P<name>.+?)\s{2,}(?P<attrs>[A-Z]+)\s+(?P<size>\d+)\s+(?P<date>.+)$", line
        )
        if not match:
            continue
        name = match.group("name").strip()
        if name in (".", ".."):
            continue
        attrs = match.group("attrs")
        is_dir = "D" in attrs
        if directories_only and not is_dir:
            continue

        entry = {
            "name": name,
            "is_dir": is_dir,
        }
        if not directories_only:
            entry["size"] = int(match.group("size"))
            entry["path"] = posixpath.join(logical_path, name) if logical_path else name
            modified = _timestamp_from_http_date(match.group("date"))
            if modified is not None:
                entry["modified"] = modified
        entries.append(entry)
    return _sort_entries(entries)


def browse_smbclient(config: dict, path: str | None, *, directories_only: bool = True) -> dict:
    """Browse einer SMB-Freigabe via smbclient ohne Mount."""

    server = str(config.get("server") or "")
    share = str(config.get("share") or "")
    username = str(config.get("username") or "")
    password = str(config.get("password") or "")
    domain = str(config.get("domain") or "")

    if not server or not share:
        raise ConnectorBrowseError("Server und Freigabe erforderlich", 400, "bad_input")
    if _contains_control(server) or _contains_control(share):
        raise ConnectorBrowseError("Ungueltige Verbindungsdaten", 400, "bad_input")
    for value in (username, password, domain):
        if _contains_control(value):
            raise ConnectorBrowseError("Ungueltige Credentials", 400, "bad_input")

    logical_path = _smb_logical_path(config, path)
    smb_browse_path = logical_path.replace("/", "\\")
    if smb_browse_path and not _SMB_PATH_RE.match(smb_browse_path):
        raise ConnectorBrowseError("Zugriff verweigert", 403, "forbidden")

    smb_path = f"//{server}/{share}"
    smb_cmd = f'cd "{smb_browse_path}"; ls' if smb_browse_path else "ls"
    cred_lines = f"username={username}\npassword={password}\n"
    if domain:
        cred_lines += f"domain={domain}\n"

    fd: int | None = None
    cred_path: str | None = None
    try:
        fd, cred_path = tempfile.mkstemp(prefix="kiara-smb-browse-", suffix=".cred")
        os.write(fd, cred_lines.encode())
        os.close(fd)
        fd = None
        os.chmod(cred_path, 0o600)
        result = subprocess.run(
            ["smbclient", smb_path, "--authentication-file", cred_path, "-c", smb_cmd],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_umount"],
        )
    except FileNotFoundError as exc:
        raise ConnectorBrowseError("smbclient nicht installiert", 502, "not_installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise ConnectorBrowseError("Timeout", 502, "timeout") from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if cred_path:
            try:
                os.unlink(cred_path)
            except OSError:
                pass

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "Verbindung fehlgeschlagen"
        raise ConnectorBrowseError(error_msg, 502, "bad_response")

    return {"entries": _parse_smbclient_stdout(result.stdout, logical_path, directories_only)}


def _webdav_target_url(config: dict, path: str | None) -> tuple[str, str]:
    url = str(config.get("url") or "")
    if not url:
        raise ConnectorBrowseError("WebDAV-URL erforderlich", 400, "bad_input")
    if _contains_control(url):
        raise ConnectorBrowseError("Ungueltige Verbindungsdaten", 400, "bad_input")

    base_path = _normalize_webdav_config_root(config.get("base_path"))
    browse_path = _normalize_webdav_browse_path(path)
    logical_path = _join_logical_paths(base_path, browse_path)

    base_url = url.rstrip("/")
    if logical_path:
        quoted_path = quote(logical_path.strip("/"), safe="/")
        return f"{base_url}/{quoted_path}/", browse_path
    return f"{base_url}/", browse_path


def _webdav_prop_text(prop: ET.Element, name: str) -> str | None:
    element = prop.find(f"{{DAV:}}{name}")
    return element.text if element is not None else None


def _parse_webdav_entries(
    content: bytes, target_url: str, browse_path: str, directories_only: bool
) -> list[dict]:
    if len(content) > _MAX_WEBDAV_RESPONSE_BYTES:
        raise ConnectorBrowseError("WebDAV-Antwort zu gross", 502, "bad_response")
    if _XML_FORBIDDEN_DECL_RE.search(content):
        raise ConnectorBrowseError("Ungueltige XML-Antwort", 502, "bad_response")

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ConnectorBrowseError("Ungueltige XML-Antwort", 502, "bad_response") from exc

    target_path = unquote(urlparse(target_url).path).rstrip("/")
    entries: list[dict] = []
    for response_el in root.findall("{DAV:}response"):
        href_el = response_el.find("{DAV:}href")
        if href_el is None:
            continue
        href = href_el.text or ""
        href_path = unquote(urlparse(href).path).rstrip("/")
        if href_path == target_path:
            continue
        name = href_path.rsplit("/", 1)[-1] if "/" in href_path else href_path
        if not name:
            continue

        is_dir = False
        size: int | None = None
        modified: float | None = None
        for propstat in response_el.findall("{DAV:}propstat"):
            prop = propstat.find("{DAV:}prop")
            if prop is None:
                continue
            rt = prop.find("{DAV:}resourcetype")
            if rt is not None and rt.find("{DAV:}collection") is not None:
                is_dir = True
            size_text = _webdav_prop_text(prop, "getcontentlength")
            if size_text:
                try:
                    size = int(size_text)
                except ValueError:
                    size = None
            modified = modified or _timestamp_from_http_date(
                _webdav_prop_text(prop, "getlastmodified")
            )

        if directories_only and not is_dir:
            continue

        entry = {
            "name": name,
            "is_dir": is_dir,
        }
        if not directories_only:
            if size is not None:
                entry["size"] = size
            if modified is not None:
                entry["modified"] = modified
            entry["path"] = posixpath.join(browse_path, name) if browse_path else name
        entries.append(entry)
    return _sort_entries(entries)


def browse_webdav_propfind(
    config: dict, path: str | None, *, directories_only: bool = True
) -> dict:
    """Browse eines WebDAV-Verzeichnisses via PROPFIND."""

    username = str(config.get("username") or "")
    password = str(config.get("password") or "")
    if _contains_control(username) or _contains_control(password):
        raise ConnectorBrowseError("Ungueltige Credentials", 400, "bad_input")

    target_url, browse_path = _webdav_target_url(config, path)
    auth = (username, password) if username else None
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUTS["http_api"]) as client:
            response = client.request(
                "PROPFIND",
                target_url,
                auth=auth,
                headers={"Depth": "1", "Content-Type": "application/xml"},
            )
    except httpx.TimeoutException as exc:
        raise ConnectorBrowseError("Timeout", 502, "timeout") from exc
    except httpx.HTTPError as exc:
        raise ConnectorBrowseError(str(exc), 502, "bad_response") from exc

    if response.status_code not in (207, 200):
        raise ConnectorBrowseError(f"HTTP {response.status_code}", 502, "bad_response")

    return {
        "entries": _parse_webdav_entries(
            response.content,
            target_url,
            browse_path,
            directories_only,
        )
    }
