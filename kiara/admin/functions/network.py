"""Netzwerk-Diagnose-Funktionen fuer den Admin-Chat.

Sprint 07: ping, portscan, dns, smb_discover, webdav_check, http_check, traceroute.
Alle subprocess-Aufrufe mit shell=False (Default), Parameter als Liste.
"""

import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile

import httpx

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.config import DEFAULT_TIMEOUTS

logger = logging.getLogger(__name__)

_OUTPUT_LIMIT = 10_000

_HOSTNAME_RE = re.compile(r"^[a-zA-Z0-9.\-]+$")


def _validate_hostname(host: str) -> str | None:
    """Validiert Hostname/IP. Gibt Fehlermeldung oder None bei OK zurueck."""
    if not host or len(host) > 255:
        return "Hostname muss 1-255 Zeichen lang sein"
    if host.startswith("-"):
        return "Hostname darf nicht mit Bindestrich beginnen"
    if not _HOSTNAME_RE.match(host):
        return "Ungueltiger Hostname: nur Buchstaben, Ziffern, Punkte und Bindestriche erlaubt"
    return None


# ---------------------------------------------------------------------------
# network.ping
# ---------------------------------------------------------------------------


@admin_function(
    name="network.ping",
    category="network",
    description="Prueft die Erreichbarkeit eines Hosts via ICMP Ping",
    help="Sendet ICMP-Ping-Pakete an einen Host und misst Antwortzeit und Paketverlust.",
    examples=["Pinge fileserver.local", "Ist gpu-server.local erreichbar?"],
    parameters={
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "IP-Adresse oder Hostname", "minLength": 1},
            "count": {
                "type": "integer",
                "description": "Anzahl Ping-Pakete",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["host"],
    },
    returns="Erreichbarkeit, Paketverlust und durchschnittliche Antwortzeit",
)
async def network_ping(user: dict, host: str, count: int = 3) -> AdminResult:
    err = _validate_hostname(host)
    if err:
        return AdminResult.fail(err)
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), "-W", "3", "--", host],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_network_ping"],
        )
    except FileNotFoundError:
        return AdminResult.fail("ping ist nicht installiert")
    except subprocess.TimeoutExpired:
        return AdminResult.fail(f"Ping-Timeout nach {DEFAULT_TIMEOUTS['subprocess_network_ping']}s")

    output = result.stdout[:_OUTPUT_LIMIT]

    # Paketloss parsen: "3 packets transmitted, 3 received, 0% packet loss"
    loss_match = re.search(r"(\d+)% packet loss", output)
    packet_loss = loss_match.group(1) + "%" if loss_match else "unbekannt"

    # RTT parsen: "rtt min/avg/max/mdev = 0.023/0.045/0.067/0.018 ms"
    rtt_match = re.search(r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", output)
    rtt_avg = rtt_match.group(2) + "ms" if rtt_match else None

    # Pakete parsen
    pkt_match = re.search(r"(\d+) packets transmitted, (\d+) received", output)
    packets_sent = int(pkt_match.group(1)) if pkt_match else count
    packets_received = int(pkt_match.group(2)) if pkt_match else 0

    reachable = packets_received > 0

    data = {
        "reachable": reachable,
        "packets_sent": packets_sent,
        "packets_received": packets_received,
        "packet_loss": packet_loss,
    }
    if rtt_avg:
        data["rtt_avg"] = rtt_avg

    if reachable:
        msg = f"{host} erreichbar — Paketverlust: {packet_loss}, RTT avg: {rtt_avg or 'n/a'}"
    else:
        msg = f"{host} nicht erreichbar — Paketverlust: {packet_loss}"

    return AdminResult.ok(msg, data) if reachable else AdminResult.fail(msg, data)


# ---------------------------------------------------------------------------
# network.portscan
# ---------------------------------------------------------------------------


@admin_function(
    name="network.portscan",
    category="network",
    description="Prueft ob TCP-Ports auf einem Host offen sind",
    help="Scannt einzelne oder mehrere TCP-Ports. Nutzt nmap falls installiert, sonst Python-Socket-Fallback.",
    examples=["Pruefe Port 445 auf fileserver.local", "Scanne Ports 80,443,8080 auf nextcloud.firma.de"],
    parameters={
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "IP-Adresse oder Hostname", "minLength": 1},
            "ports": {
                "type": "string",
                "description": "Komma-separierte Ports, z.B. '445' oder '80,443,8080'",
                "minLength": 1,
            },
        },
        "required": ["host", "ports"],
    },
    returns="Status (open/closed) pro Port, optional Service-Name",
)
async def network_portscan(user: dict, host: str, ports: str) -> AdminResult:
    err = _validate_hostname(host)
    if err:
        return AdminResult.fail(err)
    # Port-Validierung
    port_list: list[int] = []
    for p in ports.split(","):
        p = p.strip()
        try:
            port_int = int(p)
        except ValueError:
            return AdminResult.fail(f"Ungueltiger Port: '{p}'")
        if port_int < 1 or port_int > 65535:
            return AdminResult.fail(f"Ungueltiger Port: '{p}' (muss 1-65535 sein)")
        port_list.append(port_int)

    if shutil.which("nmap"):
        return await _portscan_nmap(host, port_list, ports)
    return _portscan_socket(host, port_list)


async def _portscan_nmap(host: str, port_list: list[int], ports_str: str) -> AdminResult:
    try:
        result = subprocess.run(
            ["nmap", "-Pn", "-p", ",".join(str(p) for p in port_list), "--open", "--", host],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_network_scan"],
        )
    except FileNotFoundError:
        return _portscan_socket(host, port_list)
    except subprocess.TimeoutExpired:
        return AdminResult.fail(
            f"Portscan-Timeout nach {DEFAULT_TIMEOUTS['subprocess_network_scan']}s"
        )

    output = result.stdout[:_OUTPUT_LIMIT]
    port_results = []
    for port in port_list:
        # nmap output: "445/tcp   open  microsoft-ds"
        match = re.search(rf"^{port}/tcp\s+(\S+)\s*(.*)", output, re.MULTILINE)
        if match:
            port_results.append(
                {"port": port, "state": match.group(1), "service": match.group(2).strip()}
            )
        else:
            port_results.append({"port": port, "state": "closed"})

    open_count = sum(1 for p in port_results if p["state"] == "open")
    msg = f"{host}: {open_count}/{len(port_results)} Ports offen"
    return AdminResult.ok(msg, {"host": host, "ports": port_results})


def _portscan_socket(host: str, port_list: list[int]) -> AdminResult:
    port_results = []
    for port in port_list:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            err = sock.connect_ex((host, port))
            state = "open" if err == 0 else "closed"
        except (socket.timeout, OSError):
            state = "closed"
        finally:
            sock.close()
        port_results.append({"port": port, "state": state})

    open_count = sum(1 for p in port_results if p["state"] == "open")
    msg = f"{host}: {open_count}/{len(port_results)} Ports offen (Socket-Fallback)"
    return AdminResult.ok(msg, {"host": host, "ports": port_results})


# ---------------------------------------------------------------------------
# network.dns
# ---------------------------------------------------------------------------

_ALLOWED_DNS_TYPES = {"A", "AAAA", "MX", "CNAME", "PTR"}


@admin_function(
    name="network.dns",
    category="network",
    description="Loest einen Hostnamen per DNS auf",
    help="Fuehrt eine DNS-Abfrage durch. Nutzt dig falls installiert, sonst Python socket.getaddrinfo().",
    examples=["DNS-Aufloesung fuer nextcloud.firma.de", "MX-Records fuer firma.de"],
    parameters={
        "type": "object",
        "properties": {
            "hostname": {"type": "string", "description": "Hostname oder Domain", "minLength": 1},
            "type": {
                "type": "string",
                "description": "DNS-Record-Typ",
                "enum": ["A", "AAAA", "MX", "CNAME", "PTR"],
                "default": "A",
            },
        },
        "required": ["hostname"],
    },
    returns="Liste der DNS-Records",
)
async def network_dns(user: dict, hostname: str, type: str = "A") -> AdminResult:
    err = _validate_hostname(hostname)
    if err:
        return AdminResult.fail(err)
    if type not in _ALLOWED_DNS_TYPES:
        return AdminResult.fail(
            f"Ungueltiger Record-Typ: {type} (erlaubt: {', '.join(sorted(_ALLOWED_DNS_TYPES))})"
        )

    if shutil.which("dig"):
        return _dns_dig(hostname, type)
    return _dns_socket(hostname, type)


def _dns_dig(hostname: str, record_type: str) -> AdminResult:
    try:
        result = subprocess.run(
            ["dig", "+short", "--", hostname, record_type],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_network_dns"],
        )
    except FileNotFoundError:
        return _dns_socket(hostname, record_type)
    except subprocess.TimeoutExpired:
        return AdminResult.fail(f"DNS-Timeout nach {DEFAULT_TIMEOUTS['subprocess_network_dns']}s")

    output = result.stdout.strip()[:_OUTPUT_LIMIT]
    if not output:
        return AdminResult.fail(f"Keine {record_type}-Records fuer {hostname} gefunden")

    records = [line.strip() for line in output.splitlines() if line.strip()]
    msg = f"{hostname} ({record_type}): {', '.join(records)}"
    return AdminResult.ok(msg, {"hostname": hostname, "type": record_type, "records": records})


def _dns_socket(hostname: str, record_type: str) -> AdminResult:
    if record_type not in ("A", "AAAA"):
        return AdminResult.fail(
            f"dig nicht installiert — {record_type}-Records benoetigen dig. "
            f"Nur A/AAAA-Records sind ohne dig verfuegbar."
        )
    family = socket.AF_INET if record_type == "A" else socket.AF_INET6
    try:
        results = socket.getaddrinfo(hostname, None, family)
    except socket.gaierror as exc:
        return AdminResult.fail(f"DNS-Aufloesung fehlgeschlagen fuer {hostname}: {exc}")

    records = sorted(set(r[4][0] for r in results))
    if not records:
        return AdminResult.fail(f"Keine {record_type}-Records fuer {hostname} gefunden")

    msg = f"{hostname} ({record_type}): {', '.join(records)}"
    return AdminResult.ok(msg, {"hostname": hostname, "type": record_type, "records": records})


# ---------------------------------------------------------------------------
# network.smb_discover
# ---------------------------------------------------------------------------


@admin_function(
    name="network.smb_discover",
    category="network",
    description="Listet SMB-Shares eines Servers auf",
    help="Verbindet sich per smbclient zum Server und zeigt verfuegbare Freigaben.",
    examples=["Zeige Shares auf fileserver.local", "SMB-Freigaben auf fileserver mit User admin"],
    parameters={
        "type": "object",
        "properties": {
            "host": {
                "type": "string",
                "description": "IP-Adresse oder Hostname des SMB-Servers",
                "minLength": 1,
            },
            "username": {"type": "string", "description": "SMB-Benutzername (optional)"},
            "password": {"type": "string", "description": "SMB-Passwort (optional)"},
            "domain": {"type": "string", "description": "SMB-Domain/Workgroup (optional)"},
        },
        "required": ["host"],
    },
    returns="Liste der verfuegbaren Shares mit Name, Typ und Kommentar",
)
async def network_smb_discover(
    user: dict,
    host: str,
    username: str = "",
    password: str = "",
    domain: str = "",
) -> AdminResult:
    err = _validate_hostname(host)
    if err:
        return AdminResult.fail(err)
    if domain and (not _HOSTNAME_RE.match(domain) or domain.startswith("-")):
        return AdminResult.fail("Ungueltiger Domain-Name")
    for _field, _val in [("Username", username), ("Password", password), ("Domain", domain)]:
        if "\n" in _val or "\r" in _val:
            return AdminResult.fail(f"{_field}: Zeilenumbrueche nicht erlaubt")
    cmd = ["smbclient", "-L", f"//{host}"]
    cred_path = None

    try:
        if username:
            # Authentication-File (analog smb_connector.py Z.116-126)
            cred_lines = f"username={username}\npassword={password}\n"
            if domain:
                cred_lines += f"domain={domain}\n"
            fd, cred_path = tempfile.mkstemp(prefix="kiara-smb-", suffix=".cred")
            try:
                os.write(fd, cred_lines.encode())
            finally:
                os.close(fd)
            os.chmod(cred_path, 0o600)
            cmd.append(f"--authentication-file={cred_path}")
        else:
            cmd.append("-N")  # anonymous

        if domain and not username:
            cmd.extend(["-W", domain])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_network_smb"],
        )
    except FileNotFoundError:
        return AdminResult.fail("smbclient ist nicht installiert")
    except subprocess.TimeoutExpired:
        return AdminResult.fail(f"SMB-Timeout nach {DEFAULT_TIMEOUTS['subprocess_network_smb']}s")
    finally:
        if cred_path:
            try:
                os.unlink(cred_path)
            except OSError:
                pass

    (result.stdout + result.stderr)[:_OUTPUT_LIMIT]

    if result.returncode != 0 and not result.stdout.strip():
        return AdminResult.fail(f"SMB-Verbindung fehlgeschlagen: {result.stderr.strip()[:500]}")

    # Shares parsen: "Sharename       Type      Comment"
    # Felder durch 2+ Spaces getrennt
    shares: list[dict] = []
    in_share_section = False
    for line in result.stdout[:_OUTPUT_LIMIT].splitlines():
        if re.match(r"\s*Sharename\s+Type\s+Comment", line, re.IGNORECASE):
            in_share_section = True
            continue
        if in_share_section:
            if line.strip().startswith("---"):
                continue
            if not line.strip():
                break
            # Parse: "  Daten           Disk      Gemeinsame Dateien"
            parts = re.split(r"\s{2,}", line.strip(), maxsplit=2)
            if len(parts) >= 2:
                share = {"name": parts[0], "type": parts[1]}
                if len(parts) >= 3:
                    share["comment"] = parts[2]
                else:
                    share["comment"] = ""
                shares.append(share)

    msg = f"{host}: {len(shares)} Shares gefunden"
    return AdminResult.ok(msg, {"host": host, "shares": shares})


# ---------------------------------------------------------------------------
# network.webdav_check
# ---------------------------------------------------------------------------


@admin_function(
    name="network.webdav_check",
    category="network",
    description="Prueft ob ein WebDAV-Endpunkt erreichbar und DAV-konform ist",
    help="Sendet HTTP PROPFIND/OPTIONS an die URL und prueft auf DAV-Compliance.",
    examples=[
        "Pruefe WebDAV auf https://nextcloud.firma.de/remote.php/dav",
        "WebDAV-Check nextcloud",
    ],
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "WebDAV-URL (http:// oder https://)",
                "minLength": 1,
            },
            "username": {"type": "string", "description": "WebDAV-Benutzername (optional)"},
            "password": {"type": "string", "description": "WebDAV-Passwort (optional)"},
        },
        "required": ["url"],
    },
    returns="Erreichbarkeit, Status-Code, DAV-Compliance und Server-Info",
)
async def network_webdav_check(
    user: dict, url: str, username: str = "", password: str = ""
) -> AdminResult:
    if not url.startswith(("http://", "https://")):
        return AdminResult.fail("URL muss mit http:// oder https:// beginnen")

    timeout = DEFAULT_TIMEOUTS["subprocess_network_http"]
    auth = httpx.BasicAuth(username, password) if username else None

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            response = await client.request("OPTIONS", url, auth=auth)
    except httpx.TimeoutException:
        return AdminResult.fail(f"WebDAV-Timeout nach {timeout}s")
    except httpx.ConnectError as exc:
        return AdminResult.fail(f"Verbindung fehlgeschlagen: {exc}")

    dav_header = response.headers.get("dav", "")
    dav_compliant = bool(dav_header)
    server = response.headers.get("server", "")

    data = {
        "url": url,
        "reachable": True,
        "status_code": response.status_code,
        "dav_compliant": dav_compliant,
        "server": server,
    }
    if dav_header:
        data["dav_classes"] = dav_header

    if dav_compliant:
        msg = f"WebDAV erreichbar: {url} — Status {response.status_code}, DAV-konform"
    else:
        msg = f"URL erreichbar aber NICHT DAV-konform: {url} — Status {response.status_code}, kein DAV-Header"

    return AdminResult.ok(msg, data)


# ---------------------------------------------------------------------------
# network.http_check
# ---------------------------------------------------------------------------

_ALLOWED_HTTP_METHODS = {"HEAD", "GET"}


@admin_function(
    name="network.http_check",
    category="network",
    description="Prueft die HTTP(S)-Erreichbarkeit einer URL",
    help="Sendet einen HTTP-Request (HEAD oder GET) an die URL und prueft Status-Code und Antwortzeit.",
    examples=["Pruefe https://nextcloud.firma.de", "HTTP-Check auf http://gpu-server.local:11434"],
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "HTTP(S)-URL (http:// oder https://)",
                "minLength": 1,
            },
            "method": {
                "type": "string",
                "description": "HTTP-Methode",
                "enum": ["HEAD", "GET"],
                "default": "HEAD",
            },
        },
        "required": ["url"],
    },
    returns="Erreichbarkeit, Status-Code, Antwortzeit und Server-Header",
)
async def network_http_check(user: dict, url: str, method: str = "HEAD") -> AdminResult:
    if method not in _ALLOWED_HTTP_METHODS:
        return AdminResult.fail(f"Nur {', '.join(sorted(_ALLOWED_HTTP_METHODS))} erlaubt")
    if not url.startswith(("http://", "https://")):
        return AdminResult.fail("URL muss mit http:// oder https:// beginnen")

    timeout = DEFAULT_TIMEOUTS["subprocess_network_http"]

    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            response = await client.request(method, url)
    except httpx.TimeoutException:
        return AdminResult.fail(f"HTTP-Timeout nach {timeout}s")
    except httpx.ConnectError as exc:
        return AdminResult.fail(f"Verbindung fehlgeschlagen: {exc}")

    elapsed_ms = int(response.elapsed.total_seconds() * 1000)
    server = response.headers.get("server", "")

    data = {
        "url": url,
        "reachable": True,
        "status_code": response.status_code,
        "response_time_ms": elapsed_ms,
        "server": server,
    }

    msg = f"{url} — Status {response.status_code}, {elapsed_ms}ms"
    return AdminResult.ok(msg, data)


# ---------------------------------------------------------------------------
# network.traceroute
# ---------------------------------------------------------------------------


@admin_function(
    name="network.traceroute",
    category="network",
    description="Zeigt den Netzwerkpfad zu einem Host",
    help="Fuehrt traceroute aus und zeigt alle Hops auf dem Weg zum Ziel.",
    examples=["Traceroute zu fileserver.local", "Netzwerkpfad zu nextcloud.firma.de"],
    parameters={
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "IP-Adresse oder Hostname", "minLength": 1},
        },
        "required": ["host"],
    },
    returns="Liste der Hops mit IP, Hostname und Antwortzeit",
)
async def network_traceroute(user: dict, host: str) -> AdminResult:
    err = _validate_hostname(host)
    if err:
        return AdminResult.fail(err)
    if not shutil.which("traceroute"):
        return AdminResult.fail("traceroute ist nicht installiert")

    try:
        result = subprocess.run(
            ["traceroute", "-m", "20", "-w", "3", "--", host],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_network_traceroute"],
        )
    except subprocess.TimeoutExpired:
        return AdminResult.fail(
            f"Traceroute-Timeout nach {DEFAULT_TIMEOUTS['subprocess_network_traceroute']}s"
        )

    output = result.stdout[:_OUTPUT_LIMIT]

    # Hops parsen: " 1  gateway (10.0.0.1)  0.500 ms  0.400 ms  0.350 ms"
    hops: list[dict] = []
    for line in output.splitlines():
        hop_match = re.match(
            r"\s*(\d+)\s+(\S+)\s+\(([^)]+)\)\s+([\d.]+)\s*ms",
            line,
        )
        if hop_match:
            hops.append(
                {
                    "hop": int(hop_match.group(1)),
                    "hostname": hop_match.group(2),
                    "ip": hop_match.group(3),
                    "rtt_ms": hop_match.group(4),
                }
            )
            continue
        # Timeout-Hop: " 2  * * *"
        timeout_match = re.match(r"\s*(\d+)\s+\*", line)
        if timeout_match:
            hops.append(
                {
                    "hop": int(timeout_match.group(1)),
                    "hostname": "*",
                    "ip": "*",
                    "rtt_ms": "*",
                }
            )

    msg = f"Traceroute zu {host}: {len(hops)} Hops"
    return AdminResult.ok(msg, {"host": host, "hops": hops})
