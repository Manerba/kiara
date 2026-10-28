"""Trusted-Origin-Bildung fuer CORS- und CSRF-Origin-Validation.

`compute_trusted_origins(config, *, ports)` baut eine deduplizierte Liste
vollstaendiger Origins (scheme://host[:port]) aus:
  - localhost/127.0.0.1 pro angefordertem Port
  - resolvedem Hostname (socket.gethostname()) pro Port
  - sichtbaren LAN-Interfaces (netifaces mit UDP-Socket-Fallback)
  - additiv angehaengten `web.trusted_origins`-Eintraegen aus der Config

Crash-sicher: socket.gaierror, ImportError (netifaces), UDP-Fehlschlag
liefern weiterhin eine nicht-leere Whitelist (mindestens localhost-Fallbacks
+ `web.trusted_origins`).
"""

from __future__ import annotations

import logging
import socket
from typing import Iterable

logger = logging.getLogger(__name__)


def _collect_lan_ips() -> list[str]:
    """Best-effort LAN-IP-Collection mit netifaces + UDP-Fallback."""
    ips: list[str] = []
    try:
        import netifaces  # type: ignore[import-not-found]

        for iface in netifaces.interfaces():
            try:
                addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
            except (ValueError, OSError):
                continue
            for addr in addrs:
                ip = addr.get("addr", "")
                if ip and not ip.startswith("127."):
                    ips.append(ip)
    except ImportError:
        logger.debug("netifaces nicht verfuegbar, UDP-Fallback fuer Primaer-IP")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                primary = s.getsockname()[0]
                if primary and not primary.startswith("127."):
                    ips.append(primary)
            finally:
                s.close()
        except OSError:
            pass
    except Exception as exc:
        logger.warning("Unerwarteter Fehler bei LAN-IP-Erkennung: %s", exc)
    return ips


def compute_trusted_origins(
    config: dict | None,
    *,
    ports: Iterable[int],
) -> list[str]:
    """Berechnet die Trusted-Origin-Liste fuer die angegebenen Ports.

    Args:
        config: Geladene kiara.yaml-Config (darf None sein).
        ports: Port-Nummern, fuer die Origins erzeugt werden sollen.
               Pro Port werden localhost/127.0.0.1/Hostname/LAN-IPs ergaenzt.

    Returns:
        Dedupe-Liste vollstaendiger Origin-Strings (scheme://host[:port]).
        Mindestens localhost+127.0.0.1 pro Port bei Totalausfall.
    """
    origins: list[str] = []
    seen: set[str] = set()

    def _add(origin: str) -> None:
        if origin and origin not in seen:
            seen.add(origin)
            origins.append(origin)

    port_list = list(ports)

    for port in port_list:
        _add(f"http://localhost:{port}")
        _add(f"http://127.0.0.1:{port}")

    # Hostname auto-detect
    try:
        hostname = socket.gethostname()
        if hostname:
            for port in port_list:
                _add(f"http://{hostname}:{port}")
            try:
                host_ip = socket.gethostbyname(hostname)
                if host_ip and not host_ip.startswith("127."):
                    for port in port_list:
                        _add(f"http://{host_ip}:{port}")
            except socket.gaierror:
                logger.debug("Hostname %s nicht aufloesbar", hostname)
    except OSError as exc:
        logger.debug("Hostname-Ermittlung fehlgeschlagen: %s", exc)

    # LAN-IPs via netifaces / UDP-Fallback
    for ip in _collect_lan_ips():
        for port in port_list:
            _add(f"http://{ip}:{port}")

    # Additive Origins aus Config (vollstaendige URLs, z.B. HTTPS-Proxy)
    if config:
        web_cfg = config.get("web") or {}
        for origin in web_cfg.get("trusted_origins") or []:
            if isinstance(origin, str):
                _add(origin.rstrip("/"))

    return origins
