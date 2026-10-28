"""HTTP-Pinning gegen DNS-Rebinding/SSRF-TOCTOU.

Loest DNS einmal pro Request auf, validiert die aufgeloeste IP gegen eine
Allowlist und verbindet TCP an die gepinnte IP. TLS-SNI/Hostname-Verification
und HTTP-Host-Header bleiben am Original-Host.

Vertrag (siehe docs/TODOS/TODO_BUGFIX_IP_PINNING_CODEX_V4.md):
- Nur ``http``/``https`` (case-insensitive normalisiert).
- Credentials in URLs werden abgelehnt; Log-URL wird redigiert (``***@host``).
- DNS wird einmal pro Request aufgeloest (``socket.getaddrinfo``).
- Alle aufgeloesten IPs muessen erlaubt sein. Hard-Block-Klassen
  (``is_loopback``, ``is_link_local``, ``is_unspecified``, ``is_multicast``,
  ``is_reserved``) gewinnen immer, auch bei Allowlist-Treffer.
- Allowlist als ``Sequence[str]`` (kw-only); ``str``/``bytes`` und ungueltige
  CIDRs werfen ``PinningConfigError``. CIDRs, die Hard-Block-Bereiche
  ueberlappen, sind ebenfalls ``PinningConfigError``.
- ``verify=True`` und ``follow_redirects=False`` sind verbindlich. Helfer
  lehnen reservierte Caller-Kwargs als echten ``ValueError`` ab.
- Original-``httpx.Request`` wird nicht mutiert; der Transport clont und
  setzt ``response.request = original_request`` zurueck.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_ALLOWED_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

_REQUEST_RESERVED_KWARGS = frozenset(
    {"verify", "cert", "trust_env", "event_hooks", "transport", "follow_redirects"}
)
_REQUEST_ALLOWED_KWARGS = frozenset(
    {"auth", "headers", "timeout", "content", "data", "json", "files", "params"}
)
_CLIENT_ALLOWED_KWARGS = frozenset({"auth", "headers", "timeout"})

# Reservierte Transport-Kwargs: TLS-/Trust-/Connection-Toggles, die den
# Pinning-Vertrag ("TCP an gepinnte IP" + verify=True) umgehen koennten:
# - verify/cert/trust_env: TLS-Invariante
# - proxy: Verbindungs-Umweg
# - uds: Unix-Domain-Socket statt TCP — laeuft komplett am IP-Pinning vorbei
# - local_address: Source-Bind-Override (kein direkter Bypass, aber nicht im
#   Vertrag vorgesehen — fail-closed durch Ablehnung)
# - socket_options: koennte SO_-Flags setzen, die den Connect-Pfad veraendern
_TRANSPORT_RESERVED_KWARGS = frozenset(
    {
        "verify",
        "cert",
        "trust_env",
        "proxy",
        "uds",
        "local_address",
        "socket_options",
    }
)

_HARD_BLOCK_PROPS = (
    "is_loopback",
    "is_link_local",
    "is_unspecified",
    "is_multicast",
    "is_reserved",
)

# Hard-Block-Netze pro IP-Familie. Werden nur fuer Allowlist-Validierung
# (overlaps()) genutzt; einzelne IPs werden direkt ueber die Address-Properties
# klassifiziert.
_HARD_BLOCK_V4_NETS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("0.0.0.0/32"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv4Network("240.0.0.0/4"),
)

# IPv6-_reserved_networks aus CPython 3.12 ipaddress + explizite Hard-Block.
_HARD_BLOCK_V6_NETS: tuple[ipaddress.IPv6Network, ...] = (
    ipaddress.IPv6Network("::/128"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fe80::/10"),
    ipaddress.IPv6Network("ff00::/8"),
    ipaddress.IPv6Network("::/8"),
    ipaddress.IPv6Network("100::/8"),
    ipaddress.IPv6Network("200::/7"),
    ipaddress.IPv6Network("400::/6"),
    ipaddress.IPv6Network("800::/5"),
    ipaddress.IPv6Network("1000::/4"),
    ipaddress.IPv6Network("4000::/3"),
    ipaddress.IPv6Network("6000::/3"),
    ipaddress.IPv6Network("8000::/3"),
    ipaddress.IPv6Network("a000::/3"),
    ipaddress.IPv6Network("c000::/3"),
    ipaddress.IPv6Network("e000::/4"),
    ipaddress.IPv6Network("f000::/5"),
    ipaddress.IPv6Network("f800::/6"),
    ipaddress.IPv6Network("fe00::/9"),
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PinningError(httpx.RequestError):
    """URL/IP wurde durch das Pinning blockiert.

    Subklasse von ``httpx.RequestError``, damit Caller, die ohnehin auf
    ``httpx.RequestError`` filtern, automatisch korrekt failen. Enthaelt
    ``.reason`` (Block-Grund), ``.log_url`` (redigierte Log-URL) und
    ``.request`` (echter ``httpx.Request``).
    """

    def __init__(self, reason: str, *, request: httpx.Request, log_url: str):
        super().__init__(reason, request=request)
        self.reason = reason
        self.log_url = log_url


class PinningConfigError(ValueError):
    """Ungueltige Allowlist-Config in ``allowed_private_cidrs``."""


# ---------------------------------------------------------------------------
# PinnedTarget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedTarget:
    """Ergebnis einer Pinning-Aufloesung fuer einen Request.

    ``netloc`` wird unveraendert als ``Host``-Header genutzt (inklusive
    Nicht-Default-Port). ``host`` ist nur der Hostname ohne Port und ohne
    IPv6-Brackets — er wird fuer TLS-SNI verwendet. ``pinned_url`` setzt
    die aufgeloeste IP in den Netloc; nur diese URL wird per TCP angesteuert.
    """

    original_url: str
    netloc: str
    host: str
    port: int
    scheme: str
    ip: str
    pinned_url: str


# ---------------------------------------------------------------------------
# Allowlist-Validierung
# ---------------------------------------------------------------------------


def _network_overlaps_hard_block(
    net: ipaddress.IPv4Network | ipaddress.IPv6Network,
) -> bool:
    if isinstance(net, ipaddress.IPv4Network):
        return any(net.overlaps(b) for b in _HARD_BLOCK_V4_NETS)
    return any(net.overlaps(b) for b in _HARD_BLOCK_V6_NETS)


def _parse_cidrs(
    allowed_private_cidrs: Sequence[str] | None,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    if allowed_private_cidrs is None:
        return ()
    if isinstance(allowed_private_cidrs, (str, bytes)):
        raise PinningConfigError(
            "allowed_private_cidrs darf kein str/bytes sein, "
            "sondern eine Sequenz von Strings (z. B. Liste)"
        )
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for item in allowed_private_cidrs:
        if not isinstance(item, str):
            raise PinningConfigError(
                f"allowed_private_cidrs-Element ist kein str: {type(item).__name__}"
            )
        try:
            net = ipaddress.ip_network(item, strict=False)
        except (ValueError, TypeError) as exc:
            raise PinningConfigError(f"ungueltige CIDR: {item!r} ({exc})") from exc
        if _network_overlaps_hard_block(net):
            raise PinningConfigError(f"CIDR {item!r} ueberlappt Hard-Block-Bereich")
        networks.append(net)
    return tuple(networks)


def validate_allowed_private_cidrs(
    allowed_private_cidrs: Sequence[str] | None,
) -> None:
    """Validiert eine Allowlist eagerly. Wirft ``PinningConfigError``."""
    _parse_cidrs(allowed_private_cidrs)


# ---------------------------------------------------------------------------
# URL-Validierung und Redaction
# ---------------------------------------------------------------------------


_PLACEHOLDER_URL = "http://invalid.invalid/"
_UNPARSABLE_URL_LABEL = "<unparsable URL>"


def _raw_userinfo_present(url: str) -> bool:
    """Heuristisch: enthaelt der raw netloc-Bereich (vor erstem '/') ein '@'?

    Wird benutzt, um auch bei nicht parsebaren URLs (urlsplit -> ValueError)
    Credential-Leaks im Log zu verhindern. Konservativ: ein einsames '@'
    ohne ``://`` wird ebenfalls als verdaechtig betrachtet.
    """
    if not isinstance(url, str) or "@" not in url:
        return False
    after_scheme = url.split("://", 1)[1] if "://" in url else url
    netloc_part = after_scheme.split("/", 1)[0]
    return "@" in netloc_part


def _safe_attr(obj: object, name: str, default=None):
    """Liest ein Attribut robust; faengt ValueError/TypeError aus Lazy-Parsing."""
    try:
        return getattr(obj, name, default)
    except (ValueError, TypeError):
        return default


def _redact_url(url: str) -> str:
    """Ersetzt ``user:pw@`` durch ``***@`` in einer URL fuer Logs/Errors.

    Garantiert exception-frei, auch bei kaputten Eingaben (Bad-Port,
    unvollstaendige IPv6-Brackets etc.). Bei nicht parsebaren URLs wird ein
    konstanter Platzhalter zurueckgegeben, damit weder die Originalform noch
    Credential-Reste in Logs landen.
    """
    if not isinstance(url, str) or not url:
        return _UNPARSABLE_URL_LABEL
    try:
        parts = urlsplit(url)
    except (ValueError, TypeError):
        return _UNPARSABLE_URL_LABEL
    username = _safe_attr(parts, "username")
    password = _safe_attr(parts, "password")
    if username is None and password is None:
        return url
    raw_host = _safe_attr(parts, "hostname") or ""
    port_val = _safe_attr(parts, "port")
    if port_val is not None:
        host_part = f"[{raw_host}]:{port_val}" if ":" in raw_host else f"{raw_host}:{port_val}"
    else:
        host_part = f"[{raw_host}]" if ":" in raw_host else raw_host
    new_netloc = f"***@{host_part}"
    scheme = parts.scheme or "http"
    try:
        return urlunsplit((scheme, new_netloc, parts.path, parts.query, parts.fragment))
    except (ValueError, TypeError):
        return f"{scheme}://{new_netloc}"


def _make_dummy_request(method: str, url: str) -> httpx.Request:
    """Erstellt einen leichtgewichtigen ``httpx.Request`` fuer Exception-Kontext.

    Robust gegen ``httpx.InvalidURL`` / ``ValueError`` aus kaputten Probe-URLs:
    fallback auf einen Platzhalter, damit der ``PinningError``-Pfad nicht in
    einen Programmierfehler eskaliert.
    """
    method_upper = (
        method.upper() if isinstance(method, str) and method.upper() in _ALLOWED_METHODS else "GET"
    )
    try:
        return httpx.Request(method_upper, url)
    except (httpx.InvalidURL, ValueError, TypeError):
        return httpx.Request(method_upper, _PLACEHOLDER_URL)


def _raise_pin(reason: str, *, original_url: str, method: str, redact: bool = False) -> None:
    """Erzeuge ``PinningError`` mit konsistenter Log-/Request-URL-Redaktion.

    Auch wenn der Caller ``redact=False`` setzt, wird heuristisch redaktiert,
    sobald der raw netloc ein ``@`` enthaelt. Das schuetzt vor Credential-
    Leaks bei nicht parsebaren URLs (urlsplit-Fehler) — z. B.
    ``http://user:pw@[::1/`` schaffte es sonst mit raw URL ins Log.
    """
    raw_url = original_url if isinstance(original_url, str) else str(original_url)
    log_url = _redact_url(raw_url) if redact or _raw_userinfo_present(raw_url) else raw_url
    request_url = log_url if log_url != raw_url else raw_url
    request = _make_dummy_request(method, request_url)
    message = f"{reason} (URL: {log_url})"
    logger.warning("Pinning blockiert: %s", message)
    raise PinningError(message, request=request, log_url=log_url)


def _validate_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError(f"method muss str sein, ist {type(method).__name__}")
    upper = method.upper()
    if upper not in _ALLOWED_METHODS:
        raise ValueError(f"Methode nicht erlaubt: {method!r} (erlaubt: {sorted(_ALLOWED_METHODS)})")
    return upper


def _split_url(method: str, url: str) -> tuple[str, str, int, bool, str]:
    """Validiert URL und gibt ``(scheme, host, port, has_credentials, netloc)`` zurueck.

    Wirft ``PinningError`` bei Schema/Host/Credential/Port-Verletzungen.
    Bewusst robust gegen Lazy-Parsing-Fehler in ``SplitResult`` (z. B.
    ``parts.username`` kann bei manchen kaputten netloc-Formen ValueError
    werfen) — alle solchen Faelle enden in ``PinningError``, nicht in einem
    Programmierfehler nach aussen.
    """
    if not isinstance(url, str) or not url:
        _raise_pin("EmptyHost: URL leer", original_url=str(url), method=method)
    try:
        parts = urlsplit(url)
    except (ValueError, TypeError):
        _raise_pin("Scheme: URL nicht parsebar", original_url=url, method=method)
    scheme = (parts.scheme or "").lower()
    if scheme not in _DEFAULT_PORTS:
        _raise_pin(
            f"Scheme: nur http/https erlaubt, nicht {parts.scheme!r}",
            original_url=url,
            method=method,
        )
    # Lazy-Parsing der SplitResult-Properties (username/password/hostname/port)
    # kann bei kaputten netloc-Formen ValueError werfen. Wir behandeln das
    # konsistent als Credentials/Scheme-Fehler statt es nach aussen zu lassen.
    try:
        username_present = parts.username is not None or parts.password is not None
    except (ValueError, TypeError):
        _raise_pin(
            "Scheme: URL nicht parsebar",
            original_url=url,
            method=method,
        )
    if username_present:
        _raise_pin(
            "Credentials: URL enthaelt Benutzer/Passwort",
            original_url=url,
            method=method,
            redact=True,
        )
    try:
        host = parts.hostname
    except (ValueError, TypeError):
        _raise_pin(
            "Scheme: URL nicht parsebar",
            original_url=url,
            method=method,
        )
    if not host:
        _raise_pin("EmptyHost: kein Host in URL", original_url=url, method=method)
    try:
        raw_port = parts.port
    except (ValueError, TypeError):
        _raise_pin(
            "Scheme: ungueltiger Port",
            original_url=url,
            method=method,
        )
    port = raw_port if raw_port is not None else _DEFAULT_PORTS[scheme]
    if raw_port is not None:
        netloc = f"[{host}]:{raw_port}" if ":" in host else f"{host}:{raw_port}"
    else:
        netloc = f"[{host}]" if ":" in host else host
    return scheme, host, port, False, netloc


# ---------------------------------------------------------------------------
# IP-Klassifikation und DNS
# ---------------------------------------------------------------------------


def _is_ip_allowed(
    ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
    networks: Iterable[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> tuple[bool, str]:
    """Prueft IP gegen Hard-Block + Allowlist. Gibt ``(allowed, reason)`` zurueck.

    ``reason`` ist nur fuer den Block-Fall relevant (deterministische
    Nachricht ``"Klassifikation: <ip> ist <Grund>"`` bzw. ``"Allowlist: <ip>"``).
    """
    for prop in _HARD_BLOCK_PROPS:
        if getattr(ip_obj, prop, False):
            return False, f"Klassifikation: {ip_obj} ist {prop[3:]}"
    if ip_obj.is_global:
        return True, ""
    for net in networks:
        if (
            (isinstance(ip_obj, ipaddress.IPv4Address) and isinstance(net, ipaddress.IPv4Network))
            or (
                isinstance(ip_obj, ipaddress.IPv6Address) and isinstance(net, ipaddress.IPv6Network)
            )
        ) and ip_obj in net:
            return True, ""
    return False, f"Allowlist: {ip_obj} nicht in erlaubten privaten Bereichen"


def _resolve_host_sync(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise _DNSError(f"DNS: Aufloesung fehlgeschlagen ({exc})") from exc
    return _extract_addresses(infos)


async def _resolve_host_async(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise _DNSError(f"DNS: Aufloesung fehlgeschlagen ({exc})") from exc
    return _extract_addresses(infos)


def _extract_addresses(infos: list[Any]) -> list[str]:
    addresses: list[str] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        if addr and addr not in seen:
            seen.add(addr)
            addresses.append(addr)
    if not addresses:
        raise _DNSError("DNS: keine IP zurueckgegeben")
    return addresses


class _DNSError(Exception):
    """Internes Marker-Exception fuer DNS-Fehler (in PinningError gewrapt)."""


def _ip_obj_for_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """IP-Literal-Erkennung. Gibt die IP zurueck oder ``None`` wenn Hostname."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _build_pinned_url(scheme: str, ip: str, port: int, parts_url: str) -> str:
    parts = urlsplit(parts_url)
    netloc_ip = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
    return urlunsplit((scheme, netloc_ip, parts.path or "/", parts.query, parts.fragment))


def _resolve_target_common(
    method: str,
    url: str,
    *,
    allowed_private_cidrs: Sequence[str] | None,
    addresses: list[str],
) -> PinnedTarget:
    networks = _parse_cidrs(allowed_private_cidrs)
    scheme, host, port, _, netloc = _split_url(method, url)
    chosen_ip: str | None = None
    chosen_obj: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    for addr in addresses:
        try:
            ip_obj = ipaddress.ip_address(addr)
        except ValueError:
            continue
        allowed, reason = _is_ip_allowed(ip_obj, networks)
        if not allowed:
            _raise_pin(reason, original_url=url, method=method)
        if chosen_ip is None:
            chosen_ip = addr
            chosen_obj = ip_obj
    if chosen_ip is None or chosen_obj is None:
        _raise_pin(
            "DNS: keine verwendbare IP nach Klassifikation",
            original_url=url,
            method=method,
        )
    pinned_url = _build_pinned_url(scheme, chosen_ip, port, url)
    return PinnedTarget(
        original_url=url,
        netloc=netloc,
        host=host,
        port=port,
        scheme=scheme,
        ip=chosen_ip,
        pinned_url=pinned_url,
    )


def _resolve_pinned_target_sync(
    method: str,
    url: str,
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
) -> PinnedTarget:
    _scheme, host, port, _, _ = _split_url(method, url)
    literal_ip = _ip_obj_for_host(host)
    if literal_ip is not None:
        addresses = [literal_ip.compressed]
    else:
        try:
            addresses = _resolve_host_sync(host, port)
        except _DNSError as exc:
            _raise_pin(str(exc), original_url=url, method=method)
    return _resolve_target_common(
        method,
        url,
        allowed_private_cidrs=allowed_private_cidrs,
        addresses=addresses,
    )


async def _resolve_pinned_target_async(
    method: str,
    url: str,
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
) -> PinnedTarget:
    _scheme, host, port, _, _ = _split_url(method, url)
    literal_ip = _ip_obj_for_host(host)
    if literal_ip is not None:
        addresses = [literal_ip.compressed]
    else:
        try:
            addresses = await _resolve_host_async(host, port)
        except _DNSError as exc:
            _raise_pin(str(exc), original_url=url, method=method)
    return _resolve_target_common(
        method,
        url,
        allowed_private_cidrs=allowed_private_cidrs,
        addresses=addresses,
    )


# ---------------------------------------------------------------------------
# Header- und Kwargs-Validierung
# ---------------------------------------------------------------------------


def _check_no_caller_host_header(headers: Any) -> None:
    if headers is None:
        return
    if isinstance(headers, dict):
        items = headers.items()
    elif isinstance(headers, (list, tuple)):
        items = headers
    else:
        try:
            items = list(headers.items())
        except (AttributeError, TypeError):
            raise ValueError("headers hat eine ungueltige Form (erwartet dict oder Items-iterable)")
    for entry in items:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("headers-Eintrag hat eine ungueltige Form")
        name = entry[0]
        if not isinstance(name, (str, bytes)):
            raise ValueError("headers-Schluessel muss str oder bytes sein")
        name_str = name.decode("ascii", errors="replace") if isinstance(name, bytes) else name
        if name_str.lower() == "host":
            raise ValueError(
                "Caller-supplied Host-Header ist nicht erlaubt; "
                "Host wird durch den Pinning-Helper gesetzt"
            )


def _check_request_kwargs(kwargs: dict[str, Any]) -> None:
    for name in kwargs:
        if name in _REQUEST_RESERVED_KWARGS:
            raise ValueError(
                f"Caller-Kwarg {name!r} ist reserviert und darf nicht uebergeben werden"
            )
        if name not in _REQUEST_ALLOWED_KWARGS:
            raise ValueError(f"Caller-Kwarg {name!r} ist nicht erlaubt fuer pinned_request")
    _check_no_caller_host_header(kwargs.get("headers"))


def _check_client_kwargs(kwargs: dict[str, Any]) -> None:
    for name in kwargs:
        if name in _REQUEST_RESERVED_KWARGS:
            raise ValueError(
                f"Caller-Kwarg {name!r} ist reserviert und darf nicht uebergeben werden"
            )
        if name not in _CLIENT_ALLOWED_KWARGS:
            raise ValueError(f"Caller-Kwarg {name!r} ist nicht erlaubt fuer pinned-Client-Factory")
    _check_no_caller_host_header(kwargs.get("headers"))


# ---------------------------------------------------------------------------
# Default-Limits (Keep-Alive aus)
# ---------------------------------------------------------------------------


def _default_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=100,
        max_keepalive_connections=0,
        keepalive_expiry=0,
    )


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


def _validate_transport_request(request: httpx.Request, target: PinnedTarget) -> None:
    """Prueft, dass httpx den Original-Host gesetzt hat. Wirft ``ValueError``."""
    host_header = request.headers.get("host")
    if host_header is not None and host_header.lower() != target.netloc.lower():
        raise ValueError(
            f"httpx hat unerwarteten Host-Header gesetzt: {host_header!r}"
            f" (erwartet {target.netloc!r})"
        )


def _clone_request_for_ip(request: httpx.Request, target: PinnedTarget) -> httpx.Request:
    new_extensions = dict(request.extensions or {})
    new_extensions["sni_hostname"] = target.host
    cloned = httpx.Request(
        method=request.method,
        url=httpx.URL(target.pinned_url),
        headers=request.headers,
        stream=request.stream,
        extensions=new_extensions,
    )
    cloned.headers["host"] = target.netloc
    return cloned


class PinnedHTTPTransport(httpx.HTTPTransport):
    """Sync-Transport, das pro Request DNS aufloest und IP-pinned verbindet.

    Akzeptiert keine TLS-/Trust-Override-Kwargs (``verify``, ``cert``,
    ``trust_env``, ``proxy``); ``verify=True`` ist invariant. Caller, die
    eine eigene TLS-Konfig brauchen, muessen sie ueber httpx-eigene Pfade
    machen, NICHT ueber den Pinning-Transport.
    """

    def __init__(
        self,
        *,
        allowed_private_cidrs: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        for name in kwargs:
            if name in _TRANSPORT_RESERVED_KWARGS:
                raise ValueError(
                    f"Caller-Kwarg {name!r} ist reserviert und darf nicht "
                    "an PinnedHTTPTransport uebergeben werden"
                )
        self._allowed_cidrs = allowed_private_cidrs
        # Eager-Validate, damit ungueltige Config sofort knallt.
        _parse_cidrs(allowed_private_cidrs)
        kwargs["verify"] = True
        kwargs.setdefault("limits", _default_limits())
        super().__init__(**kwargs)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)
        target = _resolve_pinned_target_sync(
            request.method,
            original_url,
            allowed_private_cidrs=self._allowed_cidrs,
        )
        _validate_transport_request(request, target)
        cloned = _clone_request_for_ip(request, target)
        response = super().handle_request(cloned)
        response.request = request
        return response


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """Async-Pendant zu :class:`PinnedHTTPTransport` mit identischen Invarianten."""

    def __init__(
        self,
        *,
        allowed_private_cidrs: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        for name in kwargs:
            if name in _TRANSPORT_RESERVED_KWARGS:
                raise ValueError(
                    f"Caller-Kwarg {name!r} ist reserviert und darf nicht "
                    "an PinnedAsyncHTTPTransport uebergeben werden"
                )
        self._allowed_cidrs = allowed_private_cidrs
        _parse_cidrs(allowed_private_cidrs)
        kwargs["verify"] = True
        kwargs.setdefault("limits", _default_limits())
        super().__init__(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)
        target = await _resolve_pinned_target_async(
            request.method,
            original_url,
            allowed_private_cidrs=self._allowed_cidrs,
        )
        _validate_transport_request(request, target)
        cloned = _clone_request_for_ip(request, target)
        response = await super().handle_async_request(cloned)
        response.request = request
        return response


# ---------------------------------------------------------------------------
# Client-Factories
# ---------------------------------------------------------------------------


def create_pinned_client_sync(
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
    **client_kwargs: Any,
) -> httpx.Client:
    """Erstellt einen ``httpx.Client`` mit gepintetem Transport."""
    _check_client_kwargs(client_kwargs)
    transport = PinnedHTTPTransport(
        allowed_private_cidrs=allowed_private_cidrs,
    )
    return httpx.Client(
        transport=transport,
        verify=True,
        follow_redirects=False,
        **client_kwargs,
    )


def create_pinned_client_async(
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
    **client_kwargs: Any,
) -> httpx.AsyncClient:
    """Erstellt einen ``httpx.AsyncClient`` mit gepintetem Transport."""
    _check_client_kwargs(client_kwargs)
    transport = PinnedAsyncHTTPTransport(
        allowed_private_cidrs=allowed_private_cidrs,
    )
    return httpx.AsyncClient(
        transport=transport,
        verify=True,
        follow_redirects=False,
        **client_kwargs,
    )


# ---------------------------------------------------------------------------
# One-shot-Helfer
# ---------------------------------------------------------------------------


def pinned_request_sync(
    method: str,
    url: str,
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
    **httpx_kwargs: Any,
) -> httpx.Response:
    """Sync-One-Shot-Request mit IP-Pinning.

    Erwartete Caller-Kwargs (siehe ``_REQUEST_ALLOWED_KWARGS``):
    ``auth``, ``headers``, ``timeout``, ``content``, ``data``, ``json``,
    ``files``, ``params``. Andere Kwargs sind ein echter ``ValueError``.

    URL-Syntax-Fehler (Bad-Port, kaputte IPv6-Brackets, etc.) werden vor
    der httpx-Konstruktion in ``PinningError`` umgewandelt — der Caller
    sieht nie ``httpx.InvalidURL``.
    """
    upper_method = _validate_method(method)
    _check_request_kwargs(httpx_kwargs)
    # Pre-validate URL-Syntax: alle URL-Probleme werden hier zu PinningError,
    # damit httpx.InvalidURL nicht aus der Helper-Boundary leakt.
    _split_url(upper_method, url)
    transport = PinnedHTTPTransport(allowed_private_cidrs=allowed_private_cidrs)
    with httpx.Client(
        transport=transport,
        verify=True,
        follow_redirects=False,
    ) as client:
        return client.request(upper_method, url, **httpx_kwargs)


async def pinned_request_async(
    method: str,
    url: str,
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
    **httpx_kwargs: Any,
) -> httpx.Response:
    """Async-One-Shot-Request mit IP-Pinning. URL-Syntax-Fehler werden in
    ``PinningError`` umgewandelt (siehe :func:`pinned_request_sync`).
    """
    upper_method = _validate_method(method)
    _check_request_kwargs(httpx_kwargs)
    _split_url(upper_method, url)
    transport = PinnedAsyncHTTPTransport(allowed_private_cidrs=allowed_private_cidrs)
    async with httpx.AsyncClient(
        transport=transport,
        verify=True,
        follow_redirects=False,
    ) as client:
        return await client.request(upper_method, url, **httpx_kwargs)
