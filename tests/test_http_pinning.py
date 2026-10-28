"""Tests fuer kiara.shared.http_pinning.

Decken den Pinning-Vertrag ab: Schema-/Host-/Credential-Pruefung,
DNS-Aufloesung mit IP-Klassifikation, Allowlist-Validierung, kw-only-Kontrakt,
Transport-Verhalten (SNI/Host-Header/clone) und Error-Modell.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket

import httpx
import pytest

from kiara.shared.http_pinning import (
    PinnedAsyncHTTPTransport,
    PinnedHTTPTransport,
    PinnedTarget,
    PinningConfigError,
    PinningError,
    _resolve_pinned_target_async,
    _resolve_pinned_target_sync,
    create_pinned_client_sync,
    pinned_request_async,
    pinned_request_sync,
    validate_allowed_private_cidrs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addrinfo(addr: str, port: int = 443):
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET
    sockaddr = (addr, port, 0, 0) if family == socket.AF_INET6 else (addr, port)
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]


def _patch_dns(monkeypatch, mapping: dict[str, list[str]]):
    """Patch ``socket.getaddrinfo`` mit einem expliziten Host->IPs Mapping."""

    def fake(host, port, *args, **kwargs):
        if host in mapping:
            infos = []
            for ip in mapping[host]:
                infos.extend(_addrinfo(ip, port if port else 0))
            return infos
        raise socket.gaierror(f"unknown host: {host}")

    monkeypatch.setattr(socket, "getaddrinfo", fake)


def _capture_transport(monkeypatch, *, fail: bool = False):
    """Faengt ``httpx.HTTPTransport.handle_request`` ab und liefert die Liste."""
    captured: list[httpx.Request] = []

    def fake_handle(self, request):
        captured.append(request)
        if fail:
            raise httpx.ConnectError("simuliert", request=request)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
    return captured


def _capture_async_transport(monkeypatch):
    captured: list[httpx.Request] = []

    async def fake_handle(self, request):
        captured.append(request)
        return httpx.Response(200, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_handle)
    return captured


# ---------------------------------------------------------------------------
# Allowlist-Validierung
# ---------------------------------------------------------------------------


class TestValidateAllowlist:
    def test_none_is_ok(self):
        validate_allowed_private_cidrs(None)

    def test_empty_list_is_ok(self):
        validate_allowed_private_cidrs([])

    def test_valid_cidr(self):
        validate_allowed_private_cidrs(["10.0.0.0/8", "192.168.0.0/16"])

    def test_str_input_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs("10.0.0.0/8")  # type: ignore[arg-type]

    def test_bytes_input_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs(b"10.0.0.0/8")  # type: ignore[arg-type]

    def test_non_string_element_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs([123])  # type: ignore[list-item]

    def test_unparseable_cidr(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs(["abc"])

    def test_loopback_cidr_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs(["127.0.0.0/8"])

    def test_multicast_cidr_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs(["224.0.0.0/4"])

    def test_link_local_cidr_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs(["169.254.0.0/16"])

    def test_ipv6_loopback_cidr_rejected(self):
        with pytest.raises(PinningConfigError):
            validate_allowed_private_cidrs(["::1/128"])

    def test_ipv6_ula_allowed(self):
        # ULA fc00::/7 ist is_private aber NICHT is_reserved → erlaubt
        validate_allowed_private_cidrs(["fc00::/7"])

    def test_pinning_config_error_is_value_error(self):
        # PinningConfigError ist ValueError, aber NICHT httpx.RequestError.
        assert issubclass(PinningConfigError, ValueError)
        assert not issubclass(PinningConfigError, httpx.RequestError)


# ---------------------------------------------------------------------------
# URL-Validierung (Scheme/Host/Credentials)
# ---------------------------------------------------------------------------


class TestUrlValidation:
    def test_blocks_ftp_scheme(self, monkeypatch):
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "ftp://example.de/")
        assert "Scheme" in exc.value.reason

    def test_blocks_empty_host(self):
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "http:///path")
        assert "EmptyHost" in exc.value.reason

    def test_case_insensitive_scheme(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        target = _resolve_pinned_target_sync("GET", "HTTPS://EXAMPLE.de:8443/x")
        assert target.scheme == "https"
        assert target.host == "example.de"
        assert target.port == 8443
        # Netloc enthaelt Original-Host (lowercased durch urlsplit) plus Port.
        assert target.netloc == "example.de:8443"

    def test_blocks_credentials_in_url_with_redact(self, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="kiara.shared.http_pinning")
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "http://user:pw@example.de/")
        assert "Credentials" in exc.value.reason
        # Log darf user:pw NICHT enthalten.
        assert "user:pw" not in str(exc.value)
        assert "***@" in exc.value.log_url
        joined = " ".join(rec.getMessage() for rec in caplog.records)
        assert "user:pw" not in joined
        assert "***@" in joined


# ---------------------------------------------------------------------------
# IP-Klassifikation
# ---------------------------------------------------------------------------


class TestIpClassification:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "::1",
            "10.0.0.5",
            "192.168.1.1",
            "169.254.1.2",
            "100.64.0.1",
            "0.0.0.0",
            "240.0.0.1",
            "224.0.0.1",
            "ff00::1",
            "fe80::1",
            "::ffff:8.8.8.8",
            "::ffff:127.0.0.1",
            "255.255.255.255",
        ],
    )
    def test_blocks_private_or_hardblock(self, host):
        # IP-Literal: kein DNS noetig
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", f"https://{_brackets(host)}/")
        assert "Klassifikation" in exc.value.reason or "Allowlist" in exc.value.reason

    def test_multicast_blocked_despite_is_global(self):
        # 224.0.0.1 hat is_global=True in CPython 3.12, muss aber blockiert werden.
        addr = ipaddress.ip_address("224.0.0.1")
        assert addr.is_global is True
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "https://224.0.0.1/")
        assert "Klassifikation" in exc.value.reason
        assert "multicast" in exc.value.reason

    def test_allows_global_ipv4(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        target = _resolve_pinned_target_sync("GET", "https://example.de/")
        assert target.ip == "93.184.216.34"
        assert target.host == "example.de"

    def test_ipv4_literal_no_dns(self, monkeypatch):
        # getaddrinfo darf NICHT aufgerufen werden.
        called = {"n": 0}

        def fake(*args, **kwargs):
            called["n"] += 1
            return _addrinfo("0.0.0.0")

        monkeypatch.setattr(socket, "getaddrinfo", fake)
        target = _resolve_pinned_target_sync("GET", "https://93.184.216.34/")
        assert called["n"] == 0
        assert target.ip == "93.184.216.34"

    def test_ipv6_literal_no_dns(self, monkeypatch):
        called = {"n": 0}

        def fake(*args, **kwargs):
            called["n"] += 1
            return _addrinfo("::1")

        monkeypatch.setattr(socket, "getaddrinfo", fake)
        target = _resolve_pinned_target_sync("GET", "https://[2606:4700:4700::1111]:8080/")
        assert called["n"] == 0
        assert target.ip in ("2606:4700:4700::1111",)
        assert target.netloc == "[2606:4700:4700::1111]:8080"

    def test_mixed_dns_blocks_when_any_is_blocked(self, monkeypatch):
        # Eine IP ist global, eine ist Loopback → komplett blockieren.
        _patch_dns(
            monkeypatch,
            {"mixed.example.de": ["93.184.216.34", "127.0.0.1"]},
        )
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "https://mixed.example.de/")
        assert "Klassifikation" in exc.value.reason

    def test_dns_failure_blocked(self, monkeypatch):
        def fake(*args, **kwargs):
            raise socket.gaierror("unknown")

        monkeypatch.setattr(socket, "getaddrinfo", fake)
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "https://nonexistent.example.de/")
        assert "DNS" in exc.value.reason


def _brackets(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


# ---------------------------------------------------------------------------
# Allowlist-Verhalten
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_none_and_empty_equivalent(self, monkeypatch):
        _patch_dns(monkeypatch, {"i.example.de": ["10.0.0.5"]})
        with pytest.raises(PinningError):
            _resolve_pinned_target_sync("GET", "https://i.example.de/", allowed_private_cidrs=None)
        with pytest.raises(PinningError):
            _resolve_pinned_target_sync("GET", "https://i.example.de/", allowed_private_cidrs=[])

    def test_allowed_private_cidr_permits(self, monkeypatch):
        _patch_dns(monkeypatch, {"i.example.de": ["10.0.0.5"]})
        target = _resolve_pinned_target_sync(
            "GET",
            "https://i.example.de/",
            allowed_private_cidrs=["10.0.0.0/8"],
        )
        assert target.ip == "10.0.0.5"

    def test_allowlist_does_not_override_hard_block(self):
        # Allowlist enthaelt 10.0.0.0/8, aber 127.0.0.1 ist Hard-Block.
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync(
                "GET",
                "https://127.0.0.1/",
                allowed_private_cidrs=["10.0.0.0/8"],
            )
        assert "Klassifikation" in exc.value.reason


# ---------------------------------------------------------------------------
# Transport: SNI, Host-Header, Original-Request bleibt
# ---------------------------------------------------------------------------


class TestPinnedTransport:
    def test_sets_sni_hostname_and_host(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        captured = _capture_transport(monkeypatch)

        transport = PinnedHTTPTransport()
        original = httpx.Request("GET", "https://example.de:8443/abc?q=1")
        response = transport.handle_request(original)

        assert response.request is original  # Originals werden nicht ersetzt
        assert len(captured) == 1
        pinned = captured[0]
        assert str(pinned.url) == "https://93.184.216.34:8443/abc?q=1"
        assert pinned.headers["host"].lower() == "example.de:8443"
        assert pinned.extensions.get("sni_hostname") == "example.de"

    def test_default_port_omitted_in_host_header(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        captured = _capture_transport(monkeypatch)

        transport = PinnedHTTPTransport()
        original = httpx.Request("GET", "https://example.de/")
        transport.handle_request(original)

        assert captured[0].headers["host"].lower() == "example.de"

    def test_does_not_mutate_original_request(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        _capture_transport(monkeypatch)

        transport = PinnedHTTPTransport()
        original = httpx.Request("GET", "https://example.de/abc")
        url_before = str(original.url)
        host_before = original.headers.get("host")
        transport.handle_request(original)
        assert str(original.url) == url_before
        assert original.headers.get("host") == host_before

    def test_keep_alive_disabled_by_default(self):
        # Einfach pruefen, dass der Default-Limits-Wert keep-alive=0 setzt.
        from kiara.shared.http_pinning import _default_limits

        limits = _default_limits()
        assert limits.max_keepalive_connections == 0
        assert limits.keepalive_expiry == 0

    def test_pinning_config_error_raised_on_init(self):
        with pytest.raises(PinningConfigError):
            PinnedHTTPTransport(allowed_private_cidrs=["abc"])

    def test_response_request_url_is_original_netloc(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        _capture_transport(monkeypatch)

        transport = PinnedHTTPTransport()
        original = httpx.Request("GET", "https://example.de:8443/abc")
        response = transport.handle_request(original)
        assert response.request.url.host == "example.de"
        assert response.request.url.port == 8443


# ---------------------------------------------------------------------------
# Async-Transport
# ---------------------------------------------------------------------------


class TestPinnedAsyncTransport:
    def test_async_resolve_uses_loop(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})

        async def run():
            target = await _resolve_pinned_target_async("GET", "https://example.de/")
            return target

        target = asyncio.run(run())
        assert target.ip == "93.184.216.34"

    def test_async_transport_sets_sni(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        captured = _capture_async_transport(monkeypatch)

        async def run():
            transport = PinnedAsyncHTTPTransport()
            original = httpx.Request("GET", "https://example.de:8443/abc")
            response = await transport.handle_async_request(original)
            return response, original

        response, original = asyncio.run(run())
        assert response.request is original
        assert captured[0].extensions.get("sni_hostname") == "example.de"
        assert captured[0].headers["host"].lower() == "example.de:8443"


# ---------------------------------------------------------------------------
# Caller-Kwargs / Methoden-Allowlist
# ---------------------------------------------------------------------------


class TestCallerKwargs:
    def test_invalid_method_raises_value_error(self):
        with pytest.raises(ValueError):
            pinned_request_sync("FOOBAR", "https://example.de/")

    def test_caller_supplied_host_header_raises_value_error(self):
        with pytest.raises(ValueError):
            pinned_request_sync(
                "GET",
                "https://example.de/",
                headers={"Host": "evil.example"},
            )

    def test_invalid_header_shape_raises_value_error(self):
        with pytest.raises(ValueError):
            pinned_request_sync("GET", "https://example.de/", headers=42)

    @pytest.mark.parametrize(
        "kw",
        [
            "verify",
            "cert",
            "trust_env",
            "event_hooks",
            "transport",
            "follow_redirects",
        ],
    )
    def test_reserved_kwargs_raise_value_error(self, kw):
        with pytest.raises(ValueError):
            pinned_request_sync("GET", "https://example.de/", **{kw: True})

    def test_disallowed_kwarg_raises_value_error(self):
        with pytest.raises(ValueError):
            pinned_request_sync("GET", "https://example.de/", extensions={})
        with pytest.raises(ValueError):
            pinned_request_sync("GET", "https://example.de/", proxy="http://p")
        with pytest.raises(ValueError):
            pinned_request_sync("GET", "https://example.de/", cookies={})

    def test_client_factory_rejects_request_only_kwargs(self):
        with pytest.raises(ValueError):
            create_pinned_client_sync(content=b"x")
        with pytest.raises(ValueError):
            create_pinned_client_sync(json={"a": 1})
        with pytest.raises(ValueError):
            create_pinned_client_sync(follow_redirects=True)

    def test_client_factory_accepts_session_kwargs(self):
        with create_pinned_client_sync(
            timeout=5.0, headers={"User-Agent": "x"}, auth=None
        ) as client:
            assert client.timeout.read == 5.0


# ---------------------------------------------------------------------------
# pinned_request_sync — End-to-End mit gemocktem Transport
# ---------------------------------------------------------------------------


class TestPinnedRequestSync:
    def test_happy_path_global_ip(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        captured = _capture_transport(monkeypatch)

        resp = pinned_request_sync("GET", "https://example.de/abc")
        assert resp.status_code == 200
        assert captured[0].extensions.get("sni_hostname") == "example.de"

    def test_blocks_private_without_allowlist(self, monkeypatch):
        _patch_dns(monkeypatch, {"intern.example.de": ["10.0.0.1"]})
        with pytest.raises(PinningError):
            pinned_request_sync("GET", "https://intern.example.de/")

    def test_allows_private_with_allowlist(self, monkeypatch):
        _patch_dns(monkeypatch, {"intern.example.de": ["10.0.0.1"]})
        captured = _capture_transport(monkeypatch)
        resp = pinned_request_sync(
            "GET",
            "https://intern.example.de/",
            allowed_private_cidrs=["10.0.0.0/8"],
        )
        assert resp.status_code == 200
        assert captured[0].extensions.get("sni_hostname") == "intern.example.de"


# ---------------------------------------------------------------------------
# Async-One-Shot
# ---------------------------------------------------------------------------


class TestPinnedRequestAsync:
    def test_async_happy_path(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        captured = _capture_async_transport(monkeypatch)

        async def run():
            return await pinned_request_async("GET", "https://example.de/abc")

        resp = asyncio.run(run())
        assert resp.status_code == 200
        assert captured[0].extensions.get("sni_hostname") == "example.de"

    def test_async_blocks_private(self, monkeypatch):
        _patch_dns(monkeypatch, {"intern.example.de": ["10.0.0.1"]})

        async def run():
            return await pinned_request_async("GET", "https://intern.example.de/")

        with pytest.raises(PinningError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# PinningError-API
# ---------------------------------------------------------------------------


class TestPinningErrorApi:
    def test_pinning_error_is_request_error(self):
        assert issubclass(PinningError, httpx.RequestError)

    def test_message_contains_address_and_reason(self):
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "https://127.0.0.1/")
        assert "127.0.0.1" in str(exc.value)
        assert "Klassifikation" in str(exc.value)

    def test_request_attribute_is_real_httpx_request(self):
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "https://127.0.0.1/")
        assert isinstance(exc.value.request, httpx.Request)

    def test_log_url_redacted_for_credentials(self):
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "http://user:secret@example.de/")
        assert "secret" not in exc.value.log_url
        assert "secret" not in str(exc.value)
        assert "***@" in exc.value.log_url


# ---------------------------------------------------------------------------
# PinnedTarget
# ---------------------------------------------------------------------------


class TestPinnedTarget:
    def test_target_is_frozen_dataclass(self, monkeypatch):
        _patch_dns(monkeypatch, {"example.de": ["93.184.216.34"]})
        target = _resolve_pinned_target_sync("GET", "https://example.de:8443/x")
        assert isinstance(target, PinnedTarget)
        assert target.original_url == "https://example.de:8443/x"
        assert target.host == "example.de"
        assert target.netloc == "example.de:8443"
        assert target.port == 8443
        assert target.scheme == "https"
        assert target.ip == "93.184.216.34"
        assert target.pinned_url.startswith("https://93.184.216.34:8443/")
        with pytest.raises((AttributeError, Exception)):
            target.host = "evil.example"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Malformed URLs (Finding 1) — fail-closed via PinningError, kein 500
# ---------------------------------------------------------------------------


class TestMalformedUrls:
    """User-/Admin-Probe-URLs muessen IMMER als PinningError enden,
    nicht als Programmierfehler (ValueError, httpx.InvalidURL).
    """

    @pytest.mark.parametrize(
        "url,expected_reason_substring",
        [
            ("http://example.de:bad/", "Port"),
            ("http://example.de:99999999/", "Port"),
            ("http://[::1/", "URL nicht parsebar"),
            ("http://[::1]:bad/", "Port"),
            ("not a url", "Scheme"),
            ("", "EmptyHost"),
        ],
    )
    def test_malformed_url_raises_pinning_error(self, url, expected_reason_substring):
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", url)
        # PinningError ist httpx.RequestError, nicht ValueError/InvalidURL
        assert isinstance(exc.value, httpx.RequestError)
        assert not isinstance(exc.value, httpx.InvalidURL)
        assert expected_reason_substring.lower() in exc.value.reason.lower()

    def test_malformed_url_with_credentials_redacts_log(self, caplog):
        """user:pw@host:bad muss credentials redaktieren UND keinen 500 werfen."""
        caplog.set_level(logging.WARNING, logger="kiara.shared.http_pinning")
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", "http://user:supersecret@example.de:bad/")
        assert "supersecret" not in str(exc.value)
        assert "supersecret" not in exc.value.log_url
        assert "***@" in exc.value.log_url
        joined = " ".join(rec.getMessage() for rec in caplog.records)
        assert "supersecret" not in joined

    def test_async_path_handles_malformed_urls(self):
        async def run():
            with pytest.raises(PinningError):
                await _resolve_pinned_target_async("GET", "http://x:bad/")

        asyncio.run(run())

    @pytest.mark.parametrize(
        "url,secret",
        [
            ("http://user:supersecret@[::1/", "supersecret"),
            ("http://user:topsecret@[bad-ipv6/", "topsecret"),
            ("http://user:sneaky@example.de:bad@/", "sneaky"),
        ],
    )
    def test_unparsable_url_with_credentials_no_leak(self, url, secret, caplog):
        """Auch wenn urlsplit fehlschlaegt: Userinfo darf NICHT ins Log
        oder in str(exc) leaken."""
        caplog.set_level(logging.WARNING, logger="kiara.shared.http_pinning")
        with pytest.raises(PinningError) as exc:
            _resolve_pinned_target_sync("GET", url)
        # Weder log_url noch Exception-Message duerfen das Secret enthalten
        assert secret not in exc.value.log_url
        assert secret not in str(exc.value)
        assert secret not in exc.value.reason
        joined = " ".join(rec.getMessage() for rec in caplog.records)
        assert secret not in joined


# ---------------------------------------------------------------------------
# Transport-Invarianten (Finding 3) — kein verify=False oder Trust-Override
# ---------------------------------------------------------------------------


class TestTransportInvariants:
    @pytest.mark.parametrize(
        "kw,val",
        [
            ("verify", False),
            ("cert", "x"),
            ("trust_env", False),
            ("proxy", "http://p"),
            ("uds", "/tmp/x.sock"),
            ("local_address", "1.2.3.4"),
            ("socket_options", []),
        ],
    )
    def test_sync_transport_rejects_reserved_kwarg(self, kw, val):
        with pytest.raises(ValueError):
            PinnedHTTPTransport(**{kw: val})

    @pytest.mark.parametrize(
        "kw,val",
        [
            ("verify", False),
            ("cert", "x"),
            ("trust_env", False),
            ("proxy", "http://p"),
            ("uds", "/tmp/x.sock"),
            ("local_address", "1.2.3.4"),
            ("socket_options", []),
        ],
    )
    def test_async_transport_rejects_reserved_kwarg(self, kw, val):
        with pytest.raises(ValueError):
            PinnedAsyncHTTPTransport(**{kw: val})

    def test_sync_transport_explicit_verify_false_blocked(self):
        with pytest.raises(ValueError):
            PinnedHTTPTransport(verify=False)

    def test_sync_transport_uds_blocked_does_not_set_pool_uds(self):
        """Pinning-Transport-UDS-Bypass: Connection-Pool darf kein _uds bekommen."""
        with pytest.raises(ValueError):
            PinnedHTTPTransport(uds="/tmp/x.sock")

    def test_async_transport_uds_blocked(self):
        with pytest.raises(ValueError):
            PinnedAsyncHTTPTransport(uds="/tmp/x.sock")


# ---------------------------------------------------------------------------
# httpcore SNI-Vertrag (Finding 2) — Frühwarnsystem
# ---------------------------------------------------------------------------


class TestHttpcoreSniContract:
    def test_httpcore_supports_sni_hostname_extension(self):
        """httpcore liest extensions['sni_hostname'] in der Connection.

        Frueherkennung: wird die interne API entfernt (z. B. in einer neuen
        Minor-Version), bricht der Pinning-Vertrag — der Test schlaegt dann
        an, bevor Production-Pfade Zertifikatsfehler werfen.
        """
        import inspect

        modules = []
        try:
            from httpcore._sync import connection as sync_conn  # type: ignore

            modules.append(sync_conn)
        except ImportError:
            pass
        try:
            from httpcore._async import connection as async_conn  # type: ignore

            modules.append(async_conn)
        except ImportError:
            pass

        assert modules, "httpcore-Connection-Module nicht gefunden"

        found = False
        for mod in modules:
            try:
                src = inspect.getsource(mod)
            except OSError:
                continue
            if "sni_hostname" in src:
                found = True
                break
        assert found, (
            "httpcore unterstuetzt 'sni_hostname' nicht mehr — Pinning-Vertrag "
            "gebrochen! Pinning-Helper anpassen oder Dependency-Pin anpassen."
        )
