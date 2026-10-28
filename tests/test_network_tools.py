"""Tests fuer Sprint 07: Netzwerk-Diagnose-Tools und Service-Management.

34 Tests, alle mit Mocks — kein echtes Netzwerk, kein systemctl.
"""

import asyncio
import inspect
import socket
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiara.admin.result import AdminResult

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


USER = {"id": 1}


# ===========================================================================
# network.ping
# ===========================================================================


class TestNetworkPing:
    def test_network_ping_success(self):
        """Mock subprocess: ping erfolgreich — Output-Parsing, AdminResult."""
        from kiara.admin.functions.network import network_ping

        mock_result = MagicMock()
        mock_result.stdout = (
            "PING localhost (127.0.0.1) 56(84) bytes of data.\n"
            "64 bytes from localhost (127.0.0.1): icmp_seq=1 ttl=64 time=0.023 ms\n"
            "64 bytes from localhost (127.0.0.1): icmp_seq=2 ttl=64 time=0.045 ms\n"
            "64 bytes from localhost (127.0.0.1): icmp_seq=3 ttl=64 time=0.067 ms\n"
            "\n"
            "--- localhost ping statistics ---\n"
            "3 packets transmitted, 3 received, 0% packet loss, time 2003ms\n"
            "rtt min/avg/max/mdev = 0.023/0.045/0.067/0.018 ms\n"
        )
        mock_result.returncode = 0

        with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
            result = _run(network_ping(USER, "localhost"))

        assert result.success is True
        assert result.data["reachable"] is True
        assert result.data["packets_sent"] == 3
        assert result.data["packets_received"] == 3
        assert result.data["packet_loss"] == "0%"
        assert result.data["rtt_avg"] == "0.045ms"

    def test_network_ping_unreachable(self):
        """Mock subprocess: 100% Paketverlust — Fehlerbehandlung."""
        from kiara.admin.functions.network import network_ping

        mock_result = MagicMock()
        mock_result.stdout = (
            "PING 192.168.99.99 (192.168.99.99) 56(84) bytes of data.\n"
            "\n"
            "--- 192.168.99.99 ping statistics ---\n"
            "3 packets transmitted, 0 received, 100% packet loss, time 2002ms\n"
        )
        mock_result.returncode = 1

        with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
            result = _run(network_ping(USER, "192.168.99.99"))

        assert result.success is False
        assert result.data["reachable"] is False
        assert result.data["packet_loss"] == "100%"

    def test_network_ping_timeout(self):
        """Mock: TimeoutExpired — Timeout-Handling."""
        from kiara.admin.functions.network import network_ping

        with patch(
            "kiara.admin.functions.network.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ping", timeout=15),
        ):
            result = _run(network_ping(USER, "10.0.0.1"))

        assert result.success is False
        assert "Timeout" in result.message

    def test_network_ping_not_installed(self):
        """Mock FileNotFoundError fuer ping — ping-Binary fehlt."""
        from kiara.admin.functions.network import network_ping

        with patch(
            "kiara.admin.functions.network.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = _run(network_ping(USER, "10.0.0.1"))

        assert result.success is False
        assert "nicht installiert" in result.message


# ===========================================================================
# network.portscan
# ===========================================================================


class TestNetworkPortscan:
    def test_network_portscan_nmap(self):
        """Mock subprocess: nmap output — nmap-Parsing."""
        from kiara.admin.functions.network import network_portscan

        mock_result = MagicMock()
        mock_result.stdout = (
            "Starting Nmap 7.80 ...\n"
            "PORT    STATE  SERVICE\n"
            "445/tcp open   microsoft-ds\n"
            "80/tcp  closed http\n"
            "Nmap done: 1 IP address (1 host up)\n"
        )
        mock_result.returncode = 0

        with patch("kiara.admin.functions.network.shutil.which", return_value="/usr/bin/nmap"):
            with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
                result = _run(network_portscan(USER, "fileserver.local", "445,80"))

        assert result.success is True
        assert result.data["ports"][0]["port"] == 445
        assert result.data["ports"][0]["state"] == "open"
        assert result.data["ports"][1]["port"] == 80
        assert result.data["ports"][1]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_network_portscan_socket_fallback(self):
        """Mock shutil.which=None, mock socket — Fallback."""
        from kiara.admin.functions.network import network_portscan

        def mock_connect_ex(addr):
            return 0 if addr[1] == 445 else 1

        with patch("kiara.admin.functions.network.shutil.which", return_value=None):
            with patch("kiara.admin.functions.network.socket.socket") as mock_sock_cls:
                mock_sock = MagicMock()
                mock_sock.connect_ex = mock_connect_ex
                mock_sock_cls.return_value = mock_sock
                result = await network_portscan(USER, "fileserver.local", "445,80")

        assert result.success is True
        assert result.data["ports"][0]["state"] == "open"
        assert result.data["ports"][1]["state"] == "closed"
        assert "Socket-Fallback" in result.message

    def test_portscan_invalid_port(self):
        """ports="abc" oder "99999" — Port-Validierung (1-65535)."""
        from kiara.admin.functions.network import network_portscan

        result = _run(network_portscan(USER, "10.0.0.1", "abc"))
        assert result.success is False
        assert "Ungueltiger Port" in result.message

        result = _run(network_portscan(USER, "10.0.0.1", "99999"))
        assert result.success is False
        assert "Ungueltiger Port" in result.message


# ===========================================================================
# network.dns
# ===========================================================================


class TestNetworkDns:
    def test_network_dns_dig(self):
        """Mock subprocess: dig output — dig-Parsing."""
        from kiara.admin.functions.network import network_dns

        mock_result = MagicMock()
        mock_result.stdout = "otel-collector.local\n"
        mock_result.returncode = 0

        with patch("kiara.admin.functions.network.shutil.which", return_value="/usr/bin/dig"):
            with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
                result = _run(network_dns(USER, "nextcloud.firma.de"))

        assert result.success is True
        assert "otel-collector.local" in result.data["records"]
        assert result.data["type"] == "A"

    def test_network_dns_socket_fallback(self):
        """Mock shutil.which=None, mock socket — Fallback."""
        from kiara.admin.functions.network import network_dns

        mock_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("otel-collector.local", 0)),
        ]

        with patch("kiara.admin.functions.network.shutil.which", return_value=None):
            with patch(
                "kiara.admin.functions.network.socket.getaddrinfo", return_value=mock_addrinfo
            ):
                result = _run(network_dns(USER, "nextcloud.firma.de"))

        assert result.success is True
        assert "otel-collector.local" in result.data["records"]

    def test_network_dns_mx_without_dig(self):
        """Mock shutil.which=None, type="MX" — MX/CNAME/PTR ohne dig -> AdminResult.fail."""
        from kiara.admin.functions.network import network_dns

        with patch("kiara.admin.functions.network.shutil.which", return_value=None):
            result = _run(network_dns(USER, "firma.de", type="MX"))

        assert result.success is False
        assert "dig nicht installiert" in result.message

    def test_dns_gaierror(self):
        """Mock socket.gaierror — Unbekannter Hostname."""
        from kiara.admin.functions.network import network_dns

        with (
            patch("kiara.admin.functions.network.shutil.which", return_value=None),
            patch(
                "kiara.admin.functions.network.socket.getaddrinfo",
                side_effect=socket.gaierror("Name or service not known"),
            ),
        ):
            result = _run(network_dns(USER, "nichtexistent.invalid"))

        assert result.success is False
        assert "fehlgeschlagen" in result.message


# ===========================================================================
# network.smb_discover
# ===========================================================================


class TestNetworkSmbDiscover:
    def test_network_smb_discover_auth_file(self):
        """Mock subprocess + tempfile — Authentication-File-Pattern."""
        from kiara.admin.functions.network import network_smb_discover

        mock_result = MagicMock()
        mock_result.stdout = (
            "\tSharename       Type      Comment\n"
            "\t---------       ----      -------\n"
            "\tDaten           Disk      Gemeinsame Dateien\n"
            "\tIPC$            IPC       IPC Service\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.network.subprocess.run", return_value=mock_result
        ) as mock_run:
            with patch(
                "kiara.admin.functions.network.tempfile.mkstemp",
                return_value=(99, "/tmp/test.cred"),
            ):
                with patch("kiara.admin.functions.network.os.write"):
                    with patch("kiara.admin.functions.network.os.close"):
                        with patch("kiara.admin.functions.network.os.chmod"):
                            with patch("kiara.admin.functions.network.os.unlink"):
                                result = _run(
                                    network_smb_discover(
                                        USER,
                                        "fileserver.local",
                                        username="admin",
                                        password="secret",
                                        domain="FIRMA",
                                    )
                                )

        assert result.success is True
        assert len(result.data["shares"]) == 2
        assert result.data["shares"][0]["name"] == "Daten"
        assert result.data["shares"][0]["type"] == "Disk"
        # Authentication-File muss benutzt worden sein (nicht -U user%pass)
        call_args = mock_run.call_args[0][0]
        assert any("--authentication-file=" in arg for arg in call_args)

    def test_network_smb_discover_anonymous(self):
        """Mock subprocess mit -N — Anonymer Zugriff."""
        from kiara.admin.functions.network import network_smb_discover

        mock_result = MagicMock()
        mock_result.stdout = (
            "\tSharename       Type      Comment\n"
            "\t---------       ----      -------\n"
            "\tPublic          Disk      Public share\n"
        )
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.network.subprocess.run", return_value=mock_result
        ) as mock_run:
            result = _run(network_smb_discover(USER, "fileserver.local"))

        assert result.success is True
        assert len(result.data["shares"]) == 1
        # -N muss benutzt worden sein (anonymer Zugriff)
        call_args = mock_run.call_args[0][0]
        assert "-N" in call_args

    def test_smb_discover_not_installed(self):
        """Mock FileNotFoundError — smbclient fehlt."""
        from kiara.admin.functions.network import network_smb_discover

        with patch(
            "kiara.admin.functions.network.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = _run(network_smb_discover(USER, "fileserver.local"))

        assert result.success is False
        assert "nicht installiert" in result.message


# ===========================================================================
# network.webdav_check
# ===========================================================================


class TestNetworkWebdavCheck:
    def test_network_webdav_check_dav(self):
        """Mock httpx: 207 + DAV-Header — DAV-Compliance."""
        from kiara.admin.functions.network import network_webdav_check

        mock_response = MagicMock()
        mock_response.status_code = 207
        mock_response.headers = {"dav": "1, 2, 3", "server": "Apache/2.4"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_webdav_check(USER, "https://nextcloud.firma.de/remote.php/dav"))

        assert result.success is True
        assert result.data["dav_compliant"] is True
        assert result.data["status_code"] == 207

    def test_webdav_check_not_dav(self):
        """Mock httpx: 200 ohne DAV-Header — Nicht-DAV-Erkennung."""
        from kiara.admin.functions.network import network_webdav_check

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"server": "nginx/1.24"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_webdav_check(USER, "https://example.com"))

        assert result.success is True
        assert result.data["dav_compliant"] is False

    def test_webdav_check_invalid_scheme(self):
        """url="ftp://..." — URL-Schema-Validierung."""
        from kiara.admin.functions.network import network_webdav_check

        result = _run(network_webdav_check(USER, "ftp://example.com"))
        assert result.success is False
        assert "http://" in result.message

    def test_webdav_check_timeout(self):
        """Mock httpx.TimeoutException — Timeout-Handling."""
        import httpx as _httpx

        from kiara.admin.functions.network import network_webdav_check

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_webdav_check(USER, "https://slow.example.com"))

        assert result.success is False
        assert "Timeout" in result.message


# ===========================================================================
# network.http_check
# ===========================================================================


class TestNetworkHttpCheck:
    def test_network_http_check_success(self):
        """Mock httpx: 200 — HTTP-Check."""
        from datetime import timedelta

        from kiara.admin.functions.network import network_http_check

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"server": "nginx/1.24"}
        mock_response.elapsed = timedelta(milliseconds=45)

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_http_check(USER, "https://nextcloud.firma.de"))

        assert result.success is True
        assert result.data["status_code"] == 200
        assert result.data["response_time_ms"] == 45

    def test_http_check_invalid_scheme(self):
        """url="ftp://..." — URL-Schema-Validierung."""
        from kiara.admin.functions.network import network_http_check

        result = _run(network_http_check(USER, "ftp://example.com"))
        assert result.success is False
        assert "http://" in result.message

    def test_http_check_timeout(self):
        """Mock httpx.TimeoutException — Timeout-Handling."""
        import httpx as _httpx

        from kiara.admin.functions.network import network_http_check

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_http_check(USER, "https://slow.example.com"))

        assert result.success is False
        assert "Timeout" in result.message


# ===========================================================================
# network.traceroute
# ===========================================================================


class TestNetworkTraceroute:
    def test_network_traceroute_success(self):
        """Mock subprocess: traceroute output — Output-Parsing."""
        from kiara.admin.functions.network import network_traceroute

        mock_result = MagicMock()
        mock_result.stdout = (
            "traceroute to fileserver.local (fileserver.local), 20 hops max, 60 byte packets\n"
            " 1  gateway (10.0.0.1)  0.500 ms  0.400 ms  0.350 ms\n"
            " 2  fileserver (fileserver.local)  1.200 ms  1.100 ms  1.050 ms\n"
        )
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.network.shutil.which", return_value="/usr/bin/traceroute"
        ):
            with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
                result = _run(network_traceroute(USER, "fileserver.local"))

        assert result.success is True
        assert len(result.data["hops"]) == 2
        assert result.data["hops"][0]["hop"] == 1
        assert result.data["hops"][0]["ip"] == "10.0.0.1"
        assert result.data["hops"][1]["hostname"] == "fileserver"

    def test_network_traceroute_not_installed(self):
        """Mock shutil.which=None — Fehlermeldung."""
        from kiara.admin.functions.network import network_traceroute

        with patch("kiara.admin.functions.network.shutil.which", return_value=None):
            result = _run(network_traceroute(USER, "fileserver.local"))

        assert result.success is False
        assert "nicht installiert" in result.message


# ===========================================================================
# services.status
# ===========================================================================


class TestServicesStatus:
    def test_services_status_allowed(self):
        """Mock subprocess: systemctl status — Whitelist-Check."""
        from kiara.admin.functions.services import services_status

        mock_result = MagicMock()
        mock_result.stdout = (
            "● kiara-worker.service - KIara Worker\n"
            "     Loaded: loaded (/etc/systemd/system/kiara-worker.service; enabled)\n"
            "     Active: active (running) since Thu 2026-03-27 08:00:00 CET; 2h 15min ago\n"
            "   Main PID: 12345 (python3)\n"
            "     Memory: 128.0M\n"
        )
        mock_result.returncode = 0

        with patch("kiara.admin.functions.services.subprocess.run", return_value=mock_result):
            result = _run(services_status(USER, "kiara-worker"))

        assert result.success is True
        assert result.data["service"] == "kiara-worker.service"
        assert result.data["status"] == "active"
        assert result.data["pid"] == 12345
        assert result.data["memory"] == "128.0M"

    def test_services_status_denied(self):
        """Service nicht in Whitelist — Whitelist-Ablehnung."""
        from kiara.admin.functions.services import services_status

        result = _run(services_status(USER, "sshd"))
        assert result.success is False
        assert "nicht erlaubt" in result.message


# ===========================================================================
# services.list
# ===========================================================================


class TestServicesList:
    def test_services_list_all(self):
        """Mock subprocess: alle Services — Batch-Status."""
        from kiara.admin.functions.services import ALLOWED_SERVICES, services_list

        output = "\n\n".join(
            f"Id={svc}\nLoadState=loaded\n"
            f"ActiveState={'active' if svc in ('kiara.service', 'kiara-worker.service') else 'inactive'}\n"
            "ActiveEnterTimestamp=@1700000000\nJob=\n"
            for svc in ALLOWED_SERVICES
        )
        with patch(
            "kiara.admin.functions.services.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, output, ""),
        ) as run:
            result = _run(services_list(USER))

        assert result.success is True
        assert len(result.data["services"]) == len(ALLOWED_SERVICES)
        active_count = sum(1 for s in result.data["services"] if s["status"] == "active")
        assert active_count == 2
        run.assert_called_once()
        assert "--timestamp=unix" in run.call_args.args[0]
        for service in result.data["services"]:
            assert service["description"]
            assert service["job_pending"] is False
            expected = "2023-11-14T22:13:20+00:00" if service["status"] == "active" else None
            assert service["active_since"] == expected

    def test_missing_service_and_pending_job(self):
        from kiara.admin.functions.services import services_list

        output = (
            "Id=kiara.service\nLoadState=loaded\nActiveState=active\n"
            "ActiveEnterTimestamp=invalid\nJob=123\n\n"
            "Id=kiara-mcp.service\nLoadState=not-found\nActiveState=inactive\nJob=\n\n"
            "Id=unrelated.service\nLoadState=loaded\nActiveState=active\n"
        )
        with patch(
            "kiara.admin.functions.services.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, output, ""),
        ):
            result = _run(services_list(USER))
        services = {service["name"]: service for service in result.data["services"]}
        assert "unrelated.service" not in services
        assert services["kiara.service"]["job_pending"] is True
        assert services["kiara.service"]["active_since"] is None
        assert services["kiara-mcp.service"]["load_state"] == "not-found"
        assert services["kiara-admin.service"]["status"] == "unknown"
        assert services["kiara-admin.service"]["active_since"] is None

    @pytest.mark.parametrize(
        "error", [subprocess.TimeoutExpired("systemctl", 5), OSError("missing")]
    )
    def test_list_error(self, error):
        from kiara.admin.functions.services import services_list

        with patch("kiara.admin.functions.services.subprocess.run", side_effect=error):
            result = _run(services_list(USER))
        assert result.success is False

    def test_list_does_not_trust_output_after_failed_command(self):
        from kiara.admin.functions.services import services_list

        with patch(
            "kiara.admin.functions.services.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "ActiveState=active", "access denied"),
        ):
            result = _run(services_list(USER))
        assert result.success is False
        assert "access denied" in result.message


# ===========================================================================
# services.logs
# ===========================================================================


class TestServicesLogs:
    def test_services_logs(self):
        """Mock subprocess: journalctl — Log-Ausgabe."""
        from kiara.admin.functions.services import services_logs

        mock_result = MagicMock()
        mock_result.stdout = "2026-03-27T08:00:00+01:00 rag kiara[12345]: Starting worker...\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.services.subprocess.run", return_value=mock_result
        ) as mock_run:
            result = _run(services_logs(USER, "kiara-worker", lines=50))

        assert result.success is True
        assert "Starting worker" in result.data["output"]
        # Argument-Reihenfolge pruefen (muss mit sudoers uebereinstimmen)
        call_args = mock_run.call_args[0][0]
        assert call_args[:4] == ["sudo", "journalctl", "--no-pager", "--output=short-iso"]


# ===========================================================================
# services.restart
# ===========================================================================


class TestServicesActions:
    @pytest.mark.parametrize("action", ["start", "stop", "restart"])
    def test_services_action_confirmation(self, action):
        """Prueft requires_confirmation=True — Confirmation-Flag."""

        # Zugriff auf die registrierte AdminFunction ueber die globale Registry
        from kiara.admin.registry import admin_registry

        func = admin_registry.get(f"services.{action}")
        assert func is not None
        assert func.requires_confirmation is True
        assert func.http_method == "POST"
        assert func.http_path == f"/api/monitoring/services/{{service}}/{action}"
        assert func.changelog_action == action

    @pytest.mark.parametrize("action", ["start", "stop", "restart"])
    @pytest.mark.parametrize(
        "service", ["kiara", "kiara-admin", "kiara-worker", "kiara-mcp", "kiara-admin-chat"]
    )
    def test_services_action_is_queued(self, action, service):
        """Auch eigene Dienste werden ohne Warten auf das Ende des systemd-Jobs gesteuert."""
        from kiara.admin.functions import services

        with patch(
            "kiara.admin.functions.services.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as run:
            result = _run(getattr(services, f"services_{action}")(USER, service))
        assert result.success is True
        assert "angefordert" in result.message
        assert result.data == {"service": f"{service}.service", "action": action}
        run.assert_called_once()
        assert run.call_args.args[0] == [
            "sudo",
            "-n",
            "systemctl",
            "--no-block",
            action,
            f"{service}.service",
        ]

    @pytest.mark.parametrize("action", ["start", "stop", "restart"])
    @pytest.mark.parametrize("service", ["sshd", "kiara;id", "--all", "../kiara", ""])
    def test_disallowed_service_never_executes(self, action, service):
        from kiara.admin.functions import services

        with patch("kiara.admin.functions.services.subprocess.run") as run:
            result = _run(getattr(services, f"services_{action}")(USER, service))
        assert result.success is False
        run.assert_not_called()

    @pytest.mark.parametrize("action", ["start", "stop", "restart"])
    @pytest.mark.parametrize(
        "error", [subprocess.TimeoutExpired("systemctl", 5), OSError("missing")]
    )
    def test_services_action_error(self, action, error):
        from kiara.admin.functions import services

        with patch("kiara.admin.functions.services.subprocess.run", side_effect=error):
            result = _run(getattr(services, f"services_{action}")(USER, "kiara"))
        assert result.success is False

    @pytest.mark.parametrize(
        "name", ["services_list", "services_start", "services_stop", "services_restart"]
    )
    def test_systemctl_does_not_block_event_loop(self, name):
        import threading

        from kiara.admin.functions import services

        caller_thread = threading.get_ident()

        def run_command(*args, **kwargs):
            assert threading.get_ident() != caller_thread
            return subprocess.CompletedProcess([], 0, "", "")

        with patch("kiara.admin.functions.services.subprocess.run", side_effect=run_command):
            fn = getattr(services, name)
            result = _run(fn(USER) if name == "services_list" else fn(USER, "kiara"))
        assert result.success is True


# ===========================================================================
# Sicherheits- und Edge-Case-Tests
# ===========================================================================


class TestSecurity:
    def test_output_truncation(self):
        """Output > 10000 Zeichen — Output-Begrenzung."""
        from kiara.admin.functions.network import network_ping

        mock_result = MagicMock()
        mock_result.stdout = (
            "x" * 20_000
            + "\n3 packets transmitted, 3 received, 0% packet loss\nrtt min/avg/max/mdev = 1/2/3/0 ms\n"
        )
        mock_result.returncode = 0

        with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
            result = _run(network_ping(USER, "localhost"))

        # Die Funktion sollte nicht abstuerzen bei langem Output
        assert isinstance(result, AdminResult)

    def test_shell_false(self):
        """Kein shell=True in network.py/services.py — Security-Audit."""
        import kiara.admin.functions.network as net_mod
        import kiara.admin.functions.services as svc_mod

        net_source = inspect.getsource(net_mod)
        svc_source = inspect.getsource(svc_mod)

        assert "shell=True" not in net_source, "shell=True gefunden in network.py!"
        assert "shell=True" not in svc_source, "shell=True gefunden in services.py!"

    def test_no_command_injection(self):
        """Host='fileserver.local; rm -rf /' — Injection wird durch Validierung blockiert."""
        from kiara.admin.functions.network import network_ping

        # Hostname-Validierung lehnt Sonderzeichen ab, bevor subprocess.run()
        # aufgerufen wird — der Injection-String erreicht nie die Shell.
        result = _run(network_ping(USER, "fileserver.local; rm -rf /"))
        assert result.success is False
        assert "Ungueltiger Hostname" in result.message

    def test_validate_service_normalization(self):
        """'kiara-worker' -> 'kiara-worker.service' — _validate_service() Normalisierung."""
        from kiara.admin.functions.services import _validate_service

        service, error = _validate_service("kiara-worker")
        assert service == "kiara-worker.service"
        assert error is None

        service, error = _validate_service("kiara-worker.service")
        assert service == "kiara-worker.service"
        assert error is None

        service, error = _validate_service("sshd")
        assert error is not None
        assert "nicht erlaubt" in error


# ===========================================================================
# Zusaetzliche Tests: ungetestete Code-Pfade
# ===========================================================================


class TestAdditionalNetworkPaths:
    def test_dns_hostname_dash_validation(self):
        """Hostname mit '-' am Anfang wird abgelehnt (dig hat kein -- Support)."""
        from kiara.admin.functions.network import network_dns

        result = _run(network_dns(USER, "-evil"))
        assert result.success is False
        assert "darf nicht mit Bindestrich beginnen" in result.message

    def test_traceroute_timeout_hops(self):
        """Traceroute mit * * * Timeout-Hops — haeufig bei Firewalls."""
        from kiara.admin.functions.network import network_traceroute

        mock_result = MagicMock()
        mock_result.stdout = (
            "traceroute to fileserver.local (fileserver.local), 20 hops max, 60 byte packets\n"
            " 1  gateway (10.0.0.1)  0.500 ms  0.400 ms  0.350 ms\n"
            " 2  * * *\n"
            " 3  fileserver (fileserver.local)  1.200 ms  1.100 ms  1.050 ms\n"
        )
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.network.shutil.which", return_value="/usr/bin/traceroute"
        ):
            with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
                result = _run(network_traceroute(USER, "fileserver.local"))

        assert result.success is True
        assert len(result.data["hops"]) == 3
        assert result.data["hops"][1]["ip"] == "*"
        assert result.data["hops"][1]["rtt_ms"] == "*"
        assert result.data["hops"][0]["ip"] == "10.0.0.1"
        assert result.data["hops"][2]["ip"] == "fileserver.local"

    def test_smb_connection_failure(self):
        """SMB-Verbindung fehlgeschlagen (rc!=0, kein stdout) — stderr-Extraktion."""
        from kiara.admin.functions.network import network_smb_discover

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "NT_STATUS_ACCESS_DENIED"
        mock_result.returncode = 1

        with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
            result = _run(network_smb_discover(USER, "fileserver.local"))

        assert result.success is False
        assert "NT_STATUS_ACCESS_DENIED" in result.message

    def test_http_connect_error(self):
        """httpx.ConnectError bei http_check — nicht routbare IP."""
        import httpx as _httpx

        from kiara.admin.functions.network import network_http_check

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=_httpx.ConnectError("Connection refused"),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_http_check(USER, "http://10.99.99.99:9999"))

        assert result.success is False
        assert "Verbindung fehlgeschlagen" in result.message

    def test_webdav_connect_error(self):
        """httpx.ConnectError bei webdav_check — Server nicht erreichbar."""
        import httpx as _httpx

        from kiara.admin.functions.network import network_webdav_check

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=_httpx.ConnectError("Connection refused"),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("kiara.admin.functions.network.httpx.AsyncClient", return_value=mock_client):
            result = _run(network_webdav_check(USER, "https://dav.invalid:443"))

        assert result.success is False
        assert "Verbindung fehlgeschlagen" in result.message


class TestAdditionalServicesPaths:
    def test_services_logs_with_since(self):
        """journalctl mit --since Parameter — Argument-Reihenfolge sudoers-kritisch."""
        from kiara.admin.functions.services import services_logs

        mock_result = MagicMock()
        mock_result.stdout = "2026-03-27T10:00:00+01:00 rag kiara[12345]: Log line\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.services.subprocess.run", return_value=mock_result
        ) as mock_run:
            result = _run(services_logs(USER, "kiara-worker", lines=20, since="1 hour ago"))

        assert result.success is True
        call_args = mock_run.call_args[0][0]
        # Reihenfolge: sudo journalctl --no-pager --output=short-iso -u SERVICE -n LINES --since SINCE
        assert call_args[:4] == ["sudo", "journalctl", "--no-pager", "--output=short-iso"]
        assert call_args[4:6] == ["-u", "kiara-worker.service"]
        assert call_args[6:8] == ["-n", "20"]
        assert call_args[8:10] == ["--since", "1 hour ago"]

    def test_services_logs_rejects_unsafe_since(self):
        """journalctl --since akzeptiert nur unterstuetzte Zeitformate."""
        from kiara.admin.functions.services import services_logs

        with patch("kiara.admin.functions.services.subprocess.run") as mock_run:
            result = _run(
                services_logs(
                    USER,
                    "kiara-worker",
                    lines=20,
                    since="1 hour ago --no-pager",
                )
            )

        assert result.success is False
        assert "Zeitfilter" in result.message
        mock_run.assert_not_called()

    def test_services_restart_failure(self):
        """systemctl restart gibt non-zero zurueck — Fehler aus stderr."""
        from kiara.admin.functions.services import services_restart

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Failed to restart kiara.service: Unit not found."
        mock_result.stdout = ""

        with patch("kiara.admin.functions.services.subprocess.run", return_value=mock_result):
            result = _run(services_restart(USER, "kiara"))

        assert result.success is False
        assert "fehlgeschlagen" in result.message
        assert "Unit not found" in result.message

    def test_services_status_inactive(self):
        """Gestoppter Service — kein PID, kein Memory, kein Uptime."""
        from kiara.admin.functions.services import services_status

        mock_result = MagicMock()
        mock_result.stdout = (
            "○ kiara-mcp.service - KIara MCP Server\n"
            "     Loaded: loaded (/etc/systemd/system/kiara-mcp.service; disabled)\n"
            "     Active: inactive (dead)\n"
        )
        mock_result.returncode = 3

        with patch("kiara.admin.functions.services.subprocess.run", return_value=mock_result):
            result = _run(services_status(USER, "kiara-mcp"))

        assert result.success is True
        assert result.data["status"] == "inactive"
        assert "pid" not in result.data
        assert "uptime" not in result.data
        assert "memory" not in result.data

    def test_dns_dig_empty_result(self):
        """dig gibt leeren Output zurueck — keine Records gefunden."""
        from kiara.admin.functions.network import network_dns

        mock_result = MagicMock()
        mock_result.stdout = "\n"
        mock_result.returncode = 0

        with patch("kiara.admin.functions.network.shutil.which", return_value="/usr/bin/dig"):
            with patch("kiara.admin.functions.network.subprocess.run", return_value=mock_result):
                result = _run(network_dns(USER, "nichtexistent.invalid"))

        assert result.success is False
        assert "Keine" in result.message

    def test_ping_custom_count(self):
        """count=5 wird als -c 5 uebergeben."""
        from kiara.admin.functions.network import network_ping

        mock_result = MagicMock()
        mock_result.stdout = (
            "5 packets transmitted, 5 received, 0% packet loss\n"
            "rtt min/avg/max/mdev = 0.1/0.2/0.3/0.1 ms\n"
        )
        mock_result.returncode = 0

        with patch(
            "kiara.admin.functions.network.subprocess.run", return_value=mock_result
        ) as mock_run:
            result = _run(network_ping(USER, "10.0.0.1", count=5))

        call_args = mock_run.call_args[0][0]
        assert "-c" in call_args
        assert call_args[call_args.index("-c") + 1] == "5"
        assert result.data["packets_sent"] == 5
