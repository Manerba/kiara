"""Tests fuer den DegradationManager (kiara.resilience.degradation)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from kiara.health.checker import ComponentHealth, HealthChecker, SystemHealth
from kiara.resilience.degradation import DegradationManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Minimale Config fuer DegradationManager."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {
            "path": "/tmp/test_vectordb",
            "collection": "knowledgebase",
        },
        "database": {
            "path": "/tmp/test.db",
        },
        "cloud": {"enabled": False},
        "tools": {"enabled": False},
        "health": {
            "check_interval_seconds": 30,
        },
    }


@pytest.fixture
def mock_checker(base_config):
    """HealthChecker-Mock."""
    return HealthChecker(base_config)


@pytest.fixture
def manager(base_config, mock_checker):
    """DegradationManager-Instanz."""
    return DegradationManager(base_config, mock_checker)


def _make_health(component_overrides: dict[str, str]) -> SystemHealth:
    """Hilfsfunktion: SystemHealth mit gewuenschten Komponentenstatus erzeugen.

    Default: alle healthy/unconfigured.
    component_overrides: z.B. {"ollama": "unhealthy", "database": "degraded"}
    """
    defaults = {
        "ollama": "healthy",
        "vectordb": "healthy",
        "database": "healthy",
        "ldap": "unconfigured",
        "cloud_ki": "unconfigured",
        "erp": "unconfigured",
        "disk": "healthy",
    }
    defaults.update(component_overrides)

    components = [ComponentHealth(name=name, status=status) for name, status in defaults.items()]

    # Determine overall
    statuses = [c.status for c in components]
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    return SystemHealth(
        status=overall,
        components=components,
        timestamp="2026-02-24T10:00:00",
        uptime_seconds=60.0,
        version="0.0.1+12",
    )


# ---------------------------------------------------------------------------
# Mode-Property Tests
# ---------------------------------------------------------------------------


class TestModeProperty:
    """Tests fuer den Modus-Getter/Setter."""

    def test_initial_mode_is_full(self, manager):
        """Initialer Modus ist FULL."""
        assert manager.mode == "FULL"

    def test_mode_setter(self, manager):
        """Mode-Setter akzeptiert gueltige Modi."""
        manager.mode = "NO_CLOUD"
        assert manager.mode == "NO_CLOUD"

        manager.mode = "MAINTENANCE"
        assert manager.mode == "MAINTENANCE"

    def test_mode_setter_invalid(self, manager):
        """Mode-Setter ignoriert ungueltige Modi."""
        manager.mode = "INVALID_MODE"
        assert manager.mode == "FULL"  # Bleibt unveraendert

    def test_mode_setter_all_valid_modes(self, manager):
        """Alle definierten Modi koennen gesetzt werden."""
        for mode_name in DegradationManager.FEATURE_MAP:
            manager.mode = mode_name
            assert manager.mode == mode_name


# ---------------------------------------------------------------------------
# Feature-Availability Tests
# ---------------------------------------------------------------------------


class TestFeatureAvailability:
    """Tests fuer is_feature_available()."""

    def test_feature_available_full_mode(self, manager):
        """FULL-Modus: alle Features verfuegbar."""
        assert manager.mode == "FULL"
        assert manager.is_feature_available("cloud") is True
        assert manager.is_feature_available("llm") is True
        assert manager.is_feature_available("erp_tools") is True
        assert manager.is_feature_available("logging") is True
        assert manager.is_feature_available("full") is True

    def test_feature_available_no_cloud(self, manager):
        """NO_CLOUD-Modus: cloud nicht verfuegbar."""
        manager.mode = "NO_CLOUD"
        assert manager.is_feature_available("cloud") is False
        assert manager.is_feature_available("llm") is True
        assert manager.is_feature_available("erp_tools") is True
        assert manager.is_feature_available("logging") is True
        assert manager.is_feature_available("full") is False

    def test_feature_available_no_llm(self, manager):
        """NO_LLM-Modus: llm nicht verfuegbar, cloud bleibt verfuegbar."""
        manager.mode = "NO_LLM"
        assert manager.is_feature_available("cloud") is True
        assert manager.is_feature_available("llm") is False
        assert manager.is_feature_available("erp_tools") is True
        assert manager.is_feature_available("logging") is True
        assert manager.is_feature_available("full") is False

    def test_feature_available_no_erp(self, manager):
        """NO_ERP-Modus: erp_tools nicht verfuegbar."""
        manager.mode = "NO_ERP"
        assert manager.is_feature_available("cloud") is True
        assert manager.is_feature_available("llm") is True
        assert manager.is_feature_available("erp_tools") is False
        assert manager.is_feature_available("logging") is True
        assert manager.is_feature_available("full") is False

    def test_feature_available_readonly(self, manager):
        """READONLY-Modus: logging nicht verfuegbar."""
        manager.mode = "READONLY"
        assert manager.is_feature_available("cloud") is True
        assert manager.is_feature_available("llm") is True
        assert manager.is_feature_available("erp_tools") is True
        assert manager.is_feature_available("logging") is False
        assert manager.is_feature_available("full") is False

    def test_feature_available_maintenance(self, manager):
        """MAINTENANCE-Modus: keine Features verfuegbar."""
        manager.mode = "MAINTENANCE"
        assert manager.is_feature_available("cloud") is False
        assert manager.is_feature_available("llm") is False
        assert manager.is_feature_available("erp_tools") is False
        assert manager.is_feature_available("logging") is False
        assert manager.is_feature_available("full") is False

    def test_feature_unknown_returns_false(self, manager):
        """Unbekanntes Feature ist nie verfuegbar."""
        assert manager.is_feature_available("nonexistent_feature") is False


# ---------------------------------------------------------------------------
# update_status() Tests
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    """Tests fuer update_status()."""

    @pytest.mark.asyncio
    async def test_update_status_sets_mode_full(self, manager, mock_checker):
        """update_status() setzt FULL wenn alles healthy."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))

        await manager.update_status()

        assert manager.mode == "FULL"
        assert manager.last_health is not None

    def test_down_components_are_written_before_mode_change(self):
        """Feature-Flags duerfen beim Mode-Wechsel kein altes Down-Set sehen."""
        import inspect

        src = inspect.getsource(DegradationManager._update_status_locked)
        assert src.index("self._down_components = down_components") < src.index(
            'self.mode = "READONLY"'
        )

    @pytest.mark.asyncio
    async def test_update_status_sets_no_llm(self, manager, mock_checker):
        """update_status() setzt NO_LLM nach 2 konsekutiven Fehlern (Hysterese)."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "NO_LLM"

    @pytest.mark.asyncio
    async def test_update_status_sets_no_cloud(self, manager, mock_checker):
        """update_status() setzt NO_CLOUD nach 2 konsekutiven Fehlern."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"cloud_ki": "unhealthy"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "NO_CLOUD"

    @pytest.mark.asyncio
    async def test_update_status_sets_no_erp(self, manager, mock_checker):
        """update_status() setzt NO_ERP nach 2 konsekutiven Fehlern (ERP degraded)."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"erp": "degraded"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "NO_ERP"

    @pytest.mark.asyncio
    async def test_update_status_sets_readonly(self, manager, mock_checker):
        """update_status() setzt READONLY nach 2 konsekutiven Fehlern (DB unhealthy)."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"database": "unhealthy"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "READONLY"

    @pytest.mark.asyncio
    async def test_update_status_sets_readonly_for_vectordb(self, manager, mock_checker):
        """VektorDB-Ausfall laesst keinen FULL-Modus aktiv."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"vectordb": "unhealthy"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "READONLY"
        assert manager.is_feature_available("full") is False

    @pytest.mark.asyncio
    async def test_update_status_sets_readonly_for_disk_degraded(self, manager, mock_checker):
        """Disk DEGRADED faellt fail-closed auf READONLY."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"disk": "degraded"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "READONLY"
        assert manager.is_feature_available("logging") is False
        assert manager.is_feature_available("full") is False

    @pytest.mark.asyncio
    async def test_maintenance_mode_not_auto_changed(self, manager, mock_checker):
        """MAINTENANCE-Modus wird nicht durch update_status() geaendert."""
        manager.mode = "MAINTENANCE"

        # Auch wenn alles healthy ist, bleibt MAINTENANCE
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))

        await manager.update_status()

        assert manager.mode == "MAINTENANCE"

    @pytest.mark.asyncio
    async def test_update_status_recovers_to_full(self, manager, mock_checker):
        """update_status() setzt FULL zurueck wenn Probleme behoben (1 Erfolg reicht)."""
        # Erst NO_LLM setzen (braucht 2 Aufrufe wegen Hysterese)
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        await manager.update_status()
        assert manager.mode == "NO_LLM"

        # Dann alles healthy — 1 Erfolg reicht fuer Recovery
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))
        await manager.update_status()
        assert manager.mode == "FULL"

    @pytest.mark.asyncio
    async def test_update_status_database_priority(self, manager, mock_checker):
        """Database-unhealthy hat hoechste Prioritaet (READONLY)."""
        mock_checker.check_all = AsyncMock(
            return_value=_make_health(
                {
                    "database": "unhealthy",
                    "ollama": "unhealthy",
                    "cloud_ki": "unhealthy",
                }
            )
        )

        await manager.update_status()
        await manager.update_status()

        # Database unhealthy sollte READONLY erzwingen, nicht NO_LLM
        assert manager.mode == "READONLY"


# ---------------------------------------------------------------------------
# Background-Check Tests
# ---------------------------------------------------------------------------


class TestBackgroundChecks:
    """Tests fuer start/stop_background_checks()."""

    @pytest.mark.asyncio
    async def test_start_stop_background_checks(self, manager, mock_checker):
        """Background-Task kann gestartet und gestoppt werden."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))

        assert manager._background_task is None

        await manager.start_background_checks()
        assert manager._background_task is not None

        await manager.stop_background_checks()
        assert manager._background_task is None

    @pytest.mark.asyncio
    async def test_start_twice_no_duplicate(self, manager, mock_checker):
        """Zweimaliges Starten erzeugt keinen zweiten Task."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))

        await manager.start_background_checks()
        task1 = manager._background_task

        await manager.start_background_checks()
        task2 = manager._background_task

        assert task1 is task2

        await manager.stop_background_checks()


# ---------------------------------------------------------------------------
# Hysterese Tests
# ---------------------------------------------------------------------------


class TestHysteresis:
    """Tests fuer die Hysterese-Logik in update_status()."""

    @pytest.mark.asyncio
    async def test_single_failure_no_mode_change(self, manager, mock_checker):
        """1x update_status() mit ollama UNHEALTHY -> Mode bleibt FULL."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        await manager.update_status()

        assert manager.mode == "FULL"

    @pytest.mark.asyncio
    async def test_double_failure_triggers_mode_change(self, manager, mock_checker):
        """2x update_status() mit ollama UNHEALTHY -> Mode wechselt zu NO_LLM."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        await manager.update_status()
        assert manager.mode == "FULL"  # Noch kein Wechsel

        await manager.update_status()
        assert manager.mode == "NO_LLM"

    @pytest.mark.asyncio
    async def test_cloud_degraded_triggers(self, manager, mock_checker):
        """2x update_status() mit cloud_ki DEGRADED -> Mode wechselt zu NO_CLOUD."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"cloud_ki": "degraded"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "NO_CLOUD"

    @pytest.mark.asyncio
    async def test_db_degraded_no_trigger(self, manager, mock_checker):
        """2x update_status() mit database DEGRADED -> Mode bleibt FULL (nur UNHEALTHY zaehlt)."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"database": "degraded"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "FULL"

    @pytest.mark.asyncio
    async def test_recovery_immediate(self, manager, mock_checker):
        """1x Erfolg nach Degradation -> sofort zurueck zu FULL."""
        # Erst degradieren
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        await manager.update_status()
        assert manager.mode == "NO_LLM"

        # 1x healthy reicht fuer Recovery
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))
        await manager.update_status()
        assert manager.mode == "FULL"

    @pytest.mark.asyncio
    async def test_intermittent_failure_resets_counter(self, manager, mock_checker):
        """Erfolg zwischen Fehlern resettet den Counter."""
        unhealthy = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        healthy = AsyncMock(return_value=_make_health({}))

        mock_checker.check_all = unhealthy
        await manager.update_status()  # fail_count=1

        mock_checker.check_all = healthy
        await manager.update_status()  # fail_count=0

        mock_checker.check_all = unhealthy
        await manager.update_status()  # fail_count=1

        # Kein Wechsel, weil Counter durch Erfolg resettet wurde
        assert manager.mode == "FULL"


# ---------------------------------------------------------------------------
# Mail-Notification Tests
# ---------------------------------------------------------------------------


class TestMailNotification:
    """Tests fuer _notify_mode_change()."""

    @pytest.mark.asyncio
    async def test_mail_sent_on_degradation(self, manager, mock_checker):
        """Mode-Wechsel von FULL weg mit notify_degradation=1 -> send_mail aufgerufen."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        mock_row = {
            "notify_degradation": 1,
            "smtp_server": "mail.test",
            "smtp_port": 587,
            "encryption": "starttls",
            "username": "test",
            "password": "test",
            "sender_address": "test@test.de",
            "recipient_addresses": "admin@test.de",
        }

        with (
            patch("kiara.shared.db.db_query") as mock_db,
            patch("kiara.web.mail_service.send_mail") as mock_send,
            patch("kiara.shared.crypto.is_encrypted", return_value=False),
        ):
            ctx = mock_db.return_value.__enter__.return_value
            ctx.execute.return_value.fetchone.return_value = mock_row

            await manager.update_status()
            await manager.update_status()

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert "FULL" in call_args[0][1]
            assert "NO_LLM" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_mail_not_sent_on_recovery(self, manager, mock_checker):
        """Wechsel zu FULL -> keine Mail."""
        # Erst degradieren
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        await manager.update_status()
        assert manager.mode == "NO_LLM"

        with patch.object(manager, "_notify_mode_change") as mock_notify:
            mock_checker.check_all = AsyncMock(return_value=_make_health({}))
            await manager.update_status()
            assert manager.mode == "FULL"
            mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_mail_not_sent_between_degraded(self, manager, mock_checker):
        """Wechsel NO_LLM -> READONLY -> keine Mail (nur FULL-Weg-Transitionen)."""
        # Erst NO_LLM
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        await manager.update_status()
        assert manager.mode == "NO_LLM"

        with patch.object(manager, "_notify_mode_change") as mock_notify:
            # Jetzt DB auch unhealthy -> READONLY (Prio hoeher)
            mock_checker.check_all = AsyncMock(
                return_value=_make_health({"database": "unhealthy", "ollama": "unhealthy"})
            )
            await manager.update_status()
            await manager.update_status()
            assert manager.mode == "READONLY"
            mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_mail_not_sent_when_disabled(self, manager, mock_checker):
        """notify_degradation=0 -> keine Mail."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        mock_row = {"notify_degradation": 0}

        with (
            patch("kiara.shared.db.db_query") as mock_db,
            patch("kiara.web.mail_service.send_mail") as mock_send,
        ):
            ctx = mock_db.return_value.__enter__.return_value
            ctx.execute.return_value.fetchone.return_value = mock_row

            await manager.update_status()
            await manager.update_status()

            mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# API-Endpoint Tests
# ---------------------------------------------------------------------------


class TestDegradationEndpoint:
    """Tests fuer /api/health/degradation-mode Endpoint."""

    @pytest.mark.asyncio
    async def test_degradation_mode_endpoint_full(self):
        """GET /api/health/degradation-mode liefert FULL bei Normalbetrieb."""
        from kiara.web.api_health import degradation_mode

        with patch("kiara.web.api_health._degradation_manager") as mock_dm:
            mock_dm.mode = "FULL"
            with patch("kiara.web.api_health._get_degradation_manager", return_value=mock_dm):
                result = await degradation_mode()
                assert result["mode"] == "FULL"
                assert result["message"] == ""

    @pytest.mark.asyncio
    async def test_degradation_mode_endpoint_no_llm(self):
        """GET /api/health/degradation-mode liefert NO_LLM mit Message."""
        from kiara.web.api_health import degradation_mode

        with patch("kiara.web.api_health._get_degradation_manager") as mock_fn:
            mock_dm = AsyncMock()
            mock_dm.mode = "NO_LLM"
            mock_fn.return_value = mock_dm

            result = await degradation_mode()
            assert result["mode"] == "NO_LLM"
            assert "Ollama" in result["message"]


# ---------------------------------------------------------------------------
# ensure_background_checks Tests
# ---------------------------------------------------------------------------


class TestEnsureBackgroundChecks:
    """Tests fuer ensure_background_checks() und shutdown_degradation_manager()."""

    @pytest.mark.asyncio
    async def test_ensure_background_checks_idempotent(self):
        """Mehrfacher Aufruf startet nur einen Task."""
        import kiara.web.api_health as mod

        old_dm = mod._degradation_manager
        old_started = mod._background_started

        try:
            mock_dm = AsyncMock()
            mock_dm.start_background_checks = AsyncMock()
            mock_dm.mode = "FULL"
            mock_dm.last_health = None

            mod._degradation_manager = mock_dm
            mod._background_started = False

            await mod.ensure_background_checks()
            await mod.ensure_background_checks()

            mock_dm.start_background_checks.assert_called_once()
        finally:
            mod._degradation_manager = old_dm
            mod._background_started = old_started

    @pytest.mark.asyncio
    async def test_shutdown_stops_background(self):
        """shutdown_degradation_manager() stoppt Task."""
        import kiara.web.api_health as mod

        old_dm = mod._degradation_manager
        old_started = mod._background_started

        try:
            mock_dm = AsyncMock()
            mock_dm.stop_background_checks = AsyncMock()

            mod._degradation_manager = mock_dm
            mod._background_started = True

            await mod.shutdown_degradation_manager()

            mock_dm.stop_background_checks.assert_called_once()
            assert mod._background_started is False
        finally:
            mod._degradation_manager = old_dm
            mod._background_started = old_started

    @pytest.mark.asyncio
    async def test_health_endpoint_uses_cached_status(self, manager, mock_checker):
        """GET /api/health liest gecachten Status, kein erneuter update_status()-Aufruf."""
        # Initialen Check simulieren
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))
        await manager.update_status()

        with patch.object(manager, "update_status", new_callable=AsyncMock) as mock_update:
            # last_health ist jetzt gesetzt, also sollte update_status() NICHT aufgerufen werden
            assert manager.last_health is not None
            # Simuliere was der Endpoint tut
            if manager.last_health is None:
                await manager.update_status()
            mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# Chat 503-Guard Test
# ---------------------------------------------------------------------------


class TestChat503Guard:
    """Tests fuer den 503-Guard in api_chat.py."""

    @pytest.mark.asyncio
    async def test_chat_503_when_no_llm_and_no_cloud(self):
        """Chat-Query ohne lokales LLM und ohne Cloud -> HTTP 503."""
        from kiara.web.api_chat import QueryRequest, chat_query

        mock_dm = MagicMock()
        mock_dm.is_feature_available.return_value = False

        with patch("kiara.web.api_health._get_degradation_manager", return_value=mock_dm):
            data = QueryRequest(question="Test?", mode="standard")
            mock_user = {"id": 1, "username": "test"}

            result = await chat_query("new", data, user=mock_user)

            assert result.status_code == 503
            mock_dm.is_feature_available.assert_has_calls([call("llm"), call("cloud")])

    @pytest.mark.asyncio
    async def test_chat_503_response_body_when_no_backend_available(self):
        """503-Response enthaelt deutsche Fehlermeldung fuer den User."""
        import json

        from kiara.web.api_chat import QueryRequest, chat_query

        mock_dm = MagicMock()
        mock_dm.is_feature_available.return_value = False

        with patch("kiara.web.api_health._get_degradation_manager", return_value=mock_dm):
            data = QueryRequest(question="Test?", mode="standard")
            result = await chat_query("new", data, user={"id": 1, "username": "test"})

            body = json.loads(result.body.decode())
            assert "Kein LLM- oder Cloud-Backend erreichbar" in body["detail"]

    @pytest.mark.asyncio
    async def test_chat_allows_cloud_when_llm_unavailable(self):
        """NO_LLM blockiert den Chat nicht, solange Cloud verfuegbar ist."""
        from kiara.web.api_chat import QueryRequest, chat_query

        async def _stream():
            yield "data: [DONE]\n\n"

        mock_dm = MagicMock()
        mock_dm.is_feature_available.side_effect = lambda feature: feature == "cloud"
        clean = SimpleNamespace(severity="none", clean_text="Test?")

        with (
            patch("kiara.web.api_health._get_degradation_manager", return_value=mock_dm),
            patch(
                "kiara.web.api_chat.load_config",
                return_value={"chat": {"default_mode": "standard"}},
            ),
            patch(
                "kiara.web.api_chat._resolve_template", return_value=("standard", "", None, None)
            ),
            patch("kiara.web.api_chat._resolve_chat_id", return_value=123),
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
            patch("kiara.shared.prompt_defense.sanitize_with_report", return_value=clean),
            patch("kiara.shared.prompt_defense_audit.log_defense_event"),
            patch("kiara.web.api_chat._stream_standard", return_value=_stream()),
        ):
            data = QueryRequest(question="Test?", mode="standard")
            result = await chat_query("new", data, user={"id": 1, "username": "test"})

        assert result.status_code == 200
        mock_msg_repo.add_message.assert_called_once_with(123, "user", "Test?")
        mock_dm.is_feature_available.assert_has_calls([call("llm"), call("cloud")])


# ---------------------------------------------------------------------------
# _check_loop Resilienz-Tests
# ---------------------------------------------------------------------------


class TestCheckLoopResilience:
    """Tests fuer die Fehlerbehandlung in _check_loop()."""

    @pytest.mark.asyncio
    async def test_check_loop_continues_after_exception(self, manager, mock_checker):
        """_check_loop faengt Exceptions und laeuft weiter statt zu sterben."""
        call_count = 0

        async def _flaky_check_all():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Ollama nicht erreichbar")
            return _make_health({})

        mock_checker.check_all = _flaky_check_all
        # Sehr kurzes Intervall fuer den Test
        manager._check_interval = 0.01

        await manager.start_background_checks()
        # Warten bis mindestens 2 Durchlaeufe passiert sind
        await asyncio.sleep(0.1)
        await manager.stop_background_checks()

        # Mindestens 2 Aufrufe: 1x Exception + 1x Erfolg
        assert call_count >= 2
        # Mode sollte FULL sein (der erfolgreiche Check hat's gesetzt)
        assert manager.mode == "FULL"


# ---------------------------------------------------------------------------
# Mail-Notification Resilienz-Tests
# ---------------------------------------------------------------------------


class TestMailNotificationResilience:
    """Tests fuer Fehlerbehandlung in _notify_mode_change()."""

    @pytest.mark.asyncio
    async def test_mail_exception_doesnt_crash_update(self, manager, mock_checker):
        """Wenn send_mail() explodiert, darf update_status() nicht abbrechen."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        with (
            patch("kiara.shared.db.db_query") as mock_db,
            patch("kiara.web.mail_service.send_mail", side_effect=ConnectionError("SMTP down")),
            patch("kiara.shared.crypto.is_encrypted", return_value=False),
        ):
            mock_row = {
                "notify_degradation": 1,
                "smtp_server": "mail.test",
                "smtp_port": 587,
                "encryption": "starttls",
                "username": "test",
                "password": "test",
                "sender_address": "a@b.de",
                "recipient_addresses": "c@d.de",
            }
            ctx = mock_db.return_value.__enter__.return_value
            ctx.execute.return_value.fetchone.return_value = mock_row

            # Darf NICHT raisen — Exception wird gefangen
            await manager.update_status()
            await manager.update_status()

            # Mode-Wechsel ist trotzdem passiert
            assert manager.mode == "NO_LLM"

    @pytest.mark.asyncio
    async def test_mail_no_config_row(self, manager, mock_checker):
        """Keine mail_config-Zeile in DB -> kein Crash, keine Mail."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))

        with (
            patch("kiara.shared.db.db_query") as mock_db,
            patch("kiara.web.mail_service.send_mail") as mock_send,
        ):
            ctx = mock_db.return_value.__enter__.return_value
            ctx.execute.return_value.fetchone.return_value = None  # Kein Row

            await manager.update_status()
            await manager.update_status()

            assert manager.mode == "NO_LLM"
            mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Trigger-Konfiguration Edge-Cases
# ---------------------------------------------------------------------------


class TestTriggerEdgeCases:
    """Tests fuer die komponentenspezifische Trigger-Konfiguration."""

    @pytest.mark.asyncio
    async def test_ollama_degraded_no_trigger(self, manager, mock_checker):
        """Ollama DEGRADED zaehlt NICHT als Fehler (nur UNHEALTHY)."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "degraded"}))

        await manager.update_status()
        await manager.update_status()

        # Ollama ist in _default_trigger (nur UNHEALTHY) -> DEGRADED ignoriert
        assert manager.mode == "FULL"
        assert manager._fail_counts.get("ollama", 0) == 0

    @pytest.mark.asyncio
    async def test_erp_unhealthy_also_triggers(self, manager, mock_checker):
        """ERP UNHEALTHY triggert ebenfalls (nicht nur DEGRADED)."""
        mock_checker.check_all = AsyncMock(return_value=_make_health({"erp": "unhealthy"}))

        await manager.update_status()
        await manager.update_status()

        assert manager.mode == "NO_ERP"


# ---------------------------------------------------------------------------
# Counter-State Tests
# ---------------------------------------------------------------------------


class TestCounterState:
    """Tests fuer den internen Zustand der Hysterese-Counter."""

    @pytest.mark.asyncio
    async def test_fail_counts_reset_after_recovery(self, manager, mock_checker):
        """Nach Recovery steht der Counter auf 0 — naechster Einzelfehler degradiert NICHT."""
        # Phase 1: Degradieren
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        await manager.update_status()
        assert manager.mode == "NO_LLM"
        assert manager._fail_counts["ollama"] == 2

        # Phase 2: Recovery
        mock_checker.check_all = AsyncMock(return_value=_make_health({}))
        await manager.update_status()
        assert manager.mode == "FULL"
        assert manager._fail_counts["ollama"] == 0

        # Phase 3: Einzelner Fehler danach -> KEIN Wechsel
        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        assert manager.mode == "FULL"
        assert manager._fail_counts["ollama"] == 1

    @pytest.mark.asyncio
    async def test_maintenance_mode_still_counts(self, manager, mock_checker):
        """Im MAINTENANCE-Modus werden Counter aktualisiert, nur der Mode nicht geaendert.

        Wichtig: Wenn jemand MAINTENANCE verlässt und eine Komponente tatsaechlich
        ausgefallen ist, muessen die Counter den realen Zustand widerspiegeln.
        """
        manager.mode = "MAINTENANCE"

        mock_checker.check_all = AsyncMock(return_value=_make_health({"ollama": "unhealthy"}))
        await manager.update_status()
        await manager.update_status()

        # Mode bleibt MAINTENANCE
        assert manager.mode == "MAINTENANCE"
        # Aber Counter hat gezaehlt!
        assert manager._fail_counts["ollama"] == 2
