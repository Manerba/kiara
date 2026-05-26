"""Orchestrierung des Bestaetigungs-Flows — verbindet DB, Funktionsausfuehrung, Changelog."""

import json
import logging

from kiara.admin.registry import admin_registry
from kiara.repositories.admin_chat import AdminConfirmationRepository

logger = logging.getLogger(__name__)


class ConfirmationManager:
    """Server-seitige Bestaetigungslogik fuer destruktive Admin-Aktionen."""

    def __init__(self) -> None:
        self._repo = AdminConfirmationRepository()

    def get_pending(self, chat_id: int) -> dict | None:
        """Holt die aelteste pending Confirmation (mit Ablauf-Pruefung)."""
        all_pending = self.get_all_pending(chat_id)
        return all_pending[0] if all_pending else None

    def get_all_pending(self, chat_id: int) -> list[dict]:
        """Holt alle pending Confirmations; Ablauf ist DB-persistent."""
        return self._repo.list_pending(chat_id)

    def check_and_handle(self, chat_id: int, user_input: str, user: dict):
        """Sync-Wrapper fuer check_and_handle_async.

        Wird aus dem sync-Generator _stream_admin_query() in api_admin_chat.py
        aufgerufen. Leitet intern ueber asyncio.run() auf den async-Pfad weiter.

        Returns:
            tuple (handled: bool, events: list[tuple[str, dict]])
            - handled=True: Confirmation wurde verarbeitet, Agent NICHT aufrufen
            - handled=False: Keine pending Confirmation, normal weiter
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.check_and_handle_async(chat_id, user_input, user))
        raise RuntimeError("check_and_handle_async() im async Kontext verwenden")

    async def check_and_handle_async(self, chat_id: int, user_input: str, user: dict):
        """Prueft ob user_input eine pending Confirmation bestaetigt oder verwirft.

        Matcht gegen ALLE pending Confirmations (nicht nur die aelteste),
        damit bei mehreren Confirmations pro Chat die richtige bestaetigt wird.
        """
        all_pending = self.get_all_pending(chat_id)
        if not all_pending:
            return False, []

        from kiara.admin.router import _normalize_confirm_string

        user_clean = _normalize_confirm_string(user_input)

        for pending in all_pending:
            confirm_string = _normalize_confirm_string(pending["confirm_string"])
            if user_clean == confirm_string:
                return await self._execute_confirmed_async(pending, user)

        # Kein Match — aelteste Confirmation verwerfen (FIFO)
        return self._dismiss(all_pending[0])

    async def _execute_confirmed_async(self, pending: dict, user: dict):
        """Bestaetigung erhalten — Funktion async ausfuehren."""
        if not self._repo.confirm(pending["action_id"]):
            return True, [
                (
                    "confirmation_result",
                    {
                        "action_id": pending["action_id"],
                        "confirmed": False,
                        "result": {
                            "success": False,
                            "message": "Aktion wurde bereits verarbeitet oder ist abgelaufen",
                        },
                    },
                )
            ]
        func = admin_registry.get(pending["function_name"])
        if not func:
            return True, [
                (
                    "confirmation_result",
                    {
                        "action_id": pending["action_id"],
                        "confirmed": True,
                        "result": {
                            "success": False,
                            "message": f"Funktion '{pending['function_name']}' nicht mehr verfuegbar",
                        },
                    },
                )
            ]

        try:
            params = (
                json.loads(pending["params"])
                if isinstance(pending["params"], str)
                else pending["params"]
            )
            result = await func.fn(user=user, **params)
            if result.success:
                self._log_changelog(func, params, user, result)
            self._complete_wizard(pending, func)
            return True, [
                (
                    "confirmation_result",
                    {
                        "action_id": pending["action_id"],
                        "confirmed": True,
                        "result": {"success": result.success, "message": result.message},
                    },
                )
            ]
        except Exception as exc:
            logger.exception("Confirmed action failed: %s", pending["action_id"])
            return True, [
                (
                    "confirmation_result",
                    {
                        "action_id": pending["action_id"],
                        "confirmed": True,
                        "result": {"success": False, "message": str(exc)},
                    },
                )
            ]

    def _dismiss(self, pending: dict):
        """Bestaetigungsstring nicht erkannt — Aktion verwerfen."""
        self._repo.dismiss(pending["action_id"])
        return False, [
            (
                "action_dismissed",
                {
                    "action_id": pending["action_id"],
                    "reason": "Bestaetigung nicht erhalten",
                },
            )
        ]

    def _complete_wizard(self, pending: dict, func):
        """Aktiven Wizard nach Confirmation-Ausfuehrung abschliessen.

        Im direkten Ausfuehrungspfad (AdminFunctionTool.execute) wird
        wizard_complete via _pending-Seitenkanal signalisiert. Im
        Confirmation-Pfad fehlt das — daher hier nachholen.

        Atomares update_where statt get_active+complete, um TOCTOU zu vermeiden.
        """
        try:
            from kiara_db.base import RawSQL

            from kiara.repositories.admin_chat import AdminWizardRepository

            wizard_repo = AdminWizardRepository()
            wizard_repo.update_where(
                {"chat_id": pending["chat_id"], "status": "collecting", "function_name": func.name},
                status="executed",
                updated_at=RawSQL("NOW()"),
            )
        except Exception as exc:
            logger.warning("Wizard-Completion nach Confirmation fehlgeschlagen: %s", exc)

    def _log_changelog(self, func, params, user, result):
        """Changelog-Eintrag fuer bestaetigte Aktion (nur bei Erfolg)."""
        if not result or not result.success:
            return
        if not func.changelog_area or not func.changelog_action:
            return
        from kiara.web.config_changelog import log_config_change

        target_id = str(params.get(func.pk_param, "")) if func.pk_param else ""
        target_label = (
            result.data.get("name", "") if result and isinstance(result.data, dict) else ""
        )
        try:
            log_config_change(
                user=user,
                area=func.changelog_area,
                action=func.changelog_action,
                target_id=target_id,
                target_label=target_label,
                source="chat",
            )
        except Exception as exc:
            logger.warning("Changelog-Eintrag fehlgeschlagen: %s", exc)
