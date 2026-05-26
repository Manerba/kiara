"""Orchestrierung des Wizard-Flows — Parameter-Sammlung ueber mehrere Nachrichten."""

import logging

from kiara_db.repositories.admin_chat import mask_wizard_secret_params
from psycopg2.errors import UniqueViolation

from kiara.repositories.admin_chat import AdminWizardRepository

logger = logging.getLogger(__name__)


class WizardManager:
    """Verwaltet mehrstufige Parameter-Erfassung fuer Admin-Funktionen.

    Secret-Felder in ``collected_params`` werden vor der DB-Persistierung
    ueber das Repository per Fernet verschluesselt (#558) und beim Lesen
    transparent wieder entschluesselt.
    """

    def __init__(self) -> None:
        self._repo = AdminWizardRepository()

    def handle_wizard_result(self, chat_id: int, user_id: int, tool_result_data: dict) -> dict:
        """Erstellt oder aktualisiert Wizard-State basierend auf ToolResult.

        Args:
            chat_id: Aktuelle Chat-ID
            user_id: Aktueller User
            tool_result_data: Das data-Dict aus dem ToolResult (mit needs_wizard)

        Returns:
            Dict mit wizard_id und aktuellem State fuer SSE-Event
        """
        function_name = tool_result_data["function_name"]
        collected = tool_result_data.get("collected", {})

        # Bestehenden Wizard suchen oder neuen erstellen
        active = self._repo.get_active(chat_id)
        if active and active["function_name"] == function_name:
            # Bestehenden Wizard aktualisieren
            self._repo.update_params(active["id"], collected)
            wizard_id = active["id"]
        else:
            # Alten Wizard ggf. abbrechen
            if active:
                self._repo.cancel(active["id"])
            # Neuen Wizard erstellen (Repository verschluesselt Secrets)
            try:
                wizard_id = self._repo.create_wizard(
                    chat_id=chat_id,
                    user_id=user_id,
                    function_name=function_name,
                    collected_params=collected,
                )
            except UniqueViolation:
                # UNIQUE INDEX Kollision bei parallelem Request — bestehenden nutzen
                active = self._repo.get_active(chat_id)
                if active and active["function_name"] == function_name:
                    self._repo.update_params(active["id"], collected)
                    wizard_id = active["id"]
                else:
                    raise

        return {
            "wizard_id": wizard_id,
            "function_name": function_name,
            "collected": mask_wizard_secret_params(collected),
            "missing_required": tool_result_data.get("missing_required", []),
            "optional": tool_result_data.get("optional", []),
            "parameter_help": tool_result_data.get("parameter_help", {}),
        }

    def complete(self, wizard_id: int) -> None:
        """Markiert Wizard als ausgefuehrt."""
        self._repo.complete(wizard_id)

    def cancel_active(self, chat_id: int) -> None:
        """Bricht den aktiven Wizard eines Chats ab."""
        active = self._repo.get_active(chat_id)
        if active:
            self._repo.cancel(active["id"])
