import json
import logging
from typing import Generator

from kiara.generation.adapters import create_tool_adapter
from kiara.generation.agent_chain import AgentChain, AgentEvent
from kiara.shared.secret_redact import redact_secrets_recursive
from kiara.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_ADMIN_SYSTEM_PROMPT_FALLBACK = """Du bist KIara, der intelligente KI-Administrations-Assistent.
Du bist das zentrale Interface fuer die Verwaltung des KIara-Systems.
Administratoren interagieren mit dir ueber diesen Chat, um das System zu steuern.

## Deine Rolle
Du bist KEIN reiner Suchassistent. Du bist ein System-Administrator-Assistent
mit direktem Zugriff auf alle Verwaltungsfunktionen. Nutze bevorzugt die
Admin-Tools (dashboard.status, datasource.*, jobs.*, users.*, ki.*, system.*,
network.*, services.*, backup.*, compliance.*, tools.*, etc.).

## Wann welches Tool nutzen
- Fragen zum System (Status, Config, Benutzer, Jobs): Admin-Tools nutzen
- Unbekannte Funktion: help.search nutzen um verfuegbare Tools zu finden
- Netzwerk-Probleme: Erst network.ping, dann network.portscan, dann Dienst pruefen

## Gefuehrte Workflows
- Nutze workflow.list fuer verfuegbare Guided Workflows
- Nutze workflow.start, um den Schrittplan zu laden
- workflow.start fuehrt selbst keine Aktion aus; rufe anschliessend die in den Schritten referenzierten Admin-Funktionen direkt auf
- Wenn ein Schritt `workflow_ref` enthaelt, lade zuerst den referenzierten Teilworkflow
- Nutze `step.fixed_params` und `step.param_mapping` fuer den konkreten Funktionsaufruf

## Dokumentation
- Nutze wiki.search fuer Fragen zur Admin-Dokumentation und zu Setup-Anleitungen
- Nutze help.search fuer unbekannte Admin-Funktionen

## Verhaltensregeln
- Antworte immer auf Deutsch
- Sei praezise und hilfreich
- Bei destruktiven Aktionen wirst du automatisch zur Bestaetigung aufgefordert
- Wenn du unsicher bist, nutze help.search um verfuegbare Funktionen zu finden
- Zeige Ergebnisse uebersichtlich formatiert (Tabellen, Listen)
"""

_ADMIN_TEMPLATE_NAME = "system_admin_agent"


def _load_admin_system_prompt() -> str:
    """Laedt den Admin-Chat System-Prompt aus der DB (prompt_templates).

    Editierbar unter KI > Chat im Admin-UI. Fallback auf den hardcodierten
    Prompt wenn das Template nicht existiert (z.B. vor der Migration).
    """
    try:
        from kiara.repositories.prompt import PromptTemplateRepository

        repo = PromptTemplateRepository()
        template = repo.get_by_name(_ADMIN_TEMPLATE_NAME)
        if template and template.get("content"):
            return template["content"]
    except Exception as exc:
        logger.warning("Admin-System-Prompt konnte nicht aus DB geladen werden: %s", exc)
    return _ADMIN_SYSTEM_PROMPT_FALLBACK


class AdminAgentChain:
    """Wrapper um AgentChain fuer den Admin-Chat-Kontext.

    Erstellt intern einen AgentChain mit:
    - Admin-spezifischem System-Prompt
    - Wiki-/Hilfstools (wiki.search, ListDatasourcesTool)
    - Admin-Funktions-Tools (via AdminFunctionRegistry, inkl. help.search)
    - Seitenkanal fuer Confirmation/Wizard-Events
    """

    def __init__(
        self,
        config: dict,
        backend: dict,
        user_id: int | None = None,
        anonymizer=None,
        chat_id: int = 0,
    ) -> None:
        self._user_id = user_id
        self._chat_id = chat_id

        tool_registry = ToolRegistry()

        # Wiki-/Hilfstools registrieren
        from kiara.admin.wiki_tool import WikiSearchTool
        from kiara.tools.builtin_tools import ListDatasourcesTool

        tool_registry.register(WikiSearchTool())
        tool_registry.register(ListDatasourcesTool())

        # Admin-Funktionen aus Registry als AdminFunctionTool registrieren
        self._admin_tools = []
        from kiara.admin.function_tool import AdminFunctionTool
        from kiara.admin.registry import admin_registry

        for func in admin_registry.list_all():
            tool = AdminFunctionTool(func, chat_id=chat_id)
            tool_registry.register(tool)
            self._admin_tools.append(tool)

        adapter = create_tool_adapter(backend, config, anonymizer)
        self._chain = AgentChain(
            config=config,
            tool_registry=tool_registry,
            adapter=adapter,
            user_id=user_id,
            system_prompt_override=_load_admin_system_prompt(),
            force_initial_search=False,
        )

    @property
    def last_usage(self) -> dict | None:
        """Token-Usage des letzten Aufrufs (delegiert an AgentChain)."""
        return self._chain.last_usage

    def query_stream(
        self,
        question: str,
        history_block: str = "",
        think: bool = False,
    ) -> Generator[AgentEvent, None, None]:
        """Delegiert an AgentChain.query_stream() mit Seitenkanal-Drain."""
        for event in self._chain.query_stream(question, history_block, think):
            yield event
            # Nach jedem tool_result: Seitenkanal aller AdminFunctionTools pruefen
            if event.event_type == "tool_result":
                yield from self._drain_pending_actions()

    def _drain_pending_actions(self):
        """Fragt alle AdminFunctionTool-Instanzen nach pending Actions ab."""
        from kiara.admin.wizard import WizardManager
        from kiara.repositories.admin_chat import AdminConfirmationRepository

        for tool in self._admin_tools:
            # Alle Actions auf einmal holen (pop_pending leert die Liste atomar).
            # Ueber lokale Kopie iterieren — kein erneuter pop_pending() noetig.
            pending = tool.pop_pending()
            for action in pending:
                try:
                    if action["type"] == "confirmation":
                        # Secrets rekursiv redacten vor Persist + Streaming
                        # (Top-Level UND verschachtelte Dicts/Lists wie connector_config, extra_config)
                        safe_params = redact_secrets_recursive(action.get("params", {}))
                        # In DB speichern (redacted)
                        repo = AdminConfirmationRepository()
                        repo.create(
                            chat_id=self._chat_id,
                            action_id=action["action_id"],
                            function_name=action["function_name"],
                            params=json.dumps(safe_params),
                            confirm_string=" ".join(str(action["confirm_string"]).split()),
                            user_id=self._user_id,
                        )
                        # Streame redacted version
                        safe_action = {**action, "params": safe_params}
                        yield AgentEvent("confirmation_required", safe_action)

                    elif action["type"] == "wizard":
                        wizard_mgr = WizardManager()
                        wizard_data = wizard_mgr.handle_wizard_result(
                            self._chat_id,
                            self._user_id,
                            action,
                        )
                        yield AgentEvent("wizard_update", wizard_data)

                    elif action["type"] == "wizard_complete":
                        # Aktiven Wizard als 'executed' markieren (Datenhygiene)
                        from kiara.repositories.admin_chat import AdminWizardRepository

                        wizard_repo = AdminWizardRepository()
                        active = wizard_repo.get_active(self._chat_id)
                        if active and active["function_name"] == action["function_name"]:
                            wizard_repo.complete(active["id"])
                except Exception as exc:
                    logger.exception("Seitenkanal-Aktion fehlgeschlagen: %s", action.get("type"))
                    yield AgentEvent("error", {"message": f"Interner Fehler: {exc}"})
