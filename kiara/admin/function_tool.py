"""BaseTool-Wrapper fuer AdminFunction — integriert Admin-Funktionen in den Agent-Tool-Loop."""

import json
import logging
import threading
import uuid

from kiara.admin.function import AdminFunction
from kiara.admin.result import AdminResult
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class AdminFunctionTool(BaseTool):
    """Wrapper: Macht eine AdminFunction als BaseTool fuer AgentChain nutzbar.

    Bei requires_confirmation=True wird die Funktion NICHT ausgefuehrt,
    sondern ein ToolResult mit needs_confirmation=True zurueckgegeben.
    Die eigentliche Ausfuehrung erfolgt erst nach Bestaetigung durch
    den ConfirmationManager.
    """

    def __init__(self, func: AdminFunction, chat_id: int | None = None) -> None:
        self._func = func
        self._chat_id = chat_id
        self._pending: list[dict] = []  # Seitenkanal fuer Confirmation/Wizard-Daten
        self._pending_lock = threading.Lock()

    def pop_pending(self) -> list[dict]:
        """Holt und leert die pending Actions dieser Tool-Instanz.

        Thread-sicher durch Lock: _execute_tool_batch() laeuft in
        ThreadPoolExecutor und kann parallel zu pop_pending() laufen.
        """
        with self._pending_lock:
            actions = self._pending.copy()
            self._pending = []
        return actions

    def validate_params(self, params: dict) -> tuple[bool, str]:
        """Validiert ohne Required-Pruefung — Wizard-Gate in execute() behandelt fehlende Pflichtfelder.

        BaseTool.validate_params() prueft required-Felder VOR execute().
        AdminFunctionTool.execute() hat einen eigenen Wizard-Gate, der
        fehlende Pflichtfelder behandelt und den Wizard-State mergt.
        Die Required-Pruefung hier wuerde den Wizard-Flow blockieren.
        """
        if not isinstance(params, dict):
            return False, "Parameter muessen ein Dictionary sein."

        defn = self.definition()
        properties: dict = defn.parameters.get("properties", {})

        for param_name, param_value in params.items():
            if param_name not in properties:
                return False, f"Unbekannter Parameter: '{param_name}'"
            prop_schema = properties[param_name]
            for check in (
                self._validate_type(param_name, param_value, prop_schema),
                self._validate_enum(param_name, param_value, prop_schema),
            ):
                if check:
                    return False, check
            for constraint_errors in (
                self._validate_string_constraints(param_name, param_value, prop_schema),
                self._validate_numeric_constraints(param_name, param_value, prop_schema),
            ):
                if constraint_errors:
                    return False, constraint_errors[0]

        return True, "ok"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._func.name,
            description=self._func.description,
            parameters=self._func.parameters,
            requires_confirmation=self._func.requires_confirmation,
            category=self._func.category,
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        func = self._func
        schema = func.parameters  # JSON-Schema mit "required" und "properties"

        # --- Wizard-State mergen: gespeicherte + aktuelle Parameter vereinen ---
        if self._chat_id:
            try:
                from kiara.repositories.admin_chat import AdminWizardRepository

                active_wiz = AdminWizardRepository().get_active(self._chat_id)
                if active_wiz and active_wiz["function_name"] == func.name:
                    stored = json.loads(active_wiz.get("collected_params") or "{}")
                    params = {**stored, **params}
            except Exception as exc:
                logger.warning(
                    "Wizard-State laden fehlgeschlagen fuer %s: %s", func.name, exc, exc_info=True
                )

        # --- Wizard-Gate: Fehlende Pflichtfelder pruefen ---
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = [r for r in required if r not in params]
        if missing:
            # Nicht alle Pflichtfelder angegeben → Wizard-Daten im Seitenkanal
            optional = [p for p in properties if p not in required and p not in params]
            with self._pending_lock:
                self._pending.append(
                    {
                        "type": "wizard",
                        "function_name": func.name,
                        "collected": params,
                        "missing_required": missing,
                        "optional": optional,
                        "parameter_help": {
                            k: {
                                "type": v.get("type", "string"),
                                "description": v.get("description", ""),
                            }
                            for k, v in properties.items()
                        },
                    }
                )
            missing_str = ", ".join(missing)
            return ToolResult(
                success=True,
                data=None,
                error=None,
                display_text=f"Fehlende Pflichtfelder: {missing_str}",
            )

        # --- Confirmation-Gate ---
        if func.requires_confirmation:
            action_id = str(uuid.uuid4())
            try:
                # Bool-Werte normalisieren: Python True/False → "true"/"false"
                fmt_params = {
                    k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()
                }
                confirm_string = (
                    func.confirm_template.format(**fmt_params)
                    if func.confirm_template
                    else func.name
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "confirm_template.format() fehlgeschlagen fuer %s: %s", func.name, exc
                )
                confirm_string = func.name
            # Guard: Leerer confirm_string — Fallback auf func.name
            if not confirm_string or not confirm_string.strip():
                confirm_string = func.name
            confirm_string = " ".join(confirm_string.split())
            with self._pending_lock:
                self._pending.append(
                    {
                        "type": "confirmation",
                        "action_id": action_id,
                        "confirm_string": confirm_string,
                        "function_name": func.name,
                        "params": params,
                        "message": f"Bitte bestaetigen: {confirm_string}",
                    }
                )
            # Kein wizard_complete hier — wird erst nach Confirmation-Ausfuehrung gesendet.
            # display_text muss EINDEUTIG signalisieren, dass die Aktion NICHT
            # ausgefuehrt wurde — sonst plant der Agent abhaengige Folgeaktionen.
            return ToolResult(
                success=True,
                data=None,
                error=None,
                display_text=(
                    f"AKTION WURDE NICHT AUSGEFUEHRT. {func.name} erfordert eine "
                    f"Bestaetigung durch den Admin. Fuehre keine weiteren "
                    f"abhaengigen Aktionen aus, bis die Bestaetigung vorliegt."
                ),
            )

        # --- Direkte Ausfuehrung ---
        try:
            user = {"id": user_id} if user_id else {"id": 0}
            result: AdminResult = await func.fn(user=user, **params)
            # Wizard-Completion signalisieren (falls ein aktiver Wizard existierte)
            with self._pending_lock:
                self._pending.append(
                    {
                        "type": "wizard_complete",
                        "function_name": func.name,
                    }
                )
            return ToolResult(
                success=result.success,
                data=result.data,
                error=result.error,
                display_text=result.message,
            )
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=f"Fehler bei {func.name}: {exc}",
            )
