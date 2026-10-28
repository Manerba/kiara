"""Basisklassen fuer das KIara Tool-System.

Definiert ToolResult, ToolDefinition und die abstrakte BaseTool-Klasse,
von der alle konkreten Tools erben.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Mapping von JSON-Schema-Typ auf Python-Typen
_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass
class ToolResult:
    """Ergebnis einer Tool-Ausfuehrung.

    Attributes:
        success: True wenn die Ausfuehrung erfolgreich war.
        data: Die zurueckgegebenen Daten (dict, list, str oder None).
        error: Fehlermeldung bei Misserfolg, sonst None.
        display_text: Menschenlesbarer Text fuer die Anzeige im Chat.
        retryable: True nur fuer transiente Fehler, bei denen ein erneuter
            Tool-Aufruf sinnvoll ist. Fachliche Fehler bleiben False.
    """

    success: bool
    data: dict | list | str | None
    error: str | None
    display_text: str
    retryable: bool = False
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Beschreibung eines Tools fuer Registry und LLM-Prompt.

    Attributes:
        name: Eindeutiger technischer Name des Tools.
        description: Menschenlesbare Beschreibung (wird im LLM-Prompt verwendet).
        parameters: JSON-Schema-Dict, das die erwarteten Parameter beschreibt.
        requires_confirmation: Ob der User vor der Ausfuehrung bestaetigen muss.
        required_permissions: Liste von Gruppen/Rollen, die das Tool nutzen duerfen.
                             Leere Liste = fuer alle verfuegbar.
        category: Kategorie des Tools (z.B. "builtin", "erp"). Leer = Standard.
    """

    name: str
    description: str
    parameters: dict
    requires_confirmation: bool = False
    required_permissions: list[str] = field(default_factory=list)
    category: str = ""


class BaseTool(ABC):
    """Abstrakte Basisklasse fuer alle KIara-Tools.

    Jedes Tool muss ``definition()`` und ``execute()`` implementieren.
    Die ``validate_params()``-Methode steht als Standardimplementierung zur
    Verfuegung und prueft Parameter gegen das JSON-Schema der Definition.
    """

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition (Name, Beschreibung, Parameter-Schema).

        Returns:
            ToolDefinition-Instanz, die das Tool beschreibt.
        """
        ...

    @abstractmethod
    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Fuehrt das Tool mit den gegebenen Parametern aus.

        Args:
            params: Dictionary mit den Tool-Parametern.
            user_id: Optionale User-ID fuer Audit/Berechtigungspruefungen.

        Returns:
            ToolResult mit Erfolg/Misserfolg, Daten und Anzeigetext.
        """
        ...

    def _validate_or_error(self, params: dict) -> ToolResult | None:
        """Validiert Parameter und liefert ein Fehler-ToolResult oder None.

        Convenience-Wrapper um ``validate_params()``. Gibt ``None`` zurueck
        wenn die Validierung erfolgreich ist, sonst ein fertiges
        ``ToolResult(success=False, ...)``.

        Args:
            params: Die zu validierenden Parameter.

        Returns:
            None bei Erfolg, ToolResult mit Fehlerdetails bei Misserfolg.
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return ToolResult(success=False, data=None, error=msg, display_text=msg)
        return None

    @staticmethod
    def _validate_required(params: dict, schema: dict) -> list[str]:
        """Prueft ob alle Pflichtparameter vorhanden sind.

        Args:
            params: Die uebergebenen Parameter.
            schema: Das JSON-Schema mit 'required'-Feld.

        Returns:
            Liste von Fehlermeldungen (leer wenn alle Pflichtfelder vorhanden).
        """
        errors: list[str] = []
        for field_name in schema.get("required", []):
            if field_name not in params:
                errors.append(f"Pflichtparameter fehlt: '{field_name}'")
        return errors

    @staticmethod
    def _validate_type(name: str, value: object, field_schema: dict) -> str | None:
        """Prueft ob der Wert dem erwarteten JSON-Schema-Typ entspricht.

        Beruecksichtigt den Sonderfall, dass bool in Python ein Subtyp von int
        ist und daher fuer 'integer'-Felder explizit abgelehnt werden muss.

        Args:
            name: Parametername (fuer Fehlermeldung).
            value: Der zu pruefende Wert.
            field_schema: Schema-Definition des Feldes (mit optionalem 'type').

        Returns:
            Fehlermeldung als String oder None wenn der Typ korrekt ist.
        """
        expected_type = field_schema.get("type")
        if not expected_type:
            return None
        if isinstance(expected_type, list):
            # JSON-Schema erlaubt type-Listen wie ["string", "null"]
            if "null" in expected_type and value is None:
                return None
            known = [t for t in expected_type if t in _JSON_TYPE_MAP]
            unknown = [t for t in expected_type if t != "null" and t not in _JSON_TYPE_MAP]
            if unknown:
                return (
                    f"Parameter '{name}': Schema-Definition fehlerhaft - "
                    f"unbekannte Typen {unknown} "
                    f"(erlaubt: {', '.join(sorted(_JSON_TYPE_MAP))})"
                )
            if not known:
                return None
            # bool ist ein Subtyp von int — fuer integer/number ablehnen,
            # ausser boolean ist explizit in der Liste erlaubt
            if (
                isinstance(value, bool)
                and "boolean" not in known
                and ("integer" in known or "number" in known)
            ):
                return (
                    f"Parameter '{name}': erwartet Typ '{expected_type}', "
                    f"erhalten '{type(value).__name__}'"
                )
            allowed = tuple(t for k in known for t in _JSON_TYPE_MAP[k])
            if not isinstance(value, allowed):
                return (
                    f"Parameter '{name}': erwartet Typ '{expected_type}', "
                    f"erhalten '{type(value).__name__}'"
                )
            return None
        if expected_type not in _JSON_TYPE_MAP:
            return (
                f"Parameter '{name}': Schema-Definition fehlerhaft - "
                f"unbekannter Typ '{expected_type}' "
                f"(erlaubt: {', '.join(sorted(_JSON_TYPE_MAP))})"
            )
        allowed_types = _JSON_TYPE_MAP[expected_type]
        # bool ist ein Subtyp von int in Python — fuer integer und number
        # explizit ausschliessen (boolean-Schema bleibt davon unberuehrt)
        if expected_type in ("integer", "number") and isinstance(value, bool):
            return (
                f"Parameter '{name}': erwartet Typ '{expected_type}', "
                f"erhalten '{type(value).__name__}'"
            )
        if not isinstance(value, allowed_types):
            return (
                f"Parameter '{name}': erwartet Typ '{expected_type}', "
                f"erhalten '{type(value).__name__}'"
            )
        return None

    @staticmethod
    def _validate_enum(name: str, value: object, field_schema: dict) -> str | None:
        """Prueft ob der Wert in der erlaubten Enum-Liste enthalten ist.

        Args:
            name: Parametername (fuer Fehlermeldung).
            value: Der zu pruefende Wert.
            field_schema: Schema-Definition des Feldes (mit optionalem 'enum').

        Returns:
            Fehlermeldung als String oder None wenn der Wert erlaubt ist.
        """
        allowed_values = field_schema.get("enum")
        if allowed_values is not None and value not in allowed_values:
            return (
                f"Parameter '{name}': Wert '{value}' nicht erlaubt. "
                f"Erlaubte Werte: {allowed_values}"
            )
        return None

    @staticmethod
    def _validate_string_constraints(name: str, value: object, field_schema: dict) -> list[str]:
        """Prueft minLength/maxLength-Constraints fuer String-Parameter.

        Args:
            name: Parametername (fuer Fehlermeldung).
            value: Der zu pruefende Wert.
            field_schema: Schema-Definition des Feldes.

        Returns:
            Liste von Fehlermeldungen (leer wenn alle Constraints erfuellt).
        """
        errors: list[str] = []
        expected_type = field_schema.get("type")
        # Type-Listen wie ["string", "null"] muessen ebenfalls beruecksichtigt werden
        if isinstance(expected_type, list):
            is_string = "string" in expected_type
        else:
            is_string = expected_type == "string"
        if not is_string or not isinstance(value, str):
            return errors
        min_len = field_schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"Parameter '{name}': Mindestlaenge {min_len}, erhalten {len(value)}")
        max_len = field_schema.get("maxLength")
        if max_len is not None and len(value) > max_len:
            errors.append(f"Parameter '{name}': Maximallaenge {max_len}, erhalten {len(value)}")
        return errors

    @staticmethod
    def _validate_numeric_constraints(name: str, value: object, field_schema: dict) -> list[str]:
        """Prueft minimum/maximum-Constraints fuer numerische Parameter.

        Args:
            name: Parametername (fuer Fehlermeldung).
            value: Der zu pruefende Wert.
            field_schema: Schema-Definition des Feldes.

        Returns:
            Liste von Fehlermeldungen (leer wenn alle Constraints erfuellt).
        """
        errors: list[str] = []
        expected_type = field_schema.get("type")
        # Type-Listen wie ["integer", "null"] muessen ebenfalls beruecksichtigt werden
        if isinstance(expected_type, list):
            is_numeric = "integer" in expected_type or "number" in expected_type
        else:
            is_numeric = expected_type in ("integer", "number")
        if not is_numeric or isinstance(value, bool) or not isinstance(value, (int, float)):
            return errors
        minimum = field_schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(f"Parameter '{name}': Minimalwert {minimum}, erhalten {value}")
        maximum = field_schema.get("maximum")
        if maximum is not None and value > maximum:
            errors.append(f"Parameter '{name}': Maximalwert {maximum}, erhalten {value}")
        return errors

    def validate_params(self, params: dict) -> tuple[bool, str]:
        """Validiert Parameter gegen das JSON-Schema der Tool-Definition.

        Prueft required-Felder, Typ-Kompatibilitaet und enum-Werte.
        Nutzt eine eigene Implementierung ohne externe jsonschema-Dependency.

        Args:
            params: Die zu validierenden Parameter.

        Returns:
            Tuple (valid, message). Bei Erfolg (True, "ok"),
            bei Fehler (False, Fehlerbeschreibung).
        """
        defn = self.definition()
        schema = defn.parameters

        if not isinstance(params, dict):
            return False, "Parameter muessen ein Dictionary sein."

        # Required fields pruefen
        req_errors = self._validate_required(params, schema)
        if req_errors:
            return False, req_errors[0]

        # Jeden uebergebenen Parameter pruefen
        properties: dict = schema.get("properties") or {}
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
