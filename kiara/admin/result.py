from dataclasses import dataclass


@dataclass
class AdminResult:
    success: bool
    message: str  # Menschenlesbare Zusammenfassung
    data: dict | None  # Strukturierte Daten (optional)
    error: str | None  # Fehlermeldung (bei success=False)
    http_status: int = 200  # HTTP-Status-Code (fuer Router, default 200 bei ok, 400 bei fail)

    @classmethod
    def ok(cls, message: str, data: dict | None = None) -> "AdminResult":
        return cls(success=True, message=message, data=data, error=None, http_status=200)

    @classmethod
    def fail(cls, message: str, data: dict | None = None, http_status: int = 400) -> "AdminResult":
        """Erzeugt ein Fehler-Ergebnis. Name 'fail' statt 'error' um
        Namenskonflikt mit dem Feld 'error' zu vermeiden.

        http_status: HTTP-Status-Code (default 400, z.B. 404 fuer 'nicht gefunden').
        """
        return cls(
            success=False, message=message, data=data, error=message, http_status=http_status
        )
