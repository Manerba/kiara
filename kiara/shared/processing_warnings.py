"""Antwortbezogene Verarbeitungshinweise fuer UI und Persistierung.

Dieser Kanal enthaelt keine Modellnachrichten. Jeder Chat-Stream besitzt
einen eigenen Collector; kopierte Thread-Kontexte teilen nur diesen Collector.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

_COMPONENT_MESSAGES = {
    "reranking": "Die Relevanzsortierung ist ausgefallen.",
    "graph": "Die ergänzende Suche im Wissensgraphen ist ausgefallen.",
    "query_expansion": "Die Erweiterung der Suchanfrage ist ausgefallen.",
    "query_variant": "Eine ergänzende Suchanfrage konnte nicht ausgeführt werden.",
    "rewrite": "Die Umformulierung der Suchanfrage ist ausgefallen; gesucht wurde mit der ursprünglichen Frage.",
    "document_expansion": "Zusätzliche Dokumentabschnitte konnten nicht nachgeladen werden.",
}

_CONTEXT_MESSAGES = {
    "budget_limited": (
        "Die gefundenen Inhalte überschreiten das verfügbare Kontextbudget oder die "
        "Quellenobergrenze. Das Sprachmodell erhält nur einen Teil der Inhalte."
    ),
    "budget_exhausted": (
        "Das Kontextbudget reicht für keine weiteren Dokumentinhalte. Ein Suchergebnis "
        "oder angeforderter Dokumentabschnitt wurde dem Sprachmodell nicht übergeben."
    ),
    "chunk_limit": (
        "Die Quellenobergrenze wurde erreicht. Einige gefundene Textabschnitte "
        "wurden dem Sprachmodell nicht übergeben."
    ),
    "window_estimate_exceeded": (
        "Der vollständige Prompt überschreitet nach einer Schätzung das konfigurierte Kontextfenster "
        "des Sprachmodells. Inhalte könnten bei der Verarbeitung verloren gehen."
    ),
    "document_partial": (
        "Das Dokument wurde nur abschnittsweise gelesen. Weitere Bereiche "
        "sind noch nicht ausgewertet."
    ),
}

_RESEARCH_MESSAGES = {
    "supported": "Die Antwort wurde anhand nachgeladener Originalstellen geprüft.",
    "not_found_in_scope": "Im geprüften Umfang wurde keine ausreichend belegte Antwort gefunden.",
    "incomplete": "Die Recherche ist unvollständig. Ungeprüfte Bereiche und erreichte Grenzen bleiben offen.",
    "failed": "Die Recherche ist fehlgeschlagen; es liegt keine geprüfte Antwort vor.",
}

_RESEARCH_LIMIT_MESSAGES = {
    "disabled": "Recherche ist in der Konfiguration deaktiviert.",
    "unsupported_backend": "Das gewählte Backend wird für Recherche noch nicht unterstützt.",
    "model_output_limit": "Ein Modellaufruf hat sein Antwortlimit erreicht; der Befund bleibt ungeprüft.",
    "active_run_limit": "Alle Rechercheplätze sind belegt. Bitte später erneut versuchen.",
    "time_limit": "Zeitlimit erreicht.",
    "model_call_limit": "Limit der Modellaufrufe erreicht.",
    "read_limit": "Limit der Leserunden erreicht.",
    "store_limit": "Ergebnisablage voll; weitere Treffer wurden ausgelassen.",
    "document_window_store_limit": "Der geladene Dokumentbereich überschreitet das Ablagebudget.",
    "context_budget": "Modellaufruf wegen fehlenden Kontextbudgets abgelehnt.",
    "question_or_history_exceeds_window": "Frage oder Verlauf überschreitet das Kontextbudget.",
    "reading_window_too_small": "Das Kontextfenster reicht für keinen weiteren Dokumentbereich.",
    "final_evidence_limit": "Nicht alle gefundenen Belege passen in die abschließende Prüfung.",
    "unknown_model_window": "Die Modellgrenze konnte nicht ermittelt werden.",
    "measured_context_overflow": "Die gemessene Tokenzahl überschreitet die reservierte Grenze.",
    "invalid_reader_output": "Ein Leserbefund war ungültig und wurde nicht als Beleg übernommen.",
    "unversioned_result": "Ein Suchtreffer ohne prüfbare Version wurde ausgelassen.",
    "unread_document_ranges": "Teile der gefundenen Dokumente sind noch ungeprüft.",
    "repeated_search": "Es wurde kein neuer Suchansatz gefunden.",
    "uncertain_readings": "Einige gelesene Bereiche bleiben unklar.",
    "conflicting_evidence": "Die Originalbelege widersprechen sich.",
    "answer_verification_failed": "Der Antwortentwurf hat die Belegprüfung nicht bestanden.",
    "answer_not_supported": "Die vorliegenden Belege reichen nicht für eine Antwort aus.",
    "cancelled": "Die Recherche wurde abgebrochen.",
    "research_error": "Ein Such-, Lese- oder Modellaufruf ist fehlgeschlagen.",
}


class ProcessingWarnings:
    """Sammelt und zaehlt gleiche Ursachen innerhalb genau einer Antwort."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], dict] = {}
        self._lock = Lock()

    def add(self, component: str, reason: str, *, details: dict | None = None) -> None:
        message = (
            _RESEARCH_MESSAGES[reason]
            if component == "research"
            else _CONTEXT_MESSAGES[reason]
            if component == "context"
            else _COMPONENT_MESSAGES[component]
        )
        if component == "research" and details:
            limits = [
                _RESEARCH_LIMIT_MESSAGES[r]
                for r in details.get("limitations", [])
                if r in _RESEARCH_LIMIT_MESSAGES
            ]
            if limits:
                message += " " + " ".join(limits)
        if component == "reranking" and reason == "cuda_oom":
            message = "Die Relevanzsortierung ist wegen fehlenden GPU-Speichers ausgefallen."
        elif component == "reranking" and reason == "out_of_memory":
            message = "Die Relevanzsortierung ist wegen Speichermangels ausgefallen."
        elif component == "reranking" and reason == "timeout":
            message = "Die Relevanzsortierung hat nicht rechtzeitig geantwortet."
        key = (component, reason)
        with self._lock:
            if key in self._items:
                self._items[key]["count"] += 1
            else:
                self._items[key] = {
                    "code": "research_status"
                    if component == "research"
                    else "context_incomplete"
                    if component == "context"
                    else f"{component}_unavailable",
                    "component": component,
                    "reason": reason,
                    "message": message,
                    "count": 1,
                }
            if details is not None:
                self._items[key]["details"] = dict(details)

    def snapshot(self) -> list[dict]:
        with self._lock:
            items = [dict(self._items[key]) for key in sorted(self._items)]
            for item in items:
                if "details" in item:
                    item["details"] = dict(item["details"])
            return items


_current: ContextVar[ProcessingWarnings | None] = ContextVar("processing_warnings", default=None)


@contextmanager
def processing_warning_scope(collector: ProcessingWarnings):
    """Bindet den Collector nur fuer den aktuellen Verarbeitungsschritt."""
    token = _current.set(collector)
    try:
        yield collector
    finally:
        _current.reset(token)


def report_processing_warning(
    component: str, reason: str = "unavailable", *, details: dict | None = None
) -> None:
    collector = _current.get()
    if collector is not None:
        collector.add(component, reason, details=details)


def current_processing_warnings() -> list[dict]:
    collector = _current.get()
    return collector.snapshot() if collector is not None else []
