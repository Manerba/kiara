"""Audit-Logging fuer Anonymisierungs-Events.

Schreibt in die Tabelle audit_anonymization,
damit nachvollziehbar ist, welche Entitaeten wann anonymisiert wurden.
"""

import hashlib
import json
import logging

from kiara.repositories.audit import AuditAnonymizationRepository

logger = logging.getLogger(__name__)

# Module-level Repo-Instanz
_anon_audit_repo = AuditAnonymizationRepository()


def log_anonymization_event(
    user_id: int | None,
    session_id: str,
    question: str,
    entity_count: int,
    entity_types: dict,
    route: str,
    cloud_provider: str | None = None,
    cloud_model: str | None = None,
    response_time_ms: int | None = None,
    correlation_id: str | None = None,
) -> None:
    """Schreibt einen Anonymisierungs-Audit-Eintrag in die DB."""
    try:
        if isinstance(question, bytes):
            q_bytes = question
        else:
            q_bytes = (question or "").encode("utf-8", errors="replace")
        question_hash = hashlib.sha256(q_bytes).hexdigest()[:16]
        entity_types_json = json.dumps(entity_types, ensure_ascii=False)

        fields = dict(
            user_id=user_id,
            session_id=session_id,
            question_hash=question_hash,
            entity_count=entity_count,
            entity_types=entity_types_json,
            route=route,
            cloud_provider=cloud_provider,
            cloud_model=cloud_model,
            response_time_ms=response_time_ms,
        )
        if correlation_id:
            fields["correlation_id"] = correlation_id
        _anon_audit_repo.create(**fields)
    except Exception as exc:
        logger.critical(
            "Anonymization-Audit-Persistenz fehlgeschlagen "
            "(Cloud-Send wird fail-closed abgebrochen): %s",
            exc,
            exc_info=True,
        )
        raise RuntimeError("Anonymization-Audit-Persistenz fehlgeschlagen") from exc
