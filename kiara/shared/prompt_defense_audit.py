"""Audit-Logging fuer Prompt-Defense-Events.

Schreibt in die Tabelle prompt_defense_events,
damit nachvollziehbar ist, welche Injection-Versuche wann erkannt wurden.
"""

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from kiara.repositories.audit import PromptDefenseRepository

if TYPE_CHECKING:
    from kiara.shared.prompt_defense import SanitizeReport

logger = logging.getLogger(__name__)

_repo = PromptDefenseRepository()


def log_defense_event(
    source_type: str,
    report: "SanitizeReport",
    user_id: int | None = None,
    question: str | None = None,
    chat_id: int | None = None,
    document_source: str | None = None,
) -> None:
    """Schreibt ein Prompt-Defense-Event in die DB.

    Wird nur aufgerufen wenn report.severity != 'none' (also etwas entfernt wurde).

    Args:
        source_type: 'user_input', 'retrieval_context', 'tool_result', 'chat_history'
        report: SanitizeReport aus sanitize_with_report()
        user_id: User-ID (None bei Kontext-Sanitisierung ohne User-Bezug)
        question: Bereinigte Frage fuer Hash-Korrelation mit query_log
                  (muss der sanitisierten Version entsprechen, die _log_query_audit erhaelt)
        chat_id: Chat-ID zur Zuordnung
        document_source: Dateipfad bei retrieval_context
    """
    if report.severity == "none":
        return

    try:
        question_hash = hashlib.sha256(question.encode()).hexdigest()[:16] if question else None
        _repo.create(
            user_id=user_id,
            source_type=source_type,
            matched_patterns=json.dumps(report.matched_patterns),
            removed_snippets=json.dumps(report.removed_snippets, ensure_ascii=False),
            chars_removed=report.chars_removed,
            original_length=report.chars_removed + len(report.clean_text),
            severity=report.severity,
            question_hash=question_hash,
            chat_id=chat_id,
            document_source=document_source,
        )
    except Exception as exc:
        logger.error("Prompt-Defense-Audit Fehler: %s", exc)
