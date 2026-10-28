"""
KIara - Chat API-Router
Chat CRUD, Messages, RAG-Query.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from functools import wraps
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kiara.config import load_config
from kiara.exceptions import RetrievalError
from kiara.repositories.audit import QueryLogRepository
from kiara.repositories.chat import ChatRepository, MessageRepository
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.repositories.permission import PermissionRepository
from kiara.repositories.prompt import PromptTemplateRepository
from kiara.shared.constants import (
    ADMIN_PORT_DEFAULT,
    CHAT_HISTORY_LIMIT,
    CHAT_MSG_EXCERPT_CHARS,
    EXCERPT_LENGTH,
    MAX_AUDIT_TEXT,
)
from kiara.shared.db import db_query, db_transaction
from kiara.shared.processing_warnings import (
    ProcessingWarnings,
    current_processing_warnings,
    processing_warning_scope,
)
from kiara.web.auth import get_chat_user
from kiara.web.logging_config import get_security_logger

logger = logging.getLogger(__name__)
security_log = get_security_logger()

# --- Repository-Instanzen (Modul-Ebene) ---
chat_repo = ChatRepository()
msg_repo = MessageRepository()
audit_repo = QueryLogRepository()
prompt_repo = PromptTemplateRepository()
ds_repo = DatasourceRepository()
embedding_repo = DatasourceEmbeddingRepository()
perm_repo = PermissionRepository()


def _log_query_audit(
    user_id: int,
    question: str,
    answer: str,
    routed_to: str = "local",
    model_used: str = "",
    sources_count: int = 0,
    tools_used: str = "",
    chunks_retrieved: int = 0,
    entities_anonymized: int = 0,
    duration_ms: int = 0,
    chat_id: int | None = None,
    chain_type: str = "",
    full_prompt: str = "",
    full_answer: str = "",
    tool_calls_json: str = "",
    context_sources: str = "",
    trace_id: str | None = None,
    correlation_id: str = "",
    request_settings: dict | None = None,
):
    """Schreibt einen Eintrag in query_log fuer den Audit-Trail."""
    if full_prompt and len(full_prompt) > MAX_AUDIT_TEXT:
        full_prompt = full_prompt[:MAX_AUDIT_TEXT] + "\n[...gekuerzt]"
    if full_answer and len(full_answer) > MAX_AUDIT_TEXT:
        full_answer = full_answer[:MAX_AUDIT_TEXT] + "\n[...gekuerzt]"
    try:
        # bugscan:by-design[raw-query-audit] - query_log speichert Rohfrage,
        # Rohantwort, Prompt und Quellen fuer Audit/QS/Retrieval-Analyse. Das
        # ist nicht der Cloud-Privacy-Auditpfad; der liegt in cloud_audit_log
        # und audit_anonymization.
        audit_repo.create(
            user_id=user_id,
            question_hash=hashlib.sha256(question.encode()).hexdigest()[:16],
            question_excerpt=question[:EXCERPT_LENGTH],
            answer_excerpt=answer[:EXCERPT_LENGTH],
            routed_to=routed_to,
            model_used=model_used,
            sources_count=sources_count,
            tools_used=tools_used,
            chunks_retrieved=chunks_retrieved,
            entities_anonymized=entities_anonymized,
            duration_ms=duration_ms,
            chat_id=chat_id,
            chain_type=chain_type,
            full_prompt=full_prompt,
            full_answer=full_answer,
            tool_calls=tool_calls_json,
            context_sources=context_sources,
            trace_id=trace_id,
            correlation_id=correlation_id,
            request_settings=json.dumps(request_settings or {}, ensure_ascii=False),
        )
    except Exception as exc:
        logger.error("Audit-Log query_log Fehler: %s", exc, exc_info=True)


# --- Think-Block Stripping (qwen3 <think>-Protokoll) ---
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Entfernt <think>-Bloecke aus LLM-Antworten (geschlossen + offen).

    Idempotent — auf sauberem Text ein no-op.
    """
    text = _THINK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def _sse_event(event: str, data: dict | list) -> str:
    """Formatiert ein einzelnes SSE-Event als String."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _with_processing_warnings(stream_function):
    """Erfasst UI-Warnungen pro Antwort, auch aus spaeten Generator-Schritten.

    Der Kontext ist vor jedem yield wieder zurueckgesetzt. So bleiben
    verschachtelte Streams, Abbruch und parallele Requests voneinander isoliert.
    Die vollstaendige Liste ersetzt den bisherigen UI-Stand (idempotent).
    """

    @wraps(stream_function)
    async def stream(*args, **kwargs):
        collector = ProcessingWarnings()
        previous = []
        generator = stream_function(*args, **kwargs)
        try:
            while True:
                with processing_warning_scope(collector):
                    try:
                        event = await anext(generator)
                    except StopAsyncIteration:
                        break
                warnings = collector.snapshot()
                if warnings != previous:
                    yield _sse_event("processing_warnings", {"warnings": warnings})
                    previous = warnings
                yield event
        finally:
            with processing_warning_scope(collector):
                await generator.aclose()

    return stream


def _sse_error_done(chat_id: int, message: str) -> tuple[str, str]:
    """Erzeugt ein Error-Event + Done-Event als Tupel.

    Usage in Generatoren::

        err, done = _sse_error_done(chat_id, "Fehlertext")
        yield err
        yield done
    """
    err = _sse_event("error", {"message": message})
    done = _sse_event("done", {"chat_id": chat_id})
    return err, done


def _sse_index_compatibility_error_done(
    chat_id: int,
    exc,
    *,
    consumer: str,
) -> tuple[str, str]:
    """Terminate an SSE request while preserving the complete guard diagnostic."""
    diagnostic = exc.to_dict()
    err = _sse_event(
        "error",
        {
            "message": str(exc),
            "reason_code": exc.reason_code,
            "operation": exc.operation,
            "consumer": consumer,
            "index_compatibility_error": diagnostic,
        },
    )
    done = _sse_event("done", {"chat_id": chat_id, "status": "error"})
    return err, done


def _safe_trace_id(value) -> str | None:
    """Nur nicht-leere String-Trace-IDs an SSE/Audit weiterreichen.

    Schuetzt bestehende Tests mit MagicMock-Chain-Objekten davor, dass
    `json.dumps()` an einem nicht-serialisierbaren Mock scheitert.
    """
    return value if isinstance(value, str) and value else None


def _is_cloud_provider(backend) -> bool:
    """Prueft ob ein Backend ein Cloud-Provider ist (anthropic/openai).

    Akzeptiert sowohl Backend-Objekte (.provider) als auch Dicts (["provider"]).
    """
    provider = (
        backend.get("provider", "ollama")
        if isinstance(backend, dict)
        else getattr(backend, "provider", "ollama")
    )
    return provider in ("anthropic", "openai")


def _build_context_override(context_chunks: list[dict] | None) -> list[dict] | None:
    """Baut context_override aus Such-Tab-Chunks (kein Retrieval).

    Filtert ungueltige Eintraege (leerer Text, kein String) und
    gibt None zurueck wenn keine gueltigen Chunks vorhanden.
    """
    if not context_chunks:
        return None
    result = []
    for chunk in context_chunks:
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            continue
        result.append(
            {
                "text": chunk["text"],
                "metadata": {"source": chunk.get("source", "")},
                "score": float(chunk.get("score") or 0.0),
                "_doc_id": chunk.get("id"),
            }
        )
    return result if result else None


def _build_context_sources(sources: list[dict]) -> list[dict]:
    """Baut Kontext-Metadaten fuer Cloud-Audit aus Quellen-Liste.

    Aggregiert Chunks pro Quelldatei (kein Volltext, nur Metadaten).

    Returns:
        Liste von Dicts mit file und chunks pro Datei.
    """
    if not sources:
        return []
    file_counts: dict[str, int] = {}
    for s in sources:
        fname = s.get("source", "") or s.get("file", "")
        if fname:
            file_counts[fname] = file_counts.get(fname, 0) + 1
    return [{"file": f, "chunks": c} for f, c in file_counts.items()]


def _log_cloud_audit(
    config: dict,
    user_id: int,
    question: str,
    answer_text: str,
    anon_question: str,
    anon_answer: str,
    provider: str,
    model_name: str,
    usage: dict | None,
    entities_count: int,
    duration_ms: int,
    chain_type: str,
    context_sources: list[dict] | None = None,
    pseudonymizer_stats: dict | None = None,
    correlation_id: str = "",
) -> None:
    """Gemeinsames Cloud-Audit-Logging fuer alle Streaming-Modi.

    Schreibt sowohl den Cloud-Request-Audit (AuditLogger) als auch
    das Anonymisierungs-Event (log_anonymization_event).  Wird nur
    aufgerufen wenn route == "cloud".

    Args:
        config:              Globale Config.
        user_id:             User-ID.
        question:            Original-Frage (unanonymisiert).
        answer_text:         Bereinigte Antwort (Think-Blocks entfernt).
        anon_question:       Anonymisierte Frage (fuer Cloud-Audit).
        anon_answer:         Anonymisierte Antwort (fuer Cloud-Audit).
        provider:            Cloud-Provider (anthropic/openai).
        model_name:          Modellname.
        usage:               Token-Usage-Dict (oder None).
        entities_count:      Anzahl anonymisierter Entities.
        duration_ms:         Dauer in Millisekunden.
        chain_type:          Ketten-Typ (standard/unchained/agent).
        context_sources:     Aggregierte Quellen-Metadaten (optional).
        pseudonymizer_stats: Stats-Dict vom Pseudonymizer (optional).
    """
    from kiara.cloud.audit import AuditLogger

    audit = AuditLogger(config)
    kwargs = dict(
        user_id=user_id,
        question_anonymized=anon_question,
        answer_anonymized=anon_answer,
        provider=provider,
        model=model_name,
        tokens_in=(usage or {}).get("prompt_tokens", 0),
        tokens_out=(usage or {}).get("response_tokens", (usage or {}).get("completion_tokens", 0)),
        entities_anonymized=entities_count,
        duration_ms=duration_ms,
    )
    if context_sources is not None:
        kwargs["context_sources"] = context_sources
    if correlation_id:
        kwargs["correlation_id"] = correlation_id
    audit.log_request(**kwargs)

    from kiara.anonymization.audit import log_anonymization_event

    stats = pseudonymizer_stats or {"total_entities": 0, "entity_types": {}}
    log_anonymization_event(
        user_id=user_id,
        session_id=correlation_id or str(uuid.uuid4()),
        question=question,
        entity_count=stats.get("total_entities", 0),
        entity_types=stats.get("entity_types", {}),
        route="cloud",
        cloud_provider=provider,
        cloud_model=model_name,
        correlation_id=correlation_id or None,
    )


def _emit_cloud_audit(
    config: dict,
    user: dict,
    question: str,
    metrics: dict,
    chain_type: str,
    *,
    anon_question: str = "[Anonymisierung nicht aktiv]",
    anon_answer: str = "[Anonymisierung nicht aktiv]",
    provider: str = "",
    model_name: str = "",
    sources: list[dict] | None = None,
    pseudonymizer_stats: dict | None = None,
    correlation_id: str = "",
) -> None:
    """Cloud-Audit ausfuehren (OHNE Route-Check — Caller muss pruefen!)."""
    _log_cloud_audit(
        config=config,
        user_id=user["id"],
        question=question,
        answer_text=metrics["answer_text"],
        anon_question=anon_question,
        anon_answer=anon_answer,
        provider=provider,
        model_name=model_name,
        usage=metrics["usage"],
        entities_count=metrics["entities_count"],
        duration_ms=metrics["duration_ms"],
        chain_type=chain_type,
        context_sources=_build_context_sources(sources) if sources is not None else None,
        pseudonymizer_stats=pseudonymizer_stats,
        correlation_id=correlation_id,
    )


def _emit_query_audit(
    user: dict,
    question: str,
    metrics: dict,
    chain_type: str,
    chat_id: int,
    *,
    routed_to: str = "local",
    full_prompt: str = "",
    sources: list[dict] | None = None,
    trace_id: str | None = None,
    tools_used: str = "",
    tool_calls_json: str = "",
    correlation_id: str = "",
    request_settings: dict | None = None,
) -> None:
    """Schreibt Query-Audit-Eintrag mit Metriken aus _collect_stream_metrics()."""
    # bugscan:by-design[raw-query-audit] - _emit_query_audit reicht bewusst
    # rohe Query-/Promptdaten an query_log durch; siehe _log_query_audit.
    _log_query_audit(
        user_id=user["id"],
        question=question,
        answer=metrics["answer_text"],
        routed_to=routed_to,
        model_used=metrics["model_name"],
        sources_count=len(sources) if sources else 0,
        chunks_retrieved=len(sources) if sources else 0,
        entities_anonymized=metrics["entities_count"],
        duration_ms=metrics["duration_ms"],
        chat_id=chat_id,
        chain_type=chain_type,
        full_prompt=full_prompt,
        full_answer=metrics["full_answer_raw"],
        context_sources=json.dumps(sources) if sources else "",
        trace_id=trace_id,
        tools_used=tools_used,
        tool_calls_json=tool_calls_json,
        correlation_id=correlation_id,
        request_settings=request_settings,
    )


router = APIRouter(prefix="/api", tags=["chat"])


class CreateChatRequest(BaseModel):
    """Request zum Erstellen eines neuen Chats."""

    title: str = Field(..., max_length=500)
    context_chunks: list[dict] | None = None


class UpdateChatRequest(BaseModel):
    """Request zum Aktualisieren eines bestehenden Chats."""

    title: str | None = Field(None, max_length=500)
    status: Literal["active", "archived"] | None = None


class QueryRequest(BaseModel):
    """Request fuer eine RAG-Query mit optionalen Filtern und Modus-Auswahl."""

    question: str = Field(..., max_length=10_000)
    k: int | None = Field(None, ge=0, strict=True)
    num_ctx: int | None = Field(None, ge=256, le=131072)
    mode: str = "standard"
    template_id: int | None = None
    filter_dir: str | None = None
    filter_type: str | None = None
    filter_datasource_ids: list[int] | None = None
    filter_datasource_exclude: bool = False
    embedding_model: str | None = None
    llm_backend_id: int | None = None
    use_history: bool = False
    think: bool = False
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tool_repeats: int = Field(5, ge=0, le=50)
    context_chunks: list[dict] | None = None  # [{id, text, source, score}] aus Such-Tab


@router.get("/chat/modes")
async def get_chat_modes(user=Depends(get_chat_user)):
    """Liefert alle aktiven Chat-Modi (prompt_templates mit show_in_chat=1)."""
    # Rewriter-Modus nur anzeigen wenn ein lokales Ollama-Backend verfuegbar ist
    from kiara.backend_selector import get_all_active
    from kiara.retrieval.source_policy import get_source_profiles

    source_profiles = get_source_profiles(load_config())
    from kiara.generation.research_config import research_settings

    research_modes = (
        [
            {
                "id": 0,
                "name": "research",
                "label": "Recherche",
                "chain_type": "research",
                "badge_color": "#0891b2",
                "sort_order": 40,
                "description": "Sucht schrittweise und prüft die Belege.",
            }
        ]
        if research_settings(load_config())["enabled"]
        else []
    )
    has_ollama = any(b.get("provider") == "ollama" for b in get_all_active("llm"))

    try:
        allowed_chain_types = {"standard", "unchained", "agent", "rewriter"}
        modes = [m for m in prompt_repo.list_chat_modes() if m["chain_type"] in allowed_chain_types]
        if not has_ollama:
            modes = [m for m in modes if m["chain_type"] != "rewriter"]
        return {"modes": modes + research_modes, "source_profiles": source_profiles}
    except Exception as exc:
        logger.warning(
            "Chat-Modes DB-Abfrage fehlgeschlagen, nutze Fallback: %s",
            exc,
            exc_info=True,
        )
        # Fallback: Tabelle existiert noch nicht (vor Migration)
        modes = [
            {
                "id": 0,
                "name": "system_standard",
                "label": "Standard",
                "chain_type": "standard",
                "badge_color": "",
                "sort_order": 10,
            },
            {
                "id": 0,
                "name": "system_rewriter",
                "label": "Hoch",
                "chain_type": "rewriter",
                "badge_color": "#10b981",
                "sort_order": 15,
            },
            {
                "id": 0,
                "name": "system_unchained",
                "label": "Unchained",
                "chain_type": "unchained",
                "badge_color": "#f59e0b",
                "sort_order": 20,
            },
            {
                "id": 0,
                "name": "system_agent",
                "label": "Agent",
                "chain_type": "agent",
                "badge_color": "#8b5cf6",
                "sort_order": 30,
            },
        ]
        if not has_ollama:
            modes = [m for m in modes if m["chain_type"] != "rewriter"]
        return {"modes": modes + research_modes, "source_profiles": source_profiles}


@router.get("/chat/filters")
async def get_chat_filters(user=Depends(get_chat_user)):
    """Get available filter values for chat (datasources, file types)."""
    try:
        # Datasources from DB — fast and scalable
        ds_ids_with_chunks = embedding_repo.get_ds_ids_with_chunks()
        if user.get("is_admin"):
            ds_list, _ = ds_repo.list_filtered(
                {"is_active": True},
                sort="name ASC",
                limit=10000,
                offset=0,
            )
            rows = [r for r in ds_list if r.get("id") in ds_ids_with_chunks]
        else:
            user_ds_ids = user.get("datasource_ids", [])
            public_ids = perm_repo.get_public_datasource_ids()
            allowed_ids = list(set(user_ds_ids + public_ids))
            if allowed_ids:
                ds_list, _ = ds_repo.list_filtered(
                    {"is_active": True, "id__in": allowed_ids},
                    sort="name ASC",
                    limit=10000,
                    offset=0,
                )
                rows = [r for r in ds_list if r.get("id") in ds_ids_with_chunks]
            else:
                rows = []

        # Distinct embedding models with data
        emb_models = embedding_repo.get_active_models()

        datasources = [{"id": r["id"], "name": r["name"]} for r in rows]

        # Aktive Embedding-Backends fuer has_backend Flag
        from kiara.backend_selector import get_all_active, normalize_model_name

        active_embedding_models = {
            normalize_model_name(b["model"]) for b in get_all_active("embedding")
        }

        embedding_models = [
            {
                "model": m,
                "has_backend": normalize_model_name(m) in active_embedding_models,
            }
            for m in emb_models
            if m  # leere Strings filtern
        ]

        # File types from config
        config = load_config()
        file_types = config["datasource"].get("file_types", [])
        from kiara.config import get_embedding_model

        default_model = get_embedding_model(config)

        # Aktive LLM-Backends inkl. Cloud (mit num_ctx aus extra_config)
        llm_backends_raw = get_all_active("llm") + get_all_active("cloud")

        def _parse_num_ctx(b):
            try:
                return json.loads(b.get("extra_config") or "{}").get("num_ctx")
            except (ValueError, TypeError):
                return None

        llm_backends = [
            {
                "id": b["id"],
                "name": b["name"],
                "model": b["model"],
                "provider": b.get("provider", "ollama"),
                "num_ctx": _parse_num_ctx(b),
            }
            for b in llm_backends_raw
        ]

        return {
            "datasources": datasources,
            "file_types": file_types,
            "embedding_models": embedding_models,
            "default_embedding_model": default_model,
            "llm_backends": llm_backends,
        }
    except Exception as exc:
        logger.warning("get_chat_filters fehlgeschlagen: %s", exc, exc_info=True)
        return {
            "datasources": [],
            "file_types": [],
            "embedding_models": [],
            "default_embedding_model": "",
            "llm_backends": [],
        }


def _ensure_chat_owner(chat_id: int, user_id: int) -> dict:
    """Prueft ob der Chat existiert und dem User gehoert.

    Raises:
        HTTPException(404): Wenn Chat nicht existiert oder fremder User.
    """
    chat = chat_repo.get_owned(chat_id, user_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat nicht gefunden")
    return chat


@router.post("/chat")
async def create_chat(data: CreateChatRequest, user=Depends(get_chat_user)):
    """Create a new chat."""
    with db_transaction() as db:
        chat_id = chat_repo.create(user_id=user["id"], title=data.title, conn=db)
        # Kontext-Chunks als Messages mit role='context' speichern
        if data.context_chunks:
            for chunk in data.context_chunks:
                if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                    continue
                content = json.dumps(
                    {
                        "text": chunk["text"],
                        "source": chunk.get("source", ""),
                        "score": float(chunk.get("score") or 0.0),
                    },
                    ensure_ascii=False,
                )
                msg_repo.add_message(chat_id, "context", content, conn=db)

    return {"id": chat_id, "title": data.title}


def _load_chat_trace_rows(chat_id: int) -> list[dict]:
    """Liest Trace-Rows fuer Chat-History-Rehydration.

    Die spaetere Zuordnung erfolgt NICHT rein positionsbasiert, weil
    `query_log`-Rows und `chat_messages` bei identischem `created_at`
    unterschiedlich sortiert sein koennen. Stattdessen wird spaeter zuerst
    gegen `full_answer` und nur legacy-kompatibel gegen `answer_excerpt`
    gematcht.
    """
    with db_query() as conn:
        rows = conn.execute(
            """
            SELECT trace_id, answer_excerpt, full_answer
            FROM query_log
            WHERE chat_id = %s
              AND trace_id IS NOT NULL
            ORDER BY created_at ASC, id ASC
            """,
            (chat_id,),
        ).fetchall()
    return [
        {
            "trace_id": row["trace_id"],
            "answer_excerpt": (row["answer_excerpt"] or ""),
            "full_answer": row["full_answer"] or "",
        }
        for row in rows
        if row["trace_id"]
    ]


def _find_trace_row_for_message(content: str, remaining: list[dict]) -> int | None:
    """Bevorzugt exakten Match ueber full_answer, faellt sonst auf Excerpt zurueck."""
    normalized_content = _strip_think_blocks(content or "")
    prefix_match_idx = None
    for idx, row in enumerate(remaining):
        normalized_full_answer = _strip_think_blocks(row["full_answer"] or "")
        if normalized_full_answer == normalized_content == "":
            return idx
        if normalized_full_answer and normalized_full_answer == normalized_content:
            return idx

        excerpt = _strip_think_blocks(row["answer_excerpt"] or "")
        if prefix_match_idx is None and excerpt and normalized_content.startswith(excerpt):
            prefix_match_idx = idx

    return prefix_match_idx


def _message_sort_key(message: dict) -> tuple[str, int]:
    """Stabilisiert `chat_messages` auch bei `created_at=None`."""
    created_at = message.get("created_at")
    return ("\uffff" if created_at is None else str(created_at), message["id"])


def _serialize_chat_messages(messages, trace_rows: list[dict]):
    remaining = list(trace_rows)
    result = []
    for m in messages:
        item = {
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "sources": json.loads(m["sources"]) if m["sources"] else [],
            "warnings": m.get("warnings") or [],
            "created_at": m["created_at"],
        }
        trace_id = None
        if m["role"] == "assistant":
            match_idx = _find_trace_row_for_message(m["content"] or "", remaining)
            if match_idx is not None:
                trace_id = remaining.pop(match_idx)["trace_id"]
        item["trace_id"] = trace_id
        result.append(item)
    return result


@router.get("/chat/{chat_id}")
async def get_chat(chat_id: int, user=Depends(get_chat_user)):
    chat = _ensure_chat_owner(chat_id, user["id"])
    messages = sorted(
        msg_repo.list_by_chat(chat_id),
        key=_message_sort_key,
    )
    trace_rows = _load_chat_trace_rows(chat_id)

    return {
        "id": chat["id"],
        "title": chat["title"],
        "status": chat["status"],
        "created_at": chat["created_at"],
        "messages": _serialize_chat_messages(messages, trace_rows),
    }


@router.put("/chat/{chat_id}")
async def update_chat(chat_id: int, data: UpdateChatRequest, user=Depends(get_chat_user)):
    """Update chat (title, status)."""
    _ensure_chat_owner(chat_id, user["id"])

    fields = {}
    if data.title is not None:
        fields["title"] = data.title
    if data.status is not None:
        fields["status"] = data.status
    if fields:
        from kiara.repositories.base import RawSQL

        fields["updated_at"] = RawSQL("NOW()")
        chat_repo.update(chat_id, **fields)
    else:
        chat_repo.touch(chat_id)

    return {"message": "Chat aktualisiert"}


@router.delete("/chat/{chat_id}")
async def delete_chat(chat_id: int, user=Depends(get_chat_user)):
    """Delete a chat and its messages."""
    _ensure_chat_owner(chat_id, user["id"])
    with db_transaction() as db:
        msg_repo.delete_by_chat(chat_id, conn=db)
        chat_repo.delete(chat_id, conn=db)

    return {"message": "Chat geloescht"}


@router.get("/chats")
async def list_chats(user=Depends(get_chat_user)):
    """List all chats for the current user."""
    chats = chat_repo.list_by_user(user["id"])

    return {
        "data": [
            {
                "id": c["id"],
                "title": c["title"],
                "status": c["status"],
                "message_count": c["message_count"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
            }
            for c in chats
        ]
    }


def _resolve_chat_id(chat_id: str, user: dict, question: str) -> int:
    """Chat-ID aufloesen: neuen Chat erstellen oder bestehenden validieren.

    Args:
        chat_id:  ``"new"`` fuer neuen Chat, sonst numerische ID.
        user:     Aktueller User-Dict (muss ``id`` enthalten).
        question: Frage-Text (wird als Titel bei neuem Chat verwendet).

    Returns:
        Numerische Chat-ID.

    Raises:
        HTTPException(404): Wenn bestehender Chat nicht gefunden oder fremder User.
    """
    if chat_id == "new":
        return chat_repo.create(user_id=user["id"], title=question[:100])
    try:
        actual_id = int(chat_id)
    except (ValueError, TypeError):
        raise HTTPException(400, f"Ungueltige Chat-ID: {chat_id}")
    _ensure_chat_owner(actual_id, user["id"])
    return actual_id


def _resolve_template(
    mode: str, template_id: int | None
) -> tuple[str, str | None, str | None, str | None]:
    """Loesung des Template-Inhalts basierend auf template_id oder Legacy-Mode.

    Args:
        mode:        Chat-Modus-String (standard/unchained/agent).
        template_id: Optionale Template-ID aus der DB.

    Returns:
        Tuple von (chain_type, template_content, final_prompt, cloud_system_prompt).
        template_content/final_prompt/cloud_system_prompt ist None wenn kein Override.
    """
    if mode == "research":
        return "research", None, None, None
    if template_id:
        try:
            row = prompt_repo.get_active(template_id)
            if (
                row
                and row.get("show_in_chat")
                and not row.get("is_cloud")
                and row.get("chain_type") in {"standard", "unchained", "agent", "rewriter"}
            ):
                fp = row["final_prompt"] if row["final_prompt"] else None
                csp = _effective_cloud_system_prompt(row)
                return row["chain_type"], row["content"], fp, csp
        except Exception as exc:
            logger.debug("Template-Resolution fehlgeschlagen (id=%s): %s", template_id, exc)

    # Legacy-Mapping: Mode-String → system_* Template
    _LEGACY_MAP = {
        "standard": "system_standard",
        "rewriter": "system_rewriter",
        "unchained": "system_unchained",
        "agent": "system_agent",
    }
    template_name = _LEGACY_MAP.get(mode, "system_standard")

    try:
        row = prompt_repo.get_by_name(template_name)
        if row:
            fp = row["final_prompt"] if row["final_prompt"] else None
            csp = _effective_cloud_system_prompt(row)
            return row["chain_type"], row["content"], fp, csp
    except Exception as exc:
        logger.debug("Legacy-Template-Resolution fehlgeschlagen (%s): %s", template_name, exc)

    # Absoluter Fallback: kein DB-Zugriff moeglich
    return mode, None, None, None


def _effective_cloud_system_prompt(row: dict) -> str | None:
    """Ermittelt den effektiven Cloud-System-Prompt fuer ein Template.

    Bei unchained/is_cloud-Templates wird content direkt als Cloud-System-Prompt
    verwendet (da keine Platzhalter). Bei standard/agent-Templates wird das
    separate cloud_system_prompt-Feld genutzt.
    """
    explicit = row.get("cloud_system_prompt") or ""
    if explicit:
        return explicit
    # Unchained + is_cloud: content IST der System-Prompt (keine Platzhalter)
    if row.get("chain_type") == "unchained" or row.get("is_cloud"):
        return row.get("content") or None
    return None


def _build_datasource_filter(
    user: dict, selected_ds_ids: list[int] | None, exclude: bool = False
) -> list[int] | None:
    """Permission-basierter Datasource-Filter fuer RAG-Queries.

    Admins erhalten None (= alle Datasources). Normale User erhalten
    ihre Gruppen-Datasources + oeffentliche Datasources (ohne Gruppenzuordnung).
    Im Exclude-Modus werden die ausgewaehlten IDs von den erlaubten abgezogen.

    Args:
        user:             User-Dict mit ``is_admin``, ``datasource_ids``.
        selected_ds_ids:  Vom User im UI ausgewaehlte Datasource-IDs (optional).
        exclude:          True = NOT-Modus (Auswahl ausschliessen).

    Returns:
        Liste erlaubter Datasource-IDs, oder None fuer uneingeschraenkten Zugriff.
    """
    if user.get("is_admin"):
        if not selected_ds_ids:
            return None
        if not exclude:
            return selected_ds_ids
        # Exclude: alle aktiven minus Auswahl
        all_active, _ = ds_repo.list_filtered(
            {"is_active": True},
            sort="id ASC",
            limit=10000,
            offset=0,
        )
        remaining = [r["id"] for r in all_active if r["id"] not in selected_ds_ids]
        return remaining if remaining else []

    user_ds_ids = user.get("datasource_ids", [])
    public_ids = perm_repo.get_public_datasource_ids()
    allowed = list(set(user_ds_ids + public_ids))

    if not selected_ds_ids:
        return allowed

    if exclude:
        remaining = [a for a in allowed if a not in selected_ds_ids]
        return remaining if remaining else []

    return [sid for sid in selected_ds_ids if sid in allowed] or []


def _build_restriction_filter(
    user: dict,
    selected_ds_ids: list[int] | None,
    exclude: bool = False,
) -> list[int]:
    """Restriction-basierter Datasource-Filter fuer Cloud-Policy.

    Wie _build_datasource_filter(), aber materialisiert fuer Admins ohne Filter
    alle aktiven DS-IDs statt None zurueckzugeben. Cloud-Policy braucht immer
    eine explizite ID-Liste.

    Returns:
        Liste von DS-IDs (nie None).

    Raises:
        RuntimeError: Bei DB-Fehler waehrend der Materialisierung.
    """
    retrieval_filter = _build_datasource_filter(user, selected_ds_ids, exclude)
    if retrieval_filter is not None:
        return retrieval_filter

    # Admin + kein Filter -> alle aktiven DS-IDs materialisieren
    try:
        all_active = ds_repo.get_active()
        return sorted([r["id"] for r in all_active])
    except Exception as exc:
        raise RuntimeError(
            f"Restriction-Filter: aktive Datasources konnten nicht geladen werden: {exc}"
        ) from exc


def _build_filter_pair(
    user: dict,
    data: QueryRequest,
) -> tuple[list[int] | None, list[int]]:
    """Erzeugt Retrieval-Filter + Restriction-Filter in einem Schritt.

    Returns:
        (filter_datasource_ids, restriction_datasource_ids)
        Erster Wert kann None sein (Admin ohne Selektion = alle).
        Zweiter Wert ist immer eine explizite ID-Liste.
    """
    return (
        _build_datasource_filter(user, data.filter_datasource_ids, data.filter_datasource_exclude),
        _build_restriction_filter(user, data.filter_datasource_ids, data.filter_datasource_exclude),
    )


def _save_assistant_message(
    chat_id: int, answer_text: str, sources: list[dict] | None = None
) -> None:
    """Speichert die Assistenten-Antwort in der DB.

    Args:
        chat_id: Die Chat-ID.
        answer_text: Der vollstaendige Antworttext.
        sources: Optionale Quellen-Liste (wird JSON-serialisiert).
    """
    with db_transaction() as db:
        msg_repo.add_message(
            chat_id,
            "assistant",
            answer_text,
            sources,
            warnings=current_processing_warnings(),
            conn=db,
        )
        chat_repo.touch(chat_id, conn=db)


def _save_successful_assistant_message(
    chat_id: int,
    answer_text: str,
    sources: list[dict] | None = None,
) -> None:
    """Persistiert auch leere erfolgreiche Antworten fuer query_log-Korrelation."""
    if sources is None:
        _save_assistant_message(chat_id, answer_text)
    else:
        _save_assistant_message(chat_id, answer_text, sources)


def _load_chat_history(chat_id: int) -> list[dict]:
    """Laedt die letzten Nachrichten eines Chats als Konversationskontext.

    Laedt CHAT_HISTORY_LIMIT+1 Nachrichten (DESC), kehrt die Reihenfolge um und
    schneidet die letzte User-Nachricht ab (= aktuelle Frage).

    Returns:
        Liste von Dicts mit 'role' und 'content'.
    """
    rows = msg_repo.load_history(chat_id, limit=CHAT_HISTORY_LIMIT + 1)
    if not rows:
        return []
    messages = [{"role": r["role"], "content": r["content"]} for r in rows]
    # Letzte User-Nachricht entfernen (= aktuelle Frage, bereits separat im Prompt)
    if messages and messages[-1]["role"] == "user":
        messages = messages[:-1]
    return messages


def _format_history_block(history: list[dict]) -> str:
    """Formatiert Chat-History als Prompt-Block.

    Args:
        history: Liste von Dicts mit 'role' und 'content'.

    Returns:
        Formatierter String oder leerer String wenn keine History.
    """
    if not history:
        return ""
    lines = ["Bisheriger Gespraechsverlauf:"]
    for msg in history:
        role = "Benutzer" if msg["role"] == "user" else "Assistent"
        content = msg["content"] or ""
        if len(content) > CHAT_MSG_EXCERPT_CHARS:
            content = content[:CHAT_MSG_EXCERPT_CHARS] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _prepend_chat_created(generator, chat_id: int):
    """Yieldet ein fruehes chat_created-Event und delegiert dann an den Stream.

    Das Frontend (chat.js-Wiring) setzt damit currentChatId generation-gebunden
    VOR einem moeglichen Abort oder Netzwerkverlust. Legacy done.chat_id bleibt
    zusaetzlich als Ende-Signal.
    """
    yield _sse_event("chat_created", {"chat_id": chat_id})
    async for event in generator:
        yield event


@router.post("/chat/{chat_id}/query")
async def chat_query(chat_id: str, data: QueryRequest, user=Depends(get_chat_user)):
    """Execute a query with SSE streaming — dispatched by mode."""
    from kiara.web.api_health import _get_degradation_manager

    dm = _get_degradation_manager()
    llm_available = dm.is_feature_available("llm")
    cloud_available = dm.is_feature_available("cloud")
    if not (llm_available or cloud_available):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Kein LLM- oder Cloud-Backend erreichbar. Bitte spaeter erneut versuchen."
            },
        )

    from kiara.shared.enums import ChatMode

    config = load_config()
    mode = data.mode or config.get("chat", {}).get("default_mode", ChatMode.STANDARD)

    # Template-Aufloesung: template_id → (chain_type, content, final_prompt, cloud_system_prompt)
    chain_type, template_content, final_prompt, cloud_system_prompt = _resolve_template(
        mode, data.template_id
    )

    from kiara.retrieval.source_policy import resolve_chat_sources

    try:
        config, effective_k = resolve_chat_sources(config, chain_type, data.k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    data = data.model_copy(update={"k": effective_k})

    # Prompt-Defense: User-Input sanitisieren
    from kiara.shared.prompt_defense import sanitize_with_report
    from kiara.shared.prompt_defense_audit import log_defense_event

    _pd_report = sanitize_with_report(data.question)
    clean_question = _pd_report.clean_text

    was_new_chat = chat_id == "new"
    if not clean_question.strip():
        actual_chat_id = None
        if not was_new_chat:
            actual_chat_id = _resolve_chat_id(chat_id, user, "")
        if _pd_report.severity != "none":
            log_defense_event(
                "user_input",
                _pd_report,
                user_id=user.get("id"),
                question=clean_question,
                chat_id=actual_chat_id,
            )

        async def _empty_gen():
            message = "Frage ist nach Bereinigung leer."
            yield _sse_event("error", {"message": message, "content": message})
            yield _sse_event("done", {"chat_id": actual_chat_id})

        return StreamingResponse(_empty_gen(), media_type="text/event-stream")

    data.question = clean_question
    actual_chat_id = _resolve_chat_id(chat_id, user, data.question)

    if _pd_report.severity != "none":
        log_defense_event(
            "user_input",
            _pd_report,
            user_id=user.get("id"),
            question=data.question,
            chat_id=actual_chat_id,
        )

    # Save sanitized user message only.
    msg_repo.add_message(actual_chat_id, "user", data.question)

    # Gespraechskontext laden wenn aktiviert
    history_block = ""
    if data.use_history:
        history = _load_chat_history(actual_chat_id)
        history_block = _format_history_block(history)

    if chain_type == ChatMode.RESEARCH:
        stream = _stream_research(config, data, actual_chat_id, user, history_block)
    elif chain_type == ChatMode.UNCHAINED:
        stream = _stream_unchained(
            config,
            data,
            actual_chat_id,
            user,
            template_content,
            history_block,
            cloud_system_prompt=cloud_system_prompt,
        )
    elif chain_type == ChatMode.AGENT:
        stream = _stream_agent(
            config, data, actual_chat_id, user, template_content, history_block, final_prompt
        )
    elif chain_type == ChatMode.REWRITER:
        stream = _stream_rewriter(
            config,
            data,
            actual_chat_id,
            user,
            template_content,
            history_block,
            cloud_system_prompt=cloud_system_prompt,
        )
    else:
        stream = _stream_standard(
            config,
            data,
            actual_chat_id,
            user,
            template_content,
            history_block,
            cloud_system_prompt=cloud_system_prompt,
        )

    if was_new_chat:
        stream = _prepend_chat_created(stream, actual_chat_id)

    return StreamingResponse(stream, media_type="text/event-stream")


@_with_processing_warnings
async def _stream_research(config, data, actual_chat_id, user, history_block=""):
    from kiara.services.research import create_research
    from kiara.shared.processing_warnings import report_processing_warning

    chain = None
    started = time.monotonic()
    yield _sse_event("mode", {"mode": "research"})
    try:
        if data.context_chunks:
            raise ValueError(
                "Recherche benötigt versionierte Originalquellen; Such-Tab-Auszüge bitte abwählen."
            )
        chain = await asyncio.to_thread(
            create_research,
            config,
            user,
            data.question,
            k=data.k,
            num_ctx=data.num_ctx,
            history=history_block,
            backend_id=data.llm_backend_id,
            embedding_model=data.embedding_model,
            selected=data.filter_datasource_ids,
            exclude=data.filter_datasource_exclude,
            filter_dir=data.filter_dir,
            filter_type=data.filter_type,
        )
        async for progress in chain.run():
            yield _sse_event("research_progress", progress)
        result = chain.result
        _save_successful_assistant_message(actual_chat_id, result["answer"], result["sources"])
        _log_query_audit(
            user["id"],
            data.question,
            result["answer"],
            duration_ms=int((time.monotonic() - started) * 1000),
            model_used=result["model"],
            chain_type="research",
            chat_id=actual_chat_id,
            routed_to="local",
            correlation_id=chain.run_id,
            context_sources=json.dumps(result["sources"]),
            request_settings={"research": result},
        )
        yield _sse_event(
            "research_status",
            {
                k: result[k]
                for k in ("run_id", "status", "evidence_status", "coverage", "limitations")
            },
        )
        yield _sse_event("sources", result["sources"])
        yield _sse_event("chunk", {"text": result["answer"]})
    except (asyncio.CancelledError, GeneratorExit):
        if chain:
            chain.cancelled.set()
        report_processing_warning(
            "research", "incomplete", details={"status": "incomplete", "limitations": ["cancelled"]}
        )
        _save_assistant_message(
            actual_chat_id,
            "Recherche abgebrochen. Es liegt keine abschließend geprüfte Antwort vor.",
        )
        raise
    except Exception as exc:
        logger.exception("Recherche-Chat fehlgeschlagen")
        report_processing_warning("research", "failed", details={"status": "failed"})
        message = (
            str(exc)
            if isinstance(exc, ValueError)
            else "Die Recherche konnte nicht durchgeführt werden. Bitte Backend, Quellen und Recherche-Konfiguration prüfen."
        )
        _save_assistant_message(actual_chat_id, message)
        yield _sse_event("error", {"message": message})
    yield _sse_event(
        "done", {"chat_id": actual_chat_id, "research_run_id": chain.run_id if chain else None}
    )


def _prepare_standard_context(config, data, user, template_content, cloud_system_prompt=None):
    """Bereitet Backend + Chain + Context-Override fuer Standard-RAG vor.

    Erstellt den Datasource-Filter (Berechtigungspruefung), konvertiert
    optionale Such-Tab-Chunks in das Context-Override-Format, erstellt
    das Backend mit Routing-Logik und instanziiert die RAGChain.

    Args:
        config:               Globale Config.
        data:                 QueryRequest mit Filtern und Context-Chunks.
        user:                 User-Dict (fuer Datasource-Berechtigungen).
        template_content:     Optionaler Prompt-Template-Override.
        cloud_system_prompt:  Optionaler System-Prompt fuer Cloud-Backends.

    Returns:
        Tuple von (chain, filter_datasource_ids, context_override).
    """
    filter_datasource_ids, restriction_datasource_ids = _build_filter_pair(user, data)

    context_override = _build_context_override(data.context_chunks)

    # Backend erstellen mit Routing-Logik
    from kiara.generation.cloud_routing import _create_chain_backend

    backend = _create_chain_backend(
        config,
        question=data.question,
        llm_backend_id=data.llm_backend_id,
        filter_datasource_ids=restriction_datasource_ids,
    )

    effective_system_prompt = (cloud_system_prompt or "") if _is_cloud_provider(backend) else ""

    from kiara.generation.chain import RAGChain

    chain = RAGChain(
        config,
        backend=backend,
        embedding_model=data.embedding_model,
        prompt_template_override=template_content,
        system_prompt=effective_system_prompt,
    )
    return chain, filter_datasource_ids, context_override


def _collect_stream_metrics(
    full_answer: list[str],
    chain,
    t_start: float,
    *,
    model_name_override: str | None = None,
    entities_count_override: int | None = None,
) -> dict:
    """Sammelt Post-Stream-Metriken aus Chain + Antwort-Chunks.

    Args:
        model_name_override:    Fuer Agent-Modus (model_name aus Backend-Dict statt usage).
        entities_count_override: Fuer Agent-Modus (entities aus Anonymizer statt Chain).

    Returns:
        Dict mit keys: full_answer_raw, answer_text, duration_ms,
        usage, model_name, entities_count.
    """
    raw = "".join(full_answer)
    usage = chain.last_usage
    return {
        "full_answer_raw": raw,
        "answer_text": _strip_think_blocks(raw),
        "duration_ms": int((time.monotonic() - t_start) * 1000),
        "usage": usage,
        "model_name": model_name_override
        if model_name_override is not None
        else (usage or {}).get("model", ""),
        "entities_count": entities_count_override
        if entities_count_override is not None
        else chain.last_entities_anonymized,
    }


def _collect_stream_error_metrics(
    full_answer: list[str],
    chain,
    t_start: float,
    error_message: str,
    *,
    model_name_override: str | None = None,
    entities_count_override: int | None = None,
) -> dict:
    """Metriken fuer fehlgeschlagene Streams inklusive sichtbarem Audit-Marker."""
    if chain is not None:
        try:
            metrics = _collect_stream_metrics(
                full_answer,
                chain,
                t_start,
                model_name_override=model_name_override,
                entities_count_override=entities_count_override,
            )
        except Exception:
            logger.warning(
                "Stream-Error-Metrics konnten nicht aus Chain abgeleitet werden — nutze Fallback",
                exc_info=True,
            )
            metrics = None
    else:
        metrics = None

    if metrics is None:
        metrics = {
            "full_answer_raw": "".join(full_answer),
            "answer_text": _strip_think_blocks("".join(full_answer)),
            "duration_ms": int((time.monotonic() - t_start) * 1000),
            "usage": None,
            "model_name": model_name_override or "",
            "entities_count": entities_count_override or 0,
        }

    marker = f"[STREAM_ERROR] {error_message}"
    raw = metrics.get("full_answer_raw", "") or ""
    raw = f"{raw}\n{marker}" if raw else marker
    metrics["full_answer_raw"] = raw
    metrics["answer_text"] = _strip_think_blocks(raw)
    return metrics


def _audit_stream_failure(
    config: dict,
    user: dict,
    question: str,
    chat_id: int,
    chain_type: str,
    error_message: str,
    full_answer: list[str],
    t_start: float,
    *,
    chain=None,
    sources: list[dict] | None = None,
    routed_to: str | None = None,
    full_prompt: str = "",
    trace_id: str | None = None,
    tools_used: str = "",
    tool_calls_json: str = "",
    correlation_id: str = "",
    model_name_override: str | None = None,
    entities_count_override: int | None = None,
    cloud_provider: str = "",
    cloud_model: str = "",
    anon_question: str = "[Anonymisierung nicht aktiv]",
    anon_answer: str = "[Anonymisierung nicht aktiv]",
    pseudonymizer_stats: dict | None = None,
) -> None:
    """Schreibt Audit-Trail fuer Streams, die vor dem normalen Epilog scheitern."""
    metrics = _collect_stream_error_metrics(
        full_answer,
        chain,
        t_start,
        error_message,
        model_name_override=model_name_override,
        entities_count_override=entities_count_override,
    )
    route = routed_to
    if route is None and chain is not None:
        route = getattr(chain, "route_used", None)
    if not isinstance(route, str) or not route:
        route = "local"

    if not full_prompt and chain is not None:
        full_prompt = getattr(chain, "last_prompt", "") or ""

    _emit_query_audit(
        user,
        question,
        metrics,
        chain_type,
        chat_id,
        routed_to=route,
        full_prompt=full_prompt,
        sources=sources,
        trace_id=trace_id,
        tools_used=tools_used,
        tool_calls_json=tool_calls_json,
        correlation_id=correlation_id,
        request_settings=config.get("_chat_source_policy", {}),
    )

    if route == "cloud":
        backend = getattr(chain, "backend", None) if chain is not None else None
        provider = cloud_provider or getattr(backend, "provider", "")
        model = cloud_model or getattr(backend, "model", "") or metrics["model_name"]
        _emit_cloud_audit(
            config,
            user,
            question,
            metrics,
            chain_type,
            anon_question=anon_question,
            anon_answer=anon_answer,
            provider=provider,
            model_name=model,
            sources=sources,
            pseudonymizer_stats=pseudonymizer_stats,
            correlation_id=correlation_id,
        )


def _stream_epilogue(
    chat_id: int,
    usage: dict | None,
    trace_id: str | None = None,
    *,
    include_trace_id: bool = True,
) -> list[str]:
    """Erzeugt die abschliessenden SSE-Events (usage + done).

    Args:
        include_trace_id: Wenn True, wird trace_id (auch None) ins done-Event
                          aufgenommen. Wenn False, wird der Key weggelassen.

    Returns:
        Liste von SSE-Event-Strings (0-2 Elemente).
    """
    events = []
    if usage:
        events.append(_sse_event("usage", usage))
    done_data = {"chat_id": chat_id}
    if include_trace_id:
        done_data["trace_id"] = trace_id
    events.append(_sse_event("done", done_data))
    return events


@_with_processing_warnings
async def _stream_standard(
    config,
    data,
    actual_chat_id,
    user,
    template_content=None,
    history_block="",
    cloud_system_prompt=None,
):
    """Standard-RAG-Streaming (bestehende Logik)."""
    correlation_id = str(uuid.uuid4())
    t_start = time.monotonic()
    from kiara.backend_selector import BackendUnavailableError
    from kiara.retrieval.index_compatibility import IndexCompatibilityError

    try:
        chain, filter_datasource_ids, context_override = _prepare_standard_context(
            config,
            data,
            user,
            template_content,
            cloud_system_prompt=cloud_system_prompt,
        )

        generator, sources = chain.query_stream(
            question=data.question,
            k=data.k,
            num_ctx=data.num_ctx,
            filter_dir=data.filter_dir,
            filter_type=data.filter_type,
            filter_datasource_ids=filter_datasource_ids,
            history_block=history_block,
            think=data.think,
            context_override=context_override,
        )
    except IndexCompatibilityError as exc:
        logger.error("Index-Kompatibilitaetsguard fuer Standard-Chat: %s", exc)
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "standard",
            str(exc),
            [],
            t_start,
            correlation_id=correlation_id,
        )
        err, done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_standard_retrieval",
        )
        yield err
        yield done
        return
    except (BackendUnavailableError, RetrievalError, RuntimeError, ValueError) as exc:
        logger.error("Backend-Fehler fuer Chat-Anfrage: %s", exc)
        message = f"Backend-Fehler: {exc}"
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "standard",
            message,
            [],
            t_start,
            correlation_id=correlation_id,
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    yield _sse_event("mode", {"mode": "standard"})

    yield _sse_event("sources", sources)

    # Query-Expansion Info
    expansions = getattr(chain, "last_expansions", [])
    if len(expansions) > 1:
        yield _sse_event("expansion", {"variants": expansions[1:]})

    yield _sse_event("route", {"route": chain.route_used})

    # Stream answer chunks
    full_answer = []
    try:
        for chunk in generator:
            full_answer.append(chunk)
            yield _sse_event("chunk", {"text": chunk})
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "Standard-Stream fuer Chat %s abgebrochen; kein Epilog und keine Teilantwort-Persistierung",
            actual_chat_id,
        )
        raise
    except IndexCompatibilityError as exc:
        logger.error(
            "Index-Kompatibilitaetsguard im Standard-Stream: %s",
            exc,
            exc_info=True,
        )
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "standard",
            str(exc),
            full_answer,
            t_start,
            chain=chain,
            sources=sources,
            routed_to=chain.route_used or "local",
            full_prompt=chain.last_prompt,
            trace_id=_safe_trace_id(chain.last_trace_id),
            correlation_id=correlation_id,
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[STREAM_ERROR]",
            pseudonymizer_stats=getattr(getattr(chain, "backend", None), "anonymizer_stats", None),
        )
        err, done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_standard_stream",
        )
        yield err
        yield done
        return
    except Exception as exc:
        logger.exception("LLM-Fehler beim Streaming: %s", exc)
        security_log.error(
            "LLM_BACKEND_ERROR mode=standard chat_id=%s",
            actual_chat_id,
            exc_info=True,
        )
        message = "Ein Fehler ist aufgetreten. Bitte erneut versuchen."
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "standard",
            message,
            full_answer,
            t_start,
            chain=chain,
            sources=sources,
            routed_to=chain.route_used or "local",
            full_prompt=chain.last_prompt,
            trace_id=_safe_trace_id(chain.last_trace_id),
            correlation_id=correlation_id,
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[STREAM_ERROR]",
            pseudonymizer_stats=getattr(getattr(chain, "backend", None), "anonymizer_stats", None),
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    m = _collect_stream_metrics(full_answer, chain, t_start)

    if chain.route_used == "cloud":
        backend = chain.backend
        _emit_cloud_audit(
            config,
            user,
            data.question,
            m,
            "standard",
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[Anonymisierung nicht aktiv]",
            provider=backend.provider,
            model_name=backend.model,
            sources=sources,
            pseudonymizer_stats=getattr(backend, "anonymizer_stats", None),
            correlation_id=correlation_id,
        )

    _save_successful_assistant_message(actual_chat_id, m["answer_text"], sources)

    _emit_query_audit(
        user,
        data.question,
        m,
        "standard",
        actual_chat_id,
        routed_to=chain.route_used or "local",
        full_prompt=chain.last_prompt,
        sources=sources,
        trace_id=_safe_trace_id(chain.last_trace_id),
        correlation_id=correlation_id,
        request_settings=config.get("_chat_source_policy", {}),
    )

    for ev in _stream_epilogue(actual_chat_id, m["usage"], _safe_trace_id(chain.last_trace_id)):
        yield ev


@_with_processing_warnings
async def _stream_rewriter(
    config,
    data,
    actual_chat_id,
    user,
    template_content=None,
    history_block="",
    cloud_system_prompt=None,
):
    """Rewriter-RAG-Streaming: LLM-optimierter Suchbegriff + Standard-RAG."""
    correlation_id = str(uuid.uuid4())
    t_start = time.monotonic()
    from kiara.backend_selector import BackendUnavailableError
    from kiara.generation.cloud_routing import _create_chain_backend
    from kiara.generation.rewriter_chain import RewriterChain
    from kiara.retrieval.index_compatibility import IndexCompatibilityError

    try:
        filter_datasource_ids, restriction_datasource_ids = _build_filter_pair(user, data)
        context_override = _build_context_override(data.context_chunks)
        rag_backend = _create_chain_backend(
            config,
            question=data.question,
            llm_backend_id=data.llm_backend_id,
            filter_datasource_ids=restriction_datasource_ids,
        )
        chain = RewriterChain(
            config,
            rag_backend=rag_backend,
            embedding_model=data.embedding_model,
            prompt_template_override=template_content,
            system_prompt=((cloud_system_prompt or "") if _is_cloud_provider(rag_backend) else ""),
        )
    except IndexCompatibilityError as exc:
        logger.error("Index-Kompatibilitaetsguard fuer Rewriter-Initialisierung: %s", exc)
        err, done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_rewriter_initialization",
        )
        yield err
        yield done
        return
    except (BackendUnavailableError, RetrievalError, RuntimeError, ValueError) as exc:
        logger.error("Rewriter-Backend nicht verfuegbar: %s", exc)
        message = f"Backend-Fehler: {exc}"
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "rewriter",
            message,
            [],
            t_start,
            correlation_id=correlation_id,
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    try:
        generator, sources = chain.query_stream(
            question=data.question,
            k=data.k,
            num_ctx=data.num_ctx,
            filter_dir=data.filter_dir,
            filter_type=data.filter_type,
            filter_datasource_ids=filter_datasource_ids,
            history_block=history_block,
            think=data.think,
            context_override=context_override,
        )
    except IndexCompatibilityError as exc:
        logger.error("Index-Kompatibilitaetsguard fuer Rewriter-Retrieval: %s", exc)
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "rewriter",
            str(exc),
            [],
            t_start,
            chain=chain,
            routed_to=getattr(chain, "route_used", "local") or "local",
            full_prompt=getattr(chain, "last_prompt", "") or "",
            trace_id=_safe_trace_id(getattr(chain, "last_trace_id", None)),
            correlation_id=correlation_id,
        )
        err, done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_rewriter_retrieval",
        )
        yield err
        yield done
        return
    except (BackendUnavailableError, RetrievalError, RuntimeError, ValueError) as exc:
        logger.error("Embedding-Backend nicht erreichbar fuer Rewriter-Anfrage: %s", exc)
        message = f"Backend-Fehler: {exc}"
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "rewriter",
            message,
            [],
            t_start,
            chain=chain,
            routed_to=getattr(chain, "route_used", "local") or "local",
            full_prompt=getattr(chain, "last_prompt", "") or "",
            trace_id=_safe_trace_id(getattr(chain, "last_trace_id", None)),
            correlation_id=correlation_id,
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    yield _sse_event("mode", {"mode": "rewriter"})

    # Rewrite-Event (nur wenn tatsaechlich umgeschrieben)
    if chain.rewritten_query != data.question:
        yield _sse_event("rewrite", {"original": data.question, "rewritten": chain.rewritten_query})

    yield _sse_event("sources", sources)

    # Query-Expansion Info
    expansions = getattr(chain, "last_expansions", [])
    if len(expansions) > 1:
        yield _sse_event("expansion", {"variants": expansions[1:]})

    yield _sse_event("route", {"route": chain.route_used})

    # Stream answer chunks
    full_answer = []
    try:
        for chunk in generator:
            full_answer.append(chunk)
            yield _sse_event("chunk", {"text": chunk})
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "Rewriter-Stream fuer Chat %s abgebrochen; kein Epilog und keine Teilantwort-Persistierung",
            actual_chat_id,
        )
        raise
    except IndexCompatibilityError as exc:
        logger.error(
            "Index-Kompatibilitaetsguard im Rewriter-Stream: %s",
            exc,
            exc_info=True,
        )
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "rewriter",
            str(exc),
            full_answer,
            t_start,
            chain=chain,
            sources=sources,
            routed_to=chain.route_used or "local",
            full_prompt=chain.last_prompt,
            trace_id=_safe_trace_id(chain.last_trace_id),
            correlation_id=correlation_id,
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[STREAM_ERROR]",
            pseudonymizer_stats=getattr(getattr(chain, "backend", None), "anonymizer_stats", None),
        )
        err, done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_rewriter_stream",
        )
        yield err
        yield done
        return
    except Exception as exc:
        logger.exception("LLM-Fehler beim Rewriter-Streaming: %s", exc)
        security_log.error(
            "LLM_BACKEND_ERROR mode=rewriter chat_id=%s",
            actual_chat_id,
            exc_info=True,
        )
        message = "Ein Fehler ist aufgetreten. Bitte erneut versuchen."
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "rewriter",
            message,
            full_answer,
            t_start,
            chain=chain,
            sources=sources,
            routed_to=chain.route_used or "local",
            full_prompt=chain.last_prompt,
            trace_id=_safe_trace_id(chain.last_trace_id),
            correlation_id=correlation_id,
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[STREAM_ERROR]",
            pseudonymizer_stats=getattr(getattr(chain, "backend", None), "anonymizer_stats", None),
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    m = _collect_stream_metrics(full_answer, chain, t_start)

    if chain.route_used == "cloud":
        backend = chain.backend
        _emit_cloud_audit(
            config,
            user,
            data.question,
            m,
            "rewriter",
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[Anonymisierung nicht aktiv]",
            provider=backend.provider,
            model_name=backend.model,
            sources=sources,
            pseudonymizer_stats=getattr(backend, "anonymizer_stats", None),
            correlation_id=correlation_id,
        )

    _save_successful_assistant_message(actual_chat_id, m["answer_text"], sources)

    _emit_query_audit(
        user,
        data.question,
        m,
        "rewriter",
        actual_chat_id,
        routed_to=chain.route_used or "local",
        full_prompt=chain.last_prompt,
        sources=sources,
        trace_id=_safe_trace_id(chain.last_trace_id),
        correlation_id=correlation_id,
        request_settings=config.get("_chat_source_policy", {}),
    )

    for ev in _stream_epilogue(actual_chat_id, m["usage"], _safe_trace_id(chain.last_trace_id)):
        yield ev


async def _stream_unchained(
    config,
    data,
    actual_chat_id,
    user,
    template_content=None,
    history_block="",
    cloud_system_prompt=None,
):
    """Unchained-Streaming: LLM direkt, ohne RAG-Kontext."""
    correlation_id = str(uuid.uuid4())
    t_start = time.monotonic()
    from kiara.backend_selector import BackendUnavailableError
    from kiara.generation.backends import create_backend, wrap_with_anonymization
    from kiara.generation.unchained_chain import UnchainedChain

    try:
        # Backend erstellen (kein Routing bei Unchained — User waehlt explizit)
        backend = create_backend(config, data.llm_backend_id)
        if _is_cloud_provider(backend) and data.llm_backend_id is not None:
            from kiara.backend_selector import get_backend_by_id

            backend_dict = get_backend_by_id(data.llm_backend_id)
            trusted = bool(backend_dict.get("trusted", 0)) if backend_dict else False
            if not trusted:
                backend = wrap_with_anonymization(backend, config, restriction_levels=None)
        elif _is_cloud_provider(backend):
            backend = wrap_with_anonymization(backend, config, restriction_levels=None)

        # System-Prompt: explizites Template hat Vorrang
        effective_system_prompt = (
            cloud_system_prompt if cloud_system_prompt is not None else template_content
        )
        chain = UnchainedChain(
            config,
            backend=backend,
            system_prompt=effective_system_prompt,
        )
    except (BackendUnavailableError, RuntimeError, ValueError) as exc:
        logger.error("Backend nicht verfuegbar fuer Unchained-Anfrage: %s", exc)
        message = f"Backend nicht verfuegbar: {exc}"
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "unchained",
            message,
            [],
            t_start,
            correlation_id=correlation_id,
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    yield _sse_event("mode", {"mode": "unchained"})

    full_answer = []
    try:
        for chunk in chain.query_stream(
            data.question,
            num_ctx=data.num_ctx,
            history_block=history_block,
            think=data.think,
            temperature=data.temperature,
        ):
            full_answer.append(chunk)
            yield _sse_event("chunk", {"text": chunk})
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "Unchained-Stream fuer Chat %s abgebrochen; kein Epilog und keine Teilantwort-Persistierung",
            actual_chat_id,
        )
        raise
    except Exception as exc:
        logger.exception("LLM-Fehler beim Streaming: %s", exc)
        security_log.error(
            "LLM_BACKEND_ERROR mode=unchained chat_id=%s",
            actual_chat_id,
            exc_info=True,
        )
        message = "Ein Fehler ist aufgetreten. Bitte erneut versuchen."
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "unchained",
            message,
            full_answer,
            t_start,
            chain=chain,
            routed_to=chain.route_used,
            full_prompt=chain.last_prompt,
            correlation_id=correlation_id,
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[STREAM_ERROR]",
            pseudonymizer_stats=getattr(getattr(chain, "backend", None), "anonymizer_stats", None),
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return

    m = _collect_stream_metrics(full_answer, chain, t_start)

    if chain.route_used == "cloud":
        backend = chain.backend
        _emit_cloud_audit(
            config,
            user,
            data.question,
            m,
            "unchained",
            anon_question=chain.last_anon_question or "[Anonymisierung nicht aktiv]",
            anon_answer=chain.last_anon_answer or "[Anonymisierung nicht aktiv]",
            provider=backend.provider,
            model_name=backend.model,
            pseudonymizer_stats=getattr(backend, "anonymizer_stats", None),
            correlation_id=correlation_id,
        )

    _save_successful_assistant_message(actual_chat_id, m["answer_text"])

    _emit_query_audit(
        user,
        data.question,
        m,
        "unchained",
        actual_chat_id,
        routed_to=chain.route_used,
        full_prompt=chain.last_prompt,
        correlation_id=correlation_id,
        request_settings=config.get("_chat_source_policy", {}),
    )

    for ev in _stream_epilogue(actual_chat_id, m["usage"]):
        yield ev


def _create_agent_anonymizer(
    config: dict,
    backend: dict,
    restriction_levels: list[str] | None = None,
):
    """Erstellt Anonymizer gemaess globalem Schalter und Quellenregeln.

    Gibt None zurueck bei:
    - Ollama-Backends (lokales LLM, keine PII-Gefahr)
    - trusted=True Backends
    - Global deaktivierter Anonymisierung (auch bei anonymize-Quellen)
    - Ausschliesslich public-Quellen

    Ohne Quellenkontext (None oder []) gilt die globale Einstellung.

    Raises:
        ValueError: Bei blocked/unknown in restriction_levels.
        RuntimeError: Bei Init-Fehler wenn anonymize erzwungen wird.
    """
    if not _cloud_requires_agent_anonymizer(config, backend, restriction_levels):
        logger.info("Agent-Backend ohne Anonymisierung (Backend/globale Einstellung/Quellenregeln)")
        return None
    try:
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.llm_detector import build_llm_detector
        from kiara.anonymization.stammdaten import build_automaton_from_sources
        from kiara.generation.backends.factory import _normalize_config_for_anonymizer

        normalized_config = _normalize_config_for_anonymizer(config)
        automaton = build_automaton_from_sources(normalized_config)
        if automaton is None or automaton.entry_count == 0:
            raise RuntimeError("Anonymisierung nicht verfuegbar (keine Stammdaten geladen)")
        llm_detector = build_llm_detector(normalized_config)
        return Anonymizer(automaton, normalized_config, llm_detector=llm_detector)
    except Exception as exc:
        if not restriction_levels:
            raise RuntimeError(
                f"Anonymisierung ist aktiviert (anonymization.enabled=True, ggf. Default) "
                f"aber konnte nicht initialisiert werden: {exc}"
            ) from exc
        raise RuntimeError(
            f"Agent-Anonymisierung fuer explizites anonymize-Level konnte nicht "
            f"initialisiert werden: {exc}"
        ) from exc


def _cloud_requires_agent_anonymizer(
    config: dict,
    backend: dict,
    restriction_levels: list[str] | None = None,
) -> bool:
    """Policy-Flag fuer Cloud-Adapter: wann fehlender Anonymizer hart fehlschlaegt."""
    if backend.get("provider", "ollama") == "ollama":
        return False
    if bool(backend.get("trusted", 0)):
        return False
    from kiara.generation._validators import needs_anonymization

    return needs_anonymization(config, restriction_levels)


def _build_agent_tool_registry(config, data, allowed_ds):
    """Erstellt die Tool-Registry fuer den Agent.

    Args:
        config: Globale Config.
        data: QueryRequest (Embedding-Modell, k, Filter).
        allowed_ds: Bereits permission-gefilterte Datasource-IDs oder None.

    Returns:
        ToolRegistry mit allen erlaubten Agent-Tools.

    Raises:
        BackendUnavailableError: Wenn das Embedding-Backend nicht erreichbar ist.
    """
    from kiara.retrieval.reranker import create_reranker
    from kiara.retrieval.retriever import HybridRetriever
    from kiara.tools.builtin_tools import CalculateTool, ListDatasourcesTool, SearchDocumentsTool
    from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool
    from kiara.tools.registry import ToolRegistry, load_config_tool_permissions

    retriever = HybridRetriever(config, embedding_model=data.embedding_model)
    reranker = create_reranker(config)

    registry = ToolRegistry(permission_provider=load_config_tool_permissions)
    references = DocumentReferences(str(retriever._store.get_collection_name()))
    registry.register(
        SearchDocumentsTool(
            retriever,
            reranker,
            config,
            allowed_datasource_ids=allowed_ds,
            default_k=data.k,
            default_filter_type=data.filter_type,
            default_filter_dir=data.filter_dir,
            document_references=references,
        )
    )
    registry.register(ReadDocumentTool(retriever, config, references, allowed_ds))
    registry.register(ListDatasourcesTool(allowed_datasource_ids=allowed_ds))
    registry.register(CalculateTool())

    # ERP-Tools registrieren (falls Feature verfuegbar und Verbindungen konfiguriert)
    tools_cfg = config.get("tools", {})
    if not tools_cfg.get("enabled", False):
        return registry

    from kiara.web.api_health import _get_degradation_manager

    dm = _get_degradation_manager()
    if dm.is_feature_available("erp_tools"):
        try:
            from kiara.tools.erp_backend import ERPBackend
            from kiara.tools.erp_connectors import ConnectionRegistry
            from kiara.tools.erp_tools import (
                ERPCustomerInfoTool,
                ERPOrderStatusTool,
                ERPOrdersTool,
                ERPProductionStatusTool,
                ERPStockLevelTool,
            )

            conn_registry = ConnectionRegistry()
            conn_registry.load_from_config(config)
            backend = ERPBackend(config, connection_registry=conn_registry)

            registry.register(ERPOrderStatusTool(backend))
            registry.register(ERPOrdersTool(backend))
            registry.register(ERPStockLevelTool(backend))
            registry.register(ERPCustomerInfoTool(backend))
            registry.register(ERPProductionStatusTool(backend))
        except Exception as exc:
            logger.warning(
                "ERP-Tools konnten nicht fuer Agent registriert werden: %s",
                exc,
                exc_info=True,
            )

    return registry


def _prepare_agent_context(config, data, user, template_content, final_prompt):
    """Bereitet Backend, Retriever, Tools, Adapter und Chain fuer den Agent vor."""
    from kiara.generation.agent_chain import AgentChain
    from kiara.generation.cloud_routing import _resolve_backend_dict
    from kiara.generation.tool_adapter import create_tool_adapter
    from kiara.shared.constants import (
        AGENT_MAX_ITERATIONS_CLOUD,
        AGENT_MAX_ITERATIONS_OLLAMA,
    )

    allowed_ds, restriction_ds = _build_filter_pair(user, data)

    backend = _resolve_backend_dict(
        config,
        llm_backend_id=data.llm_backend_id,
        filter_datasource_ids=restriction_ds,
    )
    registry = _build_agent_tool_registry(config, data, allowed_ds)

    # Restriction-Levels fuer Agent-Anonymizer ermitteln
    trusted = bool(backend.get("trusted", 0))
    restriction_levels = None
    if _is_cloud_provider(backend) and not trusted:
        from kiara.generation.cloud_routing import _load_restriction_levels

        restriction_levels = _load_restriction_levels(restriction_ds)

    anonymizer = _create_agent_anonymizer(config, backend, restriction_levels=restriction_levels)
    adapter_config = dict(config)
    adapter_config["_kiara_cloud_requires_anonymizer"] = _cloud_requires_agent_anonymizer(
        config,
        backend,
        restriction_levels=restriction_levels,
    )
    adapter = create_tool_adapter(backend, adapter_config, anonymizer)

    if data.num_ctx and hasattr(adapter, "_num_ctx"):
        adapter._num_ctx = data.num_ctx
    if final_prompt and hasattr(adapter, "_final_prompt_override"):
        adapter._final_prompt_override = final_prompt

    max_iter = (
        AGENT_MAX_ITERATIONS_CLOUD if _is_cloud_provider(backend) else AGENT_MAX_ITERATIONS_OLLAMA
    )

    # Admin-User bekommen Wildcard-Zugriff auf alle Tools
    if user.get("is_admin"):
        user_groups = ["*"]
    elif user.get("group_ids"):
        with db_query() as conn:
            placeholders = ",".join("%s" for _ in user["group_ids"])
            rows = conn.execute(
                f"SELECT cn, dn, name FROM groups WHERE id IN ({placeholders})",
                user["group_ids"],
            ).fetchall()
            user_groups = []
            for row in rows:
                for value in (row["dn"], row["cn"], row["name"]):
                    if value and value not in user_groups:
                        user_groups.append(value)
    else:
        user_groups = []

    chain = AgentChain(
        config,
        registry,
        adapter=adapter,
        user_id=user["id"],
        max_iterations=max_iter,
        max_same_tool_calls=data.max_tool_repeats,
        system_prompt_override=template_content,
        user_groups=user_groups,
    )
    return chain, backend, anonymizer


@_with_processing_warnings
async def _stream_agent(
    config, data, actual_chat_id, user, template_content=None, history_block="", final_prompt=None
):
    """Agent-Streaming: Backend-agnostischer ReAct-Loop mit Tool-Use."""
    correlation_id = str(uuid.uuid4())
    t_start = time.monotonic()
    from kiara.backend_selector import BackendUnavailableError
    from kiara.retrieval.index_compatibility import IndexCompatibilityError

    try:
        chain, backend, anonymizer = _prepare_agent_context(
            config,
            data,
            user,
            template_content,
            final_prompt,
        )
    except IndexCompatibilityError as exc:
        logger.error("Index-Kompatibilitaetsguard fuer Agent-Initialisierung: %s", exc)
        err, done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_agent_initialization",
        )
        yield err
        yield done
        return
    except BackendUnavailableError as exc:
        logger.error("Embedding-Backend nicht erreichbar fuer Agent-Anfrage: %s", exc)
        message = f"Embedding-Backend nicht erreichbar: {exc}"
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "agent",
            message,
            [],
            t_start,
            correlation_id=correlation_id,
        )
        err, done = _sse_error_done(actual_chat_id, message)
        yield err
        yield done
        return
    except (RetrievalError, RuntimeError, ValueError) as exc:
        logger.error("Backend nicht verfuegbar fuer Agent-Anfrage: %s", exc)
        message = f"Backend nicht verfuegbar: {exc}"
        _audit_stream_failure(
            config,
            user,
            data.question,
            actual_chat_id,
            "agent",
            message,
            [],
            t_start,
            correlation_id=correlation_id,
        )
        err, done = _sse_error_done(
            actual_chat_id,
            message,
        )
        yield err
        yield done
        return

    yield _sse_event("mode", {"mode": "agent"})

    full_answer = []
    all_sources = []
    had_error = False
    stream_error_message = ""
    guard_done_event = None

    try:
        for event in chain.query_stream(
            data.question, history_block=history_block, think=data.think
        ):
            if event.event_type == "tool_call":
                yield _sse_event("tool_call", event.data)
            elif event.event_type == "tool_result":
                yield _sse_event("tool_result", event.data)
            elif event.event_type == "sources":
                all_sources = event.data or []
                yield _sse_event("sources", all_sources)
            elif event.event_type == "chunk":
                text = event.data.get("text", "") if isinstance(event.data, dict) else ""
                full_answer.append(text)
                yield _sse_event("chunk", {"text": text})
            elif event.event_type == "error":
                had_error = True
                if isinstance(event.data, dict):
                    stream_error_message = str(event.data.get("message") or "")
                # Adapter-Fehler (z.B. Ollama 503, Anthropic Rate-Limit) werden
                # vom _step_with_retry() als ToolAdapterError gefangen und als
                # weicher AgentEvent("error", ...) durchgereicht — kein Python-
                # Exception bubbelt zum aeusseren Catch, der security_log triggern
                # wuerde. Daher hier explizit als LLM_BACKEND_ERROR loggen, damit
                # der Vorfall im System-Log-Tab sichtbar ist.
                security_log.error(
                    "LLM_BACKEND_ERROR mode=agent chat_id=%s soft=true",
                    actual_chat_id,
                    extra={
                        "kiara_details": {
                            "error_message": stream_error_message[:1000],
                        }
                    },
                )
                yield _sse_event("error", event.data or {})
            elif event.event_type == "done":
                pass  # done wird unten gesendet
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "Agent-Stream fuer Chat %s abgebrochen; kein Epilog und keine Teilantwort-Persistierung",
            actual_chat_id,
        )
        raise
    except IndexCompatibilityError as exc:
        had_error = True
        stream_error_message = str(exc)
        logger.error("Index-Kompatibilitaetsguard im Agent-Loop: %s", exc, exc_info=True)
        err, _done = _sse_index_compatibility_error_done(
            actual_chat_id,
            exc,
            consumer="chat_agent_tool_retrieval",
        )
        guard_done_event = _done
        yield err
    except Exception as exc:
        had_error = True
        stream_error_message = "Ein Fehler ist aufgetreten. Bitte erneut versuchen."
        logger.exception("Agent-Loop Fehler: %s", exc)
        security_log.error(
            "LLM_BACKEND_ERROR mode=agent chat_id=%s",
            actual_chat_id,
            exc_info=True,
        )
        yield _sse_event("error", {"message": stream_error_message})

    anon_stats_count = 0
    if anonymizer:
        anon_stats_count = getattr(anonymizer, "stats", {}).get("total_entities", 0)

    if had_error:
        m = _collect_stream_error_metrics(
            full_answer,
            chain,
            t_start,
            stream_error_message or "Agent-Stream fehlgeschlagen.",
            model_name_override=backend.get("model", "") if backend else "",
            entities_count_override=anon_stats_count,
        )
    else:
        m = _collect_stream_metrics(
            full_answer,
            chain,
            t_start,
            model_name_override=backend.get("model", "") if backend else "",
            entities_count_override=anon_stats_count,
        )
    answer_text = m["answer_text"]
    route = "cloud" if _is_cloud_provider(backend) else "local"

    # Genutzte Tools sammeln
    _tevts = getattr(chain, "_tool_events", None)
    tools_used = ",".join(dict.fromkeys(ev.get("tool", "") for ev in _tevts)) if _tevts else ""

    tool_calls_json = ""
    if _tevts:
        try:
            tool_calls_json = json.dumps(_tevts, ensure_ascii=False)
        except (TypeError, ValueError):
            logger.warning("Tool-Events JSON-Serialisierung fehlgeschlagen")

    if route == "cloud":
        anon_q = "[Anonymisierung nicht aktiv]"
        anon_a = "[Anonymisierung nicht aktiv]"
        if anonymizer:
            try:
                anon_q = anonymizer.anonymize(data.question)[:200]
                anon_a = anonymizer.anonymize(answer_text) if answer_text else "[leere Antwort]"
            except Exception as exc:
                logger.warning(
                    "Anonymisierung fuer Cloud-Audit fehlgeschlagen: %s",
                    exc,
                    exc_info=True,
                )
                anon_q = "[Anonymisierung fehlgeschlagen]"
                anon_a = "[Anonymisierung fehlgeschlagen]"
        _emit_cloud_audit(
            config,
            user,
            data.question,
            m,
            "agent",
            anon_question=anon_q,
            anon_answer=anon_a,
            provider=backend.get("provider", ""),
            model_name=m["model_name"],
            sources=all_sources,
            pseudonymizer_stats=anonymizer.stats if anonymizer else None,
            correlation_id=correlation_id,
        )

    if not had_error:
        _save_assistant_message(actual_chat_id, answer_text, all_sources or None)

    _emit_query_audit(
        user,
        data.question,
        m,
        "agent",
        actual_chat_id,
        routed_to=route,
        full_prompt=chain.last_conversation,
        sources=all_sources,
        trace_id=None if had_error else _safe_trace_id(chain.last_trace_id),
        tools_used=tools_used,
        tool_calls_json=tool_calls_json,
        correlation_id=correlation_id,
        request_settings=config.get("_chat_source_policy", {}),
    )

    if guard_done_event is not None:
        yield guard_done_event
        return

    for ev in _stream_epilogue(
        actual_chat_id,
        m["usage"],
        trace_id=_safe_trace_id(chain.last_trace_id),
        include_trace_id=not had_error,
    ):
        yield ev


@router.put("/chat/{chat_id}/archive")
async def archive_chat(chat_id: int, user=Depends(get_chat_user)):
    """Archive a chat."""
    _ensure_chat_owner(chat_id, user["id"])
    chat_repo.archive(chat_id)
    return {"message": "Chat archiviert"}


@router.put("/chat/{chat_id}/restore")
async def restore_chat(chat_id: int, user=Depends(get_chat_user)):
    """Restore an archived chat."""
    _ensure_chat_owner(chat_id, user["id"])
    chat_repo.restore(chat_id)
    return {"message": "Chat wiederhergestellt"}


# ---------------------------------------------------------------------------
# Chat-Debug-Endpoints (S7)
# ---------------------------------------------------------------------------


@router.get("/chat-debug/status")
async def chat_debug_status(user: dict = Depends(get_chat_user)):
    """Gibt zurueck ob Chat-Debug aktiv ist (Config-gesteuert)."""
    config = load_config()
    enabled = config.get("tracing", {}).get("chat_debug", False)
    return {
        "enabled": enabled,
        "admin_port": config.get("web", {}).get("admin_port", ADMIN_PORT_DEFAULT),
    }


class ClientErrorReport(BaseModel):
    """Frontend-Beacon-Payload fuer Stream-Abrisse, die der Server nicht sieht.

    Wird vom Browser geschickt, wenn ein SSE-Stream abreisst, ohne dass eine
    strukturierte Error-Antwort vom Backend kam (Worker-Tod mid-stream,
    Connection-Reset, Proxy-Timeout, Frontend-AbortController-Fehler).
    """

    chat_id: str | int | None = None
    # Allowlist statt freier String — verhindert KV-Injection ueber das
    # Beacon-Payload, falls untrusted Werte spaeter doch ins Message-KV-
    # Format wandern. Audit-Anforderung 0.0.1+447-Audit-Run.
    error_kind: Literal["network", "http", "unauthorized", "unknown"] = "unknown"
    error_message: str = Field("", max_length=2000)
    mode: Literal["agent", "standard", "rewriter", "unchained", "unknown"] = "unknown"


@router.post("/chat-debug/client-error")
async def chat_debug_client_error(data: ClientErrorReport, user: dict = Depends(get_chat_user)):
    """Frontend-Beacon-Endpunkt fuer Stream-Abrisse ohne Server-Sicht.

    Schreibt einen `CHAT_CLIENT_ERROR`-Event in `system_events`. Mit dem
    fail-safe security_log-Pfad: bei DB-Ausfall wird der Request nicht
    blockiert, der Beacon ist zur Diagnose, nicht zur User-Funktion.

    Sicherheitsvertrag: untrusted Beacon-Felder (mode, kind, error_message,
    chat_id) werden NICHT ins KV-Message-Format gemerged, weil der
    SystemEventDBHandler dort `key=value`-Paare parst und ins indexierte
    `reason`-Feld extrahiert. Sie wandern ausschliesslich in `kiara_details`,
    wo der Handler sie via `setdefault` einbettet, ohne die Top-Level-
    Indexspalten zu verfaelschen.
    """
    chat_id_str = str(data.chat_id)[:64] if data.chat_id is not None else "unknown"
    security_log.error(
        "CHAT_CLIENT_ERROR user=%s",
        user.get("username", "?"),
        extra={
            "kiara_details": {
                "mode": data.mode,
                "kind": data.error_kind,
                "chat_id": chat_id_str,
                "error_message": data.error_message[:1000],
            }
        },
    )
    return {"ok": True}


@router.get("/chat-debug/trace/{trace_id}")
async def chat_debug_trace(trace_id: str, user: dict = Depends(get_chat_user)):
    """Trace-Daten fuer Chat-Debug-Bar. Nur fuer eigene Traces des Users."""
    config = load_config()
    if not config.get("tracing", {}).get("chat_debug", False):
        raise HTTPException(status_code=404, detail="Chat-Debug nicht aktiviert")

    with db_query() as conn:
        owner = conn.execute(
            "SELECT 1 FROM query_log WHERE trace_id = %s AND user_id = %s LIMIT 1",
            (trace_id, user["id"]),
        ).fetchone()
    if not owner:
        raise HTTPException(status_code=404, detail="Trace nicht gefunden")

    from kiara.web.api_tracing import _get_trace_by_id

    result = _get_trace_by_id(trace_id)
    if not result:
        raise HTTPException(status_code=404, detail="Trace nicht gefunden")
    return result
