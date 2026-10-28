import asyncio
import json
import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from kiara_db.repositories.admin_chat import mask_wizard_secret_params
from pydantic import BaseModel, Field

from kiara.repositories.admin_chat import (
    AdminChatMessageRepository,
    AdminChatRepository,
    AdminWizardRepository,
)
from kiara.shared.constants import DEFAULT_CLOUD_MODEL, DEFAULT_OPENAI_MODEL
from kiara.shared.db import db_transaction
from kiara.web.auth import get_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin-chat", tags=["admin-chat"])

# Metadata-Typen fuer Setup-Messages (aus LLM-History gefiltert)
_SETUP_META_TYPES = {"setup_wizard", "setup_response", "setup_input", "greeting"}


class AdminChatQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    llm_backend_id: int | None = None
    mode: str = "agent"
    use_history: bool = True
    think: bool = False


class AdminChatUpdateRequest(BaseModel):
    title: str | None = Field(None, max_length=200)
    status: Literal["active", "archived"] | None = None


# --- Chat-Listen & CRUD ---


@router.get("/chats")
async def list_chats(
    status: str | None = None,
    user: dict = Depends(get_admin_user),
):
    """Alle Chats des Users, optional nach Status gefiltert."""
    chat_repo = AdminChatRepository()
    return chat_repo.list_by_user(user["id"], status=status)


@router.post("/chats", status_code=201)
async def create_chat(request: Request, user: dict = Depends(get_admin_user)):
    """Neuen Chat erstellen mit Begruessungsnachricht.

    Optionaler Body ``{"greeting": "...", "greeting_type": "..."}`` — uebergibt
    die im Draft-Modus bereits angezeigte Begruessung. Der Server prueft ob der
    greeting_type zum aktuellen Backend-Status passt (F44 Race-Schutz).
    """
    body_greeting = None
    body_type = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            body_greeting = body.get("greeting")
            body_type = body.get("greeting_type")
    except Exception:
        pass

    computed_greeting, meta_type = _build_chat_greeting(user)
    # Nur Client-Greeting verwenden wenn der Typ zum Server-Modus passt (F44)
    greeting_text = body_greeting if body_greeting and body_type == meta_type else computed_greeting

    chat_repo = AdminChatRepository()
    msg_repo = AdminChatMessageRepository()
    chat_id = chat_repo.create(user_id=user["id"])

    msg_repo.add_message(
        chat_id,
        "assistant",
        greeting_text,
        metadata={"type": meta_type},
    )

    chat = chat_repo.get_by_id(chat_id)
    chat["messages"] = msg_repo.list_by_chat(chat_id)
    return chat


def _has_active_llm_backend() -> bool:
    """True wenn mindestens ein aktives LLM- oder Cloud-Backend existiert.

    WICHTIG: Der Setup-Wizard legt neue Backends zunaechst mit is_active=0 an
    und aktiviert sie erst nach erfolgreichem Test/Modellwahl. Dadurch reicht
    active_only=True, ohne dass der Wizard sich selbst vorzeitig beendet.
    """
    try:
        from kiara.repositories.backend import KiBackendRepository

        repo = KiBackendRepository()
        for category in ("llm", "cloud"):
            if repo.list_by_category(category, active_only=True):
                return True
    except Exception:
        logger.warning("Backend-Erkennung fehlgeschlagen, Fallback auf Setup-Modus")
    return False


def _setup_greeting_text() -> str:
    """Statischer Greeting-Text fuer den Ersteinrichtungs-Modus (kein LLM noetig)."""
    return (
        "Willkommen bei KIara! Es ist noch kein KI-Backend konfiguriert.\n\n"
        "Bitte waehlen Sie eine Option:\n\n"
        "  [1] **Cloud-Backend** einrichten\n"
        "      `cloud anthropic` oder `cloud openai`\n\n"
        "  [2] **Lokales Backend** (Ollama) einrichten\n"
        "      `lokal http://<host>:11434`\n\n"
        "  [3] **Hilfe** anzeigen\n"
        "      `hilfe`\n\n"
        "Geben Sie den gewuenschten Befehl ein."
    )


def _help_text() -> str:
    """Ausfuehrlicher Hilfe-Text fuer den Setup-Modus."""
    return (
        "## Einrichtungshilfe\n\n"
        "KIara benoetigt ein KI-Backend um Fragen beantworten zu koennen.\n\n"
        "### Cloud-Backend (Anthropic / OpenAI)\n"
        "1. `cloud anthropic` oder `cloud openai` eingeben\n"
        "2. API-Key eingeben (wird maskiert gespeichert)\n"
        "3. Backend wird automatisch getestet und aktiviert\n\n"
        "### Lokales Backend (Ollama)\n"
        "1. `lokal http://<host>:11434` eingeben (URL Ihres Ollama-Servers)\n"
        "2. Server wird getestet, verfuegbare Modelle aufgelistet\n"
        "3. Modell auswaehlen (Nummer oder Name)\n\n"
        "### Weitere Befehle\n"
        "- `abbrechen` — Einrichtung abbrechen und zuruecksetzen\n"
        "- `hilfe` — Diese Hilfe erneut anzeigen\n\n"
        "**Tipp**: Fuer eine schluesselfertige lokale Loesung siehe "
        "[Kiron](https://wiki.2ia.de/kiron) — die KIara-Appliance mit "
        "vorinstalliertem Ollama."
    )


_kpi_cache: dict = {}
_kpi_cache_ts: float = 0.0
_KPI_CACHE_TTL: float = 60.0


def _get_cached_kpis() -> dict:
    """KPIs mit 60s TTL-Cache (vermeidet DB-Queries bei jedem /greeting)."""
    global _kpi_cache, _kpi_cache_ts
    now = time.monotonic()
    if now - _kpi_cache_ts > _KPI_CACHE_TTL or not _kpi_cache:
        from kiara.config import load_config
        from kiara.web.api_dashboard import get_dashboard_kpis

        config = load_config()
        _kpi_cache = get_dashboard_kpis(config)
        _kpi_cache_ts = now
    return _kpi_cache


def _build_chat_greeting(user: dict) -> tuple[str, str]:
    """Liefert (greeting_text, meta_type)."""
    if _has_active_llm_backend():
        kpis = _get_cached_kpis()
        username = user.get("vorname") or user.get("username", "Admin")
        return _format_greeting(username, kpis), "greeting"
    return _setup_greeting_text(), "setup_wizard"


def _format_greeting(username: str, kpis: dict) -> str:
    """Formatiert die Begruessungsnachricht aus KPI-Daten.

    KPI-Dict-Struktur (von get_dashboard_kpis):
      backends:    {total, active, by_category}
      datasources: {total, active, by_type}
      vectordb:    {status, size_bytes, chunks_by_model: [{model, chunks}], error}
      jobs:        {running: None|{id,status,...}, pending: int}
    """
    b = kpis.get("backends", {})
    ds = kpis.get("datasources", {})
    vdb = kpis.get("vectordb", {})
    jobs = kpis.get("jobs", {})

    # VektorDB: Chunks aus chunks_by_model summieren
    total_chunks = sum(m.get("chunks", 0) for m in vdb.get("chunks_by_model", []))
    # Jobs: running ist None oder Dict, nicht int
    running_job = jobs.get("running")
    jobs_info = (
        f"1 laufend ({running_job.get('datasource', '?')})" if running_job else "keine laufend"
    )

    return (
        f"Hallo {username}! Hier ist der aktuelle Systemstatus:\n\n"
        f"+-- KI-Backends:  {b.get('active', 0)}/{b.get('total', 0)} aktiv\n"
        f"|-- Datenquellen: {ds.get('active', 0)}/{ds.get('total', 0)} aktiv\n"
        f"|-- VektorDB:     {total_chunks} Chunks\n"
        f"|-- Jobs:         {jobs_info}\n"
        f"+-- System:       OK\n\n"
        f"Wie kann ich helfen?"
    )


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: int, user: dict = Depends(get_admin_user)):
    """Chat mit allen Nachrichten, pending Confirmations und aktivem Wizard laden."""
    chat_repo = AdminChatRepository()
    chat = chat_repo.get_owned(chat_id, user["id"])
    if not chat:
        raise HTTPException(404, "Chat nicht gefunden")
    msg_repo = AdminChatMessageRepository()
    chat["messages"] = msg_repo.list_by_chat(chat_id)
    from kiara.admin.confirmation import ConfirmationManager

    chat["pending_confirmations"] = ConfirmationManager().get_all_pending(chat_id)
    active_wizard = AdminWizardRepository().get_active(chat_id)
    if active_wizard:
        collected = {}
        try:
            raw = json.loads(active_wizard.get("collected_params") or "{}")
            if isinstance(raw, dict):
                collected = mask_wizard_secret_params(raw)
                active_wizard["collected_params"] = json.dumps(
                    collected,
                    ensure_ascii=False,
                )
        except (ValueError, TypeError):
            pass
        try:
            from kiara.admin.registry import admin_registry, load_admin_functions

            load_admin_functions()
            func = admin_registry.get(active_wizard.get("function_name"))
            schema = (func.parameters if func else {}) or {}
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            active_wizard["collected"] = collected
            active_wizard["missing_required"] = [key for key in required if key not in collected]
            active_wizard["optional"] = [
                key for key in properties if key not in required and key not in collected
            ]
            active_wizard["parameter_help"] = {
                key: {
                    "type": value.get("type", "string"),
                    "description": value.get("description", ""),
                }
                for key, value in properties.items()
            }
        except Exception:
            active_wizard["collected"] = collected
            active_wizard["missing_required"] = []
            active_wizard["optional"] = []
            active_wizard["parameter_help"] = {}
    chat["active_wizard"] = active_wizard
    return chat


@router.put("/chats/{chat_id}")
async def update_chat(
    chat_id: int,
    data: AdminChatUpdateRequest,
    user: dict = Depends(get_admin_user),
):
    """Chat aktualisieren (Titel und/oder Status)."""
    chat_repo = AdminChatRepository()
    chat = chat_repo.get_owned(chat_id, user["id"])
    if not chat:
        raise HTTPException(404, "Chat nicht gefunden")
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Keine Aenderungen angegeben")
    chat_repo.update(chat_id, **updates)
    chat_repo.touch(chat_id)
    return chat_repo.get_by_id(chat_id)


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: int, user: dict = Depends(get_admin_user)):
    """Chat loeschen (CASCADE loescht Messages automatisch)."""
    chat_repo = AdminChatRepository()
    chat = chat_repo.get_owned(chat_id, user["id"])
    if not chat:
        raise HTTPException(404, "Chat nicht gefunden")
    chat_repo.delete(chat_id)


# --- SSE-Streaming Query ---


@router.post("/chats/{chat_id}/query")
async def admin_chat_query(
    chat_id: int,
    data: AdminChatQueryRequest,
    user: dict = Depends(get_admin_user),
):
    chat_repo = AdminChatRepository()
    chat = chat_repo.get_owned(chat_id, user["id"])
    if not chat:
        raise HTTPException(404, "Chat nicht gefunden")
    if chat.get("status") == "archived":
        raise HTTPException(409, "Chat ist archiviert")

    # Setup-Branch: kein aktives LLM-Backend → regelbasierter Handler (kein LLM)
    # WICHTIG: _handle_setup_mode ist async — muss hier im async Endpoint
    # aufgerufen werden, NICHT im sync SSE-Generator (_stream_admin_query).
    if not _has_active_llm_backend():
        response, mask = await _handle_setup_mode(chat_id, data.question, user)
        return StreamingResponse(
            _setup_sse_generator(chat_id, data.question, response, mask_input=mask),
            media_type="text/event-stream",
        )

    # WICHTIG: User-Nachricht wird NICHT hier gespeichert, sondern im Generator
    # NACH dem History-Loading — sonst erscheint die Frage doppelt im Kontext
    return StreamingResponse(
        _stream_admin_query(chat_id, data, user),
        media_type="text/event-stream",
    )


def _stream_admin_query(chat_id: int, data: AdminChatQueryRequest, user: dict):
    """Generator der SSE-Events yieldet. Kern der Streaming-Logik.

    Analog zu api_chat.py _stream_agent(): Gesamter Body in try/except,
    Post-Processing (Save, Usage, Done) in finally-Block.

    1. History laden (optional) — VOR User-Nachricht-Speicherung
    2. User-Nachricht in DB speichern
    3. Backend aufloesen
    4. AdminAgentChain erstellen
    5. Agent-Loop ausfuehren, Events als SSE weiterleiten
    6. Assistant-Nachricht in DB speichern (finally)
    """
    from kiara.config import load_config
    from kiara.generation.admin_agent_chain import AdminAgentChain
    from kiara.retrieval.index_compatibility import IndexCompatibilityError
    from kiara.web.api_chat import _strip_think_blocks

    config = load_config()
    msg_repo = AdminChatMessageRepository()
    chat_repo = AdminChatRepository()

    chain = None
    full_answer = []
    sources = []
    tool_events = []
    _early_done = False  # Flag: done-Event wurde bereits im try-Block gesendet
    _error_sent = False  # Flag: error-Event wurde bereits gesendet (kein Duplikat im finally)
    _guard_done_event = None
    # Cloud-Audit-Kontext (im finally ausgewertet) — Issue #1904
    backend = None
    anonymizer = None
    t_start = time.monotonic()
    correlation_id = str(uuid.uuid4())

    try:
        # 1. Chat-History laden ZUERST (vor User-Nachricht-Speicherung,
        #    damit die aktuelle Frage nicht doppelt im Kontext erscheint)
        history_block = ""
        if data.use_history:
            history = msg_repo.load_history(chat_id)
            if history:
                lines = []
                for msg in history:
                    # Setup-Messages und maskierte API-Key-Eingaben filtern (F40)
                    if msg["content"] == "(API-Key eingegeben)":
                        continue
                    meta_raw = msg.get("metadata")
                    if meta_raw:
                        try:
                            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                        if meta.get("type") in _SETUP_META_TYPES:
                            continue
                    role = "User" if msg["role"] == "user" else "Assistent"
                    lines.append(f"{role}: {msg['content']}")
                history_block = "\n".join(lines)

        # 2. User-Nachricht speichern (NACH History-Loading)
        msg_repo.add_message(chat_id, "user", data.question)
        chat_repo.touch(chat_id)

        # 3. Confirmation-Check VOR Agent-Aufruf
        from kiara.admin.confirmation import ConfirmationManager

        confirm_mgr = ConfirmationManager()
        handled, events = confirm_mgr.check_and_handle(chat_id, data.question, user)

        # Vorab-Events senden (action_dismissed falls Verwurf)
        for event_type, event_data in events:
            yield f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            # Dismiss-Meldung in DB persistieren (sonst fehlt sie nach Reload)
            if event_type == "action_dismissed":
                msg_repo.add_message(
                    chat_id, "assistant", "Aktion verworfen \u2014 Best\u00e4tigung nicht erhalten."
                )

        if handled:
            # Confirmation wurde bestaetigt und ausgefuehrt — kein Agent noetig
            result_msg = ""
            for _etype, _edata in events:
                if _etype == "confirmation_result":
                    result_msg = _edata.get("result", {}).get("message", "")
                    break
            if result_msg:
                msg_repo.add_message(chat_id, "assistant", result_msg)
            yield f"event: done\ndata: {json.dumps({'chat_id': chat_id}, ensure_ascii=False)}\n\n"
            _early_done = True
            return

        # 4. LLM-Backend aufloesen (Shared-Resolver)
        from kiara.generation.cloud_routing import _resolve_backend_dict

        backend = _resolve_backend_dict(
            config,
            llm_backend_id=data.llm_backend_id,
            filter_datasource_ids=None,
        )

        # 4b. Anonymizer fuer Cloud-Backends erzeugen — sonst geht Admin-Chat
        # ungeschuetzt an Anthropic/OpenAI (Issue #1057). Logik identisch zum
        # User-Agent-Pfad (api_chat.py: _prepare_agent_context).
        from kiara.web.api_chat import _cloud_requires_agent_anonymizer, _create_agent_anonymizer

        anonymizer = _create_agent_anonymizer(config, backend, restriction_levels=None)
        chain_config = dict(config)
        chain_config["_kiara_cloud_requires_anonymizer"] = _cloud_requires_agent_anonymizer(
            config,
            backend,
            restriction_levels=None,
        )

        # Mode-Event (von API-Schicht gesendet, nicht von AgentChain)
        yield f"event: mode\ndata: {json.dumps({'mode': data.mode}, ensure_ascii=False)}\n\n"

        # 5. AdminAgentChain erstellen und ausfuehren
        chain = AdminAgentChain(
            chain_config,
            backend,
            user_id=user["id"],
            chat_id=chat_id,
            anonymizer=anonymizer,
        )

        for event in chain.query_stream(data.question, history_block, data.think):
            if event.event_type == "chunk":
                text = event.data.get("text", "") if isinstance(event.data, dict) else ""
                full_answer.append(text)
                yield f"event: chunk\ndata: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            elif event.event_type == "tool_call":
                tool_events.append(event.data)
                yield f"event: tool_call\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
            elif event.event_type == "tool_result":
                if tool_events:
                    tool_events[-1]["success"] = event.data.get("success", True)
                    tool_events[-1]["summary"] = event.data.get("summary", "")
                yield f"event: tool_result\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
            elif event.event_type == "sources":
                sources = event.data if isinstance(event.data, list) else []
                yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
            # Confirmation/Wizard Events (von AdminAgentChain Seitenkanal)
            elif event.event_type == "confirmation_required":
                yield f"event: confirmation_required\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
            elif event.event_type == "wizard_update":
                yield f"event: wizard_update\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
            elif event.event_type == "error":
                yield f"event: error\ndata: {json.dumps(event.data or {}, ensure_ascii=False)}\n\n"
                _error_sent = True
            # "done" von AgentChain (data=None) wird ignoriert —
            # eigenes "done" mit chat_id kommt am Ende

    except IndexCompatibilityError as exc:
        from kiara.web.api_chat import _sse_index_compatibility_error_done

        logger.error("Index-Kompatibilitaetsguard im Admin-Chat: %s", exc, exc_info=True)
        error_event, _guard_done_event = _sse_index_compatibility_error_done(
            chat_id,
            exc,
            consumer="admin_chat_tool_retrieval",
        )
        yield error_event
        _error_sent = True
    except Exception as exc:
        logger.exception("Admin-Chat-Query fehlgeschlagen")
        yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
        _error_sent = True

    finally:
        try:
            # Think-Bloecke VOR Speicherung strippen — sonst landen sie
            # via load_history() im LLM-Kontext (History-Leak, Kontext-Bloat)
            raw_answer = "".join(full_answer)
            # Unclosed <think>-Tags sauber schliessen bevor Strip
            if "<think>" in raw_answer and "</think>" not in raw_answer:
                raw_answer += "</think>"
            answer_text = _strip_think_blocks(raw_answer)

            # Usage sammeln
            usage = chain.last_usage if chain is not None else None
            meta = {}
            if usage:
                meta["usage"] = usage
                yield f"event: usage\ndata: {json.dumps(usage, ensure_ascii=False)}\n\n"

            # 6. Assistant-Nachricht in DB speichern.
            # Bei Fehler IMMER Fehlerhinweis statt partieller Antwort persistieren \u2014
            # sonst landet eine abgeschnittene Antwort als normale Message in der History.
            if _error_sent:
                msg_repo.add_message(
                    chat_id,
                    "assistant",
                    "\u26a0 Fehler bei der Verarbeitung \u2014 bitte erneut versuchen.",
                    metadata={"type": "error"},
                )
            elif answer_text.strip():
                msg_repo.add_message(
                    chat_id,
                    "assistant",
                    answer_text.strip(),
                    sources=sources or None,
                    tool_calls=tool_events or None,
                    metadata=meta or None,
                )
        except Exception as exc:
            logger.exception("Fehler im Finally-Block der Admin-Chat-Query: %s", exc)
            if not _error_sent:
                yield f"event: error\ndata: {json.dumps({'message': 'Interner Fehler bei Antwort-Verarbeitung'}, ensure_ascii=False)}\n\n"

        # Cloud-Audit-Logging (Issue #1904) — nur wenn ein Cloud-Backend
        # tatsaechlich aufgerufen wurde (chain ist nur dann gesetzt).
        # Confirmation-only-Pfade (_early_done vor backend-resolve) ueberspringen.
        try:
            from kiara.web.api_chat import (
                _collect_stream_metrics,
                _emit_cloud_audit,
                _is_cloud_provider,
            )

            if chain is not None and backend is not None and _is_cloud_provider(backend):
                anon_stats_count = 0
                if anonymizer:
                    anon_stats_count = getattr(anonymizer, "stats", {}).get("total_entities", 0)
                m = _collect_stream_metrics(
                    full_answer,
                    chain,
                    t_start,
                    model_name_override=backend.get("model", "") if backend else "",
                    entities_count_override=anon_stats_count,
                )
                anon_q = "[Anonymisierung nicht aktiv]"
                anon_a = "[Anonymisierung nicht aktiv]"
                if anonymizer:
                    try:
                        anon_q = anonymizer.anonymize(data.question)[:200]
                        anon_a = (
                            anonymizer.anonymize(m["answer_text"])
                            if m["answer_text"]
                            else "[leere Antwort]"
                        )
                    except Exception as exc:
                        logger.warning(
                            "Anonymisierung fuer Admin-Chat-Cloud-Audit fehlgeschlagen: %s", exc
                        )
                        anon_q = "[Anonymisierung fehlgeschlagen]"
                        anon_a = "[Anonymisierung fehlgeschlagen]"
                _emit_cloud_audit(
                    config,
                    user,
                    data.question,
                    m,
                    "admin_agent",
                    anon_question=anon_q,
                    anon_answer=anon_a,
                    provider=backend.get("provider", ""),
                    model_name=m["model_name"],
                    sources=sources or None,
                    pseudonymizer_stats=anonymizer.stats if anonymizer else None,
                    correlation_id=correlation_id,
                )
        except Exception as exc:
            logger.warning("Admin-Chat Cloud-Audit fehlgeschlagen: %s", exc)

        # done-Event IMMER senden — ausser es wurde bereits im try-Block gesendet (Confirmation-Pfad)
        if _guard_done_event is not None:
            yield _guard_done_event
        elif not _early_done:
            yield f"event: done\ndata: {json.dumps({'chat_id': chat_id}, ensure_ascii=False)}\n\n"


# --- Greeting ---


@router.get("/greeting")
async def admin_chat_greeting(user: dict = Depends(get_admin_user)):
    """Formatierte Begruessungsnachricht fuer Draft-Modus (kein DB-Eintrag)."""
    greeting, meta_type = _build_chat_greeting(user)
    username = user.get("vorname") or user.get("username", "Admin")
    return {"greeting": greeting, "type": meta_type, "username": username}


# --- Backends ---


@router.get("/backends")
async def list_backends(user: dict = Depends(get_admin_user)):
    """Verfuegbare LLM-Backends fuer Dropdown (ohne API-Keys).

    Nutzt list_by_category() fuer LLM + Cloud-Backends.
    """
    from kiara.repositories.backend import KiBackendRepository

    repo = KiBackendRepository()
    backends = []
    for category in ("llm", "cloud"):
        backends.extend(repo.list_by_category(category, active_only=True))
    # Nur sichere Felder zurueckgeben (kein api_key!)
    return [
        {
            "id": b["id"],
            "name": b.get("name", b.get("model", "")),
            "provider": b.get("provider", "ollama"),
            "model": b.get("model", ""),
        }
        for b in backends
    ]


# --- Ersteinrichtungs-Flow (ohne LLM) ---

# Prozesslokaler In-Memory Setup-State. Zugriff immer ueber die async Helper,
# damit parallele Requests im gleichen Worker keine Dict-Operationen verzahnen.
_setup_state: dict[int, dict] = {}
_setup_state_lock = asyncio.Lock()
_SETUP_STATE_TTL_SECONDS = 1800


def _copy_setup_state(state: dict | None) -> dict:
    if not state:
        return {}
    copied = dict(state)
    if isinstance(copied.get("models"), list):
        copied["models"] = list(copied["models"])
    return copied


async def _get_setup_state(chat_id: int) -> dict:
    async with _setup_state_lock:
        return _copy_setup_state(_setup_state.get(chat_id))


async def _set_setup_state(chat_id: int, state: dict) -> None:
    async with _setup_state_lock:
        _setup_state[chat_id] = _copy_setup_state(state)


async def _pop_setup_state(chat_id: int) -> dict:
    async with _setup_state_lock:
        return _copy_setup_state(_setup_state.pop(chat_id, None))


async def _pop_stale_setup_states(now: float) -> list[dict]:
    stale: list[dict] = []
    async with _setup_state_lock:
        for chat_id, state in list(_setup_state.items()):
            if now - state.get("ts", 0) > _SETUP_STATE_TTL_SECONDS:
                stale.append(_copy_setup_state(_setup_state.pop(chat_id, None)))
    return stale


async def _delete_pending_setup_backend(backend_id: int) -> None:
    from kiara.admin.services import ki_backend_service

    try:
        await ki_backend_service.delete_backend(backend_id)
    except Exception as exc:
        logger.warning("Setup-Backend %s konnte nicht geloescht werden: %s", backend_id, exc)


async def _cleanup_stale_setup_states(now: float) -> None:
    stale_states = await _pop_stale_setup_states(now)
    for state in stale_states:
        backend_id = state.get("backend_id")
        if backend_id:
            await _delete_pending_setup_backend(backend_id)


async def _cleanup_setup_state(chat_id: int) -> None:
    """Loescht Wizard-State und ein ggf. angelegtes, noch nicht aktiviertes Backend."""
    state = await _pop_setup_state(chat_id)
    backend_id = state.get("backend_id")
    if backend_id:
        await _delete_pending_setup_backend(backend_id)


async def _handle_setup_mode(chat_id: int, user_input: str, user: dict) -> tuple[str, bool]:
    """Regelbasierter Setup-Handler (async, kein LLM).

    Returns:
        (response_text, mask_input) — mask_input=True wenn die User-Eingabe
        ein API-Key war und maskiert gespeichert werden soll.
    """
    from kiara.admin.services import ki_backend_service

    # TTL-Cleanup: abgelaufene States entfernen (30 Min)
    now = time.time()
    await _cleanup_stale_setup_states(now)

    text = user_input.strip().lower()
    state = await _get_setup_state(chat_id)

    # Escape-Befehle (in jedem State)
    if text in ("abbrechen", "zurueck", "reset", "cancel"):
        await _cleanup_setup_state(chat_id)
        return (_setup_greeting_text(), False)

    if text in ("hilfe", "3", "help"):
        await _cleanup_setup_state(chat_id)
        return (_help_text(), False)

    # State: Warten auf Cloud API-Key
    if state.get("state") == "awaiting_cloud_key":
        provider = state.get("provider", "anthropic")
        model = DEFAULT_CLOUD_MODEL if provider == "anthropic" else DEFAULT_OPENAI_MODEL
        backend_id = None
        try:
            result = await ki_backend_service.create_backend(
                category="cloud",
                provider=provider,
                name=f"Cloud ({provider.title()})",
                api_key=user_input.strip(),
                model=model,
                is_active=False,
            )
            backend_id = result["id"]
            test_result = await ki_backend_service.test_backend(backend_id)
            models = test_result.get("models", [])
            if models and model not in models:
                raise ValueError(f"Modell '{model}' ist fuer diesen API-Key nicht verfuegbar")
            await ki_backend_service.update_backend(backend_id, is_active=True)
            await _pop_setup_state(chat_id)
            return (
                f"Cloud-Backend konfiguriert! Modell: {model}.\n\n"
                "Der Admin-Chat ist jetzt bereit. Stellen Sie jetzt Ihre erste Frage.",
                True,
            )
        except Exception as exc:
            if backend_id is not None:
                try:
                    await ki_backend_service.delete_backend(backend_id)
                except Exception:
                    logger.exception("Rollback fuer Cloud-Setup fehlgeschlagen")
            return (
                f"Cloud-Backend konnte nicht eingerichtet werden: {exc}\n\n"
                "Bitte erneut versuchen, 'hilfe' oder 'abbrechen' eingeben.",
                True,
            )

    # State: Warten auf Modell-Auswahl (Ollama)
    if state.get("state") == "awaiting_model_choice":
        backend_id = state.get("backend_id")
        models = state.get("models", [])
        chosen = None
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(models):
                chosen = models[idx]
        else:
            for model_name in models:
                if text == model_name.lower():
                    chosen = model_name
                    break

        if not chosen:
            options = "\n".join(f"  [{i + 1}] {m}" for i, m in enumerate(models))
            return (f"Ungueltige Auswahl. Bitte waehlen Sie:\n{options}", False)

        try:
            await ki_backend_service.update_backend(
                backend_id,
                model=chosen,
                is_active=True,
            )
            await _pop_setup_state(chat_id)
            return (
                f"Modell '{chosen}' ausgewaehlt. Der Admin-Chat ist jetzt bereit.\n\n"
                "Stellen Sie jetzt Ihre erste Frage.",
                False,
            )
        except Exception as exc:
            return (f"Fehler beim Aktivieren des lokalen Backends: {exc}", False)

    # Standalone-Nummern: User tippt nur "1" oder "2" (Greeting zeigt [1]/[2])
    if text == "1":
        return (
            "Bitte waehlen Sie einen Cloud-Provider:\n\n"
            "  `cloud anthropic` — Claude (Anthropic)\n"
            "  `cloud openai` — GPT (OpenAI)",
            False,
        )

    if text == "2":
        return (
            "Bitte geben Sie die Ollama-URL an, z.B.:\n\n  `lokal http://gpu-server.local:11434`",
            False,
        )

    # Initialbefehle: Cloud-Setup
    if text in ("cloud anthropic", "1 anthropic"):
        await _set_setup_state(
            chat_id,
            {"state": "awaiting_cloud_key", "provider": "anthropic", "ts": now},
        )
        return ("Bitte geben Sie Ihren Anthropic API-Key ein:", False)

    if text in ("cloud openai", "1 openai"):
        await _set_setup_state(
            chat_id,
            {"state": "awaiting_cloud_key", "provider": "openai", "ts": now},
        )
        return ("Bitte geben Sie Ihren OpenAI API-Key ein:", False)

    # Initialbefehle: Lokales Setup
    if text.startswith("lokal ") or text.startswith("2 "):
        url = user_input.strip().split(None, 1)[1] if " " in user_input.strip() else ""
        if not url:
            return ("Bitte URL angeben, z.B.: lokal http://gpu-server.local:11434", False)

        backend_id = None
        try:
            result = await ki_backend_service.create_backend(
                category="llm",
                provider="ollama",
                name="Lokal (Ollama)",
                url=url,
                model="qwen3:8b",  # Platzhalter bis Modellliste geladen ist
                is_active=False,
            )
            backend_id = result["id"]
            test_result = await ki_backend_service.test_backend(backend_id)
            models = test_result.get("models", [])

            if not models:
                raise ValueError("Server erreichbar, aber es wurde kein LLM-Modell gefunden")

            if len(models) == 1:
                chosen = models[0]
                await ki_backend_service.update_backend(
                    backend_id,
                    model=chosen,
                    is_active=True,
                )
                await _pop_setup_state(chat_id)
                return (
                    f"Lokales Backend konfiguriert! Modell: {chosen}.\n\n"
                    "Der Admin-Chat ist jetzt bereit. Stellen Sie Ihre erste Frage.",
                    False,
                )

            await _set_setup_state(
                chat_id,
                {
                    "state": "awaiting_model_choice",
                    "backend_id": backend_id,
                    "models": models,
                    "url": url,
                    "ts": time.time(),
                },
            )
            options = "\n".join(f"  [{i + 1}] {m}" for i, m in enumerate(models))
            return (
                f"Lokales Backend verbunden! Verfuegbare Modelle:\n{options}\n\n"
                "Welches Modell soll verwendet werden?",
                False,
            )
        except Exception as exc:
            if backend_id is not None:
                try:
                    await ki_backend_service.delete_backend(backend_id)
                except Exception:
                    logger.exception("Rollback fuer lokales Setup fehlgeschlagen")
            return (f"Lokales Backend konnte nicht eingerichtet werden: {exc}", False)

    # Unbekannter Befehl im Setup-Modus
    return (
        "Ich bin im Einrichtungsmodus und kann nur vordefinierte Befehle verarbeiten.\n"
        "Geben Sie 'hilfe' oder 'abbrechen' ein fuer Optionen.",
        False,
    )


def _setup_sse_generator(chat_id: int, question: str, response: str, *, mask_input: bool = False):
    """Sync SSE-Generator fuer Setup-Antworten (kein LLM).

    Speichert User-Nachricht und Antwort in der DB, sendet SSE-Events.
    Bei mask_input=True wird die User-Nachricht als "(API-Key eingegeben)"
    gespeichert und ein user_masked-Event gesendet (F24).
    """
    msg_repo = AdminChatMessageRepository()
    chat_repo = AdminChatRepository()
    stored_question = "(API-Key eingegeben)" if mask_input else question

    try:
        with db_transaction() as conn:
            conn.execute("SET LOCAL statement_timeout = '30000'")  # 30s
            msg_repo.add_message(
                chat_id, "user", stored_question, metadata={"type": "setup_input"}, conn=conn
            )
            chat_repo.touch(chat_id, conn=conn)

        if mask_input:
            yield (
                "event: user_masked\n"
                f"data: {json.dumps({'text': stored_question}, ensure_ascii=False)}\n\n"
            )

        yield f"event: chunk\ndata: {json.dumps({'text': response}, ensure_ascii=False)}\n\n"
        with db_transaction() as conn:
            conn.execute("SET LOCAL statement_timeout = '30000'")  # 30s
            msg_repo.add_message(
                chat_id,
                "assistant",
                response,
                metadata={"type": "setup_response"},
                conn=conn,
            )
    except Exception as exc:
        logger.exception("Setup-SSE fehlgeschlagen")
        yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
    finally:
        yield f"event: done\ndata: {json.dumps({'chat_id': chat_id}, ensure_ascii=False)}\n\n"
