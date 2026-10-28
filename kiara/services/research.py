"""Gemeinsamer authentifizierter Recherche-Einstieg fuer Chat und MCP."""

import asyncio
import copy
import json
import logging
import threading
import time
from collections import OrderedDict

from kiara.generation.research_config import research_settings

logger = logging.getLogger(__name__)


def _page_wire_size(page):
    """FastMCP liefert Dicts als Text UND structuredContent, beide zaehlen.

    Einschliesslich JSON-Escaping und Reserve fuer JSON-RPC/Ergebnis-Huelle.
    SSE-Keepalives waehrend langer Laeufe sind keine Modell-/Toolausgabe.
    """
    content = json.dumps(page, ensure_ascii=False, separators=(",", ":"))
    result = {
        "content": [{"type": "text", "text": content}],
        "structuredContent": page,
        "isError": False,
    }
    return len(json.dumps(result, ensure_ascii=False)) + 256


def refresh_scope(user, selected=None, exclude=False):
    """Rechte nicht aus einem minutenalten Request-Snapshot uebernehmen."""
    from kiara.repositories.permission import PermissionRepository
    from kiara.shared.db import db_query
    from kiara.web.config_utils import load_user_permissions

    if not user or not user.get("id"):
        raise ValueError("Authentifizierung erforderlich")
    with db_query() as db:
        if (
            user.get("api_key_id")
            and not db.execute(
                "SELECT id FROM api_keys WHERE id=%s AND user_id=%s AND is_active=true "
                "AND (expires_at IS NULL OR expires_at::timestamp > NOW())",
                (user["api_key_id"], user["id"]),
            ).fetchone()
        ):
            raise ValueError("API-Zugang nicht mehr aktiv")
        if user.get("_session_token"):
            import hashlib

            token_hash = hashlib.sha256(user["_session_token"].encode()).hexdigest()
            if not db.execute(
                "SELECT id FROM sessions WHERE token=%s AND user_id=%s AND expires_at::timestamp > NOW()",
                (token_hash, user["id"]),
            ).fetchone():
                raise ValueError("Sitzung nicht mehr aktiv")
        current = db.execute(
            "SELECT id,is_admin,is_active FROM users WHERE id=%s", (user["id"],)
        ).fetchone()
        if not current or not current["is_active"]:
            raise ValueError("Benutzer nicht mehr aktiv")
        active = {
            r["id"]
            for r in db.execute("SELECT id FROM datasources WHERE is_active=true").fetchall()
        }
        if current["is_admin"]:
            allowed = active
        else:
            _, ds_ids = load_user_permissions(db, user["id"])
            allowed = (
                set(ds_ids) | set(PermissionRepository().get_public_datasource_ids())
            ) & active
    # OIDC/API-Scopes koennen die DB-Rechte zusaetzlich einschraenken.
    if not user.get("is_admin"):
        allowed &= set(user.get("datasource_ids", [])) | set(
            PermissionRepository().get_public_datasource_ids()
        )
    if selected:
        allowed = allowed - set(selected) if exclude else allowed & set(selected)
    return sorted(allowed)


def create_research(
    config,
    user,
    question,
    *,
    k=None,
    num_ctx=None,
    history="",
    backend_id=None,
    embedding_model=None,
    selected=None,
    exclude=False,
    filter_dir=None,
    filter_type=None,
):
    from kiara.generation.backends import create_backend
    from kiara.generation.research_chain import ResearchChain
    from kiara.retrieval.reranker import create_reranker
    from kiara.retrieval.retriever import HybridRetriever
    from kiara.retrieval.source_policy import resolve_chat_sources
    from kiara.tools.builtin_tools import SearchDocumentsTool
    from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool

    settings = research_settings(config)
    if not settings["enabled"]:
        raise ValueError("Recherche ist deaktiviert")
    if not isinstance(question, str) or not question.strip() or len(question) > 10000:
        raise ValueError("Eine Frage mit maximal 10000 Zeichen ist erforderlich")
    if num_ctx is not None and (type(num_ctx) is not int or not 256 <= num_ctx <= 131072):
        raise ValueError("num_ctx muss zwischen 256 und 131072 liegen")
    if selected is not None and (
        not isinstance(selected, list) or any(type(i) is not int or i < 1 for i in selected)
    ):
        raise ValueError("Datenquellen müssen als Liste positiver IDs angegeben werden")
    config, effective_k = resolve_chat_sources(copy.deepcopy(config), "research", k)
    config["retrieval"]["keyword_match_operator"] = settings["keyword_match_operator"]
    config["retrieval"]["keyword_content_only"] = settings["keyword_content_only"]
    config["retrieval"]["elasticsearch_fuzziness"] = settings["keyword_fuzziness"]
    config.setdefault("chat", {}).setdefault("agent", {})["document_read_max_chunks"] = settings[
        "read_chunks"
    ]

    def authorize():
        return refresh_scope(user, selected, exclude)

    allowed = authorize()
    backend = create_backend(config, backend_id)
    # Kein impliziter Cloud- oder Anonymisierungs-Unteraufruf in dieser Stufe.
    if backend.provider != "ollama":
        raise ValueError("Recherche unterstützt in dieser Ausbaustufe lokale Ollama-Backends")
    retriever = HybridRetriever(config, embedding_model=embedding_model)
    refs = DocumentReferences(retriever._store.get_collection_name())
    search = SearchDocumentsTool(
        retriever,
        create_reranker(config),
        config,
        allowed_datasource_ids=allowed,
        default_k=effective_k,
        default_filter_dir=filter_dir,
        default_filter_type=filter_type,
        document_references=refs,
    )
    reader = ReadDocumentTool(retriever, config, refs, allowed)
    return ResearchChain(
        config,
        backend,
        search,
        reader,
        authorize=authorize,
        num_ctx=num_ctx,
        history=history,
        question=question,
    )


class ResearchResults:
    """Begrenzte MCP-Ergebnisfortsetzung, an Benutzer und Quellenversion gebunden.

    Keine Wiederaufnahme eines laufenden Agents. Nach TTL/Prozessneustart muss
    eine neue Recherche starten. JSON-Seiten muessen nach Offset zusammengesetzt
    werden; Ausschnitte allein gelten nicht als vollstaendige Antwort.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._timers = {}

    def _remove(self, key):
        self._entries.pop(key, None)
        timer = self._timers.pop(key, None)
        if timer:
            timer.cancel()

    def _expire(self, key):
        with self._lock:
            self._remove(key)

    def add(self, chain, user):
        with self._lock:
            self._purge()
            while len(self._entries) >= chain.settings["mcp_max_sessions"]:
                self._remove(next(iter(self._entries)))
            self._entries[chain.run_id] = (
                time.monotonic() + chain.settings["mcp_ttl_seconds"],
                user["id"],
                chain,
            )
            timer = threading.Timer(
                chain.settings["mcp_ttl_seconds"], self._expire, args=(chain.run_id,)
            )
            timer.daemon = True
            self._timers[chain.run_id] = timer
            timer.start()

    def _purge(self):
        now = time.monotonic()
        for key in [k for k, v in self._entries.items() if v[0] <= now]:
            self._remove(key)

    async def page(self, run_id, user, offset=0, max_chars=20000):
        with self._lock:
            self._purge()
            entry = self._entries.get(run_id)
        if not entry or not user or entry[1] != user.get("id"):
            raise ValueError("Recherche-Ergebnis unbekannt, abgelaufen oder nicht freigegeben")
        chain = entry[2]
        if (
            type(offset) is not int
            or offset < 0
            or type(max_chars) is not int
            or not 2000 <= max_chars <= chain.settings["mcp_max_output_chars"]
        ):
            raise ValueError("Ungueltiges Seitenbudget oder Offset")
        # Auch Ergebnisse ohne positive Belege brauchen aktuelle Zugriffsrechte.
        await chain.validate_sources()
        payload = json.dumps(chain.result, ensure_ascii=False, separators=(",", ":"))
        if offset > len(payload):
            raise ValueError("Offset liegt hinter dem Ergebnis")
        response = {
            "run_id": run_id,
            "route": "research",
            "status": chain.result["status"],
            "evidence_status": chain.result["evidence_status"],
            "encoding": "json_fragment",
            "offset": offset,
            "next_offset": None,
            "total_chars": len(payload),
            "content": "",
            "client_context_checked": False,
            "notice": "JSON-Fragmente in Offset-Reihenfolge zusammensetzen. Status, Grenzen und Abdeckung ausgeben; fremdes Kontextfenster unbekannt.",
        }
        low, high = 0, len(payload) - offset
        while low < high:
            middle = (low + high + 1) // 2
            response.update(content=payload[offset : offset + middle], next_offset=offset + middle)
            if _page_wire_size(response) <= max_chars:
                low = middle
            else:
                high = middle - 1
        response.update(
            content=payload[offset : offset + low],
            next_offset=offset + low if offset + low < len(payload) else None,
        )
        response["output_truncated"] = response["next_offset"] is not None or offset > 0
        # Reserve for the output_truncated field and null/number variation.
        while _page_wire_size(response) > max_chars:
            low -= 1
            response.update(
                content=payload[offset : offset + low],
                next_offset=offset + low,
                output_truncated=True,
            )
        return response


research_results = ResearchResults()


def audit_research(user, chain):
    """MCP hat dieselbe Run-/Unteraufruf-Korrelation im Query-Audit wie Chat."""
    import hashlib

    from kiara.repositories.audit import QueryLogRepository

    result = chain.result
    if result is None:
        return
    try:
        QueryLogRepository().create(
            user_id=user["id"],
            question_hash=hashlib.sha256(chain.question.encode()).hexdigest(),
            question_excerpt=chain.question[:500],
            answer_excerpt=result["answer"][:500],
            routed_to="mcp:research",
            model_used=result["model"],
            chain_type="research",
            sources_count=len(result["sources"]),
            duration_ms=result["resources"]["duration_ms"],
            full_answer=result["answer"],
            context_sources=json.dumps(result["sources"], ensure_ascii=False),
            correlation_id=chain.run_id,
            request_settings=json.dumps({"research": result}, ensure_ascii=False),
        )
    except Exception:
        logger.exception("MCP-Recherche-Audit fehlgeschlagen run=%s", chain.run_id)


async def run_mcp_research(
    user, question, *, k=None, max_chars=20000, backend_id=None, num_ctx=None, selected=None
):
    from kiara.config import load_config
    from kiara.shared.processing_warnings import ProcessingWarnings, processing_warning_scope

    config = load_config()
    if (
        type(max_chars) is not int
        or not 2000 <= max_chars <= research_settings(config)["mcp_max_output_chars"]
    ):
        raise ValueError("Ungueltiges MCP-Ausgabebudget")
    chain = await asyncio.to_thread(
        create_research,
        config,
        user,
        question,
        k=k,
        backend_id=backend_id,
        num_ctx=num_ctx,
        selected=selected,
    )
    warnings = ProcessingWarnings()
    try:
        with processing_warning_scope(warnings):
            async for _ in chain.run():
                pass
    finally:
        if chain.result is not None:
            chain.result["warnings"] = warnings.snapshot()
            await asyncio.to_thread(audit_research, user, chain)
    chain.result["warnings"] = warnings.snapshot()
    research_results.add(chain, user)
    return await research_results.page(chain.run_id, user, max_chars=max_chars)
