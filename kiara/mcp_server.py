"""KIara MCP Server — Model Context Protocol Endpunkt.

Stellt die KIara-Wissensbasis als MCP-Tool bereit, damit externe
LLM-Clients (z.B. Claude Desktop) Firmenwissen abfragen koennen.

Architektur:
    FastAPI-App (Auth-Middleware)
      +-- REST-Retrieval unter "/api/retrieval/*" mit OIDC/JWKS
      +-- FastMCP optional unter "/mcp" mit API-Key

    Die Auth-Middleware entscheidet pfadspezifisch. Retrieval-REST akzeptiert
    nur openDesk-Keycloak Bearer-JWTs, MCP nur orag_... API-Keys. Der
    authentifizierte User wird in eine ContextVar gelegt. Das MCP-Tool liest
    den User ueber Closure-Capture aus dem async-Kontext (NICHT ueber
    ContextVar im Worker-Thread — der User wird vor asyncio.to_thread()
    gelesen).

Error-Recovery im query()-Tool:
    Die query()-Funktion faengt ALLE Exceptions aus dem RAG-Pipeline-Aufruf
    ab und gibt sie als strukturierte Fehlerantwort zurueck, statt sie als
    MCP-Fehler hochzuwerfen. Das ist bewusst so:
    - MCP-Clients (wie Claude Desktop) zeigen MCP-Exceptions als
      Verbindungsfehler an, nicht als hilfreiche Fehlermeldung.
    - Durch die Rueckgabe als ``{"answer": "Fehler: ...", "route": "error"}``
      kann der Client den Fehler dem User sinnvoll kommunizieren.
    - Audit-Logging ist best-effort: Fehler beim Loggen erzeugen nur
      Warnings, unterbrechen aber nie die Antwort an den Client.

    Nur Validierungsfehler (leere Frage, zu lang, nicht authentifiziert)
    werden als ValueError geworfen — diese sind vom Client behebbar.
"""

import asyncio
import hashlib
import json
import logging
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from kiara.shared.constants import (
    EXCERPT_LENGTH,
    MAX_AUDIT_TEXT,
    MCP_MAX_K,
    MCP_MAX_QUESTION_LENGTH,
)
from kiara.shared.processing_warnings import ProcessingWarnings, processing_warning_scope
from kiara.shared.text_utils import sanitize_utf8

logger = logging.getLogger(__name__)

# ContextVar fuer den authentifizierten User — wird von der Middleware gesetzt
# und vom MCP-Tool gelesen. Thread-safe dank ContextVar.
_current_user: ContextVar[dict | None] = ContextVar("_current_user", default=None)


def _run_rag_query(user, question, k):
    """Synchroner RAG-Query — laeuft in einem Worker-Thread.

    Berechnet Berechtigungen und fuehrt die RAG-Chain aus.

    Args:
        user: Authentifizierter User-Dict.
        question: Frage in natuerlicher Sprache.
        k: Anzahl der Kontext-Chunks.

    Returns:
        Tuple (answer, sources, route, duration_ms, model, chain).
    """
    from kiara.config import get_embedding_model, load_config
    from kiara.generation.chain import RAGChain
    from kiara.generation.cloud_routing import _create_chain_backend
    from kiara.repositories.permission import PermissionRepository

    config = load_config()
    t0 = time.monotonic()

    # Berechtigungsfilter: Admins sehen alles (None = kein Filter fuer Retrieval).
    # Nicht-Admins bekommen die Vereinigung aus:
    #   - user_ds_ids: Datasources, fuer die der User per Gruppen-Zuordnung berechtigt ist
    #   - public_ids:  Datasources OHNE Gruppenzuordnung (= oeffentlich fuer alle User)
    filter_datasource_ids = None
    if not user.get("is_admin"):
        user_ds_ids = user.get("datasource_ids", [])
        perm_repo = PermissionRepository()
        public_ids = perm_repo.get_public_datasource_ids()
        filter_datasource_ids = list(set(user_ds_ids + public_ids))

    # Restriction-DS-IDs: Fuer Cloud-Policy immer explizit (auch fuer Admins)
    if filter_datasource_ids is not None:
        restriction_datasource_ids = filter_datasource_ids
    else:
        from kiara.repositories.datasource import DatasourceRepository

        ds_repo = DatasourceRepository()
        all_active = ds_repo.get_active()
        restriction_datasource_ids = sorted([r["id"] for r in all_active])

    embedding_model = get_embedding_model(config)

    backend = _create_chain_backend(
        config,
        question=question,
        llm_backend_id=None,
        filter_datasource_ids=restriction_datasource_ids,
    )
    chain = RAGChain(config, backend=backend, embedding_model=embedding_model)
    gen, sources = chain.query_stream(
        question=question,
        k=k,
        filter_datasource_ids=filter_datasource_ids,
    )
    answer = "".join(gen)
    route = chain.route_used
    duration_ms = int((time.monotonic() - t0) * 1000)
    model = chain.backend.model

    return answer, sources, route, duration_ms, model, chain


def _run_retrieval_search(user: dict | None, params: dict[str, Any]) -> dict[str, Any]:
    """Synchroner Retrieval-only Suchaufruf fuer Worker-Threads."""
    from kiara.services.retrieval_only import RetrievalOnlyService

    return RetrievalOnlyService().search_documents(user, params)


def _run_retrieval_list_datasources(user: dict | None) -> dict[str, Any]:
    """Synchroner Retrieval-only Datasource-Listenaufruf fuer Worker-Threads."""
    from kiara.services.retrieval_only import RetrievalOnlyService

    return RetrievalOnlyService().list_datasources(user)


def _run_retrieval_get_context(user: dict | None, params: dict[str, Any]) -> dict[str, Any]:
    """Synchroner Retrieval-only Kontextaufruf fuer Worker-Threads."""
    from kiara.services.retrieval_only import RetrievalOnlyService

    return RetrievalOnlyService().get_document_context(user, params)


def _mcp_retrieval_error_response(exc: Exception) -> dict[str, Any]:
    """Formatiert kontrollierte Retrieval-only Fehler fuer MCP-Tools."""
    from kiara.retrieval.index_compatibility import IndexCompatibilityError
    from kiara.services.retrieval_only import RetrievalOnlyError

    if isinstance(exc, IndexCompatibilityError):
        return _index_compatibility_error_response(
            exc,
            consumer="mcp_retrieval_only",
            route="retrieval-only",
        )
    if isinstance(exc, RetrievalOnlyError):
        return exc.to_response()
    logger.error("MCP Retrieval-only Fehler: %s", exc, exc_info=True)
    return {
        "ok": False,
        "route": "retrieval-only",
        "error": {
            "code": "internal_error",
            "message": "Retrieval-only Anfrage konnte nicht verarbeitet werden.",
        },
    }


def _retrieval_http_error_response(exc: Exception):
    """Uebersetzt Retrieval-only Fehler in HTTP-Responses."""
    from fastapi.responses import JSONResponse

    from kiara.retrieval.index_compatibility import IndexCompatibilityError
    from kiara.services.retrieval_only import RetrievalOnlyError

    if isinstance(exc, IndexCompatibilityError):
        return JSONResponse(
            status_code=409,
            content=_index_compatibility_error_response(
                exc,
                consumer="http_retrieval_only",
                route="retrieval-only",
            ),
        )
    if isinstance(exc, RetrievalOnlyError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_response())
    logger.error("HTTP Retrieval-only Fehler: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "route": "retrieval-only",
            "error": {
                "code": "internal_error",
                "message": "Retrieval-only Anfrage konnte nicht verarbeitet werden.",
            },
        },
    )


def _index_compatibility_error_response(
    exc,
    *,
    consumer: str,
    route: str,
) -> dict[str, Any]:
    """Preserve the complete S4 diagnostic at MCP/HTTP boundaries."""
    return {
        "ok": False,
        "route": route,
        "error": {
            "code": "index_compatibility_blocked",
            "message": str(exc),
            "reason_code": exc.reason_code,
            "operation": exc.operation,
            "consumer": consumer,
            "index_compatibility_error": exc.to_dict(),
        },
    }


class _NoopMCPRegistry:
    """No-op Decorator, wenn nur REST-Retrieval ohne FastMCP laeuft."""

    def tool(self, func=None, **_kwargs):
        def decorator(inner):
            return inner

        if func is not None:
            return decorator(func)
        return decorator


def _create_app(config: dict | None = None):
    """Erzeugt die FastAPI-App mit Auth-Middleware und gemounteter MCP-App.

    Lazy Imports, damit fastmcp nur beim MCP-Service benoetigt wird.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from kiara.config import load_config
    from kiara.web.api_key_auth import validate_api_key
    from kiara.web.oidc_auth import (
        OIDCRetrievalAuthenticator,
        OIDCUnauthorizedError,
        OIDCUnavailableError,
    )

    config = config or load_config()
    web_config = config.get("web") if isinstance(config, dict) else {}
    web_config = web_config if isinstance(web_config, dict) else {}
    mcp_endpoint_enabled = web_config.get("mcp_enabled") is True
    retrieval_enabled = web_config.get("mcp_retrieval_enabled") is True
    oidc_config = web_config.get("oidc_retrieval") if isinstance(web_config, dict) else {}
    oidc_enabled = isinstance(oidc_config, dict) and oidc_config.get("enabled") is True
    oidc_auth = OIDCRetrievalAuthenticator(config)

    mcp_http_app = None
    if mcp_endpoint_enabled:
        from fastmcp import FastMCP

        mcp = FastMCP(
            "KIara",
            instructions=(
                "KIara ist die Wissensbasis deines Unternehmens. "
                "Nutze das 'query'-Tool, um Fragen zu stellen. "
                "Antworten basieren auf den intern indexierten Dokumenten. "
                "Gib Verarbeitungshinweise aus 'warnings' an den Nutzer weiter. "
                "Bei unvollstaendigen Belegen darf eine fehlende Angabe nicht als "
                "Nachweis ihres Fehlens im Datenbestand gelten. Lade bei Bedarf "
                "gezielt Dokumentkontext nach. Das Kontextfenster des aufrufenden "
                "Modells ist KIara bei Retrieval-only-Aufrufen nicht bekannt."
            ),
        )
    else:
        mcp = _NoopMCPRegistry()

    @mcp.tool
    async def query(question: str, k: int = 5) -> dict[str, Any]:
        """Frage an die Firmenwissensbasis stellen.

        Stellt eine Frage an die intern indexierten Dokumente und gibt eine
        Antwort zurueck, die auf den relevantesten Dokumenten basiert.

        Args:
            question: Die Frage in natuerlicher Sprache.
            k: Anzahl der Kontext-Chunks fuer die Antwortgenerierung (1-20, Default 5).

        Returns:
            Dict mit 'answer' (Antwort-Text) und 'sources' (Quellenangaben).
        """
        # ContextVar HIER lesen, nicht im Worker-Thread — asyncio.to_thread()
        # kopiert den Context nicht automatisch in den Thread-Pool
        user = _current_user.get()
        if not user:
            raise ValueError(
                "Nicht authentifiziert. Authorization: Bearer orag_... Header erforderlich."
            )

        if not question or not question.strip():
            raise ValueError("Frage darf nicht leer sein.")
        if len(question) > MCP_MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Frage zu lang ({len(question)} Zeichen, max. {MCP_MAX_QUESTION_LENGTH:,})."
            )

        k = max(1, min(k, MCP_MAX_K))
        correlation_id = str(uuid4())
        warnings = ProcessingWarnings()

        try:
            # User als explizites Argument uebergeben (nicht per ContextVar im Thread)
            with processing_warning_scope(warnings):
                answer, sources, route, duration_ms, model, chain = await asyncio.to_thread(
                    _run_rag_query,
                    user,
                    question,
                    k,
                )
        except Exception as exc:
            logger.error("MCP Query fehlgeschlagen: %s: %s", type(exc).__name__, exc)
            logger.debug("MCP Query Exception Details", exc_info=True)
            try:
                _log_mcp_query(
                    user,
                    question,
                    answer="",
                    sources=[],
                    route="error",
                    duration_ms=0,
                    model="",
                    full_answer=f"Exception: {type(exc).__name__}: {exc}",
                    correlation_id=correlation_id,
                )
            except Exception:
                logger.warning("MCP Audit-Log im Error-Handler fehlgeschlagen")
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                response = _index_compatibility_error_response(
                    exc,
                    consumer="mcp_rag_query",
                    route="error",
                )
                response.update({"answer": "", "sources": [], "warnings": warnings.snapshot()})
                return response
            return {
                "answer": "Fehler bei der Verarbeitung. Bitte spaeter erneut versuchen.",
                "sources": [],
                "route": "error",
                "warnings": warnings.snapshot(),
            }

        try:
            _log_mcp_query(
                user,
                question,
                answer,
                sources,
                route,
                duration_ms,
                model,
                entities_anonymized=getattr(chain, "last_entities_anonymized", 0),
                full_prompt=chain.last_prompt,
                full_answer=answer,
                context_sources=json.dumps(sources, default=str) if sources else "",
                correlation_id=correlation_id,
            )
        except Exception as exc:
            logger.warning("MCP Audit-Log Fehler: %s", exc)

        # Cloud-Audit bei Cloud-Route
        if route == "cloud":
            try:
                _log_mcp_cloud_audit(
                    user,
                    question,
                    answer,
                    sources,
                    chain,
                    duration_ms,
                    correlation_id=correlation_id,
                )
            except Exception as exc:
                logger.error("MCP Cloud-Audit Fehler (fail-closed): %s", exc)
                return {
                    "answer": "Cloud-Audit konnte nicht geschrieben werden. Die Anfrage wurde nicht als erfolgreich abgeschlossen.",
                    "sources": [],
                    "route": "error",
                    "warnings": warnings.snapshot(),
                }

        formatted_sources = []
        for s in sources:
            raw_score = s.get("score", 0)
            formatted_sources.append(
                {
                    "file": sanitize_utf8(s.get("source", "")),
                    "relevance": round(raw_score, 3) if isinstance(raw_score, (int, float)) else 0,
                }
            )

        return {
            "answer": sanitize_utf8(answer),
            "sources": formatted_sources,
            "route": route,
            "warnings": warnings.snapshot(),
        }

    @mcp.tool
    async def research(
        question: str,
        k: int = 20,
        max_chars: int = 20000,
        llm_backend_id: int | None = None,
        num_ctx: int | None = None,
        datasource_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Gruendliche Recherche mit sequenziellen Lese-Unteragenten und Originalbelegpruefung.

        Derselbe Kern wie Kiara-Chat/Recherche. JSON-Fragmente anhand next_offset
        mit research_result zusammensetzen. Status, Grenzen, Warnungen und
        geprueften Umfang dem Nutzer anzeigen. Unvollstaendig ist kein negativer
        Sachbeweis. max_chars begrenzt die serialisierte Seite (mindestens 2000).
        Das Kontextfenster des externen Clients ist unbekannt und nicht geprueft.
        """
        from kiara.services.research import run_mcp_research

        try:
            return await run_mcp_research(
                _current_user.get(),
                question,
                k=k,
                max_chars=max_chars,
                backend_id=llm_backend_id,
                num_ctx=num_ctx,
                selected=datasource_ids,
            )
        except Exception as exc:
            logger.exception("MCP-Recherche fehlgeschlagen")
            return {
                "route": "research",
                "status": "failed",
                "error": str(exc),
                "client_context_checked": False,
            }

    @mcp.tool
    async def research_result(
        run_id: str, offset: int = 0, max_chars: int = 20000
    ) -> dict[str, Any]:
        """Naechstes JSON-Fragment einer Recherche; Rechte und Originalversionen werden erneut geprueft.

        Offset exakt aus next_offset uebernehmen. Fragmente zusammensetzen und
        Einschraenkungen sichtbar weitergeben. Ergebnisse verfallen nach der
        konfigurierten TTL bzw. bei Dienstneustart; dann neue Recherche starten.
        """
        from kiara.services.research import research_results

        try:
            return await research_results.page(run_id, _current_user.get(), offset, max_chars)
        except Exception as exc:
            return {
                "route": "research",
                "status": "failed",
                "error": str(exc),
                "client_context_checked": False,
            }

    @mcp.tool
    async def search_documents(
        query: str,
        k: int = 10,
        datasource_id: int | None = None,
        file_type: str | None = None,
        directory: str | None = None,
        source_contains: str | None = None,
        include_text: bool | None = None,
        max_chars_per_result: int | None = None,
    ) -> dict[str, Any]:
        """Retrieval-only Suche ohne KIara-Antwortgenerierung.

        Gibt Chunks, Quellen, Scores, Metadaten und Retriever-Diagnostik
        strukturiert zurueck. Volltext wird nur ausgegeben, wenn die
        Rohchunk-Policy dies erlaubt.
        """
        user = _current_user.get()
        params = {
            "query": query,
            "k": k,
            "datasource_id": datasource_id,
            "file_type": file_type,
            "directory": directory,
            "source_contains": source_contains,
            "include_text": include_text,
            "max_chars_per_result": max_chars_per_result,
        }
        try:
            return await asyncio.to_thread(_run_retrieval_search, user, params)
        except Exception as exc:
            return _mcp_retrieval_error_response(exc)

    @mcp.tool
    async def list_datasources() -> dict[str, Any]:
        """Listet fuer den API-Key sichtbare Datenquellen."""
        user = _current_user.get()
        try:
            return await asyncio.to_thread(_run_retrieval_list_datasources, user)
        except Exception as exc:
            return _mcp_retrieval_error_response(exc)

    @mcp.tool
    async def get_document_context(
        source: str,
        datasource_id: int,
        chunk_index: int,
        window: int = 2,
        max_chunks: int = 10,
        include_text: bool | None = None,
        max_chars_total: int | None = None,
    ) -> dict[str, Any]:
        """Laedt kontrolliert Nachbar-Chunks einer gefundenen Quelle."""
        user = _current_user.get()
        params = {
            "source": source,
            "datasource_id": datasource_id,
            "chunk_index": chunk_index,
            "window": window,
            "max_chunks": max_chunks,
            "include_text": include_text,
            "max_chars_total": max_chars_total,
        }
        try:
            return await asyncio.to_thread(_run_retrieval_get_context, user, params)
        except Exception as exc:
            return _mcp_retrieval_error_response(exc)

    # MCP ASGI-App nur erzeugen, wenn der MCP-Endpunkt aktiv ist.
    if mcp_endpoint_enabled:
        mcp_http_app = mcp.http_app(path="/", stateless_http=True)

    app_kwargs = {"title": "KIara MCP", "docs_url": None, "redoc_url": None}
    if mcp_http_app is not None:
        app_kwargs["lifespan"] = mcp_http_app.lifespan
    app = FastAPI(**app_kwargs)

    # Health-Check VOR mount registrieren (FastAPI Routes vor Mounts)
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "kiara-mcp"}

    @app.post("/api/retrieval/search_documents")
    async def api_search_documents(request: Request):
        """REST-Fassade fuer Retrieval-only Suche."""
        user = _current_user.get()
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                payload = {}
            return await asyncio.to_thread(_run_retrieval_search, user, payload)
        except Exception as exc:
            return _retrieval_http_error_response(exc)

    @app.get("/api/retrieval/list_datasources")
    @app.post("/api/retrieval/list_datasources")
    async def api_list_datasources():
        """REST-Fassade fuer sichtbare Datenquellen."""
        user = _current_user.get()
        try:
            return await asyncio.to_thread(_run_retrieval_list_datasources, user)
        except Exception as exc:
            return _retrieval_http_error_response(exc)

    @app.post("/api/retrieval/get_document_context")
    async def api_get_document_context(request: Request):
        """REST-Fassade fuer kontrolliertes Kontext-Nachladen."""
        user = _current_user.get()
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                payload = {}
            return await asyncio.to_thread(_run_retrieval_get_context, user, payload)
        except Exception as exc:
            return _retrieval_http_error_response(exc)

    @app.middleware("http")
    async def mcp_auth_middleware(request: Request, call_next):
        """Pfadspezifische Auth fuer Health, Retrieval-REST und MCP."""
        path = request.url.path
        if path == "/health":
            return await call_next(request)

        if path.startswith("/api/retrieval/"):
            if not retrieval_enabled:
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": "retrieval_only_disabled"}},
                )
            if not oidc_enabled:
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": "oidc_retrieval_disabled"}},
                )
            auth_header = request.headers.get("Authorization", "")
            try:
                user = await asyncio.to_thread(
                    oidc_auth.authenticate,
                    auth_header,
                    path=path,
                )
            except OIDCUnavailableError:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"code": "oidc_retrieval_unavailable"}},
                )
            except OIDCUnauthorizedError:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"code": "unauthorized"}},
                )
        elif path == "/mcp" or path.startswith("/mcp/"):
            if not mcp_endpoint_enabled:
                return JSONResponse(status_code=403, content={"error": "MCP-Endpunkt deaktiviert"})
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authorization: Bearer <API-Key> Header erforderlich"},
                )

            key = auth_header[7:]
            if not key.startswith("orag_"):
                return JSONResponse(
                    status_code=401,
                    content={"error": "Ungueltiges Key-Format. Erwartet: orag_..."},
                )

            user = await asyncio.to_thread(validate_api_key, key)
            if not user:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Ungueltiger oder abgelaufener API-Key"},
                )
        else:
            return await call_next(request)

        # ContextVar.set() gibt ein Token zurueck, mit dem reset() den
        # vorherigen Wert wiederherstellt — wichtig bei verschachtelten Requests
        token = _current_user.set(user)
        try:
            response = await call_next(request)
        finally:
            _current_user.reset(token)

        return response

    # Mount NACH allen @app-Routes und nur unter dem expliziten MCP-Pfad.
    if mcp_http_app is not None:
        app.mount("/mcp", mcp_http_app)

    return app


def _log_mcp_query(
    user: dict,
    question: str,
    answer: str,
    sources: list,
    route: str,
    duration_ms: int,
    model: str,
    entities_anonymized: int = 0,
    full_prompt: str = "",
    full_answer: str = "",
    context_sources: str = "",
    correlation_id: str = "",
):
    """Audit-Log fuer MCP-Anfragen in die bestehende query_log-Tabelle."""
    from kiara.repositories.audit import QueryLogRepository
    from kiara.repositories.base import RawSQL

    query_log_repo = QueryLogRepository()
    if full_prompt and len(full_prompt) > MAX_AUDIT_TEXT:
        full_prompt = full_prompt[:MAX_AUDIT_TEXT] + "\n[...gekuerzt]"
    if full_answer and len(full_answer) > MAX_AUDIT_TEXT:
        full_answer = full_answer[:MAX_AUDIT_TEXT] + "\n[...gekuerzt]"
    if context_sources and len(context_sources) > MAX_AUDIT_TEXT:
        context_sources = context_sources[:MAX_AUDIT_TEXT] + "\n[...gekuerzt]"
    try:
        query_log_repo.create(
            user_id=user["id"],
            question_hash=hashlib.sha256(question.encode()).hexdigest(),
            question_excerpt=question[:EXCERPT_LENGTH],
            answer_excerpt=answer[:EXCERPT_LENGTH],
            routed_to=f"mcp:{route}",
            model_used=model,
            sources_count=len(sources),
            entities_anonymized=entities_anonymized,
            duration_ms=duration_ms,
            chat_id=None,
            chain_type="standard",
            full_prompt=full_prompt,
            full_answer=full_answer,
            context_sources=context_sources,
            correlation_id=correlation_id,
            created_at=RawSQL("NOW()"),
        )
    except Exception as exc:
        logger.warning("MCP Audit-Log query_log Fehler: %s", exc)


def _log_mcp_cloud_audit(
    user: dict,
    question: str,
    answer: str,
    sources: list,
    chain,
    duration_ms: int,
    correlation_id: str = "",
):
    """Cloud-Audit-Logging fuer MCP-Anfragen (cloud_audit_log + audit_anonymization)."""
    from kiara.cloud.audit import AuditLogger
    from kiara.config import load_config

    config = load_config()
    audit = AuditLogger(config)
    usage = chain.last_usage or {}

    # Context-Sources: Datei-Metadaten
    file_counts: dict[str, int] = {}
    for s in sources:
        fname = s.get("source", "") or s.get("file", "")
        if fname:
            file_counts[fname] = file_counts.get(fname, 0) + 1
    context_sources = [{"file": f, "chunks": c} for f, c in file_counts.items()]

    backend = chain.backend
    entities_count = chain.last_entities_anonymized

    audit.log_request(
        user_id=user["id"],
        question_anonymized=chain.last_anon_question or "[nicht anonymisiert]",
        answer_anonymized=chain.last_anon_answer or "[nicht anonymisiert]",
        provider=backend.provider,
        model=backend.model,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("response_tokens", usage.get("completion_tokens", 0)),
        entities_anonymized=entities_count,
        duration_ms=duration_ms,
        context_sources=context_sources,
        correlation_id=correlation_id,
    )

    # audit_anonymization Eintrag
    from kiara.anonymization.audit import log_anonymization_event

    stats = getattr(backend, "anonymizer_stats", None) or {
        "total_entities": chain.last_entities_anonymized,
        "entity_types": {},
    }
    log_anonymization_event(
        user_id=user["id"],
        session_id=correlation_id or str(uuid4()),
        question=question,
        entity_count=stats.get("total_entities", 0),
        entity_types=stats.get("entity_types", {}),
        route="mcp:cloud",
        cloud_provider=backend.provider,
        cloud_model=backend.model,
        correlation_id=correlation_id or None,
    )


def run_mcp_server() -> None:
    """Startet den MCP-Server. Wird von kiara-mcp.service aufgerufen."""
    from kiara.config import load_config

    logging.basicConfig(level=logging.INFO)
    config = load_config()
    web_config = config.get("web", {})
    service_should_run = (
        web_config.get("mcp_enabled") is True or web_config.get("mcp_retrieval_enabled") is True
    )
    if not service_should_run:
        logger.info(
            "kiara-mcp.service beendet sich sauber: web.mcp_enabled=false und "
            "web.mcp_retrieval_enabled=false"
        )
        return

    from kiara.shared.db import init_db_backend

    init_db_backend(config)

    from kiara.shared.tracing import init_tracing

    init_tracing(config)

    host = web_config.get("host", "0.0.0.0")
    port = web_config.get("mcp_port", 8504)

    logger.info("MCP-Server startet auf %s:%d", host, port)

    import uvicorn

    app = _create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_mcp_server()
