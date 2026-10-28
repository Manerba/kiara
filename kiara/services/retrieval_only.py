"""Retrieval-only Fassade fuer MCP/API-Clients.

Der Service erzwingt Auth, Datasource-ACLs, Rohchunk-Policy und harte Limits,
nutzt aber fuer die Suche ausschliesslich die bestehende Retrieval-Pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any
from uuid import uuid4

from kiara.shared.constants import EXCERPT_LENGTH, MAX_AUDIT_TEXT
from kiara.shared.text_utils import sanitize_utf8

logger = logging.getLogger(__name__)

_DEFAULT_MAX_K = 50
_DEFAULT_MAX_QUERY_LENGTH = 10_000
_DEFAULT_MAX_CHARS_PER_RESULT = 4_000
_DEFAULT_MAX_CONTEXT_WINDOW = 5
_DEFAULT_MAX_CONTEXT_CHUNKS = 10
_DEFAULT_MAX_CHARS_TOTAL = 20_000
_FILTER_STRING_MAX_CHARS = 2_048
_FILE_TYPE_MAX_CHARS = 255
_PATHS = ("semantic", "keyword", "graph", "colbert")
_TRIVIAL_NUMERIC_QUERY_RE = re.compile(r"^\d{1,4}$")
_BARE_YEAR_SOURCE_FILTER_RE = re.compile(r"^20\d{2}$")


class RetrievalOnlyError(Exception):
    """Kontrollierter Fehler fuer Retrieval-only Requests."""

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status

    def to_response(self) -> dict[str, Any]:
        return {
            "ok": False,
            "route": "retrieval-only",
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }


def _now_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return sanitize_utf8(value) if isinstance(value, str) else value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    return sanitize_utf8(str(value))


def _normalize_ds_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list | tuple | set) else [raw]
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed not in result:
            result.append(parsed)
    return result


def _filename_from_source(source: str) -> str:
    normalized = source.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def _score_value(result: dict) -> tuple[Any, str]:
    if result.get("rerank_score") is not None:
        return result.get("rerank_score"), "rerank_score"
    if result.get("rrf_score") is not None:
        return result.get("rrf_score"), "rrf_score"
    return result.get("score", 0), "score"


class RetrievalOnlyService:
    """Zustandslose Fassade ueber KIaras bestehende Retrieval-Pipeline."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        retriever_factory=None,
        reranker_factory=None,
        permission_repo=None,
        datasource_repo=None,
        audit_repo=None,
        embedding_model: str | None = None,
    ) -> None:
        if config is None:
            from kiara.config import load_config

            config = load_config()
        self._config = config
        self._retriever_factory = retriever_factory
        self._reranker_factory = reranker_factory
        self._permission_repo = permission_repo
        self._datasource_repo = datasource_repo
        self._audit_repo = audit_repo
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Config, Auth, Limits
    # ------------------------------------------------------------------

    @property
    def _web_config(self) -> dict:
        web = self._config.get("web") if isinstance(self._config, dict) else None
        return web if isinstance(web, dict) else {}

    def _require_enabled(self) -> None:
        if self._web_config.get("mcp_retrieval_enabled") is not True:
            raise RetrievalOnlyError(
                "retrieval_only_disabled",
                "Retrieval-only MCP/API ist serverseitig deaktiviert.",
                403,
            )

    @staticmethod
    def _require_user(user: dict | None) -> dict:
        if not isinstance(user, dict) or not user.get("id"):
            raise RetrievalOnlyError(
                "auth_required",
                "Authentifizierung erforderlich.",
                401,
            )
        return user

    def _limit(self, key: str, default: int) -> int:
        value = self._web_config.get(key, default)
        if isinstance(value, bool):
            raise RetrievalOnlyError("invalid_config", f"Ungueltiger Config-Wert: web.{key}", 500)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RetrievalOnlyError(
                "invalid_config", f"Ungueltiger Config-Wert: web.{key}", 500
            ) from exc
        if parsed < 1:
            raise RetrievalOnlyError("invalid_config", f"Ungueltiger Config-Wert: web.{key}", 500)
        return parsed

    def _limits(self) -> dict[str, int]:
        return {
            "max_k": self._limit("mcp_retrieval_max_k", _DEFAULT_MAX_K),
            "max_query_length": self._limit(
                "mcp_retrieval_max_query_length", _DEFAULT_MAX_QUERY_LENGTH
            ),
            "max_chars_per_result": self._limit(
                "mcp_retrieval_max_chars_per_result", _DEFAULT_MAX_CHARS_PER_RESULT
            ),
            "max_context_window": self._limit(
                "mcp_retrieval_max_context_window", _DEFAULT_MAX_CONTEXT_WINDOW
            ),
            "max_context_chunks": self._limit(
                "mcp_retrieval_max_context_chunks", _DEFAULT_MAX_CONTEXT_CHUNKS
            ),
            "max_chars_total": self._limit(
                "mcp_retrieval_max_chars_total", _DEFAULT_MAX_CHARS_TOTAL
            ),
        }

    @staticmethod
    def _required_str(params: dict, key: str, *, max_chars: int) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RetrievalOnlyError("invalid_request", f"{key} ist erforderlich.", 400)
        value = value.strip()
        if len(value) > max_chars:
            raise RetrievalOnlyError(
                "limit_exceeded",
                f"{key} ist zu lang ({len(value)} Zeichen, max. {max_chars}).",
                400,
            )
        return value

    @staticmethod
    def _reject_trivial_query(query: str) -> None:
        stripped = query.strip()
        if _TRIVIAL_NUMERIC_QUERY_RE.fullmatch(stripped):
            raise RetrievalOnlyError(
                "query_too_broad",
                (
                    f"Suchanfrage '{stripped}' ist zu unspezifisch (reine Zahl). "
                    "Bitte eine inhaltliche Suchanfrage formulieren."
                ),
                400,
            )

    @staticmethod
    def _optional_str(params: dict, key: str, *, max_chars: int) -> str | None:
        value = params.get(key)
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise RetrievalOnlyError("invalid_request", f"{key} muss ein String sein.", 400)
        value = value.strip()
        if not value:
            return None
        if len(value) > max_chars:
            raise RetrievalOnlyError(
                "limit_exceeded",
                f"{key} ist zu lang ({len(value)} Zeichen, max. {max_chars}).",
                400,
            )
        return value

    @staticmethod
    def _effective_source_filter(source_contains: str | None, query: str) -> str | None:
        if (
            source_contains
            and _BARE_YEAR_SOURCE_FILTER_RE.fullmatch(source_contains)
            and source_contains not in query
        ):
            logger.info(
                "retrieval-only: source_contains=%r ignoriert, weil die Query "
                "keine entsprechende Jahreszahl enthaelt",
                source_contains,
            )
            return None
        return source_contains

    @staticmethod
    def _optional_bool(params: dict, key: str) -> bool | None:
        value = params.get(key)
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "ja"}:
                return True
            if normalized in {"false", "0", "no", "nein"}:
                return False
        raise RetrievalOnlyError("invalid_request", f"{key} muss boolean sein.", 400)

    @staticmethod
    def _int_param(
        params: dict,
        key: str,
        default: int | None,
        *,
        min_value: int,
        max_value: int,
    ) -> int:
        value = params.get(key, default)
        if value is None or isinstance(value, bool):
            raise RetrievalOnlyError("invalid_request", f"{key} ist erforderlich.", 400)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RetrievalOnlyError("invalid_request", f"{key} muss Integer sein.", 400) from exc
        if parsed < min_value or parsed > max_value:
            raise RetrievalOnlyError(
                "limit_exceeded",
                f"{key} muss zwischen {min_value} und {max_value} liegen.",
                400,
            )
        return parsed

    def _include_text(self, params: dict) -> bool:
        include_text = self._optional_bool(params, "include_text")
        if include_text is None:
            include_text = self._web_config.get("mcp_retrieval_include_text_default") is True
        if include_text and self._web_config.get("mcp_retrieval_allow_raw_chunks") is not True:
            raise RetrievalOnlyError(
                "raw_chunks_disabled",
                "Rohchunk-Ausgabe ist serverseitig deaktiviert.",
                403,
            )
        return include_text

    # ------------------------------------------------------------------
    # ACL / Datasources
    # ------------------------------------------------------------------

    def _permission_repo_instance(self):
        if self._permission_repo is not None:
            return self._permission_repo
        from kiara.repositories.permission import PermissionRepository

        self._permission_repo = PermissionRepository()
        return self._permission_repo

    def _datasource_repo_instance(self):
        if self._datasource_repo is not None:
            return self._datasource_repo
        from kiara.repositories.datasource import DatasourceRepository

        self._datasource_repo = DatasourceRepository()
        return self._datasource_repo

    def _audit_repo_instance(self):
        if self._audit_repo is not None:
            return self._audit_repo
        from kiara.repositories.audit import QueryLogRepository

        self._audit_repo = QueryLogRepository()
        return self._audit_repo

    def _public_datasource_ids(self) -> list[int]:
        try:
            return _normalize_ds_ids(self._permission_repo_instance().get_public_datasource_ids())
        except Exception as exc:
            logger.warning("Public-Datasource-IDs konnten nicht geladen werden: %s", exc)
            raise RetrievalOnlyError(
                "permission_unavailable",
                "Datasource-Berechtigungen konnten nicht ermittelt werden.",
                503,
            ) from exc

    def _active_datasource_ids(self) -> set[int]:
        try:
            rows = self._datasource_repo_instance().get_active()
        except Exception as exc:
            logger.warning("Aktive Datasources konnten nicht geladen werden: %s", exc)
            raise RetrievalOnlyError(
                "permission_unavailable",
                "Datasource-Berechtigungen konnten nicht ermittelt werden.",
                503,
            ) from exc
        return {int(r["id"]) for r in rows if r.get("id") is not None}

    @staticmethod
    def _selected_datasource_ids(params: dict) -> list[int] | None:
        if "datasource_id" not in params or params.get("datasource_id") in (None, ""):
            return None
        value = params.get("datasource_id")
        if isinstance(value, bool):
            raise RetrievalOnlyError("invalid_request", "datasource_id muss Integer sein.", 400)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RetrievalOnlyError(
                "invalid_request", "datasource_id muss Integer sein.", 400
            ) from exc
        if parsed < 1:
            raise RetrievalOnlyError("invalid_request", "datasource_id muss positiv sein.", 400)
        return [parsed]

    def _allowed_datasource_filter(
        self,
        user: dict,
        selected_ids: list[int] | None,
    ) -> list[int] | None:
        if user.get("is_admin"):
            if selected_ids is None:
                return None
            active_ids = self._active_datasource_ids()
            missing = [ds_id for ds_id in selected_ids if ds_id not in active_ids]
            if missing:
                raise RetrievalOnlyError(
                    "datasource_forbidden",
                    "Datenquelle ist nicht verfuegbar oder nicht erlaubt.",
                    403,
                )
            return selected_ids

        allowed = sorted(
            set(_normalize_ds_ids(user.get("datasource_ids")) + self._public_datasource_ids())
        )
        if selected_ids is None:
            return allowed
        if any(ds_id not in allowed for ds_id in selected_ids):
            raise RetrievalOnlyError(
                "datasource_forbidden",
                "Datenquelle ist nicht verfuegbar oder nicht erlaubt.",
                403,
            )
        return selected_ids

    # ------------------------------------------------------------------
    # Retrieval backend
    # ------------------------------------------------------------------

    def _embedding_model_name(self) -> str:
        if self._embedding_model:
            return self._embedding_model
        from kiara.config import get_embedding_model

        try:
            self._embedding_model = get_embedding_model(self._config)
        except Exception as exc:
            raise RetrievalOnlyError(
                "retrieval_unavailable",
                "Embedding-Modell konnte nicht ermittelt werden.",
                503,
            ) from exc
        return self._embedding_model

    def _retriever(self):
        if self._retriever_factory is not None:
            return self._retriever_factory(self._config, self._embedding_model_name())
        from kiara.retrieval.retriever import HybridRetriever

        return HybridRetriever(self._config, embedding_model=self._embedding_model_name())

    def _reranker(self):
        if self._reranker_factory is not None:
            return self._reranker_factory(self._config)
        from kiara.retrieval.reranker import create_reranker

        return create_reranker(self._config)

    @staticmethod
    def _with_final_context_ranks(results: list[dict]) -> list[dict]:
        ranked = []
        for rank, result in enumerate(results, 1):
            item = result.copy()
            item["final_context_rank"] = rank
            ranked.append(item)
        return ranked

    def _sanitize_diagnostic_hit(self, hit: dict, max_chars: int) -> dict:
        clean = dict(hit)
        excerpt = clean.get("excerpt")
        if isinstance(excerpt, str):
            excerpt_chars = min(EXCERPT_LENGTH, max_chars)
            clean["excerpt"] = sanitize_utf8(excerpt[:excerpt_chars])
            clean["excerpt_truncated"] = len(excerpt) > excerpt_chars
        return _jsonable(clean)

    def _sanitize_retrieval_diagnostics(
        self,
        diagnostics: dict | None,
        *,
        max_chars: int,
    ) -> dict | None:
        if not isinstance(diagnostics, dict):
            return None
        clean = dict(diagnostics)
        paths = clean.get("paths")
        if isinstance(paths, dict):
            clean_paths = {}
            for name, payload in paths.items():
                path_payload = dict(payload) if isinstance(payload, dict) else {}
                hits = path_payload.get("hits")
                if isinstance(hits, list):
                    path_payload["hits"] = [
                        self._sanitize_diagnostic_hit(h, max_chars)
                        for h in hits
                        if isinstance(h, dict)
                    ]
                clean_paths[name] = _jsonable(path_payload)
            clean["paths"] = clean_paths
        final_results = clean.get("final_results")
        if isinstance(final_results, list):
            clean["final_results"] = [
                self._sanitize_diagnostic_hit(h, max_chars)
                for h in final_results
                if isinstance(h, dict)
            ]
        return _jsonable(clean)

    @staticmethod
    def _path_states(diagnostics: dict | None) -> dict[str, str]:
        if not isinstance(diagnostics, dict):
            return {}
        paths = diagnostics.get("paths")
        if not isinstance(paths, dict):
            return {}
        states = {}
        for path in _PATHS:
            payload = paths.get(path)
            if isinstance(payload, dict):
                states[path] = str(payload.get("status") or "unknown")
        rrf = diagnostics.get("rrf")
        if isinstance(rrf, dict):
            states["rrf"] = str(rrf.get("status") or "unknown")
        return states

    @staticmethod
    def _path_latencies(diagnostics: dict | None) -> dict[str, int]:
        if not isinstance(diagnostics, dict):
            return {}
        latencies = {}
        paths = diagnostics.get("paths")
        if isinstance(paths, dict):
            for path, payload in paths.items():
                if isinstance(payload, dict) and isinstance(payload.get("latency_ms"), int | float):
                    latencies[str(path)] = int(payload["latency_ms"])
        rrf = diagnostics.get("rrf")
        if isinstance(rrf, dict) and isinstance(rrf.get("latency_ms"), int | float):
            latencies["rrf"] = int(rrf["latency_ms"])
        return latencies

    def _format_result(
        self,
        result: dict,
        rank: int,
        *,
        include_text: bool,
        max_chars: int,
    ) -> dict[str, Any]:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        metadata = dict(metadata)
        source = sanitize_utf8(str(metadata.get("source") or ""))
        text = sanitize_utf8(str(result.get("text") or ""))
        ds_ids = _normalize_ds_ids(metadata.get("ds_ids"))
        score, score_type = _score_value(result)
        chunk_index = metadata.get("chunk_index")
        chunk_total = metadata.get("chunk_total")
        chunk_id = (
            result.get("chunk_id")
            or result.get("point_id")
            or result.get("_doc_id")
            or hashlib.sha256(
                f"{source}:{chunk_index}:{','.join(map(str, ds_ids))}".encode()
            ).hexdigest()
        )
        excerpt_chars = min(EXCERPT_LENGTH, max_chars)
        item = {
            "id": str(chunk_id),
            "rank": rank,
            "final_context_rank": result.get("final_context_rank", rank),
            "source": source,
            "path": source,
            "filename": _filename_from_source(source),
            "datasource_ids": ds_ids,
            "file_type": sanitize_utf8(str(metadata.get("file_type") or "")),
            "directory": sanitize_utf8(str(metadata.get("directory") or "")),
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "score": score,
            "score_type": score_type,
            "scores": {
                "score": result.get("score"),
                "rerank_score": result.get("rerank_score"),
                "rrf_score": result.get("rrf_score"),
                "semantic_score": result.get("semantic_score"),
                "keyword_score": result.get("keyword_score"),
                "graph_score": result.get("graph_score"),
                "colbert_score": result.get("colbert_score"),
            },
            "ranks": {
                "semantic_rank": result.get("semantic_rank"),
                "keyword_rank": result.get("keyword_rank"),
                "graph_rank": result.get("graph_rank"),
                "colbert_rank": result.get("colbert_rank"),
                "rrf_rank": result.get("rrf_rank"),
                "final_context_rank": result.get("final_context_rank", rank),
            },
            "rrf_contributions": dict(result.get("rrf_contributions") or {}),
            "retriever_paths": list(result.get("retriever_paths") or []),
            "metadata": _jsonable(metadata),
            "excerpt": text[:excerpt_chars],
            "text_chars": len(text),
            "text_included": include_text,
        }
        item["excerpt_truncated"] = len(text) > excerpt_chars
        if include_text:
            item["text"] = text[:max_chars]
            item["text_truncated"] = len(text) > max_chars
        return _jsonable(item)

    def _diagnostics_payload(
        self,
        *,
        tool: str,
        correlation_id: str,
        started: float,
        k: int | None,
        result_count: int,
        retrieval_ms: int = 0,
        rerank_ms: int = 0,
        retrieval_diagnostics: dict | None = None,
        max_chars: int = EXCERPT_LENGTH,
    ) -> dict[str, Any]:
        clean_diag = self._sanitize_retrieval_diagnostics(
            retrieval_diagnostics,
            max_chars=max_chars,
        )
        total_ms = _now_ms(started)
        return {
            "schema": "kiara-retrieval-only-diagnostics",
            "schema_version": 1,
            "route": "retrieval-only",
            "tool": tool,
            "correlation_id": correlation_id,
            "embedding_model": self._embedding_model_name(),
            "k": k,
            "result_count": result_count,
            "duration_ms": total_ms,
            "latencies_ms": {
                "retrieval": retrieval_ms,
                "reranking": rerank_ms,
                "total": total_ms,
            },
            "path_states": self._path_states(clean_diag),
            "path_latencies_ms": self._path_latencies(clean_diag),
            "retrieval_diagnostics": clean_diag,
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @staticmethod
    def _content_coverage(
        results: list[dict], *, available_chunks: int, include_text: bool, scope: str
    ) -> dict:
        """Meldet ausgelassene Belege, ohne ein fremdes Modellfenster zu behaupten.

        Die Abdeckung bezieht sich nur auf die ausgewaehlten Suchtreffer bzw.
        das angeforderte Dokumentfenster, nie auf den gesamten Datenbestand.
        """
        truncated = sum(
            bool(item.get("text_truncated" if include_text else "excerpt_truncated"))
            for item in results
        )
        omitted = max(0, available_chunks - len(results))
        coverage = {
            "scope": scope,
            "content_complete": not (truncated or omitted),
            "available_chunks": available_chunks,
            "returned_chunks": len(results),
            "truncated_chunks": truncated,
            "omitted_chunks": omitted,
            "text_mode": "full_text" if include_text else "excerpts",
            "caller_context_window_known": False,
        }
        warnings = []
        if truncated or omitted:
            warnings.append(
                {
                    "code": "retrieval_content_incomplete",
                    "component": "context",
                    "reason": "response_limit" if include_text else "excerpt_only",
                    "message": (
                        "Die zurückgegebenen Belege sind wegen der Ausgabelimits unvollständig: "
                        "Texte wurden gekürzt oder Abschnitte ausgelassen. Fehlende Angaben "
                        "sind kein Nachweis, dass sie im Datenbestand fehlen. "
                        "Lade benötigte Bereiche mit get_document_context gezielt nach."
                    ),
                    "count": 1,
                    "details": dict(coverage),
                }
            )
        return {"content_coverage": coverage, "warnings": warnings}

    def search_documents(self, user: dict | None, params: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        correlation_id = str(uuid4())
        self._require_enabled()
        user = self._require_user(user)
        limits = self._limits()
        query = self._required_str(params, "query", max_chars=limits["max_query_length"])
        self._reject_trivial_query(query)
        k = self._int_param(params, "k", 10, min_value=1, max_value=limits["max_k"])
        max_chars = self._int_param(
            params,
            "max_chars_per_result",
            limits["max_chars_per_result"],
            min_value=1,
            max_value=limits["max_chars_per_result"],
        )
        include_text = self._include_text(params)
        selected_ids = self._selected_datasource_ids(params)
        filter_ds_ids = self._allowed_datasource_filter(user, selected_ids)
        file_type = self._optional_str(params, "file_type", max_chars=_FILE_TYPE_MAX_CHARS)
        directory = self._optional_str(params, "directory", max_chars=_FILTER_STRING_MAX_CHARS)
        source_contains = self._optional_str(
            params,
            "source_contains",
            max_chars=_FILTER_STRING_MAX_CHARS,
        )
        source_contains = self._effective_source_filter(source_contains, query)

        retriever = self._retriever()
        reranker = self._reranker()
        fetch_k = k
        if getattr(reranker, "enabled", False):
            multiplier = self._config.get("retrieval", {}).get("rerank_fetch_multiplier", 3)
            fetch_k = k * int(multiplier)

        retrieval_started = time.monotonic()
        try:
            raw_results = retriever.retrieve(
                query=query,
                k=fetch_k,
                filter_dir=directory,
                filter_type=file_type,
                filter_datasource_ids=filter_ds_ids,
                filter_source=source_contains,
            )
        except RetrievalOnlyError:
            raise
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.error("Retrieval-only Suche fehlgeschlagen: %s", exc, exc_info=True)
            raise RetrievalOnlyError(
                "retrieval_failed",
                "Retrieval konnte nicht ausgefuehrt werden.",
                503,
            ) from exc
        retrieval_ms = _now_ms(retrieval_started)

        rerank_started = time.monotonic()
        if getattr(reranker, "enabled", False):
            final_results = reranker.rerank(query, raw_results, top_k=k)
        else:
            final_results = raw_results[:k]
        rerank_ms = _now_ms(rerank_started)

        final_results = self._with_final_context_ranks(final_results)
        retrieval_diagnostics = None
        getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
        if callable(getter):
            retrieval_diagnostics = getter()

        results = [
            self._format_result(
                result,
                rank,
                include_text=include_text,
                max_chars=max_chars,
            )
            for rank, result in enumerate(final_results, 1)
        ]

        response = {
            "ok": True,
            "route": "retrieval-only",
            "tool": "search_documents",
            "query": query,
            "results": results,
            **self._content_coverage(
                results,
                available_chunks=len(final_results),
                include_text=include_text,
                scope="selected_search_hits",
            ),
            "diagnostics": self._diagnostics_payload(
                tool="search_documents",
                correlation_id=correlation_id,
                started=started,
                k=k,
                result_count=len(results),
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                retrieval_diagnostics=retrieval_diagnostics,
                max_chars=max_chars,
            ),
            "limits": {
                "max_k": limits["max_k"],
                "max_chars_per_result": limits["max_chars_per_result"],
                "raw_chunks_allowed": self._web_config.get("mcp_retrieval_allow_raw_chunks")
                is True,
            },
        }
        self._audit(user, "search_documents", query, response, results, correlation_id)
        return response

    def list_datasources(self, user: dict | None) -> dict[str, Any]:
        started = time.monotonic()
        correlation_id = str(uuid4())
        self._require_enabled()
        user = self._require_user(user)
        allowed = self._allowed_datasource_filter(user, None)
        try:
            rows = self._datasource_repo_instance().list_active_with_chunks()
        except Exception as exc:
            logger.error("Retrieval-only Datasource-Liste fehlgeschlagen: %s", exc, exc_info=True)
            raise RetrievalOnlyError(
                "datasource_unavailable",
                "Datenquellen konnten nicht geladen werden.",
                503,
            ) from exc
        if allowed is not None:
            allowed_set = set(allowed)
            rows = [row for row in rows if row.get("id") in allowed_set]
        datasources = [
            {
                "id": int(row["id"]),
                "name": sanitize_utf8(str(row.get("name") or "")),
                "type": sanitize_utf8(str(row.get("type") or "")),
                "chunk_count": int(row.get("chunk_count") or 0),
                "file_count": int(row.get("file_count") or 0),
            }
            for row in rows
        ]
        response = {
            "ok": True,
            "route": "retrieval-only",
            "tool": "list_datasources",
            "datasources": datasources,
            "diagnostics": self._diagnostics_payload(
                tool="list_datasources",
                correlation_id=correlation_id,
                started=started,
                k=None,
                result_count=len(datasources),
            ),
        }
        self._audit(
            user, "list_datasources", "list_datasources", response, datasources, correlation_id
        )
        return response

    def get_document_context(self, user: dict | None, params: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        correlation_id = str(uuid4())
        self._require_enabled()
        user = self._require_user(user)
        limits = self._limits()
        source = self._required_str(params, "source", max_chars=_FILTER_STRING_MAX_CHARS)
        selected_ids = self._selected_datasource_ids(params)
        if not selected_ids:
            raise RetrievalOnlyError("invalid_request", "datasource_id ist erforderlich.", 400)
        datasource_id = selected_ids[0]
        filter_ds_ids = self._allowed_datasource_filter(user, [datasource_id])
        chunk_index = self._int_param(
            params,
            "chunk_index",
            None,
            min_value=0,
            max_value=10_000_000,
        )
        window = self._int_param(
            params,
            "window",
            2,
            min_value=0,
            max_value=limits["max_context_window"],
        )
        max_chunks = self._int_param(
            params,
            "max_chunks",
            limits["max_context_chunks"],
            min_value=1,
            max_value=limits["max_context_chunks"],
        )
        max_chars_total = self._int_param(
            params,
            "max_chars_total",
            limits["max_chars_total"],
            min_value=1,
            max_value=limits["max_chars_total"],
        )
        include_text = self._include_text(params)

        retriever = self._retriever()
        retrieval_started = time.monotonic()
        try:
            chunks = retriever.load_chunks_by_source(source, filter_ds_ids)
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.error("Retrieval-only Kontext fehlgeschlagen: %s", exc, exc_info=True)
            raise RetrievalOnlyError(
                "retrieval_failed",
                "Dokumentkontext konnte nicht geladen werden.",
                503,
            ) from exc
        retrieval_ms = _now_ms(retrieval_started)

        eligible = []
        for chunk in chunks:
            meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            if meta.get("source") != source:
                continue
            if datasource_id not in _normalize_ds_ids(meta.get("ds_ids")):
                continue
            try:
                ci = int(meta.get("chunk_index"))
            except (TypeError, ValueError):
                continue
            if chunk_index - window <= ci <= chunk_index + window:
                eligible.append(chunk)

        eligible.sort(key=lambda item: int(item.get("metadata", {}).get("chunk_index", 0)))
        available_chunks = len(eligible)
        if len(eligible) > max_chunks:
            eligible = sorted(
                eligible,
                key=lambda item: abs(
                    int(item.get("metadata", {}).get("chunk_index", 0)) - chunk_index
                ),
            )[:max_chunks]
            eligible.sort(key=lambda item: int(item.get("metadata", {}).get("chunk_index", 0)))

        remaining_chars = max_chars_total
        context_chunks = []
        for rank, chunk in enumerate(eligible, 1):
            max_chars = max(1, remaining_chars) if include_text else EXCERPT_LENGTH
            item = self._format_result(
                chunk,
                rank,
                include_text=include_text,
                max_chars=max_chars,
            )
            if include_text:
                used = len(item.get("text", ""))
                remaining_chars -= used
                if remaining_chars <= 0:
                    context_chunks.append(item)
                    break
            context_chunks.append(item)

        response = {
            "ok": True,
            "route": "retrieval-only",
            "tool": "get_document_context",
            "source": source,
            "datasource_id": datasource_id,
            "chunk_index": chunk_index,
            "window": window,
            "context_chunks": context_chunks,
            **self._content_coverage(
                context_chunks,
                available_chunks=available_chunks,
                include_text=include_text,
                scope="requested_document_window",
            ),
            "diagnostics": self._diagnostics_payload(
                tool="get_document_context",
                correlation_id=correlation_id,
                started=started,
                k=max_chunks,
                result_count=len(context_chunks),
                retrieval_ms=retrieval_ms,
                max_chars=EXCERPT_LENGTH,
            ),
            "limits": {
                "max_context_window": limits["max_context_window"],
                "max_context_chunks": limits["max_context_chunks"],
                "max_chars_total": limits["max_chars_total"],
                "raw_chunks_allowed": self._web_config.get("mcp_retrieval_allow_raw_chunks")
                is True,
            },
        }
        self._audit(
            user,
            "get_document_context",
            source,
            response,
            context_chunks,
            correlation_id,
        )
        return response

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit(
        self,
        user: dict,
        tool: str,
        query: str,
        response: dict,
        sources: list[dict],
        correlation_id: str,
    ) -> None:
        duration_ms = int(response.get("diagnostics", {}).get("duration_ms") or 0)
        context_sources = [
            {
                "source": item.get("source") or item.get("path") or "",
                "datasource_ids": item.get("datasource_ids")
                or ([item.get("id")] if tool == "list_datasources" and item.get("id") else []),
                "score": item.get("score"),
                "score_type": item.get("score_type"),
                "rank": item.get("rank"),
                "filename": item.get("filename"),
            }
            for item in sources[:100]
        ]
        context_sources_json = json.dumps(context_sources, ensure_ascii=False, default=str)
        if len(context_sources_json) > MAX_AUDIT_TEXT:
            context_sources_json = context_sources_json[:MAX_AUDIT_TEXT] + "\n[...gekuerzt]"
        answer_excerpt = f"{tool}: {len(sources)} results"
        try:
            from kiara.repositories.base import RawSQL

            self._audit_repo_instance().create(
                user_id=user["id"],
                question_hash=hashlib.sha256(query.encode()).hexdigest(),
                question_excerpt=query[:EXCERPT_LENGTH],
                answer_excerpt=answer_excerpt[:EXCERPT_LENGTH],
                routed_to="mcp:retrieval",
                model_used=self._embedding_model_name(),
                sources_count=len(sources),
                chunks_retrieved=len(sources),
                duration_ms=duration_ms,
                chain_type="retrieval-only",
                full_prompt="",
                full_answer=answer_excerpt,
                context_sources=context_sources_json,
                correlation_id=correlation_id,
                created_at=RawSQL("NOW()"),
            )
        except Exception as exc:
            logger.warning("Retrieval-only Audit-Log fehlgeschlagen: %s", exc)
