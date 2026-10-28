"""Repository fuer additive S4-Indexvertragsdiagnosen."""

from __future__ import annotations

from typing import Any

from kiara.retrieval.index_compatibility import list_persisted_contracts
from kiara.shared.db import get_db


class IndexContractRepository:
    """Detached reads; writes remain in fenced migration/backfill transactions."""

    def list_all(self) -> list[dict[str, Any]]:
        conn = get_db()
        try:
            return [dict(row) for row in list_persisted_contracts(conn)]
        finally:
            conn.close()

    def get_by_logical_key(self, logical_index_key: str) -> dict[str, Any] | None:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT c.*, e.max_abs_error, e.max_cosine_distance, "
                "e.abs_tolerance, e.cosine_tolerance, "
                "e.collection_config_sha256, e.sample_set_sha256, "
                "e.contract_snapshot_sha256 AS evidence_contract_snapshot_sha256, "
                "e.references_json_text, e.diagnostics_json_text, "
                "e.created_at AS evidence_created_at "
                "FROM index_contracts c "
                "JOIN index_contract_evidence e ON e.id = c.evidence_id "
                "WHERE c.logical_index_key = %s",
                (logical_index_key,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_evidence(self, logical_index_key: str) -> list[dict[str, Any]]:
        conn = get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM index_contract_evidence "
                "WHERE logical_index_key = %s ORDER BY created_at DESC, id DESC",
                (logical_index_key,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
