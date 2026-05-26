"""Workflow-Export-Repository.

Ersetzt SQL in: web/api_workflows.py
"""

from __future__ import annotations

from kiara_db.base import BaseRepository


class WorkflowExportRepository(BaseRepository):
    table = "workflow_exports"
    default_sort = "created_at DESC"
