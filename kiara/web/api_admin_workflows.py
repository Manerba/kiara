"""Admin-only Zusatzrouten fuer Dokumenten-Workflows.

Die JSON-Workflow-Endpunkte werden im Admin-Prozess aus AdminFunctions
generiert. FileResponse-basierte Downloads bleiben als dedizierte
Admin-Route erhalten, weil der Auto-Router JSON-Antworten erzeugt.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from kiara.config import load_config
from kiara.web.auth import get_admin_user
from kiara.workflows.exporter import DocumentExporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["admin-workflows"])


@router.get("/exports/{filename}/download")
async def download_export(filename: str, user=Depends(get_admin_user)):
    """Exportierte Workflow-Datei im Admin-Prozess herunterladen."""
    config = load_config()
    exporter = DocumentExporter(config)
    out_dir = Path(exporter.output_dir).resolve()
    safe_path = (out_dir / filename).resolve()

    if not safe_path.is_relative_to(out_dir):
        raise HTTPException(status_code=400, detail="Ungueltiger Dateiname.")
    if not safe_path.exists() or not safe_path.is_file():
        raise HTTPException(status_code=404, detail=f"Datei '{filename}' nicht gefunden.")

    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    return FileResponse(
        path=str(safe_path),
        filename=filename,
        media_type=media_types.get(safe_path.suffix.lower(), "application/octet-stream"),
    )
