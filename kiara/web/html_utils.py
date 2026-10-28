"""Request-spezifische HTML-Helfer fuer statische Templates."""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import Mapping

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)


def get_request_csp_nonce(request: Request | None) -> str:
    """Liest den request-spezifischen CSP-Nonce aus ``request.state``."""
    if request is None:
        return ""
    return str(getattr(getattr(request, "state", None), "csp_nonce", "") or "")


def render_html_file(
    path: Path,
    request: Request | None = None,
    *,
    replacements: Mapping[str, object] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    """Rendert eine statische HTML-Datei mit einfacher Platzhalter-Ersetzung."""
    try:
        html = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("HTML-Template nicht gefunden: %s", path)
        raise HTTPException(status_code=500, detail="Template nicht verfuegbar")
    except PermissionError as exc:
        logger.error("HTML-Template nicht lesbar: %s", exc)
        raise HTTPException(status_code=500, detail="Template nicht lesbar")

    values: dict[str, object] = {"csp_nonce": get_request_csp_nonce(request)}
    if replacements:
        values.update(replacements)

    for key, value in values.items():
        html = html.replace(f"{{{{ {key} }}}}", escape(str(value)))

    return HTMLResponse(
        content=html,
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )
