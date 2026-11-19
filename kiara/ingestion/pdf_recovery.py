"""Recover text from individual PDF pages after a document conversion failure."""

from __future__ import annotations

import re
from itertools import groupby

import httpx
from pypdf import PdfReader

_IMAGE_MARKER = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)


def has_text(markdown: str) -> bool:
    return any(char.isalnum() for char in _IMAGE_MARKER.sub("", markdown))


def page_ranges(numbers) -> str:
    parts = []
    for _, group in groupby(enumerate(sorted(numbers)), lambda item: item[1] - item[0]):
        pages = [page for _, page in group]
        parts.append(str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}")
    return ",".join(parts)


class PageRecoveryError(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind


def _request_ocr(path, backend, pages, page_count):
    if len(pages) > 128:
        raise PageRecoveryError(
            "document_error", f"OCR-Seitenlimit: {len(pages)} Seiten angefordert"
        )
    try:
        with open(path, "rb") as source:
            response = httpx.post(
                backend["url"].rstrip("/") + "/v1/convert/pdf-pages/ocr",
                files={"file": ("document.pdf", source, "application/pdf")},
                data={
                    "pages": [str(page) for page in pages],
                    "page_count": str(page_count),
                    "languages": backend.get("ocr_lang", ["de", "en"]),
                },
                timeout=min(620, backend.get("timeout") or 620),
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        kind = (
            "document_error"
            if code == 422
            else "http_error"
            if code >= 500 or code in (408, 409, 425, 429)
            else "request_error"
        )
        raise PageRecoveryError(kind, f"Seiten-OCR HTTP {code}: {exc.response.text[:320]}") from exc
    except httpx.TimeoutException as exc:
        raise PageRecoveryError("timeout", "Zeitbudget der Seiten-OCR erreicht") from exc
    except httpx.RequestError as exc:
        raise PageRecoveryError("connection", f"Seiten-OCR: {exc}") from exc

    try:
        payload = response.json()
        rows = payload["pages"]
        if not page_count:
            count = payload["page_count"]
            if type(count) is not int or not 1 <= count <= 128:
                raise ValueError("Ungueltige PDFium-Seitenzahl")
            pages = range(1, count + 1)
        if not isinstance(rows, list) or len(rows) != len(pages):
            raise ValueError("Unvollstaendige Seitenliste")
        text = {}
        for row in rows:
            n, status, content = row["page"], row["status"], row["text"]
            if type(n) is not int or n in text or not isinstance(content, str):
                raise ValueError("Ungueltige OCR-Seite")
            if status not in {"text", "no_text"} or (status == "text") != has_text(content):
                raise ValueError("Inkonsistenter OCR-Status")
            text[n] = content if status == "text" else ""
        if set(text) != set(pages):
            raise ValueError("OCR-Seitennummern stimmen nicht ueberein")
        return text
    except (KeyError, ValueError, TypeError) as exc:
        raise PageRecoveryError("http_error", f"Ungueltige Seiten-OCR-Antwort: {exc}") from exc


def recover_pdf(path, result, backend, metadata):
    """Keep Docling's usable Markdown and read only pages missing from that result."""
    primary = result.content or ""
    if not has_text(primary):
        primary = ""
    try:
        with open(path, "rb") as source:
            reader = PdfReader(source)
            count = len(reader.pages)
            if count <= 0 or (result.page_count and result.page_count != count):
                raise ValueError(f"Seitenzahl pypdf={count}, Docling={result.page_count}")
            requested = result.missing_pages if primary else range(1, count + 1)
            texts = {}
            for n in requested:
                try:
                    value = (reader.pages[n - 1].extract_text() or "").replace("\x00", "").strip()
                except Exception as exc:
                    if not backend.get("do_ocr", True):
                        raise ValueError(f"Seite {n}: {exc}") from exc
                    value = ""  # Ein Fehler im Textlayer muss das Rendern nicht verhindern.
                texts[n] = value if has_text(value) else ""
    except Exception as exc:
        reason = f"PDF-Loader: {type(exc).__name__}: {exc}"
        if not backend.get("do_ocr", True) or not (result.task_id or result.page_count):
            raise PageRecoveryError("document_error", reason) from exc
        # PDFium may still render a page tree that pypdf cannot read.
        try:
            if primary and result.page_count:
                recovered = _request_ocr(
                    path, backend, list(result.missing_pages), result.page_count
                )
                count = result.page_count
            else:
                recovered = _request_ocr(path, backend, [], 0)
                count = len(recovered)
        except PageRecoveryError as ocr_error:
            raise PageRecoveryError(ocr_error.kind, f"{reason}; {ocr_error}") from ocr_error
        texts = recovered
        native, missing = [], []
        ocr_pages = [n for n, text in texts.items() if text]
    else:
        native = [n for n, text in texts.items() if text]
        missing = [n for n, text in texts.items() if not text]
        ocr_pages = []

    if missing and backend.get("do_ocr", True):
        try:
            recovered = _request_ocr(path, backend, missing, count)
        except PageRecoveryError as exc:
            raise PageRecoveryError(exc.kind, f"{exc} (Seiten {page_ranges(missing)})") from exc
        texts.update(recovered)
        ocr_pages = [n for n, text in recovered.items() if text]
    unresolved = [n for n, text in texts.items() if not text]
    if not primary and not any(texts.values()):
        note = "nach Seiten-OCR" if backend.get("do_ocr", True) else "bei deaktivierter OCR"
        raise PageRecoveryError(
            "document_error", f"Kein verwertbarer Text {note} (Seiten {page_ranges(unresolved)})"
        )
    metadata.update(
        parser_fallback=True,
        pdf_page_count=count,
        pdf_native_pages=page_ranges(native),
        pdf_ocr_pages=page_ranges(ocr_pages),
        pdf_without_text=page_ranges(unresolved),
    )
    note = [f"{count} Seiten"]
    for label, pages in (("pypdf", native), ("OCR", ocr_pages), ("ohne Text", unresolved)):
        if pages:
            note.append(f"{label}: {page_ranges(pages)}")
    metadata["pdf_recovery_note"] = " | ".join(note)
    additions = [f"[Seite {n}]\n{text}" for n, text in sorted(texts.items()) if text]
    return "\n\n".join(([primary] if primary else []) + additions)
