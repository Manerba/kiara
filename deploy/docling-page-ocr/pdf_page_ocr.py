"""Bounded, page-selective PDF OCR for the Docling conversion service."""

import csv
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

MAX_PAGES = 128
MAX_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 18_000_000
MAX_SECONDS = 600
_SLOTS = threading.BoundedSemaphore(2)
_LANGUAGE = {"de": "deu", "en": "eng", "fr": "fra", "it": "ita", "es": "spa"}


def read_words(tsv):
    lines = {}
    credible = False
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if row["level"] != "5" or not text or float(row["conf"]) < 45:
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        lines.setdefault(key, []).append(text)
        if float(row["conf"]) >= 70 and any(c.isalnum() for c in text):
            credible = True
    return "\n".join(" ".join(words) for words in lines.values()) if credible else ""


def recognize_page(page, lang):
    width, height = page.get_size()
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Invalid PDF page dimensions")
    scale = min(150 / 72, math.sqrt(MAX_PIXELS / (width * height)), 16384 / max(width, height))
    bitmap = page.render(scale=scale, grayscale=True)
    try:
        with bitmap.to_pil() as image:
            png = io.BytesIO()
            image.save(png, "PNG")
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", lang, "--psm", "11", "tsv"],
            input=png.getvalue(),
            capture_output=True,
            timeout=45,
            check=True,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
        return read_words(result.stdout.decode("utf-8", "replace"))
    finally:
        bitmap.close()


def extract(stream, pages, count, lang):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(stream)
    try:
        if count == 0:
            if not 1 <= len(pdf) <= MAX_PAGES:
                raise ValueError("PDF exceeds the full-page OCR budget")
            count = len(pdf)
            pages = range(1, count + 1)
        if len(pdf) != count:
            raise ValueError(f"Page count differs: PDFium={len(pdf)}, requested={count}")
        result = []
        for number in pages:
            page = pdf[number - 1]
            try:
                text = recognize_page(page, lang)
                result.append(
                    {
                        "page": number,
                        "status": "text" if text else "no_text",
                        "text": text,
                    }
                )
            finally:
                page.close()
        return result
    finally:
        pdf.close()


def register(app, require_auth):
    from typing import Annotated

    from fastapi import Depends, Form, HTTPException, UploadFile

    @app.post("/v1/convert/pdf-pages/ocr", dependencies=[Depends(require_auth)])
    def page_ocr(
        file: UploadFile,
        page_count: Annotated[int, Form(ge=0)],
        languages: Annotated[list[str], Form()],
        pages: Annotated[list[int] | None, Form()] = None,
    ):
        pages = pages or []
        if (
            bool(pages) != bool(page_count)
            or len(pages) > MAX_PAGES
            or len(set(pages)) != len(pages)
            or (pages and min(pages) < 1)
            or (pages and max(pages) > page_count)
            or file.size is None
            or file.size > MAX_BYTES
        ):
            raise HTTPException(422, "Invalid page selection or PDF too large")
        codes = list(dict.fromkeys(_LANGUAGE.get(s, s) for s in languages))
        model_dir = Path(os.environ["TESSDATA_PREFIX"])
        if not codes or any(
            not s.isascii()
            or not s.isalpha()
            or len(s) != 3
            or not (model_dir / f"{s}.traineddata").is_file()
            for s in codes
        ):
            raise HTTPException(422, "Requested OCR language is unavailable")
        if not _SLOTS.acquire(timeout=3):
            raise HTTPException(503, "PDF OCR workers are busy")
        try:
            file.file.seek(0)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    json.dumps(pages),
                    str(page_count),
                    "+".join(codes),
                ],
                stdin=file.file,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                output, _ = proc.communicate(timeout=MAX_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise HTTPException(422, "Page OCR time budget exhausted")
            if proc.returncode:
                raise HTTPException(503, "Page OCR worker unavailable")
            data = json.loads(output)
            if "error" in data:
                raise HTTPException(data["http_status"], data["error"])
            return data
        finally:
            _SLOTS.release()


if __name__ == "__main__":
    import pypdfium2 as pdfium

    try:
        rows = extract(
            io.BytesIO(sys.stdin.buffer.read()),
            json.loads(sys.argv[1]),
            int(sys.argv[2]),
            sys.argv[3],
        )
        print(
            json.dumps(
                {"pages": rows, "page_count": int(sys.argv[2]) or len(rows)},
                ensure_ascii=False,
            )
        )
    except (pdfium.PdfiumError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"http_status": 422, "error": str(exc)[:240]}))
