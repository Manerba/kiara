"""Allgemeine Dokument-Loader (Docx, PDF, XLSX, ODT, PPTX, CSV, JSON, Plaintext, Code)."""

import codecs
import csv
import io
import json
import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


_CODE_LANGUAGES = {
    ".cs": "csharp",
    ".vb": "vb",
    ".ps1": "powershell",
    ".py": "python",
    ".pl": "perl",
    ".js": "javascript",
    ".bat": "batch",
    ".sh": "bash",
    ".css": "css",
    ".html": "html",
    ".sln": "text",
}


# UTF-32 vor UTF-16: das UTF-32-LE-BOM beginnt mit dem UTF-16-LE-BOM.
_WIDE_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_SNIFF_BYTES = 4096
# BOM-loses UTF-16 erkennt man an NUL-Bytes auf nur einer der beiden Byte-Positionen.
_UTF16_NUL_MIN_RATIO = 0.10
_UTF16_NUL_OTHER_MAX_RATIO = 0.01
_MAX_CONTROL_CHAR_RATIO = 0.05


def _sniff_wide_encoding(sample: bytes) -> str | None:
    """Erkennt UTF-16/UTF-32 per BOM oder (nur UTF-16) per NUL-Byte-Muster.

    Gibt den Codec zurueck. Bei BOM wird der BOM-konsumierende Codec
    (``utf-16``/``utf-32``) geliefert, ohne BOM ``utf-16-le``/``utf-16-be``.
    """
    for bom, codec in _WIDE_BOMS:
        if sample.startswith(bom):
            return codec
    n = len(sample) // 2 * 2
    if n < 4:
        return None
    even_nul = sample[0:n:2].count(0) / (n // 2)
    odd_nul = sample[1:n:2].count(0) / (n // 2)
    if odd_nul >= _UTF16_NUL_MIN_RATIO and even_nul <= _UTF16_NUL_OTHER_MAX_RATIO:
        return "utf-16-le"
    if even_nul >= _UTF16_NUL_MIN_RATIO and odd_nul <= _UTF16_NUL_OTHER_MAX_RATIO:
        return "utf-16-be"
    return None


def _decode_wide(raw: bytes, codec: str) -> str | None:
    """Dekodiert UTF-16/UTF-32; None, wenn das Ergebnis kein plausibler Text ist."""
    width = 4 if codec.startswith("utf-32") else 2
    try:
        text = raw[: len(raw) // width * width].decode(codec)
    except UnicodeDecodeError:
        return None
    # Binaerdaten mit zufaellig passendem NUL-Muster ergeben Steuerzeichen-Muell.
    controls = sum(1 for c in text if c < " " and c not in "\t\r\n\f")
    if text and controls / len(text) > _MAX_CONTROL_CHAR_RATIO:
        return None
    return text


def _load_wide_text(path: str) -> str | None:
    """Laedt UTF-16/UTF-32-Textdateien, sonst None."""
    with open(path, "rb") as f:
        codec = _sniff_wide_encoding(f.read(_SNIFF_BYTES))
        if codec is None:
            return None
        f.seek(0)
        return _decode_wide(f.read(), codec)


def _load_plaintext(path: str) -> str:
    """Load plain text files with automatic encoding detection."""
    wide = _load_wide_text(path)
    if wide is not None:
        return wide
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        pass
    raw = Path(path).read_bytes()
    # CP1252 heuristic: bytes 0x80-0x9F are printable in CP1252 (€ „ " " – —)
    # but C1 controls in ISO-8859-1. charset_normalizer can misidentify short
    # German texts containing these bytes.
    if any(0x80 <= b <= 0x9F for b in raw):
        try:
            return raw.decode("cp1252")
        except (UnicodeDecodeError, LookupError):
            pass
    try:
        from charset_normalizer import from_bytes

        result = from_bytes(raw)
        best = result.best()
        if best is not None:
            return str(best)
    except (ImportError, OSError, ValueError, LookupError, TypeError):
        pass
    text = raw.decode("utf-8", errors="replace")
    if "\ufffd" in text:
        count = text.count("\ufffd")
        logger.warning(
            "Encoding-Fallback fuer %s: %d Zeichen durch U+FFFD ersetzt"
            " (charset_normalizer fehlgeschlagen, UTF-8 errors='replace' aktiv)",
            path,
            count,
        )
    return text


def _load_code(path: str) -> str:
    """Load code files with language annotation."""
    ext = Path(path).suffix.lower()
    lang = _CODE_LANGUAGES.get(ext, "")
    content = _load_plaintext(path)
    if lang:
        return f"```{lang}\n# File: {Path(path).name}\n{content}\n```"
    return content


_RELATIONSHIP_TAG = re.compile(rb"<Relationship\b[^>]*>")
_INTERNAL_TARGET = re.compile(rb"""\bTarget=["']#""")


def _mark_internal_targets_external(match: re.Match) -> bytes:
    tag = match.group(0)
    if not _INTERNAL_TARGET.search(tag) or b"TargetMode=" in tag:
        return tag
    end = -2 if tag.endswith(b"/>") else -1
    return tag[:end].rstrip() + b' TargetMode="External"' + tag[end:]


def _repair_internal_hyperlink_targets(data: bytes) -> bytes:
    """Markiert Sprungmarken-Relationships (Target="#...") in .rels-Dateien als External.

    Word/LibreOffice speichern Hyperlinks auf Bookmarks teils ohne TargetMode="External".
    python-docx sucht dann ein Package-Teil mit diesem Namen und bricht mit KeyError ab.
    """
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            blob = src.read(info.filename)
            if info.filename.endswith(".rels"):
                blob = _RELATIONSHIP_TAG.sub(_mark_internal_targets_external, blob)
            dst.writestr(info, blob)
    return out.getvalue()


def _open_docx(path: str):
    """Oeffnet eine .docx; repariert bei KeyError Sprungmarken-Relationships und versucht es erneut."""
    from docx import Document

    try:
        with open(path, "rb") as f:
            return Document(f)
    except KeyError:
        data = Path(path).read_bytes()
        repaired = _repair_internal_hyperlink_targets(data)
        if repaired == data:
            raise
        logger.info("DOCX %s: interne Hyperlink-Relationships repariert", path)
        return Document(io.BytesIO(repaired))


def _load_docx(path: str) -> str:
    """Load .docx files."""
    doc = _open_docx(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n\n".join(paragraphs)


def _load_pdf(path: str) -> str:
    """Load .pdf files."""
    from pypdf import PdfReader

    with open(path, "rb") as f:
        reader = PdfReader(f)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.replace("\x00", ""))  # pypdf Encoding-Artefakte
        return "\n\n".join(pages)


def _load_xlsx(path: str) -> str:
    """Load .xlsx/.xlsm files."""
    from openpyxl import load_workbook

    wb = None
    stream = None
    try:
        # openpyxl prueft String-Endungen; ein XLSX-Inhalt darf auch .xls heissen.
        stream = open(path, "rb")
        wb = load_workbook(stream, read_only=True, data_only=True)
        sheets = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))
            if rows:
                sheets.append(f"## Sheet: {sheet_name}\n" + "\n".join(rows))
        return "\n\n".join(sheets)
    finally:
        if wb is not None:
            wb.close()
        if stream is not None:
            stream.close()


def _load_xls(path: str) -> str:
    """Liest BIFF/XLS (auch bei falscher Dateiendung), ohne Office-Prozess."""
    import xlrd

    workbook = xlrd.open_workbook(path, on_demand=True, logfile=io.StringIO())
    try:
        sheets = []
        for sheet in workbook.sheets():
            rows = []
            for row_index in range(sheet.nrows):
                cells = []
                for cell in sheet.row(row_index):
                    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
                        value = ""
                    elif cell.ctype == xlrd.XL_CELL_DATE:
                        value = xlrd.xldate_as_datetime(cell.value, workbook.datemode).isoformat(
                            sep=" "
                        )
                    elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                        value = "TRUE" if cell.value else "FALSE"
                    elif cell.ctype == xlrd.XL_CELL_ERROR:
                        value = xlrd.error_text_from_code.get(cell.value, "#ERROR")
                    elif cell.ctype == xlrd.XL_CELL_NUMBER and float(cell.value).is_integer():
                        value = str(int(cell.value))
                    else:
                        value = str(cell.value)
                    cells.append(value)
                if any(value.strip() for value in cells):
                    rows.append(" | ".join(cells))
            if rows:
                sheets.append(f"## Sheet: {sheet.name}\n" + "\n".join(rows))
        return "\n\n".join(sheets)
    finally:
        workbook.release_resources()


def _load_odt(path: str) -> str:
    """Load .odt files."""
    from odf import text as odf_text
    from odf.opendocument import load

    doc = load(path)
    paragraphs = []
    for p in doc.getElementsByType(odf_text.P):
        content = ""
        for node in p.childNodes:
            if hasattr(node, "data"):
                content += node.data
            else:
                content += str(node)
        if content.strip():
            paragraphs.append(content)
    return "\n\n".join(paragraphs)


def _load_pptx(path: str) -> str:
    """Load .pptx files."""
    from pptx import Presentation

    with open(path, "rb") as f:
        prs = Presentation(f)
        slides = []
        for i, slide in enumerate(prs.slides, 1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            texts.append(text)
            if texts:
                slides.append(f"## Slide {i}\n" + "\n".join(texts))
    return "\n\n".join(slides)


def _load_csv_file(path: str) -> str:
    """Load .csv files as readable text."""
    content = _load_plaintext(path)
    try:
        reader = csv.reader(io.StringIO(content))
        rows = []
        for row in reader:
            rows.append(" | ".join(row))
        return "\n".join(rows)
    except csv.Error:
        return content


def _load_json_file(path: str) -> str:
    """Load .json files with pretty formatting."""
    content = _load_plaintext(path)
    try:
        data = json.loads(content)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content
