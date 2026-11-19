"""Content-basierte Dateierkennung.

Erkennt ob eine Datei lesbaren Text enthaelt, unabhaengig von der Dateiendung.
Nutzt charset_normalizer fuer Encoding-Erkennung und einen Non-Printable-Ratio
Check als Binary-Guard.
"""

import codecs
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

_DEFAULT_BINARY_THRESHOLD = 0.05
_DEFAULT_CHUNK_SIZE = 8192

# BOM-Signaturen (laengere zuerst pruefen)
_BOM_UTF8 = b"\xef\xbb\xbf"
_BOM_UTF16_LE = b"\xff\xfe"
_BOM_UTF16_BE = b"\xfe\xff"


@dataclass(frozen=True)
class FileFormat:
    """Inhaltsformat; extension ist die kanonische Endung, nicht der Dateiname."""

    extension: str
    mime_type: str
    encoding: str | None = None


_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".doc": "application/msword",
    ".ppt": "application/vnd.ms-powerpoint",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".html": "text/html",
    ".xml": "application/xml",
    ".svg": "image/svg+xml",
    ".rtf": "application/rtf",
    ".zip": "application/zip",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".tiff": "image/tiff",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}
_TEXT_HINTS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".config",
    ".ini",
    ".log",
    ".cs",
    ".vb",
    ".ps1",
    ".py",
    ".pl",
    ".js",
    ".bat",
    ".sh",
    ".css",
    ".sln",
    ".dxf",
    ".step",
    ".stp",
    ".iges",
    ".igs",
}


def _format(extension: str, encoding: str | None = None) -> FileFormat:
    return FileFormat(
        extension,
        _MIME_TYPES.get(extension, "text/plain" if encoding else "application/octet-stream"),
        encoding,
    )


def _zip_format(path: str) -> FileFormat:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "mimetype" in names:
            if archive.getinfo("mimetype").file_size > 256:
                raise ValueError("Ungueltiger OpenDocument-MIME-Eintrag")
            mime = archive.read("mimetype").decode("ascii").strip()
            for extension in (".odt", ".ods", ".odp"):
                if mime == _MIME_TYPES[extension]:
                    return _format(extension)
        if "[Content_Types].xml" in names:
            if archive.getinfo("[Content_Types].xml").file_size > 1024 * 1024:
                raise ValueError("OOXML Content_Types groesser als 1 MiB")
            root = ElementTree.fromstring(archive.read("[Content_Types].xml"))
            types = {e.get("ContentType", "") for e in root}
            candidates = []
            for extension, part, content_type in (
                (
                    ".docx",
                    "word/document.xml",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
                ),
                (
                    ".xlsx",
                    "xl/workbook.xml",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
                ),
                (
                    ".xlsm",
                    "xl/workbook.xml",
                    "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
                ),
                (
                    ".pptx",
                    "ppt/presentation.xml",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
                ),
            ):
                if part in names and content_type in types:
                    candidates.append(extension)
            if len(candidates) == 1:
                return _format(candidates[0])
            if len(candidates) > 1:
                raise ValueError("Mehrdeutiges OOXML-Dokumentformat")
        return _format(".zip")


def _ole_format(path: str) -> FileFormat:
    import olefile

    with olefile.OleFileIO(path) as document:
        # Nur Root-Streams: eingebettete Excel-Objekte machen Word nicht zu XLS.
        streams = {parts[0].lower() for parts in document.listdir() if len(parts) == 1}
    if {"encryptedpackage", "encryptioninfo"} <= streams:
        return _format(".encrypted-office")
    candidates = [
        ext
        for ext, names in (
            (".xls", {"workbook", "book"}),
            (".doc", {"worddocument"}),
            (".ppt", {"powerpoint document"}),
        )
        if names & streams
    ]
    if len(candidates) > 1:
        raise ValueError("Mehrdeutiges OLE-Dokumentformat")
    return _format(candidates[0] if candidates else ".ole")


def detect_file_format(path: str) -> FileFormat:
    """Erkennt Dokumentcontainer und Text am Inhalt, ohne Dateien umzubenennen.

    Die Endung unterscheidet nur Textdialekte (Code/CSV/CAD). Nicht erkannte
    Binaerdaten landen niemals beim Plaintext-Loader. Defekte Container werfen
    einen Fehler, damit sie nicht als erfolgreiche Leerdatei gespeichert werden.
    """
    with open(path, "rb") as stream:
        sample = stream.read(_DEFAULT_CHUNK_SIZE)
    if sample.lstrip().startswith(b"%PDF-"):
        return _format(".pdf")
    if sample.startswith(b"AutoCAD Binary DXF\r\n\x1a\x00"):
        return _format(".dxf")
    if sample.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return _ole_format(path)
    if sample.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return _zip_format(path)
    # BIFF2/3/4 ohne OLE-Container: BOF, Recordlaenge, Worksheet-Typ.
    if (
        sample[:2] in (b"\x09\x00", b"\x09\x02", b"\x09\x04", b"\x09\x08")
        and len(sample) >= 8
        and sample[6:8] in (b"\x10\x00", b"\x05\x00")
    ):
        return _format(".xls")
    for signature, extension in (
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"II*\x00", ".tiff"),
        (b"MM\x00*", ".tiff"),
        (b"II+\x00", ".tiff"),
        (b"MM\x00+", ".tiff"),
        (b"GIF87a", ".gif"),
        (b"GIF89a", ".gif"),
        (b"\x00\x00\x01\x00", ".ico"),
    ):
        if sample.startswith(signature):
            return _format(extension)
    # "BM" allein trifft auch Text ("BMW-Teileliste"): reservierte Felder und
    # DIB-Kopfgroesse muessen zu einer echten Bitmap passen.
    if (
        sample.startswith(b"BM")
        and len(sample) >= 18
        and sample[6:10] == b"\x00\x00\x00\x00"
        and int.from_bytes(sample[14:18], "little") in {12, 40, 52, 56, 64, 108, 124}
    ):
        return _format(".bmp")
    if sample[:4] == b"RIFF" and sample[8:12] == b"WEBP":
        return _format(".webp")
    if sample.startswith(b"{\\rtf"):
        return _format(".rtf")

    # Auch PPM/PCX/TGA und andere Formate pruefen, bevor ein ASCII-Header
    # versehentlich als Text gilt. Image.open liest nur Header, keine Pixel.
    from PIL import Image

    try:
        with Image.open(path) as image:
            extension = {"JPEG": ".jpg", "TIFF": ".tiff", "JPEG2000": ".jp2"}.get(
                image.format, "." + (image.format or "").lower()
            )
            return FileFormat(extension, Image.MIME.get(image.format, "application/octet-stream"))
    except Exception:
        # Nicht nur UnidentifiedImageError: Text wie "P1 ..." beginnt PIL als
        # PPM zu lesen und wirft ValueError. Das ist kein Bild, weiter mit Text.
        pass

    # Derselbe Wide-Text-Decoder wie im Loader; UTF-32 zuerst, BOM konsumieren.
    from kiara_document_loaders.general import _decode_wide, _sniff_wide_encoding

    encoding = _sniff_wide_encoding(sample)
    text = _decode_wide(sample, encoding) if encoding else None
    if text is None and sample.startswith(codecs.BOM_UTF8):
        try:
            text = codecs.getincrementaldecoder("utf-8-sig")().decode(sample, final=False)
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            pass
    if text is None:
        is_text, encoding, _ = detect_content(path)
        if is_text:
            text = sample.decode(encoding or "utf-8", errors="replace").lstrip("\ufeff")
    if text is not None:
        start = re.sub(r"<!--.*?-->", "", text, flags=re.S).lstrip()
        if re.match(
            r"(?:<\?xml[^>]*>\s*)?(?:<!doctype\s+html\b|<(?:html|head|body|table|p|div|h[1-6])(?:\s|>))",
            start,
            re.I,
        ):
            return _format(".html", encoding)
        if re.match(r"(?:<\?xml[^>]*>\s*)?<svg(?:\s|>)", start, re.I):
            return _format(".svg", encoding)
        if start.startswith("<?xml") or re.match(r"<[A-Za-z_][\w:.-]*(?:\s[^<>]*?)?/?>", start):
            # Markup ohne XML-Deklaration ist bei HTML-Endung ein HTML-Fragment
            # (z. B. beginnend mit <title>, <meta> oder <span>), kein XML.
            if not start.startswith("<?xml") and Path(path).suffix.lower() in {".html", ".htm"}:
                return _format(".html", encoding)
            return _format(".xml", encoding)
        hint = Path(path).suffix.lower()
        return _format(hint if hint in _TEXT_HINTS else ".txt", encoding or "utf-8")

    return _format(".binary")


def detect_content(
    path: str, *, binary_threshold=_DEFAULT_BINARY_THRESHOLD, chunk_size=_DEFAULT_CHUNK_SIZE
) -> tuple[bool, str | None, float]:
    """Erkennt ob eine Datei lesbaren Text enthaelt.

    Args:
        path: Dateipfad.
        binary_threshold: Anteil Non-Printable-Zeichen ab dem binaer (default 0.05).
        chunk_size: Bytes zum Lesen fuer die Erkennung (default 8192).

    Returns:
        Tuple (is_text, encoding, confidence).
        - is_text: True wenn die Datei als Text erkannt wurde.
        - encoding: Erkanntes Encoding (z.B. 'utf-8', 'latin-1') oder None.
        - confidence: Konfidenz der Erkennung (0.0 - 1.0).
    """
    from charset_normalizer import from_bytes  # lazy — war vorher Modul-Ebene

    # Leere Datei
    try:
        st = os.stat(path)
    except (OSError, PermissionError):
        return (False, None, 0.0)

    if not stat.S_ISREG(st.st_mode):
        return (False, None, 0.0)

    size = st.st_size
    if size == 0:
        return (True, "utf-8", 1.0)

    # Erste Bytes lesen
    try:
        with open(path, "rb") as f:
            chunk = f.read(chunk_size)
    except (OSError, PermissionError):
        return (False, None, 0.0)

    # BOM-Check
    encoding = None
    text = None

    if chunk.startswith(_BOM_UTF8):
        encoding = "utf-8-sig"
        try:
            text = chunk.decode("utf-8-sig")
        except UnicodeDecodeError:
            return (False, None, 0.5)
    elif chunk.startswith(_BOM_UTF16_LE):
        encoding = "utf-16-le"
        try:
            raw = chunk[2:]
            raw = raw[: len(raw) - len(raw) % 2]  # gerade Byte-Anzahl
            if not raw:
                return (False, None, 0.5)
            text = raw.decode("utf-16-le")
        except UnicodeDecodeError:
            return (False, None, 0.5)
    elif chunk.startswith(_BOM_UTF16_BE):
        encoding = "utf-16-be"
        try:
            raw = chunk[2:]
            raw = raw[: len(raw) - len(raw) % 2]  # gerade Byte-Anzahl
            if not raw:
                return (False, None, 0.5)
            text = raw.decode("utf-16-be")
        except UnicodeDecodeError:
            return (False, None, 0.5)

    # Kein BOM → charset_normalizer
    confidence = 1.0
    if text is None:
        try:
            result = from_bytes(chunk)
        except Exception:
            return (False, None, 0.5)
        best = result.best()
        if best is not None:
            encoding = best.encoding
            # UTF-16/UTF-32 ohne BOM ist ein bekannter False-Positive:
            # Binaerdaten (PNG-/EXE-Header) decoden in CJK-Codepoints,
            # die isprintable() den Binary-Guard ueberlisten lassen.
            # Echte UTF-16/32-Dateien tragen praktisch immer ein BOM
            # (Behandlung weiter oben).
            enc_norm = encoding.replace("_", "-").lower() if encoding else ""
            if enc_norm.startswith("utf-16") or enc_norm.startswith("utf-32"):
                return (False, encoding, 0.0)
            confidence = max(0.0, min(1.0, 1.0 - best.chaos))
            if confidence < 0.5:
                return (False, encoding, confidence)
            text = str(best)
        else:
            # charset_normalizer findet kein Encoding → binaer
            return (False, None, 0.0)

    # Binary-Check auf dekodiertem Text
    if not text:
        return (False, None, confidence)

    non_printable = sum(
        1 for c in text if not c.isprintable() and c not in ("\n", "\r", "\t", "\f")
    )
    ratio = non_printable / len(text) if text else 1.0

    if ratio > binary_threshold:
        return (False, encoding, confidence)

    return (True, encoding, confidence)


def is_text_file(path: str) -> bool:
    """Prueft ob eine Datei lesbaren Text enthaelt.

    Wrapper um detect_content() — gibt nur den is_text-Wert zurueck.
    """
    is_text, _, _ = detect_content(path)
    return is_text
