"""Automatische Chunk-Groessen-Erkennung basierend auf Dateityp und Textinhalt.

Erkennt die optimale Chunk-Strategie (prose/tabular/code/structured/email)
anhand eines zweistufigen Algorithmus:

1. **Dateityp-Heuristik**: Extension und Quellpfad bestimmen eine Basiskategorie.
2. **Textanalyse-Korrektur**: Bei unsicheren Kategorien (prose, default) wird der
   Textinhalt analysiert und die Kategorie ggf. ueberschrieben.

Die Schwellwerte wurden empirisch anhand eines Korpus aus Unternehmens-
dokumenten (Berichte, Tabellen, E-Mails, Quellcode, Formulare) bestimmt
und bieten eine gute Balance zwischen Precision und Recall.
"""

import logging
import os
import re

from kiara.config import DEFAULT_PATHS
from kiara.shared.constants import CODE_EXTENSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Textanalyse-Schwellwerte (empirisch bestimmt)
# ---------------------------------------------------------------------------

PIPE_RATIO_THRESHOLD: float = 0.4
"""Anteil der Zeilen mit Pipe-Zeichen (|) ab dem ein Text als tabellarisch gilt.

Markdown-Tabellen und CSV-aehnliche Formate verwenden | als Spalten-Trenner.
Bei >40% Pipe-Zeilen ist die Wahrscheinlichkeit hoch, dass es sich um
tabellarische Daten handelt (typische Dokumente haben <5%).
"""

KV_RATIO_THRESHOLD: float = 0.5
"""Anteil der Zeilen mit Key-Value-Pattern ab dem ein Text als strukturiert gilt.

Formulare, Konfigurationsdateien und Header-Bloecke enthalten typischerweise
>50% Zeilen im Format 'Schluessel: Wert' oder 'Schluessel = Wert'.
"""

KV_RATIO_FALLBACK: float = 0.3
"""Niedrigerer KV-Schwellwert fuer Texte mit kurzen Zeilen.

Wenn die durchschnittliche Zeilenlaenge unter SHORT_LINE_THRESHOLD liegt,
genuegen bereits 30% KV-Zeilen fuer die Einstufung als 'structured'.
Kurze Zeilen + moderate KV-Dichte deuten auf Listen oder Steckbriefe hin.
"""

SHORT_LINE_THRESHOLD: int = 50
"""Durchschnittliche Zeilenlaenge (Zeichen) unterhalb derer ein Text als
'kurzzeilig' gilt und der KV_RATIO_FALLBACK statt KV_RATIO_THRESHOLD greift.

50 Zeichen entsprechen etwa einem halben Satz — typisch fuer Listen, Tabellen
und strukturierte Formulare, nicht fuer Fliesstext.
"""

# ---------------------------------------------------------------------------
# Pfad-Segmente fuer Quellpfad-Erkennung (aus DEFAULT_PATHS abgeleitet)
# ---------------------------------------------------------------------------

_MAIL_CACHE_SEGMENT: str = os.sep + os.path.basename(DEFAULT_PATHS["mail_cache"]) + os.sep
"""Pfad-Segment zur Erkennung von E-Mail-Cache-Dateien (z.B. '/mail-cache/').

Abgeleitet aus DEFAULT_PATHS['mail_cache'], um Aenderungen am Cache-Pfad
automatisch zu uebernehmen.
"""

_SQL_CACHE_SEGMENT: str = os.sep + os.path.basename(DEFAULT_PATHS["sql_cache"]) + os.sep
"""Pfad-Segment zur Erkennung von SQL-Cache-Dateien (z.B. '/sql-cache/').

Abgeleitet aus DEFAULT_PATHS['sql_cache'], um Aenderungen am Cache-Pfad
automatisch zu uebernehmen.
"""

# ---------------------------------------------------------------------------
# Chunk-Strategien und Dateityp-Kategorien
# ---------------------------------------------------------------------------

# Chunk-Strategien: (chunk_size, chunk_overlap)
STRATEGIES = {
    "prose": (768, 150),  # Fliesstext: Berichte, Handbuecher, Mails
    "tabular": (256, 0),  # Tabellen: CSV, Excel, SQL-Zeilen
    "code": (512, 100),  # Quellcode
    "structured": (384, 50),  # Formulare, Key-Value-Listen
    "email": (640, 100),  # E-Mails (Mischung aus Header + Fliesstext)
    "default": (1000, 200),  # Fallback = globaler Default
}

# Dateityp -> Basiskategorie
_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xlsm", ".json"}
_PROSE_EXTENSIONS = {".docx", ".pdf", ".odt", ".pptx", ".md", ".txt"}
_CAD_EXTENSIONS = {".dxf", ".step", ".stp", ".iges", ".igs"}
_IMAGE_EXTENSIONS = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}

# Kompiliertes Pattern fuer Key-Value-Erkennung (einmalig, nicht pro Aufruf)
_KV_PATTERN = re.compile(r"^[\w\s\-]{2,30}:\s+\S|^[\w\s\-]{2,30}\s*=\s*\S")


def _analyze_text(text: str) -> str | None:
    """Analysiert Textinhalt und gibt Kategorie zurueck.

    Berechnet alle drei Metriken (Pipe-Ratio, KV-Ratio, durchschnittliche
    Zeilenlaenge) in einem einzigen Loop ueber die Zeilen.

    Returns:
        "prose" | "tabular" | "structured" | None (= Basis beibehalten)
    """
    stripped = text.strip()
    if not stripped:
        return None
    lines = stripped.split("\n")

    pipe_count = 0
    kv_count = 0
    total_len = 0

    for line in lines:
        line_stripped = line.strip()
        if "|" in line_stripped:
            pipe_count += 1
        if _KV_PATTERN.match(line_stripped):
            kv_count += 1
        total_len += len(line_stripped)

    n = len(lines)
    pipe_ratio = pipe_count / n
    kv_ratio = kv_count / n
    avg_line_len = total_len / n

    # Metrik 1: Tabellen-Erkennung ueber Pipe-Zeichen
    if pipe_ratio > PIPE_RATIO_THRESHOLD:
        return "tabular"

    # Metrik 2: Key-Value-Erkennung
    if kv_ratio > KV_RATIO_THRESHOLD:
        return "structured"

    # Metrik 3: Kurze Zeilen als Korrekturfaktor
    if avg_line_len < SHORT_LINE_THRESHOLD:
        return "structured" if kv_ratio > KV_RATIO_FALLBACK else "tabular"

    return None


def detect_chunk_params(text: str, metadata: dict) -> tuple[int, int]:
    """Bestimmt optimale (chunk_size, chunk_overlap) fuer ein Dokument.

    Zweistufiger Erkennungsalgorithmus:

    **Stufe 1 — Dateityp-Heuristik** (Prioritaet absteigend):
        1. Tabellarische Extensions (.csv, .xlsx, ...) → ``tabular``
        2. CAD-Extensions (.dxf, .step, ...) → ``structured``
        3. Bild-Extensions (.png, .jpg, ...) → ``prose`` (VLM-Beschreibung)
        4. Code-Extensions (.py, .cs, ...) → ``code`` (unveraenderbar)
        5. .txt aus Mail-Cache-Verzeichnis → ``email``
        6. .txt aus SQL-Cache-Verzeichnis → ``tabular``
        7. Prosa-Extensions (.docx, .pdf, ...) → ``prose``
        8. Alles andere → ``default``

    **Stufe 2 — Textanalyse-Korrektur** (nur bei ``prose`` oder ``default``):
        Analysiert Pipe-Ratio, KV-Ratio und durchschnittliche Zeilenlaenge,
        um die Basiskategorie ggf. auf ``tabular`` oder ``structured`` zu
        korrigieren. Sichere Kategorien (code, email, tabular aus Extension
        oder Pfad) werden NICHT ueberschrieben.

    Schwellwerte (siehe Modul-Konstanten):
        - PIPE_RATIO_THRESHOLD (0.4): >40% Pipe-Zeilen → tabular
        - KV_RATIO_THRESHOLD (0.5): >50% Key-Value-Zeilen → structured
        - SHORT_LINE_THRESHOLD (50): Durchschnitt <50 Zeichen → kurzzeilig
        - KV_RATIO_FALLBACK (0.3): Bei kurzen Zeilen genuegen 30% KV → structured

    Setzt zusaetzlich metadata["chunk_strategy"] auf den erkannten Strategie-Namen.

    Args:
        text: Der extrahierte Dokumenttext.
        metadata: Metadaten aus load_file() (enthaelt file_type, source, etc.)

    Returns:
        (chunk_size, chunk_overlap) — Werte aus dem STRATEGIES-Dict.
    """
    file_type = metadata.get("file_type", "").lower()
    source = metadata.get("source", "")

    # --- Stufe 1: Dateityp-Heuristik ---
    if file_type in _TABULAR_EXTENSIONS:
        category = "tabular"
    elif file_type in _CAD_EXTENSIONS:
        category = "structured"
    elif file_type in _IMAGE_EXTENSIONS:
        category = "prose"
    elif file_type in CODE_EXTENSIONS:
        category = "code"
    elif file_type == ".txt" and _MAIL_CACHE_SEGMENT in source:
        category = "email"
    elif file_type == ".txt" and _SQL_CACHE_SEGMENT in source:
        category = "tabular"
    elif file_type in _PROSE_EXTENSIONS:
        category = "prose"
    else:
        # Unbekannte Extension — Textanalyse entscheidet
        category = "default"

    # --- Stufe 2: Textanalyse-Korrektur ---
    # Textanalyse nur bei unsicheren Kategorien (prose, default).
    # Sichere Kategorien (code, email, tabular aus Extension/Pfad) bleiben bestehen.
    if category in ("prose", "default") and text and text.strip():
        text_category = _analyze_text(text)
        if text_category is not None:
            category = text_category
    elif not text or not text.strip():
        category = "default"

    size, overlap = STRATEGIES[category]

    # Strategie in Metadaten schreiben
    metadata["chunk_strategy"] = category

    filename = metadata.get("source", "unbekannt")
    logger.debug("Chunk-Strategie fuer %s: %s (%d/%d)", filename, category, size, overlap)

    return size, overlap
