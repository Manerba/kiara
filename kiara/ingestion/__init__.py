"""Ingestion-Pipeline: Dateien scannen, laden, chunken, embedden, speichern.

Module:
    pipeline  — Orchestrierung (run_ingestion, run_all_datasources)
    loader    — Datei-Loader (25+ Formate, DDU/Docling, VLM)
    chunker   — Text/Code-Chunking (LangChain Splitter)
    _file_utils — Hashing, Scanning, Pattern-Matching
    chunk_strategy — Auto-Erkennung optimaler Chunk-Parameter
"""

from kiara_document_loaders.vision import describe_image

from kiara.ingestion.loader import load_file
from kiara.ingestion.pipeline import run_all_datasources, run_ingestion, scan_files
