"""Loader-Submodule fuer verschiedene Dateiformate.

- general: Allgemeine Dokumentformate (Docx, PDF, XLSX, ODT, PPTX, CSV, JSON, Plaintext, Code)
- cad: CAD-Formate (DXF, STEP, IGES)
- vision: Vision Language Model Bildbeschreibung (VLM)
"""

from kiara_document_loaders.cad import (
    _BOM_TAGS,
    _TITLE_BLOCK_TAGS,
    _dxf_clean_mtext,
    _dxf_extract_bom,
    _dxf_extract_dimensions,
    _dxf_extract_leaders,
    _dxf_extract_texts_by_layer,
    _dxf_extract_title_block,
    _load_dxf,
    _load_iges,
    _load_step,
)
from kiara_document_loaders.general import (
    _CODE_LANGUAGES,
    _load_code,
    _load_csv_file,
    _load_docx,
    _load_json_file,
    _load_odt,
    _load_pdf,
    _load_plaintext,
    _load_pptx,
    _load_xlsx,
)
from kiara_document_loaders.vision import (
    _OLLAMA_NATIVE,
    MAX_VLM_IMAGE_SIZE,
    _to_png_bytes,
    _vlm_prompt_de,
    describe_image,
)
