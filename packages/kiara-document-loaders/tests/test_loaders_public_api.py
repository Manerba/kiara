import pytest
from kiara_document_loaders import (
    _load_code,
    _load_plaintext,
    detect_content,
    is_text_file,
)


def test_public_api_exports_exist():
    assert callable(_load_plaintext)
    assert callable(_load_code)
    assert callable(detect_content)
    assert callable(is_text_file)


def test_load_plaintext_reads_utf8_file(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Hallo Welt\n", encoding="utf-8")

    assert _load_plaintext(str(path)) == "Hallo Welt\n"


def test_load_code_wraps_python_file(tmp_path):
    path = tmp_path / "demo.py"
    path.write_text("print('hi')\n", encoding="utf-8")

    loaded = _load_code(str(path))

    assert loaded.startswith("```python\n# File: demo.py\n")
    assert "print('hi')" in loaded


def test_detect_content_and_is_text_file_for_plaintext(tmp_path):
    pytest.importorskip("charset_normalizer", reason="charset_normalizer nicht installiert")

    path = tmp_path / "readme.xyz"
    path.write_text("Nur Text\n", encoding="utf-8")

    is_text, encoding, confidence = detect_content(str(path))

    assert is_text is True
    assert encoding is not None
    assert confidence >= 0.0
    assert is_text_file(str(path)) is True
