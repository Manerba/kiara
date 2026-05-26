"""Tests fuer get_ddu_backends() und get_ddu_backend() in kiara/config.py.

Testet Default-Werte, Custom extra_config, Remote-Only-Filter, Wrapper,
Exception-Handling, invalid JSON, null extra_config, model-Default
und concurrency_limit.
"""

from unittest.mock import patch

from kiara.config import DEFAULT_TIMEOUTS, get_ddu_backend, get_ddu_backends

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_backend(url="http://ddu:5001", model="docling", extra_config="{}"):
    """Erzeugt ein Backend-Dict wie get_all_active() es zurueckgibt."""
    return {"url": url, "model": model, "extra_config": extra_config, "name": "Test"}


# ── get_ddu_backends() Tests ─────────────────────────────────────────────────


@patch("kiara.backend_selector.get_all_active")
def test_backends_returns_list_with_defaults(mock_all):
    """Leere extra_config → Default table_mode/do_ocr/ocr_lang/timeout/concurrency_limit."""
    mock_all.return_value = [_mock_backend()]
    result = get_ddu_backends({})

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["url"] == "http://ddu:5001"
    assert result[0]["is_local"] is False
    assert result[0]["table_mode"] == "accurate"
    assert result[0]["do_ocr"] is True
    assert result[0]["ocr_lang"] == ["de", "en"]
    assert result[0]["timeout"] == DEFAULT_TIMEOUTS["ddu_convert"]
    assert result[0]["concurrency_limit"] is None


@patch("kiara.backend_selector.get_all_active")
def test_backends_ignores_empty_url(mock_all):
    """url leer → Backend wird ignoriert."""
    mock_all.return_value = [_mock_backend(url="")]
    result = get_ddu_backends({})

    assert result == []


@patch("kiara.backend_selector.get_all_active")
def test_backends_ignores_none_url(mock_all):
    """url=None → Backend wird ignoriert."""
    mock_all.return_value = [
        {"url": None, "model": "docling", "extra_config": "{}", "name": "Local"}
    ]
    result = get_ddu_backends({})

    assert result == []


@patch("kiara.backend_selector.get_all_active")
def test_backends_returns_custom_extra_config(mock_all):
    """JSON mit table_mode=fast, timeout=60, concurrency_limit=4."""
    mock_all.return_value = [
        _mock_backend(extra_config='{"table_mode": "fast", "timeout": 60, "concurrency_limit": 4}')
    ]
    result = get_ddu_backends({})

    assert result[0]["table_mode"] == "fast"
    assert result[0]["timeout"] == 60
    assert result[0]["concurrency_limit"] == 4


@patch("kiara.backend_selector.get_all_active")
def test_backends_returns_empty_no_backend(mock_all):
    """get_all_active returns [] → []."""
    mock_all.return_value = []
    assert get_ddu_backends({}) == []


@patch("kiara.backend_selector.get_all_active")
def test_backends_returns_empty_on_exception(mock_all):
    """get_all_active raises RuntimeError → []."""
    mock_all.side_effect = RuntimeError("DB down")
    assert get_ddu_backends({}) == []


@patch("kiara.backend_selector.get_all_active")
def test_backends_invalid_json_uses_defaults(mock_all):
    """extra_config='invalid{json' → Defaults, kein Crash."""
    mock_all.return_value = [_mock_backend(extra_config="invalid{json")]
    result = get_ddu_backends({})

    assert len(result) == 1
    assert result[0]["table_mode"] == "accurate"
    assert result[0]["do_ocr"] is True


@patch("kiara.backend_selector.get_all_active")
def test_backends_null_extra_config_uses_defaults(mock_all):
    """extra_config=None → Defaults."""
    mock_all.return_value = [_mock_backend(extra_config=None)]
    result = get_ddu_backends({})

    assert len(result) == 1
    assert result[0]["table_mode"] == "accurate"
    assert result[0]["timeout"] == DEFAULT_TIMEOUTS["ddu_convert"]


@patch("kiara.backend_selector.get_all_active")
def test_backends_multiple_sorted(mock_all):
    """Remote + leere URL → nur Remote-Eintrag bleibt erhalten."""
    mock_all.return_value = [
        _mock_backend(url="http://ddu:5001"),
        _mock_backend(url=""),
    ]
    result = get_ddu_backends({})

    assert len(result) == 1
    assert result[0]["is_local"] is False
    assert result[0]["url"] == "http://ddu:5001"


@patch("kiara.backend_selector.get_all_active")
def test_backends_model_default_docling(mock_all):
    """Backend ohne model Key → result['model'] == 'docling'."""
    mock_all.return_value = [{"url": "http://ddu:5001", "extra_config": "{}", "name": "Test"}]
    result = get_ddu_backends({})

    assert result[0]["model"] == "docling"


@patch("kiara.backend_selector.get_all_active")
def test_backends_concurrency_limit_in_return(mock_all):
    """concurrency_limit ist im Return-Dict vorhanden."""
    mock_all.return_value = [_mock_backend(extra_config='{"concurrency_limit": 8}')]
    result = get_ddu_backends({})

    assert "concurrency_limit" in result[0]
    assert result[0]["concurrency_limit"] == 8


# ── get_ddu_backend() Wrapper Tests ──────────────────────────────────────────


@patch("kiara.backend_selector.get_all_active")
def test_wrapper_returns_first(mock_all):
    """get_ddu_backend() gibt das erste Backend zurueck."""
    mock_all.return_value = [_mock_backend()]
    result = get_ddu_backend({})

    assert result is not None
    assert result["url"] == "http://ddu:5001"


@patch("kiara.backend_selector.get_all_active")
def test_wrapper_returns_none_no_backend(mock_all):
    """get_ddu_backend() gibt None zurueck wenn keine Backends."""
    mock_all.return_value = []
    assert get_ddu_backend({}) is None


@patch("kiara.backend_selector.get_all_active")
def test_wrapper_returns_none_on_exception(mock_all):
    """get_ddu_backend() gibt None zurueck bei Exception."""
    mock_all.side_effect = RuntimeError("DB down")
    assert get_ddu_backend({}) is None
