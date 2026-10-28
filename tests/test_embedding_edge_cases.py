"""Edge-Case-Tests fuer das Embedding-Modul.

Testet unbekannte Modelle, Sonderzeichen in Collection-Namen,
leere Eingaben und spezifische Modell-Prefixes.
"""

from collections import OrderedDict
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

from kiara.retrieval.embedding import (
    _BATCH_CHAR_BUDGETS,
    _BATCH_LIMITS,
    _iter_token_aware_batches,
    collection_name_for_model,
    embed_texts,
    model_name_from_collection,
)

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

CONFIG = {
    "ollama": {
        "url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "embedding_url": "",
    },
}


# ---------------------------------------------------------------------------
# 1. test_embed_query_no_prefix_unknown_model
# ---------------------------------------------------------------------------


def test_model_name_role_helpers_and_local_prefix_table_are_removed():
    """S5 kennt weder Nomic-Sonderrollen noch lokale Query-Praefixe."""
    from kiara.retrieval import embedding

    assert not hasattr(embedding, "_get_query_prefix")
    assert not hasattr(embedding, "_nomic_input_type")
    assert not hasattr(embedding, "_EMBEDDING_QUERY_PREFIXES")


# ---------------------------------------------------------------------------
# 2. test_collection_name_special_chars
# ---------------------------------------------------------------------------


def test_collection_name_special_chars():
    """Modellname mit /, @, Leerzeichen -> reversibles _hh-Byte-Escape."""
    result = collection_name_for_model("my model/v2@beta")
    assert result == "kb_my_20model_2fv2_40beta"

    result2 = collection_name_for_model("model with spaces")
    assert result2 == "kb_model_20with_20spaces"

    result3 = collection_name_for_model("model.v1.2")
    assert result3 == "kb_model_2ev1_2e2"

    # Gross-/Kleinschreibung wird zentral normalisiert
    assert collection_name_for_model("Nomic-Embed-Text") == "kb_nomic-embed-text"


def test_client_cache_prunes_lru_entries():
    """Client-Caches bleiben bounded und entfernen den aeltesten Eintrag."""
    from kiara.retrieval import embedding

    cache = OrderedDict()
    for i in range(embedding._CLIENT_CACHE_MAXSIZE + 1):
        embedding._remember_client(cache, f"k{i}", object())

    assert len(cache) == embedding._CLIENT_CACHE_MAXSIZE
    assert "k0" not in cache
    assert f"k{embedding._CLIENT_CACHE_MAXSIZE}" in cache


# ---------------------------------------------------------------------------
# 3. test_collection_name_empty_string
# ---------------------------------------------------------------------------


def test_collection_name_empty_string():
    """Leerer Modellname wirft ValueError."""
    with pytest.raises(ValueError, match="nicht-leeren"):
        collection_name_for_model("")


def test_collection_name_latest_only():
    """':latest' allein normalisiert zu '' -> ValueError."""
    with pytest.raises(ValueError, match="nicht-leeren"):
        collection_name_for_model(":latest")


def test_model_name_from_collection_roundtrip():
    """Roundtrip-Test fuer diverse Sonderzeichen."""
    cases = [
        ("nomic-embed-text", "kb_nomic-embed-text"),
        ("hellord/e5-mistral-7b-instruct:q4_0", "kb_hellord_2fe5-mistral-7b-instruct_3aq4_5f0"),
        ("my model/v2@beta", "kb_my_20model_2fv2_40beta"),
    ]
    for model, expected_coll in cases:
        assert collection_name_for_model(model) == expected_coll
        assert model_name_from_collection(expected_coll) == model


def test_model_name_from_collection_invalid_escape():
    """Unvollstaendige Escape-Sequenz am Ende wirft ValueError."""
    with pytest.raises(ValueError, match="Ungueltige Escape-Sequenz"):
        model_name_from_collection("kb_test_a")


# ---------------------------------------------------------------------------
# 4. test_embed_texts_empty_list
# ---------------------------------------------------------------------------

_FAKE_BACKEND = {
    "url": "http://localhost:11434",
    "provider": "ollama",
    "model": "nomic-embed-text",
    "name": "test",
}


@patch("kiara.retrieval.embedding.httpx.post")
@patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
def test_embed_texts_empty_list(mock_backend, mock_post):
    """embed_texts(config, model, []) -> leere Liste, kein API-Call."""
    result = embed_texts(CONFIG, "nomic-embed-text", [])

    # Leere Input-Liste wird frueh abgefangen — kein HTTP-Call
    mock_post.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# 5. test_mxbai_model_prefix
# ---------------------------------------------------------------------------


def test_mxbai_name_is_not_present_in_role_dispatch_source():
    """Der historische mxbai-Sonderfall ist aus dem Requestpfad entfernt."""
    import inspect

    from kiara.retrieval import embedding

    source = inspect.getsource(embedding.embed_query.__wrapped__)
    assert "mxbai" not in source.lower()


# ---------------------------------------------------------------------------
# 6. test_snowflake_model_prefix
# ---------------------------------------------------------------------------


def test_snowflake_name_is_not_present_in_role_dispatch_source():
    """Der historische Snowflake-Sonderfall ist aus dem Requestpfad entfernt."""
    import inspect

    from kiara.retrieval import embedding

    source = inspect.getsource(embedding.embed_query.__wrapped__)
    assert "snowflake" not in source.lower()


# ---------------------------------------------------------------------------
# 7. Token-aware Batch-Splitting (M-031: Provider-Token-Limits)
# ---------------------------------------------------------------------------


def test_iter_token_aware_batches_empty():
    """Leerer Input liefert keine Batches."""
    assert list(_iter_token_aware_batches([], 100, 1000)) == []


def test_iter_token_aware_batches_item_limit():
    """Item-Count-Limit greift bei vielen kleinen Texten."""
    texts = ["abc"] * 25
    batches = list(_iter_token_aware_batches(texts, item_limit=10, char_budget=10_000))
    assert [len(b) for b in batches] == [10, 10, 5]


def test_iter_token_aware_batches_char_budget_splits_below_item_limit():
    """Char-Budget greift, BEVOR Item-Limit erreicht ist (M-031-Regression).

    Vor dem Fix: 5 Texte * 80k chars wurden in einem Batch versendet
    (unter Item-Limit), wodurch das Provider-Token-Limit ueberschritten
    wurde. Jetzt wird nach char_budget gesplittet.
    """
    big = "x" * 80_000
    batches = list(_iter_token_aware_batches([big] * 5, item_limit=1000, char_budget=200_000))
    # 5 * 80k = 400k chars > 200k Budget -> mehrere Batches
    assert len(batches) >= 2
    for b in batches:
        # Kein Batch ueber dem Budget (Solo-Ausnahme nicht relevant: 80k < 200k)
        assert sum(len(t) for t in b) <= 200_000


def test_iter_token_aware_batches_oversized_solo():
    """Einzeltext >= Char-Budget kommt als Solo-Batch — Provider entscheidet
    ueber Per-Item-Limit (kein internes Aufsplitten eines Items)."""
    huge = "x" * 300_000
    batches = list(
        _iter_token_aware_batches([huge, "a", "b"], item_limit=1000, char_budget=200_000)
    )
    assert batches[0] == [huge]
    # Kleine Items kommen in nachfolgendem Batch
    assert batches[1] == ["a", "b"]


def test_iter_token_aware_batches_preserves_order():
    """Reihenfolge der Texte bleibt ueber alle Batches hinweg erhalten."""
    texts = [f"t{i:04d}" for i in range(50)]
    batches = list(_iter_token_aware_batches(texts, item_limit=7, char_budget=10_000))
    flat = [t for b in batches for t in b]
    assert flat == texts


def test_batch_char_budgets_have_safety_margin():
    """Sanity: Char-Budgets sind klein genug, dass ein voller Batch beim
    konservativsten Token-Estimate (1 Token = 2 chars) unter dem dokumentierten
    Provider-Token-Limit bleibt."""
    # OpenAI: 300k Tokens/Request -> 600k chars worst case
    assert _BATCH_CHAR_BUDGETS["openai"] <= 600_000
    # Voyage Free Tier: 120k Tokens/Request -> 240k chars worst case
    assert _BATCH_CHAR_BUDGETS["voyage"] <= 240_000
    # Item-Limits unveraendert
    assert _BATCH_LIMITS["openai"] == 2048
    assert _BATCH_LIMITS["voyage"] == 1000
