"""Tests fuer classify_ollama_model() — Modell-Kategorisierung.

Testet die Heuristik-Kaskade:
1. Embedding-Familien (bert, nomic-bert, xlm-roberta)
2. VLM-Familien (mllama)
3. VLM-Namensmuster (vision, vl, llava, moondream, minicpm-v)
4. Embedding-Namensmuster (embed, bge-, gte-, e5-, arctic-embed)
5. Fallback → llm
"""

from kiara.web.api_ki import classify_ollama_model


def _model(name, families=None):
    """Helper: Erzeugt ein minimales Ollama-Modell-Dict."""
    m = {"name": name, "details": {}}
    if families is not None:
        m["details"]["families"] = families
    return m


# -- 1. Embedding-Familien --


class TestEmbeddingFamilies:
    def test_bert_family(self):
        assert classify_ollama_model(_model("some-model:latest", ["bert"])) == "embedding"

    def test_nomic_bert_family(self):
        assert (
            classify_ollama_model(_model("nomic-embed-text:latest", ["nomic-bert"])) == "embedding"
        )

    def test_xlm_roberta_family(self):
        assert (
            classify_ollama_model(_model("multilingual-e5:latest", ["xlm-roberta"])) == "embedding"
        )

    def test_bert_among_multiple_families(self):
        """bert in einer Liste mit anderen Familien → trotzdem embedding."""
        assert (
            classify_ollama_model(_model("custom:latest", ["transformer", "bert"])) == "embedding"
        )


# -- 2. VLM-Familien --


class TestVlmFamilies:
    def test_mllama_family(self):
        assert classify_ollama_model(_model("llama3.2-vision:latest", ["mllama"])) == "vlm"

    def test_mllama_with_other_families(self):
        assert classify_ollama_model(_model("custom-vision:latest", ["llama", "mllama"])) == "vlm"


# -- 3. VLM-Namensmuster --


class TestVlmNamePatterns:
    def test_vision_in_name(self):
        assert classify_ollama_model(_model("llama3.2-vision:11b")) == "vlm"

    def test_llava_in_name(self):
        assert classify_ollama_model(_model("llava:13b")) == "vlm"

    def test_moondream_in_name(self):
        assert classify_ollama_model(_model("moondream:latest")) == "vlm"

    def test_minicpm_v_in_name(self):
        assert classify_ollama_model(_model("minicpm-v:latest")) == "vlm"

    def test_vl_suffix(self):
        """qwen2.5vl → das 'vl' soll matchen (Word-Boundary-artig)."""
        assert classify_ollama_model(_model("qwen2.5vl:7b")) == "vlm"

    def test_vl_not_in_middle_of_word(self):
        """'evolved' enthaelt 'vl' nicht als eigenstaendiges Muster."""
        assert classify_ollama_model(_model("evolved-model:latest")) != "vlm"


# -- 4. Embedding-Namensmuster --


class TestEmbeddingNamePatterns:
    def test_embed_in_name(self):
        assert classify_ollama_model(_model("nomic-embed-text:latest")) == "embedding"

    def test_bge_prefix(self):
        assert classify_ollama_model(_model("bge-large-en:latest")) == "embedding"

    def test_gte_prefix(self):
        assert classify_ollama_model(_model("gte-small:latest")) == "embedding"

    def test_e5_prefix(self):
        assert classify_ollama_model(_model("e5-mistral:latest")) == "embedding"

    def test_arctic_embed(self):
        assert classify_ollama_model(_model("snowflake-arctic-embed:latest")) == "embedding"


# -- 5. LLM-Fallback --


class TestLlmFallback:
    def test_plain_llm(self):
        assert classify_ollama_model(_model("qwen3:8b")) == "llm"

    def test_llama_is_llm(self):
        assert classify_ollama_model(_model("llama3.1:8b")) == "llm"

    def test_mistral_is_llm(self):
        assert classify_ollama_model(_model("mistral:7b")) == "llm"

    def test_codellama_is_llm(self):
        assert classify_ollama_model(_model("codellama:13b")) == "llm"


# -- Edge-Cases --


class TestEdgeCases:
    def test_empty_model(self):
        """Leeres Dict → llm-Fallback."""
        assert classify_ollama_model({"name": "", "details": {}}) == "llm"

    def test_no_details(self):
        """Fehlendes details-Feld → llm-Fallback."""
        assert classify_ollama_model({"name": "qwen3:8b"}) == "llm"

    def test_none_families(self):
        """families=None → kein Crash, llm-Fallback."""
        assert (
            classify_ollama_model({"name": "test:latest", "details": {"families": None}}) == "llm"
        )

    def test_embedding_family_beats_vlm_name(self):
        """Familien-Check hat Vorrang vor Namensmuster."""
        m = _model("vision-embed:latest", ["bert"])
        assert classify_ollama_model(m) == "embedding"

    def test_vlm_family_beats_embedding_name(self):
        """VLM-Familie hat Vorrang vor Embedding-Namensmuster."""
        m = _model("embed-vision:latest", ["mllama"])
        assert classify_ollama_model(m) == "vlm"

    def test_real_server_nomic(self):
        """Reales Beispiel: nomic-embed-text vom Server."""
        m = {
            "name": "nomic-embed-text:latest",
            "details": {"families": ["nomic-bert"], "parameter_size": "137M"},
        }
        assert classify_ollama_model(m) == "embedding"

    def test_real_server_qwen3(self):
        """Reales Beispiel: qwen3:8b vom Server."""
        m = {
            "name": "qwen3:8b",
            "details": {"families": ["qwen3"], "parameter_size": "8.2B"},
        }
        assert classify_ollama_model(m) == "llm"
