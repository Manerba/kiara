"""Frontend-Guards fuer tab_ki_modelle.js Cloud-Backend-Flows.

    Statische Pattern-Tests fuer Backend-Modal-Guards:
    - DDU/Docling ist remote-only und braucht eine URL
- #1051: _testCloudKey Shortcut nur bei unveraenderten Cloud-Feldern
- #1052: embedding_dim-Probe lazy validiert + ins Save-Payload propagiert
"""

from pathlib import Path

import pytest

JS_PATH = Path(__file__).resolve().parent.parent / "kiara/web/static/js/tab_ki_modelle.js"


@pytest.fixture(scope="module")
def src() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _function_block(src: str, signature: str, max_chars: int = 4000) -> str:
    """Liefert den Quelltext-Block ab dem Funktionskopf bis +max_chars."""
    idx = src.find(signature)
    assert idx != -1, f"Funktionsanker nicht gefunden: {signature!r}"
    return src[idx : idx + max_chars]


# ---------------------------------------------------------------------------
# DDU/Docling remote-only
# ---------------------------------------------------------------------------


class TestDoclingRemoteOnly:
    def test_docling_url_is_required(self, src):
        """Docling-Provider muss das URL-Feld als Pflichtfeld markieren."""
        block = _function_block(src, "function _onBeFormProviderChange")
        assert "urlInput.required = true" in block
        assert "http://docling-proxy:5001" in block
        assert "Leer = Lokales Docling" not in src

    def test_docling_empty_url_is_blocked(self, src):
        """Leere Docling-URL darf keinen lokalen Testpfad mehr starten."""
        block = _function_block(src, "async function _testDoclingBackend")
        assert "Bitte Remote-DDU-URL eingeben" in block
        assert "Lokales Docling verfuegbar" not in block
        assert "data.mode === 'local'" not in block

    def test_docling_save_requires_url(self, src):
        """Save-Pfad blockiert Docling ohne Remote-URL vor dem API-Call."""
        block = _function_block(src, "async function _saveBackend")
        assert "provider === 'docling'" in block
        assert "Bitte Remote-DDU-URL eingeben" in block


# ---------------------------------------------------------------------------
# #1051: _testCloudKey Shortcut-Guard
# ---------------------------------------------------------------------------


class TestTestCloudKeyShortcutGuard:
    def test_cloud_fields_unchanged_helper_exists(self, src):
        """_cloudFieldsUnchanged() muss als eigenstaendige Helper-Funktion existieren."""
        assert "function _cloudFieldsUnchanged" in src

    def test_cloud_fields_unchanged_compares_provider_model_baseurl(self, src):
        """_cloudFieldsUnchanged() muss provider, model und base_url vergleichen."""
        block = _function_block(src, "function _cloudFieldsUnchanged", max_chars=600)
        assert "_beFormBaseline.provider" in block
        assert "_beFormBaseline.model" in block
        assert "_beFormBaseline.base_url" in block

    def test_test_cloud_key_uses_unchanged_guard(self, src):
        """_testCloudKey muss _cloudFieldsUnchanged() aufrufen, bevor Shortcut greift."""
        block = _function_block(src, "async function _testCloudKey")
        assert "_cloudFieldsUnchanged()" in block

    def test_test_cloud_key_blocks_when_fields_changed(self, src):
        """Bei geaenderten Cloud-Feldern und maskiertem Key: harter Block + notify."""
        block = _function_block(src, "async function _testCloudKey")
        # Block-Pfad muss eine Notify-Show enthalten zwischen Shortcut-Check und naechster Logik
        unchanged_pos = block.find("_cloudFieldsUnchanged()")
        notify_pos = block.find("Cloud-Felder", unchanged_pos)
        assert notify_pos > unchanged_pos, "Block-Notify muss nach dem Shortcut-Check kommen"
        # Dass es ein 'error' notify ist
        notify_block = block[notify_pos : notify_pos + 300]
        assert "'error'" in notify_block

    def test_baseline_set_on_edit_form_open(self, src):
        """_openBackendForm muss _beFormBaseline im Edit-Modus aus data setzen."""
        block = _function_block(src, "function _openBackendForm")
        # Edit-Branch
        edit_idx = block.find("mode === 'edit'")
        assert edit_idx != -1
        edit_block = block[edit_idx : edit_idx + 600]
        assert "_beFormBaseline" in edit_block
        assert "data.provider" in edit_block
        assert "data.model" in edit_block
        assert "data.base_url" in edit_block

    def test_baseline_null_on_add_form_open(self, src):
        """_openBackendForm muss _beFormBaseline im Add-Modus auf null setzen."""
        block = _function_block(src, "function _openBackendForm")
        assert "_beFormBaseline = null" in block

    def test_baseline_cleared_on_form_invalidation(self, src):
        """_invalidateBackendForm muss _beFormBaseline zuruecksetzen."""
        block = _function_block(src, "function _invalidateBackendForm", max_chars=400)
        assert "_beFormBaseline = null" in block


# ---------------------------------------------------------------------------
# #1052: embedding_dim-Invalidation und Save-Payload-Propagation
# ---------------------------------------------------------------------------


class TestEmbeddingDimProbeLifecycle:
    def test_probe_state_declared(self, src):
        """Modul-State _embeddingDimProbe muss als let-Variable existieren."""
        assert "let _embeddingDimProbe = null" in src

    def test_probe_set_after_successful_test(self, src):
        """_testEmbeddingCloud muss nach erfolgreichem Test _embeddingDimProbe setzen."""
        block = _function_block(src, "async function _testEmbeddingCloud")
        assert "_embeddingDimProbe = {" in block
        # Snapshot muss alle vier identitaetsstiftenden Felder enthalten
        # (api_key wird in _embeddingProbeMatchesCurrent ueber den Live-Wert geprueft).
        assert "model" in block
        assert "provider" in block
        assert "base_url" in block

    def test_probe_match_helper_exists(self, src):
        """_embeddingProbeMatchesCurrent() muss als Helper existieren."""
        assert "function _embeddingProbeMatchesCurrent" in src

    def test_probe_match_invalidates_on_changed_apikey(self, src):
        """Helper muss bei neu eingegebenem (nicht maskiertem) Key falsch zurueckgeben."""
        block = _function_block(src, "function _embeddingProbeMatchesCurrent", max_chars=800)
        assert "KI_PASSWORD_MASK" in block
        assert "apiKeyChanged" in block

    def test_probe_propagated_to_save_payload(self, src):
        """_saveBackend muss die Probe-Dim ins extra_config.embedding_dim propagieren."""
        block = _function_block(src, "async function _saveBackend")
        assert "_embeddingProbeMatchesCurrent()" in block
        assert "embedding_dim" in block
        assert "extra_config" in block

    def test_probe_cleared_on_form_invalidation(self, src):
        """_invalidateBackendForm muss _embeddingDimProbe zuruecksetzen."""
        block = _function_block(src, "function _invalidateBackendForm", max_chars=400)
        assert "_embeddingDimProbe = null" in block

    def test_probe_cleared_on_form_open(self, src):
        """_openBackendForm muss _embeddingDimProbe bei jedem Open zuruecksetzen."""
        block = _function_block(src, "function _openBackendForm")
        assert "_embeddingDimProbe = null" in block


# ---------------------------------------------------------------------------
# #1214: Providerwechsel darf kein altes Secret uebernehmen
# ---------------------------------------------------------------------------


class TestSaveProviderChangeSecretGuard:
    def test_provider_needs_api_key_helper_exists(self, src):
        """Provider-Key-Pflicht muss zentral im Frontend pruefbar sein."""
        block = _function_block(src, "function _providerNeedsApiKey", max_chars=500)
        assert "anthropic" in block
        assert "openai" in block
        assert "voyage" in block

    def test_save_blocks_provider_change_without_new_key(self, src):
        """Edit-Save muss Providerwechsel zu Key-Provider ohne neuen Key blocken."""
        block = _function_block(src, "async function _saveBackend")
        assert "providerChanged" in block
        assert "_providerNeedsApiKey(provider)" in block
        assert "apiKey === KI_PASSWORD_MASK" in block
        assert "Provider geaendert" in block
        assert "'error'" in block

    def test_save_clears_key_when_provider_changed_to_no_key(self, src):
        """Providerwechsel zu Provider ohne Key muss payload.api_key='' senden."""
        block = _function_block(src, "async function _saveBackend")
        assert "providerChanged && !needsApiKey" in block
        assert "payload.api_key = ''" in block
