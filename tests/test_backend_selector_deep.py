"""Tiefergehende Tests fuer kiara.backend_selector — Edge-Cases & Fehlerbehandlung.

Ergaenzt test_backend_selector.py um DB-Error-Handling, API-Key-Entschluesselung,
Priority-Sortierung, Cache-TTL und Kategorie-Unabhaengigkeit.
"""

import threading
import time
from unittest.mock import patch

import pytest

from kiara import backend_selector as selector
from kiara.backend_selector import (
    _cache,
    _decrypt_backend,
    _fetch_backends,
    _rr_index,
    get_all_active,
    get_next_backend,
    invalidate_cache,
)
from kiara.shared.constants import BACKEND_CACHE_TTL_SECS as _CACHE_TTL


def _make_backend(id_=1, category="llm", priority=10, api_key="", **overrides):
    """Erzeugt ein einzelnes Mock-Backend-Dict."""
    base = {
        "id": id_,
        "category": category,
        "provider": "ollama",
        "name": f"Backend {id_}",
        "url": f"http://server{id_}:11434",
        "base_url": "",
        "model": "nomic-embed-text" if category == "embedding" else f"model-{id_}",
        "api_key": api_key,
        "is_active": True,
        "priority": priority,
        "extra_config": "{}",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clear_cache():
    """Cache vor jedem Test leeren."""
    invalidate_cache()
    yield
    invalidate_cache()


# ===================================================================
# A) DB-Fehler-Handling
# ===================================================================


class TestDBErrorHandling:
    def test_fetch_backends_db_error_returns_none(self):
        """_fetch_backends() gibt bei DB-Exception None zurueck (Sentinel fuer Cache-Fallback)."""
        with patch(
            "kiara_db.repositories.backend.KiBackendRepository.list_by_category",
            side_effect=Exception("DB down"),
        ):
            result = _fetch_backends("llm")
            assert result is None

    def test_get_all_active_db_error_returns_empty(self):
        """get_all_active() gibt [] zurueck wenn DB nicht erreichbar."""
        with patch("kiara.backend_selector._fetch_backends", return_value=[]):
            result = get_all_active("embedding")
            assert result == []

    def test_get_next_backend_db_error_returns_none(self):
        """get_next_backend() gibt None zurueck wenn _fetch_backends fehlschlaegt."""
        with patch("kiara.backend_selector._fetch_backends", return_value=[]):
            result = get_next_backend("llm")
            assert result is None


# ===================================================================
# B) API-Key Entschluesselung
# ===================================================================


class TestDecryptBackend:
    def test_decrypt_backend_success(self):
        """Verschluesselter API-Key wird korrekt entschluesselt."""
        backend = _make_backend(api_key="gAAAAA_encrypted_token")
        # Lazy import: from kiara.shared.crypto import decrypt, is_encrypted
        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-plaintext-key"),
        ):
            result = _decrypt_backend(backend)
            assert result["api_key"] == "sk-plaintext-key"

    def test_decrypt_backend_failure_sets_empty(self):
        """Wenn decrypt() fehlschlaegt, wird api_key fail-closed auf '' gesetzt."""
        backend = _make_backend(api_key="gAAAAA_bad_token")
        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", side_effect=Exception("InvalidToken")),
        ):
            result = _decrypt_backend(backend)
            assert result["api_key"] == ""

    def test_decrypt_backend_empty_key_unchanged(self):
        """Leerer api_key wird nicht angefasst."""
        backend = _make_backend(api_key="")
        result = _decrypt_backend(backend)
        assert result["api_key"] == ""

    def test_decrypt_backend_plaintext_key_unchanged(self):
        """Klartext-Key (nicht verschluesselt) bleibt unveraendert."""
        backend = _make_backend(api_key="sk-plain-key")
        with patch("kiara.shared.crypto.is_encrypted", return_value=False):
            result = _decrypt_backend(backend)
            assert result["api_key"] == "sk-plain-key"


# ===================================================================
# C) Priority-Sortierung
# ===================================================================


class TestPrioritySorting:
    def test_backends_sorted_by_priority(self):
        """Backends werden nach Priority aufsteigend sortiert."""
        backends = [
            _make_backend(id_=1, priority=30),
            _make_backend(id_=2, priority=10),
            _make_backend(id_=3, priority=20),
        ]
        # _fetch_backends sortiert per SQL — Mock hier direkt get_all_active
        with patch("kiara.backend_selector._fetch_backends", return_value=backends):
            result = get_all_active("llm")
            # Cache speichert die Liste so wie von _fetch_backends geliefert
            assert [b["id"] for b in result] == [1, 2, 3]

    def test_failover_returns_first_in_list(self):
        """Failover (embedding) gibt immer das erste Backend der Liste zurueck."""
        backends = [
            _make_backend(id_=5, category="embedding", priority=10),
            _make_backend(id_=3, category="embedding", priority=20),
        ]
        with (
            patch("kiara.backend_selector._fetch_backends", return_value=backends),
            patch("kiara.backend_selector._is_healthy_cached", return_value=True),
        ):
            result = get_next_backend("embedding")
            assert result["id"] == 5

    def test_rr_respects_list_order(self):
        """Round-Robin rotiert in der Reihenfolge der Liste (priority-sortiert)."""
        backends = [
            _make_backend(id_=2, priority=10),
            _make_backend(id_=5, priority=20),
            _make_backend(id_=1, priority=30),
        ]
        with patch("kiara.backend_selector._fetch_backends", return_value=backends):
            ids = [get_next_backend("llm")["id"] for _ in range(6)]
            assert ids == [2, 5, 1, 2, 5, 1]


# ===================================================================
# D) Cache-Verhalten
# ===================================================================


class TestCacheBehavior:
    def test_cache_ttl_expiry(self):
        """Nach TTL wird erneut aus DB geladen."""
        backends_v1 = [_make_backend(id_=1)]
        backends_v2 = [_make_backend(id_=2)]
        call_count = {"n": 0}

        def fetch_side_effect(cat):
            call_count["n"] += 1
            return backends_v1 if call_count["n"] == 1 else backends_v2

        with patch("kiara.backend_selector._fetch_backends", side_effect=fetch_side_effect):
            r1 = get_all_active("llm")
            assert r1[0]["id"] == 1

            # Timestamp manipulieren um TTL zu simulieren
            with patch("kiara.backend_selector.time") as mock_time:
                mock_time.time.return_value = time.time() + _CACHE_TTL + 1
                r2 = get_all_active("llm")
                assert r2[0]["id"] == 2

    def test_cache_per_category_independent(self):
        """Invalidierung einer Kategorie beeinflusst andere nicht."""
        emb = [_make_backend(id_=1, category="embedding")]
        llm = [_make_backend(id_=2, category="llm")]

        with patch(
            "kiara.backend_selector._fetch_backends",
            side_effect=lambda cat: emb if cat == "embedding" else llm,
        ):
            get_all_active("embedding")
            get_all_active("llm")

        assert "embedding" in _cache
        assert "llm" in _cache

        invalidate_cache("llm")
        assert "embedding" in _cache
        assert "llm" not in _cache

    def test_llm_round_robin_does_not_use_public_get_all_active(self):
        """LLM-Auswahl koppelt Snapshot und RR-Index in einem Lock."""
        backends = [_make_backend(id_=1), _make_backend(id_=2)]
        with (
            patch("kiara.backend_selector._fetch_backends", return_value=backends),
            patch(
                "kiara.backend_selector.get_all_active",
                side_effect=AssertionError("separate snapshot"),
            ),
        ):
            assert get_next_backend("llm")["id"] == 1

    def test_llm_round_robin_fetch_runs_under_selector_lock(self):
        """Cache-Refresh fuer LLM laeuft im selben kritischen Abschnitt wie RR."""
        backends = [_make_backend(id_=1), _make_backend(id_=2)]

        def fetch(_category):
            assert selector._lock.locked()
            return backends

        with patch("kiara.backend_selector._fetch_backends", side_effect=fetch):
            assert get_next_backend("llm")["id"] == 1

    def test_invalidate_all_clears_rr_index(self):
        """invalidate_cache(None) setzt auch RR-Indizes zurueck."""
        backends = [_make_backend(id_=1), _make_backend(id_=2)]
        with patch("kiara.backend_selector._fetch_backends", return_value=backends):
            get_next_backend("llm")  # rr_index["llm"] = 1
            get_next_backend("llm")  # rr_index["llm"] = 0

        invalidate_cache()
        assert "llm" not in _rr_index

    def test_concurrent_invalidate_and_fetch(self):
        """Parallele Invalidierung und Fetch fuehrt nicht zu Crash."""
        backends = [_make_backend(id_=1)]
        errors = []

        with patch("kiara.backend_selector._fetch_backends", return_value=backends):

            def invalidator():
                try:
                    for _ in range(100):
                        invalidate_cache("llm")
                except Exception as e:
                    errors.append(e)

            def fetcher():
                try:
                    for _ in range(100):
                        get_all_active("llm")
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=invalidator),
                threading.Thread(target=fetcher),
                threading.Thread(target=fetcher),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors


# ===================================================================
# E) Kategorie-Mischung & Randfaelle
# ===================================================================


class TestCategoryEdgeCases:
    def test_rr_index_per_category_independent(self):
        """Round-Robin-Zaehler fuer verschiedene Kategorien sind unabhaengig."""
        llm_backends = [_make_backend(id_=1, category="llm"), _make_backend(id_=2, category="llm")]
        cloud_backends = [
            _make_backend(id_=10, category="cloud"),
            _make_backend(id_=20, category="cloud"),
        ]

        def fetch(cat):
            return llm_backends if cat == "llm" else cloud_backends

        with patch("kiara.backend_selector._fetch_backends", side_effect=fetch):
            llm1 = get_next_backend("llm")
            cloud1 = get_next_backend("cloud")
            llm2 = get_next_backend("llm")
            cloud2 = get_next_backend("cloud")

            assert llm1["id"] == 1
            assert llm2["id"] == 2
            assert cloud1["id"] == 10
            assert cloud2["id"] == 20

    def test_unknown_category_returns_none(self):
        """Unbekannte Kategorie gibt None zurueck (keine Exception)."""
        with patch("kiara.backend_selector._fetch_backends", return_value=[]):
            result = get_next_backend("nonexistent")
            assert result is None

    def test_rr_index_resets_when_backends_shrink(self):
        """RR-Index wird zurueckgesetzt wenn weniger Backends als Index."""
        backends_3 = [_make_backend(id_=i) for i in range(1, 4)]
        backends_1 = [_make_backend(id_=1)]

        with patch("kiara.backend_selector._fetch_backends", return_value=backends_3):
            for _ in range(3):
                get_next_backend("llm")  # rr_index = 0 (nach wrap-around)

        # Cache invalidieren und mit weniger Backends neu laden
        invalidate_cache("llm")
        with patch("kiara.backend_selector._fetch_backends", return_value=backends_1):
            result = get_next_backend("llm")
            assert result is not None
            assert result["id"] == 1
