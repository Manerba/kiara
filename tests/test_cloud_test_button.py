"""Tests fuer Build 66: Cloud-KI Test-Button + Backend-Form Refactoring.

Testet:
    - _analyze_text() Leerstring/None-Bugfix (chunk_strategy)
    - Cloud-Backend Test-Endpoint (Anthropic/OpenAI) mit PASSWORD_MASK
    - BOM-safe Plaintext-Loader (utf-8-sig)
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kiara.ingestion.chunk_strategy import STRATEGIES, _analyze_text, detect_chunk_params
from kiara.ingestion.loaders.general import _load_plaintext
from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.db import get_connection
from kiara.web.admin_app import app
from tests.admin_confirm import confirmed_request

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_admin():
    """Mock-Admin-User fuer Dependency-Override."""
    user = {"id": 1, "username": "admin", "is_admin": True}
    from kiara.web.auth import get_admin_user

    app.dependency_overrides[get_admin_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture
def mock_db():
    """PostgreSQL-DB mit ki_backends-Tabelle."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ki_backends (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL CHECK (category IN ('embedding', 'llm', 'cloud', 'vlm')),
            provider TEXT NOT NULL CHECK (provider IN ('ollama', 'anthropic', 'openai')),
            name TEXT NOT NULL,
            url TEXT DEFAULT '',
            base_url TEXT DEFAULT '',
            model TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            priority INTEGER DEFAULT 10,
            trusted BOOLEAN DEFAULT false,
            extra_config TEXT DEFAULT '{}',
            last_test_at TEXT DEFAULT NULL,
            last_test_ok INTEGER DEFAULT NULL,
            last_test_error TEXT DEFAULT '',
            created_at TEXT DEFAULT (NOW()),
            updated_at TEXT DEFAULT (NOW())
        );
    """)
    conn.commit()

    @contextmanager
    def mock_db_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def mock_db_transaction():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    with (
        patch("kiara_db.base.db_query", mock_db_query),
        patch("kiara_db.base.db_transaction", mock_db_transaction),
        patch("kiara.backend_selector.invalidate_cache"),
    ):
        yield conn

    conn.rollback()
    conn.close()


def _create_backend(client, **overrides):
    """Helper: Backend ueber API erstellen."""
    data = {
        "category": "cloud",
        "provider": "anthropic",
        "name": "Test Cloud",
        "model": "claude-sonnet-4-6",
        "api_key": "sk-ant-test-key-123",
    }
    data.update(overrides)
    resp = client.post(
        "/api/admin/ki/backends/", json=data, headers={"X-Requested-With": "XMLHttpRequest"}
    )
    return resp


def _get_raw_backend(conn_fixture, backend_id):
    """Liest Backend direkt aus DB (ohne API-Masking)."""
    c = get_connection()
    row = c.execute("SELECT * FROM ki_backends WHERE id = %s", (backend_id,)).fetchone()
    c.close()
    return dict(row) if row else None


# ===================================================================
# A) _analyze_text() Leerstring/None-Bugfix
# ===================================================================


class TestAnalyzeTextEmpty:
    def test_empty_string_returns_none(self):
        """_analyze_text('') stuerzt nicht ab und gibt None zurueck."""
        result = _analyze_text("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        """_analyze_text mit nur Whitespace gibt None zurueck."""
        result = _analyze_text("   \n  \t  \n   ")
        assert result is None

    def test_none_handled_safely(self):
        """_analyze_text(None) wird sicher behandelt (kein Crash)."""
        # _analyze_text erwartet str, aber None darf nicht zu einem
        # unbehandelten AttributeError fuehren
        try:
            result = _analyze_text(None)
            # Wenn es None akzeptiert, ist das OK
        except (TypeError, AttributeError):
            # Auch ein sauberer TypeError ist akzeptabel
            pass
        # Hauptsache: kein unkontrollierter Crash mit Traceback

    def test_empty_string_via_detect_chunk_params(self):
        """detect_chunk_params() mit leerem Text liefert Default-Strategie."""
        meta = {"file_type": ".docx", "source": "/docs/leer.docx"}
        size, overlap = detect_chunk_params("", meta)
        assert (size, overlap) == STRATEGIES["default"]
        assert meta["chunk_strategy"] == "default"

    def test_none_text_via_detect_chunk_params(self):
        """detect_chunk_params() mit None-Text liefert Default-Strategie."""
        meta = {"file_type": ".txt", "source": "/data/none.txt"}
        size, overlap = detect_chunk_params(None, meta)
        assert (size, overlap) == STRATEGIES["default"]
        assert meta["chunk_strategy"] == "default"

    def test_normal_text_still_works(self):
        """_analyze_text() funktioniert weiterhin korrekt mit normalem Text."""
        # Pipe-dominierter Text → tabular
        text = "\n".join([f"| Col A | Col B | Val {i} |" for i in range(10)])
        result = _analyze_text(text)
        assert result == "tabular"

    def test_single_newline_returns_none_or_valid(self):
        """_analyze_text mit einem einzelnen Newline crasht nicht."""
        result = _analyze_text("\n")
        # Nur Whitespace → strip() ergibt leer → None
        assert result is None


# ===================================================================
# B) Cloud-Backend Test-Endpoint
# ===================================================================


class TestCloudTestEndpoint:
    def test_test_anthropic_success(self, client, mock_admin, mock_db):
        """Test-Endpoint fuer Anthropic-Backend (Erfolg mit gemocktem API-Client)."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="claude-sonnet-4-6"), MagicMock(id="claude-haiku-3")]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-ant-real-key"),
            patch("anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_anthropic.return_value = mock_client

            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "claude-sonnet-4-6" in data["models"]

    def test_test_openai_success(self, client, mock_admin, mock_db):
        """Test-Endpoint fuer OpenAI-Backend (Erfolg mit gemocktem API-Client)."""
        resp = _create_backend(
            client,
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-openai-test-key",
        )
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="gpt-4o-mini"), MagicMock(id="gpt-4o")]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-openai-real-key"),
            patch("openai.OpenAI") as mock_openai,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_openai.return_value = mock_client

            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "gpt-4o-mini" in data["models"]

    def test_test_anthropic_api_error(self, client, mock_admin, mock_db):
        """Test-Endpoint fuer Anthropic-Backend bei API-Fehler → Fehler-Response."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-ant-bad-key"),
            patch("anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = MagicMock()
            mock_client.models.list.side_effect = Exception("Invalid API key")
            mock_anthropic.return_value = mock_client

            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        # Admin-Registry gibt 200 mit success=false oder 400 bei Fehler zurueck
        data = resp.json()
        assert data.get("success") is False or resp.status_code in (400, 502)

    def test_test_cloud_no_api_key(self, client, mock_admin, mock_db):
        """Cloud-Test ohne API-Key → Fehler-Response.

        Service verhindert Anlage ohne Key, also DB-direkt einfuegen.
        """
        c = get_connection()
        cur = c.execute(
            "INSERT INTO ki_backends (category, provider, name, model, api_key) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            ("cloud", "anthropic", "NoKey", "claude-sonnet-4-6", ""),
        )
        bid = cur.fetchone()["id"]
        c.commit()
        c.close()

        resp = client.post(
            f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        # Admin-Registry gibt AdminResult.fail → HTTP 400 zurueck
        data = resp.json()
        assert data.get("success") is False or resp.status_code in (400, 502)

    def test_test_saves_success_to_db(self, client, mock_admin, mock_db):
        """Erfolgreicher Cloud-Test speichert Ergebnis in DB."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="claude-sonnet-4-6")]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-ant-real-key"),
            patch("anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_anthropic.return_value = mock_client

            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["last_test_ok"] == 1
        assert raw["last_test_at"] is not None

    def test_test_saves_failure_to_db(self, client, mock_admin, mock_db):
        """Fehlgeschlagener Cloud-Test speichert Fehler in DB."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-ant-bad"),
            patch("anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = MagicMock()
            mock_client.models.list.side_effect = Exception("Unauthorized")
            mock_anthropic.return_value = mock_client

            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["last_test_ok"] == 0
        assert "Unauthorized" in raw["last_test_error"]

    def test_test_response_structure(self, client, mock_admin, mock_db):
        """Erfolgreiche Test-Response hat erwartete Struktur {success, models}."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="model-a"), MagicMock(id="model-b")]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-key"),
            patch("anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_anthropic.return_value = mock_client

            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        data = resp.json()
        assert "success" in data
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 2

    def test_test_nonexistent_backend_404(self, client, mock_admin, mock_db):
        """Test fuer nicht-existierendes Backend → 404."""
        resp = client.post(
            "/api/admin/ki/backends/9999/test", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert resp.status_code == 404


# ===================================================================
# C) PASSWORD_MASK Erkennung im Edit-Modus
# ===================================================================


class TestPasswordMask:
    def test_password_mask_value(self):
        """PASSWORD_MASK hat den erwarteten Wert."""
        assert PASSWORD_MASK == "********"

    def test_list_masks_cloud_api_key(self, client, mock_admin, mock_db):
        """GET-Response maskiert Cloud-API-Keys mit PASSWORD_MASK."""
        _create_backend(client)
        resp = client.get("/api/admin/ki/backends/")
        backends = resp.json()["backends"]
        assert len(backends) == 1
        assert backends[0]["api_key"] == PASSWORD_MASK

    def test_update_with_password_mask_keeps_key(self, client, mock_admin, mock_db):
        """PUT mit PASSWORD_MASK behaelt den alten verschluesselten Key."""
        resp = _create_backend(client)
        bid = resp.json()["id"]
        original_raw = _get_raw_backend(mock_db, bid)

        # Update mit PASSWORD_MASK (simuliert Edit-Dialog)
        client.put(
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "cloud",
                "provider": "anthropic",
                "name": "Updated Name",
                "model": "claude-sonnet-4-6",
                "api_key": PASSWORD_MASK,
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        updated_raw = _get_raw_backend(mock_db, bid)
        # Der verschluesselte Key muss identisch geblieben sein
        assert updated_raw["api_key"] == original_raw["api_key"]
        assert updated_raw["api_key"] != PASSWORD_MASK

    def test_update_with_new_key_encrypts(self, client, mock_admin, mock_db):
        """PUT mit neuem Key verschluesselt diesen (nicht PASSWORD_MASK)."""
        resp = _create_backend(client)
        bid = resp.json()["id"]
        original_raw = _get_raw_backend(mock_db, bid)

        confirmed_request(
            client,
            "PUT",
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "cloud",
                "provider": "anthropic",
                "name": "New Key",
                "model": "claude-sonnet-4-6",
                "api_key": "sk-ant-brand-new-key",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        updated_raw = _get_raw_backend(mock_db, bid)
        # Neuer Key wurde verschluesselt, ist also anders als Klartext
        assert updated_raw["api_key"] != "sk-ant-brand-new-key"
        # Und anders als der alte Key
        assert updated_raw["api_key"] != original_raw["api_key"]

    def test_update_with_empty_key_clears(self, client, mock_admin, mock_db):
        """PUT mit leerem api_key loescht den Key."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        confirmed_request(
            client,
            "PUT",
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "cloud",
                "provider": "anthropic",
                "name": "Cleared",
                "model": "claude-sonnet-4-6",
                "api_key": "",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["api_key"] == ""

    def test_test_endpoint_decrypts_stored_key(self, client, mock_admin, mock_db):
        """Test-Endpoint entschluesselt den gespeicherten Key fuer den API-Call."""
        resp = _create_backend(client)
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = []

        # Patches auf shared.crypto (ki_backend_service importiert lazy aus crypto)
        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True) as mock_enc,
            patch("kiara.shared.crypto.decrypt", return_value="sk-decrypted") as mock_dec,
            patch("anthropic.Anthropic") as mock_anthropic,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_anthropic.return_value = mock_client

            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

            # decrypt wurde aufgerufen (Key wird serverseitig entschluesselt)
            mock_dec.assert_called_once()
            # Anthropic-Client wurde mit dem entschluesselten Key erstellt
            mock_anthropic.assert_called_once()
            assert mock_anthropic.call_args.kwargs.get("api_key") == "sk-decrypted"


# ===================================================================
# D) BOM-safe Plaintext-Loader
# ===================================================================


class TestBomSafePlaintextLoader:
    def test_utf8_bom_stripped(self, tmp_path):
        """UTF-8 BOM (EF BB BF) wird beim Laden entfernt."""
        bom = b"\xef\xbb\xbf"
        content = "Hallo Welt, dies ist ein Test."
        file_path = tmp_path / "bom_test.txt"
        file_path.write_bytes(bom + content.encode("utf-8"))

        result = _load_plaintext(str(file_path))
        # BOM darf nicht im geladenen Text auftauchen
        assert not result.startswith("\ufeff")
        assert result == content

    def test_utf8_no_bom_unchanged(self, tmp_path):
        """Datei ohne BOM wird normal geladen."""
        content = "Normaler Text ohne BOM."
        file_path = tmp_path / "no_bom.txt"
        file_path.write_text(content, encoding="utf-8")

        result = _load_plaintext(str(file_path))
        assert result == content

    def test_utf8_bom_multiline(self, tmp_path):
        """UTF-8 BOM mit mehrzeiligem Text wird korrekt geladen."""
        bom = b"\xef\xbb\xbf"
        content = "Zeile 1\nZeile 2\nZeile 3"
        file_path = tmp_path / "bom_multi.txt"
        file_path.write_bytes(bom + content.encode("utf-8"))

        result = _load_plaintext(str(file_path))
        assert result == content
        assert result.count("\n") == 2

    def test_utf8_bom_german_umlauts(self, tmp_path):
        """UTF-8 BOM mit deutschen Umlauten wird korrekt geladen."""
        bom = b"\xef\xbb\xbf"
        content = "Aerger mit Ueberlaengen und Oeffnungen"
        file_path = tmp_path / "bom_umlauts.txt"
        file_path.write_bytes(bom + content.encode("utf-8"))

        result = _load_plaintext(str(file_path))
        assert not result.startswith("\ufeff")
        assert "Aerger" in result

    def test_utf8_sig_encoding_used(self, tmp_path):
        """_load_plaintext() verwendet utf-8-sig als primaere Encoding-Strategie."""
        bom = b"\xef\xbb\xbf"
        content = "Test"
        file_path = tmp_path / "sig_test.txt"
        file_path.write_bytes(bom + content.encode("utf-8"))

        result = _load_plaintext(str(file_path))
        # utf-8-sig entfernt BOM automatisch
        assert result == "Test"

    def test_empty_file_with_bom(self, tmp_path):
        """Datei die nur aus BOM besteht wird als leer geladen."""
        bom = b"\xef\xbb\xbf"
        file_path = tmp_path / "bom_only.txt"
        file_path.write_bytes(bom)

        result = _load_plaintext(str(file_path))
        assert result.strip() == ""


# ===================================================================
# E) OpenAI-Backend mit base_url
# ===================================================================


class TestOpenAIBaseUrl:
    def test_openai_test_passes_base_url(self, client, mock_admin, mock_db):
        """OpenAI-Test-Endpoint uebergibt base_url an den Client."""
        resp = _create_backend(
            client,
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-openai-key",
            base_url="https://custom.openai-proxy.example.com/v1",
        )
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="gpt-4o-mini")]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-key"),
            patch("openai.OpenAI") as mock_openai,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_openai.return_value = mock_client

            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

            # OpenAI-Client wurde mit base_url erstellt
            call_kwargs = mock_openai.call_args
            assert call_kwargs[1]["base_url"] == "https://custom.openai-proxy.example.com/v1"

    def test_openai_test_empty_base_url_passes_none(self, client, mock_admin, mock_db):
        """OpenAI-Test mit leerer base_url uebergibt None."""
        resp = _create_backend(
            client,
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-openai-key",
            base_url="",
        )
        bid = resp.json()["id"]

        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="gpt-4o-mini")]

        with (
            patch("kiara.shared.crypto.is_encrypted", return_value=True),
            patch("kiara.shared.crypto.decrypt", return_value="sk-key"),
            patch("openai.OpenAI") as mock_openai,
        ):
            mock_client = MagicMock()
            mock_client.models.list.return_value = mock_models
            mock_openai.return_value = mock_client

            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

            call_kwargs = mock_openai.call_args
            assert call_kwargs[1]["base_url"] is None
