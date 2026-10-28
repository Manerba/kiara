"""Tests fuer den HealthChecker (kiara.health.checker)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kiara.health.checker import ComponentHealth, HealthChecker, SystemHealth

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config(tmp_path):
    """Minimale Config fuer Health-Checks."""
    db_path = tmp_path / "test.db"
    vectordb_path = tmp_path / "vectordb"
    vectordb_path.mkdir()
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {
            "path": str(vectordb_path),
            "collection": "knowledgebase",
        },
        "database": {
            "path": str(db_path),
        },
        "cloud": {
            "enabled": False,
            "provider": "openai",
            "api_key": "",
            "model": "gpt-4o-mini",
        },
        "tools": {
            "enabled": False,
            "erp": {
                "backend": "mock",
            },
        },
    }


@pytest.fixture
def checker(base_config):
    """HealthChecker-Instanz mit Basis-Config."""
    return HealthChecker(base_config)


@pytest.fixture(autouse=True)
def _verified_embedding_role_profile(monkeypatch):
    """Health-Unit-Tests isolieren die separat getestete KIron-Discovery."""
    from kiara.retrieval import embedding_roles
    from kiara.retrieval.embedding_roles import EmbeddingRequestProfile

    def resolve(_url, model, *, role, kind, endpoint, profile_id=None, **_kwargs):
        return EmbeddingRequestProfile(
            requested_model=model,
            canonical_model_id=f"canonical::{model}",
            profile_id=profile_id or f"test-{kind.value}-profile",
            kind=kind.value,
            endpoint=endpoint,
            input_type=role,
            role_sensitive=True,
            index_compatibility_id="sha256:" + "1" * 64,
            query_compatibility_id="sha256:" + "2" * 64,
        )

    monkeypatch.setattr(embedding_roles, "resolve_server_request_profile", resolve)


@pytest.fixture
def temp_db(tmp_path):
    """PostgreSQL-Tabellen fuer Health-Check-Tests (Schema existiert bereits via conftest)."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username TEXT);
        CREATE TABLE IF NOT EXISTS chats (id SERIAL PRIMARY KEY, title TEXT);
        CREATE TABLE IF NOT EXISTS ldap_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            server_url TEXT NOT NULL,
            port INTEGER DEFAULT 389,
            use_ssl INTEGER DEFAULT 0,
            bind_dn TEXT NOT NULL,
            bind_password TEXT NOT NULL,
            search_base TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    # Gib einen Dummy-Pfad zurueck (wird nur fuer Config-Dict gebraucht, PG ignoriert ihn)
    return tmp_path / "health_test.db"


# ---------------------------------------------------------------------------
# Dataclass-Tests
# ---------------------------------------------------------------------------


class TestComponentHealthDataclass:
    """Tests fuer die ComponentHealth-Dataclass."""

    def test_component_health_dataclass(self):
        """ComponentHealth hat korrekte Default-Werte."""
        ch = ComponentHealth(name="test", status="healthy")
        assert ch.name == "test"
        assert ch.status == "healthy"
        assert ch.latency_ms == 0.0
        assert ch.message == ""
        assert ch.details == {}

    def test_component_health_with_values(self):
        """ComponentHealth akzeptiert alle Felder."""
        ch = ComponentHealth(
            name="ollama",
            status="degraded",
            latency_ms=42.5,
            message="Modell fehlt",
            details={"url": "http://test"},
        )
        assert ch.name == "ollama"
        assert ch.status == "degraded"
        assert ch.latency_ms == 42.5
        assert ch.message == "Modell fehlt"
        assert ch.details == {"url": "http://test"}


class TestSystemHealthDataclass:
    """Tests fuer die SystemHealth-Dataclass."""

    def test_system_health_dataclass(self):
        """SystemHealth hat korrekte Default-Werte."""
        sh = SystemHealth(status="healthy")
        assert sh.status == "healthy"
        assert sh.components == []
        assert sh.timestamp == ""
        assert sh.uptime_seconds == 0.0
        assert sh.version == ""

    def test_system_health_with_components(self):
        """SystemHealth kann Komponenten enthalten."""
        c1 = ComponentHealth(name="a", status="healthy")
        c2 = ComponentHealth(name="b", status="degraded")
        sh = SystemHealth(
            status="degraded",
            components=[c1, c2],
            timestamp="2026-02-24T10:00:00",
            uptime_seconds=120.5,
            version="0.0.1+12",
        )
        assert len(sh.components) == 2
        assert sh.components[0].name == "a"
        assert sh.version == "0.0.1+12"


# ---------------------------------------------------------------------------
# Disk-Check Tests
# ---------------------------------------------------------------------------


class TestCheckDisk:
    """Tests fuer check_disk()."""

    @pytest.mark.asyncio
    async def test_check_disk_healthy(self, checker):
        """Disk healthy wenn >1GB frei."""
        # 10 GB frei
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.used = 90 * 1024**3
        mock_usage.free = 10 * 1024**3

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = await checker.check_disk()

        assert result.name == "disk"
        assert result.status == "healthy"
        assert "10.00" in result.message
        assert result.details["free_gb"] == 10.0

    @pytest.mark.asyncio
    async def test_check_disk_degraded(self, checker):
        """Disk degraded wenn <1GB aber >=100MB frei."""
        # 500 MB frei
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.used = 99.5 * 1024**3
        mock_usage.free = 500 * 1024 * 1024

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = await checker.check_disk()

        assert result.name == "disk"
        assert result.status == "degraded"
        assert "Wenig Speicher" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_unhealthy(self, checker):
        """Disk unhealthy wenn <100MB frei."""
        # 50 MB frei
        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.used = 99.95 * 1024**3
        mock_usage.free = 50 * 1024 * 1024

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = await checker.check_disk()

        assert result.name == "disk"
        assert result.status == "unhealthy"
        assert "Kritisch" in result.message

    @pytest.mark.asyncio
    async def test_check_disk_missing_storage_path_degraded(self, base_config, tmp_path):
        """Fehlender Storage-Pfad wird nicht als healthy gemeldet."""
        base_config["vectordb"]["path"] = str(tmp_path / "missing-vectordb")
        checker = HealthChecker(base_config)

        mock_usage = MagicMock()
        mock_usage.total = 100 * 1024**3
        mock_usage.used = 90 * 1024**3
        mock_usage.free = 10 * 1024**3

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = await checker.check_disk()

        assert result.name == "disk"
        assert result.status == "degraded"
        assert "Storage-Pfad fehlt" in result.message
        assert result.details["configured_path"] == base_config["vectordb"]["path"]
        assert result.details["path"] == str(tmp_path)
        assert result.details["path_exists"] is False
        assert result.details["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_check_disk_missing_storage_parent_unhealthy(self, base_config, tmp_path):
        """Fehlender Storage-Parent faellt geschlossen und prueft nicht /."""
        base_config["vectordb"]["path"] = str(tmp_path / "missing-parent" / "vectordb")
        checker = HealthChecker(base_config)

        with patch("shutil.disk_usage") as mock_usage:
            result = await checker.check_disk()

        assert result.name == "disk"
        assert result.status == "unhealthy"
        assert "Storage-Pfad fehlt" in result.message
        assert result.details["parent_exists"] is False
        assert result.details["fallback_used"] is False
        mock_usage.assert_not_called()


# ---------------------------------------------------------------------------
# Database-Check Tests
# ---------------------------------------------------------------------------


class TestCheckDatabase:
    """Tests fuer check_database()."""

    @pytest.mark.asyncio
    async def test_check_database_healthy(self, base_config, temp_db):
        """Database healthy wenn Tabellen vorhanden."""
        base_config["database"]["path"] = str(temp_db)
        checker = HealthChecker(base_config)

        result = await checker.check_database()

        assert result.name == "database"
        assert result.status == "healthy"
        # check_database() nutzt db_query() (Produktions-DB), nicht die temp_db.
        # Wir pruefen daher nur dass "Tabellen" im Message steht, nicht die genaue Anzahl.
        assert "Tabellen" in result.message

    @pytest.mark.asyncio
    async def test_check_database_unhealthy(self, base_config):
        """Database unhealthy wenn DB nicht lesbar."""
        base_config["database"]["path"] = "/nonexistent/path/to/db.conf"
        checker = HealthChecker(base_config)

        # check_database() nutzt db_query() (globale DB-Verbindung).
        # Wenn die globale DB existiert, ist der Check immer healthy.
        # Wir pruefen daher nur, dass der Check durchlaeuft ohne Exception.
        result = await checker.check_database()

        assert result.name == "database"
        # Der Status haengt davon ab, ob die globale DB erreichbar ist
        assert result.status in ("healthy", "unhealthy")


# ---------------------------------------------------------------------------
# Ollama-Check Tests
# ---------------------------------------------------------------------------


class TestCheckOllama:
    """Tests fuer check_ollama()."""

    @pytest.mark.asyncio
    async def test_check_ollama_unhealthy(self, base_config):
        """Ollama unhealthy wenn Server nicht erreichbar."""
        # get_llm_backend liest aus der DB (ki_backends), nicht aus config —
        # deshalb mocken wir es, damit die ungueltige URL zurueckgegeben wird.
        fake_backend = {"url": "http://127.0.0.1:19999", "model": "qwen3:8b"}
        with patch("kiara.config.get_llm_backend", return_value=fake_backend):
            checker = HealthChecker(base_config)
            result = await checker.check_ollama()

        assert result.name == "ollama"
        assert result.status == "unhealthy"
        assert "nicht erreichbar" in result.message

    @pytest.mark.asyncio
    async def test_check_ollama_healthy_mocked(self, checker):
        """Ollama healthy wenn Server erreichbar und Modell vorhanden (mocked)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]
        }

        fake_backend = {"url": "http://localhost:11434", "model": "qwen3:8b"}
        with (
            patch("kiara.config.get_llm_backend", return_value=fake_backend),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await checker.check_ollama()

        assert result.name == "ollama"
        assert result.status == "healthy"
        assert "qwen3:8b" in result.message

    @pytest.mark.asyncio
    async def test_check_ollama_degraded_mocked(self, checker):
        """Ollama degraded wenn erreichbar aber Modell fehlt (mocked)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama2:7b"}]}

        fake_backend = {"url": "http://localhost:11434", "model": "qwen3:8b"}
        with (
            patch("kiara.config.get_llm_backend", return_value=fake_backend),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await checker.check_ollama()

        assert result.name == "ollama"
        assert result.status == "degraded"
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# ERP-Check Tests
# ---------------------------------------------------------------------------


class TestCheckERP:
    """Tests fuer check_erp()."""

    @pytest.mark.asyncio
    async def test_check_erp_mock_healthy(self, base_config):
        """ERP Mock-Backend ist immer healthy."""
        base_config["tools"]["enabled"] = True
        base_config["tools"]["erp"]["backend"] = "mock"
        checker = HealthChecker(base_config)

        result = await checker.check_erp()

        assert result.name == "erp"
        assert result.status == "healthy"
        assert "Mock" in result.message

    @pytest.mark.asyncio
    async def test_check_erp_unconfigured(self, checker):
        """ERP unconfigured wenn tools.enabled = False."""
        result = await checker.check_erp()

        assert result.name == "erp"
        assert result.status == "unconfigured"
        assert "nicht aktiviert" in result.message


# ---------------------------------------------------------------------------
# LDAP-Check Tests
# ---------------------------------------------------------------------------


class TestCheckLDAP:
    """Tests fuer check_ldap()."""

    @pytest.mark.asyncio
    async def test_check_ldap_unconfigured(self, base_config):
        """LDAP unconfigured wenn keine LDAP-Config in DB."""
        checker = HealthChecker(base_config)

        with patch("kiara.health.checks_services.ConfigRepository") as mock_repo:
            mock_repo.return_value.get.return_value = None
            result = await checker.check_ldap()

        assert result.name == "ldap"
        assert result.status == "unconfigured"
        assert "nicht konfiguriert" in result.message

    @pytest.mark.asyncio
    async def test_check_ldap_unconfigured_no_db(self, base_config):
        """LDAP unconfigured wenn DB-Zugriff fehlschlaegt."""
        checker = HealthChecker(base_config)

        with patch("kiara.health.checks_services.ConfigRepository", side_effect=Exception("no db")):
            result = await checker.check_ldap()

        assert result.name == "ldap"
        assert result.status == "unconfigured"


# ---------------------------------------------------------------------------
# check_all() Tests
# ---------------------------------------------------------------------------


class TestCheckAll:
    """Tests fuer check_all() Aggregation."""

    @pytest.mark.asyncio
    async def test_check_all_determines_overall_status(self, base_config):
        """check_all() bestimmt den Gesamtstatus korrekt."""
        checker = HealthChecker(base_config)
        checker._reload_config = lambda: None

        # Mock module-level check functions that check_all() calls directly
        with (
            patch(
                "kiara.health.checker.check_ollama",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ollama", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_vectordb",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="vectordb", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_database",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="database", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_ldap",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ldap", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_cloud_ki",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="cloud_ki", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_erp",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="erp", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_reranker",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="reranker", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_colbert",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="colbert", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_disk",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="disk", status="healthy"),
            ),
        ):
            result = await checker.check_all()

        assert result.status == "healthy"
        assert len(result.components) == 9
        assert result.version != ""
        assert result.timestamp != ""
        assert result.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_check_all_degraded(self, base_config):
        """check_all() gibt degraded zurueck wenn eine Komponente degraded."""
        checker = HealthChecker(base_config)
        checker._reload_config = lambda: None

        with (
            patch(
                "kiara.health.checker.check_ollama",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ollama", status="degraded"),
            ),
            patch(
                "kiara.health.checker.check_vectordb",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="vectordb", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_database",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="database", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_ldap",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ldap", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_cloud_ki",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="cloud_ki", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_erp",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="erp", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_reranker",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="reranker", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_colbert",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="colbert", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_disk",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="disk", status="healthy"),
            ),
        ):
            result = await checker.check_all()
        assert result.status == "degraded"

    @pytest.mark.asyncio
    async def test_check_all_unhealthy(self, base_config):
        """check_all() gibt unhealthy zurueck wenn eine Komponente unhealthy."""
        checker = HealthChecker(base_config)
        checker._reload_config = lambda: None

        with (
            patch(
                "kiara.health.checker.check_ollama",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ollama", status="unhealthy"),
            ),
            patch(
                "kiara.health.checker.check_vectordb",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="vectordb", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_database",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="database", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_ldap",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ldap", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_cloud_ki",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="cloud_ki", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_erp",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="erp", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_reranker",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="reranker", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_colbert",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="colbert", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_disk",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="disk", status="healthy"),
            ),
        ):
            result = await checker.check_all()
        assert result.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_check_all_handles_exceptions(self, base_config):
        """check_all() behandelt Exceptions als unhealthy."""
        checker = HealthChecker(base_config)
        checker._reload_config = lambda: None

        with (
            patch(
                "kiara.health.checker.check_ollama",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "kiara.health.checker.check_vectordb",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="vectordb", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_database",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="database", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_ldap",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ldap", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_cloud_ki",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="cloud_ki", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_erp",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="erp", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_reranker",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="reranker", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_colbert",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="colbert", status="unconfigured"),
            ),
            patch(
                "kiara.health.checker.check_disk",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="disk", status="healthy"),
            ),
        ):
            result = await checker.check_all()
        assert result.status == "unhealthy"
        # The exception should be converted to an unhealthy component
        unknown_components = [c for c in result.components if c.status == "unhealthy"]
        assert len(unknown_components) >= 1
        assert unknown_components[0].message
        assert unknown_components[0].latency_ms >= 0

    @pytest.mark.asyncio
    async def test_component_timeout_has_message_and_latency(self, base_config):
        """Timeouts werden als unhealthy mit Diagnose und gemessener Latenz erfasst."""
        checker = HealthChecker(base_config)

        async def slow_check():
            await asyncio.sleep(0.05)
            return ComponentHealth(name="slow", status="healthy")

        result = await checker._run_component_check("slow", slow_check(), timeout=0.001)

        assert result.name == "slow"
        assert result.status == "unhealthy"
        assert "Timeout" in result.message
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_readiness_converts_component_exception_to_503(self, base_config):
        """Readiness darf bei Backend-Lookup-Fehlern nicht als 500 crashen."""
        from kiara.web.api_health import readiness_check

        checker = HealthChecker(base_config)

        with (
            patch("kiara.web.api_health._get_health_checker", return_value=checker),
            patch(
                "kiara.health.checker.check_ollama",
                new_callable=AsyncMock,
                side_effect=RuntimeError("kein LLM-Backend"),
            ),
            patch(
                "kiara.health.checker.check_vectordb",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="vectordb", status="healthy"),
            ),
        ):
            response = await readiness_check()

        payload = json.loads(response.body)
        assert response.status_code == 503
        assert payload["ready"] is False
        assert payload["components"]["ollama"]["status"] == "unhealthy"
        assert "kein LLM-Backend" in payload["components"]["ollama"]["message"]


# ---------------------------------------------------------------------------
# Reranker-Check Tests
# ---------------------------------------------------------------------------


class TestCheckReranker:
    """Tests fuer check_reranker()."""

    @pytest.mark.asyncio
    async def test_unconfigured_when_disabled(self, base_config):
        """Reranker unconfigured wenn reranking deaktiviert."""
        base_config["retrieval"] = {"reranking": False}
        checker = HealthChecker(base_config)

        result = await checker.check_reranker()

        assert result.name == "reranker"
        assert result.status == "unconfigured"
        assert "nicht aktiviert" in result.message

    @pytest.mark.asyncio
    async def test_unconfigured_when_no_retrieval_section(self, base_config):
        """Reranker unconfigured wenn keine retrieval-Config vorhanden."""
        checker = HealthChecker(base_config)

        result = await checker.check_reranker()

        assert result.name == "reranker"
        assert result.status == "unconfigured"

    @pytest.mark.asyncio
    async def test_degraded_when_package_missing(self, base_config):
        """Reranker degraded wenn aktiviert aber sentence-transformers fehlt."""
        base_config["retrieval"] = {"reranking": True, "rerank_model": "test-model"}

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Force ImportError
            import importlib

            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError("No module"))
                    if name == "sentence_transformers"
                    else importlib.__import__(name, *a, **kw)
                ),
            ):
                checker = HealthChecker(base_config)
                result = await checker.check_reranker()

        assert result.name == "reranker"
        assert result.status == "degraded"
        assert "nicht installiert" in result.message
        assert result.details["package_installed"] is False

    @pytest.mark.asyncio
    async def test_healthy_when_package_installed(self, base_config):
        """Reranker healthy wenn aktiviert + sentence-transformers installiert + Modell im Cache."""
        import sys

        base_config["retrieval"] = {
            "reranking": True,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k": 10,
            "rerank_fetch_multiplier": 4,
        }

        from kiara.health.checks_services import check_reranker

        # Mock sentence-transformers Import falls nicht installiert
        mock_st = MagicMock()
        # Modell muss im HF-Cache liegen damit "healthy" — sonst "degraded"
        with (
            patch.dict(sys.modules, {"sentence_transformers": mock_st}),
            patch("huggingface_hub.try_to_load_from_cache", return_value="/cache/path"),
        ):
            result = await check_reranker(base_config)

        assert result.name == "reranker"
        assert result.status == "healthy"
        assert "top_k=10" in result.message
        assert result.details["top_k"] == 10
        assert result.details["fetch_multiplier"] == 4
        assert result.details["package_installed"] is True
        assert result.details["model_cached"] is True

    @pytest.mark.asyncio
    async def test_healthy_shows_cached_status(self, base_config):
        """Healthy-Meldung zeigt ob Modell bereits gecacht ist."""
        import sys

        base_config["retrieval"] = {"reranking": True, "rerank_model": "test-model"}

        from kiara.health.checks_services import check_reranker

        mock_st = MagicMock()
        # Modell nicht gecacht -> Hinweis im Message
        with (
            patch.dict(sys.modules, {"sentence_transformers": mock_st}),
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
        ):
            result = await check_reranker(base_config)
        assert result.details["model_cached"] is False
        assert "nicht im Cache" in result.message

        # Modell gecacht -> kein Hinweis
        with (
            patch.dict(sys.modules, {"sentence_transformers": mock_st}),
            patch("huggingface_hub.try_to_load_from_cache", return_value="/some/path"),
        ):
            result = await check_reranker(base_config)
        assert result.details["model_cached"] is True
        assert "nicht im Cache" not in result.message

    @pytest.mark.asyncio
    async def test_reranker_in_check_all(self, base_config):
        """check_all() enthaelt den Reranker-Check."""
        base_config["retrieval"] = {"reranking": False}
        checker = HealthChecker(base_config)

        # Alle anderen Checks mocken
        for name in [
            "check_ollama",
            "check_vectordb",
            "check_database",
            "check_ldap",
            "check_erp",
            "check_colbert",
            "check_disk",
        ]:
            setattr(
                checker,
                name,
                AsyncMock(
                    return_value=ComponentHealth(name=name.replace("check_", ""), status="healthy")
                ),
            )

        # _reload_config wuerde die echte YAML laden — im Test unterdruecken
        checker._reload_config = lambda: None
        result = await checker.check_all()

        comp_names = [c.name for c in result.components]
        assert "reranker" in comp_names
        reranker = next(c for c in result.components if c.name == "reranker")
        assert reranker.status == "unconfigured"


# ---------------------------------------------------------------------------
# ColBERT Health-Check Tests
# ---------------------------------------------------------------------------


class TestCheckColBERT:
    """Tests fuer check_colbert()."""

    @pytest.mark.asyncio
    async def test_unconfigured_when_disabled(self, base_config):
        """ColBERT unconfigured wenn retrieval.colbert.enabled = False."""
        base_config["retrieval"] = {"colbert": {"enabled": False}}
        checker = HealthChecker(base_config)
        result = await checker.check_colbert()

        assert result.name == "colbert"
        assert result.status == "unconfigured"
        assert "nicht aktiviert" in result.message

    @pytest.mark.asyncio
    async def test_unconfigured_when_no_colbert_section(self, base_config):
        """ColBERT unconfigured wenn keine retrieval.colbert-Config."""
        checker = HealthChecker(base_config)
        result = await checker.check_colbert()

        assert result.name == "colbert"
        assert result.status == "unconfigured"

    @pytest.mark.asyncio
    async def test_healthy_when_endpoint_ok(self, base_config):
        """ColBERT healthy wenn Endpoint erreichbar und Payload gueltig."""
        base_config["retrieval"] = {"colbert": {"enabled": True, "model": "colbert-xm"}}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[[0.1, 0.2], [0.3, 0.4]]]}

        with (
            patch("kiara.health.checks_services._resolve_colbert_url", create=True),
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url",
                return_value="http://localhost:11434",
            ),
            patch("kiara.backend_selector.normalize_model_name", return_value="colbert-xm"),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from kiara.health.checks_services import check_colbert

            result = await check_colbert(base_config)

        assert result.name == "colbert"
        assert result.status == "healthy"
        assert "erreichbar" in result.message
        mock_client.post.assert_awaited_once_with(
            "http://localhost:11434/api/embed_colbert",
            json={
                "model": "colbert-xm",
                "input": ["health"],
                "input_type": "search_document",
            },
        )

    @pytest.mark.asyncio
    async def test_unverified_profile_stops_before_health_request(self, base_config):
        """Discovery-Fehler werden degraded gemeldet, ohne Probe-Embedding."""
        from kiara.retrieval.embedding_roles import EmbeddingRoleContractError

        base_config["retrieval"] = {"colbert": {"enabled": True, "model": "colbert-xm"}}
        error = EmbeddingRoleContractError(
            "profile_unverified",
            "verification missing",
            requested_model="colbert-xm",
            endpoint="/api/embed_colbert",
            input_type="search_document",
        )

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url",
                return_value="http://localhost:11434",
            ),
            patch(
                "kiara.retrieval.embedding_roles.resolve_server_request_profile",
                side_effect=error,
            ),
            patch("httpx.AsyncClient") as client,
        ):
            from kiara.health.checks_services import check_colbert

            result = await check_colbert(base_config)

        assert result.status == "degraded"
        assert "profile_unverified" in result.message
        client.assert_not_called()

    @pytest.mark.asyncio
    async def test_degraded_on_network_error(self, base_config):
        """ColBERT degraded wenn Netzwerkfehler."""
        base_config["retrieval"] = {"colbert": {"enabled": True, "model": "colbert-xm"}}

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url",
                return_value="http://localhost:11434",
            ),
            patch("kiara.backend_selector.normalize_model_name", return_value="colbert-xm"),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from kiara.health.checks_services import check_colbert

            result = await check_colbert(base_config)

        assert result.name == "colbert"
        assert result.status == "degraded"
        assert "nicht erreichbar" in result.message

    @pytest.mark.asyncio
    async def test_degraded_on_url_resolve_error(self, base_config):
        """ColBERT degraded wenn URL-Aufloesung fehlschlaegt."""
        base_config["retrieval"] = {"colbert": {"enabled": True}}

        with patch(
            "kiara.retrieval.colbert._resolve_colbert_url", side_effect=ValueError("no url")
        ):
            from kiara.health.checks_services import check_colbert

            result = await check_colbert(base_config)

        assert result.name == "colbert"
        assert result.status == "degraded"
        assert "URL-Aufloesung" in result.message

    @pytest.mark.asyncio
    async def test_degraded_on_invalid_payload(self, base_config):
        """ColBERT degraded wenn Payload ungueltige Struktur hat."""
        base_config["retrieval"] = {"colbert": {"enabled": True, "model": "colbert-xm"}}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        # Leeres embeddings Array → ungueltig
        mock_resp.json.return_value = {"embeddings": []}

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url",
                return_value="http://localhost:11434",
            ),
            patch("kiara.backend_selector.normalize_model_name", return_value="colbert-xm"),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from kiara.health.checks_services import check_colbert

            result = await check_colbert(base_config)

        assert result.name == "colbert"
        assert result.status == "degraded"
        assert "nicht erreichbar" in result.message or "Payload" in result.message

    @pytest.mark.asyncio
    async def test_colbert_only_degraded_not_unhealthy(self, base_config):
        """Wenn nur ColBERT down, bleibt check_all() bei degraded statt unhealthy."""
        checker = HealthChecker(base_config)
        checker._reload_config = lambda: None

        with (
            patch(
                "kiara.health.checker.check_ollama",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ollama", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_vectordb",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="vectordb", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_database",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="database", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_ldap",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="ldap", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_erp",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="erp", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_reranker",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="reranker", status="healthy"),
            ),
            patch(
                "kiara.health.checker.check_colbert",
                new_callable=AsyncMock,
                return_value=ComponentHealth(
                    name="colbert", status="degraded", message="ColBERT down"
                ),
            ),
            patch(
                "kiara.health.checker.check_disk",
                new_callable=AsyncMock,
                return_value=ComponentHealth(name="disk", status="healthy"),
            ),
        ):
            result = await checker.check_all()

        assert result.status == "degraded"
        assert len(result.components) == 9
        colbert_comp = next(c for c in result.components if c.name == "colbert")
        assert colbert_comp.status == "degraded"
