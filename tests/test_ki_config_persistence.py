"""KI-Konfiguration mit echtem PostgreSQL-Schema und temporaerer YAML speichern."""

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.admin.functions.ki import ki_config_update
from kiara.config import load_config, save_config
from kiara.repositories.config import ConfigRepository
from kiara.shared.db import get_connection
from kiara.web import api_ki
from kiara.web.auth import get_admin_user

USER = {"id": 1, "username": "admin", "is_admin": True}


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Schema aus schema.sql nutzen, ohne persistente Config-Zeilen anzufassen."""
    schema = (Path(__file__).parents[1] / "kiara/web/schema.sql").read_text()
    columns = schema.split("CREATE TABLE public.ai_config (", 1)[1].split("\n);", 1)[0]
    conn = get_connection()
    try:
        conn.execute(f"CREATE TEMP TABLE ai_config ({columns}, PRIMARY KEY (id))")
        original_get = ConfigRepository.get
        original_set = ConfigRepository.set
        monkeypatch.setattr(
            ConfigRepository,
            "get",
            lambda self, table: original_get(self, table, conn=conn),
        )
        monkeypatch.setattr(
            ConfigRepository,
            "set",
            lambda self, table, fields: original_set(self, table, fields, conn=conn),
        )
        path = tmp_path / "kiara.yaml"
        path.write_text("{}")
        monkeypatch.setattr(api_ki, "load_config", lambda: load_config(str(path)))
        monkeypatch.setattr(api_ki, "save_config", lambda cfg: save_config(cfg, str(path)))
        monkeypatch.setattr(api_ki, "log_config_change", lambda *args, **kwargs: None)
        yield path
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize("entrypoint", ["http", "admin_chat"])
def test_anonymization_toggle_persists_in_postgres_and_yaml(isolated_config, entrypoint):
    app = FastAPI()
    app.include_router(api_ki.router)
    app.dependency_overrides[get_admin_user] = lambda: USER
    repo = ConfigRepository()

    # Bestehende Konfiguration wie beim Preread der Compliance-Oberflaeche.
    repo.set("ai_config", {"anon_enabled": True, "anon_min_term_length": 5})
    with TestClient(app) as client:
        for enabled in (False, True):
            changes = {
                "cloud_enabled": enabled,
                "anon_enabled": enabled,
                "anon_erp_enabled": enabled,
                "anon_ldap_enabled": enabled,
                "anon_llm_enabled": enabled,
                "anon_word_boundaries": enabled,
                "anon_audit_log": enabled,
            }
            if entrypoint == "http":
                payload = asyncio.run(api_ki.get_ki_config(user=USER))["config"]
                response = client.post("/api/admin/ki/config", json={**payload, **changes})
                assert response.status_code == 200, response.text
            else:
                result = asyncio.run(ki_config_update(user=USER, **changes))
                assert result.success, result.message

            row = repo.get("ai_config")
            for key in (
                "cloud_enabled",
                "anon_enabled",
                "anon_erp_enabled",
                "anon_ldap_enabled",
                "anon_llm_enabled",
            ):
                assert row[key] is enabled
            for key in ("anon_word_boundaries", "anon_audit_log"):
                assert type(row[key]) is int
                assert row[key] == int(enabled)
            assert row["anon_min_term_length"] == 5
            config = load_config(str(isolated_config))
            assert config["cloud"]["enabled"] is enabled
            anon = config["anonymization"]
            assert anon["enabled"] is enabled
            assert anon["erp_source"]["enabled"] is enabled
            assert anon["ldap_source"]["enabled"] is enabled
            assert anon["llm_detection"]["enabled"] is enabled
            assert anon["word_boundaries"] is enabled
            assert anon["audit_log"] is enabled
            assert anon["min_term_length"] == 5
