"""Tests fuer bestehende Module: config.py und db_utils.py.

Testet Config-Laden/Speichern/Validierung, Default-Werte, Deep-Merge,
sowie validate_identifier, insert_row und update_row.
"""

from pathlib import Path

import pytest
import yaml

from kiara.config import (
    DEFAULT_CONFIG,
    _deep_merge,
    load_config,
    save_config,
)
from kiara.shared.db import (
    ALLOWED_COLUMNS,
    ALLOWED_TABLES,
    insert_row,
    update_row,
    validate_identifier,
)

# ---------------------------------------------------------------------------
# Config: load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_config_from_file(self, tmp_path):
        """Laedt Config aus existierender YAML-Datei."""
        config_file = tmp_path / "kiara.yaml"
        config_data = {
            "datasource": {"path": "/custom/path"},
            "ollama": {
                "url": "http://custom:11434",
                "llm_model": "custom-model",
                "embedding_model": "custom-embed",
            },
            "vectordb": {
                "path": "/custom/vectordb",
                "collection": "custom_kb",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        result = load_config(str(config_file))

        assert result["datasource"]["path"] == "/custom/path"
        assert result["ollama"]["url"] == "http://custom:11434"
        assert result["ollama"]["llm_model"] == "custom-model"
        assert result["vectordb"]["collection"] == "custom_kb"

    def test_load_config_merges_with_defaults(self, tmp_path):
        """User-Config wird mit Defaults gemerged, nicht-spezifizierte Werte bleiben."""
        config_file = tmp_path / "kiara.yaml"
        # Nur datasource.path ueberschreiben
        config_data = {
            "datasource": {"path": "/new/path"},
        }
        config_file.write_text(yaml.dump(config_data))

        result = load_config(str(config_file))

        # Ueberschrieben
        assert result["datasource"]["path"] == "/new/path"
        # Defaults beibehalten
        assert result["ingestion"]["chunk_size"] == DEFAULT_CONFIG["ingestion"]["chunk_size"]
        assert result["web"]["port"] == DEFAULT_CONFIG["web"]["port"]
        assert result["ollama"]["llm_model"] == DEFAULT_CONFIG["ollama"]["llm_model"]

    def test_load_config_file_not_found_raises(self, tmp_path):
        """Expliziter Pfad zu nicht-existenter Datei wirft FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_load_config_default_path_missing_returns_defaults(self, monkeypatch):
        """Wenn Default-Pfad nicht existiert, werden Defaults zurueckgegeben."""
        monkeypatch.delenv("KIARA_CONFIG", raising=False)
        # Verwende einen nicht-existierenden Default-Pfad
        monkeypatch.setattr("kiara.config.DEFAULT_CONFIG_PATH", "/tmp/nonexistent_kiara_test.yaml")

        result = load_config()

        assert result["datasource"]["path"] == DEFAULT_CONFIG["datasource"]["path"]
        assert result["ollama"]["url"] == DEFAULT_CONFIG["ollama"]["url"]

    def test_load_config_env_var(self, tmp_path, monkeypatch):
        """Config wird aus KIARA_CONFIG Env-Variable geladen."""
        config_file = tmp_path / "env_config.yaml"
        config_data = {
            "datasource": {"path": "/env/path"},
            "ollama": {
                "url": "http://env:11434",
                "llm_model": "env-model",
                "embedding_model": "env-embed",
            },
            "vectordb": {
                "path": "/env/vectordb",
                "collection": "env_kb",
            },
        }
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.setenv("KIARA_CONFIG", str(config_file))

        result = load_config()

        assert result["datasource"]["path"] == "/env/path"

    def test_load_config_empty_yaml(self, tmp_path):
        """Leere YAML-Datei gibt Defaults zurueck."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = load_config(str(config_file))

        assert result == DEFAULT_CONFIG

    def test_load_config_validates(self, tmp_path):
        """Config mit fehlenden Pflichtfeldern wirft einen Fehler.

        _deep_merge ersetzt den gesamten 'ollama'-Dict durch None, wenn
        der User 'ollama: null' setzt. _validate() scheitert dann,
        da es versucht 'url' in None nachzuschlagen.
        """
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("ollama: null\n")

        with pytest.raises((ValueError, TypeError)):
            load_config(str(config_file))


# ---------------------------------------------------------------------------
# Config: save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_save_config_writes_yaml(self, tmp_path):
        """save_config schreibt eine gueltige YAML-Datei."""
        config_file = tmp_path / "output.yaml"
        config = DEFAULT_CONFIG.copy()

        save_config(config, str(config_file))

        assert config_file.exists()
        loaded = yaml.safe_load(config_file.read_text())
        assert loaded["datasource"]["path"] == config["datasource"]["path"]

    def test_save_config_creates_directories(self, tmp_path):
        """save_config erstellt fehlende Verzeichnisse."""
        config_file = tmp_path / "sub" / "dir" / "config.yaml"

        save_config(DEFAULT_CONFIG.copy(), str(config_file))

        assert config_file.exists()

    def test_save_config_validates(self, tmp_path):
        """save_config validiert die Config vor dem Schreiben."""
        config_file = tmp_path / "invalid.yaml"
        bad_config = {"datasource": {}}  # Fehlende Felder

        with pytest.raises(ValueError):
            save_config(bad_config, str(config_file))

    def test_save_and_reload_roundtrip(self, tmp_path):
        """Config speichern und erneut laden ergibt dieselben Werte."""
        config_file = tmp_path / "roundtrip.yaml"
        original = DEFAULT_CONFIG.copy()
        original["datasource"]["path"] = "/roundtrip/test"

        save_config(original, str(config_file))
        reloaded = load_config(str(config_file))

        assert reloaded["datasource"]["path"] == "/roundtrip/test"
        assert reloaded["ollama"]["url"] == original["ollama"]["url"]
        assert reloaded["vectordb"]["collection"] == original["vectordb"]["collection"]


# ---------------------------------------------------------------------------
# Config: _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_override(self):
        """Einfache Werte werden ueberschrieben."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        """Verschachtelte Dicts werden rekursiv gemerged."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_new_keys_added(self):
        """Neue Keys aus override werden hinzugefuegt."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_deep_nested_merge(self):
        """Drei Ebenen tief."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 99}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 99}}}

    def test_override_dict_with_scalar(self):
        """Dict wird durch Skalar ersetzt wenn override kein Dict."""
        base = {"a": {"x": 1}}
        override = {"a": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"a": "replaced"}

    def test_override_scalar_with_dict(self):
        """Skalar wird durch Dict ersetzt."""
        base = {"a": "original"}
        override = {"a": {"x": 1}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1}}

    def test_base_not_modified(self):
        """Original-Dict wird nicht veraendert."""
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 99}}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": {"c": 2}}

    def test_empty_override(self):
        """Leeres Override aendert nichts."""
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {})
        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        """Leeres Base ergibt Override."""
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_list_values_replaced(self):
        """Listen werden ersetzt, nicht gemerged."""
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"a": [4, 5]}


# ---------------------------------------------------------------------------
# Config: Default-Werte
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_config_has_required_sections(self):
        """DEFAULT_CONFIG enthaelt alle erforderlichen Sektionen."""
        required_sections = [
            "datasource",
            "ollama",
            "vectordb",
            "ingestion",
            "web",
            "database",
            "cloud",
            "anonymization",
            "security",
        ]
        for section in required_sections:
            assert section in DEFAULT_CONFIG, f"Sektion '{section}' fehlt"

    def test_default_config_passes_validation(self):
        """DEFAULT_CONFIG besteht die Validierung."""
        from kiara.config import _validate

        _validate(DEFAULT_CONFIG)  # Darf keine Exception werfen

    def test_default_cloud_disabled(self):
        """Cloud ist standardmaessig deaktiviert."""
        assert DEFAULT_CONFIG["cloud"]["enabled"] is False

    def test_default_anonymization_enabled(self):
        """Anonymisierung ist standardmaessig aktiviert."""
        assert DEFAULT_CONFIG["anonymization"]["enabled"] is True

    def test_default_web_port(self):
        """Standard Web-Port ist 8501."""
        assert DEFAULT_CONFIG["web"]["port"] == 8501

    def test_default_chunk_size(self):
        """Standard Chunk-Groesse."""
        assert DEFAULT_CONFIG["ingestion"]["chunk_size"] == 1000
        assert DEFAULT_CONFIG["ingestion"]["chunk_overlap"] == 200

    def test_default_config_enrichment_late_chunking(self):
        """enrichment.late_chunking ist standardmaessig True und enrichment ist eine Pflichtsektion."""
        assert "enrichment" in DEFAULT_CONFIG, "Sektion 'enrichment' fehlt in DEFAULT_CONFIG"
        assert DEFAULT_CONFIG["enrichment"]["late_chunking"] is True

    def test_config_yaml_parity_enrichment(self):
        """Beide YAML-Configs enthalten late_chunking: true unter enrichment."""
        # debian/kiara/etc/kiara/kiara.yaml ist ein Build-Artefakt von dpkg-buildpackage;
        # auf nicht-gebauten Systemen (z.B. Deploy via rsync) nur die Source-Config pruefen.
        yaml_paths = [
            Path(__file__).resolve().parent.parent / "config" / "kiara.yaml",
            Path(__file__).resolve().parent.parent
            / "debian"
            / "kiara"
            / "etc"
            / "kiara"
            / "kiara.yaml",
        ]
        existing = [p for p in yaml_paths if p.exists()]
        assert existing, f"Keine YAML gefunden in: {yaml_paths}"
        for yaml_path in existing:
            data = yaml.safe_load(yaml_path.read_text())
            assert "enrichment" in data, f"enrichment fehlt in {yaml_path}"
            assert data["enrichment"].get("late_chunking") is True, (
                f"late_chunking != true in {yaml_path}"
            )

    def test_default_security_flags(self):
        """Security-Defaults bleiben konservativ und rollback-faehig."""
        security = DEFAULT_CONFIG["security"]
        assert security["cookie_secure"] is False
        assert security["hsts_max_age"] == 0
        # V30 Welle 5 (#1049): Default-Flip — Strict-CSP im Enforce-Modus.
        assert security["csp_strict"] is True
        assert security["csp_report_only"] is True
        assert security["csp_report_uri"] == "/api/admin/csp-report"

    def test_config_yaml_parity_security(self):
        """Beide YAML-Configs enthalten den sichtbaren security-Block."""
        yaml_paths = [
            Path(__file__).resolve().parent.parent / "config" / "kiara.yaml",
            Path(__file__).resolve().parent.parent
            / "debian"
            / "kiara"
            / "etc"
            / "kiara"
            / "kiara.yaml",
        ]
        existing = [p for p in yaml_paths if p.exists()]
        assert existing, f"Keine YAML gefunden in: {yaml_paths}"
        for yaml_path in existing:
            data = yaml.safe_load(yaml_path.read_text())
            assert "security" in data, f"security fehlt in {yaml_path}"
            # V30 Welle 5: Strict-CSP ist Default in beiden YAML-Configs.
            assert data["security"].get("csp_strict") is True
            assert data["security"].get("csp_report_only") is True
            assert data["security"].get("csp_report_uri") == "/api/admin/csp-report"

    def test_config_yaml_parity_oidc_retrieval(self):
        """Beide YAML-Configs enthalten den sichtbaren OIDC-Retrieval-Block."""
        yaml_paths = [
            Path(__file__).resolve().parent.parent / "config" / "kiara.yaml",
            Path(__file__).resolve().parent.parent
            / "debian"
            / "kiara"
            / "etc"
            / "kiara"
            / "kiara.yaml",
        ]
        existing = [p for p in yaml_paths if p.exists()]
        assert existing, f"Keine YAML gefunden in: {yaml_paths}"
        for yaml_path in existing:
            data = yaml.safe_load(yaml_path.read_text())
            oidc = data.get("web", {}).get("oidc_retrieval")
            assert isinstance(oidc, dict), f"web.oidc_retrieval fehlt in {yaml_path}"
            assert oidc["enabled"] is False
            assert oidc["issuer"] == "https://id.opendesk.hirschgasse5.de/realms/opendesk"
            assert oidc["audience"] == "kiara-retrieval"
            assert oidc["client_binding_claim"] == "azp"
            assert oidc["client_binding_value"] == "open-webui"
            assert oidc["identity_claim"] == "ad_sid"


# ---------------------------------------------------------------------------
# db_utils: validate_identifier
# ---------------------------------------------------------------------------


class TestValidateIdentifier:
    def test_valid_table_name(self):
        """Erlaubter Tabellenname wird zurueckgegeben."""
        result = validate_identifier("datasources", ALLOWED_TABLES, "table")
        assert result == "datasources"

    def test_valid_column_name(self):
        """Erlaubter Spaltenname wird zurueckgegeben."""
        result = validate_identifier("name", ALLOWED_COLUMNS, "column")
        assert result == "name"

    def test_invalid_table_name_raises(self):
        """Nicht erlaubter Tabellenname wirft ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL table"):
            validate_identifier("evil_table", ALLOWED_TABLES, "table")

    def test_invalid_column_name_raises(self):
        """Nicht erlaubter Spaltenname wirft ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL column"):
            validate_identifier("DROP TABLE users", ALLOWED_COLUMNS, "column")

    def test_sql_injection_blocked(self):
        """SQL-Injection-Versuche werden blockiert."""
        dangerous_inputs = [
            "users; DROP TABLE users",
            "1 OR 1=1",
            "' OR ''='",
            "cloud_audit_log; --",
        ]
        for dangerous in dangerous_inputs:
            with pytest.raises(ValueError):
                validate_identifier(dangerous, ALLOWED_TABLES, "table")

    def test_all_allowed_tables_pass(self):
        """Alle erlaubten Tabellen bestehen die Validierung."""
        for table in ALLOWED_TABLES:
            assert validate_identifier(table, ALLOWED_TABLES, "table") == table

    def test_all_allowed_columns_pass(self):
        """Alle erlaubten Spalten bestehen die Validierung."""
        for col in ALLOWED_COLUMNS:
            assert validate_identifier(col, ALLOWED_COLUMNS, "column") == col

    def test_empty_string_rejected(self):
        """Leerer String wird abgelehnt."""
        with pytest.raises(ValueError):
            validate_identifier("", ALLOWED_TABLES, "table")

    def test_case_sensitive(self):
        """Validierung ist case-sensitive."""
        with pytest.raises(ValueError):
            validate_identifier("DATASOURCES", ALLOWED_TABLES, "table")

    def test_cloud_audit_log_allowed(self):
        """cloud_audit_log ist in ALLOWED_TABLES."""
        assert "cloud_audit_log" in ALLOWED_TABLES
        assert validate_identifier("cloud_audit_log", ALLOWED_TABLES, "table") == "cloud_audit_log"


# ---------------------------------------------------------------------------
# db_utils: insert_row / update_row (in-memory DB)
# ---------------------------------------------------------------------------


@pytest.fixture
def inmemory_db():
    """PostgreSQL-Verbindung mit Test-Tabellen fuer insert_row/update_row."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    # DROP und Neuanlage damit das Test-Schema garantiert stimmt
    conn.executescript("""
        DROP TABLE IF EXISTS cloud_audit_log CASCADE;
        DROP TABLE IF EXISTS datasources CASCADE;
        DROP TABLE IF EXISTS groups CASCADE;
        DROP TABLE IF EXISTS ai_config CASCADE;
        DROP TABLE IF EXISTS users CASCADE;

        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            vorname TEXT DEFAULT '',
            nachname TEXT DEFAULT '',
            email TEXT DEFAULT '',
            is_admin BOOLEAN DEFAULT false,
            is_active BOOLEAN DEFAULT true,
            auth_source TEXT DEFAULT 'local',
            must_change_password BOOLEAN DEFAULT false,
            created_at TEXT DEFAULT (NOW())
        );

        CREATE TABLE cloud_audit_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            question_anonymized TEXT NOT NULL,
            answer_anonymized TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            entities_anonymized INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            context_sources TEXT,
            created_at TEXT DEFAULT (NOW()),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE datasources (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'local',
            path TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            restriction_level TEXT DEFAULT 'anonymize',
            ds_contextual_retrieval INTEGER DEFAULT 0,
            ds_enrichment_prompt_id INTEGER DEFAULT NULL,
            ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false
        );

        CREATE TABLE groups (
            id SERIAL PRIMARY KEY,
            cn TEXT NOT NULL,
            dn TEXT NOT NULL UNIQUE,
            member_count INTEGER DEFAULT 0
        );

        CREATE TABLE ai_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            embedding_url TEXT NOT NULL DEFAULT 'http://localhost:11434',
            embedding_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
            llm_url TEXT NOT NULL DEFAULT 'http://gpu-server.local:11434',
            llm_model TEXT NOT NULL DEFAULT 'qwen3:8b',
            prompt_template TEXT DEFAULT ''
        );
    """)
    # Testdaten einfuegen
    conn.execute("DELETE FROM cloud_audit_log")
    conn.execute("DELETE FROM datasources")
    conn.execute("DELETE FROM groups")
    conn.execute("DELETE FROM users")
    conn.execute(
        "INSERT INTO users (id, username) VALUES (1, 'testuser') ON CONFLICT (id) DO NOTHING"
    )
    conn.commit()
    yield conn
    conn.rollback()
    conn.close()

    # Tabellen wiederherstellen (DROP+CREATE hat Produktions-Schema zerstoert)
    restore = get_connection()
    try:
        # Minimal-Tabellen droppen, damit create_tables() sie korrekt neu anlegt
        restore.execute("DROP TABLE IF EXISTS cloud_audit_log CASCADE")
        restore.execute("DROP TABLE IF EXISTS datasources CASCADE")
        restore.execute("DROP TABLE IF EXISTS groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS ai_config CASCADE")
        restore.execute("DROP TABLE IF EXISTS users CASCADE")
        from kiara.web.schema import create_tables

        create_tables(restore)
        restore.commit()
    except Exception:
        restore.rollback()
    restore.close()


class TestInsertRow:
    def test_insert_returns_row_id(self, inmemory_db):
        """insert_row gibt die neue Row-ID zurueck."""
        row_id = insert_row(
            inmemory_db,
            "datasources",
            {
                "name": "Test DS",
                "type": "local",
                "path": "/test/path",
            },
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_insert_persists_data(self, inmemory_db):
        """Eingefuegte Daten sind in der DB."""
        row_id = insert_row(
            inmemory_db,
            "datasources",
            {
                "name": "Produktiv-KB",
                "type": "local",
                "path": "/mnt/kb",
                "is_active": True,
            },
        )
        inmemory_db.commit()

        row = inmemory_db.execute("SELECT * FROM datasources WHERE id = %s", (row_id,)).fetchone()
        assert row["name"] == "Produktiv-KB"
        assert row["type"] == "local"
        assert row["path"] == "/mnt/kb"

    def test_insert_invalid_table_raises(self, inmemory_db):
        """insert_row mit unerlaubtem Tabellennamen wirft ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL table"):
            insert_row(inmemory_db, "evil_table", {"name": "test"})

    def test_insert_invalid_column_raises(self, inmemory_db):
        """insert_row mit unerlaubtem Spaltennamen wirft ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL column"):
            insert_row(inmemory_db, "datasources", {"evil_col": "test"})

    def test_insert_multiple_rows(self, inmemory_db):
        """Mehrere Zeilen einfuegen."""
        ids = []
        for i in range(3):
            row_id = insert_row(
                inmemory_db,
                "datasources",
                {
                    "name": f"DS-{i}",
                    "type": "local",
                },
            )
            ids.append(row_id)
        inmemory_db.commit()

        assert len(set(ids)) == 3
        count = inmemory_db.execute("SELECT COUNT(*) AS cnt FROM datasources").fetchone()["cnt"]
        assert count == 3

    def test_insert_into_cloud_audit_log(self, inmemory_db):
        """Insert in cloud_audit_log funktioniert."""
        row_id = insert_row(
            inmemory_db,
            "cloud_audit_log",
            {
                "user_id": 1,
                "question_anonymized": "Test-Frage [KUNDE_001]",
                "answer_anonymized": "Test-Antwort",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "tokens_in": 100,
                "tokens_out": 50,
            },
        )
        inmemory_db.commit()

        row = inmemory_db.execute(
            "SELECT * FROM cloud_audit_log WHERE id = %s", (row_id,)
        ).fetchone()
        assert row["provider"] == "openai"
        assert row["tokens_in"] == 100

    def test_insert_into_groups(self, inmemory_db):
        """Insert in groups-Tabelle."""
        row_id = insert_row(
            inmemory_db,
            "groups",
            {
                "cn": "IT-Abteilung",
                "dn": "cn=IT,ou=Groups,dc=example,dc=com",
                "member_count": 15,
            },
        )
        inmemory_db.commit()

        row = inmemory_db.execute("SELECT * FROM groups WHERE id = %s", (row_id,)).fetchone()
        assert row["cn"] == "IT-Abteilung"
        assert row["member_count"] == 15


class TestUpdateRow:
    def test_update_modifies_data(self, inmemory_db):
        """update_row aendert die Daten korrekt."""
        row_id = insert_row(
            inmemory_db,
            "datasources",
            {
                "name": "Original",
                "type": "local",
                "path": "/old/path",
            },
        )
        inmemory_db.commit()

        update_row(
            inmemory_db,
            "datasources",
            row_id,
            {
                "name": "Updated",
                "path": "/new/path",
            },
        )
        inmemory_db.commit()

        row = inmemory_db.execute("SELECT * FROM datasources WHERE id = %s", (row_id,)).fetchone()
        assert row["name"] == "Updated"
        assert row["path"] == "/new/path"
        assert row["type"] == "local"  # Unveraendert

    def test_update_invalid_table_raises(self, inmemory_db):
        """update_row mit unerlaubtem Tabellennamen wirft ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL table"):
            update_row(inmemory_db, "evil_table", 1, {"name": "test"})

    def test_update_invalid_column_raises(self, inmemory_db):
        """update_row mit unerlaubtem Spaltennamen wirft ValueError."""
        with pytest.raises(ValueError, match="Invalid SQL column"):
            update_row(inmemory_db, "datasources", 1, {"evil_col": "test"})

    def test_update_nonexistent_row_no_error(self, inmemory_db):
        """Update einer nicht-existierenden Row verursacht keinen Fehler."""
        # DB fuehrt das UPDATE aus, trifft aber keine Zeile
        update_row(
            inmemory_db,
            "datasources",
            9999,
            {
                "name": "Ghost",
            },
        )
        inmemory_db.commit()

        row = inmemory_db.execute("SELECT * FROM datasources WHERE id = 9999").fetchone()
        assert row is None

    def test_update_single_field(self, inmemory_db):
        """Update eines einzelnen Feldes."""
        row_id = insert_row(
            inmemory_db,
            "datasources",
            {
                "name": "DS",
                "type": "local",
                "is_active": True,
            },
        )
        inmemory_db.commit()

        update_row(
            inmemory_db,
            "datasources",
            row_id,
            {
                "is_active": False,
            },
        )
        inmemory_db.commit()

        row = inmemory_db.execute("SELECT * FROM datasources WHERE id = %s", (row_id,)).fetchone()
        assert not row["is_active"]
        assert row["name"] == "DS"  # Unveraendert


class TestTracingNestedConfigMerge:
    def test_deep_merge_preserves_nested_tracing_defaults(self):
        from kiara.config import DEFAULT_CONFIG, _deep_merge

        merged = _deep_merge(
            DEFAULT_CONFIG,
            {
                "tracing": {
                    "pipelines": {"retrieval": False},
                    "otlp": {"endpoint": "http://collector:4317"},
                },
            },
        )

        assert merged["tracing"]["pipelines"] == {
            "retrieval": False,
            "agent": True,
            "ingestion": True,
        }
        assert merged["tracing"]["otlp"] == {
            "endpoint": "http://collector:4317",
            "insecure": True,
        }
        assert merged["tracing"]["chat_debug"] is False

    def test_load_config_merges_nested_tracing_overrides(self, tmp_path):
        from kiara.config import load_config

        cfg = tmp_path / "kiara.yaml"
        cfg.write_text(
            "tracing:\n"
            "  pipelines:\n"
            "    retrieval: false\n"
            "  otlp:\n"
            "    endpoint: http://collector:4317\n",
            encoding="utf-8",
        )

        merged = load_config(str(cfg))
        assert merged["tracing"]["pipelines"] == {
            "retrieval": False,
            "agent": True,
            "ingestion": True,
        }
        assert merged["tracing"]["otlp"] == {
            "endpoint": "http://collector:4317",
            "insecure": True,
        }
        assert merged["tracing"]["chat_debug"] is False
