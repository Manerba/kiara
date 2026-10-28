"""Tests fuer DatasourceRequest (Phase 11.1 Refactoring).

DatasourceRequest nutzt jetzt connector_config dict statt
30+ individueller Felder.
"""

from kiara.web.api_datasource import DatasourceRequest


class TestDatasourceRequestConnectorConfig:
    """connector_config dict statt Einzelfelder."""

    def test_imap_via_connector_config(self):
        req = DatasourceRequest(
            name="Mail-Server",
            type="imap",
            connector_config={
                "server": "mail.firma.de",
                "port": 993,
                "ssl": True,
                "username": "user@firma.de",
                "password": "geheim",
                "folders": "INBOX, Sent",
                "since_days": 30,
                "download_attachments": True,
                "max_attachment_mb": 10,
            },
        )
        assert req.type == "imap"
        assert req.connector_config["server"] == "mail.firma.de"
        assert req.connector_config["port"] == 993
        assert req.connector_config["ssl"] is True
        assert req.connector_config["username"] == "user@firma.de"
        assert req.connector_config["password"] == "geheim"

    def test_sql_via_connector_config(self):
        req = DatasourceRequest(
            name="ERP-DB",
            type="sql",
            connector_config={
                "dsn": "postgresql://user:pass@localhost/erp",
                "driver": "postgresql",
                "query": "SELECT * FROM protokolle",
                "text_columns": "titel, beschreibung",
                "id_column": "protokoll_id",
            },
        )
        assert req.type == "sql"
        assert req.connector_config["dsn"] == "postgresql://user:pass@localhost/erp"
        assert req.connector_config["driver"] == "postgresql"
        assert req.connector_config["query"] == "SELECT * FROM protokolle"

    def test_empty_connector_config_default(self):
        req = DatasourceRequest(name="Test", type="local")
        assert req.connector_config == {}

    def test_local_with_path_in_config(self):
        req = DatasourceRequest(
            name="Lokal",
            type="local",
            connector_config={"path": "/mnt/data"},
        )
        assert req.type == "local"
        assert req.connector_config["path"] == "/mnt/data"

    def test_smb_via_connector_config(self):
        req = DatasourceRequest(
            name="SMB",
            type="smb",
            connector_config={
                "server": "192.168.1.10",
                "share": "Daten",
                "username": "admin",
                "password": "pw",
                "domain": "WORKGROUP",
                "sub_path": "Technik",
            },
        )
        assert req.type == "smb"
        assert req.connector_config["server"] == "192.168.1.10"
        assert req.connector_config["share"] == "Daten"

    def test_webdav_via_connector_config(self):
        req = DatasourceRequest(
            name="WebDAV",
            type="webdav",
            connector_config={
                "url": "https://nc.firma.de/remote.php/dav/",
                "username": "user",
                "password": "pw",
                "base_path": "/Docs",
            },
        )
        assert req.type == "webdav"
        assert req.connector_config["url"] == "https://nc.firma.de/remote.php/dav/"


class TestDatasourceRequestPerDSFields:
    """Per-DS Felder bleiben als Einzelfelder erhalten."""

    def test_ds_fields_defaults(self):
        req = DatasourceRequest(name="Test", type="local")
        assert req.ds_chunk_strategy == "auto"
        assert req.ds_chunk_size is None
        assert req.ds_chunk_overlap is None
        assert req.ds_file_types is None
        assert req.ds_exclude_patterns is None
        assert req.ds_ddu_enabled is False
        assert req.ds_vlm_enabled is False
        assert req.sync_interval == "global"
        assert req.restriction_level == "anonymize"

    def test_ds_fields_set(self):
        req = DatasourceRequest(
            name="Test",
            type="local",
            ds_chunk_strategy="prose",
            ds_chunk_size=1000,
            ds_chunk_overlap=200,
            ds_ddu_enabled=True,
            ds_vlm_enabled=True,
            sync_interval="hourly",
            restriction_level="blocked",
        )
        assert req.ds_chunk_strategy == "prose"
        assert req.ds_chunk_size == 1000
        assert req.ds_chunk_overlap == 200
        assert req.ds_ddu_enabled is True
        assert req.ds_vlm_enabled is True
        assert req.sync_interval == "hourly"
        assert req.restriction_level == "blocked"

    def test_restriction_level_values(self):
        """restriction_level akzeptiert alle gueltigen Werte."""
        for level in ("public", "anonymize", "blocked"):
            req = DatasourceRequest(name="Test", type="local", restriction_level=level)
            assert req.restriction_level == level
