"""Tests fuer Sprint 09: Compliance Admin-Funktionen.

Testet compliance.audit, .audit_detail, .permissions_matrix,
.anonymization_stats, .governance, .vvt, .dsfa, .dashboard — 14 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# compliance.audit
# ---------------------------------------------------------------------------


class TestComplianceAudit:
    @patch("kiara.repositories.audit.QueryLogRepository.list_with_joins")
    def test_audit_paginated(self, mock_list):
        """Test 1: Paginiertes Audit-Log."""
        mock_list.return_value = {
            "data": [{"id": 1, "question": "test"}, {"id": 2, "question": "test2"}],
            "pagination": {"page": 1, "page_size": 50, "total_items": 2, "total_pages": 1},
        }

        from kiara.admin.functions.compliance import compliance_audit

        result = _run(compliance_audit(MOCK_USER, page=1, page_size=50))

        assert result.success is True
        assert len(result.data["data"]) == 2
        assert "pagination" in result.data

    @patch("kiara.repositories.audit.QueryLogRepository.list_with_joins")
    def test_audit_error_fallback(self, mock_list):
        """Test 2: DB-Fehler → leere Liste statt Exception."""
        mock_list.side_effect = Exception("DB down")

        from kiara.admin.functions.compliance import compliance_audit

        result = _run(compliance_audit(MOCK_USER))

        assert result.success is True
        assert result.data["data"] == []
        assert result.data["pagination"]["total_items"] == 0

    @patch("kiara.repositories.audit.QueryLogRepository.list_with_joins")
    def test_audit_filters(self, mock_list):
        """Test 3: Filter-Parameter werden durchgereicht."""
        mock_list.return_value = {"data": [], "pagination": {}}

        from kiara.admin.functions.compliance import compliance_audit

        _run(
            compliance_audit(
                MOCK_USER,
                user_id=5,
                date_from="2026-03-01",
                routed_to="cloud",
                search_text="test",
            )
        )

        mock_list.assert_called_once_with(
            user_id=5,
            date_from="2026-03-01",
            date_to=None,
            routed_to="cloud",
            chain_type=None,
            search_text="test",
            sort_field="created_at",
            sort_dir="desc",
            page=1,
            page_size=50,
        )


# ---------------------------------------------------------------------------
# compliance.audit_detail
# ---------------------------------------------------------------------------


class TestComplianceAuditDetail:
    @patch("kiara.repositories.audit.CloudAuditLogRepository.find_matching_entry")
    @patch("kiara.repositories.audit.QueryLogRepository.get_detail_with_joins")
    def test_audit_detail_found(self, mock_detail, mock_cloud):
        """Test 4: Audit-Detail mit geparsten JSON-Feldern und correlation_id."""
        mock_detail.return_value = {
            "id": 42,
            "user_id": 1,
            "question": "Test?",
            "tool_calls": '[{"name": "search"}]',
            "context_sources": '["doc1.pdf"]',
            "created_at": "2026-03-28 12:00:00",
            "correlation_id": "test-corr-42",
        }
        mock_cloud.return_value = {"provider": "openai", "tokens": 100}

        from kiara.admin.functions.compliance import compliance_audit_detail

        result = _run(compliance_audit_detail(MOCK_USER, audit_id=42))

        assert result.success is True
        assert result.data["tool_calls"] == [{"name": "search"}]
        assert result.data["context_sources"] == ["doc1.pdf"]
        assert result.data["cloud_audit"]["provider"] == "openai"
        mock_cloud.assert_called_once_with("test-corr-42")

    @patch("kiara.repositories.audit.QueryLogRepository.get_detail_with_joins")
    def test_audit_detail_not_found(self, mock_detail):
        """Test 5: Nicht existenter Eintrag → fail."""
        mock_detail.return_value = None

        from kiara.admin.functions.compliance import compliance_audit_detail

        result = _run(compliance_audit_detail(MOCK_USER, audit_id=9999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.repositories.audit.QueryLogRepository.get_detail_with_joins")
    def test_audit_detail_no_correlation_id(self, mock_detail):
        """Test 5b: Eintrag ohne correlation_id → kein Cloud-Match."""
        mock_detail.return_value = {
            "id": 43,
            "user_id": 1,
            "question": "Alt?",
            "tool_calls": None,
            "context_sources": None,
            "created_at": "2026-03-28 12:00:00",
        }

        from kiara.admin.functions.compliance import compliance_audit_detail

        result = _run(compliance_audit_detail(MOCK_USER, audit_id=43))

        assert result.success is True
        assert "cloud_audit" not in result.data


# ---------------------------------------------------------------------------
# compliance.permissions_matrix
# ---------------------------------------------------------------------------


class TestCompliancePermissionsMatrix:
    @patch("kiara.repositories.permission.PermissionRepository.get_group_datasources")
    @patch("kiara.repositories.group.GroupRepository.get_members")
    @patch("kiara.repositories.group.GroupRepository.list_all")
    @patch("kiara.config.load_config")
    def test_permissions_matrix(self, mock_config, mock_list, mock_members, mock_ds):
        """Test 6: Gruppen mit Datenquellen und Mitgliedern."""
        mock_config.return_value = {"cloud": {"enabled": True}}
        mock_list.return_value = [
            {"id": 1, "cn": "Admins", "dn": "cn=Admins,ou=groups", "member_count": 2},
        ]
        mock_members.return_value = [
            {"id": 1, "username": "admin", "vorname": "Max", "nachname": "Admin"},
        ]
        mock_ds.return_value = [
            {
                "id": 10,
                "name": "Docs",
                "type": "local",
                "is_active": True,
                "restriction_level": "anonymize",
            },
        ]

        from kiara.admin.functions.compliance import compliance_permissions_matrix

        result = _run(compliance_permissions_matrix(MOCK_USER))

        assert result.success is True
        groups = result.data["groups"]
        assert len(groups) == 1
        assert groups[0]["name"] == "Admins"
        assert len(groups[0]["members"]) == 1
        assert len(groups[0]["datasources"]) == 1
        assert "cloud_ki" in groups[0]["tools"]


# ---------------------------------------------------------------------------
# compliance.anonymization_stats
# ---------------------------------------------------------------------------


class TestComplianceAnonymizationStats:
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.by_route")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.daily_since")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.stats_since")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.last_timestamp")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.all_entity_types")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.total_entities")
    def test_anonymization_stats(
        self, mock_total, mock_types, mock_last, mock_stats, mock_daily, mock_route
    ):
        """Test 7: Vollstaendige Anonymisierungsstatistiken."""
        mock_total.return_value = 5000
        mock_types.return_value = [
            {"entity_types": '{"Kunden": 100, "IBAN": 50}'},
            {"entity_types": '{"Kunden": 200}'},
        ]
        mock_last.return_value = "2026-03-28 10:00:00"
        mock_stats.return_value = {"total_30d": 300, "requests_30d": 150}
        mock_daily.return_value = [{"date": "2026-03-28", "count": 10}]
        mock_route.return_value = [{"route": "/api/chat", "count": 100}]

        from kiara.admin.functions.compliance import compliance_anonymization_stats

        result = _run(compliance_anonymization_stats(MOCK_USER))

        assert result.success is True
        assert result.data["total_entities"] == 5000
        assert result.data["by_category"]["Kunden"] == 300
        assert result.data["by_category"]["IBAN"] == 50
        assert result.data["total_30d"] == 300

    @patch("kiara.repositories.audit.AuditAnonymizationRepository.total_entities")
    def test_anonymization_stats_error(self, mock_total):
        """Test 8: DB-Fehler → Fallback-Daten."""
        mock_total.side_effect = Exception("DB error")

        from kiara.admin.functions.compliance import compliance_anonymization_stats

        result = _run(compliance_anonymization_stats(MOCK_USER))

        assert result.success is True
        assert result.data["total_entities"] == 0
        assert result.data["by_category"] == {}


# ---------------------------------------------------------------------------
# compliance.governance
# ---------------------------------------------------------------------------


class TestComplianceGovernance:
    @patch("kiara.shared.db.db_query")
    @patch("kiara.repositories.datasource.DatasourceEmbeddingRepository.list_filtered")
    @patch("kiara.repositories.audit.QueryLogRepository.count_since")
    @patch("kiara.repositories.permission.PermissionRepository.get_datasource_groups")
    @patch("kiara.repositories.datasource.DatasourceRepository.list_all")
    def test_governance(self, mock_ds, mock_groups, mock_count, mock_emb, mock_db):
        """Test 9: Datenquellen-Governance mit Zugriffsgruppen."""
        mock_ds.return_value = [
            {
                "id": 1,
                "name": "Dokumente",
                "type": "local",
                "is_active": True,
                "path": "/data/docs",
                "restriction_level": "anonymize",
            }
        ]
        mock_groups.return_value = [{"id": 1, "cn": "Alle"}]
        mock_count.return_value = 42
        mock_emb.return_value = ([], 0)
        # Mock fuer Embedding-Aggregate Query
        mock_cursor = MagicMock()
        mock_cursor.execute.return_value.fetchall.return_value = [
            {
                "datasource_id": 1,
                "chunk_count": 500,
                "file_count": 100,
                "last_ingested_at": "2026-03-27",
            },
        ]
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.admin.functions.compliance import compliance_governance

        result = _run(compliance_governance(MOCK_USER))

        assert result.success is True
        ds = result.data["datasources"]
        assert len(ds) == 1
        assert ds[0]["name"] == "Dokumente"
        assert ds[0]["access_groups"] == [{"id": 1, "name": "Alle"}]


# ---------------------------------------------------------------------------
# compliance.vvt + compliance.dsfa
# ---------------------------------------------------------------------------


class TestComplianceVvtDsfa:
    @patch("kiara.admin.functions.compliance._load_cloud_backends_for_report", return_value=[])
    @patch("kiara.compliance.ComplianceReportGenerator.generate_vvt_entry")
    @patch("kiara.config.load_config")
    def test_vvt(self, mock_config, mock_vvt, mock_backends):
        """Test 10: VVT-Eintrag."""
        mock_config.return_value = {}
        mock_vvt.return_value = {
            "name": "KIara",
            "zweck": "RAG",
            "rechtsgrundlage": "Art. 6",
            "betroffene": "Mitarbeiter",
            "datenkategorien": "Dokumente",
            "empfaenger": "Intern",
            "loeschfristen": "Nach Bedarf",
            "toms": "AES-256",
            "anonymisierung": "Aho-Corasick",
            "erstellt_am": "2026-03-28",
        }

        from kiara.admin.functions.compliance import compliance_vvt

        result = _run(compliance_vvt(MOCK_USER))

        assert result.success is True
        assert result.data["name"] == "KIara"
        assert "zweck" in result.data
        mock_backends.assert_called_once()

    @patch("kiara.admin.functions.compliance._load_cloud_backends_for_report", return_value=[])
    @patch("kiara.compliance.ComplianceReportGenerator.generate_dsfa")
    @patch("kiara.config.load_config")
    def test_dsfa(self, mock_config, mock_dsfa, mock_backends):
        """Test 11: DSFA."""
        mock_config.return_value = {}
        mock_dsfa.return_value = {
            "erforderlich": True,
            "system": "KIara",
            "beschreibung": "RAG-System",
            "notwendigkeit_und_verhaeltnismaessigkeit": "Ja",
            "risiken": ["Datenverlust"],
            "ergebnis": "Akzeptabel",
            "erstellt_am": "2026-03-28",
        }

        from kiara.admin.functions.compliance import compliance_dsfa

        result = _run(compliance_dsfa(MOCK_USER))

        assert result.success is True
        assert result.data["erforderlich"] is True
        assert "risiken" in result.data
        mock_backends.assert_called_once()


# ---------------------------------------------------------------------------
# compliance.dashboard
# ---------------------------------------------------------------------------


class TestComplianceDashboard:
    @patch("kiara.repositories.permission.PermissionRepository.count_unassigned_datasources")
    @patch("kiara.repositories.group.GroupRepository.count")
    @patch("kiara.repositories.datasource.DatasourceRepository.count")
    @patch("kiara.repositories.audit.QueryLogRepository.count_distinct_users_since")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.last_timestamp")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.stats_since")
    @patch("kiara.repositories.audit.QueryLogRepository.count_since")
    @patch("kiara.config.load_config")
    def test_dashboard(
        self,
        mock_config,
        mock_count,
        mock_anon_stats,
        mock_last,
        mock_users,
        mock_ds_count,
        mock_group_count,
        mock_unassigned,
    ):
        """Test 12: Dashboard-KPIs."""
        mock_config.return_value = {"cloud": {"enabled": False}, "anonymization": {"enabled": True}}
        mock_count.side_effect = [100, 10]  # total, cloud
        mock_anon_stats.return_value = {"total_30d": 500}
        mock_last.return_value = "2026-03-28"
        mock_users.return_value = 5
        mock_ds_count.return_value = 15
        mock_group_count.return_value = 3
        mock_unassigned.return_value = 2

        from kiara.admin.functions.compliance import compliance_dashboard

        result = _run(compliance_dashboard(MOCK_USER))

        assert result.success is True
        assert result.data["total_queries_30d"] == 100
        assert result.data["cloud_queries_30d"] == 10
        assert result.data["cloud_percentage"] == 10.0
        assert result.data["entities_anonymized_30d"] == 500
        assert result.data["datasource_count"] == 15
        assert any("2 aktive" in w for w in result.data["warnings"])

    @patch("kiara.repositories.audit.QueryLogRepository.count_since")
    @patch("kiara.config.load_config")
    def test_dashboard_error_fallback(self, mock_config, mock_count):
        """Test 13: DB-Fehler → Fallback-Daten mit Warnung."""
        mock_config.return_value = {}
        mock_count.side_effect = Exception("DB down")

        from kiara.admin.functions.compliance import compliance_dashboard

        result = _run(compliance_dashboard(MOCK_USER))

        assert result.success is True
        assert result.data["total_queries_30d"] == 0
        assert "Datenbankfehler" in result.data["warnings"][0]

    @patch("kiara.repositories.permission.PermissionRepository.count_unassigned_datasources")
    @patch("kiara.repositories.group.GroupRepository.count")
    @patch("kiara.repositories.datasource.DatasourceRepository.count")
    @patch("kiara.repositories.audit.QueryLogRepository.count_distinct_users_since")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.last_timestamp")
    @patch("kiara.repositories.audit.AuditAnonymizationRepository.stats_since")
    @patch("kiara.repositories.audit.QueryLogRepository.count_since")
    @patch("kiara.config.load_config")
    def test_dashboard_cloud_anon_warning(
        self,
        mock_config,
        mock_count,
        mock_anon_stats,
        mock_last,
        mock_users,
        mock_ds_count,
        mock_group_count,
        mock_unassigned,
    ):
        """Test 14: Warnung wenn Cloud aktiv aber Anonymisierung aus."""
        mock_config.return_value = {"cloud": {"enabled": True}, "anonymization": {"enabled": False}}
        mock_count.return_value = 0
        mock_anon_stats.return_value = {"total_30d": 0}
        mock_last.return_value = None
        mock_users.return_value = 0
        mock_ds_count.return_value = 0
        mock_group_count.return_value = 0
        mock_unassigned.return_value = 0

        from kiara.admin.functions.compliance import compliance_dashboard

        result = _run(compliance_dashboard(MOCK_USER))

        assert any("KRITISCH" in w for w in result.data["warnings"])
