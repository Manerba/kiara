"""Tests fuer den ComplianceReportGenerator (kiara.compliance.report_generator)."""

from pathlib import Path

import pytest

from kiara.compliance.report_generator import ComplianceReportGenerator

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_cloud_disabled():
    """Config mit deaktivierter Cloud."""
    return {
        "cloud": {
            "enabled": False,
            "provider": "openai",
        },
        "anonymization": {
            "enabled": True,
        },
    }


@pytest.fixture
def config_cloud_enabled():
    """Config mit aktivierter Cloud."""
    return {
        "cloud": {
            "enabled": True,
            "provider": "openai",
        },
        "anonymization": {
            "enabled": True,
        },
    }


# ---------------------------------------------------------------------------
# Tests: VVT
# ---------------------------------------------------------------------------


class TestVVTEntry:
    def test_vvt_entry_fields(self, config_cloud_disabled):
        """VVT-Eintrag enthaelt alle Pflichtfelder."""
        gen = ComplianceReportGenerator(config_cloud_disabled)
        vvt = gen.generate_vvt_entry()

        assert "name" in vvt
        assert "zweck" in vvt
        assert "rechtsgrundlage" in vvt
        assert "betroffene" in vvt
        assert "datenkategorien" in vvt
        assert "empfaenger" in vvt
        assert "drittlandtransfer" in vvt
        assert "loeschfristen" in vvt
        assert "toms" in vvt
        assert "anonymisierung" in vvt
        assert "erstellt_am" in vvt

        # Sinnvolle Inhalte pruefen
        assert isinstance(vvt["betroffene"], list)
        assert len(vvt["betroffene"]) > 0
        assert isinstance(vvt["toms"], list)
        assert len(vvt["toms"]) > 0
        assert isinstance(vvt["datenkategorien"], list)

    def test_vvt_cloud_disabled(self, config_cloud_disabled):
        """Keine Cloud-Empfaenger wenn Cloud deaktiviert."""
        gen = ComplianceReportGenerator(config_cloud_disabled)
        vvt = gen.generate_vvt_entry()

        empfaenger = vvt["empfaenger"]
        assert len(empfaenger) == 1
        assert "Interne Mitarbeiter" in empfaenger[0]
        # Kein Cloud-Anbieter in der Liste
        for e in empfaenger:
            assert "Cloud-KI-Anbieter" not in e

    def test_vvt_cloud_enabled(self, config_cloud_enabled):
        """Cloud-Anbieter als Empfaenger wenn Cloud aktiv."""
        gen = ComplianceReportGenerator(
            config_cloud_enabled,
            cloud_backends=[
                {"category": "cloud", "provider": "openai", "trusted": False, "is_active": True}
            ],
        )
        vvt = gen.generate_vvt_entry()

        empfaenger = vvt["empfaenger"]
        assert len(empfaenger) == 2
        # Zweiter Eintrag ist der Cloud-Anbieter
        cloud_entry = empfaenger[1]
        assert "Cloud-KI-Anbieter" in cloud_entry
        assert "openai" in cloud_entry
        assert "nur anonymisierte Daten" not in cloud_entry
        assert "nicht trusted" in cloud_entry

    def test_vvt_trusted_cloud_backend_documents_raw_path(self, config_cloud_enabled):
        """Trusted Cloud-Backends duerfen nicht als nur anonymisiert beschrieben werden."""
        gen = ComplianceReportGenerator(
            config_cloud_enabled,
            cloud_backends=[
                {"category": "cloud", "provider": "anthropic", "trusted": True, "is_active": True}
            ],
        )
        vvt = gen.generate_vvt_entry()

        combined = " ".join(
            vvt["empfaenger"] + vvt["toms"] + [vvt["anonymisierung"]["cloud_datenpfad"]]
        )
        assert "Originaltextuebermittlung" in combined
        assert "nur anonymisierte Daten" not in combined
        assert "Kein Originaltext" not in combined
        assert "Nur anonymisierte Kopien" not in combined

    def test_vvt_unknown_backend_flags_fail_closed_in_text(self, config_cloud_enabled):
        """Ohne Backend-Flags darf der Report keine Anonymisierung garantieren."""
        gen = ComplianceReportGenerator(config_cloud_enabled)
        vvt = gen.generate_vvt_entry()

        combined = " ".join(
            vvt["empfaenger"] + vvt["toms"] + [vvt["anonymisierung"]["cloud_datenpfad"]]
        )
        assert "Backend-Flags nicht ermittelt" in combined
        assert "Originaltextuebermittlung" in combined
        assert "nur anonymisierte Daten" not in combined

    def test_vvt_drittlandtransfer_cloud_disabled(self, config_cloud_disabled):
        """Drittlandtransfer-Feld dokumentiert klar, dass keine Uebermittlung erfolgt."""
        gen = ComplianceReportGenerator(config_cloud_disabled)
        vvt = gen.generate_vvt_entry()

        drittland = vvt["drittlandtransfer"]
        assert drittland["erfolgt"] is False
        assert drittland["anbieter"] == []
        assert "Cloud-KI deaktiviert" in drittland["drittland_hinweis"]

    def test_vvt_drittlandtransfer_cloud_enabled(self, config_cloud_enabled):
        """Drittlandtransfer-Feld weist Anbieter und DSGVO-Garantien aus, wenn Cloud aktiv."""
        gen = ComplianceReportGenerator(
            config_cloud_enabled,
            cloud_backends=[
                {"category": "cloud", "provider": "openai", "trusted": False, "is_active": True}
            ],
        )
        vvt = gen.generate_vvt_entry()

        drittland = vvt["drittlandtransfer"]
        assert drittland["erfolgt"] is True
        assert "openai" in drittland["anbieter"]
        assert "Art. 44" in drittland["drittland_hinweis"]
        assert "Art. 46" in drittland["garantien"]

    def test_vvt_retention_matches_current_implementation(self, config_cloud_disabled):
        """Report darf keine automatische Retention ohne Cleanup-Pfad behaupten."""
        gen = ComplianceReportGenerator(config_cloud_disabled)
        vvt = gen.generate_vvt_entry()

        loeschfristen = vvt["loeschfristen"]
        combined = " ".join(loeschfristen.values())
        assert "90 Tage" not in combined
        assert "365 Tage" not in combined
        assert "automatisch nach" not in combined
        assert "audit_anonymization" in loeschfristen
        assert "anonymisierung_audit" not in loeschfristen
        assert "Keine automatische Loeschung" in loeschfristen["query_log"]

    def test_wiki_retention_text_matches_current_implementation(self):
        """Wiki darf keine festen Audit-Loeschfristen behaupten."""
        src = (BASE_DIR / "kiara/web/templates/wiki/guide_compliance.html").read_text(
            encoding="utf-8"
        )
        retention_idx = src.index("<strong>Löschfristen:</strong>")
        retention_block = src[retention_idx : retention_idx + 350]
        assert "90 Tage" not in retention_block
        assert "365 Tage" not in retention_block
        assert "anonymisierung_audit" not in retention_block
        assert "audit_anonymization" in retention_block
        assert "keine automatische Löschung" in retention_block

    def test_wiki_cloud_report_text_matches_current_backend_policy(self):
        """Wiki darf keine pauschale nur-anonymisiert Cloud-Behauptung enthalten."""
        src = (BASE_DIR / "kiara/web/templates/wiki/guide_compliance.html").read_text(
            encoding="utf-8"
        )
        report_idx = src.index("<h3>VVT")
        report_block = src[report_idx : report_idx + 2200]
        assert "nur anonymisiert" not in report_block
        assert "SHA256-Hash" not in report_block
        assert "trusted kann Originaltext erhalten" in report_block
        assert "Bewertung erforderlich" in report_block


# ---------------------------------------------------------------------------
# Tests: DSFA
# ---------------------------------------------------------------------------


class TestDSFA:
    def test_dsfa_cloud_disabled(self, config_cloud_disabled):
        """DSFA nicht erforderlich wenn Cloud deaktiviert."""
        gen = ComplianceReportGenerator(config_cloud_disabled)
        dsfa = gen.generate_dsfa()

        assert dsfa["erforderlich"] is False
        assert (
            "nicht aktiviert" in dsfa["ergebnis"].lower()
            or "nicht erforderlich" in dsfa["ergebnis"].lower()
        )

    def test_dsfa_cloud_enabled(self, config_cloud_enabled):
        """DSFA erforderlich wenn Cloud aktiv."""
        gen = ComplianceReportGenerator(
            config_cloud_enabled,
            cloud_backends=[
                {"category": "cloud", "provider": "openai", "trusted": False, "is_active": True}
            ],
        )
        dsfa = gen.generate_dsfa()

        assert dsfa["erforderlich"] is True
        assert "risiken" in dsfa
        assert isinstance(dsfa["risiken"], list)
        assert len(dsfa["risiken"]) > 0
        assert "AKZEPTABEL" in dsfa["ergebnis"]
        assert "erstellt_am" in dsfa

    def test_dsfa_trusted_cloud_backend_does_not_claim_no_originaltext(self, config_cloud_enabled):
        """DSFA muss trusted-Rohpfade ausweisen statt sie als anonymisiert zu verkaufen."""
        gen = ComplianceReportGenerator(
            config_cloud_enabled,
            cloud_backends=[
                {"category": "cloud", "provider": "anthropic", "trusted": True, "is_active": True}
            ],
        )
        dsfa = gen.generate_dsfa()

        combined = str(dsfa)
        assert dsfa["erforderlich"] is True
        assert "Bewertung erforderlich" in dsfa["ergebnis"]
        assert "Originaltext" in combined
        assert "vollstaendige Anonymisierung vor Uebermittlung" not in combined
        assert "Kein Originaltext an Cloud" not in combined
        assert "Nur anonymisierte Kopien an Cloud" not in combined
