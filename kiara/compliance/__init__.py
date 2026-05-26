"""Compliance: Berichtgenerierung fuer Datenschutz und Audit.

Generiert Compliance-Berichte fuer Datenschutzbeauftragte und Auditoren.
Dokumentiert Datenquellen, Anonymisierungsregeln, Cloud-Nutzung und
Zugriffsrechte in strukturierter Form.

Beziehungen:
    - anonymization liefert Anonymisierungs-Statistiken
    - cloud.audit liefert Cloud-Nutzungsdaten
    - web.api_compliance stellt die Download-API bereit

Module:
    report_generator — ComplianceReportGenerator (HTML/PDF-Berichte)
"""

from .report_generator import ComplianceReportGenerator
