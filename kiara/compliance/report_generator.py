"""DSGVO-Compliance-Report-Generierung.

Erzeugt Berichte ueber Cloud-Nutzung, Anonymisierungsstatistiken
und Datenverarbeitungsaktivitaeten fuer das Compliance-Dashboard.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ComplianceReportGenerator:
    """Generiert DSGVO-Compliance-Reports."""

    def __init__(self, config: dict, cloud_backends: list[dict] | None = None):
        self._config = config
        self._cloud_backends = cloud_backends

    def _cloud_profile(self) -> dict:
        """Beschreibt den tatsaechlichen Cloud-Datenpfad fuer Reporttexte."""
        cloud_cfg = self._config.get("cloud", {})
        cloud_enabled_cfg = bool(cloud_cfg.get("enabled", False))
        configured_provider = str(cloud_cfg.get("provider", "") or "").strip()

        active_backends: list[dict] = []
        if self._cloud_backends is not None:
            for backend in self._cloud_backends:
                if "is_active" in backend and not bool(backend.get("is_active")):
                    continue
                category = str(backend.get("category", "cloud") or "").lower()
                if category and category != "cloud":
                    continue
                active_backends.append(backend)

        providers: list[str] = []
        if active_backends:
            for backend in active_backends:
                provider = str(backend.get("provider") or backend.get("name") or "").strip()
                if provider and provider not in providers:
                    providers.append(provider)
        elif configured_provider:
            providers.append(configured_provider)

        trusted_count = sum(1 for backend in active_backends if bool(backend.get("trusted", False)))
        untrusted_count = len(active_backends) - trusted_count
        enabled = cloud_enabled_cfg or bool(active_backends)
        backend_flags_unknown = self._cloud_backends is None and cloud_enabled_cfg

        if not enabled:
            policy_text = "Cloud-KI ist deaktiviert; es erfolgt keine Cloud-Uebermittlung."
        elif backend_flags_unknown:
            policy_text = (
                "Cloud-KI ist aktiviert, die aktiven Backend-Flags wurden fuer diesen Report "
                "nicht ermittelt. Trusted Cloud-Backends koennen Originaltexte erhalten; "
                "nicht-trusted Backends werden ueber Anonymisierungs- und Restriction-Policy gefuehrt."
            )
        elif trusted_count and untrusted_count:
            policy_text = (
                "Gemischte Cloud-Konfiguration: trusted Cloud-Backends koennen Originaltexte "
                "erhalten; nicht-trusted Cloud-Backends werden ueber Anonymisierungs- und "
                "Restriction-Policy gefuehrt."
            )
        elif trusted_count:
            policy_text = (
                "Alle aktiven Cloud-Backends sind als trusted markiert; "
                "Originaltextuebermittlung an diese Cloud-Anbieter ist moeglich."
            )
        elif untrusted_count:
            policy_text = (
                "Aktive Cloud-Backends sind nicht trusted; Cloud-Uebermittlung wird ueber "
                "Anonymisierungs- und Restriction-Policy gefuehrt."
            )
        else:
            policy_text = (
                "Cloud-KI ist aktiviert, aber es wurden keine aktiven Cloud-Backends im "
                "Backend-Register gefunden."
            )

        return {
            "enabled": enabled,
            "providers": providers,
            "trusted_count": trusted_count,
            "untrusted_count": untrusted_count,
            "backend_flags_unknown": backend_flags_unknown,
            "policy_text": policy_text,
        }

    @staticmethod
    def _cloud_recipient_text(profile: dict) -> str:
        provider_text = ", ".join(profile["providers"]) if profile["providers"] else "konfiguriert"

        if profile["backend_flags_unknown"]:
            suffix = "Backend-Flags nicht ermittelt; Originaltextuebermittlung je nach trusted-Konfiguration moeglich"
        elif profile["trusted_count"] and profile["untrusted_count"]:
            suffix = "gemischt: trusted kann Originaltext erhalten, nicht-trusted nutzt Anonymisierung/Restriction-Policy"
        elif profile["trusted_count"]:
            suffix = "trusted; Originaltextuebermittlung moeglich"
        elif profile["untrusted_count"]:
            suffix = "nicht trusted; Anonymisierung/Restriction-Policy vor Cloud-Versand"
        else:
            suffix = "aktiviert, aber kein aktives Cloud-Backend im Register gefunden"

        return f"Cloud-KI-Anbieter: {provider_text} ({suffix})"

    @staticmethod
    def _drittlandtransfer(profile: dict) -> dict:
        """Drittland-Transfer-Beschreibung gemaess Art. 30 Abs. 1 lit. e DSGVO.

        Fail-closed: Konkrete Drittlaender werden NICHT geraten, sondern explizit
        als pro-Anbietervertrag zu dokumentierend ausgewiesen. KIara hat keine
        zuverlaessige Information ueber den Datenverarbeitungsort einzelner Cloud-
        Anbieter; diese gehoert in den Auftragsverarbeitungsvertrag.
        """
        if not profile["enabled"]:
            return {
                "erfolgt": False,
                "anbieter": [],
                "drittland_hinweis": "Keine Uebermittlung in Drittlaender (Cloud-KI deaktiviert).",
                "garantien": "Nicht erforderlich.",
            }

        return {
            "erfolgt": True,
            "anbieter": list(profile["providers"]),
            "drittland_hinweis": (
                "Uebermittlung personenbezogener Daten an Cloud-KI-Anbieter. "
                "Sitz und Datenverarbeitungsorte der Anbieter sind je Auftragsverarbeitungs"
                "vertrag zu dokumentieren; bei Anbietern mit Sitz oder Datenverarbeitung "
                "ausserhalb der EU/EWR ist eine Uebermittlung in Drittlaender im Sinne "
                "Art. 44 ff. DSGVO anzunehmen."
            ),
            "garantien": (
                "Geeignete Garantien gemaess Art. 46 DSGVO sind je Anbieter zu dokumentieren "
                "(z.B. EU-Standardvertragsklauseln, EU-US Data Privacy Framework, "
                "Angemessenheitsbeschluss). Trusted-Backends erlauben Originaltext"
                "uebermittlung; nicht-trusted Backends werden ueber Anonymisierungs- und "
                "Restriction-Policy gefuehrt."
            ),
        }

    @staticmethod
    def _cloud_toms(profile: dict) -> list[str]:
        toms: list[str] = []
        if not profile["enabled"]:
            return toms

        if profile["backend_flags_unknown"]:
            toms.append(
                "Cloud-Backend-Flags als Grundlage der Datenschutzbewertung; trusted Backends koennen Originaltexte erhalten"
            )
        if profile["trusted_count"]:
            toms.append(
                "Explizites trusted-Flag fuer Cloud-Backends mit moeglicher Originaltextuebermittlung"
            )
        if profile["untrusted_count"] or profile["backend_flags_unknown"]:
            toms.append("Anonymisierungs- und Restriction-Policy fuer nicht-trusted Cloud-Backends")

        toms.extend(
            [
                "Verschluesselte Uebertragung (HTTPS/TLS) zu Cloud-Anbietern",
                "Audit-Logging aller Cloud-Anfragen",
                "Budget-Limitierung fuer Cloud-KI-Nutzung",
            ]
        )
        return toms

    def generate_vvt_entry(self) -> dict:
        """VVT-Eintrag (Verzeichnis der Verarbeitungstaetigkeiten).

        Returns dict with keys:
        - name: str
        - verantwortlicher: str (Name/Kontakt des Verantwortlichen, Art. 30 Abs. 1 lit. a)
        - zweck: str
        - rechtsgrundlage: str
        - betroffene: list[str]
        - datenkategorien: list[str]
        - empfaenger: list[str] (Cloud-Anbieter wenn aktiviert)
        - drittlandtransfer: dict (Art. 30 Abs. 1 lit. e DSGVO)
        - loeschfristen: dict
        - toms: list[str] (Technisch-organisatorische Massnahmen)
        - anonymisierung: dict (Details zum Anonymisierungsverfahren)
        - erstellt_am: str (ISO timestamp)
        """
        cloud_profile = self._cloud_profile()
        compliance_cfg = self._config.get("compliance", {})
        verantwortlicher = compliance_cfg.get(
            "verantwortlicher",
            "Nicht konfiguriert – bitte in /etc/kiara/kiara.yaml unter 'compliance.verantwortlicher' hinterlegen",
        )
        dsb_kontakt = compliance_cfg.get(
            "dsb_kontakt",
            "Nicht konfiguriert – bitte in /etc/kiara/kiara.yaml unter 'compliance.dsb_kontakt' hinterlegen",
        )

        empfaenger = ["Interne Mitarbeiter (Nutzer des RAG-Systems)"]
        if cloud_profile["enabled"]:
            empfaenger.append(self._cloud_recipient_text(cloud_profile))

        toms = [
            "Deterministische Pseudonymisierung personenbezogener Daten fuer nicht-trusted Datenpfade",
            "Aho-Corasick-basierte Entitaetserkennung mit Regex-Fallback",
            "Inhaltsklassifizierung (BLOCKED/HIGH/MEDIUM/LOW) pro Datenquelle",
            "Rollenbasierte Zugriffskontrolle (RBAC) ueber LDAP-Gruppen",
            "Audit-Logging aller Anfragen",
            "Lokale Vektordatenbank (Qdrant) - keine Datenuebertragung fuer Retrieval",
        ]
        toms.extend(self._cloud_toms(cloud_profile))

        return {
            "name": "KI-gestuetztes Wissensmanagement (KIara)",
            "verantwortlicher": verantwortlicher,
            "dsb_kontakt": dsb_kontakt,
            "zweck": "Bereitstellung eines KI-gestuetzten Frage-Antwort-Systems fuer die interne IT-Wissensbasis zur Unterstuetzung von Mitarbeitern bei technischen Fragestellungen.",
            "rechtsgrundlage": "Art. 6 Abs. 1 lit. f DSGVO (berechtigtes Interesse an effizienter Wissensnutzung)",
            "betroffene": [
                "Mitarbeiter",
                "Kunden (in Dokumenten referenziert)",
                "Lieferanten (in Dokumenten referenziert)",
            ],
            "datenkategorien": [
                "Technische Dokumentation und Wissensbasis-Inhalte",
                "Benutzeranfragen im Klartext (query_log.question_excerpt, query_log.full_prompt)",
                "Antworten im Klartext (query_log.answer_excerpt, query_log.full_answer)",
                "Quell-/Kontextverweise im Klartext (query_log.context_sources)",
                "Tool-Call-Protokolle (query_log.tool_calls)",
                "Nutzungsstatistiken (aggregiert)",
            ],
            "empfaenger": empfaenger,
            "drittlandtransfer": self._drittlandtransfer(cloud_profile),
            "loeschfristen": {
                "query_log": "Keine automatische Loeschung im aktuellen Build; Aufbewahrung bis zur manuellen Bereinigung",
                "cloud_audit_log": "Keine automatische Loeschung im aktuellen Build; Aufbewahrung bis zur manuellen Bereinigung",
                "chat_messages": "Durch Benutzer loeschbar; keine separate automatische Retention im aktuellen Build",
                "audit_anonymization": "Keine automatische Loeschung im aktuellen Build; Aufbewahrung bis zur manuellen Bereinigung",
            },
            "toms": toms,
            "anonymisierung": {
                "verfahren": "Deterministische Pseudonymisierung",
                "engine": "Aho-Corasick + Regex-Fallback",
                "kategorien": [
                    "Kunden",
                    "Lieferanten",
                    "Mitarbeiter",
                    "Produkte",
                    "Standorte",
                    "IBAN",
                    "E-Mail",
                    "Telefon",
                    "Steuer-ID",
                    "IP-Adressen",
                ],
                "original_persistiert": True,
                "hash_verfahren": "SHA256",
                "cloud_datenpfad": cloud_profile["policy_text"],
            },
            "erstellt_am": datetime.now(timezone.utc).isoformat(),
        }

    def generate_dsfa(self) -> dict:
        """Datenschutz-Folgenabschaetzung (nur relevant wenn Cloud-KI aktiviert)."""
        cloud_profile = self._cloud_profile()
        cloud_enabled = cloud_profile["enabled"]
        if cloud_profile["trusted_count"] or cloud_profile["backend_flags_unknown"]:
            cloud_result = (
                "Bewertung erforderlich: trusted Cloud-Backends koennen Originaltext erhalten; "
                "vertragliche und organisatorische Freigabe dokumentieren."
            )
        else:
            cloud_result = "Risiko nach Umsetzung der Massnahmen: AKZEPTABEL"

        return {
            "erforderlich": cloud_enabled,
            "system": "KIara – KI-gestuetztes Wissensmanagement",
            "beschreibung": "Systematische Beschreibung der Verarbeitungstaetigkeit bei Nutzung externer Cloud-KI-Dienste.",
            "notwendigkeit_und_verhaeltnismaessigkeit": {
                "zweck": "Hoehere Antwortqualitaet bei komplexen strategischen Fragen durch leistungsfaehigere Cloud-Modelle",
                "verhaeltnismaessigkeit": cloud_profile["policy_text"],
                "minimierung": "Lokale Verarbeitung als Standard; Cloud nur bei unzureichender lokaler Antwortqualitaet",
            },
            "risiken": [
                {
                    "risiko": "Offenlegung von Originaltexten oder pseudonymisierten Daten an Cloud-Anbieter",
                    "eintrittswahrscheinlichkeit": "mittel"
                    if cloud_profile["trusted_count"] or cloud_profile["backend_flags_unknown"]
                    else "gering",
                    "schwere": "hoch"
                    if cloud_profile["trusted_count"] or cloud_profile["backend_flags_unknown"]
                    else "mittel",
                    "massnahmen": [
                        "Trusted-Flag als explizite Backend-Freigabe dokumentieren",
                        "Anonymisierungs- und Restriction-Policy fuer nicht-trusted Cloud-Backends",
                        "Audit-Logging aller Cloud-Aufrufe",
                    ],
                },
                {
                    "risiko": "Unbefugter Zugriff auf Cloud-API-Schluessel",
                    "eintrittswahrscheinlichkeit": "gering",
                    "schwere": "hoch",
                    "massnahmen": [
                        "API-Schluessel in geschuetzter Konfigurationsdatei",
                        "Budget-Limitierung",
                        "Audit-Logging",
                    ],
                },
                {
                    "risiko": "Datenverlust oder -manipulation durch Cloud-Anbieter",
                    "eintrittswahrscheinlichkeit": "sehr gering",
                    "schwere": "mittel",
                    "massnahmen": [
                        "Originaldaten bleiben lokal gespeichert",
                        "Cloud-Anbieter erhalten keinen Schreibzugriff",
                        "Cloud-Antworten werden lokal auditiert",
                    ],
                },
            ]
            if cloud_enabled
            else [],
            "ergebnis": cloud_result
            if cloud_enabled
            else "Keine DSFA erforderlich (Cloud-KI nicht aktiviert)",
            "erstellt_am": datetime.now(timezone.utc).isoformat(),
        }
