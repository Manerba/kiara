# Sicherheitsrichtlinie

## Unterstützte Versionen

Kiara befindet sich in aktiver Alpha-Entwicklung (`0.0.x`). Sicherheitsupdates
werden ausschließlich für den aktuellen `master`-Branch bereitgestellt. Es gibt
noch keine Stable-Linie mit rückwirkenden Patches.

| Version            | Unterstützt |
| ------------------ | ----------- |
| `master` (`0.0.x`) | ✅          |
| ältere Tags        | ❌          |

## Eine Schwachstelle melden

Bitte melde Sicherheitslücken **nicht über öffentliche GitHub-Issues**.

Bevorzugter Weg:

1. **GitHub Security Advisories** — Schaltfläche „Report a vulnerability" im
   Reiter *Security* dieses Repositories (privates, koordiniertes Disclosure).
2. Alternativ per E-Mail an **security@kiara-ai.de** (gerne PGP-verschlüsselt).

Bitte gib soweit möglich an:

- betroffene Komponente / Datei / Endpunkt,
- Reproduktionsschritte oder einen Proof-of-Concept,
- eine Einschätzung der Auswirkung.

## Ablauf

- Bestätigung des Eingangs: innerhalb von **72 Stunden**.
- Erste Einschätzung: innerhalb von **7 Tagen**.
- Wir halten dich über den Fortschritt auf dem Laufenden und stimmen einen
  Offenlegungszeitpunkt ab (Coordinated Disclosure).

## Geltungsbereich

Als sicherheitsrelevant gelten insbesondere:

- Umgehung der **Anonymisierung/Pseudonymisierung** vor Cloud-LLM-Aufrufen,
- Umgehung der **Berechtigungs- bzw. Datenquellen-Filter** im Retrieval,
- **Authentifizierungs- oder Session-Schwächen** der Web-, Admin- oder
  MCP-Endpunkte,
- **Secret-Leaks** (Klartext-Credentials, DSNs, API-Keys),
- **Remote Code Execution** über die Ingestion (präparierte Dokumente) oder den
  Tool-Use des Agenten.

Bewusste Designentscheidungen sind **kein** Sicherheitsbug, z. B. das opt-in
gehaltene LDAP-Zertifikats-Handling für interne Self-Signed-ADs oder die
nicht-anonymisierten Roh-Audit-Logs. Diese sind in der Dokumentation als
gewolltes Verhalten beschrieben.
