# Admin-Agent Konzept

## Ueberblick

Der Admin-Agent ist das zentrale Interface fuer die konversationelle Verwaltung des KIara-Systems. Administratoren steuern alle Funktionen ueber natuerliche Sprache im Admin-Chat. Der Agent basiert auf dem ReAct-Pattern und nutzt eine deklarative Tool-Registry, die gleichzeitig als Schema fuer die REST-API, die GUI und das Hilfesystem dient.

## ReAct-Loop

Der Admin-Agent folgt dem ReAct-Pattern (Reasoning + Acting) in einer kontrollierten Schleife:

1. **Input:** Admin stellt eine Frage oder gibt einen Auftrag im Chat
2. **Reasoning:** Das LLM analysiert den Kontext (Chat-Historie, verfuegbare Tools, aktueller Wizard-State) und entscheidet ueber die naechste Aktion
3. **Action:** Ein Tool wird serverseitig ausgefuehrt (`AdminFunctionTool`)
4. **Observation:** Das Tool-Ergebnis (JSON) wird dem LLM als Beobachtung zurueckgefuehrt
5. **Answer:** Das LLM formuliert eine verstaendliche Antwort fuer den Administrator

Die maximale Anzahl an Iterationen ist konfigurierbar und begrenzt, um Endlosschleifen zu vermeiden. Zwischen den Iterationen wird der vollstaendige Message-Verlauf an das LLM uebergeben, sodass es den Kontext vorheriger Tool-Aufrufe beruecksichtigen kann.

### Streaming

Die Antworten des Agenten werden per **SSE (Server-Sent Events)** gestreamt. Der Benutzer sieht die Antwort zeichenweise erscheinen. Tool-Aufrufe werden als eigene SSE-Events signalisiert (`tool_call`, `tool_result`), sodass das Frontend den Fortschritt visualisieren kann.

## AdminFunctionRegistry

Die AdminFunctionRegistry ist die **Single Source of Truth** fuer alle Admin-Funktionen. Jede Funktion wird einmal deklarativ definiert mit:

| Attribut | Beschreibung |
|----------|-------------|
| `name` | Eindeutiger Name im Dot-Namespace (z.B. `datasource.create`, `ki.backends.test`) |
| `description` | Kurzbeschreibung fuer das LLM und das Hilfesystem |
| `help_text` | Ausfuehrlicher Hilfetext mit Beispielen |
| `parameters` | JSON-Schema der Eingabeparameter (Typ, Pflichtfelder, Defaults) |
| `requires_confirmation` | Boolean: Erfordert explizite Bestaetigung vor Ausfuehrung |
| `category` | Funktionskategorie (datasource, ki, users, system, jobs, ...) |
| `http_method` / `http_path` | Optionales HTTP-Mapping fuer GUI-Kompatibilitaet |

Die Registry wird beim Start des Admin-Chat-Services geladen und dient gleichzeitig als:

- **Tool-Definition** fuer den ReAct-Agenten
- **API-Schema** fuer den Auto-Router (REST-Endpunkte)
- **Hilfesystem** fuer den `help`-Befehl und das Wiki

### Funktions-Namespaces

| Namespace | Beispiel-Funktionen |
|-----------|-------------------|
| `datasource.*` | `datasource.create`, `datasource.list`, `datasource.delete`, `datasource.test_connection` |
| `ki.*` | `ki.backends.create`, `ki.backends.test`, `ki.backends.list`, `ki.backends.categories` |
| `jobs.*` | `jobs.start`, `jobs.status`, `jobs.cancel`, `jobs.list` |
| `users.*` | `users.create`, `users.update`, `users.ldap.sync`, `users.groups.list` |
| `system.*` | `system.status`, `system.service.restart`, `system.integrity.check`, `system.journal` |
| `compliance.*` | `compliance.audit.list`, `compliance.anonymize.test` |
| `tools.*` | `tools.erp.connections.list`, `tools.mcp.toggle`, `tools.mcp.keys.create` |
| `backup.*` | `backup.create`, `backup.list`, `backup.restore` |
| `network.*` | `network.ping`, `network.dns`, `network.portscan` |
| `wiki.*` | `wiki.search` |
| `workflow.*` | `workflow.list`, `workflow.start` |

## Confirmation-Flow

Destruktive Aktionen erfordern eine explizite Bestaetigung. Der Flow ist mehrstufig und manipulationssicher:

1. **Tool-Aufruf:** Agent ruft eine Funktion mit `requires_confirmation=True` auf (z.B. `datasource.delete`)
2. **Backend:** Generiert einen eindeutigen Bestaetigungs-String (z.B. "LOESCHE-DS-7") und speichert ihn serverseitig
3. **SSE-Event:** `confirmation_required` wird an das Frontend gesendet
4. **Frontend:** Zeigt eine System-Bubble (nicht LLM-generiert) mit Warnung und Bestaetigungsfeld
5. **Admin:** Gibt den exakten Bestaetigungs-String ein
6. **Backend:** Validiert den String, fuehrt die Aktion aus und sendet das Ergebnis

Dieser Flow verhindert, dass das LLM selbststaendig destruktive Aktionen ausfuehrt oder den Benutzer zur Bestaetigung manipuliert.

### Aktionen mit Bestaetigung

- `datasource.delete` -- Datenquelle und alle Chunks loeschen
- `backup.restore` -- Backup wiederherstellen (ueberschreibt aktuelle Daten)
- `users.ldap.config.delete` -- LDAP-Konfiguration loeschen (deaktiviert alle LDAP-Benutzer)
- `system.service.restart` -- Service neustarten
- Voll-Indexierung (setzt File-Hashes zurueck)

## Wizard-State

Funktionen mit vielen Pflichtfeldern nutzen einen mehrstufigen Wizard, der den Administrator Schritt fuer Schritt durch die Konfiguration fuehrt:

- **Server-seitiger State:** Wizard-Daten werden in PostgreSQL gespeichert und ueberleben Seitenneuladungen
- **Schrittweise Erfassung:** Der Agent fragt Parameter einzeln oder in logischen Gruppen ab
- **Fortschrittsanzeige:** Eine Checkliste zeigt, welche Parameter bereits erfasst sind
- **Validierung:** Jeder Schritt validiert die Eingabe, bevor zum naechsten uebergegangen wird
- **Abbruch:** Der Wizard kann jederzeit abgebrochen werden (`wizard.cancel`)

### Ersteinrichtungs-Wizard

Ein spezieller regelbasierter Wizard (ohne LLM-Abhaengigkeit) fuehrt durch die Ersteinrichtung:

1. Deployment-Typ waehlen (Cloud oder Ollama)
2. API-Key / Ollama-URL eingeben und testen
3. Modell auswaehlen
4. Backends automatisch anlegen

Der Ersteinrichtungs-Wizard ist crash-sicher und kann nach einem Neustart fortgesetzt werden.

## Guided Workflows

Workflows sind vordefinierte Ablaufplaene fuer komplexe Aufgaben:

- `workflow.list` -- Zeigt alle verfuegbaren Workflows mit Beschreibung
- `workflow.start` -- Liefert den Schrittplan (Markdown) an den Agenten

Der Agent fuehrt die referenzierten Admin-Funktionen einzeln auf und nutzt den Schrittplan als Leitfaden. Es gibt keinen serverseitigen Workflow-State -- der Agent arbeitet den Plan stateless ab. Bei Fehlern kann der Administrator korrigierend eingreifen.

## WikiSearchTool

Fuer Dokumentations-Fragen nutzt der Agent das Tool `wiki.search`:

- Durchsucht lokale Markdown-Dateien (Guides, Concepts, Workflows)
- Kein Retriever oder Vektor-DB noetig (einfache Textsuche mit Scoring)
- Liefert relevante Abschnitte mit Titel und Slug
- Unterstuetzt Slug-Filter fuer gezielte Suche in bestimmten Dokumenten

## LLM-Backend-Auswahl

Der Admin-Chat unterstuetzt drei LLM-Backends, die per Dropdown im Frontend gewaehlt werden:

| Backend | Vorteile | Einschraenkungen |
|---------|---------|-----------------|
| **Ollama** (lokal) | Datenschutz, keine Cloud-Abhaengigkeit, keine Kosten | Benoetigt GPU, begrenzte Modellgroesse |
| **Anthropic** (Claude) | Hohe Qualitaet, natives Tool-Use, lange Kontextfenster | API-Key noetig, Kosten pro Token |
| **OpenAI** (GPT) | Breite Modellauswahl, schnelle Antworten | API-Key noetig, Kosten pro Token |

Die Adapter (`adapters/ollama.py`, `adapters/anthropic.py`, `adapters/openai.py`) uebersetzen das einheitliche Tool-Schema in die provider-spezifischen Formate.
