# Workflow: Ersteinrichtung

## Ueberblick

Dieser Workflow fuehrt durch die komplette Ersteinrichtung einer neuen KIara-Installation. Am Ende ist das System bereit fuer den produktiven Einsatz: LLM-Backend konfiguriert, Datenquellen angebunden, Indexierung durchgefuehrt und optional Benutzer eingerichtet.

Die Ersteinrichtung besteht aus drei Pflicht-Schritten und vier optionalen Schritten. Die Reihenfolge der Pflicht-Schritte ist verbindlich, da jeder auf dem vorherigen aufbaut.

## Voraussetzungen

- KIara ist installiert und die Services laufen (`kiara.service`, `kiara-admin.service`, `kiara-worker.service`)
- Qdrant laeuft (Docker-Container auf Port 6333)
- PostgreSQL ist konfiguriert und erreichbar
- Netzwerkzugang zu den gewuenschten Datenquellen

## Uebersicht der Schritte

| # | Schritt | Dauer | Pflicht | Verwandter Workflow |
|---|---------|-------|---------|-------------------|
| 1 | KI-Backend einrichten | 5-10 Min | Ja | setup-llm-backend |
| 2 | VektorDB pruefen | 1-2 Min | Ja | -- |
| 3 | Datenquelle anlegen | 5-10 Min | Ja | setup-datasource |
| 4 | Indexierung starten | Min. bis Std. | Ja | setup-ingestion |
| 5 | Benutzer einrichten | 5-15 Min | Optional | -- |
| 6 | Anonymisierung konfigurieren | 5-10 Min | Empfohlen | -- |
| 7 | Chat testen | 2-5 Min | Ja | -- |

## Schritt 1: KI-Backend einrichten

Ohne LLM- und Embedding-Backend kann KIara nicht arbeiten. Es werden mindestens zwei Backends benoetigt:

- **Embedding-Backend** (Pflicht): Vektorisiert Text-Chunks fuer die Suche
- **LLM- oder Cloud-Backend** (Pflicht): Generiert Antworten auf Benutzer-Fragen

### Option A: Lokales Backend (Ollama)

1. Ollama-Erreichbarkeit pruefen (`ki.backends.test`)
2. Embedding-Backend anlegen: `ki.backends.create` mit `category=embedding`, `provider=ollama`, `model=nomic-embed-text`
3. LLM-Backend anlegen: `ki.backends.create` mit `category=llm`, `provider=ollama`, `model=qwen3:8b`
4. Beide testen: `ki.backends.test`

### Option B: Cloud-Backend (Anthropic/OpenAI)

1. Cloud-Backend anlegen: `ki.backends.create` mit `category=cloud`, `provider=anthropic`, API-Key angeben
2. Embedding-Backend anlegen (lokal oder Cloud)
3. Beide testen

Detaillierte Anleitung: Siehe Workflow "KI-Backend einrichten".

## Schritt 2: VektorDB pruefen

Sicherstellen, dass Qdrant erreichbar ist:

- `system.status` prueft alle Komponenten, einschliesslich Qdrant
- Bei Problemen: `docker ps | grep qdrant`, `curl http://localhost:6333/healthz`

## Schritt 3: Datenquelle anlegen

Mindestens eine Datenquelle konfigurieren:

1. Konnektor-Typ waehlen (`datasource.connector_types`)
2. Verbindungsdaten eingeben (Server, Pfad, Credentials je nach Typ)
3. Verbindung testen (`datasource.test_connection`)
4. Datenquelle anlegen (`datasource.create`)

Unterstuetzte Typen: local, smb, webdav, imap, exchange_online, sql.
Beim Typ `sql` wird die konkrete Datenbank ueber `config.driver` gewaehlt;
gueltige Driver-Optionen sind `mssql`, `postgresql` und `mysql`.

Detaillierte Anleitung: Siehe Workflow "Datenquelle einrichten".

## Schritt 4: Indexierung starten

Nach der Anlage der Datenquelle die Indexierung starten:

1. Job starten: `jobs.start` (inkrementell, empfohlen)
2. Fortschritt ueberwachen: `jobs.status`
3. Ergebnis pruefen: Chunks in der VektorDB vorhanden?

Die Jobs werden sequentiell abgearbeitet (keine GPU-Konkurrenz). Bei mehreren Datenquellen werden sie nacheinander verarbeitet.

Detaillierte Anleitung: Siehe Workflow "Indexierung einrichten".

## Schritt 5: Benutzer einrichten (optional)

Fuer den produktiven Einsatz Benutzer anlegen:

### Lokale Benutzer

- `users.create` -- Benutzername, Passwort, Name und optional Admin-Rolle
- Geeignet fuer kleine Teams oder Testumgebungen

### LDAP/Active-Directory

1. LDAP-Verbindung konfigurieren (Server, Port, SSL, Bind-DN, Search Base)
2. AD-Gruppe auswaehlen und Benutzer synchronisieren (`users.ldap.sync`)
3. Optional: Gruppen importieren fuer Datenquellen-Zugriffskontrolle

Empfohlene Reihenfolge:
1. LDAP-Verbindung konfigurieren
2. AD-Gruppe auswaehlen und Benutzer-Sync
3. Gruppen importieren
4. Mitglieder synchronisieren
5. Datenquellen den Gruppen zuordnen

## Schritt 6: Anonymisierung konfigurieren (empfohlen)

Besonders wichtig bei Cloud-Backends:

1. Pruefen ob Anonymisierung aktiviert ist (`system.status`)
2. Bei LDAP: LDAP-Mitarbeiter in Anonymisierung einbeziehen (automatisch wenn LDAP konfiguriert)
3. Bei ERP-Daten: Stammdaten-CSVs unter `/etc/kiara/stammdaten` ablegen und ERP-Import aktivieren
4. Anonymisierung mit realistischem Text testen (`compliance.anonymize.test`)
5. Cloud-Content-Klasse pruefen (Standard: LOW -- empfohlen)

## Schritt 7: Chat testen

Nachdem die Indexierung abgeschlossen ist:

1. **Admin-Chat** testen (Port 8502): Administrative Fragen stellen
2. **Benutzer-Chat** testen (Port 8501): Wissensbasis-Fragen stellen

Beispiel-Fragen:
- "Was steht in der technischen Dokumentation ueber [Thema]?"
- "Fasse das Dokument [Dateiname] zusammen."
- "Welche Informationen gibt es zu [Suchbegriff]?"

Falls keine Ergebnisse kommen:
- Indexierung abgeschlossen? `jobs.list` pruefen
- Embedding-Backend aktiv? `ki.backends.list` pruefen
- Qdrant erreichbar? `system.status` pruefen
- Chunks vorhanden? Im VektorDB-Tab pruefen

## Checkliste Ersteinrichtung

- [ ] Embedding-Backend konfiguriert und getestet
- [ ] LLM-Backend (lokal oder Cloud) konfiguriert und getestet
- [ ] Qdrant erreichbar und Collection vorhanden
- [ ] Mindestens eine Datenquelle angelegt und Verbindungstest bestanden
- [ ] Indexierung erfolgreich durchgefuehrt (Chunks in VektorDB)
- [ ] Benutzer angelegt (lokal oder LDAP)
- [ ] Anonymisierung aktiviert (besonders bei Cloud-Backends)
- [ ] Chat getestet und Ergebnisse erhalten

## Crash-Recovery

Der Ersteinrichtungs-Wizard ist crash-sicher. Bei Unterbrechung koennen folgende Funktionen den aktuellen Stand pruefen:

- `ki.backends.list` -- Welche Backends existieren?
- `datasource.list` -- Welche Datenquellen sind konfiguriert?
- `jobs.list` -- Welche Jobs laufen oder sind abgeschlossen?
- `system.status` -- Gesamtstatus des Systems

Der Wizard kann nach einem Neustart an der Stelle fortgesetzt werden, an der er unterbrochen wurde.

## Verwandte Workflows

- **KI-Backend einrichten** (`setup-llm-backend`) -- Detaillierte Anleitung fuer LLM-/Embedding-Setup
- **Datenquelle einrichten** (`setup-datasource`) -- Alle Konnektor-Typen und ihre Konfiguration
- **Indexierung einrichten** (`setup-ingestion`) -- Job-Steuerung, Monitoring und Optimierung
