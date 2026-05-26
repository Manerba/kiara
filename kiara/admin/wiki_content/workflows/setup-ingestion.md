# Workflow: Indexierung einrichten

## Ueberblick

Dieser Workflow beschreibt, wie die Indexierung (Ingestion) einer Datenquelle gestartet, ueberwacht und optimiert wird. Die Indexierung ist der Prozess, bei dem Dokumente aus einer Datenquelle gelesen, in Chunks zerlegt, vektorisiert und in der VektorDB gespeichert werden. Danach sind die Inhalte im Chat durchsuchbar.

## Voraussetzungen

- Mindestens ein **Embedding-Backend** muss konfiguriert und aktiv sein (`ki.backends.list`)
- Mindestens eine **Datenquelle** muss angelegt und der Verbindungstest erfolgreich sein (`datasource.list`)
- Der **Worker-Service** muss laufen (`kiara-worker.service`)
- **Qdrant** muss erreichbar sein (`system.status`)

## Schritt 1: Worker-Status pruefen

Bevor ein Job gestartet wird, sicherstellen dass der Ingestion-Worker aktiv ist:

```
Admin: Ist der Worker-Service aktiv?

KIara: [Tool: system.service.status]
       kiara-worker.service: active (running) seit 3 Tagen
```

Falls der Worker nicht laeuft, kann er ueber `system.service.restart` gestartet werden.

## Schritt 2: Datenquellen anzeigen

`datasource.list` zeigt alle konfigurierten Datenquellen mit Status und letzter Indexierung. Waehle die zu indexierende Datenquelle aus.

```
Admin: Welche Datenquellen sind konfiguriert?

KIara: [Tool: datasource.list]
       | ID | Name                 | Typ   | Chunks | Letzter Index     |
       |----|----------------------|-------|--------|-------------------|
       | 1  | Technik-Dokumente    | SMB   | 3.456  | 29.03.2026 14:00  |
       | 2  | E-Mail Archiv        | IMAP  | 0      | (noch nie)        |
       | 3  | Quellcode            | Lokal | 890    | 28.03.2026 02:00  |
```

## Schritt 3: Indexierung starten

Es gibt zwei Arten der Indexierung:

### Inkrementelle Indexierung (empfohlen)

`jobs.start` -- Verarbeitet nur neue und geaenderte Dateien. Identifiziert Aenderungen ueber File-Hashes (Pfad + Groesse + mtime). Schnell bei wiederholter Ausfuehrung.

```
Admin: Starte die Indexierung fuer "E-Mail Archiv".

KIara: [Tool: jobs.start]
       Job gestartet (ID: 42).
       Datenquelle: E-Mail Archiv (ID: 2), Typ: Inkrementell
```

### Voll-Indexierung

`jobs.start_full` -- Setzt alle File-Hashes zurueck und verarbeitet jede Datei neu. Notwendig nach:

- Wechsel des Embedding-Modells
- Aenderung der Chunk-Strategie
- Fehlerhafter vorheriger Indexierung

## Schritt 4: Job ueberwachen

`jobs.status` zeigt den aktuellen Fortschritt:

```
Admin: Wie ist der Status des Jobs?

KIara: [Tool: jobs.status]
       Job 42: Laufend (Phase: Embedding)
       Fortschritt: 234/567 Dateien (41%)
       Chunks erzeugt: 3.456
       Geschaetzte Restzeit: 12 Minuten
```

### Steuerungsoptionen waehrend der Indexierung

| Aktion | Funktion | Beschreibung |
|--------|----------|-------------|
| Pausieren | `jobs.pause` | Unterbricht nach dem aktuellen Batch |
| Fortsetzen | `jobs.resume` | Setzt pausierten Job fort |
| Abbrechen | `jobs.cancel` | Bricht den Job endgueltig ab (erfordert Bestaetigung) |

## Schritt 5: Ergebnis pruefen

Nach Abschluss des Jobs zeigt `jobs.status` die Statistiken:

- **Dateien gesamt:** Gesamtzahl der Dateien im Scan
- **Verarbeitet:** Erfolgreich gechunkte Dateien
- **Uebersprungen:** Hash unveraendert (bei inkrementeller Indexierung)
- **Geloescht:** Dateien, die nicht mehr existieren
- **Chunks erzeugt:** Neue Text-Chunks
- **Fehler:** Dateien mit Fehlern (Details im Datei-Protokoll)

Bei Fehlern kann das Datei-Protokoll ueber `jobs.file_logs` abgefragt werden.

## Schritt 6: Schedule setzen (optional)

Fuer automatische Aktualisierungen ein Sync-Intervall konfigurieren:

| Intervall | Beschreibung | Empfohlener Einsatz |
|-----------|-------------|-------------------|
| `manual` | Keine automatische Synchronisierung | Archiv-Datenquellen |
| `daily` | Taeglich (einmal pro 24h) | E-Mail-Postfaecher |
| `weekly` | Woechentlich | Dateisysteme mit selten aenderungen |
| `monthly` | Monatlich | Archiv-Freigaben |
| `custom` | Benutzerdefinierter Cron-Ausdruck | Spezielle Anforderungen |

Beispiel-Cron-Ausdruecke:
- `0 2 * * *` -- Taeglich um 02:00
- `0 */6 * * *` -- Alle 6 Stunden
- `0 0 * * 1` -- Woechentlich Montag Mitternacht

## Schritt 7: Ingestion-Einstellungen optimieren (optional)

Falls die Ergebnisse nicht zufriedenstellend sind:

### DDU aktivieren (fuer bessere PDF/Office-Verarbeitung)

Docling Document Understanding konvertiert komplexe Layouts (Tabellen, Spalten, Grafiken) deutlich besser als Standard-Loader. Erfordert den Docling-Proxy-Service auf dem GPU-Server.

### VLM aktivieren (fuer Bilder)

Vision Language Model beschreibt Bilder als Text, sodass sie durchsuchbar werden. Empfohlen fuer Datenquellen mit technischen Zeichnungen, Fotos oder Screenshots.

### Chunk-Strategie anpassen

Bei spezialisierten Datenquellen kann eine explizite Strategie besser sein als `auto`:
- `code` fuer Quellcode-Repositories
- `email` fuer E-Mail-Datenquellen
- `prose` fuer Berichte und Dokumentationen

Nach Strategie-Aenderung: Voll-Indexierung starten.

## Pipeline-Phasen

Jeder Job durchlaeuft diese Phasen:

| # | Phase | Beschreibung | Typische Dauer |
|---|-------|-------------|----------------|
| 1 | Scanning | Dateien auflisten, Hashes berechnen | Sekunden bis Minuten |
| 2 | Diff | Neue/geaenderte/geloeschte Dateien bestimmen | Sekunden |
| 3 | Loading | Dateien lesen und in Text konvertieren | 0.1-10s pro Datei |
| 4 | VLM | Bilder beschreiben (optional) | 2-10s pro Bild |
| 5 | Chunking | Text in Abschnitte zerlegen | Millisekunden pro Datei |
| 6 | Embedding | Vektorisieren und in Qdrant speichern | 0.05-0.5s pro Chunk |
| 7 | Finalisierung | Statistiken schreiben | Sekunden |

## Troubleshooting

| Problem | Ursache | Loesung |
|---------|---------|---------|
| Job bleibt auf "pending" | Anderer Job laeuft oder Worker inaktiv | `jobs.list` pruefen, Worker-Status pruefen |
| DDU-Timeout | Docling ueberlastet oder Datei zu gross | DDU-Timeout erhoehen, Datei ausschliessen |
| "Circuit Breaker" Abbruch | 10+ aufeinanderfolgende Fehler | Datei-Protokoll pruefen, problematische Dateien ausschliessen |
| Ghost-Job | Worker-Crash waehrend Indexierung | Job abbrechen (`jobs.cancel`), neuen starten |
| Qdrant-Fehler | VektorDB nicht erreichbar | `system.status` pruefen, Qdrant-Container neustarten |

## Verwandte Funktionen

- `jobs.start` -- Inkrementelle Indexierung starten
- `jobs.start_full` -- Voll-Indexierung starten
- `jobs.status` -- Job-Status abfragen
- `jobs.list` -- Alle Jobs anzeigen
- `jobs.cancel` -- Job abbrechen
- `jobs.pause` / `jobs.resume` -- Job pausieren/fortsetzen
- `datasource.update` -- Ingestion-Einstellungen aendern
