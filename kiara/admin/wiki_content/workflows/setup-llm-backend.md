# Workflow: KI-Backend einrichten

## Ueberblick

Dieser Workflow fuehrt Schritt fuer Schritt durch die Einrichtung eines LLM-Backends. Er unterstuetzt sowohl lokale Ollama-Backends als auch Cloud-Backends (Anthropic, OpenAI). Am Ende ist ein funktionsfaehiges Backend konfiguriert und getestet, sodass der Chat und die Indexierung arbeiten koennen.

## Voraussetzungen

- Fuer lokale Backends: Ollama muss installiert und erreichbar sein (Standard: `http://localhost:11434`)
- Fuer Cloud-Backends: API-Key des jeweiligen Anbieters (Anthropic: `sk-ant-...`, OpenAI: `sk-...`)
- Admin-Zugang zur KIara-Admin-UI oder zum Admin-Chat

## Schritt 1: Deployment-Typ waehlen

Entscheide, ob ein lokales oder Cloud-Backend eingerichtet werden soll.

**Lokales Backend (Ollama):**
- Daten bleiben im Unternehmen, keine Cloud-Abhaengigkeit
- Benoetigt GPU-Hardware (z.B. RTX 3060 mit 12 GB VRAM)
- Empfohlene Modelle: `qwen3:8b` (LLM), `nomic-embed-text` (Embedding)

**Cloud-Backend (Anthropic/OpenAI):**
- Hoehere Modellqualitaet, kein eigener GPU-Server noetig
- Kosten pro Token, Internet-Abhaengigkeit
- Anonymisierung sollte aktiviert sein, um sensible Daten zu schuetzen

## Schritt 2: Verfuegbare Kategorien pruefen

Der Agent ruft `ki.backends.categories` auf, um die konfigurierbaren Backend-Kategorien anzuzeigen:

- **Embedding** -- Vektorisierung fuer die Suche (Pflicht)
- **LLM** -- Lokales Sprachmodell fuer Antworten
- **Cloud** -- Cloud-basiertes Sprachmodell
- **VLM** -- Vision Language Model fuer Bilder (optional)
- **DDU** -- Document Understanding fuer PDF/Office (optional)

Mindestens ein **Embedding**-Backend und ein **LLM** oder **Cloud**-Backend sind erforderlich.

## Schritt 3: Backend-Details sammeln

### Lokales Backend (Ollama)

| Parameter | Beschreibung | Beispielwert |
|-----------|-------------|-------------|
| `category` | Backend-Kategorie | `llm` oder `embedding` |
| `provider` | Provider-Name | `ollama` |
| `url` | Ollama-Server-URL | `http://localhost:11434` |
| `model` | Modellname | `qwen3:8b` (LLM) oder `nomic-embed-text` (Embedding) |
| `name` | Anzeigename (optional) | "Ollama GPU-Server" |
| `priority` | Prioritaet (1-100) | `10` (primaer) |

### Cloud-Backend

| Parameter | Beschreibung | Beispielwert |
|-----------|-------------|-------------|
| `category` | Backend-Kategorie | `cloud` |
| `provider` | Provider-Name | `anthropic` oder `openai` |
| `model` | Modellname | `claude-sonnet-4-20250514` oder `gpt-4o` |
| `api_key` | API-Schluessel | `sk-ant-...` |
| `name` | Anzeigename (optional) | "Anthropic Claude" |
| `base_url` | Custom-Endpunkt (optional) | (leer lassen fuer Standard) |
| `priority` | Prioritaet | `10` (primaer) oder `50` (Fallback) |

## Schritt 4: Backend anlegen

Der Agent ruft `ki.backends.create` mit den gesammelten Daten auf. Bei lokalen Backends werden `category` und `provider` automatisch gesetzt. Bei Cloud-Backends wird `category=cloud` verwendet.

Beispiel im Admin-Chat:

```
Admin: Erstelle ein lokales LLM-Backend mit Ollama auf http://gpu-server.local:11434
       und dem Modell qwen3:8b.

KIara: [Tool: ki.backends.create]
       Backend "Ollama qwen3:8b" wurde erfolgreich angelegt (ID: 1).
       Kategorie: LLM, Provider: Ollama, Prioritaet: 10.
```

## Schritt 5: Backend testen

Nach dem Anlegen wird `ki.backends.test` aufgerufen, um die Verbindung zu verifizieren:

- **Ollama:** HTTP-Erreichbarkeit + Modell-Liste abfragen
- **Anthropic/OpenAI:** API-Key validieren + Modell-Liste
- **Embedding:** Probe-Embedding generieren und Dimension pruefen

Bei Fehlern gibt der Agent Hinweise zur Fehlerbehebung:

| Fehler | Ursache | Loesung |
|--------|---------|---------|
| "Connection refused" | Ollama nicht gestartet | `systemctl status ollama` pruefen |
| "401 Unauthorized" | API-Key ungueltig | Key pruefen, ggf. neu generieren |
| "Model not found" | Modell nicht heruntergeladen | `ollama pull <modell>` ausfuehren |

## Schritt 6: Embedding-Backend einrichten (falls noetig)

Neben dem LLM wird ein Embedding-Backend benoetigt. Die Schritte sind identisch, aber mit `category=embedding`:

```
Admin: Erstelle auch ein Embedding-Backend mit nomic-embed-text auf demselben
       Ollama-Server.

KIara: [Tool: ki.backends.create]
       Embedding-Backend "Ollama nomic-embed-text" angelegt (ID: 2).
       Vektordimension: 768. Collection: kb_nomic-embed-text.
```

## Schritt 7: Aktive Backends ueberpruefen

`ki.backends.list` zeigt alle konfigurierten Backends mit Status:

```
| ID | Name              | Kategorie | Provider | Modell            | Prio | Status |
|----|-------------------|-----------|----------|-------------------|------|--------|
| 1  | Ollama qwen3:8b   | LLM       | Ollama   | qwen3:8b          | 10   | OK     |
| 2  | Ollama nomic-embed| Embedding | Ollama   | nomic-embed-text  | 10   | OK     |
```

## Verwandte Funktionen

- `ki.backends.list` -- Alle Backends anzeigen
- `ki.backends.categories` -- Verfuegbare Backend-Kategorien
- `ki.backends.update` -- Backend-Konfiguration aendern
- `ki.backends.delete` -- Backend entfernen
- `ki.backends.test` -- Verbindungstest durchfuehren

## Naechste Schritte

Nach der Backend-Einrichtung:

1. **Datenquelle einrichten** -- Siehe Workflow "Datenquelle einrichten"
2. **Indexierung starten** -- Siehe Workflow "Indexierung einrichten"
3. **Anonymisierung konfigurieren** -- Im Compliance-Tab, falls Cloud-Backends genutzt werden
