# Kiara

**Souveraene Unternehmens-KI**

Kiara ist eine KI-Plattform, die Unternehmen in eine befragbare Entitaet verwandelt. Sie verbindet alle relevanten Datenquellen — ERP, Fileserver, DMS, Mail, Ticketsystem — mit KI-gestuetzter Fragenbeantwortung, wahlweise ueber einen lokalen LLM (Ollama) oder DSGVO-konform anonymisiert ueber Cloud-KI-Dienste (Anthropic Claude, OpenAI GPT).

Kiara positioniert sich nicht als generisches RAG-Framework, sondern als fertige Plattform mit integrierter Anonymisierung, Berechtigungssteuerung und Audit-Trail. Die Ingestion laeuft vollstaendig lokal, Cloud-Anfragen werden anonymisiert — volle Datensouveraenitaet ohne Vendor-Lock-in.

---

## Features

### RAG-Kern
- **25+ Dateiformate:** DOCX, PDF, XLSX, PPTX, ODT, Markdown, Python, PowerShell, JSON, XML, CSV und weitere
- **Inkrementelle Indizierung:** Nur geaenderte Dateien werden neu verarbeitet (MD5-Hash auf Pfad+Groesse+mtime)
- **Multi-Embedding-Modelle:** Pro Embedding-Modell eigene VectorDB-Collection, Modellauswahl bei Ingestion
- **SSE-Streaming:** Antworten werden Token fuer Token in Echtzeit gestreamt
- **Quellenangaben:** Jede Antwort verweist auf die genutzten Quelldokumente mit Relevanz-Score

### Anonymisierung
- **Aho-Corasick-Engine:** Deterministische Erkennung aller ERP-Stammdaten und LDAP-Mitarbeiterdaten in einem Durchlauf (<1ms pro Chunk)
- **Regex-Auffangnetz:** IBAN, E-Mail-Adressen, Telefonnummern, Steuernummern, interne IP-Adressen
- **Reversible Pseudonymisierung:** Forward- und Reverse-Mapping ausschliesslich im RAM, nach Session-Ende unwiederbringlich verloren
- **Content-Klassen:** Pro Datenquelle konfigurierbare Risikoklasse (BLOCKED / HIGH / MEDIUM / LOW)
- **Benutzerdefinierte Muster:** Eigene Regex-Regeln und Stammdaten-Eintraege ueber die Admin-UI hinzufuegbar

### Cloud-KI-Router
- **Hybrides Routing:** Automatische oder manuelle Entscheidung lokal vs. Cloud pro Anfrage
- **Keyword-basiertes Auto-Routing:** Strategische Fragen (Analyse, Optimierung, Prognose) werden an leistungsfaehigere Cloud-Modelle delegiert
- **Provider:** Anthropic (Claude) und OpenAI (GPT) mit konfigurierbarem Modell, Temperatur und Token-Limit
- **Automatischer Fallback:** Bei Cloud-Fehler wird transparent auf den lokalen LLM zurueckgeschaltet
- **Anonymisierungs-Pipeline:** Query und Kontext werden vor dem Cloud-Request anonymisiert, die Antwort danach de-anonymisiert

### Audit und Compliance
- **Audit-Log:** Jeder Cloud-Request wird protokolliert (Zeitpunkt, Benutzer, Entitaetenzahl, Provider, Modell)
- **Datenschutz:** Originaltext der Frage wird nie im Audit-Log gespeichert (nur SHA256-Hash)
- **Nachvollziehbarkeit:** DSB kann im Admin-Dashboard einsehen, welche Anfragen die Cloud erreicht haben

### Berechtigungen
- **LDAP/Active Directory:** Gruppensync, Login, automatischer Benutzerimport
- **3-Tabellen-Berechtigungsmodell:** Benutzer -> Gruppen -> Datenquellen -> Chunks
- **Chunk-Level-Filter:** Bei RAG-Queries werden nur Chunks aus erlaubten Datenquellen zurueckgegeben

### Multi-Datenquellen
- **Lokale Verzeichnisse, SMB-Shares, WebDAV** (z.B. Nextcloud)
- **Per-Datenquelle-Konfiguration:** Chunk-Groesse, Dateitypen, Ausschlussmuster, Sync-Intervall
- **Live-Fortschrittsanzeige:** Dateien verarbeitet/gesamt + Durchsatz bei Ingestion

### Administration
- **Zwei getrennte Web-Portale:** User-Chat (Port 8501) + Admin-Dashboard (Port 8502)
- **KI-Backend-Konfiguration:** Embedding- und LLM-Server getrennt konfigurierbar mit Verbindungstest
- **E-Mail-Benachrichtigungen:** Ollama-Ausfall, Datenquellen-Probleme, Indizierungsfehler
- **OnlyOffice-Integration:** Dokumentbearbeitung direkt im Browser

---

## Quick Start

### Voraussetzungen

- Debian/Ubuntu-Server mit Python 3.12+
- Qdrant-Server; bei aktiviertem ColBERT mindestens **1.13.0** fuer den
  [`has_vector`-Filter](https://qdrant.tech/documentation/search/filtering/#has-vector)
- Ollama-Server mit einem LLM (z.B. `qwen3:8b`) und einem Embedding-Modell (z.B. `nomic-embed-text`)
- Optional: LDAP/Active Directory fuer Benutzerverwaltung
- Optional: Anthropic- oder OpenAI-API-Key fuer Cloud-KI

Der Python-Client benoetigt mindestens `qdrant-client>=1.13.0`, damit
`HasVectorCondition` importiert werden kann. Die Lockdateien pinnen den Client
auf `1.17.1`. Die Version eines separat betriebenen Qdrant-Servers muss die
oben genannte Anforderung ebenfalls erfuellen.

### Installation via Debian-Paket

```bash
# Paket bauen (auf dem Entwicklungsrechner)
make deb

# Paket installieren
sudo dpkg -i kiara_*.deb

# Fehlende Abhaengigkeiten nachinstallieren
sudo apt-get install -f
```

Das `postinst`-Script erstellt automatisch:
- Systembenutzer `kiara`
- Python-venv in `/opt/kiara/venv/`
- Alle Python-Dependencies
- systemd-Units (`kiara.service`, `kiara-admin.service`)

### Konfiguration

Die Hauptkonfiguration liegt in `/etc/kiara/kiara.yaml` und bleibt bei Upgrades erhalten:

```yaml
# Ollama-Backend
ollama:
  url: "http://gpu-server.local:11434"
  llm_model: "qwen3:8b"
  embedding_model: "nomic-embed-text"

# Vektor-Datenbank
vectordb:
  path: "/var/lib/kiara/chromadb"
  collection: "knowledgebase"

# Ingestion
ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  batch_size: 100

# Web-Server
web:
  host: "0.0.0.0"
  port: 8501
  admin_port: 8502

# Cloud-KI (optional)
cloud:
  enabled: false
  provider: "anthropic"       # oder "openai"
  api_key: ""
  model: "claude-sonnet-4-6"
  mode: "auto"                # auto | local | cloud
  max_tokens: 4096
  temperature: 0.3

# Anonymisierung
anonymization:
  enabled: true
  word_boundaries: true
  min_term_length: 3
  max_content_class: 3        # 0=BLOCKED, 1=HIGH, 2=MEDIUM, 3=LOW
  audit_log: true
  erp_source:
    enabled: false
    type: "csv"
    path: "/etc/kiara/stammdaten"
  ldap_source:
    enabled: true

# PostgreSQL-Datenbank
database:
  host: "localhost"
  port: 5432
  name: "kiara"
```

### Erster Start

```bash
# Services starten
sudo systemctl start kiara kiara-admin

# Status pruefen
sudo systemctl status kiara kiara-admin

# Logs verfolgen
sudo journalctl -u kiara -f
```

Danach ist das System erreichbar unter:
- **User-Portal:** `http://<server>:8501` (Standard-Login: `admin` / `admin`)
- **Admin-Dashboard:** `http://<server>:8502`

### Erste Schritte im Admin-Dashboard

1. **Passwort aendern:** Sofort das Standard-Passwort des Admin-Benutzers aendern
2. **KI-Backend konfigurieren:** Im KI-Tab die Ollama-URL und Modelle pruefen (Verbindungstest)
3. **Datenquelle anlegen:** Im Datenquellen-Tab eine lokale oder SMB/WebDAV-Quelle hinzufuegen
4. **Indizierung starten:** Datenquelle auswaehlen und "Indizieren" klicken
5. **Optional -- LDAP:** Im LDAP-Tab die Verbindung konfigurieren und Gruppen importieren
6. **Optional -- Cloud-KI:** Im KI-Tab Cloud-Provider aktivieren und API-Key hinterlegen

---

## Architektur

Kiara ist als geschichtete Plattform aufgebaut. Eine ausfuehrliche Architektur-Dokumentation mit Schichtendiagramm, Datenfluss, Modul-Uebersicht, API-Referenz und Sicherheitsmodell findet sich in:

**[docs/architecture.md](docs/architecture.md)**

Das Betriebs- und Diagnoseverfahren fuer den versionierten KIron/KIara-
Embedding-Vertrag steht in **[docs/embedding-contract-v1-operations.md](docs/embedding-contract-v1-operations.md)**.

### Kurzuebersicht

```
+-----------------------------------------------------------+
|   Praesentationsschicht (FastAPI + HTML/JS SPA)           |
|   User-Portal :8501  |  Admin-Dashboard :8502             |
+-----------------------------------------------------------+
|   Sicherheitsschicht (LDAP, Sessions, Berechtigungen)     |
+-----------------------------------------------------------+
|   Anonymisierungsschicht (Aho-Corasick + Regex)           |
+-----------------------------------------------------------+
|   Cloud-KI-Router (Lokal/Cloud, Anonymisierung)           |
+-----------------------------------------------------------+
|   RAG-Kern (Ingestion, Qdrant, Ollama-LLM)               |
+-----------------------------------------------------------+
|   Datenhaltung (Qdrant, PostgreSQL, YAML-Config)           |
+-----------------------------------------------------------+
```

### Projekt-Struktur

```
kiara/
|-- anonymization/     # Anonymisierungsmodul (Aho-Corasick, Regex, Audit)
|-- generation/        # RAG-Chain, LLM-Client (Ollama + Cloud)
|-- ingestion/         # Datei-Loader, Chunker, Pipeline
|-- retrieval/         # VectorStore-Factory, Retriever
`-- web/               # FastAPI-Apps, Auth, API-Router, Templates, Static
```

---

## Konfigurationsreferenz

### Abschnitt `ollama`

| Parameter         | Standard                      | Beschreibung                                          |
|-------------------|-------------------------------|-------------------------------------------------------|
| `url`             | `http://gpu-server.local:11434`     | Ollama-Server-URL (Fallback fuer llm_url/embedding_url) |
| `llm_url`         | (leer, nutzt `url`)          | Separate URL fuer LLM-Backend                         |
| `embedding_url`   | (leer, nutzt `url`)          | Separate URL fuer Embedding-Backend                   |
| `llm_model`       | `qwen3:8b`                   | LLM-Modellname auf dem Ollama-Server                  |
| `embedding_model` | `nomic-embed-text`            | Embedding-Modellname auf dem Ollama-Server             |
| `prompt_template` | (leer, nutzt Standard)        | Benutzerdefiniertes Prompt-Template (muss `{context}` und `{question}` enthalten) |

### Abschnitt `vectordb`

| Parameter    | Standard                       | Beschreibung                          |
|--------------|--------------------------------|---------------------------------------|
| `path`       | `/var/lib/kiara/chromadb`    | Speicherort der VectorDB-Daten        |
| `collection` | `knowledgebase`                | Name der Standard-Collection          |

### Abschnitt `ingestion`

| Parameter       | Standard | Beschreibung                              |
|-----------------|----------|-------------------------------------------|
| `chunk_size`    | 1000     | Maximale Zeichenzahl pro Chunk            |
| `chunk_overlap` | 200      | Ueberlappung zwischen Chunks in Zeichen   |
| `batch_size`    | 100      | Anzahl Chunks pro VectorDB-Batch          |

### Abschnitt `cloud`

| Parameter     | Standard           | Beschreibung                                        |
|---------------|--------------------|-----------------------------------------------------|
| `enabled`     | `false`            | Cloud-KI aktivieren                                 |
| `provider`    | `anthropic`        | Cloud-Provider (`anthropic` oder `openai`)          |
| `api_key`     | (leer)             | API-Schluessel fuer den Cloud-Provider              |
| `model`       | `claude-sonnet-4-6`| Cloud-Modellname                                    |
| `base_url`    | (leer)             | Benutzerdefinierte Base-URL (z.B. fuer Azure OpenAI)|
| `mode`        | `auto`             | Routing-Modus: `auto`, `local` oder `cloud`         |
| `max_tokens`  | 4096               | Maximale Token-Anzahl fuer Cloud-Antworten          |
| `temperature` | 0.3                | Sampling-Temperatur fuer Cloud-Modell               |

### Abschnitt `anonymization`

| Parameter            | Standard                    | Beschreibung                                           |
|----------------------|-----------------------------|--------------------------------------------------------|
| `enabled`            | `true`                      | Anonymisierung aktivieren                              |
| `word_boundaries`    | `true`                      | Nur an Wortgrenzen matchen                             |
| `min_term_length`    | 3                           | Minimale Laenge fuer Stammdaten-Treffer                |
| `max_content_class`  | 3                           | Maximale Content-Klasse fuer Cloud-Upload (0-3)        |
| `audit_log`          | `true`                      | Audit-Logging fuer Cloud-Requests aktivieren           |
| `erp_source.enabled` | `false`                     | ERP-Stammdaten-Import aktivieren                       |
| `erp_source.type`    | `csv`                       | Importformat (`csv`)                                   |
| `erp_source.path`    | `/etc/kiara/stammdaten`   | Pfad zu CSV/JSON-Stammdaten                            |
| `ldap_source.enabled`| `true`                      | LDAP-Mitarbeiterdaten fuer Anonymisierung nutzen       |
| `excluded_terms`     | `[]`                        | Begriffe, die nicht anonymisiert werden sollen         |
| `custom_patterns`    | `[]`                        | Benutzerdefinierte Regex-Muster                        |
| `custom_entries`     | `[]`                        | Benutzerdefinierte Stammdaten-Eintraege                |

### Abschnitt `web`

| Parameter    | Standard   | Beschreibung                     |
|--------------|------------|----------------------------------|
| `host`       | `0.0.0.0`  | Bind-Adresse des Web-Servers     |
| `port`       | 8501       | Port fuer das User-Portal        |
| `admin_port` | 8502       | Port fuer das Admin-Dashboard    |

### Abschnitt `database`

| Parameter | Standard                        | Beschreibung                  |
|-----------|---------------------------------|-------------------------------|
| `host`    | `localhost`                     | PostgreSQL-Host               |
| `port`    | `5432`                          | PostgreSQL-Port               |
| `name`    | `kiara`                         | PostgreSQL-Datenbankname      |

---

## CLI

```bash
kiara index          # Wissensbasis indizieren
kiara query "Frage"  # RAG-Query im Terminal
kiara status         # System-Status anzeigen
kiara serve          # Web-Server starten
kiara config         # Konfiguration anzeigen/bearbeiten
```

`kiara index` ohne `--datasource-id` verwendet den Pfad aus der Konfiguration
und die persistierte Datenquelle `0` (`CLI (Konfiguration)`) für
Chunk-Zuordnungen, Datei-Hashes und Statistiken. Diese Datenquelle wird nicht
automatisch synchronisiert. Für administrierte Datenquellen dient
`kiara index --datasource-id <ID>`.

---

## Technologie-Stack

| Komponente       | Technologie                                    |
|------------------|------------------------------------------------|
| Sprache          | Python 3.12                                    |
| Web-Framework    | FastAPI + Uvicorn                              |
| Vektor-DB        | Qdrant (Docker, localhost:6333)                |
| LLM (lokal)      | Ollama (qwen3:8b)                              |
| LLM (Cloud)      | Anthropic Claude, OpenAI GPT                   |
| Embeddings       | Ollama (nomic-embed-text)                      |
| Anonymisierung   | pyahocorasick (Aho-Corasick-Automat)           |
| Authentifizierung| bcrypt + itsdangerous (signed cookies)         |
| Datenbank        | PostgreSQL (via psycopg2)                      |
| LDAP             | python-ldap (Active Directory)                 |
| Deployment       | Debian-Paket (.deb) + systemd                  |
| Text-Splitter    | LangChain Text Splitters                       |

---

## Lizenz

Dieses Projekt steht unter der [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-or-later).
