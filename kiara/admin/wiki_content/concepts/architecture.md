# KIara Architektur

## Ueberblick

KIara ist eine Unternehmens-KI-Plattform, die den gesamten relevanten Datenbestand einer Organisation einer KI zugaenglich macht. Die Architektur basiert auf vier Saeulen: Hybrid-Retrieval, ReAct-Agent, Dokumentenverarbeitung und Anonymisierung. LLM-Backends sind austauschbar: lokales Ollama, Anthropic (Claude), OpenAI (GPT).

## Hybrid-Retrieval (BM25 + Vektor + GraphRAG)

Die Suche kombiniert drei Verfahren fuer maximale Trefferqualitaet:

- **Vektor-Suche** (Qdrant + Embedding-Modelle) -- Semantische Aehnlichkeit: Findet Dokumente, die thematisch zur Anfrage passen, auch wenn keine exakten Schlagwoerter uebereinstimmen. Embedding-Modelle (z.B. `nomic-embed-text`) uebersetzen Text in hochdimensionale Vektoren.
- **BM25** (rank_bm25 oder Elasticsearch) -- Keyword-basierte Suche: Stark bei exakten Begriffen, Fachbegriffen und Nummern (Auftragsnummern, Produktbezeichnungen). BM25 findet Treffer, die der Vektor-Suche entgehen.
- **GraphRAG** -- Wissensgraph-Traversal: Der Graph speichert Entitaeten (Kunden, Produkte, Mitarbeiter) und ihre Beziehungen. Traversal (BFS, Distanz-Scoring) findet verknuepfte Informationen, die weder BM25 noch Vektor-Suche erkennen.

Die Ergebnisse werden per **Reciprocal Rank Fusion (RRF)** kombiniert. Der RRF-Algorithmus gewichtet die Rangfolge aus allen Quellen und erstellt eine einheitliche Ergebnisliste. Optional wird diese anschliessend durch einen **Cross-Encoder** (sentence-transformers) reranked, der jedes Ergebnis paarweise mit der Anfrage vergleicht und nach tatsaechlicher Relevanz sortiert.

### Retrieval-Pipeline

```
Benutzer-Anfrage
    |
    +-- [Query-Expansion] --> Alternative Formulierungen (optional)
    |
    +-- Vektor-Suche (Qdrant) ------+
    |                                |
    +-- BM25/Elasticsearch ----------+--> RRF-Fusion --> Reranking --> Top-K Chunks
    |                                |
    +-- Graph-Traversal -------------+
    |
    +--> Chunks + Kontext --> LLM --> Antwort
```

## ReAct-Agent mit Tool-Use

Der Agent folgt dem ReAct-Pattern (Reasoning + Acting):

1. **Observe:** LLM erhaelt die Benutzer-Anfrage, Chat-Historie und verfuegbare Tools
2. **Think:** LLM analysiert den Kontext und entscheidet, ob ein Tool-Aufruf noetig ist
3. **Act:** Tool wird serverseitig ausgefuehrt (z.B. `search_documents`, `erp_order_status`)
4. **Observe:** Tool-Ergebnis wird dem LLM zurueckgefuehrt
5. **Repeat:** Schleife bis zur finalen Antwort (maximale Iterationen begrenzt)

Die Tool-Adapter (`adapters/ollama.py`, `adapters/anthropic.py`, `adapters/openai.py`) uebersetzen zwischen dem einheitlichen KIara-Tool-Schema und den provider-spezifischen Formaten. Ollama verwendet Regex-basiertes Parsing, Anthropic und OpenAI nutzen native Tool-Use-APIs.

### Verfuegbare Tool-Kategorien

| Kategorie | Tools | Beschreibung |
|-----------|-------|-------------|
| **Builtin** | `search_documents`, `calculate` | Grundfunktionen fuer alle Benutzer |
| **ERP** | `erp_order_status`, `erp_stock_level`, `erp_customer_info`, `erp_production_status` | Live-Abfragen im ERP-System |
| **Action** | `generate_status_mail`, `generate_checklist` | Dokumenten-Generierung |
| **Workflow** | `summarize_document`, `generate_from_template`, `export_document` | Template-basierte Workflows |

## Dokumentenverarbeitung (Ingestion-Pipeline)

KIara verarbeitet 25+ Dateiformate ueber mehrere Loader-Stufen:

1. **Docling (DDU)** -- Deep Document Understanding: Konvertiert PDF, Office-Dokumente und Bilder ueber ein Deep-Learning-Modell zu Markdown. Erkennt komplexe Layouts (Tabellen, Spalten, Grafiken). Laeuft als separater Service (Docling-Proxy, Port 5001).
2. **VLM** -- Vision Language Model: Beschreibt Bilder (PNG, JPG, 20+ Formate) als Text. Der generierte Text wird indexiert, sodass Bilder durchsuchbar werden.
3. **CAD-Parser** -- Verarbeitet technische Zeichnungen (DXF, STEP, IGES).
4. **Standard-Loader** -- Fallback-Textextraktion fuer alle unterstuetzten Formate.

Die **Chunk-Strategie** (auto/prose/code/email/tabular) bestimmt, wie Dokumente in indexierbare Abschnitte zerlegt werden. Der `auto`-Modus erkennt den Dokumenttyp automatisch und waehlt die optimale Strategie.

## Anonymisierung

Die Anonymisierung schuetzt sensible Daten in 4 Schichten:

| Schicht | Methode | Performance | Erkennungsrate |
|---------|---------|-------------|---------------|
| 1 | **Aho-Corasick Automaton** | <10ms fuer 50k+ Entities | Hoch (exakte Treffer) |
| 2 | **LDAP/AD-Stammdaten** | <1ms (In-Memory) | Hoch (bekannte Mitarbeiter) |
| 3 | **Regex-Muster** | <1ms | Hoch (strukturierte Daten: IBAN, E-Mail, Telefon) |
| 4 | **LLM-Erkennung** | 1-5s | Mittel (unbekannte Entitaeten) |

Erkannte Entitaeten werden durch **deterministische Pseudonyme** ersetzt (`[KUNDE_001]`, `[IBAN_002]`), die innerhalb einer Anfrage konsistent bleiben. Bei Cloud-Backends wird eine **Content-Klassifizierung** vorgeschaltet, die bestimmt, welche Inhalte ueberhaupt an die Cloud gesendet werden duerfen.

## Konnektoren

8 Datenquellen-Konnektoren integrieren verschiedene Unternehmenssysteme:

- **Dateisystem** (lokal) -- Direkter Zugriff auf Verzeichnisse des KIara-Servers
- **SMB** -- Windows-Dateifreigaben mit Domain-Authentifizierung
- **WebDAV** -- Nextcloud, ownCloud, SharePoint
- **IMAP** -- E-Mail-Postfaecher mit SSL und Ordner-Auswahl
- **Exchange** -- Microsoft 365 via OAuth2 (App-Registrierung)
- **SQL** -- PostgreSQL, MSSQL, MySQL mit konfigurierbarem Text-Template

Jeder Konnektor implementiert `ConnectorBase` mit Schema-Definition, Scan-Methoden und Progress-Reporting. Passwort-Felder werden AES-verschluesselt in PostgreSQL gespeichert.

## 4-Schichten-Datenmodell

Jeder Chunk existiert in bis zu 4 voneinander abhaengigen Schichten:

| # | Schicht | Speicherort | Zweck |
|---|---------|-------------|-------|
| 1 | **Qdrant** | Collection `kb_<modell>` | Vektor-Embeddings + Payload (Text, Metadaten, ds_ids) |
| 2 | **chunk_datasources** | PostgreSQL | n:m-Zuordnung Chunk zu Datenquelle (Source of Truth fuer Zugehoerigkeit) |
| 3 | **file_hashes** | PostgreSQL | Hash pro Datei+DS+Modell fuer inkrementelles Indexing |
| 4 | **datasource_embeddings** | PostgreSQL | Aggregierte Statistik (chunk_count, file_count) pro DS+Modell |

Bei Loeschungen muessen **alle 4 Schichten** konsistent bereinigt werden. Die Integritaetspruefung erkennt Abweichungen und bietet Reparatur-Aktionen an.

## Multi-Modell VectorDB-Architektur

KIara unterstuetzt mehrere Embedding-Modelle parallel. Jedes Modell bekommt eine eigene Qdrant-Collection (`kb_<modellname>`). Der `VectorStore` oeffnet automatisch die Collection des aktuell konfigurierten Modells. Fuer systemweite Abfragen (Statistiken, Admin-Dashboards) wird `list_collections()` verwendet, das alle `kb_*`-Collections mit Chunk-Counts liefert.

## LLM-Backends

Drei Backend-Typen sind austauschbar konfigurierbar:

| Backend | Provider | Vorteil | Einschraenkung |
|---------|---------|---------|---------------|
| **Lokal** | Ollama | Datenschutz, keine Kosten, keine Abhaengigkeit | Benoetigt GPU, begrenzte Modellgroesse |
| **Cloud (Anthropic)** | Claude | Hohe Qualitaet, natives Tool-Use | Kosten pro Token, Internet-Abhaengigkeit |
| **Cloud (OpenAI)** | GPT | Breite Modellauswahl | Kosten pro Token, Internet-Abhaengigkeit |

Die Prioritaets-basierte Auswahl ermoeglicht Fallback-Konfigurationen: Wenn das primaere Backend (z.B. lokales Ollama) nicht erreichbar ist, wird automatisch auf das Cloud-Backend mit der naechstniedrigeren Prioritaet ausgewichen.
