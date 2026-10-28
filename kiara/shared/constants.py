"""Benannte Konstanten fuer KIara.

Zentrale Stelle fuer alle Schwellwerte, Sentinel-Werte und
konfigurationsunabhaengige Defaults, die in der Codebasis
verwendet werden. Verhindert Magic Numbers und dokumentiert
die Bedeutung jedes Wertes.

Beziehungen:
    - kiara.config referenziert diese Werte in DEFAULT_CONFIG
    - kiara.cli nutzt ADMIN_PORT_DEFAULT, WORKER_LOCK_PATH
    - kiara.generation.* nutzen EXCERPT_LENGTH, CHARS_PER_TOKEN, Timeout-Konstanten
    - kiara.ingestion.* nutzen CONTENT_DETECT_*, IMAP_*, HASH_*, BACKEND_WAIT_*
    - kiara.retrieval.* nutzen QDRANT_*, BM25_*, ES_*, KEYWORD_OVERFETCH_FACTOR
    - kiara.web.* nutzen RATE_LIMIT_*, CLOUD_*, CHAT_*, SEARCH_*, LOG_*
    - kiara.cloud.client nutzt CLOUD_*_TIMEOUT, ANTHROPIC_API_VERSION
    - kiara.backend_selector nutzt BACKEND_CACHE_TTL, BACKEND_HEALTH_TTL
    - kiara.backend_warmup nutzt WARMUP_* Konstanten
    - kiara.health.checks_services nutzt DEFAULT_OLLAMA_URL als Fallback
    - kiara.graph.extractor nutzt DEFAULT_OLLAMA_URL als Fallback
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Netzwerk-Defaults (Fallback-URLs fuer nicht konfigurierte Services)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
"""Standard-URL fuer den Ollama-Server (lokal).

Wird als Fallback verwendet, wenn weder ``ollama.llm_url`` noch
``ollama.url`` in der Config gesetzt sind.
"""

DEFAULT_QDRANT_URL: str = "http://localhost:6333"
"""Standard-URL fuer den Qdrant-Vektordatenbank-Server."""

DEFAULT_ELASTICSEARCH_URL: str = "http://localhost:9200"
"""Standard-URL fuer den Elasticsearch-Server (Hybrid-Suche)."""

# ---------------------------------------------------------------------------
# Netzwerk-Timeouts
# ---------------------------------------------------------------------------

CLOUD_CONNECT_TIMEOUT_SECS: float = 60.0
"""HTTP Connect-Timeout fuer Cloud-KI-APIs (OpenAI/Anthropic)."""

CLOUD_READ_TIMEOUT_SECS: float = 120.0
"""HTTP Read-Timeout fuer Cloud-KI-APIs (OpenAI/Anthropic)."""

CLOUD_MAX_RETRIES: int = 3
"""Maximale Anzahl Retry-Versuche bei transienten Cloud-API-Fehlern."""

OLLAMA_CONNECT_TIMEOUT_SECS: float = 10.0
"""HTTP Connect-Timeout fuer lokales Ollama."""

OLLAMA_READ_TIMEOUT_SECS: float = 600.0
"""HTTP Read-Timeout fuer Ollama (10 min fuer lange Generierungen bei grossem num_ctx)."""

OLLAMA_WRITE_TIMEOUT_SECS: float = 30.0
"""HTTP Write-Timeout fuer Ollama-Requests."""

OLLAMA_POOL_TIMEOUT_SECS: float = 10.0
"""Connection-Pool-Timeout fuer Ollama."""

INDEX_CONTRACT_DISCOVERY_TIMEOUT_SECS_DEFAULT: float = 10.0
"""Timeout je vollstaendiger KIron-Contract-Discovery-Anfrage."""

INDEX_CONTRACT_DISCOVERY_ATTEMPTS_DEFAULT: int = 3
"""Versuche fuer transiente ``/api/tags``-/``/api/show``-Ausfaelle."""

INDEX_CONTRACT_DISCOVERY_BACKOFF_INITIAL_SECS_DEFAULT: float = 0.25
"""Initialer Backoff zwischen transient fehlgeschlagenen Discoveries."""

INDEX_CONTRACT_DISCOVERY_BACKOFF_MAX_SECS_DEFAULT: float = 2.0
"""Obergrenze fuer den exponentiellen Discovery-Backoff."""

INDEX_CONTRACT_RECOVERY_POLL_SECS_DEFAULT: float = 5.0
"""Worker-Pollintervall waehrend einer nicht verfuegbaren Vertragspruefung."""

ANTHROPIC_API_VERSION: str = "2023-06-01"
"""Anthropic API-Version (messages-Endpoint)."""

# ---------------------------------------------------------------------------
# LLM & Generation
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS: int = 4096
"""Maximale Token-Anzahl fuer Cloud-KI-Antworten."""

DEFAULT_MAX_CONTEXT_TOKENS: int = 6000
"""Maximale Token-Anzahl fuer Kontext in Workflow-Zusammenfassungen."""

DEFAULT_SUMMARY_CHUNK_SIZE: int = 4000
"""Chunk-Groesse fuer Map-Reduce-Zusammenfassungen (Zeichen)."""

DEFAULT_LLM_TEMPERATURE: float = 0.3
"""Standard-Temperatur fuer LLM-Anfragen."""

DEFAULT_LLM_NUM_CTX: int = 8192
"""Standard-Kontextfenster-Groesse fuer lokale LLM-Anfragen (Ollama)."""

DEFAULT_CLOUD_MODEL: str = "claude-sonnet-4-6"
"""Standard-Modell fuer Cloud-KI-Anfragen (Anthropic)."""

DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
"""Standard-Modell fuer OpenAI-Cloud-KI-Anfragen."""

DEFAULT_CLOUD_PROVIDER: str = "openai"
"""Standard-Provider fuer Cloud-KI (openai oder anthropic)."""

DEFAULT_LLM_MODEL: str = "qwen3:8b"
"""Standard-Modell fuer lokale LLM-Anfragen (Ollama)."""

DEFAULT_EMBEDDING_MODEL: str = "nomic-embed-text"
"""Standard-Modell fuer Embedding-Anfragen (Ollama)."""

DEFAULT_VECTOR_SIZE: int = 768
"""Standard-Vektorgroesse fuer Embeddings (Dimensionen)."""

KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    # Ollama
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Voyage AI
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
}

DEFAULT_MAX_CONTEXT_CHUNKS: int = 80
"""Maximale Anzahl Chunks, die an das LLM als Kontext uebergeben werden.

Schuetzt vor Kontextfenster-Ueberlauf bei mehrfacher Suche (Agent-Modus)
oder hohen k-Werten (Standard-Modus). Chunks sind nach Relevanz sortiert,
die relevantesten werden behalten.
"""

DEFAULT_QUERY_EXPANSION_NUM_CTX: int = 2048
"""Kontextfenster-Groesse fuer Query-Expansion (kleiner als LLM-Kontext)."""

QUERY_EXPANSION_TIMEOUT_SECS: float = 10.0
"""Per-Call Timeout fuer Query-Expansion LLM-Aufrufe (Sekunden).

Query-Expansion ist eine Optimierung, kein kritischer Pfad — bei haengendem
Ollama soll der Search-Request nicht minutenlang blockiert werden.
"""

DEFAULT_REWRITER_NUM_CTX: int = 2048
"""Kontextfenster-Groesse fuer den Query-Rewriter-Modus (kleines Fenster, nur Suchbegriff)."""

DEFAULT_AGENT_NUM_CTX: int = 16384
"""Kontextfenster-Groesse fuer den Agent-Modus (groesser wegen Tool-Loop)."""

DEFAULT_AGENT_MAX_ITERATIONS: int = 15
"""Maximale Anzahl Tool-Iterationen im Agent-Modus."""

AGENT_MAX_ITERATIONS_OLLAMA: int = 15
"""Max-Iterationen fuer Ollama-Agent (lokales LLM, mehr Schritte noetig)."""

AGENT_MAX_ITERATIONS_CLOUD: int = 8
"""Max-Iterationen fuer Cloud-Agent (staerkere Modelle, weniger Schritte)."""

AGENT_RESULT_FLOOR_CHARS: int = 500
"""Minimale Zeichenlaenge fuer Tool-Ergebnisse im Agent-Loop.

Auch wenn das Gesamt-Budget erschoepft ist, werden Ergebnisse
nie kuerzer als dieser Floor angezeigt — verhindert nutzlos
gekuerzte Ergebnisse.
"""

AGENT_CONTEXT_BUDGET_CHARS: int = 32000
"""Gesamtbudget fuer aufbereitete Dokumentinhalte pro Agent-Anfrage."""

AGENT_CONTEXT_RESULT_CHARS: int = 24000
"""Maximale Laenge eines strukturerhaltend aufbereiteten Dokument-Ergebnisses."""

AGENT_CONTEXT_READ_RESERVE_CHARS: int = 8000
"""Innerhalb des Kontextbudgets fuer gezieltes Dokumentlesen reservierter Platz."""

AGENT_DOCUMENT_READ_MAX_CHUNKS: int = 8
"""Maximale Zahl gespeicherter Chunks pro read_document-Aufruf."""

AGENT_RESULT_TRUNCATE_OLLAMA: int = 4000
"""Maximale Zeichenlaenge fuer Tool-Ergebnisse im Ollama-Agent-Kontext.

Bei 16k Token num_ctx (~56k Zeichen) und max 15 Iterationen muss Platz
fuer System-Prompt, bisherige Ergebnisse und Generierung bleiben.
4000 Zeichen erlaubt ~7-8 vollstaendige Suchergebnisse pro Iteration.
"""

AGENT_RESULT_TRUNCATE_CLOUD: int = 24000
"""Maximale Zeichenlaenge fuer einzelne Tool-Ergebnisse im Cloud-Agent-Kontext.

Cloud-Modelle haben 128k+ Token — 24000 Zeichen ~ 6000 Tokens.
Wird ggf. durch das Gesamt-Budget weiter begrenzt.
"""

AGENT_TOTAL_RESULT_BUDGET: int = 96000
"""Gesamt-Zeichenbudget fuer alle Tool-Ergebnisse im Agent-Loop.

96000 Zeichen ~ 24000 Tokens. Bei 128k-Kontext bleiben >100k Tokens
fuer System-Prompt, History, Reasoning und Antwort.
Wenn das Budget erschoepft ist, werden neue Ergebnisse staerker gekuerzt.
"""

AGENT_MAX_SAME_TOOL_CALLS: int = 5
"""Default-Limit fuer identische Tool-Aufrufe im Agent-Loop.

Wenn ein Tool N-mal mit verschiedenen Parametern aufgerufen wurde,
wird es als Loop erkannt und uebersprungen.
"""

AGENT_MAX_PARALLEL_TOOL_WORKERS: int = 4
"""Maximale Anzahl paralleler Tool-Worker-Threads im Agent.

Jeder Thread startet Retriever + VectorDB-Query + BM25 + ggf. Reranking.
4 ist konservativ: 2-3 parallele Calls sind typisch, >4 extrem selten.
"""

DEFAULT_UNCHAINED_TEMPERATURE: float = 0.7
"""Standard-Temperatur fuer den Unchained-Modus (kreativer als RAG)."""

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 1000
"""Standard-Chunk-Groesse fuer Text-Splitting (Zeichen)."""

DEFAULT_CHUNK_OVERLAP: int = 200
"""Standard-Ueberlappung zwischen aufeinanderfolgenden Chunks (Zeichen)."""

CHARS_PER_TOKEN: float = 3.5
"""Konservative Schaetzung Zeichen-pro-Token fuer mehrsprachige Modelle (qwen3).

Wird fuer Token-Schaetzungen verwendet, wenn kein exakter Tokenizer
verfuegbar ist (z.B. bei Cloud-Backends oder Kontext-Overflow-Pruefung).
"""

DEFAULT_VLM_PROMPT: str = (
    "Beschreibe den Inhalt dieses Bildes detailliert auf Deutsch. "
    "Wenn es sich um eine technische Zeichnung handelt, nenne Masse, Werkstoffe, "
    "Artikelnummern und Fertigungshinweise. "
    "Wenn es sich um ein Typenschild handelt, nenne alle ablesbaren Daten. "
    "Wenn es sich um ein Foto handelt, beschreibe was zu sehen ist."
)
"""Standard-Prompt fuer VLM-Bildbeschreibung bei der Indexierung.

Kann ueber ``ingestion.vlm_prompt`` in der Config ueberschrieben werden.
"""

CONTENT_DETECT_CHUNK_SIZE: int = 8192
"""Chunk-Groesse beim Lesen fuer Content-Type-Detection (Bytes)."""

BINARY_CONTENT_THRESHOLD: float = 0.05
"""Anteil Non-Printable-Zeichen ab dem eine Datei als binaer gilt (5%).

Experimentell optimiert: niedrigere Werte erzeugen False-Negatives
(Binaer wird als Text erkannt), hoehere Werte False-Positives
(Text mit Encoding-Problemen wird faelschlich als binaer erkannt).
"""

DDU_DEFAULT_CONCURRENCY: int = 2
"""Standard-Concurrency-Limit fuer DDU-GPU (docling-serve).

Begrenzt parallele GPU-Requests pro Python-Prozess via Semaphore.
"""

DDU_DEFAULT_OCR_LANGUAGES: tuple[str, ...] = ("de", "en")
"""Standard-Sprachen fuer OCR-Erkennung im DDU-Backend.

Wird als Default in convert_document() verwendet. Deutsch und Englisch decken
den typischen Unternehmenskontext ab.
"""

DDU_DEFAULT_TABLE_MODE: str = "accurate"
"""Standard-Tabellen-Erkennungsmodus fuer DDU (docling-serve).

'accurate' liefert bessere Ergebnisse, ist aber langsamer als 'fast'.
"""

HASH_CHECKPOINT_INTERVAL: int = 50
"""Batch-Groesse fuer File-Hash-Checkpoints waehrend der Ingestion.

Nach jeweils N verarbeiteten Dateien werden die Hashes persistent
gespeichert, um bei Abbruch nicht alle Hashes zu verlieren.
"""

BACKEND_WAIT_INTERVAL_SECS: int = 30
"""Sekunden zwischen Health-Checks in _wait_for_backend().

Balance zwischen schneller Erkennung und Server-Last.
"""

IMAP_MAX_FILENAME_LENGTH: int = 100
"""Maximale Laenge fuer E-Mail-Anhang-Dateinamen (Truncation)."""

IMAP_FETCH_BATCH_SIZE: int = 50
"""Batch-Groesse fuer IMAP FETCH-Operationen (UIDs pro Roundtrip)."""

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".cs",
        ".vb",
        ".ps1",
        ".py",
        ".pl",
        ".js",
        ".ts",
        ".bat",
        ".sh",
        ".css",
        ".html",
        ".sln",
        ".java",
        ".rs",
        ".go",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
    }
)
"""Dateierweiterungen, die als Quellcode behandelt werden.

Bestimmt Chunk-Strategie (code-aware Splitting), Chunk-Groesse
und Syntax-Annotation in Loaders.
"""

ACCEPT_ALL_BLOCKLIST_DEFAULT: frozenset[str] = frozenset(
    {
        # Kryptographisch / Signaturen / Zertifikate
        ".p7s",
        ".p7m",
        ".p7b",
        ".p7c",
        ".pem",
        ".cer",
        ".crt",
        ".der",
        ".key",
        ".sig",
        ".asc",
        ".gpg",
        ".pgp",
        # Archive / komprimiert
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".tgz",
        # Binaer-Anhang-Formate (E-Mail)
        ".dat",
        ".winmail",
        ".dvbvkey",
        # Kalender / vCard
        ".ics",
        ".vcf",
        ".vcs",
        # Bilder (ohne VLM nicht sinnvoll als Text)
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".ico",
        ".webp",
        ".svg",
        ".heic",
        ".heif",
        # Audio / Video
        ".mp3",
        ".wav",
        ".ogg",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".flac",
        # Ausfuehrbare / Binaer
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".msi",
        ".deb",
        ".rpm",
        # Datenbanken / Lock / Temp
        ".db",
        ".sqlite",
        ".lock",
        ".tmp",
        ".bak",
        ".swp",
        # Schriftarten
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
    }
)
"""Dateitypen, die auch bei accept_all_text nie indexiert werden.

Kryptographische Formate, Archive, Binaer-Anhang-Formate, Kalender etc.
Wird als Default-Fallback in der Ingestion-Pipeline verwendet und als
Default-Wert in DEFAULT_CONFIG["ingestion"]["accept_all_blocklist"].
"""

# ---------------------------------------------------------------------------
# Web-Server Ports
# ---------------------------------------------------------------------------

WEB_PORT_DEFAULT: int = 8501
"""Standard-Port fuer die Chat-UI (FastAPI, Port 8501)."""

ADMIN_PORT_DEFAULT: int = 8502
"""Standard-Port fuer das Admin-Dashboard (FastAPI).

Wird als Fallback verwendet, wenn ``web.admin_port`` nicht in der
Config gesetzt ist.
"""

ADMIN_CHAT_PORT_DEFAULT: int = 8503
"""Standard-Port fuer den separaten Admin-Chat-Service."""

MCP_PORT_DEFAULT: int = 8504
"""Standard-Port fuer den MCP-Server (Model Context Protocol)."""

# ---------------------------------------------------------------------------
# Protokoll-Ports (IANA-Standards)
# ---------------------------------------------------------------------------

IMAP_PORT_SSL: int = 993
"""Standard-Port fuer IMAP ueber SSL/TLS (IANA IMAPS, RFC 3501)."""

SMTP_PORT_STARTTLS: int = 587
"""Standard-Port fuer SMTP mit STARTTLS (IANA Submission, RFC 2587)."""

# ---------------------------------------------------------------------------
# Mail-Connector-Defaults
# ---------------------------------------------------------------------------

IMAP_DEFAULT_SINCE_DAYS: int = 90
"""Standard-Zeitraum fuer IMAP-Abruf (nur Mails der letzten N Tage)."""

MAIL_ARCHIVE_MAX_BODY_BYTES_DEFAULT: int = 25 * 1024 * 1024
"""Obergrenze des ausgewählten Originalbodys je archivierter Mail."""

IMAP_DEFAULT_MAX_ATTACHMENT_MB: int = 25
"""Standard-Obergrenze fuer Anhangsgroesse in MB bei IMAP-Abruf."""

SCAN_CACHE_MAX_AGE_SECS: int = 86400
"""Maximales Alter des Mail-Scan-Caches in Sekunden (24 Stunden).

Nach dieser Zeit wird der Cache als veraltet betrachtet und neu aufgebaut.
"""

# ---------------------------------------------------------------------------
# UI / Darstellung
# ---------------------------------------------------------------------------

EXCERPT_LENGTH: int = 200
"""Maximale Laenge fuer Text-Auszuege (Excerpts) in der Quellenliste.

Wird in RAGChain._build_sources(), SearchDocumentsTool.execute() und
DocumentSummarizer.summarize_multiple() konsistent verwendet.
"""

MAX_AUDIT_TEXT: int = 102_400
"""Maximale Zeichenzahl fuer full_prompt/full_answer in Audit-Logs (100 KB).

Verhindert uebergrosse DB-Eintraege bei langen Agent-Konversationen.
"""

CHAT_HISTORY_LIMIT: int = 10
"""Standard-Anzahl Chat-Messages im Konversationskontext."""

CHAT_MSG_EXCERPT_CHARS: int = 500
"""Maximale Zeichenlaenge fuer Chat-Message-Auszuege im History-Kontext."""

SEARCH_MAX_PAGE_SIZE: int = 1000
"""Maximale Seitengroesse fuer Search-API-Ergebnisse."""

ES_MAX_RESULT_WINDOW: int = 10_000
"""Elasticsearch max_result_window (from + size darf diesen Wert nicht ueberschreiten)."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE_MAX_BYTES: int = 5_000_000
"""Maximale Log-Datei-Groesse vor Rotation (5 MB)."""

LOG_FILE_BACKUP_COUNT: int = 3
"""Anzahl rotierter Backup-Log-Dateien."""

# ---------------------------------------------------------------------------
# Rate-Limiting (Login)
# ---------------------------------------------------------------------------

RATE_LIMIT_THRESHOLD: int = 3
"""Fehlgeschlagene Login-Versuche bis zur Blockierung."""

RATE_LIMIT_BASE_DELAY_SECS: int = 10
"""Basis-Verzoegerung nach Erreichen des Thresholds (Sekunden).

Rate-Limit-Algorithmus:
  wait = BASE_DELAY + (Versuche - THRESHOLD) * INCREMENT
  3 Fehlversuche → 10s, 4 → 15s, 5 → 20s (progressive Backoff).
"""

RATE_LIMIT_INCREMENT_SECS: int = 5
"""Zusaetzliche Sekunden pro weiterem Fehlversuch ueber dem Threshold."""

# ---------------------------------------------------------------------------
# Retrieval & Ranking
# ---------------------------------------------------------------------------

KEYWORD_OVERFETCH_FACTOR: int = 4
"""Overfetch-Multiplikator fuer BM25- und Elasticsearch-Suche.

Holt mehr Kandidaten als angefordert, um nach DS-Filterung und
Source-Filtering noch genuegend Ergebnisse zu haben.
Beispiel: k=10 → 40 Kandidaten werden geholt, dann gefiltert.
"""

QDRANT_UPSERT_BATCH_SIZE: int = 256
"""Batch-Groesse fuer Qdrant-Upsert-Operationen.

Optimiert fuer GPU-Memory bei Nomic-Embed-Text Embeddings.
Groessere Batches bringen kaum Geschwindigkeitsgewinn,
koennen aber den Memory-Verbrauch sprengen.
"""

QDRANT_DELETE_BATCH_SIZE: int = 500
"""Batch-Groesse fuer Qdrant-Delete/Scroll-Operationen.

Hoeher als Upsert, da keine Embeddings berechnet werden.
"""

QDRANT_SCROLL_BATCH_SIZE: int = 1000
"""Batch-Groesse fuer Qdrant-Scroll-Operationen (Iteration ueber Collection).

Wird in allen scroll()-Aufrufen als Default-Limit verwendet, sofern kein
expliziter batch_size-Parameter uebergeben wird.
"""

BM25_CACHE_TTL_SECS: int = 3600
"""TTL fuer den BM25-Index-Cache (1 Stunde).

Bei 629k+ Chunks dauert der Index-Rebuild ~90 Sekunden.
5-Minuten-TTL wuerde zu haeufige Rebuilds erzeugen.
1 Stunde balanciert Staleness gegen Performance.
Bei Aenderungen wird der Cache via invalidate_bm25_cache() sofort invalidiert.
"""

BM25_MAX_CORPUS_DOCS: int = 1_000_000
"""Maximaler Korpus-Umfang (Anzahl Chunks) fuer den BM25-Index-Build.

BM25Okapi (rank_bm25) haelt das gesamte tokenisierte Korpus inklusive
Original-Texte und Metadaten im Prozess-Speicher. Pro Chunk fallen ca.
3-5 KB an (Token-Liste + Strings + BM25-State), bei Millionen-Korpora
kann der Worker-Memory dadurch reproduzierbar ausgeschoepft werden.

Bei Ueberschreiten dieses Limits wird der BM25-Build fail-closed
uebersprungen: _bm25_search() liefert leere Resultate, die Hybrid-RRF
faellt auf reine Vektor-/Graph-Suche zurueck. Fuer grosse Collections
sollte 'elasticsearch' als retrieval.hybrid_search-Modus konfiguriert
werden — ES haelt den Index extern und skaliert horizontal.

Konfigurierbar via retrieval.bm25_max_corpus_docs.
0 = unbegrenzt (Power-User, deaktiviert den Schutz vollstaendig).
"""

ES_BULK_CHUNK_SIZE: int = 1000
"""Batch-Groesse fuer Elasticsearch Bulk-Write-Operationen."""

ES_TERMS_CHUNK_SIZE: int = 1024
"""Batch-Groesse fuer ES terms-Queries (file_paths-Listen).

Elasticsearch expandiert `terms` intern in `bool/should`-Klauseln. Das Default-
Limit `indices.query.bool.max_clause_count` liegt je nach Version bei 1024-4096,
zusaetzlich greift `index.max_terms_count` (Default 65536). Bei sehr grossen
Datenquellen wuerde eine ungesplittete Liste mit `too_many_clauses` brechen.
1024 funktioniert konservativ auf allen ES-Versionen.
"""

ES_PROGRESS_CALLBACK_INTERVAL: int = 5000
"""Frequenz fuer Progress-Callbacks bei ES-Rebuild (alle N Dokumente)."""

ES_VERIFY_SAMPLE_SIZE: int = 500
"""Stichprobengroesse fuer ES-Integritaetspruefung (verify_integrity)."""

PARENT_DOC_MAX_CHUNKS: int = 30
"""Maximale Chunk-Anzahl eines Dokuments fuer Parent-Document-Retrieval.

Wenn ein Chunk eines Dokuments mit <= N Chunks als relevant eingestuft
wird, werden ALLE Chunks dieses Dokuments nachgeladen. Dies implementiert
das 'Parent Document Retrieval'-Pattern fuer kleine Dokumente.
"""

BACKEND_CACHE_TTL_SECS: int = 30
"""TTL fuer den Backend-Status-Cache (Sekunden).

Balance zwischen Server-Last (seltenes Polling) und Freshness
(Aenderungen propagieren innerhalb von 30s).
"""

BACKEND_HEALTH_TTL_SECS: float = 30.0
"""TTL fuer Health-Check-Ergebnisse pro Backend-URL (Sekunden).

Schuetzt vor parallelen Health-Checks derselben URL (Thundering-Herd).
"""

BACKEND_LOG_RATE_LIMIT_SECS: float = 300.0
"""Mindestabstand zwischen WARNING-Logs fuer dieselbe nicht erreichbare Backend-URL.

Verhindert Log-Spam bei Cluster-Ausfall: ohne Rate-Limit wuerde pro
Anfrage und nicht erreichbarem Backend eine Warning entstehen
(z.B. 100 RPS x 10 Down-Backends = 1000 WARN/sec). Mit 5min Abstand
bleibt jede ausgefallene URL sichtbar, ohne den Logger zu fluten.
"""

RRF_K_DEFAULT: int = 60
"""Reciprocal Rank Fusion k-Parameter (Default).

Steuert die Gewichtung der Rang-Position in der RRF-Formel:
``score = 1.0 / (k + rank)``.  Hoehere Werte glaetten den
Einfluss der Position.
"""

COLBERT_DEFAULT_MODEL: str = "colbert-xm"
"""Standard-Modell fuer ColBERT Token-Level-Embeddings (Multi-Vector).

antoinelouis/colbert-xm: multilinguales ColBERT-Modell (dim=128),
explizit auf Deutsch trainiert, sentence-transformers-kompatibel.
"""

NO_ACCESS_SENTINEL: int = -1
"""Sentinel-Wert fuer Datasource-Filter bei leerem Zugriff.

Wird als ``datasource_id``-Filter verwendet, wenn der Benutzer auf
keine Datenquelle Zugriff hat.  Da keine reale ``datasource_id``
jemals -1 ist, liefert der Filter garantiert 0 Treffer.
"""

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

GRAPH_EXTRACTION_LIMIT: int = 200
"""Maximale Anzahl bestehender Graph-Knoten, die bei der
Entity-Extraktion geladen werden (Deduplizierungs-Check).
"""

WORKER_LOCK_PATH: str = "/var/lib/kiara/worker.lock"
"""Pfad fuer das flock()-basierte Lock-File des Ingestion-Workers.

Stellt sicher, dass nur ein Worker-Prozess gleichzeitig laeuft
(Singleton-Schutz im ``serve-worker`` CLI-Command).
"""

HEALTH_BACKGROUND_LOCK_PATH: str = "/var/lib/kiara/health-background.lock"
"""Pfad fuer das flock()-basierte Lock-File des Health-Background-Loops.

Stellt sicher, dass bei mehreren Web-Worker-Prozessen nur ein Prozess den
periodischen Degradation-Check ausfuehrt.
"""

CRON_LOCK_PATH: str = "/var/lib/kiara/cron.lock"
"""Pfad fuer das flock()-basierte Lock-File der Cron-Schedule-Writes.

Synchronisiert write_schedule/remove_schedule-Calls prozess-uebergreifend
(Gunicorn/uvicorn mit Workers>1 — threading.Lock reicht nicht).
"""

DAVFS_SECRETS_LOCK_PATH: str = "/var/lib/kiara/davfs-secrets.lock"
"""Pfad fuer das flock()-basierte Lock-File der /etc/davfs2/secrets-Writes.

Serialisiert _ensure_davfs_secret/_remove_davfs_secret prozess-uebergreifend.
Ohne dieses Lock koennen parallele WebDAV-Mounts unterschiedlicher
Datenquellen (kiara/kiara-admin/kiara-worker) per Read-Modify-Write die
secrets-Datei ueberschreiben (Lost-Update-Race).
"""

INGESTION_WORKER_POLL_INTERVAL: int = 2
"""Sekunden zwischen Queue-Polls im Ingestion-Worker."""


INGESTION_STALL_POLL_INTERVAL_SECS_DEFAULT: float = 5.0
"""Polling-Intervall fuer die diagnostische Ingestion-D-State-Erkennung."""


INGESTION_D_STATE_WARNING_SECS_DEFAULT: float = 30.0
"""Dauer eines durchgehenden D-State bis zum ersten Diagnose-Event."""


INGESTION_D_STATE_TIMEOUT_SECS_DEFAULT: float = 120.0
"""Dauer eines bestaetigten CIFS-D-State-Haengers.

120 Sekunden entsprechen zugleich dem Linux-Default fuer die Kernel-
Hung-Task-Warnung. Ein fehlender Datei- oder Chunk-Fortschritt allein reicht
nicht aus; die Klassifikation prueft zusaetzlich den nativen Worker-Thread.
"""


INGESTION_HEARTBEAT_TIMEOUT: int = 300
"""Heartbeat-Timeout in Sekunden (5 Minuten).

Ein laufender Job gilt als tot, wenn sein Heartbeat aelter als
dieser Wert ist. Wird in is_job_alive() geprueft.
"""

INGESTION_STARTUP_GRACE: int = 60
"""Grace-Period in Sekunden nach Job-Start ohne Heartbeat.

Frisch gestartete Jobs haben noch keinen Heartbeat — innerhalb dieser
Frist gelten sie trotzdem als lebendig.
"""

INGESTION_BATCH_SIZE: int = 500
"""Max IN-clause Size fuer Batch-Queries (Hash-Store, Chunk-DS)."""

INGESTION_DATASOURCE_LOCK_NAMESPACE: int = 681030
"""PostgreSQL-Advisory-Lock-Namespace fuer DS-Job-State-Transitions."""

INGESTION_MAX_CONSECUTIVE_ERRORS: int = 10
"""Circuit-Breaker: maximale Anzahl aufeinanderfolgender Fehler in der
Ingestion-Pipeline, bevor der Job automatisch abgebrochen wird.

Schuetzt vor sinnlosem Weiterlaufen, wenn ein Backend (Embedding-Server,
DDU) ausgefallen ist. Einzelne transiente Fehler (< Schwellwert) werden
toleriert; bei einem systematischen Ausfall (>= Schwellwert) bricht die
Pipeline sofort mit RuntimeError ab.
"""

# ---------------------------------------------------------------------------
# Contextual Retrieval (Phase 16)
# ---------------------------------------------------------------------------

ENRICHMENT_MAX_CONTEXT_CHARS: int = 6000
"""Maximale Gesamtlaenge des Dokumentkontexts fuer den Enrichment-Prompt (Zeichen)."""

ENRICHMENT_HEADER_CHARS: int = 2000
"""Zeichen vom Dokumentanfang (Header) die in den Kontext einfliessen."""

ENRICHMENT_NEIGHBOR_CHARS: int = 500
"""Maximale Zeichen pro Nachbar-Chunk im Enrichment-Kontext."""

ENRICHMENT_MAX_PREFIX_CHARS: int = 500
"""Maximale Laenge des generierten Kontext-Prefix (Zeichen). Wird abgeschnitten."""

ENRICHMENT_MIN_CHUNK_CHARS: int = 50
"""Chunks unter dieser Zeichenlaenge werden nicht enriched (zu kurz fuer Kontext).
Pruefung ist strict less-than: Chunks mit exakt 50 Zeichen werden enriched."""

# ---------------------------------------------------------------------------
# Connector-Timeouts
# ---------------------------------------------------------------------------

CONNECTOR_CONNECT_TIMEOUT: int = 30
"""Timeout in Sekunden fuer den Verbindungsaufbau bei Connectoren (IMAP,
Exchange, SQL).  Schuetzt gegen Endlos-Haenger bei Firewall-Drops oder
nicht erreichbaren Servern."""

SQL_STATEMENT_TIMEOUT: int = 30
"""Timeout in Sekunden fuer einzelne SQL-Statements.

Dieser Wert wird treiber-/serverseitig gesetzt, damit blockierende DB-API-
Aufrufe nicht nach einem asyncio-Timeout im Hintergrund weiterlaufen.
"""

SQL_QUERY_MAX_ROWS_DEFAULT: int = 10000
"""Standard-Obergrenze fuer SQL-Abfrageergebnisse in execute_query().

Schuetzt gegen versehentliche Full-Table-Scans bei grossen Tabellen.
"""

CONNECTOR_XMLRPC_TIMEOUT: int = 30
"""Timeout in Sekunden fuer XML-RPC-Aufrufe (Odoo ERP-Konnektor)."""

ERP_ORDERS_DEFAULT_LIMIT: int = 10
"""Standardanzahl fuer das begrenzte Agent-Tool ``erp_orders``."""

ERP_ORDERS_MAX_LIMIT: int = 50
"""Harte Obergrenze der an ein LLM ausgegebenen Auftragsdatensaetze."""

ERP_ORDERS_MAX_OFFSET: int = 1000
"""Maximal erlaubter Offset fuer paginierte Agent-Abfragen."""

ERP_ORDERS_MAX_FILTERS: int = 8
"""Maximale Anzahl strukturierter Filter pro ``erp_orders``-Aufruf."""

ERP_ORDERS_MAX_FIELDS: int = 12
"""Maximale Anzahl projizierter Auftragsfelder pro Datensatz."""

ERP_ORDERS_MAX_IN_VALUES: int = 20
"""Maximale Wertanzahl eines ``in``-Filters."""

ERP_ORDERS_MAX_FILTER_TEXT: int = 200
"""Maximale Laenge eines einzelnen textuellen Filterwerts."""

ERP_ORDERS_MAX_SCAN: int = 2000
"""Harte Odoo-Kandidatenobergrenze fuer abgeleitete Sortierungen/Filter."""

# ---------------------------------------------------------------------------
# Backend-Warmup
# ---------------------------------------------------------------------------

WARMUP_TIMEOUT_DEFAULT: int = 300
"""Timeout in Sekunden fuer einzelne Warmup-Requests (Modell-Laden kann
bei kaltem Start 30-120s dauern, 5 Min deckt auch langsame Hardware ab)."""

WARMUP_TIMEOUT_MULTIPLIER: int = 2
"""Multiplikator fuer den Gesamt-Timeout bei parallelem Warmup.

Worst-Case pro Task mit 1 Retry: 1. Versuch Client-Timeout + Delay +
2. Versuch Client-Timeout = 2 * timeout + 2s.  Der Gesamt-Timeout
wird als ``WARMUP_TIMEOUT_MULTIPLIER * timeout + WARMUP_TIMEOUT_OFFSET``
berechnet, um diesen Worst-Case sicher abzudecken.
"""

WARMUP_TIMEOUT_OFFSET: int = 10
"""Fester Puffer (Sekunden) auf den Gesamt-Warmup-Timeout.

Zusammen mit WARMUP_TIMEOUT_MULTIPLIER ergibt sich:
``overall_timeout = 2 * timeout + 10`` (bei Default-Werten).
"""

WARMUP_RETRY_DELAY: int = 2
"""Sekunden Pause zwischen Retry-Versuchen bei transienten Fehlern."""

WARMUP_TRANSIENT_RETRIES: int = 1
"""Anzahl Retries bei transienten Fehlern (Netzwerk, HTTP 5xx)."""

WARMUP_DUMMY_TEXT: str = "warmup"
"""Dummy-Text fuer Embedding-Warmup (minimal, nur zum Modell-Laden)."""

WARMUP_KEEP_ALIVE: str = "2h"
"""Ollama keep_alive fuer Warmup-Requests — haelt Modelle waehrend der
Sync-/Hashing-Phase im VRAM (laenger als Ollamas Default von 5 Minuten)."""

# ---------------------------------------------------------------------------
# Cloud-Routing & Budget
# ---------------------------------------------------------------------------

CLOUD_MONTHLY_BUDGET_EUR: float = 50.0
"""Standard-Monatsbudget fuer Cloud-KI-Anfragen (Euro)."""

CLOUD_BUDGET_ALERT_THRESHOLD_PCT: int = 80
"""Alert-Schwelle in Prozent des Monatsbudgets."""

CLOUD_ROUTING_CONFIDENCE_THRESHOLD: float = 0.3
"""Confidence-Score-Schwelle fuer Auto-Routing zu Cloud-KI.

Wenn der lokale Retriever weniger als diesen Score liefert,
wird die Anfrage an die Cloud weitergeleitet (im Auto-Modus).
"""

CLOUD_ROUTING_MIN_CHUNKS: int = 2
"""Minimale Anzahl relevanter Chunks fuer Cloud-Routing-Entscheidung."""

# ---------------------------------------------------------------------------
# Config / Credentials
# ---------------------------------------------------------------------------

PASSWORD_MASK: str = "********"
"""Sentinel-Wert fuer maskierte Passwort-Felder in API-Responses.

Wird in GET-Endpunkten als Platzhalter fuer gespeicherte Passwoerter
zurueckgegeben. Beim Speichern (POST/PUT) signalisiert dieser Wert,
dass das bestehende Passwort beibehalten werden soll.
Verwendet in: config_utils, api_ldap, api_mail, api_onlyoffice,
api_datasource, api_tools.
"""

# ---------------------------------------------------------------------------
# Health-Checks & Disk
# ---------------------------------------------------------------------------

DISK_WARN_BYTES: int = 1024**3
"""Schwellwert fuer Disk-Health 'degraded' (1 GB frei)."""

DISK_CRIT_BYTES: int = 100 * 1024**2
"""Schwellwert fuer Disk-Health 'unhealthy' (100 MB frei)."""

MAX_WALK_FILE_COUNT: int = 10_000
"""Obergrenze fuer os.walk-basierte Datei-Zaehlung in Datenquellen."""

# ---------------------------------------------------------------------------
# Anonymisierung
# ---------------------------------------------------------------------------

ANON_LLM_TIMEOUT_SECS: int = 10
"""Timeout fuer LLM-basierte Anomalie-Erkennung in der Anonymisierung."""

ANON_LLM_CACHE_SIZE: int = 256
"""Cache-Groesse fuer Anonymisierungs-LLM-Detector-Ergebnisse."""

ANON_LLM_NUM_CTX: int = 4096
"""Token-Fenster fuer das Anonymisierungs-LLM."""

# ---------------------------------------------------------------------------
# Text-Verarbeitung / Tokenisierung
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset(
    {
        # German
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "und",
        "oder",
        "aber",
        "ist",
        "sind",
        "war",
        "hat",
        "haben",
        "wird",
        "werden",
        "kann",
        "können",
        "mit",
        "von",
        "für",
        "auf",
        "in",
        "an",
        "zu",
        "bei",
        "nach",
        "aus",
        "über",
        "wie",
        "was",
        "wer",
        "den",
        "dem",
        "des",
        "im",
        "am",
        "es",
        "ich",
        "du",
        "er",
        "sie",
        "wir",
        "ihr",
        "nicht",
        "auch",
        "noch",
        "nur",
        "als",
        "wenn",
        "dann",
        "so",
        "da",
        "schon",
        "man",
        "sich",
        # English
        "the",
        "a",
        "and",
        "or",
        "but",
        "is",
        "are",
        "has",
        "have",
        "will",
        "be",
        "been",
        "can",
        "do",
        "does",
        "did",
        "with",
        "from",
        "for",
        "on",
        "at",
        "to",
        "by",
        "of",
        "it",
        "its",
        "this",
        "that",
        "not",
        "also",
        "how",
        "what",
        "who",
        "which",
    }
)
"""Deutsch/Englische Stoppwoerter fuer BM25 und Graph-Tokenisierung."""

CLOUD_KEYWORDS: frozenset[str] = frozenset(
    {
        "strateg",
        "analyse",
        "empfehl",
        "optimier",
        "einspar",
        "vergleich",
        "bewert",
        "prognos",
        "trend",
        "markt",
        "potential",
        "risiko",
        "chance",
        "swot",
        "benchmark",
        "korrelat",
        "zusammenhang",
        "ursache",
        "auswirkung",
    }
)
"""Keywords, die im Auto-Routing-Modus eine Cloud-Anfrage ausloesen.

Werden als Substring-Match gegen die (lowercased) Benutzerfrage geprueft.
Beispiel: "strategisch" matched wegen "strateg".
"""

# ---------------------------------------------------------------------------
# MCP-Server
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# System-Monitor
# ---------------------------------------------------------------------------

METRICS_COLLECT_INTERVAL_SECS: int = 1
"""Live-Sampling-Intervall (1s Takt fuer SSE-Stream)."""

METRICS_PERSIST_INTERVAL_SECS: int = 60
"""DB-Write-Intervall (1 Zeile pro Minute in system_metrics)."""

METRICS_DOWNSAMPLE_HOUR: int = 3
"""Uhrzeit (Stunde) fuer taegliches Downsampling (03:00 Uhr)."""

METRICS_RAW_RETENTION_HOURS: int = 24
"""Raw-Daten (1/min) werden nach N Stunden zu 5min-AVG komprimiert."""

METRICS_5MIN_RETENTION_DAYS: int = 30
"""5min-Aggregate werden nach N Tagen zu 1h-AVG komprimiert."""

METRICS_1H_RETENTION_DAYS: int = 365
"""1h-Aggregate werden nach N Tagen geloescht."""

METRICS_MAX_DATAPOINTS: int = 500
"""Maximale Datenpunkte pro Chart-Abfrage (Ausdünnung bei Ueberschreitung)."""

# Tracing (Phase 17)
TRACING_DEFAULT_RETENTION_DAYS = 7
TRACING_BATCH_SCHEDULE_MS = 2000  # BatchSpanProcessor Schedule-Delay
TRACING_SERVICE_NAME = "kiara"

# ---------------------------------------------------------------------------
# MCP-Server
# ---------------------------------------------------------------------------

MCP_MAX_QUESTION_LENGTH: int = 10_000
"""Maximale Zeichenlaenge fuer MCP-Fragen (Validierung im query-Tool)."""

MCP_MAX_K: int = 20
"""Maximale Anzahl Kontext-Chunks fuer MCP-Anfragen (Clamping auf 1..N)."""

# ---------------------------------------------------------------------------
# Agent-Loop
# ---------------------------------------------------------------------------

AGENT_RETRY_SLEEP_SECONDS: int = 2
"""Wartezeit in Sekunden vor einem Retry bei transientem LLM-Fehler im Agent."""

AGENT_TOOL_MAX_RETRIES: int = 3
"""Maximale Anzahl Ausfuehrungs-Versuche pro Tool-Call im Agent-Loop."""

AGENT_TOOL_RETRY_DELAY_SECS: int = 3
"""Wartezeit in Sekunden zwischen Tool-Execution-Retries."""

# ---------------------------------------------------------------------------
# Cloud-Client
# ---------------------------------------------------------------------------

CLOUD_PING_MAX_TOKENS: int = 10
"""Max-Tokens fuer den Cloud-Verbindungstest (Ping). Minimaler Wert,
da nur die Erreichbarkeit geprueft wird."""

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

BACKUP_TIMESTAMP_FORMAT: str = "%Y%m%d-%H%M%S"
"""Zeitstempel-Format fuer Backup-Dateinamen."""

# ---------------------------------------------------------------------------
# Session & Auth
# ---------------------------------------------------------------------------

ADMIN_TOOL_TEST_SESSION_TTL_SECS: int = 1800
ADMIN_TOOL_TEST_MAX_SESSIONS: int = 128

SESSION_MAX_AGE_SECONDS: int = 365 * 86400
"""Maximale Session-Lebensdauer in Sekunden (1 Jahr).

Bewusst lang gewaehlt: KIara laeuft im Unternehmensnetzwerk, Admins sollen
sich nicht taeglich neu anmelden muessen. Session-Invalidierung bei
Benutzer-Deaktivierung ist serverseitig implementiert (UserRepository).
"""

# ---------------------------------------------------------------------------
# Security-Logging
# ---------------------------------------------------------------------------

LOG_MAX_BYTES: int = 10 * 1024 * 1024
"""Maximale Groesse der Security-Log-Datei vor Rotation (10 MB)."""

# ---------------------------------------------------------------------------
# Cache-Refresh (Admin-UI)
# ---------------------------------------------------------------------------

CACHE_THROTTLE_SECONDS: float = 3.0
"""Minimaler Abstand in Sekunden zwischen Cache-Refreshes im Worker.

Drosselt Progress-Callback-getriebene Refreshes, die pro Datei feuern.
"""

# ---------------------------------------------------------------------------
# Config-Defaults (fuer DEFAULT_CONFIG in config.py)
# ---------------------------------------------------------------------------

INGESTION_DEFAULT_BATCH_SIZE: int = 100
"""Standard-Batch-Groesse fuer die Ingestion-Pipeline (Dateien pro Batch)."""

RERANK_TOP_K_DEFAULT: int = 5
"""Standard-Anzahl Ergebnisse nach Reranking."""

RERANK_FETCH_MULTIPLIER_DEFAULT: int = 3
"""Multiplikator fuer Kandidaten-Fetch vor Reranking (top_k * N)."""

RERANK_BATCH_SIZE_DEFAULT: int = 32
"""Standard-Batchgroesse fuer Cross-Encoder-Predict.

Bei OOM (CUDA/CPU) wird zur Laufzeit halbiert, bis batch_size=1.
"""

RERANK_BACKEND_DEFAULT: str = "local"
"""Standard-Reranker-Backend: "local" (in-process CrossEncoder) oder "kiron" (Remote-HTTP)."""

RERANK_TIMEOUT_SECS_DEFAULT: int = 30
"""HTTP-Timeout fuer Remote-Reranking.

Der Kiron-Rerank-Service laedt Modelle lazy beim ersten Request; ein
Cold-Start (Modell-Load auf GPU) kann >10s dauern und muss ins Timeout passen.
"""

RERANK_FAIL_MODE_DEFAULT: str = "degrade"
"""Verhalten bei Remote-Rerank-Fehlern: "degrade" (RRF-Reihenfolge behalten) oder "error"."""

RERANK_REMOTE_MAX_DOCUMENTS: int = 256
"""API-Limit des Kiron-Rerank-Endpoints: maximale Kandidatenzahl pro Request."""

RERANK_REMOTE_MAX_DOC_CHARS: int = 50_000
"""API-Limit des Kiron-Rerank-Endpoints: maximale Zeichen pro Dokument."""

ABSTAIN_MIN_RERANK_SCORE_DEFAULT: float = 0.02
"""Abstain-Score-Gate: Liegt der beste Reranker-Score darunter, wird ohne
LLM-Call mit der Abstain-Antwort geantwortet.

Kalibrierung 2026-07-07 (Golden v1 + CRAG Full136): Der Cross-Encoder-Score
misst topische Relevanz, nicht Beantwortbarkeit — hohe Schwellen opfern
korrekte Antworten. Konservativ als "gar nichts Relevantes gefunden"-Fang.
"""

ABSTAIN_MESSAGE_DEFAULT: str = (
    "Dazu finde ich keine Informationen in der Wissensbasis. "
    "Die Frage kann aus den vorhandenen Quellen nicht beantwortet werden."
)
"""Abstain-Antworttext.

Enthaelt bewusst die Marker "keine Information" und "nicht beantwortet",
damit die Benchmark-Auswertungen (crag._MISSING_PATTERNS, golden) die
Antwort als missing statt incorrect werten.
"""

ABSTAIN_ANSWER_MARKERS: tuple[str, ...] = (
    "keine informationen in der wissensbasis",
    "keine information",
    "nicht beantwortet werden",
    "ich weiss es nicht",
    "ich weiß es nicht",
)
"""Marker zur Laufzeit-Erkennung einer Abstain-Antwort (lowercase-Substring)."""

GROUNDING_MODEL_DEFAULT: str = "mdeberta-v3-xnli"
"""NLI-Modell fuer den Grounding-Check (kiron-deberta /api/score, multilingual)."""

GROUNDING_MIN_ENTAILMENT_DEFAULT: float = 0.5
"""Mindest-Entailment-Wahrscheinlichkeit, damit eine Retry-Antwort als vom
Kontext gedeckt gilt (max ueber die geprueften Chunks)."""

GROUNDING_MAX_CHUNKS_DEFAULT: int = 5
"""Wieviele Kontext-Chunks als NLI-Praemissen geprueft werden."""

GROUNDING_PREMISE_MAX_CHARS: int = 1500
"""Zeichenkappung pro Praemisse (NLI-Modelle trunkieren bei ~512 Tokens)."""

GROUNDING_HYPOTHESIS_MAX_CHARS: int = 1000
"""Zeichenkappung der Antwort-Hypothese fuer den NLI-Check."""

GROUNDING_TIMEOUT_SECS_DEFAULT: int = 30
"""HTTP-Timeout fuer den Grounding-Score-Call (Cold-Load des NLI-Modells)."""

ABSTAIN_INSTRUCTION_MAX_SCORE_DEFAULT: float = 0.0
"""Score-Band fuer die konditionale Abstain-Instruktion (Ampel-Prinzip).

0.0 = Instruktion immer einblenden (wenn abstain_instruction aktiv).
>0  = Instruktion nur einblenden, wenn der beste Reranker-Score darunter
liegt — starke Treffer behalten den cleanen Prompt (Golden-Messung:
pauschale Instruktion machte qwen3:8b uebervorsichtig, 27->18 correct),
schwache Kontexte bekommen den Abstain-Ausweg (CRAG: halbierte
Halluzinationsrate). Ohne verfuegbare Scores (Reranker aus/degraded)
gilt die Bedingung konservativ als erfuellt.
"""

ABSTAIN_INSTRUCTION_TEXT: str = (
    "Wenn der Kontext die Frage nicht beantwortet, antworte genau: "
    '"Dazu finde ich keine Informationen in der Wissensbasis." '
    "Erfinde keine Fakten; Zahlen, Namen und Bezeichnungen duerfen nur aus "
    "dem Kontext stammen."
)
"""Konditionale Abstain-Instruktion fuer den Generator-Prompt (Stufe 2).

Moderat formuliert: erlaubt Antworten aus dem Kontext weiterhin frei,
verbietet nur erfundene Fakten und gibt einen expliziten Abstain-Ausweg
(Sprint-5-Lehre: hartes "nur woertlich zitieren" drueckte correct auf 0).
"""

EXPORT_MAX_AGE_DAYS_DEFAULT: int = 30
"""Standard-Aufbewahrungsfrist fuer Workflow-Exporte in Tagen."""

HEALTH_CHECK_INTERVAL_SECS_DEFAULT: int = 60
"""Standard-Intervall fuer Health-Checks (Sekunden)."""

HEALTH_LOG_INTERVAL_SECS_DEFAULT: int = 300
"""Standard-Intervall fuer Health-Log-Eintraege (Sekunden, 5 Minuten)."""

BACKUP_MAX_COUNT_DEFAULT: int = 5
"""Standard-Obergrenze fuer aufbewahrte Backups."""

CACHE_MAX_SIZE_DEFAULT: int = 1000
"""Standard-Obergrenze fuer Cache-Eintraege."""

CACHE_TTL_SECS_DEFAULT: int = 3600
"""Standard-TTL fuer Cache-Eintraege (1 Stunde)."""

GRAPH_MAX_TRAVERSAL_DEPTH_DEFAULT: int = 2
"""Standard-Traversierungstiefe fuer Graph-Retrieval."""

GRAPH_MAX_CHUNKS_DEFAULT: int = 20
"""Standard-Anzahl Graph-Chunks im Retrieval-Kontext."""

ANON_MIN_TERM_LENGTH_DEFAULT: int = 3
"""Standard-Minimale Termlaenge fuer Anonymisierung."""

ANON_LLM_CONFIDENCE_THRESHOLD_DEFAULT: float = 0.7
"""Standard-Confidence-Schwelle fuer LLM-basierte Anonymisierungs-Erkennung."""
