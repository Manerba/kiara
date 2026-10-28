# Gesamtarchitektur

[Karten mit Codebelegen öffnen](/wiki/guide/architecture)

Geprüft: 2026-10-22. Geprüfter Arbeitsbaum einschließlich vorhandener uncommittierter Research-, Retrieval- und Chat-Änderungen. Kein Release- oder Laufzeitnachweis.

## Gesamtarchitektur

Ein Dokument gelangt über die Ingestion in den geprüften Bestand. RAG, Agent, Research und MCP/REST-Retrieval nutzen die gemeinsame Suche. Der Such-Tab besitzt einen abweichenden direkten Pfad, dessen Prüfgrenze in der Retrieval-Ansicht erklärt wird. Linien zeigen ausgewählte wesentliche Übergänge, keine vollständige Aufrufhierarchie. Gemeinsame Funktionen laufen im aufrufenden Prozess; die Speichergruppe fasst externe Dienste und den lokalen BM25-Cache fachlich zusammen.

### O1 Job anfordern

Admin-Funktion oder Queue-CLI legt einen modellgebundenen Auftrag an.

DS-ID + Modell → Job-ID

Konfiguration: Jobtyp incremental/full; aktives Backend

Fehler/Grenzen: Doppelte oder inaktive DS-Aufträge und Vertragsfehler werden abgewiesen.

- [kiara/admin/functions/jobs.py:220 – jobs_start](/wiki/architecture/code/jobs#L220)
- [kiara/admin/functions/jobs.py:270 – jobs_start_full](/wiki/architecture/code/jobs_full#L270)
- [kiara/web/ingestion_worker.py:2720 – enqueue_job](/wiki/architecture/code/enqueue#L2720)
- [kiara/cli.py:407 – queue_ingest](/wiki/architecture/code/queue_cli#L407)

### O2 Auftrag bearbeiten

Worker pollt PostgreSQL und führt Vorbereitung, Scan, Pipeline und Abschluss aus.

Jobzeile → Statistik/Status

Konfiguration: Connector-Eigenschaften; Backend-Konfiguration

Fehler/Grenzen: Vorübergehender Claim-Ausfall bleibt pending; Vertragsbruch wird error.

- [kiara/web/ingestion_worker.py:852 – IngestionWorker._poll_loop](/wiki/architecture/code/poll#L852)
- [kiara/web/ingestion_worker.py:877 – IngestionWorker._fetch_next_job](/wiki/architecture/code/claim#L877)
- [kiara/web/ingestion_worker.py:1012 – IngestionWorker._execute_job](/wiki/architecture/code/worker#L1012)

### O3 Direkt indexieren

kiara index führt dieselbe Pipeline im CLI-Prozess aus.

Pfad/DS/alle DS → Statistik

Konfiguration: --incremental, --datasource-id, --all-datasources

Fehler/Grenzen: Ohne Worker-Control keine dauernde Backend-Warteschleife am Circuit Breaker.

- [kiara/cli.py:67 – index](/wiki/architecture/code/cli#L67)
- [kiara/cli.py:30 – _prepare_datasource_index](/wiki/architecture/code/cli_prepare#L30)
- [kiara/ingestion/pipeline.py:1802 – run_all_datasources](/wiki/architecture/code/all_ds#L1802)

### O4 Dokument verarbeiten

Scan, Hashvergleich, Parsing, Chunking, Speicherung und dokumentweite Freigabe.

Datei → gespeicherte Chunks und Abschluss

Konfiguration: DS-Overrides; Modell und Verarbeitungsprofil

Fehler/Grenzen: Fehler lassen den Dateiabschluss offen; Cleanup und Retry sind explizite Pfade.

- [kiara/ingestion/pipeline.py:1494 – run_ingestion](/wiki/architecture/code/pipeline#L1494)
- [kiara/ingestion/runtime.py:3417 – _process_file_with_publication](/wiki/architecture/code/file_loop#L3417)
- [kiara/ingestion/pipeline.py:690 – _complete_file](/wiki/architecture/code/complete#L690)

### O5 Frage oder Suche

Chat routet den Modus; MCP/REST bietet auch direkte Suche.

Frage + Benutzer + Filter → Modus/Scope

Konfiguration: chat.default_mode, Quellenprofile, MCP-Schalter

Fehler/Grenzen: Rechte und Eingaben werden vor Zugriff geprüft.

- [kiara/web/api_chat.py:1082 – chat_query](/wiki/architecture/code/chat#L1082)
- [kiara/web/api_chat.py:902 – _build_datasource_filter](/wiki/architecture/code/acl#L902)
- [kiara/mcp_server.py:122 – _run_retrieval_search](/wiki/architecture/code/mcp_search#L122)
- [kiara/services/retrieval_only.py:665 – RetrievalOnlyService.search_documents](/wiki/architecture/code/direct#L665)

### O6 Hybrid suchen

Semantische Suche, optional Keyword, Graph und ColBERT; gemeinsame Freigabe.

Suchtext + Scope → geprüfte Treffer

Konfiguration: retrieval.*, graph.enabled

Fehler/Grenzen: Kritische Such-/Vertragsfehler brechen ab; Graph kann mit Warnung entfallen.

- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)
- [kiara/retrieval/retriever.py:207 – Retriever.filter_available_results](/wiki/architecture/code/available#L207)

### O7 Kontext und Antwort

Standard, Agent und Research wählen und prüfen Belege unterschiedlich; direkte Suche gibt Treffer aus.

Treffer → Kontext/Antwort oder JSON

Konfiguration: Reranking, Kontextbudget, Modus

Fehler/Grenzen: Kein Treffer, unzureichende Belege oder Budgetgrenzen werden sichtbar.

- [kiara/generation/chain.py:874 – RAGChain.query_stream](/wiki/architecture/code/rag#L874)
- [kiara/generation/agent_chain.py:1049 – AgentChain.query_stream](/wiki/architecture/code/agent#L1049)
- [kiara/generation/research_chain.py:622 – ResearchChain.run](/wiki/architecture/code/research#L622)
- [kiara/services/retrieval_only.py:665 – RetrievalOnlyService.search_documents](/wiki/architecture/code/direct#L665)

### O8 Qdrant

Speichert Dokumenttext, Payload und Dense-/Late-Vektoren; ColBERT separat.

Chunks/Vektoren ↔ Treffer

Konfiguration: Modellgebundene kb_*- bzw. colbert_*-Collection

Fehler/Grenzen: Indexverträge schützen Modell-/Profilzuordnung.

- [kiara/retrieval/vectorstore_qdrant.py:388 – QdrantVectorStore.add](/wiki/architecture/code/qadd#L388)
- [kiara/retrieval/vectorstore_qdrant.py:441 – QdrantVectorStore.query](/wiki/architecture/code/qquery#L441)
- [kiara/retrieval/embedding.py:415 – collection_name_for_model](/wiki/architecture/code/collection#L415)
- [kiara/retrieval/retriever.py:669 – HybridRetriever._colbert_retrieve_result](/wiki/architecture/code/cb_search#L669)

### O9 Elasticsearch / BM25

ES hält den externen konfigurierten Keyword-Index; der lokale BM25-Cache lebt im abfragenden Prozess und wird aus dem Bestand aufgebaut.

Text/Metadaten ↔ Rangliste

Konfiguration: retrieval.hybrid_search: none/bm25/elasticsearch

Fehler/Grenzen: Konfigurierter ES-Schreib-/Suchfehler ist kritisch; BM25-Aufbau kann vorübergehend fehlen.

- [kiara/ingestion/runtime.py:1365 – _write_to_elasticsearch](/wiki/architecture/code/write_es#L1365)
- [kiara/retrieval/retriever.py:1504 – HybridRetriever._keyword_retrieve](/wiki/architecture/code/keyword#L1504)
- [kiara/retrieval/retriever.py:1563 – HybridRetriever._bm25_search](/wiki/architecture/code/bm25#L1563)
- [kiara/retrieval/retriever.py:1920 – HybridRetriever._es_search](/wiki/architecture/code/es_search#L1920)

### O10 PostgreSQL Dokumentzustand

DS-Zugehörigkeit, Datei-Hashes, Zähler, Publikationsfreigaben, Graph und Traces; die Jobqueue ist separat eingezeichnet.

Persistenter Zustand ↔ geprüfte Übergänge

Konfiguration: embedding_model ist Teil der Zuordnung

Fehler/Grenzen: PG ist maßgeblich für Owner/Freigabe; keine gemeinsame Transaktion mit Qdrant/ES.

- [kiara/web/ingestion_worker.py:2720 – enqueue_job](/wiki/architecture/code/enqueue#L2720)
- [kiara/ingestion/pipeline.py:690 – _complete_file](/wiki/architecture/code/complete#L690)
- [kiara/ingestion/document_publication.py:321 – assert_documents_published](/wiki/architecture/code/published#L321)
- [kiara/retrieval/retriever.py:207 – Retriever.filter_available_results](/wiki/architecture/code/available#L207)
- [kiara/ingestion/runtime.py:2331 – _persist_graph_and_associations](/wiki/architecture/code/graph_write#L2331)
- [kiara/shared/_pg_span_exporter.py:53 – PgSpanExporter.export](/wiki/architecture/code/export#L53)

### O11 Antwortmodelle

Generieren Antworten bzw. Agententscheidungen. Embedding-/DDU-/VLM-Aufrufe sind in den Detailkarten belegt.

Prompt/Frage → Antwort/Modellentscheidung

Konfiguration: Backend-Auswahl und Modus; Research derzeit lokal

Fehler/Grenzen: Modusabhängige Fehlerbehandlung; kein pauschaler Cloud-Fallback behauptet.

- [kiara/generation/chain.py:874 – RAGChain.query_stream](/wiki/architecture/code/rag#L874)
- [kiara/generation/agent_chain.py:1049 – AgentChain.query_stream](/wiki/architecture/code/agent#L1049)
- [kiara/services/research.py:84 – create_research](/wiki/architecture/code/research_setup#L84)
- [kiara/generation/research_chain.py:522 – ResearchChain._answer](/wiki/architecture/code/research_answer#L522)

### O12 Such-Tab

Separate Dokumentensuche über /api/search; Text und Semantik werden direkt ausgeführt.

Suchtext + Seiten-/Quellfilter → Trefferseite

Konfiguration: mode text/semantic, embedding_model, Datum/Pfad/Typ

Fehler/Grenzen: Textzweig prüft PG/Freigabe. Im semantischen Zweig fehlt diese zusätzliche Prüfung im geprüften Code; siehe eigene Detailkarte.

- [kiara/web/api_search.py:95 – search](/wiki/architecture/code/search_tab#L95)
- [kiara/web/api_search.py:172 – _search_text](/wiki/architecture/code/search_text#L172)
- [kiara/web/api_search.py:307 – _search_semantic](/wiki/architecture/code/search_semantic#L307)

### O13 PostgreSQL Jobqueue

Persistente Grenze zwischen Anforderung und separatem Worker.

Auftrag mit Modellvertrag → claimbare Jobzeile

Konfiguration: Ingestion-Jobstatus und aktive Datenquelle

Fehler/Grenzen: SKIP LOCKED und Vertragsprüfung schützen den Claim; der Worker kann unabhängig vom API-Prozess weiterarbeiten.

- [kiara/web/ingestion_worker.py:2720 – enqueue_job](/wiki/architecture/code/enqueue#L2720)
- [kiara/web/ingestion_worker.py:877 – IngestionWorker._fetch_next_job](/wiki/architecture/code/claim#L877)
- [kiara/web/ingestion_worker.py:852 – IngestionWorker._poll_loop](/wiki/architecture/code/poll#L852)

## Ungeprüft und Grenzen

- Ungeprüft: Welche optionalen Pfade eine konkrete produktive Datenquelle oder Anfrage tatsächlich durchläuft. Die Karten beschreiben Codebedingungen, keine ausgelesene Produktivkonfiguration.
- Ungeprüft: End-to-End-Verhalten bei realen DDU-/Embedding-/ES-Ausfällen und konkreten Dokumenten. Dafür sind die Beispielabläufe in #416 vorgesehen.
- Aus dem Code geprüft: Ein OTel-Span kann auch bei deaktiviertem Teilverfahren entstehen. enabled/configured/attempted und Trefferzahl müssen gemeinsam gelesen werden.
- Bewusste Grenze: Spezialfälle der HTML-/Mailmigration, ERP-/Aktionswerkzeuge, Rewriter und Unchained sind nur als Abgrenzung erfasst. Unchained umgeht den RAG-Pfad; die Karten fokussieren Standard, Agent, Research und direkte Suche.
