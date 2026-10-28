# Retrieval: gemeinsame Suchpipeline

[Karten mit Codebelegen öffnen](/wiki/guide/architecture-retrieval)

Geprüft: 2026-10-22. Geprüfter Arbeitsbaum einschließlich vorhandener uncommittierter Research-, Retrieval- und Chat-Änderungen. Kein Release- oder Laufzeitnachweis.

## Retrieval: gemeinsame Suchpipeline

Alle vier Hauptnutzer rufen HybridRetriever.retrieve auf. Die optionalen Suchpfade werden im Orchestrator nacheinander aufgerufen (1–4). Ihre Ergebnislisten werden anschließend kombiniert; die Darstellung behauptet keine parallele Ausführung.

### R1 Scope und Suchtext

Aufrufer bestimmt erlaubte Datenquellen sowie Datei-, Verzeichnis- und Quellfilter.

Benutzer + Anfrage → filter_datasource_ids etc.

Konfiguration: Auswahl/Ausschluss/Public-DS; Admin kann None erhalten

Fehler/Grenzen: [] bedeutet kein Zugriff/keine Treffer; None bedeutet hier keine DS-Einschränkung.

- [kiara/web/api_chat.py:902 – _build_datasource_filter](/wiki/architecture/code/acl#L902)
- [kiara/services/retrieval_only.py:365 – RetrievalOnlyService._allowed_datasource_filter](/wiki/architecture/code/direct_acl#L365)
- [kiara/tools/builtin_tools.py:585 – SearchDocumentsTool._execute_search](/wiki/architecture/code/tool#L585)
- [kiara/services/research.py:31 – refresh_scope](/wiki/architecture/code/research_scope#L31)

### R2 Suchpfade auswählen

Retriever normalisiert Filter und prüft konfigurierte bzw. verfügbare Zusatzverfahren.

Text + Scope → Pfade/Filter

Konfiguration: retrieval.hybrid_search; graph.enabled; retrieval.colbert.enabled

Fehler/Grenzen: Leerer Scope kehrt sofort zurück. Fehlt jeder Zusatzpfad, reine semantische Suche ohne RRF.

- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)
- [kiara/retrieval/retriever.py:519 – HybridRetriever._setup_hybrid_mode](/wiki/architecture/code/hybrid_setup#L519)
- [kiara/retrieval/retriever.py:568 – HybridRetriever._setup_graph](/wiki/architecture/code/graph_setup#L568)
- [kiara/retrieval/retriever.py:596 – HybridRetriever._setup_colbert](/wiki/architecture/code/colbert_setup#L596)

### R3 1 · Semantisch suchen

Frage wird eingebettet und mit Dokumentvektoren verglichen. Vorfilter schränken Kandidaten ein.

Frage → Qdrant-Treffer

Konfiguration: vector_fetch_multiplier; gebundenes Dense-/Late-Profil

Fehler/Grenzen: Semantik- oder Indexvertragsfehler brechen ab.

- [kiara/retrieval/retriever.py:1479 – HybridRetriever._semantic_retrieve](/wiki/architecture/code/semantic#L1479)
- [kiara/retrieval/vectorstore_qdrant.py:441 – QdrantVectorStore.query](/wiki/architecture/code/qquery#L441)
- [kiara/retrieval/embedding.py:775 – embed_query](/wiki/architecture/code/query_embed#L775)

### R4 2 · Optional Keyword

BM25 gewichtet passende Wörter; ES nutzt DS-spezifische Ziele, lokaler BM25 den Cache.

Suchbegriffe → Keyword-Rangliste

Konfiguration: hybrid_search; keyword_match_operator; keyword_content_only; ES-Fuzziness

Fehler/Grenzen: Aktiver Keyword-Pfad mit Fehler bricht ab; Cache-Verfügbarkeit beeinflusst lokalen BM25.

- [kiara/retrieval/retriever.py:1504 – HybridRetriever._keyword_retrieve](/wiki/architecture/code/keyword#L1504)
- [kiara/retrieval/retriever.py:1920 – HybridRetriever._es_search](/wiki/architecture/code/es_search#L1920)
- [kiara/retrieval/retriever.py:1563 – HybridRetriever._bm25_search](/wiki/architecture/code/bm25#L1563)
- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)

### R5 3 · Optional Graph

Entitäten und Beziehungen liefern verbundene Chunks; nachträglich DS-/Metadatenfilter.

Suchtext → Graph-Rangliste

Konfiguration: graph.enabled und Graph-Konfiguration

Fehler/Grenzen: Gewöhnlicher Fehler: Warnung und Fortsetzung ohne Graph; IndexCompatibilityError propagiert.

- [kiara/retrieval/retriever.py:1532 – HybridRetriever._graph_retrieve](/wiki/architecture/code/graph_search#L1532)
- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)

### R6 4 · Optional ColBERT

Tokenvektoren werden per MaxSim verglichen und gegen den primären Bestand geprüft.

Suchtext → ColBERT-Rangliste

Konfiguration: retrieval.colbert.enabled/model

Fehler/Grenzen: Aktiviertes ColBERT mit Fehler bricht ab; Initialisierungsfehler ebenso.

- [kiara/retrieval/retriever.py:669 – HybridRetriever._colbert_retrieve_result](/wiki/architecture/code/cb_search#L669)
- [kiara/retrieval/retriever.py:596 – HybridRetriever._setup_colbert](/wiki/architecture/code/colbert_setup#L596)
- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)

### R7 Ranglisten verbinden

RRF addiert 1/(rrf_k + Rang) je Pfad und Chunk-ID. Keine Vermischung inkompatibler Rohscores.

Bis zu vier Listen → eindeutige Rangliste

Konfiguration: retrieval.rrf_k

Fehler/Grenzen: Ohne eindeutige Chunk-ID wird der Treffer verworfen; jeder Pfad zählt höchstens einmal pro Chunk.

- [kiara/retrieval/retriever.py:1806 – HybridRetriever._rrf_fusion_multi](/wiki/architecture/code/rrf#L1806)

### R8 Zugehörigkeit und Version

Aktuelle PG-Owner ersetzen Payload-Owner; Publikationszustand und Version werden geprüft.

Kandidaten → freigegebene Treffer, dann Top-k

Konfiguration: DS-Scope, Modell, Collection

Fehler/Grenzen: Fehlende Owner entfernen Treffer. Offene/falsche Dokumentversion oder Nachweisfehler brechen ab.

- [kiara/retrieval/retriever.py:207 – Retriever.filter_available_results](/wiki/architecture/code/available#L207)
- [kiara/ingestion/document_publication.py:321 – assert_documents_published](/wiki/architecture/code/published#L321)
- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)

### R9 An Aufrufer zurückgeben

Standard/Tools/direct verarbeiten geprüfte Treffer weiter. Reranking liegt außerhalb des HybridRetrievers.

Treffer → Modus-spezifische Verarbeitung

Konfiguration: siehe Kontextkarte und Modustabelle

Fehler/Grenzen: Spätere Kontextprüfungen sichern Nachladen und Generierungsübergang.

- [kiara/generation/chain.py:403 – RAGChain._retrieve_and_rerank](/wiki/architecture/code/rag_retrieve#L403)
- [kiara/tools/builtin_tools.py:446 – SearchDocumentsTool._retrieve_and_rerank](/wiki/architecture/code/tool_search#L446)
- [kiara/services/retrieval_only.py:665 – RetrievalOnlyService.search_documents](/wiki/architecture/code/direct#L665)

### R10 Anfrage abbrechen

Fehlgeschlagene Pflichtpfade und Freigaben dürfen keine Teilantwort als regulären Sucherfolg liefern.

Fehler → Exception / strukturierter API-Fehler

Konfiguration: Konfigurierter Suchpfad; Indexvertrag

Fehler/Grenzen: Graph ist die dokumentierte Best-effort-Ausnahme; Reranker hat eigene Fehlerpolitik.

- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)
- [kiara/retrieval/retriever.py:207 – Retriever.filter_available_results](/wiki/architecture/code/available#L207)
- [kiara/services/retrieval_only.py:665 – RetrievalOnlyService.search_documents](/wiki/architecture/code/direct#L665)

## Detail: vom Suchtreffer zur Ausgabe

Standard-RAG sucht vor der Antwort. Agent und Research können wiederholt suchen und Originalbereiche nachlesen. Direkte Suche liefert strukturierte Treffer ohne Antwortgenerierung.

### C1 Standard-RAG

Optional Suchvarianten, Mehrfach-Retrieval, Deduplizierung, Reranking und Nachbarchunks.

Frage → geprüfte Treffer

Konfiguration: query_expansion nur mit Reranker; Expander nur Ollama

Fehler/Grenzen: Erste Suchvariante muss gelingen; zusätzliche Varianten können mit Warnung ausfallen. Vertrags-/Freigabefehler bleiben kritisch.

- [kiara/generation/chain.py:403 – RAGChain._retrieve_and_rerank](/wiki/architecture/code/rag_retrieve#L403)
- [kiara/retrieval/query_expansion.py:40 – QueryExpander.expand](/wiki/architecture/code/expansion#L40)

### C2 Agent-Schleife

Modell entscheidet über Werkzeuge; Toolrechte, Wiederholungs- und Iterationsgrenzen werden geprüft.

Frage/Toolresultate → weitere Aufrufe oder finale Antwort

Konfiguration: chat.agent.*; max_iterations; max_same_tool_calls

Fehler/Grenzen: Werkzeug-/Backendfehler und Loop-Grenzen sind sichtbar; Forced Search ist bedingt, keine universelle Suchgarantie.

- [kiara/generation/agent_chain.py:1049 – AgentChain.query_stream](/wiki/architecture/code/agent#L1049)
- [kiara/generation/agent_chain.py:1330 – AgentChain._execute_tool](/wiki/architecture/code/agent_tools#L1330)

### C3 Research-Schleife

Scope bestimmen, mehrere Suchrunden planen, Dokumentfenster lesen und Evidenz sammeln.

Frage → Belegspeicher + Befunde

Konfiguration: chat.research.*; derzeit nur lokales Ollama

Fehler/Grenzen: Zeit-/Aufruf-/Lese-/Speichergrenzen erzeugen incomplete/failed statt Vollständigkeitsbehauptung.

- [kiara/services/research.py:84 – create_research](/wiki/architecture/code/research_setup#L84)
- [kiara/generation/research_chain.py:622 – ResearchChain.run](/wiki/architecture/code/research#L622)
- [kiara/generation/research_chain.py:347 – ResearchChain._search_round](/wiki/architecture/code/research_search#L347)

### C4 Direkte Suche

MCP/REST ruft RetrievalOnlyService auf; Limits, Raw-Text-Policy und Audit.

Frage → Treffer/Diagnostik

Konfiguration: web.mcp_retrieval_enabled; mcp_retrieval_allow_raw_chunks; Limits

Fehler/Grenzen: Ungültige Filter werden abgewiesen; Retrievalfehler als 503, Vertragsfehler separat.

- [kiara/mcp_server.py:122 – _run_retrieval_search](/wiki/architecture/code/mcp_search#L122)
- [kiara/services/retrieval_only.py:665 – RetrievalOnlyService.search_documents](/wiki/architecture/code/direct#L665)
- [kiara/services/retrieval_only.py:365 – RetrievalOnlyService._allowed_datasource_filter](/wiki/architecture/code/direct_acl#L365)

### C5 Gemeinsamer Retriever

Alle Modi nutzen die oben gezeigte Suchpipeline.

Suchtext + Scope → freigegebene Chunks

Konfiguration: Modell/Hybrid-Konfiguration

Fehler/Grenzen: Gemeinsame PG-/Publikationsprüfung gilt für alle Treffer.

- [kiara/retrieval/retriever.py:1158 – HybridRetriever.retrieve](/wiki/architecture/code/retrieve#L1158)
- [kiara/retrieval/retriever.py:207 – Retriever.filter_available_results](/wiki/architecture/code/available#L207)

### C6 Suchwerkzeug

Agent und Research nutzen SearchDocumentsTool mit optionalem Reranking und Nachbarn.

Toolparameter → versionierte Referenzen/Chunks

Konfiguration: Quellenprofil, reranking, max_sibling_chunks

Fehler/Grenzen: Einmalige Wiederholung bei Nulltreffern ohne opportunistische Toolfilter; DS-Rechte und Standardfilter bleiben.

- [kiara/tools/builtin_tools.py:585 – SearchDocumentsTool._execute_search](/wiki/architecture/code/tool#L585)
- [kiara/tools/builtin_tools.py:446 – SearchDocumentsTool._retrieve_and_rerank](/wiki/architecture/code/tool_search#L446)
- [kiara/tools/document_tools.py:50 – DocumentReferences.annotate](/wiki/architecture/code/references#L50)

### C7 RAG-Kontext

Kein Treffer oder Abstain kann ohne Modellantwort enden; ansonsten Kontextlimit, Formatierung und erneute Prüfung.

Treffer → Prompt + Quellen

Konfiguration: max_context_chunks; abstain_min_rerank_score

Fehler/Grenzen: Such-Tab-Auswahl wird aus Store nachgeladen und überspringt Such-/Rerankphase. Vor Streambeginn erneut validate_context.

- [kiara/generation/chain.py:874 – RAGChain.query_stream](/wiki/architecture/code/rag#L874)
- [kiara/generation/chain.py:618 – RAGChain._abstain_gate](/wiki/architecture/code/abstain#L618)
- [kiara/retrieval/retriever.py:274 – Retriever.load_context_chunks](/wiki/architecture/code/reload#L274)

### C8 Agent-Kontext

Dokumentteile auswählen/zusammensetzen, optional final reranken und auf tatsächliche Adapterdarstellung budgetieren.

Tool-Chunks → Modell-Toolresultat

Konfiguration: context_budget_chars, context_result_chars, Leserreserve, final_reranking

Fehler/Grenzen: Budgetkürzung/ausgelassene Teile werden gemeldet; Lesen kann bisherigen Suchauszug ersetzen.

- [kiara/generation/agent_chain.py:461 – AgentChain._prepare_tool_result](/wiki/architecture/code/agent_context#L461)

### C9 Originalbereiche lesen

ReadDocumentTool löst nur zuvor erzeugte Dokumentreferenzen auf und liest begrenzte Bereiche.

Dokumentreferenz + Bereich → Originalchunks

Konfiguration: document_read_max_chunks; Leserbudget

Fehler/Grenzen: Keine neue Suche/Embeddings; falsche Version/Rechte verhindern Ausgabe.

- [kiara/tools/document_tools.py:202 – ReadDocumentTool.execute](/wiki/architecture/code/read#L202)
- [kiara/retrieval/retriever.py:346 – Retriever.load_document_range](/wiki/architecture/code/read_range#L346)

### C10 Research belegen und prüfen

Readerbefund braucht Originalzitate; Synthese und Verifier bekommen nachgeladene Originale.

Gelesene Bereiche → belegter Entwurf oder offener Befund

Konfiguration: max_evidence; Kontext- und Aufrufbudget

Fehler/Grenzen: Ungültige Zitate, unbelegte Zahlen, Versionsänderung oder ausgelassene Evidenz verhindern regulären belegten Abschluss.

- [kiara/generation/research_chain.py:419 – ResearchChain._read_page](/wiki/architecture/code/research_read#L419)
- [kiara/generation/research_chain.py:522 – ResearchChain._answer](/wiki/architecture/code/research_answer#L522)
- [kiara/generation/research_chain.py:513 – ResearchChain.validate_sources](/wiki/architecture/code/research_validate#L513)

### C11 Antwortmodell / Adapter

Standard nutzt Backendstream, Agent Adapter-Finale; Research erzeugt bereits geprüfte Ergebnisstruktur.

Prompt/Belege → Antwort

Konfiguration: Backend, Kontextgröße, Routing/Anonymisierung nach gewähltem Modus

Fehler/Grenzen: Kein impliziter allgemeiner Cloud-Fallback in dieser Karte behauptet.

- [kiara/generation/chain.py:874 – RAGChain.query_stream](/wiki/architecture/code/rag#L874)
- [kiara/generation/agent_chain.py:984 – AgentChain._stream_final_answer](/wiki/architecture/code/agent_final#L984)
- [kiara/generation/research_chain.py:522 – ResearchChain._answer](/wiki/architecture/code/research_answer#L522)

### C12 Treffer als JSON

Optionale Rerankstufe, Rangnummern, Textlimits, Abdeckung und Audit; keine LLM-Antwort.

Treffer → strukturierte Ausgabe

Konfiguration: reranking; max_k/max_chars_per_result; Raw-Policy

Fehler/Grenzen: Direktsuche führt keine Query- oder Sibling-Expansion durch.

- [kiara/services/retrieval_only.py:665 – RetrievalOnlyService.search_documents](/wiki/architecture/code/direct#L665)

## Detail: die direkte Suche im Such-Tab

Der Browser-Such-Tab /api/search ist vom RetrievalOnlyService zu unterscheiden: keine Hybrid-Fusion, keine Query-Expansion und kein Reranking. Die beiden Modi besitzen im geprüften Arbeitsstand unterschiedliche Freigabeprüfungen.

### S1 Suchmodus und Scope

Route prüft Anfrage, Pagination, Datum/Pfad und serverseitige DS-Rechte.

SearchRequest → Text- oder Semantikpfad

Konfiguration: mode text/semantic; Seitenlimit; explizite DS-Auswahl

Fehler/Grenzen: Leere DS-Auswahl liefert leer; ungültige Eingaben liefern 400.

- [kiara/web/api_search.py:95 – search](/wiki/architecture/code/search_tab#L95)
- [kiara/web/api_chat.py:902 – _build_datasource_filter](/wiki/architecture/code/acl#L902)

### S2 Text: Elasticsearch

Explizite ES-Ziele, Trefferseite, danach aktuelle PG-Zugehörigkeit und Dokumentfreigabe.

Suchbegriffe → geprüfte ES-Seite

Konfiguration: retrieval.hybrid_search=elasticsearch, Fuzziness, Datumsfilter

Fehler/Grenzen: Ein ungültiger/unveröffentlichter Seitentreffer lässt die Seite fehlschlagen; keine still falschen Gesamtzahlen.

- [kiara/web/api_search.py:172 – _search_text](/wiki/architecture/code/search_text#L172)
- [kiara/web/api_search.py:281 – _assert_text_hits_available](/wiki/architecture/code/text_available#L281)

### S3 Semantisch: direkt Qdrant

Query-Embedding, Payload-Vorfilter und paginierte query_points-Abfrage ohne HybridRetriever.

Suchtext → rohe Qdrant-Seite

Konfiguration: embedding_model; DS/Pfad/Typ/Datum; Offset

Fehler/Grenzen: Statisch bestätigte Prüflücke: _search_semantic und QdrantVectorStore.search führen keine zusätzliche aktuelle PG-Owner-/Publikationsprüfung aus. Der Indexguard prüft den Indexvertrag. Kein Laufzeitvorfall reproduziert.

- [kiara/web/api_search.py:307 – _search_semantic](/wiki/architecture/code/search_semantic#L307)
- [kiara/retrieval/vectorstore_qdrant.py:538 – QdrantVectorStore.search](/wiki/architecture/code/qsearch#L538)
- [kiara/retrieval/vectorstore_qdrant.py:118 – build_qdrant_filter](/wiki/architecture/code/qfilter#L118)
- [kiara/retrieval/vectorstore_qdrant.py:66 – _guarded_index_operation](/wiki/architecture/code/qguard#L66)

### S4 Trefferseite ausgeben

Route formatiert Text, Metadaten, Score, Trefferzahl und Laufzeit.

Treffer → JSON für Such-Tab

Konfiguration: page, page_size; ES-Score-Normalisierung

Fehler/Grenzen: Kein LLM-Aufruf. Eine spätere Übergabe als RAG-Kontext wird dort über load_context_chunks erneut gelesen und geprüft.

- [kiara/web/api_search.py:95 – search](/wiki/architecture/code/search_tab#L95)
- [kiara/web/api_search.py:172 – _search_text](/wiki/architecture/code/search_text#L172)
- [kiara/web/api_search.py:307 – _search_semantic](/wiki/architecture/code/search_semantic#L307)
- [kiara/retrieval/retriever.py:274 – Retriever.load_context_chunks](/wiki/architecture/code/reload#L274)

## Ungeprüft und Grenzen

- Ungeprüft: Welche optionalen Pfade eine konkrete produktive Datenquelle oder Anfrage tatsächlich durchläuft. Die Karten beschreiben Codebedingungen, keine ausgelesene Produktivkonfiguration.
- Ungeprüft: End-to-End-Verhalten bei realen DDU-/Embedding-/ES-Ausfällen und konkreten Dokumenten. Dafür sind die Beispielabläufe in #416 vorgesehen.
- Aus dem Code geprüft: Ein OTel-Span kann auch bei deaktiviertem Teilverfahren entstehen. enabled/configured/attempted und Trefferzahl müssen gemeinsam gelesen werden.
- Bewusste Grenze: Spezialfälle der HTML-/Mailmigration, ERP-/Aktionswerkzeuge, Rewriter und Unchained sind nur als Abgrenzung erfasst. Unchained umgeht den RAG-Pfad; die Karten fokussieren Standard, Agent, Research und direkte Suche.
