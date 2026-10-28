# Ingestion: von der Quelle zu Chunks

[Karten mit Codebelegen öffnen](/wiki/guide/architecture-ingestion)

Geprüft: 2026-10-22. Geprüfter Arbeitsbaum einschließlich vorhandener uncommittierter Research-, Retrieval- und Chat-Änderungen. Kein Release- oder Laufzeitnachweis.

## Ingestion: von der Quelle zu Chunks

Worker und CLI münden in run_ingestion. Die Pfeile zwischen Arbeitsschritten bilden den vom Orchestrator ausgeführten Ablauf ab; den tatsächlich aufrufenden Code nennt jeder Verbindungsbeleg.

### I1 Vorbereiten und scannen

Worker lädt Connector/DS und wärmt Backends auf. Ein geeigneter gefüllter Cache erlaubt parallelen Scan plus zweiten inkrementellen Durchlauf.

Job/DS → lokaler effektiver Pfad

Konfiguration: SUPPORTS_PARALLEL_SCAN; full erzwingt sequentiellen Scan; spezielle HTML-Auswahl

Fehler/Grenzen: Fehlgeschlagener Scan beendet den Job; Pass-1-Bestand bleibt bei Pass-2-Scanfehler bestehen.

- [kiara/web/ingestion_worker.py:1012 – IngestionWorker._execute_job](/wiki/architecture/code/worker#L1012)
- [kiara/web/ingestion_worker.py:1739 – IngestionWorker._run_scan_phase](/wiki/architecture/code/scan#L1739)
- [kiara/web/ingestion_worker.py:1846 – IngestionWorker._start_parallel_scan](/wiki/architecture/code/parallel#L1846)
- [kiara/web/ingestion_worker.py:2218 – IngestionWorker._run_pass2](/wiki/architecture/code/pass2#L2218)
- [kiara/web/ingestion_worker.py:1538 – IngestionWorker._run_warmup](/wiki/architecture/code/warmup#L1538)

### I2 Modell und Optionen

Pipeline löst Modell, DS-Overrides, DDU/VLM und Indexvertrag auf.

Config/Overrides → Store + ProcessingContext

Konfiguration: ingestion.*, DS ddu_enabled/vlm_enabled; aktive Backends

Fehler/Grenzen: Fehlende DDU/VLM-Backends deaktivieren das Feature; Vertragsbruch blockiert.

- [kiara/ingestion/pipeline.py:1494 – run_ingestion](/wiki/architecture/code/pipeline#L1494)
- [kiara/ingestion/pipeline.py:170 – _apply_ds_overrides](/wiki/architecture/code/overrides#L170)
- [packages/kiara-ingestion-core/kiara_ingestion_core/datasource_runtime.py:30 – build_datasource_overrides](/wiki/architecture/code/ds_overrides#L30)

### I3 Dateien vergleichen

Frischer Scan und Hash inklusive Verarbeitungsrevision bestimmen Kandidaten und entfernte Dateien. Offene Veröffentlichungen werden wiederholt.

Pfadinventar + gespeicherte Hashes → Kandidaten

Konfiguration: incremental; Connector HASH_MODE mtime/content; Dateifilter

Fehler/Grenzen: Ein Erfolgslog alleine überspringt nichts. Dateiänderungen während Verarbeitung verhindern den Abschluss.

- [kiara/ingestion/pipeline.py:495 – _load_and_diff_files](/wiki/architecture/code/diff#L495)
- [packages/kiara-ingestion-core/kiara_ingestion_core/file_diff.py:101 – _compute_file_diff](/wiki/architecture/code/hash_diff#L101)
- [packages/kiara-ingestion-core/kiara_ingestion_core/file_diff.py:80 – compute_file_hash](/wiki/architecture/code/hash#L80)
- [kiara/ingestion/_file_utils.py:47 – _LoadedFileVersion](/wiki/architecture/code/file_version#L47)
- [kiara/ingestion/_file_utils.py:87 – scan_files](/wiki/architecture/code/scan_files#L87)

### I4 Unverändert überspringen

Nur passender Abschluss-Hash ohne offene Veröffentlichung erlaubt den Skip; CLI-Owner werden zusätzlich verifiziert.

Unveränderte Datei → übersprungen

Konfiguration: Nur inkrementell

Fehler/Grenzen: No-files-Pfad führt trotzdem notwendige Zähler-/Modellbereinigung aus.

- [kiara/ingestion/pipeline.py:495 – _load_and_diff_files](/wiki/architecture/code/diff#L495)
- [kiara/ingestion/pipeline.py:1494 – run_ingestion](/wiki/architecture/code/pipeline#L1494)

### I5 Alten Bestand bereinigen

Voll-/Migrationslauf oder entfernte Dateien: quellenweite Bereinigung. Geänderte Inkrementalkandidaten erst nach Vergleich der geladenen Eingangsdarstellung.

Quellen → bereinigte Speicher + dauerhafter Retry-Anker

Konfiguration: DS/Modell, historische ColBERT-Modelle, ES-Modellzuordnung

Fehler/Grenzen: Lesesperre und Cleanup-Inventar bleiben bis bestätigter Bereinigung bestehen; fremde Owner werden berücksichtigt.

- [kiara/ingestion/pipeline.py:241 – _cleanup_stale_chunks](/wiki/architecture/code/cleanup#L241)
- [kiara/ingestion/pipeline.py:337 – _cleanup_source_batch](/wiki/architecture/code/cleanup_batch#L337)
- [kiara/ingestion/pipeline.py:643 – _prepare_file_chunks](/wiki/architecture/code/prepare#L643)
- [kiara/ingestion/pipeline.py:495 – _load_and_diff_files](/wiki/architecture/code/diff#L495)

### I6 Laden und Bilder beschreiben

Standard-/HTML-/CAD-Loader oder Remote-DDU; optional zusätzliche VLM-Beschreibung.

Datei → Text + Metadaten, optional zweiter VLM-Text

Konfiguration: DS ddu_enabled/ddu_ocr_images/vlm_enabled; DDU-Backendliste

Fehler/Grenzen: DDU-Verbindungs-/HTTP-Fehler versuchen weitere Backends; Timeout beendet DDU-Failover. Erhaltenes ddu_error verwirft auch Standard-Fallbacktext in _load_content.

- [kiara/ingestion/runtime.py:1152 – _process_single_file](/wiki/architecture/code/single#L1152)
- [kiara/ingestion/runtime.py:914 – _load_content](/wiki/architecture/code/load#L914)
- [kiara/ingestion/loader.py:189 – load_file](/wiki/architecture/code/loader#L189)
- [kiara/ingestion/loader.py:111 – _load_with_docling](/wiki/architecture/code/ddu#L111)
- [kiara/ingestion/runtime.py:994 – _handle_vlm_content](/wiki/architecture/code/vlm#L994)

### I7 In Abschnitte zerlegen

Chunk = ein Textabschnitt. Strategie: explizit, eigene Größe, automatische Erkennung. VLM liefert einen einzigen Chunk.

Text → Chunktexte, Metadaten, IDs, Zeichenbereiche

Konfiguration: chunk_strategy, ingestion.chunk_size/chunk_overlap

Fehler/Grenzen: Ungültiges Chunking führt zu Dateifehler; IDs binden Quelle, Parser und Position, Versionen zusätzlich Inhalt.

- [kiara/ingestion/runtime.py:1022 – _determine_chunks](/wiki/architecture/code/chunks#L1022)
- [kiara/ingestion/runtime.py:1086 – _chunk_and_collect](/wiki/architecture/code/collect#L1086)
- [kiara/ingestion/runtime.py:3417 – _process_file_with_publication](/wiki/architecture/code/file_loop#L3417)

### I8 Vorhandene Chunks prüfen

PG-Kandidaten werden in Qdrant bestätigt; Zuordnungen werden abgeglichen. Fehlende Punkte werden neu eingebettet.

Chunk-IDs → FULL / PARTIAL / NO_DEDUP

Konfiguration: Modell und optionale ColBERT-Aktivierung

Fehler/Grenzen: FULL-Dedup ersetzt weder ColBERT-Backfill noch Veröffentlichung/Dateiabschluss.

- [kiara/ingestion/runtime.py:2391 – _handle_chunk_dedup](/wiki/architecture/code/dedup#L2391)
- [kiara/ingestion/pipeline.py:643 – _prepare_file_chunks](/wiki/architecture/code/prepare#L643)
- [kiara/ingestion/runtime.py:3417 – _process_file_with_publication](/wiki/architecture/code/file_loop#L3417)

### I9 Optional anreichern

Zusatzkontext wird vor dem Embedding erzeugt; Originalbereiche bleiben für Late Chunking erhalten.

Neue Chunktexte → angereicherte Texte

Konfiguration: enrichment.enabled und DS-Overrides

Fehler/Grenzen: Fehlgeschlagenes Enrichment kann auf ursprüngliche Texte zurückgehen; kein Erfolg allein durch Anreicherung.

- [kiara/ingestion/runtime.py:3299 – _run_enrichment](/wiki/architecture/code/enrich#L3299)
- [kiara/ingestion/runtime.py:3103 – _should_enrich](/wiki/architecture/code/enrich_on#L3103)
- [kiara/ingestion/runtime.py:3417 – _process_file_with_publication](/wiki/architecture/code/file_loop#L3417)

### I10 Vektoren und Indizes schreiben

Embedding = numerische Textdarstellung. Dense oder explizites Late-Profil; Qdrant, optional ES, danach PG-Zuordnung und optionale Graph-Extraktion.

Neue Chunks → Vektoren/Payload, ES, PG

Konfiguration: ingestion.batch_size; enrichment.late_chunking und verifiziertes Late-Profil; retrieval.hybrid_search; graph.enabled/extract_on_ingestion

Fehler/Grenzen: ES ist bei Aktivierung Pflicht. Gewöhnliche Graph-Extraktionsfehler werden gezählt. Keine verteilte atomare Transaktion.

- [kiara/ingestion/runtime.py:2646 – _handle_embedding_phase](/wiki/architecture/code/embedding#L2646)
- [kiara/ingestion/runtime.py:1239 – _write_embeddings](/wiki/architecture/code/write#L1239)
- [kiara/retrieval/embedding.py:722 – embed_texts](/wiki/architecture/code/doc_embed#L722)
- [kiara/ingestion/runtime.py:2552 – _should_use_late_chunking](/wiki/architecture/code/late_on#L2552)
- [kiara/retrieval/embedding.py:835 – embed_texts_late](/wiki/architecture/code/late#L835)
- [kiara/ingestion/runtime.py:2331 – _persist_graph_and_associations](/wiki/architecture/code/graph_write#L2331)
- [kiara/ingestion/runtime.py:1412 – _run_graph_extraction](/wiki/architecture/code/graph_extract#L1412)

### I11 Optional ColBERT ergänzen

Zusätzliche Tokenvektoren ermöglichen MaxSim-Suche. Neue und wiederverwendete Chunks werden berücksichtigt.

Gespeicherte Chunkdarstellung → ColBERT-Index

Konfiguration: retrieval.colbert.enabled + DS colbert_enabled; Modell

Fehler/Grenzen: Aktivierter ColBERT-Schritt muss vor Dateiabschluss erfolgreich sein; eigener Retry-/Rollbackpfad.

- [kiara/ingestion/runtime.py:2056 – _handle_colbert_step](/wiki/architecture/code/colbert#L2056)
- [kiara/ingestion/runtime.py:1816 – _colbert_enabled_for_run](/wiki/architecture/code/colbert_on#L1816)
- [kiara/ingestion/runtime.py:3417 – _process_file_with_publication](/wiki/architecture/code/file_loop#L3417)

### I12 Dokument freigeben

Alle gespeicherten Chunks werden als vollständige Darstellung bestätigt; danach gemeinsamer PG-Abschluss. Detailkarte unten.

Gesamtdokument → veröffentlichte Version + Hash/Zähler

Konfiguration: Bestätigtes Modell und Owner

Fehler/Grenzen: Fehler lassen die Veröffentlichung offen; ein späterer FULL-Dedup versucht den Abschluss erneut.

- [kiara/ingestion/runtime.py:281 – _mark_file_processed](/wiki/architecture/code/mark#L281)
- [kiara/retrieval/vectorstore_qdrant.py:608 – QdrantVectorStore.finalize_document_versions](/wiki/architecture/code/versions#L608)
- [kiara/ingestion/pipeline.py:690 – _complete_file](/wiki/architecture/code/complete#L690)
- [kiara/ingestion/document_publication.py:290 – complete_document_publication](/wiki/architecture/code/pg_complete#L290)

### I13 Lauf abschließen

Weitere Dateien, Hashmodus und Recount; Worker wertet cancel/abort/shutdown aus und schreibt Abschluss.

Statistik → Job-/DS-Zustand

Konfiguration: Control-Callback, Jobtyp

Fehler/Grenzen: Ein normaler Job kann mit Dateifehlern abgeschlossen werden, solange kein abort_reason gesetzt ist; Migration ist strenger.

- [kiara/ingestion/pipeline.py:1494 – run_ingestion](/wiki/architecture/code/pipeline#L1494)
- [kiara/web/ingestion_worker.py:2346 – IngestionWorker._resolve_job_outcome](/wiki/architecture/code/outcome#L2346)
- [kiara/web/ingestion_worker.py:2380 – IngestionWorker._finalize_success](/wiki/architecture/code/finalize#L2380)

### I14 Fehler, Retry, Pause

Lade-/Speicherfehler führen zu Skip, Recovery oder Abbruch. Pause blockiert den Control-Callback; Shutdown bleibt wiederaufnehmbar.

Fehler/Steuersignal → Retry/Skip/Cancel/Abort

Konfiguration: Circuit-Breaker-Schwelle; maximal zwei Datei-Retries nach Auslösung; Worker-Control

Fehler/Grenzen: Rollbackfehler und Vertragsbruch brechen ab. Skip leert keinen bereits bestätigten Abschluss anderer Dateien.

- [kiara/ingestion/runtime.py:1539 – _handle_circuit_breaker](/wiki/architecture/code/retry#L1539)
- [kiara/ingestion/runtime.py:1668 – _handle_loading_result](/wiki/architecture/code/empty#L1668)
- [kiara/ingestion/runtime.py:456 – _rollback_embedding_persistence](/wiki/architecture/code/rollback#L456)
- [kiara/web/ingestion_worker.py:1256 – IngestionWorker._build_check_control](/wiki/architecture/code/control#L1256)
- [kiara/web/ingestion_worker.py:660 – IngestionWorker._recover_single_job](/wiki/architecture/code/recover#L660)

## Detail: wann ein Dokument sichtbar wird

Die Freigabe betrifft eine Quelle in einer Collection. Qdrant, Elasticsearch und PostgreSQL werden nicht gemeinsam atomar geschrieben. Eine dauerhafte Lesesperre und bestätigte Versionen verhindern, dass Teilstände als fertiger Kontext verwendet werden.

### P1 Quelle sperren

Advisory-Lock serialisiert Writer derselben Quelle; begin entwertet den eigenen Dateiabschluss und setzt completed_at auf NULL.

Quelle/Modell/DS → offene Generation

Konfiguration: Quell-/Indexguards

Fehler/Grenzen: Die Lock-Transaktion kann die Dateiverarbeitung umfassen; nicht als lockfreie Architektur darstellen.

- [kiara/ingestion/document_publication.py:249 – document_publication](/wiki/architecture/code/publication#L249)
- [kiara/ingestion/document_publication.py:239 – _Publication.begin](/wiki/architecture/code/begin#L239)
- [kiara/ingestion/document_publication.py:155 – invalidate_documents](/wiki/architecture/code/invalidate#L155)

### P2 Teilstände schreiben

Neue Chunks tragen processing_pending; zugehörige Speicher und optional ColBERT werden bedient.

Chunks → noch nicht freigegebener Bestand

Konfiguration: siehe Ingestion-Karte

Fehler/Grenzen: Ein Fehler kann gespeicherte Wiederanlaufbasis hinterlassen.

- [kiara/ingestion/runtime.py:3417 – _process_file_with_publication](/wiki/architecture/code/file_loop#L3417)
- [kiara/ingestion/runtime.py:2646 – _handle_embedding_phase](/wiki/architecture/code/embedding#L2646)

### P3 Gesamtdarstellung prüfen

Alle erwarteten Chunk-IDs werden aus Qdrant gelesen, Darstellung geprüft und Version aus gespeichertem Text berechnet.

Gespeicherte Texte/Metadaten → Gesamtversion

Konfiguration: Parser-/Quellgrenzen, Chunkpositionen

Fehler/Grenzen: Fehlende/mehrdeutige Chunks verhindern den Abschluss.

- [kiara/ingestion/runtime.py:281 – _mark_file_processed](/wiki/architecture/code/mark#L281)
- [kiara/retrieval/vectorstore_qdrant.py:608 – QdrantVectorStore.finalize_document_versions](/wiki/architecture/code/versions#L608)

### P4 Suchkopien bestätigen

Kanonischen Snapshot für PG-bestätigte Owner nach ES schreiben, dann Qdrant-Version/Pending-Flag setzen und zurücklesen.

Gesamtversion → konsistente Suchkopien

Konfiguration: ES nur für zugeordnetes Modell; Qdrant wait=True

Fehler/Grenzen: ES-/Payload-Abweichung verhindert PG-Freigabe. Such-Refresh folgt zusätzlich beim Pipeline-Exit.

- [kiara/ingestion/runtime.py:1335 – _publish_search_snapshot](/wiki/architecture/code/publish_es#L1335)
- [kiara/retrieval/vectorstore_qdrant.py:608 – QdrantVectorStore.finalize_document_versions](/wiki/architecture/code/versions#L608)
- [kiara/retrieval/vectorstore_qdrant.py:660 – QdrantVectorStore._publish_document_versions](/wiki/architecture/code/qpublish#L660)
- [kiara/ingestion/pipeline.py:1447 – _refresh_search_on_exit](/wiki/architecture/code/refresh#L1447)

### P5 PG gemeinsam committen

Generation und leeres Cleanup-Inventar prüfen; Freigabe, Hash und Modellzähler in derselben PG-Transaktion schreiben.

Bestätigte Version + validierter Dateiinhalt → Abschluss

Konfiguration: embedding_model; weitere ColBERT-Zähler

Fehler/Grenzen: Geänderte Datei, abweichende Generation oder offenes Cleanup verhindert Commit.

- [kiara/ingestion/pipeline.py:690 – _complete_file](/wiki/architecture/code/complete#L690)
- [kiara/ingestion/document_publication.py:290 – complete_document_publication](/wiki/architecture/code/pg_complete#L290)
- [kiara/ingestion/_file_utils.py:47 – _LoadedFileVersion](/wiki/architecture/code/file_version#L47)

### P6 Treffer erneut prüfen

Im Retriever-Pfad werden PG-Owner und Publikationsversion bei Suche/Nachladen/Kontext validiert. Der abweichende Such-Tab ist separat dokumentiert.

Treffer → freigegebener Treffer oder Fehler

Konfiguration: DS-Scope und Collection

Fehler/Grenzen: Offene/falsche Version bricht die Anfrage ab. Ohne registrierte Publikationszeile lässt die Versionsprüfung die Quelle passieren; Ownerprüfung bleibt.

- [kiara/retrieval/retriever.py:207 – Retriever.filter_available_results](/wiki/architecture/code/available#L207)
- [kiara/ingestion/document_publication.py:321 – assert_documents_published](/wiki/architecture/code/published#L321)
- [kiara/retrieval/retriever.py:267 – Retriever.validate_context](/wiki/architecture/code/validate#L267)

## Ungeprüft und Grenzen

- Ungeprüft: Welche optionalen Pfade eine konkrete produktive Datenquelle oder Anfrage tatsächlich durchläuft. Die Karten beschreiben Codebedingungen, keine ausgelesene Produktivkonfiguration.
- Ungeprüft: End-to-End-Verhalten bei realen DDU-/Embedding-/ES-Ausfällen und konkreten Dokumenten. Dafür sind die Beispielabläufe in #416 vorgesehen.
- Aus dem Code geprüft: Ein OTel-Span kann auch bei deaktiviertem Teilverfahren entstehen. enabled/configured/attempted und Trefferzahl müssen gemeinsam gelesen werden.
- Bewusste Grenze: Spezialfälle der HTML-/Mailmigration, ERP-/Aktionswerkzeuge, Rewriter und Unchained sind nur als Abgrenzung erfasst. Unchained umgeht den RAG-Pfad; die Karten fokussieren Standard, Agent, Research und direkte Suche.
