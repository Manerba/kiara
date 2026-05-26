"""Domaenen-spezifische Exception-Hierarchie fuer KIara.

Alle KIara-spezifischen Fehler erben von KiaraError.
Ermoeglicht gezielte Fehlerbehandlung statt breiter 'except Exception'-Catches.

Hierarchie:
    KiaraError
    +-- ConfigurationError    # Fehlerhafte oder fehlende Konfiguration
    +-- IngestionError        # Fehler beim Laden, Chunken oder Speichern von Dokumenten
    +-- RetrievalError        # Fehler bei Vektor-Suche oder Embedding-Berechnung
    +-- GenerationError       # Fehler bei LLM-Anfrage oder RAG-Chain
    +-- ConnectorError        # Fehler bei Datenquellen-Verbindung (SMB, WebDAV, IMAP, SQL)
        +-- StatementTimeout  # SQL-Statement wurde treiber-/serverseitig abgebrochen
    +-- StorageError          # Fehler bei Datenbank- oder VectorStore-Operationen
    +-- AuthenticationError   # Fehler bei Login, Token-Validierung oder API-Key-Pruefung

Verwendung:
    from kiara.exceptions import IngestionError

    try:
        run_ingestion(config)
    except IngestionError as exc:
        logger.error("Ingestion fehlgeschlagen: %s", exc)
"""


class KiaraError(Exception):
    """Basis-Exception fuer alle KIara-spezifischen Fehler."""


class ConfigurationError(KiaraError):
    """Fehlerhafte oder fehlende Konfiguration.

    Wird geworfen wenn Config-Werte fehlen, ungueltig sind
    oder nicht geladen werden koennen.
    """


class IngestionError(KiaraError):
    """Fehler waehrend der Dokument-Ingestion.

    Umfasst Datei-Laden, Chunking, Embedding-Berechnung
    und Speicherung in der Vektordatenbank.
    """


class RetrievalError(KiaraError):
    """Fehler bei der Retrieval-Phase.

    Umfasst Vektor-Suche, Embedding-Berechnung fuer Queries,
    Reranking und Metadaten-Filterung.
    """


class GenerationError(KiaraError):
    """Fehler bei der Antwort-Generierung.

    Umfasst LLM-Anfragen (lokal und Cloud), Prompt-Erstellung,
    Streaming und Anonymisierung.
    """


class ConnectorError(KiaraError):
    """Fehler bei Datenquellen-Verbindungen.

    Umfasst SMB-Mounts, WebDAV-Zugriffe, IMAP-Verbindungen,
    SQL-Abfragen und Web-Crawling.
    """


class StatementTimeout(ConnectorError):
    """SQL-Statement wurde durch einen DB- oder Treiber-Timeout abgebrochen."""


class StorageError(KiaraError):
    """Fehler bei Datenbank- oder VectorStore-Operationen.

    Umfasst PostgreSQL-Fehler, Qdrant-Fehler,
    Migrationen und Schema-Probleme.
    """


class VectorDBConnectionError(StorageError):
    """Qdrant-Backend nicht erreichbar oder Upsert fehlgeschlagen.

    Wird geworfen, wenn isolierte Qdrant-Write-/Sync-Operationen
    (client.upsert, set_payload nach Retries) scheitern. Embedding-Fehler
    (Ollama/OpenAI/Voyage) bleiben bei BackendUnavailableError.
    """


class AuthenticationError(KiaraError):
    """Fehler bei Authentifizierung oder Autorisierung.

    Umfasst Login-Fehler, Token-Validierung, API-Key-Pruefung
    und LDAP-Authentifizierung.
    """
