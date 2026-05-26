"""Gemeinsam genutzte Utilities fuer Core- und Web-Module.

Dieses Paket enthaelt Infrastruktur-Code, der sowohl von Core-Modulen
(anonymization, generation, ingestion, graph, cloud) als auch von
Web-Modulen (web/api_*, web/auth) benoetigt wird.

Durch die Trennung in ein eigenes Paket wird vermieden, dass Core-Module
aus dem Web-Layer importieren muessen (Clean-Architecture-Verletzung).

Hinweis: ``__all__`` wird projektweit nicht verwendet. Oeffentliche API
wird ueber die Underscore-Konvention (``_name`` = intern) markiert.

Submodule:
    db      - PostgreSQL-Verbindungsverwaltung, Context-Manager, SQL-Helpers
    crypto  - Fernet-Verschluesselung, Secret-Key-Verwaltung
    mount   - Generische Filesystem-Mount-/Unmount-Operationen
    ldap    - LDAP-Verbindung, Gruppen, Authentifizierung (ldap3)
    enums       - Zentrale Enum-Definitionen (IngestionStatus, HealthStatus, RoutingMode)
    constants   - Benannte Konstanten (RRF_K_DEFAULT, Sentinels, Cloud-Keywords)
    date_utils  - Datums-Hilfsfunktionen
    cron_utils  - Cron-Schedule-Verwaltung (/etc/cron.d/kiara)
"""
