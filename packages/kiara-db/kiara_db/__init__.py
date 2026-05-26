"""kiara-db — Repository-Layer und Connection-Management fuer KIara.

Standalone-Package: kapselt den gesamten Datenbank-Zugriff
(BaseRepository, 30 konkrete Repositories, Connection-Management).
Ohne weitere kiara-Dependencies nutzbar.

Schnellstart::

    from kiara_db import init_db, DatasourceRepository

    init_db("postgresql://user:pass@localhost/kiara")
    repo = DatasourceRepository()
    print(repo.list_all())
"""

# Connection-Management
# Base
from kiara_db.base import BaseRepository, RawSQL
from kiara_db.connection import (
    ConnectionWrapper,
    connect_query,
    connect_transaction,
    db_query,
    db_transaction,
    get_connection,
    get_db,
    init_db,
    register_dsn_provider,
    set_statement_timeout_ms,
)

# Konkrete Repositories
from kiara_db.repositories import (
    AdminChatMessageRepository,
    AdminChatRepository,
    AdminConfirmationRepository,
    AdminWizardRepository,
    ApiKeyRepository,
    AuditAnonymizationRepository,
    CacheRepository,
    ChatRepository,
    ChunkDatasourceRepository,
    CloudAuditLogRepository,
    ConfigChangelogRepository,
    ConfigRepository,
    DatasourceEmbeddingRepository,
    DatasourceRepository,
    EdgeRepository,
    FileHashRepository,
    FileLogRepository,
    GroupRepository,
    HealthLogRepository,
    IngestionJobRepository,
    KiBackendRepository,
    MessageRepository,
    NodeRepository,
    PermissionRepository,
    PromptTemplateRepository,
    QueryLogRepository,
    SessionRepository,
    SystemEventRepository,
    UserRepository,
    WopiRepository,
    WorkflowExportRepository,
    _coerce_system_event_filters,
)
