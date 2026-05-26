"""KIara Repository-Layer — zentralisierte Datenbankzugriffe.

Alle SQL-Operationen werden ueber Repository-Klassen abstrahiert.
API-Router und Business-Logic importieren nur Repositories, kein direktes SQL.

Seit der kiara-db-Extraktion sind die Implementierungen in kiara_db.
Dieses Modul re-exportiert alles fuer Rueckwaertskompatibilitaet.
"""

import kiara.shared.db
from kiara.repositories.admin_chat import (
    AdminChatMessageRepository,
    AdminChatRepository,
    AdminConfirmationRepository,
    AdminWizardRepository,
)
from kiara.repositories.api_key import ApiKeyRepository
from kiara.repositories.audit import (
    AuditAnonymizationRepository,
    CloudAuditLogRepository,
    QueryLogRepository,
)
from kiara.repositories.backend import KiBackendRepository
from kiara.repositories.base import BaseRepository, RawSQL
from kiara.repositories.cache import CacheRepository
from kiara.repositories.changelog import ConfigChangelogRepository

# Konkrete Repositories
from kiara.repositories.chat import ChatRepository, MessageRepository
from kiara.repositories.config import ConfigRepository
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.repositories.file_hash import ChunkDatasourceRepository, FileHashRepository
from kiara.repositories.graph import EdgeRepository, NodeRepository
from kiara.repositories.group import GroupRepository
from kiara.repositories.health import HealthLogRepository
from kiara.repositories.job import FileLogRepository, IngestionJobRepository
from kiara.repositories.permission import PermissionRepository
from kiara.repositories.prompt import PromptTemplateRepository
from kiara.repositories.system_events import (
    SystemEventRepository,
    _coerce_system_event_filters,
)
from kiara.repositories.user import SessionRepository, UserRepository
from kiara.repositories.wopi import WopiRepository
from kiara.repositories.workflow import WorkflowExportRepository
