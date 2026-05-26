"""Monorepo-internes Ingestion-Core-Package.

Dieses Package stellt nur die aus dem Hauptprojekt `kiara` extrahierten
Ingestion-Bausteine bereit. Es ist kein stabilisiertes Standalone-SDK; direkte
Abhaengigkeiten auf `kiara.shared.*`, `kiara.repositories.*` und die
Monorepo-DB-Infrastruktur sind weiterhin Teil des ehrlichen Contracts.
"""

# Re-Export der monorepo-intern genutzten Public API.
from kiara_ingestion_core.datasource_runtime import (
    build_datasource_overrides,
    load_connector_config,
)
from kiara_ingestion_core.file_diff import compute_file_hash
from kiara_ingestion_core.index_state import ChunkDatasources, HashStore
from kiara_ingestion_core.models import FileLogEntry, ProcessingContext
