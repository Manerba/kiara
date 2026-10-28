"""Stub — re-exportiert aus kiara_db.repositories.datasource."""

from kiara_db.repositories.datasource import (  # noqa: F401
    DatasourceEmbeddingRepository,
    DatasourceRepository,
)

import kiara.shared.db  # noqa: F401 — DSN-Provider-Bootstrap
