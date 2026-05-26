"""Stub — re-exportiert aus kiara_db.repositories.system_events."""

from kiara_db.repositories.system_events import (  # noqa: F401
    SystemEventRepository,
    _coerce_system_event_filters,
)

import kiara.shared.db  # noqa: F401 — DSN-Provider-Bootstrap
