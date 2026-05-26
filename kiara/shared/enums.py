"""Zentrale Enum-Definitionen fuer KIara.

Alle Status-Werte, die als Strings in der Codebasis verwendet werden,
sind hier als ``str``-Enums definiert.  Durch Vererbung von ``str``
sind die Werte direkt JSON-serialisierbar und abwaertskompatibel
mit bestehendem Code, der String-Vergleiche nutzt.

Verwendung::

    from kiara.shared.enums import IngestionStatus

    job.status = IngestionStatus.RUNNING
    if job.status == "running":   # funktioniert weiterhin
        ...
"""

from __future__ import annotations

from enum import Enum


class IngestionStatus(str, Enum):
    """Status eines Ingestion-Jobs."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    IDLE = "idle"
    PAUSED = "paused"


class HealthStatus(str, Enum):
    """Status einer Health-Check-Komponente."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNCONFIGURED = "unconfigured"


class RoutingMode(str, Enum):
    """Cloud/Local-Routing-Modus fuer die RAG-Chain."""

    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"


class ChatMode(str, Enum):
    """Chat-Modus: Standard-RAG, Unchained (freies LLM) oder Agent (Tool-Use)."""

    STANDARD = "standard"
    REWRITER = "rewriter"
    UNCHAINED = "unchained"
    AGENT = "agent"


class ConnectorType(str, Enum):
    """Typ eines Datenquellen-Konnektors.

    ``(str, Enum)`` erlaubt direkten String-Vergleich::

        ConnectorType.LOCAL == "local"   # True
        "imap" == ConnectorType.IMAP     # True

    Damit bleibt bestehender Code, der den Typ als String vergleicht,
    vollstaendig kompatibel.
    """

    LOCAL = "local"
    SMB = "smb"
    WEBDAV = "webdav"
    IMAP = "imap"
    EXCHANGE_ONLINE = "exchange_online"
    SQL = "sql"
