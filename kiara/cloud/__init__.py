"""Cloud-AI-Integration: Client-Anbindung und Audit-Logging.

Ermoeglicht die Nutzung externer KI-Dienste (OpenAI, Anthropic) als
Ergaenzung zum lokalen Ollama-Backend. Der CloudKIClient kapselt die
HTTP-Kommunikation inkl. Retry und Streaming.

Beziehungen:
    - generation.cloud_routing nutzt CloudKIClient fuer Admin-Cloud-Funktionen
    - anonymization sorgt fuer Datenschutz vor Cloud-Versand

Module:
    client  — CloudKIClient (OpenAI/Anthropic HTTP-Client)
    audit   — AuditLogger (Cloud-Anfragen protokollieren)
"""

from .audit import AuditLogger
from .client import CloudKIClient
