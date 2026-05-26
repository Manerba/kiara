"""Resilience: Graceful Degradation bei Backend-Ausfaellen.

Verwaltet den Systemzustand bei Teilausfaellen einzelner Backends
(Ollama, VectorDB, LDAP, Cloud-KI). Der DegradationManager bestimmt,
welche Features verfuegbar sind und welche Fallbacks greifen.

Beziehungen:
    - health.checker liefert den aktuellen Status aller Komponenten
    - generation.chain nutzt Degradation fuer Cloud-Fallback
    - web.app zeigt Warnhinweise bei degradiertem Betrieb

Module:
    degradation — DegradationManager (Feature-Flags, Fallback-Logik)
"""

from .degradation import DegradationManager
