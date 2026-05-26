"""Health-Check-System fuer KIara-Komponenten.

Prueft Erreichbarkeit und Status von Ollama, VectorDB, LDAP,
Cloud-KI, ERP, Datenbank und Festplatte.

Module:
    checker          — HealthChecker Orchestrator
    models           — ComponentHealth, SystemHealth Dataclasses
    checks_vectordb  — Qdrant-Checks
    checks_services  — Ollama/LDAP/Cloud-KI/ERP-Checks
    checks_infra     — Datenbank/Disk-Checks
"""

from .checker import HealthChecker
from .models import ComponentHealth, SystemHealth
