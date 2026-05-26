"""ERP-Konnektor-Schicht fuer KIara.

Stellt die abstrakte Basis, eine Connection-Registry und konkrete
Konnektoren (Odoo, Generic SQL) bereit. Registriert die Konnektor-Typen
beim Import automatisch.
"""

from kiara.tools.erp_connectors.base import (
    ConnectionInfo,
    ConnectorDefinition,
    ERPConnectorBase,
)
from kiara.tools.erp_connectors.generic_sql import GenericSQLConnector
from kiara.tools.erp_connectors.odoo import OdooConnector
from kiara.tools.erp_connectors.registry import ConnectionRegistry

# Konnektor-Typen registrieren
ConnectionRegistry.register_type(OdooConnector)
ConnectionRegistry.register_type(GenericSQLConnector)
