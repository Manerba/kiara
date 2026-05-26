"""Admin-Funktions-Framework fuer den KIara Admin-Chat.

Re-exportiert die Kern-Interfaces. Circular-Imports sind vermieden:
function.py nutzt einen Lazy-Import fuer admin_registry im Decorator.
"""

from kiara.admin.function import AdminFunction, admin_function
from kiara.admin.registry import AdminFunctionRegistry, admin_registry, load_admin_functions
from kiara.admin.result import AdminResult
