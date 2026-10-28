"""ERP-Stammdaten-Import in den Wissensgraphen."""

import logging
import time

from kiara.graph.normalization import normalize_node_id

logger = logging.getLogger(__name__)

_ERP_SYNC_LOCK_KEY = 681030
_READ_TOOLS = (
    "erp_order_status",
    "erp_stock_level",
    "erp_customer_info",
    "erp_production_status",
)


class ERPSyncAlreadyRunningError(RuntimeError):
    """Ein anderer ERP-Graph-Sync haelt den Sync-Lock."""


class ERPSyncConfigurationError(RuntimeError):
    """ERP-Graph-Sync-Konfiguration ist inkonsistent."""


class ERPGraphSync:
    """Synchronisiert ERP-Stammdaten in den Wissensgraphen."""

    def __init__(self, config: dict, graph_store, erp_backend=None):
        self._config = config
        self._graph = graph_store
        if erp_backend is not None:
            self._erp = erp_backend
            self._registry = None
        else:
            from kiara.tools.erp_backend import ERPBackend
            from kiara.tools.erp_connectors import ConnectionRegistry

            registry = ConnectionRegistry()
            registry.load_from_config(config)
            self._registry = registry
            self._erp = ERPBackend(config, registry)

    async def sync_all(self) -> dict:
        """Importiert alle ERP-Stammdaten als Knoten + Beziehungen.

        Bulk-Fetch und ID-Preflight laufen vor der Graph-Transaktion. Erst wenn
        alle ERP-Daten konsistent geladen wurden, wird ``erp_import`` geloescht
        und neu aufgebaut.

        Returns:
            {"nodes_created": int, "edges_created": int, "duration_ms": float}
        """
        from kiara.shared.db import db_transaction

        start = time.time()
        nodes_created = 0
        edges_created = 0
        lock_conn = self._acquire_sync_lock()
        try:
            self._validate_assignment_policy()
            data = await self._fetch_bulk_data()
            self._preflight_normalized_ids(data)

            with db_transaction() as conn:
                deleted = self._graph.clear(source="erp_import", conn=conn)
                if deleted:
                    logger.info(
                        "ERP-Sync: %d alte ERP-Import-Node-Kandidaten bereinigt",
                        deleted,
                    )

                nodes_created += self._sync_customers(conn, data)
                n, e = self._sync_articles_and_suppliers(conn, data)
                nodes_created += n
                edges_created += e
                n, e = self._sync_orders(conn, data)
                nodes_created += n
                edges_created += e
                n, e = self._sync_production_orders(conn, data)
                nodes_created += n
                edges_created += e
        finally:
            self._release_sync_lock(lock_conn)

        duration_ms = (time.time() - start) * 1000
        logger.info(
            "ERP-Sync abgeschlossen: %d Knoten, %d Kanten in %.0f ms",
            nodes_created,
            edges_created,
            duration_ms,
        )
        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "duration_ms": round(duration_ms, 1),
        }

    def _acquire_sync_lock(self):
        from kiara.shared.db import get_db

        conn = get_db()
        try:
            row = conn.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired",
                (_ERP_SYNC_LOCK_KEY,),
            ).fetchone()
            conn.commit()
            if not row or not row["acquired"]:
                conn.close()
                raise ERPSyncAlreadyRunningError("ERP-Graph-Sync laeuft bereits.")
            return conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _release_sync_lock(conn) -> None:
        if conn is None:
            return
        try:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_ERP_SYNC_LOCK_KEY,))
            conn.commit()
        except Exception as exc:
            logger.warning("ERP-Sync-Lock konnte nicht freigegeben werden: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            conn.close()

    def _validate_assignment_policy(self) -> None:
        if not self._registry:
            return
        assignments = self._registry.get_tool_assignments()
        assigned = {tool: assignments.get(tool) for tool in _READ_TOOLS if assignments.get(tool)}
        if not assigned:
            return
        if len(assigned) != len(_READ_TOOLS):
            raise ERPSyncConfigurationError(
                "ERP-Graph-Sync erfordert entweder keine Tool-Zuordnung oder "
                "dieselbe Connection fuer alle vier ERP-Read-Tools."
            )
        if len(set(assigned.values())) != 1:
            raise ERPSyncConfigurationError(
                "ERP-Graph-Sync kann keine gemischten ERP-Connections verwenden."
            )

    async def _fetch_bulk_data(self) -> dict:
        import asyncio

        from kiara.tools.erp_backend import ERPUpstreamError

        try:
            return {
                "customers": await self._erp.list_all_customers(),
                "articles": await self._erp.list_all_articles(),
                "orders": await self._erp.list_all_orders(),
                "production_orders": await self._erp.list_all_production_orders(),
            }
        except ValueError as exc:
            raise ERPSyncConfigurationError(str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise ERPUpstreamError("ERP-Bulk-Timeout beim Laden der Stammdaten.") from exc

    def _preflight_normalized_ids(self, data: dict) -> None:
        seen: dict[str, tuple[str, str]] = {}
        specs = (
            ("customers", "customer", "customer_id", "name"),
            ("articles", "article", "article_id", "description"),
            ("orders", "order", "order_id", "order_id"),
            ("production_orders", "production_order", "production_order_id", "production_order_id"),
        )
        for collection, entity_type, id_key, label_key in specs:
            for row in data.get(collection, []):
                raw_id = row.get(id_key)
                label = row.get(label_key) or raw_id
                self._track_normalized_collision(
                    seen,
                    entity_type,
                    raw_id,
                    label,
                    f"{collection}.{id_key}",
                    required=False,
                )
        for article in data.get("articles", []):
            supplier = article.get("supplier")
            if supplier:
                self._track_normalized_collision(
                    seen,
                    "supplier",
                    supplier,
                    supplier,
                    "articles.supplier",
                    required=False,
                )

    def _track_normalized_collision(
        self,
        seen: dict[str, tuple[str, str]],
        entity_type: str,
        raw_id,
        label,
        context: str,
        *,
        required: bool,
    ) -> None:
        try:
            node_id = normalize_node_id(entity_type, raw_id)
        except ValueError as exc:
            if required:
                raise
            logger.warning("ERP-Preflight: %s uebersprungen: %s", context, exc)
            return
        signature = (str(raw_id), str(label or ""))
        previous = seen.get(node_id)
        if previous is not None and previous != signature:
            raise ERPSyncConfigurationError(
                f"ERP-ID-Kollision nach Normalisierung: {previous[0]!r} und "
                f"{raw_id!r} ergeben beide {node_id!r}."
            )
        seen[node_id] = signature

    @staticmethod
    def _normalize_required(entity_type: str, value, context: str) -> str | None:
        try:
            return normalize_node_id(entity_type, value)
        except ValueError as exc:
            logger.warning("%s uebersprungen (ungueltige ID): %s", context, exc)
            return None

    @staticmethod
    def _normalize_optional(entity_type: str, value, context: str) -> str | None:
        if value is None or value == "":
            return None
        try:
            return normalize_node_id(entity_type, value)
        except ValueError as exc:
            logger.warning("%s-Kante uebersprungen (ungueltige Referenz): %s", context, exc)
            return None

    def _sync_customers(self, conn, data) -> int:
        """Kunden-Knoten aus ERP-Stammdaten erstellen."""
        nodes = 0
        for customer in data.get("customers", []):
            node_id = self._normalize_required(
                "customer",
                customer.get("customer_id"),
                "Kunde",
            )
            if not node_id:
                continue
            props = {}
            address = customer.get("address") or {}
            if isinstance(address, dict) and address.get("city"):
                props["city"] = address["city"]
            if customer.get("payment_terms"):
                props["payment_terms"] = customer["payment_terms"]
            if customer.get("customer_since"):
                props["customer_since"] = customer["customer_since"]
            label = customer.get("name") or str(customer.get("customer_id"))
            self._graph.add_node(
                node_id,
                "customer",
                label,
                properties=props,
                source="erp_import",
                conn=conn,
            )
            nodes += 1
        return nodes

    def _sync_articles_and_suppliers(self, conn, data) -> tuple[int, int]:
        """Artikel-Knoten + Lieferanten-Knoten/-Kanten erstellen."""
        nodes = 0
        edges = 0
        supplier_slugs: dict[str, str] = {}
        for article in data.get("articles", []):
            art_id = self._normalize_required(
                "article",
                article.get("article_id"),
                "Artikel",
            )
            if not art_id:
                continue
            props = {
                "category": article.get("category", ""),
                "warehouse": article.get("warehouse", ""),
            }
            label = article.get("description") or str(article.get("article_id"))
            self._graph.add_node(
                art_id,
                "article",
                label,
                properties=props,
                source="erp_import",
                conn=conn,
            )
            nodes += 1

            supplier_name = article.get("supplier", "")
            if supplier_name:
                supplier_id = self._normalize_optional(
                    "supplier",
                    supplier_name,
                    "Lieferant",
                )
                if not supplier_id:
                    continue
                if supplier_id not in supplier_slugs:
                    self._graph.add_node(
                        supplier_id,
                        "supplier",
                        supplier_name,
                        source="erp_import",
                        conn=conn,
                    )
                    supplier_slugs[supplier_id] = supplier_name
                    nodes += 1
                self._graph.add_edge(supplier_id, art_id, "liefert", source="erp_import", conn=conn)
                edges += 1
        return nodes, edges

    def _sync_orders(self, conn, data) -> tuple[int, int]:
        """Auftrags-Knoten + Kunden-/Artikel-Kanten erstellen."""
        nodes = 0
        edges = 0
        for order in data.get("orders", []):
            raw_order_id = order.get("order_id")
            order_id = self._normalize_required("order", raw_order_id, "Auftrag")
            if not order_id:
                continue
            props = {
                "status": order.get("status", ""),
                "order_date": order.get("order_date", ""),
                "total_amount": order.get("total_amount", 0),
            }
            self._graph.add_node(
                order_id,
                "order",
                f"Auftrag {raw_order_id}",
                properties=props,
                source="erp_import",
                conn=conn,
            )
            nodes += 1
            customer_id = self._normalize_optional(
                "customer",
                order.get("customer_id"),
                "Auftrag.customer_id",
            )
            if customer_id:
                if self._graph.node_exists(customer_id, conn=conn):
                    self._graph.add_edge(
                        order_id, customer_id, "bestellt_bei", source="erp_import", conn=conn
                    )
                    edges += 1
                else:
                    logger.warning("Kante uebersprungen: Ziel-Node %s existiert nicht", customer_id)
            positions = order.get("positions", [])
            if not isinstance(positions, list):
                logger.warning(
                    "Auftrag %s: 'positions' ist kein list (%s), uebersprungen",
                    raw_order_id,
                    type(positions).__name__,
                )
                positions = []
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                art_id = self._normalize_optional(
                    "article",
                    pos.get("article_id"),
                    "Auftragsposition.article_id",
                )
                if art_id:
                    if self._graph.node_exists(art_id, conn=conn):
                        self._graph.add_edge(
                            order_id, art_id, "enthaelt", source="erp_import", conn=conn
                        )
                        edges += 1
                    else:
                        logger.warning("Kante uebersprungen: Ziel-Node %s existiert nicht", art_id)
        return nodes, edges

    def _sync_production_orders(self, conn, data) -> tuple[int, int]:
        """Fertigungsauftrags-Knoten + Order-Kanten erstellen."""
        nodes = 0
        edges = 0
        for po in data.get("production_orders", []):
            raw_po_id = po.get("production_order_id")
            po_id = self._normalize_required(
                "production_order",
                raw_po_id,
                "Fertigungsauftrag",
            )
            if not po_id:
                continue
            props = {
                "status": po.get("status", ""),
                "article_id": po.get("article_id", ""),
                "quantity": po.get("quantity", 0),
            }
            self._graph.add_node(
                po_id,
                "production_order",
                f"FA {raw_po_id}",
                properties=props,
                source="erp_import",
                conn=conn,
            )
            nodes += 1
            order_ref_id = self._normalize_optional(
                "order",
                po.get("order_reference"),
                "Fertigungsauftrag.order_reference",
            )
            if order_ref_id:
                if self._graph.node_exists(order_ref_id, conn=conn):
                    self._graph.add_edge(
                        po_id, order_ref_id, "produziert_fuer", source="erp_import", conn=conn
                    )
                    edges += 1
                else:
                    logger.warning(
                        "Kante uebersprungen: Ziel-Node %s existiert nicht", order_ref_id
                    )
        return nodes, edges
