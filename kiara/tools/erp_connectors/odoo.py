"""Odoo-Konnektor ueber XML-RPC API.

Implementiert die 4 ERP-Query-Methoden gegen eine Odoo-Instanz
(getestet mit Odoo 19.0). Nutzt ``xmlrpc.client.ServerProxy``
und wickelt die synchronen Aufrufe ueber ``asyncio.to_thread()`` ab.
"""

import asyncio
import logging
import threading
from xmlrpc.client import Fault, SafeTransport, ServerProxy, Transport

from kiara.shared.constants import CONNECTOR_XMLRPC_TIMEOUT, ERP_ORDERS_MAX_SCAN
from kiara.tools.erp_connectors.base import (
    ConnectionInfo,
    ConnectorDefinition,
    ERPConnectorBase,
)
from kiara.tools.erp_order_query import (
    ERPOrderQuery,
    apply_order_query,
)

logger = logging.getLogger(__name__)

# Status-Mappings: Odoo-State → KIara-Status
_SALE_ORDER_STATE_MAP: dict[str, str] = {
    "draft": "offen",
    "sent": "offen",
    "sale": "offen",
    "done": "geliefert",
    "locked": "geliefert",
    "cancel": "storniert",
}

_DELIVERY_STATUS_MAP: dict[str, str] = {
    "pending": "nicht_geliefert",
    "started": "in_auslieferung",
    "partial": "teilweise_geliefert",
    "full": "geliefert",
}

_ORDER_READ_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "partner_id",
    "state",
    "date_order",
    "commitment_date",
    "effective_date",
    "delivery_status",
    "amount_total",
    "currency_id",
    "order_line",
    "picking_ids",
)

_ORDER_FILTER_FIELD_MAP: dict[str, str] = {
    "order_id": "name",
    "customer_id": "partner_id.ref",
    "customer_name": "partner_id.name",
    "delivery_status": "delivery_status",
    "order_date": "date_order",
    "promised_delivery_date": "commitment_date",
    "total_amount": "amount_total",
}

_ORDER_SORT_FIELD_MAP: dict[str, str] = {
    "order_id": "name",
    "order_date": "date_order",
    "promised_delivery_date": "commitment_date",
    "total_amount": "amount_total",
}

_ODOO_OPERATOR_MAP: dict[str, str] = {
    "eq": "=",
    "ne": "!=",
    "contains": "ilike",
    "in": "in",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}

_PRODUCTION_STATE_MAP: dict[str, str] = {
    "draft": "geplant",
    "confirmed": "geplant",
    "progress": "in_bearbeitung",
    "to_close": "in_bearbeitung",
    "done": "abgeschlossen",
    "cancel": "storniert",
}

_WORKORDER_STATE_MAP: dict[str, str] = {
    "pending": "offen",
    "ready": "offen",
    "progress": "in_bearbeitung",
    "done": "abgeschlossen",
    "cancel": "storniert",
}

_XMLRPC_TIMEOUT = CONNECTOR_XMLRPC_TIMEOUT


class _TimeoutTransport(Transport):
    """XML-RPC Transport mit konfigurierbarem Timeout."""

    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


class _TimeoutSafeTransport(SafeTransport):
    """XML-RPC SafeTransport (HTTPS) mit konfigurierbarem Timeout."""

    def __init__(self, timeout: float):
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


def _make_proxy(url: str, timeout: float = _XMLRPC_TIMEOUT) -> ServerProxy:
    """Erstellt einen ServerProxy mit Timeout-Transport."""
    transport_cls = _TimeoutSafeTransport if url.lower().startswith("https") else _TimeoutTransport
    return ServerProxy(url, transport=transport_cls(timeout), allow_none=True)


class OdooConnector(ERPConnectorBase):
    """Odoo-Konnektor ueber XML-RPC API."""

    def __init__(self, connection: ConnectionInfo) -> None:
        super().__init__(connection)
        self._url: str = connection.config.get("url", "").rstrip("/")
        if not self._url:
            raise ValueError("Odoo-URL darf nicht leer sein (config key 'url')")
        self._database: str = connection.config.get("database", "")
        self._username: str = connection.config.get("username", "")
        self._password: str = connection.config.get("password", "")
        self._common: ServerProxy | None = None
        self._models: ServerProxy | None = None
        self._uid: int | None = None
        self._lock = threading.Lock()

    @classmethod
    def connector_definition(cls) -> ConnectorDefinition:
        return ConnectorDefinition(
            type_id="odoo",
            display_name="Odoo (XML-RPC)",
            config_fields=[
                {
                    "name": "url",
                    "label": "URL",
                    "type": "string",
                    "placeholder": "http://odoo.example.com:8069",
                    "required": True,
                },
                {
                    "name": "database",
                    "label": "Datenbank",
                    "type": "string",
                    "placeholder": "kiara",
                    "required": True,
                },
                {
                    "name": "username",
                    "label": "Benutzer",
                    "type": "string",
                    "placeholder": "admin",
                    "required": True,
                },
                {
                    "name": "password",
                    "label": "Passwort / API-Key",
                    "type": "password",
                    "required": True,
                },
            ],
        )

    def _ensure_authenticated(self) -> None:
        """Authentifiziert gegen Odoo (synchron). Cacht UID."""
        if self._uid is not None:
            return
        self._common = _make_proxy(f"{self._url}/xmlrpc/2/common")
        self._uid = self._common.authenticate(self._database, self._username, self._password, {})
        if not self._uid:
            raise ConnectionError(
                f"Odoo-Authentifizierung fehlgeschlagen fuer '{self._username}' "
                f"auf '{self._url}' (DB: '{self._database}')."
            )
        self._models = _make_proxy(f"{self._url}/xmlrpc/2/object")

    def _execute(self, model: str, method: str, *args) -> list | dict | int | bool:
        """Ruft ``execute_kw`` auf dem Odoo-Models-Endpoint auf (synchron)."""
        with self._lock:
            self._ensure_authenticated()
            return self._models.execute_kw(
                self._database,
                self._uid,
                self._password,
                model,
                method,
                *args,
            )

    def _search_read(
        self,
        model: str,
        domain: list,
        fields: list[str],
        limit: int = 0,
        offset: int = 0,
        order: str = "",
    ) -> list[dict]:
        """Convenience-Wrapper fuer search_read."""
        kwargs: dict = {"fields": fields}
        if limit:
            kwargs["limit"] = limit
        if offset:
            kwargs["offset"] = offset
        if order:
            kwargs["order"] = order
        return self._execute(model, "search_read", [domain], kwargs)

    def _paged_search_read(
        self,
        model: str,
        domain: list,
        fields: list[str],
        *,
        page_size: int = 500,
        order: str = "id",
    ) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            page = self._search_read(
                model,
                domain,
                fields,
                limit=page_size,
                offset=offset,
                order=order,
            )
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return rows

    @staticmethod
    def _m2o_id(value) -> str:
        """Extrahiert ID aus einem many2one-Feld (``[id, name]`` oder ``False``)."""
        if isinstance(value, (list, tuple)) and len(value) >= 1:
            return str(value[0])
        return ""

    @staticmethod
    def _m2o_name(value) -> str:
        """Extrahiert Display-Name aus einem many2one-Feld."""
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            name = value[1]
            return str(name) if name else ""
        return ""

    @staticmethod
    def _odoo_date(value) -> str:
        """Konvertiert Odoo-Datum (str oder False) in String."""
        if value and value is not False:
            return str(value)
        return ""

    @staticmethod
    def _relation_id(value) -> int | None:
        """Liest eine numerische Odoo-Many2one-ID defensiv aus."""
        if isinstance(value, (list, tuple)) and value:
            try:
                return int(value[0])
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _order_status(order: dict, production_origins: set[str] | None = None) -> str:
        """Priorisiert den echten Lieferstatus vor dem Verkaufsworkflow-State."""
        delivery_status = str(order.get("delivery_status") or "")
        if delivery_status in _DELIVERY_STATUS_MAP and delivery_status != "pending":
            return _DELIVERY_STATUS_MAP[delivery_status]

        state = str(order.get("state") or "")
        name = str(order.get("name") or "")
        if state == "sale" and production_origins and name in production_origins:
            return "in_fertigung"
        return _SALE_ORDER_STATE_MAP.get(state, "offen")

    def _build_order_record(
        self,
        order: dict,
        *,
        customer_id: str = "",
        production_origins: set[str] | None = None,
        delivery_dates: dict[int, dict[str, str | None]] | None = None,
        positions: list[dict] | None = None,
    ) -> dict:
        """Baut den normalisierten Auftragsvertrag aus einem Odoo-Record."""
        sale_id = int(order.get("id") or 0)
        dates = (delivery_dates or {}).get(sale_id, {})
        promised = self._odoo_date(order.get("commitment_date")) or None
        first = dates.get("first") or self._odoo_date(order.get("effective_date")) or None
        actual = dates.get("actual") or None
        result = {
            "order_id": order.get("name", ""),
            "customer_id": customer_id or self._m2o_id(order.get("partner_id")),
            "customer_name": self._m2o_name(order.get("partner_id")),
            "status": self._order_status(order, production_origins),
            "delivery_status": order.get("delivery_status") or None,
            "order_date": self._odoo_date(order.get("date_order")) or None,
            # delivery_date bleibt vorerst der zugesagte Termin fuer bestehende
            # Graph-/SQL-Vertraege. Neue Agent-Abfragen nutzen nur die expliziten Felder.
            "delivery_date": promised,
            "promised_delivery_date": promised,
            "first_delivery_date": first,
            "actual_delivery_date": actual,
            "total_amount": order.get("amount_total", 0),
            "currency": self._m2o_name(order.get("currency_id")) or None,
            "position_count": len(order.get("order_line") or []),
        }
        if positions is not None:
            result["positions"] = positions
        return result

    @staticmethod
    def _delivery_domain(sale_ids: list[int] | None = None) -> list:
        domain: list = [
            ["state", "=", "done"],
            ["picking_type_code", "=", "outgoing"],
            ["sale_id", "!=", False],
        ]
        if sale_ids is not None:
            domain.append(["sale_id", "in", sale_ids])
        return domain

    def _read_delivery_aggregate(
        self,
        sale_ids: list[int],
        aggregate: str,
    ) -> dict[int, str]:
        """Gruppiert Ist-Transferdaten in begrenzten Odoo-Batches."""
        if aggregate not in {"min", "max"}:
            raise ValueError("Unbekanntes Lieferdatum-Aggregat.")
        result: dict[int, str] = {}
        batch_size = 500
        for start in range(0, len(sale_ids), batch_size):
            batch = sale_ids[start : start + batch_size]
            groups = self._execute(
                "stock.picking",
                "read_group",
                [
                    self._delivery_domain(batch),
                    ["sale_id", f"date_done:{aggregate}"],
                    ["sale_id"],
                ],
                {"lazy": False, "limit": len(batch)},
            )
            if not isinstance(groups, list):
                raise RuntimeError("Odoo read_group lieferte keine Liste.")
            for group in groups:
                sale_id = self._relation_id(group.get("sale_id"))
                date_done = self._odoo_date(group.get("date_done"))
                if sale_id is not None and date_done:
                    result[sale_id] = date_done
        return result

    def _delivery_dates_by_sale_ids(
        self,
        sale_ids: list[int],
        *,
        known_first: dict[int, str] | None = None,
        known_actual: dict[int, str] | None = None,
    ) -> dict[int, dict[str, str | None]]:
        unique_ids = list(dict.fromkeys(sale_id for sale_id in sale_ids if sale_id > 0))
        if not unique_ids:
            return {}
        first = known_first or self._read_delivery_aggregate(unique_ids, "min")
        actual = known_actual or self._read_delivery_aggregate(unique_ids, "max")
        return {
            sale_id: {
                "first": first.get(sale_id),
                "actual": actual.get(sale_id),
            }
            for sale_id in unique_ids
        }

    @staticmethod
    def _odoo_filter_value(value):
        if value is None:
            return False
        if isinstance(value, list):
            return [OdooConnector._odoo_filter_value(item) for item in value]
        if isinstance(value, str):
            return value.replace("T", " ").replace("Z", "")
        return value

    def _order_domain_from_query(self, query: ERPOrderQuery) -> list:
        """Uebersetzt nur explizit erlaubte Filter auf eine Odoo-Domain."""
        domain: list = []
        for item in query.filters:
            odoo_field = _ORDER_FILTER_FIELD_MAP.get(item.field)
            if not odoo_field:
                # status und actual_delivery_date werden nach Normalisierung
                # innerhalb der harten Scan-Grenze ausgewertet.
                continue
            domain.append(
                [
                    odoo_field,
                    _ODOO_OPERATOR_MAP[item.operator],
                    self._odoo_filter_value(item.value),
                ]
            )
        return domain

    def _delivery_sorted_candidates(
        self,
        direction: str,
        aggregate: str,
    ) -> tuple[list[int], dict[int, str], bool]:
        """Liest nach erstem/letztem Ist-Transfer sortierte IDs, hart begrenzt."""
        if aggregate not in {"min", "max"}:
            raise ValueError("Unbekanntes Lieferdatum-Aggregat.")
        groups = self._execute(
            "stock.picking",
            "read_group",
            [
                self._delivery_domain(),
                ["sale_id", f"date_done:{aggregate}"],
                ["sale_id"],
            ],
            {
                "lazy": False,
                "limit": ERP_ORDERS_MAX_SCAN,
                "orderby": f"date_done:{aggregate} {direction}",
            },
        )
        if not isinstance(groups, list):
            raise RuntimeError("Odoo read_group lieferte keine Liste.")
        sale_ids: list[int] = []
        actual: dict[int, str] = {}
        for group in groups:
            sale_id = self._relation_id(group.get("sale_id"))
            date_done = self._odoo_date(group.get("date_done"))
            if sale_id is not None and date_done:
                sale_ids.append(sale_id)
                actual[sale_id] = date_done
        return sale_ids, actual, len(groups) >= ERP_ORDERS_MAX_SCAN

    # ------------------------------------------------------------------
    # Interface-Implementierung
    # ------------------------------------------------------------------

    async def test_connection(self) -> dict:
        """Authentifizierung + Version pruefen."""
        try:

            def _test() -> dict:
                common = _make_proxy(f"{self._url}/xmlrpc/2/common")
                version_info = common.version()
                uid = common.authenticate(self._database, self._username, self._password, {})
                if not uid:
                    return {
                        "success": False,
                        "message": "Authentifizierung fehlgeschlagen.",
                        "details": {},
                    }
                server_version = version_info.get("server_version", "unbekannt")
                return {
                    "success": True,
                    "message": f"Verbunden mit Odoo {server_version}",
                    "details": {
                        "version": server_version,
                        "uid": uid,
                    },
                }

            return await asyncio.to_thread(_test)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Verbindungsfehler: {exc}",
                "details": {},
            }

    async def query_order(self, order_id: str) -> dict | None:
        """sale.order + sale.order.line abfragen."""

        def _query() -> dict | None:
            orders = self._search_read(
                "sale.order",
                [["name", "=", order_id]],
                list(_ORDER_READ_FIELDS),
                limit=1,
            )
            if not orders:
                return None

            order = orders[0]

            # Positionen laden
            line_ids = order.get("order_line", [])
            positions = []
            if line_ids:
                lines = self._search_read(
                    "sale.order.line",
                    [["id", "in", line_ids]],
                    ["sequence", "product_id", "name", "product_uom_qty", "price_unit"],
                )
                # default_code (Artikelnummer) per Batch laden
                _pids = [
                    l["product_id"][0]
                    for l in lines
                    if isinstance(l.get("product_id"), (list, tuple))
                ]
                _code_map = {}
                if _pids:
                    _prods = self._search_read(
                        "product.product",
                        [["id", "in", _pids]],
                        ["default_code"],
                    )
                    _code_map = {p["id"]: p.get("default_code") or str(p["id"]) for p in _prods}
                for line in lines:
                    _pid = line.get("product_id")
                    _pid_int = _pid[0] if isinstance(_pid, (list, tuple)) else 0
                    positions.append(
                        {
                            "pos": line.get("sequence", 0),
                            "article_id": _code_map.get(_pid_int) or str(_pid_int),
                            "description": line.get("name", ""),
                            "quantity": line.get("product_uom_qty", 0),
                            "unit_price": line.get("price_unit", 0),
                        }
                    )

            sale_id = int(order.get("id") or 0)
            partner_id = int(self._m2o_id(order.get("partner_id")) or 0)
            customer_ids = self._bulk_partner_ref_map([partner_id]) if partner_id else {}
            production_origins = self._bulk_production_origins([str(order.get("name") or "")])
            delivery_dates = self._delivery_dates_by_sale_ids([sale_id]) if sale_id else {}
            return self._build_order_record(
                order,
                customer_id=customer_ids.get(partner_id, ""),
                production_origins=production_origins,
                delivery_dates=delivery_dates,
                positions=positions,
            )

        return await asyncio.to_thread(_query)

    async def query_orders(self, query: ERPOrderQuery) -> dict:
        """Fuehrt den validierten Agent-Query serverseitig begrenzt aus."""

        def _query() -> dict:
            domain = self._order_domain_from_query(query)
            has_post_filter = any(
                item.field in {"status", "first_delivery_date", "actual_delivery_date"}
                for item in query.filters
            )
            known_first: dict[int, str] | None = None
            known_actual: dict[int, str] | None = None
            scan_truncated = False

            if query.sort_by in {"first_delivery_date", "actual_delivery_date"}:
                aggregate = "min" if query.sort_by == "first_delivery_date" else "max"
                candidate_ids, known_dates, scan_truncated = self._delivery_sorted_candidates(
                    query.sort_direction,
                    aggregate,
                )
                if aggregate == "min":
                    known_first = known_dates
                else:
                    known_actual = known_dates
                if not candidate_ids:
                    return apply_order_query([], query)
                orders = self._search_read(
                    "sale.order",
                    [*domain, ["id", "in", candidate_ids]],
                    list(_ORDER_READ_FIELDS),
                    limit=ERP_ORDERS_MAX_SCAN,
                )
            else:
                if has_post_filter:
                    fetch_limit = ERP_ORDERS_MAX_SCAN
                else:
                    # Ab Offset 0 lesen und erst nach Normalisierung paginieren.
                    # Das zusaetzliche Element ermittelt has_more ohne COUNT.
                    fetch_limit = min(
                        query.offset + query.limit + 1,
                        ERP_ORDERS_MAX_SCAN,
                    )
                order_field = _ORDER_SORT_FIELD_MAP[query.sort_by]
                orders = self._search_read(
                    "sale.order",
                    domain,
                    list(_ORDER_READ_FIELDS),
                    limit=fetch_limit,
                    order=f"{order_field} {query.sort_direction}, id {query.sort_direction}",
                )
                scan_truncated = has_post_filter and len(orders) >= ERP_ORDERS_MAX_SCAN

            sale_ids = [int(order.get("id") or 0) for order in orders if order.get("id")]
            partner_ids = [
                int(self._m2o_id(order.get("partner_id")))
                for order in orders
                if self._m2o_id(order.get("partner_id"))
            ]
            names = [str(order.get("name") or "") for order in orders]
            customer_ids = self._bulk_partner_ref_map(partner_ids)
            production_origins = self._bulk_production_origins(names)
            delivery_dates = self._delivery_dates_by_sale_ids(
                sale_ids,
                known_first=known_first,
                known_actual=known_actual,
            )

            rows = []
            for order in orders:
                partner_id = int(self._m2o_id(order.get("partner_id")) or 0)
                rows.append(
                    self._build_order_record(
                        order,
                        customer_id=customer_ids.get(partner_id, ""),
                        production_origins=production_origins,
                        delivery_dates=delivery_dates,
                    )
                )
            return apply_order_query(rows, query, scan_truncated=scan_truncated)

        return await asyncio.to_thread(_query)

    def _load_supplier_info(self, product: dict) -> tuple[str, int]:
        """Laedt Lieferant und Lieferzeit fuer ein Produkt.

        Returns:
            (supplier_name, lead_time_days). Bei Fehler ("", 0).
        """
        try:
            tmpl_id = self._m2o_id(product.get("product_tmpl_id"))
            if tmpl_id:
                suppliers = self._search_read(
                    "product.supplierinfo",
                    [["product_tmpl_id", "=", int(tmpl_id)]],
                    ["partner_id", "delay"],
                    limit=1,
                    order="sequence,id",
                )
                if suppliers:
                    return (
                        self._m2o_name(suppliers[0].get("partner_id")),
                        int(suppliers[0].get("delay", 0)),
                    )
        except Exception as exc:
            logger.debug("Lieferanten-Abfrage fehlgeschlagen: %s", exc)
        return "", 0

    def _load_min_stock(self, product_id: int) -> float:
        """Laedt Mindestbestand aus stock.warehouse.orderpoint."""
        try:
            orderpoints = self._search_read(
                "stock.warehouse.orderpoint",
                [["product_id", "=", product_id]],
                ["product_min_qty"],
                limit=1,
            )
            if orderpoints:
                return orderpoints[0].get("product_min_qty", 0)
        except Exception as exc:
            logger.debug("Orderpoint-Abfrage fehlgeschlagen: %s", exc)
        return 0.0

    def _load_warehouse_location(self, product_id: int) -> str:
        """Laedt primaeren Lagerort (interner Quant mit hoechster Menge)."""
        try:
            quants = self._search_read(
                "stock.quant",
                [
                    ["product_id", "=", product_id],
                    ["location_id.usage", "=", "internal"],
                ],
                ["location_id", "quantity"],
                limit=1,
                order="quantity desc",
            )
            if quants:
                return self._m2o_name(quants[0].get("location_id"))
        except Exception as exc:
            logger.debug("Quant-Abfrage fehlgeschlagen: %s", exc)
        return ""

    async def query_stock(self, article_id: str) -> dict | None:
        """product.product + stock.quant abfragen."""

        def _query() -> dict | None:
            products = self._search_read(
                "product.product",
                [["default_code", "=", article_id]],
                [
                    "default_code",
                    "name",
                    "categ_id",
                    "uom_id",
                    "qty_available",
                    "list_price",
                    "product_tmpl_id",
                ],
                limit=1,
            )
            if not products:
                return None

            product = products[0]
            product_id = product["id"]
            supplier, lead_time_days = self._load_supplier_info(product)

            return {
                "article_id": product.get("default_code", article_id),
                "description": product.get("name", ""),
                "category": self._m2o_name(product.get("categ_id")),
                "unit": self._m2o_name(product.get("uom_id")),
                "stock_quantity": product.get("qty_available", 0),
                "min_stock": self._load_min_stock(product_id),
                "warehouse": self._load_warehouse_location(product_id),
                "unit_price": product.get("list_price", 0),
                "supplier": supplier,
                "lead_time_days": lead_time_days,
            }

        return await asyncio.to_thread(_query)

    _CUSTOMER_FIELDS = [
        "id",
        "ref",
        "name",
        "street",
        "zip",
        "city",
        "country_id",
        "phone",
        "email",
        "create_date",
        "property_payment_term_id",
        "child_ids",
    ]

    def _search_partner(
        self,
        customer_id: str,
        query: str | None,
    ) -> dict | None:
        """Sucht res.partner per Freitext, Referenz, ID oder Name-Fallback."""
        fields = self._CUSTOMER_FIELDS
        customer_id = (customer_id or "").strip()
        query = (query or "").strip() or None
        if query:
            partners = self._search_read(
                "res.partner",
                [["is_company", "=", True], ["name", "ilike", query]],
                fields,
                limit=1,
            )
        else:
            if not customer_id:
                return None
            partners = self._search_read(
                "res.partner", [["ref", "=", customer_id]], fields, limit=1
            )
            if not partners:
                try:
                    partner_int_id = int(customer_id)
                    partners = self._search_read(
                        "res.partner", [["id", "=", partner_int_id]], fields, limit=1
                    )
                except (ValueError, TypeError):
                    partners = self._search_read(
                        "res.partner",
                        [["is_company", "=", True], ["name", "ilike", customer_id]],
                        fields,
                        limit=1,
                    )
        return partners[0] if partners else None

    def _load_contact_info(self, partner: dict) -> dict[str, str]:
        """Laedt primaere Kontaktperson (erstes Kind mit type=contact)."""
        contact = {
            "name": partner.get("name", ""),
            "phone": partner.get("phone", "") or "",
            "email": partner.get("email", "") or "",
        }
        child_ids = partner.get("child_ids", [])
        if child_ids:
            try:
                children = self._search_read(
                    "res.partner",
                    [["id", "in", child_ids], ["type", "=", "contact"]],
                    ["name", "phone", "email"],
                    limit=1,
                )
                if children:
                    contact["name"] = children[0].get("name", contact["name"])
                    contact["phone"] = children[0].get("phone", "") or contact["phone"]
                    contact["email"] = children[0].get("email", "") or contact["email"]
            except Exception as exc:
                logger.debug("Kontaktperson-Abfrage fehlgeschlagen: %s", exc)
        return contact

    def _load_credit_limit(self, partner_id: int) -> float:
        """Laedt credit_limit (optional, Modul account_credit_limit)."""
        try:
            cl_data = self._search_read(
                "res.partner", [["id", "=", partner_id]], ["credit_limit"], limit=1
            )
            if cl_data:
                return cl_data[0].get("credit_limit", 0.0) or 0.0
        except (Fault, Exception) as exc:
            logger.debug("Credit-Limit nicht lesbar (optional): %s", exc)
        return 0.0

    def _load_recent_orders(self, partner_id: int, limit: int = 5) -> list[str]:
        """Laedt die letzten N Auftragsnummern eines Partners."""
        try:
            recent = self._search_read(
                "sale.order",
                [["partner_id", "=", partner_id]],
                ["name"],
                limit=limit,
                order="date_order desc",
            )
            return [o["name"] for o in recent]
        except Exception as exc:
            logger.debug("Auftragshistorie-Abfrage fehlgeschlagen: %s", exc)
        return []

    async def query_customer(self, customer_id: str, query: str | None = None) -> dict | None:
        """res.partner abfragen (ID, Referenz oder Freitext-Suche ueber query)."""

        def _query() -> dict | None:
            partner = self._search_partner(customer_id, query)
            if not partner:
                return None

            contact = self._load_contact_info(partner)
            customer_id_val = partner.get("ref") or str(partner.get("id", ""))

            return {
                "customer_id": customer_id_val,
                "name": partner.get("name", ""),
                "address": {
                    "street": partner.get("street", "") or "",
                    "zip": partner.get("zip", "") or "",
                    "city": partner.get("city", "") or "",
                    "country": self._m2o_name(partner.get("country_id")),
                },
                "contact": contact,
                "customer_since": self._odoo_date(partner.get("create_date")),
                "payment_terms": self._m2o_name(partner.get("property_payment_term_id")),
                "credit_limit": self._load_credit_limit(partner["id"]),
                "last_orders": self._load_recent_orders(partner["id"]),
            }

        return await asyncio.to_thread(_query)

    async def query_production(self, order_id: str) -> dict | None:
        """mrp.production + mrp.workorder abfragen."""

        def _query() -> dict | None:
            productions = self._search_read(
                "mrp.production",
                [["name", "=", order_id]],
                [
                    "name",
                    "origin",
                    "product_id",
                    "product_qty",
                    "state",
                    "date_start",
                    "date_deadline",
                    "workorder_ids",
                ],
                limit=1,
            )
            if not productions:
                return None

            prod = productions[0]

            # Arbeitsgaenge laden
            wo_ids = prod.get("workorder_ids", [])
            operations = []
            if wo_ids:
                try:
                    workorders = self._search_read(
                        "mrp.workorder",
                        [["id", "in", wo_ids]],
                        [
                            "sequence",
                            "name",
                            "workcenter_id",
                            "state",
                            "duration_expected",
                            "duration",
                        ],
                        order="sequence",
                    )
                    for wo in workorders:
                        planned_min = wo.get("duration_expected", 0) or 0
                        actual_min = wo.get("duration", 0) or 0
                        wo_state = wo.get("state", "")
                        operations.append(
                            {
                                "op_number": wo.get("sequence", 0),
                                "name": wo.get("name", ""),
                                "workstation": self._m2o_name(wo.get("workcenter_id")),
                                "status": _WORKORDER_STATE_MAP.get(wo_state, "offen"),
                                "planned_hours": round(planned_min / 60, 2),
                                "actual_hours": round(actual_min / 60, 2) if actual_min else None,
                            }
                        )
                except Exception as exc:
                    logger.debug("Workorder-Abfrage fehlgeschlagen: %s", exc)

            state = prod.get("state", "")

            # default_code (Artikelnummer) laden
            _prod_pid = prod.get("product_id")
            _article_code = ""
            if isinstance(_prod_pid, (list, tuple)) and _prod_pid:
                _pp = self._search_read(
                    "product.product",
                    [["id", "=", _prod_pid[0]]],
                    ["default_code"],
                    limit=1,
                )
                if _pp:
                    _article_code = _pp[0].get("default_code") or str(_prod_pid[0])

            return {
                "production_order_id": prod.get("name", order_id),
                "order_reference": prod.get("origin", "") or "",
                "article_id": _article_code,
                "article_description": self._m2o_name(prod.get("product_id")),
                "quantity": prod.get("product_qty", 0),
                "status": _PRODUCTION_STATE_MAP.get(state, "geplant"),
                "start_date": self._odoo_date(prod.get("date_start")),
                "planned_end_date": self._odoo_date(prod.get("date_deadline")),
                "operations": operations,
            }

        return await asyncio.to_thread(_query)

    # ------------------------------------------------------------------
    # Bulk-Interface fuer Graph-Sync
    # ------------------------------------------------------------------

    async def list_all_customers(self) -> list[dict]:
        def _query() -> list[dict]:
            partners = self._paged_search_read(
                "res.partner",
                [["customer_rank", ">", 0]],
                self._CUSTOMER_FIELDS,
                order="id",
            )
            result = []
            for partner in partners:
                customer_id = partner.get("ref") or str(partner.get("id", ""))
                if not customer_id:
                    continue
                result.append(
                    {
                        "customer_id": customer_id,
                        "name": partner.get("name", ""),
                        "address": {
                            "street": partner.get("street", "") or "",
                            "zip": partner.get("zip", "") or "",
                            "city": partner.get("city", "") or "",
                            "country": self._m2o_name(partner.get("country_id")),
                        },
                        "customer_since": self._odoo_date(partner.get("create_date")),
                        "payment_terms": self._m2o_name(partner.get("property_payment_term_id")),
                    }
                )
            return result

        return await asyncio.to_thread(_query)

    async def list_all_articles(self) -> list[dict]:
        def _query() -> list[dict]:
            products = self._paged_search_read(
                "product.product",
                [],
                [
                    "id",
                    "default_code",
                    "name",
                    "categ_id",
                    "uom_id",
                    "qty_available",
                    "list_price",
                    "product_tmpl_id",
                ],
                order="id",
            )
            tmpl_ids = [
                int(self._m2o_id(p.get("product_tmpl_id")))
                for p in products
                if self._m2o_id(p.get("product_tmpl_id"))
            ]
            product_ids = [p["id"] for p in products if p.get("id")]
            supplier_map = self._bulk_supplier_map(tmpl_ids)
            min_stock_map = self._bulk_min_stock_map(product_ids)
            warehouse_map = self._bulk_warehouse_map(product_ids)
            result = []
            for product in products:
                article_id = product.get("default_code") or str(product.get("id", ""))
                tmpl_id = int(self._m2o_id(product.get("product_tmpl_id")) or 0)
                supplier, lead_time = supplier_map.get(tmpl_id, ("", 0))
                result.append(
                    {
                        "article_id": article_id,
                        "description": product.get("name", ""),
                        "category": self._m2o_name(product.get("categ_id")),
                        "unit": self._m2o_name(product.get("uom_id")),
                        "stock_quantity": product.get("qty_available", 0),
                        "min_stock": min_stock_map.get(product.get("id"), 0),
                        "warehouse": warehouse_map.get(product.get("id"), ""),
                        "unit_price": product.get("list_price", 0),
                        "supplier": supplier,
                        "lead_time_days": lead_time,
                    }
                )
            return result

        return await asyncio.to_thread(_query)

    async def list_all_orders(self) -> list[dict]:
        def _query() -> list[dict]:
            orders = self._paged_search_read(
                "sale.order",
                [],
                list(_ORDER_READ_FIELDS),
                order="id",
            )
            partner_ids = [
                int(self._m2o_id(o.get("partner_id")))
                for o in orders
                if self._m2o_id(o.get("partner_id"))
            ]
            partner_ref_map = self._bulk_partner_ref_map(partner_ids)
            line_ids = []
            for order in orders:
                line_ids.extend(order.get("order_line") or [])
            lines_by_id, product_code_map = self._bulk_order_lines(line_ids)
            production_origins = self._bulk_production_origins([o.get("name", "") for o in orders])
            sale_ids = [int(order.get("id") or 0) for order in orders if order.get("id")]
            delivery_dates = self._delivery_dates_by_sale_ids(sale_ids)

            result = []
            for order in orders:
                partner_id = int(self._m2o_id(order.get("partner_id")) or 0)
                customer_id = partner_ref_map.get(partner_id) or str(partner_id or "")
                positions = []
                for line_id in order.get("order_line") or []:
                    line = lines_by_id.get(line_id)
                    if not line:
                        continue
                    product_id = line.get("_product_id_int") or 0
                    positions.append(
                        {
                            "pos": line.get("sequence", 0),
                            "article_id": product_code_map.get(product_id, ""),
                            "description": line.get("name", ""),
                            "quantity": line.get("product_uom_qty", 0),
                            "unit_price": line.get("price_unit", 0),
                        }
                    )
                result.append(
                    self._build_order_record(
                        order,
                        customer_id=customer_id,
                        production_origins=production_origins,
                        delivery_dates=delivery_dates,
                        positions=positions,
                    )
                )
            return result

        return await asyncio.to_thread(_query)

    async def list_all_production_orders(self) -> list[dict]:
        def _query() -> list[dict]:
            productions = self._paged_search_read(
                "mrp.production",
                [],
                [
                    "id",
                    "name",
                    "origin",
                    "product_id",
                    "product_qty",
                    "state",
                    "date_start",
                    "date_deadline",
                    "workorder_ids",
                ],
                order="id",
            )
            product_ids = [
                int(self._m2o_id(p.get("product_id")))
                for p in productions
                if self._m2o_id(p.get("product_id"))
            ]
            product_code_map = self._bulk_product_code_map(product_ids)
            workorder_ids = []
            for prod in productions:
                workorder_ids.extend(prod.get("workorder_ids") or [])
            workorders_by_id = self._bulk_workorders(workorder_ids)
            result = []
            for prod in productions:
                prod_product_id = int(self._m2o_id(prod.get("product_id")) or 0)
                operations = []
                for wo_id in prod.get("workorder_ids") or []:
                    wo = workorders_by_id.get(wo_id)
                    if not wo:
                        continue
                    planned_min = wo.get("duration_expected", 0) or 0
                    actual_min = wo.get("duration", 0) or 0
                    operations.append(
                        {
                            "op_number": wo.get("sequence", 0),
                            "name": wo.get("name", ""),
                            "workstation": self._m2o_name(wo.get("workcenter_id")),
                            "status": _WORKORDER_STATE_MAP.get(wo.get("state", ""), "offen"),
                            "planned_hours": round(planned_min / 60, 2),
                            "actual_hours": round(actual_min / 60, 2) if actual_min else None,
                        }
                    )
                result.append(
                    {
                        "production_order_id": prod.get("name", ""),
                        "order_reference": prod.get("origin", "") or "",
                        "article_id": product_code_map.get(prod_product_id, ""),
                        "article_description": self._m2o_name(prod.get("product_id")),
                        "quantity": prod.get("product_qty", 0),
                        "status": _PRODUCTION_STATE_MAP.get(prod.get("state", ""), "geplant"),
                        "start_date": self._odoo_date(prod.get("date_start")),
                        "planned_end_date": self._odoo_date(prod.get("date_deadline")),
                        "operations": operations,
                    }
                )
            return result

        return await asyncio.to_thread(_query)

    def _bulk_partner_ref_map(self, partner_ids: list[int]) -> dict[int, str]:
        if not partner_ids:
            return {}
        partners = self._search_read(
            "res.partner",
            [["id", "in", sorted(set(partner_ids))]],
            ["id", "ref"],
        )
        return {p["id"]: (p.get("ref") or str(p.get("id", ""))) for p in partners if p.get("id")}

    def _bulk_product_code_map(self, product_ids: list[int]) -> dict[int, str]:
        if not product_ids:
            return {}
        products = self._search_read(
            "product.product",
            [["id", "in", sorted(set(product_ids))]],
            ["id", "default_code"],
        )
        return {
            p["id"]: (p.get("default_code") or str(p.get("id", "")))
            for p in products
            if p.get("id")
        }

    def _bulk_order_lines(self, line_ids: list[int]) -> tuple[dict[int, dict], dict[int, str]]:
        if not line_ids:
            return {}, {}
        lines = self._search_read(
            "sale.order.line",
            [["id", "in", sorted(set(line_ids))]],
            ["id", "sequence", "product_id", "name", "product_uom_qty", "price_unit"],
            order="id",
        )
        product_ids = []
        lines_by_id = {}
        for line in lines:
            product_id = int(self._m2o_id(line.get("product_id")) or 0)
            line["_product_id_int"] = product_id
            if product_id:
                product_ids.append(product_id)
            lines_by_id[line["id"]] = line
        return lines_by_id, self._bulk_product_code_map(product_ids)

    def _bulk_production_origins(self, order_names: list[str]) -> set[str]:
        names = sorted({n for n in order_names if n})
        if not names:
            return set()
        try:
            productions = self._search_read(
                "mrp.production",
                [["origin", "in", names]],
                ["origin"],
            )
        except Exception:
            logger.debug(
                "Odoo mrp.production-Lookup fehlgeschlagen — "
                "Production-Origins koennen nicht ermittelt werden",
                exc_info=True,
            )
            return set()
        return {p.get("origin", "") for p in productions if p.get("origin")}

    def _bulk_supplier_map(self, tmpl_ids: list[int]) -> dict[int, tuple[str, int]]:
        if not tmpl_ids:
            return {}
        suppliers = self._search_read(
            "product.supplierinfo",
            [["product_tmpl_id", "in", sorted(set(tmpl_ids))]],
            ["product_tmpl_id", "partner_id", "delay", "sequence"],
            order="sequence,id",
        )
        result: dict[int, tuple[str, int]] = {}
        for supplier in suppliers:
            tmpl_id = int(self._m2o_id(supplier.get("product_tmpl_id")) or 0)
            if tmpl_id and tmpl_id not in result:
                result[tmpl_id] = (
                    self._m2o_name(supplier.get("partner_id")),
                    int(supplier.get("delay", 0) or 0),
                )
        return result

    def _bulk_min_stock_map(self, product_ids: list[int]) -> dict[int, float]:
        if not product_ids:
            return {}
        orderpoints = self._search_read(
            "stock.warehouse.orderpoint",
            [["product_id", "in", sorted(set(product_ids))]],
            ["product_id", "product_min_qty"],
        )
        result = {}
        for row in orderpoints:
            product_id = int(self._m2o_id(row.get("product_id")) or 0)
            result.setdefault(product_id, row.get("product_min_qty", 0) or 0)
        return result

    def _bulk_warehouse_map(self, product_ids: list[int]) -> dict[int, str]:
        if not product_ids:
            return {}
        quants = self._search_read(
            "stock.quant",
            [
                ["product_id", "in", sorted(set(product_ids))],
                ["location_id.usage", "=", "internal"],
            ],
            ["product_id", "location_id", "quantity"],
            order="quantity desc",
        )
        result = {}
        for row in quants:
            product_id = int(self._m2o_id(row.get("product_id")) or 0)
            if product_id and product_id not in result:
                result[product_id] = self._m2o_name(row.get("location_id"))
        return result

    def _bulk_workorders(self, workorder_ids: list[int]) -> dict[int, dict]:
        if not workorder_ids:
            return {}
        rows = self._search_read(
            "mrp.workorder",
            [["id", "in", sorted(set(workorder_ids))]],
            [
                "id",
                "sequence",
                "name",
                "workcenter_id",
                "state",
                "duration_expected",
                "duration",
            ],
            order="sequence,id",
        )
        return {row["id"]: row for row in rows if row.get("id")}
