"""Sicher begrenzter Query-Vertrag fuer ERP-Auftragsgrunddaten.

Das Modul kapselt die einzige vom Agent akzeptierte Abfragesprache. Freie
Odoo-Domains, Modellnamen und SQL-Fragmente sind absichtlich ausgeschlossen.
Connectoren erhalten ausschliesslich einen bereits validierten
``ERPOrderQuery`` und liefern normalisierte, projizierte Datensaetze zurueck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kiara.shared.constants import (
    ERP_ORDERS_DEFAULT_LIMIT,
    ERP_ORDERS_MAX_FIELDS,
    ERP_ORDERS_MAX_FILTER_TEXT,
    ERP_ORDERS_MAX_FILTERS,
    ERP_ORDERS_MAX_IN_VALUES,
    ERP_ORDERS_MAX_LIMIT,
    ERP_ORDERS_MAX_OFFSET,
)

ORDER_FIELD_TYPES: dict[str, str] = {
    "order_id": "string",
    "customer_id": "string",
    "customer_name": "string",
    "status": "string",
    "delivery_status": "string",
    "order_date": "datetime",
    "promised_delivery_date": "datetime",
    "first_delivery_date": "datetime",
    "actual_delivery_date": "datetime",
    "total_amount": "number",
    "currency": "string",
    "position_count": "integer",
}

ORDER_FIELDS: tuple[str, ...] = tuple(ORDER_FIELD_TYPES)
DEFAULT_ORDER_FIELDS: tuple[str, ...] = (
    "order_id",
    "customer_id",
    "customer_name",
    "status",
    "delivery_status",
    "order_date",
    "promised_delivery_date",
    "first_delivery_date",
    "actual_delivery_date",
    "total_amount",
    "currency",
)
ORDER_FILTER_FIELDS: frozenset[str] = frozenset(
    {
        "order_id",
        "customer_id",
        "customer_name",
        "status",
        "delivery_status",
        "order_date",
        "promised_delivery_date",
        "first_delivery_date",
        "actual_delivery_date",
        "total_amount",
    }
)
ORDER_SORT_FIELDS: tuple[str, ...] = (
    "order_id",
    "order_date",
    "promised_delivery_date",
    "first_delivery_date",
    "actual_delivery_date",
    "total_amount",
)

_STRING_OPERATORS = frozenset({"eq", "ne", "contains", "in"})
_ORDERED_OPERATORS = frozenset({"eq", "ne", "gt", "gte", "lt", "lte", "in"})


@dataclass(frozen=True)
class ERPOrderFilter:
    """Ein validierter Feld-/Operator-/Wert-Filter."""

    field: str
    operator: str
    value: Any


@dataclass(frozen=True)
class ERPOrderQuery:
    """Vollstaendig validierter und begrenzter Auftragsquery."""

    filters: tuple[ERPOrderFilter, ...]
    fields: tuple[str, ...]
    sort_by: str
    sort_direction: str
    limit: int
    offset: int


def build_order_query(params: dict) -> ERPOrderQuery:
    """Validiert Agent-Parameter fail-closed und baut den Query-Vertrag."""
    if not isinstance(params, dict):
        raise ValueError("ERP-Auftragsparameter muessen ein Objekt sein.")
    allowed_params = {
        "filters",
        "fields",
        "sort_by",
        "sort_direction",
        "limit",
        "offset",
    }
    unknown_params = sorted(set(params) - allowed_params)
    if unknown_params:
        raise ValueError(f"Nicht erlaubte Parameter: {unknown_params}")

    raw_fields = params.get("fields")
    if raw_fields is not None and not isinstance(raw_fields, list):
        raise ValueError("fields muss eine Liste sein.")
    fields = tuple(raw_fields) if raw_fields is not None else DEFAULT_ORDER_FIELDS
    if not fields:
        raise ValueError("fields darf nicht leer sein.")
    if len(fields) > ERP_ORDERS_MAX_FIELDS:
        raise ValueError(f"Maximal {ERP_ORDERS_MAX_FIELDS} Felder sind erlaubt.")
    if len(set(fields)) != len(fields):
        raise ValueError("fields darf keine Duplikate enthalten.")
    unknown_fields = [field for field in fields if field not in ORDER_FIELD_TYPES]
    if unknown_fields:
        raise ValueError(f"Nicht erlaubte Felder: {unknown_fields}")

    sort_by = params.get("sort_by", "order_date")
    if sort_by not in ORDER_SORT_FIELDS:
        raise ValueError(
            f"sort_by '{sort_by}' ist nicht erlaubt. Erlaubt: {list(ORDER_SORT_FIELDS)}"
        )
    sort_direction = params.get("sort_direction", "desc")
    if sort_direction not in {"asc", "desc"}:
        raise ValueError("sort_direction muss 'asc' oder 'desc' sein.")

    limit = params.get("limit", ERP_ORDERS_DEFAULT_LIMIT)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit muss eine Ganzzahl sein.")
    if not 1 <= limit <= ERP_ORDERS_MAX_LIMIT:
        raise ValueError(f"limit muss zwischen 1 und {ERP_ORDERS_MAX_LIMIT} liegen.")

    offset = params.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset muss eine Ganzzahl sein.")
    if not 0 <= offset <= ERP_ORDERS_MAX_OFFSET:
        raise ValueError(f"offset muss zwischen 0 und {ERP_ORDERS_MAX_OFFSET} liegen.")

    raw_filters = params.get("filters", [])
    if not isinstance(raw_filters, list):
        raise ValueError("filters muss eine Liste sein.")
    if len(raw_filters) > ERP_ORDERS_MAX_FILTERS:
        raise ValueError(f"Maximal {ERP_ORDERS_MAX_FILTERS} Filter sind erlaubt.")

    filters = tuple(_validate_filter(item, index) for index, item in enumerate(raw_filters))
    return ERPOrderQuery(
        filters=filters,
        fields=fields,
        sort_by=sort_by,
        sort_direction=sort_direction,
        limit=limit,
        offset=offset,
    )


def _validate_filter(raw: object, index: int) -> ERPOrderFilter:
    if not isinstance(raw, dict):
        raise ValueError(f"Filter {index + 1} muss ein Objekt sein.")
    expected_keys = {"field", "operator", "value"}
    if set(raw) != expected_keys:
        raise ValueError(f"Filter {index + 1} muss genau field, operator und value enthalten.")

    field = raw.get("field")
    operator = raw.get("operator")
    if field not in ORDER_FILTER_FIELDS:
        raise ValueError(f"Filterfeld '{field}' ist nicht erlaubt.")
    field_type = ORDER_FIELD_TYPES[field]
    allowed_ops = _STRING_OPERATORS if field_type == "string" else _ORDERED_OPERATORS
    if operator not in allowed_ops:
        raise ValueError(
            f"Operator '{operator}' ist fuer Feld '{field}' nicht erlaubt. "
            f"Erlaubt: {sorted(allowed_ops)}"
        )

    value = raw.get("value")
    if value is None:
        if operator not in {"eq", "ne"}:
            raise ValueError("null ist nur mit eq oder ne erlaubt.")
        return ERPOrderFilter(field=field, operator=operator, value=None)

    if operator == "in":
        if not isinstance(value, list) or not value:
            raise ValueError("Der Operator 'in' erwartet eine nicht-leere Liste.")
        if len(value) > ERP_ORDERS_MAX_IN_VALUES:
            raise ValueError(
                f"Ein in-Filter darf maximal {ERP_ORDERS_MAX_IN_VALUES} Werte enthalten."
            )
        for item in value:
            _validate_filter_scalar(field, field_type, item)
    else:
        _validate_filter_scalar(field, field_type, value)

    return ERPOrderFilter(field=field, operator=operator, value=value)


def _validate_filter_scalar(field: str, field_type: str, value: object) -> None:
    if field_type in {"number", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Filterwert fuer '{field}' muss numerisch sein.")
        if not math.isfinite(float(value)):
            raise ValueError(f"Filterwert fuer '{field}' muss endlich sein.")
        return
    if not isinstance(value, str):
        raise ValueError(f"Filterwert fuer '{field}' muss Text sein.")
    if not value or len(value) > ERP_ORDERS_MAX_FILTER_TEXT:
        raise ValueError(
            f"Filterwert fuer '{field}' muss 1 bis {ERP_ORDERS_MAX_FILTER_TEXT} Zeichen haben."
        )
    if field_type == "datetime":
        _datetime_key(value)


def normalize_order_record(row: dict) -> dict:
    """Normalisiert Connector-/Legacy-Felder auf den Agent-Vertrag."""
    positions = row.get("positions")
    promised = row.get("promised_delivery_date")
    if promised in (None, "", False):
        promised = row.get("delivery_date")
    return {
        "order_id": _none_for_false(row.get("order_id")),
        "customer_id": _none_for_false(row.get("customer_id")),
        "customer_name": _none_for_false(row.get("customer_name")),
        "status": _none_for_false(row.get("status")),
        "delivery_status": _none_for_false(row.get("delivery_status")),
        "order_date": _none_for_false(row.get("order_date")),
        "promised_delivery_date": _none_for_false(promised),
        "first_delivery_date": _none_for_false(row.get("first_delivery_date")),
        "actual_delivery_date": _none_for_false(row.get("actual_delivery_date")),
        "total_amount": row.get("total_amount"),
        "currency": _none_for_false(row.get("currency")),
        "position_count": (
            row.get("position_count")
            if row.get("position_count") is not None
            else len(positions)
            if isinstance(positions, list)
            else None
        ),
    }


def apply_order_query(
    rows: list[dict],
    query: ERPOrderQuery,
    *,
    scan_truncated: bool = False,
) -> dict:
    """Filtert, sortiert und projiziert ausschliesslich erlaubte Skalarfelder."""
    normalized = [normalize_order_record(row) for row in rows]
    filtered = [
        row for row in normalized if all(_matches_filter(row, item) for item in query.filters)
    ]
    populated = [row for row in filtered if row.get(query.sort_by) not in (None, "", False)]
    empty = [row for row in filtered if row.get(query.sort_by) in (None, "", False)]
    populated.sort(
        key=lambda row: _sort_key(row.get(query.sort_by), ORDER_FIELD_TYPES[query.sort_by]),
        reverse=query.sort_direction == "desc",
    )
    ordered = populated + empty

    end = query.offset + query.limit
    page = ordered[query.offset : end]
    projected = [{field: row.get(field) for field in query.fields} for row in page]
    return {
        "rows": projected,
        "returned": len(projected),
        "has_more": len(ordered) > end or scan_truncated,
        "limit": query.limit,
        "offset": query.offset,
        "fields": list(query.fields),
        "scan_truncated": bool(scan_truncated),
    }


def _matches_filter(row: dict, item: ERPOrderFilter) -> bool:
    actual = row.get(item.field)
    expected = item.value
    operator = item.operator
    if operator == "eq":
        return _comparable(actual, item.field) == _comparable(expected, item.field)
    if operator == "ne":
        return _comparable(actual, item.field) != _comparable(expected, item.field)
    if actual in (None, "", False):
        return False
    if operator == "contains":
        return str(expected).casefold() in str(actual).casefold()
    if operator == "in":
        actual_value = _comparable(actual, item.field)
        return actual_value in {_comparable(value, item.field) for value in expected}

    left = _comparable(actual, item.field)
    right = _comparable(expected, item.field)
    if operator == "gt":
        return left > right
    if operator == "gte":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "lte":
        return left <= right
    return False


def _comparable(value: object, field: str):
    if value is None:
        return None
    field_type = ORDER_FIELD_TYPES[field]
    if field_type == "datetime":
        return _datetime_key(value)
    if field_type in {"number", "integer"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")
    return str(value).casefold()


def _sort_key(value: object, field_type: str):
    if field_type == "datetime":
        return _datetime_key(value)
    if field_type in {"number", "integer"}:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")
    return str(value).casefold()


def _datetime_key(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError(f"Ungueltiger Datumswert: {value!r}")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Ungueltiger ISO-Datumswert: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _none_for_false(value):
    return None if value in (None, "", False) else value
