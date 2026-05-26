"""Zentrale Normalisierung fuer Graph-Node-IDs."""

from __future__ import annotations

import re
import unicodedata

_ENTITY_TYPE_RE = re.compile(r"^[a-z0-9_]+$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalize_node_id(entity_type: str, label_or_id) -> str:
    """Normalisiert einen Entity-Typ und Rohwert zu ``type:slug``.

    Der Helper akzeptiert bewusst keinen vollstaendigen ``type:slug``-String;
    Aufrufer splitten den Typ-Teil vorher ab.
    """
    if entity_type is None:
        raise ValueError("entity_type darf nicht None sein")
    normalized_type = str(entity_type).strip().lower()
    if not normalized_type or not _ENTITY_TYPE_RE.match(normalized_type):
        raise ValueError(f"ungueltiger entity_type: {entity_type!r}")
    if label_or_id is None:
        raise ValueError("label_or_id darf nicht None sein")
    raw = str(label_or_id).strip()
    if not raw:
        raise ValueError("label_or_id darf nicht leer sein")

    slug = unicodedata.normalize("NFKD", raw)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = slug.lower()
    slug = _SLUG_RE.sub("-", slug).strip("-")
    if not slug:
        raise ValueError(f"label_or_id ergibt leeren Slug: {label_or_id!r}")
    return f"{normalized_type}:{slug}"
