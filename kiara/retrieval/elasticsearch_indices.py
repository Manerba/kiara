"""Datasource index contract for Tareas #69 (S1).

``retrieval.elasticsearch_index`` is the configured *prefix*: ``kiara`` maps
DS 12 to ``kiara_ds_12``. Only explicit, canonical names leave this module.
Selection None means all caller-authorized sources; [] means no ES request.
The caller supplies PG membership expectations for the guarded default dense model,
including explicit zeroes for empty/new sources. Unknown is never zero.
DS 0 is the existing, PG-backed CLI owner, not a fallback for missing ownership.
run_ingestion resolves CLI None via _ensure_cli_datasource before ES writes;
an unresolved None or an unassigned chunk must fail before ES side effects.

These helpers do not authorize users, release unpublished documents, acquire
model guards, or create indices. S2-S4 must wire them into the existing paths;
the old single-index functions are not a fallback for this contract.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_PREFIX = re.compile(r"[a-z0-9][a-z0-9._-]*", re.ASCII)


def _validate_datasource_id(value: int) -> None:
    # No bools, floats, strings, truncation, or silent filtering of invalid IDs.
    if type(value) is not int or value < 0:
        raise ValueError("Elasticsearch braucht eine nichtnegative ganzzahlige Datenquellen-ID")


def _index_name(prefix: str, datasource_id: int) -> str:
    _validate_datasource_id(datasource_id)
    if not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix):
        raise ValueError("Ungueltiges Elasticsearch-Namenspraefix")
    name = f"{prefix}_ds_{datasource_id}"
    if len(name.encode("utf-8")) > 255:
        raise ValueError("Elasticsearch-Indexname ueberschreitet 255 Bytes")
    return name


def datasource_index_name(config: dict, datasource_id: int) -> str:
    """Return the sole physical name for a DS; no model/name/alias axis."""
    return _index_name(
        config.get("retrieval", {}).get("elasticsearch_index", "kiara"), datasource_id
    )


def _validate_address_index(datasource_id: int, index_name: str) -> None:
    _validate_datasource_id(datasource_id)
    if not isinstance(index_name, str):
        raise ValueError("Elasticsearch-Indexname muss eine Zeichenkette sein")
    prefix, separator, suffix = index_name.rpartition("_ds_")
    if not separator or suffix != str(datasource_id):
        raise ValueError("Elasticsearch-Indexname passt nicht zur Datenquelle")
    if _index_name(prefix, datasource_id) != index_name:
        raise ValueError("Nichtkanonischer Elasticsearch-Indexname")


def _datasource_ids(values: Iterable[int]) -> set[int]:
    if values is None or isinstance(values, (str, bytes, Mapping)):
        raise ValueError("Elasticsearch braucht eine explizite Datenquellenmenge")
    result = set()
    for value in values:
        _validate_datasource_id(value)
        result.add(value)
    return result


@dataclass(frozen=True)
class DatasourceIndexTarget:
    datasource_id: int
    index_name: str
    # Expectation is an observation, not part of the physical identity.
    expected_document_count: int = field(compare=False)

    def __post_init__(self):
        _validate_address_index(self.datasource_id, self.index_name)
        if type(self.expected_document_count) is not int or self.expected_document_count < 0:
            raise ValueError(
                "Elasticsearch-Bestandserwartung muss eine nichtnegative Ganzzahl sein"
            )


@dataclass(frozen=True)
class ElasticsearchDocumentAddress:
    """Complete ES identity; CAS/version evidence belongs to this exact address."""

    datasource_id: int
    index_name: str
    chunk_id: str

    def __post_init__(self):
        _validate_address_index(self.datasource_id, self.index_name)
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("Elasticsearch braucht eine nichtleere Chunk-ID")


def document_address(
    config: dict, datasource_id: int, chunk_id: str
) -> ElasticsearchDocumentAddress:
    """Validate a write/lookup/delete address before any ES side effect."""
    return ElasticsearchDocumentAddress(
        datasource_id, datasource_index_name(config, datasource_id), chunk_id
    )


def resolve_datasource_targets(
    config: dict,
    selected_ds_ids: Iterable[int] | None,
    *,
    allowed_ds_ids: Iterable[int],
    expected_document_counts: Mapping[int, int],
) -> tuple[DatasourceIndexTarget, ...]:
    """Resolve selection intersected with caller-authorized PG sources.

    Both [] and an empty permission set return (). Unknown/disallowed selected
    IDs cannot broaden the result. Invalid IDs anywhere in the selection fail.
    Counts must be supplied for every effective DS, even for known empty ones;
    search/write callers use batched PG EXISTS (0/1 availability expectations)
    for the bound model, not statistical counts or absent/stale aggregates
    interpreted as zero. Exact inventory counts may also be supplied. No I/O here.
    """
    # Validate configuration even when the effective selection is empty.
    datasource_index_name(config, 1)
    allowed = _datasource_ids(allowed_ds_ids)
    selected = allowed if selected_ds_ids is None else _datasource_ids(selected_ds_ids) & allowed
    if not isinstance(expected_document_counts, Mapping):
        raise ValueError("Elasticsearch-Bestandserwartungen fehlen")
    # Validate keys before lookup: True and 1.0 otherwise alias the integer 1.
    for ds_id, count in expected_document_counts.items():
        _validate_datasource_id(ds_id)
        if type(count) is not int or count < 0:
            raise ValueError("Ungueltige Elasticsearch-Bestandserwartung")
    targets = []
    for ds_id in sorted(selected):
        if ds_id not in expected_document_counts:
            raise ValueError(f"Elasticsearch-Bestandserwartung fuer DS {ds_id} fehlt")
        targets.append(
            DatasourceIndexTarget(
                ds_id, datasource_index_name(config, ds_id), expected_document_counts[ds_id]
            )
        )
    return tuple(targets)


@dataclass(frozen=True)
class DatasourceIndexAvailability:
    """Observed primaries-only physical entries, never globally unique chunks.

    Missing new/empty indices are explicit. This is an availability check,
    not a full integrity proof, publication decision or maintenance release.
    """

    present: tuple[DatasourceIndexTarget, ...]
    absent_empty: tuple[DatasourceIndexTarget, ...]
    document_counts: dict[int, int]

    @property
    def physical_documents(self) -> int:
        return sum(self.document_counts.values())
