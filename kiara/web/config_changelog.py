"""Config-Changelog: Aenderungsprotokoll fuer Konfigurationsaenderungen.

Generisches Changelog-System das zunaechst Prompt-Template-Aenderungen loggt
und zukuenftig alle Konfigurationsaenderungen aufnehmen kann.
"""

import logging

from kiara.repositories.changelog import ConfigChangelogRepository

logger = logging.getLogger(__name__)

changelog_repo = ConfigChangelogRepository()

_ALLOWED_SORT_FIELDS = {
    "id",
    "user_id",
    "created_at",
    "username",
    "area",
    "action",
    "target_id",
    "target_label",
    "field_name",
    "old_value",
    "new_value",
    "source",
}
_ALLOWED_SOURCES = {"gui", "chat"}


def _validate_source(source: str) -> str:
    """Validiert eine Changelog-Quelle gegen die zentrale Allowlist."""
    if source not in _ALLOWED_SOURCES:
        allowed = ", ".join(sorted(_ALLOWED_SOURCES))
        raise ValueError(f"Ungueltige Changelog-Quelle: {source!r}; erlaubt: {allowed}")
    return source


def log_config_change(
    user: dict,
    area: str,
    action: str,
    target_id: str = "",
    target_label: str = "",
    field_name: str = "",
    old_value: str | None = None,
    new_value: str | None = None,
    source: str = "gui",
) -> None:
    """Schreibt einen Eintrag in das Config-Changelog.

    Args:
        user: User-Dict mit 'id' und 'username'.
        area: Bereich (z.B. 'prompt_templates', 'ki_config').
        action: Aktion (z.B. 'update', 'create', 'delete').
        target_id: ID des betroffenen Objekts.
        target_label: Label/Name des betroffenen Objekts.
        field_name: Name des geaenderten Feldes (bei Updates).
        old_value: Alter Wert (bei Updates/Deletes).
        new_value: Neuer Wert (bei Updates/Creates).
        source: Quelle der Aenderung ('gui' oder 'chat').
    """
    try:
        source = _validate_source(source)
    except ValueError:
        logger.warning(
            "Config-Changelog: unbekannter source=%r, faelle auf 'gui' zurueck (area=%s, action=%s)",
            source,
            area,
            action,
        )
        source = "gui"
    try:
        changelog_repo.create(
            user_id=user.get("id"),
            username=user.get("username", ""),
            area=area,
            action=action,
            target_id=str(target_id),
            target_label=target_label,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            source=source,
        )
    except Exception as exc:
        logger.warning("Config-Changelog Eintrag fehlgeschlagen: %s", exc)


def query_changelog(
    area: str | None = None,
    source: str | None = None,
    page: int = 1,
    page_size: int = 50,
    sort_field: str = "created_at",
    sort_dir: str = "desc",
) -> dict:
    """Liest Changelog-Eintraege mit optionalen Filtern.

    Returns:
        Dict mit 'data' (Liste) und 'pagination' (ExpandableTable-kompatibel).
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 200))

    if sort_field not in _ALLOWED_SORT_FIELDS:
        sort_field = "created_at"
    if sort_dir not in ("asc", "desc"):
        sort_dir = "desc"

    filters = {}
    if area:
        filters["area"] = area
    if source:
        filters["source"] = _validate_source(source)

    offset = (page - 1) * page_size
    rows, total = changelog_repo.list_filtered(
        filters,
        sort=f"{sort_field} {sort_dir}",
        limit=page_size,
        offset=offset,
    )

    total_pages = max(1, (total + page_size - 1) // page_size)

    data = []
    for d in rows:
        data.append(
            {
                "id": d["id"],
                "user_id": d.get("user_id"),
                "username": d.get("username", ""),
                "area": d["area"],
                "action": d["action"],
                "target_id": d.get("target_id", ""),
                "target_label": d.get("target_label", ""),
                "field_name": d.get("field_name", ""),
                "old_value": d.get("old_value"),
                "new_value": d.get("new_value"),
                "source": d.get("source", "gui"),
                "created_at": d.get("created_at", ""),
            }
        )

    return {
        "data": data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total,
            "total_pages": total_pages,
        },
    }
