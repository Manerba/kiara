import pytest

from kiara.web import config_changelog


def test_query_changelog_rejects_invalid_source(monkeypatch):
    repo = type(
        "Repo",
        (),
        {
            "list_filtered": lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("DB should not be queried")
            )
        },
    )()
    monkeypatch.setattr(config_changelog, "changelog_repo", repo)

    with pytest.raises(ValueError, match="Ungueltige Changelog-Quelle"):
        config_changelog.query_changelog(source="cli")


def test_query_changelog_accepts_allowed_source(monkeypatch):
    captured = {}

    class Repo:
        def list_filtered(self, filters, sort, limit, offset):
            captured.update(
                {
                    "filters": filters,
                    "sort": sort,
                    "limit": limit,
                    "offset": offset,
                }
            )
            return [], 0

    monkeypatch.setattr(config_changelog, "changelog_repo", Repo())

    result = config_changelog.query_changelog(source="chat", page=2, page_size=10)

    assert captured["filters"] == {"source": "chat"}
    assert captured["limit"] == 10
    assert captured["offset"] == 10
    assert result["pagination"]["total_items"] == 0


def test_log_config_change_invalid_source_falls_back_to_gui(monkeypatch):
    captured = {}

    class Repo:
        def create(self, **fields):
            captured.update(fields)
            return 1

    monkeypatch.setattr(config_changelog, "changelog_repo", Repo())

    config_changelog.log_config_change(
        {"id": 1, "username": "admin"},
        area="system",
        action="update",
        source="cli",
    )

    assert captured["source"] == "gui"
