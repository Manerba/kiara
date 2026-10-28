import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


@contextmanager
def _db_rows(rows, seen_sql):
    class Conn:
        def execute(self, sql, params=None):
            seen_sql.append(sql)
            cursor = MagicMock()
            cursor.fetchall.return_value = rows
            return cursor

    yield Conn()


def test_overlapping_datasources_use_connector_effective_path(tmp_path):
    from kiara.integrity.checks import check_overlapping_datasources

    root = tmp_path / "root"
    child = root / "child"
    root.mkdir()
    child.mkdir()
    rows = [
        {
            "id": 1,
            "name": "Root",
            "type": "local",
            "connector_config": json.dumps({"path": str(root)}),
        },
        {
            "id": 2,
            "name": "Child",
            "type": "local",
            "connector_config": json.dumps({"path": str(child)}),
        },
    ]
    seen_sql = []

    with patch("kiara.shared.db.db_query", return_value=_db_rows(rows, seen_sql)):
        result = check_overlapping_datasources({})

    assert result["status"] == "warning"
    assert result["overlaps"][0]["relationship"] == "parent"
    assert result["overlaps"][0]["ds_a"]["id"] == 1
    assert result["overlaps"][0]["ds_b"]["id"] == 2
    assert "connector_config" in seen_sql[0]
    assert "smb_mount_point" not in seen_sql[0]
    assert "smb_server" not in seen_sql[0]
