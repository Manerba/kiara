"""Statische Guards fuer Datasource-Tab-Connectorfelder."""

from pathlib import Path


def test_connector_field_ids_are_derived_via_safe_helper():
    js = Path("kiara/web/static/js/tab_datenquellen.js").read_text()

    assert "function _connectorFieldDomId" in js
    assert "function _connFieldId" in js
    for raw_pattern in (
        "connField_${fieldName}",
        "btnLoadFolders_${fieldName}",
        "folderTree_${fieldName}",
        "chkAllFolders_${fieldName}",
        "folderPickerWrap_${fieldName}",
    ):
        assert raw_pattern not in js
