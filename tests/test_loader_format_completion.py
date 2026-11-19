"""Format-Skips/-Fehler duerfen keinen dauerhaften Leer-Abschluss erzeugen."""

import zipfile
from pathlib import Path

import pytest
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_ingestion_empty_completion import _assert_not_completed, _run


@pytest.mark.parametrize("broken", [False, True])
def test_format_outcome_stays_retryable_in_database(whole_pipeline, broken):
    case = whole_pipeline
    path = Path(case.path)
    if broken:
        path.write_bytes(b"PK\x03\x04abgeschnitten")
    else:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("inhalt.txt", "Kein unterstuetztes Dokumentformat")

    for _ in range(2):
        logs = []
        result = _run(case, file_log_callback=logs.append)
        assert result["errors"] == int(broken)
        assert result.get("empty_files", 0) == 0
        assert result["processed_files"] == 0
        assert len(logs) == 1
        assert logs[0].status == ("error" if broken else "skipped")
        assert logs[0].error_message
        assert not logs[0].empty_content_confirmed
        _assert_not_completed(case)
