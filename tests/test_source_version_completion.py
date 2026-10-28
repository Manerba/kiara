"""Dateifassung, erfolgreicher Abschluss und Wiederanlauf durch die ganze Pipeline."""

import os
from pathlib import Path

import pytest
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_hash_mode_completion import _assert_completed, _run

from kiara.exceptions import RetrievalError
from kiara.ingestion import pipeline, runtime


@pytest.mark.parametrize("mode", ["content", "mtime"])
def test_new_version_before_loading_gets_its_own_completion(
    colbert_pipeline, test_db, monkeypatch, mode
):
    state = colbert_pipeline
    case = state.case
    process = pipeline._process_file_in_pipeline

    def change(ctx, path):
        Path(path).write_text("B" * 300)
        case.prefix = "Neu: "
        return process(ctx, path)

    with monkeypatch.context() as timing:
        timing.setattr(pipeline, "_process_file_in_pipeline", change)
        result = _run(state, mode)
    assert result["processed_files"] == 1 and result["errors"] == 0
    _assert_completed(state, test_db, mode)
    again = _run(state, mode)
    assert again["skipped_files"] == 1 and again["processed_files"] == again["errors"] == 0


@pytest.mark.parametrize("mode", ["content", "mtime"])
@pytest.mark.parametrize("boundary", ["loading", "completion"])
@pytest.mark.parametrize("change", ["restore_bytes", "replace_inode", "delete"])
def test_source_change_prevents_completion_and_retries(
    colbert_pipeline, test_db, monkeypatch, mode, boundary, change
):
    state = colbert_pipeline
    case = state.case
    path = Path(case.path)
    original = path.read_bytes()
    before = path.stat()
    load = runtime._process_single_file
    process = pipeline._process_file_in_pipeline

    def mutate():
        if change == "delete":
            path.unlink()
        elif change == "replace_inode":
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(original)
            os.replace(replacement, path)
        else:
            path.write_text("D" * 300)
            path.write_bytes(original)
        if path.exists():
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    def loading(*args, **kwargs):
        result = load(*args, **kwargs)
        mutate()
        return result

    def completing(ctx, file_path):
        complete = ctx.file_completion_callback

        def finish(p):
            mutate()
            complete(p)

        ctx.file_completion_callback = finish
        return process(ctx, file_path)

    with monkeypatch.context() as timing:
        if boundary == "loading":
            timing.setattr(runtime, "_process_single_file", loading)
            result = _run(state, mode)
            assert result["processed_files"] == 0 and result["errors"] >= 1
        else:
            timing.setattr(pipeline, "_process_file_in_pipeline", completing)
            with pytest.raises(OSError):
                _run(state, mode)
            assert state.context.stats["processed_files"] == 0
    assert case.path not in case.hashes.load(case.ds, "test-model")
    assert (
        test_db.execute(
            "SELECT COALESCE(SUM(file_count),0) AS n FROM datasource_embeddings WHERE datasource_id=%s",
            (case.ds,),
        ).fetchone()["n"]
        == 0
    )
    if boundary == "completion":
        with pytest.raises(RetrievalError):
            case.reader()
    # Der erneute Lauf muss tatsaechliche neue Texte/Vektoren publizieren.
    path.write_text("B" * 300)
    case.prefix = "Neu: "
    result = _run(state, mode)
    assert result["processed_files"] == 1 and result["errors"] == 0
    _assert_completed(state, test_db, mode)
    assert _run(state, mode)["skipped_files"] == 1
