"""Dokumentreferenzen, begrenztes Nachlesen und Fortsetzung ohne Belegverlust."""

import asyncio
import os
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from kiara.exceptions import RetrievalError
from kiara.generation.agent_chain import AgentChain, _freeze_params
from kiara.generation.context_selection import prepare_context, prepare_document_read
from kiara.retrieval.retriever import HybridRetriever, Retriever
from kiara.tools.builtin_tools import SearchDocumentsTool
from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool


def _chunks(texts, **metadata):
    return [
        {
            "text": text,
            "metadata": {
                "source": "anleitung.txt",
                "ds_ids": [7],
                "embedding_model": "model-a",
                "chunk_index": index,
                "chunk_total": len(texts),
                "modified": "2026-09-30T12:00:00+00:00",
                "size_bytes": sum(len(text) for text in texts),
                **metadata,
            },
            "score": 0.5,
        }
        for index, text in enumerate(texts)
    ]


def _tool(chunks, allowed=None):
    references = DocumentReferences()
    reference = references.annotate(chunks)[0]["metadata"]["document_ref"]
    retriever = MagicMock()
    retriever.load_document_range.side_effect = lambda source, start, end, **kw: deepcopy(
        chunks[start:end]
    )
    return ReadDocumentTool(retriever, {}, references, allowed), reference, retriever


def _read(tool, reference, **params):
    return asyncio.run(tool.execute({"document_id": reference, **params}))


def test_search_provides_reference_that_reads_sorted_original_chunks_without_search():
    chunks = _chunks(["Anfang", "Betriebsdruck\n10 bar", "Nicht fuer Trinkwasser geeignet."])
    references = DocumentReferences()
    retriever = MagicMock()
    retriever.retrieve.return_value = [chunks[1]]
    search = SearchDocumentsTool(retriever, None, {}, document_references=references)
    result = asyncio.run(search.execute({"query": "Betriebsdruck"}))
    hit = result.data["results"][0]
    reference = hit["metadata"]["document_ref"]
    display = prepare_context([hit], budget=2000).text
    assert reference in display and "Chunks: 1" in display and "Chunks gesamt: 3" in display
    retriever.retrieve.reset_mock()
    retriever.load_document_range.return_value = list(reversed(chunks))
    read = ReadDocumentTool(retriever, {}, references, [7])
    result = _read(read, reference)
    assert result.success
    assert [c["text"] for c in result.data["results"]] == [c["text"] for c in chunks]
    retriever.load_document_range.assert_called_once_with(
        "anleitung.txt", 0, 3, filter_datasource_ids=[7], parser="standard"
    )
    retriever.retrieve.assert_not_called()


def test_document_references_separate_datasources_models_and_sessions():
    refs = DocumentReferences()
    documents = [
        _chunks(["A"])[0],
        _chunks(["B"], ds_ids=[8])[0],
        _chunks(["C"], embedding_model="model-b")[0],
    ]
    annotated = refs.annotate(documents)
    identifiers = [c["metadata"]["document_ref"] for c in annotated]
    assert len(set(identifiers)) == 3
    assert not DocumentReferences().resolve(identifiers[0])
    assert "document_ref" not in documents[0]["metadata"]
    assert refs.annotate([documents[0]])[0]["metadata"]["document_ref"] == identifiers[0]
    assert (
        DocumentReferences("kb_model_a").annotate(documents)[0]["metadata"]["document_ref"]
        != DocumentReferences("kb_model_b").annotate(documents)[0]["metadata"]["document_ref"]
    )


@pytest.mark.parametrize(
    "params",
    [
        {"document_id": "/etc/passwd"},
        {"start_chunk": -1},
        {"start_chunk": True},
        {"max_chunks": 0},
        {"max_chunks": 9},
        {"start_offset": -1},
        {"start_offset": True},
        {"start_chunk": 1.5},
        {"source": "other.txt"},
    ],
)
def test_invalid_calls_do_not_read_store(params):
    tool, reference, retriever = _tool(_chunks(["Inhalt"]))
    result = asyncio.run(tool.execute({"document_id": reference, **params}))
    assert not result.success
    retriever.load_document_range.assert_not_called()


@pytest.mark.parametrize("allowed", [[], [8]])
def test_read_rejects_datasource_outside_current_permissions(allowed):
    tool, reference, retriever = _tool(_chunks(["Inhalt"]), allowed)
    assert not _read(tool, reference).success
    retriever.load_document_range.assert_not_called()


@pytest.mark.parametrize(
    "mutation", ["gap", "duplicate", "source", "model", "ds", "total", "parser"]
)
def test_inconsistent_document_ranges_fail_closed(mutation):
    chunks = _chunks(["A", "B", "C"])
    tool, reference, retriever = _tool(chunks)
    returned = deepcopy(chunks)
    if mutation == "gap":
        returned.pop(1)
    elif mutation == "duplicate":
        returned[1] = deepcopy(returned[0])
    else:
        key, value = {
            "source": ("source", "anderes.txt"),
            "model": ("embedding_model", "model-b"),
            "ds": ("ds_ids", [8]),
            "total": ("chunk_total", 4),
            "parser": ("parser", "vlm"),
        }[mutation]
        returned[1]["metadata"][key] = value
    retriever.load_document_range.side_effect = None
    retriever.load_document_range.return_value = returned
    with pytest.raises(RetrievalError):
        _read(tool, reference)


def test_large_document_load_is_bounded_and_offset_validated():
    chunks = _chunks([f"Abschnitt {i}" for i in range(100)])
    tool, reference, retriever = _tool(chunks)
    result = _read(tool, reference, start_chunk=42)
    assert len(result.data["results"]) == 8
    retriever.load_document_range.assert_called_once_with(
        "anleitung.txt", 42, 50, filter_datasource_ids=[7], parser="standard"
    )
    assert not _read(tool, reference, start_chunk=100).success
    assert not _read(tool, reference, start_offset=9999).success


def test_small_document_is_complete_and_preserves_conditions():
    tool, reference, _ = _tool(_chunks(["Druck\n10 bar", "Nur nach Freigabe.\nKein Trinkwasser."]))
    result = _read(tool, reference)
    prepared = prepare_document_read(
        result.data["results"], request=result.data["document_read"], budget=3000, source_numbers={}
    )
    assert "10 bar" in prepared.text and "Nur nach Freigabe." in prepared.text
    assert "Kein Trinkwasser." in prepared.text
    assert "Dokument vollstaendig gelesen" in prepared.text
    reading = prepared.chunks[0]["metadata"]["document_read"]
    assert reading["complete_document"] and not reading["has_more"]
    assert reading["next_chunk"] is None


def test_pagination_recovers_all_original_characters_including_large_table():
    texts = [
        "| Teil | Bedingung |\n| --- | --- |\n| A | " + "Nur nach Freigabe " * 90 + "|",
        "Frist: 31.12.2026.\nKeine automatische Verlaengerung. " * 13,
        "Ende des Dokuments.",
    ]
    tool, reference, _ = _tool(_chunks(texts))
    start, offset = 0, 0
    recovered = ["" for _ in texts]
    calls = 0
    while True:
        result = _read(tool, reference, start_chunk=start, start_offset=offset, max_chunks=2)
        prepared = prepare_document_read(
            result.data["results"],
            request=result.data["document_read"],
            budget=1100,
            source_numbers={},
        )
        assert prepared.chunks and len(prepared.text) <= 1100
        reading = prepared.chunks[0]["metadata"]["document_read"]
        assert (reading["start_chunk"], reading["start_offset"]) == (start, offset)
        for index in range(start, reading["end_chunk"] + 1):
            low = offset if index == start else 0
            high = reading["end_offset"] if index == reading["end_chunk"] else len(texts[index])
            piece = texts[index][low:high]
            assert piece.strip() in prepared.chunks[0]["text"]
            recovered[index] += piece
        calls += 1
        assert calls < 30
        if not reading["has_more"]:
            break
        assert (reading["next_chunk"], reading["next_offset"]) > (start, offset)
        start, offset = reading["next_chunk"], reading["next_offset"]
    assert recovered == texts
    assert calls > 2


def test_retriever_enforces_source_and_acl_even_after_store_filter():
    with patch("kiara.retrieval.retriever.VectorStore") as store:
        retriever = Retriever({}, embedding_model="model-a")
        store.return_value.get_document_chunks.return_value = {
            "ids": ["chunk-1"],
            "documents": ["Inhalt"],
            "metadatas": [{"source": "anleitung.txt", "ds_ids": [8]}],
        }
        with pytest.raises(RetrievalError):
            retriever.load_document_range("anleitung.txt", 0, 1, [7])
        assert retriever.load_document_range("anleitung.txt", 0, 1, []) == []


def test_read_pagination_is_not_blocked_after_five_distinct_pages():
    history = [
        ("read_document", _freeze_params({"document_id": "doc_a", "start_chunk": i}))
        for i in range(7)
    ]
    allowed = AgentChain._filter_tool_calls(
        [("read_document", {"document_id": "doc_a", "start_chunk": 7}, "call")], history, 5
    )
    assert len(allowed.executable) == 1
    repeated = AgentChain._filter_tool_calls(
        [("read_document", {"document_id": "doc_a", "start_chunk": 6}, "call")], history, 5
    )
    assert repeated.looped


def test_oversized_search_passage_keeps_reference_for_targeted_read():
    chunk = DocumentReferences().annotate(_chunks(["| Wert |\n| --- |\n| " + "Text " * 1000 + "|"]))
    prepared = prepare_context(chunk, budget=400)
    assert not prepared.chunks
    assert chunk[0]["metadata"]["document_ref"] in prepared.text


@pytest.mark.parametrize("retriever_class", [Retriever, HybridRetriever])
@pytest.mark.parametrize("allowed_ds", [[7], None])
def test_document_range_requires_current_postgres_membership(retriever_class, allowed_ds):
    with (
        patch("kiara.retrieval.retriever.VectorStore") as store,
        patch("kiara.ingestion._file_utils.ChunkDatasources") as memberships,
    ):
        retriever = retriever_class({}, embedding_model="model-a")
        store.return_value._embed_model = "model-a"
        store.return_value.get_collection_name.return_value = "kb_model-a"
        store.return_value.get_document_chunks.return_value = {
            "ids": ["chunk-0", "chunk-1"],
            "documents": ["Anfang", "Nur ehemals freigegeben"],
            "metadatas": [{"source": "anleitung.txt", "ds_ids": [7]}] * 2,
        }
        lookup = memberships.return_value.get_datasource_ids_for_chunks
        lookup.return_value = {"chunk-0": [7]}
        with pytest.raises(RetrievalError, match="nicht vollstaendig freigegeben"):
            retriever.load_document_range("anleitung.txt", 0, 2, allowed_ds)
        lookup.assert_called_once_with(["chunk-0", "chunk-1"], "model-a")
        lookup.return_value["chunk-1"] = [7]
        assert len(retriever.load_document_range("anleitung.txt", 0, 2, allowed_ds)) == 2
        lookup.side_effect = RuntimeError("PostgreSQL offline")
        with pytest.raises(RetrievalError, match="nicht geprueft") as error:
            retriever.load_document_range("anleitung.txt", 0, 2, allowed_ds)
        assert str(error.value.__cause__) == "PostgreSQL offline"


def test_document_range_postgres_lookup_is_bound_to_chunk_id_and_model(test_db):
    ds_id = test_db.execute(
        "INSERT INTO datasources (name, type, path) VALUES ('Read Test', 'local', '/tmp/read-test') "
        "RETURNING id"
    ).fetchone()["id"]
    test_db.execute(
        "INSERT INTO chunk_datasources (chunk_id, datasource_id, embedding_model) "
        "VALUES ('read-test', %s, 'model-a'), ('read-test', %s, 'model-b')",
        (ds_id, ds_id),
    )
    test_db.commit()
    with patch("kiara.retrieval.retriever.VectorStore") as store:
        store.return_value._embed_model = "model-a"
        store.return_value.get_collection_name.return_value = "kb_model-a"
        store.return_value.get_document_chunks.return_value = {
            "ids": ["read-test"],
            "documents": ["Inhalt"],
            "metadatas": [{"source": "anleitung.txt", "ds_ids": [ds_id]}],
        }
        retriever = HybridRetriever({}, embedding_model="model-a")
        assert retriever.load_document_range("anleitung.txt", 0, 1, [ds_id])
        test_db.execute(
            "DELETE FROM chunk_datasources WHERE chunk_id = 'read-test' "
            "AND embedding_model = 'model-a'"
        )
        test_db.commit()
        # Qdrant bleibt unveraendert; die Freigabe im anderen Modell hilft nicht.
        with pytest.raises(RetrievalError, match="nicht vollstaendig freigegeben"):
            retriever.load_document_range("anleitung.txt", 0, 1, [ds_id])


@pytest.mark.parametrize(
    "metadata",
    [
        {"modified": "2026-10-01T12:00:00+00:00"},
        {"size_bytes": 999},
        {"file_hash": "changed-hash"},
        {"modified": None},
        {"size_bytes": None},
    ],
)
def test_continuation_rejects_changed_version_with_same_chunk_total(metadata):
    chunks = _chunks(["Erster Abschnitt " * 30, "Zweiter Abschnitt"])
    tool, reference, retriever = _tool(chunks)
    first = _read(tool, reference, max_chunks=1)
    assert first.success
    changed = deepcopy(chunks)
    for chunk in changed:
        chunk["metadata"].update(metadata)
    retriever.load_document_range.side_effect = lambda source, start, end, **kw: changed[start:end]
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        _read(tool, reference, start_chunk=0, start_offset=20, max_chunks=1)
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        _read(tool, reference, start_chunk=1, max_chunks=1)


def test_continuation_rejects_changed_chunk_even_with_unchanged_file_metadata():
    chunks = _chunks(["Original " * 20])
    tool, reference, retriever = _tool(chunks)
    assert _read(tool, reference).success
    chunks[0]["text"] = "Eingefuegt " + chunks[0]["text"]
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        _read(tool, reference, start_offset=10)


@pytest.mark.parametrize("changed_total", [False, True])
def test_new_search_reference_cannot_mix_versions_with_earlier_read_pages(changed_total):
    chunks = _chunks(["Alte Fassung"])
    tool, reference, retriever = _tool(chunks)
    assert _read(tool, reference).success
    changed = (
        _chunks(["Neue Fassung", "Weiterer Chunk"], size_bytes=chunks[0]["metadata"]["size_bytes"])
        if changed_total
        else _chunks(["Neue Fassung"], modified="2026-10-01T12:00:00+00:00")
    )
    new_reference = DocumentReferences().annotate(changed)[0]["metadata"]["document_ref"]
    assert new_reference != reference
    retriever.load_document_range.side_effect = None
    retriever.load_document_range.return_value = changed
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        tool._references.annotate(changed)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("modified", ["2026-10-01T16:00:00+00:00", None])
def test_search_after_read_rejects_changed_version_before_deduplication(index, modified):
    chunks = _chunks(["Alte Fassung", "Alter Nachbar"])
    reader, reference, retriever = _tool(chunks)
    search = SearchDocumentsTool(retriever, None, {}, document_references=reader._references)
    with patch.object(search, "_retrieve_and_rerank", return_value=[chunks[0]]):
        found = asyncio.run(search.execute({"query": "Dokumentinhalt pruefen"}))
    assert found.success and _read(reader, reference, max_chunks=1).success
    seen = set(search._seen_keys)
    changed = deepcopy(chunks[index])
    changed["text"] = "NEUE FASSUNG darf nicht in den Antwortkontext"
    changed["metadata"]["modified"] = modified
    with patch.object(search, "_retrieve_and_rerank", return_value=[changed]):
        with pytest.raises(RetrievalError, match="Dokumentversion"):
            asyncio.run(search.execute({"query": "Weitere Dokumentangaben"}))
    assert search._seen_keys == seen


def test_version_check_is_atomic_for_search_batch():
    chunks = _chunks(["Original"])
    reader, reference, _ = _tool(chunks)
    assert _read(reader, reference).success
    other = _chunks(["Anderes Dokument"], source="other.txt")[0]
    changed = _chunks(["Neue Fassung"])[0]
    before = deepcopy(reader._references._references)
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        reader._references.annotate([other, changed])
    assert reader._references._references == before


def test_later_search_checks_content_of_already_read_chunks():
    chunks = _chunks(["Original"])
    reader, reference, _ = _tool(chunks)
    assert _read(reader, reference).success
    changed = deepcopy(chunks)
    changed[0]["text"] = "Aenderung bei unveraenderten Dateimetadaten"
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        reader._references.annotate(changed)


def test_parser_representations_remain_distinct_in_search_and_context():
    from kiara.generation.chain import expand_sibling_chunks

    ocr = _chunks(["OCR Anfang", "OCR Ende"], parser="docling")
    vlm = _chunks(["VLM Anfang", "VLM Ende"], parser="vlm")
    references = DocumentReferences()
    annotated = references.annotate([*ocr, *vlm])
    assert annotated[0]["metadata"]["document_ref"] != annotated[2]["metadata"]["document_ref"]
    search = SearchDocumentsTool(MagicMock(), None, {}, document_references=references)
    assert len(search._deduplicate_results(annotated)) == 4
    selected = expand_sibling_chunks([ocr[0]], [*ocr, *vlm])
    assert [chunk["text"] for chunk in selected] == ["OCR Anfang", "OCR Ende"]
    prepared = prepare_context(annotated, budget=2000)
    assert len(prepared.chunks) == 2


@pytest.mark.parametrize(
    "metadata", [{"modified": None}, {"size_bytes": None}, {"size_bytes": True}]
)
def test_missing_version_does_not_issue_readable_reference(metadata):
    chunk = _chunks(["Inhalt"], document_ref="doc_stale", **metadata)
    assert "document_ref" not in DocumentReferences().annotate(chunk)[0]["metadata"]


def _ingest_file(path, *, size=100, overlap=0, context=None):
    from kiara.ingestion.runtime import ProcessingContext, _process_single_file

    texts, metas, _, _, _ = _process_single_file(
        str(path),
        {"ingestion": {"chunk_size": size, "chunk_overlap": overlap}},
        {"chunk_size": size, "chunk_overlap": overlap},
        7,
        context or ProcessingContext(),
        {"errors": 0},
    )
    # DS und Modell werden beim Schreiben/Suchen ergaenzt; die Dokument- und
    # Chunking-Metadaten stammen unveraendert aus Loader und Ingestion.
    return [
        {"text": text, "metadata": {**meta, "ds_ids": [7], "embedding_model": "model-a"}}
        for text, meta in zip(texts, metas, strict=True)
    ]


@pytest.mark.parametrize("first_size,second_size,second_overlap", [(100, 110, 0), (110, 110, 10)])
def test_reingestion_with_changed_boundaries_rejects_unread_chunk(
    tmp_path, first_size, second_size, second_overlap
):
    path = tmp_path / "document.txt"
    path.write_text("A" * 300)
    original = _ingest_file(path, size=first_size)
    reader, reference, retriever = _tool(original)
    assert _read(reader, reference, max_chunks=1).success
    changed = _ingest_file(path, size=second_size, overlap=second_overlap)
    assert len(original) == len(changed) == 3
    for field in ("modified", "size_bytes", "file_hash", "chunk_total"):
        assert original[0]["metadata"].get(field) == changed[0]["metadata"].get(field)
    assert len({c["metadata"]["processing_version"] for c in changed}) == 1
    assert (
        original[0]["metadata"]["processing_version"]
        != changed[0]["metadata"]["processing_version"]
    )
    retriever.load_document_range.side_effect = lambda source, start, end, **kw: changed[start:end]
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        _read(reader, reference, start_chunk=1, max_chunks=1)
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        reader._references.annotate([changed[2]])
    fresh, fresh_reference, _ = _tool(changed)
    assert fresh_reference != reference
    assert _read(fresh, fresh_reference).success


def test_processing_version_covers_changes_outside_already_read_chunks(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("A" * 100 + "B" * 100 + "C" * 100)
    original = _ingest_file(path)
    assert _ingest_file(path) == original
    reader, reference, retriever = _tool(original)
    assert _read(reader, reference, max_chunks=1).success
    stat = path.stat()
    path.write_text("A" * 100 + "D" * 100 + "C" * 100)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = _ingest_file(path)
    assert original[0]["text"] == changed[0]["text"]
    assert original[0]["metadata"]["modified"] == changed[0]["metadata"]["modified"]
    retriever.load_document_range.side_effect = lambda source, start, end, **kw: changed[start:end]
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        _read(reader, reference, start_chunk=1, max_chunks=1)


@pytest.mark.parametrize(
    "metadata",
    [{"chunk_size_used": 110}, {"chunk_overlap_used": 10}, {"chunk_strategy": "markdown"}],
)
def test_reference_binds_stored_chunking_parameters(metadata):
    original = _chunks(["A" * 100, "B" * 100], chunk_size_used=100, chunk_strategy="custom")
    reader, reference, retriever = _tool(original)
    assert _read(reader, reference, max_chunks=1).success
    changed = deepcopy(original)
    changed[1]["metadata"].update(metadata)
    retriever.load_document_range.side_effect = lambda source, start, end, **kw: changed[start:end]
    with pytest.raises(RetrievalError, match="Dokumentversion"):
        _read(reader, reference, start_chunk=1)


@pytest.mark.parametrize("with_ocr", [False, True])
def test_vlm_from_real_loader_and_ingestion_can_be_found_and_read(tmp_path, with_ocr):
    from PIL import Image

    from kiara.ingestion.runtime import ProcessingContext

    path = tmp_path / "diagram.png"
    Image.new("RGB", (16, 16), "white").save(path)
    description = "Typenschild der Pumpe. Betriebsdruck 10 bar. Nur nach Freigabe verwenden."
    context = ProcessingContext(
        ddu_enabled=with_ocr,
        ddu_backends=[{"url": "http://unused"}],
        vlm_enabled=True,
        vlm_model="test-vlm",
        vlm_url="http://unused",
    )
    with (
        patch("ollama.Client") as client,
        patch(
            "kiara.ingestion.loader._load_with_docling",
            return_value=("OCR-Beschriftung: Pumpe 10 bar", "", "docling-remote"),
        ),
    ):
        client.return_value.chat.return_value = {"message": {"content": description}}
        chunks = _ingest_file(path, context=context)
        client.return_value.chat.assert_called_once()
    assert len(chunks) == (2 if with_ocr else 1)
    vlm_chunks = [c for c in chunks if c["metadata"]["parser"] == "vlm"]
    assert vlm_chunks[0]["metadata"]["chunk_total"] == 1
    assert vlm_chunks[0]["metadata"]["processing_version"]
    if with_ocr:
        assert (
            chunks[0]["metadata"]["processing_version"]
            != vlm_chunks[0]["metadata"]["processing_version"]
        )
    references = DocumentReferences()
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    search = SearchDocumentsTool(retriever, None, {}, document_references=references)
    found = asyncio.run(search.execute({"query": "Betriebsdruck"}))
    found_vlm = next(c for c in found.data["results"] if c["metadata"]["parser"] == "vlm")
    reference = found_vlm["metadata"]["document_ref"]
    retriever.load_document_range.return_value = vlm_chunks
    reader = ReadDocumentTool(retriever, {}, references, [7])
    read = _read(reader, reference)
    prepared = prepare_document_read(
        read.data["results"], request=read.data["document_read"], budget=2000, source_numbers={}
    )
    assert prepared.chunks[0]["text"] == description
    assert prepared.chunks[0]["metadata"]["document_read"]["complete_document"]
    retriever.load_document_range.assert_called_once_with(
        str(path), 0, 1, filter_datasource_ids=[7], parser="vlm"
    )


@pytest.mark.parametrize(
    "text",
    [
        "A" * 260 + "\n" + " " * 400 + "B" * 300,
        " " * 900 + "Nutztext",
        "Anfang\n" + " \n\t" * 400,
    ],
)
def test_read_continuation_preserves_whitespace_and_always_advances(text):
    reader, reference, _ = _tool(_chunks([text]))
    offset, recovered = 0, ""
    for _ in range(30):
        read = _read(reader, reference, start_offset=offset)
        prepared = prepare_document_read(
            read.data["results"], request=read.data["document_read"], budget=750, source_numbers={}
        )
        assert prepared.chunks and len(prepared.text) <= 750
        reading = prepared.chunks[0]["metadata"]["document_read"]
        assert reading["end_offset"] > offset
        assert prepared.chunks[0]["text"] == text[offset : reading["end_offset"]]
        recovered += prepared.chunks[0]["text"]
        if not reading["has_more"]:
            break
        offset = reading["next_offset"]
    else:
        pytest.fail("Lesecursor kommt nicht bis zum Dokumentende")
    assert recovered == text


@pytest.mark.parametrize(
    "texts",
    [
        ["\u00ad" * 1000 + "B" * 1000],
        ["\u00ad" * 1000, "B" * 1000],
        ["\u00ad" * 1000],
        ["Anfang" + "\u00ad" * 1000],
        ["Start", " \n\t" * 100, "Ende"],
        ["<|sys\u00adtem|>" * 100 + "Text " * 300],
    ],
)
def test_read_skips_removed_characters_with_original_cursors(texts):
    from kiara.shared.prompt_defense import sanitize_with_report

    reader, reference, _ = _tool(_chunks(texts))
    index, offset, visible = 0, 0, []
    for _ in range(30):
        read = _read(reader, reference, start_chunk=index, start_offset=offset, max_chunks=1)
        prepared = prepare_document_read(
            read.data["results"], request=read.data["document_read"], budget=750, source_numbers={}
        )
        assert prepared.chunks and len(prepared.text) <= 750
        page = prepared.chunks[0]
        cursor = page["metadata"]["document_read"]
        assert (cursor["start_chunk"], cursor["start_offset"]) == (index, offset)
        assert cursor["end_offset"] > offset
        assert (
            page["text"]
            == sanitize_with_report(texts[index][offset : cursor["end_offset"]]).clean_text
        )
        visible.append(page["text"])
        if not cursor["has_more"]:
            break
        index, offset = cursor["next_chunk"], cursor["next_offset"]
    else:
        pytest.fail("Lesecursor bleibt in bereinigtem Bereich stehen")
    assert "".join(visible) == "".join(sanitize_with_report(t).clean_text for t in texts)


def test_default_budget_releases_search_with_sanitized_source():
    from kiara.tools.registry import ToolRegistry

    text = "Originaltext mit Bedingungen. " * 600
    chunk = DocumentReferences().annotate(_chunks([text], source="doc\u00adument.txt"))[0]
    reference = chunk["metadata"]["document_ref"]
    chain = AgentChain({"chat": {"agent": {"final_reranking": False}}}, ToolRegistry(), MagicMock())
    _, _, total = chain._prepare_tool_result({"raw_chunks": [chunk]}, "search_documents", "", 0)
    result = {
        "raw_chunks": [chunk],
        "document_read": {
            "document_id": reference,
            "start_chunk": 0,
            "start_offset": 0,
            "total_chunks": 1,
        },
    }
    _, read, total = chain._prepare_tool_result(result, "read_document", "", total)
    assert read[0]["text"] == text
    assert result["document_read"]["complete_document"]
    assert chain._document_context_results[0]["chunks"] == []
    assert total == chain._retrieval_context_chars < 32000


def test_removed_middle_chunk_is_processed_without_claiming_an_unloaded_gap():
    reader, reference, _ = _tool(_chunks(["Anfang", "\u00ad" * 1000, "Ende"]))
    result = _read(reader, reference, max_chunks=3)
    prepared = prepare_document_read(
        result.data["results"],
        request=result.data["document_read"],
        budget=750,
        source_numbers={},
    )
    assert prepared.chunks[0]["text"] == "Anfang\nEnde"
    assert prepared.chunks[0]["metadata"]["document_read"]["complete_document"]
    assert prepared.chunks[0]["metadata"]["context_spans"] == [
        {"chunk_index": 0, "start": 0, "end": 6},
        {"chunk_index": 2, "start": 0, "end": 4},
    ]
