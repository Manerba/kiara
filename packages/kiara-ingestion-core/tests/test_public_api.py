from kiara_ingestion_core import (
    ChunkDatasources,
    FileLogEntry,
    HashStore,
    ProcessingContext,
    build_datasource_overrides,
    compute_file_hash,
    load_connector_config,
)


class _ConnectorStub:
    @staticmethod
    def secret_fields():
        return ["token"]


def test_public_api_exports_exist():
    entry = FileLogEntry(
        file_path="/tmp/a.txt",
        file_name="a.txt",
        file_size=1,
        file_ext=".txt",
    )
    ctx = ProcessingContext()

    assert callable(build_datasource_overrides)
    assert callable(load_connector_config)
    assert ChunkDatasources.__name__ == "ChunkDatasources"
    assert HashStore.__name__ == "HashStore"
    assert entry.status == "success"
    assert ctx.vlm_enabled is False


def test_build_datasource_overrides_typical_row():
    row = {
        "ds_chunk_size": 256,
        "ds_chunk_overlap": 32,
        "ds_file_types": '[".txt", ".md"]',
        "ds_exclude_patterns": '["*/tmp/*"]',
        "ds_chunk_strategy": "prose",
        "ds_contextual_retrieval": 1,
        "ds_enrichment_prompt_id": 7,
        "ds_enrichment_backend_id": 11,
        "restriction_level": "public",
    }

    overrides = build_datasource_overrides(row)

    assert overrides == {
        "chunk_size": 256,
        "chunk_overlap": 32,
        "file_types": [".txt", ".md"],
        "exclude_patterns": ["*/tmp/*"],
        "chunk_strategy": "prose",
        "contextual_retrieval": True,
        "enrichment_prompt_id": 7,
        "enrichment_backend_id": 11,
        "restriction_level": "public",
    }


def test_load_connector_config_plain_json():
    row = {
        "connector_config": '{"url": "https://example.invalid", "token": "plain"}',
    }

    config = load_connector_config(row, _ConnectorStub)

    assert config["url"] == "https://example.invalid"
    assert config["token"] == "plain"


def test_compute_file_hash_supports_mtime_and_content(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hallo Welt\n", encoding="utf-8")

    mtime_hash = compute_file_hash(str(path), "mtime")
    content_hash = compute_file_hash(str(path), "content")

    assert len(mtime_hash) == 32
    assert len(content_hash) == 32
    assert mtime_hash != ""
    assert content_hash != ""
