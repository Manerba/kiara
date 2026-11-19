"""Deterministic CAD extraction, publication and queries."""

SCHEMA_VERSION = 1
READER_REVISION = "a9f78e632147783fc199b035a2ec5dd3392034d9"
ADAPTER_REVISION = "pkg-v6-nested-containers"
PARSER = "cad-pkg"


def revision():
    return f"{READER_REVISION}:{SCHEMA_VERSION}:{ADAPTER_REVISION}"
