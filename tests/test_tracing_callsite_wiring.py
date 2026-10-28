import ast
from pathlib import Path

import pytest

CASES = [
    ("kiara/generation/chain.py", "retrieval"),
    ("kiara/retrieval/retriever.py", "retrieval"),
    ("kiara/retrieval/reranker.py", "retrieval"),
    ("kiara/generation/agent_chain.py", "agent"),
    ("kiara/ingestion/runtime.py", "ingestion"),
]


def _extract_pipeline_argument(path: str) -> str | None:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "_tracer" for t in node.targets):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "get_tracer":
            continue
        for kw in call.keywords:
            if kw.arg == "pipeline" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    return None


@pytest.mark.parametrize(("path", "pipeline"), CASES)
def test_module_level_tracer_is_wired_to_expected_pipeline(path, pipeline):
    assert _extract_pipeline_argument(path) == pipeline
