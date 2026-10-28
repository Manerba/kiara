"""Audit regressions: real module entry, venv preflight and offline recovery."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_elasticsearch_s4 import backend, corpus, run  # noqa: F401

from kiara.retrieval import elasticsearch_maintenance as maintenance
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.embedding import collection_name_for_model

ROOT = Path(__file__).resolve().parents[1]


def test_documented_python_m_in_fresh_processes(tmp_path):
    # Isolate only configuration, transport and OS. The module/session boundary
    # and report/marker lifecycle are real, including a failed release.
    (tmp_path / "sitecustomize.py").write_text("""
import json, os, subprocess, sys
from types import SimpleNamespace
from kiara import config as cfg
from kiara.shared import db
from kiara.retrieval import elasticsearch_service as es
assert "kiara.retrieval.elasticsearch_maintenance" not in sys.modules
cfg.load_config = lambda path: json.loads(os.environ["S4_TEST_CONFIG"])
cfg.get_embedding_model = lambda config: "test-model"
db.init_db_backend = lambda config: None
subprocess.run = lambda *a, **kw: SimpleNamespace(stdout="inactive\\n")
def operation(config, **kwargs):
    from kiara.retrieval.elasticsearch_maintenance import require_session, assert_access
    require_session(config)
    assert_access(config)
    failed = os.environ.get("S4_TEST_FAIL") == "yes"
    return {"status": "error" if failed else "success", "release_ready": not failed}
es.rebuild_from_vectorstore = operation
es.verify_integrity = operation
""")
    marker = tmp_path / "marker.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)]),
        "S4_TEST_CONFIG": json.dumps(
            {"retrieval": {"elasticsearch_maintenance_file": str(marker)}}
        ),
    }
    for number, (command, failed) in enumerate(
        [
            ("begin", False),
            ("rebuild", False),
            ("verify", False),
            ("release", True),
            ("release", False),
        ]
    ):
        report = tmp_path / f"{number}-{command}.json"
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "kiara.retrieval.elasticsearch_maintenance",
                command,
                "--config",
                "isolated.yaml",
                "--report",
                str(report),
            ],
            env={**env, "S4_TEST_FAIL": "yes" if failed else "no"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == (1 if failed else 0), child.stderr
        data = json.loads(report.read_text())
        assert data["status"] == (
            "blocked" if command == "begin" else "error" if failed else "success"
        )
        assert marker.exists() == (command != "release" or failed)


def test_prepare_preflight_preserves_real_venv(tmp_path):
    # Execute the actual script prefix, ending before marker/systemd operations.
    # This uses the real venv and pip, never the destructive Prepare body.
    assert sys.prefix != sys.base_prefix, "Run this regression with the project venv"
    script = (ROOT / "scripts/elasticsearch_maintenance_prepare.sh").read_text()
    prefix = script.split('marker=$("$python"', 1)[0]
    assert "systemctl" not in prefix and "cp -a" not in prefix
    preflight = tmp_path / "preflight.sh"
    preflight.write_text(prefix)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    evidence = tmp_path / "evidence"
    child = subprocess.run(
        ["bash", str(preflight), str(config), str(evidence), sys.executable],
        env={**os.environ, "KIARA_ES_NO_OTHER_WRITERS": "yes"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    environment = json.loads((evidence / "python-environment.json").read_text())
    assert environment["prefix"] == sys.prefix
    assert environment["executable"] == os.path.abspath(sys.executable)
    assert environment["prefix"] != environment["base_prefix"]
    assert "elasticsearch==" in (evidence / "python-packages.txt").read_text().lower()


def add_publication(corpus, source="interrupted-without-points.txt", collection=None):
    corpus.db.execute(
        "INSERT INTO document_publications (collection_name,source,versions,generation) "
        "VALUES (%s,%s,%s::jsonb,1)",
        (collection or collection_name_for_model(corpus.model), source, "{}"),
    )
    corpus.db.commit()


@pytest.mark.parametrize("empty", [False, True])
def test_pointless_publication_blocks_preview_verify_release(
    corpus, backend, monkeypatch, tmp_path, empty
):
    if empty:
        corpus.db.execute("DELETE FROM chunk_datasources WHERE embedding_model=%s", (corpus.model,))
        corpus.db.commit()
        corpus.client.delete(
            corpus.store._collection, points_selector=[p.id for p in corpus.points]
        )
    assert run(corpus)["release_ready"]
    add_publication(corpus)
    with maintenance.offline_session(corpus.config):
        for result in [
            es.rebuild_from_vectorstore(corpus.config, dry_run=True),
            es.verify_integrity(corpus.config, full=True),
        ]:
            assert result["reason_code"] == "open_document_publications"
            assert result["open_publications"] == ["interrupted-without-points.txt"]
            assert not result["release_ready"]
    report = tmp_path / "release.json"
    monkeypatch.setattr("kiara.config.load_config", lambda path: corpus.config)
    monkeypatch.setattr("kiara.shared.db.init_db_backend", lambda config: None)
    monkeypatch.setattr(
        sys, "argv", ["maintenance", "release", "--config", "fixture.yaml", "--report", str(report)]
    )
    with pytest.raises(SystemExit) as exc:
        maintenance.main()
    assert exc.value.code == 1
    assert maintenance.marker_path(corpus.config).exists()
    assert not json.loads(report.read_text())["release_ready"]


def test_ledger_is_rechecked_after_full_scan(corpus, backend, monkeypatch):
    assert run(corpus)["release_ready"]
    original = backend.mget
    inserted = False

    def mget(*args, **kwargs):
        nonlocal inserted
        if not inserted:
            add_publication(corpus)
            inserted = True
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "mget", mget)
    with maintenance.offline_session(corpus.config):
        result = es.verify_integrity(corpus.config, full=True)
    assert result["matching"] == result["sampled"]
    assert result["reason_code"] == "open_document_publications"
    assert not result["release_ready"]


def test_ledger_diagnostics_are_bounded_and_model_specific(corpus, backend):
    add_publication(corpus, collection="kb_other_model")
    assert run(corpus)["release_ready"]
    for i in range(25):
        add_publication(corpus, source=f"pending-{i:02}.txt")
    result = run(corpus)
    assert not result["release_ready"]
    assert len(result["open_publications"]) == 20
    assert result["open_publications_truncated"]


def test_later_rebuild_requires_recovery_before_release(corpus, backend, monkeypatch, tmp_path):
    assert run(corpus)["release_ready"]
    backend.fail_bulk = True
    failed = run(corpus)
    assert failed["status"] == "error" and not failed["release_ready"]
    monkeypatch.setattr("kiara.config.load_config", lambda path: corpus.config)
    monkeypatch.setattr("kiara.shared.db.init_db_backend", lambda config: None)

    def release(filename):
        report = tmp_path / filename
        monkeypatch.setattr(
            sys,
            "argv",
            ["maintenance", "release", "--config", "fixture.yaml", "--report", str(report)],
        )
        maintenance.main()
        return json.loads(report.read_text())

    with pytest.raises(SystemExit):
        release("failed-release.json")
    assert maintenance.marker_path(corpus.config).exists()
    assert not json.loads((tmp_path / "failed-release.json").read_text())["release_ready"]
    backend.fail_bulk = False
    assert run(corpus)["release_ready"]
    assert maintenance.marker_path(corpus.config).exists()
    assert release("recovered-release.json")["release_ready"]
    assert not maintenance.marker_path(corpus.config).exists()


def test_verification_error_preserves_counts_and_details(corpus, backend):
    assert run(corpus)["release_ready"]
    docs = backend.data["s4_test_ds_0"]
    ids = list(docs)
    docs.pop(ids[0])
    docs[ids[1]]["text"] = "wrong"
    with maintenance.offline_session(corpus.config):
        result = es.verify_integrity(corpus.config, full=True)
    assert result["status"] == "error" and result["error"]
    assert result["missing_in_es"] == 1
    assert result["mismatch_count"] == 1
    assert result["mismatches"] == [{"chunk_id": ids[1], "index": "s4_test_ds_0"}]
    assert not result["counts_match"]


def test_admin_renders_error_evidence_and_escapes_source_values():
    script = (ROOT / "kiara/web/static/js/tab_ki_retrieval.js").read_text()
    function = script[
        script.index("function _esShowResult(result)") : script.index("// ---- User-Aktionen (AP4)")
    ]
    harness = r"""
const assert = require('node:assert/strict');
const div = {classList: {remove() {}, add() {}}, innerHTML: ''};
const document = {getElementById: () => div};
const SharedUi = {runtime: {escapeHtml: s => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')}};
_esShowResult({status: 'error', missing_in_es: 2, mismatch_count: 1,
    mismatches: [{index: '<index>', chunk_id: '<chunk>'}],
    datasources: [{index_name: '<index>', expected: 4, observed: 2}],
    open_publications: ['<source>']});
for (const expected of ['In ES fehlend: 2', 'Abweichende Inhalte: 1', '&lt;chunk&gt;',
    '&lt;index&gt;', 'erwartet 4, vorhanden 2', '&lt;source&gt;']) assert(div.innerHTML.includes(expected));
assert(!div.innerHTML.includes('Unbekannt'));
assert(!div.innerHTML.includes('<chunk>'));
_esShowResult({status: 'error', error: '<failure>'});
assert(div.innerHTML.includes('&lt;failure&gt;'));
"""
    child = subprocess.run(
        ["node", "-e", function + harness], capture_output=True, text=True, timeout=10
    )
    assert child.returncode == 0, child.stderr
