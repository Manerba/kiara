"""Offline ES maintenance entry point; see docs/operations/elasticsearch-s4.md.

The persistent marker blocks ES clients. Systemd conditions in the runbook also
block service starts by deploy.sh/reboot. Only the exclusive offline session
may bypass the marker; a successful rebuild alone never removes it.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_session = ContextVar("es_maintenance_session", default=None)
SERVICES = (
    "kiara.service",
    "kiara-admin.service",
    "kiara-admin-chat.service",
    "kiara-worker.service",
    "kiara-mcp.service",
    "kiara-sync.service",
)


def marker_path(config):
    return Path(
        config.get("retrieval", {}).get(
            "elasticsearch_maintenance_file", "/var/lib/kiara/elasticsearch-maintenance.json"
        )
    )


def binding(config):
    from kiara.config import get_embedding_model

    # Hash endpoint credentials too, without persisting them in reports.
    relevant = {
        "retrieval": config.get("retrieval", {}),
        "model": get_embedding_model(config),
        "vectordb": config.get("vectordb", {}),
        "database": config.get("database", {}),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def assert_access(config):
    path = marker_path(config)
    if path.exists() and (_session.get() is None or _session.get() != (str(path), binding(config))):
        raise RuntimeError("Elasticsearch ist fuer Wartung gesperrt; Offline-Runbook verwenden")


def require_session(config):
    path = marker_path(config)
    if not path.exists() or _session.get() != (str(path), binding(config)):
        raise RuntimeError("Offline-Wartung erforderlich: docs/operations/elasticsearch-s4.md")


def assert_stopped():
    for service in SERVICES:
        check = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True)
        if check.stdout.strip() not in {"inactive", "failed", "unknown"}:
            raise RuntimeError(f"Wartung braucht gestoppten Dienst: {service}")


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def begin(config):
    path = marker_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump({"binding": binding(config), "state": "blocked"}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)
    return {"status": "blocked", "marker": str(path)}


@contextmanager
def offline_session(config):
    path = marker_path(config)
    # Never unlink the lock inode; another waiter must not acquire a stale inode.
    with Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(path.read_text())
        if state.get("binding") != binding(config):
            raise RuntimeError("Wartungsmarker passt nicht zu Config/Modell")
        assert_stopped()
        token = _session.set((str(path), binding(config)))
        try:
            yield
        finally:
            _session.reset(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preview", "begin", "rebuild", "verify", "release", "delete", "retained"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    from kiara.config import load_config
    from kiara.retrieval import elasticsearch_service as es
    from kiara.shared.db import init_db_backend

    config = load_config(args.config)
    init_db_backend(config)
    report = Path(args.report)
    # Reserve report before any mutation; do not overwrite previous evidence.
    with report.open("x") as output:
        try:
            if args.command == "begin":
                result = begin(config)
            elif args.command == "preview" and not marker_path(config).exists():
                result = es.rebuild_from_vectorstore(config, dry_run=True)
            else:
                with offline_session(config):
                    if args.command in {"preview", "rebuild"}:
                        result = es.rebuild_from_vectorstore(
                            config, dry_run=args.command == "preview"
                        )
                    elif args.command == "retained":
                        from kiara.retrieval.elasticsearch_indices import datasource_index_name

                        # Migration evidence only, never a runtime search fallback.
                        datasource_index_name(config, 0)
                        name = config.get("retrieval", {}).get("elasticsearch_index", "kiara")
                        client = es._get_client(config)
                        if client.indices.exists(index=name):
                            stats = client.indices.stats(index=name, metric="docs", level="indices")
                            shards = stats["_shards"]
                            if shards["failed"] or shards["successful"] != shards["total"]:
                                raise RuntimeError("Altindex-Messung unvollstaendig")
                            entry = stats["indices"][name]
                            result = {
                                "index": name,
                                "uuid": entry["uuid"],
                                "primary_documents": entry["primaries"]["docs"]["count"],
                            }
                        else:
                            result = {"index": name, "absent": True}
                    elif args.command == "delete":
                        result = {"status": "success" if es.delete_index(config) else "error"}
                    else:
                        result = es.verify_integrity(config, full=True)
                    if args.command == "release" and result.get("release_ready"):
                        # Persist the full gate evidence BEFORE lifting the marker.
                        json.dump(result, output, indent=2)
                        output.flush()
                        os.fsync(output.fileno())
                        _sync_directory(report.parent)
                        marker_path(config).unlink()
                        _sync_directory(marker_path(config).parent)
                        return
            json.dump(result, output, indent=2)
        except BaseException as exc:
            if output.tell() == 0:
                json.dump(
                    {
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "release_ready": False,
                    },
                    output,
                    indent=2,
                )
                output.flush()
                os.fsync(output.fileno())
            raise
    if result.get("status") == "error" or (
        args.command == "release" and not result.get("release_ready")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    # python -m creates a separate __main__ module. Services import the canonical
    # module, so the CLI must use that module's session ContextVar as well.
    from kiara.retrieval.elasticsearch_maintenance import main as _main

    _main()
