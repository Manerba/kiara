#!/usr/bin/env python3
"""Vollstaendige HTTP-Chat-/MCP-Laeufe; nutzt die vorhandenen Benchmark-Artefakte.

Ausfuehrung lokal als Betreiber mit DB-Zugriff. Temporäre Session/API-Key werden
im finally entfernt; Chats und Audits bleiben als Abnahmebeleg erhalten.
Fall-Datei: JSON-Liste mit id, question und optional answer_checks.
"""

import argparse
import hashlib
import json
import secrets
import subprocess
import time
from pathlib import Path

import httpx

from kiara.benchmarks.agent_retrieval import AgentRuntime, answer_checks
from kiara.benchmarks.artifacts import write_private_json
from kiara.config import load_config
from kiara.repositories.api_key import ApiKeyRepository
from kiara.repositories.user import SessionRepository
from kiara.shared.db import db_query
from kiara.web.auth import create_session, hash_token, sign_token


def sse(text):
    events, kind = [], ""
    for line in text.splitlines():
        if line.startswith("event: "):
            kind = line[7:]
        elif line.startswith("data: "):
            events.append({"event": kind, "data": json.loads(line[6:])})
    return events


def resources(backend):
    data = {}
    for service in ["kiara", "kiara-mcp"]:
        pid = subprocess.check_output(
            ["systemctl", "show", service, "-p", "MainPID", "--value"], text=True
        ).strip()
        path = Path(f"/proc/{pid}/status")
        main_status = (
            [
                line
                for line in path.read_text().splitlines()
                if line.startswith(("VmRSS:", "VmHWM:"))
            ]
            if path.exists()
            else []
        )
        group = subprocess.check_output(
            ["systemctl", "show", service, "-p", "ControlGroup", "--value"], text=True
        ).strip()
        cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
        data[service] = {"main_pid_status": main_status, "cgroup": group}
        for metric in ["memory.current", "memory.peak", "memory.swap.current"]:
            path = cgroup / metric
            if path.exists():
                data[service][metric] = int(path.read_text().strip())
    try:
        data["ollama_ps"] = httpx.get(backend["url"].rstrip("/") + "/api/ps", timeout=10).json()
    except Exception as exc:
        data["ollama_ps_error"] = str(exc)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--modes", default="standard,rewriter,agent,research")
    parser.add_argument("--interfaces", default="chat,mcp")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    config = load_config()
    runtime = AgentRuntime(args.backend_id)
    from kiara.retrieval.vectorstore import VectorStore

    write_private_json(
        args.output / "fingerprint.json",
        {
            "runtime": {
                **runtime.fingerprint(),
                "scope": "authenticated user permissions",
                "user_id": args.user_id,
            },
            "research_config": config["chat"]["research"],
            "num_ctx": args.num_ctx,
            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "resources_before": resources(runtime.backend),
            "collections_before": VectorStore.list_collections(config),
        },
    )
    with db_query() as db:
        user = dict(
            db.execute(
                "SELECT id,is_admin FROM users WHERE id=%s AND is_admin=false AND is_active=true",
                (args.user_id,),
            ).fetchone()
        )
    session, csrf = create_session(user["id"])
    key = "orag_" + secrets.token_hex(32)
    repo = ApiKeyRepository()
    key_id = repo.create(
        name="P78 temporaerer Abnahme-Selftest",
        key_hash=hashlib.sha256(key.encode()).hexdigest(),
        key_prefix=key[:12],
        user_id=user["id"],
        is_active=True,
    )
    chat_url = "http://127.0.0.1:" + str(config.get("web", {}).get("port", 8501))
    mcp_url = "http://127.0.0.1:" + str(config["web"]["mcp_port"]) + "/mcp/"
    results = []
    try:
        with httpx.Client(timeout=700, follow_redirects=True) as client:
            client.cookies.set("kiara_session", sign_token(session))
            headers = {
                "X-CSRF-Token": csrf,
                "X-Requested-With": "XMLHttpRequest",
                "Origin": chat_url,
            }
            modes = client.get(chat_url + "/api/chat/modes")
            modes.raise_for_status()
            write_private_json(args.output / "chat-modes.json", modes.json())
            assert any(m["chain_type"] == "research" for m in modes.json()["modes"])
            for case in cases:
                if "chat" in args.interfaces.split(","):
                    for mode in args.modes.split(","):
                        started = time.monotonic()
                        print(case["id"], mode, "start", flush=True)
                        response = client.post(
                            chat_url + "/api/chat/new/query",
                            headers=headers,
                            json={
                                "question": case["question"],
                                "mode": mode,
                                "k": 20,
                                "llm_backend_id": args.backend_id,
                                "num_ctx": args.num_ctx,
                                "think": False,
                            },
                        )
                        response.raise_for_status()
                        events = sse(response.text)
                        done = next(
                            (e["data"] for e in reversed(events) if e["event"] == "done"), {}
                        )
                        chat_id = done.get("chat_id")
                        answer = "".join(
                            e["data"].get("text", "") for e in events if e["event"] == "chunk"
                        )
                        loaded = client.get(chat_url + f"/api/chat/{chat_id}")
                        loaded.raise_for_status()
                        with db_query() as db:
                            audit = db.execute(
                                "SELECT * FROM query_log WHERE chat_id=%s ORDER BY id DESC LIMIT 1",
                                (chat_id,),
                            ).fetchone()
                        record = {
                            "case": case,
                            "interface": "chat",
                            "mode": mode,
                            "answer": answer,
                            "latency_ms": int((time.monotonic() - started) * 1000),
                            "events": events,
                            "reloaded": loaded.json(),
                            "audit": dict(audit) if audit else None,
                            "answer_check": answer_checks(case, answer),
                            "resources": resources(runtime.backend),
                        }
                        write_private_json(
                            args.output / f"{case['id']}-{mode}.json",
                            json.loads(json.dumps(record, default=str)),
                        )
                        results.append(
                            {
                                k: record[k]
                                for k in ["interface", "mode", "latency_ms", "answer_check"]
                            }
                            | {"id": case["id"], "chat_id": chat_id}
                        )
                        print(case["id"], mode, "done", record["latency_ms"], flush=True)
                if "mcp" in args.interfaces.split(","):
                    wire_metrics = []

                    def call(name, arguments, *, metrics=wire_metrics):
                        response = client.post(
                            mcp_url,
                            headers={
                                "Authorization": "Bearer " + key,
                                "Accept": "application/json, text/event-stream",
                            },
                            json={
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "tools/call",
                                "params": {"name": name, "arguments": arguments},
                            },
                        )
                        response.raise_for_status()
                        envelope = (
                            sse(response.text)[0]["data"]
                            if "data: " in response.text
                            else response.json()
                        )
                        serialized_chars = len(
                            json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
                        )
                        metrics.append(
                            {
                                "tool": name,
                                "serialized_mcp_chars": serialized_chars,
                                "http_stream_chars": len(response.text),
                                "transport_ping_count": sum(
                                    line.startswith(": ping") for line in response.text.splitlines()
                                ),
                            }
                        )
                        if "max_chars" in arguments:
                            assert serialized_chars <= arguments["max_chars"], (
                                "Serialized MCP result budget exceeded"
                            )
                        return envelope["result"]["structuredContent"]

                    print(case["id"], "mcp", "start", flush=True)
                    started = time.monotonic()
                    page = call(
                        "research",
                        {
                            "question": case["question"],
                            "k": 20,
                            "max_chars": 12000,
                            "llm_backend_id": args.backend_id,
                            "num_ctx": args.num_ctx,
                        },
                    )
                    pages = [page]
                    while page.get("next_offset") is not None:
                        page = call(
                            "research_result",
                            {
                                "run_id": page["run_id"],
                                "offset": page["next_offset"],
                                "max_chars": 12000,
                            },
                        )
                        pages.append(page)
                    assert all(len(json.dumps(p, ensure_ascii=False)) <= 12000 for p in pages)
                    result = (
                        json.loads("".join(p["content"] for p in pages))
                        if "content" in pages[0]
                        else pages[0]
                    )
                    with db_query() as db:
                        audit = db.execute(
                            "SELECT * FROM query_log WHERE correlation_id=%s "
                            "AND routed_to='mcp:research' ORDER BY id DESC LIMIT 1",
                            (result["run_id"],),
                        ).fetchone()
                    assert audit and audit["chain_type"] == "research"
                    assert all(c["budget"]["fits"] for c in result["model_calls"])
                    assert all(
                        source["verified_original"] and source["version"] and source["chunk_ids"]
                        for source in result["sources"]
                    )
                    assert all(
                        quote in source["original_excerpt"]
                        for source in result["sources"]
                        for quote in source["quotes"]
                    )
                    record = {
                        "case": case,
                        "interface": "mcp",
                        "pages": pages,
                        "wire_metrics": wire_metrics,
                        "result": result,
                        "audit": dict(audit),
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "answer_check": answer_checks(case, result.get("answer", "")),
                        "resources": resources(runtime.backend),
                    }
                    write_private_json(
                        args.output / f"{case['id']}-mcp.json",
                        json.loads(json.dumps(record, default=str)),
                    )
                    results.append(
                        {k: record[k] for k in ["interface", "latency_ms", "answer_check"]}
                        | {"id": case["id"], "status": result["status"]}
                    )
                    print(case["id"], "mcp", "done", result["status"], flush=True)
                write_private_json(args.output / "summary.json", results)
    finally:
        SessionRepository().delete_by_token(hash_token(session))
        repo.delete(key_id)
        write_private_json(args.output / "summary.json", results)
        write_private_json(
            args.output / "resources-after.json",
            {
                "resources": resources(runtime.backend),
                "collections_after": VectorStore.list_collections(config),
                "note": "Service-cgroup includes workers; peak is since service start, not reset per query. Ollama /api/ps is a snapshot, not measured GPU peak.",
            },
        )


if __name__ == "__main__":
    main()
