#!/usr/bin/env python3
"""Build/check the reviewed P79 atlas, source evidence and offline Mermaid SVGs.

Only --capture updates the reviewed source hashes. Review changed code and the
atlas first; a successful mechanical check is not a semantic code review.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/architecture/p79"
ASSETS = ROOT / "kiara/web/static/architecture"
BUNDLE = ROOT / "kiara/web/templates/wiki/architecture/atlas.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def source_reference(reference):
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT) or path.suffix != ".py":
        raise ValueError(f"Invalid source: {path}")
    source = path.read_text(encoding="utf-8")
    current = ast.parse(source)
    for name in reference["symbol"].split("."):
        matches = [
            node
            for node in current.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Symbol not unique/found: {reference}")
        current = matches[0]
    start = min([current.lineno, *(d.lineno for d in current.decorator_list)])
    lines = source.splitlines()
    return {
        **reference,
        "start": start,
        "end": current.end_lineno,
        "sha256": digest(path.read_bytes()),
        "code": "\n".join(lines[start - 1 : current.end_lineno]),
    }


def resolve_atlas(atlas):
    refs = {key: source_reference(value) for key, value in atlas["references"].items()}
    node_ids, edge_ids, diagram_ids = set(), set(), set()
    for diagram in atlas["diagrams"]:
        if diagram["id"] in diagram_ids:
            raise ValueError("Duplicate diagram")
        diagram_ids.add(diagram["id"])
        local = {n["id"] for n in diagram["nodes"]}
        if len(local) != len(diagram["nodes"]) or node_ids & local:
            raise ValueError("Duplicate node")
        node_ids.update(local)
        for node in diagram["nodes"]:
            for field in ("purpose", "io", "config", "error", "refs"):
                if not node[field]:
                    raise ValueError(f"Incomplete node: {node['id']} / {field}")
            if set(node["refs"]) - refs.keys():
                raise ValueError(f"Unknown node reference: {node['id']}")
        for edge in diagram["edges"]:
            if edge["id"] in edge_ids or {edge["source"], edge["target"]} - local:
                raise ValueError(f"Invalid edge: {edge['id']}")
            edge_ids.add(edge["id"])
            if edge["kind"] not in {"call", "data", "job"} or not edge["evidence"]:
                raise ValueError(f"Unproven edge: {edge['id']}")
            for evidence in edge["evidence"]:
                ref = refs[evidence["ref"]]
                if not evidence["needles"]:
                    raise ValueError(f"Missing callsite evidence: {edge['id']}")
                evidence["lines"] = []
                for needle in evidence["needles"]:
                    if needle not in ref["code"]:
                        raise ValueError(f"{edge['id']}: {evidence['ref']} lacks {needle!r}")
                    evidence["lines"].append(
                        ref["start"] + ref["code"][: ref["code"].index(needle)].count("\n")
                    )
    for item in atlas["tracing"]:
        if set(item["refs"]) - refs.keys():
            raise ValueError("Unknown tracing reference")
    if node_ids & edge_ids:
        raise ValueError(f"Step and edge anchors overlap: {node_ids & edge_ids}")
    return {**atlas, "references": refs}


def mermaid(diagram):
    lines = [
        "%% Generated from atlas.json; edit the reviewed atlas, then rebuild.",
        "flowchart TB",
        f"  accTitle: {diagram['title']}",
        f"  accDescr: {diagram['intro']}",
    ]

    def emit(node, indent="  "):
        label = json.dumps(f"{node['id']} · {node['label']}", ensure_ascii=False)
        shape = "{" + label + "}" if node["kind"] == "decision" else "[" + label + "]"
        lines.append(f"{indent}{node['id']}{shape}")

    for group, title in diagram["groups"].items():
        lines.append(f"  subgraph {group}[{json.dumps(title, ensure_ascii=False)}]")
        for node in diagram["nodes"]:
            if node["group"] == group:
                emit(node, "    ")
        lines.append("  end")
    for node in diagram["nodes"]:
        if not node["group"]:
            emit(node)
    arrows = {"call": "-->", "data": "-.->", "job": "==>"}
    for edge in diagram["edges"]:
        label = json.dumps(f"{edge['id']} · {edge['label']}", ensure_ascii=False)
        lines.append(f"  {edge['source']} {arrows[edge['kind']]}|{label}| {edge['target']}")
    lines += [
        "  classDef optional fill:#f2eafe,stroke:#7252a3,color:#302344",
        "  classDef decision fill:#fff4d2,stroke:#98731d,color:#453407",
        "  classDef error fill:#ffe8e6,stroke:#b33a32,color:#6e201c",
        "  classDef store fill:#e6f3f7,stroke:#317488,color:#174555",
    ]
    for node in diagram["nodes"]:
        if node["kind"] != "step":
            lines.append(f"  class {node['id']} {node['kind']}")
    return "\n".join(lines) + "\n"


def inventory(atlas):
    state = atlas["state"]
    text = [
        "# Codepfadaufnahme – Tareas #79 / #414",
        "",
        f"Prüfdatum: {state['reviewed_at']}. Basis-Commit: `{state['base_commit']}`.",
        state["description"],
        "",
        "Kanonische Daten: [atlas.json](atlas.json). SHA-256-Dateistand: "
        "[source-state.json](source-state.json). Die Zeilenangaben beziehen sich auf diesen Arbeitsstand.",
        "",
        "Automatische Symbol-/Beleg-/Hashprüfung: `python3 scripts/build_architecture_atlas.py --check`.",
        "Die Prüfung erkennt Drift; sie ersetzt nicht das Lesen der betroffenen Implementierung.",
        "",
        "## Legende",
        *[f"- {value}" for value in atlas["legend"].values()],
        "",
        "## Erkenntnisse aus dem Code",
        *[f"- {value}" for value in atlas["findings"]],
        "",
        "## Ungeprüft und Umfangsgrenzen",
        *[f"- {value}" for value in atlas["caveats"]],
    ]

    def link(key, line=None):
        ref = atlas["references"][key]
        return f"[{ref['path']}:{line or ref['start']} – {ref['symbol']}](../../../{ref['path']})"

    for diagram in atlas["diagrams"]:
        text += ["", f"## {diagram['title']}", "", diagram["intro"], ""]
        for node in diagram["nodes"]:
            text += [
                f"### {node['id']} · {node['label']}",
                "",
                node["purpose"],
                "",
                f"- Ein-/Ausgabe: {node['io']}",
                f"- Konfiguration/Entscheidung: {node['config']}",
                f"- Fehler/Grenzen: {node['error']}",
                "- Code: " + "; ".join(link(key) for key in node["refs"]),
                "",
            ]
        text += ["### Verbindungsbelege", ""]
        for edge in diagram["edges"]:
            text += [
                f"- **{edge['id']} {edge['source']} → {edge['target']} ({edge['kind']})**: {edge['proof']}",
            ]
            for ev in edge["evidence"]:
                fragments = "; ".join(
                    f"`{needle}` (Z. {line})"
                    for needle, line in zip(ev["needles"], ev["lines"], strict=True)
                )
                text += [f"  {link(ev['ref'])}: {fragments}."]
    text += ["", "## Tracing-Zuordnung", ""]
    for item in atlas["tracing"]:
        text += [
            f"### {item['phase']}",
            "",
            item["spans"],
            "",
            item["gap"],
            "",
            "; ".join(link(key) for key in item["refs"]),
            "",
        ]
    text += [
        "## Begriffe",
        "",
        *[f"- **{key}**: {value}" for key, value in atlas["glossary"].items()],
        "",
    ]
    return "\n".join(text)


def emit_file(path, content, check):
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Generated file is stale: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def wiki_markdown(atlas, page):
    """Keep the existing wiki.search tool on the same reviewed content."""
    diagrams = [d for d in atlas["diagrams"] if d["page"] == page]
    lines = [
        f"# {diagrams[0]['title']}",
        "",
        f"[Karten mit Codebelegen öffnen](/wiki/guide/{page})",
        "",
        f"Geprüft: {atlas['state']['reviewed_at']}. {atlas['state']['description']}",
        "",
    ]
    for diagram in diagrams:
        lines += [f"## {diagram['title']}", "", diagram["intro"], ""]
        for node in diagram["nodes"]:
            lines += [
                f"### {node['id']} {node['label']}",
                "",
                node["purpose"],
                "",
                node["io"],
                "",
                f"Konfiguration: {node['config']}",
                "",
                f"Fehler/Grenzen: {node['error']}",
                "",
            ]
            for key in node["refs"]:
                ref = atlas["references"][key]
                lines += [
                    f"- [{ref['path']}:{ref['start']} – {ref['symbol']}](/wiki/architecture/code/{key}#L{ref['start']})"
                ]
            lines.append("")
    lines += ["## Ungeprüft und Grenzen", "", *[f"- {c}" for c in atlas["caveats"]], ""]
    return "\n".join(lines)


def build(*, capture=False, check=False, render=False, mmdc="mmdc", puppeteer_config=None):
    atlas = resolve_atlas(read_json(DOCS / "atlas.json"))
    files = {ref["path"]: ref["sha256"] for ref in atlas["references"].values()}
    state_path = DOCS / "source-state.json"
    if capture:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True
        )
        state = {
            **atlas["state"],
            "files": dict(sorted(files.items())),
            "reviewed_file_status": [line for line in status.splitlines() if line[3:] in files],
        }
        emit_file(state_path, json_text(state), False)
    state = read_json(state_path)
    if state["files"] != files or any(state[key] != value for key, value in atlas["state"].items()):
        raise ValueError("Reviewed source changed. Review atlas and code before --capture.")
    atlas["state"] = state
    emit_file(DOCS / "inventory.md", inventory(atlas), check)
    for page in dict.fromkeys(d["page"] for d in atlas["diagrams"]):
        emit_file(
            ROOT / f"kiara/admin/wiki_content/concepts/{page}.md", wiki_markdown(atlas, page), check
        )
    for diagram in atlas["diagrams"]:
        diagram["mermaid"] = mermaid(diagram)
        emit_file(DOCS / f"{diagram['id']}.mmd", diagram["mermaid"], check)
    emit_file(BUNDLE, json_text(atlas), check)
    if render:
        ASSETS.mkdir(parents=True, exist_ok=True)
        for diagram in atlas["diagrams"]:
            command = [
                mmdc,
                "-i",
                str(DOCS / f"{diagram['id']}.mmd"),
                "-o",
                str(ASSETS / f"{diagram['id']}.svg"),
                "-c",
                str(DOCS / "mermaid-config.json"),
                "-b",
                "white",
                "--quiet",
            ]
            if puppeteer_config:
                command += ["-p", puppeteer_config]
            subprocess.run(command, check=True)
        render_state = {
            "renderer": "@mermaid-js/mermaid-cli 11.12.0",
            "config_sha256": digest((DOCS / "mermaid-config.json").read_bytes()),
            "diagrams": {
                d["id"]: {
                    "source_sha256": digest(d["mermaid"].encode()),
                    "svg_sha256": digest((ASSETS / f"{d['id']}.svg").read_bytes()),
                }
                for d in atlas["diagrams"]
            },
        }
        emit_file(DOCS / "render-state.json", json_text(render_state), False)
    if check:
        render_state = read_json(DOCS / "render-state.json")
        if render_state["config_sha256"] != digest((DOCS / "mermaid-config.json").read_bytes()):
            raise ValueError("Mermaid config changed; rerender diagrams")
        for diagram in atlas["diagrams"]:
            svg = (ASSETS / f"{diagram['id']}.svg").read_bytes()
            expected = render_state["diagrams"][diagram["id"]]
            if expected != {
                "source_sha256": digest(diagram["mermaid"].encode()),
                "svg_sha256": digest(svg),
            }:
                raise ValueError(f"Stale diagram: {diagram['id']}")
            tree = ElementTree.fromstring(svg)
            if tree.tag != "{http://www.w3.org/2000/svg}svg" or "viewBox" not in tree.attrib:
                raise ValueError("Invalid SVG")
    print(
        f"Atlas OK: {len(files)} files, {len(atlas['references'])} symbols, "
        f"{sum(len(d['edges']) for d in atlas['diagrams'])} evidenced connections"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capture", action="store_true", help="Capture reviewed source hashes")
    mode.add_argument(
        "--check", action="store_true", help="Check sources and generated artifacts without writing"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render SVGs with Mermaid CLI 11.12.0"
    )
    parser.add_argument("--mmdc", default="mmdc")
    parser.add_argument("--puppeteer-config")
    args = parser.parse_args()
    if args.check and args.render:
        parser.error("--check cannot write --render output")
    build(**vars(args))
