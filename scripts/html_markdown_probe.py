#!/usr/bin/env python3
"""Compare HTML-to-Markdown libraries on the frozen Tareas 319 corpus.

The corpus is the results directory of docling_html_probe.py. This experiment
does not import Docling or change KIara runtime configuration or indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path

from bs4 import BeautifulSoup
from html2text import HTML2Text
from html_to_markdown import ConversionOptions, PreprocessingOptions, convert
from markdown_it import MarkdownIt
from markdownify import MarkdownConverter


def normalize(text):
    text = unicodedata.normalize("NFKC", text).replace("\\_", "_").replace("**", "")
    text = text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-"}))
    return " ".join(text.split())


def markdownify_prototype(html, parser="html.parser"):
    """The already tested preprocessing: remove CSS/scripts, unwrap explicit layout."""
    soup = BeautifulSoup(html, parser)
    for node in soup.find_all(["script", "style"]):
        node.decompose()
    for table in soup.find_all("table", attrs={"role": "presentation"}):
        cells = [
            node
            for node in table.find_all(["thead", "tbody", "tfoot", "tr", "td", "th"])
            if node.find_parent("table") is table
        ]
        for cell in cells:
            cell.name = "div"
        table.name = "div"
    return MarkdownConverter(heading_style="ATX", wrap=False).convert_soup(soup)


def serialize_native(result):
    tables = []
    for table in result.tables:
        tables.append(
            {
                "markdown": table.markdown,
                "rows": table.grid.rows,
                "cols": table.grid.cols,
                "cells": [
                    {
                        field: getattr(cell, field)
                        for field in ["content", "row", "col", "row_span", "col_span", "is_header"]
                    }
                    for cell in table.grid.cells
                ],
            }
        )
    return {
        "warnings": [str(w) for w in result.warnings],
        "tables": tables,
        "nodes": [
            {
                "id": node.id,
                "content": str(node.content),
                "parent": node.parent,
                "children": node.children,
                "attributes": node.attributes,
            }
            for node in result.document.nodes
        ]
        if result.document is not None
        else [],
    }


def native_grid_conflicts(tables):
    """Check exported coordinates independently of rendered Markdown."""
    conflicts = []
    for ti, table in enumerate(tables):
        occupied = {}
        for cell in table["cells"]:
            for row in range(cell["row"], cell["row"] + cell["row_span"]):
                for col in range(cell["col"], cell["col"] + cell["col_span"]):
                    if (row, col) in occupied:
                        conflicts.append(
                            {
                                "table": ti,
                                "row": row,
                                "col": col,
                                "first": occupied[row, col],
                                "second": cell["content"],
                            }
                        )
                    occupied[row, col] = cell["content"]
    return conflicts


def convert_candidate(name, html):
    if name == "markdownify":
        return markdownify_prototype(html), {}
    if name == "markdownify_html5lib":
        return markdownify_prototype(html, "html5lib"), {}
    if name in {"html2text", "html2text_unicode"}:
        converter = HTML2Text()
        converter.body_width = 0
        converter.unicode_snob = name == "html2text_unicode"
        return converter.handle(html), {}
    options = None
    if name == "html_to_markdown_mail":
        options = ConversionOptions(
            extract_metadata=False,
            compact_tables=True,
            include_document_structure=True,
            preprocessing=PreprocessingOptions(
                enabled=True, preset="minimal", remove_navigation=False, remove_forms=False
            ),
        )
    result = convert(html, options)
    return result.content, serialize_native(result)


def literal_checks(text, case):
    flat = normalize(text)
    return {
        "missing": [v for v in case["required"] if normalize(v) not in flat],
        "forbidden_present": [v for v in case["forbidden"] if normalize(v) in flat],
    }


def measure(markdown, html, case):
    rendered = MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown)
    soup = BeautifulSoup(rendered, "html.parser")
    # No forced separator between inline nodes: A<strong>B</strong>-123 stays AB-123.
    visible = soup.get_text()
    rows = [
        [normalize(cell.get_text()) for cell in row.find_all(["th", "td"], recursive=False)]
        for table in soup.find_all("table")
        for row in table.find_all("tr")
        if row.find_parent("table") is table
    ]
    absent_rows = [
        expected
        for expected in case["rows"]
        if not any(all(normalize(value) in row for value in expected) for row in rows)
    ]
    source = BeautifulSoup(html, "html5lib")
    for node in source.find_all(["head", "style", "script", "noscript"]):
        node.decompose()
    source_text = normalize(source.get_text(" ", strip=True))
    # Counts are diagnostic, not automatic error verdicts: responsive variants and
    # merged cells can legitimately repeat values in the original or rendered text.
    occurrences = {
        value: {
            "source": source_text.count(normalize(value)),
            "markdown": normalize(markdown).count(normalize(value)),
            "rendered": normalize(visible).count(normalize(value)),
        }
        for value in case["required"]
    }
    metrics = {
        "characters": len(markdown),
        "max_line_characters": max(map(len, markdown.splitlines()), default=0),
        "raw_checks": literal_checks(markdown, case),
        "rendered_checks": literal_checks(visible, case),
        "rendered_table_rows_missing": absent_rows,
        "max_table_columns": max(map(len, rows), default=0),
        "rendered_table_rows": rows,
        "occurrences": occurrences,
    }
    return metrics, rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expectation_bytes = (args.corpus / "expectations.json").read_bytes()
    cases = json.loads(expectation_bytes)
    names = [
        "markdownify",
        "markdownify_html5lib",
        "html2text",
        "html2text_unicode",
        "html_to_markdown",
        "html_to_markdown_mail",
    ]
    summary = {
        "versions": {
            name: version(name)
            for name in [
                "markdownify",
                "html2text",
                "html-to-markdown",
                "beautifulsoup4",
                "html5lib",
                "markdown-it-py",
            ]
        },
        "corpus": str(args.corpus),
        "expectations_sha256": hashlib.sha256(expectation_bytes).hexdigest(),
        "candidates": names,
        "cases": [],
    }
    (args.output_dir / "expectations.json").write_bytes(expectation_bytes)
    for case in cases:
        html = (args.corpus / case["id"] / "input.html").read_text()
        folder = args.output_dir / case["id"]
        folder.mkdir(exist_ok=True)
        (folder / "input.html").write_text(html)
        result = {**case, "input_sha256": hashlib.sha256(html.encode()).hexdigest(), "results": {}}
        for name in names:
            start = time.perf_counter()
            try:
                markdown, metadata = convert_candidate(name, html)
                elapsed_ms = (time.perf_counter() - start) * 1000
                metrics, rendered = measure(markdown, html, case)
                metrics.update(
                    {
                        "conversion_ms": elapsed_ms,
                        "exported_grid_conflicts": native_grid_conflicts(
                            metadata.get("tables", [])
                        ),
                    }
                )
                (folder / (name + ".md")).write_text(markdown)
                (folder / (name + ".rendered.html")).write_text(rendered)
                if metadata:
                    (folder / (name + ".metadata.json")).write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2)
                    )
                result["results"][name] = metrics
            except Exception as exc:
                result["results"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        summary["cases"].append(result)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "candidates": {
                        name: {
                            k: v
                            for k, v in metrics.items()
                            if k
                            in [
                                "error",
                                "raw_checks",
                                "rendered_checks",
                                "rendered_table_rows_missing",
                            ]
                        }
                        for name, metrics in result["results"].items()
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if any("error" in metrics for case in summary["cases"] for metrics in case["results"].values()):
        raise SystemExit("Some conversions failed; inspect summary.json.")


if __name__ == "__main__":
    main()
