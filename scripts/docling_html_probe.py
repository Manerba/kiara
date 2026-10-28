#!/usr/bin/env python3
"""Isolated HTML extraction experiment for Tareas #62 / #319.

Run in the separate environment described in the accompanying evaluation report.
No KIara services, models, indexes, mailboxes, or configuration are modified.
Original HTML and extracted text are written only to the supplied output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.base import BaseSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import ContentLayer, DoclingDocument
from markdownify import MarkdownConverter


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\\_", "_").replace("**", "")
    text = text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-"}))
    return " ".join(text.split())


def fixture(name, html, required=(), forbidden=(), rows=()):
    return {
        "id": name,
        "html": html,
        "required": list(required),
        "forbidden": list(forbidden),
        "rows": [list(row) for row in rows],
    }


def synthetic_cases():
    data_table = "<table><tr><th>Position</th><th>Betrag</th></tr><tr><td>Wartung</td><td>7,45 €</td></tr><tr><td>Gesamt</td><td>7,45 €</td></tr></table>"
    cases = [
        fixture(
            "blocks",
            "<style>.x{color:red}</style><h2>Beleg</h2><div>Datum: 10.03.2025</div><div>Gesamt: 7,45&nbsp;€</div><script>track()</script>",
            ["10.03.2025", "7,45 €"],
            ["color:red", "track()"],
        ),
        fixture(
            "table",
            data_table,
            ["Wartung", "Gesamt", "7,45 €"],
            rows=[["Wartung", "7,45 €"], ["Gesamt", "7,45 €"]],
        ),
        fixture(
            "layout_marked",
            '<table role="presentation"><tr><td><h2>Beleg</h2>' + data_table + "</td></tr></table>",
            ["Beleg", "Wartung", "Gesamt", "7,45 €"],
            rows=[["Wartung", "7,45 €"]],
        ),
        fixture(
            "layout_unmarked",
            "<table><tr><td><h2>Beleg</h2>" + data_table + "</td></tr></table>",
            ["Beleg", "Wartung", "Gesamt", "7,45 €"],
            rows=[["Wartung", "7,45 €"]],
        ),
        fixture(
            "rowspan",
            '<table><tr><th>Kunde</th><th>Position</th><th>Betrag</th></tr><tr><td rowspan="2">Firma A</td><td>Wartung</td><td>7,45 €</td></tr><tr><td>Ersatzteil</td><td>12,50 €</td></tr></table>',
            ["Firma A", "Wartung", "Ersatzteil", "12,50 €"],
            rows=[["Firma A", "Ersatzteil", "12,50 €"]],
        ),
        fixture(
            "colspan",
            '<table><thead><tr><th rowspan="2">Artikel</th><th colspan="2">Maße in mm</th></tr><tr><th>Breite</th><th>Höhe</th></tr></thead><tbody><tr><td>A-42</td><td>12,5</td><td>4,0</td></tr></tbody></table>',
            ["Artikel", "Maße in mm", "Breite", "Höhe", "A-42", "12,5", "4,0"],
            rows=[["A-42", "12,5", "4,0"]],
        ),
        fixture(
            "units_and_identifiers",
            "<p>Teilenummer: A-2025-42</p><p>Maße: 12,5 × 4,0 mm</p><p>Spannung: 24 V</p><p>Frist: 30.04.2025</p>",
            ["A-2025-42", "12,5 × 4,0 mm", "24 V", "30.04.2025"],
        ),
        fixture(
            "inline_identifier",
            "<p>Teilenummer A<strong>B</strong>-123. Spannung 24&nbsp;V.</p>",
            ["AB-123", "24 V"],
        ),
        fixture(
            "adjacent_blocks",
            "<div>Jahr 2025</div><div>2,99 €</div><p>Nicht freigegeben.</p>",
            ["Jahr 2025", "2,99 €", "Nicht freigegeben."],
            ["20252,99"],
        ),
        fixture(
            "footer",
            "<h1>Vertrag</h1><p>Artikel A-123</p><footer>Widerruf innerhalb von 14 Tagen. Gutschrift 17,50 €.</footer>",
            ["Artikel A-123", "14 Tagen", "17,50 €"],
        ),
        fixture(
            "before_heading",
            "<p>Vertrag R-291. Kündigung bis 30.04.2025.</p><h2>Details</h2><p>Nicht automatisch verlängern.</p>",
            ["R-291", "30.04.2025", "Nicht automatisch verlängern."],
        ),
        fixture(
            "hidden_inline",
            '<p style="display:none">GEHEIM_DOPPELT 999,99 €</p><p hidden>HIDDEN_ATTR 88,88 €</p><p>Gesamt 7,45 €</p>',
            ["7,45 €"],
            ["GEHEIM_DOPPELT", "HIDDEN_ATTR"],
        ),
        fixture(
            "hidden_stylesheet",
            '<style>.mobile {display:none}</style><div class="mobile">MOBILE_DUPLIKAT 999,99 €</div><div>Gesamt 7,45 €</div>',
            ["7,45 €"],
            ["MOBILE_DUPLIKAT"],
        ),
        fixture(
            "lists",
            "<h2>Freigabe</h2><ul><li>Spannung: 24 V</li><li>Nicht bei Nässe verwenden.</li></ul><dl><dt>Frist</dt><dd>30.04.2025</dd></dl>",
            ["24 V", "Nicht bei Nässe verwenden.", "30.04.2025"],
        ),
        fixture(
            "malformed_table",
            "<table><tr><th>Artikel<th>Betrag<tr><td>A-42<td>7,45 €</table>",
            ["Artikel", "Betrag", "A-42", "7,45 €"],
            rows=[["A-42", "7,45 €"]],
        ),
    ]
    rows = [[f"TEIL-{i:03}", f"{i},37 €", f"{i + 10} V"] for i in range(1, 41)]
    html = "<h2>Positionen</h2><table><tr><th>Artikel</th><th>Betrag</th><th>Spannung</th></tr>"
    html += (
        "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
        + "</table>"
    )
    cases.append(fixture("long_table", html, [v for row in rows for v in row], rows=rows))
    return cases


class ProbeSerializer(BaseSerializerProvider):
    def __init__(self, *, include_furniture=False, markdown_tables=False):
        self.include_furniture = include_furniture
        self.markdown_tables = markdown_tables

    def get_serializer(self, doc: DoclingDocument):
        layers = {ContentLayer.BODY}
        if self.include_furniture:
            layers.add(ContentLayer.FURNITURE)
        kwargs = {"table_serializer": MarkdownTableSerializer()} if self.markdown_tables else {}
        return ChunkingDocSerializer(
            doc=doc,
            params=MarkdownParams(
                layers=layers,
                image_placeholder="",
                escape_underscores=False,
                escape_html=False,
            ),
            **kwargs,
        )


def markdownify_baseline(html):
    """Same preprocessing as the existing four-case prototype."""
    soup = BeautifulSoup(html, "html.parser")
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


def checks(text, case):
    flat = normalize(text)
    return {
        "missing": [v for v in case["required"] if normalize(v) not in flat],
        "forbidden_present": [v for v in case["forbidden"] if normalize(v) in flat],
    }


def prepare_real(path):
    html = path.read_text()
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.find_all(["script", "style", "head"]):
        node.decompose()
    source_text = normalize(soup.get_text(" ", strip=True))
    # Source-derived literal amounts, plus predeclared business facts for Apple.
    required = sorted(set(re.findall(r"\b\d[\d.]*,\d{2}\s*€|\bEUR\s*\d[\d.]*,\d{2}", source_text)))
    if path.stem.startswith("apple_"):
        required += [
            "iCloud+",
            "200 GB",
            "2,99 €",
            "2,51 €",
            "0,48 €",
            "15 Tagen",
            "45 Tagen",
            "IE9700053D",
        ]
        for regex in [
            r"DOKUMENT-NR\.\s*(\d+)",
            r"Dokument:\s*(\d+)",
            r"BESTELLNR\.\s*(\w+)",
            r"Bestellnummer:\s*(\w+)",
        ]:
            required.extend(re.findall(regex, source_text))
    if path.stem.startswith("hetzner_"):
        required += ["demnächst", "HRB 6089", "DE812871812"]
        required.extend(re.findall(r"aktuelle Rechnung\s*(\d+)", source_text))
    literal_facts = {
        "payment_01": ["siku32276", "P02-6301375-8055493", "18. Januar 2025", "drei Werktagen"],
        "order_01": ["304-0218028-8235579", "3–5 Tage", "Erstattung"],
        "order_02": [
            "304-4277927-3955551",
            "Sterntaler",
            "Menge: 1",
            "noch keine Annahme",
            "9 Januar",
        ],
        "shipping_01": [
            "304-4277927-3955551",
            "JJD000390016743189104",
            "14 Tagen",
            "nicht mehr möglich",
        ],
        "shipping_02": [
            "304-5301354-4451569",
            "MEDIMOPS",
            "nicht nachverfolgbar",
            "Gebraucht - Sehr gut",
            "9 Januar",
            "11 Januar",
        ],
        "notice_01": [
            "304-0218028-8235579",
            "B09FF95DG3",
            "U=1,22 W/m2K",
            "80x55",
            "Anzahl: 2",
            "nicht weitergeleitet",
        ],
        "notice_02": [
            "SEPA-Lastschrifteinzug",
            "gesetzlichen Pflichtangaben",
            "beigefügten Anlage",
        ],
    }
    required += literal_facts.get(path.stem, [])
    assert all(normalize(v) in source_text for v in required)
    return fixture(path.stem, html, sorted(set(required)))


def run_docling(case, mode, root, tokenizer, repair_parser, table_format):
    options = HTMLBackendOptions(infer_furniture=(mode == "default"))
    start = time.perf_counter()
    html = case["html"]
    if repair_parser != "none":
        html = str(BeautifulSoup(html, repair_parser))
    repair_ms = (time.perf_counter() - start) * 1000
    stream = BytesIO(html.encode())
    start = time.perf_counter()
    input_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        backend_options=options,
        filename=case["id"] + ".html",
    )
    backend = HTMLDocumentBackend(in_doc=input_doc, path_or_stream=stream, options=options)
    doc = backend.convert()
    conversion_ms = (time.perf_counter() - start) * 1000
    layers = None if mode == "default" else {ContentLayer.BODY, ContentLayer.FURNITURE}
    md = doc.export_to_markdown(included_content_layers=layers)
    chunker_kwargs = {"tokenizer": tokenizer}
    if mode != "default" or table_format == "markdown":
        chunker_kwargs["serializer_provider"] = ProbeSerializer(
            include_furniture=(mode != "default"), markdown_tables=(table_format == "markdown")
        )
    chunker = HybridChunker(**chunker_kwargs)
    start = time.perf_counter()
    chunks = list(chunker.chunk(dl_doc=doc))
    contextualized = [chunker.contextualize(chunk=chunk) for chunk in chunks]
    chunking_ms = (time.perf_counter() - start) * 1000
    chunk_text = "\n\n".join(contextualized)
    table_grids = [
        [[normalize(cell.text) for cell in row] for row in table.data.grid] for table in doc.tables
    ]
    actual_rows = [row for grid in table_grids for row in grid]
    missing_rows = [
        row
        for row in case["rows"]
        if not any(all(normalize(v) in " | ".join(actual) for v in row) for actual in actual_rows)
    ]
    chunk_row_losses = [
        row
        for row in case["rows"]
        if not any(all(normalize(v) in normalize(text) for v in row) for text in contextualized)
    ]
    result = {
        "repair_ms": repair_ms,
        "conversion_ms": conversion_ms,
        "chunking_ms": chunking_ms,
        "markdown_chars": len(md),
        "markdown_tokens": tokenizer.count_tokens(md),
        "chunks": len(chunks),
        "chunk_tokens": [tokenizer.count_tokens(t) for t in contextualized],
        "markdown_checks": checks(md, case),
        "chunk_checks": checks(chunk_text, case),
        "missing_table_rows": missing_rows,
        "rows_split_or_missing_in_chunks": chunk_row_losses,
        "tables": len(doc.tables),
        "texts": len(doc.texts),
        "table_grids": table_grids,
        "amount_occurrences": normalize(md).count("2,99 €"),
    }
    target = root / case["id"] / mode
    target.mkdir(parents=True, exist_ok=True)
    (target / "docling-input.html").write_text(html)
    (target / "document.json").write_text(
        json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2)
    )
    (target / "document.md").write_text(md)
    (target / "chunks.json").write_text(
        json.dumps(
            [
                {"text": text, "metadata": chunk.meta.model_dump(mode="json")}
                for text, chunk in zip(contextualized, chunks, strict=True)
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    (target / "chunks.txt").write_text("\n\n--- CHUNK ---\n\n".join(contextualized))
    (target / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    backend.unload()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repair-parser", choices=["none", "lxml", "html5lib"], default="none")
    parser.add_argument("--table-format", choices=["triplet", "markdown"], default="triplet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=args.max_tokens
    )
    cases = [("synthetic", case) for case in synthetic_cases()]
    cases += [("real", prepare_real(path)) for path in sorted(args.source_dir.glob("*.html"))]
    summary = {
        "versions": {
            p: version(p)
            for p in ["docling-slim", "docling-core", "markdownify", "beautifulsoup4", "tiktoken"]
        },
        "tokenizer": "cl100k_base",
        "max_tokens": args.max_tokens,
        "repair_parser": args.repair_parser,
        "table_format": args.table_format,
        "method": "Official HTMLDocumentBackend plus HybridChunker; no OCR, LLM, rendering, external assets or embeddings.",
        "modes": {
            "default": "HTMLBackendOptions defaults; default Markdown/HybridChunker content layers",
            "preserve": "infer_furniture=False; Markdown and chunks include BODY and FURNITURE",
        },
        "cases": [],
    }
    # Freeze the expectations before running either converter.
    (args.output_dir / "expectations.json").write_text(
        json.dumps(
            [
                {"kind": kind, **{k: v for k, v in case.items() if k != "html"}}
                for kind, case in cases
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    for kind, case in cases:
        target = args.output_dir / case["id"]
        target.mkdir(exist_ok=True)
        (target / "input.html").write_text(case["html"])
        baseline = markdownify_baseline(case["html"])
        (target / "markdownify.md").write_text(baseline)
        entry = {
            "id": case["id"],
            "kind": kind,
            "required": case["required"],
            "markdownify": {
                "chars": len(baseline),
                "tokens": tokenizer.count_tokens(baseline),
                **checks(baseline, case),
            },
        }
        for mode in ["default", "preserve"]:
            try:
                entry[mode] = run_docling(
                    case, mode, args.output_dir, tokenizer, args.repair_parser, args.table_format
                )
            except Exception as exc:
                entry[mode] = {"error": f"{type(exc).__name__}: {exc}"}
        summary["cases"].append(entry)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        print(
            json.dumps(
                {
                    "id": case["id"],
                    **{
                        mode: (
                            {
                                k: entry[mode][k]
                                for k in [
                                    "chunks",
                                    "markdown_checks",
                                    "chunk_checks",
                                    "missing_table_rows",
                                    "rows_split_or_missing_in_chunks",
                                ]
                            }
                            if "error" not in entry[mode]
                            else entry[mode]
                        )
                        for mode in ["default", "preserve"]
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if any("error" in case[mode] for case in summary["cases"] for mode in ["default", "preserve"]):
        raise SystemExit("At least one conversion failed; inspect summary.json.")


if __name__ == "__main__":
    main()
