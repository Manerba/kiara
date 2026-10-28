#!/usr/bin/env python3
"""Prepare BEIR datasets as normal KIara local datasource documents.

The script intentionally stays outside the KIara runtime code path. It downloads
or reads a BEIR dataset layout, writes one Markdown file per corpus document, and
stores query/qrel/doc-map artifacts for the retrieval evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("/var/lib/kiara/benchmarks/beir")
DEFAULT_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)


@dataclass(frozen=True)
class BeirPrepareStats:
    dataset: str
    split: str
    output_dir: str
    source_dir: str
    documents_dir: str
    documents_written: int
    queries_written: int
    qrels_written: int
    qrels_dropped_missing_document: int
    max_docs: int | None


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_dataset_name(dataset: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset.strip()).strip("._-")
    if not cleaned:
        raise ValueError("dataset must not be empty")
    return cleaned


def _doc_filename(doc_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(doc_id)).strip("._-")[:80]
    if not cleaned:
        cleaned = "doc"
    digest = hashlib.sha1(str(doc_id).encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}_{digest}.md"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with target.open("wb") as out:
            shutil.copyfileobj(response, out)


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            destination = (target_dir / member.filename).resolve()
            if not destination.is_relative_to(target_root):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(target_dir)


def _find_beir_source_dir(root: Path, dataset: str) -> Path:
    candidates = [
        root,
        root / dataset,
        root / dataset.lower(),
        root / dataset.upper(),
    ]
    for candidate in candidates:
        if (candidate / "corpus.jsonl").exists() and (candidate / "queries.jsonl").exists():
            return candidate
    for candidate in root.rglob("corpus.jsonl"):
        source_dir = candidate.parent
        if (source_dir / "queries.jsonl").exists():
            return source_dir
    raise FileNotFoundError(f"No BEIR corpus.jsonl/queries.jsonl layout found below {root}")


def _ensure_source_dir(
    dataset: str,
    output_dir: Path,
    *,
    source_dir: Path | None,
    download: bool,
    url: str | None,
) -> Path:
    if source_dir is not None:
        return _find_beir_source_dir(source_dir, dataset)

    raw_dir = output_dir / "raw"
    try:
        return _find_beir_source_dir(raw_dir, dataset)
    except FileNotFoundError:
        if not download:
            raise

    dataset_url = url or DEFAULT_URL_TEMPLATE.format(dataset=dataset)
    archive = raw_dir / f"{dataset}.zip"
    print(f"Downloading BEIR dataset '{dataset}' from {dataset_url}", file=sys.stderr)
    _download(dataset_url, archive)
    _safe_extract_zip(archive, raw_dir)
    return _find_beir_source_dir(raw_dir, dataset)


def _read_qrels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_no, parts in enumerate(reader, 1):
            if not parts:
                continue
            if line_no == 1 and parts[0].strip().lower() in {"query-id", "query_id", "qid"}:
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no} has fewer than 3 TSV columns")
            query_id, doc_id, score = parts[:3]
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "score": int(float(score)),
                }
            )
    return rows


def _render_document(title: str, text: str) -> str:
    parts = []
    title = title.strip()
    text = text.strip()
    if title:
        parts.extend([f"# {title}", ""])
    if text:
        parts.append(text)
    return "\n".join(parts).strip() + "\n"


def prepare_beir_dataset(
    dataset: str = "scifact",
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_dir: Path | None = None,
    split: str = "test",
    download: bool = True,
    url: str | None = None,
    force: bool = False,
    max_docs: int | None = None,
) -> BeirPrepareStats:
    dataset = _safe_dataset_name(dataset)
    output_dir = output_root / dataset
    source = _ensure_source_dir(
        dataset,
        output_dir,
        source_dir=source_dir,
        download=download,
        url=url,
    )

    corpus_path = source / "corpus.jsonl"
    queries_path = source / "queries.jsonl"
    qrels_path = source / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = sorted(path.stem for path in (source / "qrels").glob("*.tsv"))
        raise FileNotFoundError(
            f"qrels split '{split}' not found at {qrels_path}. Available: {available}"
        )

    documents_dir = output_dir / "documents"
    if documents_dir.exists():
        if not force:
            raise FileExistsError(f"{documents_dir} already exists. Use --force to replace it.")
        shutil.rmtree(documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)

    doc_map_rows: list[dict[str, Any]] = []
    written_doc_ids: set[str] = set()
    for index, row in enumerate(iter_jsonl(corpus_path), 1):
        if max_docs is not None and index > max_docs:
            break
        doc_id = str(row.get("_id") or row.get("id") or "").strip()
        if not doc_id:
            raise ValueError(f"{corpus_path}:{index} has no _id/id")
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        target = documents_dir / _doc_filename(doc_id)
        target.write_text(_render_document(title, text), encoding="utf-8")
        written_doc_ids.add(doc_id)
        doc_map_rows.append(
            {
                "doc_id": doc_id,
                "relative_path": str(target.relative_to(output_dir)),
                "source_path": str(target.resolve()),
                "title": title,
                "title_chars": len(title),
                "text_chars": len(text),
            }
        )

    queries_by_id = {
        str(row.get("_id") or row.get("id") or ""): str(row.get("text") or "")
        for row in iter_jsonl(queries_path)
    }
    raw_qrels = _read_qrels(qrels_path)
    filtered_qrels: list[dict[str, Any]] = []
    dropped_missing_doc = 0
    query_ids_with_qrels: set[str] = set()
    for row in raw_qrels:
        if row["doc_id"] not in written_doc_ids:
            dropped_missing_doc += 1
            continue
        filtered_qrels.append(row)
        query_ids_with_qrels.add(row["query_id"])

    query_rows = [
        {"query_id": query_id, "text": queries_by_id[query_id]}
        for query_id in sorted(query_ids_with_qrels)
        if query_id in queries_by_id
    ]

    write_jsonl(output_dir / "doc_map.jsonl", doc_map_rows)
    write_jsonl(output_dir / "queries.jsonl", query_rows)
    write_jsonl(output_dir / "qrels.jsonl", filtered_qrels)

    manifest = {
        "schema": "kiara-beir-prepared-dataset",
        "dataset": dataset,
        "split": split,
        "prepared_at": _utc_timestamp(),
        "source_dir": str(source.resolve()),
        "output_dir": str(output_dir.resolve()),
        "documents_dir": str(documents_dir.resolve()),
        "files": {
            "doc_map": "doc_map.jsonl",
            "queries": "queries.jsonl",
            "qrels": "qrels.jsonl",
            "source_corpus": str(corpus_path.resolve()),
            "source_queries": str(queries_path.resolve()),
            "source_qrels": str(qrels_path.resolve()),
        },
        "counts": {
            "documents": len(doc_map_rows),
            "queries": len(query_rows),
            "qrels": len(filtered_qrels),
            "qrels_dropped_missing_document": dropped_missing_doc,
        },
        "max_docs": max_docs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# BEIR Dataset: {dataset}\n\n"
        "Prepared for KIara as a normal local datasource.\n\n"
        "- Documents: `documents/`\n"
        "- Queries: `queries.jsonl`\n"
        "- Qrels: `qrels.jsonl`\n"
        "- Doc map: `doc_map.jsonl`\n".format(dataset=dataset),
        encoding="utf-8",
    )

    return BeirPrepareStats(
        dataset=dataset,
        split=split,
        output_dir=str(output_dir),
        source_dir=str(source),
        documents_dir=str(documents_dir),
        documents_written=len(doc_map_rows),
        queries_written=len(query_rows),
        qrels_written=len(filtered_qrels),
        qrels_dropped_missing_document=dropped_missing_doc,
        max_docs=max_docs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name, default: scifact")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root, default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing BEIR dataset directory or parent directory; skips download.",
    )
    parser.add_argument("--split", default="test", help="qrels split to export, default: test")
    parser.add_argument("--url", help="Override dataset zip URL")
    parser.add_argument("--no-download", action="store_true", help="Fail if raw data is missing")
    parser.add_argument("--force", action="store_true", help="Replace existing documents directory")
    parser.add_argument("--max-docs", type=int, help="Limit documents for smoke tests")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = prepare_beir_dataset(
        dataset=args.dataset,
        output_root=args.output_root,
        source_dir=args.source_dir,
        split=args.split,
        download=not args.no_download,
        url=args.url,
        force=args.force,
        max_docs=args.max_docs,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
