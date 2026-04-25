#!/usr/bin/env python3
"""Shared utilities for OpenAI-backed lexicon check scripts.

These tools are audit-only and never rewrite source files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from openai_batch_utils import collect_batch_results, submit_batch, wait_for_batch, write_jsonl


@dataclass
class CheckResult:
    status: str
    confidence: float
    issues: list[str]
    suggestions: list[str]


def build_common_parser(description: str, default_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", default="r2-backup/words", help="Input directory (default: r2-backup/words)")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory for reports")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--glob", default="*.json", help="Filename glob pattern")
    parser.add_argument("--max-retries", type=int, default=2, help="Reserved for compatibility (unused in batch mode)")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Reserved for compatibility (unused in batch mode)")
    parser.add_argument("--completion-window", default="24h", help="Batch API completion window (default: 24h)")
    parser.add_argument("--poll-seconds", type=float, default=10.0, help="Polling interval for batch status")
    return parser


def iter_files(input_dir: Path, pattern: str, limit: int) -> Iterable[Path]:
    files = sorted(input_dir.glob(pattern))
    if limit > 0:
        files = files[:limit]
    return files


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        raise SystemExit(2)
    return api_key


def parse_check_content(content: str) -> CheckResult:
    parsed = json.loads(content)
    return CheckResult(
        status=str(parsed.get("status", "needs_review")).strip().lower(),
        confidence=float(parsed.get("confidence", 0.0)),
        issues=[str(x) for x in parsed.get("issues", [])],
        suggestions=[str(x) for x in parsed.get("suggestions", [])],
    )


def write_reports(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.jsonl"
    with report_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["report"] = str(report_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_check(
    *,
    check_name: str,
    system_prompt: str,
    payload_builder,
    input_dir: Path,
    output_dir: Path,
    model: str,
    file_glob: str,
    limit: int,
    max_retries: int,
    sleep_seconds: float,
    completion_window: str,
    poll_seconds: float,
) -> int:
    _ = max_retries, sleep_seconds

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    files = list(iter_files(input_dir, file_glob, limit))
    if not files:
        print(f"No files matched in {input_dir} with glob {file_glob}")
        return 0

    client = OpenAI(api_key=require_api_key())

    indexed: list[tuple[str, Path]] = []
    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for idx, path in enumerate(files):
        try:
            entry = load_json(path)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"file": str(path), "reason": f"json_error:{type(exc).__name__}"})
            continue

        if not isinstance(entry, dict):
            skipped.append({"file": str(path), "reason": f"unsupported_json_type:{type(entry).__name__}"})
            continue

        payload = payload_builder(entry)
        custom_id = str(len(indexed))
        indexed.append((custom_id, path))
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                },
            }
        )

    if not requests:
        write_reports(
            output_dir,
            [],
            {
                "check": check_name,
                "input_dir": str(input_dir),
                "processed": 0,
                "counts": {"pass": 0, "needs_review": 0, "error": 0},
                "skipped": skipped,
                "batch_id": None,
                "batch_status": "not_submitted",
                "batch_input": None,
            },
        )
        print(
            json.dumps(
                {
                    "check": check_name,
                    "processed": 0,
                    "counts": {"pass": 0, "needs_review": 0, "error": 0},
                    "skipped": len(skipped),
                    "output_dir": str(output_dir),
                    "batch_status": "not_submitted",
                },
                indent=2,
            )
        )
        return 0

    batch_input_path = output_dir / "batch" / "input.jsonl"
    write_jsonl(batch_input_path, requests)

    batch = submit_batch(
        client,
        input_jsonl=batch_input_path,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"check": check_name, "source": str(input_dir)},
    )
    batch = wait_for_batch(client, batch.id, poll_seconds)

    output_rows, error_rows = collect_batch_results(client, batch)

    row_by_id: dict[str, dict[str, Any]] = {}
    counts = {"pass": 0, "needs_review": 0, "error": 0}

    for custom_id, path in indexed:
        row = {
            "file": str(path),
            "check": check_name,
            "status": "error",
            "confidence": 0,
            "issues": ["missing_batch_output"],
            "suggestions": [],
        }
        row_by_id[custom_id] = row

    for out_row in output_rows:
        custom_id = str(out_row.get("custom_id", ""))
        row = row_by_id.get(custom_id)
        if not row:
            continue

        response = out_row.get("response", {})
        status_code = response.get("status_code")
        if status_code != 200:
            row["status"] = "error"
            row["issues"] = [f"http_status:{status_code}"]
            continue

        try:
            body = response.get("body", {})
            content = body["choices"][0]["message"]["content"]
            result = parse_check_content(content)
            row.update(
                {
                    "status": result.status if result.status in {"pass", "needs_review"} else "needs_review",
                    "confidence": result.confidence,
                    "issues": result.issues,
                    "suggestions": result.suggestions,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["issues"] = [f"exception:{exc}"]

    for err_row in error_rows:
        custom_id = str(err_row.get("custom_id", ""))
        row = row_by_id.get(custom_id)
        if row:
            row["status"] = "error"
            row["issues"] = [f"batch_error:{err_row.get('error', 'unknown')}" ]

    rows = [row_by_id[custom_id] for custom_id, _path in indexed]
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    write_reports(
        output_dir,
        rows,
        {
            "check": check_name,
            "input_dir": str(input_dir),
            "processed": len(rows),
            "counts": counts,
            "skipped": skipped,
            "batch_id": batch.id,
            "batch_status": batch.status,
            "batch_input": str(batch_input_path),
        },
    )

    print(
        json.dumps(
            {
                "check": check_name,
                "processed": len(rows),
                "counts": counts,
                "skipped": len(skipped),
                "output_dir": str(output_dir),
                "batch_id": batch.id,
                "batch_status": batch.status,
            },
            indent=2,
        )
    )
    return 0
