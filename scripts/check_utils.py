#!/usr/bin/env python3
"""Shared utilities for OpenAI-backed lexicon check scripts.

These tools are audit-only and never rewrite source files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


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
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for malformed output")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between requests")
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


def call_check_model(client: OpenAI, model: str, system_prompt: str, payload: dict[str, Any], max_retries: int) -> CheckResult:
    last_err: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            return CheckResult(
                status=str(parsed.get("status", "needs_review")).strip().lower(),
                confidence=float(parsed.get("confidence", 0.0)),
                issues=[str(x) for x in parsed.get("issues", [])],
                suggestions=[str(x) for x in parsed.get("suggestions", [])],
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"Model call failed after retries: {last_err}")


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
) -> int:
    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    files = list(iter_files(input_dir, file_glob, limit))
    if not files:
        print(f"No files matched in {input_dir} with glob {file_glob}")
        return 0

    client = OpenAI(api_key=require_api_key())

    rows: list[dict[str, Any]] = []
    counts = {"pass": 0, "needs_review": 0, "error": 0}

    for path in files:
        entry = load_json(path)
        row: dict[str, Any] = {
            "file": str(path),
            "check": check_name,
            "status": "error",
            "confidence": 0,
            "issues": [],
            "suggestions": [],
        }
        try:
            payload = payload_builder(entry)
            result = call_check_model(client, model, system_prompt, payload, max_retries)
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

        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_reports(
        output_dir,
        rows,
        {
            "check": check_name,
            "input_dir": str(input_dir),
            "processed": len(rows),
            "counts": counts,
        },
    )

    print(json.dumps({"check": check_name, "processed": len(rows), "counts": counts, "output_dir": str(output_dir)}, indent=2))
    return 0
