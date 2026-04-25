#!/usr/bin/env python3
"""Batch audit/rewrite Cantonese word JSON files with OpenAI.

This script is intentionally conservative:
- It keeps the original JSON shape.
- It requires the model to return structured JSON with decision/confidence/issues/updated_entry.
- It validates that critical IDs are preserved.
- It supports dry-run for audit-only workflows.
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


SYSTEM_PROMPT = """You are a Cantonese lexicography QA assistant.

Task:
Audit and, if needed, rewrite ONE Cantonese word entry JSON.

Rules:
1) Return ONLY valid JSON.
2) Output keys exactly: decision, confidence, issues, updated_entry.
3) decision must be one of: accept, rewrite, flag.
4) confidence must be a float from 0 to 1.
5) Preserve original JSON schema and all required keys.
6) Keep id, audio, and example ids unchanged unless clearly invalid.
7) Correct headword and example jyutping for Cantonese accuracy.
8) Keep translations natural and semantically faithful.
9) Do not invent new examples unless needed to fix obvious corruption.
"""


@dataclass
class ModelResult:
    decision: str
    confidence: float
    issues: list[str]
    updated_entry: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch audit/rewrite word JSON files with OpenAI")
    parser.add_argument(
        "--input-dir",
        default="r2-backup/words",
        help="Input directory containing individual word JSON files (default: r2-backup/words)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/rewritten-words",
        help="Directory where rewritten JSON files and reports are written",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--min-confidence", type=float, default=0.8, help="Minimum confidence to auto-apply rewrite")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for malformed model output")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between API requests")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write rewritten files; still writes audit report",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Filename glob relative to input directory (default: *.json)",
    )
    return parser.parse_args()


def iter_files(input_dir: Path, pattern: str, limit: int) -> Iterable[Path]:
    files = sorted(input_dir.glob(pattern))
    if limit > 0:
        files = files[:limit]
    return files


def validate_shape(original: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["id", "word", "jyutping", "meaning", "pos", "usage", "examples", "audio", "tags", "related"]:
        if key not in candidate:
            errors.append(f"missing_key:{key}")
    if original.get("id") != candidate.get("id"):
        errors.append("id_changed")

    original_examples = original.get("examples", [])
    candidate_examples = candidate.get("examples", [])
    if len(original_examples) != len(candidate_examples):
        errors.append("examples_count_changed")
    else:
        for idx, (orig_ex, new_ex) in enumerate(zip(original_examples, candidate_examples, strict=True)):
            if orig_ex.get("id") != new_ex.get("id"):
                errors.append(f"example_id_changed:{idx}")

    if "audio" in original and "audio" in candidate:
        if original["audio"] != candidate["audio"]:
            errors.append("audio_changed")

    return errors


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def call_model(
    client: OpenAI,
    model: str,
    entry: dict[str, Any],
    max_retries: int,
) -> ModelResult:
    user_payload = {
        "constraints": {
            "tone_numbers": "Use Jyutping tone digits 1-6.",
            "stability": "Preserve ids and structure.",
            "return_format": {
                "decision": "accept|rewrite|flag",
                "confidence": "float 0..1",
                "issues": "string[]",
                "updated_entry": "object",
            },
        },
        "entry": entry,
    }

    last_err: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            result = ModelResult(
                decision=str(parsed["decision"]).strip().lower(),
                confidence=float(parsed["confidence"]),
                issues=[str(x) for x in parsed.get("issues", [])],
                updated_entry=dict(parsed["updated_entry"]),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

    raise RuntimeError(f"Model call failed after retries: {last_err}")


def main() -> int:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    rewritten_dir = output_dir / "rewritten"
    reports_dir = output_dir / "reports"
    rewritten_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_files(input_dir, args.glob, args.limit))
    if not files:
        print(f"No files matched in {input_dir} with glob {args.glob}")
        return 0

    client = OpenAI(api_key=api_key)

    report_rows: list[dict[str, Any]] = []
    applied = 0

    for path in files:
        entry = load_json(path)
        row: dict[str, Any] = {
            "file": str(path),
            "status": "unknown",
            "decision": None,
            "confidence": None,
            "issues": [],
            "validation_errors": [],
        }

        try:
            result = call_model(client, args.model, entry, args.max_retries)
            row["decision"] = result.decision
            row["confidence"] = result.confidence
            row["issues"] = result.issues

            validation_errors = validate_shape(entry, result.updated_entry)
            row["validation_errors"] = validation_errors

            should_apply = (
                result.decision in {"rewrite", "accept"}
                and result.confidence >= args.min_confidence
                and not validation_errors
            )

            if should_apply:
                row["status"] = "applied" if not args.dry_run else "would_apply"
                if not args.dry_run:
                    target = rewritten_dir / path.name
                    target.write_text(json.dumps(result.updated_entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    applied += 1
            else:
                row["status"] = "review_needed"

        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["issues"] = [f"exception:{exc}"]

        report_rows.append(row)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    report_path = reports_dir / "audit-report.jsonl"
    with report_path.open("w", encoding="utf-8") as fh:
        for row in report_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_dir": str(input_dir),
        "processed": len(report_rows),
        "applied": applied,
        "dry_run": args.dry_run,
        "report": str(report_path),
        "rewritten_dir": str(rewritten_dir),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
