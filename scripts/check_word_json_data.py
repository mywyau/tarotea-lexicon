#!/usr/bin/env python3
"""Validate word JSON entries and report structural/content issues.

This script is offline-only and does not rewrite source files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Issue:
    severity: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate word JSON files for common data errors")
    parser.add_argument(
        "--input-dir",
        default="r2-backup/content",
        help="Directory containing word JSON files (default: r2-backup/content)",
    )
    parser.add_argument("--glob", default="*.json", help="Filename glob pattern (default: *.json)")
    parser.add_argument("--recursive", action="store_true", help="Find files recursively")
    parser.add_argument(
        "--audio-dir",
        default="r2-backup/audio",
        help="Audio directory used by --check-audio-files (default: r2-backup/audio)",
    )
    parser.add_argument(
        "--check-audio-files",
        action="store_true",
        help="Verify referenced audio files exist on disk",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write machine-readable report JSON",
    )
    return parser.parse_args()


def iter_json_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(input_dir.rglob(pattern))
    return sorted(input_dir.glob(pattern))


def validate_entry(path: Path, data: dict[str, Any], audio_dir: Path, check_audio_files: bool) -> list[Issue]:
    issues: list[Issue] = []

    expected_name = path.stem
    entry_id = data.get("id")
    if not isinstance(entry_id, str) or not entry_id.strip():
        issues.append(Issue("error", "Missing/invalid 'id' (non-empty string required)"))
        entry_id = ""
    elif entry_id != expected_name:
        issues.append(Issue("warning", f"Entry id '{entry_id}' does not match filename '{expected_name}'"))

    for field in ("word", "jyutping", "meaning"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(Issue("error", f"Missing/invalid '{field}' (non-empty string required)"))

    for field in ("pos", "usage", "tags", "related"):
        value = data.get(field)
        if not isinstance(value, list) or not value:
            issues.append(Issue("error", f"Missing/invalid '{field}' (non-empty list required)"))

    examples = data.get("examples")
    if not isinstance(examples, list) or not examples:
        issues.append(Issue("error", "Missing/invalid 'examples' (non-empty list required)"))
        examples = []

    seen_example_ids: set[str] = set()
    for index, ex in enumerate(examples):
        if not isinstance(ex, dict):
            issues.append(Issue("error", f"examples[{index}] must be an object"))
            continue

        ex_id = ex.get("id")
        if not isinstance(ex_id, str) or not ex_id.strip():
            issues.append(Issue("error", f"examples[{index}] missing/invalid 'id'"))
            continue

        if ex_id in seen_example_ids:
            issues.append(Issue("error", f"Duplicate example id '{ex_id}'"))
        seen_example_ids.add(ex_id)

        if entry_id and not ex_id.startswith(f"{entry_id}-example-"):
            issues.append(Issue("warning", f"Example id '{ex_id}' does not start with '{entry_id}-example-'"))

        for field in ("sentence", "jyutping", "meaning"):
            value = ex.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(Issue("error", f"examples[{index}] missing/invalid '{field}'"))

    audio = data.get("audio")
    if not isinstance(audio, dict):
        issues.append(Issue("error", "Missing/invalid 'audio' object"))
    else:
        word_audio = audio.get("word")
        if not isinstance(word_audio, str) or not word_audio.strip():
            issues.append(Issue("error", "audio.word must be a non-empty string"))
        elif entry_id and word_audio != f"{entry_id}.mp3":
            issues.append(Issue("warning", f"audio.word expected '{entry_id}.mp3' but found '{word_audio}'"))

        ex_audio = audio.get("examples")
        if not isinstance(ex_audio, list):
            issues.append(Issue("error", "audio.examples must be a list"))
            ex_audio = []

        if len(ex_audio) != len(examples):
            issues.append(Issue("warning", f"audio.examples has {len(ex_audio)} items but examples has {len(examples)} items"))

        expected_example_audio = {f"{ex_id}.mp3" for ex_id in seen_example_ids}
        actual_example_audio = {x for x in ex_audio if isinstance(x, str) and x.strip()}

        missing_audio_refs = sorted(expected_example_audio - actual_example_audio)
        extra_audio_refs = sorted(actual_example_audio - expected_example_audio)

        if missing_audio_refs:
            issues.append(Issue("warning", f"Missing example audio refs: {', '.join(missing_audio_refs[:5])}"))
        if extra_audio_refs:
            issues.append(Issue("warning", f"Unexpected example audio refs: {', '.join(extra_audio_refs[:5])}"))

        if check_audio_files:
            referenced_files = []
            if isinstance(word_audio, str) and word_audio.strip():
                referenced_files.append(word_audio)
            referenced_files.extend([x for x in ex_audio if isinstance(x, str) and x.strip()])
            for audio_name in referenced_files:
                if not (audio_dir / audio_name).exists():
                    issues.append(Issue("error", f"Referenced audio file not found: {audio_name}"))

    return issues


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    audio_dir = Path(args.audio_dir)

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}")
        return 2

    files = iter_json_files(input_dir, args.glob, args.recursive)
    if not files:
        print(f"No files matched in {input_dir} with glob '{args.glob}'")
        return 0

    report: list[dict[str, Any]] = []
    seen_entry_ids: dict[str, list[str]] = {}

    for path in files:
        item: dict[str, Any] = {"file": str(path), "errors": [], "warnings": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            item["errors"].append(f"Invalid JSON: {type(exc).__name__}: {exc}")
            report.append(item)
            continue

        if not isinstance(data, dict):
            item["errors"].append(f"Top-level JSON must be an object, got {type(data).__name__}")
            report.append(item)
            continue

        entry_id = data.get("id")
        if isinstance(entry_id, str) and entry_id.strip():
            seen_entry_ids.setdefault(entry_id, []).append(str(path))

        issues = validate_entry(path, data, audio_dir, args.check_audio_files)
        for issue in issues:
            key = "errors" if issue.severity == "error" else "warnings"
            item[key].append(issue.message)

        report.append(item)

    duplicate_errors = 0
    for entry_id, id_paths in seen_entry_ids.items():
        if len(id_paths) > 1:
            duplicate_errors += 1
            duplicate_msg = f"Duplicate entry id '{entry_id}' also appears in: {', '.join(id_paths[1:])}"
            for row in report:
                if row["file"] == id_paths[0]:
                    row["errors"].append(duplicate_msg)
                    break

    error_files = sum(1 for row in report if row["errors"])
    warning_files = sum(1 for row in report if row["warnings"])

    print(f"Checked {len(report)} file(s) in {input_dir}")
    print(f"Files with errors: {error_files}")
    print(f"Files with warnings: {warning_files}")

    for row in report:
        if not row["errors"] and not row["warnings"]:
            continue
        rel = row["file"]
        print(f"\n{rel}")
        for msg in row["errors"]:
            print(f"  [ERROR] {msg}")
        for msg in row["warnings"]:
            print(f"  [WARN ] {msg}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dir": str(input_dir),
            "files_checked": len(report),
            "error_files": error_files,
            "warning_files": warning_files,
            "duplicate_ids": duplicate_errors,
            "results": report,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote report: {out_path}")

    return 1 if error_files > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
