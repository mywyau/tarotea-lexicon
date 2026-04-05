import argparse
import json
import re
from pathlib import Path
from typing import Any

CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def find_usage_fields(data: Any, path: str = "$") -> list[tuple[str, list[Any]]]:
    results: list[tuple[str, list[Any]]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}"
            if key == "usage" and isinstance(value, list):
                results.append((child_path, value))
            else:
                results.extend(find_usage_fields(value, child_path))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            results.extend(find_usage_fields(item, f"{path}[{i}]"))

    return results


def chinese_ratio(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0

    visible_chars = [ch for ch in stripped if not ch.isspace()]
    if not visible_chars:
        return 0.0

    chinese_count = sum(1 for ch in visible_chars if CHINESE_CHAR_RE.match(ch))
    return chinese_count / len(visible_chars)


def check_file(path: Path, threshold: float) -> list[dict[str, Any]]:
    data = load_json(path)
    if data is None:
        return []

    issues: list[dict[str, Any]] = []

    for usage_path, usage_list in find_usage_fields(data):
        for idx, entry in enumerate(usage_list):
            if not isinstance(entry, str):
                issues.append(
                    {
                        "file": str(path),
                        "usage_path": usage_path,
                        "index": idx,
                        "issue": "non_string",
                        "value": entry,
                    }
                )
                continue

            ratio = chinese_ratio(entry)

            if ratio > threshold:
                issues.append(
                    {
                        "file": str(path),
                        "usage_path": usage_path,
                        "index": idx,
                        "issue": "high_chinese_ratio",
                        "chinese_ratio": round(ratio, 4),
                        "threshold": threshold,
                        "value": entry,
                    }
                )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report usage entries where Chinese content exceeds a given ratio."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root folder to scan for JSON files",
    )
    parser.add_argument(
        "--output",
        default="usage_high_chinese_ratio.json",
        help="Path to save the audit report JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Flag entries where Chinese character ratio is greater than this value (default: 0.40)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    json_files = sorted(root.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found under {root}")
        return

    all_issues: list[dict[str, Any]] = []

    for path in json_files:
        all_issues.extend(check_file(path, args.threshold))

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "root": str(root),
        "files_scanned": len(json_files),
        "threshold": args.threshold,
        "issues_found": len(all_issues),
        "issues": all_issues,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Scanned {len(json_files)} JSON files")
    print(f"Found {len(all_issues)} usage entries above Chinese ratio threshold {args.threshold:.2f}")
    print(f"Report saved to: {output_path}")

    if all_issues:
        print("\nSample flagged entries:")
        for issue in all_issues[:20]:
            print(
                f"- {issue['file']} | {issue['usage_path']}[{issue['index']}] "
                f"| ratio={issue.get('chinese_ratio', 'n/a')} | {issue['value']!r}"
            )


if __name__ == "__main__":
    main()