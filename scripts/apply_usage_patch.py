import argparse
import json
import re
from pathlib import Path
from typing import Any

CHINESE_CHAR_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extract_patch_text(row: dict[str, Any]) -> str:
    output = row["response"]["body"]["output"]

    for item in output:
        if item.get("type") != "message":
            continue

        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    raise ValueError("Could not find output_text")


def validate_patch(patch: Any, reject_chinese: bool) -> tuple[Path, list[str]]:
    if not isinstance(patch, dict):
        raise ValueError("Patch is not an object")

    file_value = patch.get("file")
    usage_value = patch.get("usage")

    if not isinstance(file_value, str) or not file_value.strip():
        raise ValueError("Missing valid file path")

    if not isinstance(usage_value, list) or not usage_value:
        raise ValueError("Missing valid usage array")

    if not all(isinstance(x, str) and x.strip() for x in usage_value):
        raise ValueError("Usage must be a non-empty list of non-empty strings")

    if reject_chinese:
        for entry in usage_value:
            if CHINESE_CHAR_RE.search(entry):
                raise ValueError(f"Chinese still present in usage entry: {entry!r}")

    return Path(file_value).resolve(), usage_value


def apply_patch(
    original_file: Path,
    usage: list[str],
    output_dir: Path | None,
    in_place: bool,
    root: Path | None,
) -> Path:
    data = load_json(original_file)

    if not isinstance(data, dict):
        raise ValueError(f"Original file is not a JSON object: {original_file}")

    data["usage"] = usage

    if in_place:
        destination = original_file
    else:
        if output_dir is None:
            raise ValueError("output_dir required unless --in-place is used")

        if root is not None:
            try:
                relative = original_file.relative_to(root)
            except ValueError:
                relative = Path(original_file.name)
        else:
            relative = Path(original_file.name)

        destination = output_dir / relative

    save_json(destination, data)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply batch usage patches locally")
    parser.add_argument("--batch-output", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-dir", default="reports/openai/applied_usage_patches")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--reject-chinese", action="store_true")
    args = parser.parse_args()

    batch_output = Path(args.batch_output).resolve()
    root = Path(args.root).resolve() if args.root else None
    output_dir = None if args.in_place else Path(args.output_dir).resolve()

    applied = 0
    failed = 0

    with batch_output.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                if row.get("error") is not None:
                    raise ValueError(f"Top-level error: {row['error']}")

                response = row.get("response")
                if not isinstance(response, dict):
                    raise ValueError("Missing response")

                if response.get("status_code") != 200:
                    raise ValueError(f"Bad status code: {response.get('status_code')}")

                patch_text = extract_patch_text(row)
                patch = json.loads(patch_text)

                original_file, usage = validate_patch(patch, reject_chinese=args.reject_chinese)

                if not original_file.exists():
                    raise FileNotFoundError(f"Original file not found: {original_file}")

                destination = apply_patch(
                    original_file=original_file,
                    usage=usage,
                    output_dir=output_dir,
                    in_place=args.in_place,
                    root=root,
                )

                print(f"[OK] {original_file} -> {destination}")
                applied += 1

            except Exception as e:
                print(f"[FAIL] line {line_number}: {e}")
                failed += 1

    print()
    print(f"Applied: {applied}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()