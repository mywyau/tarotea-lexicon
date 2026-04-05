import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_COMPLETION_WINDOW = "24h"

SYSTEM_PROMPT = """You are editing Cantonese learning content.

Task:
- You will receive one JSON word file.
- Rewrite ONLY the top-level "usage" array into natural, concise English.
- Do not change anything else.
- Return valid JSON only in exactly this shape:

{
  "file": "<same file path from the prompt>",
  "usage": ["...", "..."]
}

Rules:
- "usage" must be a JSON array of strings.
- Keep the number of usage entries the same as the original unless impossible.
- Keep the English concise, natural, and learner-friendly.
- Do not return markdown fences.
- Do not include commentary or explanations.
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def collect_flagged_files(report_path: Path) -> list[str]:
    report = load_json(report_path)
    issues = report.get("issues", [])

    files: set[str] = set()

    for issue in issues:
        if issue.get("issue") == "no_latin_translation":
            file_path = issue.get("file")
            if isinstance(file_path, str) and file_path.strip():
                files.add(file_path)

    return sorted(files)


def build_user_prompt(file_path: str, file_contents: str) -> str:
    return (
        f"File path: {file_path}\n\n"
        "Rewrite the top-level `usage` array into English.\n"
        "Return only the patch object.\n\n"
        "Original JSON file:\n"
        f"{file_contents}"
    )


def make_batch_line(
    *,
    custom_id: str,
    model: str,
    file_path: str,
    file_contents: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_user_prompt(file_path, file_contents),
                        }
                    ],
                },
            ],
        },
    }


def write_jsonl(lines: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an OpenAI Batch job from a usage audit report."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the audit report JSON, e.g. reports/usage_untranslated_only.json",
    )
    parser.add_argument(
        "--output-jsonl",
        default="reports/openai/usage_patch_batch.jsonl",
        help="Where to write the batch input JSONL",
    )
    parser.add_argument(
        "--mapping",
        default="reports/openai/usage_patch_mapping.json",
        help="Where to write custom_id -> file mapping",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--completion-window",
        default=DEFAULT_COMPLETION_WINDOW,
        help="Batch completion window",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    report_path = Path(args.report).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    mapping_path = Path(args.mapping).resolve()

    flagged_files = collect_flagged_files(report_path)
    if not flagged_files:
        print("No files with no_latin_translation found in the report.")
        return

    lines: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}

    for i, file_name in enumerate(flagged_files, start=1):
        path = Path(file_name).resolve()

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        try:
            file_contents = load_text(path)
        except Exception as e:
            print(f"Skipping unreadable file {path}: {e}")
            continue

        custom_id = f"usage-patch-{i}"
        mapping[custom_id] = str(path)

        lines.append(
            make_batch_line(
                custom_id=custom_id,
                model=args.model,
                file_path=str(path),
                file_contents=file_contents,
            )
        )

    if not lines:
        print("No valid batch requests generated.")
        return

    write_jsonl(lines, output_jsonl)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(lines)} batch requests to {output_jsonl}")
    print(f"Wrote mapping file to {mapping_path}")

    with output_jsonl.open("rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="batch",
        )

    print(f"Uploaded batch input file: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
        metadata={
            "job": "usage_patch_rewrite",
            "source_report": str(report_path),
            "request_count": str(len(lines)),
        },
    )

    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Endpoint: {batch.endpoint}")
    print(f"Completion window: {batch.completion_window}")


if __name__ == "__main__":
    main()