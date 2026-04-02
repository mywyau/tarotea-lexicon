import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------

ROOT_DIR = Path(os.getenv("ROOT_DIR", "./r2-backup/words")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./reports/example-jyutping-review")).resolve()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))

MAX_FILES = int(os.getenv("MAX_FILES", "5"))
SKIP_COPY_FILES = os.getenv("SKIP_COPY_FILES", "true").lower() == "true"

BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")
BATCH_STATUS_POLL_SECONDS = int(os.getenv("BATCH_STATUS_POLL_SECONDS", "20"))

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# Prompt + Schema
# ----------------------------

SYSTEM_PROMPT = (
    "You are reviewing Cantonese-learning example sentences for Jyutping accuracy only.\n\n"

    "For each example, compare:\n"
    "- Chinese sentence\n"
    "- Jyutping\n\n"

    "Your task is to judge whether the Jyutping matches the Chinese sentence accurately.\n\n"

    "Focus on:\n"
    "- wrong readings\n"
    "- wrong tones\n"
    "- missing syllables\n"
    "- extra syllables\n"
    "- mismatched words\n"
    "- incorrect segmentation that changes meaning\n"
    "- obvious inconsistencies between the sentence and the Jyutping\n\n"

    "Rules:\n"
    "- Do not judge sentence naturalness.\n"
    "- Do not judge English meaning.\n"
    "- Do not flag only because another valid reading might also be possible unless the provided Jyutping is likely wrong in context.\n"
    "- Be conservative.\n"
    "- If the Jyutping is clearly correct enough, return approved.\n"
    "- If something looks suspicious but you are not fully sure, return needs_manual_review.\n"
    "- If the Jyutping is likely wrong, incomplete, or mismatched, return likely_incorrect.\n"
    "- Only suggest a corrected Jyutping when confidence is reasonably high.\n"
    "- Write reasons in English only.\n"
    "- Return strict JSON only.\n\n"

    "Before deciding, check whether the provided Jyutping could plausibly be read aloud as the given Chinese sentence by a native Hong Kong Cantonese speaker."
)

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["approved", "needs_manual_review", "likely_incorrect"],
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                    "suggestedJyutping": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [
                    "id",
                    "decision",
                    "reason",
                    "confidence",
                    "suggestedJyutping",
                ],
            },
        },
    },
    "required": ["examples"],
}


# ----------------------------
# Helpers
# ----------------------------

def get_json_files(root: Path) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []

    for p in root.rglob("*.json"):
        if not p.is_file():
            continue

        if SKIP_COPY_FILES and " copy" in p.stem.lower():
            continue

        rp = p.resolve()
        if rp in seen:
            continue

        seen.add(rp)
        files.append(rp)

    return sorted(files)


def reduce_payload(data: dict) -> dict:
    return {
        "id": data.get("id"),
        "word": data.get("word"),
        "examples": [
            {
                "id": ex.get("id"),
                "sentence": ex.get("sentence"),
                "jyutping": ex.get("jyutping"),
            }
            for ex in data.get("examples", [])
            if isinstance(ex, dict)
        ],
    }


def sanitize_custom_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def build_custom_id(index: int, file_path: Path) -> str:
    rel = file_path.relative_to(ROOT_DIR).as_posix()
    return f"req-{index:06d}-{sanitize_custom_id(rel)}"


def write_text_file(path: Path, lines: list[str]) -> None:
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def extract_output_text_from_response_body(body: dict) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []

    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])

    return "".join(chunks).strip()


def parse_review_output_text(output_text: str) -> dict:
    if not output_text.strip():
        raise ValueError("Empty model output_text")
    return json.loads(output_text)


def make_result_from_review(file_path: Path, reduced: dict, review: dict) -> dict:
    issues = []

    example_lookup = {
        ex.get("id"): ex
        for ex in reduced.get("examples", [])
    }

    for ex_review in review.get("examples", []):
        if ex_review.get("decision") != "approved":
            example_id = ex_review.get("id")
            current_example = example_lookup.get(example_id, {})

            issues.append({
                "path": f"Example: {example_id}",
                "decision": ex_review.get("decision", ""),
                "reason": ex_review.get("reason", ""),
                "currentSentence": current_example.get("sentence", ""),
                "currentJyutping": current_example.get("jyutping", ""),
                "suggestedJyutping": ex_review.get("suggestedJyutping") or "",
                "confidence": ex_review.get("confidence", 0),
            })

    return {
        "file": str(file_path),
        "status": "ok" if not issues else "needs_review",
        "summary": "No likely Jyutping issues found" if not issues else "Potential Jyutping issues found",
        "lowConfidence": any(issue["decision"] == "needs_manual_review" for issue in issues),
        "issues": issues,
    }


# ----------------------------
# Batch build / submit
# ----------------------------

def build_batch_request_body(payload: dict) -> dict:
    return {
        "model": OPENAI_MODEL,
        "reasoning": {"effort": OPENAI_REASONING_EFFORT},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "example_jyutping_review",
                "strict": True,
                "schema": REVIEW_SCHEMA,
            }
        },
    }


def prepare_batch_files(files: list[Path]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "batch-manifest.json"
    batch_input_path = OUTPUT_DIR / "batch-input.jsonl"

    manifest: dict[str, dict] = {}
    lines_written = 0

    with batch_input_path.open("w", encoding="utf-8") as out:
        for index, file_path in enumerate(files, start=1):
            try:
                raw = file_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                reduced = reduce_payload(data)
            except Exception as e:
                custom_id = build_custom_id(index, file_path)
                manifest[custom_id] = {
                    "file": str(file_path),
                    "status": "parse_error",
                    "error": str(e),
                }
                continue

            custom_id = build_custom_id(index, file_path)

            manifest[custom_id] = {
                "file": str(file_path),
                "status": "queued",
                "reduced": reduced,
            }

            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": build_batch_request_body(reduced),
            }

            out.write(json.dumps(task, ensure_ascii=False) + "\n")
            lines_written += 1

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared batch JSONL: {batch_input_path}")
    print(f"Prepared manifest: {manifest_path}")
    print(f"Queued requests: {lines_written}")
    print(f"Local parse errors skipped: {sum(1 for v in manifest.values() if v['status'] == 'parse_error')}")

    return batch_input_path, manifest_path


def submit_batch(batch_input_path: Path) -> str:
    with batch_input_path.open("rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="batch",
        )

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={
            "job": "example-jyutping-review",
            "model": OPENAI_MODEL,
        },
    )

    submission = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
        "created_at": batch.created_at,
    }

    (OUTPUT_DIR / "batch-submission.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(submission, ensure_ascii=False, indent=2))
    return batch.id


# ----------------------------
# Batch status / collect
# ----------------------------

def get_batch_id(cli_batch_id: str | None = None) -> str:
    if cli_batch_id:
        return cli_batch_id

    submission_path = OUTPUT_DIR / "batch-submission.json"
    if not submission_path.exists():
        raise RuntimeError("Missing batch-submission.json and no batch_id provided.")

    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    batch_id = submission.get("batch_id")
    if not batch_id:
        raise RuntimeError("batch-submission.json does not contain batch_id")

    return batch_id


def print_batch_status(batch_id: str) -> dict:
    batch = client.batches.retrieve(batch_id)

    status_payload = {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "in_progress_at": getattr(batch, "in_progress_at", None),
        "finalizing_at": getattr(batch, "finalizing_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "failed_at": getattr(batch, "failed_at", None),
        "expired_at": getattr(batch, "expired_at", None),
        "request_counts": {
            "total": batch.request_counts.total if batch.request_counts else None,
            "completed": batch.request_counts.completed if batch.request_counts else None,
            "failed": batch.request_counts.failed if batch.request_counts else None,
        },
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }

    print(json.dumps(status_payload, ensure_ascii=False, indent=2))
    return status_payload


def wait_for_batch(batch_id: str) -> dict:
    while True:
        status_payload = print_batch_status(batch_id)
        status = status_payload["status"]

        if status in {"completed", "failed", "expired", "cancelled"}:
            return status_payload

        time.sleep(BATCH_STATUS_POLL_SECONDS)


def download_file_content(file_id: str, destination: Path) -> None:
    content = client.files.content(file_id).content
    destination.write_bytes(content)


def collect_results(batch_id: str) -> None:
    batch = client.batches.retrieve(batch_id)

    manifest_path = OUTPUT_DIR / "batch-manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing batch-manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results_file_path = OUTPUT_DIR / "batch-results.jsonl"
    errors_file_path = OUTPUT_DIR / "batch-errors.jsonl"

    if batch.output_file_id:
        download_file_content(batch.output_file_id, results_file_path)
        print(f"Downloaded output file -> {results_file_path}")

    if batch.error_file_id:
        download_file_content(batch.error_file_id, errors_file_path)
        print(f"Downloaded error file -> {errors_file_path}")

    final_results: list[dict] = []

    for custom_id, item in manifest.items():
        if item["status"] == "parse_error":
            final_results.append({
                "file": item["file"],
                "status": "parse_error",
                "summary": "Invalid JSON",
                "error": item["error"],
            })

    seen_custom_ids: set[str] = set()

    if results_file_path.exists():
        with results_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                seen_custom_ids.add(custom_id)

                manifest_item = manifest.get(custom_id)
                if not manifest_item:
                    final_results.append({
                        "file": f"<unknown:{custom_id}>",
                        "status": "api_error",
                        "summary": "Batch returned an unknown custom_id",
                        "error": "custom_id not found in local manifest",
                    })
                    continue

                file_path = Path(manifest_item["file"])
                reduced = manifest_item["reduced"]

                if row.get("error"):
                    final_results.append({
                        "file": str(file_path),
                        "status": "api_error",
                        "summary": "OpenAI batch request failed",
                        "error": row["error"].get("message", "Unknown batch error"),
                    })
                    continue

                response = row.get("response") or {}
                body = response.get("body") or {}

                try:
                    output_text = extract_output_text_from_response_body(body)
                    review = parse_review_output_text(output_text)
                    final_results.append(make_result_from_review(file_path, reduced, review))
                except Exception as e:
                    final_results.append({
                        "file": str(file_path),
                        "status": "api_error",
                        "summary": "OpenAI batch output parse failed",
                        "error": str(e),
                    })

    if errors_file_path.exists():
        with errors_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if custom_id in seen_custom_ids:
                    continue

                manifest_item = manifest.get(custom_id)
                if not manifest_item:
                    continue

                final_results.append({
                    "file": manifest_item["file"],
                    "status": "api_error",
                    "summary": "OpenAI batch request failed",
                    "error": (row.get("error") or {}).get("message", "Unknown batch error"),
                })

    for custom_id, item in manifest.items():
        if item["status"] != "queued":
            continue

        already_present = any(r["file"] == item["file"] for r in final_results)
        if already_present:
            continue

        final_results.append({
            "file": item["file"],
            "status": "api_error",
            "summary": "No output found for queued request",
            "error": "Request missing from output and error files",
        })

    final_results.sort(key=lambda r: r["file"])

    ok_files = [r for r in final_results if r["status"] == "ok"]
    needs_review = [r for r in final_results if r["status"] == "needs_review"]
    parse_errors = [r for r in final_results if r["status"] == "parse_error"]
    api_errors = [r for r in final_results if r["status"] == "api_error"]

    summary = {
        "rootDir": str(ROOT_DIR),
        "model": OPENAI_MODEL,
        "batchId": batch_id,
        "totalFiles": len(final_results),
        "okFiles": len(ok_files),
        "needsReview": len(needs_review),
        "parseErrors": len(parse_errors),
        "apiErrors": len(api_errors),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "full-report.json").write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_text_file(OUTPUT_DIR / "needs-review.txt", [r["file"] for r in needs_review])
    write_text_file(OUTPUT_DIR / "ok.txt", [r["file"] for r in ok_files])
    write_text_file(
        OUTPUT_DIR / "parse-errors.txt",
        [f'{r["file"]} :: {r.get("error", "")}' for r in parse_errors],
    )
    write_text_file(
        OUTPUT_DIR / "api-errors.txt",
        [f'{r["file"]} :: {r.get("error", "")}' for r in api_errors],
    )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ----------------------------
# Commands
# ----------------------------

def command_submit() -> None:
    print("Preparing Batch API input...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")

    files = get_json_files(ROOT_DIR)[:MAX_FILES]
    if not files:
        print(f"No JSON files found under: {ROOT_DIR}")
        return

    print(f"Found {len(files)} JSON files")
    print(f"Using model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")

    batch_input_path, _manifest_path = prepare_batch_files(files)
    batch_id = submit_batch(batch_input_path)

    print("\nSubmitted.")
    print(f"Batch ID: {batch_id}")
    print(f"Check status with: python {Path(__file__).name} status {batch_id}")
    print(f"Collect with:      python {Path(__file__).name} collect {batch_id}")


def command_status(batch_id: str | None) -> None:
    print_batch_status(get_batch_id(batch_id))


def command_wait(batch_id: str | None) -> None:
    status_payload = wait_for_batch(get_batch_id(batch_id))
    print("\nFinal status:")
    print(json.dumps(status_payload, ensure_ascii=False, indent=2))


def command_collect(batch_id: str | None) -> None:
    collect_results(get_batch_id(batch_id))


def command_run() -> None:
    print("Preparing Batch API input...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")

    files = get_json_files(ROOT_DIR)[:MAX_FILES]
    if not files:
        print(f"No JSON files found under: {ROOT_DIR}")
        return

    print(f"Found {len(files)} JSON files")
    print(f"Using model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")

    batch_input_path, _manifest_path = prepare_batch_files(files)
    batch_id = submit_batch(batch_input_path)

    print("\nWaiting for batch to finish...")
    final_status = wait_for_batch(batch_id)

    print("\nBatch finished:")
    print(json.dumps(final_status, ensure_ascii=False, indent=2))

    if final_status["status"] in {"completed", "expired"}:
        print("\nCollecting available results...")
        collect_results(batch_id)
    else:
        print(f"\nNot collecting because batch status is {final_status['status']}.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {Path(__file__).name} submit")
        print(f"  python {Path(__file__).name} status [batch_id]")
        print(f"  python {Path(__file__).name} wait [batch_id]")
        print(f"  python {Path(__file__).name} collect [batch_id]")
        print(f"  python {Path(__file__).name} run")
        sys.exit(1)

    command = sys.argv[1].strip().lower()
    batch_id = sys.argv[2].strip() if len(sys.argv) >= 3 else None

    if command == "submit":
        command_submit()
    elif command == "status":
        command_status(batch_id)
    elif command == "wait":
        command_wait(batch_id)
    elif command == "collect":
        command_collect(batch_id)
    elif command == "run":
        command_run()
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()