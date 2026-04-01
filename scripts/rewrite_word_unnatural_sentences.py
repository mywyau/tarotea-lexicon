import json
import os
import re
import sys
import time
import shutil
from pathlib import Path

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------

ROOT_DIR = Path(os.getenv("ROOT_DIR", "./r2-backup/words")).resolve()
REVIEW_REPORT_PATH = Path(
    os.getenv("REVIEW_REPORT_PATH", "./reports/example-naturalness-review/full-report.json")
).resolve()

OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", "./reports/example-naturalness-rewrite")
).resolve()

PATCHED_ROOT_DIR = Path(
    os.getenv("PATCHED_ROOT_DIR", str(OUTPUT_DIR / "rewritten-json"))
).resolve()

UNCHANGED_SOURCE_ROOT_DIR = Path(
    os.getenv("UNCHANGED_SOURCE_ROOT_DIR", str(OUTPUT_DIR / "unchanged-source-json"))
).resolve()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1400"))
# TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

MAX_FILES = int(os.getenv("MAX_FILES", "5"))
REWRITE_DECISIONS = {
    value.strip()
    for value in os.getenv(
        "REWRITE_DECISIONS", "likely_unnatural,needs_manual_review"
    ).split(",")
    if value.strip()
}

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0"))
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
    "You are an expert Hong Kong Cantonese editor.\n\n"
    "Rewrite only the flagged example sentences for a Cantonese vocabulary entry.\n\n"
    "Rules:\n"
    "- Make each example sound natural in everyday Hong Kong Cantonese.\n"
    "- Preserve the original meaning and scenario as closely as possible.\n"
    "- Prefer the smallest possible edit.\n"
    "- Do not invent a new context unless the original context is clearly unnatural or misleading.\n"
    "- Keep the headword clearly relevant.\n"
    "- Keep examples short, practical, and plausible.\n"
    "- Fix the Jyutping so it exactly matches the rewritten Chinese sentence.\n"
    "- Fix the English meaning so it exactly matches the rewritten Chinese sentence.\n"
    "- Do not rewrite examples that were not provided.\n\n"
    "Output rules:\n"
    "- Return exactly one result for each input example id.\n"
    "- Preserve each example id exactly.\n"
    "- Write changeSummary in English only and keep it brief.\n"
    "- Return strict JSON only.\n"
)

REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "sentence": {"type": "string"},
                    "jyutping": {"type": "string"},
                    "meaning": {"type": "string"},
                    "changeSummary": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "id",
                    "sentence",
                    "jyutping",
                    "meaning",
                    "changeSummary",
                    "confidence",
                ],
            },
        },
    },
    "required": ["examples"],
}


# ----------------------------
# Helpers
# ----------------------------

def sanitize_custom_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def build_custom_id(index: int, file_path: Path) -> str:
    rel = file_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    return f"rewrite-{index:06d}-{sanitize_custom_id(rel)}"


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

def parse_json_output_text(output_text: str) -> dict:
    if not output_text.strip():
        raise ValueError("Empty model output_text")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return json.loads(extract_first_json_object(output_text))
    

def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError("No complete JSON object found in output")


def parse_json_output_text(output_text: str) -> dict:
    if not output_text.strip():
        raise ValueError("Empty model output_text")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return json.loads(extract_first_json_object(output_text))

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


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_review_targets() -> list[dict]:
    if not REVIEW_REPORT_PATH.exists():
        raise RuntimeError(f"Missing review report: {REVIEW_REPORT_PATH}")

    raw = REVIEW_REPORT_PATH.read_text(encoding="utf-8")
    report = json.loads(raw)

    if not isinstance(report, list):
        raise RuntimeError("Review report must be a JSON array")

    grouped: dict[Path, list[dict]] = {}

    for item in report:
        if not isinstance(item, dict):
            continue

        if item.get("status") != "needs_review":
            continue

        file_path = Path(item.get("file", "")).resolve()
        issues = item.get("issues", [])
        if not isinstance(issues, list):
            continue

        selected_issues: list[dict] = []

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            decision = issue.get("decision", "")
            confidence = float(issue.get("confidence", 0) or 0)

            if decision not in REWRITE_DECISIONS:
                continue
            if confidence < MIN_CONFIDENCE:
                continue

            path_value = issue.get("path", "")
            match = re.match(r"Example:\s*(.+)", path_value)
            if not match:
                continue

            example_id = match.group(1).strip()
            if not example_id:
                continue

            selected_issues.append({
                "id": example_id,
                "decision": decision,
                "reason": issue.get("reason", ""),
                "confidence": confidence,
                "currentSentence": issue.get("currentSentence", ""),
                "currentJyutping": issue.get("currentJyutping", ""),
                "currentMeaning": issue.get("currentMeaning", ""),
            })

        if selected_issues:
            grouped[file_path] = selected_issues

    results = [
        {
            "file_path": file_path,
            "issues": issues,
        }
        for file_path, issues in grouped.items()
    ]

    results.sort(key=lambda x: str(x["file_path"]))
    return results[:MAX_FILES]


def build_rewrite_payload(word_data: dict, issues: list[dict]) -> dict:
    issue_lookup = {issue["id"]: issue for issue in issues}

    examples = []
    for ex in word_data.get("examples", []):
        if not isinstance(ex, dict):
            continue
        ex_id = ex.get("id")
        if ex_id not in issue_lookup:
            continue

        issue = issue_lookup[ex_id]
        examples.append({
            "id": ex_id,
            "currentSentence": ex.get("sentence", ""),
            "currentJyutping": ex.get("jyutping", ""),
            "currentMeaning": ex.get("meaning", ""),
            "decision": issue.get("decision", ""),
            "reviewReason": issue.get("reason", ""),
            "reviewConfidence": issue.get("confidence", 0),
        })

    return {
        "id": word_data.get("id"),
        "word": word_data.get("word"),
        "meaning": word_data.get("meaning"),
        "examplesToRewrite": examples,
    }


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
                "name": "example_rewrite_result",
                "strict": True,
                "schema": REWRITE_SCHEMA,
            }
        },
    }


def patch_word_json(original_data: dict, rewritten: dict) -> tuple[dict, list[dict]]:
    rewritten_lookup = {
        item["id"]: item
        for item in rewritten.get("examples", [])
        if isinstance(item, dict) and item.get("id")
    }

    patched = json.loads(json.dumps(original_data, ensure_ascii=False))
    changes: list[dict] = []

    for ex in patched.get("examples", []):
        if not isinstance(ex, dict):
            continue

        ex_id = ex.get("id")
        rewrite = rewritten_lookup.get(ex_id)
        if not rewrite:
            continue

        before = {
            "sentence": ex.get("sentence", ""),
            "jyutping": ex.get("jyutping", ""),
            "meaning": ex.get("meaning", ""),
        }

        ex["sentence"] = rewrite.get("sentence", "")
        ex["jyutping"] = rewrite.get("jyutping", "")
        ex["meaning"] = rewrite.get("meaning", "")

        after = {
            "sentence": ex.get("sentence", ""),
            "jyutping": ex.get("jyutping", ""),
            "meaning": ex.get("meaning", ""),
        }

        changes.append({
            "id": ex_id,
            "before": before,
            "after": after,
            "changeSummary": rewrite.get("changeSummary", ""),
            "confidence": rewrite.get("confidence", 0),
        })

    return patched, changes


def copy_unchanged_source_files(rewritten_source_files: set[Path]) -> tuple[int, list[str]]:
    UNCHANGED_SOURCE_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    copied_paths: list[str] = []

    for file_path in get_json_files(ROOT_DIR):
        resolved_file = file_path.resolve()

        if resolved_file in rewritten_source_files:
            continue

        relative_file = resolved_file.relative_to(ROOT_DIR.resolve())
        destination = UNCHANGED_SOURCE_ROOT_DIR / relative_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_file, destination)

        copied_paths.append(str(destination))
        count += 1

    return count, copied_paths


# ----------------------------
# Batch build / submit
# ----------------------------

def prepare_batch_files(targets: list[dict]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATCHED_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "rewrite-batch-manifest.json"
    batch_input_path = OUTPUT_DIR / "rewrite-batch-input.jsonl"

    manifest: dict[str, dict] = {}
    lines_written = 0

    with batch_input_path.open("w", encoding="utf-8") as out:
        for index, target in enumerate(targets, start=1):
            file_path: Path = target["file_path"]
            issues: list[dict] = target["issues"]

            try:
                raw = file_path.read_text(encoding="utf-8")
                word_data = json.loads(raw)
                payload = build_rewrite_payload(word_data, issues)
            except Exception as e:
                custom_id = build_custom_id(index, file_path)
                manifest[custom_id] = {
                    "file": str(file_path),
                    "relativeFile": str(file_path.resolve().relative_to(ROOT_DIR.resolve())),
                    "status": "parse_error",
                    "error": str(e),
                }
                continue

            if not payload["examplesToRewrite"]:
                custom_id = build_custom_id(index, file_path)
                manifest[custom_id] = {
                    "file": str(file_path),
                    "relativeFile": str(file_path.resolve().relative_to(ROOT_DIR.resolve())),
                    "status": "skipped",
                    "error": "No matching flagged examples found in file",
                }
                continue

            custom_id = build_custom_id(index, file_path)

            manifest[custom_id] = {
                "file": str(file_path),
                "relativeFile": str(file_path.resolve().relative_to(ROOT_DIR.resolve())),
                "status": "queued",
                "targetExampleIds": [item["id"] for item in payload["examplesToRewrite"]],
            }

            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": build_batch_request_body(payload),
            }

            out.write(json.dumps(task, ensure_ascii=False) + "\n")
            lines_written += 1

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared batch JSONL: {batch_input_path}")
    print(f"Prepared manifest: {manifest_path}")
    print(f"Queued rewrite requests: {lines_written}")
    print(
        f"Local parse/skipped count: "
        f"{sum(1 for v in manifest.values() if v['status'] in {'parse_error', 'skipped'})}"
    )

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
            "job": "example-naturalness-rewrite",
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

    (OUTPUT_DIR / "rewrite-batch-submission.json").write_text(
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

    submission_path = OUTPUT_DIR / "rewrite-batch-submission.json"
    if not submission_path.exists():
        raise RuntimeError("Missing rewrite-batch-submission.json and no batch_id provided.")

    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    batch_id = submission.get("batch_id")
    if not batch_id:
        raise RuntimeError("rewrite-batch-submission.json does not contain batch_id")

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

    manifest_path = OUTPUT_DIR / "rewrite-batch-manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing rewrite-batch-manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results_file_path = OUTPUT_DIR / "rewrite-batch-results.jsonl"
    errors_file_path = OUTPUT_DIR / "rewrite-batch-errors.jsonl"

    for stale_path in [results_file_path, errors_file_path]:
        if stale_path.exists():
            stale_path.unlink()

    reset_directory(PATCHED_ROOT_DIR)
    reset_directory(UNCHANGED_SOURCE_ROOT_DIR)

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
        elif item["status"] == "skipped":
            final_results.append({
                "file": item["file"],
                "status": "skipped",
                "summary": item["error"],
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
                    rewritten = parse_json_output_text(output_text)

                    original_data = json.loads(file_path.read_text(encoding="utf-8"))
                    patched_data, changes = patch_word_json(original_data, rewritten)

                    if changes:
                        relative_file = Path(manifest_item["relativeFile"])
                        output_file = PATCHED_ROOT_DIR / relative_file
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        output_file.write_text(
                            json.dumps(patched_data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )

                        final_results.append({
                            "file": str(file_path),
                            "rewrittenFile": str(output_file),
                            "status": "ok",
                            "summary": f"Rewrote {len(changes)} example(s)",
                            "rewrittenCount": len(changes),
                            "changes": changes,
                        })
                    else:
                        final_results.append({
                            "file": str(file_path),
                            "status": "api_error",
                            "summary": "Model returned no matching rewrites",
                            "error": "No matching example ids were returned by the model",
                        })
                except Exception as e:
                    final_results.append({
                        "file": str(file_path),
                        "status": "api_error",
                        "summary": "OpenAI batch output parse/apply failed",
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
    skipped_files = [r for r in final_results if r["status"] == "skipped"]
    parse_errors = [r for r in final_results if r["status"] == "parse_error"]
    api_errors = [r for r in final_results if r["status"] == "api_error"]

    rewritten_source_files = {
        Path(r["file"]).resolve()
        for r in ok_files
    }

    unchanged_source_count, unchanged_files = copy_unchanged_source_files(rewritten_source_files)

    summary = {
        "rootDir": str(ROOT_DIR),
        "reviewReportPath": str(REVIEW_REPORT_PATH),
        "model": OPENAI_MODEL,
        "batchId": batch_id,
        "totalFiles": len(final_results),
        "rewrittenFiles": len(ok_files),
        "skippedFiles": len(skipped_files),
        "parseErrors": len(parse_errors),
        "apiErrors": len(api_errors),
        "patchedRootDir": str(PATCHED_ROOT_DIR),
        "unchangedSourceRootDir": str(UNCHANGED_SOURCE_ROOT_DIR),
        "unchangedSourceFiles": unchanged_source_count,
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "full-report.json").write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_text_file(
        OUTPUT_DIR / "rewritten-files.txt",
        [f'{r["file"]} -> {r.get("rewrittenFile", "")}' for r in ok_files],
    )
    write_text_file(
        OUTPUT_DIR / "skipped-files.txt",
        [r["file"] for r in skipped_files],
    )
    write_text_file(
        OUTPUT_DIR / "parse-errors.txt",
        [f'{r["file"]} :: {r.get("error", "")}' for r in parse_errors],
    )
    write_text_file(
        OUTPUT_DIR / "api-errors.txt",
        [f'{r["file"]} :: {r.get("error", "")}' for r in api_errors],
    )
    write_text_file(
        OUTPUT_DIR / "unchanged-source-files.txt",
        unchanged_files,
    )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ----------------------------
# Commands
# ----------------------------

def command_submit() -> None:
    print("Preparing rewrite Batch API input...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")
    print(f"Using review report: {REVIEW_REPORT_PATH}")

    targets = load_review_targets()
    if not targets:
        print("No flagged review targets found.")
        return

    print(f"Found {len(targets)} target files")
    print(f"Using model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Patched copies dir: {PATCHED_ROOT_DIR}")

    batch_input_path, _manifest_path = prepare_batch_files(targets)
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
    print("Preparing rewrite Batch API input...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")
    print(f"Using review report: {REVIEW_REPORT_PATH}")

    targets = load_review_targets()
    if not targets:
        print("No flagged review targets found.")
        return

    print(f"Found {len(targets)} target files")
    print(f"Using model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Patched copies dir: {PATCHED_ROOT_DIR}")

    batch_input_path, _manifest_path = prepare_batch_files(targets)
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