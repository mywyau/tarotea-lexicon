import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
INPUT_PATH = Path(os.getenv("INPUT_PATH", "./lexicon/lexicon-build/lexicon-map.json")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./lexicon/audit-batch-output")).resolve()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "30"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
COMPLETION_WINDOW = os.getenv("COMPLETION_WINDOW", "24h")

AUTO_KEEP_MANUAL_REVIEW_MIN_CONFIDENCE = float(
    os.getenv("AUTO_KEEP_MANUAL_REVIEW_MIN_CONFIDENCE", "0.65")
)
AUTO_ACCEPT_CORRECTED_MIN_CONFIDENCE = float(
    os.getenv("AUTO_ACCEPT_CORRECTED_MIN_CONFIDENCE", "0.90")
)

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Prompt + Schema
# ----------------------------
SYSTEM_PROMPT = (
    "You are auditing a Cantonese learner lexicon.\n\n"
    "Each entry contains:\n"
    "- id\n"
    "- hanzi\n"
    "- existingJyutping\n\n"
    "Your task is to decide the best learner-friendly Hong Kong Cantonese Jyutping for each entry.\n\n"
    "Important decision policy:\n"
    "- Prefer making a decision rather than escalating.\n"
    "- If the existing jyutping is plausible and standard enough, keep it.\n"
    "- If the existing jyutping is likely wrong, provide a corrected jyutping.\n"
    "- For single-character entries, prefer the most common standalone learner-friendly reading.\n"
    "- Only use manual_review when the text is corrupted, non-Chinese, or genuinely impossible to resolve confidently.\n"
    "- Do not overuse manual_review.\n"
    "- Use lowercase letters, tones 1-6, and spaces between syllables.\n"
    "- Return strict JSON only.\n\n"
    "Allowed statuses:\n"
    "- keep_existing\n"
    "- corrected\n"
    "- non_chinese\n"
    "- manual_review\n"
)

AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "hanzi": {"type": "string"},
                    "existingJyutping": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "keep_existing",
                            "corrected",
                            "non_chinese",
                            "manual_review",
                        ],
                    },
                    "finalJyutping": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "id",
                    "hanzi",
                    "existingJyutping",
                    "status",
                    "finalJyutping",
                    "reason",
                    "confidence",
                ],
            },
        },
    },
    "required": ["entries"],
}

# ----------------------------
# Helpers
# ----------------------------
WHITESPACE_RE = re.compile(r"\s+")
JYUTPING_ALLOWED_RE = re.compile(r"^[a-z1-6 ]+$")
CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def normalize_hanzi(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip())


def normalize_jyutping(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip().lower())


def normalize_reason(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip())


def is_plausible_jyutping(value: str | None) -> bool:
    if not value:
        return False
    v = normalize_jyutping(value)
    return bool(v) and bool(JYUTPING_ALLOWED_RE.fullmatch(v))


def is_chinese_like(value: str | None) -> bool:
    if not value:
        return False
    return bool(CJK_RE.search(value))


def chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def is_auto_id(value: str) -> bool:
    return isinstance(value, str) and value.startswith("auto-")


def hanzi_to_codepoint_slug(hanzi: str) -> str:
    return "-".join(f"u{ord(ch):04x}" for ch in hanzi)


def make_readable_id(hanzi: str, jyutping: str | None) -> str:
    jp = normalize_jyutping(jyutping) or "unknown"
    jp = jp.replace(" ", "-")
    slug = hanzi_to_codepoint_slug(hanzi)
    return f"{jp}-{slug}"


def uniquify_id(base_id: str, used_ids: set[str]) -> str:
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id

    i = 2
    while True:
        candidate = f"{base_id}-{i}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        i += 1


def load_entries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        for entry_id, value in raw.items():
            if not isinstance(entry_id, str) or not isinstance(value, dict):
                continue

            hanzi = normalize_hanzi(value.get("hanzi"))
            jyutping = normalize_jyutping(value.get("jyutping"))

            if not hanzi:
                continue

            entries.append({
                "id": entry_id,
                "hanzi": hanzi,
                "existingJyutping": jyutping or None,
            })
        return entries

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue

            entry_id = item.get("id")
            hanzi = normalize_hanzi(item.get("hanzi"))
            jyutping = normalize_jyutping(item.get("jyutping"))

            if not isinstance(entry_id, str) or not hanzi:
                continue

            entries.append({
                "id": entry_id,
                "hanzi": hanzi,
                "existingJyutping": jyutping or None,
            })
        return entries

    raise RuntimeError("Unsupported lexicon JSON shape")


def build_request_payload(chunk: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "entries": [
            {
                "id": entry["id"],
                "hanzi": entry["hanzi"],
                "existingJyutping": entry.get("existingJyutping"),
            }
            for entry in chunk
        ]
    }


def prefilter_obvious_noise(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean: list[dict[str, Any]] = []
    obvious_noise: list[dict[str, Any]] = []

    for entry in entries:
        hanzi = entry["hanzi"]
        jyutping = entry.get("existingJyutping")

        if not is_chinese_like(hanzi):
            obvious_noise.append({
                "id": entry["id"],
                "hanzi": hanzi,
                "existingJyutping": jyutping,
                "status": "non_chinese",
                "finalJyutping": None,
                "reason": "No CJK Hanzi detected in entry",
                "confidence": 1.0,
            })
            continue

        clean.append(entry)

    return clean, obvious_noise


def validate_model_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)

    normalized["id"] = normalized.get("id")
    normalized["hanzi"] = normalize_hanzi(normalized.get("hanzi"))
    normalized["existingJyutping"] = normalize_jyutping(normalized.get("existingJyutping")) or None
    normalized["finalJyutping"] = normalize_jyutping(normalized.get("finalJyutping")) or None
    normalized["reason"] = normalize_reason(normalized.get("reason"))

    status = normalized.get("status")
    hanzi = normalized.get("hanzi")
    existing = normalized.get("existingJyutping")
    final = normalized.get("finalJyutping")

    try:
        normalized["confidence"] = float(normalized.get("confidence", 0.0) or 0.0)
    except Exception:
        normalized["confidence"] = 0.0

    if status not in {"keep_existing", "corrected", "non_chinese", "manual_review"}:
        normalized["status"] = "manual_review"
        normalized["finalJyutping"] = existing
        normalized["reason"] = f"Invalid status returned by model: {status}"
        normalized["confidence"] = 0.0
        return normalized

    if status != "non_chinese" and not is_chinese_like(hanzi):
        normalized["status"] = "non_chinese"
        normalized["finalJyutping"] = None
        normalized["reason"] = "No CJK Hanzi detected in entry"
        normalized["confidence"] = 1.0
        return normalized

    if status in {"keep_existing", "corrected"}:
        if not final or not is_plausible_jyutping(final):
            normalized["status"] = "manual_review"
            normalized["finalJyutping"] = existing
            normalized["reason"] = f"Invalid finalJyutping format from model: {final}"
            normalized["confidence"] = 0.0
            return normalized

    if status == "keep_existing" and existing and final and existing != final:
        normalized["status"] = "corrected"
        normalized["reason"] = normalized["reason"] or "Model changed jyutping, so row was treated as corrected."

    if status == "corrected" and existing and final and existing == final:
        normalized["status"] = "keep_existing"
        normalized["reason"] = normalized["reason"] or "Model returned same jyutping, so row was treated as keep_existing."

    if status == "non_chinese":
        normalized["finalJyutping"] = None

    return normalized


def soften_manual_review(row: dict[str, Any]) -> dict[str, Any]:
    softened = dict(row)

    status = softened.get("status")
    existing = normalize_jyutping(softened.get("existingJyutping")) or None
    final = normalize_jyutping(softened.get("finalJyutping")) or None
    confidence = float(softened.get("confidence", 0.0) or 0.0)

    if status == "manual_review":
        if existing and is_plausible_jyutping(existing) and confidence >= AUTO_KEEP_MANUAL_REVIEW_MIN_CONFIDENCE:
            softened["status"] = "keep_existing"
            softened["finalJyutping"] = existing
            softened["reason"] = (
                f"Auto-kept existing jyutping after validation. Original reason: {softened.get('reason', '')}"
            ).strip()

    elif status == "corrected":
        if final and is_plausible_jyutping(final) and confidence >= AUTO_ACCEPT_CORRECTED_MIN_CONFIDENCE:
            pass
        elif not final or not is_plausible_jyutping(final):
            softened["status"] = "manual_review"
            softened["finalJyutping"] = existing
            softened["reason"] = (
                f"Correction failed post-validation. Original reason: {softened.get('reason', '')}"
            ).strip()

    return softened


def build_fixed_lexicon_map_and_remap(
    original_entries: list[dict[str, Any]],
    audit_results: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    audit_by_id = {
        row["id"]: row
        for row in audit_results
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }

    used_ids: set[str] = set()
    fixed_map: dict[str, dict[str, str]] = {}
    remap: list[dict[str, str]] = []

    for entry in original_entries:
        entry_id = entry["id"]
        if not is_auto_id(entry_id):
            used_ids.add(entry_id)

    for entry in original_entries:
        old_id = entry["id"]
        hanzi = normalize_hanzi(entry.get("hanzi"))
        existing_jyutping = normalize_jyutping(entry.get("existingJyutping"))

        audit = audit_by_id.get(old_id, {})
        status = audit.get("status")
        final_jyutping = normalize_jyutping(audit.get("finalJyutping")) or existing_jyutping

        if status == "non_chinese":
            continue

        if not hanzi or not final_jyutping:
            continue

        if is_auto_id(old_id):
            base_id = make_readable_id(hanzi, final_jyutping)
            new_id = uniquify_id(base_id, used_ids)
        else:
            new_id = old_id
            used_ids.add(new_id)

        fixed_map[new_id] = {
            "hanzi": hanzi,
            "jyutping": final_jyutping,
        }

        if new_id != old_id:
            remap.append({
                "oldId": old_id,
                "newId": new_id,
                "hanzi": hanzi,
                "jyutping": final_jyutping,
            })

    return fixed_map, remap


# ----------------------------
# Batch helpers
# ----------------------------
def build_batch_request_line(chunk: list[dict[str, Any]], chunk_index: int) -> dict[str, Any]:
    payload = build_request_payload(chunk)
    custom_id = f"chunk-{chunk_index:05d}"

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": OPENAI_MODEL,
            "reasoning": {"effort": OPENAI_REASONING_EFFORT},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "lexicon_audit",
                    "strict": True,
                    "schema": AUDIT_SCHEMA,
                }
            },
        },
    }


def write_batch_jsonl(chunks: list[list[dict[str, Any]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks, start=1):
            line = build_batch_request_line(chunk, idx)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def upload_batch_input_file(path: Path) -> str:
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    return uploaded.id


def create_batch_job(input_file_id: str) -> Any:
    return client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window=COMPLETION_WINDOW,
        metadata={
            "job": "lexicon_audit",
            "model": OPENAI_MODEL,
        },
    )


def poll_batch(batch_id: str) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)

        completed = getattr(counts, "completed", None) if counts else None
        failed = getattr(counts, "failed", None) if counts else None
        total = getattr(counts, "total", None) if counts else None

        print(
            f"Batch {batch_id}: status={status}, "
            f"completed={completed}, failed={failed}, total={total}"
        )

        if status in {"completed", "failed", "expired", "cancelled"}:
            return batch

        time.sleep(POLL_SECONDS)


def download_file_text(file_id: str, path: Path) -> str:
    content = client.files.content(file_id)
    text = content.text
    path.write_text(text, encoding="utf-8")
    return text


def extract_output_text_from_batch_response_body(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response_body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)

    return "".join(chunks).strip()


def parse_batch_output_file(
    output_text: str,
    chunk_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    rows_by_custom_id: dict[str, list[dict[str, Any]]] = {}
    parse_errors_by_custom_id: dict[str, str] = {}

    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        custom_id = item.get("custom_id")
        response = item.get("response", {})
        body = response.get("body", {})

        if not isinstance(custom_id, str):
            continue

        try:
            model_text = extract_output_text_from_batch_response_body(body)
            if not model_text:
                parse_errors_by_custom_id[custom_id] = "Missing output_text for batch item"
                continue

            parsed = json.loads(model_text)
            entries = parsed.get("entries")
            if not isinstance(entries, list):
                parse_errors_by_custom_id[custom_id] = "Batch item missing entries array"
                continue

            rows_by_custom_id[custom_id] = entries

        except Exception as e:
            preview = ""
            try:
                preview = model_text[:500]
            except Exception:
                pass

            parse_errors_by_custom_id[custom_id] = (
                f"Failed to parse model JSON: {e}. Preview: {preview}"
            )

    expected_ids = {f"chunk-{i:05d}" for i in range(1, chunk_count + 1)}
    for missing_id in sorted(expected_ids - set(rows_by_custom_id.keys()) - set(parse_errors_by_custom_id.keys())):
        parse_errors_by_custom_id[missing_id] = "Missing completed batch result for chunk"

    return rows_by_custom_id, parse_errors_by_custom_id


def parse_batch_error_file(error_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in error_text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def chunk_fallback_rows(chunk: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "id": entry["id"],
            "hanzi": entry["hanzi"],
            "existingJyutping": entry.get("existingJyutping"),
            "status": "manual_review",
            "finalJyutping": entry.get("existingJyutping"),
            "reason": reason,
            "confidence": 0.0,
        }
        for entry in chunk
    ]

def build_results_from_batch(
    chunks: list[list[dict[str, Any]]],
    rows_by_custom_id: dict[str, list[dict[str, Any]]],
    batch_errors: list[dict[str, Any]],
    parse_errors_by_custom_id: dict[str, str],
) -> list[dict[str, Any]]:
    error_by_custom_id: dict[str, dict[str, Any]] = {}
    for row in batch_errors:
        custom_id = row.get("custom_id")
        if isinstance(custom_id, str):
            error_by_custom_id[custom_id] = row

    all_results: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        custom_id = f"chunk-{idx:05d}"

        if custom_id in error_by_custom_id:
            err = error_by_custom_id[custom_id]
            message = json.dumps(err, ensure_ascii=False)
            all_results.extend(chunk_fallback_rows(chunk, f"Batch item failed: {message}"))
            continue

        if custom_id in parse_errors_by_custom_id:
            all_results.extend(
                chunk_fallback_rows(
                    chunk,
                    f"Batch item parse failed: {parse_errors_by_custom_id[custom_id]}",
                )
            )
            continue

        returned_rows = rows_by_custom_id.get(custom_id)
        if returned_rows is None:
            all_results.extend(chunk_fallback_rows(chunk, "Batch item missing from output"))
            continue

        if len(returned_rows) != len(chunk):
            all_results.extend(
                chunk_fallback_rows(
                    chunk,
                    f"Batch item returned {len(returned_rows)} rows for chunk of {len(chunk)}",
                )
            )
            continue

        validated_rows = [validate_model_row(row) for row in returned_rows]
        softened_rows = [soften_manual_review(row) for row in validated_rows]
        all_results.extend(softened_rows)

    return all_results

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lexicon from: {INPUT_PATH}")
    entries = load_entries(INPUT_PATH)
    print(f"Loaded entries: {len(entries)}")

    clean_entries, obvious_noise = prefilter_obvious_noise(entries)
    print(f"Prefiltered obvious non-Chinese entries: {len(obvious_noise)}")
    print(f"Entries to audit with batch model run: {len(clean_entries)}")

    chunks = chunked(clean_entries, CHUNK_SIZE)
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total chunks: {len(chunks)}")

    batch_input_path = OUTPUT_DIR / "batch-input.jsonl"
    write_batch_jsonl(chunks, batch_input_path)
    print(f"Wrote batch input: {batch_input_path}")

    input_file_id = upload_batch_input_file(batch_input_path)
    print(f"Uploaded batch input file: {input_file_id}")

    batch = create_batch_job(input_file_id)
    batch_id = batch.id
    print(f"Created batch: {batch_id}")

    final_batch = poll_batch(batch_id)

    error_rows: list[dict[str, Any]] = []
    output_rows_by_custom_id: dict[str, list[dict[str, Any]]] = {}
    parse_errors_by_custom_id: dict[str, str] = {}

    if getattr(final_batch, "output_file_id", None):
        output_file_id = final_batch.output_file_id
        output_path = OUTPUT_DIR / "batch-output.jsonl"
        output_text = download_file_text(output_file_id, output_path)
        print(f"Downloaded batch output: {output_path}")

        output_rows_by_custom_id, parse_errors_by_custom_id = parse_batch_output_file(
            output_text,
            len(chunks),
        )

    if getattr(final_batch, "error_file_id", None):
        error_file_id = final_batch.error_file_id
        error_path = OUTPUT_DIR / "batch-errors.jsonl"
        error_text = download_file_text(error_file_id, error_path)
        print(f"Downloaded batch errors: {error_path}")
        error_rows = parse_batch_error_file(error_text)

    all_results: list[dict[str, Any]] = []
    all_results.extend(obvious_noise)

    if chunks:
        all_results.extend(
            build_results_from_batch(
                chunks,
                output_rows_by_custom_id,
                error_rows,
                parse_errors_by_custom_id,
            )
        )

    accepted_entries = [
        e for e in all_results
        if e["status"] in {"keep_existing", "corrected"}
    ]
    manual_review_entries = [
        e for e in all_results
        if e["status"] == "manual_review"
    ]
    non_chinese_entries = [
        e for e in all_results
        if e["status"] == "non_chinese"
    ]

    summary = {
        "inputPath": str(INPUT_PATH),
        "model": OPENAI_MODEL,
        "batchId": batch_id,
        "inputFileId": input_file_id,
        "batchStatus": getattr(final_batch, "status", None),
        "totalEntries": len(entries),
        "auditedEntries": len(all_results),
        "acceptedCount": len(accepted_entries),
        "manualReviewCount": len(manual_review_entries),
        "nonChineseCount": len(non_chinese_entries),
        "statusBreakdown": {},
    }

    for entry in all_results:
        status = entry["status"]
        summary["statusBreakdown"][status] = summary["statusBreakdown"].get(status, 0) + 1

    fixed_lexicon_map, id_remap = build_fixed_lexicon_map_and_remap(entries, all_results)

    fixed_runtime_min = [
        {
            "id": entry_id,
            "hanzi": value["hanzi"],
            "jyutping": value["jyutping"],
        }
        for entry_id, value in sorted(fixed_lexicon_map.items())
    ]

    (OUTPUT_DIR / "audit-all.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "audit-accepted.json").write_text(
        json.dumps(accepted_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "audit-manual-review.json").write_text(
        json.dumps(manual_review_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "audit-non-chinese.json").write_text(
        json.dumps(non_chinese_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "audit-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "lexicon-map-fixed.json").write_text(
        json.dumps(fixed_lexicon_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "lexicon-min-fixed.json").write_text(
        json.dumps(fixed_runtime_min, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "id-remap.json").write_text(
        json.dumps(id_remap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Fixed lexicon entries: {len(fixed_lexicon_map)}")
    print(f"Remapped ids: {len(id_remap)}")

if __name__ == "__main__":
    main()