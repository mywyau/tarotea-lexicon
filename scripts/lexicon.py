# import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------




ROOT_DIR = Path(os.getenv("ROOT_DIR", "./r2-backup/words")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./lexicon/lexicon-build")).resolve()

UNKNOWN_REPORT_PATH = Path(
    os.getenv("UNKNOWN_REPORT_PATH", str(OUTPUT_DIR / "unknown-chunk-report.json"))
).resolve()

MERGE_EXISTING = os.getenv("MERGE_EXISTING", "true").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

MAX_FILES = int(os.getenv("MAX_FILES", "5000"))
SKIP_COPY_FILES = os.getenv("SKIP_COPY_FILES", "true").lower() == "true"

BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")
BATCH_STATUS_POLL_SECONDS = int(os.getenv("BATCH_STATUS_POLL_SECONDS", "20"))

# Number of lexicon entries per batch request
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "40"))

# How many example sentences to include per entry for context
MAX_EXAMPLES_PER_ENTRY = int(os.getenv("MAX_EXAMPLES_PER_ENTRY", "2"))


MIN_CANDIDATE_FREQUENCY = int(os.getenv("MIN_CANDIDATE_FREQUENCY", "2"))

MIN_CANDIDATE_CHUNK_LEN = 1
MAX_CANDIDATE_CHUNK_LEN = int(os.getenv("MAX_CANDIDATE_CHUNK_LEN", "1"))

# Sharding mode for compiled runtime output:
# - none
# - jyutping_initial
# - id_hash_prefix
SHARD_MODE = os.getenv("SHARD_MODE", "jyutping_initial").strip().lower()

if SHARD_MODE not in {"none", "jyutping_initial", "id_hash_prefix"}:
    raise RuntimeError("SHARD_MODE must be one of: none, jyutping_initial, id_hash_prefix")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# Prompt + Schema
# ----------------------------

SYSTEM_PROMPT = (
    "You are building a Cantonese lexicon for learners.\n\n"
    "You will receive a list of entries. Each entry may include:\n"
    "- id\n"
    "- Chinese word/phrase\n"
    "- existing Jyutping (possibly missing or wrong)\n"
    "- meaning\n"
    "- example sentences for context\n\n"

    "Your task:\n"
    "For each entry, return the best canonical Hong Kong Cantonese Jyutping for the Chinese word or phrase.\n\n"

    "Important rules:\n"
    "- Focus only on the Chinese word/phrase itself, not the full sentence reading unless needed for disambiguation.\n"
    "- Use the examples only as context to infer the intended reading.\n"
    "- Be conservative.\n"
    "- If the existing Jyutping looks correct enough, keep it.\n"
    "- If it looks wrong and you are reasonably confident, correct it.\n"
    "- If the Chinese is ambiguous or you are not confident enough, return needs_manual_review.\n"
    "- Prefer standard learner-friendly Jyutping formatting with lowercase letters, tones 1-6, and spaces between syllables.\n"
    "- Do not invent extra meanings or metadata.\n"
    "- Do not rewrite the Chinese text.\n"
    "- Return strict JSON only.\n\n"

    "Decision meanings:\n"
    "- approved_existing: existing Jyutping is acceptable as the canonical entry\n"
    "- approved_corrected: existing Jyutping was likely wrong and you corrected it\n"
    "- approved_generated: there was no usable Jyutping and you generated one with reasonable confidence\n"
    "- needs_manual_review: you are not confident enough to approve a canonical Jyutping\n\n"

    "Before deciding, ask whether a Hong Kong Cantonese speaker would plausibly read the given Chinese word/phrase with the returned Jyutping in this lexical context."
)

LEXICON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "hanzi": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": [
                            "approved_existing",
                            "approved_corrected",
                            "approved_generated",
                            "needs_manual_review",
                        ],
                    },
                    "jyutping": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "id",
                    "hanzi",
                    "decision",
                    "jyutping",
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

JYUTPING_ALLOWED_RE = re.compile(r"^[a-z1-6 ]+$")
WHITESPACE_RE = re.compile(r"\s+")
NON_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
# LATIN_PREFIX_RE = re.compile(r"^[a-z]+")


LATIN_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9]")


def build_auto_candidate_id(chunk: str) -> str:
    digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
    return f"auto-{digest}"


def is_good_unknown_candidate(
    chunk: str,
    *,
    min_len: int = 1,
    max_len: int = 2,
    exclude_latin: bool = True,
) -> bool:
    chunk = normalize_hanzi(chunk)
    if not chunk:
        return False
    if len(chunk) < min_len or len(chunk) > max_len:
        return False
    if exclude_latin and LATIN_OR_DIGIT_RE.search(chunk):
        return False
    return True


def load_unknown_chunk_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing unknown chunk report: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("unknown-chunk-report.json must contain a top-level array")

    return raw


def build_candidate_entries_from_unknown_report(
    report_path: Path,
    *,
    min_frequency: int,
    min_len: int,
    max_len: int,
    max_examples_per_entry: int,
    exclude_latin: bool = True,
) -> list[dict[str, Any]]:
    rows = load_unknown_chunk_report(report_path)
    out: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        hanzi = normalize_hanzi(row.get("hanzi"))
        count = row.get("count", 0)
        occurrences = row.get("occurrences", [])

        if not isinstance(count, int):
            continue
        if count < min_frequency:
            continue
        if not is_good_unknown_candidate(
            hanzi,
            min_len=min_len,
            max_len=max_len,
            exclude_latin=exclude_latin,
        ):
            continue

        examples: list[dict[str, str]] = []
        seen_example_keys: set[tuple[str, str]] = set()

        if isinstance(occurrences, list):
            for occ in occurrences:
                if not isinstance(occ, dict):
                    continue

                sentence = normalize_hanzi(occ.get("sentence"))
                jyutping = normalize_jyutping(occ.get("jyutping"))

                if not sentence:
                    continue

                key = (sentence, jyutping)
                if key in seen_example_keys:
                    continue

                seen_example_keys.add(key)
                examples.append({
                    "sentence": sentence,
                    "jyutping": jyutping,
                })

                if len(examples) >= max_examples_per_entry:
                    break

        out.append({
            "id": build_auto_candidate_id(hanzi),
            "hanzi": hanzi,
            "existingJyutping": None,
            "meaning": None,
            "examples": examples,
            "sourceFile": None,
            "frequency": count,
            "entryType": "candidate_unknown_chunk",
        })

    return out

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


def build_custom_id(index: int) -> str:
    return f"req-{index:06d}"

def normalize_hanzi(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip())

def load_seed_and_candidate_entries(
    files: list[Path],
    known_hanzi: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extracted: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []

    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": str(e),
            })
            continue

        if not isinstance(data, dict):
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": "Top-level JSON is not an object",
            })
            continue

        reduced = reduce_word_file(data, file_path)
        if reduced is None:
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": "Missing usable id or word field",
            })
        else:
            extracted.append(reduced)

        candidate_entries = extract_candidate_entries_from_examples(
            data=data,
            file_path=file_path,
            known_hanzi=known_hanzi,
        )
        extracted.extend(candidate_entries)

    return extracted, parse_errors


def extract_candidate_entries_from_examples(
    data: dict[str, Any],
    file_path: Path,
    known_hanzi: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    raw_examples = data.get("examples", [])
    if not isinstance(raw_examples, list):
        return out

    for ex in raw_examples:
        if not isinstance(ex, dict):
            continue

        sentence = normalize_hanzi(ex.get("sentence"))
        if not sentence:
            continue

        ex_jyutping = normalize_jyutping(ex.get("jyutping"))

        unknown_chunks = find_unknown_chunks(sentence, known_hanzi)

        for chunk in unknown_chunks:
            out.append({
                "id": f"auto-{NON_ID_SAFE_RE.sub('_', chunk)}",
                "hanzi": chunk,
                "existingJyutping": None,
                "meaning": None,
                "examples": [
                    {
                        "sentence": sentence,
                        "jyutping": ex_jyutping,
                    }
                ],
                "sourceFile": str(file_path),
            })

    return out


def normalize_jyutping(value: str | None) -> str:
    if not value:
        return ""
    v = value.strip().lower()
    v = WHITESPACE_RE.sub(" ", v)
    return v


def is_plausible_jyutping(value: str | None) -> bool:
    if not value:
        return False
    v = normalize_jyutping(value)
    return bool(v) and bool(JYUTPING_ALLOWED_RE.fullmatch(v))


def extract_examples(data: dict[str, Any]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    raw_examples = data.get("examples", [])
    if not isinstance(raw_examples, list):
        return examples

    for ex in raw_examples:
        if not isinstance(ex, dict):
            continue

        sentence = normalize_hanzi(ex.get("sentence"))
        jyutping = normalize_jyutping(ex.get("jyutping"))

        if not sentence:
            continue

        examples.append({
            "sentence": sentence,
            "jyutping": jyutping,
        })

        if len(examples) >= MAX_EXAMPLES_PER_ENTRY:
            break

    return examples


def reduce_word_file(data: dict[str, Any], file_path: Path) -> dict[str, Any] | None:
    entry_id = data.get("id")
    hanzi = normalize_hanzi(data.get("word"))
    existing_jyutping = normalize_jyutping(data.get("jyutping"))
    meaning = data.get("meaning")

    if not isinstance(entry_id, str) or not entry_id.strip():
        return None

    if not hanzi:
        return None

    reduced = {
        "id": entry_id.strip(),
        "hanzi": hanzi,
        "existingJyutping": existing_jyutping or None,
        "meaning": meaning if isinstance(meaning, str) and meaning.strip() else None,
        "examples": extract_examples(data),
        "sourceFile": str(file_path),
    }

    return reduced


def dedupe_entries(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []

    for entry in entries:
        entry_id = entry["id"]
        current = merged.get(entry_id)

        if current is None:
            merged[entry_id] = entry
            continue

        same_hanzi = current["hanzi"] == entry["hanzi"]
        same_existing_jyutping = normalize_jyutping(current.get("existingJyutping")) == normalize_jyutping(entry.get("existingJyutping"))

        is_auto = isinstance(entry_id, str) and entry_id.startswith("auto-")

        if is_auto and same_hanzi and same_existing_jyutping:
            current_examples = current.get("examples", []) or []
            incoming_examples = entry.get("examples", []) or []

            seen_example_keys = {
                (ex.get("sentence"), ex.get("jyutping"))
                for ex in current_examples
                if isinstance(ex, dict)
            }

            for ex in incoming_examples:
                if not isinstance(ex, dict):
                    continue
                key = (ex.get("sentence"), ex.get("jyutping"))
                if key not in seen_example_keys:
                    current_examples.append(ex)
                    seen_example_keys.add(key)

            current["examples"] = current_examples[:MAX_EXAMPLES_PER_ENTRY]

            current_sources = current.get("sourceFiles", [])
            if not current_sources:
                existing_source = current.get("sourceFile")
                if existing_source:
                    current_sources = [existing_source]

            incoming_source = entry.get("sourceFile")
            if incoming_source and incoming_source not in current_sources:
                current_sources.append(incoming_source)

            current["sourceFiles"] = current_sources[:5]
            continue

        if not same_hanzi or not same_existing_jyutping:
            conflicts.append({
                "id": entry_id,
                "type": "duplicate_id_conflict",
                "existing": current,
                "incoming": entry,
            })

    return list(merged.values()), conflicts

def chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def extract_output_text_from_response_body(body: dict[str, Any]) -> str:
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


def parse_model_output_text(output_text: str) -> dict[str, Any]:
    if not output_text.strip():
        raise ValueError("Empty model output_text")
    return json.loads(output_text)


def build_batch_request_body(chunk: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "entries": [
            {
                "id": entry["id"],
                "hanzi": entry["hanzi"],
                "existingJyutping": entry.get("existingJyutping"),
                "meaning": entry.get("meaning"),
                "examples": entry.get("examples", []),
            }
            for entry in chunk
        ]
    }

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
                "name": "candidate_lexicon_build",
                "strict": True,
                "schema": LEXICON_SCHEMA,
            }
        },
    }

# ----------------------------
# Seed extraction
# ----------------------------

def load_seed_entries(files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extracted: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []

    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": str(e),
            })
            continue

        if not isinstance(data, dict):
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": "Top-level JSON is not an object",
            })
            continue

        reduced = reduce_word_file(data, file_path)
        if reduced is None:
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": "Missing usable id or word field",
            })
            continue

        extracted.append(reduced)

    return extracted, parse_errors


# ----------------------------
# Batch build / submit
# ----------------------------

def prepare_batch_files(entries: list[dict[str, Any]], parse_errors: list[dict[str, Any]], dedupe_conflicts: list[dict[str, Any]]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "batch-manifest.json"
    batch_input_path = OUTPUT_DIR / "batch-input.jsonl"
    seed_path = OUTPUT_DIR / "seed-entries.json"
    parse_errors_path = OUTPUT_DIR / "seed-parse-errors.json"
    dedupe_conflicts_path = OUTPUT_DIR / "seed-dedupe-conflicts.json"

    seed_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    parse_errors_path.write_text(json.dumps(parse_errors, ensure_ascii=False, indent=2), encoding="utf-8")
    dedupe_conflicts_path.write_text(json.dumps(dedupe_conflicts, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, dict[str, Any]] = {}
    chunks = chunked(entries, CHUNK_SIZE)
    lines_written = 0

    with batch_input_path.open("w", encoding="utf-8") as out:
        for index, chunk in enumerate(chunks, start=1):
            custom_id = build_custom_id(index)

            manifest[custom_id] = {
                "status": "queued",
                "entryIds": [entry["id"] for entry in chunk],
                "chunk": chunk,
            }

            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": build_batch_request_body(chunk),
            }

            out.write(json.dumps(task, ensure_ascii=False) + "\n")
            lines_written += 1

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared batch JSONL: {batch_input_path}")
    print(f"Prepared manifest: {manifest_path}")
    print(f"Prepared seed entries: {seed_path}")
    print(f"Queued requests: {lines_written}")
    print(f"Entries: {len(entries)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Seed parse errors: {len(parse_errors)}")
    print(f"Seed duplicate-id conflicts: {len(dedupe_conflicts)}")

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
            "job": "candidate-lexicon-build",
            "model": OPENAI_MODEL,
            "chunk_size": str(CHUNK_SIZE),
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


def print_batch_status(batch_id: str) -> dict[str, Any]:
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


def wait_for_batch(batch_id: str) -> dict[str, Any]:
    while True:
        status_payload = print_batch_status(batch_id)
        status = status_payload["status"]

        if status in {"completed", "failed", "expired", "cancelled"}:
            return status_payload

        time.sleep(BATCH_STATUS_POLL_SECONDS)


def download_file_content(file_id: str, destination: Path) -> None:
    content = client.files.content(file_id).content
    destination.write_bytes(content)


def validate_model_entry(model_entry: dict[str, Any], seed_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entry_id = model_entry.get("id")
    hanzi = normalize_hanzi(model_entry.get("hanzi"))
    decision = model_entry.get("decision")
    jyutping = normalize_jyutping(model_entry.get("jyutping"))
    reason = model_entry.get("reason") or ""
    confidence = model_entry.get("confidence")

    if not isinstance(entry_id, str) or entry_id not in seed_lookup:
        raise ValueError(f"Unknown or invalid id returned by model: {entry_id!r}")

    if not isinstance(hanzi, str) or not hanzi:
        raise ValueError(f"Missing hanzi for id={entry_id}")

    seed = seed_lookup[entry_id]
    seed_hanzi = normalize_hanzi(seed["hanzi"])
    if hanzi != seed_hanzi:
        raise ValueError(f"Hanzi mismatch for id={entry_id}: model={hanzi!r}, seed={seed_hanzi!r}")

    if decision not in {
        "approved_existing",
        "approved_corrected",
        "approved_generated",
        "needs_manual_review",
    }:
        raise ValueError(f"Invalid decision for id={entry_id}: {decision!r}")

    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Invalid confidence for id={entry_id}")

    if decision != "needs_manual_review":
        if not is_plausible_jyutping(jyutping):
            raise ValueError(f"Invalid or missing jyutping for approved id={entry_id}: {jyutping!r}")

    return {
        "id": entry_id,
        "hanzi": seed_hanzi,
        "existingJyutping": seed.get("existingJyutping"),
        "jyutping": jyutping or None,
        "decision": decision,
        "confidence": float(confidence),
        "reason": str(reason),
        "meaning": seed.get("meaning"),
        "sourceFile": seed.get("sourceFile"),
        "examples": seed.get("examples", []),
    }

def load_existing_lexicon_map() -> dict[str, dict[str, str]]:
    path = OUTPUT_DIR / "lexicon-map.json"
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}

    existing: dict[str, dict[str, str]] = {}

    for entry_id, value in raw.items():
        if (
            isinstance(entry_id, str)
            and isinstance(value, dict)
            and isinstance(value.get("hanzi"), str)
            and isinstance(value.get("jyutping"), str)
        ):
            existing[entry_id] = {
                "hanzi": value["hanzi"],
                "jyutping": value["jyutping"],
            }

    return existing


def load_overrides() -> dict[str, dict[str, str]]:
    path = OUTPUT_DIR / "lexicon-overrides.json"
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}

    overrides: dict[str, dict[str, str]] = {}

    for entry_id, value in raw.items():
        if (
            isinstance(entry_id, str)
            and isinstance(value, dict)
            and isinstance(value.get("hanzi"), str)
            and isinstance(value.get("jyutping"), str)
        ):
            overrides[entry_id] = {
                "hanzi": value["hanzi"],
                "jyutping": value["jyutping"],
            }

    return overrides


def compile_outputs(
    final_entries: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    api_errors: list[dict[str, Any]],
    dedupe_conflicts: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_map = load_existing_lexicon_map() if MERGE_EXISTING else {}
    overrides = load_overrides()

    approved_current = {
        e["id"]: {
            "hanzi": e["hanzi"],
            "jyutping": e["jyutping"],
        }
        for e in final_entries
        if e["decision"] in {"approved_existing", "approved_corrected", "approved_generated"}
        and e.get("jyutping")
    }

    merged_map: dict[str, dict[str, str]] = {}

    # 1. start with old entries
    merged_map.update(existing_map)

    # 2. overlay current run
    merged_map.update(approved_current)

    # 3. overlay manual overrides last
    merged_map.update(overrides)

    runtime_min = [
        {
            "id": entry_id,
            "hanzi": value["hanzi"],
            "jyutping": value["jyutping"],
        }
        for entry_id, value in sorted(merged_map.items())
    ]

    (OUTPUT_DIR / "lexicon-min.json").write_text(
        json.dumps(runtime_min, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "lexicon-map.json").write_text(
        json.dumps(merged_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "rootDir": str(ROOT_DIR),
        "model": OPENAI_MODEL,
        "currentRunEntries": len(final_entries),
        "approvedCurrentRunEntries": len(approved_current),
        "mergedTotalEntries": len(merged_map),
        "parseErrors": len(parse_errors),
        "apiErrors": len(api_errors),
        "dedupeConflicts": len(dedupe_conflicts),
        "overrideEntries": len(overrides),
        "existingEntriesBeforeMerge": len(existing_map),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCompile complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

from collections import Counter

def collect_candidate_chunk_counts(
    files: list[Path],
    known_hanzi: set[str],
) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    parse_errors: list[dict[str, Any]] = []

    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": str(e),
            })
            continue

        if not isinstance(data, dict):
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": "Top-level JSON is not an object",
            })
            continue

        raw_examples = data.get("examples", [])
        if not isinstance(raw_examples, list):
            continue

        for ex in raw_examples:
            if not isinstance(ex, dict):
                continue

            sentence = normalize_hanzi(ex.get("sentence"))
            if not sentence:
                continue

            unknown_chunks = find_unknown_chunks(sentence, known_hanzi)
            for chunk in unknown_chunks:
                counts[chunk] += 1

    return counts, parse_errors

import hashlib

def build_auto_candidate_id(chunk: str) -> str:
    digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
    return f"auto-{digest}"

from collections import Counter, defaultdict

def collect_candidate_chunk_data(
    files: list[Path],
    known_hanzi: set[str],
) -> tuple[Counter[str], dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    examples_by_chunk: dict[str, list[dict[str, str]]] = defaultdict(list)
    parse_errors: list[dict[str, Any]] = []

    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": str(e),
            })
            continue

        if not isinstance(data, dict):
            parse_errors.append({
                "file": str(file_path),
                "status": "parse_error",
                "error": "Top-level JSON is not an object",
            })
            continue

        raw_examples = data.get("examples", [])
        if not isinstance(raw_examples, list):
            continue

        for ex in raw_examples:
            if not isinstance(ex, dict):
                continue

            sentence = normalize_hanzi(ex.get("sentence"))
            if not sentence:
                continue

            ex_jyutping = normalize_jyutping(ex.get("jyutping"))
            unknown_chunks = find_unknown_chunks(sentence, known_hanzi)

            for chunk in unknown_chunks:
                counts[chunk] += 1

                existing = examples_by_chunk[chunk]
                key = (sentence, ex_jyutping)

                already = {
                    (item.get("sentence"), item.get("jyutping"))
                    for item in existing
                }

                if key not in already and len(existing) < MAX_EXAMPLES_PER_ENTRY:
                    existing.append({
                        "sentence": sentence,
                        "jyutping": ex_jyutping,
                    })

    return counts, examples_by_chunk, parse_errors

def build_candidate_entries_from_counts(
    counts: Counter[str],
    examples_by_chunk: dict[str, list[dict[str, str]]],
    min_frequency: int = 5,
    max_chunk_len: int = 2,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for chunk, freq in counts.most_common():
        if freq < min_frequency:
            continue
        if len(chunk) > max_chunk_len:
            continue

        out.append({
            "id": build_auto_candidate_id(chunk),
            "hanzi": chunk,
            "existingJyutping": None,
            "meaning": None,
            "examples": examples_by_chunk.get(chunk, []),
            "sourceFile": None,
            "frequency": freq,
        })

    return out

def collect_results(batch_id: str) -> None:
    batch = client.batches.retrieve(batch_id)

    manifest_path = OUTPUT_DIR / "batch-manifest.json"
    parse_errors_path = OUTPUT_DIR / "seed-parse-errors.json"
    dedupe_conflicts_path = OUTPUT_DIR / "seed-dedupe-conflicts.json"

    if not manifest_path.exists():
        raise RuntimeError("Missing batch-manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parse_errors = json.loads(parse_errors_path.read_text(encoding="utf-8")) if parse_errors_path.exists() else []
    dedupe_conflicts = json.loads(dedupe_conflicts_path.read_text(encoding="utf-8")) if dedupe_conflicts_path.exists() else []

    results_file_path = OUTPUT_DIR / "batch-results.jsonl"
    errors_file_path = OUTPUT_DIR / "batch-errors.jsonl"

    if batch.output_file_id:
        download_file_content(batch.output_file_id, results_file_path)
        print(f"Downloaded output file -> {results_file_path}")

    if batch.error_file_id:
        download_file_content(batch.error_file_id, errors_file_path)
        print(f"Downloaded error file -> {errors_file_path}")

    seed_lookup: dict[str, dict[str, Any]] = {}
    for item in manifest.values():
        for chunk_entry in item.get("chunk", []):
            seed_lookup[chunk_entry["id"]] = chunk_entry

    final_entries_by_id: dict[str, dict[str, Any]] = {}
    api_errors: list[dict[str, Any]] = []
    seen_custom_ids: set[str] = set()

    if results_file_path.exists():
        with results_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                seen_custom_ids.add(custom_id)

                manifest_item = manifest.get(custom_id)
                if not manifest_item:
                    api_errors.append({
                        "custom_id": custom_id,
                        "error": "Unknown custom_id in output file",
                    })
                    continue

                if row.get("error"):
                    api_errors.append({
                        "custom_id": custom_id,
                        "error": row["error"].get("message", "Unknown batch error"),
                        "entryIds": manifest_item.get("entryIds", []),
                    })
                    continue

                response = row.get("response") or {}
                body = response.get("body") or {}

                try:
                    output_text = extract_output_text_from_response_body(body)
                    parsed = parse_model_output_text(output_text)
                    returned_entries = parsed.get("entries", [])

                    if not isinstance(returned_entries, list):
                        raise ValueError("Model output does not contain 'entries' array")

                    for model_entry in returned_entries:
                        validated = validate_model_entry(model_entry, seed_lookup)
                        final_entries_by_id[validated["id"]] = validated

                except Exception as e:
                    api_errors.append({
                        "custom_id": custom_id,
                        "error": f"Output parse/validation failed: {e}",
                        "entryIds": manifest_item.get("entryIds", []),
                    })

    if errors_file_path.exists():
        with errors_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if custom_id in seen_custom_ids:
                    continue

                manifest_item = manifest.get(custom_id)
                api_errors.append({
                    "custom_id": custom_id,
                    "error": (row.get("error") or {}).get("message", "Unknown batch error"),
                    "entryIds": manifest_item.get("entryIds", []) if manifest_item else [],
                })

    # Add any missing entries as manual review placeholders
    for seed_id, seed in seed_lookup.items():
        if seed_id in final_entries_by_id:
            continue

        final_entries_by_id[seed_id] = {
            "id": seed_id,
            "hanzi": seed["hanzi"],
            "existingJyutping": seed.get("existingJyutping"),
            "jyutping": None,
            "decision": "needs_manual_review",
            "confidence": 0.0,
            "reason": "No model output found for this entry",
            "meaning": seed.get("meaning"),
            "sourceFile": seed.get("sourceFile"),
            "examples": seed.get("examples", []),
        }

    final_entries = sorted(final_entries_by_id.values(), key=lambda e: e["id"])

    compile_outputs(
        final_entries=final_entries,
        parse_errors=parse_errors,
        api_errors=api_errors,
        dedupe_conflicts=dedupe_conflicts,
    )


# ----------------------------
# Commands
# ----------------------------

# def command_submit() -> None:
#     print("Preparing candidate lexicon Batch API input...")
#     print(f"Using ROOT_DIR: {ROOT_DIR}")

#     files = get_json_files(ROOT_DIR)[:MAX_FILES]
#     if not files:
#         print(f"No JSON files found under: {ROOT_DIR}")
#         return

#     print(f"Found {len(files)} JSON files")
#     print(f"Using model: {OPENAI_MODEL}")
#     print(f"Output dir: {OUTPUT_DIR}")

#     known_hanzi = load_known_hanzi_from_lexicon_map(OUTPUT_DIR / "lexicon-map.json")


#     seed_entries, seed_parse_errors = load_seed_entries(files)
#     chunk_counts, examples_by_chunk, chunk_parse_errors = collect_candidate_chunk_data(files, known_hanzi)

#     candidate_entries = build_candidate_entries_from_counts(
#         counts=chunk_counts,
#         examples_by_chunk=examples_by_chunk,
#         min_frequency=MIN_CANDIDATE_FREQUENCY,
#         max_chunk_len=MAX_CANDIDATE_CHUNK_LEN,
#     )

#     extracted = seed_entries + candidate_entries
#     parse_errors = seed_parse_errors + chunk_parse_errors

#     deduped, dedupe_conflicts = dedupe_entries(extracted)

#     print(f"Loaded known lexicon hanzi: {len(known_hanzi)}")
#     print(f"Seed entries: {len(seed_entries)}")
#     print(f"Unique unknown chunks counted: {len(chunk_counts)}")
#     print(f"Candidate entries after frequency filter: {len(candidate_entries)}")
#     print(f"Combined entries before dedupe: {len(extracted)}")
#     print(f"Deduped entries: {len(deduped)}")
#     print(f"Parse errors: {len(parse_errors)}")
#     print(f"Duplicate-id conflicts: {len(dedupe_conflicts)}")

#     batch_input_path, _manifest_path = prepare_batch_files(
#         deduped,
#         parse_errors,
#         dedupe_conflicts,
#     )
#     batch_id = submit_batch(batch_input_path)

#     print("\nSubmitted.")
#     print(f"Batch ID: {batch_id}")
#     print(f"Check status with: python {Path(__file__).name} status {batch_id}")
#     print(f"Collect with:      python {Path(__file__).name} collect {batch_id}")

def command_submit() -> None:
    print("Preparing candidate lexicon Batch API input...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")

    files = get_json_files(ROOT_DIR)[:MAX_FILES]
    if not files:
        print(f"No JSON files found under: {ROOT_DIR}")
        return

    print(f"Found {len(files)} JSON files")
    print(f"Using model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Unknown report path: {UNKNOWN_REPORT_PATH}")

    seed_entries, seed_parse_errors = load_seed_entries(files)

    candidate_entries = build_candidate_entries_from_unknown_report(
        report_path=UNKNOWN_REPORT_PATH,
        min_frequency=MIN_CANDIDATE_FREQUENCY,
        min_len=MIN_CANDIDATE_CHUNK_LEN,
        max_len=MAX_CANDIDATE_CHUNK_LEN,
        max_examples_per_entry=MAX_EXAMPLES_PER_ENTRY,
    )

    extracted = seed_entries + candidate_entries
    parse_errors = seed_parse_errors

    deduped, dedupe_conflicts = dedupe_entries(extracted)

    print(f"Seed entries: {len(seed_entries)}")
    print(f"Candidate entries from Scala report: {len(candidate_entries)}")
    print(f"Combined entries before dedupe: {len(extracted)}")
    print(f"Deduped entries: {len(deduped)}")
    print(f"Parse errors: {len(parse_errors)}")
    print(f"Duplicate-id conflicts: {len(dedupe_conflicts)}")
    print(f"Candidate filters: min_frequency={MIN_CANDIDATE_FREQUENCY}, min_len={MIN_CANDIDATE_CHUNK_LEN}, max_len={MAX_CANDIDATE_CHUNK_LEN}")

    batch_input_path, _manifest_path = prepare_batch_files(
        deduped,
        parse_errors,
        dedupe_conflicts,
    )
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


# def command_run() -> None:
#     print("Preparing candidate lexicon Batch API input...")
#     print(f"Using ROOT_DIR: {ROOT_DIR}")

#     files = get_json_files(ROOT_DIR)[:MAX_FILES]
#     if not files:
#         print(f"No JSON files found under: {ROOT_DIR}")
#         return

#     print(f"Found {len(files)} JSON files")
#     print(f"Using model: {OPENAI_MODEL}")
#     print(f"Output dir: {OUTPUT_DIR}")

#     known_hanzi = load_known_hanzi_from_lexicon_map(OUTPUT_DIR / "lexicon-map.json")

#     seed_entries, seed_parse_errors = load_seed_entries(files)
#     chunk_counts, examples_by_chunk, chunk_parse_errors = collect_candidate_chunk_data(files, known_hanzi)

#     candidate_entries = build_candidate_entries_from_counts(
#         counts=chunk_counts,
#         examples_by_chunk=examples_by_chunk,
#         min_frequency=5,
#         max_chunk_len=2,
#     )

#     extracted = seed_entries + candidate_entries
#     parse_errors = seed_parse_errors + chunk_parse_errors

#     deduped, dedupe_conflicts = dedupe_entries(extracted)

#     print(f"Loaded known lexicon hanzi: {len(known_hanzi)}")
#     print(f"Seed entries: {len(seed_entries)}")
#     print(f"Unique unknown chunks counted: {len(chunk_counts)}")
#     print(f"Candidate entries after frequency filter: {len(candidate_entries)}")
#     print(f"Combined entries before dedupe: {len(extracted)}")
#     print(f"Deduped entries: {len(deduped)}")
#     print(f"Parse errors: {len(parse_errors)}")
#     print(f"Duplicate-id conflicts: {len(dedupe_conflicts)}")

#     batch_input_path, _manifest_path = prepare_batch_files(
#         deduped,
#         parse_errors,
#         dedupe_conflicts,
#     )
#     batch_id = submit_batch(batch_input_path)

#     print("\nWaiting for batch to finish...")
#     final_status = wait_for_batch(batch_id)

#     print("\nBatch finished:")
#     print(json.dumps(final_status, ensure_ascii=False, indent=2))

#     if final_status["status"] in {"completed", "expired"}:
#         print("\nCollecting available results...")
#         collect_results(batch_id)
#     else:
#         print(f"\nNot collecting because batch status is {final_status['status']}.")

def command_run() -> None:
    print("Preparing candidate lexicon Batch API input...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")

    files = get_json_files(ROOT_DIR)[:MAX_FILES]
    if not files:
        print(f"No JSON files found under: {ROOT_DIR}")
        return

    print(f"Found {len(files)} JSON files")
    print(f"Using model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Unknown report path: {UNKNOWN_REPORT_PATH}")

    seed_entries, seed_parse_errors = load_seed_entries(files)

    candidate_entries = build_candidate_entries_from_unknown_report(
        report_path=UNKNOWN_REPORT_PATH,
        min_frequency=MIN_CANDIDATE_FREQUENCY,
        min_len=MIN_CANDIDATE_CHUNK_LEN,
        max_len=MAX_CANDIDATE_CHUNK_LEN,
        max_examples_per_entry=MAX_EXAMPLES_PER_ENTRY,
    )

    extracted = seed_entries + candidate_entries
    parse_errors = seed_parse_errors

    deduped, dedupe_conflicts = dedupe_entries(extracted)

    print(f"Seed entries: {len(seed_entries)}")
    print(f"Candidate entries from Scala report: {len(candidate_entries)}")
    print(f"Combined entries before dedupe: {len(extracted)}")
    print(f"Deduped entries: {len(deduped)}")
    print(f"Parse errors: {len(parse_errors)}")
    print(f"Duplicate-id conflicts: {len(dedupe_conflicts)}")
    print(f"Candidate filters: min_frequency={MIN_CANDIDATE_FREQUENCY}, min_len={MIN_CANDIDATE_CHUNK_LEN}, max_len={MAX_CANDIDATE_CHUNK_LEN}")

    batch_input_path, _manifest_path = prepare_batch_files(
        deduped,
        parse_errors,
        dedupe_conflicts,
    )
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

def command_compile_local() -> None:
    """
    Re-compile from existing collected outputs without re-running submit.
    Useful after manual edits to output artifacts.
    """
    submission_path = OUTPUT_DIR / "batch-submission.json"
    if not submission_path.exists():
        raise RuntimeError("Missing batch-submission.json")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    batch_id = submission["batch_id"]
    collect_results(batch_id)

def load_known_hanzi_from_lexicon_map(path: Path) -> set[str]:
    if not path.exists():
        return set()

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return set()

    out: set[str] = set()

    for value in raw.values():
        if isinstance(value, dict):
            hanzi = normalize_hanzi(value.get("hanzi"))
            if hanzi:
                out.add(hanzi)

    return out


PUNCTUATION_CHARS = set(" \t\r\n。，“”‘’！？、：；,.!?:;()（）[]【】《》<>\"'")


def is_punctuation_char(ch: str) -> bool:
    return ch in PUNCTUATION_CHARS


def segment_known_unknown(sentence: str, known_hanzi: set[str]) -> list[dict[str, Any]]:
    """
    Longest-match segmentation against known lexicon entries.
    Returns tokens like:
      { "type": "known", "text": "健身房" }
      { "type": "unknown", "text": "舉" }
      { "type": "punct", "text": "。" }
    """
    if not sentence:
        return []

    sorted_known = sorted(known_hanzi, key=len, reverse=True)

    i = 0
    tokens: list[dict[str, Any]] = []

    while i < len(sentence):
        ch = sentence[i]

        if is_punctuation_char(ch):
            tokens.append({"type": "punct", "text": ch})
            i += 1
            continue

        match = None
        for candidate in sorted_known:
            if sentence.startswith(candidate, i):
                match = candidate
                break

        if match is not None:
            tokens.append({"type": "known", "text": match})
            i += len(match)
        else:
            tokens.append({"type": "unknown", "text": ch})
            i += 1

    return tokens


def collapse_unknown_tokens(tokens: list[dict[str, Any]]) -> list[str]:
    """
    Merge adjacent unknown characters into chunks.
    Example:
      我 / 喺 / 新屋 / 買 / 咗 / 張 / 床 / 。
      if 新屋 unknown as chars 新 + 屋 => returns ["新屋"]
    """
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []

    for token in tokens:
        token_type = token["type"]
        text = token["text"]

        if token_type == "unknown":
            current.append(text)
        else:
            flush()

    flush()
    return chunks

def find_unknown_chunks(sentence: str, known_hanzi: set[str]) -> list[str]:
    sentence = normalize_hanzi(sentence)
    if not sentence:
        return []

    tokens = segment_known_unknown(sentence, known_hanzi)
    chunks = collapse_unknown_tokens(tokens)

    filtered: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        if len(chunk) > 4:
            continue
        if re.search(r"[A-Za-z0-9]", chunk):
            continue
        filtered.append(chunk)

    seen: set[str] = set()
    out: list[str] = []
    for chunk in filtered:
        if chunk not in seen:
            seen.add(chunk)
            out.append(chunk)

    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {Path(__file__).name} submit")
        print(f"  python {Path(__file__).name} status [batch_id]")
        print(f"  python {Path(__file__).name} wait [batch_id]")
        print(f"  python {Path(__file__).name} collect [batch_id]")
        print(f"  python {Path(__file__).name} compile-local")
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
    elif command == "compile-local":
        command_compile_local()
    elif command == "run":
        command_run()
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()