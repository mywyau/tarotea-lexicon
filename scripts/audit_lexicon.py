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
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./lexicon/audit-output")).resolve()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "120"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "0.2"))

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Prompt + Schema
# ----------------------------
SYSTEM_PROMPT = (
    "You are auditing a Cantonese learner lexicon.\n\n"
    "You will receive a batch of lexicon entries. Each entry includes:\n"
    "- id\n"
    "- hanzi\n"
    "- existingJyutping\n\n"
    "Your task:\n"
    "Review each entry and decide whether the current jyutping looks acceptable.\n\n"
    "Rules:\n"
    "- Focus on standard Hong Kong Cantonese learner-friendly Jyutping.\n"
    "- Use lowercase letters, tones 1-6, and spaces between syllables.\n"
    "- If the existing jyutping looks fine, keep it.\n"
    "- If it looks wrong and you are reasonably confident, suggest a corrected jyutping.\n"
    "- If the entry is not meaningful Chinese lexicon content (e.g. Latin letters, punctuation, corrupted text), mark it non_chinese.\n"
    "- If the entry is Chinese but too ambiguous or uncertain, mark it needs_manual_review.\n"
    "- Do not invent meanings.\n"
    "- Return strict JSON only.\n\n"
    "Statuses:\n"
    "- ok\n"
    "- suspicious\n"
    "- duplicate_conflict\n"
    "- non_chinese\n"
    "- needs_manual_review\n"
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
                            "ok",
                            "suspicious",
                            "duplicate_conflict",
                            "non_chinese",
                            "needs_manual_review",
                        ],
                    },
                    "suggestedJyutping": {
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
                    "suggestedJyutping",
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


def extract_output_text_from_response(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "output_text":
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)

    return "".join(chunks).strip()


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


def audit_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = build_request_payload(chunk)

    response = client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "lexicon_audit",
                "strict": True,
                "schema": AUDIT_SCHEMA,
            }
        },
    )

    output_text = extract_output_text_from_response(response)
    if not output_text:
        raise RuntimeError("Empty model response")

    parsed = json.loads(output_text)
    returned_entries = parsed.get("entries")
    if not isinstance(returned_entries, list):
        raise RuntimeError("Model response missing entries array")

    return returned_entries


def prefilter_obvious_noise(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                "suggestedJyutping": None,
                "reason": "No CJK Hanzi detected in entry",
                "confidence": 1.0,
            })
            continue

        clean.append(entry)

    return clean, obvious_noise


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

    # Reserve existing non-auto ids first
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
        suggested = normalize_jyutping(audit.get("suggestedJyutping"))
        final_jyutping = suggested or existing_jyutping

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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lexicon from: {INPUT_PATH}")
    entries = load_entries(INPUT_PATH)
    print(f"Loaded entries: {len(entries)}")

    clean_entries, obvious_noise = prefilter_obvious_noise(entries)
    print(f"Prefiltered obvious non-Chinese entries: {len(obvious_noise)}")
    print(f"Entries to audit with model: {len(clean_entries)}")

    all_results: list[dict[str, Any]] = []
    all_results.extend(obvious_noise)

    chunks = chunked(clean_entries, CHUNK_SIZE)
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total chunks: {len(chunks)}")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Auditing chunk {idx}/{len(chunks)} ({len(chunk)} entries)")
        try:
            result_entries = audit_chunk(chunk)
            all_results.extend(result_entries)
        except Exception as e:
            for entry in chunk:
                all_results.append({
                    "id": entry["id"],
                    "hanzi": entry["hanzi"],
                    "existingJyutping": entry.get("existingJyutping"),
                    "status": "needs_manual_review",
                    "suggestedJyutping": None,
                    "reason": f"Chunk audit failed: {e}",
                    "confidence": 0.0,
                })
        time.sleep(SLEEP_SECONDS)

    ok_entries = [e for e in all_results if e["status"] == "ok"]
    needs_review_entries = [e for e in all_results if e["status"] != "ok"]

    summary = {
        "inputPath": str(INPUT_PATH),
        "model": OPENAI_MODEL,
        "totalEntries": len(entries),
        "auditedEntries": len(all_results),
        "okCount": len(ok_entries),
        "needsReviewCount": len(needs_review_entries),
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
    (OUTPUT_DIR / "audit-ok.json").write_text(
        json.dumps(ok_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "audit-needs-review.json").write_text(
        json.dumps(needs_review_entries, ensure_ascii=False, indent=2),
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