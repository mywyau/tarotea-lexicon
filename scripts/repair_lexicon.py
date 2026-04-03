import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
INPUT_PATH = Path(os.getenv("INPUT_PATH", "./lexicon/lexicon-build/lexicon-map.json")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./lexicon/repair-output")).resolve()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "12000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "120"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "0.2"))

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Prompt + Schema
# ----------------------------
SYSTEM_PROMPT = (
    "You are repairing a Cantonese learner lexicon.\n\n"
    "You will receive lexicon entries with:\n"
    "- id\n"
    "- hanzi\n"
    "- existingJyutping\n"
    "- existingMeaning\n\n"
    "Your task for each entry:\n"
    "1. Decide whether the current jyutping is acceptable Hong Kong Cantonese learner-friendly Jyutping.\n"
    "2. If needed, suggest a corrected jyutping.\n"
    "3. Add a short learner-friendly English meaning.\n"
    "4. Flag entries that are not valid Chinese lexicon entries or are too uncertain.\n\n"
    "Rules:\n"
    "- Use standard Hong Kong Cantonese learner-friendly Jyutping.\n"
    "- Use lowercase letters, tones 1-6, and spaces between syllables.\n"
    "- Meanings should be short, natural, and dictionary-like.\n"
    "- Prefer a core meaning, not a long explanation.\n"
    "- Do not invent rare or speculative meanings if uncertain.\n"
    "- If the entry is not meaningful Chinese lexicon content, mark it non_chinese.\n"
    "- If the entry is Chinese but too ambiguous or uncertain, mark it needs_manual_review.\n"
    "- If the existing jyutping is fine, keep it.\n"
    "- Return strict JSON only.\n\n"
    "Statuses:\n"
    "- ok\n"
    "- suspicious\n"
    "- duplicate_conflict\n"
    "- non_chinese\n"
    "- needs_manual_review\n"
)

REPAIR_SCHEMA = {
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
                    "existingMeaning": {
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
                    "suggestedMeaning": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "id",
                    "hanzi",
                    "existingJyutping",
                    "existingMeaning",
                    "status",
                    "suggestedJyutping",
                    "suggestedMeaning",
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

CANONICAL_ID_RE = re.compile(
    r"^(?P<jyutping>[a-z]+[1-6](?:-[a-z]+[1-6])*)-(?P<slug>u[0-9a-f]{4,6}(?:-u[0-9a-f]{4,6})*)(?:-(?P<suffix>\d+))?$"
)


def normalize_hanzi(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip())


def normalize_jyutping(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip().lower())


def normalize_meaning(value: str | None) -> str:
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


def hanzi_to_codepoint_slug(hanzi: str) -> str:
    return "-".join(f"u{ord(ch):04x}" for ch in hanzi)


def jyutping_to_id_part(jyutping: str) -> str:
    return normalize_jyutping(jyutping).replace(" ", "-")


def make_canonical_base_id(hanzi: str, jyutping: str) -> str:
    return f"{jyutping_to_id_part(jyutping)}-{hanzi_to_codepoint_slug(hanzi)}"


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


def parse_existing_canonical_id(value: str) -> dict[str, Any] | None:
    m = CANONICAL_ID_RE.match(value)
    if not m:
        return None
    return {
        "jyutping_part": m.group("jyutping"),
        "slug_part": m.group("slug"),
        "suffix": int(m.group("suffix")) if m.group("suffix") else None,
    }


def load_entries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        for entry_id, value in raw.items():
            if not isinstance(entry_id, str) or not isinstance(value, dict):
                continue
            hanzi = normalize_hanzi(value.get("hanzi"))
            jyutping = normalize_jyutping(value.get("jyutping"))
            meaning = normalize_meaning(value.get("meaning"))
            if not hanzi:
                continue
            entries.append({
                "id": entry_id,
                "hanzi": hanzi,
                "existingJyutping": jyutping or None,
                "existingMeaning": meaning or None,
            })
        return entries

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            entry_id = item.get("id")
            hanzi = normalize_hanzi(item.get("hanzi"))
            jyutping = normalize_jyutping(item.get("jyutping"))
            meaning = normalize_meaning(item.get("meaning"))
            if not isinstance(entry_id, str) or not hanzi:
                continue
            entries.append({
                "id": entry_id,
                "hanzi": hanzi,
                "existingJyutping": jyutping or None,
                "existingMeaning": meaning or None,
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
                "existingMeaning": entry.get("existingMeaning"),
            }
            for entry in chunk
        ]
    }


def repair_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "name": "lexicon_repair",
                "strict": True,
                "schema": REPAIR_SCHEMA,
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
        meaning = entry.get("existingMeaning")

        if not is_chinese_like(hanzi):
            obvious_noise.append({
                "id": entry["id"],
                "hanzi": hanzi,
                "existingJyutping": jyutping,
                "existingMeaning": meaning,
                "status": "non_chinese",
                "suggestedJyutping": None,
                "suggestedMeaning": None,
                "reason": "No CJK Hanzi detected in entry",
                "confidence": 1.0,
            })
            continue

        clean.append(entry)

    return clean, obvious_noise


def post_validate_model_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)

    row["hanzi"] = normalize_hanzi(row.get("hanzi"))
    row["existingJyutping"] = normalize_jyutping(row.get("existingJyutping")) or None
    row["suggestedJyutping"] = normalize_jyutping(row.get("suggestedJyutping")) or None
    row["existingMeaning"] = normalize_meaning(row.get("existingMeaning")) or None
    row["suggestedMeaning"] = normalize_meaning(row.get("suggestedMeaning")) or None
    row["reason"] = normalize_meaning(row.get("reason"))
    row["status"] = row.get("status") or "needs_manual_review"

    suggested_jp = row.get("suggestedJyutping")
    if suggested_jp and not is_plausible_jyutping(suggested_jp):
        row["status"] = "needs_manual_review"
        row["reason"] = f"Model returned invalid jyutping format: {suggested_jp}"

    return row


def build_fixed_outputs(
    original_entries: list[dict[str, Any]],
    repair_results: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    results_by_id = {
        row["id"]: post_validate_model_row(row)
        for row in repair_results
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }

    used_ids: set[str] = set()
    fixed_map: dict[str, dict[str, str]] = {}
    fixed_min: list[dict[str, Any]] = []
    remap: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for entry in original_entries:
        old_id = entry["id"]
        hanzi = normalize_hanzi(entry.get("hanzi"))
        existing_jyutping = normalize_jyutping(entry.get("existingJyutping"))
        existing_meaning = normalize_meaning(entry.get("existingMeaning"))

        repair = results_by_id.get(old_id, {})
        status = repair.get("status", "needs_manual_review")
        suggested_jyutping = normalize_jyutping(repair.get("suggestedJyutping"))
        suggested_meaning = normalize_meaning(repair.get("suggestedMeaning"))

        final_jyutping = suggested_jyutping or existing_jyutping
        final_meaning = suggested_meaning or existing_meaning

        if status == "non_chinese":
            unresolved.append({
                "oldId": old_id,
                "hanzi": hanzi,
                "existingJyutping": existing_jyutping or None,
                "existingMeaning": existing_meaning or None,
                "status": status,
                "reason": repair.get("reason"),
                "confidence": repair.get("confidence"),
            })
            continue

        if not hanzi or not final_jyutping or not final_meaning:
            unresolved.append({
                "oldId": old_id,
                "hanzi": hanzi,
                "existingJyutping": existing_jyutping or None,
                "existingMeaning": existing_meaning or None,
                "status": "needs_manual_review",
                "reason": "Missing repaired hanzi, jyutping, or meaning",
                "confidence": repair.get("confidence", 0.0),
            })
            continue

        base_id = make_canonical_base_id(hanzi, final_jyutping)
        new_id = uniquify_id(base_id, used_ids)

        fixed_map[new_id] = {
            "hanzi": hanzi,
            "jyutping": final_jyutping,
            "meaning": final_meaning,
        }

        fixed_min.append({
            "id": new_id,
            "hanzi": hanzi,
            "jyutping": final_jyutping,
            "meaning": final_meaning,
        })

        remap.append({
            "oldId": old_id,
            "newId": new_id,
            "hanzi": hanzi,
            "jyutping": final_jyutping,
            "meaning": final_meaning,
            "status": status,
            "reason": repair.get("reason"),
            "confidence": repair.get("confidence"),
        })

        if status != "ok":
            unresolved.append({
                "oldId": old_id,
                "newId": new_id,
                "hanzi": hanzi,
                "jyutping": final_jyutping,
                "meaning": final_meaning,
                "status": status,
                "reason": repair.get("reason"),
                "confidence": repair.get("confidence"),
            })

    fixed_min.sort(key=lambda x: x["id"])
    return fixed_map, fixed_min, remap, unresolved


def build_id_diagnostics(remap: list[dict[str, Any]]) -> dict[str, Any]:
    base_counter = Counter()
    by_hanzi: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in remap:
        new_id = row["newId"]
        parsed = parse_existing_canonical_id(new_id)
        if parsed:
            base = f"{parsed['jyutping_part']}-{parsed['slug_part']}"
            base_counter[base] += 1
        by_hanzi[row["hanzi"]].append(row)

    duplicate_bases = [
        {"baseId": base, "count": count}
        for base, count in sorted(base_counter.items())
        if count > 1
    ]

    multi_reading_hanzi = []
    for hanzi, rows in sorted(by_hanzi.items()):
        readings = sorted({row["jyutping"] for row in rows})
        if len(readings) > 1:
            multi_reading_hanzi.append({
                "hanzi": hanzi,
                "readings": readings,
                "ids": [row["newId"] for row in rows],
            })

    return {
        "duplicateBaseIds": duplicate_bases,
        "multiReadingHanzi": multi_reading_hanzi,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading lexicon from: {INPUT_PATH}")
    entries = load_entries(INPUT_PATH)
    print(f"Loaded entries: {len(entries)}")

    clean_entries, obvious_noise = prefilter_obvious_noise(entries)
    print(f"Prefiltered obvious non-Chinese entries: {len(obvious_noise)}")
    print(f"Entries to repair with model: {len(clean_entries)}")

    all_results: list[dict[str, Any]] = []
    all_results.extend(obvious_noise)

    chunks = chunked(clean_entries, CHUNK_SIZE)
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Total chunks: {len(chunks)}")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Repairing chunk {idx}/{len(chunks)} ({len(chunk)} entries)")
        try:
            result_entries = repair_chunk(chunk)
            all_results.extend(result_entries)
        except Exception as e:
            for entry in chunk:
                all_results.append({
                    "id": entry["id"],
                    "hanzi": entry["hanzi"],
                    "existingJyutping": entry.get("existingJyutping"),
                    "existingMeaning": entry.get("existingMeaning"),
                    "status": "needs_manual_review",
                    "suggestedJyutping": None,
                    "suggestedMeaning": None,
                    "reason": f"Chunk repair failed: {e}",
                    "confidence": 0.0,
                })
        time.sleep(SLEEP_SECONDS)

    fixed_lexicon_map, fixed_runtime_min, id_remap, unresolved = build_fixed_outputs(
        entries,
        all_results,
    )

    ok_entries = [e for e in all_results if e["status"] == "ok"]
    non_ok_entries = [e for e in all_results if e["status"] != "ok"]

    diagnostics = build_id_diagnostics(id_remap)

    summary = {
        "inputPath": str(INPUT_PATH),
        "model": OPENAI_MODEL,
        "totalEntries": len(entries),
        "processedEntries": len(all_results),
        "fixedEntries": len(fixed_lexicon_map),
        "okCount": len(ok_entries),
        "nonOkCount": len(non_ok_entries),
        "unresolvedCount": len(unresolved),
        "statusBreakdown": {},
    }

    for entry in all_results:
        status = entry["status"]
        summary["statusBreakdown"][status] = summary["statusBreakdown"].get(status, 0) + 1

    (OUTPUT_DIR / "repair-all.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "repair-ok.json").write_text(
        json.dumps(ok_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "repair-needs-review.json").write_text(
        json.dumps(non_ok_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "repair-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "lexicon-map-repaired.json").write_text(
        json.dumps(fixed_lexicon_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "lexicon-min-repaired.json").write_text(
        json.dumps(fixed_runtime_min, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "id-remap.json").write_text(
        json.dumps(id_remap, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "unresolved.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "id-diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Repaired lexicon entries: {len(fixed_lexicon_map)}")
    print(f"ID remap rows: {len(id_remap)}")
    print(f"Unresolved rows: {len(unresolved)}")


if __name__ == "__main__":
    main()