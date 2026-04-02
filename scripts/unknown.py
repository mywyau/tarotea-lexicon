import hashlib
import json
import re
from pathlib import Path
from typing import Any

LATIN_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9]")

def normalize_hanzi(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip())

def normalize_jyutping(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())

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
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("unknown-chunk-report.json must contain a top-level array")
    return raw

def build_candidate_entries_from_unknown_report(
    report_path: Path,
    *,
    min_frequency: int = 2,
    min_len: int = 1,
    max_len: int = 2,
    max_examples_per_entry: int = 2,
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

                # Scala report currently includes file + sentence only.
                # If jyutping is absent, keep it empty.
                key = (sentence, jyutping)

                if not sentence:
                    continue
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