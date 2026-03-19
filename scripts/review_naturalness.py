import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
ROOT_DIR = Path(os.getenv("ROOT_DIR", "./r2-backup/words"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./reports/example-naturalness-review"))
MAX_FILES = int(os.getenv("MAX_FILES", "10"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "1"))
SKIP_COPY_FILES = os.getenv("SKIP_COPY_FILES", "true").lower() == "true"

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# Helpers
# ----------------------------

def get_json_files(root: Path) -> list[Path]:
    files = []

    for p in root.rglob("*.json"):
        if not p.is_file():
            continue

        if SKIP_COPY_FILES and " copy" in p.stem.lower():
            continue

        files.append(p)

    return sorted(files)


def reduce_payload(data: dict) -> dict:
    return {
        "id": data.get("id"),
        "word": data.get("word"),
        "meaning": data.get("meaning"),
        "examples": [
            {
                "id": ex.get("id"),
                "sentence": ex.get("sentence"),
                "jyutping": ex.get("jyutping"),
                "meaning": ex.get("meaning"),
            }
            for ex in data.get("examples", [])
            if isinstance(ex, dict)
        ],
    }


def review_with_openai(payload: dict) -> dict:
    schema = {
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
                            "enum": ["approved", "needs_manual_review", "likely_unnatural"],
                        },
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "suggestedSentence": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                        "suggestedJyutping": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                        "suggestedMeaning": {
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
                        "suggestedSentence",
                        "suggestedJyutping",
                        "suggestedMeaning",
                    ],
                },
            },
        },
        "required": ["examples"],
    }

    response = client.responses.create(
        model=OPENAI_MODEL,
        reasoning={"effort": "low"},
        max_output_tokens=700,
        input=[
            {
                "role": "system",
                "content": (
                    "You are reviewing Cantonese-learning example sentences. "
                    "Your task is to judge whether each Chinese example sentence sounds natural and appropriate for Cantonese usage.\n\n"
                    "Focus on:\n"
                    "- awkward phrasing\n"
                    "- Mandarin-influenced wording\n"
                    "- overly literal translation feel\n"
                    "- unnatural sentence flow\n"
                    "- mismatch between the Chinese sentence and the Jyutping\n"
                    "- mismatch between the Chinese sentence and the English meaning\n\n"
                    "Rules:\n"
                    "- Only return likely_unnatural if the example is clearly awkward, unnatural, misleading, or suspicious.\n"
                    "- If the example may be acceptable but you are unsure, return needs_manual_review.\n"
                    "- Be conservative about flagging issues.\n"
                    "- Do not nitpick minor stylistic alternatives.\n"
                    "- Prefer natural Cantonese-style phrasing suitable for learners.\n"
                    "- Only suggest rewrites when confidence is reasonably high.\n"
                    "- If the sentence is natural enough, return approved.\n"
                    "- Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "example_naturalness_review",
                "strict": True,
                "schema": schema,
            }
        },
    )

    return json.loads(response.output_text)


def review_file(file_path: Path) -> dict:
    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        return {
            "file": str(file_path),
            "status": "parse_error",
            "summary": "Invalid JSON",
            "error": str(e),
        }

    try:
        reduced = reduce_payload(data)
        review = review_with_openai(reduced)

        issues = []

        example_lookup = {
            ex.get("id"): ex
            for ex in reduced.get("examples", [])
        }

        for ex_review in review["examples"]:
            if ex_review["decision"] != "approved":
                example_id = ex_review.get("id")
                current_example = example_lookup.get(example_id, {})

                issues.append({
                    "path": f"example:{example_id}",
                    "decision": ex_review["decision"],
                    "reason": ex_review["reason"],
                    "currentSentence": current_example.get("sentence", ""),
                    "currentJyutping": current_example.get("jyutping", ""),
                    "currentMeaning": current_example.get("meaning", ""),
                    "suggestedSentence": ex_review["suggestedSentence"] or "",
                    "suggestedJyutping": ex_review["suggestedJyutping"] or "",
                    "suggestedMeaning": ex_review["suggestedMeaning"] or "",
                    "confidence": ex_review["confidence"],
                })

        return {
            "file": str(file_path),
            "status": "ok" if not issues else "needs_review",
            "summary": "No likely naturalness issues found" if not issues else "Potential naturalness issues found",
            "lowConfidence": any(
                issue["decision"] == "needs_manual_review" for issue in issues
            ),
            "issues": issues,
        }

    except Exception as e:
        return {
            "file": str(file_path),
            "status": "api_error",
            "summary": "OpenAI review failed",
            "error": str(e),
        }


def write_text_file(path: Path, lines: list[str]) -> None:
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("Starting example naturalness review...")
    print(f"Using ROOT_DIR: {ROOT_DIR}")

    files = get_json_files(ROOT_DIR)[:MAX_FILES]

    if not files:
        print(f"No JSON files found under: {ROOT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} JSON files")
    print(f"Reviewing with model: {OPENAI_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")

    results: list[dict] = []

    if CONCURRENCY <= 1:
        for i, file_path in enumerate(files, start=1):
            results.append(review_file(file_path))
            if i % 25 == 0 or i == len(files):
                print(f"Processed {i}/{len(files)}")
    else:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(review_file, p): p for p in files}
            done_count = 0

            for future in as_completed(futures):
                results.append(future.result())
                done_count += 1
                if done_count % 25 == 0 or done_count == len(files):
                    print(f"Processed {done_count}/{len(files)}")

        results.sort(key=lambda r: r["file"])

    ok_files = [r for r in results if r["status"] == "ok"]
    needs_review = [r for r in results if r["status"] == "needs_review"]
    parse_errors = [r for r in results if r["status"] == "parse_error"]
    api_errors = [r for r in results if r["status"] == "api_error"]

    summary = {
        "rootDir": str(ROOT_DIR),
        "model": OPENAI_MODEL,
        "totalFiles": len(results),
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
        json.dumps(results, ensure_ascii=False, indent=2),
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


if __name__ == "__main__":
    main()