import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"Skipping non-object JSON: {path}")
            return None

        return data
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def extract_sentences_from_file(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []

    source_word_id = data.get("id", "")
    source_word = data.get("word", "")
    source_word_jyutping = data.get("jyutping", "")
    source_word_meaning = data.get("meaning", "")
    tags = data.get("tags", [])

    examples = data.get("examples", [])
    if not isinstance(examples, list):
        print(f"Skipping invalid examples array in {path}")
        return []

    rows: list[dict[str, Any]] = []

    for ex in examples:
        if not isinstance(ex, dict):
            continue

        sentence = ex.get("sentence", "")
        jyutping = ex.get("jyutping", "")
        meaning = ex.get("meaning", "")
        sentence_id = ex.get("id", "")

        if not sentence:
            continue

        rows.append({
            "sentenceId": sentence_id,
            "sentence": sentence,
            "jyutping": jyutping,
            "meaning": meaning,
            "sourceWordId": source_word_id,
            "sourceWord": source_word,
            "sourceWordJyutping": source_word_jyutping,
            "sourceWordMeaning": source_word_meaning,
            "tags": tags,
            "sourceFile": str(path),
        })

    return rows


def extract_all_sentences(input_dir: Path) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []

    for path in sorted(input_dir.rglob("*.json")):
        all_rows.extend(extract_sentences_from_file(path))

    return all_rows


def dedupe_sentences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for row in rows:
        key = row["sentence"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract example sentences from word JSON files for sentence dojo."
    )
    parser.add_argument("input_dir", help="Directory containing word JSON files")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate sentences"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    rows = extract_all_sentences(input_dir)

    if args.dedupe:
        rows = dedupe_sentences(rows)

    write_json(rows, output_file)

    print(f"Extracted {len(rows)} sentence examples to {output_file}")


if __name__ == "__main__":
    main()