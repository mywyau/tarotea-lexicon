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


def get_word_ids_from_level(level_data: dict[str, Any]) -> list[str]:
    categories = level_data.get("categories", {})
    if not isinstance(categories, dict):
        return []

    word_ids: list[str] = []

    for _, words in categories.items():
        if not isinstance(words, list):
            continue

        for word in words:
            if not isinstance(word, dict):
                continue

            word_id = word.get("id")
            if isinstance(word_id, str) and word_id.strip():
                word_ids.append(word_id)

    return word_ids


def extract_sentences_from_word_file(path: Path) -> list[dict[str, Any]]:
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


def process_level(level_file: Path, words_dir: Path, output_file: Path, dedupe: bool) -> None:
    level_data = load_json(level_file)
    if not level_data:
        print(f"Could not load level file: {level_file}")
        return

    word_ids = get_word_ids_from_level(level_data)

    rows: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for word_id in word_ids:
        word_file = words_dir / f"{word_id}.json"

        if not word_file.exists():
            missing_files.append(word_id)
            continue

        rows.extend(extract_sentences_from_word_file(word_file))

    if dedupe:
        rows = dedupe_sentences(rows)

    payload = {
        "level": level_data.get("level"),
        "title": level_data.get("title"),
        "totalWords": len(word_ids),
        "totalSentences": len(rows),
        "items": rows,
        "missingWordFiles": missing_files,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(word_ids)} word ids from {level_file.name}")
    print(f"Extracted {len(rows)} sentences")
    print(f"Missing word files: {len(missing_files)}")
    print(f"Wrote output to {output_file}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sentence dojo data for one level file or all level JSON files in a directory."
    )
    parser.add_argument("level_input", help="Path to a level JSON file or a directory of level JSON files")
    parser.add_argument("words_dir", help="Directory containing all word JSON files")
    parser.add_argument("output_path", help="Output JSON file (single mode) or output directory (directory mode)")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate sentences")
    args = parser.parse_args()

    level_input = Path(args.level_input)
    words_dir = Path(args.words_dir)
    output_path = Path(args.output_path)

    if not level_input.exists():
        raise SystemExit(f"Level input not found: {level_input}")

    if not words_dir.exists() or not words_dir.is_dir():
        raise SystemExit(f"Words directory not found or not a directory: {words_dir}")

    if level_input.is_file():
        process_level(level_input, words_dir, output_path, args.dedupe)
        return

    if level_input.is_dir():
        level_files = sorted(level_input.glob("level-*.json"))
        if not level_files:
            raise SystemExit(f"No level JSON files found in: {level_input}")

        output_path.mkdir(parents=True, exist_ok=True)

        for level_file in level_files:
            output_file = output_path / f"{level_file.stem}-sentences.json"
            process_level(level_file, words_dir, output_file, args.dedupe)
        return

    raise SystemExit(f"Unsupported input path: {level_input}")


if __name__ == "__main__":
    main()