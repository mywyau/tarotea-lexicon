import json
from pathlib import Path


TOPICS_FILE = Path("content/topics/weather.json")


def normalize_jyutping(jyutping: str) -> str:
    return "".join(jyutping.strip().split())


def extract_id_prefix(word_id: str) -> str:
    return word_id.split("-", 1)[0]


def main() -> None:
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    mismatches = []

    for category, items in data.get("categories", {}).items():
        for entry in items:
            word_id = entry.get("id", "")
            word = entry.get("word", "")
            jyutping = entry.get("jyutping", "")

            expected_prefix = normalize_jyutping(jyutping)
            actual_prefix = extract_id_prefix(word_id)

            if actual_prefix != expected_prefix:
                mismatches.append({
                    "category": category,
                    "word": word,
                    "id": word_id,
                    "jyutping": jyutping,
                    "expected_prefix": expected_prefix,
                    "actual_prefix": actual_prefix,
                    "suggested_id": f"{expected_prefix}-{word_id.split('-', 1)[1]}" if "-" in word_id else expected_prefix
                })

    if not mismatches:
        print("✅ All IDs match their Jyutping prefixes.")
        return

    print(f"❌ Found {len(mismatches)} mismatched IDs:\n")

    for m in mismatches:
        print(f"Category: {m['category']}")
        print(f"Word: {m['word']}")
        print(f"Jyutping: {m['jyutping']}")
        print(f"Current ID: {m['id']}")
        print(f"Expected prefix: {m['expected_prefix']}")
        print(f"Actual prefix: {m['actual_prefix']}")
        print(f"Suggested ID: {m['suggested_id']}")
        print("-" * 50)


if __name__ == "__main__":
    main()