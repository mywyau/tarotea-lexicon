import json
from pathlib import Path

WORDS_DIR = Path("content/words")
OUT_FILE = Path("resources/words.audio.json")


def extract_audio_entries(word_entry):
    entries = []

    # word audio
    entries.append({
        "filename": word_entry["id"],
        "id": word_entry["id"],
        "text": word_entry["word"]
    })

    # example audios
    for ex in word_entry.get("examples", []):
        entries.append({
            "filename": ex["id"],
            "id": ex["id"],
            "text": ex["sentence"]
        })

    return entries

def main():
    all_entries = []

    for word_file in sorted(WORDS_DIR.glob("*.json")):
        with open(word_file, "r", encoding="utf-8") as f:
            word_entry = json.load(f)

        all_entries.extend(extract_audio_entries(word_entry))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(all_entries)} audio resources → {OUT_FILE}")


if __name__ == "__main__":
    main()
