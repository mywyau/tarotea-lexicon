import json
import sys
from pathlib import Path


def extract_audio_entries(word_entry):
    audio_entries = []

    # word audio
    audio_entries.append({
        "filename": word_entry["id"],
        "id": word_entry["id"],
        "text": word_entry["word"]
    })

    # example audios
    for ex in word_entry.get("examples", []):
        audio_entries.append({
            "filename": ex["id"],
            "id": ex["id"],
            "text": ex["sentence"]
        })

    return audio_entries


def main(input_path, output_path):
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_entries = []

    # case 1: single word JSON
    if "word" in data and "id" in data:
        audio_entries = extract_audio_entries(data)

    # case 2: topic with many words
    elif "words" in data:
        for word in data["words"]:
            audio_entries.extend(extract_audio_entries(word))
    else:
        raise ValueError("Unrecognised JSON format")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audio_entries, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(audio_entries)} audio entries → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_audio_json.py input.json output.audio.json")
        sys.exit(1)

    main(Path(sys.argv[1]), Path(sys.argv[2]))
