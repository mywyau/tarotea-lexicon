import argparse
from pathlib import Path
from .io import load_json, ensure_dir
from .generator import generate_audio

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    for json_file in input_dir.rglob("*.json"):
        data = load_json(json_file)
        word_id = data.get("id")
        word = data.get("word")

        if not word_id or not word:
            continue

        out = output_dir / f"{word_id}.mp3"
        if out.exists():
            continue

        if args.dry_run:
            print(f"[DRY] {word} → {out}")
        else:
            generate_audio(word, out)
