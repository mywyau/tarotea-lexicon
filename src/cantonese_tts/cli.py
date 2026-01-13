import argparse
from pathlib import Path
from dotenv import load_dotenv
from .io import load_json, ensure_dir
from .generator import generate_audio

load_dotenv()

def run():
    parser = argparse.ArgumentParser(description="Generate Cantonese TTS audio")
    parser.add_argument("--resources", required=True, help="Path to audio resources JSON")
    parser.add_argument("--output", required=True, help="Output directory for audio files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    ensure_dir(output_dir)

    resources = load_json(Path(args.resources))

    for item in resources:
        filename = item.get("filename")
        word_id = item.get("id")
        text = item.get("text")

        if not word_id or not text or not filename:
            print(f"⚠️  Skipping invalid resource: {item}")
            continue

        out = output_dir / f"{filename}.mp3"

        if out.exists():
            print(f"⏭️  Skipping existing audio: {out.name}")
            continue

        if args.dry_run:
            print(f"[DRY] {text} → {out}")
        else:
            generate_audio(text, out)

    print("✅ Done")
