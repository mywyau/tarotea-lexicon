import subprocess
from pathlib import Path

# CONFIG
INPUT_DIR = Path("audio/raw")
OUTPUT_DIR = Path("audio/cleaned")
# TRIM_START = 1.2  # seconds
TRIM_START = 1.7  # seconds
# TRIM_START = 2.1  # seconds
# TRIM_START = 2.2  # seconds
# TRIM_START = 2.8  # seconds

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for audio_file in INPUT_DIR.glob("*.mp3"):
    out_file = OUTPUT_DIR / audio_file.name

    if out_file.exists():
        print(f"⏭️  Skipping existing {out_file.name}")
        continue

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_file),
        "-af", f"atrim=start={TRIM_START}",
        str(out_file)
    ]

    print(f"✂️  Trimming {audio_file.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed to process {audio_file.name}")
        print(result.stderr)
    else:
        print(f"✅ Wrote {out_file}")
