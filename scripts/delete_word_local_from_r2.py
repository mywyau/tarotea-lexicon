import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ----------------------------
# CONFIG
# ----------------------------
DRY_RUN = False  # set to False to actually delete

LOCAL_AUDIO_DIR = Path("./content/words")

R2_BUCKET = "tarotea-content"
ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]

AUDIO_PREFIX = "audio/words"

# ----------------------------
# R2 CLIENT
# ----------------------------
s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    region_name="auto",
)

# ----------------------------
# HELPERS
# ----------------------------
def delete_r2_object(key: str):
    if DRY_RUN:
        print(f"[DRY RUN] Would delete R2 object: {key}")
        return

    try:
        s3.delete_object(Bucket=R2_BUCKET, Key=key)
        print(f"Deleted R2 object: {key}")
    except Exception as e:
        print(f"Failed to delete {key}: {e}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not LOCAL_AUDIO_DIR.exists():
        raise FileNotFoundError(f"Local audio directory not found: {LOCAL_AUDIO_DIR}")

    local_files = [p for p in LOCAL_AUDIO_DIR.iterdir() if p.is_file()]
    print(f"Found {len(local_files)} local audio files in {LOCAL_AUDIO_DIR}")

    for local_file in local_files:
        filename = local_file.name
        r2_key = f"{AUDIO_PREFIX}/{filename}"
        delete_r2_object(r2_key)

if __name__ == "__main__":
    main()