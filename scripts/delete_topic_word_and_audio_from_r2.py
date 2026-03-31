import json
import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ----------------------------
# CONFIG
# ----------------------------

DRY_RUN = False  # set to False to actually delete

TOPIC_JSON_PATH = Path("./r2-backup/topics/news-media.json")

# Your local topic file only; deletions happen in R2
R2_BUCKET = "tarotea-content"
ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]

# Object prefixes inside the bucket
WORDS_PREFIX = "words"
AUDIO_PREFIX = "audio/words"


# ----------------------------
# R2 CLIENT
# ----------------------------
s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
    region_name="auto",  # required by SDK, not used by R2
)

# ----------------------------
# HELPERS
# ----------------------------
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def collect_ids_from_topic(topic_data: dict) -> list[str]:
    ids = []
    for items in topic_data.get("categories", {}).values():
        for item in items:
            word_id = item.get("id")
            if word_id:
                ids.append(word_id)
    return ids

def get_r2_json(key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Failed to read {key}: {e}")
        return None

def delete_r2_object(key: str):
    if DRY_RUN:
        print(f"[DRY RUN] Would delete R2 object: {key}")
        return

    try:
        s3.delete_object(Bucket=R2_BUCKET, Key=key)
        print(f"Deleted R2 object: {key}")
    except Exception as e:
        print(f"Failed to delete {key}: {e}")

def extract_audio_files(word_data: dict) -> list[str]:
    audio = word_data.get("audio", {})
    files = []

    word_audio = audio.get("word")
    if isinstance(word_audio, str):
        files.append(word_audio)

    example_audio = audio.get("examples", [])
    if isinstance(example_audio, list):
        files.extend(x for x in example_audio if isinstance(x, str))

    return files

# ----------------------------
# MAIN
# ----------------------------
def main():
    topic_data = load_json(TOPIC_JSON_PATH)
    word_ids = collect_ids_from_topic(topic_data)

    print(f"Found {len(word_ids)} ids in old topic JSON")

    for word_id in word_ids:
        json_key = f"{WORDS_PREFIX}/{word_id}.json"

        word_data = get_r2_json(json_key)
        if not word_data:
            print(f"Missing or unreadable word JSON in R2: {json_key}")
            continue

        audio_files = extract_audio_files(word_data)

        for filename in audio_files:
            delete_r2_object(f"{AUDIO_PREFIX}/{filename}")

        delete_r2_object(json_key)

if __name__ == "__main__":
    main()