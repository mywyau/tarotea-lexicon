import json
import os
from pathlib import Path
from openai import OpenAI

LIMIT = 200
MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "low"   # try "none" for faster/cheaper, "medium" if quality needs help

print("🚀 generate_level_word_data.py started")

SYSTEM_PROMPT = """
You are a professional Hong Kong Cantonese linguist and language teacher.

Rules:
- Use natural spoken Hong Kong Cantonese
- Use the provided jyutping as canonical
- Jyutping must be accurate and include tone numbers
- This may be a phrase, idiom, discourse marker, connector, particle, or expression
- Usage notes must be in English only
- No Chinese characters inside usage notes
- Example sentences should be natural, spoken, practical, and everyday
- Avoid textbook-style phrasing
- Output must match the provided JSON schema exactly
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "pos": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },
        "usage": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 5
        },
        "examples": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string"},
                    "jyutping": {"type": "string"},
                    "meaning": {"type": "string"}
                },
                "required": ["sentence", "jyutping", "meaning"],
                "additionalProperties": False
            }
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "related": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5
        }
    },
    "required": ["pos", "usage", "examples", "tags", "related"],
    "additionalProperties": False
}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

LEVEL_FILE = Path("content/levels/level-ten.json")
OUT_DIR = Path("content/levels/generated")


def build_prompt(job):
    return f"""Generate learning content for an advanced Cantonese expression.

Level: {job["level"]}
Category: {job["category"]}

Id: {job["id"]}
Expression: {job["word"]}
Meaning: {job["meaning"]}
Jyutping: {job["jyutping"]}

Requirements:
- Use provided jyutping as canonical
- This may be a phrase, idiom, or discourse marker
- POS should be chosen from these when appropriate:
  phrase, idiom, connector, particle, expression
- Usage:
  - 4–5 English-only learner notes
  - No Chinese characters inside usage notes
- Examples:
  - Exactly 4 natural spoken Cantonese sentences
- Related:
  - 3–5 related common Cantonese words or expressions
- Tags:
  - learner-friendly tags
"""


with open(LEVEL_FILE, "r", encoding="utf-8") as f:
    level_data = json.load(f)

level_number = level_data["level"]
count = 0

for category, items in level_data["categories"].items():
    for entry in items:
        if count >= LIMIT:
            break

        count += 1
        word_id = entry["id"]

        out_dir = OUT_DIR / f"level-{level_number}" / category
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{word_id}.json"

        if out_path.exists():
            print(f"⏭️ Skipping existing {word_id}")
            continue

        job = {
            "level": level_number,
            "category": category,
            "id": word_id,
            "word": entry["word"],
            "meaning": entry["meaning"],
            "jyutping": entry["jyutping"],
        }

        prompt = build_prompt(job)

        try:
            response = client.responses.create(
                model=MODEL,
                reasoning={"effort": REASONING_EFFORT},
                max_output_tokens=1400,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "advanced_cantonese_expression_data",
                        "schema": SCHEMA,
                        "strict": True,
                    }
                },
            )

            generated = json.loads(response.output_text)

        except Exception as e:
            print(f"❌ Failed for {entry['word']}: {e}")
            continue

        final_json = {
            "id": word_id,
            "word": entry["word"],
            "jyutping": entry["jyutping"],
            "meaning": entry["meaning"],
            "pos": generated["pos"],
            "usage": generated["usage"],
            "examples": [
                {
                    "id": f"{word_id}-example-{i+1}",
                    "sentence": ex["sentence"],
                    "jyutping": ex["jyutping"],
                    "meaning": ex["meaning"],
                }
                for i, ex in enumerate(generated["examples"])
            ],
            "audio": {
                "word": f"{word_id}.mp3",
                "examples": [
                    f"{word_id}-example-{i+1}.mp3"
                    for i in range(len(generated["examples"]))
                ],
            },
            "tags": generated["tags"] + [f"level-{level_number}"],
            "related": generated["related"],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {out_path}")