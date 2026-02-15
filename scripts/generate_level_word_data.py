import json
import os
from pathlib import Path
from openai import OpenAI

LIMIT = 200
print("🚀 generate_level_word_data.py started")

SYSTEM_PROMPT = """
You are a professional Hong Kong Cantonese linguist and language teacher.

Requirements:
- Use the provided jyutping as canonical
- In english generate four or five sentences on grammatical usage of the word in the Usage field section, please translate to english and no chinese in this section.
- Generate exactly 4 example sentences of using the word in different contexts
- Generate learner-friendly tags
- Generate 3–5 related common Cantonese words

Return JSON in this exact format (no extra fields):

{{
  "pos": [],
  "usage": [],
  "examples": [
    {{
      "sentence": "",
      "jyutping": "",
      "meaning": ""
    }}
  ],
  "tags": [],
  "related": []
}}

With filename = <id>.json
"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

LEVEL_FILE = Path("content/levels/level-fourteen.json")
OUT_DIR = Path("content/levels/generated")


def extract_json(text: str) -> str:
    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]

    start = text.find("{")
    end = text.rfind("}") + 1
    return text[start:end]


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
- POS should be: phrase, idiom, connector, particle, expression

Usage:
- 4–5 English-only learner notes
- No Chinese characters inside usage notes

Examples:
- Exactly 4 natural spoken Cantonese sentences

Return STRICT JSON only:

{{
  "pos": [],
  "usage": [],
  "examples": [
    {{
      "sentence": "",
      "jyutping": "",
      "meaning": ""
    }}
  ],
  "tags": [],
  "related": []
}}
"""


# Load level file
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
            "jyutping": entry["jyutping"]
        }

        prompt = build_prompt(job)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        raw_text = response.output_text
        json_text = extract_json(raw_text)

        try:
            generated = json.loads(json_text)
        except json.JSONDecodeError:
            print("❌ JSON parse failed:", entry["word"])
            print(raw_text)
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
                    "meaning": ex["meaning"]
                }
                for i, ex in enumerate(generated["examples"])
            ],
            "audio": {
                "word": f"{word_id}.mp3",
                "examples": [
                    f"{word_id}-example-{i+1}.mp3"
                    for i in range(len(generated["examples"]))
                ]
            },
            "tags": generated["tags"] + [f"level-{level_number}"],
            "related": generated["related"]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {out_path}")
