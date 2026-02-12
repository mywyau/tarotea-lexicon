import json
import os
from pathlib import Path
from openai import OpenAI

LIMIT = 3
print("🚀 generate_word_data.py started")


def build_prompt(job):
    return f"""Generate learning content for a Cantonese word.

Topic: {job["topic"]}
Category: {job["category"]}
Level: {job["level"]}

Word: {job["word"]}
Meaning: {job["meaning"]}
Jyutping: {job["jyutping"]}

Requirements:
- Use the provided jyutping as canonical
- Part of speech: noun (if applicable)
- Generate exactly 5 short usage notes
- Generate exactly 4 example sentences:
  - buying / shopping
  - describing freshness or quality
  - common dish or preparation
  - simple word usage
- Generate learner-friendly tags
- Generate 3–5 related common Cantonese words
- can the related field be ids of other words following <jyutping>-<word> e.g. coi3-vegetables

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
"""


def make_word_id(jyutping, meaning):
    return f"{jyutping.replace(' ', '')}-{meaning.replace(' ', '-')}"


SYSTEM_PROMPT = """
You are a professional Hong Kong Cantonese linguist and language teacher.

Rules:
- Use natural spoken Hong Kong Cantonese
- Jyutping must be accurate and include tone numbers
- Sentences should be short, practical, and everyday
- Avoid textbook-style phrasing
- Usage notes should be concrete and learner-friendly in english
- Output STRICT JSON only
- No explanations
"""


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOPICS_FILE = Path("content/topics/fruits-vegetables.json")
OUT_DIR = Path("content/topics/words")

with open(TOPICS_FILE, "r", encoding="utf-8") as f:
    topic_data = json.load(f)

topic_name = topic_data["topic"]

count = 0

for category, items in topic_data["categories"].items():
    for entry in items:
        if count >= LIMIT:
            break
        count += 1

        job = {
            "topic": topic_name,
            "category": category,
            "word": entry["word"],
            "meaning": entry["meaning"],
            "jyutping": entry["jyutping"],
            "level": 1
        }

        word_id = make_word_id(entry["jyutping"], entry["meaning"])

        out_dir = OUT_DIR / topic_name / category
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{word_id}.json"

        # Optional: skip already-generated words
        if out_path.exists():
            print(f"⏭️  Skipping existing {word_id}")
            continue

        prompt = build_prompt(job)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.output_text.strip()

        try:
            generated = json.loads(text)
        except json.JSONDecodeError:
            print("❌ JSON parse failed for:", entry["word"])
            print(text)
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
            "tags": generated["tags"] + [f"level-{job['level']}"],
            "related": generated["related"]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {out_path}")
