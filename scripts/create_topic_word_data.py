import json
import os
from pathlib import Path
from openai import OpenAI

LIMIT = 200
MODEL = "gpt-5.4-mini"   # try "gpt-5.4" if you want higher quality
REASONING_EFFORT = "low" # try "medium" only if needed

print("🚀 generate_word_data.py started")

SYSTEM_PROMPT = """
You are a professional Hong Kong Cantonese linguist and language teacher.

Rules:
- Use natural spoken Hong Kong Cantonese
- Jyutping must be accurate and include tone numbers
- Example sentences must appear ONLY in the examples field
- The usage field must contain English teaching notes only
- Do not put full example sentences in the usage field
- Do not put Chinese example clauses or translations in the usage field
- In the usage field, you may mention the target word or very short Cantonese words/phrases only when necessary
- Usage notes should explain grammar, register, collocations, common contexts, and learner tips
- Avoid textbook-style phrasing
- Output must match the provided JSON schema exactly
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "pos": {
            "type": "array",
            "items": {"type": "string"}
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
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"}
        }
    },
    "required": ["pos", "usage", "examples", "tags", "related"],
    "additionalProperties": False
}

def build_prompt(job):
    return f"""Generate learning content for a Cantonese word.

Topic: {job["topic"]}
Category: {job["category"]}
Level: {job["level"]}

Id: {job["id"]}
Word: {job["word"]}
Meaning: {job["meaning"]}
Jyutping: {job["jyutping"]}

Requirements:
- Use the provided jyutping as canonical
- Part of speech: noun (if applicable)
- The usage field needs four bullet points on the grammitcal usage, please use english to teach the user.
- Usage field content must be in english, any references to chinese words, sentence or characters are to be in Cantonese
- Generate exactly 4 example sentences using the word in different contexts that are natural chinese
- Generate learner-friendly tags
- Generate 3–5 related common Cantonese words
- Do not rush prioritise chinese, jyutping, translation accuracy
"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TOPICS_FILE = Path("content/topics/plants-gardening.json")
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
            "id": entry["id"],
            "word": entry["word"],
            "meaning": entry["meaning"],
            "jyutping": entry["jyutping"],
            "level": 1,
        }

        word_id = entry["id"]
        out_dir = OUT_DIR / topic_name / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{word_id}.json"

        if out_path.exists():
            print(f"⏭️  Skipping existing {word_id}")
            continue

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
                        "name": "cantonese_word_data",
                        "schema": SCHEMA,
                        "strict": True,
                    }
                },
            )

            # Safety/refusal handling
            if response.output and response.output[0].content:
                first = response.output[0].content[0]
                if getattr(first, "type", None) == "refusal":
                    print(f"❌ Refused for: {entry['word']}")
                    continue

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
            "tags": generated["tags"] + [f"level-{job['level']}"],
            "related": generated["related"],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {out_path}")