from pathlib import Path
from collections import defaultdict

WORDS_DIR = Path("./r2-backup/words")

groups = defaultdict(list)

for path in WORDS_DIR.glob("*.json"):
    stem = path.stem  # filename without .json
    key = stem.split("-", 1)[0]  # everything before first hyphen
    groups[key].append(path.name)

duplicates = {k: v for k, v in groups.items() if len(v) > 1}

if not duplicates:
    print("No duplicate base IDs found.")
else:
    print(f"Found {len(duplicates)} duplicate base ID groups:\n")
    for key in sorted(duplicates):
        print(f"{key}")
        for name in sorted(duplicates[key]):
            print(f"  - {name}")
        print()