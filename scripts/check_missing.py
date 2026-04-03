import json
from collections import Counter, defaultdict
from pathlib import Path

missing_path = Path("./r2-backupV2/audio_report/audio-rename-missing.json")
rows = json.loads(missing_path.read_text(encoding="utf-8"))

print(f"Total missing rows: {len(rows)}")
print()

reason_counts = Counter(row.get("reason", "unknown") for row in rows)
print("By reason:")
for reason, count in reason_counts.most_common():
    print(f"- {reason}: {count}")

print("\nExamples:")
grouped = defaultdict(list)
for row in rows:
    grouped[row.get("reason", "unknown")].append(row)

for reason, items in grouped.items():
    print(f"\n=== {reason} ({len(items)}) ===")
    for item in items[:5]:
        print(json.dumps(item, ensure_ascii=False, indent=2))