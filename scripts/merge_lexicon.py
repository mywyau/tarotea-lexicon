import json
import os
from pathlib import Path
from typing import Any

ORIGINAL_PATH = Path(os.getenv("ORIGINAL_PATH", "./lexicon/lexicon-build/lexicon-map.json")).resolve()
FIXED_PATH = Path(os.getenv("FIXED_PATH", "./lexicon/audit-output/lexicon-map-fixed.json")).resolve()
REMAP_PATH = Path(os.getenv("REMAP_PATH", "./lexicon/audit-output/id-remap.json")).resolve()
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "./lexicon/lexicon-build/lexicon-map.merged.json")).resolve()
OUTPUT_MIN_PATH = Path(os.getenv("OUTPUT_MIN_PATH", "./lexicon/lexicon-build/lexicon-min.merged.json")).resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original = load_json(ORIGINAL_PATH)
    fixed = load_json(FIXED_PATH)
    remap = load_json(REMAP_PATH) if REMAP_PATH.exists() else []

    if not isinstance(original, dict):
        raise RuntimeError("Original lexicon must be a JSON object")
    if not isinstance(fixed, dict):
        raise RuntimeError("Fixed lexicon must be a JSON object")
    if not isinstance(remap, list):
        raise RuntimeError("Remap file must be a JSON array")

    merged: dict[str, dict[str, str]] = {}

    # Start from original
    merged.update(original)

    # Remove old auto ids that were remapped
    for item in remap:
        if not isinstance(item, dict):
            continue
        old_id = item.get("oldId")
        new_id = item.get("newId")
        if isinstance(old_id, str) and isinstance(new_id, str) and old_id != new_id:
            merged.pop(old_id, None)

    # Overlay fixed audited entries
    merged.update(fixed)

    runtime_min = [
        {
            "id": entry_id,
            "hanzi": value["hanzi"],
            "jyutping": value["jyutping"],
        }
        for entry_id, value in sorted(merged.items())
        if isinstance(value, dict)
        and isinstance(value.get("hanzi"), str)
        and isinstance(value.get("jyutping"), str)
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    OUTPUT_MIN_PATH.write_text(
        json.dumps(runtime_min, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote merged map: {OUTPUT_PATH}")
    print(f"Wrote merged min: {OUTPUT_MIN_PATH}")
    print(f"Original entries: {len(original)}")
    print(f"Fixed entries: {len(fixed)}")
    print(f"Merged entries: {len(merged)}")


if __name__ == "__main__":
    main()