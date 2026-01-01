import json
from pathlib import Path

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
