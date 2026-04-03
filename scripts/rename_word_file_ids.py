import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

# ----------------------------
# Config
# ----------------------------
ROOT_DIR = Path(os.getenv("ROOT_DIR", "./r2-backup/words")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./r2-backupV2/words")).resolve()

FILE_GLOB = os.getenv("FILE_GLOB", "*.json")
RENAME_FILES = os.getenv("RENAME_FILES", "true").lower() == "true"
COPY_UNCHANGED = os.getenv("COPY_UNCHANGED", "true").lower() == "true"

# id mode:
# - jyutping-meaning
# - jyutping-meaning-u
ID_MODE = os.getenv("ID_MODE", "jyutping-meaning").strip().lower()

# if true, overwrite existing output dir contents for matching files
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"

# ----------------------------
# Helpers
# ----------------------------
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
APOSTROPHE_RE = re.compile(r"[’']")
MULTI_DASH_RE = re.compile(r"-+")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return WHITESPACE_RE.sub(" ", value.strip())


def normalize_jyutping(value: str | None) -> str:
    return normalize_text(value).lower()


def normalize_meaning(value: str | None) -> str:
    return normalize_text(value).lower()


def slugify_meaning(value: str | None) -> str:
    text = normalize_meaning(value)
    if not text:
        return "unknown"

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = APOSTROPHE_RE.sub("", text)
    text = NON_ALNUM_RE.sub("-", text)
    text = MULTI_DASH_RE.sub("-", text).strip("-")

    return text or "unknown"


def jyutping_to_slug(value: str | None) -> str:
    jp = normalize_jyutping(value)
    if not jp:
        return "unknown"
    return jp.replace(" ", "")


def hanzi_to_codepoint_slug(hanzi: str | None) -> str:
    text = normalize_text(hanzi)
    if not text:
        return "unknown"
    return "-".join(f"u{ord(ch):04x}" for ch in text)


def build_base_id(entry: dict[str, Any]) -> str:
    jyutping_slug = jyutping_to_slug(entry.get("jyutping"))
    meaning_slug = slugify_meaning(entry.get("meaning"))
    hanzi_slug = hanzi_to_codepoint_slug(entry.get("word"))

    if ID_MODE == "jyutping-meaning-u":
        return f"{jyutping_slug}-{meaning_slug}-{hanzi_slug}"

    return f"{jyutping_slug}-{meaning_slug}"


def uniquify(base_id: str, used_ids: set[str]) -> str:
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id

    i = 2
    while True:
        candidate = f"{base_id}-{i}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        i += 1


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def should_process_entry(data: Any) -> bool:
    return isinstance(data, dict) and "id" in data and "word" in data and "jyutping" in data


def rename_entry(entry: dict[str, Any], new_base_id: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(entry, ensure_ascii=False))

    old_id = updated.get("id")
    updated["id"] = new_base_id

    examples = updated.get("examples")
    if isinstance(examples, list):
        for idx, ex in enumerate(examples, start=1):
            if isinstance(ex, dict):
                ex["id"] = f"{new_base_id}-example-{idx}"

    audio = updated.get("audio")
    if isinstance(audio, dict):
        audio["word"] = f"{new_base_id}.mp3"

        examples_audio = audio.get("examples")
        if isinstance(examples_audio, list):
            audio["examples"] = [
                f"{new_base_id}-example-{idx}.mp3"
                for idx in range(1, len(examples_audio) + 1)
            ]

    return updated


def collect_json_files(root_dir: Path, pattern: str) -> list[Path]:
    return sorted(root_dir.rglob(pattern))


def safe_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not OVERWRITE:
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    shutil.copy2(src, dst)


def relative_output_path(src: Path, root_dir: Path, new_name: str | None = None) -> Path:
    rel = src.relative_to(root_dir)
    if new_name is None:
        return rel
    return rel.with_name(new_name)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not ROOT_DIR.exists():
        raise RuntimeError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_json_files(ROOT_DIR, FILE_GLOB)
    print(f"Found JSON files: {len(files)}")

    used_ids: set[str] = set()
    remap: list[dict[str, str]] = []
    processed_count = 0
    skipped_count = 0

    # First pass: reserve unique ids deterministically in file order
    planned_ids: dict[Path, str] = {}

    for path in files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping unreadable JSON: {path} ({e})")
            skipped_count += 1
            continue

        if not should_process_entry(data):
            skipped_count += 1
            continue

        base_id = build_base_id(data)
        new_id = uniquify(base_id, used_ids)
        planned_ids[path] = new_id

    # Second pass: rewrite files
    for path in files:
        rel = path.relative_to(ROOT_DIR)

        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping unreadable JSON: {path} ({e})")
            continue

        if not should_process_entry(data):
            if COPY_UNCHANGED:
                out_path = OUTPUT_DIR / rel
                safe_copy_file(path, out_path)
            continue

        old_id = str(data.get("id", ""))
        new_id = planned_ids[path]
        updated = rename_entry(data, new_id)

        out_filename = f"{new_id}.json" if RENAME_FILES else path.name
        out_rel = relative_output_path(path, ROOT_DIR, out_filename)
        out_path = OUTPUT_DIR / out_rel

        if out_path.exists() and not OVERWRITE:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

        write_json(out_path, updated)

        remap.append({
            "oldId": old_id,
            "newId": new_id,
            "oldPath": str(rel),
            "newPath": str(out_rel),
        })

        processed_count += 1

    write_json(OUTPUT_DIR / "id-remap.json", remap)

    summary = {
        "rootDir": str(ROOT_DIR),
        "outputDir": str(OUTPUT_DIR),
        "fileGlob": FILE_GLOB,
        "idMode": ID_MODE,
        "renameFiles": RENAME_FILES,
        "processedCount": processed_count,
        "skippedCount": skipped_count,
        "remapCount": len(remap),
    }
    write_json(OUTPUT_DIR / "rename-summary.json", summary)

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()