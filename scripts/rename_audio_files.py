import json
import os
import shutil
from pathlib import Path
from typing import Any

OLD_WORD_ROOT = Path(os.getenv("OLD_WORD_ROOT", "./r2-backup/words")).resolve()
NEW_WORD_ROOT = Path(os.getenv("NEW_WORD_ROOT", "./r2-backupV2/words")).resolve()

AUDIO_ROOT_DIR = Path(os.getenv("AUDIO_ROOT_DIR", "./r2-backup/audio")).resolve()
NEW_AUDIO_ROOT_DIR = Path(os.getenv("NEW_AUDIO_ROOT_DIR", "./r2-backupV2/audio")).resolve()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./r2-backupV2/audio_report")).resolve()

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
COPY_INSTEAD_OF_MOVE = os.getenv("COPY_INSTEAD_OF_MOVE", "true").lower() == "true"
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"

AUDIO_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.getenv("AUDIO_EXTENSIONS", ".mp3,.wav,.m4a,.ogg").split(",")
    if ext.strip()
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_transfer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if not OVERWRITE:
            raise FileExistsError(f"Destination already exists: {dst}")
        dst.unlink()

    if COPY_INSTEAD_OF_MOVE:
        shutil.copy2(src, dst)
    else:
        src.rename(dst)


def collect_json_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def collect_audio_files(root: Path) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        by_name.setdefault(path.name, []).append(path)
    return by_name


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def get_example_sentences(entry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    examples = entry.get("examples")
    if isinstance(examples, list):
        for ex in examples:
            if isinstance(ex, dict):
                sentence = normalize_text(ex.get("sentence"))
                if sentence:
                    out.append(sentence)
    return out


def make_match_key(entry: dict[str, Any]) -> tuple:
    return (
        normalize_text(entry.get("word")),
        normalize_text(entry.get("jyutping")).lower(),
        normalize_text(entry.get("meaning")).lower(),
        tuple(get_example_sentences(entry)),
    )


def get_audio_names(entry: dict[str, Any]) -> list[str]:
    audio = entry.get("audio")
    if not isinstance(audio, dict):
        return []

    names: list[str] = []

    word_audio = audio.get("word")
    if isinstance(word_audio, str) and word_audio.strip():
        names.append(word_audio.strip())

    example_audio = audio.get("examples")
    if isinstance(example_audio, list):
        for item in example_audio:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())

    return names


def index_entries_by_key(root: Path) -> tuple[dict[tuple, Path], list[dict[str, Any]]]:
    index: dict[tuple, Path] = {}
    collisions: list[dict[str, Any]] = []

    for path in collect_json_files(root):
        try:
            entry = load_json(path)
        except Exception as e:
            collisions.append({
                "path": str(path),
                "reason": f"Unreadable JSON: {e}",
            })
            continue

        if not isinstance(entry, dict):
            collisions.append({
                "path": str(path),
                "reason": "JSON is not an object",
            })
            continue

        key = make_match_key(entry)

        if key in index:
            collisions.append({
                "path": str(path),
                "reason": "Duplicate match key",
                "existingPath": str(index[key]),
                "keyPreview": {
                    "word": key[0],
                    "jyutping": key[1],
                    "meaning": key[2],
                    "exampleCount": len(key[3]),
                },
            })
            continue

        index[key] = path

    return index, collisions


def main() -> None:
    if not OLD_WORD_ROOT.exists():
        raise RuntimeError(f"OLD_WORD_ROOT does not exist: {OLD_WORD_ROOT}")
    if not NEW_WORD_ROOT.exists():
        raise RuntimeError(f"NEW_WORD_ROOT does not exist: {NEW_WORD_ROOT}")
    if not AUDIO_ROOT_DIR.exists():
        raise RuntimeError(f"AUDIO_ROOT_DIR does not exist: {AUDIO_ROOT_DIR}")
    

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NEW_AUDIO_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    old_index, old_collisions = index_entries_by_key(OLD_WORD_ROOT)
    new_index, new_collisions = index_entries_by_key(NEW_WORD_ROOT)
    audio_by_name = collect_audio_files(AUDIO_ROOT_DIR)

    planned: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    executed: list[dict[str, Any]] = []

    for key, old_path in old_index.items():
        new_path = new_index.get(key)
        if new_path is None:
            missing.append({
                "reason": "No matching new word file",
                "oldWordPath": str(old_path),
                "matchKey": {
                    "word": key[0],
                    "jyutping": key[1],
                    "meaning": key[2],
                    "exampleCount": len(key[3]),
                },
            })
            continue

        try:
            old_entry = load_json(old_path)
            new_entry = load_json(new_path)
        except Exception as e:
            errors.append({
                "oldWordPath": str(old_path),
                "newWordPath": str(new_path),
                "error": f"Failed reading matched word files: {e}",
            })
            continue

        old_audio_names = get_audio_names(old_entry)
        new_audio_names = get_audio_names(new_entry)

        if len(old_audio_names) != len(new_audio_names):
            missing.append({
                "reason": "Audio name count mismatch",
                "oldWordPath": str(old_path),
                "newWordPath": str(new_path),
                "oldAudioNames": old_audio_names,
                "newAudioNames": new_audio_names,
            })
            continue

        for old_name, new_name in zip(old_audio_names, new_audio_names):
            matches = audio_by_name.get(old_name, [])

            if not matches:
                missing.append({
                    "reason": "Audio file not found",
                    "oldWordPath": str(old_path),
                    "newWordPath": str(new_path),
                    "oldAudioName": old_name,
                    "newAudioName": new_name,
                })
                continue

            if len(matches) > 1:
                collisions.append({
                    "reason": "Multiple audio files share same old filename",
                    "oldWordPath": str(old_path),
                    "newWordPath": str(new_path),
                    "oldAudioName": old_name,
                    "newAudioName": new_name,
                    "matches": [str(p) for p in matches],
                })
                continue

            src = matches[0]
            rel_parent = src.relative_to(AUDIO_ROOT_DIR).parent
            dst = NEW_AUDIO_ROOT_DIR / rel_parent / new_name

            planned.append({
                "oldWordPath": str(old_path),
                "newWordPath": str(new_path),
                "src": str(src),
                "dst": str(dst),
                "oldAudioName": old_name,
                "newAudioName": new_name,
            })

    if not DRY_RUN:
        for row in planned:
            src = Path(row["src"])
            dst = Path(row["dst"])
            try:
                safe_transfer(src, dst)
                executed.append(row)
            except Exception as e:
                errors.append({
                    "src": str(src),
                    "dst": str(dst),
                    "error": str(e),
                })

    summary = {
        "oldWordRoot": str(OLD_WORD_ROOT),
        "newWordRoot": str(NEW_WORD_ROOT),
        "audioRootDir": str(AUDIO_ROOT_DIR),
        "audioRootDir": str(NEW_AUDIO_ROOT_DIR),
        "dryRun": DRY_RUN,
        "copyInsteadOfMove": COPY_INSTEAD_OF_MOVE,
        "overwrite": OVERWRITE,
        "plannedRenameCount": len(planned),
        "executedRenameCount": len(executed),
        "missingCount": len(missing),
        "collisionCount": len(collisions),
        "errorCount": len(errors),
        "oldIndexCollisionCount": len(old_collisions),
        "newIndexCollisionCount": len(new_collisions),
    }

    write_json(OUTPUT_DIR / "audio-rename-summary.json", summary)
    write_json(OUTPUT_DIR / "audio-rename-planned.json", planned)
    write_json(OUTPUT_DIR / "audio-rename-missing.json", missing)
    write_json(OUTPUT_DIR / "audio-rename-collisions.json", collisions)
    write_json(OUTPUT_DIR / "audio-rename-errors.json", errors)
    write_json(OUTPUT_DIR / "old-word-index-collisions.json", old_collisions)
    write_json(OUTPUT_DIR / "new-word-index-collisions.json", new_collisions)

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()