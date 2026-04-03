import json
import os
from pathlib import Path
from typing import Any

OLD_WORD_ROOT = Path(os.getenv("OLD_WORD_ROOT", "./r2-backup/words")).resolve()
NEW_WORD_ROOT = Path(os.getenv("NEW_WORD_ROOT", "./r2-backupV2/words")).resolve()

INPUT_CONTENT_ROOT = Path(os.getenv("INPUT_CONTENT_ROOT", "./r2-backup/levels")).resolve()
OUTPUT_CONTENT_ROOT = Path(os.getenv("OUTPUT_CONTENT_ROOT", "./r2-backupV2/levels")).resolve()

FILE_GLOB = os.getenv("FILE_GLOB", "*.json")
OVERWRITE = os.getenv("OVERWRITE", "false").lower() == "true"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_json_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(FILE_GLOB) if p.is_file())


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


def index_word_entries(root: Path) -> tuple[dict[tuple, dict[str, Any]], list[dict[str, Any]]]:
    index: dict[tuple, dict[str, Any]] = {}
    collisions: list[dict[str, Any]] = []

    for path in collect_json_files(root):
        try:
            entry = load_json(path)
        except Exception as e:
            collisions.append({"path": str(path), "reason": f"Unreadable JSON: {e}"})
            continue

        if not isinstance(entry, dict):
            continue

        if not all(k in entry for k in ("id", "word", "jyutping", "meaning")):
            continue

        key = make_match_key(entry)
        if key in index:
            collisions.append({
                "path": str(path),
                "reason": "Duplicate match key",
                "existingPath": str(index[key]["path"]),
                "word": key[0],
                "jyutping": key[1],
                "meaning": key[2],
            })
            continue

        index[key] = {
            "path": path,
            "id": entry.get("id"),
            "word": entry.get("word"),
            "jyutping": entry.get("jyutping"),
            "meaning": entry.get("meaning"),
        }

    return index, collisions


def build_id_map(
    old_root: Path,
    new_root: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    old_index, old_collisions = index_word_entries(old_root)
    new_index, new_collisions = index_word_entries(new_root)

    issues: list[dict[str, Any]] = []
    issues.extend([{"type": "old_collision", **row} for row in old_collisions])
    issues.extend([{"type": "new_collision", **row} for row in new_collisions])

    id_map: dict[str, str] = {}

    for key, old_meta in old_index.items():
        new_meta = new_index.get(key)
        if new_meta is None:
            issues.append({
                "type": "missing_new_match",
                "oldPath": str(old_meta["path"]),
                "oldId": old_meta["id"],
                "word": old_meta["word"],
                "jyutping": old_meta["jyutping"],
                "meaning": old_meta["meaning"],
            })
            continue

        old_id = old_meta["id"]
        new_id = new_meta["id"]

        if isinstance(old_id, str) and isinstance(new_id, str):
            id_map[old_id] = new_id
        else:
            issues.append({
                "type": "bad_id",
                "oldPath": str(old_meta["path"]),
                "newPath": str(new_meta["path"]),
                "oldId": old_id,
                "newId": new_id,
            })

    return id_map, issues


def rewrite_ids_recursive(value: Any, id_map: dict[str, str], replacements: list[dict[str, Any]], path: str = "$") -> Any:
    if isinstance(value, dict):
        updated = dict(value)

        looks_like_word_ref = (
            isinstance(updated.get("id"), str)
            and "word" in updated
            and "jyutping" in updated
            and "meaning" in updated
        )

        if looks_like_word_ref and updated["id"] in id_map:
            old_id = updated["id"]
            new_id = id_map[old_id]
            updated["id"] = new_id
            replacements.append({
                "path": f"{path}.id",
                "oldId": old_id,
                "newId": new_id,
            })

        for k, v in updated.items():
            if k == "id" and looks_like_word_ref:
                continue
            updated[k] = rewrite_ids_recursive(v, id_map, replacements, f"{path}.{k}")

        return updated

    if isinstance(value, list):
        return [
            rewrite_ids_recursive(item, id_map, replacements, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        ]

    return value

def main() -> None:
    if not OLD_WORD_ROOT.exists():
        raise RuntimeError(f"OLD_WORD_ROOT does not exist: {OLD_WORD_ROOT}")
    if not NEW_WORD_ROOT.exists():
        raise RuntimeError(f"NEW_WORD_ROOT does not exist: {NEW_WORD_ROOT}")
    if not INPUT_CONTENT_ROOT.exists():
        raise RuntimeError(f"INPUT_CONTENT_ROOT does not exist: {INPUT_CONTENT_ROOT}")

    OUTPUT_CONTENT_ROOT.mkdir(parents=True, exist_ok=True)

    id_map, mapping_issues = build_id_map(OLD_WORD_ROOT, NEW_WORD_ROOT)
    print(f"Built ID map entries: {len(id_map)}")
    print(f"Mapping issues: {len(mapping_issues)}")

    files = collect_json_files(INPUT_CONTENT_ROOT)
    print(f"Content files found: {len(files)}")

    file_reports: list[dict[str, Any]] = []
    total_replacements = 0

    for src in files:
        rel = src.relative_to(INPUT_CONTENT_ROOT)
        dst = OUTPUT_CONTENT_ROOT / rel

        if dst.exists() and not OVERWRITE:
            raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

        try:
            data = load_json(src)
        except Exception as e:
            file_reports.append({
                "file": str(rel),
                "status": "error",
                "error": f"Unreadable JSON: {e}",
                "replacementCount": 0,
            })
            continue

        replacements: list[dict[str, Any]] = []
        updated = rewrite_ids_recursive(data, id_map, replacements)

        write_json(dst, updated)

        file_reports.append({
            "file": str(rel),
            "status": "ok",
            "replacementCount": len(replacements),
            "replacements": replacements,
        })
        total_replacements += len(replacements)

    summary = {
        "oldWordRoot": str(OLD_WORD_ROOT),
        "newWordRoot": str(NEW_WORD_ROOT),
        "inputContentRoot": str(INPUT_CONTENT_ROOT),
        "outputContentRoot": str(OUTPUT_CONTENT_ROOT),
        "idMapCount": len(id_map),
        "mappingIssueCount": len(mapping_issues),
        "contentFileCount": len(files),
        "totalReplacements": total_replacements,
    }

    write_json(OUTPUT_CONTENT_ROOT / "_rewrite-summary.json", summary)
    write_json(OUTPUT_CONTENT_ROOT / "_rewrite-file-reports.json", file_reports)
    write_json(OUTPUT_CONTENT_ROOT / "_rewrite-mapping-issues.json", mapping_issues)
    write_json(OUTPUT_CONTENT_ROOT / "_old-to-new-id-map.json", id_map)

    print("\nDone.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()