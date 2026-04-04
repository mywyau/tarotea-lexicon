import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Skipping non-object JSON: {path}")
            return None
        return data
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def check_file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def collect_ids_recursively(value: Any, results: list[str]) -> None:
    if isinstance(value, dict):
        maybe_id = value.get("id")
        if isinstance(maybe_id, str) and maybe_id.strip():
            results.append(maybe_id.strip())

        for child in value.values():
            collect_ids_recursively(child, results)

    elif isinstance(value, list):
        for item in value:
            collect_ids_recursively(item, results)


def get_word_ids_from_level(level_data: dict[str, Any]) -> list[str]:
    collected: list[str] = []
    collect_ids_recursively(level_data, collected)

    seen: set[str] = set()
    word_ids: list[str] = []
    for item in collected:
        if item not in seen:
            seen.add(item)
            word_ids.append(item)

    return word_ids


def get_sentence_ids_from_word(word_data: dict[str, Any]) -> list[str]:
    examples = word_data.get("examples", [])
    if not isinstance(examples, list):
        return []

    sentence_ids: list[str] = []
    for example in examples:
        if not isinstance(example, dict):
            continue

        sentence_id = example.get("id")
        if isinstance(sentence_id, str) and sentence_id.strip():
            sentence_ids.append(sentence_id.strip())

    return sentence_ids


def audit_levels(
    levels_dir: Path,
    words_dir: Path,
    audio_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "summary": {
            "levels_checked": 0,
            "level_files_seen": 0,
            "word_ids_referenced": 0,
            "unique_word_ids_referenced": 0,
            "missing_word_json_count": 0,
            "missing_word_audio_count": 0,
            "sentence_examples_checked": 0,
            "missing_sentence_audio_count": 0,
        },
        "levels": [],
        "allMissingWordJsonIds": [],
        "allMissingWordAudioIds": [],
        "allMissingSentenceAudioIds": [],
    }

    seen_word_ids: set[str] = set()
    all_missing_word_json_ids: set[str] = set()
    all_missing_word_audio_ids: set[str] = set()
    all_missing_sentence_audio_ids: set[str] = set()

    level_files = sorted(levels_dir.rglob("*.json"))
    report["summary"]["level_files_seen"] = len(level_files)

    for level_file in level_files:
        level_data = load_json(level_file)
        if not level_data:
            continue

        level_id = level_data.get("id", level_file.stem)
        word_ids = get_word_ids_from_level(level_data)

        level_result: dict[str, Any] = {
            "levelFile": str(level_file),
            "levelId": level_id,
            "wordCount": len(word_ids),
            "missingWordJson": [],
            "missingWordAudio": [],
            "missingSentenceAudio": [],
        }

        report["summary"]["levels_checked"] += 1
        report["summary"]["word_ids_referenced"] += len(word_ids)

        for word_id in word_ids:
            seen_word_ids.add(word_id)

            word_json_path = words_dir / f"{word_id}.json"
            word_audio_path = audio_dir / f"{word_id}.mp3"

            word_json_exists = check_file_exists(word_json_path)
            word_audio_exists = check_file_exists(word_audio_path)

            if not word_json_exists:
                level_result["missingWordJson"].append({
                    "wordId": word_id,
                    "path": str(word_json_path),
                })
                all_missing_word_json_ids.add(word_id)
                report["summary"]["missing_word_json_count"] += 1

            if not word_audio_exists:
                level_result["missingWordAudio"].append({
                    "wordId": word_id,
                    "path": str(word_audio_path),
                })
                all_missing_word_audio_ids.add(word_id)
                report["summary"]["missing_word_audio_count"] += 1

            if word_json_exists:
                word_data = load_json(word_json_path)
                if word_data:
                    sentence_ids = get_sentence_ids_from_word(word_data)
                    report["summary"]["sentence_examples_checked"] += len(sentence_ids)

                    for sentence_id in sentence_ids:
                        sentence_audio_path = audio_dir / f"{sentence_id}.mp3"
                        sentence_audio_exists = check_file_exists(sentence_audio_path)

                        if not sentence_audio_exists:
                            level_result["missingSentenceAudio"].append({
                                "wordId": word_id,
                                "sentenceId": sentence_id,
                                "path": str(sentence_audio_path),
                            })
                            all_missing_sentence_audio_ids.add(sentence_id)
                            report["summary"]["missing_sentence_audio_count"] += 1

        report["levels"].append(level_result)

    report["summary"]["unique_word_ids_referenced"] = len(seen_word_ids)
    report["allMissingWordJsonIds"] = sorted(all_missing_word_json_ids)
    report["allMissingWordAudioIds"] = sorted(all_missing_word_audio_ids)
    report["allMissingSentenceAudioIds"] = sorted(all_missing_sentence_audio_ids)

    return report


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]

    print("\n=== Level audit complete ===\n")
    print(f"Levels checked:            {summary['levels_checked']}")
    print(f"Level files seen:          {summary['level_files_seen']}")
    print(f"Word refs:                 {summary['word_ids_referenced']}")
    print(f"Unique word refs:          {summary['unique_word_ids_referenced']}")
    print(f"Missing word JSON files:   {summary['missing_word_json_count']}")
    print(f"Missing word audio files:  {summary['missing_word_audio_count']}")
    print(f"Sentence examples checked: {summary['sentence_examples_checked']}")
    print(f"Missing sentence audio:    {summary['missing_sentence_audio_count']}")

    for level in report["levels"]:
        if (
            not level["missingWordJson"]
            and not level["missingWordAudio"]
            and not level["missingSentenceAudio"]
        ):
            continue

        print(f"\n--- Level: {level['levelId']} ---")

        if level["missingWordJson"]:
            print("\nMissing word JSON:")
            for item in level["missingWordJson"]:
                print(f"  - {item['wordId']} -> {item['path']}")

        if level["missingWordAudio"]:
            print("\nMissing word audio:")
            for item in level["missingWordAudio"]:
                print(f"  - {item['wordId']} -> {item['path']}")

        if level["missingSentenceAudio"]:
            print("\nMissing sentence audio:")
            for item in level["missingSentenceAudio"]:
                print(
                    f"  - sentenceId={item['sentenceId']} "
                    f"(wordId={item['wordId']}) -> {item['path']}"
                )


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_id_list(ids: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit level word IDs against words and audio backups."
    )
    parser.add_argument("--levels-dir", default="./levels")
    parser.add_argument("--words-dir", default="./r2-backup/words")
    parser.add_argument("--audio-dir", default="./r2-backup/audio")
    parser.add_argument("--output-dir", default="./audit-output-levels")

    args = parser.parse_args()

    levels_dir = Path(args.levels_dir)
    words_dir = Path(args.words_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    report = audit_levels(
        levels_dir=levels_dir,
        words_dir=words_dir,
        audio_dir=audio_dir,
    )

    print_summary(report)

    write_json_report(report, output_dir / "level-audit-report.json")
    write_id_list(report["allMissingWordJsonIds"], output_dir / "missing-word-json.txt")
    write_id_list(report["allMissingWordAudioIds"], output_dir / "missing-word-audio.txt")
    write_id_list(
        report["allMissingSentenceAudioIds"],
        output_dir / "missing-sentence-audio.txt",
    )

    print("\nWritten files:")
    print(f"  - {output_dir / 'level-audit-report.json'}")
    print(f"  - {output_dir / 'missing-word-json.txt'}")
    print(f"  - {output_dir / 'missing-word-audio.txt'}")
    print(f"  - {output_dir / 'missing-sentence-audio.txt'}")


if __name__ == "__main__":
    main()