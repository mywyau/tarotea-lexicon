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


def get_word_ids_from_topic(topic_data: dict[str, Any]) -> list[str]:
    categories = topic_data.get("categories", {})
    if not isinstance(categories, dict):
        return []

    word_ids: list[str] = []

    for _, words in categories.items():
        if not isinstance(words, list):
            continue

        for word in words:
            if not isinstance(word, dict):
                continue

            word_id = word.get("id")
            if isinstance(word_id, str) and word_id.strip():
                word_ids.append(word_id.strip())

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


def check_file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def audit_topics(
    topics_dir: Path,
    words_dir: Path,
    audio_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "summary": {
            "topics_checked": 0,
            "topic_files_seen": 0,
            "word_ids_referenced": 0,
            "unique_word_ids_referenced": 0,
            "missing_word_json": 0,
            "missing_word_audio": 0,
            "sentence_examples_checked": 0,
            "missing_sentence_audio": 0,
        },
        "topics": [],
    }

    seen_word_ids: set[str] = set()

    topic_files = sorted(topics_dir.rglob("*.json"))
    report["summary"]["topic_files_seen"] = len(topic_files)

    for topic_file in topic_files:
        topic_data = load_json(topic_file)
        if not topic_data:
            continue

        topic_id = topic_data.get("id", topic_file.stem)
        word_ids = get_word_ids_from_topic(topic_data)

        topic_result: dict[str, Any] = {
            "topicFile": str(topic_file),
            "topicId": topic_id,
            "wordCount": len(word_ids),
            "missingWordJson": [],
            "missingWordAudio": [],
            "words": [],
        }

        report["summary"]["topics_checked"] += 1
        report["summary"]["word_ids_referenced"] += len(word_ids)

        for word_id in word_ids:
            seen_word_ids.add(word_id)

            word_json_path = words_dir / f"{word_id}.json"
            word_audio_path = audio_dir / f"{word_id}.mp3"

            word_json_exists = check_file_exists(word_json_path)
            word_audio_exists = check_file_exists(word_audio_path)

            word_result: dict[str, Any] = {
                "wordId": word_id,
                "wordJsonPath": str(word_json_path),
                "wordAudioPath": str(word_audio_path),
                "wordJsonExists": word_json_exists,
                "wordAudioExists": word_audio_exists,
                "missingSentenceAudio": [],
            }

            if not word_json_exists:
                topic_result["missingWordJson"].append(word_id)
                report["summary"]["missing_word_json"] += 1

            if not word_audio_exists:
                topic_result["missingWordAudio"].append(word_id)
                report["summary"]["missing_word_audio"] += 1

            if word_json_exists:
                word_data = load_json(word_json_path)
                if word_data:
                    sentence_ids = get_sentence_ids_from_word(word_data)
                    report["summary"]["sentence_examples_checked"] += len(sentence_ids)

                    for sentence_id in sentence_ids:
                        sentence_audio_path = audio_dir / f"{sentence_id}.mp3"
                        sentence_audio_exists = check_file_exists(sentence_audio_path)

                        if not sentence_audio_exists:
                            word_result["missingSentenceAudio"].append({
                                "sentenceId": sentence_id,
                                "sentenceAudioPath": str(sentence_audio_path),
                            })
                            report["summary"]["missing_sentence_audio"] += 1

            topic_result["words"].append(word_result)

        report["topics"].append(topic_result)

    report["summary"]["unique_word_ids_referenced"] = len(seen_word_ids)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\nAudit complete\n")
    print(f"Topics checked:            {summary['topics_checked']}")
    print(f"Topic files seen:          {summary['topic_files_seen']}")
    print(f"Word refs:                 {summary['word_ids_referenced']}")
    print(f"Unique word refs:          {summary['unique_word_ids_referenced']}")
    print(f"Missing word JSON files:   {summary['missing_word_json']}")
    print(f"Missing word audio files:  {summary['missing_word_audio']}")
    print(f"Sentence examples checked: {summary['sentence_examples_checked']}")
    print(f"Missing sentence audio:    {summary['missing_sentence_audio']}")

    print("\nTopics with problems:\n")
    for topic in report["topics"]:
        has_problem = (
            len(topic["missingWordJson"]) > 0
            or len(topic["missingWordAudio"]) > 0
            or any(len(w["missingSentenceAudio"]) > 0 for w in topic["words"])
        )

        if not has_problem:
            continue

        print(f"- {topic['topicId']}")
        if topic["missingWordJson"]:
            print(f"  missing word json:  {len(topic['missingWordJson'])}")
        if topic["missingWordAudio"]:
            print(f"  missing word audio: {len(topic['missingWordAudio'])}")

        missing_sentence_total = sum(
            len(w["missingSentenceAudio"]) for w in topic["words"]
        )
        if missing_sentence_total:
            print(f"  missing sentence audio: {missing_sentence_total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit topic word IDs against words and audio backups."
    )
    parser.add_argument(
        "--topics-dir",
        default="./topics",
        help="Directory containing topic JSON files",
    )
    parser.add_argument(
        "--words-dir",
        default="./r2-backup/words",
        help="Directory containing word JSON files",
    )
    parser.add_argument(
        "--audio-dir",
        default="./r2-backup/audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output",
        default="./audit-output/topic-audio-word-audit.json",
        help="Path to write JSON report",
    )

    args = parser.parse_args()

    report = audit_topics(
        topics_dir=Path(args.topics_dir),
        words_dir=Path(args.words_dir),
        audio_dir=Path(args.audio_dir),
        output_path=Path(args.output),
    )
    print_summary(report)


if __name__ == "__main__":
    main()