#!/usr/bin/env python3
"""Audit-only checker for Chinese text quality in word JSON entries."""

from __future__ import annotations

from pathlib import Path

from check_utils import build_common_parser, run_check

SYSTEM_PROMPT = """You are checking Cantonese lexicon Chinese text quality.
Return JSON only with keys: status, confidence, issues, suggestions.
- status: pass | needs_review
- confidence: float 0..1
- issues: list of short issue tags
- suggestions: list of concise actionable suggestions

Check only Chinese text quality and consistency:
1) headword in `word`
2) Chinese sentences in `examples[].sentence`
3) obvious script/character errors or malformed punctuation
4) consistency with meaning context
Do NOT rewrite content. Audit only.
"""


def payload_builder(entry: dict) -> dict:
    return {
        "task": "chinese_text_check",
        "entry": {
            "id": entry.get("id"),
            "word": entry.get("word"),
            "meaning": entry.get("meaning"),
            "examples": [
                {"id": ex.get("id"), "sentence": ex.get("sentence"), "meaning": ex.get("meaning")}
                for ex in entry.get("examples", [])
            ],
        },
    }


def main() -> int:
    parser = build_common_parser("Check Chinese text quality only (no rewrites)", "output/checks/chinese")
    args = parser.parse_args()
    return run_check(
        check_name="chinese",
        system_prompt=SYSTEM_PROMPT,
        payload_builder=payload_builder,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model=args.model,
        file_glob=args.glob,
        limit=args.limit,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
