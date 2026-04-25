#!/usr/bin/env python3
"""Audit-only checker for Jyutping quality in word JSON entries."""

from __future__ import annotations

from pathlib import Path

from check_utils import build_common_parser, run_check

SYSTEM_PROMPT = """You are checking Jyutping quality for Cantonese lexicon entries.
Return JSON only with keys: status, confidence, issues, suggestions.
- status: pass | needs_review
- confidence: float 0..1
- issues: list of short issue tags
- suggestions: list of concise actionable suggestions

Check only Jyutping quality:
1) headword `jyutping` segmentation and tone numbers
2) `examples[].jyutping` alignment with `examples[].sentence`
3) obvious tone/romanization errors
4) consistency with standard Cantonese Jyutping conventions
Do NOT rewrite content. Audit only.
"""


def payload_builder(entry: dict) -> dict:
    return {
        "task": "jyutping_check",
        "entry": {
            "id": entry.get("id"),
            "word": entry.get("word"),
            "jyutping": entry.get("jyutping"),
            "examples": [
                {
                    "id": ex.get("id"),
                    "sentence": ex.get("sentence"),
                    "jyutping": ex.get("jyutping"),
                }
                for ex in entry.get("examples", [])
            ],
        },
    }


def main() -> int:
    parser = build_common_parser("Check Jyutping quality only (no rewrites)", "output/checks/jyutping")
    args = parser.parse_args()
    return run_check(
        check_name="jyutping",
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
