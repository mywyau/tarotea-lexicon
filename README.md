# tarotea-lexicon

This repository now contains **lexicon-focused tooling only**:

- lexicon generation from word JSON content
- lexicon auditing and repair workflows
- utilities for extracting/normalizing sentence data used during lexicon QA

Audio/TTS generation code and related helper scripts were removed.

## OpenAI audit/check scripts (Batch API, no source rewrites)

These check scripts **never modify source files** in `r2-backup/words`.
They use the OpenAI Batch API and only write reports to `output/checks/...`.

### Prerequisites

Install Python dependencies before running any OpenAI-backed script:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 1) Chinese checker

```bash
python3 scripts/check_chinese.py \
  --input-dir r2-backup/words \
  --output-dir output/checks/chinese \
  --model gpt-4.1 \
  --limit 20
```

Batch options available on all scripts:

- `--completion-window` (default `24h`)
- `--poll-seconds` (default `10`)

Use `--limit 0` (or omit `--limit`) to process all files.

### 2) Jyutping checker

```bash
python3 scripts/check_jyutping.py \
  --input-dir r2-backup/words \
  --output-dir output/checks/jyutping \
  --model gpt-4.1 \
  --limit 0
```

If you run the checker with a newer reasoning model (for example `gpt-5.1`) and get a very large `needs_review` bucket, treat that report as a triage queue, not a direct rewrite list.

Recommended post-check triage:

1) Summarize issue tags by frequency:

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

report = Path("output/checks/jyutping/report.jsonl")
rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines() if line.strip()]
needs_review = [r for r in rows if r.get("status") == "needs_review"]

counter = Counter()
for row in needs_review:
    counter.update(row.get("issues", []))

print(f"needs_review={len(needs_review)}")
for issue, count in counter.most_common(20):
    print(f"{count:>5}  {issue}")
PY
```

2) Create a focused file list for high-signal issues first (example: clear mismatches/romanization errors):

```bash
python3 - <<'PY'
import json
from pathlib import Path

high_signal = {"example_jyutping_mismatch", "romanization_error", "headword_tone", "character_mismatch"}
report = Path("output/checks/jyutping/report.jsonl")
out = Path("output/checks/jyutping/high-signal-files.txt")

selected = []
for line in report.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("status") != "needs_review":
        continue
    issues = set(row.get("issues", []))
    if issues & high_signal:
        selected.append(row["file"])

out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
print(f"wrote {len(selected)} files to {out}")
PY
```

3) Stage and run rewrite candidate generation only on that shortlist (after manual spot-check):

```bash
mkdir -p output/checks/jyutping/high-signal-words
while IFS= read -r file; do cp "$file" output/checks/jyutping/high-signal-words/; done < output/checks/jyutping/high-signal-files.txt

python3 scripts/batch_rewrite_words.py \
  --input-dir output/checks/jyutping/high-signal-words \
  --output-dir output/rewritten-words-jyutping-pass1 \
  --model gpt-4.1 \
  --min-confidence 0.9 \
  --dry-run
```

Then manually review `output/rewritten-words-jyutping-pass1/reports/audit-report.jsonl` before applying anything.

### 3) Translation checker

```bash
python3 scripts/check_translations.py \
  --input-dir r2-backup/words \
  --output-dir output/checks/translations \
  --model gpt-4.1 \
  --limit 20
```

### 4) Local structure checker (offline)

Use this when you want to validate word JSON structure and references without calling OpenAI:

```bash
python3 scripts/check_word_json_data.py \
  --input-dir r2-backup/words \
  --recursive \
  --output-json output/checks/word-json/report.json
```

If your files are under `r2-backup/content`, pass that path with `--input-dir r2-backup/content`.

Optional strict check for audio files on disk:

```bash
python3 scripts/check_word_json_data.py \
  --input-dir r2-backup/words \
  --check-audio-files
```

Auto-add placeholders for missing `pos`/`tags`/`related`/`audio` fields:

```bash
python3 scripts/check_word_json_data.py \
  --input-dir r2-backup/words \
  --apply-placeholders \
  --output-json output/checks/word-json/placeholders.json
```

Notes:
- `--skip-nonword-files` is enabled by default, so utility JSON files (like remap summaries) are ignored.
- Use `--no-skip-nonword-files` if you want every JSON file audited regardless of shape.

## Optional rewrite tool (OpenAI Batch API)

Use this when you want model-assisted rewrite candidates for word entries.
Like the check scripts, it processes data via OpenAI Batch API and writes output under `output/...` (not back into `r2-backup/words`).

### Basic usage

```bash
python3 scripts/batch_rewrite_words.py \
  --input-dir r2-backup/words \
  --output-dir output/rewritten-words \
  --model gpt-4.1 \
  --limit 20
```

### Dry-run audit only (recommended first pass)

```bash
python3 scripts/batch_rewrite_words.py \
  --input-dir r2-backup/words \
  --output-dir output/rewritten-words \
  --dry-run \
  --limit 20
```

### Key options

- `--min-confidence` (default `0.8`): minimum model confidence required before a rewrite is considered auto-applicable.
- `--glob` (default `*.json`): select a subset of files by pattern.
- `--completion-window` (default `24h`) and `--poll-seconds` (default `10`): Batch API processing/polling controls.
- `--temperature`: optional sampling temperature.
- `--force-json-mode`: force Chat Completions JSON mode (by default, `gpt-5*` models skip forced JSON mode for compatibility).

### Output layout

- `output/rewritten-words/batch/input.jsonl`: submitted batch payload.
- `output/rewritten-words/reports/audit-report.jsonl`: per-file decisions, issues, validation status.
- `output/rewritten-words/reports/summary.json`: run summary (counts, batch id/status, paths).
- `output/rewritten-words/rewritten/*.json`: rewritten entries that passed decision/confidence/validation checks (unless `--dry-run`).

### Safety behavior

- The script validates key shape invariants (required keys, stable IDs, unchanged example IDs/audio).
- Rewrites are only written when decision/confidence/validation gates pass.
- Original source files in `r2-backup/words` are never overwritten by this tool.

## Important notes

- Set `OPENAI_API_KEY` before running OpenAI-backed scripts.
- Start with small `--limit` batches and review generated reports first.
- This repo currently stores individual word entries in `r2-backup/words`.
