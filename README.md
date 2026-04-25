# tarotea-lexicon

This repository now contains **lexicon-focused tooling only**:

- lexicon generation from word JSON content
- lexicon auditing and repair workflows
- utilities for extracting/normalizing sentence data used during lexicon QA

Audio/TTS generation code and related helper scripts were removed.

## OpenAI batch rewriting workflow

Use the batch tool to audit and optionally rewrite individual word JSON files.

```bash
python3 scripts/batch_rewrite_words.py \
  --input-dir r2-backup/words \
  --output-dir output/rewritten-words \
  --model gpt-4.1 \
  --dry-run
```

To process all matching files and write rewritten output when confidence is high enough:

```bash
python3 scripts/batch_rewrite_words.py \
  --input-dir r2-backup/words \
  --output-dir output/rewritten-words \
  --model gpt-4.1 \
  --min-confidence 0.85
```

### Important notes

- Set `OPENAI_API_KEY` before running OpenAI-backed scripts.
- Start with `--dry-run` and small `--limit` batches, then review `output/rewritten-words/reports/audit-report.jsonl`.
- This repo currently stores individual word entries in `r2-backup/words`.
