# tarotea-lexicon

This repository now contains **lexicon-focused tooling only**:

- lexicon generation from word JSON content
- lexicon auditing and repair workflows
- utilities for extracting/normalizing sentence data used during lexicon QA

Audio/TTS generation code and related helper scripts were removed.

## OpenAI audit/check scripts (Batch API, no source rewrites)

These check scripts **never modify source files** in `r2-backup/words`.
They use the OpenAI Batch API and only write reports to `output/checks/...`.

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
  --limit 20
```

### 3) Translation checker

```bash
python3 scripts/check_translations.py \
  --input-dir r2-backup/words \
  --output-dir output/checks/translations \
  --model gpt-4.1 \
  --limit 20
```

## Optional rewrite tool

If you decide to apply rewrites later, `scripts/batch_rewrite_words.py` also uses the OpenAI Batch API, writes candidates to an output folder, and does not overwrite source files unless you manually promote reviewed output.

## Important notes

- Set `OPENAI_API_KEY` before running OpenAI-backed scripts.
- Start with small `--limit` batches and review generated reports first.
- This repo currently stores individual word entries in `r2-backup/words`.
