# tarotea-lexicon

This repository now contains **lexicon-focused tooling only**:

- lexicon generation from word JSON content
- lexicon auditing and repair workflows
- utilities for extracting/normalizing sentence data used during lexicon QA

Audio/TTS generation code and related helper scripts were removed.

## Core workflows

### 1) Generate lexicon candidates

```bash
python3 scripts/lexicon.py
```

Outputs are written to `lexicon/lexicon-build/`.

### 2) Audit generated lexicon

```bash
python3 scripts/audit_lexicon.py
```

Outputs are written to `lexicon/audit-batch-output/`.

### 3) Merge audited output back into runtime lexicon

```bash
python3 scripts/merge_lexicon.py
```

### 4) Additional audit + repair helpers

```bash
```

## Data extraction helpers

```bash
```

## Notes

- Most scripts use environment variables for paths (defaults are in each script).
- For OpenAI-backed scripts, set `OPENAI_API_KEY` before running.
