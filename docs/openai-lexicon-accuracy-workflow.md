# OpenAI workflow for improving Cantonese lexicon accuracy

This guide explains how to use OpenAI to **audit and rewrite individual word JSON files** in this repository.

## 1) Where the source JSON lives

Word entries are in:

- `r2-backup/words/*.json`

> Note: if you were looking for `r2-backup/content`, this repo currently stores word JSON in `r2-backup/words`.

## 2) Accuracy goals for each entry

For each word JSON, focus validation on:

1. **Headword consistency**
   - `word` matches `id` intent and `meaning`.
2. **Jyutping quality**
   - `jyutping` field for the headword is segmented correctly and uses valid tone numbers.
   - every `examples[].jyutping` matches the Cantonese sentence.
3. **English meaning quality**
   - natural, concise, and aligned with Cantonese usage.
4. **POS and usage notes**
   - `pos` and `usage` are coherent with the lexical item.
5. **Related items sanity**
   - `related` values are semantically relevant.
6. **Structure and IDs**
   - stable ids and JSON shape maintained for app compatibility.

## 3) Use structured outputs to avoid JSON drift

When calling OpenAI, use a strict schema so responses always return valid JSON with your expected keys.

Recommended pattern:

- send one input word JSON at a time
- ask model to return:
  - `decision`: `"accept" | "rewrite" | "flag"`
  - `confidence`: number 0-1
  - `issues`: list of detected issues
  - `updated_entry`: full JSON object in your original shape

This lets you safely automate changes while preserving app format.

## 4) Two-pass model strategy (higher accuracy)

Use two passes instead of one:

1. **Audit pass**
   - Model reviews entry and reports issues only.
2. **Rewrite pass**
   - Triggered only when issues exist.
   - Model outputs corrected JSON.

Optional third pass:

3. **Verifier pass**
   - Second model call checks rewritten JSON against source sentence and phonology constraints.

## 5) Prompting template (single entry)

System prompt essentials:

- You are a Cantonese lexicography assistant.
- Preserve JSON schema exactly.
- Keep `id`, `audio`, and example ids stable unless explicitly invalid.
- Correct Jyutping for headword + all examples.
- Do not simplify Cantonese sentence content unless needed for correctness.
- Return JSON only.

User payload:

- include the full original word JSON
- include explicit constraints:
  - use tone numbers 1-6
  - preserve punctuation where possible
  - keep translation natural (not overly literal)

## 6) Safe rollout plan

1. Start with a **small batch** (e.g., 20 files).
2. Human-review all `rewrite` and `flag` outputs.
3. Track error categories (tone, segmentation, mistranslation, POS mismatch).
4. Iterate prompts.
5. Scale to all files once spot-check quality is stable.

## 7) Suggested quality metrics

Track before/after:

- `% entries requiring rewrite`
- `% example jyutping corrections`
- `% reviewer-approved on first pass`
- average confidence by category
- top recurring issue types

## 8) Practical notes for this repo

- Keep generated updates in a separate output folder before replacing source files.
- Store model version and prompt hash next to each processed file for reproducibility.
- Add a deterministic JSON formatter step so diffs are easy to review.

