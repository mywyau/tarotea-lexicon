# openai-tts

## What this does

Batch-generate Cantonese TTS audio for dictionary content.

## Usage

python scripts/generate_words.py --input ... --output ...

source .venv/bin/activate


### Dry run is just a test run nothing will be generated :)
```
python scripts/generate_words.py \
  --resources resources/base_words.json \
  --output audio/words \
  --dry-run
```


### Production run, this will actually bill you :)
```
python scripts/generate_words.py \
  --resources resources/base_words.json \
  --output audio/words
```


### Production run, this will actually bill you :)

After installing ffmpeg we can then trim the file of the prompt that aids openai to use cantonese speech

```
ffmpeg -i 十一個人.mp3 -af "atrim=start=1.3" 十一個人2.mp3
```

