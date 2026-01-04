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

You first need to edit the words in base_words.json 

Then we can run this script to call openapi to convert out text to speech.
The cantonese context is inferred and using cantonese style langauge helps the ai speak using cantonese. 

```
python scripts/generate_words.py \
  --resources resources/base_words.json \
  --output audio/words
```

We then need to trim the sound file and remove the prefix in the next step


### Production run, this will actually bill you :)

After installing ffmpeg we can then trim the file of the prompt that aids openai to use cantonese speech

in /tarotea/openai-tts/audio/words directory we can call ffmpeg to convert

```
ffmpeg -i 四個人.mp3 -af "atrim=start=1.3" cleaned/四個人.mp3
ffmpeg -i 四杯水.mp3 -af "atrim=start=1.3" cleaned/四杯水.mp3
ffmpeg -i 四點鐘.mp3 -af "atrim=start=1.6" cleaned/四點鐘.mp3
ffmpeg -i 四十蚊.mp3 -af "atrim=start=1.3" cleaned/四十蚊.mp3
ffmpeg -i 四次.mp3 -af "atrim=start=1.3" cleaned/四次.mp3
ffmpeg -i 星期四.mp3 -af "atrim=start=1.3" cleaned/星期四.mp3
ffmpeg -i 四樓.mp3 -af "atrim=start=1.3" cleaned/四樓.mp3
ffmpeg -i 四樣嘢.mp3 -af "atrim=start=1.3" cleaned/四樣嘢.mp3
```

### Trim for single words alter the trim timing

```
ffmpeg -i 星期三.mp3 -af "atrim=start=1.5" cleaned/星期三.mp3
```

