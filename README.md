# openai-tts

## What does this app do?

Batch-generate Cantonese Text To Speech audio for TaroTea site content. Or can be used to call OpenAI's api for conversion 

## Usage

source .venv/bin/activate

### Dry run is just a test run nothing will be generated :)

```
python scripts/generate_words.py \
  --resources resources/base_words.json \
  --output audio/words \
  --dry-run
```


### Production run, this will actually bill you :)

You first need to edit the data for the words/sentences wanted in 

```
base_words.json 
```

Then we can run this script to call openapi to convert out text to speech.
The cantonese context is inferred and using cantonese style langauge helps the ai speak using cantonese. 

```
python scripts/generate_words.py \
  --resources resources/base_words.json \
  --output audio/words
```

or 

```
./run_prod.sh
```

The conversion is prefixed with "用廣東話讀" which we will trim. 

We then need to trim the sound file and remove the prefix in the next step


### Production run, this will actually bill you :)

After installing ffmpeg we can then trim the file of the prompt that aids openai to use cantonese speech

in /tarotea/openai-tts/audio/words directory we can call ffmpeg to convert and edit our audio

Some example commands

```
ffmpeg -i zoi3gin3-goodbye.mp3 -af "atrim=start=1.7" cleaned/zoi3gin3-goodbye.mp3
ffmpeg -i lei4-come-exmaple-2.mp3 -af "atrim=start=1.5" cleaned/lei4-come-exmaple-2.mp3
```

### Trim for single words alter the trim timing

```
ffmpeg -i 九月.mp3 -af "atrim=start=1.3" cleaned/九月.mp3
```

```

ffmpeg -i 九月.mp3 -af "atrim=start=1.3:end=1.8,asetpts=PTS-STARTPTS" cleaned/九月.mp3
ffmpeg -i 九月.mp3 -af "atrim=start=1.3:end=1.6" cleaned/九月.mp3
```

<audio controls src="audio/words/cleaned/.mp3" title="Title"></audio>
```
ffmpeg -i git3cuk1-end-example-1.mp3 -af "atrim=start=1.3" cleaned/git3cuk1-end-example-1.mp3
ffmpeg -i cleaned/我之前去過香港好多次.mp3 -af "atrim=end=0.5" cleaned/我之前去過香港好多次2.mp3

ffmpeg -i 我.mp3 -af "atrim=start=1.3" cleaned/我.mp3
```


ffmpeg -i co5-sit-example-2.mp3 \
  -af "atrim=start=2.4:end=2.7,atempo=0.7" \
  cleaned/co5-sit-example-2.mp3


### Download and back up locally cloudflare r2 storage 


```
aws s3 sync s3://tarotea-content ./r2-backup \
  --endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
  --profile r2
```

### Generate audio file names and generate audio


# 1. Generate audio resource list
python scripts/generate_audio_resources.py

# 2. Dry run (no API calls)
python scripts/generate_words.py \
  --resources resources/words.audio.json \
  --output audio/words \
  --dry-run

# 3. Generate actual audio
python scripts/generate_words.py \
  --resources resources/words.audio.json \
  --output audio/words