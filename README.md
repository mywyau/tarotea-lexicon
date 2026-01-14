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
ffmpeg -i ngo5dei6-we-example-1.mp3 -af "atrim=start=1.3" cleaned/ngo5dei6-we-example-1.mp3
ffmpeg -i ngo5dei6-we-example-2.mp3 -af "atrim=start=1.3" cleaned/ngo5dei6-we-example-2.mp3
```

### Trim for single words alter the trim timing

```
ffmpeg -i 九月.mp3 -af "atrim=start=1.3" cleaned/九月.mp3
```


```

ffmpeg -i 九月.mp3 -af "atrim=start=1.3:end=1.8,asetpts=PTS-STARTPTS" cleaned/九月.mp3
ffmpeg -i 九月.mp3 -af "atrim=start=1.3:end=1.6" cleaned/九月.mp3
```


```
ffmpeg -i cleaned/ngo5dei6-we-example-1.mp3 -af "atrim=end=0.5" cleaned/ngo5dei6-we.mp3
ffmpeg -i cleaned/我之前去過香港好多次.mp3 -af "atrim=end=0.5" cleaned/我之前去過香港好多次2.mp3

ffmpeg -i 我.mp3 -af "atrim=start=1.3" cleaned/我.mp3
```


