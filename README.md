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
ffmpeg -i lau1-jacket-example-4.mp3 -af "atrim=start=1.8:end=3.5" cleaned/lau1-jacket-example-4.mp3
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

### Sync and download S3 bucket locally

```
aws s3 sync s3://tarotea-content ./r2-backup \
  --endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
  --profile r2
```

### Copy words from local directory to S3 cloudflare

```
aws s3 cp ./content/words \
s3://tarotea-content/words \
--recursive \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2
```

### Copy audio from local directory to S3 cloudflare

```
aws s3 cp ./audio/words \
s3://tarotea-content/audio \
--recursive \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2
```

### Find audio files longer than 10s in wip folders

```
find audio/words -type f -name "*.mp3" -print0 |
while IFS= read -r -d '' file; do
  duration=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$file")
  if awk "BEGIN {exit !($duration > 10)}"; then
    echo "TOO LONG: $file -> ${duration}s"
  fi
done
```

### Find audio files longer than 10s in r2 local audio directory - takes a while to run

```
find r2-backup/audio -type f -name "*.mp3" -print0 |
while IFS= read -r -d '' file; do
  duration=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$file")
  if awk "BEGIN {exit !($duration > 10)}"; then
    echo "TOO LONG: $file -> ${duration}s"
  fi
done
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

### remember to export your OpenAi key -

```

export <key>

```

### find copy json file in file name in directory

```
 find . -type f -name "*copy*.json"
```

### find copy json file in file name in directory

```
 find . -type f -name "*copy*.mp3"
```

```

python3 scripts/extract_example_sentences.py /path/to/your/json-folder examples.json
```

### extract sentences from levels

```
 not this one python3 scripts/extract_level_sentences.py ./r2-backup/levels/level-one.json ./r2-backup/words ./sentences/level/level-one-sentences.json --dedupe

python3 scripts/extract_level_sentences.py \
  ./r2-backup/levels \
  ./r2-backup/words \
  ./sentences/level \
  --dedupe
```

### extract sentences from topics

```
python3 scripts/extract_topics_sentences.py ./r2-backup/topics/math.json ./r2-backup/words ./sentences/topics/math-sentences.json --dedupe

python3 scripts/extract_topics_sentences.py \
  ./r2-backup/topics \
  ./r2-backup/words \
  ./sentences/topics \
  --dedupe
```

### delete topic words from r2 for topic

```
python3 scripts/delete_topic_from_r2.py
```

```
python3 scripts/review_naturalness.py run
```

python3 scripts/rewrite_word_unnatural_sentences.py run

python3 scripts/delete_audio_local_from_r2.py

mkdir -p /Users/michaelyau/self_projects/ibm/resources/r2-backup
cp -R /Users/michaelyau/tarotea/openai-tts/r2-backup/words /Users/michaelyau/self_projects/ibm/resources/r2-backup/

cp -R /Users/michaelyau/tarotea/openai-tts/r2-backupV2/audio /Users/michaelyau/tarotea/openai-tts/audio/words

### delete and sync new audio

aws s3 sync ./r2-backupV2/audio \
s3://tarotea-content/audio \
--delete \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2

aws s3 sync ./r2-backupV2/words \
s3://tarotea-content/words \
--delete \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2

aws s3 sync ./r2-backupV2/levels \
s3://tarotea-content/levels \
--delete \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2

python scripts/audit_topics.py \
 --topics-dir ./r2-backup/topics \
 --words-dir ./r2-backup/words \
 --audio-dir ./r2-backup/audio \
 --output ./audit-output/topic-audio-word-audit.json

python scripts/audit_levels.py \
 --levels-dir ./r2-backup/levels \
 --words-dir ./r2-backup/words \
 --audio-dir ./r2-backup/audio \
 --output-dir ./audit-output-levels

python scripts/audit_usage.py ./r2-backup/words

python scripts/audit_usage.py ./r2-backup/words --output reports/usage_audit.json

python scripts/fix_usage_ai.py \
 --report reports/usage_audit.json \
 --output-jsonl reports/openai/usage_patch_batch.jsonl \
 --mapping reports/openai/usage_patch_mapping.json \
 --model gpt-5.4-mini


python scripts/apply_usage_patch.py \
 --batch-output reports/openai/usage_patch_batch_output.jsonl \
 --root ./r2-backup/words \
 --output-dir reports/openai/applied_usage_patches \
 --reject-chinese

### delete and sync from local

aws s3 sync ./content/words \
s3://tarotea-content/words \
--dryrun \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2

aws s3 sync ./content/words \
s3://tarotea-content/words \
--endpoint-url https://3ed1e60152f33852da41c3d61ddb1140.r2.cloudflarestorage.com \
--profile r2