#!/usr/bin/env bash

cd /Users/michaelyau/tarotea/openai-tts/public

for file in words/*.json; do
  name=$(basename "$file")
  echo "Uploading words/$name with UTF-8"
  wrangler r2 object put tarotea-content/words/$name \
    --file "$file" \
    --content-type "application/json; charset=utf-8" \
    --remote
done
