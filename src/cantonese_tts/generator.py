from openai import OpenAI
from pathlib import Path

MODEL = "gpt-4o-mini-tts"
VOICE = "alloy"          # also test: "cedar"
FORMAT = "mp3"           # use wav/flac for master quality
SPEED = 1.00

client = OpenAI()

def generate_audio(text: str, output_path: Path, word_id:str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"🔊 Generating audio for: {word_id}")

    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=text.strip(),
        instructions=(
            "Speak in natural Hong Kong Cantonese. "
            "Use native Hong Kong Cantonese pronunciation and intonation. "
            "Read only the text exactly as written. "
            "Do not translate. "
            "Do not add extra words. "
            "Keep the delivery clear, steady, and slightly careful."
        ),
        response_format=FORMAT,
    ) as response:
        response.stream_to_file(output_path)