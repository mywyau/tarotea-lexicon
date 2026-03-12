from openai import OpenAI
from pathlib import Path
from .config import MODEL, VOICE, FORMAT


def generate_audio(text: str, output_path: Path):
    client = OpenAI()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"🔊 Generating audio for: {text}")

    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=text,
        instructions=(
            "Speak in natural Hong Kong Cantonese. "
            "Use native Cantonese pronunciation. "
            "Do not translate. "
            "Do not add extra words."
        ),
        response_format=FORMAT
    ) as response:
        response.stream_to_file(output_path)