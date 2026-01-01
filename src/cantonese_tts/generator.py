from openai import OpenAI
from pathlib import Path
from .config import MODEL, VOICE, FORMAT

client = OpenAI()

def generate_audio(text: str, output_path: Path):
    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=text,
        format=FORMAT
    ) as response:
        response.stream_to_file(output_path)
