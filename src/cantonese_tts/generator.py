from openai import OpenAI
from pathlib import Path
from .config import MODEL, VOICE, FORMAT


def generate_audio(text: str, output_path: Path):
    
    client = OpenAI()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"🔊 Generating audio for: {text}")
    
    
    prompt = f"用廣東話讀: \n{text}"

    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=prompt,
        response_format=FORMAT
    ) as response:
        response.stream_to_file(output_path)
