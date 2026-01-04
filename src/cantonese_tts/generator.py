from openai import OpenAI
from pathlib import Path
from .config import MODEL, VOICE, FORMAT


def generate_audio(text: str, output_path: Path, jyutping: str | None = None):
    
    client = OpenAI()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"🔊 Generating audio for: {text}")
    
    if jyutping:
        prompt = f"用廣東話讀: \n{text}"
    else:
        prompt = text  # fallback (not recommended for Chinese)

    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        # input=f"用廣東話讀：{text}",
        input=prompt,
        response_format=FORMAT
    ) as response:
        response.stream_to_file(output_path)
