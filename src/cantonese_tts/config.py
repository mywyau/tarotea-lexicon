import os

MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
# VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
# VOICE = os.getenv("OPENAI_TTS_VOICE", "cedar")
FORMAT = "mp3"
