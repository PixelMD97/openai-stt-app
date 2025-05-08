import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_with_openai(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
