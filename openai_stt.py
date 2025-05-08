import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_with_openai(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript.get("text", "No transcription returned.")
