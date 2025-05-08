import os
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv

load_dotenv()

def get_speech_to_text_service():
    authenticator = IAMAuthenticator(os.getenv("WATSON_API_KEY"))
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(os.getenv("WATSON_URL"))
    return speech_to_text

def transcribe_audio(file_path: str) -> str:
    stt_service = get_speech_to_text_service()
    with open(file_path, 'rb') as audio_file:
        result = stt_service.recognize(
            audio=audio_file,
            content_type='audio/mp3',
            model='en-US_BroadbandModel',
            smart_formatting=True
        ).get_result()
        
    if result.get("results"):
        return " ".join([r["alternatives"][0]["transcript"] for r in result["results"]])
    else:
        return "No transcription available."
