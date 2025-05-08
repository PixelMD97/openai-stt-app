import streamlit as st
import tempfile
from pydub import AudioSegment
from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity
import pandas as pd

st.set_page_config(page_title="OpenAI Whisper Speech to Text Demo", layout="centered")
st.title("🎙️ OpenAI Whisper Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3) to get a transcription.")

uploaded_file = st.file_uploader("Drag and drop an MP3/WAV/MP4 file here", type=["mp3", "wav", "mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    audio = AudioSegment.from_file(tmp_path)
    wav_path = tmp_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    st.audio(wav_path, format="audio/wav")

    with st.spinner("Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("📝 Transcription")
    st.write(transcript)

    with st.spinner("Extracting food entities..."):
        food_entities = extract_food_entities(transcript)

    with st.spinner("Matching to Swiss food database..."):
        food_db = load_food_database("swiss_food_composition_database_small.csv")
        matches = [match_entity(entity, food_db) for entity in food_entities]
        df = pd.DataFrame(matches)

    st.subheader("🍽️ Food Entities Extracted & Matched")
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])
