import streamlit as st
import tempfile
from pydub import AudioSegment
from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity
import pandas as pd
import os

st.set_page_config(page_title="OpenAI Whisper Speech to Text Demo", layout="centered")
st.title("🎙️ OpenAI Whisper Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3, .wav, .ogg, .mp4) to get a transcription.")

uploaded_file = st.file_uploader("Drag and drop an MP3/WAV/OGG/MP4 file here", type=["mp3", "wav", "ogg", "mp4"])

if uploaded_file:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Convert to mp3 if needed
    if tmp_path.endswith(".ogg") or tmp_path.endswith(".wav") or tmp_path.endswith(".mp4"):
        audio = AudioSegment.from_file(tmp_path)
        tmp_path_mp3 = tmp_path + ".converted.mp3"
        audio.export(tmp_path_mp3, format="mp3")
        tmp_path = tmp_path_mp3

    # Convert MP3 to WAV for playback in browser
    audio = AudioSegment.from_file(tmp_path)
    wav_path = tmp_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    st.audio(wav_path, format="audio/wav")

    with st.spinner("Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("📝 Transcription")
    st.write(transcript)

    with st.spinner("Extracting food entities..."):
        food_entities, response_text = extract_food_entities(transcript)
        st.subheader("🧠 Raw LLM Output")
        st.code(response_text)
        st.markdown("**Extracted entities:**")
        st.write(food_entities)

    with st.spinner("Matching to Swiss food database..."):
        csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
        food_db = load_food_database(csv_path)
        matches = [match_entity(entity, food_db) for entity in food_entities]
        st.markdown("**Raw matches output:**")
        st.write(matches)

    st.subheader("🍽️ Food Entities Extracted & Matched")
    if matches:
        df = pd.DataFrame(matches)
        if all(k in df.columns for k in ["extracted", "recognized", "quantity", "unit", "ID"]):
            st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])
        else:
            st.warning("Some fields are missing in the match results.")
    else:
        st.warning("No food entities were matched. Please try with a different input.")
