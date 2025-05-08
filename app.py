import streamlit as st
import tempfile
from pydub import AudioSegment
from openai_stt import transcribe_with_openai

st.set_page_config(page_title="OpenAI STT Demo", layout="centered")
st.title("🎙️ OpenAI Whisper Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3) to get a transcription.")

uploaded_file = st.file_uploader("Drag and drop an MP3/WAV/MP4 file here", type=["mp3", "wav", "mp4"])

if uploaded_file:
    file_type = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Convert to WAV for playback
    if "mp3" in file_type:
        audio = AudioSegment.from_mp3(tmp_path)
    elif "wav" in file_type:
        audio = AudioSegment.from_wav(tmp_path)
    elif "mp4" in file_type:
        audio = AudioSegment.from_file(tmp_path, format="mp4")
    else:
        st.error("Unsupported file format.")
        st.stop()

    wav_path = tmp_path.replace(f".{file_type}", ".wav")
    audio.export(wav_path, format="wav")

    st.audio(wav_path, format="audio/wav")

    with st.spinner("Transcribing with OpenAI Whisper..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("📝 Transcription")
    st.write(transcript)



from swiss_food_matcher import load_food_database, match_entity
from entity_extractor import extract_food_entities
import pandas as pd

# Load food DB
food_db = load_food_database("swiss_food_composition_database_small.csv")

# Extract entities via OpenAI
entities = extract_food_entities(transcript)

# Match entities to DB
results = [match_entity(e, food_db) for e in entities]

# Show table
df = pd.DataFrame(results)
st.subheader("🍽️ Food Entities Extracted & Matched")
st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])
