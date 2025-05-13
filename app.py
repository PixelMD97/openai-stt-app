import streamlit as st
import streamlit.components.v1 as components
import base64
import tempfile
import os
from pydub import AudioSegment
import pandas as pd


from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity


st.set_page_config(page_title="Pathmate Speech to Text Demo", layout="centered")
st.title("Pathmate Speech to Text Demo")
st.caption("Record or upload your meal voice log to get a transcription.")


# === Option A: HTML Recording ===
st.subheader("🎤 Step 1: Record or Upload")
st.markdown("#### Option A: Record voice input")

audio_html = """
<script>
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();

    audioChunks = [];
    mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = () => {
            const base64String = reader.result.split(',')[1];
            const queryString = "?audio=" + encodeURIComponent(base64String);
            window.location.search = queryString;
        };
    };
}

function toggleRecording() {
    if (!isRecording) {
        startRecording();
        document.getElementById("recordButton").innerText = "Stop Recording";
        isRecording = true;
    } else {
        mediaRecorder.stop();
        document.getElementById("recordButton").innerText = "Start Recording";
        isRecording = false;
    }
}

window.addEventListener("DOMContentLoaded", function() {
    const button = document.createElement("button");
    button.id = "recordButton";
    button.innerText = "Start Recording";
    button.onclick = toggleRecording;
    document.body.appendChild(button);
});
</script>
"""
components.html(audio_html, height=100)


# === Option B: File Upload ===
uploaded_file = st.file_uploader("#### Option B: Or upload MP3/WAV/OGG/MP4 file", type=["mp3", "wav", "ogg", "mp4"])

# === Handle Inputs ===
tmp_path = None

# From HTML recording
query_audio = st.experimental_get_query_params().get("audio")
if query_audio:
    st.success("✅ Audio recorded.")
    audio_bytes = base64.b64decode(query_audio[0])
    st.audio(audio_bytes, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        tmp_path = f.name

# From uploaded file
elif uploaded_file:
    st.success("✅ File uploaded successfully.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if tmp_path.endswith((".ogg", ".wav", ".mp4")):
        audio = AudioSegment.from_file(tmp_path)
        tmp_path_mp3 = tmp_path + ".converted.mp3"
        audio.export(tmp_path_mp3, format="mp3")
        tmp_path = tmp_path_mp3


# === Proceed if we have audio ===
if tmp_path:
    audio = AudioSegment.from_file(tmp_path)
    wav_path = tmp_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    st.audio(wav_path, format="audio/wav")

    with st.spinner("🧠 Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("📝 Transcription")
    st.write(transcript)

    with st.spinner("🔍 Extracting food entities..."):
        food_entities, response_text = extract_food_entities(transcript)
        st.subheader("🧠 Raw LLM Output")
        st.code(response_text)
        st.markdown("**Extracted entities:**")
        st.write(food_entities)

    with st.spinner("📊 Matching to Swiss food database..."):
        csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
        food_db = load_food_database(csv_path)
        matches = [match_entity(entity, food_db) for entity in food_entities]

    st.markdown("**Raw matches output:**")
    st.write(matches)

    st.subheader("Corrections (if needed)")
    corrections = []
    for match in matches:
        if not match["recognized"] or match["ID"] is None:
            corrected = st.text_input(
                f"🤔 Could not recognize food: '{match['extracted']}'. Please specify:",
                key=f"correction_{match['extracted']}"
            )
            if corrected:
                corrected_entity = {"extracted": corrected}
                corrected_match = match_entity(corrected_entity, food_db)
                corrected_match["quantity"] = match["quantity"]
                corrected_match["unit"] = match["unit"]
                corrections.append(corrected_match)
            else:
                corrections.append(match)
        else:
            corrections.append(match)

    st.subheader("🍽️ Food Entities Extracted & Matched")
    if corrections:
        df = pd.DataFrame(corrections)
        st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])
    else:
        st.warning("No food entities were matched. Please try with a different input.")
else:
    st.info("🎤 Please record or upload a voice input to continue.")
