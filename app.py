# app.py

import streamlit as st
import tempfile
from pydub import AudioSegment
from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity
import pandas as pd
import os
import requests
import json
from datetime import datetime

# --------- Google Sheets Logger ---------
def send_to_google_sheets(meal_id, user_id, raw_text, entities, matches, prompts):
    url = "https://script.google.com/macros/s/AKfycbwMRxcoQarz8GdxxDGUTBmY-jfzPqXlBRD-DfFsiDX1PiNXieNGc4nTB1_Qo-Cj9pRd/exec"
    payload = {
        "meal_id": meal_id,
        "user_id": user_id,
        "raw_text": raw_text,
        "entities": entities,
        "matches": matches,
        "prompts": prompts,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raises error if status code is not 200
        print("✅ Logged to Google Sheets:", response.text)
    except Exception as e:
        print("❌ Failed to log to Sheets:", e)
        print("📨 Payload was:", json.dumps(payload, indent=2))
        if 'response' in locals():
            print("📬 Response:", response.status_code)
            print("📬 Response text:", response.text)
        else:
            print("📬 No HTTP response received.")

# --------- Streamlit UI ---------
st.set_page_config(page_title="PATHMATE - Speech to Text Demo", layout="centered")
st.title("Pathmate Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3, .wav, .ogg, .mp4) to get a transcription.")

uploaded_file = st.file_uploader("Drag and drop an MP3/WAV/OGG/MP4 file here", type=["mp3", "wav", "ogg", "mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Convert to mp3 if needed
    if tmp_path.endswith((".ogg", ".wav", ".mp4")):
        audio = AudioSegment.from_file(tmp_path)
        tmp_path_mp3 = tmp_path + ".converted.mp3"
        audio.export(tmp_path_mp3, format="mp3")
        tmp_path = tmp_path_mp3

    audio = AudioSegment.from_file(tmp_path)
    wav_path = tmp_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    st.audio(wav_path, format="audio/wav")

    # Transcription
    with st.spinner("Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("📝 Transcription")
    st.write(transcript)

    # Entity Extraction
    with st.spinner("Extracting food entities..."):
        food_entities, response_text = extract_food_entities(transcript)
        st.subheader("Raw LLM Output")
        st.code(response_text)
        st.markdown("**Extracted entities:**")
        st.write(food_entities)

    # Match to Swiss food DB
    with st.spinner("Matching to Swiss food database..."):
        csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
        food_db = load_food_database(csv_path)
        matches = [match_entity(entity, food_db) for entity in food_entities]

    st.markdown("**Raw matches output:**")
    st.write(matches)

    # Correction prompt
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

    # Display final table
    st.subheader("🍽️ Food Entities Extracted & Matched")
    if corrections:
        df = pd.DataFrame(corrections)
        st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

        # Send to Google Sheets
        send_to_google_sheets(
            meal_id=f"meal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id="anon_user",
            raw_text=transcript,
            entities=food_entities,
            matches=corrections,
            prompts=[]  # Optional: extend later
        )
    else:
        st.warning("No food entities were matched. Please try with a different input.")
