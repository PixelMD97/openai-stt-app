# app.py 11.45

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
import numpy as np

# --------- Google Sheets Logger ---------
def make_json_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)

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
        cleaned = json.loads(json.dumps(payload, default=make_json_serializable))
        response = requests.post(url, json=cleaned)
        response.raise_for_status()
        print("✅ Logged to Google Sheets:", response.text)
    except Exception as e:
        print("❌ Failed to log to Sheets:", e)

# --------- Highlighting ---------

import re

def highlight_transcript(text, entities):
    highlighted = text

    # Sort to replace longer matches first
    entities_sorted = sorted(entities, key=lambda e: -len(str(e.get("extracted", ""))))

    for ent in entities_sorted:
        food = ent.get("extracted", "").strip()
        quantity = str(ent.get("quantity", "")).strip()
        unit = ent.get("unit", "").strip()

        # Highlight "40 grams", "1 tablespoon", etc.
        if quantity and unit:
            pattern = rf"\b{quantity}\s+{unit}\b"
            highlighted = re.sub(pattern, 
                                 rf'<span style="background-color:#40e0d0;">\g<0></span>', 
                                 highlighted)

        # Highlight quantity alone if above didn't match
        if quantity:
            pattern = rf"\b{quantity}\b"
            highlighted = re.sub(pattern, 
                                 rf'<span style="background-color:#40e0d0;">\g<0></span>', 
                                 highlighted)

        # Highlight food term (green)
        if food:
            pattern = rf"\b{re.escape(food)}\b"
            highlighted = re.sub(pattern, 
                                 rf'<span style="background-color:#90ee90;">\g<0></span>', 
                                 highlighted, 
                                 flags=re.IGNORECASE)

    return highlighted

# --------- Streamlit UI ---------
st.set_page_config(page_title="PATHMATE - Speech to Text Demo", layout="centered")
st.title("Pathmate Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3, .wav, .ogg, .mp4) to get a transcription.")

uploaded_file = st.file_uploader("12:10 Drag and drop an MP3/WAV/OGG/MP4 file here", type=["mp3", "wav", "ogg", "mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if tmp_path.endswith((".ogg", ".wav", ".mp4")):
        audio = AudioSegment.from_file(tmp_path)
        tmp_path_mp3 = tmp_path + ".converted.mp3"
        audio.export(tmp_path_mp3, format="mp3")
        tmp_path = tmp_path_mp3

    audio = AudioSegment.from_file(tmp_path)
    wav_path = tmp_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    st.audio(wav_path, format="audio/wav")

    with st.spinner("Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("Transcription")
    st.markdown(highlight_transcript(transcript, []), unsafe_allow_html=True)

    with st.spinner("Extracting food entities..."):
        food_entities, response_text = extract_food_entities(transcript)
        st.subheader(" Raw LLM Output")
        st.code(response_text)
        st.markdown("**Extracted entities:**")
        st.write(food_entities)

    st.markdown("Highlighted Transcript with Entities")
    st.markdown(highlight_transcript(transcript, food_entities), unsafe_allow_html=True)

    with st.spinner("Matching to Swiss food database..."):
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

        # Clean data for logging/export
        serializable_corrections = json.loads(json.dumps(corrections, default=make_json_serializable))

        # ✅ Send to Google Sheets
        send_to_google_sheets(
            meal_id=f"meal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id="anon_user",
            raw_text=transcript,
            entities=food_entities,
            matches=serializable_corrections,
            prompts=[]
        )

        # ⬇️ Export buttons
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(serializable_corrections, indent=2),
            file_name="meal_log.json",
            mime="application/json"
        )

        st.download_button(
            "📥 Download CSV",
            data=df.to_csv(index=False),
            file_name="meal_log.csv",
            mime="text/csv"
        )
    else:
        st.warning("No food entities were matched. Please try with a different input.")
