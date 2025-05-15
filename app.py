#16:50 15.5

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
import re

# --------- JSON helpers ---------
def make_json_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    return str(obj)

def clean_list_for_json(data):
    return json.loads(json.dumps(data, default=make_json_serializable))

# --------- Google Sheets Logger ---------
def send_to_google_sheets(meal_id, user_id, raw_text, entities, matches, prompts):
    url = "https://script.google.com/macros/s/YOUR_SCRIPT_URL/exec"  # <<< Replace this
    payload = {
        "meal_id": meal_id,
        "user_id": user_id,
        "raw_text": raw_text,
        "entities": entities,
        "matches": matches,
        "prompts": prompts,
    }

    try:
        cleaned = clean_list_for_json(payload)
        response = requests.post(url, json=cleaned)
        response.raise_for_status()
        print("✅ Logged to Google Sheets:", response.text)
    except Exception as e:
        print("❌ Failed to log to Sheets:", e)
        print("📨 Payload was:", json.dumps(payload, indent=2, default=make_json_serializable))
        if 'response' in locals():
            print("📬 Response:", response.status_code)
            print("📬 Response text:", response.text)

# --------- Highlighting ---------
def highlight_transcript(text, entities):
    if not entities:
        return text

    highlighted = text
    entities_sorted = sorted(entities, key=lambda e: -len(str(e.get("extracted", ""))))

    for ent in entities_sorted:
        food = str(ent.get("extracted", "") or "").strip()
        quantity = str(ent.get("quantity", "") or "").strip()
        unit = str(ent.get("unit", "") or "").strip()

        if quantity and unit:
            pattern = rf"\b{re.escape(quantity)}\s+{re.escape(unit)}\b"
            highlighted = re.sub(pattern, rf'<span style="background-color:#40e0d0;">\g<0></span>', highlighted)

        if quantity:
            pattern = rf"\b{re.escape(quantity)}\b"
            highlighted = re.sub(pattern, rf'<span style="background-color:#40e0d0;">\g<0></span>', highlighted)

        if food:
            pattern = rf"\b{re.escape(food)}\b"
            highlighted = re.sub(pattern, rf'<span style="background-color:#90ee90;">\g<0></span>', highlighted, flags=re.IGNORECASE)

    return highlighted

# --------- Streamlit UI ---------
st.set_page_config(page_title="PATHMATE - Speech to Text Demo 17:15", layout="centered")
st.title("Pathmate Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3, .wav, .ogg, .mp4) to get a transcription.")

uploaded_file = st.file_uploader("Drag and drop an MP3/WAV/OGG/MP4 file here", type=["mp3", "wav", "ogg", "mp4"])

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

    # Transcription
    with st.spinner("Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("Transcription")
    st.markdown(highlight_transcript(transcript, []), unsafe_allow_html=True)

    # Entity Extraction
    with st.spinner("Extracting food entities..."):
        food_entities, response_text = extract_food_entities(transcript)
        st.subheader("Raw LLM Output")
        st.code(response_text)
        st.markdown("Extracted entities:")
        st.write(food_entities)

    # Quantity + Unit Clarification
    st.subheader("Clarify missing quantities or units")
    clarified_entities = []
    clarification_prompts = []

    for entity in food_entities:
        extracted = entity.get("extracted", "")
        quantity = entity.get("quantity")
        unit = entity.get("unit")

        # Handle "a"/"an"
        if isinstance(quantity, str) and quantity.lower() in ["a", "an"]:
            quantity = 1

        # Ask if quantity is "some", "few", or missing
        if not quantity or (isinstance(quantity, str) and quantity.lower() in ["some", "several", "few"]):
            quantity_input = st.number_input(
                f"How much {extracted}? (unit: {unit or 'portion'})",
                min_value=0.0,
                step=1.0,
                key=f"clarify_quantity_{extracted}"
            )
            if quantity_input > 0:
                quantity = quantity_input
                clarification_prompts.append({
                    "extracted": extracted,
                    "asked_for": "quantity",
                    "response": quantity_input
                })

        # Unit fallback
        if not unit or unit.strip() == "":
            unit = "portion"

        clarified_entities.append({
            "extracted": extracted,
            "quantity": quantity,
            "unit": unit
        })

    # Show updated highlight
    st.markdown("Highlighted Transcript with Entities")
    st.markdown(highlight_transcript(transcript, clarified_entities), unsafe_allow_html=True)

    # Match to DB
    with st.spinner("Matching to Swiss food database..."):
        csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
        food_db = load_food_database(csv_path)
        initial_matches = [match_entity(entity, food_db) for entity in clarified_entities]

    # Ask for unknown food correction
    final_matches = []
    for match in initial_matches:
        if not match["recognized"] or match["ID"] is None:
            correction = st.text_input(
                f"🤔 Could not recognize food: '{match['extracted']}'. What is it?",
                key=f"manual_match_{match['extracted']}"
            )
            if correction:
                corrected = match_entity({"extracted": correction}, food_db)
                corrected["quantity"] = match["quantity"]
                corrected["unit"] = match["unit"]
                final_matches.append(corrected)
            else:
                final_matches.append(match)
        else:
            final_matches.append(match)

    # Show final table
    st.subheader("Matched Results")
    df = pd.DataFrame(final_matches)
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

    # Send to Sheets
    send_to_google_sheets(
        meal_id=f"meal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        user_id="anon_user",
        raw_text=transcript,
        entities=clarified_entities,
        matches=clean_list_for_json(final_matches),
        prompts=clarification_prompts
    )

    # Download options
    st.download_button(
        "Download JSON",
        data=json.dumps(final_matches, indent=2, default=make_json_serializable),
        file_name="meal_log.json",
        mime="application/json"
    )

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="meal_log.csv",
        mime="text/csv"
    )
