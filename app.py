#17:45 15.5

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

# --- JSON helpers ---
def make_json_serializable(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, (datetime,)): return obj.isoformat()
    elif isinstance(obj, set): return list(obj)
    return str(obj)

def clean_list_for_json(data):
    return json.loads(json.dumps(data, default=make_json_serializable))

# --- Google Sheets logging ---
def send_to_google_sheets(meal_id, user_id, raw_text, entities, matches, prompts):
    url = "https://script.google.com/macros/s/YOUR_SCRIPT_URL/exec"  # Replace!
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
        print("❌ Logging failed:", e)

# --- Highlighting ---
def highlight_transcript(text, entities):
    highlighted = text
    entities_sorted = sorted(entities, key=lambda e: -len(str(e.get("extracted", ""))))
    for ent in entities_sorted:
        food = str(ent.get("extracted", "")).strip()
        quantity = str(ent.get("quantity", "")).strip()
        unit = str(ent.get("unit", "")).strip()

        if quantity and unit:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\s+{re.escape(unit)}\b",
                                 rf'<span style="background-color:#40e0d0;">\g<0></span>', highlighted)
        if quantity:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\b",
                                 rf'<span style="background-color:#40e0d0;">\g<0></span>', highlighted)
        if food:
            highlighted = re.sub(rf"\b{re.escape(food)}\b",
                                 rf'<span style="background-color:#90ee90;">\g<0></span>', highlighted, flags=re.IGNORECASE)
    return highlighted

# --- Streamlit UI ---
st.set_page_config(page_title="PATHMATE - Meal Logging Demo", layout="centered")
st.title("Pathmate Speech to Text Demo")
st.caption("Upload your meal voice log (.mp3, .wav, .ogg, .mp4) to get a transcription.")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if tmp_path.endswith((".ogg", ".wav", ".mp4")):
        audio = AudioSegment.from_file(tmp_path)
        tmp_path = tmp_path + ".converted.mp3"
        audio.export(tmp_path, format="mp3")

    wav_path = tmp_path.replace(".mp3", ".wav")
    AudioSegment.from_file(tmp_path).export(wav_path, format="wav")
    st.audio(wav_path, format="audio/wav")

    with st.spinner("Transcribing..."):
        transcript = transcribe_with_openai(tmp_path)

    st.subheader("Transcript")
    st.write(transcript)

    with st.spinner("Extracting food entities..."):
        food_entities, raw_llm = extract_food_entities(transcript)
        st.subheader("Raw LLM Output")
        st.code(raw_llm)
        st.markdown("Extracted entities:")
        st.write(food_entities)

    # Early highlighting before clarification
    st.markdown("Initial Highlight (from LLM):")
    st.markdown(highlight_transcript(transcript, food_entities), unsafe_allow_html=True)

    # --- Parallel track: quantity/unit clarification + DB matching ---
    st.subheader("Clarify quantities / units + Match foods")

    clarified_entities = []
    clarification_prompts = []
    matched_entities = []

    csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
    food_db = load_food_database(csv_path)

    for entity in food_entities:
        extracted = entity.get("extracted", "")
        quantity = entity.get("quantity")
        unit = entity.get("unit")

        # Fix basic types
        if isinstance(quantity, str) and quantity.lower() in ["a", "an"]:
            quantity = 1
        if isinstance(quantity, str) and quantity.lower() in ["some", "few", "several"]:
            quantity = None

        # Ask for quantity if missing
        if quantity in [None, ""]:
            quantity = st.number_input(
                f"How much {extracted}? (e.g. 100, 2)",
                min_value=0.0, step=1.0,
                key=f"q_input_{extracted}"
            )
            clarification_prompts.append({
                "extracted": extracted,
                "asked_for": "quantity",
                "response": quantity
            })

        # Default unit
        if not unit or unit.strip() == "":
            unit = "portion"

        clarified = {"extracted": extracted, "quantity": quantity, "unit": unit}
        clarified_entities.append(clarified)

        # Try DB match
        match = match_entity(clarified, food_db)

        # If not found, ask user
        if not match["recognized"] or match["ID"] is None:
            correction = st.text_input(
                f"Food '{extracted}' not recognized. What is it?",
                key=f"match_correction_{extracted}"
            )
            if correction:
                corrected = match_entity({"extracted": correction}, food_db)
                corrected["quantity"] = quantity
                corrected["unit"] = unit
                matched_entities.append(corrected)
            else:
                matched_entities.append(match)
        else:
            matched_entities.append(match)

    # --- Final view ---
    st.subheader("Matched Results")
    df = pd.DataFrame(matched_entities)
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

    # --- Save to Sheets ---
    send_to_google_sheets(
        meal_id=f"meal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        user_id="anon_user",
        raw_text=transcript,
        entities=clarified_entities,
        matches=clean_list_for_json(matched_entities),
        prompts=clarification_prompts
    )

    # --- Download buttons ---
    st.download_button(
        "Download JSON",
        data=json.dumps(matched_entities, indent=2, default=make_json_serializable),
        file_name="meal_log.json",
        mime="application/json"
    )

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="meal_log.csv",
        mime="text/csv"
    )
