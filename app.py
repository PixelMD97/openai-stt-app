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

# --- Load known food words from CSV ---
def load_known_food_words(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return set(df['food_name'].str.lower().str.strip())
    except Exception as e:
        print("⚠️ Failed to load known food words:", e)
        return set()

csv_foods_path = os.path.join(os.path.dirname(__file__), "csv_foods.csv")
KNOWN_FOOD_WORDS = load_known_food_words(csv_foods_path)

# --- JSON helpers ---
def make_json_serializable(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, (datetime,)): return obj.isoformat()
    elif isinstance(obj, set): return list(obj)
    return str(obj)

def clean_list_for_json(data):
    return json.loads(json.dumps(data, default=make_json_serializable))

# --- Fallback food detection ---
def find_potential_foods_simple(transcript, known_food_words, extracted_entities):
    transcript_words = set(re.findall(r'\b\w+\b', transcript.lower()))
    extracted = {ent["extracted"].lower() for ent in extracted_entities}
    return [word for word in transcript_words if word in known_food_words and word not in extracted]

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

from word2number import w2n

def normalize_numbers(text):
    words = text.split()
    converted = []
    for word in words:
        try:
            converted.append(str(w2n.word_to_num(word)))
        except:
            converted.append(word)
    return ' '.join(converted)


def highlight_transcript(text, entities):
    text = normalize_numbers(text)  
    highlighted = text
    entities_sorted = sorted(entities, key=lambda e: -len(str(e.get("extracted", ""))))
    vague_terms = {"some", "few", "several", "a", "an"}

    for ent in entities_sorted:
        food = str(ent.get("extracted", "")).strip()
        quantity = str(ent.get("quantity", "")).strip()
        unit = str(ent.get("unit", "")).strip()

        if quantity.lower() in vague_terms:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\b",
                                 rf'<span style="background-color:#ffff99;">\g<0></span>', highlighted, flags=re.IGNORECASE)
        elif quantity and unit:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\s+{re.escape(unit)}\b",
                                 rf'<span style="background-color:#40e0d0;">\g<0></span>', highlighted, flags=re.IGNORECASE)
        elif quantity:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\b",
                                 rf'<span style="background-color:#40e0d0;">\g<0></span>', highlighted, flags=re.IGNORECASE)

        if food:
            highlighted = re.sub(rf"\b{re.escape(food)}\b",
                                 rf'<span style="background-color:#90ee90;">\g<0></span>', highlighted, flags=re.IGNORECASE)
    return highlighted


# --- Streamlit UI ---
now = datetime.now().strftime("%Y-%m-%d %H:%M")
st.set_page_config(page_title=f"PATHMATE - Meal Logging {now}", layout="centered")
st.title(f"Pathmate Speech to Text Demo ({now})")
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

        # CSV-based fallback detection
        missing_foods = find_potential_foods_simple(transcript, KNOWN_FOOD_WORDS, food_entities)
        for food in missing_foods:
            if st.checkbox(f"Include '{food}' even though no quantity was mentioned?"):
                food_entities.append({
                    "extracted": food,
                    "quantity": None,
                    "unit": None
                })

    # --- Clarification & Matching ---
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

        if isinstance(quantity, str):
            q_lower = quantity.lower()
            if q_lower in ["a", "an", "one"]:
                quantity = 1
            elif q_lower in ["some", "few", "several"]:
                quantity = None

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

        if not unit or unit.strip() == "":
            unit = "portion"

        clarified = {"extracted": extracted, "quantity": quantity, "unit": unit}
        clarified_entities.append(clarified)

        match = match_entity(clarified, food_db)
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

    # --- Results ---
    st.subheader("Matched Results")
    df = pd.DataFrame(matched_entities)
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

    # --- Highlighted Transcript ---
    st.subheader("Final Highlighted Transcript")
    normalized_transcript = normalize_numbers(transcript)
    st.markdown(highlight_transcript(normalized_transcript, clarified_entities), unsafe_allow_html=True)



    st.markdown("""
    <div style='padding-top: 10px; font-size: 14px;'>
    <b>🟩 Green</b>: Food items (e.g., <i>pizza</i>)<br>
    <b>🗭 Blue</b>: Quantity and unit (e.g., <i>3 slices</i>)<br>
    <b>🟨 Yellow</b>: Vague/uncertain quantities (e.g., <i>some</i>, <i>a</i>, <i>few</i>)
    </div>
    """, unsafe_allow_html=True)

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
