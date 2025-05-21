import streamlit as st
import os
import tempfile
import subprocess
import json
import pandas as pd
from datetime import datetime
import numpy as np
import re
from word2number import w2n

from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity

# --- Helper functions (no external core.py)
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

def make_json_serializable(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, (datetime,)): return obj.isoformat()
    elif isinstance(obj, set): return list(obj)
    return str(obj)

def clean_list_for_json(data):
    return json.loads(json.dumps(data, default=make_json_serializable))

def find_potential_foods_simple(transcript, known_food_words, extracted_entities):
    transcript_words = set(re.findall(r'\b\w+\b', transcript.lower()))
    extracted = {ent["extracted"].lower() for ent in extracted_entities}
    return [word for word in transcript_words if word in known_food_words and word not in extracted]

def convert_to_mp3(input_path, output_path):
    try:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], check=True)
    except subprocess.CalledProcessError as e:
        st.error("Audio conversion failed.")
        raise e

# --- App Config
now = datetime.now().strftime("%Y-%m-%d %H:%M")
st.set_page_config(page_title="Pathmate Chat {now}", layout="centered")
st.title("Pathmate - Chat-Based Meal Logger")

with st.chat_message("assistant"):
    st.markdown("""
    Hello! This is a **prototype demo built at FHNW in collaboration with Pathmate**. The goal is to illustrate how voice or chat input can be turned into **structured meal logging** using AI.

    👉 You can tell me what you ate today, or upload a voice recording — and I’ll extract food items, quantities, and units for you. 

    note: This is an early demo, not a medical device. Results may be imperfect and are meant for research/demo purposes only.
    """)

# --- Load food database
csv_foods_path = os.path.join(os.path.dirname(__file__), "csv_foods.csv")
csv_db_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
KNOWN_FOOD_WORDS = set(pd.read_csv(csv_foods_path)['food_name'].str.lower().str.strip())
FOOD_DB = load_food_database(csv_db_path)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

# --- Input mode
input_mode = st.radio("Choose input method:", ["💬 Chat", "🎤 Voice"], horizontal=True)

def handle_transcript(transcript):
    st.chat_message("user").markdown(transcript)
    st.session_state.chat_history.append(("user", transcript))
    st.session_state.last_transcript = transcript

    with st.spinner("Extracting food entities..."):
        entities, raw_llm = extract_food_entities(transcript)

    fallback = find_potential_foods_simple(transcript, KNOWN_FOOD_WORDS, entities)
    for food in fallback:
        if st.checkbox(f"Include fallback '{food}' (no quantity)?"):
            entities.append({"extracted": food, "quantity": None, "unit": None})

    clarified_entities = []
    matched_entities = []

    st.chat_message("assistant").markdown("Here's what I found:")

    for entity in entities:
        extracted = entity.get("extracted")
        quantity = entity.get("quantity")
        unit = entity.get("unit")

        if quantity in ["a", "an", "one"]:
            quantity = 1
        elif quantity in ["some", "few", "several"]:
            quantity = None

        if quantity in [None, ""]:
            quantity = st.number_input(f"How much {extracted}?", min_value=0.0, key=f"q_{extracted}")
        if not unit or unit.strip() == "":
            unit = "portion"

        clarified = {"extracted": extracted, "quantity": quantity, "unit": unit}
        clarified_entities.append(clarified)

        match = match_entity(clarified, FOOD_DB)
        if not match["recognized"] or match["ID"] is None:
            correction = st.text_input(f"Couldn't match '{extracted}'. What is it?", key=f"corr_{extracted}")
            if correction:
                match = match_entity({"extracted": correction, "quantity": quantity, "unit": unit}, FOOD_DB)
        matched_entities.append(match)

    df = pd.DataFrame(matched_entities)
    highlighted = highlight_transcript(transcript, clarified_entities)

    st.subheader("📋 Matched Table")
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

    st.subheader("📝 Highlighted Transcript")
    st.markdown(highlighted, unsafe_allow_html=True)

    st.chat_message("assistant").markdown(f"✅ Logged {len(matched_entities)} items successfully.")

    st.download_button("📥 Download JSON", json.dumps(clean_list_for_json(matched_entities), indent=2),
                       file_name="meal_log.json", mime="application/json")

    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="meal_log.csv", mime="text/csv")

# --- Text or voice input flow
if input_mode == "💬 Chat":
    user_message = st.chat_input("What did you eat today?")
    if user_message:
        handle_transcript(user_message)

elif input_mode == "🎤 Voice":
    voice_file = st.file_uploader("Upload your voice log", type=["mp3", "wav", "ogg", "mp4"])
    if voice_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(voice_file.name)[1]) as tmp:
            tmp.write(voice_file.read())
            tmp_path = tmp.name

        converted_path = tmp_path + ".converted.mp3"
        convert_to_mp3(tmp_path, converted_path)

        with st.spinner("Transcribing audio..."):
            transcript = transcribe_with_openai(converted_path)

        handle_transcript(transcript)
