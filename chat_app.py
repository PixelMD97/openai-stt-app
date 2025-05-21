import streamlit as st
from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity
import tempfile, os, json, pandas as pd
from datetime import datetime
from pydub import AudioSegment
from word2number import w2n
import numpy as np
import re

# --- Helpers inlined here

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

# --- App config
st.set_page_config(page_title="ChatBot Meal Logger", layout="centered")
st.title("Pathmate - Chat-Based Meal Logger")

with st.chat_message("assistant"):
    st.markdown("""
    👋 Hello! This is a **prototype demo built at FHNW for Pathmate Technologies**.

    🧠 The goal is to illustrate how voice or chat input can be turned into **structured meal logging** using AI.

    👉 You can tell me what you ate today, or upload a voice recording — and I’ll extract food items, quantities, and units for you.

    ⚠️ This is an early demo, not a medical device. Results may be imperfect and are meant for research/demo purposes only.
    """)

# --- Load database
csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
food_db = load_food_database(csv_path)

chat_history = st.session_state.get("chat_history", [])
input_mode = st.radio("Input mode", ["Text", "Voice"])

# --- Handle input
if input_mode == "Voice":
    voice_file = st.file_uploader("Speak your meal...", type=["mp3", "wav", "ogg"])
    if voice_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(voice_file.read())
            tmp_path = tmp.name
        if tmp_path.endswith((".ogg", ".wav")):
            tmp_path = tmp_path + ".converted.mp3"
            audio = AudioSegment.from_file(tmp.name)
            audio.export(tmp_path, format="mp3")
        transcript = transcribe_with_openai(tmp_path)
        st.session_state["chat_history"] = chat_history + [f": {transcript}"]
else:
    user_input = st.text_input("What did you eat?")
    if user_input:
        transcript = user_input
        st.session_state["chat_history"] = chat_history + [f": {user_input}"]

# --- Main processing
if "transcript" in locals():
    with st.spinner("Extracting foods..."):
        food_entities, raw_output = extract_food_entities(transcript)

    # Fallback detection
    known_words = set(pd.read_csv("csv_foods.csv")["food_name"].str.lower().str.strip())
    fallback_foods = find_potential_foods_simple(transcript, known_words, food_entities)

    clarified_entities = []
    matched_entities = []

    for food in food_entities:
        extracted = food["extracted"]
        quantity = food.get("quantity") or st.number_input(f"How much {extracted}?", min_value=0.0)
        unit = food.get("unit") or st.text_input(f"Unit for {extracted}?", value="portion")

        clarified = {"extracted": extracted, "quantity": quantity, "unit": unit}
        clarified_entities.append(clarified)

        match = match_entity(clarified, food_db)
        matched_entities.append(match)

    # Show output
    st.session_state["chat_history"] += [f": Detected {len(matched_entities)} food items."]
    for msg in st.session_state["chat_history"]:
        st.markdown(msg)

    st.subheader("Matched Table")
    df = pd.DataFrame(matched_entities)
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

    st.subheader("Highlighted Transcript")
    st.markdown(highlight_transcript(transcript, clarified_entities), unsafe_allow_html=True)

    st.download_button("Download JSON", json.dumps(clean_list_for_json(matched_entities), indent=2), "meal_log.json")
    st.download_button("Download CSV", df.to_csv(index=False), "meal_log.csv")
