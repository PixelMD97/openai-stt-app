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
import requests

from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity

# --- Helpers ---
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
    vague_terms = {"some", "few", "several"}

    for ent in entities_sorted:
        food = str(ent.get("extracted", "")).strip()
        quantity = str(ent.get("quantity", "")).strip()
        unit = str(ent.get("unit", "")).strip()

        if quantity.lower() in vague_terms:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\b",
                                 r'<span style="background-color:#ffff99;">\g<0></span>', highlighted, flags=re.IGNORECASE)
        elif quantity and unit:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\s+{re.escape(unit)}\b",
                                 r'<span style="background-color:#40e0d0;">\g<0></span>', highlighted, flags=re.IGNORECASE)
        elif quantity:
            highlighted = re.sub(rf"\b{re.escape(quantity)}\b",
                                 r'<span style="background-color:#40e0d0;">\g<0></span>', highlighted, flags=re.IGNORECASE)

        if food:
            highlighted = re.sub(rf"\b{re.escape(food)}\b",
                                 r'<span style="background-color:#90ee90;">\g<0></span>', highlighted, flags=re.IGNORECASE)
    return highlighted

def make_json_serializable(obj):
    if isinstance(obj, np.generic): return obj.item()
    elif isinstance(obj, datetime): return obj.isoformat()
    elif isinstance(obj, set): return list(obj)
    return str(obj)

def clean_list_for_json(data):
    return json.loads(json.dumps(data, default=make_json_serializable))

def convert_to_mp3(input_path, output_path):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path], check=True)

def send_to_google_sheets(meal_id, user_id, raw_text, entities, matches, prompts):
    url = "https://script.google.com/macros/s/AKfycbwxZT_5PtOTEZpYbOsNINwDUjwHOAw7Nzm21-UfrDZsBsuojWl48wXKO1-Xvrlx_XQ7zA/exec"
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
        st.success("✅ Logged to Google Sheets!")
    except Exception as e:
        st.error("❌ Google Sheets logging failed.")
        st.exception(e)

# --- App config ---
now = datetime.now().strftime("%Y-%m-%d %H:%M")
st.set_page_config(page_title=f"Pathmate Chat - {now}", layout="centered")
st.title("Pathmate - Chat-Based Meal Logger")
st.caption(f"{now}")

with st.chat_message("assistant"):
    st.markdown("""
    Hello! This is a **prototype demo built at FHNW in collaboration with Pathmate**.  
    The goal is to illustrate how voice or chat input can be turned into **structured meal logging** using AI.  
    👉 You can tell me what you ate today, or upload a voice recording — and I’ll extract food items, quantities, and units for you.
    """)

# --- Load Swiss DB ---
db_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
FOOD_DB = load_food_database(db_path)

# --- Session state ---
for key in ["chat_history", "transcript", "pending_entities", "clarified_entities", "matched_entities"]:
    if key not in st.session_state:
        st.session_state[key] = []

# --- Input method ---
input_mode = st.radio("Choose input method:", ["💬 Chat", "🎤 Voice"], horizontal=True)

def process_new_transcript(transcript):
    st.session_state.transcript = transcript
    st.session_state.chat_history.append(("user", transcript))
    with st.spinner("Extracting food items..."):
        entities, _ = extract_food_entities(transcript)
    if not entities:
        st.error("❌ No food or drinks found. Please try again.")
        return
    st.session_state.pending_entities = entities
    st.session_state.clarified_entities = []
    st.session_state.matched_entities = []

def clarify_next_food():
    if "clarification_in_progress" not in st.session_state:
        st.session_state.clarification_in_progress = True
        st.session_state.current_entity = st.session_state.pending_entities.pop(0)

    entity = st.session_state.current_entity
    extracted = entity["extracted"]
    quantity = entity.get("quantity")
    unit = entity.get("unit")

    if isinstance(quantity, str) and quantity.lower() in ["a", "an", "one"]:
        quantity = 1
    elif isinstance(quantity, str) and quantity.lower() in ["some", "few", "several"]:
        quantity = None
        unit = None

    quantity = st.number_input(f"How much {extracted}?", min_value=0.0, key=f"q_{extracted}")
    unit = st.text_input(f"Unit for {extracted}?", value="portion", key=f"unit_{extracted}")
    confirm = st.button(f"Confirm {extracted}", key=f"confirm_{extracted}")

    if confirm:
        clarified = {"extracted": extracted, "quantity": quantity, "unit": unit}
        st.session_state.clarified_entities.append(clarified)

        match = match_entity(clarified, FOOD_DB)
        if match["score"] < 0.7 or not match["recognized"]:
            correction = st.text_input(f"'{extracted}' not recognized. What did you mean?", key=f"corr_{extracted}")
            if correction:
                match = match_entity({"extracted": correction, "quantity": quantity, "unit": unit}, FOOD_DB)

        st.session_state.matched_entities.append(match)
        del st.session_state["clarification_in_progress"]
        del st.session_state["current_entity"]
        st.rerun()




# --- Handle input
if input_mode == "💬 Chat":
    user_message = st.chat_input("What did you eat today?")
    if user_message:
        process_new_transcript(user_message)

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
        process_new_transcript(transcript)

# --- Clarification or Results
if st.session_state.pending_entities:
    clarify_next_food()
elif st.session_state.matched_entities:
    df = pd.DataFrame(st.session_state.matched_entities)
    st.subheader("📋 Matched Table")
    st.dataframe(df[["extracted", "recognized", "quantity", "unit", "ID"]])

    st.subheader("📝 Highlighted Transcript")
    st.markdown(highlight_transcript(st.session_state.transcript, st.session_state.clarified_entities), unsafe_allow_html=True)

    send_to_google_sheets(
        meal_id=f"meal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        user_id="anon_user",
        raw_text=st.session_state.transcript,
        entities=st.session_state.clarified_entities,
        matches=clean_list_for_json(st.session_state.matched_entities),
        prompts=[],
    )

    st.success("✅ Thank you for using the Pathmate Chat-Based Meal Logger!")

    st.download_button("📥 Download JSON",
                       data=json.dumps(clean_list_for_json(st.session_state.matched_entities), indent=2),
                       file_name="meal_log.json", mime="application/json")

    st.download_button("📥 Download CSV",
                       data=df.to_csv(index=False), file_name="meal_log.csv", mime="text/csv")

# --- Footer
st.markdown("""
---
⚠️ *Note: This is an early demo, not a medical device. Results may be imperfect and are meant for research/demo purposes only.*
""", unsafe_allow_html=True)
