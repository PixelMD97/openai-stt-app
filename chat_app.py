import streamlit as st
from openai_stt import transcribe_with_openai
from entity_extractor import extract_food_entities
from swiss_food_matcher import load_food_database, match_entity
from core import highlight_transcript, find_potential_foods_simple, make_json_serializable, clean_list_for_json
import tempfile, os, json, pandas as pd
from datetime import datetime
from pydub import AudioSegment

st.set_page_config(page_title="ChatBot Meal Logger", layout="centered")
st.title("Pathmate - Chat-Based Meal Logger")

csv_path = os.path.join(os.path.dirname(__file__), "swiss_food_composition_database_small.csv")
food_db = load_food_database(csv_path)

chat_history = st.session_state.get("chat_history", [])
input_mode = st.radio("Input mode", ["Text", "Voice"])

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

    st.download_button("Download JSON", json.dumps(matched_entities, indent=2), "meal_log.json")
    st.download_button("Download CSV", df.to_csv(index=False), "meal_log.csv")
