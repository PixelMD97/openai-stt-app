# entity_extractor.py

import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a nutrition assistant. Extract food items with quantities and units from the given meal description.
Return a JSON list like this:
[
  {"extracted": "banana", "quantity": 1, "unit": "piece"},
  {"extracted": "milk", "quantity": 200, "unit": "ml"}
]
Only return valid JSON.
"""

def extract_food_entities(transcript):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript}
            ],
            temperature=0,
        )
        content = response["choices"][0]["message"]["content"].strip()
        entities = json.loads(content)
        return entities, content
    except Exception as e:
        return [], f"Error: {str(e)}"
