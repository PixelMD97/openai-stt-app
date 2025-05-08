import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

EXTRACTION_PROMPT = """
Extract the food entities and quantities from the following meal description. Return them as a JSON list with keys: "name", "quantity", and "unit". If unit is missing, guess it (e.g., g for solids, ml for liquids).

Meal description:
"{text}"
"""

def extract_food_entities(text: str) -> list:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful nutrition assistant."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}
        ],
        temperature=0.2
    )
    content = response.choices[0].message.content.strip()
