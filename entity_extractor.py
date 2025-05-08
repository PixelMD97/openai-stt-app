import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = """
Extract food items from the following meal description. For each, return:
- name
- quantity
- unit (like g, ml, piece)

Respond as a Python list of dicts with keys: name, quantity, unit.

Text:
"{text}"
"""

def extract_food_entities(text: str) -> list:
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful nutrition assistant."},
            {"role": "user", "content": PROMPT.format(text=text)}
        ],
        temperature=0.2
    )
    content = response.choices[0].message.content.strip()
    try:
        return eval(content)
    except Exception as e:
        return []
