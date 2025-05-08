import openai
import json

def extract_food_entities(text: str):
    PROMPT_TEMPLATE = """
Extract a list of food items and amounts from the following meal description.
Return it as JSON list of dictionaries with keys:
- "extracted"
- "quantity"
- "unit"

Text:
{text}
"""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful nutrition assistant."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
        ],
        temperature=0.2
    )

    raw_response = response.choices[0].message.content.strip()
    print("🧠 LLM Raw Response:\n", raw_response)

    try:
        extracted_entities = json.loads(raw_response)
    except Exception as e:
        extracted_entities = []

    return extracted_entities, raw_response
