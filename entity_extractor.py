import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
Extract a list of food items with quantity and unit from the meal description below.

Return the result strictly as a JSON list like:
[
  {{ "food": "pizza", "quantity": 3, "unit": "slice" }},
  {{ "food": "tomato", "quantity": 1, "unit": "whole" }}
]

Meal description:
"{text}"
"""

def extract_food_entities(text: str) -> list:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful nutrition assistant."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
            ],
            temperature=0.2
        )

        # 👇 ADD THIS
        print("LLM Raw Response:\n", response.choices[0].message.content.strip())

        content = response.choices[0].message.content.strip()

        # Try parsing JSON directly
        return json.loads(content)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return []


response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful nutrition assistant."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
    ],
    temperature=0.2
)

# 👇 ADD THIS
print("🧠 LLM Raw Response:\n", response.choices[0].message.content.strip())
