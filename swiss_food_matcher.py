# swiss_food_matcher.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model only once (global variable)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and embed the food database
def load_food_database(csv_path):
    df = pd.read_csv(csv_path)
    df["name_clean"] = df["name"].str.lower().str.strip()
    df["embedding"] = model.encode(df["name_clean"].tolist(), convert_to_tensor=True).tolist()

    return df

# Match a single food entity to the database using semantic similarity
def match_entity(entity, food_db):
    input_text = entity["extracted"].strip().lower()
    input_embedding = model.encode(input_text, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(input_embedding, food_db["embedding"].tolist())[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    matched = food_db.iloc[best_idx]

    return {
        "extracted": entity["extracted"],
        "recognized": matched["name"],
        "quantity": entity.get("quantity"),
        "unit": entity.get("unit"),
        "ID": matched["ID"],
        "score": round(best_score, 3)
    }
