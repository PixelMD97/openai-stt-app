# swiss_food_matcher.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_food_database(csv_path):
    df = pd.read_csv(csv_path)
    df["name_clean"] = df["name"].str.strip().str.lower()
    df["embedding"] = model.encode(df["name_clean"].tolist(), convert_to_tensor=True).tolist()
    return df

def match_entity(entity, food_db, threshold=0.7):
    input_text = entity["extracted"].strip().lower()
    input_embedding = model.encode(input_text, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(input_embedding, food_db["embedding"].tolist())[0]
    top_idx = scores.argmax().item()
    top_score = scores[top_idx].item()

    if top_score >= threshold:
        matched = food_db.iloc[top_idx]
        return {
            "extracted": entity["extracted"],
            "recognized": matched["name"],
            "quantity": entity.get("quantity"),
            "unit": entity.get("unit"),
            "ID": matched["ID"],
            "score": round(top_score, 3)
        }
    else:
        return {
            "extracted": entity["extracted"],
            "recognized": None,
            "quantity": entity.get("quantity"),
            "unit": entity.get("unit"),
            "ID": None,
            "score": round(top_score, 3)
        }
