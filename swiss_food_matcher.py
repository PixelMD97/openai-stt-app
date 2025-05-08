from rapidfuzz import process
import pandas as pd

def load_food_database(csv_path):
    food_db = pd.read_csv(csv_path)
    food_db["name_lower"] = food_db["name"].str.lower().str.strip()
    return food_db

def match_entity(entity, food_db):
    extracted_name = entity["extracted"].strip().lower()

    match_name, score, idx = process.extractOne(
        extracted_name,
        food_db["name_lower"],
        score_cutoff=80  # adjust if needed
    )

    if match_name:
        matched_row = food_db[food_db["name_lower"] == match_name].iloc[0]
        return {
            "extracted": entity["extracted"],
            "recognized": matched_row["name"],
            "quantity": entity.get("quantity"),
            "unit": entity.get("unit"),
            "ID": matched_row["ID"]
        }

    return {
        "extracted": entity["extracted"],
        "recognized": None,
        "quantity": entity.get("quantity"),
        "unit": entity.get("unit"),
        "ID": None
    }
