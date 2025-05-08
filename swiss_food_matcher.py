import pandas as pd
from rapidfuzz import process

def load_food_database(csv_path):
    df = pd.read_csv(csv_path)
    df["name_clean"] = df["name"].str.lower().str.strip()
    return df

def match_entity(entity, food_db):
    extracted = entity.get("extracted", "").lower().strip()
    quantity = entity.get("quantity")
    unit = entity.get("unit")

    match_name, score, idx = process.extractOne(
        extracted,
        food_db["name_clean"],
        score_cutoff=70  # Lower to catch minor variations
    )

    if match_name:
        match_row = food_db[food_db["name_clean"] == match_name].iloc[0]
        return {
            "extracted": entity.get("extracted"),
            "recognized": match_row["name"],
            "quantity": quantity,
            "unit": unit,
            "ID": match_row["ID"]
        }

    # Fallback if no match
    return {
        "extracted": entity.get("extracted"),
        "recognized": None,
        "quantity": quantity,
        "unit": unit,
        "ID": None
    }
