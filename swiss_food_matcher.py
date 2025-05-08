from rapidfuzz import fuzz
import pandas as pd

def load_food_database(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def match_entity(entity: dict, food_db: pd.DataFrame) -> dict:
    best_match = None
    best_score = 0

    for _, row in food_db.iterrows():
        score = fuzz.partial_ratio(entity["extracted"].lower(), row["name"].lower())
        if score > best_score:
            best_score = score
            best_match = row

    result = {
        "extracted": entity.get("extracted"),
        "quantity": entity.get("quantity"),
        "unit": entity.get("unit"),
        "recognized": best_match["name"] if best_score > 70 else "No match",
        "ID": best_match["ID"] if best_score > 70 else None
    }

    return result
