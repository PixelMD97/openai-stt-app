import pandas as pd
from rapidfuzz import process, fuzz

def load_food_database(csv_path: str):
    return pd.read_csv(csv_path)

def match_food(food_name: str, food_df: pd.DataFrame, limit=1):
    choices = food_df["name"].tolist()
    matches = process.extract(food_name, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    results = []
    for match_name, score, index in matches:
        row = food_df.iloc[index]
        results.append({
            "recognized": match_name,
            "ID": row["ID"],
            "score": score
        })
    return results

def match_entity(entity: dict, food_df: pd.DataFrame) -> dict:
    best_match = match_food(entity["name"], food_df, limit=1)[0]
    return {
        "extracted": f"{entity['name']} {entity['quantity']} {entity['unit']}",
        "recognized": best_match["recognized"],
        "quantity": entity["quantity"],
        "unit": entity["unit"],
        "ID": best_match["ID"]
    }
