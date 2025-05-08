def match_entity(entity, food_df):
    name = entity["name"]
    match = match_food(name, food_df, limit=1)[0]
    return {
        "extracted": f"{entity['name']} {entity['quantity']} {entity['unit']}",
        "recognized": match["recognized"],
        "quantity": entity["quantity"],
        "unit": entity["unit"],
        "ID": match["ID"]
    }
