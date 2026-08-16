"""Loads and looks up the treatment recommendation database."""
import json
import os

_DB_PATH = os.path.join(os.path.dirname(__file__), "treatment_db.json")

with open(_DB_PATH) as _f:
    _TREATMENT_DB = json.load(_f)


def get_treatment(category: str) -> dict:
    """category is the full class name, e.g. 'Tomato_Early_blight'."""
    if category not in _TREATMENT_DB:
        raise KeyError(f"No treatment entry for category '{category}'")
    return _TREATMENT_DB[category]
