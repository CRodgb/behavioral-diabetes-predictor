# utils/defaults.py

'''
1. Load the composite CSV once
2. Compute sane defaults (median for numeric, mode for categorical)
3. Provide choices for selectboxes
'''

# --------------------------------
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Project root = one level above utils/
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "BRFSS2024_Composites.csv"

# Load composite dataset
_df = pd.read_csv(DATA_PATH, low_memory=False)

# Ensure key categorical columns are treated as strings
CATEGORICAL_COLS = [
    "State_Name",
    "BMI_Category",
    "Alcohol_Risk_Level",
    "Education_Level",
    "Household_Income_Category",
    "Race_Ethnicity_Group",
    "Biological_Sex",
    "Smoking_Status_Category",
    "Ever_Smoked_100_Cigarettes",
    "Current_Smoking_Frequency",
    "Heavy_Drinking_Flag",
    "Physical_Activity",
]

for col in CATEGORICAL_COLS:
    if col in _df.columns:
        _df[col] = _df[col].astype(str)


def get_training_df() -> pd.DataFrame:
    """Return a copy of the composite training dataframe."""
    return _df.copy()


def compute_defaults() -> Dict[str, Any]:
    """
    Compute default values for each column:
    - numeric: median
    - non-numeric: mode
    """
    df = get_training_df()
    defaults: Dict[str, Any] = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            defaults[col] = None
        elif pd.api.types.is_numeric_dtype(s):
            defaults[col] = float(s.median())
        else:
            defaults[col] = s.mode().iloc[0]
    return defaults


def get_choices(col: str) -> List[Any]:
    """
    Return sorted unique values for a column (for selectboxes).
    """
    df = get_training_df()
    if col not in df.columns:
        return []
    s = df[col].dropna().unique()
    try:
        return sorted(s)
    except TypeError:
        return list(s)
