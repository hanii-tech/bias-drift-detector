# =============================================================
#  data_loader.py  —  API DATA LOADER + NORMALIZER
#
#  This is the bridge between the new FastAPI backend and your
#  EXISTING utils.py (which is completely unchanged).
#
#  utils.py expects a DataFrame with columns:
#    Gender, probability, predicted, actual, timestamp
#
#  But the database/API gives us:
#    group_name, score, actual, timestamp
#
#  This file converts one format into the other BEFORE the data
#  ever reaches utils.py — so utils.py never needs to know the
#  API/database exists.
# =============================================================

import pandas as pd
import requests
import config


def fetch_predictions_from_api(limit=None):
    """
    Calls the FastAPI /predictions endpoint and returns the raw
    JSON response as a pandas DataFrame (still in database format
    at this point — group_name, score, actual).

    Raises an exception if the API cannot be reached, so app.py
    can show a clear error message to the user.
    """
    limit = limit or config.PREDICTIONS_LIMIT

    response = requests.get(
        config.PREDICTIONS_ENDPOINT,
        params={"limit": limit},
        timeout=60,
    )
    response.raise_for_status()   # raises an error if API call failed

    data = response.json()
    rows = data.get("predictions", [])

    if not rows:
        raise ValueError(
            "API returned 0 predictions. "
            "Did you run 'python api/seed_database.py' to load data?"
        )

    return pd.DataFrame(rows)


def normalize_api_data(raw_df):
    """
    Converts database/API column names into the names
    utils.py already expects. This is the ONLY place where
    the score → prediction and group_name → Gender conversion
    happens.

    Database format          →   utils.py format
    ─────────────────────────────────────────────
    score        (0.0–1.0)   →   probability
    score >= 0.5             →   predicted (0 or 1)
    actual       (0/1)       →   actual
    group_name   (text)      →   Gender
    timestamp                →   Timestamp
    """
    df = raw_df.copy()

    # score → probability (same values, just renamed)
    df["Probability"] = pd.to_numeric(df["score"], errors="coerce")

    # score → predicted, using the threshold from config.py
    df["Predicted"] = (df["Probability"] >= config.SCORE_TO_PREDICTION_THRESHOLD).astype(int)

    # actual stays the same, just renamed to match expected casing
    df["Attrition"] = pd.to_numeric(df["actual"], errors="coerce")

    # group_name → Gender
    df["Gender"] = df["group_name"].astype(str).str.strip().str.capitalize()

    # timestamp stays the same, just renamed to match expected casing
    df["Timestamp"] = df["timestamp"]

    # Keep only the columns utils.load_predictions() expects,
    # in the same shape as the old predictions_log.csv
    final_df = df[["Gender", "Attrition", "Predicted", "Probability", "Timestamp"]].copy()

    return final_df


def load_predictions_from_api(limit=None):
    """
    ONE FUNCTION that does the full job:
      1. Calls the API
      2. Normalizes the response
      3. Returns a DataFrame ready to be saved as a temp CSV
         or passed straight into utils.load_predictions()-style logic

    This is what app.py calls instead of pd.read_csv().
    """
    raw_df   = fetch_predictions_from_api(limit=limit)
    norm_df  = normalize_api_data(raw_df)
    return norm_df
