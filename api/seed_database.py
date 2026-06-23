# =============================================================
#  api/seed_database.py  —  LOAD CSV DATA INTO THE DATABASE
#
#  WHY THIS FILE EXISTS:
#    train_model.py still creates logs/predictions_log.csv exactly
#    as before (UNCHANGED). This script takes that CSV and inserts
#    it into the SQL database, so the FastAPI backend has data to
#    serve. This is the bridge between Version 1 (CSV) and
#    Version 2 (SQL + API).
#
#  HOW TO RUN (in order):
#    1. python train_model.py        ← creates predictions_log.csv
#    2. python api/seed_database.py  ← loads it into the database
#    3. uvicorn api.main:app --reload
#    4. streamlit run app.py
# =============================================================

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.database import SessionLocal, Prediction, init_db


def seed_from_csv(csv_path="logs/predictions_log.csv"):
    print("=" * 55)
    print("  Seeding database from predictions_log.csv")
    print("=" * 55)

    # Step 1 — make sure the table exists
    init_db()

    # Step 2 — load the CSV (same file train_model.py already creates)
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "logs", "predictions_log.csv")

    print(f"\n[1/3] Loaded {len(df)} rows from {csv_path}")

    # Step 3 — normalize column names to match the database schema
    #   predictions_log.csv  →  predictions table
    #   Probability          →  score
    #   Gender                →  group_name
    #   Attrition            →  actual
    #   Timestamp            →  timestamp
    df["timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%y", errors="coerce")
    df["score"]      = df["Probability"]
    df["group_name"] = df["Gender"].astype(str).str.strip().str.capitalize()
    df["actual"]      = df["Attrition"]

    df = df.dropna(subset=["timestamp", "score", "actual", "group_name"])
    print(f"[2/3] Normalized columns → score, actual, group_name, timestamp")

    # Step 4 — insert into database (clears old rows first to avoid duplicates)
    db = SessionLocal()
    try:
        deleted = db.query(Prediction).delete()
        if deleted:
            print(f"      Cleared {deleted} old rows from database")

        records = [
            Prediction(
                timestamp  = row["timestamp"],
                score      = float(row["score"]),
                actual     = int(row["actual"]),
                group_name = row["group_name"],
            )
            for _, row in df.iterrows()
        ]
        db.bulk_save_objects(records)
        db.commit()
        print(f"[3/3] ✅ Inserted {len(records)} rows into the database")
    finally:
        db.close()

    print("\n" + "=" * 55)
    print("  ✅ Database seeding complete!")
    print("  Now run:  uvicorn api.main:app --reload")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    seed_from_csv()
