# =============================================================
#  train_model.py  —  MODEL TRAINING SCRIPT
#
#  Dataset: hr_dataset.csv (999 rows, 13 columns)
#  Columns: Age, Gender, Department, JobRole, Education,
#           MonthlyIncome, YearsAtCompany, JobSatisfaction,
#           OverTime, DistanceFromHome, Attrition,
#           Probability, Timestamp
#
#  HOW TO RUN:
#    python train_model.py
#
#  WHAT IT CREATES:
#    model/model.pkl           ← trained Logistic Regression
#    model/scaler.pkl          ← StandardScaler
#    model/feature_names.pkl   ← list of feature columns
#    logs/predictions_log.csv  ← predictions for dashboard
# =============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("   Bias Drift Detector — Model Training")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
#  STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading hr_dataset.csv ...")

df = pd.read_csv("data/hr_dataset.csv")
print(f"      Rows: {len(df)}  |  Columns: {len(df.columns)}")
print(f"      Columns: {list(df.columns)}")
print(f"      Attrition distribution: {df['Attrition'].value_counts().to_dict()}")
print(f"      Gender distribution:    {df['Gender'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
#  STEP 2 — SAVE GENDER & TIMESTAMP BEFORE ENCODING
#  (we need original text values for the predictions log)
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Saving original columns before encoding ...")

gender_text = df["Gender"].astype(str).str.strip().str.capitalize().copy()
timestamp   = df["Timestamp"].copy() if "Timestamp" in df.columns else pd.Series(["2025-01-01"] * len(df))

# ─────────────────────────────────────────────────────────────
#  STEP 3 — PREPARE FEATURES
#
#  Features we use to train:
#    Age, Education, MonthlyIncome, YearsAtCompany,
#    JobSatisfaction, OverTime (encoded),
#    DistanceFromHome, Gender_Label,
#    Department_* (one-hot), JobRole_* (one-hot)
#
#  We do NOT use: Probability, Timestamp
#  (those are outputs, not inputs)
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Preparing features ...")

# Target
y = df["Attrition"].copy()   # already 0/1

# Drop columns that are outputs or identifiers
raw_features = df.drop(columns=["Attrition", "Probability", "Timestamp"], errors="ignore")

# Encode Gender: Male→0, Female→1
raw_features["Gender_Label"] = raw_features["Gender"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)
raw_features = raw_features.drop(columns=["Gender"])

# Encode OverTime: No→0, Yes→1
raw_features["OverTime"] = raw_features["OverTime"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

# One-hot encode Department and JobRole
raw_features = pd.get_dummies(raw_features, columns=["Department", "JobRole"], drop_first=False)

# Fill any remaining NaN
raw_features = raw_features.fillna(0)

# Convert all bool columns to int (get_dummies can produce bool)
for col in raw_features.columns:
    if raw_features[col].dtype == bool:
        raw_features[col] = raw_features[col].astype(int)

X = raw_features.copy()
feature_names = X.columns.tolist()

print(f"      Features: {len(feature_names)}")
print(f"      Feature list: {feature_names}")

# ─────────────────────────────────────────────────────────────
#  STEP 4 — TRAIN / TEST SPLIT & SCALE
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Splitting and scaling ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # keeps same class ratio in both splits
)
print(f"      Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
#  STEP 5 — TRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Training Logistic Regression ...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # handles the 912 vs 87 imbalance
    random_state=42,
    solver="lbfgs",
    C=1.0
)
model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
acc    = accuracy_score(y_test, y_pred)

print(f"\n      ✅ Training complete!")
print(f"      Test Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
print(f"\n      Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Attrition", "Attrition"]))
print(f"      Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────────────────────
#  STEP 6 — SAVE MODEL FILES
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Saving model files ...")

os.makedirs("model", exist_ok=True)
joblib.dump(model,         "model/model.pkl")
joblib.dump(scaler,        "model/scaler.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

print("      ✅ Saved → model/model.pkl")
print("      ✅ Saved → model/scaler.pkl")
print("      ✅ Saved → model/feature_names.pkl")

# ─────────────────────────────────────────────────────────────
#  STEP 7 — GENERATE PREDICTIONS LOG
#
#  Run model on FULL dataset (all 999 rows) so the dashboard
#  has enough data for drift-over-time analysis.
#
#  predictions_log.csv columns:
#    Age, Gender, Gender_Label, Education, MonthlyIncome,
#    YearsAtCompany, JobSatisfaction, OverTime,
#    DistanceFromHome, Attrition (actual), Probability,
#    Timestamp, Department_*, JobRole_*, Predicted
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Generating predictions_log.csv ...")

# Predict on full dataset
X_full_scaled  = scaler.transform(X)
full_predicted = model.predict(X_full_scaled)
full_prob      = model.predict_proba(X_full_scaled)[:, 1]

# Build log — keep original columns + add Predicted
log_df = df[["Age", "Gender", "Education", "MonthlyIncome",
             "YearsAtCompany", "JobSatisfaction", "OverTime",
             "DistanceFromHome", "Attrition", "Timestamp"]].copy()

log_df["Gender_Label"] = (log_df["Gender"].map({"Male": 0, "Female": 1})
                                           .fillna(0).astype(int))
log_df["Probability"]  = np.round(full_prob, 9)
log_df["Predicted"]    = full_predicted

# Add one-hot encoded columns that are in feature_names
for col in feature_names:
    if col.startswith("Department_") or col.startswith("JobRole_"):
        log_df[col] = X[col].values

# Reorder columns to match your original predictions_log format
col_order = [
    "Age", "Gender", "Gender_Label", "Education", "MonthlyIncome",
    "YearsAtCompany", "JobSatisfaction", "OverTime", "DistanceFromHome",
    "Attrition", "Probability", "Timestamp"
]
# Add Department and JobRole one-hot columns
dept_cols    = [c for c in log_df.columns if c.startswith("Department_")]
jobrole_cols = [c for c in log_df.columns if c.startswith("JobRole_")]
col_order   += dept_cols + jobrole_cols + ["Predicted"]
log_df = log_df[[c for c in col_order if c in log_df.columns]]

os.makedirs("logs", exist_ok=True)
log_df.to_csv("logs/predictions_log.csv", index=False)

print(f"      ✅ Saved → logs/predictions_log.csv ({len(log_df)} records)")

# Quick check
male_rate   = log_df[log_df["Gender"] == "Male"]["Predicted"].mean()
female_rate = log_df[log_df["Gender"] == "Female"]["Predicted"].mean()
print(f"\n      Male   positive prediction rate: {male_rate:.4f}")
print(f"      Female positive prediction rate: {female_rate:.4f}")
print(f"      DPD (bias level):                {abs(male_rate - female_rate):.4f}")

print(f"\n{'='*60}")
print("  ✅ ALL DONE!")
print("  Now run:   streamlit run app.py")
print(f"{'='*60}\n")
