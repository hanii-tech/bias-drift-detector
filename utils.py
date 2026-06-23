# =============================================================
#  utils.py  —  CORE ENGINE for Bias Drift Detector
#
#  All calculations happen here.
#  app.py calls these functions and shows results.
#
#  COLUMNS USED FROM predictions_log.csv:
#    Gender      → 'Male' or 'Female'  (capital G)
#    Attrition   → actual label (0 or 1)
#    Predicted   → model prediction (0 or 1)
#    Probability → confidence score (0.0 to 1.0)
#    Timestamp   → date string e.g. "03-08-25"
# =============================================================

import pandas as pd
import numpy as np
from scipy import stats
import joblib
import shap
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_predictions(filepath="logs/predictions_log.csv"):
    """
    Loads predictions_log.csv.
    Standardizes column names so all functions work correctly.
    """
    df = pd.read_csv(filepath)
    return standardize_predictions_df(df)


def standardize_predictions_df(df):
    """
    VERSION 2 ADDITION — does the exact same column standardization
    as load_predictions(), but accepts a DataFrame that is ALREADY
    in memory instead of reading from a CSV file path.

    This is used by app.py when data comes from the FastAPI backend
    (api/main.py) instead of a local CSV file. load_predictions()
    above is unchanged and still works exactly as before for CSV files.
    """
    df = df.copy()

    # ── Fix Timestamp ──
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], dayfirst=True, errors="coerce"
        )

    # ── Fix Gender column (capital G, your CSV uses "Gender") ──
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()
    elif "Gender_Label" in df.columns:
        # fallback if Gender text column is missing
        df["Gender"] = df["Gender_Label"].map({0: "Male", 1: "Female"})

    # ── Rename to standard internal names ──
    df = df.rename(columns={
        "Attrition":   "actual",
        "Predicted":   "predicted",
        "Probability": "probability",
        "Timestamp":   "timestamp",
    })

    # ── Ensure correct types ──
    df["actual"]      = pd.to_numeric(df["actual"],      errors="coerce")
    df["predicted"]   = pd.to_numeric(df["predicted"],   errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df = df.dropna(subset=["actual", "predicted", "probability"])

    return df


# ─────────────────────────────────────────────────────────────
#  SECTION 2 — LOAD MODEL (for SHAP)
# ─────────────────────────────────────────────────────────────

def load_model():
    """
    Loads the trained Logistic Regression model saved by train_model.py.
    Returns: model, scaler, feature_names
    """
    model         = joblib.load("model/model.pkl")
    scaler        = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return model, scaler, feature_names


# ─────────────────────────────────────────────────────────────
#  SECTION 3 — SPLIT BASELINE vs CURRENT
# ─────────────────────────────────────────────────────────────

def split_baseline_current(df, split_ratio=0.5):
    """
    Splits data chronologically into two halves.
    Baseline = older half (reference)
    Current  = newer half (what is happening now)
    """
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    cut      = int(len(df) * split_ratio)
    baseline = df.iloc[:cut].copy()
    current  = df.iloc[cut:].copy()
    return baseline, current


# ─────────────────────────────────────────────────────────────
#  SECTION 4 — PSI (Population Stability Index)
#
#  PSI < 0.10  → No drift      ✅
#  PSI 0.10–0.25 → Slight drift ⚠️
#  PSI > 0.25  → Major drift   🚨
# ─────────────────────────────────────────────────────────────

def calculate_psi(baseline_scores, current_scores, bins=10):
    """
    Compares probability score distributions of baseline vs current.
    Returns: (psi_total float, breakdown DataFrame)
    """
    baseline_scores = np.array(baseline_scores)
    current_scores  = np.array(current_scores)

    bin_edges      = np.linspace(0, 1, bins + 1)
    bin_edges[0]  -= 1e-6
    bin_edges[-1] += 1e-6

    b_counts, _ = np.histogram(baseline_scores, bins=bin_edges)
    c_counts, _ = np.histogram(current_scores,  bins=bin_edges)

    # Proportions — add tiny value to avoid log(0)
    b_pct = (b_counts + 1e-6) / len(baseline_scores)
    c_pct = (c_counts + 1e-6) / len(current_scores)

    psi_bins  = (c_pct - b_pct) * np.log(c_pct / b_pct)
    psi_total = np.sum(psi_bins)

    breakdown = pd.DataFrame({
        "Bin":         [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(bins)],
        "Baseline %":  np.round(b_pct * 100, 2),
        "Current %":   np.round(c_pct * 100,  2),
        "PSI per bin": np.round(psi_bins, 4),
    })
    return round(float(psi_total), 4), breakdown


def interpret_psi(psi_value):
    if psi_value < 0.10:
        return "✅ No Drift — Distribution is stable"
    elif psi_value < 0.25:
        return "⚠️ Slight Drift — Monitor closely"
    else:
        return "🚨 Major Drift — Bias drift detected!"


# ─────────────────────────────────────────────────────────────
#  SECTION 5 — KS TEST
#
#  Compares Male vs Female probability distributions.
#  p-value < 0.05 → significantly different → possible bias
# ─────────────────────────────────────────────────────────────

def run_ks_test(df):
    """
    KS test: are Male and Female score distributions different?
    Uses Gender column (capital G).
    """
    male_scores   = df[df["Gender"] == "Male"]["probability"].dropna().values
    female_scores = df[df["Gender"] == "Female"]["probability"].dropna().values

    if len(male_scores) < 5 or len(female_scores) < 5:
        return {
            "ks_statistic": None,
            "p_value":      None,
            "result":       "❌ Not enough data for Male or Female group"
        }

    ks_stat, p_value = stats.ks_2samp(male_scores, female_scores)

    result = (
        f"🚨 Significant difference between Male & Female distributions "
        f"(KS={ks_stat:.4f}, p={p_value:.4f}) — Possible bias!"
        if p_value < 0.05 else
        f"✅ No significant difference between groups "
        f"(KS={ks_stat:.4f}, p={p_value:.4f}) — Fair distribution"
    )
    return {
        "ks_statistic": round(ks_stat,  4),
        "p_value":      round(p_value,  4),
        "result":       result
    }


# ─────────────────────────────────────────────────────────────
#  SECTION 6 — FAIRNESS METRICS
#
#  DPD — Demographic Parity Difference  → ideal: < 0.10
#  EOD — Equalized Odds Difference      → ideal: < 0.10
#  DIR — Disparate Impact Ratio         → ideal: 0.80–1.20
#  AOD — Average Odds Difference        → ideal: < 0.10
# ─────────────────────────────────────────────────────────────

def compute_fairness_metrics(df):
    """
    Computes all fairness metrics comparing Male vs Female.
    Uses Gender (capital G), predicted, actual columns.
    """
    results = {}

    male   = df[df["Gender"] == "Male"]
    female = df[df["Gender"] == "Female"]

    # Positive prediction rates
    m_pos = male["predicted"].mean()   if len(male)   > 0 else 0.0
    f_pos = female["predicted"].mean() if len(female) > 0 else 0.0

    results["Male Count"]           = len(male)
    results["Female Count"]         = len(female)
    results["Male Positive Rate"]   = round(m_pos, 4)
    results["Female Positive Rate"] = round(f_pos, 4)

    # ── 1. DPD ──
    dpd = abs(m_pos - f_pos)
    results["Demographic Parity Difference"] = round(dpd, 4)
    results["DPD Status"] = (
        "✅ Fair (< 0.10)"          if dpd < 0.10 else
        "⚠️ Moderate (0.10–0.20)"   if dpd < 0.20 else
        "🚨 Biased (> 0.20)"
    )

    # ── 2. EOD ──
    m_tp = male[male["actual"]     == 1]["predicted"].mean() if len(male[male["actual"]==1])     > 0 else 0.0
    f_tp = female[female["actual"] == 1]["predicted"].mean() if len(female[female["actual"]==1]) > 0 else 0.0
    eod  = abs(m_tp - f_tp)

    results["Male True Positive Rate"]   = round(m_tp, 4)
    results["Female True Positive Rate"] = round(f_tp, 4)
    results["Equalized Odds Difference"] = round(eod,  4)
    results["EOD Status"] = (
        "✅ Fair (< 0.10)"          if eod < 0.10 else
        "⚠️ Moderate (0.10–0.20)"   if eod < 0.20 else
        "🚨 Biased (> 0.20)"
    )

    # ── 3. DIR ──
    dir_r = (f_pos / m_pos) if m_pos > 0 else 0.0
    results["Disparate Impact Ratio"] = round(dir_r, 4)
    results["DIR Status"] = (
        "✅ Fair (0.80–1.20)"           if 0.80 <= dir_r <= 1.20 else
        "🚨 Biased (outside 0.80–1.20)"
    )

    # ── 4. AOD ──
    m_fp = male[male["actual"]     == 0]["predicted"].mean() if len(male[male["actual"]==0])     > 0 else 0.0
    f_fp = female[female["actual"] == 0]["predicted"].mean() if len(female[female["actual"]==0]) > 0 else 0.0
    aod  = abs(((m_tp - f_tp) + (m_fp - f_fp)) / 2)
    results["Average Odds Difference"] = round(aod, 4)

    return results


# ─────────────────────────────────────────────────────────────
#  SECTION 7 — BIAS DRIFT OVER TIME
#
#  KEY PART: split data into time windows, compute DPD per window.
#  If DPD goes UP → bias is drifting!
# ─────────────────────────────────────────────────────────────

def compute_bias_drift_over_time(df, n_windows=5):
    """
    Divides predictions into n time windows.
    Computes DPD per window to track bias over time.
    """
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    window_size = len(df) // n_windows
    records = []

    for i in range(n_windows):
        start  = i * window_size
        end    = start + window_size if i < n_windows - 1 else len(df)
        chunk  = df.iloc[start:end]

        male   = chunk[chunk["Gender"] == "Male"]
        female = chunk[chunk["Gender"] == "Female"]

        m_rate = male["predicted"].mean()   if len(male)   > 0 else 0.0
        f_rate = female["predicted"].mean() if len(female) > 0 else 0.0
        dpd    = abs(m_rate - f_rate)

        if "timestamp" in df.columns and len(chunk) > 0:
            ts    = chunk["timestamp"].dropna()
            label = str(ts.iloc[0].date()) if len(ts) > 0 else f"W{i+1}"
        else:
            label = f"Window {i+1}"

        records.append({
            "Window":      i + 1,
            "Time Label":  label,
            "Male Rate":   round(m_rate, 4),
            "Female Rate": round(f_rate, 4),
            "DPD":         round(dpd,    4),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
#  SECTION 8 — GENDER SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def gender_summary(df):
    """Summary table of Male vs Female statistics."""
    summary = (
        df.groupby("Gender")
        .agg(
            Total_Records    = ("predicted",   "count"),
            Predicted_1      = ("predicted",   "sum"),
            Positive_Rate    = ("predicted",   "mean"),
            Avg_Probability  = ("probability", "mean"),
            Actual_Attrition = ("actual",      "mean"),
        )
        .reset_index()
    )
    for col in ["Positive_Rate", "Avg_Probability", "Actual_Attrition"]:
        summary[col] = summary[col].round(4)
    return summary


# ─────────────────────────────────────────────────────────────
#  SECTION 9 — ALERT SYSTEM
# ─────────────────────────────────────────────────────────────

def generate_alerts(psi_value, ks_result, fairness_metrics, drift_df):
    """
    Checks all metrics and returns list of alerts.
    Each alert: { level: INFO/WARNING/CRITICAL, message: str }
    """
    alerts = []

    # PSI
    if psi_value >= 0.25:
        alerts.append({"level": "CRITICAL",
                        "message": f"PSI = {psi_value} — Major drift! Model needs retraining."})
    elif psi_value >= 0.10:
        alerts.append({"level": "WARNING",
                        "message": f"PSI = {psi_value} — Slight drift in prediction scores."})
    else:
        alerts.append({"level": "INFO",
                        "message": f"PSI = {psi_value} — Prediction distribution is stable."})

    # KS Test
    p = ks_result.get("p_value")
    if p is not None and p < 0.05:
        alerts.append({"level": "CRITICAL",
                        "message": f"KS Test p={p} — Male & Female score distributions significantly different!"})
    else:
        alerts.append({"level": "INFO",
                        "message": "KS Test — No significant difference between Male and Female distributions."})

    # DPD
    dpd = fairness_metrics.get("Demographic Parity Difference", 0)
    if dpd >= 0.20:
        alerts.append({"level": "CRITICAL",
                        "message": f"DPD = {dpd} — HIGH bias! Prediction rates differ by {dpd*100:.1f}% between genders."})
    elif dpd >= 0.10:
        alerts.append({"level": "WARNING",
                        "message": f"DPD = {dpd} — Moderate bias between Male and Female. Review model."})
    else:
        alerts.append({"level": "INFO",
                        "message": f"DPD = {dpd} — Model is fair across genders."})

    # EOD
    eod = fairness_metrics.get("Equalized Odds Difference", 0)
    if eod >= 0.20:
        alerts.append({"level": "CRITICAL",
                        "message": f"EOD = {eod} — Unequal True Positive Rates across Male and Female!"})
    elif eod >= 0.10:
        alerts.append({"level": "WARNING",
                        "message": f"EOD = {eod} — Slight disparity in True Positive Rates."})
    else:
        alerts.append({"level": "INFO",
                        "message": f"EOD = {eod} — Equal True Positive Rates across genders."})

    # DIR
    dir_r = fairness_metrics.get("Disparate Impact Ratio", 1.0)
    if not (0.80 <= dir_r <= 1.20):
        alerts.append({"level": "CRITICAL",
                        "message": f"DIR = {dir_r} — Outside 0.80–1.20 range. Disparate impact detected!"})
    else:
        alerts.append({"level": "INFO",
                        "message": f"DIR = {dir_r} — Within acceptable fair range (0.80–1.20)."})

    # Drift trend
    if len(drift_df) >= 2:
        first = drift_df["DPD"].iloc[0]
        last  = drift_df["DPD"].iloc[-1]
        delta = last - first
        if delta > 0.10:
            alerts.append({"level": "CRITICAL",
                            "message": f"Bias INCREASED over time: DPD {first:.4f} → {last:.4f} (Δ=+{delta:.4f}). Drift confirmed!"})
        elif delta > 0:
            alerts.append({"level": "WARNING",
                            "message": f"Bias slowly increasing: DPD {first:.4f} → {last:.4f} (Δ=+{delta:.4f})."})
        else:
            alerts.append({"level": "INFO",
                            "message": f"Bias stable or improving: DPD {first:.4f} → {last:.4f}."})

    return alerts


# ─────────────────────────────────────────────────────────────
#  SECTION 10 — OVERALL VERDICT
# ─────────────────────────────────────────────────────────────

def overall_verdict(psi_value, fairness_metrics, ks_result):
    """Returns (title, message, color) — single final verdict."""
    critical = 0
    if psi_value >= 0.25:                                                          critical += 1
    if ks_result.get("p_value") and ks_result["p_value"] < 0.05:                  critical += 1
    if fairness_metrics.get("Demographic Parity Difference", 0)  >= 0.20:         critical += 1
    if fairness_metrics.get("Equalized Odds Difference",     0)  >= 0.20:         critical += 1
    if not (0.80 <= fairness_metrics.get("Disparate Impact Ratio", 1.0) <= 1.20): critical += 1

    if critical >= 2:
        return ("🚨 HIGH BIAS DRIFT DETECTED",
                "Multiple critical fairness violations. Immediate review required.", "red")
    elif critical == 1:
        return ("⚠️ MODERATE BIAS DRIFT",
                "One critical fairness issue detected. Model review recommended.", "orange")
    else:
        return ("✅ MODEL IS FAIR",
                "No significant bias drift detected across all metrics.", "green")


# ─────────────────────────────────────────────────────────────
#  SECTION 11 — SHAP EXPLAINABILITY
#
#  Uses model.pkl trained on hr_dataset (NOT predictions_log)
#  Explains WHICH features cause bias between Male & Female
# ─────────────────────────────────────────────────────────────
 
def compute_shap_analysis(df_log, n_sample=200):
    """
    Loads model.pkl + hr_dataset features from predictions_log,
    computes SHAP values, returns feature importance tables
    for Male vs Female separately.
 
    Parameters:
      df_log   : the loaded predictions_log DataFrame
      n_sample : how many rows to use for SHAP (keep low for speed)
 
    Returns:
      shap_df_all    : global importance for all records
      shap_df_male   : importance for Male records only
      shap_df_female : importance for Female records only
      feature_names  : list of feature names
    """
    # Load model artifacts
    model, scaler, feature_names = load_model()
 
    # ── Pull only feature columns that exist in predictions_log ──
    missing = [f for f in feature_names if f not in df_log.columns]
    if missing:
        for col in missing:
            df_log[col] = 0
 
    # ── FIX: encode any text columns before scaling ──
    # OverTime comes in as "Yes"/"No" text in predictions_log
    # but model was trained with it as 1/0 — must convert here
    for col in feature_names:
       if col in df_log.columns:

          # Handle text values like Yes/No properly
          if df_log[col].dtype == object:
            df_log[col] = (
                df_log[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0})
             )

          # Convert everything to numeric safely
          df_log[col] = pd.to_numeric(df_log[col], errors="coerce").fillna(0)
 
    X_log = df_log[feature_names].fillna(0).astype(float)
 
    # Sample for speed
    sample_idx = X_log.sample(min(n_sample, len(X_log)), random_state=42).index
    X_sample   = X_log.loc[sample_idx]
    gender_sample = df_log.loc[sample_idx, "Gender"]
 
    # Scale
    X_scaled = scaler.transform(X_sample)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X_sample.index)
 
    # SHAP — LinearExplainer is fast and correct for Logistic Regression
    explainer   = shap.LinearExplainer(model, X_scaled_df)
    shap_values = explainer.shap_values(X_scaled_df)
 
    # Convert to DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index)
 
    # ── Global importance (all records) ──
    global_imp = pd.DataFrame({
        "Feature":          feature_names,
        "SHAP Importance":  np.abs(shap_df.values).mean(axis=0)
    }).sort_values("SHAP Importance", ascending=False).reset_index(drop=True)
 
    # ── Male importance ──
    male_idx  = gender_sample[gender_sample == "Male"].index
    if len(male_idx) > 0:
        male_imp = pd.DataFrame({
            "Feature":         feature_names,
            "SHAP Importance": np.abs(shap_df.loc[male_idx].values).mean(axis=0)
        }).sort_values("SHAP Importance", ascending=False).reset_index(drop=True)
    else:
        male_imp = global_imp.copy()
 
    # ── Female importance ──
    female_idx = gender_sample[gender_sample == "Female"].index
    if len(female_idx) > 0:
        female_imp = pd.DataFrame({
            "Feature":         feature_names,
            "SHAP Importance": np.abs(shap_df.loc[female_idx].values).mean(axis=0)
        }).sort_values("SHAP Importance", ascending=False).reset_index(drop=True)
    else:
        female_imp = global_imp.copy()
 
    # ── Bias contribution: which features differ most Male vs Female ──
    merged = global_imp.copy()
    merged = merged.rename(columns={"SHAP Importance": "Global"})
    merged["Male SHAP"]   = male_imp.set_index("Feature").reindex(merged["Feature"])["SHAP Importance"].values
    merged["Female SHAP"] = female_imp.set_index("Feature").reindex(merged["Feature"])["SHAP Importance"].values
    merged["Difference"]  = abs(merged["Male SHAP"] - merged["Female SHAP"]).round(4)
    merged = merged.sort_values("Difference", ascending=False).reset_index(drop=True)
 
    return global_imp, male_imp, female_imp, merged, feature_names
 
 
# ─────────────────────────────────────────────────────────────
#  SECTION 12 — COMPLIANCE REPORT
# ─────────────────────────────────────────────────────────────

def generate_compliance_report(psi_value, ks_result, fairness_metrics, alerts):
    """Generates structured compliance report as a DataFrame."""
    critical_count = sum(1 for a in alerts if a["level"] == "CRITICAL")
    warning_count  = sum(1 for a in alerts if a["level"] == "WARNING")

    report = {
        "PSI Score":                      psi_value,
        "KS Statistic":                   ks_result.get("ks_statistic"),
        "KS P-value":                     ks_result.get("p_value"),
        "Male Positive Rate":             fairness_metrics.get("Male Positive Rate"),
        "Female Positive Rate":           fairness_metrics.get("Female Positive Rate"),
        "Demographic Parity Difference":  fairness_metrics.get("Demographic Parity Difference"),
        "Equalized Odds Difference":      fairness_metrics.get("Equalized Odds Difference"),
        "Disparate Impact Ratio":         fairness_metrics.get("Disparate Impact Ratio"),
        "Average Odds Difference":        fairness_metrics.get("Average Odds Difference"),
        "Total Alerts":                   len(alerts),
        "Critical Alerts":                critical_count,
        "Warning Alerts":                 warning_count,
        "Overall Bias Status": (
            "HIGH RISK"      if psi_value > 0.25 or fairness_metrics.get("Demographic Parity Difference", 0) > 0.20
            else "MODERATE"  if psi_value > 0.10
            else "FAIR"
        )
    }
    return pd.DataFrame([report])


# ─────────────────────────────────────────────────────────────
# SECTION 13 — EMAIL ALERT SYSTEM
# ─────────────────────────────────────────────────────────────

def build_email_message(psi_value, ks_result, fairness_metrics, dashboard_url):
    message = f"""
🚨 Bias Drift Detected - Action Required

📊 PSI: {psi_value}
📊 KS Statistic: {ks_result.get('ks_statistic')}
📊 KS P-value: {ks_result.get('p_value')}

📊 Demographic Parity Difference: {fairness_metrics.get('Demographic Parity Difference')}
📊 Equalized Odds Difference: {fairness_metrics.get('Equalized Odds Difference')}
📊 Disparate Impact Ratio: {fairness_metrics.get('Disparate Impact Ratio')}

👉 Dashboard:
{dashboard_url}

⚠️ Please review model fairness immediately.
"""
    return message


def send_email_alert(subject, message, to_email=None):
    """
    Sends the email alert.

    VERSION 2 CHANGE: sender_email and sender_password are now read
    from config.py instead of being hardcoded here. This means you
    only need to set your real Gmail + App Password in ONE place
    (config.py) instead of editing this file.

    to_email defaults to config.EMAIL_RECEIVER if not provided,
    so existing calls like send_email_alert(subject, message)
    still work exactly as before.
    """
    import config

    sender_email    = config.EMAIL_SENDER
    sender_password = config.EMAIL_APP_PASSWORD
    to_email         = to_email or config.EMAIL_RECEIVER

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)

        server.send_message(msg)
        server.quit()

        print("Email sent successfully!")
        return True, "✅ Email sent successfully!"

    except Exception as e:
        print("Email failed:", e)
        return False, f"❌ Email failed: {e}"