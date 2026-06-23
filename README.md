# Bias Drift Detector — Version 2 (API + Database)

This is the upgraded version of the Bias Drift Detector. It now simulates a
real-world AI monitoring pipeline: a SQL database stores prediction logs, a
FastAPI backend serves them as JSON, and the Streamlit dashboard fetches data
from that API instead of reading a CSV file directly.

**Your existing fairness/drift logic in `utils.py` is completely unchanged.**
Only the data-loading layer was added on top of it.

---

## What changed from Version 1

| | Version 1 (old) | Version 2 (this version) |
|---|---|---|
| Data source | `pd.read_csv("logs/predictions_log.csv")` | SQL database → FastAPI → `requests.get()` |
| Storage | CSV file only | SQLite/PostgreSQL database |
| Access pattern | Local file read | REST API call |
| `utils.py` | unchanged | **unchanged** |
| Dashboard tabs | 6 tabs | same 6 tabs, unchanged |
| Email alerts | hardcoded credentials, broken indentation | reads from `config.py`, fixed, sends automatically when critical |
| Settings | scattered across files | centralised in `config.py` |

---

## Project structure

```
bias-drift-detector/
├── api/
│   ├── __init__.py
│   ├── database.py        ← SQL database setup (SQLAlchemy)
│   ├── main.py             ← FastAPI backend (GET /predictions, GET /health)
│   └── seed_database.py    ← loads predictions_log.csv INTO the database
├── data/
│   └── hr_dataset.csv      ← put your dataset here
├── logs/
│   └── predictions_log.csv ← created by train_model.py (unchanged)
├── model/
│   ├── model.pkl            ← created by train_model.py (unchanged)
│   ├── scaler.pkl
│   └── feature_names.pkl
├── config.py                ← ALL settings: API URL, email, thresholds
├── data_loader.py            ← fetches API data + converts it to utils.py format
├── train_model.py            ← UNCHANGED — trains the Logistic Regression model
├── utils.py                  ← UNCHANGED — all drift/fairness/SHAP/email logic
├── app.py                    ← Streamlit dashboard (data loading part updated)
└── requirements.txt
```

---

## How everything connects

```
hr_dataset.csv
      │
      ▼
train_model.py                 (UNCHANGED)
  trains Logistic Regression
  saves model/model.pkl
  saves logs/predictions_log.csv
      │
      ▼
api/seed_database.py            (NEW)
  reads predictions_log.csv
  renames columns:
    Probability  → score
    Gender       → group_name
    Attrition    → actual
  inserts rows into the SQL database
      │
      ▼
SQL Database (predictions.db)   (NEW)
  table: predictions
  columns: id, timestamp, score, actual, group_name
      │
      ▼
api/main.py  (FastAPI)          (NEW)
  GET /health        → status check
  GET /predictions    → returns rows as JSON
      │
      ▼
data_loader.py                   (NEW)
  calls the API
  converts JSON → DataFrame
  renames columns BACK:
    score        → Probability, Predicted (score >= 0.5)
    group_name   → Gender
    actual       → Attrition
      │
      ▼
utils.standardize_predictions_df()   (UNCHANGED logic, new entry point)
      │
      ▼
app.py — Streamlit Dashboard         (6 tabs, all UNCHANGED)
  Tab 1: Overview
  Tab 2: PSI & KS Test
  Tab 3: Fairness Metrics
  Tab 4: Drift Over Time
  Tab 5: Alerts + automatic Email + Compliance Report download
  Tab 6: SHAP Explainability
```

**Key idea:** `utils.py` never knows the database or API exists. It only ever
sees a DataFrame with columns `Gender`, `probability`, `predicted`, `actual`,
`timestamp` — exactly like before. `data_loader.py` is the translator that
sits between the new infrastructure and your old, working logic.

---

## How to run (local development)

Run these commands **in order**, each in the project root folder:

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Train the model (creates model.pkl + predictions_log.csv)
#    — exactly the same as Version 1, nothing changed here
python train_model.py

# 3. Load predictions_log.csv into the SQL database
python api/seed_database.py

# 4. Start the FastAPI backend (keep this terminal open)
uvicorn api.main:app --reload

# 5. In a NEW terminal, start the dashboard
streamlit run app.py
```

Once both are running:
- API docs: http://127.0.0.1:8000/docs
- API health check: http://127.0.0.1:8000/health
- Dashboard: http://localhost:8501

In the dashboard sidebar, choose **"🌐 API (Database)"** as the data source.
A **"📄 CSV File (offline mode)"** option is also available if you want the
old Version 1 behaviour without running the API at all.

---

## Configuration (`config.py`)

Everything that used to be hardcoded now lives in one file:

```python
API_URL              = "http://127.0.0.1:8000"   # change for deployment
DATABASE_URL          = "sqlite:///./predictions.db"
DASHBOARD_URL          = "http://localhost:8501"
EMAIL_SENDER           = "your_email@gmail.com"     # ← fill in your Gmail
EMAIL_APP_PASSWORD     = "your_app_password"        # ← fill in App Password
EMAIL_RECEIVER         = "employee@gmail.com"
PSI_CRITICAL_THRESHOLD = 0.25
DPD_CRITICAL_THRESHOLD = 0.20
```

To get a Gmail App Password: **myaccount.google.com → Security →
2-Step Verification → App Passwords**.

---

## Deploying to production (e.g. Render)

1. **Deploy the API first.** Push the project to GitHub, create a new Web
   Service on Render, set the start command to:
   ```
   uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```
2. Once deployed, copy the live URL (e.g. `https://bias-drift-api.onrender.com`).
3. Set it as an environment variable for the Streamlit app:
   ```
   BIAS_API_URL=https://bias-drift-api.onrender.com
   ```
   `config.py` automatically reads this environment variable — **no code
   changes needed**, since `API_URL = os.getenv("BIAS_API_URL", "http://127.0.0.1:8000")`.
4. Deploy the Streamlit dashboard separately (e.g. Streamlit Community Cloud
   or another Render Web Service) and set the same environment variable there.
5. Update `BIAS_DASHBOARD_URL` similarly so email alerts link to the live
   dashboard instead of localhost.

This is why `config.py` reads from environment variables first — you can
switch between local and deployed setups without touching any code.

---

## Database schema

```sql
CREATE TABLE predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME NOT NULL,
    score       FLOAT    NOT NULL,   -- model confidence, 0.0 to 1.0
    actual      INTEGER  NOT NULL,   -- true label, 0 or 1
    group_name  VARCHAR  NOT NULL    -- demographic group, e.g. "Male" / "Female"
);
```

This intentionally uses generic column names (`score`, `group_name`) instead
of `probability`/`gender`, to simulate how a real company's existing
production database might already be structured — the translation happens
in `data_loader.py`, not by changing the database to match `utils.py`.

---

## What stayed exactly the same

- `train_model.py` — not modified at all
- `utils.py` — every existing function (`calculate_psi`, `run_ks_test`,
  `compute_fairness_metrics`, `compute_bias_drift_over_time`,
  `generate_alerts`, `overall_verdict`, `compute_shap_analysis`,
  `generate_compliance_report`) is untouched
- All 6 dashboard tabs and their charts
- The email message format (`build_email_message`)

## What was added or fixed

- `config.py`, `data_loader.py`, `api/` folder — all new
- `utils.standardize_predictions_df()` — new function added (old
  `load_predictions()` still works exactly as before for CSV files)
- `send_email_alert()` — now reads credentials from `config.py` instead of
  hardcoded placeholder text
- Fixed an indentation bug in Tab 5 of `app.py` where the email-sending code
  had fallen outside the `with tab5:` block
- Email now sends **automatically** when bias is critical — no manual form,
  no button click required
