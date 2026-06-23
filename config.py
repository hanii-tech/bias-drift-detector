# =============================================================
#  config.py  —  CENTRAL CONFIGURATION FILE
#
#  Every setting that used to be hardcoded in app.py / utils.py
#  now lives here. Change values in ONE place only.
#
#  HOW TO SWITCH LOCAL <-> DEPLOYED API:
#    Just change API_URL below. Nothing else needs to change.
# =============================================================

import os

# ─────────────────────────────────────────────────────────────
#  API SETTINGS
# ─────────────────────────────────────────────────────────────

# Local development (FastAPI running on your machine):
#   API_URL = "http://127.0.0.1:8000"
#
# Deployed (e.g. Render):
#   API_URL = "https://bias-drift-api.onrender.com"
#
# Reads from environment variable first (useful when deployed),
# falls back to localhost for local development.
API_URL = os.getenv("BIAS_API_URL", "https://bias-drift-detector.onrender.com")

# Endpoint paths (do not need to change these)
PREDICTIONS_ENDPOINT = f"{API_URL}/predictions"
HEALTH_ENDPOINT       = f"{API_URL}/health"

# How many rows to request from the API at once
PREDICTIONS_LIMIT = 1000


# ─────────────────────────────────────────────────────────────
#  DATABASE SETTINGS
# ─────────────────────────────────────────────────────────────

# SQLite by default (a single local file — no server needed).
# To use PostgreSQL instead, change this to something like:
#   "postgresql://user:password@host:5432/dbname"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")


# ─────────────────────────────────────────────────────────────
#  DASHBOARD SETTINGS
# ─────────────────────────────────────────────────────────────

# Local Streamlit dashboard:
#   DASHBOARD_URL = "http://localhost:8501"
#
# Deployed Streamlit dashboard:
#   DASHBOARD_URL = "https://your-app.streamlit.app"
DASHBOARD_URL = os.getenv("BIAS_DASHBOARD_URL", "http://localhost:8501")


# ─────────────────────────────────────────────────────────────
#  EMAIL ALERT SETTINGS
#
#  Fill these in before running the project.
#  For Gmail, use an App Password (NOT your normal password):
#    myaccount.google.com → Security → 2-Step Verification → App Passwords
# ─────────────────────────────────────────────────────────────

EMAIL_SENDER         = os.getenv("EMAIL_SENDER",   "harinit0506@gmail.com")
EMAIL_APP_PASSWORD   = os.getenv("EMAIL_PASSWORD", "edmc awju ybwm hyry")
EMAIL_RECEIVER       = os.getenv("EMAIL_RECEIVER", "har3105ini@gmail.com")


# ─────────────────────────────────────────────────────────────
#  DRIFT & FAIRNESS THRESHOLDS
#
#  Centralised so they can be tuned without touching utils.py
# ─────────────────────────────────────────────────────────────

PSI_WARNING_THRESHOLD   = 0.10
PSI_CRITICAL_THRESHOLD  = 0.25

DPD_WARNING_THRESHOLD   = 0.10
DPD_CRITICAL_THRESHOLD  = 0.20

EOD_WARNING_THRESHOLD   = 0.10
EOD_CRITICAL_THRESHOLD  = 0.20

DIR_FAIR_RANGE          = (0.80, 1.20)

# Score → predicted label threshold
# (used when normalizing API data: score >= 0.5 → predicted = 1)
SCORE_TO_PREDICTION_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────
#  DASHBOARD DEFAULTS (sliders in app.py)
# ─────────────────────────────────────────────────────────────

DEFAULT_N_WINDOWS   = 5
DEFAULT_SPLIT_RATIO = 0.5
DEFAULT_PSI_BINS    = 10
