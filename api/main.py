# =============================================================
#  api/main.py  —  FASTAPI BACKEND
#
#  This simulates a real company's AI monitoring API.
#  It reads from the SQL database and serves predictions as JSON.
#
#  HOW TO RUN:
#    uvicorn api.main:app --reload
#
#  ENDPOINTS:
#    GET /predictions   → latest 100 prediction rows (JSON)
#    GET /health        → simple status check
#
#  TEST IN BROWSER:
#    http://127.0.0.1:8000/health
#    http://127.0.0.1:8000/predictions
#    http://127.0.0.1:8000/docs   ← interactive API docs (auto-generated)
# =============================================================

from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc


from api.seed_database import seed_from_csv
from api.database import get_db, Prediction, init_db

app = FastAPI(
    title="Bias Drift Detector API",
    description="Serves AI model prediction logs for bias drift monitoring",
    version="2.0.0",
)

# Allow the Streamlit dashboard (running on a different port/domain)
# to call this API without being blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # in production, replace * with your dashboard's exact URL
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

    # seed only if DB is empty (IMPORTANT FIX)
    from api.database import SessionLocal, Prediction

    db = SessionLocal()
    try:
        exists = db.query(Prediction).first()
        if not exists:
            seed_from_csv()
    finally:
        db.close()
    
# ─────────────────────────────────────────────────────────────
#  GET /health
#
#  Simple check to confirm the API is alive.
#  The dashboard can call this first to verify connectivity
#  before trying to fetch real data.
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Bias Drift Detector API",
        "version": "2.0.0",
    }


# ─────────────────────────────────────────────────────────────
#  GET /predictions
#
#  Returns prediction rows from the database as JSON.
#  Default: latest 100 rows. Can be overridden with ?limit=500
# ─────────────────────────────────────────────────────────────

@app.get("/predictions")
def get_predictions(
    limit: int = Query(100, ge=1, le=5000, description="Number of rows to return"),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Prediction)
        .order_by(desc(Prediction.timestamp))
        .limit(limit)
        .all()
    )

    return {
        "count": len(rows),
        "predictions": [
            {
                "id":         r.id,
                "timestamp":  r.timestamp.isoformat() if r.timestamp else None,
                "score":      r.score,
                "actual":     r.actual,
                "group_name": r.group_name,
            }
            for r in rows
        ],
    }


# ─────────────────────────────────────────────────────────────
#  Root endpoint — friendly landing message
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Bias Drift Detector API is running.",
        "endpoints": ["/health", "/predictions", "/docs"],
    }
