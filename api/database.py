# =============================================================
#  api/database.py  —  DATABASE SETUP
#
#  Creates the SQL database and the "predictions" table.
#  Uses SQLite by default (a single file, zero setup needed).
#  Can switch to PostgreSQL by changing DATABASE_URL in config.py
#
#  TABLE: predictions
#    id          → auto-increment primary key
#    timestamp   → when the prediction was made
#    score       → model confidence score (0.0 to 1.0)
#    actual      → true label (0 or 1)
#    group_name  → demographic group, e.g. "Male" / "Female"
# =============================================================

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import sys
import os

# Make config.py importable from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─────────────────────────────────────────────────────────────
#  DATABASE ENGINE
# ─────────────────────────────────────────────────────────────

# `check_same_thread=False` is required for SQLite when used with FastAPI
connect_args = {"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {}

engine = create_engine(config.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────────────────────
#  TABLE DEFINITION
# ─────────────────────────────────────────────────────────────

class Prediction(Base):
    """
    One row = one model prediction event.

    NOTE: column names intentionally use 'score' and 'group_name'
    (NOT 'probability' / 'gender') to simulate a real-world company
    database schema. utils.py still expects 'probability' / 'Gender',
    so the API layer converts between the two — see api/main.py
    and the normalization logic in app.py.
    """
    __tablename__ = "predictions"

    id         = Column(Integer, primary_key=True, index=True)
    timestamp  = Column(DateTime, nullable=False)
    score      = Column(Float,    nullable=False)   # 0.0 to 1.0
    actual     = Column(Integer,  nullable=False)    # 0 or 1
    group_name = Column(String,   nullable=False)    # e.g. "Male", "Female"


# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def init_db():
    """Creates the predictions table if it doesn't already exist."""
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database ready at: {config.DATABASE_URL}")


def get_db():
    """
    FastAPI dependency — gives each request its own database session
    and closes it automatically when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Running this file directly just creates the table.
    # python api/database.py
    init_db()
