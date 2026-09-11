# Bias Drift Detector System for AI Models

## Overview

Bias Drift Detector System is a production-style AI monitoring application that detects data drift, bias drift, and fairness issues in machine learning predictions. The system provides an interactive dashboard, REST API, SQL database integration, automated compliance reporting, and email-based alerting for critical fairness violations.

This project simulates how organizations monitor deployed AI models in real-world environments.

---

## Features

### Model Monitoring

* Population Stability Index (PSI) calculation
* Kolmogorov-Smirnov (KS) statistical testing
* Bias drift tracking over time
* Fairness metric monitoring

### Fairness Analysis

* Demographic Parity Difference (DPD)
* Equalized Odds Difference (EOD)
* Disparate Impact Ratio (DIR)
* Gender-based fairness evaluation

### Dashboard

* Interactive Streamlit dashboard
* Real-time monitoring view
* Compliance reporting
* Alert management
* Downloadable reports

### API Layer

* FastAPI backend
* REST endpoints for prediction retrieval
* Health monitoring endpoint
* Swagger API documentation

### Database Integration

* SQLite database storage
* Prediction log persistence
* Automated database seeding
* Historical monitoring support

### Email Alert System

* Automatic email notifications
* Critical bias detection alerts
* Drift warning notifications
* Dashboard link included in alert emails
* Gmail SMTP integration using App Passwords

---

## System Architecture

Prediction Logs (CSV)
↓
SQLite Database
↓
FastAPI Backend
↓
Streamlit Dashboard
↓
Bias & Drift Analysis
↓
Email Alerts + Compliance Reports

---

## Technology Stack

### Backend

* Python
* FastAPI
* SQLAlchemy
* Uvicorn

### Dashboard

* Streamlit
* Plotly

### Data Processing

* Pandas
* NumPy
* SciPy

### Machine Learning

* Scikit-learn

### Database

* SQLite

### Notifications

* SMTP
* Gmail App Password Authentication

### Deployment

* Render (API)
* Streamlit Community Cloud (Dashboard)
* GitHub

---

## API Endpoints

### Health Check

GET `/health`

Returns API status information.

### Predictions

GET `/predictions`

Returns prediction records from the database.

### API Documentation

GET `/docs`

Interactive Swagger documentation.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/hanii-tech/bias-drift-detector.git
cd bias-drift-detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Local Setup

### Step 1 – Train Model

```bash
python train_model.py
```

Generates:

```text
logs/predictions_log.csv
```

### Step 2 – Seed Database

```bash
python api/seed_database.py
```

Loads prediction data into SQLite.

### Step 3 – Start API

```bash
uvicorn api.main:app --reload
```

API available at:

```text
http://127.0.0.1:8000
```

### Step 4 – Launch Dashboard

```bash
streamlit run app.py
```

Dashboard available at:

```text
http://localhost:8501
```

---

## Deployment

### API Deployment

Hosted using Render.

API URL: https://bias-drift-detector.onrender.com

### Dashboard Deployment

Hosted using Streamlit Community Cloud.

Dashboard URL: https://bias-drift-detector-system.streamlit.app

---

## Project Structure

```text
bias_drift_detector/
│
├── api/
│   ├── main.py
│   ├── database.py
│   └── seed_database.py
│
├── logs/
│   └── predictions_log.csv
│
├── app.py
├── config.py
├── data_loader.py
├── utils.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

* PostgreSQL integration
* Authentication and user management
* Multi-model monitoring
* Scheduled background jobs
* Slack and Microsoft Teams alerts
* Docker deployment
* CI/CD pipeline integration

---

## Author

Harini T 

Aspiring Software Engineer with AI and Data Focus

Focused on building AI-powered, data-driven software applications and monitoring systems.
