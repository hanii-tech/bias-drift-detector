# =============================================================
#  app.py  —  STREAMLIT DASHBOARD  
#
#  WHAT CHANGED FROM VERSION 1:
#    ❌ OLD: pd.read_csv("logs/predictions_log.csv")
#    ✅ NEW: requests.get(API_URL) → DataFrame → normalize
#
#
#  HOW TO RUN (in order):
#    1. python train_model.py          ← creates predictions_log.csv
#    2. python api/seed_database.py    ← loads CSV into SQL database
#    3. uvicorn api.main:app --reload  ← starts the API (new terminal)
#    4. streamlit run app.py           ← starts the dashboard
#
#  FOLDER STRUCTURE:
#    bias-drift-detector/
#    ├── api/
#    │   ├── __init__.py
#    │   ├── database.py       ← SQL database setup
#    │   ├── main.py            ← FastAPI backend
#    │   └── seed_database.py   ← loads CSV into database
#    ├── data/
#    │   └── hr_dataset.csv
#    ├── logs/
#    │   └── predictions_log.csv   ← created by train_model.py
#    ├── model/
#    │   ├── model.pkl
#    │   ├── scaler.pkl
#    │   └── feature_names.pkl
#    ├── config.py              ← all settings (API URL, email, thresholds)
#    ├── data_loader.py         ← fetches + normalizes API data
#    ├── train_model.py
#    ├── utils.py               ← core logic
#    ├── app.py
#    └── requirements.txt
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import utils
import config
import data_loader

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bias Drift Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<div style='background:linear-gradient(90deg,#1F3864,#2E75B6);
            padding:22px 30px;border-radius:12px;margin-bottom:18px;'>
    <h1 style='color:white;margin:0;font-size:2rem;'>
        🔍 Bias Drift Detector in AI Models
    </h1>
    <p style='color:#c8d8f0;margin:6px 0 0 0;font-size:1rem;'>
        HR Attrition Model · Monitoring Male vs Female fairness over time
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("⚙️ Settings")

# ── DATA SOURCE SELECTOR (NEW in Version 2) ──
data_source = st.sidebar.radio(
    "Data Source",
    ["🌐 API (Database)", "📄 CSV File (offline mode)"],
    help="API mode fetches live data from FastAPI + SQL database. "
         "CSV mode reads the local predictions_log.csv directly (old Version 1 behaviour)."
)

if data_source == "📄 CSV File (offline mode)":
    csv_path = st.sidebar.text_input(
        "Path to predictions_log.csv",
        value="logs/predictions_log.csv"
    )
else:
    st.sidebar.text_input("API URL (from config.py)", value=config.API_URL, disabled=True)
    st.sidebar.caption(f"Endpoint: `{config.PREDICTIONS_ENDPOINT}`")

n_windows   = st.sidebar.slider("Time Windows (drift trend)", 3, 10, config.DEFAULT_N_WINDOWS)
split_ratio = st.sidebar.slider("Baseline split ratio", 0.3, 0.7, config.DEFAULT_SPLIT_RATIO, step=0.1)
psi_bins    = st.sidebar.slider("PSI bins", 5, 20, config.DEFAULT_PSI_BINS)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Result key:**
- ✅ Green = Fair / No drift
- ⚠️ Orange = Watch closely
- 🚨 Red = Bias detected!
""")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Run `python train_model.py` → `python api/seed_database.py` → "
    "`uvicorn api.main:app --reload` before using API mode."
)

# ─────────────────────────────────────────────────────────────
#  LOAD DATA
#
#  Instead of always reading a CSV, we now check which data
#  source the user picked and load accordingly. Either path
#  ends up calling the SAME utils.py standardization logic.
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)   # re-fetch from API every 30 seconds at most
def load_from_api():
    raw_df = data_loader.load_predictions_from_api()
    return utils.standardize_predictions_df(raw_df)

@st.cache_data
def load_from_csv(path):
    return utils.load_predictions(path)

try:
    if data_source == "🌐 API (Database)":
        df = load_from_api()
        st.sidebar.success(f"✅ {len(df)} records loaded from API")
    else:
        df = load_from_csv(csv_path)
        st.sidebar.success(f"✅ {len(df)} records loaded from CSV")
except Exception as e:
    st.error(f"❌ Could not load data.\n\nError: {e}")
    if data_source == "🌐 API (Database)":
        st.info(
            "Make sure the API is running:\n\n"
            "1. `python train_model.py`\n"
            "2. `python api/seed_database.py`\n"
            "3. `uvicorn api.main:app --reload`\n\n"
            f"Then check {config.HEALTH_ENDPOINT} in your browser."
        )
    else:
        st.info("Run `python train_model.py` first to generate predictions_log.csv")
    st.stop()

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 PSI & KS Test",
    "⚖️ Fairness Metrics",
    "🕐 Drift Over Time",
    "🚨 Alerts & Report",
    "🧠 SHAP Explainability",
])


# ═══════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW   
# ═══════════════════════════════════════════════════════════

with tab1:
    st.header("📊 Dataset Overview")

    total  = len(df)
    n_male = (df["Gender"] == "Male").sum()
    n_fem  = (df["Gender"] == "Female").sum()
    avg_p  = df["probability"].mean()
    attr_r = df["actual"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",    total)
    c2.metric("Male",             n_male)
    c3.metric("Female",           n_fem)
    c4.metric("Avg Probability",  f"{avg_p:.3f}")
    c5.metric("Actual Attrition", f"{attr_r:.1%}")

    st.markdown("---")

    st.subheader("👥 Male vs Female Summary")
    st.dataframe(utils.gender_summary(df), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Gender Distribution")
        gc = df["Gender"].value_counts().reset_index()
        gc.columns = ["Gender", "Count"]
        fig = px.pie(gc, names="Gender", values="Count", hole=0.4,
                     color="Gender",
                     color_discrete_map={"Male": "#2E75B6", "Female": "#E84393"})
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Positive Prediction Rate by Gender")
        pr = df.groupby("Gender")["predicted"].mean().reset_index()
        pr.columns = ["Gender", "Rate"]
        fig2 = px.bar(pr, x="Gender", y="Rate", color="Gender", text="Rate",
                      color_discrete_map={"Male": "#2E75B6", "Female": "#E84393"})
        fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig2.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Probability Score Distribution — Male vs Female")
    fig3 = go.Figure()
    for g, col in [("Male", "#2E75B6"), ("Female", "#E84393")]:
        fig3.add_trace(go.Histogram(
            x=df[df["Gender"] == g]["probability"],
            name=g, opacity=0.65, marker_color=col, nbinsx=30
        ))
    fig3.update_layout(barmode="overlay",
                       xaxis_title="Probability Score", yaxis_title="Count",
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("🔎 View Raw Data (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  TAB 2 — PSI & KS TEST  
# ═══════════════════════════════════════════════════════════

with tab2:
    st.header("📈 Population Stability Index (PSI) & KS Test")
    st.markdown("""
    | PSI | Meaning |
    |-----|---------|
    | < 0.10 | ✅ Stable — No drift |
    | 0.10–0.25 | ⚠️ Slight drift |
    | > 0.25 | 🚨 Major drift |
    """)

    baseline, current = utils.split_baseline_current(df, split_ratio)
    st.info(f"Baseline: **{len(baseline)} records** (first {int(split_ratio*100)}%) "
            f"| Current: **{len(current)} records** (last {int((1-split_ratio)*100)}%)")

    psi_val, psi_breakdown = utils.calculate_psi(
        baseline["probability"].values,
        current["probability"].values,
        bins=psi_bins
    )
    interp = utils.interpret_psi(psi_val)
    color  = "red" if psi_val >= config.PSI_CRITICAL_THRESHOLD else \
             ("orange" if psi_val >= config.PSI_WARNING_THRESHOLD else "green")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style='text-align:center;padding:28px;background:#f0f4ff;
                    border-radius:12px;border-left:6px solid {color};'>
            <h2 style='color:{color};font-size:3rem;margin:0;'>{psi_val}</h2>
            <p style='color:#333;margin:6px 0;'>Overall PSI Score</p>
            <p style='color:{color};font-weight:bold;margin:0;'>{interp}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("PSI by Gender")
        for g in ["Male", "Female"]:
            gb = baseline[baseline["Gender"] == g]["probability"].values
            gc = current[current["Gender"]   == g]["probability"].values
            if len(gb) > 0 and len(gc) > 0:
                gp, _ = utils.calculate_psi(gb, gc, bins=psi_bins)
                st.markdown(f"**{g}:** PSI = `{gp}` — {utils.interpret_psi(gp)}")

    st.markdown("---")
    st.subheader("PSI Breakdown by Bin")
    st.dataframe(psi_breakdown, use_container_width=True)

    st.subheader("Baseline vs Current Probability Distribution")
    fig_psi = go.Figure()
    fig_psi.add_trace(go.Histogram(x=baseline["probability"], name="Baseline",
                                    opacity=0.7, marker_color="#1F3864", nbinsx=psi_bins))
    fig_psi.add_trace(go.Histogram(x=current["probability"],  name="Current",
                                    opacity=0.7, marker_color="#F4A460", nbinsx=psi_bins))
    fig_psi.update_layout(barmode="overlay",
                           xaxis_title="Probability Score", yaxis_title="Count",
                           margin=dict(t=10, b=10))
    st.plotly_chart(fig_psi, use_container_width=True)

    st.markdown("---")
    st.subheader("🧪 KS Test — Male vs Female Score Comparison")
    ks = utils.run_ks_test(df)
    k1, k2 = st.columns(2)
    k1.metric("KS Statistic", ks["ks_statistic"])
    k2.metric("p-value",      ks["p_value"])
    if ks["p_value"] and ks["p_value"] < 0.05:
        st.error(ks["result"])
    else:
        st.success(ks["result"])


# ═══════════════════════════════════════════════════════════
#  TAB 3 — FAIRNESS METRICS   
# ═══════════════════════════════════════════════════════════

with tab3:
    st.header("⚖️ Fairness Metrics")
    st.markdown("Checking whether the HR model treats **Male** and **Female** employees equally.")

    fm = utils.compute_fairness_metrics(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DPD", fm["Demographic Parity Difference"])
    c1.caption(fm["DPD Status"])
    c2.metric("EOD", fm["Equalized Odds Difference"])
    c2.caption(fm["EOD Status"])
    c3.metric("DIR", fm["Disparate Impact Ratio"])
    c3.caption(fm["DIR Status"])
    c4.metric("AOD", fm["Average Odds Difference"])
    c4.caption("Ideal: < 0.10")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Positive Prediction Rate")
        rd = pd.DataFrame({
            "Group": ["Male",    "Female"],
            "Rate":  [fm["Male Positive Rate"], fm["Female Positive Rate"]]
        })
        fig_f = px.bar(rd, x="Group", y="Rate", color="Group", text="Rate",
                       color_discrete_map={"Male": "#2E75B6", "Female": "#E84393"})
        fig_f.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_f.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig_f, use_container_width=True)

    with col_b:
        st.subheader("True Positive Rate (Equalized Odds)")
        td = pd.DataFrame({
            "Group": ["Male",    "Female"],
            "TPR":   [fm["Male True Positive Rate"], fm["Female True Positive Rate"]]
        })
        fig_t = px.bar(td, x="Group", y="TPR", color="Group", text="TPR",
                       color_discrete_map={"Male": "#2E75B6", "Female": "#E84393"})
        fig_t.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_t.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("DPD Gauge")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fm["Demographic Parity Difference"],
        title={"text": "Demographic Parity Difference"},
        gauge={
            "axis":  {"range": [0, 0.5]},
            "bar":   {"color": "darkblue"},
            "steps": [
                {"range": [0.00, 0.10], "color": "#b7e4c7"},
                {"range": [0.10, 0.20], "color": "#ffe08a"},
                {"range": [0.20, 0.50], "color": "#f4a2a2"},
            ],
            "threshold": {"line": {"color": "red", "width": 3}, "value": 0.20}
        }
    ))
    fig_g.update_layout(height=280, margin=dict(t=30, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("All Fairness Metrics")
    st.dataframe(
        pd.DataFrame(list(fm.items()), columns=["Metric", "Value"]),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════
#  TAB 4 — DRIFT OVER TIME  
# ═══════════════════════════════════════════════════════════

with tab4:
    st.header("🕐 Bias Drift Over Time")
    st.markdown("""
    **Core of this project.**
    Data is split into time windows and DPD is tracked per window.
    - Line going **UP** → bias is getting worse → **drift detected** 🚨
    - Line staying **flat** → model is stable ✅
    """)

    drift_df = utils.compute_bias_drift_over_time(df, n_windows)

    st.subheader("Window-by-Window Results")
    st.dataframe(drift_df, use_container_width=True)

    st.subheader("📉 DPD Trend — Is Bias Increasing?")
    fig_d = go.Figure()
    fig_d.add_hline(y=0.10, line_dash="dash", line_color="orange",
                    annotation_text="⚠️ Warning (0.10)")
    fig_d.add_hline(y=0.20, line_dash="dash", line_color="red",
                    annotation_text="🚨 Critical (0.20)")
    marker_colors = [
        "red" if v >= 0.20 else "orange" if v >= 0.10 else "green"
        for v in drift_df["DPD"]
    ]
    fig_d.add_trace(go.Scatter(
        x=drift_df["Time Label"], y=drift_df["DPD"],
        mode="lines+markers+text",
        name="DPD",
        line=dict(color="#1F3864", width=3),
        marker=dict(size=12, color=marker_colors),
        text=drift_df["DPD"].round(4),
        textposition="top center"
    ))
    fig_d.update_layout(
        xaxis_title="Time Window",
        yaxis_title="Demographic Parity Difference",
        yaxis_range=[0, max(drift_df["DPD"].max() + 0.05, 0.30)],
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Male vs Female Positive Rate Over Time")
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=drift_df["Time Label"], y=drift_df["Male Rate"],
                                mode="lines+markers", name="Male",
                                line=dict(color="#2E75B6", width=2), marker=dict(size=8)))
    fig_r.add_trace(go.Scatter(x=drift_df["Time Label"], y=drift_df["Female Rate"],
                                mode="lines+markers", name="Female",
                                line=dict(color="#E84393", width=2), marker=dict(size=8)))
    fig_r.update_layout(xaxis_title="Time Window",
                         yaxis_title="Positive Prediction Rate",
                         margin=dict(t=20, b=20))
    st.plotly_chart(fig_r, use_container_width=True)

    first_dpd = drift_df["DPD"].iloc[0]
    last_dpd  = drift_df["DPD"].iloc[-1]
    change    = last_dpd - first_dpd

    if change > 0.10:
        st.error(f"🚨 Bias significantly increased: {first_dpd:.4f} → {last_dpd:.4f}  (Δ = +{change:.4f})")
    elif change > 0:
        st.warning(f"⚠️ Bias slowly increasing: {first_dpd:.4f} → {last_dpd:.4f}  (Δ = +{change:.4f})")
    else:
        st.success(f"✅ Bias stable or improving: {first_dpd:.4f} → {last_dpd:.4f}  (Δ = {change:.4f})")


# ═══════════════════════════════════════════════════════════
#  TAB 5 — ALERTS & COMPLIANCE REPORT
# ═══════════════════════════════════════════════════════════

with tab5:
    st.header("🚨 Alerts & Compliance Report")

    bl2, cu2 = utils.split_baseline_current(df, split_ratio)
    psi2, _  = utils.calculate_psi(bl2["probability"].values,
                                    cu2["probability"].values, bins=psi_bins)
    ks2      = utils.run_ks_test(df)
    fm2      = utils.compute_fairness_metrics(df)
    drift2   = utils.compute_bias_drift_over_time(df, n_windows)

    # Overall verdict banner
    verdict, vmsg, vcol = utils.overall_verdict(psi2, fm2, ks2)
    bg  = {"red": "#ffeded", "orange": "#fff8e8", "green": "#edfff3"}[vcol]
    st.markdown(f"""
    <div style='text-align:center;padding:25px;border-radius:12px;
                background:{bg};border:3px solid {vcol};margin-bottom:20px;'>
        <h2 style='color:{vcol};margin:0;'>{verdict}</h2>
        <p style='font-size:1.1rem;margin-top:8px;'>{vmsg}</p>
    </div>
    """, unsafe_allow_html=True)

    # Alerts
    st.subheader("📋 All Alerts")
    alerts = utils.generate_alerts(psi2, ks2, fm2, drift2)

    for a in alerts:
        if a["level"] == "CRITICAL":
            st.error(f"🚨 CRITICAL: {a['message']}")
        elif a["level"] == "WARNING":
            st.warning(f"⚠️ WARNING: {a['message']}")
        else:
            st.success(f"✅ INFO: {a['message']}")

    st.markdown("---")

    # ── EMAIL ALERT TRIGGER (AUTOMATIC — no user input needed) ──
    st.subheader("📧 Email Alert Status")

    is_critical = (
        psi2 > config.PSI_CRITICAL_THRESHOLD
        or fm2.get("Demographic Parity Difference", 0) > config.DPD_CRITICAL_THRESHOLD
    )

    if is_critical:
        email_body = utils.build_email_message(
            psi2, ks2, fm2, config.DASHBOARD_URL
        )
        sent_ok, sent_msg = utils.send_email_alert(
            subject="🚨 Bias Drift Detected - Action Required",
            message=email_body,
        )
        if sent_ok:
            st.error("🚨 Critical bias detected — Email alert sent automatically!")
            st.caption(sent_msg)
        else:
            st.error("🚨 Critical bias detected — but the email FAILED to send.")
            st.caption(sent_msg)
            st.caption("Check EMAIL_SENDER / EMAIL_APP_PASSWORD in config.py")
    else:
        st.success("✅ No critical bias — No email alert needed.")

    st.markdown("---")

    # Compliance Report
    st.subheader("📊 Compliance Report")
    report_df = utils.generate_compliance_report(psi2, ks2, fm2, alerts)
    st.dataframe(report_df, use_container_width=True)

    csv = report_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Compliance Report as CSV",
        data=csv,
        file_name="bias_compliance_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════
#  TAB 6 — SHAP EXPLAINABILITY   (UNCHANGED FROM VERSION 1)
# ═══════════════════════════════════════════════════════════

with tab6:
    st.header("🧠 SHAP Explainability — Why Does Bias Happen?")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) tells us **which features**
    push the model toward predicting attrition — and whether those
    features affect **Male and Female employees differently**.

    > The model used here is the **Logistic Regression trained on hr_dataset.csv**
    > (loaded from `model/model.pkl`) — not the predictions log directly.
    """)

    if not os.path.exists("model/model.pkl"):
        st.error("❌ model/model.pkl not found! Run `python train_model.py` first.")
        st.stop()

    with st.spinner("Computing SHAP values... (this may take 10–20 seconds)"):
        try:
            global_imp, male_imp, female_imp, bias_contrib, feature_names = \
                utils.compute_shap_analysis(df.copy(), n_sample=200)
            shap_ok = True
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            shap_ok = False

    if shap_ok:

        st.subheader("📊 Global Feature Importance (All Records)")
        fig_g = px.bar(
            global_imp.head(15), x="SHAP Importance", y="Feature",
            orientation="h", color="SHAP Importance",
            color_continuous_scale="Blues",
            title="Top 15 Features — Mean |SHAP| Value"
        )
        fig_g.update_layout(yaxis={"autorange": "reversed"},
                             margin=dict(t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("---")

        st.subheader("👥 Feature Importance — Male vs Female")
        col_m, col_f = st.columns(2)

        with col_m:
            fig_m = px.bar(
                male_imp.head(10), x="SHAP Importance", y="Feature",
                orientation="h", color_discrete_sequence=["#2E75B6"],
                title="Top 10 — Male"
            )
            fig_m.update_layout(yaxis={"autorange": "reversed"},
                                  margin=dict(t=40, b=10))
            st.plotly_chart(fig_m, use_container_width=True)

        with col_f:
            fig_f = px.bar(
                female_imp.head(10), x="SHAP Importance", y="Feature",
                orientation="h", color_discrete_sequence=["#E84393"],
                title="Top 10 — Female"
            )
            fig_f.update_layout(yaxis={"autorange": "reversed"},
                                  margin=dict(t=40, b=10))
            st.plotly_chart(fig_f, use_container_width=True)

        st.markdown("---")

        st.subheader("⚠️ Features Contributing Most to Gender Bias")
        st.markdown("""
        The **Difference** column shows how much each feature's SHAP value
        differs between Male and Female groups.
        A **high difference** means that feature is treating genders unequally.
        """)
        st.dataframe(bias_contrib.head(15), use_container_width=True)

        top_feature = bias_contrib["Feature"].iloc[0]
        top_diff    = bias_contrib["Difference"].iloc[0]
        st.warning(f"⚠️ Top bias driver: **{top_feature}** "
                   f"(SHAP difference = {top_diff:.4f} between Male and Female)")
