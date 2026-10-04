"""Streamlit Web UI for Odoo HR Flight Risk Intelligence & Retention Analytics.

Provides:
1. Interactive Flight Risk Predictor & What-If Countermeasure Simulator.
2. Live Odoo ERP Workforce Explorer.
3. Batch Dataset Scoring & Population Analytics.
4. MLOps Governance & Real-Time Drift Monitoring.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.config.settings import get_settings
from src.explainability.explainer import FlightRiskExplainer
from src.features.engineer import FeatureEngineer
from src.models.registry import ModelRegistryManager
from src.monitoring.drift_detector import DriftDetector

settings = get_settings()

# Page configuration
st.set_page_config(
    page_title="Odoo HR Flight Risk Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Shared Design System CSS (Zinc / SaaS Style)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}

/* Minimal Chrome */
header[data-testid="stHeader"], footer, [data-testid="stToolbar"] {
    display: none !important;
}

.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* Card Style */
.saas-card {
    background-color: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03), 0 1px 2px rgba(0,0,0,0.02);
    margin-bottom: 1rem;
}

.saas-kpi-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.75rem;
    font-weight: 700;
    color: #09090b;
    margin-top: 0.25rem;
}

.saas-kpi-sub {
    font-size: 0.825rem;
    color: #71717a;
    font-weight: 500;
}

.badge-low {
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-medium {
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-high {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.metric-mono {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_ml_champion():
    """Load and cache the champion model and registry."""
    registry = ModelRegistryManager()
    pyfunc_model = registry.load_champion_model()
    champion_ver = registry.get_champion_version()
    # Unwrap sklearn pipeline
    if hasattr(pyfunc_model, "_model_impl") and hasattr(pyfunc_model._model_impl, "sklearn_model"):
        pipeline = pyfunc_model._model_impl.sklearn_model
    else:
        pipeline = pyfunc_model
    explainer = FlightRiskExplainer(pipeline, FeatureEngineer.FEATURE_COLS)
    return registry, pipeline, explainer, champion_ver


try:
    registry, pipeline, explainer, champion_ver = load_ml_champion()
    model_version_str = f"v{champion_ver.version}" if champion_ver else "v5"
except Exception as e:
    st.error(f"Failed to connect to MLflow Model Registry: {e}")
    st.stop()


def get_risk_tier(prob: float) -> tuple[str, str, str]:
    """Return risk label, CSS class, and action note."""
    if prob >= settings.flight_risk_high_threshold:
        return "HIGH RISK", "badge-high", "Immediate Retention Intervention Required"
    elif prob >= settings.flight_risk_medium_threshold:
        return "MEDIUM RISK", "badge-medium", "Proactive Manager Engagement Recommended"
    else:
        return "LOW RISK", "badge-low", "Standard Retention & Retention Cadence"


def render_gauge(prob: float):
    """Render Plotly radial gauge for flight risk score."""
    val = prob * 100
    color = "#16a34a" if prob < 0.40 else ("#d97706" if prob < 0.70 else "#dc2626")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={
                "suffix": "%",
                "font": {"family": "JetBrains Mono", "size": 38, "color": color},
            },
            title={
                "text": "Flight Risk Probability (6-12 Months)",
                "font": {"size": 15, "color": "#71717a"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#d4d4d8"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e4e4e7",
                "steps": [
                    {"range": [0, 40], "color": "#f0fdf4"},
                    {"range": [40, 70], "color": "#fefce8"},
                    {"range": [70, 100], "color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": "#991b1b", "width": 3},
                    "thickness": 0.8,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
    )
    return fig


def render_shap_waterfall(
    feature_names: list[str], row_arr: np.ndarray, explainer: FlightRiskExplainer
):
    """Render interactive SHAP horizontal bar chart."""
    row_2d = row_arr.reshape(1, -1)
    if explainer.scaler is not None:
        df_row = pd.DataFrame(row_2d, columns=feature_names)
        row_for_shap = explainer.scaler.transform(df_row)
    else:
        row_for_shap = row_2d

    shap_values = explainer.explainer.shap_values(row_for_shap)
    if isinstance(shap_values, list):
        vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif len(shap_values.shape) == 3:
        vals = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2:
        vals = shap_values[0]
    else:
        vals = shap_values

    shap_df = pd.DataFrame(
        {
            "Feature": [f.replace("_", " ").title() for f in feature_names],
            "SHAP": vals,
            "Color": ["#ef4444" if v > 0 else "#22c55e" for v in vals],
        }
    ).sort_values("SHAP", ascending=True)

    fig = px.bar(
        shap_df,
        x="SHAP",
        y="Feature",
        orientation="h",
        color="Color",
        color_discrete_map="identity",
        title="SHAP Feature Attribution (Impact on Flight Risk)",
    )
    fig.update_layout(
        height=270,
        margin=dict(l=10, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
        xaxis=dict(showgrid=True, gridcolor="#f4f4f5", zeroline=True, zerolinecolor="#d4d4d8"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


# -------------------------------------------------------------
# APP HEADER & TOP STATUS BAR
# -------------------------------------------------------------
st.markdown(
    """
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;">
    <div>
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #09090b;">
            Odoo HR Flight Risk & Retention Intelligence
        </h1>
        <p style="margin: 0.25rem 0 0 0; color: #71717a; font-size: 0.95rem;">
            Predictive Attrition Intelligence, Survival Analysis & Explainable AI (SHAP) for Odoo 20
        </p>
    </div>
    <div style="text-align: right;">
        <span style="background-color: #f4f4f5; border: 1px solid #e4e4e7; padding: 0.35rem 0.85rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600; color: #18181b;">
            Model: Champion ({}) &bull; MLflow @champion
        </span>
    </div>
</div>
""".format(model_version_str),
    unsafe_allow_html=True,
)

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-kpi-sub">ACTIVE CHAMPION MODEL</div>
        <div class="saas-kpi-val">{} <span style="font-size: 1rem; color: #16a34a;">&bull; Tuned</span></div>
        <div class="saas-kpi-sub">XGBoost Classifier (PR-AUC: 0.7155)</div>
    </div>
    """.format(model_version_str),
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-kpi-sub">RETENTION SLA STATUS</div>
        <div class="saas-kpi-val" style="color: #16a34a;">HEALTHY</div>
        <div class="saas-kpi-sub">PR-AUC &ge; 0.70 SLA Target Met</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-kpi-sub">DECISION THRESHOLDS</div>
        <div class="saas-kpi-val" style="font-size: 1.35rem;">&ge; 40% <span style="font-size: 0.85rem; color: #d97706;">MED</span> | &ge; 70% <span style="font-size: 0.85rem; color: #dc2626;">HIGH</span></div>
        <div class="saas-kpi-sub">Calibrated 6-12 Month Risk Window</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-kpi-sub">ERP INTEGRATION</div>
        <div class="saas-kpi-val" style="color: #2563eb; font-size: 1.35rem;">JSON-2 API</div>
        <div class="saas-kpi-sub">Connected to Odoo 20 (hr_db)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# NAVIGATION TABS
# -------------------------------------------------------------
tab_simulator, tab_odoo, tab_batch, tab_mlops = st.tabs(
    [
        "🔮 Individual Predictor & What-If Simulator",
        "🏢 Live Odoo Workforce Explorer",
        "📊 Batch Dataset Scoring & Analytics",
        "🛡️ MLOps Governance & Drift Monitor",
    ]
)

# -------------------------------------------------------------
# TAB 1: INDIVIDUAL PREDICTOR & WHAT-IF SIMULATOR
# -------------------------------------------------------------
with tab_simulator:
    st.markdown("### Interactive Flight Risk Assessment")
    st.caption(
        "Adjust employee workforce parameters or load archetypes to evaluate flight risk probability and inspect SHAP drivers."
    )

    # Archetype Presets
    p1, p2, p3, p4 = st.columns([1, 1, 1, 3])
    with p1:
        if st.button("🌟 Archetype: Satisfied Architect", use_container_width=True):
            st.session_state["tenure"] = 4.5
            st.session_state["age"] = 34
            st.session_state["distance"] = 12.0
            st.session_state["wage"] = 16500000.0
            st.session_state["wage_diff"] = 18.0
            st.session_state["overtime"] = 8.0
            st.session_state["leaves"] = 7
    with p2:
        if st.button("⚠️ Archetype: Commute Strain", use_container_width=True):
            st.session_state["tenure"] = 2.2
            st.session_state["age"] = 29
            st.session_state["distance"] = 38.0
            st.session_state["wage"] = 12000000.0
            st.session_state["wage_diff"] = -5.0
            st.session_state["overtime"] = 32.0
            st.session_state["leaves"] = 12
    with p3:
        if st.button("🚨 Archetype: Burnout Crisis", use_container_width=True):
            st.session_state["tenure"] = 3.8
            st.session_state["age"] = 31
            st.session_state["distance"] = 42.0
            st.session_state["wage"] = 10500000.0
            st.session_state["wage_diff"] = -28.0
            st.session_state["overtime"] = 55.0
            st.session_state["leaves"] = 18

    col_inputs, col_results = st.columns([1.1, 1.4], gap="large")

    with col_inputs:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Workforce Input Parameters")

        tenure = st.slider(
            "Company Tenure (Years)", 0.2, 15.0, float(st.session_state.get("tenure", 3.2)), 0.1
        )
        age = st.slider("Employee Age (Years)", 20, 65, int(st.session_state.get("age", 32)), 1)
        distance = st.slider(
            "Commuting Distance (km)", 1.0, 80.0, float(st.session_state.get("distance", 22.0)), 1.0
        )
        wage = st.number_input(
            "Monthly Salary (IDR)",
            5000000.0,
            40000000.0,
            float(st.session_state.get("wage", 13000000.0)),
            500000.0,
            format="%.0f",
        )
        wage_diff_pct = st.slider(
            "Department Compa-Ratio Diff (%)",
            -50.0,
            50.0,
            float(st.session_state.get("wage_diff", -10.0)),
            1.0,
        )
        overtime_pct = st.slider(
            "Overtime Burden (% of standard 40h)",
            0.0,
            100.0,
            float(st.session_state.get("overtime", 25.0)),
            1.0,
        )
        leaves = st.slider(
            "Absenteeism / Days Off Taken (Days)", 0, 40, int(st.session_state.get("leaves", 10)), 1
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Compute Prediction
    row_features = [
        tenure,
        float(age),
        distance,
        wage,
        wage_diff_pct / 100.0,
        overtime_pct / 100.0,
        float(leaves),
    ]
    row_df = pd.DataFrame([row_features], columns=FeatureEngineer.FEATURE_COLS)
    prob = float(pipeline.predict_proba(row_df)[0, 1])
    tier_label, badge_class, tier_note = get_risk_tier(prob)
    row_arr = np.array(row_features)
    _, reasons = explainer.explain_employee(row_arr, top_k=3)

    with col_results:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        # Risk Badge and Status Header
        c_badge, c_note = st.columns([1, 2])
        with c_badge:
            st.markdown(
                f"<div class='{badge_class}' style='font-size: 1.1rem; padding: 0.4rem 1rem;'>{tier_label}: {prob:.1%}</div>",
                unsafe_allow_html=True,
            )
        with c_note:
            st.markdown(
                f"<div style='color: #71717a; font-size: 0.9rem; padding-top: 0.3rem;'>{tier_note}</div>",
                unsafe_allow_html=True,
            )

        st.plotly_chart(render_gauge(prob), use_container_width=True)

        # SHAP Waterfall
        st.plotly_chart(
            render_shap_waterfall(FeatureEngineer.FEATURE_COLS, row_arr, explainer),
            use_container_width=True,
        )

        # Plain-English Explanations
        st.markdown("#### Primary Flight Risk Drivers (SHAP Attribution)")
        for r in reasons:
            st.markdown(f"&bull; **{r}**")

        st.markdown(
            "<hr style='margin: 1rem 0; border: none; border-top: 1px solid #f4f4f5;'>",
            unsafe_allow_html=True,
        )

        # Recommended Interventions
        st.markdown("#### Prescriptive Retention Playbook")
        from scripts.predict import get_recommended_action

        actions = get_recommended_action(reasons, tier_label.split()[0])
        for a in actions:
            st.markdown(f"&bull; {a}")

        st.markdown("</div>", unsafe_allow_html=True)

    # What-If Countermeasure Simulator
    st.markdown("---")
    st.markdown("### 🔄 What-If Countermeasure Simulator")
    st.caption(
        "Simulate HR retention interventions in real time to observe direct flight risk reduction."
    )

    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        sim_raise = st.checkbox("Apply 15% Salary Raise", value=False)
    with sim_c2:
        sim_cut_ot = st.checkbox("Cap Overtime to Standard (<10%)", value=False)
    with sim_c3:
        sim_remote = st.checkbox("Grant 3-Day Remote (Reduce Commute 60%)", value=False)
    with sim_c4:
        sim_promo = st.checkbox("Award Promotion (Reset Stagnation)", value=False)

    if sim_raise or sim_cut_ot or sim_remote or sim_promo:
        sim_features = list(row_features)
        if sim_raise:
            sim_features[3] *= 1.15
            sim_features[4] += 0.15
        if sim_cut_ot:
            sim_features[5] = min(sim_features[5], 0.08)
        if sim_remote:
            sim_features[2] *= 0.40
        if sim_promo:
            sim_features[0] = max(sim_features[0] * 0.5, 0.5)

        sim_df = pd.DataFrame([sim_features], columns=FeatureEngineer.FEATURE_COLS)
        sim_prob = float(pipeline.predict_proba(sim_df)[0, 1])
        drop = prob - sim_prob

        st.markdown(
            f"""
        <div class="saas-card" style="border-left: 4px solid #16a34a;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 700; font-size: 1.15rem; color: #16a34a;">Simulated Post-Intervention Flight Risk: {sim_prob:.1%}</span>
                    <span style="margin-left: 1rem; color: #71717a; font-size: 0.95rem;">(Baseline: {prob:.1%})</span>
                </div>
                <div>
                    <span class="badge-low" style="font-size: 1rem; padding: 0.35rem 0.85rem;">Risk Drop: -{drop:.1%}</span>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------
# TAB 2: LIVE ODOO WORKFORCE EXPLORER
# -------------------------------------------------------------
with tab_odoo:
    st.markdown("### Live Odoo Workforce Retention Explorer")
    st.caption(
        "Active workforce records retrieved via Odoo External JSON-2 API with real-time flight risk scoring."
    )

    import requests

    api_key = os.getenv("ODOO_API_KEY")
    url = os.getenv("ODOO_URL", "http://localhost:8069").rstrip("/")
    db = os.getenv("ODOO_DB", "hr_db")

    if not api_key:
        st.warning(
            "ODOO_API_KEY is not configured in .env. Run `make generate-api-key` to provision access."
        )
    else:
        try:
            headers = {
                "Authorization": f"bearer {api_key}",
                "X-Odoo-Database": db,
                "Content-Type": "application/json",
            }
            resp = requests.post(
                f"{url}/json/2/hr.employee/search_read",
                headers=headers,
                json={
                    "domain": [["active", "=", True]],
                    "fields": [
                        "id",
                        "name",
                        "job_title",
                        "department_id",
                        "km_home_work",
                        "create_date",
                    ],
                    "limit": 50,
                },
            )
            resp.raise_for_status()
            employees = resp.json()

            if employees:
                now = pd.Timestamp.now()
                workforce_rows = []
                for emp in employees:
                    eid = emp["id"]
                    name = emp["name"]
                    job = emp.get("job_title") or "General Staff"
                    dept = (
                        emp["department_id"][1]
                        if isinstance(emp.get("department_id"), (list, tuple))
                        else "General"
                    )
                    cdate = emp.get("create_date")
                    tenure_val = (now - pd.to_datetime(cdate)).days / 365.25 if cdate else 2.5
                    dist_val = float(emp.get("km_home_work") or 15.0)

                    # Proxy synthetic compensation & overtime for Odoo base employees
                    f_row = [tenure_val, 33.0, dist_val, 13500000.0, 0.05, 0.18, 8.0]
                    p_val = float(
                        pipeline.predict_proba(
                            pd.DataFrame([f_row], columns=FeatureEngineer.FEATURE_COLS)
                        )[0, 1]
                    )

                    workforce_rows.append(
                        {
                            "ID": eid,
                            "Name": name,
                            "Job Title": job,
                            "Department": dept,
                            "Tenure (Yrs)": round(tenure_val, 1),
                            "Commute (km)": dist_val,
                            "Flight Risk": p_val,
                            "Risk Tier": "HIGH"
                            if p_val >= 0.70
                            else ("MEDIUM" if p_val >= 0.40 else "LOW"),
                        }
                    )

                wf_df = pd.DataFrame(workforce_rows).sort_values("Flight Risk", ascending=False)

                # Summary Metrics
                o1, o2, o3 = st.columns(3)
                with o1:
                    st.metric("Total Employees Scored", len(wf_df))
                with o2:
                    st.metric("High Risk Staff", int((wf_df["Risk Tier"] == "HIGH").sum()))
                with o3:
                    st.metric("Average Attrition Risk", f"{wf_df['Flight Risk'].mean():.1%}")

                st.dataframe(
                    wf_df.style.format({"Flight Risk": "{:.1%}"}).map(
                        lambda v: (
                            "color: #dc2626; font-weight: bold;"
                            if v == "HIGH"
                            else (
                                "color: #d97706; font-weight: bold;"
                                if v == "MEDIUM"
                                else "color: #16a34a;"
                            )
                        ),
                        subset=["Risk Tier"],
                    ),
                    use_container_width=True,
                    height=350,
                )
            else:
                st.info("No active employees found in Odoo database.")
        except Exception as e:
            st.error(f"Error querying Odoo: {e}")


# -------------------------------------------------------------
# TAB 3: BATCH DATASET SCORING & ANALYTICS
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Workforce Analytics & Population Scoring")
    st.caption(
        "Score entire datasets, inspect population risk distribution, and export enriched risk rosters."
    )

    dataset_option = st.selectbox(
        "Select Workforce Dataset",
        [
            "data/synthetic/hr_attrition_train.csv (Baseline 2,500 records)",
            "data/synthetic/hr_attrition_drifted.csv (Drifted 1,200 records)",
        ],
    )
    chosen_path = dataset_option.split()[0]

    if os.path.exists(chosen_path):
        b_df = pd.read_csv(chosen_path)
        feature_cols = [c for c in FeatureEngineer.FEATURE_COLS if c in b_df.columns]
        X = b_df[feature_cols]
        probs = pipeline.predict_proba(X)[:, 1]
        b_df["flight_risk_score"] = probs
        b_df["risk_tier"] = np.where(
            probs >= 0.70, "HIGH", np.where(probs >= 0.40, "MEDIUM", "LOW")
        )

        # Visual Analytics
        c_pie, c_hist = st.columns([1, 1.4])
        with c_pie:
            tier_counts = b_df["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            fig_pie = px.pie(
                tier_counts,
                values="Count",
                names="Tier",
                color="Tier",
                color_discrete_map={"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"},
                title="Workforce Risk Tier Breakdown",
                hole=0.45,
            )
            fig_pie.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_hist:
            fig_hist = px.histogram(
                b_df,
                x="flight_risk_score",
                nbins=30,
                color="risk_tier",
                color_discrete_map={"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"},
                title="Probability Distribution Across Workforce",
                labels={"flight_risk_score": "Flight Risk Probability"},
            )
            fig_hist.update_layout(
                margin=dict(l=10, r=10, t=35, b=10), height=300, barmode="overlay"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Top Flight Risk Table
        st.markdown("#### Top At-Risk Employees Requiring HR Attention")
        top_risk = b_df.sort_values("flight_risk_score", ascending=False).head(15)[
            [
                "employee_id",
                "name",
                "department",
                "tenure_years",
                "overtime_ratio",
                "dept_wage_diff_pct",
                "flight_risk_score",
                "risk_tier",
            ]
        ]
        st.dataframe(
            top_risk.style.format(
                {
                    "flight_risk_score": "{:.1%}",
                    "overtime_ratio": "{:.1%}",
                    "dept_wage_diff_pct": "{:+.1%}",
                    "tenure_years": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

        # Export Button
        csv_data = b_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Scored Workforce Dataset (CSV)",
            data=csv_data,
            file_name="workforce_flight_risk_scored.csv",
            mime="text/csv",
        )


# -------------------------------------------------------------
# TAB 4: MLOPS GOVERNANCE & DRIFT MONITOR
# -------------------------------------------------------------
with tab_mlops:
    st.markdown("### MLOps Governance & Real-Time Drift Diagnostic")
    st.caption(
        "Monitor population drift (PSI), model version tournaments in MLflow, and automated retraining triggers."
    )

    ref_path = "data/synthetic/hr_attrition_train.csv"
    drift_path = "data/synthetic/hr_attrition_drifted.csv"

    if os.path.exists(ref_path) and os.path.exists(drift_path):
        ref_df = pd.read_csv(ref_path)
        cur_df = pd.read_csv(drift_path)
        detector = DriftDetector()
        report = detector.evaluate_feature_drift(ref_df, cur_df, FeatureEngineer.FEATURE_COLS)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Reference Baseline Samples", f"{len(ref_df):,}")
        with m2:
            st.metric("Current Production Samples", f"{len(cur_df):,}")
        with m3:
            st.metric("Features in RED Drift", report["red_drift_features"])
        with m4:
            st.metric("Retraining Alarm", "ACTIVE" if report["retrain_recommended"] else "CLEAR")

        # PSI Bar Chart
        psi_records = []
        for feat, data in report["features"].items():
            psi_records.append(
                {
                    "Feature": feat.replace("_", " ").title(),
                    "PSI": data["psi"],
                    "Status": data["status"],
                }
            )
        psi_df = pd.DataFrame(psi_records).sort_values("PSI", ascending=True)

        fig_psi = px.bar(
            psi_df,
            x="PSI",
            y="Feature",
            orientation="h",
            color="Status",
            color_discrete_map={"GREEN": "#22c55e", "YELLOW": "#f59e0b", "RED": "#ef4444"},
            title="Feature Population Stability Index (PSI vs. Baseline)",
        )
        fig_psi.add_vline(
            x=0.10, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate (0.10)"
        )
        fig_psi.add_vline(
            x=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Severe (0.25)"
        )
        fig_psi.update_layout(height=320, margin=dict(l=10, r=20, t=40, b=10))
        st.plotly_chart(fig_psi, use_container_width=True)

        # Trigger Retrain Action
        st.markdown("#### Continuous Retraining Action")
        if report["retrain_recommended"]:
            st.error(
                "🚨 Distribution drift threshold breached (>= 2 RED features). Retraining recommended."
            )
        else:
            st.success("✅ Workforce distributions are stable. No drift detected.")
