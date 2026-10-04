"""Inference CLI for Flight Risk Prediction & Explainability.

Supports:
1. Feature arguments CLI: Score custom employee parameters directly.
2. Dataset batch/single prediction: Score records from a CSV file.
3. Live Odoo ERP employee prediction: Score active employees from Odoo JSON-2 API.
4. Curated archetypes demo: Compare Low, Medium, and High risk profiles.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Set path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.explainability.explainer import FlightRiskExplainer
from src.features.engineer import FeatureEngineer
from src.models.registry import ModelRegistryManager

settings = get_settings()


def get_risk_badge(
    prob: float, high_thresh: float = 0.70, med_thresh: float = 0.40
) -> tuple[str, str]:
    """Return colored risk badge and label based on thresholds."""
    if prob >= high_thresh:
        return f"\033[1;31m[HIGH RISK: {prob:.1%}]\033[0m", "HIGH"
    elif prob >= med_thresh:
        return f"\033[1;33m[MEDIUM RISK: {prob:.1%}]\033[0m", "MEDIUM"
    else:
        return f"\033[1;32m[LOW RISK: {prob:.1%}]\033[0m", "LOW"


def get_recommended_action(reasons: list[str], risk_level: str) -> list[str]:
    """Map top SHAP explanatory drivers to actionable HR retention interventions."""
    if risk_level == "LOW":
        return ["Maintain standard engagement cadence and routine quarterly check-ins."]

    actions = []
    text = " ".join(reasons).lower()
    if "compensation" in text or "salary" in text:
        actions.append(
            "Initiate immediate compensation review against market and department benchmark."
        )
    if "overtime" in text or "burnout" in text:
        actions.append(
            "Rebalance workload allocation, enforce overtime caps, and check for understaffing."
        )
    if "commuting" in text or "distance" in text:
        actions.append("Offer flexible hybrid/remote work arrangement (2-3 days remote/week).")
    if "tenure" in text or "advancement" in text:
        actions.append(
            "Conduct career development 1-on-1; establish promotion timeline or role expansion."
        )
    if "absenteeism" in text:
        actions.append(
            "Schedule confidential HR wellness check-in to identify potential team friction or burnout."
        )

    if not actions:
        actions.append("Conduct proactive retention conversation to understand employee sentiment.")

    return actions


def format_prediction_card(
    name: str,
    emp_id: str | int,
    features_dict: dict,
    prob: float,
    reasons: list[str],
    risk_level: str,
    badge: str,
) -> str:
    """Format single employee prediction result into terminal card."""
    lines = []
    lines.append("+" + "-" * 73 + "+")
    lines.append(f"|  EMPLOYEE FLIGHT RISK ASSESSMENT : {name:<36} ID: {emp_id:<4} |")
    lines.append("+" + "-" * 73 + "+")
    lines.append(f"|  Flight Risk Status  : {badge:<55} |")
    lines.append(
        f"|  Probability Score   : {prob:>6.2%} (Horizon: Next 6-12 Months)                   |"
    )
    lines.append("|" + " " * 73 + "|")
    lines.append("|  Employee Baseline Features:                                           |")
    lines.append(
        f"|    * Tenure: {features_dict.get('tenure_years', 0):.1f} yrs   * Age: {features_dict.get('age', 0):.0f} yrs   * Commute: {features_dict.get('distance_km', 0):.0f} km   * Salary: IDR {features_dict.get('wage', 0):>10,.0f} |"
    )
    lines.append(
        f"|    * Dept Wage Diff: {features_dict.get('dept_wage_diff_pct', 0):>+6.1%}   * Overtime: {features_dict.get('overtime_ratio', 0):>5.1%}   * Absenteeism: {features_dict.get('leave_days_taken', 0):>2.0f} days      |"
    )
    lines.append("|" + " " * 73 + "|")
    lines.append("|  Key Explanatory Risk Drivers (SHAP Attribution):                     |")
    for r in reasons:
        lines.append(f"|    -> {r:<65} |")
    lines.append("|" + " " * 73 + "|")
    lines.append("|  Recommended HR Retention Interventions:                              |")
    actions = get_recommended_action(reasons, risk_level)
    for a in actions:
        lines.append(f"|    [*] {a:<64} |")
    lines.append("+" + "-" * 73 + "+")
    return "\n".join(lines)


def predict_single(
    pipeline,
    explainer: FlightRiskExplainer,
    feature_names: list[str],
    features: dict,
    name: str = "Anonymous Employee",
    emp_id: str | int = 1,
):
    """Run prediction on a single feature dictionary."""
    row_vals = [float(features.get(f, 0.0)) for f in feature_names]
    row_arr = np.array(row_vals)
    df_row = pd.DataFrame([row_vals], columns=feature_names)

    prob = float(pipeline.predict_proba(df_row)[0, 1])
    badge, risk_level = get_risk_badge(
        prob,
        high_thresh=settings.flight_risk_high_threshold,
        med_thresh=settings.flight_risk_medium_threshold,
    )
    _, reasons = explainer.explain_employee(row_arr, top_k=settings.shap_top_k_reasons)

    print(
        "\n"
        + format_prediction_card(name, emp_id, features, prob, reasons, risk_level, badge)
        + "\n"
    )


def run_archetypes_demo(pipeline, explainer: FlightRiskExplainer, feature_names: list[str]):
    """Demonstrate inference across three realistic HR archetypes."""
    print("\n" + "=" * 75)
    print("        WORKFORCE ATTRITION PREDICTION: ARCHETYPE DEMONSTRATION")
    print("=" * 75)

    archetypes = [
        {
            "name": "Sarah Chen (Senior Architect)",
            "id": "ARCH-01",
            "features": {
                "tenure_years": 4.5,
                "age": 34,
                "distance_km": 12.0,
                "wage": 16500000.0,
                "dept_wage_diff_pct": 0.18,
                "overtime_ratio": 0.08,
                "leave_days_taken": 7.0,
            },
        },
        {
            "name": "Marcus Vance (Operations Specialist)",
            "id": "ARCH-02",
            "features": {
                "tenure_years": 2.2,
                "age": 29,
                "distance_km": 36.0,
                "wage": 12000000.0,
                "dept_wage_diff_pct": -0.05,
                "overtime_ratio": 0.32,
                "leave_days_taken": 12.0,
            },
        },
        {
            "name": "David Miller (Lead Software Engineer)",
            "id": "ARCH-03",
            "features": {
                "tenure_years": 3.8,
                "age": 31,
                "distance_km": 42.0,
                "wage": 10500000.0,
                "dept_wage_diff_pct": -0.28,
                "overtime_ratio": 0.55,
                "leave_days_taken": 18.0,
            },
        },
    ]

    for arch in archetypes:
        predict_single(
            pipeline, explainer, feature_names, arch["features"], arch["name"], arch["id"]
        )


def predict_from_dataset(
    pipeline,
    explainer: FlightRiskExplainer,
    feature_names: list[str],
    csv_path: str,
    emp_id: int | None = None,
    limit: int = 5,
):
    """Predict for records from a CSV file."""
    if not os.path.exists(csv_path):
        print(f"[-] CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} records from {csv_path}")

    if emp_id is not None:
        target = df[df["employee_id"] == emp_id]
        if target.empty:
            print(f"[-] Employee ID {emp_id} not found in dataset.")
            sys.exit(1)
        records = target.to_dict("records")
    else:
        records = df.head(limit).to_dict("records")

    for rec in records:
        name = rec.get("name", f"Employee #{rec.get('employee_id', 'N/A')}")
        eid = rec.get("employee_id", "N/A")
        predict_single(pipeline, explainer, feature_names, rec, name, eid)


def predict_from_odoo(
    pipeline,
    explainer: FlightRiskExplainer,
    feature_names: list[str],
    emp_id: int | None = None,
    limit: int = 5,
):
    """Fetch active employees from live Odoo ERP via External JSON-2 API and run predictions."""
    import requests

    api_key = os.getenv("ODOO_API_KEY")
    url = os.getenv("ODOO_URL", "http://localhost:8069").rstrip("/")
    db = os.getenv("ODOO_DB", "hr_db")

    if not api_key:
        print("[-] ODOO_API_KEY not configured. Run 'make generate-api-key' first.")
        sys.exit(1)

    headers = {
        "Authorization": f"bearer {api_key}",
        "X-Odoo-Database": db,
        "Content-Type": "application/json",
    }

    domain = [["active", "=", True]]
    if emp_id is not None:
        domain.append(["id", "=", emp_id])

    payload = {
        "domain": domain,
        "fields": ["id", "name", "job_title", "department_id", "km_home_work", "create_date"],
        "limit": limit,
    }

    try:
        resp = requests.post(f"{url}/json/2/hr.employee/search_read", headers=headers, json=payload)
        resp.raise_for_status()
        employees = resp.json()
    except Exception as e:
        print(f"[-] Failed to fetch employees from Odoo: {e}")
        sys.exit(1)

    if not employees:
        print(f"[-] No active employees found matching criteria (ID={emp_id}).")
        return

    print(f"\n[+] Scoring {len(employees)} active employees from live Odoo ERP...")

    now = pd.Timestamp.now()
    for emp in employees:
        eid = emp["id"]
        name = emp["name"]
        create_date = emp.get("create_date")
        tenure = (now - pd.to_datetime(create_date)).days / 365.25 if create_date else 2.5
        distance = float(emp.get("km_home_work") or 15.0)

        # Baseline features (using typical department wage if contract module not installed)
        feat_dict = {
            "tenure_years": tenure,
            "age": 33.0,
            "distance_km": distance,
            "wage": 13500000.0,
            "dept_wage_diff_pct": 0.05,
            "overtime_ratio": 0.18,
            "leave_days_taken": 8.0,
        }

        predict_single(pipeline, explainer, feature_names, feat_dict, name, eid)


def main():
    parser = argparse.ArgumentParser(
        description="Predict Employee Flight Risk and generate SHAP explainability insights."
    )
    parser.add_argument(
        "--sample", action="store_true", help="Run archetypes demo (Low, Medium, High risk)"
    )
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset to score")
    parser.add_argument(
        "--employee-id", type=int, default=None, help="Employee ID to score from data or Odoo"
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Max records to score when reading from data or Odoo"
    )
    parser.add_argument(
        "--odoo", action="store_true", help="Score live active employees from Odoo ERP"
    )

    # Custom ad-hoc feature inputs
    parser.add_argument("--tenure", type=float, default=None, help="Tenure at company in years")
    parser.add_argument("--age", type=float, default=None, help="Employee age in years")
    parser.add_argument("--distance", type=float, default=None, help="Commuting distance in km")
    parser.add_argument("--wage", type=float, default=None, help="Monthly base salary (IDR)")
    parser.add_argument(
        "--dept-wage-diff",
        type=float,
        default=None,
        help="Salary difference vs dept avg (-0.25 = -25%)",
    )
    parser.add_argument(
        "--overtime", type=float, default=None, help="Overtime ratio (0.30 = 30% overtime)"
    )
    parser.add_argument(
        "--leaves", type=float, default=None, help="Absenteeism / leaves taken in days"
    )

    args = parser.parse_args()

    # Load Active Champion Model from MLflow Model Registry
    registry = ModelRegistryManager()
    print("[+] Loading Champion Model from MLflow Model Registry...")
    champion = registry.get_champion_version()
    if champion:
        print(f"[+] Loaded Champion Model Version: {champion.version} (Alias: @champion)")
    else:
        print("[+] Loading Latest Registered Model...")

    pyfunc_model = registry.load_champion_model()
    # Unwrap pipeline
    if hasattr(pyfunc_model, "_model_impl") and hasattr(pyfunc_model._model_impl, "sklearn_model"):
        pipeline = pyfunc_model._model_impl.sklearn_model
    else:
        pipeline = pyfunc_model

    feature_names = FeatureEngineer.FEATURE_COLS
    explainer = FlightRiskExplainer(pipeline, feature_names)

    # Route based on arguments:
    # 1. Custom features CLI:
    if args.tenure is not None or args.overtime is not None:
        custom_feat = {
            "tenure_years": args.tenure if args.tenure is not None else 3.0,
            "age": args.age if args.age is not None else 32.0,
            "distance_km": args.distance if args.distance is not None else 20.0,
            "wage": args.wage if args.wage is not None else 13000000.0,
            "dept_wage_diff_pct": args.dept_wage_diff if args.dept_wage_diff is not None else 0.0,
            "overtime_ratio": args.overtime if args.overtime is not None else 0.15,
            "leave_days_taken": args.leaves if args.leaves is not None else 8.0,
        }
        predict_single(
            pipeline, explainer, feature_names, custom_feat, "Custom Parameter Query", "CLI-01"
        )
        return

    # 2. Live Odoo ERP:
    if args.odoo:
        predict_from_odoo(
            pipeline, explainer, feature_names, emp_id=args.employee_id, limit=args.limit
        )
        return

    # 3. CSV Dataset:
    if args.data:
        predict_from_dataset(
            pipeline, explainer, feature_names, args.data, emp_id=args.employee_id, limit=args.limit
        )
        return

    # Default / Sample Demo:
    run_archetypes_demo(pipeline, explainer, feature_names)


if __name__ == "__main__":
    main()
