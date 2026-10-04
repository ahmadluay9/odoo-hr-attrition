"""Deterministic synthetic HR workforce and attrition dataset generator.

Generates statistically grounded employee profiles and realistic voluntary exit
signals aligned with Odoo 20 data models (hr.employee, hr_version, hr.leave, hr.attendance).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

DEPARTMENTS = [
    "Engineering",
    "Sales",
    "Marketing",
    "Human Resources",
    "Finance",
    "Operations",
    "Consulting",
]

DEPARTMENT_BASE_SALARIES = {
    "Engineering": 15000000,
    "Finance": 14000000,
    "Sales": 12000000,
    "Consulting": 11500000,
    "Marketing": 11000000,
    "Operations": 9500000,
    "Human Resources": 10000000,
}


def generate_synthetic_hr_dataset(
    num_employees: int = 2500,
    seed: int = 42,
    window_months: int = 12,
) -> pd.DataFrame:
    """Generate realistic employee records with probabilistic attrition ground truth.

    Args:
        num_employees: Total number of synthetic employee rows to simulate.
        seed: Random state seed for reproducible dataset generation.
        window_months: Prediction horizon in months (6 or 12).

    Returns:
        pd.DataFrame containing demographic, compensation, operational, and target columns.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []
    for emp_id in range(1, num_employees + 1):
        dept = random.choice(DEPARTMENTS)
        age = int(np.clip(np.random.normal(36, 8.5), 21, 64))
        tenure_years = round(float(np.clip(np.random.exponential(3.8), 0.3, 20.0)), 1)
        tenure_months = int(tenure_years * 12)
        distance_km = int(np.clip(np.random.gamma(3.2, 4.5), 1, 85))

        base_dept_wage = DEPARTMENT_BASE_SALARIES[dept]
        tenure_multiplier = 1.0 + (0.045 * min(tenure_years, 10.0))
        individual_variation = np.random.normal(1.0, 0.16)
        wage = int(base_dept_wage * tenure_multiplier * individual_variation)

        # Overtime: right-skewed distribution
        overtime_ratio = round(float(np.random.beta(2.0, 5.0) * 0.85), 3)

        # Absenteeism / leaves taken over past 180 days
        leave_days = int(np.clip(np.random.poisson(9), 0, 35))

        # Time since last promotion (cannot exceed employee tenure)
        max_promo_gap = min(tenure_years, 8.0)
        years_since_promo = round(float(np.random.uniform(0.2, max_promo_gap)), 1)

        # Number of manager changes in past 24 months
        manager_changes = int(np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.11, 0.04]))

        # Performance rating (1 to 5)
        appraisal_score = round(float(np.clip(np.random.normal(3.4, 0.7), 1.0, 5.0)), 1)

        # Log-odds formulation for voluntary attrition within window_months
        # Base exit rate ~ 12-16%
        log_odds = -2.35

        # Compensation penalty: underpaid compared to expected base wage
        dept_diff_pct = (wage - base_dept_wage) / base_dept_wage
        if dept_diff_pct < -0.12:
            log_odds += 0.95
        elif dept_diff_pct > 0.15:
            log_odds -= 0.60

        # Overtime burnout penalty
        if overtime_ratio > 0.25:
            log_odds += 1.10
        elif overtime_ratio > 0.15:
            log_odds += 0.55

        # Promotion stagnation penalty
        if years_since_promo >= 3.0:
            log_odds += 0.85

        # Long commute friction
        if distance_km > 35:
            log_odds += 0.50

        # Manager churn friction
        if manager_changes >= 2:
            log_odds += 0.45

        # Early-tenure restless exit risk (< 1.5 years)
        if tenure_years < 1.5:
            log_odds += 0.40

        # Performance sentiment: low appraisal triggers exit exploration
        if appraisal_score < 2.5:
            log_odds += 0.65

        # Probability of leaving within target window
        attrition_prob = 1.0 / (1.0 + np.exp(-log_odds))
        will_leave = 1 if np.random.rand() < attrition_prob else 0

        # Time-to-event for survival analysis: duration until event or censoring
        if will_leave:
            # Expected months to exit within window
            time_to_exit = max(1, int(np.random.uniform(1, window_months)))
            duration_months = tenure_months + time_to_exit
            event_observed = 1
        else:
            duration_months = tenure_months + window_months
            event_observed = 0

        records.append(
            {
                "employee_id": emp_id,
                "name": f"Synthetic Employee {emp_id:04d}",
                "department": dept,
                "age": age,
                "tenure_years": tenure_years,
                "tenure_months": tenure_months,
                "distance_km": distance_km,
                "wage": wage,
                "dept_wage_diff_pct": round(dept_diff_pct, 4),
                "overtime_ratio": overtime_ratio,
                "leave_days_taken": leave_days,
                "years_since_promo": years_since_promo,
                "manager_change_count": manager_changes,
                "appraisal_score": appraisal_score,
                "duration_months": duration_months,
                "event_observed": event_observed,
                "attrition_prob_true": round(attrition_prob, 4),
                "will_leave": will_leave,
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic HR attrition dataset.")
    parser.add_argument("--samples", type=int, default=2500, help="Number of records to simulate")
    parser.add_argument("--output", type=str, default="data/synthetic/hr_attrition_train.csv")
    parser.add_argument("--window", type=int, default=12, help="Prediction horizon in months")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_hr_dataset(
        num_employees=args.samples,
        seed=args.seed,
        window_months=args.window,
    )
    df.to_csv(out_path, index=False)

    attrition_rate = df["will_leave"].mean()
    print(f"[SUCCESS] Synthetic dataset written to: {out_path}")
    print(f"          Total Samples : {len(df)}")
    print(
        f"          Attrition Rate: {attrition_rate:.2%} ({df['will_leave'].sum()} positive cases)"
    )
    print(f"          Features      : {len(df.columns)} columns generated.")


if __name__ == "__main__":
    main()
