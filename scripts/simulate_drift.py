"""Simulate realistic HR workforce drift scenarios to stress-test MLOps drift detection and retraining pipelines.

Supported Scenarios:
1. 'burnout': Severe overtime surge & elevated absenteeism following corporate restructuring / hiring freeze.
2. 'rto': Return-To-Office mandate creating long-distance commute friction.
3. 'market_shock': Rapid inflation & competitor wage spikes causing internal compensation to lag.
4. 'compound': Enterprise-wide workforce shock combining overtime burnout, commute friction, and wage disparity.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.engineer import FeatureEngineer
from src.monitoring.drift_detector import DriftDetector


def simulate_burnout_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate severe overtime burden and absenteeism shock."""
    drifted = df.copy()
    # Overtime shifts upward significantly (+0.20 to +0.35 mean shift)
    overtime_shock = rng.normal(loc=0.25, scale=0.08, size=len(drifted))
    drifted["overtime_ratio"] = np.clip(drifted["overtime_ratio"] + overtime_shock, 0.0, 1.2)

    # Sick / unplanned leave days increase dramatically
    leave_shock = rng.poisson(lam=6.5, size=len(drifted))
    drifted["leave_days_taken"] = np.clip(drifted["leave_days_taken"] + leave_shock, 0, 45)

    # Recalculate true attrition probability due to burnout
    burnout_factor = 0.35 * (drifted["overtime_ratio"] / 0.5) + 0.20 * (
        drifted["leave_days_taken"] / 20.0
    )
    new_prob = np.clip(drifted["attrition_prob_true"] + burnout_factor * 0.4, 0.05, 0.95)
    drifted["attrition_prob_true"] = new_prob
    drifted["will_leave"] = (rng.uniform(0, 1, size=len(drifted)) < new_prob).astype(int)
    drifted["event_observed"] = drifted["will_leave"]
    return drifted


def simulate_rto_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate strict Return-to-Office mandate causing commute friction."""
    drifted = df.copy()
    # Distance distribution shifts right as remote workers in suburbs are forced to commute
    commute_mult = rng.uniform(1.6, 2.4, size=len(drifted))
    drifted["distance_km"] = np.clip(drifted["distance_km"] * commute_mult, 2.0, 95.0)

    # Recalculate attrition
    rto_factor = np.where(drifted["distance_km"] > 30, 0.30, 0.05)
    new_prob = np.clip(drifted["attrition_prob_true"] + rto_factor, 0.05, 0.95)
    drifted["attrition_prob_true"] = new_prob
    drifted["will_leave"] = (rng.uniform(0, 1, size=len(drifted)) < new_prob).astype(int)
    drifted["event_observed"] = drifted["will_leave"]
    return drifted


def simulate_market_shock_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate macro inflation & external salary hikes causing internal salary lag."""
    drifted = df.copy()
    # Department wage difference shifts heavily negative (peers outside pay 25% more)
    wage_discount = rng.normal(loc=0.22, scale=0.06, size=len(drifted))
    drifted["dept_wage_diff_pct"] = np.clip(
        drifted["dept_wage_diff_pct"] - wage_discount, -0.65, 0.40
    )

    # Stagnation penalty
    comp_factor = np.where(drifted["dept_wage_diff_pct"] < -0.15, 0.35, 0.05)
    new_prob = np.clip(drifted["attrition_prob_true"] + comp_factor, 0.05, 0.95)
    drifted["attrition_prob_true"] = new_prob
    drifted["will_leave"] = (rng.uniform(0, 1, size=len(drifted)) < new_prob).astype(int)
    drifted["event_observed"] = drifted["will_leave"]
    return drifted


def simulate_compound_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate enterprise-wide workforce crisis combining burnout, commute, and compensation disparity."""
    df1 = simulate_burnout_drift(df, rng)
    df2 = simulate_rto_drift(df1, rng)
    return simulate_market_shock_drift(df2, rng)


SCENARIOS = {
    "burnout": simulate_burnout_drift,
    "rto": simulate_rto_drift,
    "market_shock": simulate_market_shock_drift,
    "compound": simulate_compound_drift,
}


def main():
    parser = argparse.ArgumentParser(
        description="Simulate realistic HR workforce data drift scenarios."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="data/synthetic/hr_attrition_train.csv",
        help="Path to baseline reference training CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/hr_attrition_drifted.csv",
        help="Path to save drifted dataset",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="compound",
        choices=list(SCENARIOS.keys()),
        help="Drift scenario to inject: 'burnout', 'rto', 'market_shock', or 'compound'",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Number of drifted production samples to synthesize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        print(f"[-] Baseline file not found at {args.baseline}. Generating baseline dataset...")
        os.system(f"python scripts/generate_hr_data.py --samples 2500 --output {args.baseline}")

    ref_df = pd.read_csv(args.baseline)
    rng = np.random.default_rng(args.seed)

    # Subsample baseline to derive drifted population
    sample_size = min(args.samples, len(ref_df))
    base_sample = ref_df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    transform_fn = SCENARIOS[args.scenario]
    drifted_df = transform_fn(base_sample, rng)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    drifted_df.to_csv(args.output, index=False)
    print(f"[+] Successfully generated drifted workforce dataset: '{args.output}'")
    print(f"[+] Scenario Applied : '{args.scenario.upper()}' ({len(drifted_df)} records)")

    # Execute drift diagnostic
    feature_cols = [c for c in FeatureEngineer.FEATURE_COLS if c in ref_df.columns]
    detector = DriftDetector()
    report = detector.evaluate_feature_drift(ref_df, drifted_df, feature_cols)

    print("\n" + "=" * 75)
    print(f"       DRIFT SIMULATION DIAGNOSTIC REPORT: SCENARIO '{args.scenario.upper()}'")
    print("=" * 75)
    print(f"  Reference Baseline Records : {len(ref_df):,}")
    print(f"  Drifted Production Records : {len(drifted_df):,}")
    print(
        f"  Features In RED Drift      : {report['red_drift_features']} / {report['total_features_evaluated']}"
    )
    print(
        f"  Features In YELLOW Drift   : {report['yellow_drift_features']} / {report['total_features_evaluated']}"
    )
    print(
        f"  Retraining Triggered       : {'YES (URGENT ACTION REQUIRED)' if report['retrain_recommended'] else 'NO'}"
    )
    print("=" * 75)
    print(
        f"  {'Feature':<22} | {'Baseline Mean':<14} | {'Drifted Mean':<14} | {'PSI':<8} | {'Status'}"
    )
    print("-" * 75)

    for feat in feature_cols:
        b_mean = ref_df[feat].mean()
        d_mean = drifted_df[feat].mean()
        f_data = report["features"].get(feat, {"psi": 0.0, "status": "GREEN"})
        psi_val = f_data["psi"]
        status = f_data["status"]

        status_str = (
            f"\033[32m{status}\033[0m"
            if status == "GREEN"
            else (
                f"\033[33m{status}\033[0m"
                if status == "YELLOW"
                else f"\033[31m{status} (DRIFT)\033[0m"
            )
        )
        print(f"  {feat:<22} | {b_mean:<14.3f} | {d_mean:<14.3f} | {psi_val:<8.4f} | {status_str}")

    print("=" * 75)
    print("  Baseline Turnover (y=1)  : {:.1%}".format(ref_df["will_leave"].mean()))
    print("  Drifted Turnover (y=1)   : {:.1%}".format(drifted_df["will_leave"].mean()))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
