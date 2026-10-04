"""Detect data drift and concept drift between reference baseline and production workforce.

Computes:
- Population Stability Index (PSI) per feature.
- Kolmogorov-Smirnov (KS) test for statistical distribution difference.
- Target/Prior probability drift.
- Logs drift statistics to MLflow.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.features.engineer import FeatureEngineer
from src.monitoring.drift_detector import DriftDetector

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(description="Detect feature and target drift.")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/synthetic/hr_attrition_train.csv",
        help="Path to reference baseline dataset",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Path to current/production dataset. If omitted, uses active workforce or creates evaluation split.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )
    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"[-] Reference dataset not found at {args.reference}")
        sys.exit(1)

    ref_df = pd.read_csv(args.reference)

    if args.current and os.path.exists(args.current):
        curr_df = pd.read_csv(args.current)
    else:
        # If no current dataset is provided, use the last 20% or active workforce simulation
        curr_df = ref_df.sample(frac=0.35, random_state=123).copy()

    feature_cols = [c for c in FeatureEngineer.FEATURE_COLS if c in ref_df.columns]

    detector = DriftDetector()
    report = detector.evaluate_feature_drift(ref_df, curr_df, feature_cols)

    # Log to MLflow
    try:
        detector.log_drift_to_mlflow(report)
    except Exception as e:
        print(f"[!] Warning: Could not log drift to MLflow: {e}")

    if args.json:
        import json

        print(json.dumps(report, indent=2))
        return

    print("\n" + "=" * 70)
    print("                    DATA DRIFT DETECTION REPORT")
    print("=" * 70)
    print(f"  Reference Samples     : {len(ref_df):,}")
    print(f"  Current/Inference Set : {len(curr_df):,}")
    print(f"  Features Evaluated    : {report['total_features_evaluated']}")
    print(
        f"  Drift Status Summary  : {report['red_drift_features']} RED, {report['yellow_drift_features']} YELLOW"
    )
    print(
        f"  Retrain Recommended   : {'YES (ACTION REQUIRED)' if report['retrain_recommended'] else 'NO (STABLE)'}"
    )
    print("=" * 70)
    print(f"  {'Feature':<25} | {'PSI':<8} | {'KS Stat':<8} | {'KS p-val':<8} | {'Status'}")
    print("-" * 70)

    for feat, data in report["features"].items():
        status_str = (
            f"\033[32m{data['status']}\033[0m"
            if data["status"] == "GREEN"
            else (
                f"\033[33m{data['status']}\033[0m"
                if data["status"] == "YELLOW"
                else f"\033[31m{data['status']}\033[0m"
            )
        )
        print(
            f"  {feat:<25} | {data['psi']:<8.4f} | {data['ks_statistic']:<8.4f} | {data['ks_pvalue']:<8.4f} | {status_str}"
        )

    print("=" * 70)
    print(
        "  Thresholds: PSI < 0.10: GREEN (No Drift) | 0.10 - 0.25: YELLOW (Moderate) | >= 0.25: RED (Significant)"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
