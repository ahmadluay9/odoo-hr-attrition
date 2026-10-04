"""Evaluate model predictions against ground truth turnover outcomes."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.features.engineer import FeatureEngineer
from src.models.registry import ModelRegistryManager
from src.monitoring.performance import PerformanceMonitor

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model performance against ground-truth departures."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/synthetic/hr_attrition_train.csv",
        help="Evaluation dataset with ground truth outcomes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.flight_risk_medium_threshold,
        help="Classification decision threshold (default: 0.40)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[-] Evaluation data file not found at {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    target_col = "will_leave" if "will_leave" in df.columns else "event_observed"
    feature_cols = [c for c in FeatureEngineer.FEATURE_COLS if c in df.columns]

    # Load active Champion model from MLflow
    registry = ModelRegistryManager()
    champ_version = registry.get_champion_version()

    if champ_version is None:
        print("[-] No Champion model found registered in MLflow. Run 'make train' first.")
        sys.exit(1)

    print(f"[+] Loaded Champion Model Version: {champ_version.version}")
    model = registry.load_champion_model()

    # Predict flight risk probabilities
    X = df[feature_cols]
    y_true = df[target_col].values

    # If the loaded model is an MLflow pyfunc model:
    try:
        y_prob = model.predict(X)
        if hasattr(y_prob, "ndim") and y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
    except Exception:
        # Fallback if scikit-learn pipeline underlying
        underlying = getattr(model, "_model_impl", model)
        y_prob = underlying.predict_proba(X)[:, 1]

    monitor = PerformanceMonitor()
    metrics = monitor.evaluate_predictions(y_true=y_true, y_prob=y_prob, threshold=args.threshold)

    # Log to MLflow
    try:
        monitor.log_performance_to_mlflow(metrics)
    except Exception as e:
        print(f"[!] Warning: Could not log performance to MLflow: {e}")

    if args.json:
        import json

        print(json.dumps(metrics, indent=2))
        return

    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 65)
    print("             MODEL PERFORMANCE & SLA MONITORING")
    print("=" * 65)
    print(f"  Champion Version    : {champ_version.version}")
    print(f"  Total Samples       : {metrics['sample_count']:,}")
    print(
        f"  Observed Departures : {metrics['positive_count']:,} ({metrics['positive_count'] / metrics['sample_count']:.1%})"
    )
    print(f"  Decision Threshold  : {metrics['threshold_used']:.2f}")
    print("-" * 65)
    print(
        f"  PR-AUC (Primary)    : {metrics['pr_auc']:.4f}  (SLA Min: {settings.retrain_prauc_threshold:.2f})"
    )
    print(f"  ROC-AUC             : {metrics['roc_auc']:.4f}")
    print(f"  Brier Score         : {metrics['brier_score']:.4f}  (Probability Error)")
    print(f"  F1-Score            : {metrics['f1_score']:.4f}")
    print(f"  Precision           : {metrics['precision']:.4f}")
    print(f"  Recall              : {metrics['recall']:.4f}")
    print("-" * 65)
    sla_status = (
        "\033[31mBREACHED (RETRAIN REQUIRED)\033[0m"
        if metrics["sla_breached"]
        else "\033[32mHEALTHY (WITHIN SLA)\033[0m"
    )
    print(f"  SLA Status          : {sla_status}")
    print("=" * 65)
    print("  CONFUSION MATRIX:")
    print(f"    True Negatives  (Retained Correctly) : {cm['true_negatives']:,}")
    print(f"    False Positives (False Alarms)        : {cm['false_positives']:,}")
    print(f"    False Negatives (Missed Departures)   : {cm['false_negatives']:,}")
    print(f"    True Positives  (Caught Departures)   : {cm['true_positives']:,}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
