"""Automated continuous retraining pipeline managing drift detection, trigger evaluation, tuning, and champion promotion."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.monitoring.retrainer import RetrainingOrchestrator

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(
        description="Continuous retraining pipeline with drift triggers and champion promotion."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/synthetic/hr_attrition_train.csv",
        help="Training dataset",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning on the challenger model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of drift or SLA triggers",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Path to current/drifted dataset to evaluate against baseline reference",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Challenger model architecture",
    )
    args = parser.parse_args()

    orchestrator = RetrainingOrchestrator()

    if not os.path.exists(args.data):
        print(f"[-] Training data file not found at {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    print(f"[+] Loaded {len(df)} reference records from {args.data}")

    # Step 1: Check Triggers (Drift & SLA)
    ref_df = df
    if args.current and os.path.exists(args.current):
        curr_df = pd.read_csv(args.current)
        print(f"[+] Loaded {len(curr_df)} current/drifted records from {args.current}")
        training_df = pd.concat([df, curr_df], ignore_index=True)
    else:
        curr_df = df.sample(frac=0.40, random_state=42)  # simulation of current active batch
        training_df = df

    print("\n[+] Evaluating Drift & Performance SLA triggers...")
    triggers = orchestrator.check_retrain_triggers(
        reference_df=ref_df,
        current_df=curr_df,
    )

    should_retrain = args.force or triggers["retrain_needed"]

    print("=" * 65)
    print("             CONTINUOUS RETRAINING TRIGGER EVALUATION")
    print("=" * 65)
    for r in triggers["reasons"]:
        print(f"  * {r}")
    print(f"  * Force Override: {args.force}")
    print(
        f"  * ACTION: {'TRIGGERING RETRAINING PIPELINE' if should_retrain else 'RETRAINING NOT TRIGGERED'}"
    )
    print("=" * 65 + "\n")

    if not should_retrain:
        print("[+] Active Champion model remains optimal. No retraining needed.")
        return

    # Step 2: Execute Retraining & Champion-Challenger Tournament
    print(
        f"[+] Launching Challenger training ({args.model_type.upper()}, Hyperparameter Tuning={args.tune})..."
    )
    result = orchestrator.execute_retraining(
        training_df=training_df,
        tune=args.tune,
        model_type=args.model_type,
        force_champion=args.force,
    )

    m = result["metrics"]
    print("\n" + "=" * 65)
    print("          CHALLENGER TRAINING & TOURNAMENT RESULTS")
    print("=" * 65)
    print(f"  Challenger Version  : {result['version']}")
    print(f"  Tuned Optimization  : {result['tuned']}")
    print(f"  Challenger PR-AUC   : {m['pr_auc']:.4f}")
    print(f"  Challenger ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"  Challenger F1-Score : {m['f1_score']:.4f}")
    print(f"  Challenger Brier    : {m['brier_score']:.4f}")
    print("-" * 65)
    status_str = (
        "\033[32mPROMOTED TO CHAMPION\033[0m"
        if result["promoted_to_champion"]
        else "\033[33mRETAINED AS CHALLENGER\033[0m"
    )
    print(f"  Tournament Outcome  : {status_str}")
    print(f"  Decision Rationale  : {result['governance_reason']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
