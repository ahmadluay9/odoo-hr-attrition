"""Train Attrition Flight Risk models with MLflow tracking, hyperparameter tuning, and Model Registry governance."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import auc, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.features.engineer import FeatureEngineer
from src.models.classifier import build_rf_pipeline, build_xgb_pipeline, tune_hyperparameters
from src.models.registry import ModelRegistryManager

settings = get_settings()


def main():
    parser = argparse.ArgumentParser(description="Train HR flight risk predictive models.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/synthetic/hr_attrition_train.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=settings.ml_model_type,
        choices=["xgboost", "random_forest"],
        help="Model architecture",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Stratified K-Fold hyperparameter tuning",
    )
    parser.add_argument(
        "--force-champion",
        action="store_true",
        help="Force register as MLflow champion regardless of previous score",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(
            f"[-] Training dataset not found at {args.data}. Generating synthetic dataset first..."
        )
        os.system(f"python scripts/generate_hr_data.py --samples 2500 --output {args.data}")

    df = pd.read_csv(args.data)
    print(f"[+] Loaded {len(df)} records from {args.data}")

    target_col = "will_leave" if "will_leave" in df.columns else "event_observed"
    feature_cols = [c for c in FeatureEngineer.FEATURE_COLS if c in df.columns]

    X = df[feature_cols]
    y = df[target_col].astype(int)

    pos_rate = y.mean()
    print(f"[+] Features: {feature_cols}")
    print(f"[+] Class Balance: {pos_rate:.1%} positive flight risk (y=1)")

    # 1. Strict Featurization Ordering: Train/Test split BEFORE fitting any pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=settings.random_state, stratify=y
    )

    # 2. Pipeline selection and training / tuning
    params = {}
    if args.tune:
        print(f"[+] Executing Stratified 5-Fold Hyperparameter Tuning for {args.model_type}...")
        pipeline, best_params, cv_score = tune_hyperparameters(
            model_type=args.model_type,
            X_train=X_train,
            y_train=y_train,
            n_iter=15,
            cv_folds=5,
            random_state=settings.random_state,
        )
        params = best_params
        params["cv_pr_auc"] = cv_score
        print(f"[+] Best Hyperparameters: {best_params} (CV PR-AUC: {cv_score:.4f})")
    else:
        if args.model_type == "xgboost":
            pos_count = int(np.sum(y_train == 1))
            neg_count = int(len(y_train) - pos_count)
            scale_weight = neg_count / max(pos_count, 1)
            pipeline = build_xgb_pipeline(
                scale_pos_weight=scale_weight, random_state=settings.random_state
            )
            params = {
                "scale_pos_weight": scale_weight,
                "model": "xgboost",
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.05,
            }
        else:
            pipeline = build_rf_pipeline(random_state=settings.random_state)
            params = {
                "model": "random_forest",
                "n_estimators": 300,
                "max_depth": 6,
            }
        pipeline.fit(X_train, y_train)

    # 3. Holdout Evaluation
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= settings.flight_risk_medium_threshold).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = float(auc(rec_curve, prec_curve))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    print("\n" + "=" * 60)
    print("           MODEL HOLDOUT EVALUATION METRICS")
    print("=" * 60)
    print(f"  * Model Architecture  : {args.model_type.upper()}")
    print(f"  * Tuned Search        : {args.tune}")
    print(f"  * PR-AUC (Primary)    : {pr_auc:.4f}")
    print(f"  * ROC-AUC             : {roc_auc:.4f}")
    print(f"  * Brier Score         : {brier:.4f} (lower is better)")
    print(f"  * F1-Score (med thresh: {f1:.4f}")
    print("=" * 60 + "\n")

    # 4. Log to MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    run_name = f"{args.model_type}_{'tuned' if args.tune else 'base'}_run"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", args.model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", feature_cols)
        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "f1_score": f1,
            }
        )

        # PR Curve Plot artifact
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec_curve, prec_curve, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {args.model_type.upper()}")
        ax.legend(loc="lower left")
        ax.grid(True, linestyle="--", alpha=0.6)
        pr_curve_path = f"/tmp/pr_curve_{args.model_type}.png"
        fig.tight_layout()
        fig.savefig(pr_curve_path)
        plt.close(fig)
        mlflow.log_artifact(pr_curve_path, artifact_path="evaluation_plots")

        # Log Model with signature
        signature = infer_signature(X_test, y_prob)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:3],
            serialization_format="cloudpickle",
        )

        # 5. MLflow Model Registry Champion-Challenger Governance
        registry = ModelRegistryManager()
        gov_res = registry.register_and_govern(
            model_uri=model_info.model_uri,
            metrics={"pr_auc": pr_auc, "f1_score": f1, "roc_auc": roc_auc, "brier_score": brier},
            params=params,
            force_champion=args.force_champion,
        )

    print(f"[+] MLflow Run Complete! Model Version: {gov_res['version']}")
    print(
        f"[+] Governance Status: {'PROMOTED TO CHAMPION' if gov_res['promoted'] else 'REGISTERED AS CHALLENGER'}"
    )
    print(f"[+] Details: {gov_res['reason']}")


if __name__ == "__main__":
    main()
