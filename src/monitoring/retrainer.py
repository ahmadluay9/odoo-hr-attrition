"""Continuous retraining orchestrator managing automated retraining triggers and champion-challenger competition."""

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.features.engineer import FeatureEngineer
from src.models.classifier import build_rf_pipeline, build_xgb_pipeline, tune_hyperparameters
from src.models.registry import ModelRegistryManager
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance import PerformanceMonitor

settings = get_settings()


class RetrainingOrchestrator:
    """Coordinates drift/performance-triggered model retraining and champion-challenger promotion."""

    def __init__(self):
        self.drift_detector = DriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.registry = ModelRegistryManager()
        self.engineer = FeatureEngineer()

    def check_retrain_triggers(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        observed_y_true: np.ndarray | None = None,
        predicted_y_prob: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate both drift and performance criteria to determine if retraining is warranted."""
        cols = feature_cols or self.engineer.FEATURE_COLS
        drift_report = self.drift_detector.evaluate_feature_drift(reference_df, current_df, cols)

        perf_report = None
        sla_breached = False
        if observed_y_true is not None and predicted_y_prob is not None:
            perf_report = self.performance_monitor.evaluate_predictions(
                y_true=observed_y_true, y_prob=predicted_y_prob
            )
            sla_breached = perf_report.get("sla_breached", False)

        retrain_needed = drift_report["retrain_recommended"] or sla_breached

        reasons = []
        if drift_report["retrain_recommended"]:
            reasons.append(
                f"Feature drift threshold reached ({drift_report['red_drift_features']} red, {drift_report['yellow_drift_features']} yellow)."
            )
        if sla_breached:
            reasons.append(
                f"Model performance dropped below SLA (PR-AUC = {perf_report.get('pr_auc')} < {settings.retrain_prauc_threshold})."
            )
        if not reasons:
            reasons.append("No drift or performance degradation detected. Retraining not required.")

        return {
            "retrain_needed": retrain_needed,
            "reasons": reasons,
            "drift_report": drift_report,
            "performance_report": perf_report,
        }

    def execute_retraining(
        self,
        training_df: pd.DataFrame,
        tune: bool = False,
        model_type: str = "xgboost",
        force_champion: bool = False,
    ) -> dict[str, Any]:
        """Execute automated retraining, register challenger, and promote if superior.

        Enforces strict featurization ordering: dataset split occurs before any pipeline fitting.
        """
        feature_cols = [c for c in self.engineer.FEATURE_COLS if c in training_df.columns]
        target_col = "will_leave" if "will_leave" in training_df.columns else "event_observed"

        X = training_df[feature_cols]
        y = training_df[target_col].astype(int)

        # 1. Strict Featurization Ordering: Train/Test Split BEFORE fitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=settings.random_state, stratify=y
        )

        # 2. Model Training or Tuning
        params = {}
        if tune:
            print(f"[Retrainer] Running hyperparameter tuning for {model_type}...")
            pipeline, best_params, cv_score = tune_hyperparameters(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                n_iter=15,
                cv_folds=5,
                random_state=settings.random_state,
            )
            params = best_params
            params["cv_pr_auc"] = cv_score
        else:
            if model_type.lower() == "xgboost":
                pos_count = int(np.sum(y_train == 1))
                neg_count = int(len(y_train) - pos_count)
                scale_weight = neg_count / max(pos_count, 1)
                pipeline = build_xgb_pipeline(
                    scale_pos_weight=scale_weight, random_state=settings.random_state
                )
                params = {"scale_pos_weight": scale_weight, "model": "xgboost"}
            else:
                pipeline = build_rf_pipeline(random_state=settings.random_state)
                params = {"model": "random_forest"}

            pipeline.fit(X_train, y_train)

        # 3. Evaluate Challenger on Holdout Set
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = self.performance_monitor.evaluate_predictions(y_true=y_test.values, y_prob=y_prob)

        # 4. Log Challenger Run to MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        run_name = f"retrain_{model_type}_{'tuned' if tune else 'standard'}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features", feature_cols)
            mlflow.log_metrics(
                {
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "f1_score": metrics["f1_score"],
                    "brier_score": metrics["brier_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )

            signature = infer_signature(X_test, y_prob)
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_test.iloc[:3],
                serialization_format="cloudpickle",
            )

            # 5. Champion-Challenger Promotion via Model Registry
            gov_res = self.registry.register_and_govern(
                model_uri=model_info.model_uri,
                metrics={
                    "pr_auc": metrics["pr_auc"],
                    "f1_score": metrics["f1_score"],
                    "roc_auc": metrics["roc_auc"],
                },
                params=params,
                force_champion=force_champion,
            )

        return {
            "status": "success",
            "run_id": run.info.run_id,
            "version": gov_res["version"],
            "promoted_to_champion": gov_res["promoted"],
            "governance_reason": gov_res["reason"],
            "metrics": metrics,
            "tuned": tune,
        }
