"""Model performance monitoring engine tracking accuracy, PR-AUC, and calibration."""

from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.settings import get_settings

settings = get_settings()


class PerformanceMonitor:
    """Tracks model performance on ground truth and evaluates SLA compliance."""

    def __init__(self, prauc_threshold: float = settings.retrain_prauc_threshold):
        self.prauc_threshold = prauc_threshold

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Compute full suite of classification and calibration metrics."""
        y_true_arr = np.asarray(y_true).astype(int)
        y_prob_arr = np.asarray(y_prob).astype(float)
        y_pred = (y_prob_arr >= threshold).astype(int)

        # Handle edge cases with zero departures or zero retentions
        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) < 2:
            return {
                "sample_count": len(y_true_arr),
                "positive_count": int(np.sum(y_true_arr)),
                "negative_count": int(len(y_true_arr) - np.sum(y_true_arr)),
                "error": "Single class present in evaluation set. Insufficient variation.",
                "sla_breached": False,
            }

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_auc = float(auc(recall_curve, precision_curve))

        # ROC-AUC
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

        # Brier Score (probability calibration error, closer to 0 is better)
        brier = float(brier_score_loss(y_true_arr, y_prob_arr))

        # Precision, Recall, F1
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

        sla_breached = pr_auc < self.prauc_threshold

        return {
            "sample_count": len(y_true_arr),
            "positive_count": int(np.sum(y_true_arr)),
            "negative_count": int(len(y_true_arr) - np.sum(y_true_arr)),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "sla_breached": sla_breached,
            "threshold_used": threshold,
        }

    def log_performance_to_mlflow(
        self,
        eval_metrics: dict[str, Any],
        experiment_name: str = "odoo-hr-monitoring",
    ) -> None:
        """Log performance evaluation results to MLflow."""
        if "error" in eval_metrics:
            return

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="model_performance_evaluation"):
            mlflow.log_metrics(
                {
                    "eval_pr_auc": eval_metrics["pr_auc"],
                    "eval_roc_auc": eval_metrics["roc_auc"],
                    "eval_brier_score": eval_metrics["brier_score"],
                    "eval_f1_score": eval_metrics["f1_score"],
                    "eval_precision": eval_metrics["precision"],
                    "eval_recall": eval_metrics["recall"],
                    "sla_breached": 1.0 if eval_metrics["sla_breached"] else 0.0,
                }
            )
            mlflow.log_params(
                {
                    "eval_sample_count": eval_metrics["sample_count"],
                    "eval_positive_count": eval_metrics["positive_count"],
                    "decision_threshold": eval_metrics["threshold_used"],
                }
            )
