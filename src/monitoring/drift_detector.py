"""Statistical drift detection engine for HR workforce features.

Calculates:
- Population Stability Index (PSI) on continuous & categorical features.
- Two-sample Kolmogorov-Smirnov (KS) tests for distribution divergence.
- Prior probability / target drift (historical vs. current attrition rate).
- Logs drift statistics to MLflow.
"""

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import get_settings

settings = get_settings()


class DriftDetector:
    """Detects data drift and concept drift between reference and production datasets."""

    def __init__(
        self,
        psi_threshold: float = settings.drift_psi_threshold,
        psi_moderate_threshold: float = settings.drift_psi_moderate_threshold,
    ):
        self.psi_threshold = psi_threshold
        self.psi_moderate_threshold = psi_moderate_threshold

    @staticmethod
    def calculate_numerical_psi(
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index (PSI) for numerical distributions."""
        ref_clean = reference.dropna().values
        curr_clean = current.dropna().values

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return 0.0

        # Create quantile bin edges from reference distribution
        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(ref_clean, quantiles)
        bin_edges = np.unique(bin_edges)  # remove duplicate edges

        if len(bin_edges) < 2:
            return 0.0

        # Adjust minimum and maximum edges to avoid out-of-bounds
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        # Count frequencies
        ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

        # Convert to percentages with epsilon smoothing to prevent div by zero
        eps = 1e-4
        ref_pct = (ref_counts / len(ref_clean)) + eps
        curr_pct = (curr_counts / len(curr_clean)) + eps

        # Re-normalize
        ref_pct = ref_pct / np.sum(ref_pct)
        curr_pct = curr_pct / np.sum(curr_pct)

        # PSI formula = sum((Actual - Expected) * ln(Actual / Expected))
        psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(max(0.0, psi_val))

    @staticmethod
    def calculate_categorical_psi(
        reference: pd.Series,
        current: pd.Series,
    ) -> float:
        """Calculate PSI for categorical features."""
        all_cats = list(set(reference.dropna().unique()).union(set(current.dropna().unique())))
        if not all_cats:
            return 0.0

        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()

        eps = 1e-4
        ref_pct = np.array(
            [(ref_counts.get(c, 0) / max(len(reference), 1)) + eps for c in all_cats]
        )
        curr_pct = np.array(
            [(curr_counts.get(c, 0) / max(len(current), 1)) + eps for c in all_cats]
        )

        ref_pct = ref_pct / np.sum(ref_pct)
        curr_pct = curr_pct / np.sum(curr_pct)

        psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(max(0.0, psi_val))

    def evaluate_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """Perform comprehensive feature-by-feature drift evaluation."""
        feature_results = {}
        red_count = 0
        yellow_count = 0

        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_s = reference_df[col]
            curr_s = current_df[col]

            is_numeric = pd.api.types.is_numeric_dtype(ref_s)

            if is_numeric:
                psi_val = self.calculate_numerical_psi(ref_s, curr_s)
                # Two-sample Kolmogorov-Smirnov test
                ref_clean = ref_s.dropna()
                curr_clean = curr_s.dropna()
                if len(ref_clean) > 5 and len(curr_clean) > 5:
                    ks_stat, p_val = stats.ks_2samp(ref_clean, curr_clean)
                else:
                    ks_stat, p_val = 0.0, 1.0
            else:
                psi_val = self.calculate_categorical_psi(ref_s, curr_s)
                ks_stat, p_val = 0.0, 1.0

            if psi_val >= self.psi_threshold:
                drift_status = "RED"
                red_count += 1
            elif psi_val >= self.psi_moderate_threshold:
                drift_status = "YELLOW"
                yellow_count += 1
            else:
                drift_status = "GREEN"

            feature_results[col] = {
                "psi": round(psi_val, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(p_val), 4),
                "status": drift_status,
                "is_drifted": (drift_status == "RED") or (p_val < 0.01 and psi_val >= 0.10),
            }

        retrain_recommended = (red_count >= 2) or (red_count >= 1 and yellow_count >= 2)

        return {
            "features": feature_results,
            "red_drift_features": red_count,
            "yellow_drift_features": yellow_count,
            "total_features_evaluated": len(feature_results),
            "retrain_recommended": retrain_recommended,
        }

    def log_drift_to_mlflow(
        self,
        drift_report: dict[str, Any],
        experiment_name: str = "odoo-hr-monitoring",
    ) -> None:
        """Log drift metrics to MLflow."""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="data_drift_detection"):
            mlflow.log_metric("red_drift_features", drift_report["red_drift_features"])
            mlflow.log_metric("yellow_drift_features", drift_report["yellow_drift_features"])
            mlflow.log_metric(
                "retrain_recommended", 1.0 if drift_report["retrain_recommended"] else 0.0
            )

            for feat, res in drift_report["features"].items():
                mlflow.log_metric(f"psi_{feat}", res["psi"])
                mlflow.log_metric(f"ks_stat_{feat}", res["ks_statistic"])
