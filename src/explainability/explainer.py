"""Local and global explainability engine utilizing SHAP."""

from typing import Any

import numpy as np
import pandas as pd
import shap

FEATURE_READABLE_NAMES = {
    "dept_wage_diff_pct": "Salary vs. Department Average",
    "overtime_ratio": "Overtime Burden / Burnout",
    "tenure_years": "Tenure at Company",
    "age": "Employee Age",
    "distance_km": "Commute Distance",
    "leave_days_taken": "Absenteeism Frequency",
    "wage": "Monthly Base Salary",
}


class FlightRiskExplainer:
    """Computes SHAP values and translates top contributors into plain English."""

    def __init__(self, model: Any, feature_names: list[str]):
        self.feature_names = feature_names

        # Unwrap MLflow PyFunc or Sklearn wrapper if present
        if hasattr(model, "_model_impl") and hasattr(model._model_impl, "sklearn_model"):
            model = model._model_impl.sklearn_model

        self.pipeline = model
        self.scaler = getattr(model, "named_steps", {}).get("scaler", None)
        self.tree_model = getattr(model, "named_steps", {}).get("model", model)
        self.explainer = shap.TreeExplainer(self.tree_model)

    def explain_employee(self, feature_row: np.ndarray, top_k: int = 3) -> tuple[float, list[str]]:
        """Explain an individual employee's prediction with top K risk factors.

        Returns:
            Tuple of (base_expected_value, list_of_top_reasons)
        """
        row_2d = feature_row.reshape(1, -1)
        if self.scaler is not None:
            # Transform for SHAP evaluation matching the model's feature space
            df_row = pd.DataFrame(row_2d, columns=self.feature_names)
            row_for_shap = self.scaler.transform(df_row)
        else:
            row_for_shap = row_2d

        shap_values = self.explainer.shap_values(row_for_shap)

        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        top_indices = np.argsort(vals)[::-1][:top_k]

        reasons = []
        for idx in top_indices:
            feat = self.feature_names[idx]
            impact = vals[idx]
            val = feature_row[idx]  # Use raw value for readable context
            readable = FEATURE_READABLE_NAMES.get(feat, feat)

            if impact > 0.01:
                if feat == "dept_wage_diff_pct" and val < 0:
                    reasons.append(f"Compensation is {abs(val):.1%} below departmental average")
                elif feat == "overtime_ratio" and val > 0.10:
                    reasons.append(f"High overtime burden ({val:.1%} above standard hours)")
                elif feat == "distance_km" and val > 20:
                    reasons.append(f"Long commuting distance ({val:.0f} km)")
                elif feat == "tenure_years" and val > 2.5:
                    reasons.append(f"Extended tenure without advancement ({val:.1f} years)")
                elif feat == "leave_days_taken" and val > 12:
                    reasons.append(f"Elevated absenteeism ({val:.0f} days taken)")
                else:
                    reasons.append(f"{readable} increases flight risk (+{impact:.2f} SHAP)")

        if not reasons:
            reasons.append("Workforce retention indicators within standard baseline")

        return float(np.sum(vals)), reasons
