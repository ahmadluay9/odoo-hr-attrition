"""Cox Proportional Hazards survival analysis model using lifelines."""

import pandas as pd
from lifelines import CoxPHFitter


class FlightRiskSurvivalModel:
    """Fits Cox Proportional Hazards model to predict survival curves."""

    def __init__(self, penalizer: float = 0.1):
        self.cph = CoxPHFitter(penalizer=penalizer)

    def fit(self, df: pd.DataFrame, duration_col: str, event_col: str) -> None:
        """Fit Cox model on duration and event status."""
        self.cph.fit(df, duration_col=duration_col, event_col=event_col)

    def predict_flight_risk_window(self, df: pd.DataFrame, window_months: int = 12) -> pd.Series:
        """Calculate probability of leaving within the specified month window."""
        surv_funcs = self.cph.predict_survival_function(df)
        risks = []
        for i, row in df.iterrows():
            curr_t = row.get("tenure_months", row.get("tenure_years", 1.0) * 12.0)
            target_t = curr_t + window_months
            surv_prob = surv_funcs[i].asof(target_t) if target_t in surv_funcs[i].index else 0.5
            risks.append(round((1.0 - surv_prob) * 100, 1))
        return pd.Series(risks, index=df.index, name="survival_flight_risk_score")
