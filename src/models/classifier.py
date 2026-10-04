"""XGBoost and Random Forest binary flight risk classifiers with hyperparameter tuning."""

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_xgb_pipeline(
    scale_pos_weight: float = 5.0,
    n_estimators: int = 250,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """Build standardized XGBoost classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def build_rf_pipeline(
    n_estimators: int = 300,
    max_depth: int = 6,
    min_samples_split: int = 8,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> Pipeline:
    """Build standardized Random Forest classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def tune_hyperparameters(
    model_type: str,
    X_train,
    y_train,
    n_iter: int = 15,
    cv_folds: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], float]:
    """Perform Stratified K-Fold hyperparameter tuning using RandomizedSearchCV.

    Optimizes for PR-AUC ('average_precision') by default to handle class imbalance.
    Enforces strict featurization ordering (StandardScaler is fit inside CV folds).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if model_type.lower() == "xgboost":
        # Calculate positive class imbalance weight
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(len(y_train) - pos_count)
        base_weight = neg_count / max(pos_count, 1)

        base_pipe = build_xgb_pipeline(scale_pos_weight=base_weight, random_state=random_state)
        param_dist = {
            "model__n_estimators": [100, 150, 200, 300],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "model__scale_pos_weight": [base_weight * 0.8, base_weight, base_weight * 1.2],
        }
    elif model_type.lower() == "random_forest":
        base_pipe = build_rf_pipeline(random_state=random_state)
        param_dist = {
            "model__n_estimators": [150, 200, 300, 400],
            "model__max_depth": [4, 6, 8, 10, None],
            "model__min_samples_split": [4, 6, 8, 12],
            "model__min_samples_leaf": [2, 3, 5, 8],
            "model__max_features": ["sqrt", "log2", 0.7, 0.8],
        }
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'xgboost' or 'random_forest'.")

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_score = float(search.best_score_)

    return best_pipe, best_params, best_score
