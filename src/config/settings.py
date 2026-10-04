"""Application settings validated using Pydantic Settings and Python 3.12."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Odoo Connection
    odoo_url: str = "http://localhost:8069"
    odoo_db: str = "hr_db"
    odoo_user: str = "admin@admin.com"
    odoo_password: str = ""
    odoo_api_key: str | None = None

    # PostgreSQL Connection (for direct high-speed queries)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hr_db"
    postgres_user: str = "odoo"
    postgres_password: str = ""

    # MLflow MLOps Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "odoo-hr-attrition"
    mlflow_model_name: str = "odoo-flight-risk-classifier"

    # Machine Learning Configuration
    ml_model_type: Literal["xgboost", "random_forest", "cox_ph"] = "xgboost"
    attrition_window_months: int = 12
    flight_risk_high_threshold: float = 0.70
    flight_risk_medium_threshold: float = 0.40
    shap_top_k_reasons: int = 3
    random_state: int = 42

    # MLOps Monitoring & Drift Detection Configuration
    drift_psi_threshold: float = 0.25
    drift_psi_moderate_threshold: float = 0.10
    retrain_prauc_threshold: float = 0.70
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton instance."""
    return Settings()
