# GEMINI.md

This document serves as the single source of truth for Antigravity agents operating on this repository. It defines technical standards, architectural boundaries, Odoo 20 data models, machine learning feature engineering, predictive flight risk modeling, explainable AI (SHAP), custom Odoo module development, XML-RPC bi-directional integration, and ethical AI operational guardrails for the Employee Attrition Prediction system.

---

## 1. Antigravity Project Persona & Mission

You are an expert **HR Predictive Analytics & MLOps Engineer** specializing in employee retention intelligence, survival analysis, explainable machine learning (XAI), and enterprise ERP integration with Odoo 20.

### Primary Objectives

* **Proactive Flight Risk Identification:** Predict with high precision and recall which employees are at risk of leaving the company within the next 6–12 months, providing HR managers actionable lead time for retention interventions.
* **Explainable & Actionable Insights (SHAP):** Every prediction must be accompanied by local feature attribution (SHAP values) translated into transparent, plain-English drivers (e.g., *"High risk due to 3.2 years without a promotion and compensation 18% below departmental average"*).
* **Closed-Loop Odoo 20 ERP Integration:** Seamlessly extract historical workforce data from Odoo tables (`hr.employee`, `hr.contract`, `hr.leave`, `hr.attendance`, `hr.appraisal`), train/score predictive models, and push flight risk scores back into Odoo via the XML-RPC API.
* **Strict Role-Based Access Control (RBAC):** Restrict all flight risk scores, risk levels, and attrition explanations strictly to HR Managers (`hr.group_hr_manager`). Flight risk data must never be visible to regular employees, department officers, or unauthorized users.
* **Rigorous ML Best Practices & Zero Data Leakage:** Enforce strict featurization ordering: split datasets into training, validation, and test splits *before* fitting any imputer, scaler, or categorical encoder. Prevent temporal and group leakage across employee tenures.
* **Dual Modeling Strategy:** Support both binary classification (XGBoost / Random Forest) for discrete 6–12 month attrition probability and Survival Analysis (Cox Proportional Hazards) for time-to-event estimation (expected months to exit).
* **MLflow Experiment Tracking & Model Governance:** Track every training run, hyperparameter set, and evaluation metric (PR-AUC, F1, LogLoss) in MLflow; manage model promotion through the MLflow Model Registry (`Staging` -> `Production`) so inference workers always score employees using the approved champion model.
* **Modern Python 3.12 & Astral uv Toolchain:** Standardize runtime on Python 3.12 managed via Astral uv (`uv python pin 3.12`, `pyproject.toml`) alongside Docker Compose orchestration.

---

## 2. Architecture & Tech Stack

| Layer | Technology | Primary Location | Purpose |
| :--- | :--- | :--- | :--- |
| **ERP Application Core** | Odoo 20 (Community) | Docker container (`web`) | Primary HR operational system managing employees, contracts, attendance, leaves, and appraisals. |
| **Transactional Database** | PostgreSQL 18 | Docker container (`db`) | Relational persistence for all Odoo models (`hr_db`). |
| **Database Administration**| pgAdmin 4 | Docker container (`pgadmin`)| Web-based SQL exploration and database management interface on port 5050. |
| **MLOps & Tracking** | MLflow | Docker container (`mlflow`)| Experiment tracking, hyperparameter logging, model artifact storage, and Model Registry on port 5000. |
| **Runtime & Dependency Mgr**| Python 3.12 (`uv`) | Local / ML Worker | High-performance Python runtime and deterministic dependency management. |
| **ETL & Data Extraction** | Python `xmlrpc.client` / `psycopg2` | `src/data/` | Extracts raw tables from Odoo via XML-RPC or direct read-only SQL queries. |
| **Feature Engineering** | `pandas`, `numpy`, `scipy` | `src/features/` | Compa-ratio, overtime burden, promotion velocity, absenteeism rate, manager churn. |
| **Classification Models** | `xgboost`, `scikit-learn` | `src/models/` | XGBoost Classifier and Random Forest for 6–12 month flight risk probability. |
| **Survival Analysis** | `lifelines` (Cox PH) | `src/models/` | Cox Proportional Hazards for time-to-event modeling (predicting *when* exit occurs). |
| **Explainable AI (XAI)** | `shap` (SHapley Additive exPlanations) | `src/explainability/` | Computes global and local feature contributions; generates natural-language risk reasons. |
| **Interactive UI & Simulator** | Streamlit, Plotly | `app.py` | Web dashboard for What-If flight risk simulation, Odoo workforce exploration, batch analytics, and drift monitoring on port 8501. |
| **Prediction CLI** | Python CLI | `scripts/predict.py` | Multi-mode inference CLI (custom features, live Odoo ID, dataset batch, archetypes). |
| **Odoo Custom Addon** | Python / XML (`hr_flight_risk`) | `addons/hr_flight_risk` | Extends `hr.employee` form view with flight risk fields, badges, and security groups. |
| **API Sync Worker** | Python XML-RPC / JSON-2 Client | `src/integration/` | Automated batch service pushing risk scores (0–100%) and reasons to `hr.employee`. |
| **Synthetic Data Generator**| `scripts/generate_hr_data.py` | Host CLI / scripts | Generates statistically sound synthetic HR history aligned with Odoo schema for training. |

---

## 3. Directory Structure

```text
odoo-hr-attrition/
├── addons/
│   └── hr_flight_risk/               # Custom Odoo 20 module for Flight Risk display
│       ├── __init__.py
│       ├── __manifest__.py           # Module manifest (depends: ['hr'])
│       ├── models/
│       │   ├── __init__.py
│       │   └── hr_employee.py        # Model extension (flight_risk_score, etc.)
│       ├── security/
│       │   └── ir.model.access.csv   # RBAC: HR Manager exclusive access
│       └── views/
│           └── hr_employee_views.xml # Form view inheritance & risk badges
├── config/
│   ├── odoo.conf.example             # Git-tracked Odoo configuration template
│   └── odoo.conf                     # Active Odoo config (gitignored)
├── data/
│   ├── raw/                          # Extracted raw data snapshots (gitignored)
│   ├── processed/                    # Feature-engineered training/inference datasets
│   └── synthetic/                    # Generated synthetic training corpus (baseline & drifted)
├── docker-compose.yml                # Multi-container orchestration (Odoo, Postgres, pgAdmin)
├── .env.example                      # Environment variables template
├── .env                              # Active environment credentials (gitignored)
├── .gitignore                        # Git exclusion rules
├── Makefile                          # Unified developer automation commands
├── pyproject.toml                    # Python project dependencies managed via uv
├── app.py                            # Streamlit Web UI for Prediction & Retention Intelligence
├── GEMINI.md                         # Project source of truth & technical standards
├── README.md                         # Project overview and setup instructions
├── scripts/
│   ├── generate_api_key.py           # Programmatic Odoo API key generation via container shell
│   ├── generate_hr_data.py           # Synthetic HR attrition dataset generator
│   ├── ingest_to_odoo.py             # Ingests synthetic employees & attendance into Odoo JSON-2
│   ├── get_odoo_stats.py             # Quick workforce audit & stats extractor
│   ├── train_model.py                # Model training, 5-fold CV, and hyperparameter tuning
│   ├── simulate_drift.py             # Workforce drift simulator (burnout, rto, compound)
│   ├── detect_drift.py               # Feature drift evaluation (PSI & Kolmogorov-Smirnov)
│   ├── monitor_performance.py        # Model performance monitoring against SLA targets
│   ├── retrain_pipeline.py           # Automated retraining & champion-challenger tournament
│   ├── predict.py                    # Multi-mode inference CLI (features, Odoo live, CSV, archetypes)
│   └── sync_flight_risk.py           # XML-RPC batch inference & Odoo sync script
└── src/
    ├── __init__.py
    ├── config/
    │   └── settings.py               # Pydantic Settings strongly typed configuration
    ├── data/
    │   ├── __init__.py
    │   └── odoo_extractor.py         # Odoo XML-RPC / Postgres data extraction client
    ├── features/
    │   ├── __init__.py
    │   └── engineer.py               # Feature transformation & compa-ratio calculations
    ├── models/
    │   ├── __init__.py
    │   ├── classifier.py             # XGBoost & Random Forest flight risk classifier
    │   ├── survival.py               # Cox Proportional Hazards survival analysis model
    │   └── registry.py               # MLflow Champion/Challenger model registry manager
    ├── monitoring/
    │   ├── __init__.py
    │   ├── drift_detector.py         # Population Stability Index (PSI) & KS test engine
    │   └── evaluator.py              # Performance SLA & calibration evaluator
    ├── explainability/
    │   ├── __init__.py
    │   └── explainer.py              # SHAP TreeExplainer & natural language generator
    └── integration/
        ├── __init__.py
        └── odoo_client.py            # XML-RPC client for pushing scores to Odoo
```

---

## 4. Environment Configuration & .env Lifecycle

Application parameters, database credentials, Odoo XML-RPC authentication, and ML hyperparameters derive strictly from environment variables validated via `pydantic-settings`.

### 4.1 `.env.example` Template

```env
# =====================================================================
# Odoo HR Attrition & Flight Risk Intelligence - Environment Config
# =====================================================================

# Environment & Runtime
ENVIRONMENT=development
LOG_LEVEL=INFO

# Odoo ERP Service Connection
ODOO_URL=http://localhost:8069
ODOO_HOST=localhost
ODOO_PORT=8069
ODOO_DB=hr_db
ODOO_ADMIN_PASSWORD=your_admin_master_pwd
ODOO_USER=admin
ODOO_PASSWORD=your_odoo_user_password

# PostgreSQL Database Service
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hr_db
POSTGRES_USER=odoo
POSTGRES_PASSWORD=your_postgres_password

# pgAdmin 4 Administration UI
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=your_pgadmin_password

# MLOps & Experiment Tracking (MLflow)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=odoo-hr-attrition
MLFLOW_MODEL_NAME=odoo-flight-risk-classifier

# Machine Learning & Attrition Pipeline
ML_MODEL_TYPE=xgboost                    # 'xgboost', 'random_forest', or 'cox_ph'
ATTRITION_WINDOW_MONTHS=12               # Prediction horizon (6 or 12 months)
FLIGHT_RISK_HIGH_THRESHOLD=0.70          # Score >= 70% flagged as HIGH RISK
FLIGHT_RISK_MEDIUM_THRESHOLD=0.40        # Score 40-69% flagged as MEDIUM RISK
SHAP_TOP_K_REASONS=3                     # Top N explanatory factors to store in Odoo
RANDOM_STATE=42
```

### 4.2 Canonical `pydantic-settings` Implementation (`src/config/settings.py`)

```python
"""Application settings validated using Pydantic Settings and Python 3.12."""

from functools import lru_cache
from typing import Literal
from pydantic import Field
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
    odoo_user: str = "admin"
    odoo_password: str = Field(..., description="Odoo API/User password")

    # PostgreSQL Connection (for direct high-speed queries)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hr_db"
    postgres_user: str = "odoo"
    postgres_password: str = Field(..., description="PostgreSQL password")

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


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton instance."""
    return Settings()
```

---

## 5. Odoo Data Sources & Extraction Architecture

The pipeline gathers signals from five core Odoo operational tables. In Community Edition, where `hr_appraisal` is an Enterprise module, the system supports both standard Odoo Survey assessments (`survey.user_input`) and synthetic evaluation metrics as a fallback.

### 5.1 Odoo Model Field Mapping

| Odoo Model | Technical Fields Extracted | Extracted Feature Signal |
| :--- | :--- | :--- |
| **`hr.employee`** | `id`, `name`, `department_id`, `parent_id` (manager), `job_id`, `birthday`, `km_home_work` (distance), `create_date` | Age, tenure, department, manager identity, commuting distance. |
| **`hr.contract`** | `employee_id`, `wage`, `date_start`, `date_end`, `state`, `structure_type_id`, `schedule_pay` | Current wage, historical wage trajectory, salary growth rate, contract stability. |
| **`hr.leave`** | `employee_id`, `date_from`, `date_to`, `number_of_days`, `holiday_status_id`, `state`, `create_date` | Absenteeism frequency, unplanned/short-notice leave ratio, sick leave velocity. |
| **`hr.attendance`** | `employee_id`, `check_in`, `check_out`, `worked_hours` | Actual weekly hours, overtime burden, irregular work hours, burnout index. |
| **`hr.appraisal`** *(or `survey`)* | `employee_id`, `rating` / `score`, `date_close` | Recent performance evaluations, rating drops, manager sentiment. |

### 5.2 Odoo XML-RPC Data Extraction Client (`src/data/odoo_extractor.py`)

```python
"""Odoo XML-RPC data extractor client for workforce tables."""

import xmlrpc.client
import pandas as pd
from src.config.settings import get_settings

settings = get_settings()


class OdooExtractor:
    """Extracts raw employee, contract, leave, and attendance records from Odoo."""

    def __init__(self):
        self.url = settings.odoo_url
        self.db = settings.odoo_db
        self.username = settings.odoo_user
        self.password = settings.odoo_password
        self.common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common")
        self.models = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object")
        self.uid = self.common.authenticate(self.db, self.username, self.password, {})
        if not self.uid:
            raise ConnectionError("Failed to authenticate with Odoo via XML-RPC.")

    def fetch_employees(self) -> pd.DataFrame:
        """Fetch active employee baseline demographics."""
        fields = [
            "id",
            "name",
            "department_id",
            "parent_id",
            "job_id",
            "birthday",
            "km_home_work",
            "create_date",
            "active",
        ]
        records = self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            "hr.employee",
            "search_read",
            [[["active", "in", [True, False]]]],
            {"fields": fields},
        )
        return pd.DataFrame(records)

    def fetch_contracts(self) -> pd.DataFrame:
        """Fetch historical and active employment contracts."""
        fields = ["id", "employee_id", "wage", "date_start", "date_end", "state"]
        records = self.models.execute_kw(
            self.db, self.uid, self.password, "hr.contract", "search_read", [[]], {"fields": fields}
        )
        return pd.DataFrame(records)

    def fetch_leaves(self) -> pd.DataFrame:
        """Fetch leave history to calculate absenteeism patterns."""
        fields = [
            "id",
            "employee_id",
            "date_from",
            "date_to",
            "number_of_days",
            "holiday_status_id",
            "state",
            "create_date",
        ]
        records = self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            "hr.leave",
            "search_read",
            [[["state", "=", "validate"]]],
            {"fields": fields},
        )
        return pd.DataFrame(records)

    def fetch_attendance(self) -> pd.DataFrame:
        """Fetch attendance punches to compute overtime burden."""
        fields = ["id", "employee_id", "check_in", "check_out", "worked_hours"]
        records = self.models.execute_kw(
            self.db,
            self.uid,
            self.password,
            "hr.attendance",
            "search_read",
            [[]],
            {"fields": fields},
        )
        return pd.DataFrame(records)
```

### 5.3 Modern External JSON-2 API & XML-RPC Deprecation Migration

> [!WARNING]
> **XML-RPC & JSON-RPC Deprecation Notice (Odoo 19 & 20):**
> Endpoints `/xmlrpc`, `/xmlrpc/2`, and `/jsonrpc` are officially deprecated in Odoo 19 and scheduled for complete removal in Odoo 22. In Odoo 20, unmuted XML-RPC calls emit:
> `WARNING hr_db odoo.addons.rpc.controllers.xmlrpc: The /xmlrpc, /xmlrpc/2 and /jsonrpc endpoints are deprecated in Odoo 19 and scheduled for removal in Odoo 22.`
> See official docs: [Migrating from XML-RPC / JSON-RPC to External JSON-2 API](https://www.odoo.com/documentation/latest/developer/reference/external_api.html#migrating-from-xml-rpc-json-rpc).

#### 1. Muting XML-RPC Deprecation Warnings
To suppress this warning during transition, configure the logger level to `ERROR` for the XML-RPC controller:
* **In `config/odoo.conf`**:
  ```ini
  log_handler = :INFO,odoo.addons.rpc.controllers.xmlrpc:ERROR
  ```
* **In CLI / Docker Compose (`command:`)**:
  ```bash
  odoo --log-handler odoo.addons.rpc.controllers.xmlrpc:ERROR
  ```

#### 2. Migrating to Odoo External JSON-2 API (`/json/2/`)
Odoo 19 & 20 introduce the **External JSON-2 API** which replaces the legacy XML-RPC object service:
* **Endpoint Pattern:** `POST {ODOO_URL}/json/2/{model}/{method}`
* **Authentication:** API Key passed via HTTP Header `Authorization: bearer <API_KEY>` (no passwords or user IDs transmitted in requests).
* **Database Header:** `X-Odoo-Database: {ODOO_DB}`
* **Payload Structure:** Clean JSON with named kwargs (`domain`, `fields`, `ids`, `values`, `context`).

#### Python Modern JSON-2 Extractor / Client:
```python
import requests
import pandas as pd
from src.config.settings import get_settings

settings = get_settings()


class OdooJson2Client:
    """Modern External JSON-2 client for Odoo 20."""

    def __init__(self):
        self.base_url = settings.odoo_url.rstrip("/")
        self.headers = {
            "Authorization": f"bearer {settings.odoo_api_key}",
            "X-Odoo-Database": settings.odoo_db,
            "Content-Type": "application/json",
        }

    def call(self, model: str, method: str, **kwargs):
        url = f"{self.base_url}/json/2/{model}/{method}"
        resp = requests.post(url, headers=self.headers, json=kwargs)
        resp.raise_for_status()
        return resp.json()

    def fetch_employees(self) -> pd.DataFrame:
        """Fetch active employee records via modern JSON-2 API."""
        fields = ["id", "name", "job_title", "wage", "km_home_work", "department_id", "active"]
        data = self.call(
            "hr.employee", "search_read", domain=[["active", "in", [True, False]]], fields=fields
        )
        return pd.DataFrame(data)
```

#### 3. Automated API Key Generation via Makefile
To provision an Odoo API key programmatically without manually configuring it through the web UI:
```bash
make generate-api-key
```
This triggers `scripts/generate_api_key.py`, which invokes `res.users.apikeys.with_user(user)._generate()` inside the Odoo container, extracts the provisioned key, and automatically persists it to `.env` as `ODOO_API_KEY`.

---

## 6. Feature Engineering Pipeline

Feature engineering transforms raw transactional timestamps and logs into high-signal attrition predictors.

```
+--------------------+   +---------------------+   +---------------------+
|    hr.employee     |   |     hr.contract     |   |    hr.attendance    |
| (Tenure, Age, Dist)|   | (Wage, Compa-Ratio) |   |  (Overtime Burden)  |
+---------+----------+   +----------+----------+   +----------+----------+
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
                                    v
                     +-----------------------------+
                     | Engineered Feature Vector   |
                     |  - compa_ratio_dept_diff    |
                     |  - overtime_to_std_ratio    |
                     |  - tenure_years             |
                     |  - time_since_promotion     |
                     |  - manager_change_count     |
                     |  - short_notice_leave_rate  |
                     +--------------+--------------+
                                    |
                                    v
                     +-----------------------------+
                     | ML Classifier / Survival    |
                     | (XGBoost / Cox PH) + SHAP   |
                     +-----------------------------+
```

### 6.1 Engineered Feature Definitions

1. **Departmental Compa-Ratio Deviation (`compa_ratio_diff`):**
   $$\text{Compa-Ratio Diff} = \frac{\text{Employee Wage} - \overline{\text{Department Wage}}}{\overline{\text{Department Wage}}}$$
   Measures relative compensation satisfaction against peers in the same department. Negative values indicate underpaid risk.
2. **Overtime Burden Ratio (`overtime_ratio`):**
   $$\text{Overtime Ratio} = \frac{\text{Actual Worked Hours} - \text{Standard Contract Hours}}{\text{Standard Contract Hours}}$$
   Tracks sustained burnout risk where actual hours consistently exceed contractual 40 hours/week.
3. **Time Since Last Promotion / Role Change (`years_since_promotion`):**
   Calculated from job change history or contract updates. Stagnation beyond 2.5–3 years strongly correlates with voluntary exit.
4. **Manager Turnover Index (`manager_change_count`):**
   Number of times the employee's `parent_id` changed over the past 24 months. Frequent manager churn increases flight risk.
5. **Absenteeism Velocity & Short-Notice Leaves (`short_notice_leave_rate`):**
   Ratio of leaves submitted less than 48 hours before start date versus planned annual leaves. Often signals interview activity.
6. **Recent Performance Drop (`appraisal_score_delta`):**
   Change in evaluation score between the two most recent performance reviews.

### 6.2 Feature Engineering Module (`src/features/engineer.py`)

```python
"""Feature engineering pipeline for Odoo workforce tables."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Transforms raw Odoo DataFrames into tabular ML features."""

    def build_features(
        self,
        employees_df: pd.DataFrame,
        contracts_df: pd.DataFrame,
        leaves_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Execute end-to-end feature aggregation."""
        ref_date = reference_date or pd.Timestamp.now()

        # 1. Base Employee Features
        df = employees_df.copy()
        df["tenure_years"] = (ref_date - pd.to_datetime(df["create_date"])).dt.days / 365.25
        df["age"] = (ref_date - pd.to_datetime(df["birthday"])).dt.days / 365.25
        df["distance_km"] = df["km_home_work"].fillna(df["km_home_work"].median())

        # Extract Department ID
        df["department_id_clean"] = df["department_id"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) else (x if pd.notna(x) else 0)
        )

        # 2. Wage & Department Compa-Ratio
        latest_contracts = (
            contracts_df.sort_values("date_start").groupby("employee_id").last().reset_index()
        )
        # Clean employee_id if returned as [id, name]
        latest_contracts["emp_id"] = latest_contracts["employee_id"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) else x
        )
        df = df.merge(
            latest_contracts[["emp_id", "wage"]], left_on="id", right_on="emp_id", how="left"
        )
        df["wage"] = df["wage"].fillna(df["wage"].median())

        # Department Average Wage
        dept_avg_wage = df.groupby("department_id_clean")["wage"].transform("mean")
        df["dept_wage_diff_pct"] = (df["wage"] - dept_avg_wage) / (dept_avg_wage + 1e-6)

        # 3. Attendance & Overtime Burden (past 90 days)
        if not attendance_df.empty:
            att = attendance_df.copy()
            att["emp_id"] = att["employee_id"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else x
            )
            att_summary = att.groupby("emp_id")["worked_hours"].agg(["count", "sum"]).reset_index()
            # Standard hours based on 8 hrs/day
            att_summary["std_hours"] = att_summary["count"] * 8.0
            att_summary["overtime_ratio"] = (att_summary["sum"] - att_summary["std_hours"]) / (
                att_summary["std_hours"] + 1e-6
            )
            df = df.merge(
                att_summary[["emp_id", "overtime_ratio"]],
                left_on="id",
                right_on="emp_id",
                how="left",
            )
        else:
            df["overtime_ratio"] = 0.0
        df["overtime_ratio"] = df["overtime_ratio"].fillna(0.0).clip(lower=-0.5, upper=1.5)

        # 4. Absenteeism & Leave Velocity (past 180 days)
        if not leaves_df.empty:
            lv = leaves_df.copy()
            lv["emp_id"] = lv["employee_id"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else x
            )
            leave_summary = lv.groupby("emp_id")["number_of_days"].sum().reset_index()
            leave_summary.rename(columns={"number_of_days": "leave_days_taken"}, inplace=True)
            df = df.merge(leave_summary, left_on="id", right_on="emp_id", how="left")
        else:
            df["leave_days_taken"] = 0.0
        df["leave_days_taken"] = df["leave_days_taken"].fillna(0.0)

        # Select final model feature matrix
        feature_cols = [
            "id",
            "name",
            "tenure_years",
            "age",
            "distance_km",
            "wage",
            "dept_wage_diff_pct",
            "overtime_ratio",
            "leave_days_taken",
        ]
        return df[feature_cols]
```

---

## 7. Machine Learning Modeling & Survival Analysis

### 7.1 Binary Classification: XGBoost & Random Forest

* **Target Variable ($y$):** Binary indicator (`1` = Employee left voluntarily within 6–12 months, `0` = Employee retained).
* **Class Imbalance Handling:** Employee attrition typically exhibits severe imbalance (10–18% positive class). Implement:
  1. `scale_pos_weight = (count(negative) / count(positive))` in `XGBClassifier`.
  2. Balanced class weighting (`class_weight='balanced'`) in `RandomForestClassifier`.
  3. Stratified $K$-Fold cross-validation ($k=5$).
* **Primary Evaluation Metrics:** **PR-AUC (Precision-Recall AUC)** and **Recall@Top-Decile**, aligning with the HR operational requirement to avoid missing true flight risks without overwhelming HR with false alarms.

```python
"""XGBoost and Random Forest binary flight risk classifiers."""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_xgb_pipeline(scale_pos_weight: float = 5.0) -> Pipeline:
    """Build standardized XGBoost classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def build_rf_pipeline() -> Pipeline:
    """Build standardized Random Forest classification pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_split=8,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
```

### 7.2 Alternative Approach: Survival Analysis (Cox Proportional Hazards)

For granular time-to-event predictions estimating *when* an employee will exit over continuous time:

* **Event Indicator ($E$):** `1` if observed attrition occurred; `0` if censored (still employed).
* **Duration ($T$):** Tenure duration in months until attrition or censoring date.
* **Hazard Function:**
  $$h(t | X) = h_0(t) \exp\left(\sum_{j=1}^p \beta_j X_j\right)$$
* **Advantage:** Accurately handles censored data (current employees who have not left yet) without discarding them or introducing bias.

```python
"""Cox Proportional Hazards survival analysis model using lifelines."""

import pandas as pd
from lifelines import CoxPHFitter


class FlightRiskSurvivalModel:
    """Fits Cox Proportional Hazards model to predict survival curves."""

    def __init__(self):
        self.cph = CoxPHFitter(penalizer=0.1)

    def fit(self, df: pd.DataFrame, duration_col: str, event_col: str) -> None:
        """Fit Cox model on duration and event status."""
        self.cph.fit(df, duration_col=duration_col, event_col=event_col)

    def predict_flight_risk_window(self, df: pd.DataFrame, window_months: int = 12) -> pd.Series:
        """Calculate probability of leaving within the specified month window."""
        # Risk = 1 - Survival Probability at t = current_tenure + window_months
        surv_funcs = self.cph.predict_survival_function(df)
        risks = []
        for i, row in df.iterrows():
            curr_t = row["tenure_months"]
            target_t = curr_t + window_months
            surv_prob = surv_funcs[i].asof(target_t) if target_t in surv_funcs[i].index else 0.5
            risks.append(round((1.0 - surv_prob) * 100, 1))
        return pd.Series(risks, index=df.index, name="survival_flight_risk_score")
```

### 7.3 MLOps Workflow: MLflow Experiment Tracking & Model Registry

All model training runs, hyperparameters, evaluation metrics, and artifacts are tracked via **MLflow**. The best-performing model is registered in the **MLflow Model Registry** for controlled deployment to production.

```python
"""MLflow experiment tracking and model registration workflow."""

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
from src.config.settings import get_settings

settings = get_settings()


def log_model_run_to_mlflow(
    pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    model_family: str,
    params: dict,
    register_as_champion: bool = False,
):
    """Log hyperparams, PR-AUC, artifacts, and optionally register model in MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{model_family}_attrition_{settings.attrition_window_months}m"):
        # 1. Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", model_family)
        mlflow.log_param("window_months", settings.attrition_window_months)
        mlflow.log_param("train_samples", len(X_train))

        # 2. Fit and Evaluate
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= settings.flight_risk_medium_threshold).astype(int)

        precision, recall, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)

        # 3. Log Metrics
        mlflow.log_metrics(
            {
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "f1_score": float(f1),
            }
        )

        # 4. Log Precision-Recall Curve Plot as Artifact
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve - {model_family}")
        ax.legend()
        pr_curve_path = f"/tmp/pr_curve_{model_family}.png"
        fig.savefig(pr_curve_path)
        plt.close(fig)
        mlflow.log_artifact(pr_curve_path, artifact_path="evaluation_plots")

        # 5. Log Model with Signature
        signature = infer_signature(X_val, y_prob)
        input_example = X_val.iloc[:3] if hasattr(X_val, "iloc") else X_val[:3]
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # 6. Model Registry Governance
        if register_as_champion:
            model_uri = model_info.model_uri
            registered_model = mlflow.register_model(model_uri, settings.mlflow_model_name)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                name=settings.mlflow_model_name,
                alias="champion",
                version=registered_model.version,
            )
            print(
                f"[MLflow] Model version {registered_model.version} promoted to 'champion' alias."
            )

        return model_info
```

### 7.4 MLOps Lifecycle: Versioning, Drift Detection, Performance Monitoring & Retraining

To prevent model degradation over time, the system implements a production MLOps closed loop:

```
+--------------------------------------------------------------------------+
|                          MLOps Continuous Loop                           |
|                                                                          |
|   +-------------------+         +--------------------+                   |
|   | Reference Data    |         | Live Workforce     |                   |
|   | (Baseline Corpus) |         | (Odoo Production)  |                   |
|   +---------+---------+         +---------+----------+                   |
|             |                             |                              |
|             +--------------+--------------+                              |
|                            |                                             |
|                            v                                             |
|             +-------------------------------+                            |
|             | Statistical Drift Detector    |                            |
|             |  - Population Stability (PSI) |                            |
|             |  - 2-Sample KS Tests (p-val)  |                            |
|             +--------------+----------------+                            |
|                            |                                             |
|        [PSI >= 0.25 on >= 2 Features] OR [SLA Breach (PR-AUC < 0.70)]   |
|                            |                                             |
|                            v                                             |
|             +-------------------------------+                            |
|             | Automated Retraining Pipeline |                            |
|             |  - Stratified 5-Fold Tuning   |                            |
|             |  - Strict Featurization Split |                            |
|             +--------------+----------------+                            |
|                            |                                             |
|                            v                                             |
|             +-------------------------------+                            |
|             | MLflow Model Registry Tourney |                            |
|             |  - Challenger vs Champion     |                            |
|             |  - Promote if PR-AUC exceeds  |                            |
|             +--------------+----------------+                            |
|                            |                                             |
|                            v                                             |
|             +-------------------------------+                            |
|             | Push Updated Scores to Odoo   |                            |
|             +-------------------------------+                            |
+--------------------------------------------------------------------------+
```

#### 1. Model Versioning & Registry Governance (`src/models/registry.py`)
* **Tracking Server:** MLflow 3.16.1 running on port 5000.
* **Registered Model:** `odoo-flight-risk-classifier`.
* **Aliases & Tags:**
  * `@champion`: The verified production model used for all inference scoring.
  * `@challenger`: Candidate models trained on recent data competing against the champion.
* **Promotion Rule:** When a challenger is evaluated against the test set, it is automatically promoted to `@champion` if its PR-AUC exceeds the current champion's PR-AUC. Otherwise, it remains tagged as `@challenger` without displacing production.

#### 2. Stratified Hyperparameter Tuning (`src/models/classifier.py`)
* **Method:** `RandomizedSearchCV` wrapped in `tune_hyperparameters()`.
* **Cross-Validation:** 5-Fold `StratifiedKFold` with strict featurization ordering (preprocessors fit inside CV folds to prevent data leakage).
* **Objective Metric:** `average_precision` (PR-AUC) to explicitly handle workforce class imbalance (10–20% positive departure rates).
* **CLI Trigger:** `make tune` or `uv run python scripts/train_model.py --tune`.

#### 3. Statistical Drift Detection (`src/monitoring/drift_detector.py`)
* **Population Stability Index (PSI):** Measures distribution drift between baseline training data and live inference data:
  * **$PSI < 0.10$ (GREEN):** Stable distribution, no action required.
  * **$0.10 \le PSI < 0.25$ (YELLOW):** Moderate drift, watch list.
  * **$PSI \ge 0.25$ (RED):** Significant distribution shift.
* **Two-Sample Kolmogorov-Smirnov (KS) Test:** Evaluates non-parametric shape divergence for continuous features (`distance_km`, `wage`, `overtime_ratio`, `tenure_years`, `leave_days_taken`).
* **CLI Command:** `make detect-drift` or `uv run python scripts/detect_drift.py`.

#### 4. Ground Truth Performance Monitoring (`src/monitoring/performance.py`)
* **Delayed Ground Truth:** When an employee departs in Odoo (`active = False`), the feedback loop records an observed exit ($y = 1$).
* **Metrics Computed:** PR-AUC, ROC-AUC, Brier score (probability calibration), F1-Score, Recall, Precision, and Confusion Matrix.
* **SLA Threshold:** Flagged as `BREACHED` if PR-AUC falls below 0.70.
* **CLI Command:** `make monitor-perf` or `uv run python scripts/monitor_performance.py`.

#### 5. Automated Closed-Loop Retraining (`src/monitoring/retrainer.py`)
* **Retraining Trigger Criteria:**
  1. Feature drift alarm ($\ge 2$ features with $PSI \ge 0.25$).
  2. Performance degradation below SLA ($PR\text{-}AUC < 0.70$).
  3. Forced retrain manual trigger (`--force`).
* **Automated Tournament:** Automatically trains candidate models, logs runs to MLflow, and executes the Model Registry champion-challenger promotion tournament.
* **CLI Commands:**
  * `make retrain`: Evaluates triggers and retrains if drift or degradation is detected.
  * `make retrain-force`: Forces challenger retraining with hyperparameter tuning and promotes to champion.

#### 6. Workforce Drift Simulation Engine (`scripts/simulate_drift.py`)
To stress-test MLOps drift detectors, alarms, and retraining triggers under realistic workforce conditions:
* **Scenario `burnout`:** Post-restructuring overtime surge ($\Delta \text{overtime} \ge +25\%$) accompanied by elevated unplanned sick leaves ($\Delta \text{leaves} \ge +6.5\text{ days}$).
* **Scenario `rto`:** Strict Return-to-Office mandate causing suburban commute friction ($\Delta \text{distance} \ge 2\times$).
* **Scenario `market_shock`:** Macro inflation causing internal pay bands to lag competitor salaries ($\Delta \text{wage diff} \le -22\%$).
* **Scenario `compound`:** Enterprise-wide crisis combining burnout, commute strain, and compensation compression.
* **Testing Command:** `make test-drift-scenario` (runs simulation -> PSI alarm check -> automated challenger retraining & promotion tournament).

---

## 8. Model Explainability: SHAP & Narrative Generation

HR intervention requires clear explanation of *why* an employee is classified as flight risk. Opaque black-box scores are unacceptable in enterprise HR.

### 8.1 SHAP TreeExplainer & Narrative Formatter (`src/explainability/explainer.py`)

```python
"""Local and global explainability engine utilizing SHAP."""

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

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def explain_employee(self, feature_row: np.ndarray, top_k: int = 3) -> tuple[float, str]:
        """Explain an individual employee's prediction with top K risk factors."""
        shap_values = self.explainer.shap_values(feature_row.reshape(1, -1))
        # Handle binary classification output shape
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        # Sort indices by positive contribution to flight risk
        top_indices = np.argsort(vals)[::-1][:top_k]

        reasons = []
        for idx in top_indices:
            feat = self.feature_names[idx]
            impact = vals[idx]
            val = feature_row[idx]
            readable = FEATURE_READABLE_NAMES.get(feat, feat)

            if impact > 0.05:  # Positive driver of risk
                if feat == "dept_wage_diff_pct" and val < 0:
                    reasons.append(f"Compensation is {abs(val):.1%} below departmental average")
                elif feat == "overtime_ratio" and val > 0.15:
                    reasons.append(f"High overtime burden ({val:.1%} above standard hours)")
                elif feat == "distance_km" and val > 30:
                    reasons.append(f"Long commuting distance ({val:.0f} km)")
                elif feat == "tenure_years" and val > 3:
                    reasons.append(f"Extended tenure without recent advancement ({val:.1f} years)")
                else:
                    reasons.append(f"{readable} contributes significantly to flight risk")

        narrative = "; ".join(reasons) if reasons else "Standard retention factors observed"
        return float(np.sum(vals)), narrative
```

---

## 9. Odoo Custom Addon: `hr_flight_risk`

To display risk scores directly on the `hr.employee` form view with strict HR Manager access restrictions, the repository includes a custom Odoo 20 addon in `addons/hr_flight_risk`.

### 9.1 Module Manifest (`addons/hr_flight_risk/__manifest__.py`)

```python
{
    "name": "HR Employee Flight Risk Intelligence",
    "version": "20.0.1.0.0",
    "category": "Human Resources",
    "summary": "Predictive attrition flight risk scoring and explainability",
    "description": """
        Proactively monitors employee flight risk:
        - Flight Risk Score (0-100%)
        - Risk Level Badge (Low, Medium, High, Critical)
        - Key Explanatory Risk Factors (SHAP AI Narrative)
        - Last Calculated Timestamp
        - Restricted strictly to HR Managers
    """,
    "author": "Antigravity AI",
    "depends": ["hr"],
    "data": [
        "security/ir.model.access.csv",
        "views/hr_employee_views.xml",
    ],
    "installable": True,
    "application": False,
    "license": "LGPL-3",
}
```

### 9.2 Model Extension (`addons/hr_flight_risk/models/hr_employee.py`)

```python
from odoo import fields, models


class HrEmployee(models.Model):
    _inherit = "hr.employee"

    flight_risk_score = fields.Float(
        string="Flight Risk Score (%)",
        digits=(5, 2),
        help="Predicted probability of employee leaving within 6-12 months",
        groups="hr.group_hr_manager",
    )
    flight_risk_level = fields.Selection(
        selection=[
            ("low", "Low Risk (<40%)"),
            ("medium", "Medium Risk (40-69%)"),
            ("high", "High Risk (70-84%)"),
            ("critical", "Critical Risk (85%+)"),
        ],
        string="Risk Level",
        compute="_compute_flight_risk_level",
        store=True,
        groups="hr.group_hr_manager",
    )
    flight_risk_factors = fields.Text(
        string="Primary Risk Drivers (XAI)",
        help="Top SHAP-derived factors contributing to flight risk",
        groups="hr.group_hr_manager",
    )
    flight_risk_last_updated = fields.Datetime(
        string="Risk Last Evaluated",
        groups="hr.group_hr_manager",
    )

    def _compute_flight_risk_level(self):
        for record in self:
            score = record.flight_risk_score or 0.0
            if score >= 85.0:
                record.flight_risk_level = "critical"
            elif score >= 70.0:
                record.flight_risk_level = "high"
            elif score >= 40.0:
                record.flight_risk_level = "medium"
            else:
                record.flight_risk_level = "low"
```

### 9.3 Form View Inheritance (`addons/hr_flight_risk/views/hr_employee_views.xml`)

```xml
<?xml version="1.0" encoding="utf-8"?>
<odoo>
    <record id="view_employee_form_flight_risk" model="ir.ui.view">
        <field name="name">hr.employee.form.flight.risk</field>
        <field name="model">hr.employee</field>
        <field name="inherit_id" ref="hr.view_employee_form"/>
        <field name="arch" type="xml">
            <!-- Add Flight Risk Notebook Page visible strictly to HR Managers -->
            <xpath expr="//notebook" position="inside">
                <page string="Flight Risk Analytics" groups="hr.group_hr_manager" name="flight_risk_page">
                    <group>
                        <group string="Attrition Risk Assessment">
                            <field name="flight_risk_score" widget="progressbar"/>
                            <field name="flight_risk_level" widget="badge" 
                                   decoration-success="flight_risk_level == 'low'"
                                   decoration-warning="flight_risk_level == 'medium'"
                                   decoration-danger="flight_risk_level in ('high', 'critical')"/>
                            <field name="flight_risk_last_updated"/>
                        </group>
                        <group string="Explainable AI (SHAP Insights)">
                            <field name="flight_risk_factors" readonly="1" nolabel="1" placeholder="Risk factor drivers will appear here after evaluation..."/>
                        </group>
                    </group>
                </page>
            </xpath>
        </field>
    </record>
</odoo>
```

---

## 10. Closed-Loop Odoo XML-RPC Sync Service

The sync service evaluates active employees in batch and pushes calculated flight risk scores, levels, and SHAP narratives directly into Odoo.

```python
"""Batch sync worker pushing flight risk scores to Odoo via XML-RPC."""

import xmlrpc.client
from datetime import datetime, timezone
import pandas as pd
from src.config.settings import get_settings

settings = get_settings()


class OdooSyncService:
    """Pushes scored employee records back to Odoo hr.employee model."""

    def __init__(self):
        self.common = xmlrpc.client.ServerProxy(f"{settings.odoo_url}/xmlrpc/2/common")
        self.models = xmlrpc.client.ServerProxy(f"{settings.odoo_url}/xmlrpc/2/object")
        self.uid = self.common.authenticate(
            settings.odoo_db, settings.odoo_user, settings.odoo_password, {}
        )
        if not self.uid:
            raise ConnectionError("Odoo authentication failed during sync service init.")

    @classmethod
    def load_champion_model(cls):
        """Retrieve latest approved champion model directly from MLflow Model Registry."""
        import mlflow.pyfunc

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.mlflow_model_name}@champion"
        return mlflow.pyfunc.load_model(model_uri)

    def sync_batch(self, scored_df: pd.DataFrame) -> int:
        """Batch update employee records with flight risk attributes."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        updated_count = 0

        for _, row in scored_df.iterrows():
            emp_id = int(row["id"])
            vals = {
                "flight_risk_score": float(row["flight_risk_score"]),
                "flight_risk_factors": str(row["flight_risk_factors"]),
                "flight_risk_last_updated": now_str,
            }
            success = self.models.execute_kw(
                settings.odoo_db,
                self.uid,
                settings.odoo_password,
                "hr.employee",
                "write",
                [[emp_id], vals],
            )
            if success:
                updated_count += 1

        return updated_count
```

---

## 11. Synthetic Training Corpus Generator (`scripts/generate_hr_data.py`)

To bootstrap and rigorously benchmark the model prior to historical data accumulation, the pipeline provides a deterministic generator modeling IBM HR Analytics and realistic Odoo workforce dynamics.

```python
"""Deterministic synthetic HR workforce and attrition dataset generator."""

import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Human Resources", "Finance", "Operations"]


def generate_synthetic_hr_dataset(num_employees: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic employee records with probabilistic attrition ground truth."""
    np.random.seed(seed)
    random.seed(seed)

    records = []
    for i in range(1, num_employees + 1):
        dept = random.choice(DEPARTMENTS)
        age = int(np.random.normal(35, 8).clip(21, 62))
        tenure = round(float(np.random.exponential(3.5).clip(0.3, 18.0)), 1)
        dist_km = int(np.random.gamma(3, 4).clip(1, 80))

        # Base wages by department
        dept_base = {
            "Engineering": 15000000,
            "Finance": 14000000,
            "Sales": 12000000,
            "Marketing": 11000000,
            "Operations": 9000000,
            "Human Resources": 10000000,
        }
        base_wage = dept_base[dept]
        wage = int(np.random.normal(base_wage * (1 + 0.05 * tenure), base_wage * 0.15))

        overtime_ratio = round(float(np.random.beta(2, 5) * 0.8), 2)
        leave_days = int(np.random.poisson(10).clip(0, 30))
        years_since_promo = round(float(np.random.uniform(0.2, min(tenure, 7.0))), 1)

        # Probabilistic Ground Truth Attrition Calculation (6-12 month window)
        # Log-odds formulation
        log_odds = -2.2
        if wage < base_wage * 0.85:
            log_odds += 0.85  # Underpaid relative to dept
        if overtime_ratio > 0.30:
            log_odds += 0.90  # Heavy overtime burnout
        if years_since_promo > 3.0:
            log_odds += 0.75  # Lack of promotion / stagnation
        if dist_km > 35:
            log_odds += 0.50  # Long commute
        if tenure < 1.5:
            log_odds += 0.40  # Early career restlessness

        attrition_prob = 1.0 / (1.0 + np.exp(-log_odds))
        attrition_label = 1 if np.random.rand() < attrition_prob else 0

        records.append(
            {
                "employee_id": i,
                "name": f"Employee {i}",
                "department": dept,
                "age": age,
                "tenure_years": tenure,
                "distance_km": dist_km,
                "wage": wage,
                "overtime_ratio": overtime_ratio,
                "leave_days_taken": leave_days,
                "years_since_promo": years_since_promo,
                "attrition_prob_true": round(attrition_prob, 3),
                "will_leave": attrition_label,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic HR attrition data.")
    parser.add_argument("--samples", type=int, default=1500, help="Number of employee rows")
    parser.add_argument("--output", type=str, default="data/synthetic/hr_attrition_train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_hr_dataset(args.samples, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        f"[SUCCESS] Generated {len(df)} records. Base Attrition Rate: {df['will_leave'].mean():.2%}"
    )
```

---

## 11.5 Streamlit Web UI Architecture (`app.py`)

A full-featured Streamlit web application (`app.py`) provides HR business partners, people analytics leaders, and department managers an intuitive interface for flight risk scoring, What-If simulation, live ERP exploration, and MLOps governance.

* **Port:** `8501` (accessible at `http://localhost:8501`, triggered via `make ui`).
* **Design Standards:** Adheres to the Zinc SaaS minimalism system:
  * Minimal chrome (`stHeader`, toolbar, and footers hidden).
  * DM Sans typography for narrative headings and body; JetBrains Mono for metrics and KPIs.
  * Bordered cards with subtle box-shadows and semantic badges (Green for Low Risk, Amber for Medium Risk, Red for High Risk).
* **Application Tabs & Functional Capabilities:**
  1. **Individual Predictor & What-If Simulator:**
     * **Archetype Presets:** Instant loading of curated employee personas (*"Satisfied Architect"*, *"Commute Strain"*, *"Burnout Crisis"*).
     * **Feature Sliders:** Dynamic input for tenure, age, commute distance, salary, compa-ratio delta, overtime ratio, and absenteeism days.
     * **Plotly Radial Gauge:** Displays calibrated 0–100% flight risk with standard decision thresholds (Low <40%, Medium 40–70%, High $\ge$70%).
     * **Interactive SHAP Waterfall / Attribution:** Visualizes positive and negative feature pushes toward voluntary attrition.
     * **Plain-English Explanations:** Driver narratives generated by `FlightRiskExplainer`.
     * **Prescriptive Retention Playbook:** Automatic mapping of top risk factors to targeted HR retention interventions.
     * **What-If Countermeasure Simulator:** Checkbox scenarios (*"Apply 15% Salary Raise"*, *"Cap Overtime to Standard"*, *"Grant 3-Day Remote"*) that immediately recalculate and display post-intervention risk drops in real time.
  2. **Live Odoo Workforce Explorer:**
     * Direct integration with Odoo 20 using the modern External JSON-2 API (`POST /json/2/hr.employee/search_read`).
     * Scans active workforce records, computes live flight risk scores, and displays an interactive, sortable workforce roster.
  3. **Batch Dataset Scoring & Analytics:**
     * Evaluates baseline or drifted datasets (`hr_attrition_train.csv`, `hr_attrition_drifted.csv`).
     * Renders workforce risk tier donuts, probability distributions, top at-risk employee lists, and CSV export.
  4. **MLOps Governance & Real-Time Drift Monitor:**
     * Live PSI evaluations against the training baseline across all 7 features.
     * Visual PSI thresholds (0.10 moderate drift, 0.25 severe drift) and automated retraining alarms.

---

## 11.6 Flight Risk Prediction CLI (`scripts/predict.py`)

A unified inference CLI supporting four flexible scoring modalities:
1. **Curated Archetypes Demo:** `make predict` or `uv run python scripts/predict.py --sample`.
2. **Custom Parameter Inputs:** `uv run python scripts/predict.py --tenure 4.2 --age 33 --distance 45 --wage 9000000 --dept-wage-diff -0.30 --overtime 0.60 --leaves 20`.
3. **Live Odoo ERP Employees:** `uv run python scripts/predict.py --odoo --limit 5` (or `--employee-id <ID>`).
4. **Batch CSV Scoring:** `uv run python scripts/predict.py --data data/synthetic/hr_attrition_train.csv --limit 10`.

---

## 12. Makefile Automation Specification

A canonical `Makefile` standardizes all environment initialization, Docker orchestration, data generation, training, explainability, and Odoo XML-RPC synchronization.

```make
# =====================================================================
# Odoo HR Attrition & Flight Risk Intelligence - Automation Makefile
# =====================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help setup env-init compose-up compose-down compose-restart compose-logs compose-logs50 \
        generate-api-key generate-data ingest-odoo odoo-stats stats train tune simulate-drift detect-drift monitor-perf retrain retrain-force test-drift-scenario predict ui explain mlflow-ui sync-odoo test lint format clean

help: ## Show available commands
	@echo "Odoo HR Attrition Flight Risk Intelligence - Developer Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initialize Python 3.12 virtualenv and sync packages with Astral uv
	@echo "[+] Pinning Python 3.12 and syncing dependencies via uv..."
	uv python pin 3.12
	uv sync --extra dev

env-init: ## Initialize .env and config/odoo.conf from example templates
	@if [ ! -f .env ]; then \
		echo "[+] Copying .env.example to .env..."; \
		cp .env.example .env; \
	fi
	@if [ ! -f config/odoo.conf ]; then \
		echo "[+] Copying config/odoo.conf.example to config/odoo.conf..."; \
		cp config/odoo.conf.example config/odoo.conf; \
	fi

compose-up: ## Start Odoo, PostgreSQL, pgAdmin, and MLflow containers
	@echo "[+] Starting Docker Compose services in background..."
	docker compose up -d

compose-down: ## Stop Docker Compose containers
	@echo "[-] Tearing down Docker Compose containers..."
	docker compose down

compose-restart: ## Restart Odoo web container
	@echo "[+] Restarting Odoo web service..."
	docker compose restart web

compose-logs: ## View real-time logs from Odoo web container
	docker compose logs -f web

compose-logs50: ## View real-time logs from Odoo web container with last 50 lines
	docker compose logs -f web --tail 50

generate-api-key: ## Provision Odoo External JSON-2 API key and persist to .env
	@echo "[+] Provisioning Odoo External JSON-2 API key..."
	uv run python scripts/generate_api_key.py

generate-data: ## Generate synthetic HR attrition training dataset
	@echo "[+] Generating synthetic HR attrition dataset..."
	uv run python scripts/generate_hr_data.py --samples 2500 --output data/synthetic/hr_attrition_train.csv

ingest-odoo: ## Ingest synthetic workforce records into Odoo ERP via XML-RPC
	@echo "[+] Ingesting synthetic records into Odoo ERP..."
	uv run python scripts/ingest_to_odoo.py --limit 120

odoo-stats: ## Display workforce metrics from Odoo (Active, Departed y=1, Attendance)
	@echo "[+] Fetching workforce metrics from Odoo..."
	uv run python scripts/get_odoo_stats.py

stats: odoo-stats ## Alias for odoo-stats

train: ## Train XGBoost, Random Forest & Survival models with MLflow tracking
	@echo "[+] Training Attrition Flight Risk models..."
	uv run python scripts/train_model.py

tune: ## Train with Stratified 5-Fold hyperparameter search (RandomizedSearchCV)
	@echo "[+] Executing hyperparameter tuning and model optimization..."
	uv run python scripts/train_model.py --tune

simulate-drift: ## Simulate workforce drift scenarios ('compound', 'burnout', 'rto', 'market_shock')
	@echo "[+] Simulating workforce drift scenario..."
	uv run python scripts/simulate_drift.py --scenario compound

detect-drift: ## Detect population feature drift (PSI) and Kolmogorov-Smirnov distribution shifts
	@echo "[+] Inspecting dataset and feature drift against baseline..."
	uv run python scripts/detect_drift.py

monitor-perf: ## Evaluate model predictions against ground truth exits and SLA thresholds
	@echo "[+] Evaluating model accuracy, PR-AUC, and calibration against ground truth..."
	uv run python scripts/monitor_performance.py

retrain: ## Automated continuous retraining triggered by drift or performance degradation
	@echo "[+] Evaluating retraining triggers and executing champion-challenger pipeline..."
	uv run python scripts/retrain_pipeline.py

retrain-force: ## Force retrain and hyperparameter tune a challenger, promoting to champion
	@echo "[+] Force retraining challenger with hyperparameter tuning..."
	uv run python scripts/retrain_pipeline.py --force --tune

test-drift-scenario: ## End-to-end drift test: simulate drift, detect alarm, trigger retrain & tournament
	@echo "[+] Step 1/3: Simulating workforce drift crisis..."
	uv run python scripts/simulate_drift.py --scenario compound
	@echo "[+] Step 2/3: Detecting feature drift against baseline..."
	uv run python scripts/detect_drift.py --current data/synthetic/hr_attrition_drifted.csv
	@echo "[+] Step 3/3: Triggering automated continuous retraining & model tournament..."
	uv run python scripts/retrain_pipeline.py --current data/synthetic/hr_attrition_drifted.csv --tune

predict: ## Predict flight risk and generate SHAP explanations for sample archetypes
	@echo "[+] Running Flight Risk prediction & SHAP explainability demo..."
	uv run python scripts/predict.py --sample

ui: ## Launch Streamlit Flight Risk Prediction & Retention Dashboard
	@echo "[+] Starting HR Flight Risk Prediction Dashboard on http://localhost:8501..."
	uv run streamlit run app.py --server.port 8501 --server.headless true

explain: ## Generate SHAP explainability summary plots and test employee narrative
	@echo "[+] Computing SHAP values and feature attribution..."
	uv run python -m src.explainability.explainer

mlflow-ui: ## Launch MLflow UI tracking dashboard on port 5000
	@echo "[+] Launching MLflow UI on http://localhost:5000..."
	uv run mlflow ui --host 0.0.0.0 --port 5000

sync-odoo: ## Batch score active employees and push scores to Odoo via XML-RPC
	@echo "[+] Running inference and syncing flight risk scores to Odoo..."
	uv run python scripts/sync_flight_risk.py

## Run unit and integration test suite
test:
	uv run pytest tests/ -v

## Lint codebase with Ruff
lint:
	uv run ruff check .

## Auto-format codebase with Ruff
format:
	uv run ruff format .

## Clean temporary bytecode and build caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
```

---

## 13. Conventional Commits Standard

Every commit must strictly follow the Conventional Commits specification:

```text
<type>(<scope>): <short summary in imperative mood>

[optional body describing rationale, architecture decisions, trade-offs]

[optional footer(s) referencing issue IDs]
```

### Allowed Types

* `feat`: A new feature (e.g., custom Odoo addon, SHAP explainer, XML-RPC client, feature pipeline).
* `fix`: A bug fix (e.g., handling missing wage records, XML-RPC timeout, zero division in compa-ratio).
* `refactor`: Code restructuring without modifying behavior.
* `perf`: Performance improvements (e.g., vectorized pandas calculations, batch XML-RPC calls).
* `docs`: Documentation updates (e.g., GEMINI.md, README.md).
* `test`: Adding or enhancing test suites.
* `chore`: Dependency updates (`uv.lock`), Docker config tweaks, Makefile targets.

### Allowed Scopes

`(addon)`, `(data)`, `(features)`, `(models)`, `(survival)`, `(shap)`, `(xmlrpc)`, `(config)`, `(docker)`, `(make)`.

---

## 14. Antigravity Operational Protocol & Guardrails

Prior to generating, editing, or executing code in this repository, Antigravity agents must satisfy these operational checkpoints:

1. **Strict Featurization Ordering (No Data Leakage):**
   * **Mandatory Rule:** In all supervised model training, **always** split the dataset into train, validation, and test splits *before* fitting any scaler (`StandardScaler`), imputer (`SimpleImputer`), or encoder (`OneHotEncoder`).
   * Never compute department averages across the entire dataset before splitting; compute aggregations on training splits and map to validation/test sets to prevent target or feature leakage.

2. **Handling Missing or NULL Values:**
   * Attendance or leave records may be null for new hires. Impute with contextual domain defaults (e.g., zero overtime, zero unplanned leaves, median commuting distance) and explicitly document rationale.

3. **Privacy & Ethical AI Guardrails:**
   * **Zero Sensitive Attribute Weighting:** Protect against discriminatory bias. The model must not use protected attributes (gender, marital status, nationality, religion) as features.
   * **Strict Access Isolation:** Flight risk scores and SHAP explanations must **never** be exposed to non-managerial staff. Only users in the `hr.group_hr_manager` group may view or query risk fields.
   * **Constructive Purpose:** Flight risk intelligence is an intervention tool for career pathing, retention conversations, and compensation reviews—never for punitive measures or preemptive termination.

4. **Community vs. Enterprise Module Fallback:**
   * When connecting to Odoo Community Edition (where `hr_appraisal` is uninstallable/Enterprise-only), seamlessly fall back to Odoo `survey` evaluation records or synthetic performance scores without raising unhandled exceptions.

5. **Tooling & Dependency Discipline:**
   * Manage dependencies exclusively through Astral `uv` (`pyproject.toml`). Never run global `pip install`.
   * Keep all linters clean (`make lint` / `ruff check .`) before declaring tasks complete.
