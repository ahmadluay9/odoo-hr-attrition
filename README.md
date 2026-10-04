# Odoo HR Flight Risk & Retention Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Odoo 20](https://img.shields.io/badge/odoo-20.0-purple.svg)](https://www.odoo.com/)
[![MLflow](https://img.shields.io/badge/mlflow-3.16-0194E2.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An enterprise-grade, explainable machine learning platform and retention intelligence system integrated with **Odoo 20 ERP**. It predicts employee flight risk 6–12 months in advance, translates predictive signals into plain-English drivers using **SHAP**, provides real-time **What-If Countermeasure Simulation** in an interactive **Streamlit** dashboard, and automates continuous MLOps governance with **MLflow** drift detection and model retraining tournaments.

---

## 1. System Architecture & Services

The platform operates as a hybrid architecture connecting containerized services via Docker Compose with high-performance local ML worker and UI processes managed by Astral `uv`:

```
+---------------------------------------------------------------------------------------------------+
|                                      DOCKER COMPOSE STACK                                         |
|                                                                                                   |
|  +--------------------+    +--------------------+    +--------------------+    +----------------+ |
|  |     Odoo 20 ERP    |    |   PostgreSQL 18    |    |     pgAdmin 4      |    |   MLflow UI    | |
|  |   Community: 8069  |    |    Database: 5432  |    |    Admin UI: 5050  |    | Tracking: 5000 | |
|  +---------+----------+    +---------+----------+    +--------------------+    +-------+--------+ |
|            ^                         ^                                                 ^          |
+------------|-------------------------|-------------------------------------------------|----------+
             |                         |                                                 |
             | JSON-2 API / SQL        | Read-only Queries                               | Runs/Registry
             v                         v                                                 v
+---------------------------------------------------------------------------------------------------+
|                                 PYTHON 3.12 ML & STREAMLIT PIPELINE                              |
|                                                                                                   |
|  +---------------------+   +-----------------------+   +-------------------+   +----------------+ |
|  | Odoo JSON-2 Client  |-->| Feature Engineering   |-->| ML Classifiers    |-->| SHAP Explainer | |
|  | (API Key Auth)      |   | (Compa-Ratio, Overtime|   | (XGBoost/Cox PH)  |   | (TreeExplainer)| |
|  +---------------------+   +-----------------------+   +-------------------+   +--------+-------+ |
|                                                                                         |         |
|  +---------------------------------------------------------------------------------+    |         |
|  | STREAMLIT WEB UI (Port 8501)                                                    |<---+         |
|  | * Interactive Predictor & What-If Countermeasure Simulator                       |             |
|  | * Live Odoo 20 Workforce Explorer                                               |             |
|  | * Batch Dataset Scoring & Analytics                                             |             |
|  | * MLOps Drift Monitor & Population Stability Index (PSI)                        |             |
|  +---------------------------------------------------------------------------------+              |
+---------------------------------------------------------------------------------------------------+
```

### Containerized Infrastructure

| Service | Image | Internal Host | Exposed Port | Description |
| :--- | :--- | :--- | :--- | :--- |
| **web** | `odoo:20` | `web` | `8069` | Odoo 20 Community Application Server |
| **db** | `postgres:18` | `db` | `5432` | Relational Persistence (`hr_db`) |
| **pgadmin** | `dpage/pgadmin4:latest` | `pgadmin` | `5050` | Database Administration Web UI |
| **mlflow** | Local / Container | `localhost` | `5000` | Experiment Tracking & Model Registry |
| **streamlit** | Local / Python | `localhost` | `8501` | Interactive Flight Risk & Retention UI |

---

## 2. Key Capabilities

* **🔮 Proactive Flight Risk Identification:** Predicts whether an employee is at risk of voluntary exit within the next 6–12 months with high precision (PR-AUC $\ge$ 0.70).
* **💡 Explainable AI (SHAP):** Every prediction includes local feature attribution transformed into plain-English HR explanations (e.g., *"Compensation is 28% below department average"*, *"High overtime burden (55% above standard 40h)"*).
* **🔄 What-If Countermeasure Simulation:** Interactive sliders allow HR business partners to test retention actions (e.g., *"What happens if we grant a 15% salary raise or 2-day remote work?"*) and see the flight risk drop immediately.
* **🏢 Odoo 20 JSON-2 API Integration:** Full compatibility with modern Odoo External JSON-2 API (`POST /json/2/{model}/{method}`) utilizing bearer token authentication.
* **🛡️ Continuous MLOps & Drift Governance:** Tracks models in the MLflow Model Registry (`@champion` vs `@challenger`), monitors Population Stability Index (PSI) drift alarms, and automates retraining tournaments upon workforce distribution shifts.
* **🔒 Strict Role-Based Access Control (RBAC):** Flight risk scores, risk levels, and SHAP narratives are restricted strictly to HR Managers (`hr.group_hr_manager`).

---

## 3. Quickstart Guide

### Prerequisites
* [Docker Engine](https://docs.docker.com/engine/install/) (v20.10+) & [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
* [Astral uv](https://github.com/astral-sh/uv) (v0.1.20+)

### Step 1: Environment Setup
Initialize the configuration files and Python 3.12 virtual environment:
```bash
# Initialize local .env and config/odoo.conf
make env-init

# Set up Python 3.12 virtualenv and dependencies via uv
make setup
```

### Step 2: Start Background Infrastructure
Launch Odoo 20, PostgreSQL, and pgAdmin containers:
```bash
make compose-up
```

### Step 3: Provision Odoo API Key
Generate an External JSON-2 API key automatically via the Odoo container shell:
```bash
make generate-api-key
```
*This extracts the provisioned key and saves it to `.env` as `ODOO_API_KEY`.*

### Step 4: Generate Synthetic Workforce & Ingest to Odoo
```bash
# Generate 2,500 baseline employee records
make generate-data

# Ingest active workforce and attendance history into Odoo ERP
make ingest-odoo

# Verify workforce statistics from Odoo
make stats
```

### Step 5: Train & Tune the Champion Model
```bash
# Train baseline XGBoost classifier with MLflow tracking
make train

# Or run 5-Fold Stratified hyperparameter search (RandomizedSearchCV)
make tune
```

### Step 6: Launch the Web UI
Start the interactive Streamlit dashboard:
```bash
make ui
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 4. Making Predictions

### Option A: Interactive Web UI (`make ui`)
Launch the Streamlit dashboard on `http://localhost:8501` featuring:
1. **What-If Simulator**: Preset personas, feature sliders, Plotly radial gauge, SHAP waterfall chart, and dynamic countermeasure toggles.
2. **Live Odoo Explorer**: Real-time workforce scanning directly from Odoo tables with sortable risk rosters.
3. **Batch Scoring**: Full CSV dataset processing and export.
4. **MLOps Monitor**: Live PSI drift indicators.

### Option B: Quick CLI Demonstration (`make predict`)
Evaluates sample HR archetypes (Low, Medium, and High risk profiles):
```bash
make predict
```

### Option C: Score a Custom Employee Profile via CLI Flags
```bash
uv run python scripts/predict.py \
  --tenure 4.2 \
  --age 33 \
  --distance 45 \
  --wage 9000000 \
  --dept-wage-diff -0.30 \
  --overtime 0.60 \
  --leaves 20
```

### Option D: Score Live Employees from Odoo ERP
```bash
# Score the first 5 active employees from Odoo
uv run python scripts/predict.py --odoo --limit 5

# Score an employee by their Odoo ID
uv run python scripts/predict.py --odoo --employee-id 7
```

### Option E: Programmatic Python API
Load the active champion model directly from MLflow:
```python
import pandas as pd
from src.features.engineer import FeatureEngineer
from src.explainability.explainer import FlightRiskExplainer
from src.models.registry import ModelRegistryManager

# Load approved champion model from MLflow registry
registry = ModelRegistryManager()
model = registry.load_champion_model()
pipeline = model._model_impl.sklearn_model

# Prepare employee feature vector
employee = pd.DataFrame([[3.5, 30.0, 35.0, 11000000.0, -0.15, 0.40, 14.0]],
                        columns=FeatureEngineer.FEATURE_COLS)

# Compute probability and explainability drivers
prob = pipeline.predict_proba(employee)[0, 1]
explainer = FlightRiskExplainer(pipeline, FeatureEngineer.FEATURE_COLS)
_, reasons = explainer.explain_employee(employee.values[0], top_k=3)

print(f"Flight Risk: {prob:.1%}")
print("Drivers:", reasons)
```

---

## 5. MLOps Governance, Drift Detection & Retraining

To prevent model degradation over time, the system continuously monitors data distributions and model metrics:

```
+-------------------------------------------------------------------------------------+
|                              CONTINUOUS MLOPS LIFECYCLE                             |
|                                                                                     |
|   1. Production Scoring  -->  2. Drift Detection (PSI)  -->  3. Retraining Trigger  |
|                               (Alert if >= 2 RED features)   (Baseline + Drifted)   |
|                                                                       |             |
|   5. Model Registry      <--  4. Model Tournament       <-------------+             |
|   (Promote to @champion       (Challenger vs Champion                               |
|    if PR-AUC improved)         5-Fold Stratified Tuning)                            |
+-------------------------------------------------------------------------------------+
```

### Testing the Drift Scenario End-to-End
Execute the full automated drift crisis and retraining tournament with a single command:
```bash
make test-drift-scenario
```
This recipe:
1. **Simulates workforce shock**: Injects a compound crisis (+25% overtime surge, 2x commute distance, -22% real wage decline).
2. **Detects feature drift**: Flags population distribution shifts ($PSI \ge 0.25$ on 4 features).
3. **Executes model tournament**: Retrains with 5-Fold Stratified tuning, logs runs to MLflow, and evaluates challenger metrics against champion.

### Inspecting Models in MLflow UI
View all runs, parameter comparisons, and registered model versions:
```bash
make mlflow-ui
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 6. Developer Commands Reference (`Makefile`)

| Command | Purpose |
| :--- | :--- |
| `make help` | Show all available automation commands |
| `make setup` | Initialize Python 3.12 virtualenv and sync packages with Astral `uv` |
| `make env-init` | Initialize local `.env` and `config/odoo.conf` from examples |
| `make compose-up` | Start Odoo, PostgreSQL, pgAdmin, and MLflow containers |
| `make compose-down` | Stop and tear down Docker Compose containers |
| `make compose-restart` | Restart the Odoo application container |
| `make compose-logs` | Stream real-time logs from Odoo web container |
| `make generate-api-key` | Provision Odoo External JSON-2 API key and persist to `.env` |
| `make generate-data` | Generate synthetic HR attrition training dataset (2,500 samples) |
| `make ingest-odoo` | Ingest synthetic workforce records into Odoo ERP |
| `make stats` | Display workforce metrics from Odoo (Active, Departed, Attendance) |
| `make train` | Train XGBoost flight risk classifier with MLflow tracking |
| `make tune` | Train with Stratified 5-Fold hyperparameter search (`RandomizedSearchCV`) |
| `make simulate-drift` | Simulate workforce drift scenario (`compound`, `burnout`, `rto`) |
| `make detect-drift` | Detect feature drift against baseline using PSI and KS tests |
| `make monitor-perf` | Evaluate model accuracy and PR-AUC against ground truth SLA targets |
| `make retrain` | Automated continuous retraining triggered by drift or performance decay |
| `make test-drift-scenario` | End-to-end drift crisis simulation, detection, and retraining tournament |
| `make predict` | Run interactive sample flight risk prediction with champion model |
| `make ui` | Launch Streamlit Flight Risk Prediction & Retention Dashboard on port 8501 |
| `make explain` | Generate SHAP explainability summary plots and test employee narrative |
| `make mlflow-ui` | Launch MLflow UI tracking dashboard on port 5000 |
| `make sync-odoo` | Batch score active employees and push risk scores to Odoo via API |
| `make test` | Run unit and integration test suite via `pytest` |
| `make lint` | Lint codebase with `ruff check .` |
| `make format` | Auto-format codebase with `ruff format .` |
| `make clean` | Remove temporary cache files and artifacts |

---

## 7. Security, Privacy & Ethical AI Guardrails

1. **Strict Role-Based Access Control (RBAC):** Flight risk scores and explanations are restricted exclusively to HR Managers (`hr.group_hr_manager`). Flight risk fields are hidden from non-managerial staff.
2. **Zero Protected Attribute Weighting:** The predictive models strictly exclude protected demographic attributes (gender, race, religion, marital status) to eliminate algorithmic discrimination.
3. **Constructive Retention Interventions:** Flight risk scores are intended solely as early indicators for proactive career development, workload rebalancing, and compensation reviews—never for punitive actions.
4. **Zero Data Leakage:** All data preprocessing (imputation, scaling) is fit strictly on training splits *after* train-test splitting to ensure zero target or distribution leakage.

---

## 8. License

This project is licensed under the [Apache License 2.0](LICENSE).
