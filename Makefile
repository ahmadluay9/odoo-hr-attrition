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

test: ## Run unit and integration test suite
	uv run pytest tests/ -v

lint: ## Lint codebase with Ruff
	uv run ruff check .

format: ## Auto-format codebase with Ruff
	uv run ruff format .

clean: ## Clean temporary bytecode and build caches
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
