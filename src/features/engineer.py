"""Feature engineering pipeline for Odoo workforce tables."""

import pandas as pd


class FeatureEngineer:
    """Transforms raw Odoo DataFrames into tabular ML features."""

    FEATURE_COLS = [
        "tenure_years",
        "age",
        "distance_km",
        "wage",
        "dept_wage_diff_pct",
        "overtime_ratio",
        "leave_days_taken",
    ]

    def build_features(
        self,
        employees_df: pd.DataFrame,
        contracts_df: pd.DataFrame | None = None,
        leaves_df: pd.DataFrame | None = None,
        attendance_df: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Execute end-to-end feature aggregation for inference or training."""
        ref_date = reference_date or pd.Timestamp.now()

        df = employees_df.copy()

        # 1. Base Employee Features
        if "create_date" in df.columns:
            df["tenure_years"] = (
                ref_date - pd.to_datetime(df["create_date"])
            ).dt.total_seconds() / (365.25 * 86400)
        elif "tenure_years" not in df.columns:
            df["tenure_years"] = 1.0
        df["tenure_years"] = df["tenure_years"].fillna(1.0).clip(lower=0.1, upper=40.0)

        if "birthday" in df.columns:
            df["age"] = (ref_date - pd.to_datetime(df["birthday"])).dt.total_seconds() / (
                365.25 * 86400
            )
        elif "age" not in df.columns:
            df["age"] = 35.0
        df["age"] = df["age"].fillna(35.0).clip(lower=18.0, upper=75.0)

        if "km_home_work" in df.columns:
            df["distance_km"] = df["km_home_work"].fillna(
                df["km_home_work"].median() if not df["km_home_work"].dropna().empty else 10.0
            )
        elif "distance_km" not in df.columns:
            df["distance_km"] = 10.0
        df["distance_km"] = df["distance_km"].fillna(10.0).clip(lower=0.0, upper=150.0)

        # Clean Department
        if "department_id" in df.columns:
            df["department_id_clean"] = df["department_id"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else (x if pd.notna(x) else 0)
            )
        else:
            df["department_id_clean"] = 0

        # 2. Wage & Department Compa-Ratio
        if contracts_df is not None and not contracts_df.empty and "wage" in contracts_df.columns:
            contracts = contracts_df.copy()
            if "date_start" in contracts.columns:
                contracts = contracts.sort_values("date_start")
            contracts["emp_id"] = contracts["employee_id"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else x
            )
            latest_contracts = contracts.groupby("emp_id").last().reset_index()
            df = df.merge(
                latest_contracts[["emp_id", "wage"]],
                left_on="id",
                right_on="emp_id",
                how="left",
                suffixes=("", "_contract"),
            )
        if "wage" not in df.columns:
            df["wage"] = 5000.0
        df["wage"] = df["wage"].fillna(
            df["wage"].median() if not df["wage"].dropna().empty else 5000.0
        )

        # Department Average Wage
        dept_avg = df.groupby("department_id_clean")["wage"].transform("mean")
        df["dept_wage_diff_pct"] = (df["wage"] - dept_avg) / (dept_avg + 1e-6)
        df["dept_wage_diff_pct"] = df["dept_wage_diff_pct"].fillna(0.0).clip(lower=-0.9, upper=2.0)

        # 3. Attendance & Overtime Burden
        if attendance_df is not None and not attendance_df.empty:
            att = attendance_df.copy()
            att["emp_id"] = att["employee_id"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else x
            )
            att_summary = att.groupby("emp_id")["worked_hours"].agg(["count", "sum"]).reset_index()
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
        if "overtime_ratio" not in df.columns:
            df["overtime_ratio"] = 0.0
        df["overtime_ratio"] = df["overtime_ratio"].fillna(0.0).clip(lower=-0.5, upper=1.5)

        # 4. Absenteeism & Leave Velocity
        if leaves_df is not None and not leaves_df.empty:
            lv = leaves_df.copy()
            lv["emp_id"] = lv["employee_id"].apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) else x
            )
            leave_summary = lv.groupby("emp_id")["number_of_days"].sum().reset_index()
            leave_summary.rename(columns={"number_of_days": "leave_days_taken"}, inplace=True)
            df = df.merge(leave_summary, left_on="id", right_on="emp_id", how="left")
        if "leave_days_taken" not in df.columns:
            df["leave_days_taken"] = 0.0
        df["leave_days_taken"] = df["leave_days_taken"].fillna(0.0).clip(lower=0.0, upper=60.0)

        meta_cols = [c for c in ["id", "name", "active"] if c in df.columns]
        return df[meta_cols + self.FEATURE_COLS]
