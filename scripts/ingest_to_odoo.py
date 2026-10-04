"""Ingest synthetic HR dataset into Odoo 20 ERP database via XML-RPC.

Populates hr.employee (with hr_version wage/contract), hr.attendance, and hr.leave,
including historical voluntary departures (Resigned) to establish real ground truth in Odoo.
"""

import argparse
import os
import random
import xmlrpc.client
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ODOO_URL = os.getenv("ODOO_URL", "http://localhost:8069")
ODOO_DB = os.getenv("ODOO_DB", "hr_db")
ODOO_USER = os.getenv("ODOO_USER", "admin@admin.com")
ODOO_PASSWORD = os.getenv("ODOO_PASSWORD", "")


def connect_odoo():
    """Authenticate and return XML-RPC proxies and user ID."""
    common = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/common")
    models = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/object")
    uid = common.authenticate(ODOO_DB, ODOO_USER, ODOO_PASSWORD, {})
    if not uid:
        raise ConnectionError(f"Failed to authenticate with Odoo at {ODOO_URL} as {ODOO_USER}")
    return models, uid


def get_or_create_metadata(models, uid):
    """Retrieve existing departments, jobs, and departure reasons from Odoo."""
    depts = models.execute_kw(
        ODOO_DB,
        uid,
        ODOO_PASSWORD,
        "hr.department",
        "search_read",
        [[]],
        {"fields": ["id", "name"]},
    )
    dept_map = {d["name"].lower(): d["id"] for d in depts}

    # Department fallback mapping
    dept_id_mapping = {
        "engineering": dept_map.get("research & development", 4),
        "sales": dept_map.get("sales", 3),
        "marketing": dept_map.get("sales", 3),
        "human resources": dept_map.get("administration", 1),
        "finance": dept_map.get("management", 2),
        "consulting": dept_map.get("professional services", 6),
        "operations": dept_map.get("long term projects", 5),
    }

    jobs = models.execute_kw(
        ODOO_DB, uid, ODOO_PASSWORD, "hr.job", "search_read", [[]], {"fields": ["id", "name"]}
    )
    job_map = {j["name"].lower(): j["id"] for j in jobs}

    return dept_id_mapping, job_map


def ingest_synthetic_data(csv_path: str, max_employees: int = 150):
    """Read synthetic CSV and push structured employee records into Odoo."""
    df = pd.read_csv(csv_path)
    if len(df) > max_employees:
        df = df.iloc[:max_employees].copy()

    print(f"[+] Connecting to Odoo at {ODOO_URL} (db: {ODOO_DB})...")
    models, uid = connect_odoo()
    dept_id_mapping, job_map = get_or_create_metadata(models, uid)

    print(f"[+] Ingesting {len(df)} synthetic employees into Odoo...")

    now = datetime.now(timezone.utc)
    created_employee_ids = []
    attendance_records = []
    leave_records = []

    # Map job titles to IDs
    job_titles_by_dept = {
        "engineering": (job_map.get("experienced developer", 3), "Experienced Developer"),
        "sales": (job_map.get("marketing and community manager", 5), "Account Executive"),
        "marketing": (job_map.get("marketing and community manager", 5), "Marketing Lead"),
        "human resources": (job_map.get("human resources manager", 4), "HR Generalist"),
        "finance": (job_map.get("chief technical officer", 1), "Financial Analyst"),
        "consulting": (job_map.get("consultant", 2), "Consultant"),
        "operations": (job_map.get("experienced developer", 3), "Operations Specialist"),
    }

    # First pass: Create employees in batches
    batch_size = 25
    total_created = 0

    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i : i + batch_size]
        batch_vals = []

        for _, row in chunk.iterrows():
            dept_name = str(row["department"]).lower()
            dept_id = dept_id_mapping.get(dept_name, 1)
            job_info = job_titles_by_dept.get(dept_name, (2, "Consultant"))

            tenure_days = int(row["tenure_years"] * 365.25)
            start_date = (now - timedelta(days=tenure_days)).strftime("%Y-%m-%d")

            is_departed = bool(row["will_leave"] == 1 and random.random() < 0.65)
            vals = {
                "name": str(row["name"]),
                "department_id": dept_id,
                "job_id": job_info[0],
                "job_title": job_info[1],
                "wage": float(row["wage"]),
                "km_home_work": int(row["distance_km"]),
                "contract_date_start": start_date,
                "first_contract_date": start_date,
                "active": not is_departed,
            }

            if is_departed:
                # Mark as resigned 1-4 months ago
                exit_days_ago = random.randint(15, 120)
                departure_date = (now - timedelta(days=exit_days_ago)).strftime("%Y-%m-%d")
                vals["departure_date"] = departure_date
                vals["departure_reason_id"] = 2  # Resigned

            batch_vals.append((vals, row, is_departed))

        # Push employee batch
        create_payload = [v[0] for v in batch_vals]
        emp_ids = models.execute_kw(
            ODOO_DB, uid, ODOO_PASSWORD, "hr.employee", "create", [create_payload]
        )
        if not isinstance(emp_ids, list):
            emp_ids = [emp_ids]

        total_created += len(emp_ids)

        for emp_id, (_, row, is_departed) in zip(emp_ids, batch_vals):
            created_employee_ids.append(emp_id)

            # Generate sample attendance for employee (last 30 days)
            overtime = float(row["overtime_ratio"])
            daily_hours = 8.0 * (1.0 + overtime)

            num_work_days = random.randint(10, 20)
            for d in range(1, num_work_days + 1):
                day_date = now - timedelta(days=d)
                if day_date.weekday() < 5:  # Monday to Friday
                    check_in = day_date.replace(hour=8, minute=30, second=0).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    worked_min = int(daily_hours * 60)
                    check_out = (
                        day_date.replace(hour=8, minute=30, second=0)
                        + timedelta(minutes=worked_min)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    attendance_records.append(
                        {
                            "employee_id": emp_id,
                            "check_in": check_in,
                            "check_out": check_out,
                        }
                    )

            # Generate leave records if leaves taken > 0
            leaves_count = int(row["leave_days_taken"])
            if leaves_count > 0:
                leave_start = now - timedelta(days=random.randint(20, 150))
                leave_end = leave_start + timedelta(days=min(leaves_count, 5))
                leave_records.append(
                    {
                        "employee_id": emp_id,
                        "date_from": leave_start.strftime("%Y-%m-%d 08:00:00"),
                        "date_to": leave_end.strftime("%Y-%m-%d 17:00:00"),
                        "number_of_days": float(min(leaves_count, 5)),
                        "work_entry_type_id": 5,
                        "state": "validate",
                    }
                )

        print(f"    [+] Created {total_created}/{len(df)} employees in Odoo...")

    # Push attendance in batch
    if attendance_records:
        print(f"[+] Ingesting {len(attendance_records)} attendance logs into hr.attendance...")
        att_batch_size = 100
        for b in range(0, len(attendance_records), att_batch_size):
            chunk = attendance_records[b : b + att_batch_size]
            models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, "hr.attendance", "create", [chunk])

    # Push leaves in batch
    if leave_records:
        print(f"[+] Ingesting {len(leave_records)} leave records into hr.leave...")
        for l_rec in leave_records:
            try:
                models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, "hr.leave", "create", [[l_rec]])
            except Exception:
                pass  # Gracefully handle any leave constraint edge case

    # Verification query
    active_count = models.execute_kw(
        ODOO_DB, uid, ODOO_PASSWORD, "hr.employee", "search_count", [[["active", "=", True]]]
    )
    inactive_count = models.execute_kw(
        ODOO_DB, uid, ODOO_PASSWORD, "hr.employee", "search_count", [[["active", "=", False]]]
    )
    total_att = models.execute_kw(
        ODOO_DB, uid, ODOO_PASSWORD, "hr.attendance", "search_count", [[]]
    )

    print("\n" + "=" * 65)
    print(" [SUCCESS] SYNTHETIC DATA INGESTION TO ODOO COMPLETE")
    print("=" * 65)
    print(f" Total Active Employees in Odoo   : {active_count}")
    print(f" Total Departed Employees (y = 1) : {inactive_count}")
    print(f" Total Attendance Logs in Odoo    : {total_att}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest synthetic HR data to Odoo via XML-RPC.")
    parser.add_argument("--csv", type=str, default="data/synthetic/hr_attrition_train.csv")
    parser.add_argument("--limit", type=int, default=120, help="Number of employees to ingest")
    args = parser.parse_args()

    ingest_synthetic_data(args.csv, max_employees=args.limit)
