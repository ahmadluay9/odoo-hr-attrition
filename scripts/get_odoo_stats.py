#!/usr/bin/env python3
"""Workforce Statistics Reporter for Odoo HR Attrition Pipeline.

Queries the active Odoo ERP instance via External JSON-2 API (or XML-RPC fallback)
to retrieve workforce counts:
- Total Active Employees
- Total Departed Employees (y = 1 ground truth)
- Total Historical Workforce
- Total Attendance Logs
- Department Breakdown
"""

import argparse
import collections
import json
import os
import sys
import xmlrpc.client

from dotenv import load_dotenv

load_dotenv()


def get_stats_json2(url: str, db: str, api_key: str) -> dict:
    """Fetch workforce metrics using Odoo 20 External JSON-2 API."""
    import requests

    headers = {
        "Authorization": f"bearer {api_key}",
        "X-Odoo-Database": db,
        "Content-Type": "application/json",
    }
    base = url.rstrip("/")

    # 1. Active Employees
    res_active = requests.post(
        f"{base}/json/2/hr.employee/search_count",
        headers=headers,
        json={"domain": [["active", "=", True]]},
    )
    res_active.raise_for_status()
    active_count = res_active.json()

    # 2. Departed Employees (active = False)
    res_departed = requests.post(
        f"{base}/json/2/hr.employee/search_count",
        headers=headers,
        json={"domain": [["active", "=", False]], "context": {"active_test": False}},
    )
    res_departed.raise_for_status()
    departed_count = res_departed.json()

    # 3. Attendance Logs
    res_att = requests.post(
        f"{base}/json/2/hr.attendance/search_count",
        headers=headers,
        json={"domain": []},
    )
    res_att.raise_for_status()
    attendance_count = res_att.json()

    # 4. Department Breakdown
    res_emps = requests.post(
        f"{base}/json/2/hr.employee/search_read",
        headers=headers,
        json={
            "domain": [],
            "fields": ["name", "active", "department_id"],
            "context": {"active_test": False},
        },
    )
    res_emps.raise_for_status()
    all_employees = res_emps.json()

    dept_stats = collections.defaultdict(lambda: {"active": 0, "departed": 0})
    for emp in all_employees:
        dept = emp["department_id"][1] if emp.get("department_id") else "Unassigned"
        if emp.get("active"):
            dept_stats[dept]["active"] += 1
        else:
            dept_stats[dept]["departed"] += 1

    return {
        "api_protocol": "External JSON-2 API (/json/2/)",
        "active_employees": active_count,
        "departed_employees": departed_count,
        "total_workforce": active_count + departed_count,
        "historical_attrition_rate": (
            round((departed_count / (active_count + departed_count)) * 100, 2)
            if (active_count + departed_count) > 0
            else 0.0
        ),
        "attendance_logs": attendance_count,
        "departments": dict(dept_stats),
    }


def get_stats_xmlrpc(url: str, db: str, user: str, pwd: str) -> dict:
    """Fallback workforce metrics extraction via legacy XML-RPC."""
    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, user, pwd, {})
    if not uid:
        raise ConnectionError(f"Authentication failed for user {user}")

    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")

    # 1. Active Employees
    active_count = models.execute_kw(
        db, uid, pwd, "hr.employee", "search_count", [[["active", "=", True]]]
    )

    # 2. Departed Employees (active = False)
    departed_count = models.execute_kw(
        db,
        uid,
        pwd,
        "hr.employee",
        "search_count",
        [[["active", "=", False]]],
        {"context": {"active_test": False}},
    )

    # 3. Attendance Logs
    attendance_count = models.execute_kw(db, uid, pwd, "hr.attendance", "search_count", [[]])

    # 4. Department Breakdown
    all_employees = models.execute_kw(
        db,
        uid,
        pwd,
        "hr.employee",
        "search_read",
        [[]],
        {"fields": ["name", "active", "department_id"], "context": {"active_test": False}},
    )

    dept_stats = collections.defaultdict(lambda: {"active": 0, "departed": 0})
    for emp in all_employees:
        dept = emp["department_id"][1] if emp.get("department_id") else "Unassigned"
        if emp.get("active"):
            dept_stats[dept]["active"] += 1
        else:
            dept_stats[dept]["departed"] += 1

    return {
        "api_protocol": "XML-RPC API (/xmlrpc/2/)",
        "active_employees": active_count,
        "departed_employees": departed_count,
        "total_workforce": active_count + departed_count,
        "historical_attrition_rate": (
            round((departed_count / (active_count + departed_count)) * 100, 2)
            if (active_count + departed_count) > 0
            else 0.0
        ),
        "attendance_logs": attendance_count,
        "departments": dict(dept_stats),
    }


def print_formatted_summary(stats: dict) -> None:
    """Render beautiful formatted CLI report."""
    title = "ODOO WORKFORCE & ATTRITION STATUS REPORT"
    line = "=" * 65
    subline = "-" * 65

    print("\n" + line)
    print(f"{title:^65}")
    print(f"Protocol: {stats['api_protocol']:^55}")
    print(line)
    print(f"  * Total Active Employees (Inference Pool) : {stats['active_employees']:>6,}")
    print(f"  * Total Departed Employees (y = 1 Ground)  : {stats['departed_employees']:>6,}")
    print(f"  * Total Historical Employees               : {stats['total_workforce']:>6,}")
    print(
        f"  * Historical Attrition Rate                : {stats['historical_attrition_rate']:>6.2f} %"
    )
    print(f"  * Total Attendance Punches / Logs          : {stats['attendance_logs']:>6,}")
    print(line)
    print("  DEPARTMENT BREAKDOWN:")
    print(subline)
    print(f"  {'Department':<42} | {'Active':<6} | {'Departed':<8}")
    print(subline)

    for dept, counts in sorted(stats["departments"].items()):
        print(f"  {dept:<42} | {counts['active']:>6} | {counts['departed']:>8}")

    print(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Odoo workforce statistics")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    args = parser.parse_args()

    url = os.getenv("ODOO_URL", "http://localhost:8069")
    db = os.getenv("ODOO_DB", "hr_db")
    api_key = os.getenv("ODOO_API_KEY")
    user = os.getenv("ODOO_USER", "admin@admin.com")
    pwd = os.getenv("ODOO_PASSWORD", "")

    try:
        if api_key:
            stats = get_stats_json2(url, db, api_key)
        else:
            stats = get_stats_xmlrpc(url, db, user, pwd)

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print_formatted_summary(stats)

    except Exception as e:
        print(f"[!] Error fetching Odoo workforce stats: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
