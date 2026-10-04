#!/usr/bin/env python3
"""Automated Odoo API Key Generator for External JSON-2 API.

Connects to the running Odoo web container via Docker, provisions an API key for the
configured admin user using `res.users.apikeys._generate()`, and persists it to `.env`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def get_configured_user() -> str:
    """Read ODOO_USER from .env or return default."""
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("ODOO_USER="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "admin@admin.com"


def update_env_file(key: str) -> bool:
    """Update or append ODOO_API_KEY in .env."""
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(f"ODOO_API_KEY={key}\n", encoding="utf-8")
        return True

    content = env_file.read_text(encoding="utf-8")
    if "ODOO_API_KEY=" in content:
        new_content = re.sub(r"ODOO_API_KEY=.*", f"ODOO_API_KEY={key}", content)
    else:
        new_content = content.rstrip() + f"\nODOO_API_KEY={key}\n"

    env_file.write_text(new_content, encoding="utf-8")
    return True


def main() -> None:
    user_email = os.getenv("ODOO_USER", get_configured_user())
    key_name = os.getenv("KEY_NAME", "odoo-hr-attrition-key")

    print(f"[+] Generating Odoo API key for user: {user_email} (Name: {key_name})...")

    # Shell snippet to run in Odoo shell
    odoo_script = f"""
user = env['res.users'].search([('login', '=', '{user_email}')], limit=1)
if not user:
    print('__ERROR__:User not found with login: {user_email}')
else:
    key = env['res.users.apikeys'].with_user(user)._generate(
        scope='rpc',
        name='{key_name}',
        expiration_date=False
    )
    env.cr.commit()
    print('__KEY_RESULT__:' + key)
"""

    cmd = ["docker", "compose", "exec", "-T", "web", "odoo", "shell", "-d", "hr_db", "--no-http"]

    try:
        proc = subprocess.run(
            cmd,
            input=odoo_script,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print("[!] Error: docker command not found. Please ensure Docker is installed and in PATH.")
        sys.exit(1)

    output = proc.stdout + proc.stderr

    if "__ERROR__:" in output:
        err_msg = output.split("__ERROR__:", 1)[1].splitlines()[0]
        print(f"[!] Error: {err_msg}")
        sys.exit(1)

    match = re.search(r"__KEY_RESULT__:([a-f0-9]+)", output)
    if not match:
        print("[!] Failed to parse generated API key from Odoo output.")
        print("    Docker output:", output[-500:])
        sys.exit(1)

    api_key = match.group(1).strip()
    print(f"[SUCCESS] Generated API Key: {api_key}")

    if update_env_file(api_key):
        print("[+] Successfully saved ODOO_API_KEY to .env file.")

    print("\n[+] Verification Test:")
    print("    curl -s -X POST http://localhost:8069/json/2/hr.employee/search_count \\")
    print(f'      -H "Authorization: bearer {api_key}" \\')
    print('      -H "X-Odoo-Database: hr_db" \\')
    print('      -H "Content-Type: application/json" \\')
    print("      -d '{\"domain\": []}'\n")


if __name__ == "__main__":
    main()
