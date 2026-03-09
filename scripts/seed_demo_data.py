"""Seed a demo-ready user, account connections, and ops data for video walkthroughs.

Usage:
  python scripts/seed_demo_data.py
  python scripts/seed_demo_data.py --email demo@autopilot.local --password "DemoPass123!"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config
from services.auth_store import AuthStore
from services.ops_state_store import OpsStateStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed demo user and project data")
    parser.add_argument("--email", default="demo@autopilot.local", help="Demo account email")
    parser.add_argument("--password", default="DemoPass123!", help="Demo account password")
    parser.add_argument(
        "--skip-aws",
        action="store_true",
        help="Skip AWS credential validation and connection seeding",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub token validation and connection seeding",
    )
    return parser.parse_args()


def _get_or_create_user(auth_store: AuthStore, email: str, password: str) -> Dict[str, Any]:
    normalized_email = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()

    with auth_store._conn() as conn:  # Intentional private access for idempotent demo bootstrap.
        row = conn.execute("SELECT id FROM users WHERE email = ?", (normalized_email,)).fetchone()
        password_hash, password_salt = auth_store._hash_password(password)

        if row:
            user_id = int(row["id"])
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?, updated_at = ?
                WHERE id = ?
                """,
                (password_hash, password_salt, now, user_id),
            )
            return {"id": user_id, "email": normalized_email, "created": False}

    created = auth_store.create_user(normalized_email, password)
    return {"id": int(created["id"]), "email": normalized_email, "created": True}


def _seed_aws_connection(auth_store: AuthStore, user_id: int) -> Dict[str, Any]:
    access_key_id = (config.AWS_ACCESS_KEY_ID or "").strip()
    secret_access_key = (config.AWS_SECRET_ACCESS_KEY or "").strip()
    env_session_token = (os.getenv("AWS_SESSION_TOKEN") or "").strip() or None
    region = (config.AWS_REGION or "us-east-1").strip()

    if not access_key_id or not secret_access_key:
        return {
            "seeded": False,
            "message": "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY not found in .env",
        }

    session = boto3.session.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=env_session_token,
        region_name=region,
    )

    permissions = {
        "sts:GetCallerIdentity": False,
        "cloudwatch:ListMetrics": False,
        "ce:GetCostAndUsage": False,
    }

    try:
        identity = session.client("sts").get_caller_identity()
        permissions["sts:GetCallerIdentity"] = True
    except (ClientError, BotoCoreError, Exception) as exc:
        return {"seeded": False, "message": f"AWS validation failed: {exc}"}

    try:
        session.client("cloudwatch").list_metrics(Namespace="AWS/EC2", MetricName="CPUUtilization")
        permissions["cloudwatch:ListMetrics"] = True
    except Exception:
        pass

    try:
        now = datetime.now(timezone.utc)
        session.client("ce", region_name="us-east-1").get_cost_and_usage(
            TimePeriod={
                "Start": now.replace(day=1).date().strftime("%Y-%m-%d"),
                "End": now.date().strftime("%Y-%m-%d"),
            },
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
        )
        permissions["ce:GetCostAndUsage"] = True
    except Exception:
        pass

    auth_store.upsert_aws_connection(
        user_id=user_id,
        account_id=identity.get("Account", ""),
        arn=identity.get("Arn", ""),
        region=region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=env_session_token,
        validated_permissions=json.dumps(permissions),
    )

    return {
        "seeded": True,
        "account_id": identity.get("Account", ""),
        "region": region,
        "permissions": permissions,
    }


def _seed_github_connection(auth_store: AuthStore, user_id: int) -> Dict[str, Any]:
    token = (config.GITHUB_TOKEN or "").strip()
    repo_owner = (config.GITHUB_REPO_OWNER or "").strip() or None
    repo_name = (config.GITHUB_REPO_NAME or "").strip() or None

    if not token:
        return {"seeded": False, "message": "GITHUB_TOKEN not found in .env"}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "autopilot-ai-demo-seeder",
    }

    username = ""
    scopes = ""

    try:
        user_res = requests.get("https://api.github.com/user", headers=headers, timeout=20)
        user_res.raise_for_status()
        user_json = user_res.json()
        username = str(user_json.get("login") or "")
        scopes = user_res.headers.get("X-OAuth-Scopes", "")
    except Exception as exc:
        return {"seeded": False, "message": f"GitHub validation failed: {exc}"}

    auth_store.upsert_github_connection(
        user_id=user_id,
        username=username,
        token=token,
        scopes=scopes,
        repo_owner=repo_owner,
        repo_name=repo_name,
    )

    return {
        "seeded": True,
        "username": username,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
    }


def _seed_ops_state(ops_store: OpsStateStore) -> Dict[str, int]:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    actions = [
        {
            "title": "SRE: Investigate elevated Lambda error rates in checkout-service",
            "status": "Suggested",
        },
        {
            "title": "FinOps: Approve EC2 rightsizing plan for idle staging nodes",
            "status": "Approved",
        },
        {
            "title": "Platform: Rotate expiring IAM access keys and verify least privilege",
            "status": "Executed",
        },
        {
            "title": "Control Center: Validate alert routing for critical incidents",
            "status": "Verified",
        },
    ]

    jobs = [
        {
            "name": "Lambda Activity Digest",
            "prompt": "List all Lambda functions in my current AWS account and summarize their recent activity, invocation patterns, and error trends.",
            "schedule": "hourly",
            "enabled": True,
            "last_status": "SUCCESS",
            "last_run_at": (now - timedelta(minutes=23)).isoformat(),
        },
        {
            "name": "Daily Cost Guardrail Sweep",
            "prompt": "Analyze current AWS costs, detect anomalies by service, and propose top 3 optimization actions with ROI in INR.",
            "schedule": "daily",
            "enabled": True,
            "last_status": "SUCCESS",
            "last_run_at": (now - timedelta(hours=9)).isoformat(),
        },
        {
            "name": "Platform Reliability Review",
            "prompt": "Review ECS, RDS, and CloudWatch health signals and surface high-severity reliability risks for platform engineering.",
            "schedule": "daily",
            "enabled": True,
            "last_status": "SUCCESS",
            "last_run_at": (now - timedelta(hours=4)).isoformat(),
        },
    ]

    runs = [
        {
            "job_name": "Lambda Activity Digest",
            "status": "SUCCESS",
            "insight_count": 4,
            "recommendation_count": 3,
            "ago_minutes": 23,
        },
        {
            "job_name": "Daily Cost Guardrail Sweep",
            "status": "SUCCESS",
            "insight_count": 5,
            "recommendation_count": 4,
            "ago_minutes": 90,
        },
        {
            "job_name": "Platform Reliability Review",
            "status": "SUCCESS",
            "insight_count": 3,
            "recommendation_count": 2,
            "ago_minutes": 240,
        },
    ]

    state_path = ops_store.file_path
    state_path.parent.mkdir(parents=True, exist_ok=True)

    current: Dict[str, Any] = {"actions": [], "jobs": [], "job_runs": []}
    if state_path.exists():
        try:
            current = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            current = {"actions": [], "jobs": [], "job_runs": []}

    current_actions = current.get("actions", []) if isinstance(current, dict) else []
    current_jobs = current.get("jobs", []) if isinstance(current, dict) else []
    current_runs = current.get("job_runs", []) if isinstance(current, dict) else []

    next_id = int(now.timestamp() * 1000)

    action_by_title = {str(item.get("title", "")).strip().lower(): item for item in current_actions}
    for item in actions:
        key = item["title"].strip().lower()
        existing = action_by_title.get(key)
        if existing:
            existing["status"] = item["status"]
            existing["updated_at"] = now_iso
        else:
            next_id += 1
            current_actions.insert(
                0,
                {
                    "id": next_id,
                    "title": item["title"],
                    "status": item["status"],
                    "created_at": now_iso,
                    "updated_at": now_iso,
                },
            )

    job_by_name = {str(item.get("name", "")).strip().lower(): item for item in current_jobs}
    seeded_job_ids: Dict[str, int] = {}
    for item in jobs:
        key = item["name"].strip().lower()
        existing = job_by_name.get(key)
        if existing:
            existing.update(
                {
                    "prompt": item["prompt"],
                    "schedule": item["schedule"],
                    "enabled": item["enabled"],
                    "last_run_at": item["last_run_at"],
                    "last_status": item["last_status"],
                    "updated_at": now_iso,
                }
            )
            seeded_job_ids[item["name"]] = int(existing.get("id"))
        else:
            next_id += 1
            new_job = {
                "id": next_id,
                "name": item["name"],
                "prompt": item["prompt"],
                "schedule": item["schedule"],
                "enabled": item["enabled"],
                "last_run_at": item["last_run_at"],
                "last_status": item["last_status"],
                "created_at": now_iso,
                "updated_at": now_iso,
            }
            current_jobs.insert(0, new_job)
            seeded_job_ids[item["name"]] = int(new_job["id"])

    for run in runs:
        job_name = run["job_name"]
        job_id = seeded_job_ids.get(job_name)
        if not job_id:
            continue

        completed_at = (now - timedelta(minutes=run["ago_minutes"]))
        exists = any(
            (str(r.get("name", "")) == job_name)
            and (str(r.get("completed_at", ""))[:16] == completed_at.isoformat()[:16])
            for r in current_runs
        )
        if exists:
            continue

        current_runs.insert(
            0,
            {
                "job_id": job_id,
                "name": job_name,
                "status": run["status"],
                "summary": f"Demo seeded run for {job_name}",
                "insight_count": run["insight_count"],
                "recommendation_count": run["recommendation_count"],
                "started_at": (completed_at - timedelta(seconds=12)).isoformat(),
                "completed_at": completed_at.isoformat(),
            },
        )

    current_runs = current_runs[:100]

    state = {
        "actions": current_actions,
        "jobs": current_jobs,
        "job_runs": current_runs,
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return {
        "actions": len(current_actions),
        "jobs": len(current_jobs),
        "job_runs": len(current_runs),
    }


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")
    args = _parse_args()

    auth_store = AuthStore()
    ops_store = OpsStateStore()

    user = _get_or_create_user(auth_store, args.email, args.password)
    session_token = auth_store.create_session(user_id=user["id"], ttl_hours=24 * 7)

    aws_result = {"seeded": False, "message": "Skipped by flag"}
    if not args.skip_aws:
        aws_result = _seed_aws_connection(auth_store, user["id"])

    github_result = {"seeded": False, "message": "Skipped by flag"}
    if not args.skip_github:
        github_result = _seed_github_connection(auth_store, user["id"])

    ops_counts = _seed_ops_state(ops_store)

    print("Demo seed completed")
    print(f"Email: {args.email.strip().lower()}")
    print(f"Password: {args.password}")
    print(f"User Created: {user['created']}")
    print(f"Bearer Token: {session_token}")
    print(f"AWS Seeded: {aws_result.get('seeded')} ({aws_result.get('message', 'ok')})")
    print(f"GitHub Seeded: {github_result.get('seeded')} ({github_result.get('message', 'ok')})")
    print(
        "Ops Data: "
        f"actions={ops_counts['actions']}, jobs={ops_counts['jobs']}, job_runs={ops_counts['job_runs']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
