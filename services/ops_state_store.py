"""
Operational state persistence for Action Center and Autonomous Jobs.

Stores lightweight state in a local JSON file so the UI can survive refreshes
and backend restarts without introducing a new database dependency.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OpsStateStore:
    """Thread-safe JSON-backed store for actions, jobs, and job runs."""

    def __init__(self, file_path: str = "./local_kb/configs/ops_state.json"):
        self.file_path = Path(file_path)
        self.lock = threading.Lock()
        self._state = {
            "actions": [],
            "jobs": [],
            "job_runs": [],
        }
        self._ensure_file()
        self._load()

    def _ensure_file(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_state(self._state)

    def _load(self) -> None:
        with self.lock:
            try:
                with self.file_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self._state["actions"] = data.get("actions", [])
                    self._state["jobs"] = data.get("jobs", [])
                    self._state["job_runs"] = data.get("job_runs", [])
            except Exception as exc:
                logger.warning(f"Failed to load ops state; using defaults: {exc}")
                self._state = {"actions": [], "jobs": [], "job_runs": []}
                self._write_state(self._state)

    def _write_state(self, state: Dict[str, Any]) -> None:
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def _persist(self) -> None:
        self._write_state(self._state)

    def get_actions(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self._state["actions"])

    def create_action(self, title: str) -> Dict[str, Any]:
        with self.lock:
            action = {
                "id": int(datetime.utcnow().timestamp() * 1000),
                "title": title,
                "status": "Suggested",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            self._state["actions"].insert(0, action)
            self._persist()
            return action

    def update_action(self, action_id: int, status: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            for action in self._state["actions"]:
                if action.get("id") == action_id:
                    action["status"] = status
                    action["updated_at"] = datetime.utcnow().isoformat()
                    self._persist()
                    return action
            return None

    def get_jobs(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self._state["jobs"])

    def create_job(self, name: str, prompt: str, schedule: str) -> Dict[str, Any]:
        with self.lock:
            job = {
                "id": int(datetime.utcnow().timestamp() * 1000),
                "name": name,
                "prompt": prompt,
                "schedule": schedule,
                "enabled": True,
                "last_run_at": None,
                "last_status": None,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            self._state["jobs"].insert(0, job)
            self._persist()
            return job

    def update_job(self, job_id: int, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self.lock:
            for job in self._state["jobs"]:
                if job.get("id") == job_id:
                    for key, value in fields.items():
                        if key in {"name", "prompt", "schedule", "enabled", "last_run_at", "last_status"}:
                            job[key] = value
                    job["updated_at"] = datetime.utcnow().isoformat()
                    self._persist()
                    return job
            return None

    def append_job_run(self, run: Dict[str, Any]) -> None:
        with self.lock:
            self._state["job_runs"].insert(0, run)
            self._state["job_runs"] = self._state["job_runs"][:100]
            self._persist()

    def get_job_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self._state["job_runs"][:max(1, min(limit, 100))])
