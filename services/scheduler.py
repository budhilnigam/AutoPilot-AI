"""
Scheduler Service

Responsible for:
- Periodic execution of agent tasks (Requirement 10)
- Real-time monitoring loop
- Alert dispatching
"""

import logging
import time
import threading
import schedule
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime

from api.routes import AutoPilotAPI
from services.notification_service import NotificationService
from models.agent_protocol import Severity

logger = logging.getLogger(__name__)


class SchedulerService:
    """
    Background scheduler for AutoPilot AI.
    Runs periodic checks and dispatches alerts.
    """
    
    def __init__(self, api: AutoPilotAPI = None, on_job_complete: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize Scheduler.
        
        Args:
            api: AutoPilot API instance
        """
        self.api = api or AutoPilotAPI()
        self.notification_service = NotificationService()
        self.running = False
        self.worker_thread = None
        self.on_job_complete = on_job_complete
        self._registered_job_ids: set[str] = set()
        
        logger.info("Scheduler Service initialized")

    @staticmethod
    def _job_tag(job_id: int) -> str:
        return f"autonomous-job-{job_id}"
    
    def start(self):
        """Start the scheduler in a background thread"""
        if self.running:
            logger.warning("Scheduler already running")
            return
            
        self.running = True
        
        # Define schedule
        # In production, these would be configurable
        schedule.every(1).minutes.do(self.check_critical_metrics)
        schedule.every(5).minutes.do(self.check_build_status)
        schedule.every(1).hours.do(self.check_cost_anomalies)
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Scheduler started - Monitoring active")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def sync_autonomous_jobs(self, jobs: List[Dict[str, Any]]):
        """Sync scheduler registrations with persisted autonomous jobs."""
        incoming_ids = {str(job.get('id')) for job in jobs if job.get('id') is not None}

        # Remove jobs that no longer exist.
        stale = self._registered_job_ids - incoming_ids
        for job_id in stale:
            schedule.clear(self._job_tag(int(job_id)))
            self._registered_job_ids.discard(job_id)

        # Re-register all incoming jobs to ensure schedule/settings stay up to date.
        for job in jobs:
            self.register_autonomous_job(job)

    def register_autonomous_job(self, job: Dict[str, Any]):
        """Register or update a single autonomous job in scheduler."""
        job_id = job.get('id')
        if job_id is None:
            return

        tag = self._job_tag(job_id)
        schedule.clear(tag)

        if not job.get('enabled', True):
            self._registered_job_ids.add(str(job_id))
            return

        cadence = str(job.get('schedule', 'daily')).lower()
        if cadence == 'hourly':
            schedule.every(1).hours.do(self.execute_autonomous_job, job).tag(tag)
        elif cadence == 'weekly':
            schedule.every(1).weeks.do(self.execute_autonomous_job, job).tag(tag)
        else:
            schedule.every(1).days.do(self.execute_autonomous_job, job).tag(tag)

        self._registered_job_ids.add(str(job_id))

    def remove_autonomous_job(self, job_id: int):
        """Remove a job registration from scheduler."""
        schedule.clear(self._job_tag(job_id))
        self._registered_job_ids.discard(str(job_id))

    def trigger_job_now(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an autonomous job immediately."""
        return self.execute_autonomous_job(job)

    def execute_autonomous_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Execute autonomous job prompt through planner pipeline."""
        started_at = datetime.utcnow()
        job_id = job.get('id')
        prompt = job.get('prompt', '').strip()

        if not prompt:
            result = {
                'job_id': job_id,
                'name': job.get('name', 'Unnamed job'),
                'status': 'FAILED',
                'error': 'Job prompt is empty',
                'started_at': started_at.isoformat(),
                'completed_at': datetime.utcnow().isoformat(),
            }
            if self.on_job_complete:
                self.on_job_complete(result)
            return result

        logger.info(f"Running autonomous job {job_id}: {job.get('name')}")
        try:
            response = self.api.query(prompt, context={'source': 'autonomous_job', 'job_id': job_id})
            completed_at = datetime.utcnow()
            result = {
                'job_id': job_id,
                'name': job.get('name', 'Unnamed job'),
                'status': response.get('status', 'SUCCESS'),
                'summary': response.get('summary') or response.get('response'),
                'insight_count': len(response.get('insights', [])),
                'recommendation_count': len(response.get('recommendations', [])),
                'started_at': started_at.isoformat(),
                'completed_at': completed_at.isoformat(),
            }
            if self.on_job_complete:
                self.on_job_complete(result)
            return result
        except Exception as exc:
            logger.error(f"Autonomous job failed ({job_id}): {exc}")
            result = {
                'job_id': job_id,
                'name': job.get('name', 'Unnamed job'),
                'status': 'FAILED',
                'error': str(exc),
                'started_at': started_at.isoformat(),
                'completed_at': datetime.utcnow().isoformat(),
            }
            if self.on_job_complete:
                self.on_job_complete(result)
            return result
    
    def _run_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)  # Backoff on error
    
    def check_critical_metrics(self):
        """
        Periodic check for critical infrastructure metrics.
        Simulates retrieving recent metrics and analyzing them.
        """
        logger.info("Running scheduled task: check_critical_metrics")
        try:
            # In a real scenario, this would fetch the last 5 minutes of metrics
            # For now, we simulate a check via the Observability Agent
            
            # Use the API to run an analysis task
            # We construct a synthetic query to trigger the agent
            result = self.api.query(
                "Check for critical anomalies in CPU and Memory usage for the last 5 minutes",
                context={'source': 'scheduler'}
            )
            
            self._process_results(result)
            
        except Exception as e:
            logger.error(f"Failed to run check_critical_metrics: {e}")
    
    def check_build_status(self):
        """Periodic check for CI/CD build regressions"""
        logger.info("Running scheduled task: check_build_status")
        try:
            result = self.api.query(
                "Check recent CI/CD builds for regressions or failures",
                context={'source': 'scheduler'}
            )
            self._process_results(result)
        except Exception as e:
            logger.error(f"Failed to run check_build_status: {e}")

    def check_cost_anomalies(self):
        """Periodic check for cost spikes"""
        logger.info("Running scheduled task: check_cost_anomalies")
        try:
            result = self.api.query(
                "Analyze current costs for unexpected spikes",
                context={'source': 'scheduler'}
            )
            self._process_results(result)
        except Exception as e:
            logger.error(f"Failed to run check_cost_anomalies: {e}")

    def _process_results(self, result: Dict[str, Any]):
        """
        Process analysis results and send alerts if needed.
        
        Args:
            result: API response dictionary
        """
        if result.get('status') != 'SUCCESS':
            return
            
        insights = result.get('insights', [])
        
        # Filter for actionable alerts
        alerts = []
        for insight_dict in insights:
            # Convert dict back to object if needed, or handle as dict
            # The API returns dicts for insights
            severity_str = insight_dict.get('severity', 'LOW')
            
            if severity_str in ['HIGH', 'CRITICAL']:
                # Construct a lightweight object for the notification service
                # or modify NotificationService to accept dicts.
                # Here we'll adapt on the fly.
                
                # We need to recreate the Insight object for the notification service
                from models.agent_protocol import Insight, Severity
                
                insight = Insight(
                    summary=insight_dict.get('summary', ''),
                    business_impact=insight_dict.get('business_impact', ''),
                    severity=Severity(severity_str),
                    recommendations=insight_dict.get('recommendations', []),
                    cost_impact_inr=insight_dict.get('cost_impact_inr', 0.0),
                    confidence_score=insight_dict.get('confidence', 0.0)
                )
                alerts.append(insight)
        
        # Send alerts
        for alert in alerts:
            self.notification_service.send_alert(alert)
