"""
services/github_poller.py — Background asyncio loop that polls GitHub repos.

Responsibilities:
  - Poll one or more GitHub repositories on a configurable interval.
  - Detect new commits that touch infrastructure files and dispatch an
    alert-mode Planner task so the system proactively identifies regressions.
  - Detect failed/cancelled CI workflow runs and create alerts.
  - Track last-seen commit SHA per repo in memory so we never re-process the
    same commit twice.

Usage (API layer):
    from autopilot_ai.services.github_poller import github_poller
    github_poller.add_repo("my-org/my-service")
    await github_poller.start()      # call once at API startup
    ...
    await github_poller.stop()       # call at API shutdown

The poller is designed to be resilient: exceptions in a single poll cycle
are logged and swallowed so the background loop keeps running.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import GitHubError, GitHubRateLimitError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.github.client import GitHubClient, CommitInfo, WorkflowRunInfo
from autopilot_ai.models.insights import (
    Insight,
    InsightCategory,
    Urgency,
    Recommendation,
    ImplementationEffort,
)
from autopilot_ai.models.tasks import AgentType, Priority, Task, TaskType

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# How many commits/runs to fetch per poll cycle per repo
_MAX_COMMITS_PER_POLL = 20
_MAX_WORKFLOW_RUNS_PER_POLL = 30

# Back-off multiplier when rate-limited (capped at 5× the normal interval)
_RATE_LIMIT_BACKOFF_FACTOR = 5


class _RepoState:
    """Per-repository bookkeeping for the poller."""

    def __init__(self, repo_full_name: str) -> None:
        self.repo = repo_full_name
        self.last_seen_sha: str | None = None
        # Track run IDs we have already reported to avoid duplicate alerts
        self.reported_run_ids: set[int] = set()
        # Timestamp of the last successful poll (used to bound workflow run queries)
        self.last_polled_at: datetime | None = None


class GitHubPoller:
    """
    Background GitHub polling service.

    Thread-safety: all state is managed within the asyncio event loop.
    """

    def __init__(self) -> None:
        self._repos: dict[str, _RepoState] = {}
        self._client = GitHubClient()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # ── Public API ────────────────────────────────────────────────────────

    def add_repo(self, repo_full_name: str) -> None:
        """Register a repository to poll. Safe to call before start()."""
        if repo_full_name not in self._repos:
            self._repos[repo_full_name] = _RepoState(repo_full_name)
            logger.info("github_poller_repo_added", repo=repo_full_name)

    def remove_repo(self, repo_full_name: str) -> None:
        """Deregister a repository."""
        self._repos.pop(repo_full_name, None)
        logger.info("github_poller_repo_removed", repo=repo_full_name)

    async def start(self) -> None:
        """
        Start the background polling loop.

        Safe to call multiple times — if the loop is already running,
        this is a no-op.
        """
        if self._task and not self._task.done():
            logger.debug("github_poller_already_running")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="github_poller")
        logger.info(
            "github_poller_started",
            interval_s=settings.github_poll_interval_seconds,
            repos=list(self._repos.keys()),
        )

    async def stop(self) -> None:
        """Signal the polling loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("github_poller_stopped")

    @property
    def is_running(self) -> bool:
        return bool(self._task and not self._task.done())

    # ── Background loop ───────────────────────────────────────────────────

    async def _loop(self) -> None:
        interval = settings.github_poll_interval_seconds
        while not self._stop_event.is_set():
            start = asyncio.get_event_loop().time()
            await self._poll_all()
            elapsed = asyncio.get_event_loop().time() - start

            sleep_for = max(0.0, interval - elapsed)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass  # normal — sleep expired, loop again

    async def _poll_all(self) -> None:
        """Run one poll cycle across all registered repos, sequentially."""
        for repo_full_name, state in list(self._repos.items()):
            if self._stop_event.is_set():
                break
            try:
                await self._poll_repo(state)
            except GitHubRateLimitError:
                wait = settings.github_poll_interval_seconds * _RATE_LIMIT_BACKOFF_FACTOR
                logger.warning(
                    "github_poller_rate_limited",
                    repo=repo_full_name,
                    backoff_s=wait,
                )
                # Back off by sleeping the extra time
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
            except Exception as exc:
                logger.error(
                    "github_poller_error",
                    repo=repo_full_name,
                    error=str(exc)[:200],
                    exc_info=True,
                )

    # ── Per-repo poll ─────────────────────────────────────────────────────

    async def _poll_repo(self, state: _RepoState) -> None:
        """
        Fetch new commits and workflow runs for one repo.

        1. Pull commits since last_seen_sha (or last 5 minutes on first run).
        2. For infrastructure-touching commits, dispatch a Planner alert task.
        3. Pull failed workflow runs; create alerts for new failures.
        """
        now = datetime.now(tz=timezone.utc)

        # ── Commits ──────────────────────────────────────────────────────
        since_dt = state.last_polled_at or (now - timedelta(minutes=5))
        commits = await self._client.get_recent_commits(
            state.repo,
            since_sha=state.last_seen_sha,
            since=since_dt,
            max_count=_MAX_COMMITS_PER_POLL,
        )

        if commits:
            # newest-first from GitHub; process oldest-first to track SHA correctly
            for commit in reversed(commits):
                await self._handle_commit(state, commit)
            state.last_seen_sha = commits[0].sha  # track the newest SHA seen

        # ── Workflow runs ─────────────────────────────────────────────────
        runs = await self._client.get_workflow_runs(
            state.repo,
            since=since_dt,
            max_count=_MAX_WORKFLOW_RUNS_PER_POLL,
        )
        for run in runs:
            await self._handle_workflow_run(state, run)

        state.last_polled_at = now
        logger.debug(
            "github_poller_cycle_done",
            repo=state.repo,
            new_commits=len(commits),
            new_runs=len([r for r in runs if r.run_id not in state.reported_run_ids]),
        )

    # ── Event handlers ────────────────────────────────────────────────────

    async def _handle_commit(
        self, state: _RepoState, commit: CommitInfo
    ) -> None:
        """
        If the commit touches infrastructure files, dispatch an alert-mode
        Planner task so agents proactively check for regressions.
        """
        if not commit.is_infrastructure_change:
            return

        logger.info(
            "github_poller_infra_commit",
            repo=state.repo,
            sha=commit.sha[:8],
            message=commit.message[:80],
            files=len(commit.files_changed),
        )

        await self._dispatch_planner_task(
            query=(
                f"Infrastructure files were changed in commit {commit.sha[:8]} "
                f"({commit.message[:120]}) by {commit.author_name}. "
                f"Check for Dockerfile/Terraform/ECS task definition regressions and "
                f"assess whether this change could cause performance or cost issues."
            ),
            context={
                "mode": "alert",
                "commit_sha": commit.sha,
                "repo": state.repo,
                "changed_files": commit.files_changed[:20],
                "author": commit.author_name,
            },
        )

    async def _handle_workflow_run(
        self, state: _RepoState, run: WorkflowRunInfo
    ) -> None:
        """
        Create an alert for failed or cancelled CI workflow runs if we have
        not already reported this run_id.
        """
        if run.run_id in state.reported_run_ids:
            return

        if run.conclusion not in ("failure", "cancelled", "timed_out"):
            return

        state.reported_run_ids.add(run.run_id)
        # Bound the set size to avoid unbounded growth over many days
        if len(state.reported_run_ids) > 10_000:
            # Discard oldest half (sets are unordered; this is a best-effort prune)
            state.reported_run_ids = set(list(state.reported_run_ids)[5_000:])

        urgency = Urgency.HIGH if run.conclusion == "failure" else Urgency.MEDIUM
        title = (
            f"CI {run.conclusion}: {run.workflow_name} on {run.branch} "
            f"(commit {run.commit_sha[:8]})"
        )

        logger.info(
            "github_poller_workflow_failure",
            repo=state.repo,
            run_id=run.run_id,
            workflow=run.workflow_name,
            conclusion=run.conclusion,
            branch=run.branch,
            sha=run.commit_sha[:8],
        )

        insight = Insight(
            category=InsightCategory.PERFORMANCE,  # CICD is tracked under performance
            component=f"ci/{run.workflow_name}",
            title=title,
            business_context=(
                f"The '{run.workflow_name}' pipeline {run.conclusion}d on branch "
                f"'{run.branch}'. This may block deployments and delay feature delivery."
            ),
            urgency=urgency,
            confidence=0.95,
            recommendations=[
                Recommendation(
                    action=f"Investigate the failed workflow run #{run.run_id}",
                    rationale="CI failures block deployments and increase development cycle time.",
                    steps=[
                        f"Open the run at {run.html_url}",
                        "Identify the failing step and review logs",
                        "Fix the root cause or re-run flaky tests",
                    ],
                    expected_benefit="Restore green CI pipeline and unblock deployments.",
                    effort=ImplementationEffort.LOW,
                    related_commit_sha=run.commit_sha,
                )
            ],
            supporting_data={
                "run_id": run.run_id,
                "workflow_name": run.workflow_name,
                "branch": run.branch,
                "commit_sha": run.commit_sha,
                "duration_seconds": run.duration_seconds,
                "html_url": run.html_url,
            },
        )

        # Import lazily to avoid circular imports at module load time
        from autopilot_ai.services.alerting import alerting_service  # noqa: PLC0415
        await alerting_service.create_alert(insight, commit_sha=run.commit_sha)

    # ── Planner dispatch ──────────────────────────────────────────────────

    async def _dispatch_planner_task(
        self,
        query: str,
        context: dict,
    ) -> None:
        """
        Build a PLAN_QUERY task and hand it to the Planner Agent.
        The result is fed to the alerting service so any high-urgency insights
        become live alerts.
        """
        # Lazy import to avoid circular dependency at module load time
        from autopilot_ai.agents.planner import planner  # noqa: PLC0415
        from autopilot_ai.services.alerting import alerting_service  # noqa: PLC0415

        task = Task(
            task_type=TaskType.PLAN_QUERY,
            agent_type=AgentType.PLANNER,
            priority=Priority.HIGH,
            parameters={
                "query": query,
                "context": context,
                "mode": "alert",
            },
        )

        try:
            response = await planner.execute(task)
            # Extract the inner list of AgentResponses from the data field
            query_response_data = response.data.get("query_response", {})
            sub_responses_raw = query_response_data.get("agent_responses", [])

            from autopilot_ai.models.responses import AgentResponse  # noqa: PLC0415
            sub_responses = [
                AgentResponse.model_validate(r) for r in sub_responses_raw
            ]

            await alerting_service.process_responses(
                sub_responses,
                commit_sha=context.get("commit_sha"),
            )
        except Exception as exc:
            logger.error(
                "github_poller_planner_dispatch_failed",
                query=query[:80],
                error=str(exc)[:200],
                exc_info=True,
            )


# Module-level singleton
github_poller = GitHubPoller()
