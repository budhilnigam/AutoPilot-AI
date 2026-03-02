"""
integrations/github/client.py — GitHub API client using PyGithub.

Provides:
  get_recent_commits(repo, since_sha, since)  -> list[CommitInfo]
  get_workflow_runs(repo, since, branch)       -> list[WorkflowRunInfo]
  get_commit_diff(repo, sha)                   -> CommitDiff
  get_file_content(repo, path, ref)            -> str
  list_infrastructure_files(repo, ref)         -> list[str]

All methods are async via run_in_executor (PyGithub is synchronous).
Raises GitHubError on API errors, GitHubRateLimitError on rate-limit exhaustion.

Returns typed Pydantic models — agents never interact with raw PyGithub objects.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import partial
from typing import Any

from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository as GHRepo
from pydantic import BaseModel, Field

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import GitHubError, GitHubRateLimitError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.models.domain import BuildData, BuildStatus

logger = get_logger(__name__)

# File extensions / names that count as infrastructure-impacting (Property 32)
_INFRA_FILE_PATTERNS = frozenset(
    [
        "dockerfile",
        "docker-compose",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".tf",             # Terraform
        ".tfvars",
        "taskdef.json",
        "task-definition.json",
        ".github/workflows",
    ]
)


# ── Response models ────────────────────────────────────────────────────────


class CommitInfo(BaseModel):
    """Typed representation of a GitHub commit."""

    sha: str = Field(min_length=7)
    message: str
    author_name: str
    author_email: str
    timestamp: datetime
    files_changed: list[str] = Field(default_factory=list)
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)
    is_infrastructure_change: bool = False


class FileDiff(BaseModel):
    """Changed file with diff patch."""

    filename: str
    status: str  # "added", "modified", "removed", "renamed"
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str | None = None  # unified diff text; None for binary files


class CommitDiff(BaseModel):
    """Full diff for a commit — used for infrastructure drift detection."""

    sha: str
    files: list[FileDiff] = Field(default_factory=list)
    infrastructure_files: list[str] = Field(
        default_factory=list,
        description="Subset of changed files that are infrastructure-related (Property 32)",
    )


class WorkflowRunInfo(BaseModel):
    """
    Typed representation of a GitHub Actions workflow run.
    Maps directly to BuildData for CICD Agent use.
    Validates Property 33.
    """

    run_id: int
    workflow_name: str
    commit_sha: str
    branch: str
    status: str
    conclusion: str | None
    started_at: datetime
    updated_at: datetime
    duration_seconds: float = Field(ge=0.0)
    html_url: str


# ── Client class ───────────────────────────────────────────────────────────


class GitHubClient:
    """
    Async wrapper around PyGithub.

    Usage:
        client = GitHubClient()
        commits = await client.get_recent_commits("org/repo", since_sha="abc123")
    """

    def __init__(self, token: str | None = None) -> None:
        tok = token or settings.github_token
        if not tok:
            logger.warning("github_client_no_token — GitHub API rate limits apply")
        self._gh = Github(tok) if tok else Github()

    # ── Private rate-limit helper ──────────────────────────────────────────

    def _wrap_github_call(self, func, *args, **kwargs):
        """
        Execute a PyGithub call and translate exceptions to our hierarchy.
        Call this inside run_in_executor.
        """
        try:
            return func(*args, **kwargs)
        except RateLimitExceededException as e:
            raise GitHubRateLimitError(
                "GitHub API rate limit exhausted. Poller will back off."
            ) from e
        except GithubException as e:
            raise GitHubError(
                f"GitHub API error {e.status}: {e.data}",
                status=e.status,
            ) from e

    # ── Sync helpers (run in executor) ─────────────────────────────────────

    def _get_repo_sync(self, repo_full_name: str) -> GHRepo:
        return self._wrap_github_call(self._gh.get_repo, repo_full_name)

    def _get_recent_commits_sync(
        self,
        repo_full_name: str,
        since_sha: str | None,
        since: datetime | None,
        max_count: int,
    ) -> list[CommitInfo]:
        repo = self._get_repo_sync(repo_full_name)

        kwargs: dict[str, Any] = {}
        if since:
            kwargs["since"] = since

        commits = self._wrap_github_call(repo.get_commits, **kwargs)

        results: list[CommitInfo] = []
        for commit in commits:
            # Stop when we reach the already-seen SHA
            if since_sha and commit.sha == since_sha:
                break
            if len(results) >= max_count:
                break

            files_changed = []
            additions = 0
            deletions = 0
            is_infra = False

            # Fetch file stats — requires an extra API call per commit
            try:
                detailed = self._wrap_github_call(repo.get_commit, commit.sha)
                for f in detailed.files:
                    files_changed.append(f.filename)
                    additions += f.additions or 0
                    deletions += f.deletions or 0
                    lower = f.filename.lower()
                    if any(pat in lower for pat in _INFRA_FILE_PATTERNS):
                        is_infra = True
            except (GitHubError, GitHubRateLimitError):
                pass  # partial data is acceptable

            ts = commit.commit.author.date
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            results.append(
                CommitInfo(
                    sha=commit.sha,
                    message=commit.commit.message.split("\n")[0][:200],
                    author_name=commit.commit.author.name or "",
                    author_email=commit.commit.author.email or "",
                    timestamp=ts,
                    files_changed=files_changed,
                    additions=additions,
                    deletions=deletions,
                    is_infrastructure_change=is_infra,
                )
            )

        return results

    def _get_workflow_runs_sync(
        self,
        repo_full_name: str,
        since: datetime | None,
        branch: str | None,
        max_count: int,
    ) -> list[WorkflowRunInfo]:
        repo = self._get_repo_sync(repo_full_name)
        kwargs: dict[str, Any] = {}
        if branch:
            kwargs["branch"] = branch

        workflows = self._wrap_github_call(repo.get_workflows)
        results: list[WorkflowRunInfo] = []

        for workflow in workflows:
            if len(results) >= max_count:
                break
            run_kwargs: dict[str, Any] = {}
            if branch:
                run_kwargs["branch"] = branch
            runs = self._wrap_github_call(workflow.get_runs, **run_kwargs)

            for run in runs:
                if len(results) >= max_count:
                    break

                started = run.run_started_at or run.created_at
                updated = run.updated_at or started

                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)

                if since and started < since:
                    break  # runs are ordered newest-first

                duration = max((updated - started).total_seconds(), 0.0)

                results.append(
                    WorkflowRunInfo(
                        run_id=run.id,
                        workflow_name=workflow.name,
                        commit_sha=run.head_sha,
                        branch=run.head_branch or "unknown",
                        status=run.status or "unknown",
                        conclusion=run.conclusion,
                        started_at=started,
                        updated_at=updated,
                        duration_seconds=duration,
                        html_url=run.html_url,
                    )
                )

        return results

    def _get_commit_diff_sync(
        self, repo_full_name: str, sha: str
    ) -> CommitDiff:
        repo = self._get_repo_sync(repo_full_name)
        commit = self._wrap_github_call(repo.get_commit, sha)

        file_diffs: list[FileDiff] = []
        infra_files: list[str] = []

        for f in commit.files:
            fd = FileDiff(
                filename=f.filename,
                status=f.status,
                additions=f.additions or 0,
                deletions=f.deletions or 0,
                patch=f.patch,  # None for binary / large files
            )
            file_diffs.append(fd)
            lower = f.filename.lower()
            if any(pat in lower for pat in _INFRA_FILE_PATTERNS):
                infra_files.append(f.filename)

        return CommitDiff(
            sha=sha,
            files=file_diffs,
            infrastructure_files=infra_files,
        )

    def _get_file_content_sync(
        self, repo_full_name: str, path: str, ref: str
    ) -> str:
        repo = self._get_repo_sync(repo_full_name)
        contents = self._wrap_github_call(repo.get_contents, path, ref=ref)
        if isinstance(contents, list):
            raise GitHubError(
                f"Path '{path}' is a directory, not a file",
                repo=repo_full_name,
                path=path,
            )
        return contents.decoded_content.decode("utf-8", errors="replace")

    # ── Public async API ───────────────────────────────────────────────────

    @with_retry(retry_on=(GitHubRateLimitError,))
    async def get_recent_commits(
        self,
        repo_full_name: str,
        since_sha: str | None = None,
        since: datetime | None = None,
        max_count: int = 50,
    ) -> list[CommitInfo]:
        """
        Fetch commits newer than `since_sha` or `since` datetime.

        Used by the GitHub Poller to detect new pushes.
        Validates Property 32 (infrastructure change detection).

        Args:
            repo_full_name: "owner/repo" format.
            since_sha:      Stop fetching when this SHA is encountered.
            since:          Fetch commits after this datetime.
            max_count:      Hard cap on number of commits fetched.

        Returns:
            List of CommitInfo, newest first.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._get_recent_commits_sync,
                repo_full_name,
                since_sha,
                since,
                max_count,
            ),
        )

    @with_retry(retry_on=(GitHubRateLimitError,))
    async def get_workflow_runs(
        self,
        repo_full_name: str,
        since: datetime | None = None,
        branch: str | None = None,
        max_count: int = 100,
    ) -> list[WorkflowRunInfo]:
        """
        Fetch GitHub Actions workflow runs.

        Validates Property 33 (workflow execution tracking).

        Returns:
            List of WorkflowRunInfo for use by the CICD Agent.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._get_workflow_runs_sync,
                repo_full_name,
                since,
                branch,
                max_count,
            ),
        )

    @with_retry(retry_on=(GitHubRateLimitError,))
    async def get_commit_diff(
        self, repo_full_name: str, sha: str
    ) -> CommitDiff:
        """
        Get the full file diff for a commit.

        Validates Property 32 (infrastructure change analysis).

        Returns:
            CommitDiff with per-file patches and infrastructure_files list.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._get_commit_diff_sync, repo_full_name, sha),
        )

    @with_retry(retry_on=(GitHubRateLimitError,))
    async def get_file_content(
        self, repo_full_name: str, path: str, ref: str = "HEAD"
    ) -> str:
        """
        Fetch the raw text content of a file at a given ref (branch/SHA/tag).

        Used to read Dockerfiles, Terraform files, GitHub Actions YAML, etc.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._get_file_content_sync, repo_full_name, path, ref),
        )

    def workflow_run_to_build_data(self, run: WorkflowRunInfo) -> BuildData:
        """Convert a WorkflowRunInfo into the BuildData model used by CICD Agent."""
        status_map = {
            "success": BuildStatus.SUCCESS,
            "failure": BuildStatus.FAILURE,
            "cancelled": BuildStatus.CANCELLED,
            "in_progress": BuildStatus.IN_PROGRESS,
        }
        status = status_map.get(
            (run.conclusion or run.status).lower(), BuildStatus.PENDING
        )
        return BuildData(
            workflow_run_id=run.run_id,
            commit_sha=run.commit_sha,
            workflow_name=run.workflow_name,
            branch=run.branch,
            duration_seconds=run.duration_seconds,
            status=status,
            started_at=run.started_at,
            finished_at=run.updated_at,
        )


# Module-level singleton
github_client = GitHubClient()
