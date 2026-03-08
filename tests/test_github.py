"""
test_github.py — Quick connectivity test for GitHub API.

Tests each method of GitHubClient in isolation to verify:
  1. Token is loaded correctly from .env
  2. PyGithub can reach api.github.com
  3. Commits, workflow runs, PRs, and repo info can be fetched

Run with:  python test_github.py
Set GITHUB_MONITORED_REPOS in .env (or the fallback below) before running.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Load .env before importing anything from autopilot_ai ─────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ── Config: override here if not in .env ──────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT", "")
# Fallback repo if GITHUB_MONITORED_REPOS is not set
FALLBACK_REPO = "octocat/Hello-World"   # public repo, works without a token
RAW_REPOS = os.getenv("GITHUB_MONITORED_REPOS", "")
TEST_REPO = RAW_REPOS.split(",")[0].strip() if RAW_REPOS.strip() else FALLBACK_REPO


def _sep(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print('─' * 60)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


# ── 1. Token sanity check ──────────────────────────────────────────────────

_sep("1. Token & Config")
if GITHUB_TOKEN:
    masked = GITHUB_TOKEN[:4] + "*" * (len(GITHUB_TOKEN) - 6) + GITHUB_TOKEN[-2:]
    _ok(f"GITHUB_TOKEN found → {masked}")
else:
    _fail("GITHUB_TOKEN not set — unauthenticated calls (60 req/hr limit)")
    _info("Add GITHUB_TOKEN=ghp_... to your .env file")

_info(f"Testing against repo: {TEST_REPO}")
if TEST_REPO == FALLBACK_REPO:
    _info("(using public fallback repo — set GITHUB_MONITORED_REPOS in .env for your own repo)")


# ── 2. Raw PyGithub connectivity ───────────────────────────────────────────

_sep("2. Raw PyGithub connectivity")
try:
    from github import Github, GithubException, Auth
    auth = Auth.Token(GITHUB_TOKEN) if GITHUB_TOKEN else None
    gh = Github(auth=auth) if auth else Github()

    # Rate limit check — attribute changed in PyGithub 2.x
    rate = gh.get_rate_limit()
    core = getattr(rate, 'core', None) or getattr(rate, 'rate', None)
    if core:
        _ok(f"Connected — rate limit: {core.remaining}/{core.limit} remaining")
        if core.remaining < 10:
            _fail("Rate limit nearly exhausted — results may be incomplete")
    else:
        _ok("Connected — (rate limit info unavailable in this PyGithub version)")

    # Authenticated user (only if token provided)
    if GITHUB_TOKEN:
        try:
            user = gh.get_user()
            _ok(f"Authenticated as: {user.login} ({user.name or 'no display name'})")
        except GithubException as e:
            _fail(f"get_user() failed → {e.status}: {e.data}")

except Exception as e:
    _fail(f"PyGithub import or connection failed → {type(e).__name__}: {e}")
    sys.exit(1)


# ── 3. Repo access ─────────────────────────────────────────────────────────

_sep(f"3. Repo access: {TEST_REPO}")
try:
    repo = gh.get_repo(TEST_REPO)
    _ok(f"Repo found: {repo.full_name}")
    _info(f"  Description : {repo.description or '(none)'}")
    _info(f"  Default branch: {repo.default_branch}")
    _info(f"  Stars: {repo.stargazers_count}   Forks: {repo.forks_count}")
    _info(f"  Open issues: {repo.open_issues_count}")
    _info(f"  Last pushed: {repo.pushed_at}")
except Exception as e:
    _fail(f"Could not access repo → {type(e).__name__}: {e}")


# ── 4. Recent commits (last 7 days) ───────────────────────────────────────

_sep("4. Recent commits (last 7 days, max 5)")
try:
    since = datetime.now(timezone.utc) - timedelta(days=7)
    commits = repo.get_commits(since=since)
    count = 0
    for c in commits:
        if count >= 5:
            break
        msg = c.commit.message.split("\n")[0][:80]
        ts = c.commit.author.date
        _ok(f"{c.sha[:7]}  {ts.strftime('%Y-%m-%d %H:%M')}  {msg}")
        count += 1

    if count == 0:
        _info("No commits in the last 7 days — try a more active repo or widen the window")
    else:
        _info(f"Fetched {count} commit(s)")

except Exception as e:
    _fail(f"get_commits() failed → {type(e).__name__}: {e}")


# ── 5. Pull requests ───────────────────────────────────────────────────────

_sep("5. Pull requests (open, max 5)")
try:
    pulls = repo.get_pulls(state="open", sort="updated", direction="desc")
    count = 0
    for pr in pulls:
        if count >= 5:
            break
        _ok(f"#{pr.number}  [{pr.state}]  {pr.title[:70]}")
        _info(f"   Author: {pr.user.login}   Updated: {pr.updated_at}")
        count += 1

    if count == 0:
        _info("No open PRs found — trying merged PRs...")
        merged = repo.get_pulls(state="closed", sort="updated", direction="desc")
        for pr in merged:
            if count >= 3:
                break
            _ok(f"#{pr.number}  [closed]  {pr.title[:70]}")
            count += 1

    _info(f"Fetched {count} PR(s)")

except Exception as e:
    _fail(f"get_pulls() failed → {type(e).__name__}: {e}")


# ── 6. GitHub Actions workflow runs ───────────────────────────────────────

_sep("6. Workflow runs (last 30 days, max 5)")
try:
    since_30 = datetime.now(timezone.utc) - timedelta(days=30)
    workflows = repo.get_workflows()
    run_count = 0
    wf_count = 0

    for wf in workflows:
        wf_count += 1
        if run_count >= 5:
            break
        runs = wf.get_runs()
        for run in runs:
            if run_count >= 5:
                break
            started = run.run_started_at or run.created_at
            if started and started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if started and started < since_30:
                break
            conclusion = run.conclusion or run.status or "in_progress"
            _ok(f"[{wf.name}]  run #{run.id}  {conclusion.upper()}  {started.strftime('%Y-%m-%d %H:%M') if started else '?'}")
            run_count += 1

    if wf_count == 0:
        _info("No workflows found in this repo (no .github/workflows/ files?)")
    elif run_count == 0:
        _info("Workflows found but no runs in the last 30 days")
    else:
        _info(f"Fetched {run_count} workflow run(s) across {wf_count} workflow(s)")

except Exception as e:
    _fail(f"get_workflows() / get_runs() failed → {type(e).__name__}: {e}")


# ── 7. Full GitHubClient (via autopilot_ai) ────────────────────────────────

_sep("7. GitHubClient wrapper (autopilot_ai integration)")
try:
    from autopilot_ai.integrations.github.client import GitHubClient

    client = GitHubClient(token=GITHUB_TOKEN or None)

    async def _check_client() -> None:
        # get_recent_commits
        try:
            since = datetime.now(timezone.utc) - timedelta(days=7)
            commits = await client.get_recent_commits(TEST_REPO, since=since, max_count=3)
            _ok(f"get_recent_commits() → {len(commits)} result(s)")
            for c in commits:
                _info(f"  {c.sha[:7]}  {c.timestamp.strftime('%Y-%m-%d')}  {c.message[:60]}")
        except Exception as e:
            _fail(f"get_recent_commits() → {type(e).__name__}: {e}")

        # get_workflow_runs
        try:
            since = datetime.now(timezone.utc) - timedelta(days=30)
            runs = await client.get_workflow_runs(TEST_REPO, since=since, max_count=3)
            _ok(f"get_workflow_runs() → {len(runs)} result(s)")
            for r in runs:
                _info(f"  #{r.run_id}  {r.workflow_name}  {r.conclusion or r.status}  {r.started_at.strftime('%Y-%m-%d')}")
        except Exception as e:
            _fail(f"get_workflow_runs() → {type(e).__name__}: {e}")

        # get_pull_requests
        try:
            prs = await client.get_pull_requests(TEST_REPO, state="open", max_count=3)
            _ok(f"get_pull_requests() → {len(prs)} result(s)")
            for pr in prs:
                _info(f"  #{pr.get('number')}  {str(pr.get('title', ''))[:60]}")
        except Exception as e:
            _fail(f"get_pull_requests() → {type(e).__name__}: {e}")

        # get_repo_info
        try:
            info = await client.get_repo_info(TEST_REPO)
            _ok(f"get_repo_info() → {info.get('name')}  ★{info.get('stars')}  forks:{info.get('forks')}")
            _info(f"  Languages: {list(info.get('languages', {}).keys())}")
            _info(f"  Visibility: {info.get('visibility')}   Archived: {info.get('archived')}")
        except Exception as e:
            _fail(f"get_repo_info() → {type(e).__name__}: {e}")

        # get_user_repos
        try:
            user_repos = await client.get_user_repos(max_count=10)
            _ok(f"get_user_repos() → {len(user_repos)} result(s) (showing first 5)")
            for r in user_repos[:5]:
                vis = "private" if r.get("visibility") == "private" else "public"
                _info(f"  [{vis}] {r['name']}  ★{r['stars']}  pushed:{(r.get('pushed_at') or '')[:10]}")
        except Exception as e:
            _fail(f"get_user_repos() → {type(e).__name__}: {e}")

    asyncio.run(_check_client())

except Exception as e:
    _fail(f"GitHubClient import failed → {type(e).__name__}: {e}")


# ── Summary ────────────────────────────────────────────────────────────────

_sep("Done")
print()
print("  If everything shows [OK], the GitHub integration is working.")
print("  If you see [FAIL] on raw connectivity, check your GITHUB_TOKEN in .env.")
print("  If raw PyGithub works but GitHubClient fails, check autopilot_ai imports.")
print()
