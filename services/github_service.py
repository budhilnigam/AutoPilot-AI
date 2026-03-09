"""
GitHub Service

High-level wrapper around GitHub client for repository monitoring and CI/CD analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from config import config
from services.github_client import GitHubClient
from models.core_models import BuildData

logger = logging.getLogger(__name__)


class GitHubService:
    """
    GitHub Service for repository monitoring and CI/CD analysis.
    
    Provides convenient methods for:
    - Repository information
    - Workflow run tracking
    - Build trend analysis
    - Commit history analysis
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub Service.
        
        Args:
            token: GitHub personal access token (defaults to env config)
        """
        self.token = token or config.GITHUB_TOKEN
        self.repo_owner = config.GITHUB_REPO_OWNER
        self.repo_name = config.GITHUB_REPO_NAME
        
        self.client = GitHubClient(token=self.token)
        
        if not self.is_configured():
            logger.warning("GitHub service requires GITHUB_TOKEN for private/public repository access.")
        else:
            logger.info("GitHub Service initialized with token-based access")
    
    def is_configured(self) -> bool:
        """Check if GitHub service is properly configured"""
        return bool(self.token)

    def _resolve_repo(self, owner: Optional[str] = None, repo: Optional[str] = None) -> Optional[tuple]:
        """Resolve target repository from explicit args or defaults."""
        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            cleaned = value.strip()
            if not cleaned:
                return None
            placeholders = {"your-github-username", "your-repo-name"}
            if cleaned.lower() in placeholders:
                return None
            return cleaned

        resolved_owner = _clean(owner) or _clean(self.repo_owner)
        resolved_repo = _clean(repo) or _clean(self.repo_name)

        if resolved_owner and resolved_repo:
            return resolved_owner, resolved_repo

        # Fallback: choose a reachable repository when defaults are incomplete.
        if self.is_configured():
            try:
                candidates = self.client._make_request(
                    'GET',
                    '/user/repos',
                    params={'per_page': 50, 'sort': 'updated'}
                )
                if isinstance(candidates, list) and candidates:
                    if resolved_owner:
                        owner_filtered = [
                            item for item in candidates
                            if isinstance(item, dict) and ((item.get('owner') or {}).get('login') == resolved_owner)
                        ]
                        candidates = owner_filtered or candidates

                    selected = candidates[0] if isinstance(candidates[0], dict) else None
                    full_name = (selected or {}).get('full_name')
                    if full_name and '/' in full_name:
                        fallback_owner, fallback_repo = full_name.split('/', 1)
                        logger.info(
                            f"Auto-resolved repository target to {fallback_owner}/{fallback_repo} "
                            "because owner/repo defaults were incomplete"
                        )
                        return fallback_owner, fallback_repo
            except Exception as e:
                logger.warning(f"Automatic repository resolution failed: {e}")

        if not resolved_owner or not resolved_repo:
            return None

        return resolved_owner, resolved_repo

    def list_accessible_repositories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List repositories accessible by the configured token (includes private repos if authorized).

        Args:
            limit: Maximum repositories to return

        Returns:
            List of repository summaries
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return []

        try:
            repos = self.client._make_request(
                'GET',
                '/user/repos',
                params={'per_page': min(limit, 100), 'sort': 'updated'}
            )
            summaries = [
                {
                    'name': r.get('name'),
                    'full_name': r.get('full_name'),
                    'private': r.get('private', False),
                    'default_branch': r.get('default_branch'),
                    'updated_at': r.get('updated_at'),
                    'stargazers_count': r.get('stargazers_count', 0),
                    'forks_count': r.get('forks_count', 0),
                    'language': r.get('language'),
                }
                for r in repos[:limit]
            ]
            logger.info(f"Retrieved {len(summaries)} accessible repositories")
            return summaries
        except Exception as e:
            logger.error(f"Failed to list accessible repositories: {e}")
            return []
    
    def get_repository_info(self, owner: Optional[str] = None, repo: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get current repository information.
        
        Returns:
            Repository metadata or None
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return None

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return None
        repo_owner, repo_name = target
        
        try:
            repo_info = self.client.get_repository_info(repo_owner, repo_name)
            logger.info(f"Retrieved repository info for {repo_owner}/{repo_name}")
            return repo_info
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return None
    
    def get_recent_builds(self, limit: int = 10, owner: Optional[str] = None, repo: Optional[str] = None) -> List[BuildData]:
        """
        Get recent workflow runs and convert them to BuildData.
        
        Args:
            limit: Maximum number of builds to retrieve
            
        Returns:
            List of BuildData objects
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return []

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return []
        repo_owner, repo_name = target
        
        try:
            runs = self.client.get_workflow_runs(repo_owner, repo_name, limit=limit)
            build_list = []
            
            for run in runs:
                build = self.client.get_build_data(repo_owner, repo_name, run['id'])
                if build:
                    build_list.append(build)
            
            logger.info(f"Retrieved {len(build_list)} recent builds")
            return build_list
        except Exception as e:
            logger.error(f"Failed to get recent builds: {e}")
            return []
    
    def get_build_trends(self, days: int = 7, owner: Optional[str] = None, repo: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze build trends over a time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Build trend analysis
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return {}

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return {'status': 'error', 'message': 'Repository target missing. Provide owner/repo.'}
        repo_owner, repo_name = target
        
        try:
            trends = self.client.analyze_build_trends(repo_owner, repo_name)
            return {
                'status': 'success',
                'build_analysis': trends,
                'analysis_date': datetime.utcnow().isoformat(),
                'period_days': days,
                'repository': f"{repo_owner}/{repo_name}",
            }
        except Exception as e:
            logger.error(f"Failed to analyze build trends: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_failed_builds(self, limit: int = 5, owner: Optional[str] = None, repo: Optional[str] = None) -> List[BuildData]:
        """
        Get failed workflow runs.
        
        Args:
            limit: Maximum number of failed builds to retrieve
            
        Returns:
            List of failed BuildData objects
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return []

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return []
        repo_owner, repo_name = target
        
        try:
            runs = self.client.get_workflow_runs(repo_owner, repo_name, limit=limit*2)
            failed_builds = []
            
            for run in runs:
                if run['conclusion'] in ['failure', 'timed_out', 'cancelled']:
                    build = self.client.get_build_data(repo_owner, repo_name, run['id'])
                    if build:
                        failed_builds.append(build)
                    if len(failed_builds) >= limit:
                        break
            
            logger.info(f"Retrieved {len(failed_builds)} failed builds")
            return failed_builds
        except Exception as e:
            logger.error(f"Failed to get failed builds: {e}")
            return []
    
    def get_commit_history(
        self,
        branch: str = 'main',
        limit: int = 20,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get commit history for a branch.
        
        Args:
            branch: Branch name
            limit: Maximum commits to retrieve
            
        Returns:
            List of commit data
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return []

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return []
        repo_owner, repo_name = target
        
        try:
            commits = self.client.get_commit_history(
                repo_owner,
                repo_name,
                branch=branch,
                limit=limit
            )
            logger.info(f"Retrieved {len(commits)} commits from {repo_owner}/{repo_name}/{branch}")
            return commits
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            return []
    
    def get_workflow_runs(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 10,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get workflow runs for the repository.
        
        Args:
            workflow_id: Optional specific workflow ID
            limit: Maximum runs to retrieve
            
        Returns:
            List of workflow runs
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return []

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return []
        repo_owner, repo_name = target
        
        try:
            runs = self.client.get_workflow_runs(
                repo_owner,
                repo_name,
                workflow_id=workflow_id,
                limit=limit
            )
            logger.info(f"Retrieved {len(runs)} workflow runs for {repo_owner}/{repo_name}")
            return runs
        except Exception as e:
            logger.error(f"Failed to get workflow runs: {e}")
            return []
    
    def get_build_health_summary(self, owner: Optional[str] = None, repo: Optional[str] = None) -> Dict[str, Any]:
        """
        Get overall build health summary.
        
        Returns:
            Build health metrics
        """
        if not self.is_configured():
            logger.warning("GitHub token is not configured")
            return {'status': 'unconfigured'}

        target = self._resolve_repo(owner=owner, repo=repo)
        if not target:
            logger.warning("Repository target missing. Provide owner/repo in query or set defaults in .env")
            return {
                'status': 'error',
                'message': 'Repository target missing. Provide owner/repo.',
            }
        repo_owner, repo_name = target
        
        try:
            recent_builds = self.get_recent_builds(limit=30, owner=repo_owner, repo=repo_name)
            
            if not recent_builds:
                return {
                    'status': 'unknown',
                    'total_builds': 0,
                    'health': 'N/A',
                    'repository': f"{repo_owner}/{repo_name}"
                }
            
            success_count = sum(1 for b in recent_builds if b.status == 'success')
            failed_count = sum(1 for b in recent_builds if b.status in ['failure', 'failed'])
            total_count = len(recent_builds)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            avg_build_time = sum(b.build_time_seconds for b in recent_builds) / len(recent_builds)
            
            return {
                'status': 'success',
                'repository': f"{repo_owner}/{repo_name}",
                'total_builds': total_count,
                'successful_builds': success_count,
                'failed_builds': failed_count,
                'success_rate_percent': round(success_rate, 2),
                'average_build_time_seconds': round(avg_build_time, 2),
                'health': 'healthy' if success_rate >= 90 else 'degraded' if success_rate >= 70 else 'poor',
                'last_analyzed': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get build health summary: {e}")
            return {'status': 'error', 'message': str(e)}
