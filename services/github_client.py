"""
GitHub Client

Interacts with GitHub API for repository data, CI/CD metrics, and commit history.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

from models.core_models import BuildData

logger = logging.getLogger(__name__)


class GitHubClient:
    """Client for GitHub API"""
    
    def __init__(self, token: str = None):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token (defaults to env GITHUB_TOKEN)
        """
        self.token = token or os.getenv('GITHUB_TOKEN')
        if not self.token:
            logger.warning("GitHub token not provided. API rate limits will be restrictive.")
        
        self.base_url = "https://api.github.com"
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
        }
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
        
        logger.info("GitHub client initialized")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make HTTP request to GitHub API"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                timeout=30,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            raise
    
    def get_workflow_runs(
        self,
        owner: str,
        repo: str,
        workflow_id: Optional[str] = None,
        branch: Optional[str] = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get GitHub Actions workflow runs.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Optional specific workflow
            branch: Optional branch filter
            limit: Max results
            
        Returns:
            List of workflow runs
        """
        endpoint = f"repos/{owner}/{repo}/actions/runs"
        params = {'per_page': min(limit, 100)}
        
        if branch:
            params['branch'] = branch
        
        response = self._make_request('GET', endpoint, params=params)
        runs = response.get('workflow_runs', [])
        
        logger.info(f"Retrieved {len(runs)} workflow runs for {owner}/{repo}")
        return runs[:limit]
    
    def get_build_data(
        self,
        owner: str,
        repo: str,
        run_id: int
    ) -> Optional[BuildData]:
        """
        Get detailed build data for a workflow run.
        
        Args:
            owner: Repository owner
            repo: Repository name
            run_id: Workflow run ID
            
        Returns:
            BuildData object or None
        """
        try:
            # Get run details
            endpoint = f"repos/{owner}/{repo}/actions/runs/{run_id}"
            run = self._make_request('GET', endpoint)
            
            # Get job details
            jobs_endpoint = f"repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
            jobs_response = self._make_request('GET', jobs_endpoint)
            jobs = jobs_response.get('jobs', [])
            
            # Calculate total build time
            created_at = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
            updated_at = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
            build_time = (updated_at - created_at).total_seconds()
            
            # Extract steps from jobs
            steps = []
            for job in jobs:
                for step in job.get('steps', []):
                    steps.append({
                        'name': step['name'],
                        'status': step['status'],
                        'conclusion': step.get('conclusion'),
                    })
            
            build_data = BuildData(
                build_id=str(run_id),
                commit_sha=run['head_sha'],
                build_time_seconds=build_time,
                status=run['conclusion'] or run['status'],
                timestamp=run['created_at'],
                repository=f"{owner}/{repo}",
                branch=run['head_branch'],
                triggering_event=run['event'],
                steps=steps
            )
            
            return build_data
            
        except Exception as e:
            logger.error(f"Failed to get build data: {e}")
            return None
    
    def get_commit_history(
        self,
        owner: str,
        repo: str,
        branch: str = 'main',
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get commit history.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            limit: Max results
            
        Returns:
            List of commits
        """
        endpoint = f"repos/{owner}/{repo}/commits"
        params = {
            'sha': branch,
            'per_page': min(limit, 100)
        }
        
        commits = self._make_request('GET', endpoint, params=params)
        logger.info(f"Retrieved {len(commits)} commits for {owner}/{repo}/{branch}")
        return commits[:limit]
    
    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository data
        """
        endpoint = f"repos/{owner}/{repo}"
        return self._make_request('GET', endpoint)
    
    def analyze_build_trends(
        self,
        owner: str,
        repo: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze build time trends over recent runs.
        
        Args:
            owner: Repository owner
            repo: Repository name
            limit: Number of runs to analyze
            
        Returns:
            Trend analysis data
        """
        runs = self.get_workflow_runs(owner, repo, limit=limit)
        
        build_times = []
        successful_runs = 0
        failed_runs = 0
        
        for run in runs:
            created = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
            updated = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
            duration = (updated - created).total_seconds()
            
            build_times.append(duration)
            
            if run['conclusion'] == 'success':
                successful_runs += 1
            elif run['conclusion'] == 'failure':
                failed_runs += 1
        
        avg_build_time = sum(build_times) / len(build_times) if build_times else 0
        max_build_time = max(build_times) if build_times else 0
        min_build_time = min(build_times) if build_times else 0
        
        return {
            'total_runs': len(runs),
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': successful_runs / len(runs) if runs else 0,
            'avg_build_time_seconds': avg_build_time,
            'max_build_time_seconds': max_build_time,
            'min_build_time_seconds': min_build_time,
            'build_times': build_times,
        }
