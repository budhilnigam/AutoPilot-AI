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
            logger.warning("GitHub service not fully configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME.")
        else:
            logger.info(f"GitHub Service initialized for {self.repo_owner}/{self.repo_name}")
    
    def is_configured(self) -> bool:
        """Check if GitHub service is properly configured"""
        return bool(self.token and self.repo_owner and self.repo_name)
    
    def get_repository_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current repository information.
        
        Returns:
            Repository metadata or None
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return None
        
        try:
            repo_info = self.client.get_repository_info(self.repo_owner, self.repo_name)
            logger.info(f"Retrieved repository info for {self.repo_owner}/{self.repo_name}")
            return repo_info
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return None
    
    def get_recent_builds(self, limit: int = 10) -> List[BuildData]:
        """
        Get recent workflow runs and convert them to BuildData.
        
        Args:
            limit: Maximum number of builds to retrieve
            
        Returns:
            List of BuildData objects
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return []
        
        try:
            runs = self.client.get_workflow_runs(self.repo_owner, self.repo_name, limit=limit)
            build_list = []
            
            for run in runs:
                build = self.client.get_build_data(self.repo_owner, self.repo_name, run['id'])
                if build:
                    build_list.append(build)
            
            logger.info(f"Retrieved {len(build_list)} recent builds")
            return build_list
        except Exception as e:
            logger.error(f"Failed to get recent builds: {e}")
            return []
    
    def get_build_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze build trends over a time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Build trend analysis
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return {}
        
        try:
            trends = self.client.analyze_build_trends(self.repo_owner, self.repo_name)
            return {
                'status': 'success',
                'build_analysis': trends,
                'analysis_date': datetime.utcnow().isoformat(),
                'period_days': days
            }
        except Exception as e:
            logger.error(f"Failed to analyze build trends: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_failed_builds(self, limit: int = 5) -> List[BuildData]:
        """
        Get failed workflow runs.
        
        Args:
            limit: Maximum number of failed builds to retrieve
            
        Returns:
            List of failed BuildData objects
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return []
        
        try:
            runs = self.client.get_workflow_runs(self.repo_owner, self.repo_name, limit=limit*2)
            failed_builds = []
            
            for run in runs:
                if run['conclusion'] in ['failure', 'timed_out', 'cancelled']:
                    build = self.client.get_build_data(self.repo_owner, self.repo_name, run['id'])
                    if build:
                        failed_builds.append(build)
                    if len(failed_builds) >= limit:
                        break
            
            logger.info(f"Retrieved {len(failed_builds)} failed builds")
            return failed_builds
        except Exception as e:
            logger.error(f"Failed to get failed builds: {e}")
            return []
    
    def get_commit_history(self, branch: str = 'main', limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get commit history for a branch.
        
        Args:
            branch: Branch name
            limit: Maximum commits to retrieve
            
        Returns:
            List of commit data
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return []
        
        try:
            commits = self.client.get_commit_history(
                self.repo_owner,
                self.repo_name,
                branch=branch,
                limit=limit
            )
            logger.info(f"Retrieved {len(commits)} commits from {branch}")
            return commits
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            return []
    
    def get_workflow_runs(self, workflow_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get workflow runs for the repository.
        
        Args:
            workflow_id: Optional specific workflow ID
            limit: Maximum runs to retrieve
            
        Returns:
            List of workflow runs
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return []
        
        try:
            runs = self.client.get_workflow_runs(
                self.repo_owner,
                self.repo_name,
                workflow_id=workflow_id,
                limit=limit
            )
            logger.info(f"Retrieved {len(runs)} workflow runs")
            return runs
        except Exception as e:
            logger.error(f"Failed to get workflow runs: {e}")
            return []
    
    def get_build_health_summary(self) -> Dict[str, Any]:
        """
        Get overall build health summary.
        
        Returns:
            Build health metrics
        """
        if not self.is_configured():
            logger.warning("GitHub service not configured")
            return {'status': 'unconfigured'}
        
        try:
            recent_builds = self.get_recent_builds(limit=30)
            
            if not recent_builds:
                return {
                    'status': 'unknown',
                    'total_builds': 0,
                    'health': 'N/A',
                    'repository': f"{self.repo_owner}/{self.repo_name}"
                }
            
            success_count = sum(1 for b in recent_builds if b.status == 'success')
            failed_count = sum(1 for b in recent_builds if b.status in ['failure', 'failed'])
            total_count = len(recent_builds)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            avg_build_time = sum(b.build_time_seconds for b in recent_builds) / len(recent_builds)
            
            return {
                'status': 'success',
                'repository': f"{self.repo_owner}/{self.repo_name}",
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
